#!/usr/bin/env python3
"""bayes_zen_uarms_build.py (stock_decision_v1, BAYES-DATA): the U0 / U1 / U1p / U2 / U3 exposure lists on the published Stage-2
pilot mixtures M0 and M1 (PILOT-DATA stage0/render/exposures_pilot_M{0,1}.jsonl.gz), from the bayes_zen_v2_2 U block.

  python3 bayes_zen_uarms_build.py --code C --sd0 SD0 --mix MIXDIR --keys KEYSDIR --corpus V22DIR --tok TOKDIR --out OUT

Per base tag:  M0  = M0 with its bayes_pref exposures swapped for same-stratum replays from M0 itself (U-arm build only)
               M1s = M1 with its bayes_pref exposures swapped the same way (both bases: every arm, U0 included, has 0 bayes_pref)
SWAP: each bayes_pref exposure (stratum choice | K 3-5 | 513-2048 | soft) is replaced, in position order, by a second exposure
  (exposure_k = 1) of a non-bayes_pref decision of the same stratum taken in ascending sha256(NS|replay|tag|uid) order, distinct
  decisions; option order by PILOT-DATA's rule default_rng(sha256(seed|mixture|uid|1)); rendered with the R2 renderer; a replay
  whose length bucket changes is skipped. U0 = the swapped base.
REPLACEMENT (coordinator decision 2026-10-14): U1 / U1p / U2 / U3 replace the SAME 1,920 slots spread over the WHOLE mixture,
  STRATIFIED BY UPSTREAM SOURCE: quotas proportional to each source's exposure count (largest remainder, ties by
  sha256(NS|uquota|source)), swap rows excluded, rows within a source by smallest sha256(NS|uslot|exposure_id); U rows fill the
  slots in block order. Every source loses ~5% (= the addendum's 5% replacement). No K / length matching: U arms carry ~+7%
  prompt tokens vs U0 (reported per arm). STOP RULE: a source losing more than 2x its proportional share (beyond a +1 rounding
  unit) aborts the build.
RENDERER: every U exposure and every replay is re-rendered here with trainers/fullft_render.py's pinned scorer (e0672bd0) +
  pinned tokenizer, noul_defs=True, max_len 8192; renderer_sha256 = FR.renderer_spec(...) must equal the mixtures' 0fab3c66...
  Decision-row parity: each replayed decision's exposure_k=0 prompt is re-rendered from its decision row and must reproduce
  prompt_sha256_first_exposure.
COMPONENT CHECK: the U block's model-visible keys (Stage 0 sd0_textkeys: identical state, shared >=20-word units with the DF
  rules, MinHash near-duplicates with the LSH / bucket / Jaccard rules) against every Stage 0 pool and against bayes_zen's own
  held-out splits; U records take the components they link to, then sd0_pools.validate_disjoint must pass. Counts only.
"""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xxhash

TOK_REV = "google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475"
REN_EXPECT = "0fab3c66ad3bb2044c27a849022912e3d83cf46fb51e9f5b1807fbaa821884e3"
SEED = 20261013
ARMS = ("U1", "U1p", "U2", "U3")
CELL = ("3-5", "513-2048")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(m)
    return m


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def kb(k):
    return next(f"{lo}-{hi}" for lo, hi in ((2, 2), (3, 5), (6, 16), (17, 32), (33, 52)) if lo <= int(k) <= hi)


def lb(n):
    n = int(n)
    return "1-512" if n <= 512 else "513-2048" if n <= 2048 else "2049-8192" if n <= 8192 else ">8192"


def letters(j):
    return chr(65 + j) if j < 26 else chr(97 + j - 26)


class Renderer:
    def __init__(self, code, tok_dir):
        self.FR = load(os.path.join(code, "trainers", "fullft_render.py"), "fullft_render")
        self.ls, md5s = self.FR.load_scorer(os.path.join(code, "trainers"))
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        self.spec, self.ren = self.FR.renderer_spec(md5s, tok_dir, 8192, True, TOK_REV)
        assert self.ren == REN_EXPECT, (self.ren, "renderer differs from the mixtures' R2")

    def render(self, u, order):
        text = self.ls.build_prompt(self.tok, u, order, 8192, noul_defs=True)
        if text is None or self.FR.was_cut(self.ls, self.tok, u, order, text, 8192, True):
            raise SystemExit(f"{u.get('qid')}: does not fit 8192 without cutting")
        return sha(text), len(self.tok(text, add_special_tokens=False)["input_ids"])


def unit_from_decision(d):
    keys = [o["key"] for o in d["ordered_options"]]
    return {"qid": d["decision_uid"], "qtype": d["kind"], "state": d["state"], "instructions": d["question"], "option_keys": keys,
            "option_texts": [o["text"] for o in d["ordered_options"]], "noul_criteria": None,
            "target": [d["gold_by_option_key"][k] for k in keys]}


def pilot_order(mixture, uid, k, kind, K):
    if kind == "choice" and K >= 2:
        s = int(sha(f"{SEED}|{mixture}|{uid}|{k}")[:16], 16)
        return [int(x) for x in np.random.default_rng(s).permutation(K)]
    return list(range(K))


# ------------------------------------------------------------------------------------------------ U block
def load_ublock(R, corpus, bz):
    ub = os.path.join(corpus, "u_block")
    blocks, recs, bad = {}, {}, collections.Counter()
    for arm in ARMS:
        rows = [json.loads(l) for l in open(os.path.join(ub, f"exposures_{arm}.jsonl"))]
        rr = {}
        for fn in os.listdir(os.path.join(ub, arm, "train")):
            for l in open(os.path.join(ub, arm, "train", fn)):
                r = json.loads(l)
                rr[r["id"]] = r
        for row in rows:
            rec = rr[row["decision_uid"]]
            qid, q = next(iter(rec["questions"].items()))
            keys, texts, target = R.ls.cf.question_options(q, rec["labels"][qid])
            u = {"qid": qid, "qtype": q["type"], "state": rec["state"], "instructions": q["instructions"], "option_keys": keys,
                 "option_texts": texts, "noul_criteria": None, "target": target}
            order = [keys.index(k) for k in row["option_order"]]
            ps, nt = R.render(u, order)
            bad["prompt_sha_mismatch"] += ps != row["prompt_sha256"]
            bad["n_tokens_mismatch"] += nt != row["n_tokens"]
            row.update({"renderer_sha256": R.ren, "slot_map": {letters(j): keys[o] for j, o in enumerate(order)}, "K": len(keys)})
        blocks[arm], recs[arm] = rows, rr
    n = len(blocks["U1"])
    assert all(len(blocks[a]) == n for a in ARMS)
    assert all(len({blocks[a][i]["decision_group"] for a in ARMS}) == 1 for i in range(n))
    return blocks, recs, dict(bad)


def ublock_decision_rows(R, blocks, recs):
    """U-block decisions in PILOT-DATA's decision-row schema (the trainer reads state / question / options / gold here)."""
    out = {}
    for arm in ARMS:
        rows = []
        for row in blocks[arm]:
            rec = recs[arm][row["decision_uid"]]
            qid, q = next(iter(rec["questions"].items()))
            keys, texts, target = R.ls.cf.question_options(q, rec["labels"][qid])
            sd = rec["meta"]["sd"]
            gold = {k: float(v) for k, v in zip(keys, target)}
            opts = [{"key": k, "text": t} for k, t in zip(keys, texts)]
            rows.append({"decision_uid": row["decision_uid"], "record_uid": row["decision_uid"], "parent_uid": sd["parent_uid"],
                         "ancestry_component_id": sd["ancestry_component_id"], "source_repo": sd["source_repo"],
                         "source_revision": sd["source_revision"], "source_record_id": sd["source_record_id"],
                         "source_family": sd["source_family"], "upstream_source": "zenodo17677329", "skill": sd["skill"],
                         "domain": sd["domain"], "area": "knowledge", "generator_family": sd["generator_family"],
                         "generator_revision": sd["generator_revision"], "split": "train", "state": rec["state"],
                         "question": q["instructions"], "kind": q["type"], "ordered_options": opts, "semantic_order": False,
                         "gold_by_option_key": gold, "target_basis": sd["target_basis"], "label_verification": sd["label_verification"],
                         "label_uncertainty": {"label_kind": rec["label_kind"][qid], "tmax": max(target), "hard": max(target) >= 0.999999},
                         "license_use": {"licence": rec["license"], "line": "clean (CC BY 4.0)"}, "teacher_runs": [],
                         "counterfactual_parent_uid": None, "renderer_sha256": R.ren, "prompt_sha256_first_exposure": row["prompt_sha256"],
                         "option_sha256": sha(json.dumps(opts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                         "gold_sha256": sha(json.dumps(gold, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                         "tokenizer_revision": TOK_REV, "rendered_token_count": row["n_tokens"], "K": len(keys), "k_bucket": kb(len(keys)),
                         "len_bucket": lb(row["n_tokens"]), "weight": 1.0, "u_arm": arm, "di_ancestry": False, "m1_category": "probabilistic_disagreement"})
        out[arm] = rows
    return out


# ------------------------------------------------------------------------------------------------ bases
def build_base(R, bz, mixdir, mixture, tag, swap, blocks, report):
    ex = [json.loads(l) for l in gzip.open(os.path.join(mixdir, f"exposures_pilot_{mixture}.jsonl.gz"), "rt")]
    D = {}
    for l in gzip.open(os.path.join(mixdir, f"decisions_pilot_{mixture}.jsonl.gz"), "rt"):
        d = json.loads(l)
        D[d["decision_uid"]] = d
    man = json.load(open(os.path.join(mixdir, f"{mixture}.manifest.json")))
    assert [e["pos"] for e in ex] == list(range(len(ex)))
    rep = {"base": mixture, "tag": tag, "swap": swap, "base_exposures": len(ex), "base_tokens": int(sum(e["n_tokens"] for e in ex)),
           "base_renderer": sorted({e["renderer_sha256"] for e in ex})}
    bp = [e["pos"] for e in ex if "bayes_pref" in e["decision_uid"]]
    rep["bayes_pref_in_base"] = len(bp)
    if mixture == "M0":
        assert bp == man["bayes_pref_exposures"]["positions"], "bayes_pref positions differ from the M0 manifest"
    rows = [dict(e) for e in ex]
    swapped = set()
    if swap:
        strat = lambda e: (D[e["decision_uid"]]["kind"], kb(D[e["decision_uid"]]["K"]), lb(e["n_tokens"]),  # noqa: E731
                           "hard" if D[e["decision_uid"]]["label_uncertainty"]["hard"] else "soft")
        want = collections.Counter(strat(ex[p]) for p in bp)
        assert list(want) == [("choice", "3-5", "513-2048", "soft")], want
        cands = sorted({e["decision_uid"] for e in ex if "bayes_pref" not in e["decision_uid"] and e["exposure_k"] == 0
                        and strat(e) == ("choice", "3-5", "513-2048", "soft")}, key=lambda u: bz.hrank(bz.NS, "replay", tag, u))
        parity = collections.Counter()
        it = iter(cands)
        used = []
        for p in bp:
            while True:
                uid = next(it)
                d = D[uid]
                u = unit_from_decision(d)
                o0 = pilot_order(mixture, uid, 0, d["kind"], d["K"])
                ps0, _ = R.render(u, o0)
                parity["first_exposure_match" if ps0 == d["prompt_sha256_first_exposure"] else "first_exposure_MISMATCH"] += 1
                if ps0 != d["prompt_sha256_first_exposure"]:
                    continue
                o1 = pilot_order(mixture, uid, 1, d["kind"], d["K"])
                ps1, nt1 = R.render(u, o1)
                if lb(nt1) != "513-2048":
                    parity["replay_skipped_bucket_change"] += 1
                    continue
                break
            keys = [o["key"] for o in d["ordered_options"]]
            rows[p] = {"pos": p, "exposure_id": f"bzenU_{tag}_swap:{p}", "decision_uid": uid, "exposure_k": 1,
                       "option_order": [keys[i] for i in o1], "slot_map": {letters(j): keys[o] for j, o in enumerate(o1)},
                       "prompt_sha256": ps1, "n_tokens": nt1, "renderer_sha256": R.ren,
                       "swap": {"replaces_exposure_id": ex[p]["exposure_id"], "replaces_decision_uid": ex[p]["decision_uid"],
                                "rule": "same-stratum replay from the base (choice|3-5|513-2048|soft), U-arm build only"}}
            swapped.add(p)
            used.append(uid)
        rep["swap_report"] = {"swapped": len(swapped), "replayed_decisions_distinct": len(set(used)), "parity": dict(parity),
                              "tokens_removed": int(sum(ex[p]["n_tokens"] for p in bp)),
                              "tokens_added": int(sum(rows[p]["n_tokens"] for p in bp))}
    bp_after = sum(1 for r in rows if "bayes_pref" in r["decision_uid"])
    rep["bayes_pref_after_swap"] = bp_after
    # replacement slots
    rep["u_block_k_length"] = dict(collections.Counter(f"K{kb(r['K'])}|{lb(r['n_tokens'])}" for r in blocks["U3"]))
    cand = [i for i, r in enumerate(rows) if i not in swapped]          # whole mixture (swap rows excluded)
    rep["cell_supply"] = len(cand)
    need_n = len(blocks["U3"])
    src_of = lambda i: str(D[rows[i]["decision_uid"]].get("upstream_source"))   # noqa: E731
    by_src = collections.defaultdict(list)
    for i in cand:
        by_src[src_of(i)].append(i)
    share = {k: need_n * len(v) / len(cand) for k, v in by_src.items()}
    quota = {k: int(np.floor(x)) for k, x in share.items()}
    left = need_n - sum(quota.values())
    for k in sorted(share, key=lambda k: (-(share[k] - quota[k]), bz.hrank(bz.NS, "uquota", k)))[:left]:
        quota[k] += 1
    slots = []
    for k, v in by_src.items():
        slots += sorted(v, key=lambda i: bz.hrank(bz.NS, "uslot", rows[i]["exposure_id"]))[:quota[k]]
    slots = sorted(slots)
    assert len(slots) == need_n and len(set(slots)) == need_n
    # per-source before / after (cell and overall), and the 2x stop rule
    overall = collections.Counter(src_of(i) for i in range(len(rows)))
    tab = {}
    for k in sorted(by_src, key=lambda k: -len(by_src[k])):
        tab[k] = {"cell_before": len(by_src[k]), "cell_after": len(by_src[k]) - quota[k], "removed": quota[k],
                  "proportional_share": round(share[k], 3), "overall_before": overall[k], "overall_after": overall[k] - quota[k],
                  "overall_loss_frac": round(quota[k] / overall[k], 4)}
    viol = {k: v for k, v in tab.items() if v["removed"] > 2 * v["proportional_share"] and v["removed"] - v["proportional_share"] > 1}
    rep["replacement_by_source"] = tab
    rep["replacement_rule"] = ("whole-mixture spreading, stratified by upstream_source: quotas proportional to each source's exposures "
                               "(largest remainder), swap rows excluded; no K / length matching")
    rep["max_removed_over_share"] = round(max(v["removed"] / v["proportional_share"] for v in tab.values() if v["proportional_share"] > 0), 3)
    rep["cell_removed_frac"] = round(need_n / len(cand), 4)
    rep["replaced_by_k_and_length"] = dict(collections.Counter(f"K{kb(D[rows[i]['decision_uid']]['K'])}|{lb(rows[i]['n_tokens'])}" for i in slots))
    rep["max_overall_loss_frac"] = max((v["overall_loss_frac"], k) for k, v in tab.items())
    rep["stop_rule_violations"] = viol
    if viol:
        report[tag] = rep
        raise SystemExit(f"{tag}: STOP: sources losing more than 2x their proportional share: {viol}")
    arms = {"U0": rows}
    for arm in ARMS:
        a = list(rows)
        for j, i in enumerate(slots):
            b = blocks[arm][j]
            a[i] = {"pos": i, "exposure_id": f"bzenU_{tag}_{arm}:{i}", "decision_uid": b["decision_uid"], "exposure_k": 0,
                    "option_order": b["option_order"], "slot_map": b["slot_map"], "prompt_sha256": b["prompt_sha256"],
                    "n_tokens": b["n_tokens"], "renderer_sha256": R.ren, "u_arm": arm, "replaces": rows[i]["exposure_id"],
                    "replaces_decision_uid": rows[i]["decision_uid"], "decision_group": b["decision_group"],
                    "target_basis_kind": b["target_basis_kind"], "parent_route": "none", "weight": 1.0, "gold": b["gold"]}
        arms[arm] = a
    rep["replaced_slots"] = len(slots)
    rep["replaced_share"] = round(len(slots) / len(rows), 6)
    rep["replaced_by_source"] = dict(collections.Counter(D[rows[i]["decision_uid"]]["upstream_source"] for i in slots).most_common(12))
    rep["replaced_bayes_pref"] = sum(1 for i in slots if "bayes_pref" in rows[i]["decision_uid"])
    rep["replaced_tokens"] = int(sum(rows[i]["n_tokens"] for i in slots))
    tok = {a: int(sum(r["n_tokens"] for r in v)) for a, v in arms.items()}
    rep["tokens"] = tok
    rep["token_delta_vs_published_base"] = {a: [tok[a] - rep["base_tokens"], round(tok[a] / rep["base_tokens"] - 1, 5)] for a in tok}
    rep["token_delta_vs_U0"] = {a: [tok[a] - tok["U0"], round(tok[a] / tok["U0"] - 1, 5)] for a in tok}
    rep["renderers"] = {a: sorted({r["renderer_sha256"] for r in v}) for a, v in arms.items()}
    rep["exposures"] = {a: len(v) for a, v in arms.items()}
    rep["bayes_pref_in_arm"] = {a: sum(1 for r in v if "bayes_pref" in r["decision_uid"]) for a, v in arms.items()}
    same_slots = all([i for i, r in enumerate(arms[a]) if r.get("u_arm")] == slots for a in ARMS)
    rep["same_slots_all_arms"] = same_slots
    report[tag] = rep
    return arms


# ------------------------------------------------------------------------------------------------ component check
def ukeys(TK, recs_iter):
    """sd0_textkeys keys for a list of records: state hash, >=12-word units (with field and word count), minhash, lineage."""
    out = {"lin": [], "sh": [], "mh": [], "tu": []}
    for ri, rec in enumerate(recs_iter):
        st = TK.as_text(rec.get("state"))
        ns = TK.norm(st)
        w = ns.split()
        out["lin"].append(str((rec.get("meta") or {}).get("group_id") or rec.get("base_id") or rec.get("id")))
        out["sh"].append(xxhash.xxh64_intdigest(ns.encode()) if len(w) >= TK.MIN_STATE_WORDS else 0)
        out["mh"].append(TK.minhash(w))
        seen = {}
        for u in TK.units_of(st):
            seen.setdefault(u, TK.FIELDS["state"])
        for qid, q in (rec.get("questions") or {}).items():
            for u in TK.units_of(TK.as_text(q.get("instructions") or "")):
                seen.setdefault(u, TK.FIELDS["instructions"])
            for v in (q.get("criteria") or {}).values() if isinstance(q.get("criteria"), dict) else []:
                for u in TK.units_of(TK.as_text(v)):
                    seen.setdefault(u, TK.FIELDS["criteria"])
        for u, f in seen.items():
            out["tu"].append((xxhash.xxh64_intdigest(u.encode()), ri, f, len(u.split())))
    out["sh"] = np.array(out["sh"], dtype=np.uint64)
    out["mh"] = np.stack(out["mh"])
    return out


def band_keys(mh, b, ROWS):
    band = mh[:, b * ROWS:(b + 1) * ROWS].astype(np.uint64)
    return (band[:, 0] << np.uint64(32)) ^ (band[:, 1] << np.uint64(21)) ^ (band[:, 2] << np.uint64(11)) ^ band[:, 3]


def buckets(uk, ui, ok, oi):
    """(U indices, other indices) for every band key present in both, by sorted search."""
    common = np.intersect1d(uk, ok)
    if not len(common):
        return
    so = np.argsort(ok, kind="stable")
    oks = ok[so]
    su = np.argsort(uk, kind="stable")
    uks = uk[su]
    for key in common:
        a0, a1 = np.searchsorted(oks, key, "left"), np.searchsorted(oks, key, "right")
        b0, b1 = np.searchsorted(uks, key, "left"), np.searchsorted(uks, key, "right")
        yield ui[su[b0:b1]], oi[so[a0:a1]]


def near_dups(CMP, Umh, Omh, cnt, links):
    """validator N rule: LSH bands, Jaccard >= 0.8, buckets above BUCKET_MAX skipped (counted)."""
    oi = np.flatnonzero(Omh.any(axis=1))
    ui_ = np.flatnonzero(Umh.any(axis=1))
    for b in range(CMP.BANDS):
        for us, os_ in buckets(band_keys(Umh[ui_], b, CMP.ROWS), ui_, band_keys(Omh[oi], b, CMP.ROWS), oi):
            if len(us) + len(os_) > CMP.BUCKET_MAX:
                cnt["N_buckets_over_max_(validator_skips)"] += 1
                if len(us) * len(os_) <= 2_000_000:
                    eq = (Umh[us][:, None, :] == Omh[os_][None, :, :]).mean(axis=2)
                    cnt["N_pairs_J>=0.8_in_skipped_buckets"] += int((eq >= CMP.JACCARD).sum())
                continue
            eq = (Umh[us][:, None, :] == Omh[os_][None, :, :]).mean(axis=2)
            ii, jj = np.nonzero(eq >= CMP.JACCARD)
            for a_, b_ in zip(ii, jj):
                if links is not None:
                    links[int(us[a_])].add(int(os_[b_]))
                cnt["N_near_dup_links"] += 1


def component_check(TK, SP, CMP, keysdir, mixdir, corpus, blocks, recs, rep):
    t0 = time.time()
    # U-block records: next_choice twins (U1 / U1p / U3 share their visible content) and recommend (U2)
    urecs = [recs["U3"][r["decision_uid"]] for r in blocks["U3"]] + [recs["U2"][r["decision_uid"]] for r in blocks["U2"]]
    U = ukeys(TK, urecs)
    ulin = set(U["lin"])
    res = {"u_records_keyed": len(urecs), "u_state_hashed": int((U["sh"] != 0).sum()), "u_minhashed": int(U["mh"].any(axis=1).sum()),
           "u_text_units_ge20_nonInstr": sum(1 for t in U["tu"] if t[3] >= CMP.EDGE_WORDS and t[2] != 2)}
    up = pd.read_parquet(os.path.join(mixdir, "units_pools.parquet"), columns=["rk", "pool_assign", "comp"])
    grp = {}
    for g, tags in SP.OWNER_GROUPS.items():
        for t in tags:
            grp[t] = g
    linked_rk = collections.Counter()
    per_pool = {}
    uh_set = set(int(t[0]) for t in U["tu"] if t[3] >= CMP.EDGE_WORDS and t[2] != 2)
    for fn in sorted(os.listdir(keysdir)):
        if not fn.startswith("keys_"):
            continue
        pool = fn[5:-4]
        rr = [json.loads(l) for l in gzip.open(os.path.join(keysdir, f"recs_{pool}.jsonl.gz"), "rt")]
        d = np.load(os.path.join(keysdir, fn))
        cnt = collections.Counter()
        links = collections.defaultdict(set)
        # S
        osh = d["state_hash"]
        hit = np.flatnonzero(np.isin(osh, U["sh"][U["sh"] != 0]))
        for j in hit:
            for ui in np.flatnonzero(U["sh"] == osh[j]):
                links[int(ui)].add(int(j))
                cnt["S_state"] += 1
        # T
        tw, tf, th, tr = d["tu_words"], d["tu_field"], d["tu_hash"], d["tu_rec"]
        sel = np.flatnonzero((tw >= CMP.EDGE_WORDS) & (tf != 2))
        th_s = th[sel]
        m = np.isin(th_s, np.fromiter(uh_set, dtype=np.uint64, count=len(uh_set)))
        cnt["T_units_shared_raw"] = int(m.sum())
        if m.any():
            occ = collections.defaultdict(list)                  # unit hash -> pool occurrences (record, field)
            for j in sel[np.flatnonzero(m)]:
                occ[int(th[j])].append((int(tr[j]), int(tf[j])))
            for h, oc in occ.items():
                uo = [(t[1], t[2]) for t in U["tu"] if t[0] == h]
                lins = {"o|" + rr[r_][1] for r_, _ in oc} | {"u|" + U["lin"][u] for u, _ in uo}
                stt = all(f == 1 for _, f in oc) and all(f == 1 for _, f in uo)
                nl = len(lins)      # DF within this pool + U (the validator counts all pools: more lineages, fewer edges)
                if (stt and 2 <= nl <= CMP.DF_MAX) or ((not stt) and 2 <= nl <= CMP.DF_MAX_OPT):
                    for u, _ in uo:
                        for r_, _ in oc:
                            links[u].add(r_)
                    cnt["T_text_unit_links"] += 1
                else:
                    cnt["T_units_boilerplate_over_df"] += 1
        # N
        near_dups(CMP, U["mh"], d["minhash"], cnt, links)
        for u, js in links.items():
            for j in js:
                linked_rk[rr[j][0]] += 1
        per_pool[pool] = {k: int(v) for k, v in cnt.items()} or {"links": 0}
        per_pool[pool]["u_records_linked"] = len(links)
        print(f"[{time.time() - t0:.0f}s] keys {pool}: {per_pool[pool]}", flush=True)
        del d, rr
    res["stage0_pools"] = per_pool
    sub = up[up.rk.isin(set(linked_rk))]
    linked_groups = collections.Counter(grp.get(pa, f"unassigned:{pa}") for pa in sub.pool_assign)
    linked_comps = set(int(c) for c in sub.comp)
    missing = set(linked_rk) - set(sub.rk)
    if missing:
        linked_groups["record_not_in_units_pools"] += len(missing)
    res["linked_records"] = len(linked_rk)
    res["linked_groups"] = dict(linked_groups)
    res["linked_components"] = len(linked_comps)
    non_training = {g: n for g, n in linked_groups.items() if g != "training"}
    res["collision_with_heldout_or_confirm_or_other_nontraining_group"] = bool(non_training)
    # validator: U rows join the training group with the components they link to (or a new component each)
    aug = up[["pool_assign", "comp"]].copy()
    newc = int(aug.comp.max()) + 1
    add = []
    for i in range(len(urecs)):
        add.append({"pool_assign": "pilot", "comp": newc + i})
    for c in linked_comps:
        add.append({"pool_assign": "pilot", "comp": int(c)})
    aug = pd.concat([aug, pd.DataFrame(add)], ignore_index=True)
    try:
        res["validate_disjoint"] = SP.validate_disjoint(aug)
        res["validate_disjoint"]["status"] = "PASS"
    except SystemExit as e:
        res["validate_disjoint"] = {"status": "FAIL", "detail": str(e)[:500]}
    # own held-out splits (bayes_zen dev splits + generator holdout): S / T / N against U, same rules
    own = collections.Counter()
    drecs, dsplit = [], []
    for split in sorted(os.listdir(os.path.join(corpus, "data"))):
        if split == "train":
            continue
        for fn in sorted(os.listdir(os.path.join(corpus, "data", split))):
            if not (fn.endswith("next_choice.jsonl") or fn.endswith("recommend.jsonl") or "genholdout" in fn):
                continue
            for l in open(os.path.join(corpus, "data", split, fn)):
                drecs.append(json.loads(l))
                dsplit.append(split)
    Dk = ukeys(TK, drecs)
    own["dev_records_keyed"] = len(drecs)
    own["S_state_links"] = int(np.isin(U["sh"][U["sh"] != 0], Dk["sh"][Dk["sh"] != 0]).sum())
    dth = {}
    for t in Dk["tu"]:
        if t[3] >= CMP.EDGE_WORDS and t[2] != 2:
            dth.setdefault(t[0], set()).add(Dk["lin"][t[1]])
    tl = 0
    for t in U["tu"]:
        if t[3] >= CMP.EDGE_WORDS and t[2] != 2 and t[0] in dth:
            nl = len(dth[t[0]] | {U["lin"][t[1]]})
            stt = t[2] == 1
            if (stt and nl <= CMP.DF_MAX) or ((not stt) and nl <= CMP.DF_MAX_OPT):
                tl += 1
    own["T_text_unit_links_within_df"] = tl
    near_dups(CMP, U["mh"], Dk["mh"], own, None)
    res["own_heldout_splits"] = dict(own)
    res["seconds"] = round(time.time() - t0, 1)
    rep["component_check"] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ("code", "sd0", "mix", "keys", "corpus", "tok", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    bz = load(os.path.join(a.code, "stock_decision_v1", "code", "bayes_zen_convert.py"), "bz")
    TK = load(os.path.join(a.sd0, "sd0_textkeys.py"), "sd0_textkeys")
    SP = load(os.path.join(a.sd0, "sd0_pools.py"), "sd0_pools")
    CMP = load(os.path.join(a.sd0, "sd0_components.py"), "sd0_components")
    R = Renderer(a.code, a.tok)
    rep = {"builder": "bayes_zen_uarms_build.py", "builder_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "renderer_sha256": R.ren, "corpus": a.corpus,
           "mixture_inputs": {f: hashlib.md5(open(os.path.join(a.mix, f), "rb").read()).hexdigest() for f in sorted(os.listdir(a.mix))
                              if f.startswith(("exposures_pilot_", "decisions_pilot_")) or f.endswith(".manifest.json")},
           "template_near_dup_note": ("U-block states are MinHash near-duplicates (Stage 0 N rule, J>=0.8) of some bayes_zen flight dev states "
                                      "(dev_profile / dev_flight_eval) because they share the declared-model paragraph; 0 pairs share a profile, "
                                      "option set or state. Accepted. If those flight dev splits ever enter the Stage 0 registry, exempt the "
                                      "bayes_zen declared-model template as boilerplate (strip it before shingling) or the N rule will merge "
                                      "them with training components. Hotel, human and generator-holdout splits have no links.")}
    blocks, recs, bad = load_ublock(R, a.corpus, bz)
    rep["ublock_render_check"] = {"rows_per_arm": len(blocks["U1"]), **{k: int(v) for k, v in bad.items()},
                                  "renderer_sha256": R.ren}
    assert not any(bad.values()), bad
    print("U block renders identically under", R.ren, flush=True)
    dec = ublock_decision_rows(R, blocks, recs)
    for arm, rows in dec.items():
        with gzip.open(os.path.join(a.out, f"decisions_ublock_{arm}.jsonl.gz"), "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for mixture, tag, swap in (("M0", "M0", True), ("M1", "M1s", True)):
        arms = build_base(R, bz, a.mix, mixture, tag, swap, blocks, rep)
        for arm, rows in arms.items():
            with gzip.open(os.path.join(a.out, f"exposures_{tag}_{arm}.jsonl.gz"), "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(tag, json.dumps({k: rep[tag][k] for k in ("bayes_pref_in_base", "bayes_pref_after_swap", "cell_supply", "replaced_slots",
                                                          "token_delta_vs_published_base", "token_delta_vs_U0")}), flush=True)
    component_check(TK, SP, CMP, a.keys, a.mix, a.corpus, blocks, recs, rep)
    files = {}
    for fn in sorted(os.listdir(a.out)):
        p = os.path.join(a.out, fn)
        b = open(p, "rb").read()
        files[fn] = {"sha256": hashlib.sha256(b).hexdigest(), "md5": hashlib.md5(b).hexdigest(), "bytes": len(b)}
    rep["files"] = files
    json.dump(rep, open(os.path.join(a.out, "uarms_report.json"), "w"), indent=1, default=str)
    print(json.dumps(rep["component_check"], indent=1, default=str))


if __name__ == "__main__":
    main()
