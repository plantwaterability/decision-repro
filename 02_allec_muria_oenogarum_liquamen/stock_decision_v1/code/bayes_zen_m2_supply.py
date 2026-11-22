#!/usr/bin/env python3
"""bayes_zen_m2_supply.py (stock_decision_v1, BAYES-DATA): Mixture M2 Bucket 2 sub-allocation bayes_zen_v2_3_U3_exact_posterior_and_eu
(8% = 3,840 exposures) in the Stage 0 exposure / decision format, R2-rendered, parent_route "none".

  python3 bayes_zen_m2_supply.py --code C --sd0 SD0 --mix MIX_R2 --keys KEYS --corpus V23 --tok TOK --schema LABEL_SEMANTICS --out OUT

SOURCE: bayes_zen_v2_3 train split only (500 training profiles of the Zenodo 17677329 flight training file). Never dev_profile, the
  eval interaction files, hotel or human.
SELECTION (fixed before rendering): 960 training episodes, round-robin over profiles (each profile's episodes in sha256(NS|m2_ep|uid)
  order; at most 2 per profile), split by sha256(NS|m2_q|uid) into 672 next-choice episodes and 288 EU-action episodes; each episode
  contributes 4 of its 5 prefixes (the dropped r = sha256(NS|m2_drop|uid) mod 5), i.e. <= 4 siblings per parent and r ~ balanced.
  -> 2,688 exact posterior-predictive next-choice decisions (70%) + 1,152 EU-action decisions (30%), one exposure each.
TARGETS (bayes_zen_v2_3 primary, release-process filtered model declared in every state):
  next_choice -> target_basis exact_posterior (posterior predictive of the stated model), soft gold, loss_by_option null
  recommend   -> target_basis optimal_action, one-hot (exact ties share the mass, tie-accepting HARD), loss_by_option = expected loss
                 -E[w . phi(option)] under the same posterior (lower is better; regret = loss - min loss)
FORMAT: decision rows with sd0_render.py's fields (+ parent_route "none", m2 bucket tags); exposure rows (pos, exposure_id, decision_uid,
  exposure_k 0, option_order by default_rng(sha256(20261013|M2|uid|0)), slot_map, prompt_sha256, n_tokens, renderer_sha256).
CHECKS: renderer_sha256 must equal 0fab3c66 (fullft_render.renderer_spec); no truncation; Stage 0 component check (keys of every
  Stage 0 pool + r2 units_pools) must pass with no link into a non-training group.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    for k in ("code", "sd0", "mix", "keys", "corpus", "tok", "schema", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n-next", type=int, default=2688)
    ap.add_argument("--n-eu", type=int, default=1152)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    UB = load(os.path.join(HERE, "bayes_zen_uarms_build.py"), "ub")
    bz = load(os.path.join(HERE, "bayes_zen_convert.py"), "bz")
    TK = load(os.path.join(a.sd0, "sd0_textkeys.py"), "sd0_textkeys")
    SP = load(os.path.join(a.sd0, "sd0_pools.py"), "sd0_pools")
    CMP = load(os.path.join(a.sd0, "sd0_components.py"), "sd0_components")
    R = UB.Renderer(a.code, a.tok)
    schema_sha = hashlib.sha256(open(a.schema, "rb").read()).hexdigest()
    corpus_man = json.load(open(os.path.join(a.corpus, "manifest.json")))
    # ---- training records (next_choice + recommend), by decision group
    recs = {}
    tdir = os.path.join(a.corpus, "data", "train")
    for fn in sorted(os.listdir(tdir)):
        if not (fn.endswith("next_choice.jsonl") or fn.endswith("recommend.jsonl")):
            continue
        for l in open(os.path.join(tdir, fn)):
            r = json.loads(l)
            assert r["meta"]["sd"]["split"] == "train" and not r["meta"].get("quarantine")
            recs[r["id"]] = r
    eps = collections.defaultdict(list)
    for rid, r in recs.items():
        if rid.endswith("/next_choice"):
            eps[r["meta"]["profile_idx"]].append(r["meta"]["episode_uid"])
    eps = {p: sorted(set(v), key=lambda u: bz.hrank(bz.NS, "m2_ep", u)) for p, v in eps.items()}
    assert len(eps) == 500 and all(len(v) == 10 for v in eps.values())
    n_ep = (a.n_next + a.n_eu) // 4
    order = []
    for rnd in range(10):
        for p in sorted(eps, key=lambda p: bz.hrank(bz.NS, "m2_prof", rnd, p)):
            order.append(eps[p][rnd])
            if len(order) == n_ep:
                break
        if len(order) == n_ep:
            break
    chosen = sorted(order, key=lambda u: bz.hrank(bz.NS, "m2_q", u))
    nxt_eps, eu_eps = chosen[:a.n_next // 4], chosen[a.n_next // 4:]
    sel = []
    for q, group in (("next_choice", nxt_eps), ("recommend", eu_eps)):
        for u in group:
            drop = int(bz.hrank(bz.NS, "m2_drop", u)[:8], 16) % 5
            for r in range(5):
                if r != drop:
                    sel.append(recs[f"{u}/r{r}/{q}"])
    assert len(sel) == a.n_next + a.n_eu
    sel.sort(key=lambda r: bz.hrank(bz.NS, "m2_pos", r["id"]))
    # ---- render + rows
    drows, erows = [], []
    tokc = collections.Counter()
    for pos, rec in enumerate(sel):
        qid, q = next(iter(rec["questions"].items()))
        keys, texts, target = R.ls.cf.question_options(q, rec["labels"][qid])
        uid = f"bayes_zen_v2_3|{rec['id']}"
        u = {"qid": qid, "qtype": q["type"], "state": rec["state"], "instructions": q["instructions"], "option_keys": keys,
             "option_texts": texts, "noul_criteria": None, "target": target}
        o = UB.pilot_order("M2", uid, 0, "choice", len(keys))
        ps, nt = R.render(u, o)
        ps0, _ = R.render(u, list(range(len(keys))))
        m, sd = rec["meta"], rec["meta"]["sd"]
        gold = {k: float(v) for k, v in zip(keys, target)}
        opts = [{"key": k, "text": t} for k, t in zip(keys, texts)]
        is_eu = qid == "recommend"
        tmax = max(target)
        nz = [v for v in target if v > 0]
        hard = (tmax >= 0.999999) or (is_eu and len(set(round(v, 9) for v in nz)) == 1)
        loss = {k: -float(e) for k, e in zip(keys, m["eu_by_option"])} if is_eu else None
        if is_eu:
            best = min(loss.values())
            assert all((gold[k] > 0) == (abs(loss[k] - best) <= 1e-12) for k in keys), rec["id"]
        drows.append({
            "decision_uid": uid, "record_uid": uid, "parent_uid": f"bayes_zen_v2_3|{m['episode_uid']}",
            "ancestry_component_id": sd["ancestry_component_id"], "source_repo": "zenodo:17677329 via s3://YOUR_BUCKET/jev-race/stock_decision_v1/bayes_zen_v2_3/bayes_zen_v2_3.tgz",
            "source_revision": "tgz md5 9e80a90b937c8d96f9218cee1dae24f4; zenodo data.zip md5 a341ca74418ed9152a31cc3f91e261dd",
            "source_record_id": rec["id"], "source_family": "zenodo17677329_bayesian_teaching", "upstream_source": "zenodo17677329",
            "skill": "preference_posterior", "domain": "travel", "area": "knowledge", "generator_family": sd["generator_family"],
            "generator_revision": f"{corpus_man['builder']} (sha256 {corpus_man['builder_sha256'][:16]}...)", "split": "training",
            "state": rec["state"], "question": q["instructions"], "kind": "choice", "ordered_options": opts, "semantic_order": False,
            "gold_by_option_key": gold, "target_basis": "optimal_action" if is_eu else "exact_posterior",
            "target_basis_source": ("bayes_zen_v2_3 meta.question_basis=eu_recommendation: argmax E_post[w.phi] under the release-process "
                                    "(filtered) posterior; loss_by_option = expected loss -E_post[w.phi(option)]") if is_eu else
                                   ("bayes_zen_v2_3 meta.question_basis=next_choice_predictive: exact posterior predictive of the stated "
                                    "release-process (filtered) model"),
            "loss_by_option": loss, "label_verification": "exact_oracle", "label_provenance": "exact_posterior",
            "label_provenance_detail": "m2_bucket=2_exact_bayesian_posteriors_and_eu_actions; sub_allocation=bayes_zen_v2_3_U3_exact_posterior_and_eu; upstream_source=zenodo17677329",
            "evidence_refs": [f"source_record:{rec['id']}", "source:zenodo:17677329 (data.zip md5 a341ca74418ed9152a31cc3f91e261dd)",
                              f"generator:{sd['generator_family']}@{sd['generator_revision']}", f"oracle:{sd['label_verification'].get('oracle_id')}"],
            "label_semantics_schema_sha256": schema_sha,
            "label_uncertainty": {"label_kind": rec["label_kind"][qid], "tmax": float(tmax), "hard": bool(hard)},
            "license_use": {"licence": rec["license"], "line": "clean"}, "teacher_runs": [], "counterfactual_parent_uid": None,
            "renderer_sha256": R.ren, "prompt_sha256_first_exposure": ps, "prompt_sha256_stored_order": ps0,
            "option_sha256": hashlib.sha256(canon(opts).encode()).hexdigest(), "gold_sha256": hashlib.sha256(canon(gold).encode()).hexdigest(),
            "tokenizer_revision": UB.TOK_REV, "rendered_token_count": nt, "K": len(keys), "k_bucket": UB.kb(len(keys)),
            "len_bucket": {"1-512": "<=512"}.get(UB.lb(nt), UB.lb(nt)), "weight": 1.0, "di_ancestry": False,
            "m1_category": "probabilistic_disagreement", "parent_route": "none", "rounds_observed": m["rounds_observed"],
            "profile_idx": m["profile_idx"], "m2_bucket": "2_exact_bayesian_posteriors_and_eu_actions",
            "m2_sub_allocation": "bayes_zen_v2_3_U3_exact_posterior_and_eu"})
        erows.append({"pos": pos, "exposure_id": f"bzen_M2U3:{pos}", "decision_uid": uid, "exposure_k": 0,
                      "option_order": [keys[i] for i in o], "slot_map": {UB.letters(j): keys[i] for j, i in enumerate(o)},
                      "prompt_sha256": ps, "n_tokens": nt, "renderer_sha256": R.ren, "parent_route": "none"})
        tokc["tokens"] += nt
        tokc[f"len_{drows[-1]['len_bucket']}"] += 1
    dp = os.path.join(a.out, "decisions_bzen_m2_u3.jsonl.gz")
    ep = os.path.join(a.out, "exposures_bzen_m2_u3.jsonl.gz")
    with gzip.open(dp, "wt", encoding="utf-8") as f:
        for r in drows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(ep, "wt", encoding="utf-8") as f:
        for r in erows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # ---- component check (same rules as the U-arm build): next-choice records as "U3", EU records as "U2"
    blocks = {"U3": [{"decision_uid": r["id"]} for r in sel if r["id"].endswith("/next_choice")],
              "U2": [{"decision_uid": r["id"]} for r in sel if r["id"].endswith("/recommend")]}
    rr = {"U3": {r["id"]: r for r in sel if r["id"].endswith("/next_choice")}, "U2": {r["id"]: r for r in sel if r["id"].endswith("/recommend")}}
    rep = {}
    cc = UB.component_check(TK, SP, CMP, a.keys, a.mix, a.corpus, blocks, rr, rep)
    assert cc["validate_disjoint"]["status"] == "PASS" and not cc["collision_with_heldout_or_confirm_or_other_nontraining_group"], cc
    hdr = {"tag": "bzen_m2_u3", "mixture": "M2", "bucket": "2_exact_bayesian_posteriors_and_eu_actions",
           "sub_allocation": "bayes_zen_v2_3_U3_exact_posterior_and_eu", "share": 0.08, "builder": "bayes_zen_m2_supply.py",
           "builder_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "renderer_sha256": R.ren, "renderer_spec": R.spec,
           "renderer": "R2 = production build_prompt (claude_lmscore e0672bd0) with noul_defs=True, max_len 8192, margin 16, no truncation",
           "option_order_rule": "choice: numpy default_rng(sha256(20261013|M2|uid|exposure_k)[:16]) permutation",
           "label_semantics_schema_sha256": schema_sha, "corpus": {"name": "bayes_zen_v2_3", "tgz_md5": "9e80a90b937c8d96f9218cee1dae24f4",
                                                                   "builder_sha256": corpus_man["builder_sha256"]},
           "selection": {"episodes": n_ep, "profiles": len({r["profile_idx"] for r in drows}), "max_per_profile": max(collections.Counter(r["profile_idx"] for r in drows).values()),
                         "prefixes_per_episode": 4, "split_source": "train only (500 profiles); no dev_profile / eval / hotel / human",
                         "by_target_basis": dict(collections.Counter(r["target_basis"] for r in drows)),
                         "by_rounds_observed": {tb: dict(sorted(collections.Counter(r["rounds_observed"] for r in drows if r["target_basis"] == tb).items()))
                                                for tb in ("exact_posterior", "optimal_action")},
                         "hard_soft": {tb: dict(collections.Counter("hard" if r["label_uncertainty"]["hard"] else "soft" for r in drows if r["target_basis"] == tb))
                                       for tb in ("exact_posterior", "optimal_action")}},
           "counts": {"exposures": len(erows), "unique_decisions": len({r["decision_uid"] for r in drows}), **{k: int(v) for k, v in tokc.items()}},
           "component_check": {k: cc[k] for k in ("linked_records", "linked_groups", "collision_with_heldout_or_confirm_or_other_nontraining_group", "validate_disjoint", "own_heldout_splits")},
           "template_near_dup_note": ("states are MinHash near-duplicates of some bayes_zen flight dev states through the shared declared-model "
                                      "paragraph (0 shared profiles / option sets / states); exempt that template as boilerplate if those dev "
                                      "splits ever enter the Stage 0 registry"),
           "parent_route": "none", "files": {}}
    for p in (dp, ep):
        b = open(p, "rb").read()
        hdr["files"][os.path.basename(p)] = {"sha256": hashlib.sha256(b).hexdigest(), "md5": hashlib.md5(b).hexdigest(), "bytes": len(b)}
    json.dump(hdr, open(os.path.join(a.out, "render_bzen_m2_u3.json"), "w"), indent=1, default=str)
    print(json.dumps({k: hdr[k] for k in ("selection", "counts", "files")}, indent=1, default=str))
    print("component", cc["validate_disjoint"]["status"], "links", cc["linked_records"])


if __name__ == "__main__":
    main()
