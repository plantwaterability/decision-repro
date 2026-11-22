#!/usr/bin/env python3
"""FS5 training manifest = FS1 manifest (8461365d) MINUS procedural_typed_decisions/* (except partial_observation_calibration and
policy_under_uncertainty) MINUS synth_workflows/*, PLUS a supplement of real train-split rows for under-covered DI gap task types.

Conventions copied from the earlier builds: DI-format rows (kit normalizer twins, as pools_v8.py), rendered with
build_manifest_v3.make_render (renderer R3 = ddfaf6c0, options shuffled; noul rows rendered as the engine renders RAGTruth:
Yes / No with definitions, fixed order), gated with build_manifest_v3.Gate (exact state >= 25 chars + 13-grams of the item's own
content + short-item containment) over the DI 0.3 public suite + FS1's held-out list + v8's held-out list + cov2 + MMLU-CF val
(SuperGPQA filter), per-benchmark id bans (When2Call test questions, Habermas test policy questions, RAGTruth test source ids,
SuperGPQA uuids already trained / held out), deduplicated against every FS5 base row by (state, instructions) as fs1_build.py
(Habermas: + option texts, because one round has two candidate sets), quotas in sha1(id) order. Rows keep FS1's 23 keys exactly.
Sealed sets, the guard set, typed-decision splits, the ESCI test split and benchmark test splits are never read as rows.

fs5_build.py --w WORKDIR   (layout made by stage.sh)"""
import argparse, collections, csv, gzip, hashlib, json, math, random, re, sys, ast

ap = argparse.ArgumentParser()
ap.add_argument("--w", default="/data/shared/fs5")
ap.add_argument("--suite", default="/data/local/phish_v73h/suite/suite-0.3")
ap.add_argument("--seed", type=int, default=20261113)
ap.add_argument("--caps", default="vast=9000,acos=8000,when2call=6000,ragtruth=6000,habermas=3000,sgpqa_gpqa=5000,sgpqa_mmlupro=3000")
ap.add_argument("--only-pools", action="store_true", help="build pools + report, skip the manifest assembly")
a = ap.parse_args()
W = a.w
sys.path[:0] = [f"{W}/code/esbar_code_v2e/trainers", f"{W}/scripts"]
import build_manifest_v3 as BM
import claude_lmscore as clm
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{W}/tok")
CAPS = {k: int(v) for k, v in (x.split("=") for x in a.caps.split(","))}
REP = {"caps": CAPS, "pools": {}}
USER, QUE, OPT = "<|turn>user\n", "\n\nQuestion: ", "\n\nOptions:\n"
KEEP_PROC = {"partial_observation_calibration", "policy_under_uncertainty"}
EXCLUDED_QIDS = {"action", "needs_review", "outcome", "risk", "urgency", "category", "churn_risk", "needs_human", "discrepancy_severity",
                 "disposition", "duplicate", "matches_order", "credential_compromise", "severity", "true_positive"}   # synth_workflows.exclusion


def jl(p):
    with (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def md5d(s):
    return hashlib.md5(s.encode()).digest()


def parse_prompt(p):                       # = fs1_build.parse_prompt
    io_ = p.rfind(OPT); iq = p.rfind(QUE, 0, io_); iu = p.find(USER)
    if io_ < 0 or iu < 0:
        return None
    if iq < 0:
        body = p[iu + len(USER):io_]
        return ("", body[len("Question: "):]) if body.startswith("Question: ") else None
    return p[iu + len(USER):iq], p[iq + len(QUE):io_]


def as_text(x):
    return "" if x is None else x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def key_of(st, ins, opts=None):
    k = BM.norm_text(st) + "|" + BM.norm_text(ins)
    if opts is not None:
        k += "|" + "|".join(sorted(BM.norm_text(o) for o in opts))
    return md5d(k)


def is_removed(r):
    """user rule 11-12: drop procedural_typed_decisions/* except the two exact-Bayes configs, and every synth_workflows/* row."""
    tags = [str(x) for x in (r.get("source"), r.get("source_family")) if x]
    uid = r["decision_uid"]
    if any(t.startswith("synth_workflows/") for t in tags) or uid.startswith("synth_wf:"):
        return "synth_workflows"
    for t in tags:
        if t.startswith("procedural_typed_decisions/"):
            return None if t.split("/", 1)[1] in KEEP_PROC else "procedural_typed_decisions"
    if uid.startswith(("proc_harvest_", "proc_td_")):
        raise SystemExit(f"procedural uid without a procedural source tag: {uid}")
    return None


# ------------------------------------------------------------------ base = FS1 minus the removed rows (streamed, order kept)
FS1 = f"{W}/in/esbar_stage1_fs1.jsonl"
seen, uids, sgpqa_used, seen_opt = set(), set(), set(), set()


def prompt_options(p, k):
    """option texts shown in a rendered prompt (letters stripped), or None if the block does not split into k options"""
    io_ = p.rfind(OPT); ia = p.rfind("\n\nAnswer with")
    if io_ < 0 or ia < io_:
        return None
    parts = re.split(r"\n(?=[A-Za-z]\) )", p[io_ + len(OPT):ia])
    if len(parts) != k:
        return None
    return [re.sub(r"^[A-Za-z]\) ", "", x, count=1) for x in parts]


KEYS = None; base_c = collections.Counter(); removed = collections.Counter(); kept_proc = collections.Counter()
base_rows_by_src = collections.Counter(); base_parent = 0; base_w = 0.0
h_fs1 = hashlib.sha256()
TMP_BASE = f"{W}/out/fs5_base.jsonl"
import os
os.makedirs(f"{W}/out/pools", exist_ok=True)
with open(TMP_BASE, "w", encoding="utf-8") as fb:
    for l in open(FS1, encoding="utf-8"):
        h_fs1.update(l.encode())
        r = json.loads(l)
        if KEYS is None:
            KEYS = list(r.keys())
        why = is_removed(r)
        if why:
            removed[(why, r.get("source_family") or r.get("source"))] += 1
            continue
        if str(r.get("source_family") or "").startswith("procedural_typed_decisions/"):
            kept_proc[r["source_family"]] += 1
        pp = parse_prompt(r["prompt"]) if r.get("kind") != "gen" else None
        if pp:
            seen.add(key_of(*pp))
            po = prompt_options(r["prompt"], len(r["option_order"]))
            if po is not None:
                seen_opt.add(key_of(pp[0], pp[1], po))
            else:
                base_c["options_unparsed"] += 1
        else:
            base_c["unparsed"] += 1
        for m in re.finditer(r"V7-sgpqa:(?:GPQA|MMLU):([0-9a-f]{32})", r["decision_uid"]):
            sgpqa_used.add(m.group(1))
        assert r["decision_uid"] not in uids
        uids.add(r["decision_uid"])
        fb.write(l); base_c["rows"] += 1
        base_rows_by_src[r.get("source_family") or r.get("source")] += 1
        base_parent += r.get("parent_preservation") is not None
        base_w += float(r.get("weight", 1.0))
assert h_fs1.hexdigest() == "8461365da84aaf5e0f3528900b314755b080212d6ea2cda9ceb0f41111cad644", "FS1 input sha mismatch"
REP["base"] = {**base_c, "fs1_rows_sha256": h_fs1.hexdigest(), "removed": {f"{k[0]}::{k[1]}": v for k, v in sorted(removed.items())},
               "removed_total": sum(removed.values()), "kept_procedural": dict(kept_proc), "rows_with_parent": base_parent,
               "sgpqa_uuids_in_base": len(sgpqa_used)}
print("BASE", json.dumps(REP["base"])[:3000], flush=True)

# ------------------------------------------------------------------ gate
gate = BM.Gate(); g = collections.Counter()
suite_rows = []
for f in ("selected-rows", "added-rows", "gsm8k-rows"):
    for r in jl(f"{a.suite}/{f}.jsonl.gz"):
        gate.add(r); g["suite"] += 1
        fam = str(r.get("family") or r.get("benchmark") or r.get("id", "")).lower()
        if "habermas" in fam:
            suite_rows.append(("habermas", BM.norm_text((r.get("state") or {}).get("policy_question", ""))))
        elif "when2call" in fam or str(r.get("id", "")).startswith("62:"):
            st = r.get("state") or {}
            suite_rows.append(("when2call", BM.norm_text(st.get("question", "") if isinstance(st, dict) else "")))
HO = [f"{W}/gate/{x}" for x in ("dev.jsonl", "eval_fresh.jsonl", "cal.jsonl", "headfit.jsonl", "when2call_4opt_clean_dev.jsonl",
                                "official_dev_splits_eval.jsonl", "official_dev_splits_eval_v2.jsonl", "libeval_clean.jsonl")]
HO += sorted(__import__("glob").glob(f"{W}/gate/HO_*_v9.jsonl.gz")) + sorted(__import__("glob").glob(f"{W}/gate/v7/HO_*.jsonl"))
w2c_dev_q = set()
for path in HO:
    for r in jl(path):
        if r.get("manifest_header"):
            continue
        if "prompt" in r:
            pp = parse_prompt(r["prompt"])
            if not pp:
                g["heldout_unparsed"] += 1; continue
            if path.endswith("when2call_4opt_clean_dev.jsonl"):
                try:
                    w2c_dev_q.add(BM.norm_text(json.loads(pp[0]).get("question", "")))
                except Exception:
                    pass
            r = {"state": pp[0], "questions": {"q": {"instructions": pp[1]}}}
        gate.add(r); g[path.rsplit("/", 1)[-1]] += 1
for r in jl(f"{W}/gate/cov2_heldout_excl.jsonl.gz"):
    for t in [r.get("q")] + list(r.get("parts") or []):
        if t:
            gate.exact.add(BM.norm_text(t)); gate.grams |= BM.grams13(t); g["cov2_texts"] += 1
import pyarrow.parquet as pq
for p in sorted(__import__("glob").glob(f"{W}/src/mmlucf/**/*.parquet", recursive=True)):
    for r in pq.read_table(p).to_pylist():
        q = r.get("Question") or r.get("question") or ""
        opts = [str(r.get(k) or "") for k in ("A", "B", "C", "D")]
        gate.add({"state": "", "questions": {"q": {"instructions": q + "\n" + "\n".join(opts)}}})
        gate.exact.add(BM.norm_text(q)); g["mmlu_cf_val"] += 1
REP["gate"] = {**g, "exact": len(gate.exact), "grams13": len(gate.grams), "suite_ban_rows": dict(collections.Counter(x[0] for x in suite_rows))}
print("GATE", json.dumps(REP["gate"]), flush=True)
BAN_HAB_Q = {q for k, q in suite_rows if k == "habermas" and q}
BAN_W2C_Q = {q for k, q in suite_rows if k == "when2call" and q} | w2c_dev_q
for r in jl(f"{W}/src/when2call_test_mcq.jsonl"):        # test file: question ban only
    BAN_W2C_Q.add(BM.norm_text(r["question"]))
REP["bans"] = {"habermas_test_questions": len(BAN_HAB_Q), "when2call_test_or_dev_questions": len(BAN_W2C_Q)}

rng = random.Random(a.seed)
render = BM.make_render(tok, clm, rng, split="train", tag="fs5")
SUFFIX = BM.SUFFIX


def render_noul(uid, src, state, instructions, criteria, gold_bool):
    """noul question rendered as the engine renders RAGTruth / train_ragtruth: keys yes/no, texts Yes/No + definitions, fixed order."""
    unit = {"state": as_text(state), "instructions": as_text(instructions), "qtype": "noul", "option_keys": ["yes", "no"],
            "option_texts": ["Yes", "No"], "noul_criteria": criteria}
    res = clm.build_prompt(tok, unit, [0, 1], max_len=32768, noul_defs=True)
    prompt = res[0] if isinstance(res, tuple) else res
    assert prompt.endswith(SUFFIX)
    nt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    if nt > BM.MAX_TOKENS:
        return None
    g = "yes" if gold_bool else "no"
    return {"ancestry_component_id": int(hashlib.md5(uid.encode()).hexdigest()[:8], 16), "decision_uid": uid, "di_ancestry": False,
            "exposure_id": f"fs5:{uid}", "exposure_k": 0, "gold": {"yes": 1.0 if g == "yes" else 0.0, "no": 1.0 if g == "no" else 0.0},
            "kind": "noul", "loss_by_option": None, "n_tokens": nt, "numeric_family": False, "option_order": ["yes", "no"],
            "parent_id": None, "parent_preservation": None, "parent_route": "none", "parent_uid": uid.rsplit("|", 1)[0],
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "renderer_sha256": BM.R3,
            "source": src, "source_family": src, "split": "train", "target_basis": "hard_fact", "weight": 1.0}


POOL_ROWS = {}     # pool -> list of (row, meta)


def emit(pool, cands, line, licence, content_of, key_opts=False, balance=None):
    """cands: list of dicts {id, state, qk, q, gold, soft?, meta, noul?}; gate -> dedup -> quota (sha1(id) order, optional label balance)
    -> render. content_of(c) -> (content for 13-grams, exact keys)."""
    c = collections.Counter(); ok = []
    cap = CAPS[pool]
    golds = collections.defaultdict(set); local = set()
    for x in cands:                                    # same item twice in the source with different labels -> ambiguous, dropped
        st, ins = as_text(x["state"]), as_text(x["q"]["instructions"])
        opts = [as_text(v) for v in (x["q"].get("criteria") or {}).values()] if key_opts else None
        x["_key"] = key_of(st, ins, opts)
        golds[x["_key"]].add(json.dumps([x["gold"] if not x.get("noul") else bool(x["gold"]), x.get("soft")], sort_keys=True,
                                        default=str) if key_opts else json.dumps(x["q"]["criteria"].get(x["gold"]) if not x.get("noul") else bool(x["gold"])))
    for x in sorted(cands, key=lambda x: sha1(x["id"])):
        assert x["qk"] not in EXCLUDED_QIDS, x["qk"]
        k = x["_key"]
        if len(golds[k]) > 1:
            c["dropped_conflicting_label_in_source"] += 1; continue
        if k in local:
            c["dropped_dup_within_source"] += 1; continue
        local.add(k)
        if k in seen or (key_opts and k in seen_opt):
            c["dropped_dup_of_base_or_pool"] += 1; continue
        content, ekeys = content_of(x)
        if not gate.clean(content, exact_keys=ekeys):
            c["dropped_gate"] += 1
            if (x.get("meta") or {}).get("task_type"):
                c["dropped_gate::" + x["meta"]["task_type"]] += 1
            continue
        ok.append((x, k))
    c["eligible"] = len(ok)
    take = []
    if balance:
        lab = collections.Counter(balance(x) for x, _ in ok)
        quota = math.ceil(cap / max(1, len(lab)))
        lc = collections.Counter(); rest = []
        for x, k in ok:
            l = balance(x)
            if lc[l] < quota and len(take) < cap:
                lc[l] += 1; take.append((x, k))
            else:
                rest.append((x, k))
        take += rest[:max(0, cap - len(take))]
    else:
        take = ok[:cap]
    rows = []
    for x, k in take:
        uid = f"fs5|{pool}|{x['id']}"
        if x.get("noul"):
            row = render_noul(uid, f"fs5_{pool}", x["state"], x["q"]["instructions"], x["q"]["criteria"], x["gold"])
        else:
            row = render(uid, f"fs5_{pool}", x["state"], x["q"]["instructions"], x["q"]["criteria"], x["gold"], f"fs5_{pool}", soft=x.get("soft"))
        if row is None:
            c["dropped_over_4096_tokens"] += 1; continue
        if row["decision_uid"] in uids:
            c["dropped_dup_uid"] += 1; continue
        seen.add(k); uids.add(row["decision_uid"])
        row = {kk: row.get(kk) for kk in KEYS}
        rows.append((row, {"uid": uid, "pool": pool, "line": line, "licence": licence, **(x.get("meta") or {})}))
    c["kept"] = len(rows)
    gold_top = collections.Counter(max(r["gold"], key=r["gold"].get) if r["kind"] != "noul" else max(r["gold"], key=r["gold"].get) for r, _ in rows)
    REP["pools"][pool] = {"candidates": len(cands), **c, "line": line, "licence": licence, "gold_top": dict(gold_top.most_common(12)),
                          "k_hist": dict(collections.Counter(len(r["option_order"]) for r, _ in rows)),
                          "tokens_mean": round(sum(r["n_tokens"] for r, _ in rows) / max(1, len(rows)), 1),
                          "tokens": sum(r["n_tokens"] for r, _ in rows)}
    POOL_ROWS[pool] = rows
    with open(f"{W}/out/pools/fs5_{pool}.rows.jsonl", "w", encoding="utf-8") as f:
        for r, _ in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(f"{W}/out/pools/fs5_{pool}.meta.jsonl", "w", encoding="utf-8") as f:
        for _, m in rows:
            f.write(json.dumps({kk: vv for kk, vv in m.items() if not kk.startswith("_")}, ensure_ascii=False) + "\n")
    print("POOL", pool, json.dumps(REP["pools"][pool]), flush=True)


S = f"{W}/src"
# ------------------------------------------------------------------ VAST (official train split; kit normalize_text.vast twin)
labels = ["against", "favor", "neutral"]
cands = []
with open(f"{S}/vast_train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        lab = int(r["label"]); assert 0 <= lab < 3
        ins = f"Topic: {r['topic_str']}\nPost: {r['post']}\nDetermine the stance of the post toward the topic."
        cands.append({"id": f"VAST:train:{r['new_id']}", "state": {}, "qk": "q1", "q": {"type": "choice", "instructions": ins,
                      "criteria": dict(zip("ABC", labels))}, "gold": "ABC"[lab],
                      "meta": {"new_id": r["new_id"], "type_idx": r["type_idx"], "seen": r.get("seen?"), "_post": r["post"]}})
emit("vast", cands, "no-licence-stated", "states no licence (github.com/emilyallaway/zero-shot-stance @e7c47751: README has no licence; no LICENSE file)",
     lambda x: (x["meta"]["_post"], (x["q"]["instructions"],)), balance=lambda x: x["gold"])

# ------------------------------------------------------------------ ACOS (official train split; kit adapters_mechanical twin, one question per row)
def acos_rows(path):
    out = []
    for line in open(path, encoding="utf-8").read().splitlines():
        if not line:
            continue
        fields = line.rstrip("\n").split("\t"); pairs = set()
        for quad in fields[1:]:
            bits = quad.split()
            if len(bits) != 4:
                raise ValueError(quad)
            pairs.add((bits[1], bits[2]))
        out.append((fields[0], pairs))
    return out


SENT = {"0": "negative", "1": "neutral", "2": "positive"}
ACOS_INS = ("Given the review below, decide whether it expresses at least one aspect-category/sentiment pair ({c}, {s}) anywhere in the review. "
            "Answer yes only when that exact category and sentiment pair is present; answer no otherwise. Multiple opposite sentiments are separate valid questions.")
ACOS_CRIT = {"yes": "The exact category/sentiment pair is present.", "no": "The exact category/sentiment pair is absent."}
cands = []; ac = collections.Counter()
for dom, stem in (("laptop", "laptop"), ("rest16", "rest16")):
    tr = acos_rows(f"{S}/acos_{stem}_quad_train.tsv"); dv = acos_rows(f"{S}/acos_{stem}_quad_dev.tsv")
    cats = sorted({p[0] for _, ps in tr + dv for p in ps})              # inventory from train + dev (test file not read)
    rr = random.Random(f"{a.seed}:acos:{dom}")
    for i, (review, pairs) in enumerate(tr, 1):
        st = {"review": review, "task": "ACOS fixed category/sentiment presence"}
        picks = [(c_, s_, "yes") for c_, s_ in sorted(pairs)]
        hard = sorted({(c_, s2) for c_, _ in pairs for s2 in SENT if (c_, s2) not in pairs})
        if hard:
            picks.append(hard[rr.randrange(len(hard))] + ("no",))
        others = [c_ for c_ in cats if c_ not in {p[0] for p in pairs}]
        for c_ in rr.sample(others, min(2, len(others))):
            picks.append((c_, rr.choice(list(SENT)), "no"))
        for c_, s_, yn in picks:
            sname = SENT[s_]
            cands.append({"id": f"acos_{dom}:train:line-{i}|{c_}__sentiment_{sname}", "state": st, "qk": f"{c_}__sentiment_{sname}",
                          "q": {"type": "choice", "instructions": ACOS_INS.format(c=c_, s=sname), "criteria": ACOS_CRIT}, "gold": yn,
                          "meta": {"domain": dom, "src_line": i, "review_group": f"{dom}:{i}"}})
            ac[(dom, yn)] += 1
REP["acos_candidates"] = {f"{k[0]}/{k[1]}": v for k, v in ac.items()}
# quota by REVIEW (all questions of a review together), reviews in sha1 order
byrev = collections.defaultdict(list)
for x in cands:
    byrev[x["meta"]["review_group"]].append(x)
order = sorted(byrev, key=sha1); cand2 = []
for rg in order:
    cand2.extend(byrev[rg])
_cap = CAPS["acos"]; CAPS["acos"] = 10 ** 9
def _acos_content(x):
    return x["state"]["review"], (json.dumps(x["state"], ensure_ascii=False),)
# emit sorts by sha1(id); keep review grouping by giving ids a review-hash prefix
for x in cand2:
    x["id"] = f"{sha1(x['meta']['review_group'])[:10]}|{x['id']}"
emit("acos", cand2, "no-licence-stated", "states no licence (github.com/NUSTM/ACOS @45d179a3: README asks only for a citation; no LICENSE file)", _acos_content)
# apply the row cap by whole reviews after gating
rows = POOL_ROWS["acos"]; keep = []; n = 0; revs = collections.OrderedDict()
for r, m in rows:
    revs.setdefault(m["review_group"], []).append((r, m))
for rg in sorted(revs, key=sha1):
    if n + len(revs[rg]) > _cap:
        break
    keep.extend(revs[rg]); n += len(revs[rg])
CAPS["acos"] = _cap
dropped_cap = len(rows) - len(keep)
kept_uids = {r["decision_uid"] for r, _ in keep}
for r, m in rows:
    if r["decision_uid"] not in kept_uids:
        uids.discard(r["decision_uid"])
POOL_ROWS["acos"] = keep
REP["pools"]["acos"].update({"kept_before_review_cap": len(rows), "kept": len(keep), "reviews": len({m['review_group'] for _, m in keep}),
                             "gold_top": dict(collections.Counter(max(r["gold"], key=r["gold"].get) for r, _ in keep))})
with open(f"{W}/out/pools/fs5_acos.rows.jsonl", "w", encoding="utf-8") as f:
    for r, _ in keep:
        f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
with open(f"{W}/out/pools/fs5_acos.meta.jsonl", "w", encoding="utf-8") as f:
    for _, m in keep:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print("POOL acos (review cap)", json.dumps(REP["pools"]["acos"]), flush=True)

# ------------------------------------------------------------------ When2Call train_pref (authentic chosen vs rejected; kit when2call twin state/instructions)
W2C_INS = "Which response should the assistant give to the user's question, given the available tools?"


def w2c_text(msg):
    t = (msg or {}).get("content") or ""
    m = re.fullmatch(r"\s*<TOOLCALL>(.*)</TOOLCALL>\s*", t, re.S)
    if not m:
        return t.strip(), None
    try:
        calls = json.loads(m.group(1))
    except Exception:
        return None, "bad_toolcall_json"
    if not isinstance(calls, list) or not calls:
        return None, "bad_toolcall_json"
    if any(not c.get("arguments") for c in calls):
        return None, "zero_argument_toolcall"
    return (json.dumps(calls[0], ensure_ascii=False) if len(calls) == 1 else json.dumps(calls, ensure_ascii=False)), None


groups = collections.OrderedDict(); wc = collections.Counter()
for i, r in enumerate(jl(f"{S}/when2call_train_pref.jsonl")):
    users = [m for m in r["messages"] if m.get("role") == "user"]
    if len(r["messages"]) != 1 or len(users) != 1:
        wc["skip_multi_turn"] += 1; continue
    q = users[0]["content"]
    try:
        tools = [json.loads(t) for t in r["tools"]]
    except Exception:
        wc["skip_bad_tools"] += 1; continue
    k = (q, json.dumps(tools, sort_keys=True))
    groups.setdefault(k, []).append((i, r, tools))
cands = []
for (q, _), items in groups.items():
    if BM.norm_text(q) in BAN_W2C_Q:
        wc["banned_test_or_dev_question"] += 1; continue
    ch = {json.dumps(it[1]["chosen_response"], sort_keys=True) for it in items}
    if len(ch) != 1:
        wc["skip_inconsistent_chosen"] += 1; continue
    chosen, err = w2c_text(items[0][1]["chosen_response"])
    if chosen is None:
        wc[f"skip_chosen_{err}"] += 1; continue
    rej = []
    for it in items:
        t, err = w2c_text(it[1]["rejected_response"])
        if t is None:
            wc[f"skip_rejected_{err}"] += 1; continue
        if t and BM.norm_text(t) != BM.norm_text(chosen) and BM.norm_text(t) not in {BM.norm_text(x) for x in rej}:
            rej.append(t)
    if not rej or not chosen:
        wc["skip_no_distinct_rejected"] += 1; continue
    opts = [chosen] + rej[:3]
    keys = [chr(65 + j) for j in range(len(opts))]
    cands.append({"id": f"When2Call:train_pref:{items[0][0]}", "state": {"tools": items[0][2], "question": q}, "qk": "q",
                  "q": {"type": "choice", "instructions": W2C_INS, "criteria": dict(zip(keys, opts))}, "gold": "A",
                  "meta": {"pref_rows": [it[0] for it in items], "k": len(opts), "_question": q}})
REP["when2call_build"] = {**wc, "groups": len(groups), "candidates": len(cands), "k_hist": dict(collections.Counter(len(c["q"]["criteria"]) for c in cands))}
emit("when2call", cands, "clean", "CC BY 4.0 (nvidia/When2Call @0582f774 card: 'licensed under a Creative Commons Attribution 4.0 International License')",
     lambda x: (x["meta"]["_question"], (x["meta"]["_question"],)))

# ------------------------------------------------------------------ RAGTruth train responses (open-model responses only; kit ragtruth twin, noul)
RT_INS = "The response contains content that is not supported by the context in the prompt."
RT_CRIT = {"true": "The response contains content that is not supported by the context in the prompt.", "false": "All content of the response is supported by the context in the prompt."}
OPEN_MODELS = {"llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat", "mistral-7B-instruct"}
src_info = {x["source_id"]: x for x in jl(f"{S}/ragtruth_source_info.jsonl")}
resp = list(jl(f"{S}/ragtruth_response.jsonl"))
test_src = {r["source_id"] for r in resp if r["split"] == "test"}
cands = []; rc = collections.Counter()
for r in resp:
    if r["split"] != "train":
        continue
    if r["model"] not in OPEN_MODELS:
        rc["skip_closed_model_response"] += 1; continue
    if r["source_id"] in test_src:
        rc["banned_test_source"] += 1; continue
    s = src_info[r["source_id"]]
    si = s["source_info"]
    if s["task_type"] == "Data2txt" and isinstance(si, dict):      # Yelp hours / attribute fields are shared boilerplate, not item content
        si = {k: v for k, v in si.items() if k not in ("hours", "attributes")}
    cands.append({"id": f"RAGTruth:train:{r['id']}", "state": {"prompt": s["prompt"], "response": r["response"]}, "qk": "q",
                  "q": {"type": "noul", "instructions": RT_INS, "criteria": RT_CRIT}, "gold": len(r["labels"]) > 0, "noul": True,
                  "meta": {"source_id": r["source_id"], "task_type": s["task_type"], "source": s["source"], "model": r["model"],
                           "_content": r["response"] + "\n" + as_text(si)}})
REP["ragtruth_build"] = {**rc, "candidates": len(cands)}
emit("ragtruth", cands, "NC", "annotations MIT (github.com/ParticleMedia/RAGTruth @c103204b LICENSE); passages: MS MARCO (non-commercial research terms), Yelp Open Dataset (Yelp dataset terms), CNN/DM news -> NC line as train_ragtruth in FS1",
     lambda x: (x["meta"]["_content"], (as_text(x["state"]),)), balance=lambda x: (x["meta"]["task_type"], x["gold"]))
del resp, src_info
REP["pools"]["ragtruth"]["by_task"] = dict(collections.Counter(m["task_type"] for _, m in POOL_ROWS["ragtruth"]))
REP["pools"]["ragtruth"]["by_model"] = dict(collections.Counter(m["model"] for _, m in POOL_ROWS["ragtruth"]))

# ------------------------------------------------------------------ Habermas Machine TRAINING_DAT / question split TRAIN (kit habermas twin)
HAB_INS = ("Choose the consensus statement you predict this group would rank highest on average, given their expressed opinions. Each participant has equal weight. "
           "Assess the group's preferences, rather than your own policy preference.")
cols = ["metadata.version", "metadata.status", "round_id", "iteration_index", "question.split", "question.text", "rankings.metadata.status",
        "rankings.candidate_ids", "rankings.numerical_ranks", "candidates.metadata.id", "candidates.text", "own_opinion.metadata.id", "own_opinion.text",
        "other_opinions.metadata.id"]
raw = pq.read_table(f"{S}/hm_all_candidate_comparisons.parquet", columns=cols).to_pylist()
grp = collections.defaultdict(list); hc = collections.Counter()
for row in raw:
    if row["question.split"] != "TRAIN":
        hc["not_train_split"] += 1; continue
    if row["metadata.status"] != "COMPLETED" or row["rankings.metadata.status"] != "COMPLETED":
        hc["incomplete"] += 1; continue
    grp[(row["round_id"], row["iteration_index"], tuple(sorted(row["rankings.candidate_ids"] or [])))].append(row)
cands = []
for (rid, it, cids), panel in sorted(grp.items()):
    ref = panel[0]
    if BM.norm_text(ref["question.text"]) in BAN_HAB_Q:
        hc["banned_test_question"] += 1; continue
    expected = set(ref["other_opinions.metadata.id"]) | {ref["own_opinion.metadata.id"]}
    actual = [r["own_opinion.metadata.id"] for r in panel]
    if set(actual) != expected or len(actual) != len(set(actual)):
        hc["incomplete_panel"] += 1; continue
    if not 2 <= len(cids) <= 52:
        hc["candidate_count"] += 1; continue
    tmap = dict(zip(ref["candidates.metadata.id"], ref["candidates.text"]))
    if not all(isinstance(tmap.get(c_), str) and tmap[c_].strip() for c_ in cids) or len({" ".join(tmap[c_].split()) for c_ in cids}) != len(cids):
        hc["blank_or_duplicate_candidate"] += 1; continue
    sums = dict.fromkeys(cids, 0); ops = {}; bad = False
    for r in panel:
        if r["question.text"] != ref["question.text"] or dict(zip(r["candidates.metadata.id"], r["candidates.text"])) != tmap:
            bad = True; break
        ranks = dict(zip(r["rankings.candidate_ids"], r["rankings.numerical_ranks"]))
        if set(ranks) != set(cids):
            bad = True; break
        ops[r["own_opinion.metadata.id"]] = r["own_opinion.text"]
        for c_, v in ranks.items():
            sums[c_] += v
    if bad:
        hc["inconsistent_panel"] += 1; continue
    order = sorted(cids, key=lambda c_: hashlib.sha256((rid + c_).encode()).digest())
    keys = {c_: f"option_{i}" for i, c_ in enumerate(order)}
    best = [keys[c_] for c_ in order if sums[c_] == min(sums.values())]
    soft = {keys[c_]: (1.0 / len(best) if keys[c_] in best else 0.0) for c_ in order} if len(best) > 1 else None
    st = {"policy_question": ref["question.text"], "participant_opinions": [ops[k] for k in sorted(ops)]}
    cands.append({"id": f"Habermas:train:{rid}:it{it}", "state": st, "qk": "consensus",
                  "q": {"type": "choice", "instructions": HAB_INS, "criteria": {keys[c_]: tmap[c_] for c_ in order}}, "gold": best[0], "soft": soft,
                  "meta": {"round_id": rid, "iteration": it, "panel": len(panel), "tied_best": len(best),
                           "_content": "\n".join(st["participant_opinions"] + [tmap[c_] for c_ in order])}})
REP["habermas_build"] = {**hc, "panels": len(grp), "candidates": len(cands)}
emit("habermas", cands, "clean", "CC BY 4.0 data / Apache-2.0 code (github.com/google-deepmind/habermas_machine @7923b719 README 'All other materials are licensed under the Creative Commons Attribution 4.0'); parquet sha256 7cf8d5ce",
     lambda x: (x["meta"]["_content"], (as_text(x["state"]),)), key_opts=True)
del raw, grp

# ------------------------------------------------------------------ SuperGPQA (ODC-BY): new uuids only, GPQA shape (Science / Medicine / Engineering, middle+hard) and MMLU-Pro shape
GPQA_HO = set()
for r in jl(f"{W}/gate/v7/HO_sgpqa_v7.jsonl"):
    for m in re.finditer(r"([0-9a-f]{32})", json.dumps(r)):
        GPQA_HO.add(m.group(1))
sg = list(jl(f"{S}/SuperGPQA-all.jsonl")); sc = collections.Counter(); free = []
for x in sg:
    opts = x["options"] if isinstance(x["options"], list) else ast.literal_eval(x["options"])
    if x["uuid"] in sgpqa_used:
        sc["skip_uuid_in_base"] += 1; continue
    if x["uuid"] in GPQA_HO:
        sc["skip_uuid_heldout"] += 1; continue
    L = [chr(65 + i) for i in range(len(opts))]
    if x["answer_letter"] not in L or opts[L.index(x["answer_letter"])].strip() != str(x["answer"]).strip() or len({BM.norm_text(o) for o in opts}) != len(opts):
        sc["skip_bad_options_or_answer"] += 1; continue
    free.append((x, opts))
REP["sgpqa_build"] = {**sc, "free": len(free)}
gp, mp = [], []
for x, opts in free:
    (gp if x["discipline"] in ("Science", "Medicine") and x["difficulty"] in ("middle", "hard") else mp).append((x, opts))
cands = []
for x, opts in gp:
    rr = random.Random(f"{a.seed}:gpqa:{x['uuid']}")
    gi = ord(x["answer_letter"]) - 65
    dis = rr.sample([i for i in range(len(opts)) if i != gi], 3)
    pick = [gi] + dis; rr.shuffle(pick)
    crit = {chr(65 + j): opts[i] for j, i in enumerate(pick)}
    cands.append({"id": f"SuperGPQA:gpqa4:{x['uuid']}", "state": "", "qk": "answer", "q": {"type": "choice", "instructions": x["question"], "criteria": crit},
                  "gold": chr(65 + pick.index(gi)), "meta": {"uuid": x["uuid"], "discipline": x["discipline"], "field": x["field"], "difficulty": x["difficulty"]}})
emit("sgpqa_gpqa", cands, "clean", "ODC-BY 1.0 (m-a-p/SuperGPQA @4430d445 card); MMLU-Pro (suite) and MMLU-CF val text-match filtered",
     lambda x: (x["q"]["instructions"] + ("\n" + "\n".join(x["q"]["criteria"].values()) if len(x["q"]["instructions"].split()) < 13 else ""), (x["q"]["instructions"],)))
REP["pools"]["sgpqa_gpqa"]["by_discipline"] = dict(collections.Counter(m["discipline"] for _, m in POOL_ROWS["sgpqa_gpqa"]))
REP["pools"]["sgpqa_gpqa"]["by_difficulty"] = dict(collections.Counter(m["difficulty"] for _, m in POOL_ROWS["sgpqa_gpqa"]))
cands = []
for x, opts in mp:
    L = [chr(65 + i) for i in range(len(opts))]
    cands.append({"id": f"SuperGPQA:mmlupro:{x['uuid']}", "state": x["question"], "qk": "q", "q": {"type": "choice", "instructions": "Which option is the correct answer?",
                  "criteria": dict(zip(L, opts))}, "gold": x["answer_letter"], "meta": {"uuid": x["uuid"], "discipline": x["discipline"], "difficulty": x["difficulty"]}})
emit("sgpqa_mmlupro", cands, "clean", "ODC-BY 1.0 (m-a-p/SuperGPQA @4430d445 card); MMLU-Pro (suite) and MMLU-CF val text-match filtered",
     lambda x: (x["state"] + ("\n" + "\n".join(x["q"]["criteria"].values()) if len(x["state"].split()) < 13 else ""), (x["state"],)),
     balance=lambda x: x["meta"]["discipline"])
REP["pools"]["sgpqa_mmlupro"]["by_discipline"] = dict(collections.Counter(m["discipline"] for _, m in POOL_ROWS["sgpqa_mmlupro"]))

json.dump(REP, open(f"{W}/out/fs5_pools_report.json", "w"), indent=1)
sup = sum(len(v) for v in POOL_ROWS.values())
print("SUPPLEMENT", sup, {k: len(v) for k, v in POOL_ROWS.items()}, flush=True)
if a.only_pools:
    sys.exit(0)

# ------------------------------------------------------------------ assemble FS5 = base (FS1 order) + supplement (pool order)
OUT = f"{W}/out/esbar_stage1_fs5.jsonl"
h = hashlib.sha256(); n = 0; wsum = base_w; per_src = collections.Counter(base_rows_by_src)
with open(OUT, "w", encoding="utf-8") as fo:
    for l in open(TMP_BASE, encoding="utf-8"):
        fo.write(l); h.update(l.encode()); n += 1
    for pool in ("vast", "acos", "when2call", "ragtruth", "habermas", "sgpqa_gpqa", "sgpqa_mmlupro"):
        for r, _ in POOL_ROWS.get(pool, []):
            l = json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            fo.write(l); h.update(l.encode()); n += 1; wsum += float(r.get("weight", 1.0)); per_src[r["source_family"]] += 1
fs1h = json.load(open(FS1 + ".header.json"))
hdr = {"manifest_header": True, "schema_version": fs1h["schema_version"], "renderer_sha256": fs1h["renderer_sha256"], "rows": n, "rows_sha256": h.hexdigest(),
       "total_decision_weight": float(wsum),
       "parent_preservation": {**fs1h["parent_preservation"], "rows_with_parent": base_parent,
                               "note": "only FS1's v8 rows carry E1's parent targets; breadth_v1, pools_v4 and the FS5 supplement rows have none"},
       "fs5_derivation": {"fs1_rows": REP["base"]["rows"] + REP["base"]["removed_total"], "fs1_rows_sha256": fs1h["rows_sha256"],
                          "removed_procedural_and_synth_workflows": REP["base"]["removed_total"], "kept_procedural": REP["base"]["kept_procedural"],
                          "base_rows": REP["base"]["rows"], "supplement_rows": sup, "supplement_by_pool": {k: len(v) for k, v in POOL_ROWS.items()},
                          "steps": -(-n // 64), "builder": "fs5_build.py"}}
assert fs1h["renderer_sha256"] == BM.R3
json.dump(hdr, open(OUT + ".header.json", "w"), indent=2)
REP["total"] = {"rows": n, "rows_sha256": hdr["rows_sha256"], "steps": -(-n // 64), "total_decision_weight": wsum, "per_source_family": dict(sorted(per_src.items()))}
json.dump(REP, open(OUT + ".report.json", "w"), indent=1)
print("TOTAL", json.dumps({k: v for k, v in REP["total"].items() if k != "per_source_family"}), flush=True)
