#!/usr/bin/env python3
"""breadth_v1 pools: per-source small quotas from tasksource-jev-typed-decisions @8df2e5c9 (train split, base rows only; no packed /
verification / procedural typed-decision variants), in DI multiple-choice format with the source's native option count, rendered
with build_manifest_v3.make_render (renderer R3, as every v7.x / v8 pool) and gated with build_manifest_v3.Gate (exact state +
13-grams + short items) against the DI 0.3 public suite and our internal held-out sets; exact-state dedupe against the v8 manifest.

  python3 build.py --kit KIT --code CODE --scripts SCRIPTS --tok TOK --suite SUITE --heldout F1,F2 --v8 MANIFEST --cand CAND --decisions D.json --out OUT
Writes OUT/<line>/pools/<pool>.rows.jsonl (rendered), OUT/<line>/di_rows/<pool>.jsonl (DI rows), OUT/pools_report.json."""
import argparse, collections, gzip, hashlib, json, math, os, random, re, sys

ap = argparse.ArgumentParser()
for k in ("kit", "code", "scripts", "tok", "suite", "heldout", "v8", "cand", "decisions", "out"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--heldout_exact", default="")            # large proxy sets (GDE-1): exact normalised state only, no 13-grams
ap.add_argument("--seed", type=int, default=20261110)
ap.add_argument("--only", default="")
a = ap.parse_args()
sys.path[:0] = [a.kit, a.code, a.scripts]
import build_manifest_v3 as BM
import claude_lmscore as clm
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.tok)
REPORT = collections.defaultdict(dict)
def slug(s): return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_")
def jl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)
def parse_prompt(p):
    a0 = p.find("<|turn>user\n"); a0 = a0 + len("<|turn>user\n") if a0 >= 0 else 0
    io_ = p.rfind("\n\nOptions:\n"); iq = p.rfind("\n\nQuestion: ", 0, io_) if io_ >= 0 else -1
    if io_ < 0 or iq < a0:                                 # other renderers: the whole user turn is the content
        e = p.find("<turn|>", a0)
        return p[a0:e if e >= 0 else len(p)], ""
    return p[a0:iq], p[iq + len("\n\nQuestion: "):io_]

gate = BM.Gate(); n_suite = 0
for f in ("selected-rows", "added-rows", "gsm8k-rows"):
    for r in jl(f"{a.suite}/{f}.jsonl.gz"):
        gate.add(r); n_suite += 1
n_ho = collections.Counter()
for path in [p for p in a.heldout.split(",") if p]:
    for r in jl(path):
        if r.get("manifest_header"):
            continue
        if "prompt" in r:
            st, ins = parse_prompt(r["prompt"]); r = {"state": st, "questions": {"q": {"instructions": ins}}}
        gate.add(r); n_ho[path.rsplit("/", 1)[-1]] += 1
n_hx = 0
for path in [p for p in a.heldout_exact.split(",") if p]:
    for r in jl(path):
        st = r.get("state"); s = BM.norm_text(st if isinstance(st, str) else json.dumps(st, ensure_ascii=False))
        if len(s) >= 25:
            gate.exact.add(s); n_hx += 1
v8 = set(); n_v8 = 0
for r in jl(a.v8):
    if "prompt" not in r:
        continue
    try:
        st, _ = parse_prompt(r["prompt"])
    except ValueError:
        continue
    s = BM.norm_text(st); n_v8 += 1
    if len(s) >= 25:
        v8.add(hashlib.md5(s.encode()).digest())
REPORT["gate"] = {"suite_rows": n_suite, "heldout_rows": dict(n_ho), "heldout_exact_rows": n_hx, "exact_keys": len(gate.exact), "grams13": len(gate.grams), "v8_rows": n_v8, "v8_states": len(v8)}
print("gate", json.dumps(REPORT["gate"]), flush=True)

DEC = json.load(open(a.decisions))["sources"]
ONLY = set(x for x in a.only.split(",") if x)
# a tasksource question used by >= 4 source families is a generic wrapper, not the task's own wording: replaced by neutral wording
qsrc = collections.defaultdict(set); CAND = {}
for src, d in DEC.items():
    p = f"{a.cand}/{d.get('cand_line', d['line'])}/{slug(src)}.jsonl"
    if not os.path.exists(p):
        REPORT[src]["status"] = "no_candidate_rows"; continue
    CAND[src] = list(jl(p))
    for r in CAND[src]:
        qsrc[(r.get("question") or "").strip()].add(d["family"])
GENERIC = {q for q, s in qsrc.items() if len(s) >= 4}
REPORT["generic_questions"] = sorted(GENERIC)[:200]
NEUTRAL = ["Which option fits the text above?", "Select the option that applies.", "Which of the options is correct?", "Pick the option that best matches.",
           "Which label applies here?", "Choose the correct option.", "Which option describes this best?", "Decide which option applies."]
def keys_for(n): return [chr(97 + j) if j < 26 else f"option_{j + 1}" for j in range(n)]
rng = random.Random(a.seed)
render = BM.make_render(tok, clm, rng, split="train", tag="breadth_v1")
seen_global = set()
for src in sorted(CAND):
    d = DEC[src]; pool = "bv1_" + slug(src)
    if ONLY and src not in ONLY:
        continue
    c = collections.Counter(); items = []; seen = set()
    rows = sorted(CAND[src], key=lambda r: hashlib.md5(str(r["group_id"]).encode()).hexdigest())
    for r in rows:
        opts = [str(o).strip() for o in (r.get("options") or [])]; tgt = list(r.get("target") or [])
        st = (r.get("state") or "").strip()
        if r.get("split") != "train" or r.get("kind") not in ("choice", "score"): c["drop_not_train_choice"] += 1; continue
        if len(opts) < 2 or len(opts) != len(tgt) or not all(opts) or len({BM.norm_text(o) for o in opts}) != len(opts): c["drop_bad_options"] += 1; continue
        if not (5 <= len(st) <= 12000): c["drop_state_length"] += 1; continue
        top = sorted(tgt, reverse=True)
        if top[0] - top[1] < 0.2: c["drop_ambiguous_label"] += 1; continue
        gi = tgt.index(top[0]); ns = BM.norm_text(st); nk = " ".join(st.lower().split())
        q = (r.get("question") or "").strip()
        k = (nk, q if q not in GENERIC else "", tuple(sorted(opts)))
        if k in seen or (src.split("/")[0], k) in seen_global: c["drop_duplicate"] += 1; continue
        if len(ns) >= 25 and hashlib.md5(ns.encode()).digest() in v8: c["drop_in_v8"] += 1; continue
        if not gate.clean(st + ("\n" + "\n".join(opts) if sum(map(len, opts)) > 200 else ""), exact_keys=(st,)): c["drop_gate"] += 1; continue
        seen.add(k); items.append((r, opts, gi, q))
    quota = int(d["quota"]); fixed = len(set(tuple(sorted(x[1])) for x in items[:50])) == 1      # fixed label set (classification) vs per-row options
    labels = collections.Counter((o[gi] if fixed else gi) for r, o, gi, q in items)
    cap = math.ceil(quota / max(1, len(labels))); took, rest, lc = [], [], collections.Counter()
    for it in items:
        lab = it[1][it[2]] if fixed else it[2]
        if lc[lab] < cap and len(took) < quota: lc[lab] += 1; took.append(it)
        else: rest.append(it)
    took += rest[:max(0, quota - len(took))]
    out, di = [], []
    for r, opts, gi, q in took:
        order = list(range(len(opts)))
        if r["kind"] == "choice": rng.shuffle(order)          # score levels keep their order (ordered scale)
        keys = keys_for(len(opts)); crit = {keys[j]: opts[o] for j, o in enumerate(order)}; gold = keys[order.index(gi)]
        ins = q if q and q not in GENERIC else rng.choice(NEUTRAL)
        rid = hashlib.md5(str(r["id"]).encode()).hexdigest()[:16]
        row = render(f"bv1|{pool}|{rid}", pool, r["state"].strip(), ins, crit, gold, pool)
        if row is None: c["drop_over_4096_tokens"] += 1; continue
        row["source_family"] = "bv1_" + slug(d["family"])
        row["bv1_meta"] = {"source": src, "ts_id": r["id"], "line": d["line"], "tier": d["tier"], "n_options": len(opts), "kind": r["kind"], "own_question": bool(q and q not in GENERIC)}
        out.append(row); di.append({"id": f"{pool}:{rid}", "state": r["state"].strip(), "questions": {"q": {"type": "choice", "instructions": ins, "criteria": crit}}, "expected": {"q": gold}})
        seen_global.add((src.split("/")[0], (" ".join(r["state"].strip().lower().split()), q if q not in GENERIC else "", tuple(sorted(opts)))))
    for sub, data in (("pools", out), ("di_rows", di)):
        os.makedirs(f"{a.out}/{d['line']}/{sub}", exist_ok=True)
        with open(f"{a.out}/{d['line']}/{sub}/{pool}" + (".rows.jsonl" if sub == "pools" else ".jsonl"), "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    REPORT[src].update({"pool": pool, "line": d["line"], "tier": d["tier"], "family": d["family"], "quota": quota, "candidates": len(rows), "eligible": len(items), "kept": len(out), **c,
                        "n_options": dict(collections.Counter(x["bv1_meta"]["n_options"] for x in out)), "gold_top": collections.Counter(max(x["gold"], key=x["gold"].get) for x in out).most_common(3),
                        "tokens_mean": round(sum(x["n_tokens"] for x in out) / max(1, len(out)), 1)})
    print(pool, json.dumps(REPORT[src]), flush=True)
json.dump(REPORT, open(f"{a.out}/pools_report.json", "w"), indent=1)
print("BUILD_DONE", sum(v.get("kept", 0) for v in REPORT.values() if isinstance(v, dict)), flush=True)
