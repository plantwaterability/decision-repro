#!/usr/bin/env python3
# breadth_v1 step 1: per-source candidate rows from tasksource-jev-typed-decisions @8df2e5c9 (train split, base variants only).
# clean line <- filtered-full (commercial-only, reviewed); NC line <- full rows with license_use == non-commercial.
import os, re, json, glob, hashlib, heapq, collections
import pyarrow.parquet as pq
D = "/data/local/breadth_v1"; RAW = D + "/ts_raw"; OUT = D + "/cand"
CAP = 1500
PRI = {"direct": 0, "instruction_paraphrase": 1, "criteria_permutation": 2, "paired_text_format": 3}
BLOCK = ("procedural-typed-decisions", "typed-decisions")
def slug(s): return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_")
stats = collections.defaultdict(collections.Counter)
best = {}      # (line, source, group_id) -> (pri, file_idx, row_idx)
cols = ["source", "kind", "variant", "split", "group_id", "license_use"]
files = {"clean": sorted(glob.glob(RAW + "/filtered-full/train-*.parquet")), "nc": sorted(glob.glob(RAW + "/full/train-*.parquet"))}
for line, fl in files.items():
    for fi, f in enumerate(fl):
        t = pq.read_table(f, columns=cols).to_pydict()
        for ri in range(len(t["source"])):
            s, k, v, sp, g, lu = (t[c][ri] for c in cols)
            stats[(line, s)]["rows_all"] += 1
            if line == "nc":
                stats[("full", s)]["lu_" + str(lu)] += 1
                if lu != "non-commercial": continue
            elif lu != "commercial": stats[(line, s)]["not_commercial"] += 1; continue
            if sp != "train" or s.startswith(BLOCK) or k not in ("choice", "score") or v not in PRI: continue
            key = (line, s, g); p = (PRI[v], fi, ri)
            if key not in best or p < best[key]: best[key] = p
        print(line, fi, len(best), flush=True)
per = collections.defaultdict(list)
for (line, s, g), p in best.items():
    per[(line, s)].append((hashlib.md5(g.encode()).hexdigest(), p))
want = collections.defaultdict(dict)   # (line, fi) -> {ri: (line, s)}
for (line, s), l in per.items():
    stats[(line, s)]["groups"] = len(l)
    for _, (pr, fi, ri) in heapq.nsmallest(CAP, l):
        want[(line, fi)][ri] = s
outs = {}
for (line, fi), m in sorted(want.items()):
    t = pq.read_table(files[line][fi])
    idx = sorted(m)
    rows = t.take(idx).to_pylist()
    for ri, r in zip(idx, rows):
        s = m[ri]; p = f"{OUT}/{line}/{slug(s)}.jsonl"
        if p not in outs:
            os.makedirs(os.path.dirname(p), exist_ok=True); outs[p] = open(p, "w")
        outs[p].write(json.dumps(r, ensure_ascii=False) + "\n"); stats[(line, s)]["cand"] += 1
    print("wrote", line, fi, len(idx), flush=True)
for f in outs.values(): f.close()
json.dump({f"{a}|{b}": dict(c) for (a, b), c in stats.items()}, open(OUT + "/sample_stats.json", "w"), indent=0)
print("SAMPLE_DONE", len(outs), flush=True)
