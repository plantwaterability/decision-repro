#!/usr/bin/env python3
# breadth_v1 wave 1b: candidate rows for sources whose licence was read at the original repository (decisions_src.json).
# Rows come from the `full` config @8df2e5c9 (train split, base variants only); output dirs cand/clean_src and cand/nc_src.
import os, re, sys, json, glob, hashlib, heapq, collections
import pyarrow.parquet as pq
D = "/data/local/breadth_v1"; RAW = D + "/ts_raw"; OUT = D + "/cand"
CAP = 1500
PRI = {"direct": 0, "instruction_paraphrase": 1, "criteria_permutation": 2, "paired_text_format": 3}
def slug(s): return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_")
dec = json.load(open(sys.argv[1]))["sources"]
stats = collections.defaultdict(collections.Counter)
best = {}
cols = ["source", "kind", "variant", "split", "group_id", "license_use"]
files = sorted(glob.glob(RAW + "/full/train-*.parquet"))
for fi, f in enumerate(files):
    t = pq.read_table(f, columns=cols).to_pydict()
    for ri in range(len(t["source"])):
        s, k, v, sp, g, lu = (t[c][ri] for c in cols)
        if s not in dec: continue
        stats[s]["rows_all"] += 1; stats[s]["lu_" + str(lu)] += 1
        if sp != "train" or k not in ("choice", "score") or v not in PRI: continue
        key = (s, g); p = (PRI[v], fi, ri)
        if key not in best or p < best[key]: best[key] = p
    print(fi, len(best), flush=True)
per = collections.defaultdict(list)
for (s, g), p in best.items(): per[s].append((hashlib.md5(g.encode()).hexdigest(), p))
want = collections.defaultdict(dict)
for s, l in per.items():
    stats[s]["groups"] = len(l)
    for _, (pr, fi, ri) in heapq.nsmallest(CAP, l): want[fi][ri] = s
outs = {}
for fi, m in sorted(want.items()):
    t = pq.read_table(files[fi]); idx = sorted(m); rows = t.take(idx).to_pylist(); del t
    for ri, r in zip(idx, rows):
        s = m[ri]; p = f"{OUT}/{dec[s]['cand_line']}/{slug(s)}.jsonl"
        if p not in outs:
            os.makedirs(os.path.dirname(p), exist_ok=True); outs[p] = open(p, "w")
        outs[p].write(json.dumps(r, ensure_ascii=False) + "\n"); stats[s]["cand"] += 1
    print("wrote", fi, len(idx), flush=True)
for f in outs.values(): f.close()
json.dump({a: dict(c) for a, c in stats.items()}, open(OUT + "/sample_src_stats.json", "w"), indent=0)
print("SAMPLE_SRC_DONE", len(outs), sorted(set(dec) - set(stats)), flush=True)
