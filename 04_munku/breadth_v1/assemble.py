#!/usr/bin/env python3
# breadth_v1: per-line manifests from the built pools + summary (rows, pools, families, option counts, md5).
import os, sys, json, glob, hashlib, collections
D = "/data/local/breadth_v1"; DEL = D + "/deliver"
os.makedirs(DEL + "/manifests", exist_ok=True)
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
summ = {"lines": {}, "sources": {}}
for root in sys.argv[1:]:
    for ld in sorted(glob.glob(root + "/*/pools")):
        line = ld.split("/")[-2]
        for kind, sub, suf in (("rows", "pools", ".rows.jsonl"), ("di", "di_rows", ".jsonl")):
            outp = f"{DEL}/manifests/breadth_v1_{line}.{kind}.jsonl"; n = 0
            fams = set(); nopt = collections.Counter(); kinds = collections.Counter(); uids = set()
            with open(outp, "w") as o:
                for f in sorted(glob.glob(f"{root}/{line}/{sub}/*{suf}")):
                    k = 0
                    for l in open(f):
                        o.write(l); n += 1; k += 1
                        if kind == "rows":
                            r = json.loads(l); m = r.get("bv1_meta", {})
                            fams.add(r.get("source_family")); nopt[m.get("n_options")] += 1; kinds[m.get("kind")] += 1
                            uids.add(r.get("decision_uid") or r.get("uid"))
                            s = summ["sources"].setdefault(m.get("source"), {"line": line, "rows": 0, "n_options": collections.Counter()})
                            s["rows"] += 1; s["n_options"][m.get("n_options")] += 1
            e = summ["lines"].setdefault(line, {})
            e[kind] = {"path": os.path.basename(outp), "rows": n, "md5": md5(outp)}
            if kind == "rows":
                e.update(pools=len(glob.glob(f"{root}/{line}/pools/*.rows.jsonl")), families=len(fams), unique_uids=len(uids),
                         n_options={str(a): b for a, b in sorted(nopt.items(), key=lambda x: (x[0] is None, x[0]))}, kinds=dict(kinds))
for s in summ["sources"].values(): s["n_options"] = {str(a): b for a, b in sorted(s["n_options"].items(), key=lambda x: (x[0] is None, x[0]))}
json.dump(summ, open(DEL + "/build_summary.json", "w"), indent=1)
with open(DEL + "/built_rows.tsv", "w") as o:
    o.write("source\tline\trows\tn_options\n")
    for k, v in sorted(summ["sources"].items(), key=lambda x: (x[1]["line"], str(x[0]))): o.write(f"{k}\t{v['line']}\t{v['rows']}\t{json.dumps(v['n_options'])}\n")
for line, e in summ["lines"].items(): print(line, json.dumps(e))
print("ASSEMBLE_DONE", sum(e["rows"]["rows"] for e in summ["lines"].values()))
