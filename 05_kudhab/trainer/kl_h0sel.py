#!/usr/bin/env python3
"""kl_h0sel.py MANIFEST.jsonl H0_REQUESTS.jsonl.gz OUT_IDS.txt OUT_REPORT.json NAME:COUNT[,NAME:COUNT..] (KL agent, 11-16): the exposure ids of the H0 rows
(ids in H0_REQUESTS) whose manifest source is one of NAME, for kl_data.py --exclude-ids. The field is the one (source or source_family) on which EVERY name's
count equals the expected COUNT exactly; if no field gives exactly the expected counts, it stops (no silent mismatch)."""
import gzip, json, sys, collections
man, h0f, out_ids, out_rep, spec = sys.argv[1:6]
want = {kv.rsplit(":", 1)[0]: int(kv.rsplit(":", 1)[1]) for kv in spec.split(",") if kv}
h0 = set(json.loads(l)["id"] for l in gzip.open(h0f, "rt"))
src = {}
for l in open(man):
    r = json.loads(l)
    if r.get("manifest_header"): continue
    if r["exposure_id"] in h0: src[r["exposure_id"]] = (r.get("source"), r.get("source_family"))
assert len(src) == len(h0), f"{len(h0) - len(src)} H0 ids not in the manifest"
rep = {"h0_rows": len(h0), "expected": want}
for fi, field in enumerate(("source", "source_family")):
    c = collections.Counter(v[fi] for v in src.values())
    got = {n: c.get(n, 0) for n in want}; rep[f"counts_by_{field}"] = got
    if got == want:
        ids = sorted(i for i, v in src.items() if v[fi] in want)
        open(out_ids, "w").write("\n".join(ids) + "\n")
        kept = collections.Counter(v[fi] for i, v in src.items() if v[fi] not in want)
        rep.update({"field": field, "excluded": len(ids), "kept_h0_rows": len(h0) - len(ids), "kept_by_" + field: dict(kept.most_common())})
        json.dump(rep, open(out_rep, "w"), indent=1); print("KLH0SEL_DONE", json.dumps(rep)[:900]); sys.exit(0)
json.dump(rep, open(out_rep, "w"), indent=1); print("KLH0SEL_FAILED no field gives the expected counts", json.dumps(rep)); sys.exit(1)
