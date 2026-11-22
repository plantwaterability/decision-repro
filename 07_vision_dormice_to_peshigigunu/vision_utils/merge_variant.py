"""Make a full-sample run dir from a base run with some rows replaced by a variant run (e.g. readout rows answered without
the 'The answer is (' prefix).  usage: python merge_variant.py <base_run> <variant_run> <out_run>
"""
import json
import shutil
import sys
from pathlib import Path

base, var, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
v = {json.loads(l)["run_id"]: l for l in open(var / "results.jsonl")}
n = 0
with open(out / "results.jsonl", "w") as f:
    for l in open(base / "results.jsonl"):
        rid = json.loads(l)["run_id"]
        if rid in v:
            l = v[rid]
            n += 1
        f.write(l if l.endswith("\n") else l + "\n")
shutil.copy(base / "environment.json", out / "environment.json")
print(json.dumps({"replaced": n, "variant_rows": len(v)}))
