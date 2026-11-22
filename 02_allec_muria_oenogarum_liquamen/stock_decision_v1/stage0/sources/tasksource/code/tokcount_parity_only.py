#!/usr/bin/env python3
"""(stock_decision_v1) finish a tokcount run whose main pass completed (OUT.partial holds the tokens table, summary, inputs) but whose
parity step crashed: re-stage the same input, run tokcount.parity on OUT.partial, write parity_<pool>.json, rename to OUT."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokcount as T

ap = argparse.ArgumentParser()
for k in ("--tokenizer", "--code", "--input", "--pool", "--work", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--md5", default="")
ap.add_argument("--splits", default="train")
ap.add_argument("--lines-per-chunk", type=int, default=4000)
ap.add_argument("--parity-every", type=int, default=300)
a = ap.parse_args()
a.splits = [s for s in a.splits.split(",") if s]
tmp = Path(a.out + ".partial")
assert (tmp / f"units_tokens_{a.pool}.jsonl.gz").exists() and not Path(a.out).exists()
prod = T.Prod(a.code)
files, info = T.stage_input(a, Path(a.work))
par = T.parity(a, files, tmp, prod)
(tmp / f"parity_{a.pool}.json").write_text(json.dumps(par, indent=1) + "\n")
print("PARITY", json.dumps({k: v for k, v in par.items() if k != "mismatch_examples"}), flush=True)
os.replace(tmp, a.out)
print("DONE_PARITY_ONLY", par["pass"])
