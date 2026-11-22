#!/usr/bin/env python3
"""Held-out DI-format rows from the program library's EVAL split (groups excluded from Stage-1 v2 training): one row per problem
(the K=4 variant when there is one), a seeded sample of up to N problems per pool, formulas only from problems not seen by final v1.
Pool rows are already DI-format; the engine eval reads them as the held-out pool files.

  python3 make_library_heldout.py --pools DIR --eval-manifest library_eval_manifest.jsonl --out FILE [--per-pool 500]
"""
import argparse
import hashlib
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-pool", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261102)
    a = ap.parse_args()
    meta = {}
    for l in open(a.eval_manifest):
        r = json.loads(l)
        meta[r["group"]] = r
    rng = random.Random(a.seed)
    out, counts = [], {}
    for name in ("arith_gsm8k", "arith_aqua", "formulas", "stem_mds", "stem_math"):
        by = {}
        for l in open(os.path.join(a.pools, name + ".jsonl")):
            r = json.loads(l)
            g = r.get("group") or r["id"]
            m = meta.get(g)
            if m is None or m.get("seen_by_final_v1"):
                continue
            by.setdefault(g, []).append(r)
        gs = sorted(by)
        rng.shuffle(gs)
        for g in gs[:a.per_pool]:
            rows = by[g]
            k4 = [r for r in rows if len(r["questions"]["answer"]["criteria"]) == 4]
            r = dict((k4 or rows)[0])
            r["metadata"] = dict(r.get("metadata") or {}, track=f"libeval_{name}", split_key=meta[g].get("split_key"))
            out.append(r)
        counts[name] = min(len(gs), a.per_pool)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    open(a.out, "w").write(body)
    print(json.dumps({"rows": len(out), "per_pool": counts, "sha256": hashlib.sha256(body.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
