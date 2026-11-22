#!/usr/bin/env python3
"""FS5 manifest check with the trainer's own code (esbar_code_v2e, fullft_train md5 4048f97c): sha256_file == header rows_sha256 (the
--expect-rows-sha256 logic), read_manifest passes (renderer, prompt sha, option order, gold distribution, parent fields, weights),
steps = ceil(rows / 64), n_tokens / K ranges against FS1's."""
import collections, hashlib, json, math, sys
W = sys.argv[1] if len(sys.argv) > 1 else "/data/shared/fs5"
C = f"{W}/code/esbar_code_v2e"
sys.path[:0] = [f"{C}/trainers", f"{C}/engine", C]
assert hashlib.md5(open(f"{C}/trainers/fullft_train.py", "rb").read()).hexdigest().startswith("4048f97c")
import fullft_train as ft
M = f"{W}/out/esbar_stage1_fs5.jsonl"
hdr = json.load(open(M + ".header.json"))
got = ft.sha256_file(M)
assert got == hdr["rows_sha256"], (got, hdr["rows_sha256"])
header, rows = ft.read_manifest(M)
assert len(rows) == hdr["rows"]
tw = sum(float(r.get("weight", 1.0)) for r in rows)
out = {"READ_MANIFEST_OK": len(rows), "sha256_file": got, "header_rows_sha256": hdr["rows_sha256"], "steps": math.ceil(len(rows) / 64),
       "total_decision_weight_recomputed": tw, "header_total_decision_weight": hdr["total_decision_weight"],
       "max_n_tokens": max(int(r["n_tokens"]) for r in rows), "k_min": min(len(r["option_order"]) for r in rows), "k_max": max(len(r["option_order"]) for r in rows),
       "kinds": dict(collections.Counter(r["kind"] for r in rows)), "tokens": sum(int(r["n_tokens"]) for r in rows),
       "uids_unique": len({r["decision_uid"] for r in rows}) == len(rows), "rows_with_parent": sum(r.get("parent_preservation") is not None for r in rows),
       "procedural_left": dict(collections.Counter(str(r.get("source_family")) for r in rows if str(r.get("source_family") or r.get("source") or "").startswith(("procedural_typed_decisions/", "synth_workflows/")))),
       "fs5_rows": sum(str(r.get("source_family")).startswith("fs5_") for r in rows)}
json.dump(out, open(M + ".validate.json", "w"), indent=1)
print("VALIDATE", json.dumps(out))
