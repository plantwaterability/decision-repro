#!/usr/bin/env python3
"""REVIEW_2026-10-13_2105 finding 4 (schema_repro), integration form: REAL rendered Stage 0 discovery rows -> the teacher failure
ledger (stock_decision_v1/teacher/sdv1_failure_ledger.py) with FABRICATED predictions (every model puts 0.9 on a wrong key).
Checks: every row loads (gold_of), label fields follow the shared schema (target_basis / label_verification / label_provenance
are distinct fields; label_verification is a schema key), and the category depends on the verification tier:
independent or source_gold tier + hard gold -> common_teacher_failure (with needs_independent_check), unverified tier ->
ambiguous_or_defective_label, soft gold -> soft_target_metric_only. Nothing real is scored.
  python3 test_schema_integration.py DECISIONS_discovery.jsonl.gz TEACHER_DIR SCHEMA.json"""
import collections
import gzip
import json
import sys

dec_path, teacher_dir, schema_path = sys.argv[1:4]
sys.path.insert(0, teacher_dir)
import sdv1_failure_ledger as fl  # noqa: E402

S = json.load(open(schema_path))
rows, seen = [], collections.Counter()
with gzip.open(dec_path, "rt") as g:
    for line in g:
        r = json.loads(line)
        k = (r["label_provenance"], r["label_uncertainty"]["hard"])
        if seen[k] < 5:
            seen[k] += 1
            rows.append(r)
fails, cats = [], collections.Counter()
for r in rows:
    if r["label_verification"] not in S["label_verification"] or r["target_basis"] not in S["target_basis"]:
        fails.append(f"{r['decision_uid']}: fields not schema keys")
    pm = S["stage0_provenance_map"][r["label_provenance"]]
    want_tb = pm["target_basis"] if r.get("target_basis_source", "").startswith("stage0_provenance_map") else r["target_basis"]
    if r["label_verification"] != pm["label_verification"] or r["target_basis"] != want_tb:
        fails.append(f"{r['decision_uid']}: semantic fields are not the audited conversion of {r['label_provenance']}")
    g = fl.gold_of(r)
    wrong = min(g["gold"], key=lambda k: (g["gold"][k], k))
    preds = [dict(decision_uid=g["decision_uid"], teacher_id=m, transform="base", status="ok",
                  probabilities={k: (0.9 if k == wrong else 0.1 / (len(g["keys"]) - 1)) for k in g["keys"]},
                  prediction_sha256="fabricated", n_input_tokens=5) for m in ["stock", "garas", "d3", "pplx", "olmo"]]
    out = fl.ledger_row(g, preds)
    tier = S["label_verification"][r["label_verification"]]["tier"]
    tbs = S["target_basis"][r["target_basis"]]["scoring"]
    sc = pm["scoring"] if pm["target_basis"] == r["target_basis"] else tbs
    soft = sc == "soft" or (sc == "from_gold" and not r["label_uncertainty"]["hard"])
    want = "soft_target_metric_only" if soft else ("ambiguous_or_defective_label" if tier == "unverified" else "common_teacher_failure")
    cats[(r["label_provenance"], r["label_verification"], tier, "soft" if soft else "hard", out["category"])] += 1
    if out["category"] != want:
        fails.append(f"{r['decision_uid']} {r['label_provenance']}/{r['label_verification']} ({tier}, {'soft' if soft else 'hard'}): {out['category']} != {want}")
for k, v in sorted(cats.items()):
    print(v, k)
print("rows", len(rows), "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails[:5]}")
sys.exit(1 if fails else 0)
