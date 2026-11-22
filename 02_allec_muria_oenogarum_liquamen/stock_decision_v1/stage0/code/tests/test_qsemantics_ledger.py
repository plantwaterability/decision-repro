#!/usr/bin/env python3
"""REVIEW_2026-10-13_2300 finding 2, integration form: from REAL rendered Stage 0 decision rows, pick, for each Bayesian generator
that declares both kinds, one belief / probability question and one ACTION question from the SAME generator, and pass both through
the teacher failure ledger with FABRICATED all-wrong predictions (0.9 on the lowest-gold key). Expected: the probability row keeps
target_basis exact_posterior and lands in soft_target_metric_only (an exact predictive is soft, never a 'miss'), the action row has
target_basis optimal_action (scored hard, action_regret when a loss table is present) and lands in common_teacher_failure
(exact_oracle is an independent tier). Meaning comes from the generator's declaration, not from one-hotness.
  python3 test_qsemantics_ledger.py TEACHER_DIR SCHEMA decisions_*.jsonl.gz ..."""
import gzip
import json
import sys

teacher_dir, schema = sys.argv[1], sys.argv[2]
sys.path.insert(0, teacher_dir)
import sdv1_failure_ledger as fl  # noqa: E402

pairs = {}
for path in sys.argv[3:]:
    with gzip.open(path, "rt") as g:
        for line in g:
            r = json.loads(line)
            gen = r.get("generator_family")
            if not gen or r["target_basis"] not in ("exact_posterior", "optimal_action"):
                continue
            slot = pairs.setdefault(gen, {})
            slot.setdefault(r["target_basis"], r)
fails, n = [], 0
for gen, d in sorted(pairs.items()):
    if set(d) != {"exact_posterior", "optimal_action"}:
        continue
    n += 1
    for tb, want in (("exact_posterior", "soft_target_metric_only"), ("optimal_action", "common_teacher_failure")):
        r = d[tb]
        r = dict(r, split="training") if r.get("split") not in ("discovery", "train", "training") else r
        g = fl.gold_of(r)
        wrong = min(g["gold"], key=lambda k: (g["gold"][k], k))
        preds = [dict(decision_uid=g["decision_uid"], teacher_id=m, transform="base", status="ok",
                      probabilities={k: (0.9 if k == wrong else 0.1 / (len(g["keys"]) - 1)) for k in g["keys"]},
                      prediction_sha256="fabricated", n_input_tokens=5) for m in ["stock", "garas", "d3", "pplx", "olmo"]]
        out = fl.ledger_row(g, preds)
        print(gen, tb, r["target_basis_source"], "loss" if r.get("loss_by_option") else "-", "->", out["category"])
        if out["category"] != want:
            fails.append(f"{gen}/{tb}: {out['category']} != {want}")
print("generators with both kinds:", n, "ALL PASS" if (n and not fails) else f"FAILED {fails or 'no pair found'}")
sys.exit(0 if n and not fails else 1)
