#!/usr/bin/env python3
"""Tier accounting for rendered decision rows: counts by (label_provenance, label_verification, target_basis, tier), and rows whose
tier differs from (a) the pre-evidence Tasksource mapping (every Tasksource llm-reviewed source unverified) and (b) the
source-level-only mapping (no per-question declarations).
  python3 sd0_tier_report.py SCHEMA decisions_X.jsonl.gz ... > tier_report.json"""
import collections
import gzip
import json
import sys

S = json.load(open(sys.argv[1]))
tier = lambda lv: S["label_verification"][lv]["tier"]
out = {}
for path in sys.argv[2:]:
    c = collections.Counter()
    moved_ts = collections.Counter()
    moved_q = collections.Counter()
    n = 0
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        n += 1
        t = tier(r["label_verification"])
        c[(r["label_provenance"], r["label_verification"], r["target_basis"], t)] += 1
        if r["decision_uid"].startswith("tasksource_ff_v1|") and r["label_provenance"] == "human_annotation":
            moved_ts[f"unverified -> {t}"] += 1
        base = S["stage0_provenance_map"][r["label_provenance"]]
        if (r["target_basis"], r["label_verification"]) != (base["target_basis"], base["label_verification"]):
            moved_q[f"{base['target_basis']}/{base['label_verification']} -> {r['target_basis']}/{r['label_verification']} ({tier(base['label_verification'])} -> {t})"] += 1
    out[path.split("/")[-1]] = {"rows": n, "by_tier": dict(collections.Counter({k[3]: 0 for k in c}) + collections.Counter({k: 0 for k in []}) or {}),
                                "tier_counts": {k: sum(v for kk, v in c.items() if kk[3] == k) for k in ("independent", "source_gold", "unverified")},
                                "tasksource_moved_by_card_evidence": dict(moved_ts), "per_question_overrides": dict(moved_q),
                                "detail": {" | ".join(k): v for k, v in sorted(c.items(), key=lambda x: -x[1])}}
print(json.dumps(out, indent=1))
