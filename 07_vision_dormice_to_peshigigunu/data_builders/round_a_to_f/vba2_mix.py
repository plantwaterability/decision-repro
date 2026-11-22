"""Arm A2 manifest: round A (arm_a_nc) + round-B gap pools (b_gap_nc), explicit per-family targets, 64,000 rows (1,000 x 64).
Gap families get two thirds of the rows; the families Arm A already wins keep a third."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

D = Path("/data/local/visdata")
SEED = 20261110
FROM_A = {"Mind2Web": 5000, "KIE (CORD+FUNSD)": 4500, "RealWorldQA": 3994, "InfographicVQA": 3368, "Moderation (Hateful Memes)": 3000,
          "Winoground": 1138, "CharXiv": 4500, "BLINK": 4000, "CV-Bench": 4000}
FROM_B = {"R-Bench-M": 9000, "MMMU-Pro vision": 9000, "CharXiv": 4500, "BLINK": 4000, "CV-Bench": 4000}
GAP = {"R-Bench-M", "MMMU-Pro vision", "CharXiv", "BLINK", "CV-Bench"}

a = [json.loads(l) for l in open(D / "manifests" / "arm_a_nc.jsonl")]
b = [json.loads(l) for l in open(D / "manifests_b" / "b_gap_nc.jsonl")]
rows = []
short = {}
for src_rows, targets, tag in ((a, FROM_A, "A"), (b, FROM_B, "B")):
    for fam, n in targets.items():
        pool = sorted((r for r in src_rows if r["family"] == fam), key=lambda r: r["id"])
        random.Random(f"{SEED}:{tag}:{fam}").shuffle(pool)
        if len(pool) < n:
            short[f"{tag}:{fam}"] = [n, len(pool)]
        for r in pool[:n]:
            rows.append({**r, "a2_part": tag})
assert len({r["id"] for r in rows}) == len(rows), "duplicate ids across parts"
random.Random(f"{SEED}:order").shuffle(rows)
p = D / "manifests" / "a2_mix.jsonl"
with open(p, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fam = Counter(r["family"] for r in rows)
hdr = {"manifest": "a2_mix", "rows": len(rows), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
       "targets_from_round_A": FROM_A, "targets_from_round_B": FROM_B, "short": short,
       "per_family": dict(fam), "per_family_share": {k: round(v / len(rows), 4) for k, v in fam.items()},
       "gap_family_share": round(sum(v for k, v in fam.items() if k in GAP) / len(rows), 4),
       "per_part": dict(Counter(r["a2_part"] for r in rows)), "per_source": dict(Counter(r["source"] for r in rows)),
       "per_line": dict(Counter(r["line"] for r in rows)), "gold_letters": dict(Counter(r["gold"] for r in rows)),
       "readout_rows": sum(1 for r in rows if (r.get("prompt") or {}).get("readout")),
       "options_in_image_rows": sum(1 for r in rows if (r.get("prompt") or {}).get("options_in_image")),
       "rbench_other_option_rows": sum(1 for r in rows if r["family"] == "R-Bench-M" and "Other options are incorrect" in r["options"]),
       "rbench_other_is_gold_rows": sum(1 for r in rows if r["family"] == "R-Bench-M" and r.get("subtask") == "other_correct"),
       "inputs": {"arm_a_nc": "2604f2be007fd6aa0cddc9a7bcd2f203f7189f9e7a144c563640093329fb8057",
                  "b_gap_nc": "bc4025b60f31da6345d72879907c5ee64019329dbd162cb486658d0a40f7c1ad"}}
(D / "manifests" / "a2_mix.header.json").write_text(json.dumps(hdr, indent=1, ensure_ascii=False))
print(json.dumps({k: hdr[k] for k in ("rows", "rows_sha256", "per_family", "gap_family_share", "per_line", "short", "rbench_other_option_rows",
                                      "rbench_other_is_gold_rows", "options_in_image_rows")}))
