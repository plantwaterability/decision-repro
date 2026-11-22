"""Arm A3 manifest (64,000 rows = 1,000 x 64): round A + round B (R-Bench rows rebuilt with the fixed 'Other' construction) +
round C (Hypersim multi-view / visual correspondence). R-Bench v2 rows get the same public-image dedupe as every round."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/local/visdata/scripts")
from vb_manifest import PH_T, public_hashes  # noqa: E402

D = Path("/data/local/visdata")
C = Path("/data/shared/visdata_c")
SEED = 20261112
FROM_A = {"Mind2Web": 5000, "KIE (CORD+FUNSD)": 4500, "RealWorldQA": 3994, "InfographicVQA": 2368, "Moderation (Hateful Memes)": 3000,
          "Winoground": 1138, "CharXiv": 4000, "BLINK": 2500, "CV-Bench": 4000}
FROM_B = {"MMMU-Pro vision": 9000, "CharXiv": 4000, "BLINK": 2500, "CV-Bench": 4000}   # from b_gap_nc (R-Bench rows NOT taken from it)
RB2 = 9000                                                                               # R-Bench-M, fixed construction (items_b3)
HS = 5000                                                                                # BLINK from round C; shortfall -> round-A BLINK
GAP = {"R-Bench-M", "MMMU-Pro vision", "CharXiv", "BLINK", "CV-Bench"}

pub = public_hashes()
pub_sha = {r["sha256"] for r in pub}
pub_ph = np.array([int(r["phash"], 16) for r in pub if r["phash"]], dtype=np.uint64)


def clean_of_public(it):
    for s1, s2, ph in zip(it["img_src_sha256"], it["img_sha256"], it["img_phash"]):
        if s1 in pub_sha or s2 in pub_sha or int(np.bitwise_count(pub_ph ^ np.uint64(int(ph, 16))).min()) <= PH_T:
            return False
    return True


a = [json.loads(l) for l in open(D / "manifests" / "arm_a_nc.jsonl")]
b = [json.loads(l) for l in open(D / "manifests_b" / "b_gap_nc.jsonl")]
rb2 = [json.loads(l) for f in sorted((D / "items_b3").glob("*.jsonl")) for l in open(f)]
rb2_kept = [r for r in rb2 if clean_of_public(r)]
c = [json.loads(l) for l in open(C / "manifests_c" / "c_gap_clean.jsonl")]


def take(rows, fam, n, tag):
    pool = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["id"])
    random.Random(f"{SEED}:{tag}:{fam}").shuffle(pool)
    return pool[:n]


out, short = [], {}
hs = take(c, "BLINK", HS, "C")
if len(hs) < HS:
    short["C:BLINK"] = [HS, len(hs)]
    FROM_A["BLINK"] += HS - len(hs)
for tag, rows, targets in (("A", a, FROM_A), ("B", b, FROM_B)):
    for fam, n in targets.items():
        got = take(rows, fam, n, tag)
        if len(got) < n:
            short[f"{tag}:{fam}"] = [n, len(got)]
        out += [{**r, "a3_part": tag} for r in got]
rbs = take(rb2_kept, "R-Bench-M", RB2, "RB2")
if len(rbs) < RB2:
    short["RB2"] = [RB2, len(rbs)]
out += [{**r, "a3_part": "B_rb2"} for r in rbs] + [{**r, "a3_part": "C"} for r in hs]
assert len({r["id"] for r in out}) == len(out), "duplicate ids"
random.Random(f"{SEED}:order").shuffle(out)
p = D / "manifests" / "a3_mix.jsonl"
with open(p, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fam = Counter(r["family"] for r in out)
rbrows = [r for r in out if r["family"] == "R-Bench-M"]
hdr = {"manifest": "a3_mix", "rows": len(out), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "short": short,
       "targets": {"from_A": FROM_A, "from_B": FROM_B, "rbench_v2": RB2, "hypersim": HS},
       "per_family": dict(fam), "per_family_share": {k: round(v / len(out), 4) for k, v in fam.items()},
       "gap_family_share": round(sum(v for k, v in fam.items() if k in GAP) / len(out), 4),
       "per_part": dict(Counter(r["a3_part"] for r in out)), "per_source": dict(Counter(r["source"] for r in out)),
       "per_line": dict(Counter(r["line"] for r in out)),
       "rbench_rows_with_other": sum(1 for r in rbrows if "Other options are incorrect" in r["options"]),
       "rbench_other_is_gold": sum(1 for r in rbrows if r["options"][ord(r["gold"]) - 65] == "Other options are incorrect"),
       "rbench_rows": len(rbrows), "rbench_v2_dedupe_drops": len(rb2) - len(rb2_kept),
       "inputs": {"arm_a_nc": "2604f2be...", "b_gap_nc": "bc4025b6...", "c_gap_clean": json.load(open(C / "manifests_c" / "c_gap_clean.header.json"))["rows_sha256"]}}
(D / "manifests" / "a3_mix.header.json").write_text(json.dumps(hdr, indent=1, ensure_ascii=False))
print(json.dumps({k: hdr[k] for k in ("rows", "rows_sha256", "per_family", "gap_family_share", "per_line", "short", "rbench_rows",
                                      "rbench_rows_with_other", "rbench_other_is_gold", "rbench_v2_dedupe_drops")}))
