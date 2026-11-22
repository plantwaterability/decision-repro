"""Arm A4 manifest (64,000 rows): A3's families re-weighted + round D (VQAv2 / VG real-world QA, TallyQA counting).
Round-D rows get the public-image dedupe here (sha256 + pHash <= 6 vs the same 10,742-image index)."""
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
DD = Path("/data/shared/visdata_d")
SEED = 20261113
pub = public_hashes()
pub_sha = {r["sha256"] for r in pub}
pub_ph = np.array([int(r["phash"], 16) for r in pub if r["phash"]], dtype=np.uint64)


def clean(it):
    for s1, s2, ph in zip(it["img_src_sha256"], it["img_sha256"], it["img_phash"]):
        if s1 in pub_sha or s2 in pub_sha or int(np.bitwise_count(pub_ph ^ np.uint64(int(ph, 16))).min()) <= PH_T:
            return False
    return True


load = lambda p: [json.loads(l) for l in open(p)]
A = load(D / "manifests" / "arm_a_nc.jsonl")
Bm = load(D / "manifests_b" / "b_gap_nc.jsonl")
RB2 = [r for f in sorted((D / "items_b3").glob("*.jsonl")) for r in load(f)]
Cm = load(C / "manifests_c" / "c_gap_clean.jsonl")
Draw = {f.stem: load(f) for f in sorted((DD / "items_d").glob("*.jsonl"))}
Dk = {k: [r for r in v if clean(r)] for k, v in Draw.items()}
d_drops = {k: len(Draw[k]) - len(Dk[k]) for k in Draw}
RB2 = [r for r in RB2 if clean(r)]

PLAN = [  # (pool, selector, n, part)
    ("A", lambda r: r["family"] == "RealWorldQA", 3000, "A"),
    ("D", lambda r: r["source"] == "vqa2_rw", 6000, "D"),
    ("D", lambda r: r["source"] == "vg_rw", 3000, "D"),
    ("A", lambda r: r["family"] == "CV-Bench", 3000, "A"),
    ("B", lambda r: r["family"] == "CV-Bench", 2500, "B"),
    ("D", lambda r: r["source"] == "tally_cnt", 3000, "D"),
    ("C", lambda r: r["family"] == "BLINK", 5500, "C"),
    ("B", lambda r: r["family"] == "BLINK", 2000, "B"),
    ("A", lambda r: r["family"] == "BLINK", 2000, "A"),
    ("RB2", lambda r: r["family"] == "R-Bench-M", 7000, "B_rb2"),
    ("B", lambda r: r["family"] == "MMMU-Pro vision", 7000, "B"),
    ("A", lambda r: r["family"] == "CharXiv", 3431, "A"),
    ("B", lambda r: r["family"] == "CharXiv", 3431, "B"),
    ("A", lambda r: r["family"] == "Mind2Web", 4000, "A"),
    ("A", lambda r: r["family"] == "KIE (CORD+FUNSD)", 3500, "A"),
    ("A", lambda r: r["family"] == "InfographicVQA", 2000, "A"),
    ("A", lambda r: r["family"] == "Moderation (Hateful Memes)", 2500, "A"),
    ("A", lambda r: r["family"] == "Winoground", 1138, "A"),
]
POOLS = {"A": A, "B": Bm, "C": Cm, "RB2": RB2, "D": [r for v in Dk.values() for r in v]}
out, short = [], {}
for i, (pool, sel, n, part) in enumerate(PLAN):
    rows = sorted((r for r in POOLS[pool] if sel(r)), key=lambda r: r["id"])
    random.Random(f"{SEED}:{i}").shuffle(rows)
    if len(rows) < n:
        short[f"{i}:{pool}"] = [n, len(rows)]
    out += [{**r, "a4_part": part} for r in rows[:n]]
assert len({r["id"] for r in out}) == len(out), "duplicate ids"
random.Random(f"{SEED}:order").shuffle(out)
p = D / "manifests" / "a4_mix.jsonl"
with open(p, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fam = Counter(r["family"] for r in out)
hdr = {"manifest": "a4_mix", "rows": len(out), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "short": short,
       "per_family": dict(fam), "per_family_share": {k: round(v / len(out), 4) for k, v in fam.items()},
       "per_part": dict(Counter(r["a4_part"] for r in out)), "per_source": dict(Counter(r["source"] for r in out)),
       "per_line": dict(Counter(r["line"] for r in out)), "round_d_dedupe_drops": d_drops,
       "rbench_rows_with_other": sum(1 for r in out if r["family"] == "R-Bench-M" and "Other options are incorrect" in r["options"]),
       "rbench_other_is_gold": sum(1 for r in out if r["family"] == "R-Bench-M" and r["options"][ord(r["gold"]) - 65] == "Other options are incorrect"),
       "plan": [(pool, n, part) for pool, _, n, part in PLAN]}
(D / "manifests" / "a4_mix.header.json").write_text(json.dumps(hdr, indent=1, ensure_ascii=False))
print(json.dumps({k: hdr[k] for k in ("rows", "rows_sha256", "per_family", "per_line", "short", "round_d_dedupe_drops", "rbench_rows_with_other",
                                      "rbench_other_is_gold")}))
