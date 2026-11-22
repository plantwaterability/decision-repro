"""Arm U manifest: UNION of rounds A, B (R-Bench rows = B3 fixed construction), C (Hypersim) and D (VQAv2/VG/TallyQA), every row
deduped vs the public images; family shares = A3's (none below A3; BLINK/Hypersim at least A3's level); 3,000 steps x 64 = 192,000
rows; pools smaller than their target repeat (whole-pool passes, then a seeded partial pass; repeat ids get a '#rK' suffix)."""
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
SEED = 20261114
TOTAL = 192000
SHARE = {"R-Bench-M": 9000, "MMMU-Pro vision": 9000, "BLINK": 10000, "CharXiv": 8000, "CV-Bench": 8000, "Mind2Web": 5000,
         "KIE (CORD+FUNSD)": 4500, "RealWorldQA": 3994, "Moderation (Hateful Memes)": 3000, "InfographicVQA": 2368, "Winoground": 1138}  # A3 counts / 64,000
HYPERSIM_MIN = 5000 / 64000
pub = public_hashes()
psha = {r["sha256"] for r in pub}
pph = np.array([int(r["phash"], 16) for r in pub if r["phash"]], dtype=np.uint64)


def clean(it):
    return not any(s1 in psha or s2 in psha or int(np.bitwise_count(pph ^ np.uint64(int(p, 16))).min()) <= PH_T
                   for s1, s2, p in zip(it["img_src_sha256"], it["img_sha256"], it["img_phash"]))


def load(paths):
    out = []
    for p in paths:
        out += [json.loads(l) for l in open(p)]
    return out


a_drop = {json.loads(l)["id"] for l in open(D / "manifests" / "dedupe_drops.jsonl")}
b_drop = {json.loads(l)["id"] for l in open(D / "manifests_b" / "dedupe_drops_b.jsonl")}
pool = [r for r in load(sorted((D / "items").glob("*.jsonl"))) if r["id"] not in a_drop]
pool += [r for r in load(sorted((D / "items_b").glob("*.jsonl"))) if r["id"] not in b_drop and r["family"] != "R-Bench-M"]
pool += [r for r in load(sorted((D / "items_b3").glob("*.jsonl"))) if clean(r)]
pool += load([Path("/data/shared/visdata_c/manifests_c/c_gap_clean.jsonl")])
pool += [r for r in load(sorted(Path("/data/shared/visdata_d/items_d").glob("*.jsonl"))) if clean(r)]
meme = [r for r in pool if r["source"] == "meme_v1"]
pool = [r for r in pool if r["source"] != "meme_v1"] + sorted(meme, key=lambda r: r["id"])[:3000]  # meme pool stays capped
fam_pool = {}
for r in pool:
    fam_pool.setdefault(r["family"], []).append(r)

out, reps = [], {}


def draw(rows, n, tag):
    rows = sorted(rows, key=lambda r: r["id"])
    got, k = [], 0
    while len(got) < n:
        order = rows[:]
        random.Random(f"{SEED}:{tag}:{k}").shuffle(order)
        need = n - len(got)
        got += [{**r, "id": r["id"] if k == 0 else f"{r['id']}#r{k}"} for r in order[:need]]
        k += 1
    reps[tag] = round(n / max(1, len(rows)), 2)
    return got


for fam, cnt in SHARE.items():
    n = round(TOTAL * cnt / 64000)
    rows = fam_pool[fam]
    if fam == "BLINK":  # Hypersim at least A3's share of the WHOLE manifest
        hs = [r for r in rows if r["source"].startswith("hs_")]
        n_hs = max(round(TOTAL * HYPERSIM_MIN), round(n * len(hs) / len(rows)))
        out += draw(hs, n_hs, "BLINK:hypersim") + draw([r for r in rows if not r["source"].startswith("hs_")], n - n_hs, "BLINK:other")
    else:
        out += draw(rows, n, fam)
out = out[:TOTAL] if len(out) > TOTAL else out
assert len({r["id"] for r in out}) == len(out), "duplicate ids"
random.Random(f"{SEED}:order").shuffle(out)
p = D / "manifests" / "u_union.jsonl"
with open(p, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fam = Counter(r["family"] for r in out)
hdr = {"manifest": "u_union", "rows": len(out), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
       "per_family": dict(fam), "per_family_share": {k: round(v / len(out), 4) for k, v in fam.items()},
       "a3_share": {k: round(v / 64000, 4) for k, v in SHARE.items()}, "pool_sizes": {k: len(v) for k, v in fam_pool.items()},
       "repeat_factor": reps, "hypersim_share": round(sum(1 for r in out if r["source"].startswith("hs_")) / len(out), 4),
       "per_source": dict(Counter(r["source"] for r in out)), "per_line": dict(Counter(r["line"] for r in out)),
       "rbench_other_is_gold": sum(1 for r in out if r["family"] == "R-Bench-M" and r["options"][ord(r["gold"]) - 65] == "Other options are incorrect")}
(D / "manifests" / "u_union.header.json").write_text(json.dumps(hdr, indent=1, ensure_ascii=False))
print(json.dumps({k: hdr[k] for k in ("rows", "rows_sha256", "per_family", "pool_sizes", "repeat_factor", "hypersim_share", "per_line")}))
