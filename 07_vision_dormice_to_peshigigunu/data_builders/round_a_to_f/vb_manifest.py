"""Dedupe gate + quota sampling + the two manifests (NC line for Arm A, clean-only twin).

Dedupe: every training image vs every public vision image on disk (suite_noWG: 11,410 rows' images incl. Mind2Web marked
and clip crops; Winoground is not on disk = recorded gap). Drop an item if any of its images matches a public image by
exact sha256 (source bytes or saved bytes) or by 64-bit DCT pHash Hamming distance <= PH_T.
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vb_build import phash  # noqa: E402

D = Path("/data/local/visdata")
PUB = D / "public" / "suite"
PH_T = 6
SEED = 20261109

QUOTA = {
    "nc": {"m2w": 7000, "cord_train": 5000, "funsd_train": 1500, "infovqa_train": 6000, "docvqa": 4000, "chartnet": 9000,
           "coco_count": 6000, "vrd25": 2500, "vsr": 4000, "mindcube": 4000, "sat": 4000, "hm_train": 5000, "meme_v1": 1500,
           "coco_wino": 3000},
    "clean": {"m2w": 7000, "cord_train": 5000, "docvqa": 8000, "chartnet": 9000, "coco_count": 6000, "vrd25": 2500, "vsr": 4000,
              "mindcube": 4000, "sat": 4000, "meme_v1": 3000, "coco_wino": 3000},
}
FAMILIES9 = ["CV-Bench", "BLINK", "RealWorldQA", "CharXiv", "InfographicVQA", "Mind2Web", "Winoground", "KIE (CORD+FUNSD)",
             "Moderation (Hateful Memes)"]


def hash_one(p):
    from PIL import Image

    b = p.read_bytes()
    try:
        im = Image.open(p)
        im.load()
        ph = phash(im)
    except Exception:
        ph = None
    return str(p.relative_to(PUB)), hashlib.sha256(b).hexdigest(), ph


def public_hashes():
    cache = D / "public_hashes.jsonl"
    if cache.exists():
        return [json.loads(l) for l in open(cache)]
    files = [p for p in PUB.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif")]
    with Pool(4) as pool:
        res = pool.map(hash_one, files, chunksize=64)
    out = [{"path": r[0], "sha256": r[1], "phash": None if r[2] is None else f"{r[2]:016x}"} for r in res]
    with open(cache, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def main():
    pub = public_hashes()
    pub_sha = {r["sha256"]: r["path"] for r in pub}
    pub_ph = np.array([int(r["phash"], 16) for r in pub if r["phash"]], dtype=np.uint64)
    pub_ph_path = [r["path"] for r in pub if r["phash"]]
    items = []
    for f in sorted((D / "items").glob("*.jsonl")):
        items += [json.loads(l) for l in open(f)]
    drops, keep = [], []
    stats = defaultdict(Counter)
    for it in items:
        why = None
        for s1, s2, ph in zip(it["img_src_sha256"], it["img_sha256"], it["img_phash"]):
            if s1 in pub_sha or s2 in pub_sha:
                why = {"kind": "sha256", "public": pub_sha.get(s1) or pub_sha.get(s2)}
                break
            d = np.bitwise_count(pub_ph ^ np.uint64(int(ph, 16)))
            j = int(d.argmin())
            if int(d[j]) <= PH_T:
                why = {"kind": "phash", "distance": int(d[j]), "public": pub_ph_path[j]}
                break
        stats[it["source"]]["checked"] += 1
        if why:
            stats[it["source"]][f"drop_{why['kind']}"] += 1
            drops.append({"id": it["id"], "source": it["source"], "images": it["images"], **why})
        else:
            keep.append(it)
    by = defaultdict(list)
    for it in keep:
        by[it["source"]].append(it)
    (D / "manifests").mkdir(exist_ok=True)
    report = {"public_images": len(pub), "public_with_phash": int(len(pub_ph)), "phash_threshold": PH_T,
              "winoground_public_images": "NOT ON DISK (gated) - not deduped against; recorded gap",
              "per_source": {k: dict(v) for k, v in stats.items()}, "drops": len(drops)}
    (D / "manifests" / "dedupe_report.json").write_text(json.dumps(report, indent=1))
    with open(D / "manifests" / "dedupe_drops.jsonl", "w") as f:
        for r in drops:
            f.write(json.dumps(r) + "\n")
    for line, quota in QUOTA.items():
        rows = []
        for src, q in quota.items():
            pool = sorted(by.get(src, []), key=lambda it: it["id"])
            if line == "clean":
                pool = [it for it in pool if it["line"] == "clean"]
            random.Random(f"{SEED}:{line}:{src}").shuffle(pool)
            rows += pool[:q]
        assert line == "nc" or all(r["line"] == "clean" for r in rows)
        random.Random(f"{SEED}:{line}:order").shuffle(rows)
        name = "arm_a_nc" if line == "nc" else "twin_clean"
        p = D / "manifests" / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        fam = Counter(r["family"] for r in rows)
        hdr = {"manifest": name, "rows": len(rows), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
               "per_source": dict(Counter(r["source"] for r in rows)), "per_family": dict(fam),
               "per_line": dict(Counter(r["line"] for r in rows)), "quota": quota,
               "families_missing": [f for f in FAMILIES9 if fam.get(f, 0) == 0],
               "licences": {r["source"]: [r["line"], r["licence"]] for r in rows},
               "gold_letters": dict(Counter(r["gold"] for r in rows)),
               "dedupe": {"phash_threshold": PH_T, "drops_total": len(drops), "winoground": "not on disk"},
               "excluded": {"MMMU dev/val": "MMMU-Pro is built from MMMU", "BDD100K": "dataset terms unresolved at source",
                            "R-Bench-M / MMMU-Pro style": "no clean source in round A (not in the private sets)",
                            "GroundCUA": "MIT (resolved) but screenshot-to-element mapping not in the instruction files; not built in round A"}}
        (D / "manifests" / f"{name}.header.json").write_text(json.dumps(hdr, indent=1, ensure_ascii=False))
        print(json.dumps({k: hdr[k] for k in ("manifest", "rows", "rows_sha256", "per_family", "per_line", "families_missing")}), flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
