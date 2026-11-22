"""Round-B gap manifests: dedupe items_b vs every public image (same index as round A) + quotas -> manifests_b/
b_gap_nc (clean + NC) and b_gap_clean (clean only). Pools for the continuation decision; not mixed into Arm A."""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vb_manifest import PH_T, public_hashes  # noqa: E402

D = Path("/data/local/visdata")
SEED = 20261109
QUOTA = {
    "nc": {"ai2d_rb": 6000, "sqa_rb": 3000, "chartnet_rb": 4000, "ai2d_mmmu": 6000, "sqa_mmmu": 3000, "chartnet_mmmu": 4000,
           "chartnet_b": 12000, "vrd_b": 8000, "coco_b": 7000},
    "clean": {"chartnet_rb": 4000, "chartnet_mmmu": 4000, "chartnet_b": 12000, "vrd_b": 8000, "coco_b": 7000},
}


def main():
    pub = public_hashes()
    pub_sha = {r["sha256"]: r["path"] for r in pub}
    pub_ph = np.array([int(r["phash"], 16) for r in pub if r["phash"]], dtype=np.uint64)
    pub_ph_path = [r["path"] for r in pub if r["phash"]]
    items = []
    for f in sorted((D / "items_b").glob("*.jsonl")):
        items += [json.loads(l) for l in open(f)]
    drops, keep, stats = [], [], defaultdict(Counter)
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
            drops.append({"id": it["id"], "source": it["source"], **why})
        else:
            keep.append(it)
    out = D / "manifests_b"
    out.mkdir(exist_ok=True)
    rep = {"public_images": len(pub), "phash_threshold": PH_T, "winoground": "not on disk", "per_source": {k: dict(v) for k, v in stats.items()},
           "drops": len(drops)}
    (out / "dedupe_report_b.json").write_text(json.dumps(rep, indent=1))
    with open(out / "dedupe_drops_b.jsonl", "w") as f:
        for r in drops:
            f.write(json.dumps(r) + "\n")
    by = defaultdict(list)
    for it in keep:
        by[it["source"]].append(it)
    for line, quota in QUOTA.items():
        rows = []
        for src, q in quota.items():
            pool = sorted(by.get(src, []), key=lambda it: it["id"])
            pool = [it for it in pool if line == "nc" or it["line"] == "clean"]
            random.Random(f"{SEED}:b:{line}:{src}").shuffle(pool)
            rows += pool[:q]
        random.Random(f"{SEED}:b:{line}:order").shuffle(rows)
        name = f"b_gap_{line}"
        p = out / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        hdr = {"manifest": name, "rows": len(rows), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
               "per_source": dict(Counter(r["source"] for r in rows)), "per_family": dict(Counter(r["family"] for r in rows)),
               "per_line": dict(Counter(r["line"] for r in rows)), "quota": quota,
               "licences": {r["source"]: [r["line"], r["licence"]] for r in rows},
               "gold_letters": dict(Counter(r["gold"] for r in rows)),
               "readout_rows": sum(1 for r in rows if (r.get("prompt") or {}).get("readout")),
               "options_in_image_rows": sum(1 for r in rows if (r.get("prompt") or {}).get("options_in_image")),
               "excluded": {"MMMU dev/val": "MMMU-Pro overlap", "A-OKVQA": "data licence unresolved (repo LICENSE covers code)",
                            "GeomVerse": "no dataset licence", "TQA": "not fetched (NC anyway)",
                            "CV-Bench 3D Distance": "no clean 3D-box source resolved this round",
                            "AI2D": "NC line only: the zip's license.txt is an AI2 non-commercial licence (web pages say CC BY-SA)"}}
        (out / f"{name}.header.json").write_text(json.dumps(hdr, indent=1, ensure_ascii=False))
        print(json.dumps({k: hdr[k] for k in ("manifest", "rows", "rows_sha256", "per_family", "per_line", "readout_rows", "options_in_image_rows")}), flush=True)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
