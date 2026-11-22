"""which images of a UG manifest have no OCR text yet (the OCR member needs it). usage: vg_ocrneed.py <manifest name> <ocr cache jsonl> [...]
The caches are keyed by image path ("<source>/<file>"). Writes manifests_g/ocr_needed_<manifest>.jsonl: one line per uncovered image with its
sha256 and, when a byte-identical image already has OCR text under another path, that path (same bytes -> same OCR output)."""
import json, sys
from collections import Counter
from pathlib import Path
G = Path("/data/shared/visdata_g")
man = sys.argv[1]
keys = set()
for c in sys.argv[2:]:
    for l in open(c):
        i = l.find('"image": "')
        if i >= 0:
            j = l.index('"', i + 10)
            keys.add(l[i + 10:j])
need, sha_cached = {}, {}
for l in open(G / "manifests_g" / f"{man}.jsonl"):
    r = json.loads(l)
    for p, s in zip(r["images"], r["img_sha256"]):
        if p in keys:
            sha_cached.setdefault(s, p)
        else:
            need.setdefault(p, (r["source"], s))
out = G / "manifests_g" / f"ocr_needed_{man}.jsonl"
per, reuse = Counter(), Counter()
with open(out, "w") as f:
    for p, (src, s) in sorted(need.items()):
        same = sha_cached.get(s)
        per[src] += 1
        reuse[src] += same is not None
        f.write(json.dumps({"image": p, "source": src, "img_sha256": s, "same_bytes_cached_as": same}) + "\n")
print(json.dumps({"manifest": man, "cache_keys": len(keys), "images_without_ocr": len(need), "per_source": dict(per),
                  "of_which_byte_identical_to_a_cached_image": dict(reuse), "file": str(out)}))
