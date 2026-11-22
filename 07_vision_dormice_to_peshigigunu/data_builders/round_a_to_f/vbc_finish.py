"""Round C: dedupe items_c vs every public image (same index), write c_gap_clean manifest (all Hypersim rows are clean, CC BY-SA)."""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/local/visdata/scripts")
from vb_manifest import PH_T, public_hashes  # noqa: E402

C = Path("/data/shared/visdata_c")
pub = public_hashes()
pub_sha = {r["sha256"]: r["path"] for r in pub}
pub_ph = np.array([int(r["phash"], 16) for r in pub if r["phash"]], dtype=np.uint64)
items = []
for f in sorted((C / "items_c").glob("*.jsonl")):
    items += [json.loads(l) for l in open(f)]
keep, drops, stats = [], [], defaultdict(Counter)
for it in items:
    bad = None
    for s1, s2, ph in zip(it["img_src_sha256"], it["img_sha256"], it["img_phash"]):
        if s1 in pub_sha or s2 in pub_sha:
            bad = "sha256"
            break
        if int(np.bitwise_count(pub_ph ^ np.uint64(int(ph, 16))).min()) <= PH_T:
            bad = "phash"
            break
    stats[it["source"]]["checked"] += 1
    if bad:
        stats[it["source"]][f"drop_{bad}"] += 1
        drops.append({"id": it["id"], "kind": bad})
    else:
        keep.append(it)
random.Random("20261111:c:order").shuffle(keep)
out = C / "manifests_c"
out.mkdir(exist_ok=True)
p = out / "c_gap_clean.jsonl"
with open(p, "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
hdr = {"manifest": "c_gap_clean", "rows": len(keep), "rows_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
       "per_source": dict(Counter(r["source"] for r in keep)), "per_subtask": dict(Counter(r["subtask"] for r in keep)),
       "per_line": dict(Counter(r["line"] for r in keep)), "gold_letters": dict(Counter(r["gold"] for r in keep)),
       "dedupe": {"phash_threshold": PH_T, "per_source": {k: dict(v) for k, v in stats.items()}, "drops": len(drops)},
       "licences": {r["source"]: [r["line"], r["licence"]] for r in keep},
       "not_built": {"BLINK Semantic/Functional correspondence": "no licence-clear source (SPair-71k / FunKPoint have no data licence; PF-PASCAL/WILLOW unresolved)",
                     "MegaDepth": "derived files CC BY 4.0 but the Flickr photos keep their own (unknown) licences",
                     "HPatches": "licence not found at source", "CO3D": "CC BY-NC 4.0 (NC line); not built this round (download size)"}}
(out / "c_gap_clean.header.json").write_text(json.dumps(hdr, indent=1))
print(json.dumps({k: hdr[k] for k in ("rows", "rows_sha256", "per_source", "per_subtask", "gold_letters", "dedupe")}))
