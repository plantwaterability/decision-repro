#!/usr/bin/env python3
"""kl_subset.py SRC_PREFIX DST_PREFIX (KL agent): the rows of a kl_data.py set that carry a teacher target (hasq), same order, same arrays."""
import hashlib, json, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
info = json.load(open(src + ".info.json")); ids = np.load(src + ".ids.npy", mmap_mode="r"); m = np.load(src + ".meta.npz")
sel = m["hasq"].astype(bool); ln = np.diff(m["off"]); K = m["K"].astype(np.int64)
ids2 = np.asarray(ids)[np.repeat(sel, ln)].astype(np.int32); ln2 = ln[sel]; K2 = m["K"][sel]
off = np.concatenate([[0], np.cumsum(ln2)]); ooff = np.concatenate([[0], np.cumsum(K2.astype(np.int64))])
om = np.repeat(sel, K); gold, q, w, hasq = m["gold"][om], m["q"][om], m["w"][sel], m["hasq"][sel]
assert len(ids2) == off[-1] and len(gold) == ooff[-1] == len(q) and hasq.all()
h = hashlib.sha256()
for x in (ids2, off, K2, ooff, gold, q, hasq, w): h.update(np.ascontiguousarray(x).tobytes())
np.save(dst + ".ids.npy", ids2); np.savez(dst + ".meta.npz", off=off, K=K2, w=w, ooff=ooff, gold=gold, q=q, hasq=hasq)
out = {"data_sha256": h.hexdigest(), "rows": int(sel.sum()), "teacher_rows": int(hasq.sum()), "tokens": int(ln2.sum()), "len_max": int(ln2.max()), "weight": float(w.sum()), "K_max": int(K2.max()),
       "subset_of": info["data_sha256"], "selection": "rows with a karpo teacher target (hasq)", "template": "karpo-blocks-v1", "karpo_code": "a6fa38bf"}
json.dump(out, open(dst + ".info.json", "w"), indent=1); print("KLSUBSET_DONE", json.dumps(out))
