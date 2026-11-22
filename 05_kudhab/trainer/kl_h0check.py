#!/usr/bin/env python3
"""kl_h0check.py FS5KT_PREFIX FS5K_ROWIDS FS5K_META FS5KHFULL_PREFIX FS5KH_PREFIX OUT.json (KL agent, 11-16): every fs5kt row must sit unchanged inside fs5kh
(tokens, option count, gold, teacher q, weight), in the same relative order; and fs5kh = the teacher rows of fs5khfull. Prints KLH0CHECK {...}."""
import json, sys
import numpy as np
kt, k_ids, k_meta, hf, hk, out = sys.argv[1:7]
def ids_of(rowids, meta):
    r = open(rowids).read().split("\n"); r = r[:-1] if r and r[-1] == "" else r; h = np.load(meta)["hasq"].astype(bool); assert len(r) == len(h); return [i for i, x in zip(r, h) if x]
def load(p): z = np.load(p + ".meta.npz"); m = {k: z[k] for k in z.files}; return np.load(p + ".ids.npy"), m      # arrays read ONCE (an NpzFile re-reads an array on every [] access)
a_ids = ids_of(k_ids, k_meta); b_ids = ids_of(hf + ".rowids.txt", hf + ".meta.npz")
A, am = load(kt); B, bm = load(hk)
assert len(a_ids) == len(am["K"]) and len(b_ids) == len(bm["K"]), (len(a_ids), len(am["K"]), len(b_ids), len(bm["K"]))
pos = {i: j for j, i in enumerate(b_ids)}; bad = {"missing": 0, "tokens": 0, "K": 0, "gold": 0, "q": 0, "w": 0}; last = -1; order_ok = True
for j, i in enumerate(a_ids):
    b = pos.get(i)
    if b is None: bad["missing"] += 1; continue
    if b < last: order_ok = False
    last = b
    if not np.array_equal(A[am["off"][j]:am["off"][j + 1]], B[bm["off"][b]:bm["off"][b + 1]]): bad["tokens"] += 1
    if am["K"][j] != bm["K"][b]: bad["K"] += 1; continue
    sa, sb = slice(int(am["ooff"][j]), int(am["ooff"][j + 1])), slice(int(bm["ooff"][b]), int(bm["ooff"][b + 1]))
    bad["gold"] += int(not np.array_equal(am["gold"][sa], bm["gold"][sb])); bad["q"] += int(not np.array_equal(am["q"][sa], bm["q"][sb])); bad["w"] += int(am["w"][j] != bm["w"][b])
res = {"fs5kt_rows": len(a_ids), "fs5kh_rows": len(b_ids), "new_rows": len(b_ids) - len(a_ids) + bad["missing"], "mismatch": bad, "order_kept": order_ok,
       "identical": len(a_ids) - bad["missing"] - max(bad["tokens"], bad["K"], bad["gold"], bad["q"], bad["w"]) if not any(bad.values()) else None}
res["ok"] = (not any(bad.values())) and order_ok
json.dump(res, open(out, "w"), indent=1); print("KLH0CHECK", json.dumps(res))
