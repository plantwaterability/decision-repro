#!/usr/bin/env python3
"""kl_pick.py SRC_PREFIX FS5K_ROWIDS FS5K_META NC_IDS SEED OUT_DIR (KL agent, 11-16): three training sets, by row id, from ONE kl_data.py build SRC (with .rowids.txt) of
FS5 + round H0 + round H (its exclusions already applied), each in kl_data's format + .rowids.txt:
  fs5hh  (K21) = every SRC row with a teacher target except the NC ids;
  fs5hn  (K23) = the NEW rows (not fs5kt rows, not NC) + an equal-size replay sample of the fs5kt rows present in SRC (numpy default_rng(SEED).choice, no replacement);
  fs5hnc (K22) = fs5hn + the NC rows.   Rows keep SRC (= manifest) order; the trainer shuffles. Prints KLPICK {...}."""
import hashlib, json, sys
import numpy as np
src, k_rowids, k_meta, nc_f, seed, outdir = sys.argv[1:7]; seed = int(seed)
ids = np.load(src + ".ids.npy"); z = np.load(src + ".meta.npz"); m = {k: z[k] for k in z.files}
def lines(f): x = open(f).read().split("\n"); return [l for l in x if l.strip() and not l.startswith("#")]
rowids = lines(src + ".rowids.txt"); n = len(m["K"]); assert len(rowids) == n, (len(rowids), n)
r = lines(k_rowids); h = np.load(k_meta)["hasq"].astype(bool); assert len(r) == len(h)
kt = set(i for i, x in zip(r, h) if x); nc = set(lines(nc_f))
hasq = m["hasq"].astype(bool); is_nc = np.array([i in nc for i in rowids]); is_kt = np.array([i in kt for i in rowids]); is_h = np.array([i.startswith("fs5h:") for i in rowids])
base = hasq & ~is_nc; new = np.nonzero(base & ~is_kt)[0]; pool = np.nonzero(base & is_kt)[0]; ncrows = np.nonzero(hasq & is_nc)[0]
assert len(pool) >= len(new), (len(pool), len(new))
rep = np.sort(np.random.default_rng(seed).choice(pool, size=len(new), replace=False))
sets = {"fs5hh": np.nonzero(base)[0], "fs5hn": np.sort(np.concatenate([new, rep])), "fs5hnc": np.sort(np.concatenate([new, rep, ncrows]))}
ln_all = np.diff(m["off"]); K_all = m["K"].astype(np.int64); res = {"src_rows": n, "src_without_teacher": int((~hasq).sum()), "fs5kt_rows_in_src": int((hasq & is_kt).sum()),
     "new_rows": len(new), "new_h0": int((~is_h[new]).sum()), "new_h": int(is_h[new].sum()), "replay": len(rep), "nc_rows": len(ncrows), "seed": seed}
for name, sel in sets.items():
    mask = np.zeros(n, bool); mask[sel] = True
    tok = ids[np.repeat(mask, ln_all)].astype(np.int32); ln = ln_all[sel]; off = np.concatenate([[0], np.cumsum(ln)])
    K = m["K"][sel]; ooff = np.concatenate([[0], np.cumsum(K.astype(np.int64))]); om = np.repeat(mask, K_all)
    gold, q, hq, w = m["gold"][om], m["q"][om], m["hasq"][sel], m["w"][sel]
    assert len(tok) == off[-1] and len(gold) == ooff[-1] == len(q) and hq.all()
    hs = hashlib.sha256()
    for x in (tok, off, K, ooff, gold, q, hq, w): hs.update(np.ascontiguousarray(x).tobytes())
    p = f"{outdir}/{name}"; np.save(p + ".ids.npy", tok); np.savez(p + ".meta.npz", off=off, K=K, w=w, ooff=ooff, gold=gold, q=q, hasq=hq)
    open(p + ".rowids.txt", "w").write("\n".join(rowids[i] for i in sel) + "\n")
    info = {"data_sha256": hs.hexdigest(), "rows": int(len(sel)), "teacher_rows": int(hq.sum()), "tokens": int(ln.sum()), "len_max": int(ln.max()), "over_8192": int((ln > 8192).sum()),
            "weight": float(w.sum()), "K_max": int(K.max()), "rows_fs5kt": int(is_kt[sel].sum()), "rows_h0": int((~is_kt[sel] & ~is_h[sel]).sum()), "rows_h": int(is_h[sel].sum()),
            "rows_nc": int(is_nc[sel].sum()), "subset_of": json.load(open(src + ".info.json"))["data_sha256"], "template": "karpo-blocks-v1", "karpo_code": "a6fa38bf", "pick": res}
    json.dump(info, open(p + ".info.json", "w"), indent=1); res[name] = {k: info[k] for k in ("rows", "tokens", "len_max", "over_8192", "K_max", "rows_fs5kt", "rows_h0", "rows_h", "rows_nc", "data_sha256")}
print("KLPICK", json.dumps(res))
