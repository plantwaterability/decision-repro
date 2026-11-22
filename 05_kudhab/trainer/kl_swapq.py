#!/usr/bin/env python3
"""kl_swapq.py SRC_PREFIX(fs5kt) ROWIDS(fs5k.rowids.txt) HASQ_META(fs5k.meta.npz) REQUESTS(fs5.jsonl.gz) TEACHER.jsonl.gz DST_PREFIX (KL agent): the same rows,
tokens, gold and weights as SRC with the teacher targets replaced by another teacher file ({"id", "p": {option_key: prob}}, teacher_karpo_fs5's format).
Checks: every SRC row has exactly one teacher row; its keys = the row's option keys (taken from the request, shown order); probabilities finite, >= 0, sum 1 +- 1e-3
(renormalised). Run with teacher_karpo_fs5 itself it must reproduce SRC bit for bit (same data_sha256): the test of the id / key mapping."""
import gzip, hashlib, json, sys
import numpy as np
src, rowids_f, meta_f, req_f, teach_f, dst = sys.argv[1:7]
info = json.load(open(src + ".info.json")); m = np.load(src + ".meta.npz"); K = m["K"].astype(np.int64); ooff = m["ooff"]; n = len(K)
allids = open(rowids_f).read().split("\n"); allids = allids[:-1] if allids[-1] == "" else allids
hasq = np.load(meta_f)["hasq"].astype(bool); assert len(allids) == len(hasq), (len(allids), len(hasq))
ids = [i for i, h in zip(allids, hasq) if h]; assert len(ids) == n == info["rows"], (len(ids), n)
keys = {}
want = set(ids)
for l in gzip.open(req_f, "rt"):
    r = json.loads(l)
    if r["id"] in want: keys[r["id"]] = list(r["request"]["questions"]["q"]["criteria"])
assert len(keys) == n, f"requests for {len(keys)} of {n} rows"
teach = {}; extra = 0; th = hashlib.sha256(open(teach_f, "rb").read()).hexdigest()
for l in gzip.open(teach_f, "rt"):
    r = json.loads(l)
    if "p" not in r: continue
    if r["id"] in want:
        assert r["id"] not in teach, f"duplicate teacher row {r['id']}"; teach[r["id"]] = r["p"]
    else: extra += 1
missing = [i for i in ids if i not in teach]; assert not missing, f"{len(missing)} rows without a teacher target, e.g. {missing[:3]}"
q = np.zeros(int(ooff[-1]), dtype=np.float32); oldq = m["q"]; maxdev = 0.0; same_arg = 0; l1 = 0.0
for j, i in enumerate(ids):
    ks = keys[i]; p = teach[i]; assert len(ks) == K[j] and set(p) == set(ks), f"keys differ on {i}"
    v = np.array([float(p[k]) for k in ks], dtype=np.float64); assert np.isfinite(v).all() and (v >= 0).all(), i
    s = v.sum(); maxdev = max(maxdev, abs(s - 1.0)); assert abs(s - 1.0) < 1e-3, f"{i} sums to {s}"
    a, b = int(ooff[j]), int(ooff[j]) + int(K[j]); q[a:b] = v.astype(np.float32) if abs(s - 1.0) < 1e-9 else (v / s).astype(np.float32)
    o = oldq[a:b]; same_arg += int(np.argmax(q[a:b]) == np.argmax(o)); l1 += float(np.abs(q[a:b].astype(np.float64) - o).sum())
gold = m["gold"]; ag = sum(int(np.argmax(q[int(ooff[j]):int(ooff[j]) + int(K[j])]) == np.argmax(gold[int(ooff[j]):int(ooff[j]) + int(K[j])])) for j in range(n))
idsarr = np.load(src + ".ids.npy", mmap_mode="r"); h = hashlib.sha256()
for x in (np.asarray(idsarr), m["off"], m["K"], m["ooff"], m["gold"], q, m["hasq"], m["w"]): h.update(np.ascontiguousarray(x).tobytes())
np.save(dst + ".ids.npy", np.asarray(idsarr)); np.savez(dst + ".meta.npz", off=m["off"], K=m["K"], w=m["w"], ooff=m["ooff"], gold=m["gold"], q=q, hasq=m["hasq"])
out = {"data_sha256": h.hexdigest(), "rows": n, "teacher_rows": n, "tokens": info["tokens"], "len_max": info["len_max"], "weight": info["weight"], "K_max": info["K_max"],
       "same_rows_as": info["data_sha256"], "ids_equal_to_source": True, "teacher_file": teach_f.split("/")[-1], "teacher_file_sha256": th, "teacher_rows_not_in_set": extra,
       "max_abs_sum_minus_1": maxdev, "argmax_same_as_source_teacher": same_arg, "mean_L1_to_source_teacher": l1 / n, "teacher_argmax_is_gold_argmax": ag,
       "template": "karpo-blocks-v1", "karpo_code": "a6fa38bf"}
json.dump(out, open(dst + ".info.json", "w"), indent=1); print("KLSWAPQ_DONE", json.dumps(out))
