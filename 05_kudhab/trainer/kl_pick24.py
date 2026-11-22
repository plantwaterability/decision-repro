#!/usr/bin/env python3
"""kl_pick24.py SRC_PREFIX FS5K_ROWIDS FS5K_META H_META.jsonl MANIFEST[,MANIFEST..] EXCLUDED_IDS.txt SEED OUT_PREFIX (TA agent, 11-16; coordinator ~22:30Z): K24's set = K23's
(kl_pick.py's fs5hn) without the HellaSwag pool, from SRC = fs5hh (every row with a karpo target of the FS5 + round H0 + round H build, cfcolor / H0 chess / bv1_discovery /
DH's drop list already left out):
  NEW    = SRC rows that are not fs5kt rows, minus the round-H pools hellaswag and cfcolor (= H0 rows + pools sata_mrc, sata_goemo, gsm8k_ag, banking77; pool from DH's meta file)
  REPLAY = an equal-size sample of the fs5kt rows present in SRC (numpy default_rng(SEED).choice, no replacement), as kl_pick.py.
Rows keep SRC order (the trainer shuffles). It STOPS unless: 0 hellaswag / cfcolor rows by pool, by id text and by manifest source text; every pool count is exactly the expected
one; no excluded id (H0 chess / bv1_discovery, drop list) is present; H ids and meta agree. Output in kl_data's format + .rowids.txt + .info.json. Prints KLPICK24 {...}."""
import collections, hashlib, json, re, sys
import numpy as np
src, k_rowids, k_meta, h_meta, mans, excl_f, seed, out = sys.argv[1:9]; seed = int(seed)
WANT = {"h0": 33998, "sata_mrc": 11217, "sata_goemo": 5001, "gsm8k_ag": 10000, "banking77": 8638}; OUT_POOLS = {"hellaswag": 12000, "cfcolor": 0}      # expected in SRC; OUT_POOLS are removed
BAD = re.compile(r"hella.?swag|cf.?colou?r", re.I)
ids = np.load(src + ".ids.npy"); z = np.load(src + ".meta.npz"); m = {k: z[k] for k in z.files}
def lines(f): x = open(f).read().split("\n"); return [l for l in x if l.strip() and not l.startswith("#")]
rowids = lines(src + ".rowids.txt"); n = len(m["K"]); assert len(rowids) == n == len(set(rowids)), (len(rowids), n)
r = lines(k_rowids); h = np.load(k_meta)["hasq"].astype(bool); assert len(r) == len(h)
kt = set(i for i, x in zip(r, h) if x); hasq = m["hasq"].astype(bool); assert hasq.all(), "SRC must hold teacher rows only (fs5hh)"
pool_of = {}
for l in open(h_meta):
    x = json.loads(l); pool_of[x["exposure_id"]] = x["pool"]
is_kt = np.array([i in kt for i in rowids]); is_h = np.array([i.startswith("fs5h:") for i in rowids])
assert not (is_kt & is_h).any()
pools = np.array([pool_of[i] if hh else ("fs5kt" if k else "h0") for i, hh, k in zip(rowids, is_h, is_kt)])      # KeyError = an H row that DH's meta does not know
for i in np.nonzero(is_h)[0][:: max(1, int(is_h.sum()) // 5000)]:
    assert f"|{pools[i]}|" in rowids[i], ("H id does not carry its meta pool", rowids[i], pools[i])
src_counts = dict(collections.Counter(pools.tolist()))
for p, c in {**WANT, **OUT_POOLS}.items(): assert src_counts.get(p, 0) == c, ("SRC pool count", p, src_counts.get(p, 0), c)
assert set(src_counts) <= set(WANT) | set(OUT_POOLS) | {"fs5kt"}, ("unknown pool in SRC", src_counts)
drop = np.isin(pools, list(OUT_POOLS)); new = np.nonzero(~is_kt & ~drop)[0]; pool = np.nonzero(is_kt)[0]
assert len(pool) >= len(new) == sum(WANT.values()), (len(pool), len(new))
rep = np.sort(np.random.default_rng(seed).choice(pool, size=len(new), replace=False))
sel = np.sort(np.concatenate([new, rep])); assert len(sel) == len(set(sel.tolist())) == 2 * len(new)
selids = [rowids[i] for i in sel]; sset = set(selids)
# ---- checks on the built set
by_pool = dict(collections.Counter(pools[sel].tolist()))
assert by_pool.get("hellaswag", 0) == 0 and by_pool.get("cfcolor", 0) == 0, by_pool
for p, c in WANT.items(): assert by_pool.get(p, 0) == c, ("pool count", p, by_pool.get(p, 0), c)
assert by_pool.get("fs5kt", 0) == len(new), by_pool
id_text = sum(1 for i in selids if BAD.search(i)); assert id_text == 0, ("hellaswag / cfcolor in an id", id_text)
excl = set(lines(excl_f)); in_excl = len(sset & excl); assert in_excl == 0, ("excluded ids present", in_excl)
fam = collections.Counter(); man_bad = 0; man_seen = 0; swag = 0; colour = 0
for mf in mans.split(","):
    for l in open(mf):
        x = json.loads(l)
        if x.get("manifest_header") or x.get("exposure_id") not in sset: continue
        man_seen += 1; s = f"{x.get('source')} {x.get('source_family')} {x.get('dataset', '')}"; man_bad += bool(BAD.search(s)); swag += "swag" in s.lower(); colour += bool(re.search(r"colou?r", s, re.I))
        fam[(("replay:" if x["exposure_id"] in kt else "h0:" if not x["exposure_id"].startswith("fs5h:") else "h:") + str(x.get("source_family")))] += 1
assert man_seen == len(sel), ("selected rows found in the manifests", man_seen, len(sel)); assert man_bad == 0, ("hellaswag / cfcolor in a manifest source", man_bad)
# ---- write (kl_pick.py's arrays)
ln_all = np.diff(m["off"]); K_all = m["K"].astype(np.int64); mask = np.zeros(n, bool); mask[sel] = True
tok = ids[np.repeat(mask, ln_all)].astype(np.int32); ln = ln_all[sel]; off = np.concatenate([[0], np.cumsum(ln)])
K = m["K"][sel]; ooff = np.concatenate([[0], np.cumsum(K.astype(np.int64))]); om = np.repeat(mask, K_all)
gold, q, hq, w = m["gold"][om], m["q"][om], m["hasq"][sel], m["w"][sel]
assert len(tok) == off[-1] and len(gold) == ooff[-1] == len(q) and hq.all()
hs = hashlib.sha256()
for x in (tok, off, K, ooff, gold, q, hq, w): hs.update(np.ascontiguousarray(x).tobytes())
np.save(out + ".ids.npy", tok); np.savez(out + ".meta.npz", off=off, K=K, w=w, ooff=ooff, gold=gold, q=q, hasq=hq)
open(out + ".rowids.txt", "w").write("\n".join(selids) + "\n")
tokp = collections.Counter()
for p, t in zip(pools[sel].tolist(), ln.tolist()): tokp[p] += int(t)
top = lambda pre: {k[len(pre):]: v for k, v in fam.most_common() if k.startswith(pre)}
info = {"data_sha256": hs.hexdigest(), "rows": int(len(sel)), "teacher_rows": int(hq.sum()), "tokens": int(ln.sum()), "len_max": int(ln.max()), "over_8192": int((ln > 8192).sum()),
        "weight": float(w.sum()), "K_max": int(K.max()), "rows_fs5kt": int(is_kt[sel].sum()), "rows_h0": int((~is_kt[sel] & ~is_h[sel]).sum()), "rows_h": int(is_h[sel].sum()), "rows_nc": 0,
        "subset_of": json.load(open(src + ".info.json"))["data_sha256"], "template": "karpo-blocks-v1", "karpo_code": "a6fa38bf",
        "pick": {"rule": "fs5hn's rule without the hellaswag pool (kl_pick24.py)", "seed": seed, "src_rows": n, "src_pools": src_counts, "new_rows": int(len(new)), "replay": int(len(rep)), "replay_pool": int(len(pool))},
        "pools": {p: int(by_pool.get(p, 0)) for p in ("h0", "sata_mrc", "sata_goemo", "gsm8k_ag", "banking77", "hellaswag", "cfcolor", "fs5kt")}, "pool_tokens": dict(tokp),
        "checks": {"hellaswag_rows_by_pool": 0, "cfcolor_rows_by_pool": 0, "ids_matching_hellaswag_or_cfcolor": id_text, "manifest_sources_matching_hellaswag_or_cfcolor": man_bad,
                   "manifest_sources_containing_swag": swag, "manifest_sources_containing_colour": colour, "excluded_ids_present": in_excl, "excluded_ids_listed": len(excl), "selected_rows_found_in_manifests": man_seen},
        "h0_by_source_family": dict(list(top("h0:").items())[:25]), "h_by_source_family": top("h:"), "replay_source_families": len(top("replay:")), "replay_top_source_families": dict(list(top("replay:").items())[:12])}
json.dump(info, open(out + ".info.json", "w"), indent=1)
print("KLPICK24", json.dumps({k: info[k] for k in ("rows", "tokens", "len_max", "over_8192", "K_max", "rows_fs5kt", "rows_h0", "rows_h", "pools", "pool_tokens", "checks", "pick", "data_sha256")}))
