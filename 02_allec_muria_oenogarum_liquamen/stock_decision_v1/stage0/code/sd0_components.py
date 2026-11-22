#!/usr/bin/env python3
"""Stage 0 ancestry components over ALL candidate pools (stock_decision_v1).

Records (rk) are joined by union-find when they share, in this fixed order:
  L  lineage: the same lineage key inside a pool (meta group_id / base_id), and the same parent key;
  S  an identical normalised state of >= 8 words (same document);
  T  an identical normalised text unit of >= EDGE_WORDS (20) words: from the STATE with document frequency 2..DF_MAX (200)
     distinct lineages, or from option / criteria text with document frequency 2..DF_MAX_OPT (5); instructions never make an
     edge (they are task templates). Units above the DF limits are boilerplate / template text (counted and reported);
  N  near-duplicate states: MinHash (48 permutations, 8-word shingles, states >= 20 words), LSH 12 bands x 4 rows, a bucket of
     more than BUCKET_MAX records is skipped (template bucket, reported), candidate pairs need estimated Jaccard >= 0.8.
Decisions: exact duplicates (same state, instructions, options and gold) and conflicting duplicates (same content, different gold)
are flagged from the dec_* tables (keep-one rule applied later; conflicts are quarantined, all copies).
Output: comp_<tag>.parquet (rk, comp, pool, lineage, parent), edges summary, component statistics.
Thresholds are fixed here before any split is drawn.

  python3 sd0_components.py --keys DIR --pools p1,p2,... --out DIR
"""
import argparse
import collections
import gzip
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DF_MAX, DF_MAX_OPT, EDGE_WORDS, BUCKET_MAX, JACCARD, BANDS, ROWS = 200, 5, 20, 50, 0.8, 12, 4


def group_edges(keys, recs):
    """star edges joining every record that shares a key value (keys / recs aligned int arrays)"""
    if len(keys) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = np.argsort(keys, kind="stable")
    k, r = keys[o], recs[o]
    new = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(new) - 1
    first = r[np.flatnonzero(new)][gid]
    m = ~new
    return first[m], r[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pools = a.pools.split(",")
    rk, lin, par, pl = [], [], [], []
    sh, mh, tuh, tur, tuf, tuw = [], [], [], [], [], []
    off = 0
    for t in pools:
        with gzip.open(os.path.join(a.keys, f"recs_{t}.jsonl.gz"), "rt", encoding="utf-8") as g:
            rr = [json.loads(l) for l in g]
        d = np.load(os.path.join(a.keys, f"keys_{t}.npz"))
        rk += [x[0] for x in rr]
        lin += [f"{t}|{x[1]}" for x in rr]
        par += [f"{t}|{x[2]}" for x in rr]
        pl += [t] * len(rr)
        sh.append(d["state_hash"])
        mh.append(d["minhash"])
        tuh.append(d["tu_hash"])
        tur.append(d["tu_rec"].astype(np.int64) + off)
        tuf.append(d["tu_field"])
        tuw.append(d["tu_words"])
        off += len(rr)
        print(t, len(rr), flush=True)
    n = len(rk)
    sh = np.concatenate(sh)
    mh = np.concatenate(mh)
    tuh = np.concatenate(tuh)
    tur = np.concatenate(tur)
    tuf = np.concatenate(tuf)
    tuw = np.concatenate(tuw)
    cnt = collections.Counter()
    E1, E2 = [], []

    def add(e, name):
        E1.append(e[0])
        E2.append(e[1])
        cnt[name] += int(len(e[0]))
    # L: lineage and parent keys
    li = pd.factorize(pd.Series(lin))[0]
    add(group_edges(li, np.arange(n)), "L_lineage")
    pi = pd.factorize(pd.Series(par))[0]
    add(group_edges(pi, np.arange(n)), "L_parent")
    # S: identical states
    m = sh != 0
    add(group_edges(sh[m].view(np.int64), np.flatnonzero(m)), "S_state")
    # T: shared text units, document frequency counted in distinct lineages
    sel = (tuw >= EDGE_WORDS) & (tuf != 2)
    df = pd.DataFrame({"h": tuh[sel].view(np.int64), "l": li[tur[sel]], "r": tur[sel], "st": tuf[sel] == 1})
    g_ = df.groupby("h").agg(nl=("l", "nunique"), st=("st", "all"))
    ok_ = (g_.nl >= 2) & (((g_.st) & (g_.nl <= DF_MAX)) | ((~g_.st) & (g_.nl <= DF_MAX_OPT)))
    keep = g_.index[ok_]
    cnt["T_units_considered"] = int(len(g_))
    cnt["T_units_shared_kept"] = int(len(keep))
    cnt["T_units_boilerplate_over_df"] = int(((g_.nl >= 2) & ~ok_).sum())
    sub = df[df.h.isin(keep)]
    add(group_edges(sub.h.values, sub.r.values), "T_text_unit")
    del df, sub
    # N: near-duplicate states via LSH
    idx = np.flatnonzero(mh.any(axis=1))
    cnt["N_minhashed"] = int(len(idx))
    n1, n2 = [], []
    for b in range(BANDS):
        band = mh[idx, b * ROWS:(b + 1) * ROWS].astype(np.uint64)
        key = (band[:, 0] << np.uint64(32)) ^ (band[:, 1] << np.uint64(21)) ^ (band[:, 2] << np.uint64(11)) ^ band[:, 3]
        o = np.argsort(key, kind="stable")
        k, r = key[o], idx[o]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        ends = np.r_[starts[1:], len(k)]
        sizes = ends - starts
        for s_, e_ in zip(starts[sizes > BUCKET_MAX], ends[sizes > BUCKET_MAX]):
            cnt["N_buckets_skipped_gt_max"] += 1
            cnt["N_records_in_skipped_buckets"] += int(e_ - s_)
        for s_, e_ in zip(starts[(sizes >= 2) & (sizes <= BUCKET_MAX)], ends[(sizes >= 2) & (sizes <= BUCKET_MAX)]):
            mem = r[s_:e_]
            sig = mh[mem]
            eq = (sig[:, None, :] == sig[None, :, :]).mean(axis=2)
            ii, jj = np.nonzero(np.triu(eq >= JACCARD, 1))
            if len(ii):
                n1.append(mem[ii])
                n2.append(mem[jj])
    if n1:
        add((np.concatenate(n1), np.concatenate(n2)), "N_near_dup_pairs")
    e1 = np.concatenate(E1)
    e2 = np.concatenate(E2)
    g = coo_matrix((np.ones(len(e1), dtype=np.int8), (e1, e2)), shape=(n, n))
    ncomp, comp = connected_components(g, directed=False)
    out = pd.DataFrame({"rk": rk, "pool": pl, "lineage": lin, "parent": par, "comp": comp})
    os.makedirs(a.out, exist_ok=True)
    out.to_parquet(os.path.join(a.out, "components.parquet"), index=False)
    cs = out.groupby("comp").agg(size=("rk", "size"), pools=("pool", "nunique"), lineages=("lineage", "nunique"))
    stats = {"thresholds": {"DF_MAX": DF_MAX, "DF_MAX_OPT": DF_MAX_OPT, "EDGE_WORDS": EDGE_WORDS, "BUCKET_MAX": BUCKET_MAX, "JACCARD": JACCARD, "BANDS": BANDS, "ROWS": ROWS, "NPERM": 48,
                            "SHINGLE": 8, "MIN_TU_WORDS": 12, "MIN_STATE_WORDS": 8, "MIN_MH_WORDS": 20},
             "records": n, "components": int(len(cs)), "edges": dict(cnt),
             "largest_components": cs.sort_values("size", ascending=False).head(25).reset_index().to_dict("records"),
             "multi_pool_components": int((cs.pools > 1).sum()), "multi_lineage_components": int((cs.lineages > 1).sum())}
    json.dump(stats, open(os.path.join(a.out, "components_stats.json"), "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in stats.items() if k != "largest_components"}, default=int))


if __name__ == "__main__":
    main()
