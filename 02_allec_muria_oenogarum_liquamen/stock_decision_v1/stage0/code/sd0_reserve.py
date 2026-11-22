#!/usr/bin/env python3
"""Stage 0 M1 reserve (stock_decision_v1): freeze up to N_RESERVE accepted M1 decisions (the pilot tranche first), drawn from the
pilot pool of M1-eligible units with the same category shares and caps scaled to the reserve (2% source, 6% family, 2% generator
of N_RESERVE), at most 4 decisions per parent, seeded hash order. Not rendered here (only the pilot tranche is rendered).

  python3 sd0_reserve.py --units units_pools.parquet --pilot pilot_M1.parquet --config pilot_config.json --n 200000 --out DIR
"""
import argparse
import collections
import hashlib
import json
import os

import pandas as pd


def h(*x):
    return hashlib.sha256("|".join(map(str, x)).encode()).hexdigest()


def capped_fill(supply, groups, total):
    """water-fill `total` over sources with per-source upper bounds `supply`, then clip every group (list of (members, cap)) and
    redistribute until stable"""
    ub = dict(supply)
    for _ in range(50):
        alloc = {k: 0 for k in ub}
        left, active = total, sorted(k for k in ub if ub[k] > 0)
        while left > 0 and active:
            share = max(1, left // len(active))
            nxt = []
            for k in active:
                g = min(share, ub[k] - alloc[k], left)
                alloc[k] += g
                left -= g
                if alloc[k] < ub[k]:
                    nxt.append(k)
                if left == 0:
                    break
            active = nxt
        changed = False
        for mem, cap in groups:
            s = sum(alloc.get(k, 0) for k in mem)
            if s > cap:
                changed = True
                scale = cap / s
                for k in mem:
                    ub[k] = min(ub[k], int(alloc.get(k, 0) * scale))
        if not changed:
            return alloc
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    seed = cfg["seed"]
    d = pd.read_parquet(a.units)
    d = d[(d.pool_assign == "pilot") & d.m1_eligible].copy()
    d["_k"] = [h(seed, "reserve", u) for u in d.uid]
    d = d.sort_values("_k").groupby("parent_key", sort=False).head(4)
    pilot = set(pd.read_parquet(a.pilot).uid)
    N = a.n
    caps = {"source": int(0.02 * N), "family": int(0.06 * N), "generator": int(0.02 * N)}
    out_uids = []
    rep = {"n_target": N, "caps": caps, "categories": {}}
    for cat, share in sorted(cfg["m1_shares"].items()):
        tot = int(round(share * N))
        g = d[d.m1_category == cat]
        sup = g.groupby("upstream_source").size().to_dict()
        groups = [(list(x.upstream_source.unique()), caps["family"]) for _, x in g.groupby("canonical_family")]
        groups += [(list(x.upstream_source.unique()), caps["generator"]) for _, x in g.dropna(subset=["generator_mechanism"]).groupby("generator_mechanism")]
        alloc = capped_fill({s: min(v, caps["source"]) for s, v in sup.items()}, groups, tot)
        got = 0
        for s, n in sorted(alloc.items()):
            gs = g[g.upstream_source == s]
            first = gs[gs.uid.isin(pilot)]
            rest = gs[~gs.uid.isin(pilot)]
            take = list(first.uid) + list(rest.uid[: max(0, n - len(first))])
            out_uids += take
            got += len(take)
        rep["categories"][cat] = {"target": tot, "allocated": got, "sources": len(alloc)}
    sel = d.set_index("uid").loc[sorted(set(out_uids) | pilot & set(d.uid))].reset_index()
    rep["n"] = int(len(sel))
    rep["pilot_included"] = int(sel.uid.isin(pilot).sum())
    rep["by_category"] = sel.groupby("m1_category").size().to_dict()
    rep["max_source"] = int(sel.groupby("upstream_source").size().max())
    os.makedirs(a.out, exist_ok=True)
    sel.drop(columns=["_k"]).to_parquet(os.path.join(a.out, "m1_reserve.parquet"), index=False)
    json.dump(rep, open(os.path.join(a.out, "m1_reserve_report.json"), "w"), indent=1, default=str)
    print(json.dumps(rep, default=str)[:1500])


if __name__ == "__main__":
    main()
