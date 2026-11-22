#!/usr/bin/env python3
"""Stage 0 step A (stock_decision_v1): apply the identity checks to the candidate table and build the family table.

  1. base eligibility (sd0_candidates.base_ok);
  2. protected overlap (REQUIRED gate): every included pool must have a COMPLETE scan-manifest entry (sd0_scangate.py) and its
     hits file (md5-checked); any STRONG hit of the all-field protected index -> excluded. Missing scans refuse the run;
  3. exact duplicate decisions (same state, instructions, kind, options and gold): keep the first uid in sorted order;
     conflicting duplicates (same content, different gold): every copy quarantined;
  4. ancestry components (sd0_components) joined on the record key;
  5. family table: per canonical family -> units, parents, components, area, pools, M0 / M1 roles, DI ancestry, label basis;
     family links: two families are linked when they share >= LINK_MIN components AND >= LINK_FRAC of the smaller family's
     components (a single shared Wikipedia sentence does not merge two corpora; a shared document collection does);
     independent family clusters = connected components of the link graph. Weaker cross-family contacts are 'bridges': those
     records are removed from the training side once a family is reserved for development / confirmation.
Outputs: units_eligible.parquet (uid-level, with excl_reason and comp), families.parquet, family_links.json.

  python3 sd0_families.py --cand cand_core.parquet [--cand cand_ts.parquet] --comp components.parquet --dec DIR --pools ... \
        [--hits DIR] --out DIR
"""
import argparse
import collections
import gzip
import hashlib
import json
import os

import numpy as np
import pandas as pd

LINK_MIN, LINK_FRAC = 3, 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--hits", required=True)
    ap.add_argument("--scan-manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.concat([pd.read_parquet(p) for p in a.cand], ignore_index=True)
    d["excl"] = np.where(d.base_ok, "", d.base_reason)
    # 2. protected overlap: REQUIRED. Every included pool needs a COMPLETE entry in the scan manifest (sd0_scangate.py) and its
    #    hits file; a missing scan is a refusal, never 'no hits'.
    sm = json.load(open(a.scan_manifest))
    pools_in = sorted(set(d.pool))
    bad = {t: (sm.get("pools", {}).get(t) or {}).get("reasons") or ["no scan-manifest entry"] for t in pools_in
           if (sm.get("pools", {}).get(t) or {}).get("status") != "COMPLETE"}
    if bad:
        raise SystemExit(f"protected-overlap gate: incomplete or missing scans {bad}")
    strong = set()
    for t in pools_in:
        p = os.path.join(a.hits, f"hits_{t}.jsonl.gz")
        if not os.path.exists(p):
            raise SystemExit(f"protected-overlap gate: hits file missing for {t}")
        h_ = hashlib.md5(open(p, "rb").read()).hexdigest()
        if h_ != sm["pools"][t]["hits"]["md5_local"]:
            raise SystemExit(f"protected-overlap gate: {t} hits md5 {h_} != scan manifest")
        with gzip.open(p, "rt") as g:
            for l in g:
                r = json.loads(l)
                if r.get("s"):
                    strong.add(r["uid"])
    m = (d.excl == "") & d.uid.isin(strong)
    d.loc[m, "excl"] = "protected_overlap"
    # 3. duplicates / conflicts
    dec = []
    for t in a.pools.split(","):
        with gzip.open(os.path.join(a.dec, f"dec_{t}.jsonl.gz"), "rt") as g:
            dec += [json.loads(l) for l in g]
    dec = pd.DataFrame(dec)[["uid", "dec", "dec_nogold"]]
    d = d.merge(dec, on="uid", how="left")
    live = d.excl == ""
    ng = d[live].groupby("dec_nogold")["dec"].nunique()
    conflict = set(ng[ng > 1].index)
    m = live & d.dec_nogold.isin(conflict)
    d.loc[m, "excl"] = "conflicting_target"
    live = d.excl == ""
    d = d.sort_values("uid", kind="stable")
    dup = live & d.duplicated("dec", keep="first") & d.dec.notna()
    d.loc[dup, "excl"] = "exact_duplicate"
    # 4. components
    comp = pd.read_parquet(a.comp)[["rk", "comp"]]
    d = d.merge(comp, on="rk", how="left")
    if d.comp.isna().any():
        raise SystemExit(f"{int(d.comp.isna().sum())} units without a component")
    d["comp"] = d.comp.astype(np.int64)
    # 5. families
    e = d[d.excl == ""]
    fam = e.groupby("canonical_family").agg(units=("uid", "size"), parents=("parent_key", "nunique"), comps=("comp", "nunique"),
                                            area=("area", "first"), sources=("upstream_source", "nunique"), pools=("pool", lambda x: ",".join(sorted(set(x)))),
                                            m0=("m0_root", "max"), m1=("m1_eligible", "max"), m1_category=("m1_category", "first"),
                                            di=("di_ancestry", "max"), basis=("label_basis", "first"),
                                            mech=("generator_mechanism", lambda x: ",".join(sorted(set(str(v) for v in x if v)))))
    # family links through shared components
    cf = e[["comp", "canonical_family"]].drop_duplicates()
    multi = cf.groupby("comp").canonical_family.nunique()
    cf = cf[cf.comp.isin(multi[multi > 1].index)]
    pair = collections.Counter()
    for c, g in cf.groupby("comp"):
        fs = sorted(g.canonical_family)
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                pair[(fs[i], fs[j])] += 1
    links, bridges = [], []
    for (f1, f2), n in pair.items():
        small = min(fam.at[f1, "comps"], fam.at[f2, "comps"])
        rec = {"f1": f1, "f2": f2, "shared_components": n, "frac_of_smaller": round(n / small, 5)}
        (links if n >= LINK_MIN and n / small >= LINK_FRAC else bridges).append(rec)
    par = {f: f for f in fam.index}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for r in links:
        par[find(r["f1"])] = find(r["f2"])
    fam["cluster"] = [find(f) for f in fam.index]
    cl = fam.groupby("cluster").size()
    fam["cluster_size"] = fam.cluster.map(cl)
    d.to_parquet(os.path.join(a.out, "units_eligible.parquet"), index=False)
    fam.reset_index().to_parquet(os.path.join(a.out, "families.parquet"), index=False)
    summ = {"rules": {"LINK_MIN": LINK_MIN, "LINK_FRAC": LINK_FRAC}, "units": int(len(d)), "eligible": int((d.excl == "").sum()),
            "exclusions": d.loc[d.excl != "", "excl"].str.split(";").explode().value_counts().to_dict(),
            "exclusions_by_pool": {k: v for k, v in d[d.excl != ""].groupby(["pool", "excl"]).size().reset_index().groupby("pool")
                                   .apply(lambda g: dict(zip(g.excl, g[0]))).items()},
            "conflict_groups": len(conflict), "families": int(len(fam)), "clusters": int(fam.cluster.nunique()),
            "multi_family_clusters": {k: sorted(fam.index[fam.cluster == k]) for k in cl[cl > 1].index},
            "links": sorted(links, key=lambda r: -r["shared_components"]), "bridges_top": sorted(bridges, key=lambda r: -r["shared_components"])[:50],
            "n_bridges": len(bridges), "strong_hit_uids_loaded": len(strong)}
    json.dump(summ, open(os.path.join(a.out, "family_links.json"), "w"), indent=1, default=int)
    print(json.dumps({k: summ[k] for k in ("units", "eligible", "exclusions", "conflict_groups", "families", "clusters", "n_bridges")}, default=int))


if __name__ == "__main__":
    main()
