#!/usr/bin/env python3
"""Family-held head-fit / calibration pair for the final head (stock_decision_v1): head_fit_famsplit_clean2 + cal_famheld_clean2.

Why: cal_select shared ~all of head_fit's families, so head / temperature selection on it rewards family-specific fitting.
Here the training-side families are split into two DISJOINT sets, H (head) and C (calibration), at the level of family groups:
a family cluster (sd0_families: families linked through shared components), with the generator suites bayes_families/* and
garas_synth/* each held as one group. The split is a seeded MILP (scipy / HiGHS): minimise the decision shortfall of both pools in
every stratum (kind, K bucket, length bucket, area) given the supply on each side, with C taking frac_lo..frac_hi of the groups in
every area and about target_frac overall (seeded hash tie-break).
  head_fit_famsplit_clean2: head_fit_clean2 rows of H groups are kept; rows of C groups are replaced from H supply in the same cell
    (kind, K, length, hard, tier, mixture half, M1 category); fallback the same stratum; new components only, <= 4 per parent.
  cal_famheld_clean2: cal_size decisions from C groups, stratum shares = head_fit_clean2's (largest remainder); inside a stratum
    water-filled equally over C families, then hash order, new parents first, <= 4 decisions per parent.
  Shortfalls fall back level by level: cell+area (head only) -> stratum -> (kind, K, area) -> (kind, area) -> kind.
Supply on both sides: units of the head_fit / cal_select (r2 and clean2) components and of the free pilot reserve (components not
used by M0, M1, M1 reserve or M2), excl == '', licence-registry line clean (NC / quarantine / unmapped out; unresolved terms kept),
not an excluded family. A drawn decision brings its whole component. Components of the old head / cal pools that are not reused
go back to the pilot reserve. Gates: family (and group) disjointness, component disjointness of all owner groups, licence sweep,
0 intersection with training selections. The r2 and clean2 pools are not modified.
  python3 sd0_famheld.py --units units_pools_clean2.parquet --families families.parquet --registry license_registry.json
      --exclude-family doc_nli --exclude-family goal_step_wikihow --used NAME=uids.txt ... --out DIR"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT = ["kind", "k_bucket", "len_bucket", "area"]
CELL = ["kind", "k_bucket", "len_bucket", "hard", "tier", "half", "m1_category"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def h(*x):
    return hashlib.sha256("|".join(map(str, x)).encode()).hexdigest()


def group_of(fam, cluster):
    for suite in ("bayes_families/", "garas_synth/"):
        if fam.startswith(suite):
            return suite.rstrip("/")
    return "cl:" + str(cluster)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--exclude-family", action="append", required=True)
    ap.add_argument("--used", action="append", required=True)
    ap.add_argument("--head-size", type=int, default=16384)
    ap.add_argument("--cal-size", type=int, default=4086)
    ap.add_argument("--target-frac", type=float, default=0.30)
    ap.add_argument("--frac-lo", type=float, default=0.20)
    ap.add_argument("--frac-hi", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    SP, CH = load("sd0_pools"), load("sd0_clean_heads")
    R = json.load(open(a.registry))["entries"]
    d = pd.read_parquet(a.units)
    fam = pd.read_parquet(a.families).set_index("canonical_family")
    excl = set(a.exclude_family)
    d["half"] = d.m0_root.map({True: "M0", False: "M1"})
    d["lic_line"] = d.root.map(lambda r: (R.get(CH.reg_key(r, R)) or {}).get("line", "unmapped"))
    d["group"] = [group_of(f, fam.cluster.get(f, f)) for f in d.canonical_family.astype(str)]
    used = set()
    for spec in a.used:
        used |= {x.strip() for x in open(spec.split("=", 1)[1]) if x.strip()}
    used_comps = set(d.loc[d.uid.isin(used), "comp"])
    old_tags = {"head_fit_clean2", "head_fit_clean2_unused", "cal_select_clean2", "cal_select_clean2_unused", "head_fit_unused",
                "cal_select_unused", "head_fit_clean2_removed", "cal_select_clean2_removed"}
    old_comps = set(d.loc[d.pool_assign_clean.isin(old_tags), "comp"])
    free_comps = set(d.loc[d.pool_assign_clean == "pilot", "comp"]) - used_comps
    if old_comps & used_comps:
        raise SystemExit("old head / cal components intersect training selections")
    ownable = old_comps | free_comps
    unit_ok = (d.excl == "") & (d.lic_line == "clean") & (d.licence_line == "clean") & ~d.canonical_family.isin(excl)
    cand = d[d.comp.isin(ownable) & unit_ok & d.pool_assign_clean.isin(old_tags | {"pilot"})].copy()
    cand["_k"] = [h(a.seed, x) for x in cand.uid]
    cand["strat"] = cand[STRAT].astype(str).agg("|".join, axis=1)
    cand["cell"] = cand[CELL].astype(str).agg("|".join, axis=1)
    hf0 = cand[cand.pool_assign_clean == "head_fit_clean2"]
    if len(hf0) != int((d.pool_assign_clean == "head_fit_clean2").sum()):
        raise SystemExit("head_fit_clean2 rows fail the unit filters")
    hgroups = set(hf0.group)
    rep0 = {"families_outside_head_fit_clean2_not_used": sorted(set(cand.loc[~cand.group.isin(hgroups), "canonical_family"]))}
    cand = cand[cand.group.isin(hgroups)]               # the family universe is head_fit_clean2's (no new families on either side)
    groups = sorted(hgroups)
    # supply per group per stratum: <= 4 decisions per parent
    cand["_r"] = cand.sort_values("_k").groupby(["pool", "parent_key"]).cumcount().reindex(cand.index)
    sup = cand[cand._r < 4].groupby(["group", "strat"]).size().unstack(fill_value=0).reindex(groups, fill_value=0)
    Th = hf0.strat.value_counts().to_dict()
    scale = a.head_size / len(hf0)
    Th = SP.lr_alloc({s: v for s, v in Th.items()}, a.head_size) if scale != 1 else Th
    Tc = SP.lr_alloc(hf0.strat.value_counts().to_dict(), a.cal_size)
    strata = sorted(set(Th) | set(Tc))
    sup = sup.reindex(columns=strata, fill_value=0)
    garea = hf0.groupby("group").area.agg(lambda x: x.value_counts().index[0])
    garea = pd.Series({g: garea.get(g, cand[cand.group == g].area.mode().iloc[0]) for g in groups})
    G, S = len(groups), len(strata)
    A = sup.values.astype(float)                       # G x S
    # variables: x (G, binary), yc (S), yh (S), dpos, dneg
    nv = G + 2 * S + 2
    c = np.zeros(nv)
    c[G:G + 2 * S] = 1000.0
    c[G + 2 * S:] = 1.0
    c[:G] = [int(h(a.seed, "tie", g)[:6], 16) / 16 ** 6 * 1e-3 for g in groups]
    rows, lo, hi = [], [], []
    for j, s in enumerate(strata):
        r = np.zeros(nv); r[:G] = A[:, j]; r[G + j] = 1; rows.append(r); lo.append(Tc.get(s, 0)); hi.append(np.inf)
        r = np.zeros(nv); r[:G] = -A[:, j]; r[G + S + j] = 1; rows.append(r); lo.append(Th.get(s, 0) - A[:, j].sum()); hi.append(np.inf)
    area_n = garea[garea.index.isin([g for g in groups if g in set(hf0.group)])].value_counts().to_dict()
    for ar, n in area_n.items():
        r = np.zeros(nv)
        for i, g in enumerate(groups):
            if garea[g] == ar and g in set(hf0.group):
                r[i] = 1
        rows.append(r); lo.append(max(1, int(np.floor(a.frac_lo * n)))); hi.append(max(1, int(np.ceil(a.frac_hi * n))))
    r = np.zeros(nv); r[:G] = 1; r[G + 2 * S] = -1; r[G + 2 * S + 1] = 1
    tgt = round(a.target_frac * len(set(hf0.group)))
    rows.append(r); lo.append(tgt); hi.append(tgt)
    integrality = np.zeros(nv); integrality[:G] = 1
    ub = np.full(nv, np.inf); ub[:G] = 1
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality, bounds=Bounds(np.zeros(nv), ub),
               options={"time_limit": 600})
    if res.x is None:
        raise SystemExit(f"MILP failed: {res.message}")
    x = np.round(res.x[:G]).astype(int)
    C = {g for g, v in zip(groups, x) if v}
    H = set(groups) - C
    yc = dict(zip(strata, np.round(res.x[G:G + S], 3)))
    yh = dict(zip(strata, np.round(res.x[G + S:G + 2 * S], 3)))
    rep = {**rep0, "seed": a.seed, "sizes": {"head": a.head_size, "cal": a.cal_size}, "milp": {"status": res.message, "objective": float(res.fun)},
           "groups": {"total": G, "C": len(C), "H": len(H), "C_list": sorted(C)},
           "milp_shortfall": {"cal": {s: v for s, v in yc.items() if v > 0}, "head": {s: v for s, v in yh.items() if v > 0}},
           "area_groups": {ar: {"groups_in_head": int(n), "C": int(sum(1 for g in C if garea[g] == ar))} for ar, n in area_n.items()}}
    print(json.dumps({k: rep[k] for k in ("milp", "groups", "milp_shortfall", "area_groups")}, default=str)[:3000])
    if a.dry:
        json.dump(rep, open(os.path.join(a.out, "famheld_dry.json"), "w"), indent=1, default=str)
        return
    LV = ["L0", "L1", "L2", "L3", "L4"]                # cell+area -> stratum -> (kind, K, area) -> (kind, area) -> kind
    for df in (cand,):
        df["L0"] = df[CELL + ["area"]].astype(str).agg("|".join, axis=1)
        df["L1"] = df.strat
        df["L2"] = df[["kind", "k_bucket", "area"]].astype(str).agg("|".join, axis=1)
        df["L3"] = df[["kind", "area"]].astype(str).agg("|".join, axis=1)
        df["L4"] = df.kind.astype(str)
    hf0 = cand[cand.pool_assign_clean == "head_fit_clean2"]
    proj = [dict(cand[[LV[i], LV[i + 1]]].drop_duplicates().values) for i in range(4)]
    per_parent = collections.Counter()
    taken = set()

    def draw(sup, need, start, one_per_comp, wf):
        """draw need[key at level start] decisions from sup; shortfalls fall back to coarser levels. Returns picks, fallback, short."""
        picks, fallback, short = [], collections.Counter(), dict(need)
        final = {}
        for li in range(start, 5):
            key = LV[li]
            nxt = collections.Counter()
            for k, n in sorted(short.items()):
                if n <= 0:
                    continue
                cs = sup[(sup[key] == k) & ~sup.index.isin(picks)]
                order = []
                if wf:
                    alloc = SP.waterfill(cs.canonical_family.value_counts().to_dict(), n)
                    for f, q in sorted(alloc.items()):
                        order.append(cs[cs.canonical_family == f].sort_values(["_r", "_k"]).head(4 * q + 50))
                    order.append(cs.sort_values(["_r", "_k"]))
                    quota = dict(alloc)
                else:
                    order.append(cs.sort_values("_k"))
                    quota = None
                got = 0
                fam_got = collections.Counter()
                for pi, part in enumerate(order):
                    last = pi == len(order) - 1
                    for r in part.itertuples():
                        if got >= n:
                            break
                        if quota is not None and not last and fam_got[r.canonical_family] >= quota.get(r.canonical_family, 0):
                            break
                        if r.Index in picked_set:
                            continue
                        if (one_per_comp and r.comp in taken) or per_parent[(r.pool, r.parent_key)] >= 4:
                            continue
                        taken.add(r.comp); per_parent[(r.pool, r.parent_key)] += 1; picks.append(r.Index); picked_set.add(r.Index)
                        fam_got[r.canonical_family] += 1; got += 1
                if li > start and got:
                    fallback[f"{key}:{k}"] += got
                if got < n:
                    if li < 4:
                        nxt[proj[li][k]] += n - got
                    else:
                        final[k] = n - got
            short = nxt
        return picks, dict(fallback), final

    picked_set = set()
    # ---- head: keep H rows, refill C rows from H supply (new components only)
    keep = hf0[hf0.group.isin(H)]
    drop = hf0[hf0.group.isin(C)]
    taken |= set(keep.comp)
    per_parent.update(zip(keep.pool, keep.parent_key))
    hsup = cand[cand.group.isin(H) & ~cand.comp.isin(taken)]
    hp, hfb, hshort = draw(hsup, drop.L0.value_counts().to_dict(), 0, True, False)
    head = pd.concat([keep, cand.loc[hp]])
    rep["head"] = {"kept": int(len(keep)), "replaced": int(len(drop)), "refilled": int(len(hp)),
                   "refilled_same_cell": int(len(hp) - sum(hfb.values())), "fallback": hfb, "unfilled": hshort}
    # ---- cal: from C groups, stratum targets, water-filled over families, new parents first
    csup = cand[cand.group.isin(C) & ~cand.comp.isin(taken)].copy()
    csup["_r"] = csup.sort_values("_k").groupby(["pool", "parent_key"]).cumcount().reindex(csup.index)
    csup = csup[csup._r < 4]
    cp, cfb, cshort = draw(csup, Tc, 1, False, True)
    cal = cand.loc[cp]
    rep["cal"] = {"target": a.cal_size, "drawn": int(len(cal)), "same_stratum": int(len(cp) - sum(cfb.values())), "fallback": cfb, "unfilled": cshort}
    # ---- assignment
    col = "pool_assign_fs"
    d[col] = d.pool_assign_clean.where(~d.pool_assign_clean.isin(old_tags), "pilot")   # old head / cal units back to the reserve
    for tag, sel in (("head_fit_famsplit_clean2", head), ("cal_famheld_clean2", cal)):
        d.loc[d.comp.isin(set(sel.comp)) & (d[col] == "pilot"), col] = tag + "_unused"
        d.loc[d.uid.isin(set(sel.uid)), col] = tag
    hcomp, ccomp = set(head.comp), set(cal.comp)
    if hcomp & ccomp:
        raise SystemExit("head and cal share components")
    groups_v = {"dev": ("dev_select", "dev_select_unused", "dev_select_family_unused"), "confirm_seal": ("confirm_seal", "confirm_seal_unused"),
                "confirm_seal_reserve": ("confirm_seal_reserve", "confirm_seal_reserve_unused"), "confirm_family_unused": ("confirm_seal_family_unused",),
                "head_fit": ("head_fit_famsplit_clean2", "head_fit_famsplit_clean2_unused"), "cal_select": ("cal_famheld_clean2", "cal_famheld_clean2_unused"),
                "discovery": ("discovery", "discovery_unused"), "training": ("pilot",)}
    rep["component_disjointness"] = SP.validate_disjoint(d.assign(pool_assign=d[col]), groups=groups_v)
    rep["vs_training_selections_component_intersection"] = len((hcomp | ccomp) & used_comps)
    if rep["vs_training_selections_component_intersection"]:
        raise SystemExit("new pools share components with a training selection")
    hf_f, cal_f = set(head.canonical_family), set(cal.canonical_family)
    rep["family_disjoint"] = {"head_families": len(hf_f), "cal_families": len(cal_f), "shared_families": sorted(hf_f & cal_f),
                              "head_groups": int(head.group.nunique()), "cal_groups": int(cal.group.nunique()),
                              "shared_groups": sorted(set(head.group) & set(cal.group)),
                              "head_clusters": int(head.canonical_family.map(fam.cluster).nunique()),
                              "cal_clusters": int(cal.canonical_family.map(fam.cluster).nunique())}
    if rep["family_disjoint"]["shared_families"] or rep["family_disjoint"]["shared_groups"]:
        raise SystemExit("family overlap between head and cal")
    rep["licence_sweep"] = CH.licence_sweep(d, R, ["head_fit_famsplit_clean2", "cal_famheld_clean2"], col)
    if any(v["NC"] or v["quarantine"] or v["unmapped"] for v in rep["licence_sweep"].values()):
        raise SystemExit("licence sweep failed")
    rep["excluded_family_rows"] = int(head.canonical_family.isin(excl).sum() + cal.canonical_family.isin(excl).sum())

    def dist(x):
        return (x.strat.value_counts() / len(x)) if len(x) else pd.Series(dtype=float)
    dh, dc, d0 = dist(head), dist(cal), dist(hf0)
    idx = sorted(set(dh.index) | set(dc.index) | set(d0.index))
    tv = lambda p, q: float(0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())
    rep["strata_match"] = {"tv_cal_vs_head": tv(dc, dh), "tv_head_vs_head_fit_clean2": tv(dh, d0), "tv_cal_vs_head_fit_clean2": tv(dc, d0)}
    for nm, sel in (("head", head), ("cal", cal)):
        rep[nm].update({"decisions": int(len(sel)), "parents": int(sel.groupby(["pool", "parent_key"]).ngroups),
                        "families": int(sel.canonical_family.nunique()), "max_family_share": float(sel.canonical_family.value_counts(normalize=True).iloc[0]),
                        "top_families": {k: int(v) for k, v in sel.canonical_family.value_counts().head(8).items()},
                        "by_area": {k: int(v) for k, v in sel.area.value_counts().items()},
                        "by_kind": {k: int(v) for k, v in sel.kind.value_counts().items()},
                        "families_by_area": {k: int(v) for k, v in sel.groupby("area").canonical_family.nunique().items()}})
    rep["cal"]["family_list"] = sorted(cal_f)
    tmp = ["half", "lic_line", "group"]
    d.drop(columns=tmp).to_parquet(os.path.join(a.out, "units_pools_famsplit.parquet"), index=False)
    for tag in ("head_fit_famsplit_clean2", "cal_famheld_clean2"):
        d[d[col] == tag].drop(columns=tmp).to_parquet(os.path.join(a.out, f"sel_{tag}.parquet"), index=False)
    json.dump(rep, open(os.path.join(a.out, "famheld_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("head", "cal", "family_disjoint", "strata_match", "licence_sweep")}, default=str)[:4000])


if __name__ == "__main__":
    main()
