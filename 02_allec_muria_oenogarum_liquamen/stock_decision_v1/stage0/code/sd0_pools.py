#!/usr/bin/env python3
"""Stage 0 pools (stock_decision_v1): reserve whole family clusters for development and confirmation, then carve parent-disjoint
(component-disjoint) head-fit, calibration-selection and discovery pools from the training families; everything left is the pilot
reserve. All draws are seeded (seed 20261013) hash orders; rules and sizes come from the config and are fixed before drawing.

Panels (dev_select, confirm_seal): eligible family CLUSTERS (sd0_families: families linked through shared components) whose
families are all clean-licence, not Decision-Index ancestry, not model-labelled / unverified / workflow-imitation, have >= min_parents
parents, are not in the current mixture's roots (M0) unless listed in panel_allow, and are not a scarce verified-synthetic /
exact-probability generator mechanism. Per area, clusters in hash order are dealt alternately to dev and confirm until each panel
has >= min_families_per_area clusters and enough parents for its area quota (DI 0.3 area weights x panel size, largest remainder).
Inside an area, parents are water-filled equally over the panel's families (families sampled before records), <= 4 decisions per
parent. A confirmation reserve (sealed, for enlargement) takes up to reserve_frac more parents from confirmation families. Every
other record of a reserved family is unusable for training; every record of another family that shares a component with a reserved
family is removed from the training side ('bridge').
Training pools take whole components; a component with more than comp_max_parents parents is never carved (left to the pilot).
head_fit (decisions) and cal_select (parents) mirror the two mixtures half / half by upstream-source shares; discovery (parents)
is allocated over areas by the DI 0.3 weights and water-filled over families inside an area. Per-family carve limits protect
the pilot supply.

  python3 sd0_pools.py --units units_eligible.parquet --families families.parquet --config pools_config.json --out DIR
"""
import argparse
import collections
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

AREAS = ["knowledge", "language", "retrieval", "tools", "arts"]


def h(*xs):
    return hashlib.sha256("|".join(str(x) for x in xs).encode()).hexdigest()


def lr_alloc(weights, total):
    """largest-remainder integer allocation"""
    tw = sum(weights.values())
    raw = {k: total * w / tw for k, w in weights.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def waterfill(supply, total):
    alloc = {k: 0 for k in supply}
    left = total
    active = sorted(k for k in supply if supply[k] > 0)
    while left > 0 and active:
        share = max(1, left // len(active))
        nxt = []
        for k in active:
            g = min(share, supply[k] - alloc[k], left)
            alloc[k] += g
            left -= g
            if alloc[k] < supply[k]:
                nxt.append(k)
            if left == 0:
                break
        active = nxt
    return alloc


OWNER_GROUPS = {"dev": ("dev_select", "dev_select_unused", "dev_select_family_unused"),
                "confirm_seal": ("confirm_seal", "confirm_seal_unused"),
                "confirm_seal_reserve": ("confirm_seal_reserve", "confirm_seal_reserve_unused"),
                "confirm_family_unused": ("confirm_seal_family_unused",),
                "head_fit": ("head_fit", "head_fit_unused"), "cal_select": ("cal_select", "cal_select_unused"),
                "discovery": ("discovery", "discovery_unused"), "training": ("pilot",)}


def validate_disjoint(d, groups=OWNER_GROUPS):
    """publication gate: the ancestry-component sets of dev, confirmation (main, reserve, unused family records), head-fit,
    calibration, discovery and training must be pairwise disjoint. Raises on any intersection."""
    sets = {g: set(d.loc[d.pool_assign.isin(tags), "comp"]) for g, tags in groups.items()}
    names = sorted(sets)
    bad = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            inter = sets[x] & sets[y]
            if inter:
                bad[f"{x}&{y}"] = sorted(int(c) for c in inter)[:20]
    if bad:
        raise SystemExit(f"COMPONENT DISJOINTNESS FAILED: {bad}")
    return {"pairs_checked": len(names) * (len(names) - 1) // 2, "components_by_group": {g: len(v) for g, v in sets.items()}, "intersections": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    seed = cfg["seed"]
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_parquet(a.units)
    fam = pd.read_parquet(a.families).set_index("canonical_family")
    live = d.excl == ""
    d["pool_assign"] = np.where(live, "", "excluded")
    d["pkey"] = d.pool + "|" + d.parent_key.astype(str)
    rep = {"config": cfg}
    # ---------------- panels
    bad_basis = set(cfg["panel_bad_basis"])
    allow = set(cfg.get("panel_allow", []))
    scarce = set(cfg.get("panel_exclude_mechanisms", []))
    fam["mechs"] = fam.mech.fillna("").map(lambda s: set(x for x in s.split(",") if x))
    cl = fam.reset_index().groupby("cluster").agg(families=("canonical_family", list), parents=("parents", "sum"),
                                                     di=("di", "max"), m0=("m0", "max"), m1=("m1", "max"))
    def area_of(c):
        sub = fam.loc[cl.at[c, "families"]]
        return sub.sort_values("parents", ascending=False).area.iloc[0]
    cl["area"] = [area_of(c) for c in cl.index]
    ok = []
    why = collections.Counter()
    for c, r in cl.iterrows():
        sub = fam.loc[r.families]
        reason = None
        if r.di:
            reason = "di_ancestry"
        elif sub.basis.isin(bad_basis).any():
            reason = "label_basis"
        elif any(m in scarce for ms in sub.mechs for m in ms):
            reason = "scarce_mechanism"
        elif r.m0 and not (set(r.families) & allow):
            reason = "current_mixture_root"
        elif r.parents < cfg["min_parents"]:
            reason = "too_few_parents"
        elif r.area not in AREAS:
            reason = "area_unclear"
        why[reason or "eligible"] += 1
        if reason is None:
            ok.append(c)
    rep["panel_cluster_eligibility"] = dict(why)
    psize = cfg["panel_parents"]
    aq = lr_alloc(cfg["area_weights"], psize)
    fcap = cfg["panel_family_cap"]
    minc = cfg["min_clusters_per_area"]                   # {panel: {area: n}}: INDEPENDENT family clusters, not families
    min_total = cfg.get("min_clusters_total", {})         # {panel: n}
    approved = cfg.get("approved_amendments", {})         # {"panel/area" or "panel/total": approved minimum}: set only by the coordinator
    fam_parents = d[live].groupby("canonical_family").pkey.nunique()
    panel_fams = {"dev_select": collections.defaultdict(list), "confirm_seal": collections.defaultdict(list)}
    capacity = {p: {} for p in panel_fams}
    nclu_tot = {p: 0 for p in panel_fams}
    short, stop = {}, {}
    for ar in AREAS:
        cands = sorted([c for c in ok if cl.at[c, "area"] == ar], key=lambda c: h(seed, "panel", c))
        cap = {p: 0 for p in panel_fams}
        nclu = {p: 0 for p in panel_fams}
        for c in cands:
            need = {p: nclu[p] < minc[p][ar] or cap[p] < aq[ar] for p in panel_fams}
            if not any(need.values()):
                break
            gap = {q: minc[q][ar] - nclu[q] for q in panel_fams}
            if need["dev_select"] and need["confirm_seal"]:
                p = max(panel_fams, key=lambda q: (gap[q], q == "dev_select"))
            else:
                p = "dev_select" if need["dev_select"] else "confirm_seal"
            panel_fams[p][ar].append(c)
            nclu[p] += 1                                  # one independent cluster, however many families it holds
            cap[p] += int(sum(min(fam_parents.get(f, 0), fcap) for f in cl.at[c, "families"]))
        for p in panel_fams:
            capacity[p][ar] = cap[p]
            nclu_tot[p] += nclu[p]
            if cap[p] < aq[ar]:
                short[f"{p}/{ar}/parents"] = {"parent_capacity": cap[p], "area_quota": aq[ar], "rule": "reallocated to other areas by weight"}
            if nclu[p] < minc[p][ar]:
                k = f"{p}/{ar}"
                rec = {"clusters": nclu[p], "min_clusters": minc[p][ar], "eligible_clusters_in_area": len(cands)}
                short[k] = rec
                if not (k in approved and nclu[p] >= approved[k]):
                    stop[k] = rec
    for p in panel_fams:
        if nclu_tot[p] < min_total.get(p, 0):
            k = f"{p}/total"
            rec = {"clusters": nclu_tot[p], "min_clusters": min_total[p]}
            short[k] = rec
            if not (k in approved and nclu_tot[p] >= approved[k]):
                stop[k] = rec
    rep["panel_clusters"] = {p: {ar: [{"cluster": c, "families": list(cl.at[c, "families"])} for c in cs] for ar, cs in byar.items()}
                             for p, byar in panel_fams.items()}
    rep["panel_cluster_counts"] = {p: {"total": nclu_tot[p], **{ar: len(panel_fams[p][ar]) for ar in AREAS}} for p in panel_fams}
    rep["panel_shortfalls_by_area"] = short
    if stop:
        rep["STOPPED"] = {"reason": "independent-cluster minimum not met by supply; an explicit amendment (approved_amendments) is required",
                          "unmet": stop}
        json.dump(rep, open(os.path.join(a.out, "pools_report.json"), "w"), indent=1, default=str)
        print(json.dumps({"STOPPED": stop, "panel_cluster_counts": rep["panel_cluster_counts"]}, default=str))
        raise SystemExit(3)
    reserved_fams = {}
    for p, byar in panel_fams.items():
        for ar, cs in byar.items():
            for c in cs:
                for f in cl.at[c, "families"]:
                    reserved_fams[f] = p
    d["reserved_for"] = d.canonical_family.map(reserved_fams).fillna("")
    # every ancestry component gets ONE owner: a component touching dev AND confirmation families is quarantined (both sides),
    # a component touching a reserved family and a training family loses its training-family records ('bridge'); both happen
    # BEFORE any parent is drawn, so the panel quotas are filled from what remains
    own = d[live & (d.reserved_for != "")].groupby("comp").reserved_for.nunique()
    cross = set(own[own > 1].index)
    m = live & d.comp.isin(cross)
    d.loc[m, "pool_assign"] = "cross_panel_bridge"
    rep["cross_panel_components"] = len(cross)
    rep["cross_panel_units"] = int(m.sum())
    rc = set(d.loc[live & (d.reserved_for != ""), "comp"])
    m = live & (d.pool_assign == "") & (d.reserved_for == "") & d.comp.isin(rc)
    d.loc[m, "pool_assign"] = "bridge_reserved_component"
    rep["bridge_units"] = int(m.sum())
    comp_idx = d.groupby("comp").indices                  # positional (d has a RangeIndex)
    livev = live.values
    taken_comp = set()

    def take_components(tag, mask, quota_parents, seed_tag):
        """draw parents in hash order; each drawn parent brings its WHOLE component to `tag` (<= max_sib decisions per parent,
        the rest '<tag>_unused'); returns parents taken (all parents of the taken components)"""
        u = d[mask & (d.pool_assign == "")]
        pars = sorted(set(u.pkey), key=lambda x: h(seed, seed_tag, x))
        pc = u.groupby("pkey").comp.first()
        got = 0
        for pk in pars:
            if got >= quota_parents:
                break
            c = pc[pk]
            if c in taken_comp:
                continue
            taken_comp.add(c)
            pos = comp_idx[c]
            pos = pos[livev[pos] & (d.pool_assign.values[pos] == "")]
            rows_ = d.iloc[pos].assign(_k=lambda x: [h(seed, tag, u_) for u_ in x.uid]).sort_values("_k")
            keep = rows_.groupby("pkey").head(cfg["max_sib"]).index
            d.loc[rows_.index, "pool_assign"] = tag + "_unused"
            d.loc[keep, "pool_assign"] = tag
            got += rows_.pkey.nunique()
        return got
    fam_parents_now = d[live & (d.pool_assign == "") & (d.reserved_for != "")].groupby("canonical_family").pkey.nunique()
    for p in panel_fams:                                  # refill: capacities after the quarantine
        cap_now = {ar: int(sum(min(fam_parents_now.get(f, 0), fcap) for c in panel_fams[p][ar] for f in cl.at[c, "families"])) for ar in AREAS}
        q = {ar: min(aq[ar], cap_now[ar]) for ar in AREAS}
        left = psize - sum(q.values())
        while left > 0:
            room = {ar: cap_now[ar] - q[ar] for ar in AREAS if cap_now[ar] > q[ar]}
            if not room:
                break
            add = lr_alloc({ar: cfg["area_weights"][ar] for ar in room}, left)
            moved = 0
            for ar, n in add.items():
                g = min(n, room[ar])
                q[ar] += g
                moved += g
            left -= moved
            if moved == 0:
                break
        rep[f"{p}_area_quota_after_reallocation"] = q
        rep[f"{p}_unfilled_parents"] = left
        res_quota = {}
        for ar in AREAS:
            fams_ = [f for c in panel_fams[p][ar] for f in cl.at[c, "families"]]
            if not fams_:
                continue
            alloc = waterfill({f: min(int(fam_parents_now.get(f, 0)), fcap) for f in fams_}, q[ar])
            for f, n in sorted(alloc.items()):
                if n > 0:
                    take_components(p, live & (d.canonical_family == f), n, p)
                    res_quota[f] = int(math.ceil(n * cfg["confirm_reserve_frac"])) if p == "confirm_seal" else 0
        for f, n in sorted(res_quota.items()):            # the sealed confirmation reserve: further whole components
            if n > 0:
                take_components(p + "_reserve", live & (d.canonical_family == f), n, p + "_reserve")
        # exact parent counts: trim surplus parents (their components stay owned by the panel)
        for tag in ((p, p + "_reserve") if p == "confirm_seal" else (p,)):
            s_ = d[d.pool_assign == tag]
            target = psize if tag == p else int(sum(res_quota.values()))
            extra = s_.pkey.nunique() - target
            if extra > 0:
                drop = sorted(set(s_.pkey), key=lambda x: h(seed, tag, "trim", x))[:extra]
                d.loc[d.pkey.isin(set(drop)) & (d.pool_assign == tag), "pool_assign"] = tag + "_unused"
    for p in panel_fams:
        m = live & (d.reserved_for == p) & (d.pool_assign == "")
        d.loc[m, "pool_assign"] = p + "_family_unused"
    # ---------------- training carving
    free = live & (d.pool_assign == "")
    comp_par = d[free].groupby("comp").pkey.nunique()
    big = set(comp_par[comp_par > cfg["comp_max_parents"]].index)
    rep["components_too_big_to_carve"] = len(big)
    src_parents = d[free].groupby("upstream_source").pkey.nunique()

    def carve(tag, mask, quotas_by_source, unit, limit_frac):
        """quotas_by_source: {source: n parents or decisions}; whole components; returns count taken"""
        got = collections.Counter()
        for s_, q in sorted(quotas_by_source.items()):
            lim = int(limit_frac * src_parents.get(s_, 0))
            q = min(q, lim)
            if q <= 0:
                continue
            u = d[mask & (d.upstream_source == s_) & (d.pool_assign == "") & ~d.comp.isin(big)]
            pars = sorted(set(u.pkey), key=lambda x: h(seed, tag, x))
            pc = u.groupby("pkey").comp.first()
            for pk in pars:
                if got[s_] >= q:
                    break
                c = pc[pk]
                if c in taken_comp:
                    continue
                taken_comp.add(c)
                pos = comp_idx[c]
                pos = pos[livev[pos] & (d.pool_assign.values[pos] == "")]
                rows_ = d.iloc[pos].assign(_k=lambda x: [h(seed, tag, u_) for u_ in x.uid]).sort_values("_k")
                keep = rows_.groupby("pkey").head(cfg["max_sib"]).index
                d.loc[rows_.index, "pool_assign"] = tag + "_unused"
                d.loc[keep, "pool_assign"] = tag
                got[s_] += len(keep) if unit == "decisions" else rows_.pkey.nunique()
        return got

    def mirror_quotas(mask, n, mixture):
        if mixture == "M0":
            sh = cfg["m0_recipeA_roots"]
            rs = d[mask].groupby("root").upstream_source.first()
            w = collections.Counter()
            for r_, x in sh.items():
                if r_ in rs:
                    w[rs[r_]] += x
        else:
            w = collections.Counter()
            cats = d[mask].groupby("upstream_source").m1_category.first()
            for c_, share in cfg["m1_shares"].items():
                srcs = [s_ for s_ in cats.index if cats[s_] == c_]
                for s_ in srcs:
                    w[s_] += share / max(len(srcs), 1)
        return lr_alloc(dict(w), n) if w else {}
    m0mask = free & d.m0_root
    m1mask = free & d.m1_eligible

    def exact(tag, mixmask, target, unit, frac, mixture):
        """mirror the mixture's source shares, then fill to `target` from any source of that mixture (frac relaxed x1.5), then trim"""
        got = carve(tag, mixmask, mirror_quotas(mixmask, target, mixture), unit, frac)
        def count():
            s_ = d[(d.pool_assign == tag) & mixmask]
            return len(s_) if unit == "decisions" else s_.pkey.nunique()
        n = count()
        if n < target:
            srcs = d[mixmask & (d.pool_assign == "")].groupby("upstream_source").pkey.nunique()
            more = waterfill({s_: int(v) for s_, v in srcs.items()}, 3 * (target - n))
            carve(tag, mixmask, more, unit, frac * 1.5)
            n = count()
        if n > target:
            s_ = d[(d.pool_assign == tag) & mixmask]
            if unit == "decisions":
                drop = sorted(s_.uid, key=lambda u_: h(seed, tag, "trim", u_))[: n - target]
                d.loc[d.uid.isin(set(drop)) & (d.pool_assign == tag), "pool_assign"] = tag + "_unused"
            else:
                drop = sorted(set(s_.pkey), key=lambda x: h(seed, tag, "trim", x))[: n - target]
                d.loc[d.pkey.isin(set(drop)) & (d.pool_assign == tag) & mixmask, "pool_assign"] = tag + "_unused"
        return count()
    rep["head_fit"] = {"M0": exact("head_fit", m0mask, cfg["head_fit_decisions"] // 2, "decisions", cfg["head_fit_frac"], "M0")}
    rep["head_fit"]["M1"] = exact("head_fit", m1mask & ~m0mask, cfg["head_fit_decisions"] - rep["head_fit"]["M0"], "decisions", cfg["head_fit_frac"], "M1")
    rep["cal_select"] = {"M0": exact("cal_select", m0mask, cfg["cal_parents"] // 2, "parents", cfg["cal_frac"], "M0")}
    rep["cal_select"]["M1"] = exact("cal_select", m1mask & ~m0mask, cfg["cal_parents"] - rep["cal_select"]["M0"], "parents", cfg["cal_frac"], "M1")
    def trim_total(tag, target, unit):
        s_ = d[d.pool_assign == tag]
        n = len(s_) if unit == "decisions" else s_.pkey.nunique()
        if n > target:
            if unit == "decisions":
                drop = sorted(s_.uid, key=lambda u_: h(seed, tag, "trim_total", u_))[: n - target]
                d.loc[d.uid.isin(set(drop)) & (d.pool_assign == tag), "pool_assign"] = tag + "_unused"
            else:
                drop = sorted(set(s_.pkey), key=lambda x: h(seed, tag, "trim_total", x))[: n - target]
                d.loc[d.pkey.isin(set(drop)) & (d.pool_assign == tag), "pool_assign"] = tag + "_unused"
        s_ = d[d.pool_assign == tag]
        return len(s_) if unit == "decisions" else s_.pkey.nunique()
    rep["head_fit_total"] = trim_total("head_fit", cfg["head_fit_decisions"], "decisions")
    rep["cal_select_total"] = trim_total("cal_select", cfg["cal_parents"], "parents")
    disc_mask = free & ~d.label_basis.isin(set(cfg["discovery_bad_basis"])) & ~d.di_ancestry & (d.m0_root | d.m1_eligible)
    dq = lr_alloc(cfg["area_weights"], cfg["discovery_parents"])
    rep["discovery_alloc_initial"] = dict(dq)
    disc_n = {}

    def disc_count(ar):
        s_ = d[(d.pool_assign == "discovery") & (d.area == ar)]
        return s_.pkey.nunique()
    for rnd in range(3):
        for ar in AREAS:
            want = dq[ar] - disc_count(ar)
            if want <= 0:
                continue
            mk = disc_mask & (d.area == ar) & (d.pool_assign == "")
            fams_ = d[mk].groupby("upstream_source").pkey.nunique()
            lim = {s_: int(cfg["discovery_frac"] * (1 + rnd) * src_parents.get(s_, 0)) for s_ in fams_.index}
            alloc = waterfill({s_: min(int(fams_[s_]), lim[s_]) for s_ in fams_.index}, want)
            carve("discovery", mk, alloc, "parents", 1.0)
        disc_n = {ar: disc_count(ar) for ar in AREAS}
        short_ = {ar: dq[ar] - disc_n[ar] for ar in AREAS if disc_n[ar] < dq[ar]}
        if not short_:
            break
        # nearest feasible allocation: an area without supply passes its remainder to the others by weight
        dead = {ar for ar in short_ if not (disc_mask & (d.area == ar) & (d.pool_assign == "")).any()}
        give = sum(short_[ar] for ar in dead)
        for ar in dead:
            dq[ar] = disc_n[ar]
        if give:
            add = lr_alloc({ar: cfg["area_weights"][ar] for ar in AREAS if ar not in dead}, give)
            for ar, x in add.items():
                dq[ar] += x
    tot = sum(disc_n.values())
    if tot > cfg["discovery_parents"]:
        s_ = d[d.pool_assign == "discovery"]
        drop = sorted(set(s_.pkey), key=lambda x: h(seed, "discovery", "trim", x))[: tot - cfg["discovery_parents"]]
        d.loc[d.pkey.isin(set(drop)) & (d.pool_assign == "discovery"), "pool_assign"] = "discovery_unused"
        disc_n = {ar: disc_count(ar) for ar in AREAS}
    rep["discovery_alloc_final"] = dq
    rep["discovery"] = disc_n
    m = live & (d.pool_assign == "")
    d.loc[m, "pool_assign"] = "pilot"
    rep["component_disjointness"] = validate_disjoint(d)
    d.drop(columns=["pkey"]).to_parquet(os.path.join(a.out, "units_pools.parquet"), index=False)
    summ = d.groupby("pool_assign").agg(units=("uid", "size"), parents=("parent_key", "nunique"),
                                        families=("canonical_family", "nunique"), comps=("comp", "nunique")).to_dict("index")
    rep["summary"] = summ
    rep["panel_families"] = {p: {ar: [f for c in cs for f in cl.at[c, "families"]] for ar, cs in byar.items()} for p, byar in panel_fams.items()}
    rep["panel_independent_clusters"] = {p: int(sum(len(cs) for cs in byar.values())) for p, byar in panel_fams.items()}
    for p in ("dev_select", "confirm_seal"):
        s = d[d.pool_assign == p]
        rep[p + "_stats"] = {"parents": int(s.groupby(["pool", "parent_key"]).ngroups), "decisions": int(len(s)),
                             "families": int(s.canonical_family.nunique()), "clusters": int(s.canonical_family.map(fam.cluster).nunique()),
                             "by_area_parents": s.groupby("area").apply(lambda g: g.groupby(["pool", "parent_key"]).ngroups).to_dict()}
    if (rep.get("confirm_seal_stats") or {}).get("by_area_parents") is not None:
        pass
    json.dump(rep, open(os.path.join(a.out, "pools_report.json"), "w"), indent=1, default=str)
    print(json.dumps({"summary": summ, "shortfalls": short, "dev": rep["dev_select_stats"], "confirm": {k: v for k, v in rep["confirm_seal_stats"].items() if k != "by_area_parents"}}, default=str))


if __name__ == "__main__":
    main()
