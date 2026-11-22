#!/usr/bin/env python3
"""Stage 0 matched pilot selection (stock_decision_v1): M0 and M1 pilot tranches of N exposures each, with IDENTICAL stratum
counts over (kind, K bucket, rendered-length bucket, hard/soft, exposure repetition), source / family / generator caps and the M1
category shares, then token-balanced to <= TOKEN_TOL relative difference (proxy R2 counts; the real render re-checks).

Quotas (fixed before solving):
  M0: recipe A's root composition (garas_gemma_vbll_v2 freeze_r2 manifest_A roots, scaled to N), then water-filled under the M0
      caps (source, family, generator) and the supply left after reservations;
  M1: category totals from the (possibly amended) shares; inside a category, water-filling across upstream sources (equal
      shares up to supply), under the 2% source, 6% family and 2% generator caps.
Matching: one integer program (scipy milp / HiGHS): x0[source, stratum], x1[source, stratum], T[stratum]; sum_source x = T for both
mixtures; caps and M1 category totals are hard; minimise |X0 - q0| + |X1 - q1| + 0.2 |T - T_nat| (T_nat = mean of the two
mixtures' natural stratum mixes at their quotas). Every exposure is a first exposure (repetition 1) unless --allow-repeat.
Within a (source, stratum) cell units are drawn by a seeded shuffle, at most MAX_SIB decisions per parent per mixture, new parents
first. Token balancing swaps units inside the same (source, stratum) cell only.

  python3 sd0_pilot.py --units units_pools.parquet --config pilot_config.json --out DIR
"""
import argparse
import collections
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

MAX_SIB = 4


def waterfill(supply, caps, total):
    """max-min fair integer allocation of `total` over keys with per-key upper bounds min(supply, cap)"""
    ub = {k: min(supply[k], caps.get(k, 10 ** 9)) for k in supply}
    alloc = {k: 0 for k in supply}
    left = total
    active = [k for k in ub if ub[k] > 0]
    while left > 0 and active:
        share = max(1, left // len(active))
        nxt = []
        for k in sorted(active):
            give = min(share, ub[k] - alloc[k], left)
            alloc[k] += give
            left -= give
            if alloc[k] < ub[k]:
                nxt.append(k)
            if left == 0:
                break
        active = nxt
    return alloc


def strata_key(df):
    return df.kind.astype(str) + "|" + df.k_bucket.astype(str) + "|" + df.len_bucket.astype(str) + "|" + np.where(df.hard, "hard", "soft")


def solve(cells0, cells1, q0, q1, caps0, caps1, cat_tot, src_cat, n_total, time_limit):
    S = sorted(set(cells0.stratum) | set(cells1.stratum))
    sidx = {s: i for i, s in enumerate(S)}
    v0 = cells0.reset_index(drop=True)
    v1 = cells1.reset_index(drop=True)
    n0, n1, nS = len(v0), len(v1), len(S)
    src0 = sorted(q0)
    src1 = sorted(q1)
    # variable layout: x0 | x1 | T | d0+ d0- (per M0 source) | d1+ d1- (per M1 source) | dt+ dt- (per stratum)
    o_x1, o_T = n0, n0 + n1
    o_d0 = o_T + nS
    o_d1 = o_d0 + 2 * len(src0)
    o_dt = o_d1 + 2 * len(src1)
    nv = o_dt + 2 * nS
    rows, lo, hi = [], [], []
    A = lil_matrix((0, nv))

    def add(coefs, l, h):
        rows.append(coefs)
        lo.append(l)
        hi.append(h)
    for s in S:     # sum x0 = T, sum x1 = T
        c = {i: 1.0 for i in np.flatnonzero(v0.stratum.values == s)}
        c[o_T + sidx[s]] = -1.0
        add(c, 0, 0)
        c = {o_x1 + i: 1.0 for i in np.flatnonzero(v1.stratum.values == s)}
        c[o_T + sidx[s]] = -1.0
        add(c, 0, 0)
    add({o_T + i: 1.0 for i in range(nS)}, n_total, n_total)
    for (mix, v, off, caps) in ((0, v0, 0, caps0), (1, v1, o_x1, caps1)):
        for col, lim in (("upstream_source", caps["source"]), ("canonical_family", caps["family"]), ("generator_mechanism", caps["generator"])):
            for key, g in v.groupby(col):
                if key in (None, "", "None") or (isinstance(key, float) and np.isnan(key)):
                    continue
                lim_k = caps.get("source_override", {}).get(key, lim) if col == "upstream_source" else lim
                add({off + i: 1.0 for i in g.index}, 0, lim_k)
    for cat, tot in cat_tot.items():     # M1 category totals
        idx = [o_x1 + i for i in np.flatnonzero(v1.upstream_source.map(src_cat).values == cat)]
        add({i: 1.0 for i in idx}, tot, tot)
    for j, s_ in enumerate(src0):        # |X0 - q0|
        c = {i: 1.0 for i in np.flatnonzero(v0.upstream_source.values == s_)}
        c[o_d0 + 2 * j] = -1.0
        c[o_d0 + 2 * j + 1] = 1.0
        add(c, q0[s_], q0[s_])
    for j, s_ in enumerate(src1):
        c = {o_x1 + i: 1.0 for i in np.flatnonzero(v1.upstream_source.values == s_)}
        c[o_d1 + 2 * j] = -1.0
        c[o_d1 + 2 * j + 1] = 1.0
        add(c, q1[s_], q1[s_])
    nat0 = v0.groupby("stratum").nat.sum()
    nat1 = v1.groupby("stratum").nat.sum()
    tnat = ((nat0.reindex(S, fill_value=0) / max(nat0.sum(), 1) + nat1.reindex(S, fill_value=0) / max(nat1.sum(), 1)) / 2 * n_total)
    for s in S:
        add({o_T + sidx[s]: 1.0, o_dt + 2 * sidx[s]: -1.0, o_dt + 2 * sidx[s] + 1: 1.0}, float(tnat[s]), float(tnat[s]))
    A = lil_matrix((len(rows), nv))
    for r, c in enumerate(rows):
        for k, val in c.items():
            A[r, k] = val
    ub = np.full(nv, np.inf)
    ub[:n0] = v0.supply.values
    ub[o_x1:o_x1 + n1] = v1.supply.values
    cost = np.zeros(nv)
    cost[o_d0:o_dt] = 1.0
    cost[o_dt:] = 0.2
    integ = np.zeros(nv)
    integ[:o_d0] = 1
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), bounds=Bounds(np.zeros(nv), ub), integrality=integ,
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None, {"status": res.status, "message": res.message}
    x = np.round(res.x).astype(np.int64)
    v0 = v0.assign(ntake=x[:n0])
    v1 = v1.assign(ntake=x[o_x1:o_x1 + n1])
    T = dict(zip(S, x[o_T:o_T + nS]))
    return (v0, v1, T), {"status": int(res.status), "message": str(res.message), "objective": float(res.fun),
                         "mip_gap": float(getattr(res, "mip_gap", 0) or 0), "T_nat": {s: round(float(tnat[s]), 1) for s in S}}


def draw(units, cells, seed, tag):
    """pick `take` units per (source, stratum) cell; <= MAX_SIB per parent; new parents first"""
    picked = []
    per_parent = collections.Counter()
    by_cell = {k: g for k, g in units.groupby(["upstream_source", "stratum"])}
    cells = cells.groupby(["upstream_source", "stratum"], as_index=False)["ntake"].sum()   # one draw per (source, stratum) cell
    for _, c in cells[cells["ntake"] > 0].sort_values(["upstream_source", "stratum"]).iterrows():
        g = by_cell[(c.upstream_source, c.stratum)]
        h = g.uid.map(lambda u: hashlib.sha256(f"{seed}|{tag}|{u}".encode()).hexdigest())
        g = g.assign(_h=h.values).sort_values("_h")
        got = []
        for pass_ in (0, 1):
            for r in g.itertuples():
                if len(got) >= c["ntake"]:
                    break
                if r.uid in got:
                    continue
                n = per_parent[r.parent_key]
                if (pass_ == 0 and n == 0) or (pass_ == 1 and n < MAX_SIB):
                    got.append(r.uid)
                    per_parent[r.parent_key] += 1
        if len(got) < c["ntake"]:
            raise SystemExit(f"{tag}: cell {c.upstream_source} {c.stratum}: {len(got)} < {c['ntake']} under the sibling cap")
        picked += got
    return picked


def _swaps(P, sel_list, want, seed, sign):
    """swaps inside (source, stratum) cells that move the mixture's token total by about `want` tokens in direction `sign`
    (-1 shorter, +1 longer); returns (new selection, achieved change, n swaps, n candidates)"""
    chosen = set(sel_list)
    P = P.assign(_sel=P.uid.isin(chosen), _h=[hashlib.sha256(f"{seed}|tb|{u}".encode()).hexdigest() for u in P.uid])
    cand = []
    for key, g in P.groupby(["upstream_source", "stratum"], sort=True):
        a_ = g[g._sel].sort_values(["tok_r2", "_h"], ascending=[sign > 0, True])     # shorten: longest selected first
        f_ = g[~g._sel].sort_values(["tok_r2", "_h"], ascending=[sign < 0, True])    # shorten: shortest free first
        k = min(len(a_), len(f_))
        if k == 0:
            continue
        dl = (f_.tok_r2.values[:k].astype(np.int64) - a_.tok_r2.values[:k].astype(np.int64)) * sign
        for i in np.flatnonzero(dl > 0):
            cand.append((-int(dl[i]), a_.uid.values[i], f_.uid.values[i]))
    cand.sort(key=lambda x: (x[0], x[1]))
    done, sw = 0, {}
    for negd, u_old, u_new in cand:
        if done >= want:
            break
        if done - negd <= want * 1.02:
            sw[u_old] = u_new
            done -= negd
    return [sw.get(u, u) for u in sel_list], done * sign, len(sw), len(cand)


def token_balance(sel, pools, tol, seed):
    """Close the token gap between M0 and M1 to <= tol/2 by within-cell swaps, half of the gap from each side (the longer
    mixture gets shorter units, the shorter one longer units, same (source, stratum) cell): stratum and source counts unchanged."""
    def total(m):
        return int(pools[m].set_index("uid").loc[sel[m]].tok_r2.sum())
    tot = {m: total(m) for m in pools}
    info = {"initial_rel_gap": (tot["M0"] - tot["M1"]) / max(tot["M1"], 1)}
    big, small = ("M0", "M1") if tot["M0"] > tot["M1"] else ("M1", "M0")
    gap = tot[big] - tot[small] - int(tol * 0.5 * tot[small])
    if gap > 0:
        sel[big], d1, n1, c1 = _swaps(pools[big], sel[big], gap // 2, seed, -1)
        sel[small], d2, n2, c2 = _swaps(pools[small], sel[small], gap + d1, seed, +1)
        info.update(swaps={big: n1, small: n2}, candidates={big: c1, small: c2})
    tot = {m: total(m) for m in pools}
    info.update(tokens_m0=tot["M0"], tokens_m1=tot["M1"], rel_gap=(tot["M0"] - tot["M1"]) / max(tot["M1"], 1))
    return sel, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    N = int(cfg["n_exposures"])
    seed = int(cfg["seed"])
    d = pd.read_parquet(a.units)
    d = d[d.pool_assign == "pilot"].copy()
    d["stratum"] = strata_key(d)
    def cap_sib(u, tag):
        """at most MAX_SIB candidate decisions per parent (seeded hash order), so no draw can break the sibling cap"""
        u = u.assign(_k=[hashlib.sha256(f"{seed}|sib|{tag}|{x}".encode()).hexdigest() for x in u.uid]).sort_values("_k")
        return u.groupby("parent_key", sort=False).head(MAX_SIB).drop(columns=["_k"])
    m0u = cap_sib(d[d.m0_root].copy(), "M0")
    m1u = cap_sib(d[d.m1_eligible].copy(), "M1")
    src_cat = m1u.groupby("upstream_source").m1_category.first().to_dict()
    # quotas
    a_roots = cfg["m0_recipeA_roots"]                       # root -> recipe A exposures (of 120,000)
    root_src = m0u.groupby("root").upstream_source.first().to_dict()
    tgt = collections.Counter()
    for r_, n_ in a_roots.items():
        if r_ in root_src:
            tgt[root_src[r_]] += n_ * N / 120000.0
    sup0 = m0u.groupby("upstream_source").size().to_dict()
    cat_tot = {}
    shares = cfg["m1_shares"]
    rem = N
    for i, (c, sh) in enumerate(sorted(shares.items())):
        cat_tot[c] = int(round(sh * N)) if i < len(shares) - 1 else rem
        rem -= cat_tot[c]
    cap1 = cfg["m1_caps"]
    sup1 = m1u.groupby("upstream_source").size().to_dict()
    q1 = {}
    for c, tot in cat_tot.items():
        ss = {s: sup1[s] for s in sup1 if src_cat.get(s) == c}
        q1.update(waterfill(ss, {s: cap1["source"] for s in ss}, tot))
    report = {"config": cfg, "q1": q1, "cat_tot": cat_tot, "supply_m0_by_source": sup0, "supply_m1_by_source": sup1,
              "m1_category_supply": {c: int(sum(sup1[s] for s in sup1 if src_cat.get(s) == c)) for c in cat_tot},
              "m1_category_quota_sum": {c: int(sum(q1[s] for s in q1 if src_cat.get(s) == c)) for c in cat_tot}, "ladder": []}
    os.makedirs(a.out, exist_ok=True)
    for c, tot in cat_tot.items():
        if report["m1_category_quota_sum"][c] < tot:
            report["m1_category_shortfall"] = {c: tot - report["m1_category_quota_sum"][c]}
            json.dump(report, open(os.path.join(a.out, "pilot_report.json"), "w"), indent=1, default=str)
            raise SystemExit(f"M1 category {c}: capped supply {report['m1_category_quota_sum'][c]} < {tot}: amend the shares")

    def cells(u, q):
        c = u.groupby(["upstream_source", "canonical_family", "generator_mechanism", "stratum"], dropna=False).size().rename("supply").reset_index()
        tot = c.groupby("upstream_source").supply.transform("sum")
        c["nat"] = c.upstream_source.map(q).fillna(0) * c.supply / tot
        return c
    sol = None
    for cap_src in cfg.get("m0_source_cap_ladder", [cfg["m0_caps"]["source"]]):
        cap0 = dict(cfg["m0_caps"], source=cap_src)
        caps_src0 = {s: cap_src for s in sup0}
        q0 = waterfill({s: min(sup0.get(s, 0), int(round(tgt.get(s, 0)))) if tgt.get(s, 0) else 0 for s in sup0}, caps_src0, N)
        if sum(q0.values()) < N:
            room = {s: min(sup0[s], caps_src0[s]) - q0[s] for s in sup0 if s in tgt}
            extra = waterfill(room, {}, N - sum(q0.values()))
            q0 = {s: q0[s] + extra.get(s, 0) for s in q0}
        c0, c1 = cells(m0u, q0), cells(m1u, q1)
        if sum(q0.values()) < N:
            report["ladder"].append({"m0_source_cap": cap_src, "status": "m0 capped supply short", "q0_sum": sum(q0.values())})
            continue
        sol, info = solve(c0, c1, q0, q1, cap0, cap1, cat_tot, src_cat, N, cfg.get("time_limit", 600))
        report["ladder"].append({"m0_source_cap": cap_src, "status": info.get("status"), "message": info.get("message")})
        if sol is not None:
            report.update(q0=q0, m0_caps_used=cap0, solver=info)
            break
    if sol is None:
        json.dump(report, open(os.path.join(a.out, "pilot_report.json"), "w"), indent=1, default=str)
        raise SystemExit(f"INFEASIBLE on the whole M0 cap ladder: {report['ladder']}")
    v0, v1, T = sol
    json.dump(report, open(os.path.join(a.out, "pilot_report.json"), "w"), indent=1, default=str)
    p0 = draw(m0u, v0, seed, "M0")
    p1 = draw(m1u, v1, seed, "M1")
    sel, tb = token_balance({"M0": list(p0), "M1": list(p1)}, {"M0": m0u, "M1": m1u}, cfg.get("token_tol", 0.005), seed)
    for m_ in ("M0", "M1"):
        if len(set(sel[m_])) != len(sel[m_]):
            raise SystemExit(f"{m_}: duplicate decisions in the selection")
    s0 = m0u.set_index("uid").loc[sel["M0"]].reset_index()
    s1 = m1u.set_index("uid").loc[sel["M1"]].reset_index()
    for tag, s in (("M0", s0), ("M1", s1)):
        s = s.copy()
        s["pos_key"] = s.uid.map(lambda u: hashlib.sha256(f"{seed}|pos|{tag}|{u}".encode()).hexdigest())
        s = s.sort_values("pos_key").reset_index(drop=True)
        s["pos"] = np.arange(len(s))
        s["exposure_k"] = 0
        s.drop(columns=["pos_key"]).to_parquet(os.path.join(a.out, f"pilot_{tag}.parquet"), index=False)
    chk = {}
    for tag, s in (("M0", s0), ("M1", s1)):
        chk[tag] = {"n": len(s), "unique": int(s.uid.nunique()), "tokens_r2": int(s.tok_r2.sum()),
                    "strata": s.groupby("stratum").size().to_dict(),
                    "by_source": s.groupby("upstream_source").size().sort_values(ascending=False).to_dict(),
                    "by_family_max": int(s.groupby("canonical_family").size().max()),
                    "by_generator": s.dropna(subset=["generator_mechanism"]).groupby("generator_mechanism").size().to_dict(),
                    "by_category": s.groupby("m1_category", dropna=False).size().to_dict() if tag == "M1" else None,
                    "by_area": s.groupby("area").size().to_dict(), "kind": s.groupby("kind").size().to_dict(),
                    "max_sib": int(s.groupby("parent_key").size().max()), "hard_share": float(s.hard.mean())}
    report.update(T={k: int(v) for k, v in T.items()}, token_balance=tb, check=chk,
                  strata_equal=chk["M0"]["strata"] == chk["M1"]["strata"])
    json.dump(report, open(os.path.join(a.out, "pilot_report.json"), "w"), indent=1, default=str)
    print(json.dumps({"strata_equal": report["strata_equal"], "token_balance": tb, "solver": {k: info[k] for k in ("status", "objective")},
                      "M0_sources": len(chk["M0"]["by_source"]), "M1_sources": len(chk["M1"]["by_source"])}))


if __name__ == "__main__":
    main()
