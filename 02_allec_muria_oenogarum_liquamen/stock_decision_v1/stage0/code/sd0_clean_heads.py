#!/usr/bin/env python3
"""head_fit_clean / cal_select_clean (stock_decision_v1): the r2 head-fit and calibration pools without any rows of an excluded family
(doc_nli: DocNLI re-hosts facebook/anli pairs, CC BY-NC 4.0), refilled to the same decision counts from the same cell:
(kind, K bucket, length bucket, hard/soft, label-verification tier, mixture half [M0-root / M1-eligible], M1 category).
Refill supply: units of the r2 PILOT pool (component-disjoint from every carved pool by construction) whose ancestry component is not
used by any training selection (M0, M1, M1 reserve, M2) and that are not of an excluded family; draws in a seeded hash order, new
parents first, <= 4 decisions per parent; a drawn decision brings its whole component (other units of it are marked unused).
The r2 pools are not modified (the 12B screen uses them). Output: units_pools_clean.parquet (r2 + the two clean pools), report.
With --registry R.json --exclude-lines NC,quarantine (clean2): a unit whose licence-registry line is listed is also excluded (dropped
from the pools and barred from refill); the sweep must then show 0 rows on those lines; unresolved entries (custom terms, unresolved
generation permission, composite review) are reported, not excluded. --strict-licence also excludes unresolved entries.
--suffix names the output pools (default _clean -> head_fit_clean / cal_select_clean).
  python3 sd0_clean_heads.py --units units_pools_r2.parquet --schema S.json --exclude-family doc_nli --used NAME=uids.txt ... --out DIR
  python3 sd0_clean_heads.py ... --registry license_registry.json --strict-licence --suffix _clean2 --out DIR2"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def h(*x):
    return hashlib.sha256("|".join(map(str, x)).encode()).hexdigest()


def reg_key(root, R):
    if root.startswith("ts:"):
        return "tasksource:" + root[3:]
    if root in R:
        return root
    if root.startswith("bf_"):
        return "bayes_families"
    return None


def reg_problem(k, R):
    v = R.get(k)
    if v is None:
        return "unmapped"
    why = []
    if v.get("line") != "clean":
        why.append(f"line={v.get('line')}")
    if v.get("commercial_use") != "yes":
        why.append(f"commercial_use={v.get('commercial_use')}")
    if v.get("custom_or_other_terms"):
        why.append("custom_terms")
    if v.get("generation_permission") == "unresolved":
        why.append("generation_permission=unresolved")
    if v.get("composite_review"):
        why.append("composite_review=" + str(v["composite_review"].get("status")))
    return ",".join(why)


def licence_sweep(d, R, pools, col):
    """counts of rows in each pool by registry status: NC / unresolved / quarantine / unmapped, and offending keys."""
    out = {}
    for p in pools:
        s = d[d[col] == p]
        keys = s.root.map(lambda r: reg_key(r, R))
        lines = keys.map(lambda k: (R.get(k) or {}).get("line", "unmapped"))
        prob = keys.map(lambda k: reg_problem(k, R))
        out[p] = {"rows": int(len(s)), "NC": int((lines == "NC").sum()), "quarantine": int((lines == "quarantine").sum()),
                  "unmapped": int((lines == "unmapped").sum()),
                  "unresolved": int(((prob != "") & (lines == "clean")).sum()),
                  "registry_keys": int(keys.nunique()),
                  "offending": {f"{k} [{w}]": int(n) for (k, w), n in pd.DataFrame({"k": keys, "w": prob})[prob != ""].value_counts().items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--exclude-family", action="append", required=True)
    ap.add_argument("--used", action="append", required=True)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--registry", default="")
    ap.add_argument("--strict-licence", action="store_true")
    ap.add_argument("--exclude-lines", default="")
    ap.add_argument("--suffix", default="_clean")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xlines = {x for x in a.exclude_lines.split(",") if x}
    if (a.strict_licence or xlines) and not a.registry:
        raise SystemExit("--strict-licence / --exclude-lines need --registry")
    sfx = a.suffix
    os.makedirs(a.out, exist_ok=True)
    M2 = load("sd0_m2")
    SP = load("sd0_pools")
    S = json.load(open(a.schema))
    d = M2.semantics(pd.read_parquet(a.units), S)
    d["half"] = d.m0_root.map({True: "M0", False: "M1"})
    d["cell"] = (d.kind.astype(str) + "|" + d.k_bucket.astype(str) + "|" + d.len_bucket.astype(str) + "|" + d.hard.astype(str) + "|"
                 + d.tier.astype(str) + "|" + d.half + "|" + d.m1_category.astype(str))
    excl = set(a.exclude_family)
    R = json.load(open(a.registry))["entries"] if a.registry else {}
    d["lic_key"] = d.root.map(lambda r: reg_key(r, R)) if R else None
    d["lic_problem"] = d.lic_key.map(lambda k: reg_problem(k, R)) if R else ""
    d["lic_line"] = d.lic_key.map(lambda k: (R.get(k) or {}).get("line", "unmapped")) if R else ""
    d["xbad"] = (d.canonical_family.isin(excl) | ((d.lic_problem != "") if a.strict_licence else False)
                 | (d.lic_line.isin(xlines | {"unmapped"}) if xlines else False))
    used = set()
    for spec in a.used:
        name, path = spec.split("=", 1)
        used |= {x.strip() for x in open(path) if x.strip()}
    used_comps = set(d.loc[d.uid.isin(used), "comp"])
    rep = {"exclude_families": sorted(excl), "strict_licence": bool(a.strict_licence), "exclude_lines": sorted(xlines), "suffix": sfx,
           "registry_md5": hashlib.md5(open(a.registry, "rb").read()).hexdigest() if a.registry else None,
           "used_selections": [s.split("=")[0] for s in a.used], "used_components": len(used_comps)}
    d["pool_assign_clean"] = d.pool_assign
    free = (d.pool_assign == "pilot") & ~d.comp.isin(used_comps) & ~d.xbad & (d.excl == "") & (d.licence_line == "clean")
    comp_idx = d.groupby("comp").indices
    taken = set()
    for pool in ("head_fit", "cal_select"):
        cur = d[d.pool_assign == pool]
        drop = cur[cur.xbad]
        new = pool + sfx
        d.loc[cur.index, "pool_assign_clean"] = new
        d.loc[drop.index, "pool_assign_clean"] = new + "_removed"
        need = drop.cell.value_counts().to_dict()
        got = collections.Counter()
        short = {}
        for cell, n in sorted(need.items()):
            cand = d[free & (d.cell == cell) & (d.pool_assign_clean == "pilot")]
            cand = cand.assign(_k=[h(a.seed, new, x) for x in cand.uid]).sort_values("_k")
            per_parent = collections.Counter()
            for r in cand.itertuples():
                if got[cell] >= n:
                    break
                if r.comp in taken or d.at[r.Index, "pool_assign_clean"] != "pilot":
                    continue
                if per_parent[r.parent_key] >= 4:
                    continue
                taken.add(r.comp)
                pos = comp_idx[r.comp]
                d.iloc[pos, d.columns.get_loc("pool_assign_clean")] = new + "_unused"
                d.at[r.Index, "pool_assign_clean"] = new
                per_parent[r.parent_key] += 1
                got[cell] += 1
            if got[cell] < n:
                short[cell] = n - got[cell]
        rep[new] = {"original_decisions": int(len(cur)), "removed": int(len(drop)), "refilled": int(sum(got.values())), "unfilled": short,
                    "final_decisions": int((d.pool_assign_clean == new).sum()),
                    "final_parents": int(d[d.pool_assign_clean == new].groupby(["pool", "parent_key"]).ngroups),
                    "removed_by_cell": {k: int(v) for k, v in need.items()},
                    "removed_by_family": {k: int(v) for k, v in drop.canonical_family.value_counts().items()},
                    "removed_by_reason": {k: int(v) for k, v in (drop.canonical_family.isin(excl).map({True: "excluded_family", False: ""})
                                          + "|" + drop.lic_line).value_counts().items()} if R else {},
                    "excluded_family_rows_left": int(((d.pool_assign_clean == new) & d.canonical_family.isin(excl)).sum()),
                    "excluded_rows_left": int(((d.pool_assign_clean == new) & d.xbad).sum())}
    # component disjointness over the clean assignment
    v = d.assign(pool_assign=d.pool_assign_clean.replace({"head_fit" + sfx: "head_fit", "head_fit" + sfx + "_unused": "head_fit_unused",
                                                          "cal_select" + sfx: "cal_select", "cal_select" + sfx + "_unused": "cal_select_unused"}))
    v = v[~v.pool_assign_clean.isin(["head_fit" + sfx + "_removed", "cal_select" + sfx + "_removed"])]
    rep["component_disjointness"] = SP.validate_disjoint(v)
    if R:
        rep["licence_sweep"] = licence_sweep(d, R, ["head_fit" + sfx, "cal_select" + sfx], "pool_assign_clean")
        bad = lambda x: (x["unresolved"] if a.strict_licence else 0) + sum(x.get(l, 0) for l in xlines | {"unmapped"})
        if (a.strict_licence or xlines) and any(bad(x) for x in rep["licence_sweep"].values()):
            raise SystemExit("licence sweep failed: " + json.dumps(rep["licence_sweep"]))
    clean_comps = set(d.loc[d.pool_assign_clean.isin(["head_fit" + sfx, "cal_select" + sfx]), "comp"])
    rep["clean_vs_used_selections_component_intersection"] = len(clean_comps & used_comps)
    if rep["clean_vs_used_selections_component_intersection"]:
        raise SystemExit("clean head pools share components with a training selection")
    tmp = ["cell", "half", "lic_key", "lic_problem", "lic_line", "xbad"]
    d.drop(columns=tmp).to_parquet(os.path.join(a.out, f"units_pools{sfx}.parquet"), index=False)
    for new in ("head_fit" + sfx, "cal_select" + sfx):
        d[d.pool_assign_clean == new].drop(columns=tmp).to_parquet(os.path.join(a.out, f"sel_{new}.parquet"), index=False)
    json.dump(rep, open(os.path.join(a.out, f"clean_heads{sfx}_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep.get(k) for k in ("head_fit" + sfx, "cal_select" + sfx, "clean_vs_used_selections_component_intersection", "licence_sweep")}, default=str))


if __name__ == "__main__":
    main()
