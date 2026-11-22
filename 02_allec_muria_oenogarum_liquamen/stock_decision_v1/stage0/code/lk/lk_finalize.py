#!/usr/bin/env python3
"""long_knowledge_v1 unit table + eligibility + report (stock_decision_v1).

Eligibility of a unit (excl == '' keeps it):
  base       candidate-table reasons (no token count, > 8,192 R2 tokens, K < 2 or > 52, duplicate option text, unmapped source)
  component  its ancestry component (recomputed over all eleven pools: nine Stage 0 pools + gap_supply_v2 + long_knowledge_v1)
             holds a record of another pool that is not UNUSED PILOT RESERVE (every unit pool_assign 'pilot', not excluded, Stage 0
             component used by no selection: M0, M1, M1 reserve, M2) -> 'component_link:<class>' (panel / used / excluded / gap).
             Reserve records in an eligible LK component are consumed (listed; LK owns the component from now on).
  protected  STRONG after the T1 / T2 template rules -> 'protected_strong'. A unit whose ONLY strong kind is text_long and whose
             every matched text unit was adjudicated text_template in a Stage 0 source-pool adjudication (DF >= 25 there; the
             multi-record states repeat Stage 0 record lines) is demoted to weak ('text_template_stage0'); eval_in_cand / near_dup
             hits are never demoted.
Report: per family records / decisions (all, eligible), exclusions by reason, length and K buckets, per-question target basis /
label verification (label_semantics schema), share-alike share, protected hits by protected population (DI / fitted-benchmark
overlap), label verification results, consumed reserve records.

  python3 lk_finalize.py --cand cand_lk.parquet --qsem qsem_lk.parquet --comp comp12/components.parquet --free-rk free_rk.txt
        --adj scans_lk/hits_long_knowledge_v1.adj.jsonl.gz --scan-manifest scan_manifest_lk.json --verify verify.json
        --pool-manifest long_knowledge_manifest.json --schema label_semantics_v1.json --out DIR"""
import argparse
import collections
import gzip
import importlib.util
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    for k in ("cand", "comp", "free-rk", "adj", "scan-manifest", "verify", "pool-manifest", "schema", "out", "units-s0", "out-dir-s0adj"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--used", action="append", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M2 = load("sd0_m2", os.path.join(HERE, "..", "sd0_m2.py"))
    S = json.load(open(a.schema))
    c = pd.read_parquet(a.cand)
    c = M2.semantics(c, S)
    comp = pd.read_parquet(a.comp, columns=["rk", "comp", "pool"])
    c = c.merge(comp[["rk", "comp"]], on="rk", how="left")
    assert c.comp.notna().all()
    u = pd.read_parquet(a.units_s0, columns=["uid", "rk", "comp", "excl", "pool_assign_fs"])
    used = set()
    for spec in a.used:
        used |= {x.strip() for x in open(spec.split("=", 1)[1]) if x.strip()}
    uc = set(u.loc[u.uid.isin(used), "comp"])
    st = u.groupby("rk").agg(pa=("pool_assign_fs", lambda s_: ",".join(sorted(set(s_)))), ex=("excl", lambda s_: bool((s_ != "").any())),
                            c10=("comp", "first"))
    st["used"] = st.c10.isin(uc)
    cls = lambda r: ("gap" if r.pool == "gap_supply_v2" else "missing" if pd.isna(r.pa) else "excluded" if r.ex else
                     "used_by_selection" if r.used else "pilot_free" if r.pa == "pilot" else "panel")
    lk_comps = set(c.comp)
    others = comp[comp.comp.isin(lk_comps) & (comp.pool != "long_knowledge_v1")].merge(st, left_on="rk", right_index=True, how="left")
    others["cls"] = [cls(r) for r in others.itertuples()]
    bad_links = others[others.cls != "pilot_free"]
    link_pool = bad_links.groupby("comp").cls.agg(lambda s_: ",".join(sorted(set(s_))))
    ok_comps = lk_comps - set(link_pool.index)
    consumed = sorted(set(others[others.comp.isin(ok_comps)].rk))
    tpl = set()
    for fn in sorted(os.listdir(a.out_dir_s0adj)):
        if fn.endswith(".adj.jsonl.gz"):
            for line in gzip.open(os.path.join(a.out_dir_s0adj, fn), "rt"):
                r = json.loads(line)
                if "text_long" in ((r.get("adj") or {}).get("demoted") or []):
                    tpl.update(r.get("tk") or [])
    strong, demoted = set(), 0
    for line in gzip.open(a.adj, "rt"):
        r = json.loads(line)
        if r["s"]:
            if set(r["s"]) == {"text_long"} and r.get("tk") and set(r["tk"]) <= tpl:
                demoted += 1
                continue
            strong.add(r["uid"])
    reasons = []
    for r in c.itertuples():
        x = []
        if r.base_reason:
            x.append(r.base_reason)
        if r.comp in link_pool.index:
            x.append("component_link:" + link_pool[r.comp])
        if r.uid in strong:
            x.append("protected_strong")
        reasons.append(";".join(x))
    c["excl"] = reasons
    c["share_alike"] = c.root.isin(["lk_musique", "lk_tydi"])
    c.to_parquet(os.path.join(a.out, "units_long_knowledge_v1.parquet"), index=False)
    e = c[c.excl == ""]
    rep = {"stage0_template_text_keys": len(tpl), "strong_text_long_demoted_stage0_template": demoted,
           "units_total": int(len(c)), "units_eligible": int(len(e)), "records_total": int(c.rk.nunique()), "records_eligible": int(e.rk.nunique()),
           "by_family": {}, "exclusions": {k: int(v) for k, v in collections.Counter(
               x.split(":")[0] for s in c.excl if s for x in s.split(";")).items()},
           "component_links_by_class": {k: int(v) for k, v in collections.Counter(
               p for s in c.excl if "component_link" in s for x in s.split(";") if x.startswith("component_link") for p in x.split(":", 1)[1].split(",")).items()},
           "reserve_records_consumed": len(consumed),
           "eligible_len_buckets": {k: int(v) for k, v in e.len_bucket.value_counts().items()},
           "eligible_k_buckets": {k: int(v) for k, v in e.k_bucket.value_counts().items()},
           "eligible_tokens_r2": float(e.tok_r2.sum()),
           "eligible_share_alike_units": int(e.share_alike.sum())}
    for f_, g in c.groupby("root"):
        ge = g[g.excl == ""]
        rep["by_family"][f_] = {"records": int(g.rk.nunique()), "units": int(len(g)), "eligible_records": int(ge.rk.nunique()),
                                "eligible_units": int(len(ge)), "parents_eligible": int(ge.parent_key.nunique()),
                                "len_buckets": {k: int(v) for k, v in ge.len_bucket.value_counts().items()},
                                "k_buckets": {k: int(v) for k, v in ge.k_bucket.value_counts().items()},
                                "kinds": {k: int(v) for k, v in ge.kind.value_counts().items()},
                                "target_basis": {k: int(v) for k, v in ge.target_basis.value_counts().items()},
                                "label_verification": {k: int(v) for k, v in ge.label_verification.value_counts().items()},
                                "splits": {k: int(v) for k, v in ge.split.value_counts().items()},
                                "excluded_by_reason": {k: int(v) for k, v in collections.Counter(
                                    x.split(":")[0] for s in g.excl if s for x in s.split(";")).items()},
                                "share_alike": bool(g.share_alike.iloc[0])}
    sm = json.load(open(a.scan_manifest))
    rep["protected_scan"] = sm
    hits_pop = collections.Counter()
    for line in gzip.open(a.adj, "rt"):
        r = json.loads(line)
        for p in (r.get("sp") or []) if r["s"] else []:
            hits_pop[f"{r['uid'].split('|')[1].split('/')[-1][:-6]}|{p}"] += 1
    rep["strong_units_by_family_and_protected_population"] = dict(hits_pop.most_common(60))
    rep["label_verification_check"] = json.load(open(a.verify))
    rep["pool_manifest"] = json.load(open(a.pool_manifest))
    json.dump(rep, open(os.path.join(a.out, "long_knowledge_v1_report.json"), "w"), indent=1, sort_keys=True, default=str)
    open(os.path.join(a.out, "reserve_records_consumed.txt"), "w").write("\n".join(consumed))
    print(json.dumps({k: rep[k] for k in ("strong_text_long_demoted_stage0_template", "units_total", "units_eligible", "records_eligible", "exclusions", "component_links_by_class",
                                          "eligible_len_buckets", "eligible_k_buckets", "reserve_records_consumed")}))
    print(json.dumps({f: {k: v[k] for k in ("eligible_records", "eligible_units", "len_buckets", "k_buckets", "excluded_by_reason")}
                      for f, v in rep["by_family"].items()})[:4000])
    print(json.dumps(rep["strong_units_by_family_and_protected_population"])[:1500])


if __name__ == "__main__":
    main()
