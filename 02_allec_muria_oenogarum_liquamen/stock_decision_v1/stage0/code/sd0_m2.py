#!/usr/bin/env python3
"""Mixture M2 assembly (stock_decision_v1; SOTA_EXECUTION_PACK_2026-10-14/mixture_M2_spec.json), PILOT-DATA build.

Buckets (48,000 exposures):
  1  core real decision / grounding / NLI / human disagreement, 14,400: Stage 0 r2 PILOT-pool units whose label_verification
     (schema label_semantics_v1, per-question value first) is source_human_gold / retained_human_distribution / outcome_record /
     execution_check / construction_rule -- i.e. tiers independent + source_gold minus synthetic exact_oracle rows -- from clean,
     non-DI, non-model-labelled, non-TD sources; ChaosNLI and contract_nli EXCLUDED (protected / DI root; coordinator 2026-10-14);
     <= 2% of the mixture (960 exposures) per upstream source; within the bucket the (length, K) cells are steered toward the
     spec's overall k/length targets net of the fixed buckets.
  2  exact Bayesian posteriors and EU actions, 12,000: bayes_zen U3 (BAYES-DATA, 3,840 rendered exposures, taken as delivered),
     bayes_families r2 4,320 and garas_synth r2 3,840 (rows whose per-question target_basis is exact_posterior or optimal_action),
     70 / 30 posterior / action inside each sub-allocation.
  3  gap closers, 14,400: gap_supply_v1 TRAIN split per sub-allocation (tool 3,840, appliance 2,880, causal 2,880, music 2,400,
     pyexec 2,400); synth_tool_policy rows keep td_workflow_imitation.
  4  teacher-failure-mined discovery rows, 7,200: the TEACHER export (sdv1_m2_export.py) -- verified gold targets only.
Exposures: unique decisions get 1 or 2 exposures (exposure_k 0 / 1) so each bucket hits its exposure count at ~1.5 exposures per
decision (U3 and Bucket 4 as delivered); <= 4 decisions per parent per bucket. A second exposure only goes to a choice question
(K >= 3 first, then K = 2), whose option order differs between exposures; fixed-order score / noul rows get one exposure and the
unique count grows instead (v2; v1 gave identical second prompts to score / noul rows). The renderer refuses two exposures of one
decision with the same prompt (--require-distinct-prompts).
parent_route (backbone_objective.routing_by_target_basis): exact_posterior / optimal_action / synthetic exact_oracle hard_fact ->
none; realised_outcome / human_distribution / graded_rating / outcome_probability -> garas; other hard_fact (language / retrieval /
grounding, source gold) -> stock.

  python3 sd0_m2.py --units units_pools.parquet --gap units_gap.parquet --schema label_semantics_v1.json --config m2_config.json
        [--bucket4 export.jsonl.gz] --out DIR
"""
import argparse
import collections
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

KB = ["2", "3-5", "6-16", "17-32", "33-52"]
LB = ["<=512", "513-2048", "2049-8192"]
B1_VERIF = {"source_human_gold", "retained_human_distribution", "outcome_record", "execution_check", "construction_rule"}
GARAS_BASIS = {"realised_outcome", "human_distribution", "graded_rating", "outcome_probability"}


def h(*x):
    return hashlib.sha256("|".join(map(str, x)).encode()).hexdigest()


def lr_alloc(w, total):
    tw = sum(w.values())
    if tw <= 0:
        return {k: 0 for k in w}
    raw = {k: total * v / tw for k, v in w.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def semantics(df, S):
    pm = S["stage0_provenance_map"]
    tb = df.label_basis.map(lambda b: pm[b]["target_basis"])
    lv = df.label_basis.map(lambda b: pm[b]["label_verification"])
    if "target_basis_q" in df:
        tb = df.target_basis_q.where(df.target_basis_q.notna() & (df.target_basis_q != ""), tb)
    if "label_verification_q" in df:
        lv = df.label_verification_q.where(df.label_verification_q.notna() & (df.label_verification_q != ""), lv)
    return df.assign(target_basis=tb, label_verification=lv, tier=lv.map(lambda v: S["label_verification"][v]["tier"]))


def route(r):
    if r.target_basis in ("exact_posterior", "optimal_action") or r.label_verification == "exact_oracle":
        return "none"
    if r.target_basis in GARAS_BASIS:
        return "garas"
    return "stock"


def pick(units, n_exposures, seed, tag, per_decision=1.5, max_unique=None):
    """choose unique decisions (hash order, new parents first, <= 4 per parent) and exposures (1 or 2 each) summing to n_exposures.
    A second exposure only goes to a decision whose option order can change (choice with K >= 3 first, then choice with K = 2);
    fixed-order rows (score / noul) get one exposure -- the unique count grows instead. pick.fixed_second counts any fixed-order
    second exposures that could not be avoided (supply exhausted)."""
    if len(units) == 0 or n_exposures <= 0:
        pick.fixed_second = 0
        return units.iloc[:0].assign(n_exp=0)
    u = units.assign(_k=[h(seed, tag, x) for x in units.uid]).sort_values("_k")
    u["_r"] = u.groupby("parent_key").cumcount()
    u = u[u._r < 4].sort_values(["_r", "_k"]).reset_index(drop=True)
    perm3 = ((u.kind == "choice") & (u.K >= 3)).values
    perm2 = ((u.kind == "choice") & (u.K == 2)).values
    permc = (perm3 | perm2).cumsum()
    n_exposures = min(n_exposures, len(u) + int(permc[-1]))   # what this supply can carry without a fixed-order second exposure
    if n_exposures <= 0:
        pick.fixed_second = 0
        return units.iloc[:0].assign(n_exp=0)
    n_unique = int(round(n_exposures / per_decision))
    if max_unique is not None:
        n_unique = min(n_unique, max_unique)
    n_unique = max(1, min(n_unique, len(u)))
    while n_unique < len(u) and n_exposures - n_unique > permc[n_unique - 1]:
        n_unique += 1                                      # not enough permutable rows for the seconds: take more unique decisions
    n_unique = min(n_unique, n_exposures, len(u))
    sel = u.head(n_unique).copy()
    twice = max(0, n_exposures - n_unique)
    if twice > n_unique:
        raise SystemExit(f"{tag}: {n_exposures} exposures need more than 2 per decision (unique supply {n_unique})")
    pr = np.where(((sel.kind == "choice") & (sel.K >= 3)).values, 0, np.where(((sel.kind == "choice") & (sel.K == 2)).values, 1, 2))
    order = np.lexsort((sel._k.values, pr))
    sel["n_exp"] = 1
    sel.iloc[order[:twice], sel.columns.get_loc("n_exp")] = 2
    pick.fixed_second = int(((sel.n_exp == 2) & (pr == 2)).sum())
    return sel.drop(columns=["_k", "_r"])


pick.fixed_second = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--gap", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bucket4", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    S = json.load(open(a.schema))
    seed = cfg["seed"]
    N = cfg["total_exposures"]
    os.makedirs(a.out, exist_ok=True)
    rep = {"config": cfg}
    d = semantics(pd.read_parquet(a.units), S)
    d = d[(d.pool_assign == "pilot") & (d.excl == "")]
    gap = semantics(pd.read_parquet(a.gap), S)
    gap = gap[gap.split == "train"]
    sel = []
    FIXED2 = {}
    # ---------------- bucket 2 (fixed composition)
    for sub, mask, n in (("bayes_families_r2", d.root.astype(str).str.startswith("bf_"), cfg["b2"]["bayes_families_r2"]),
                         ("garas_synth_r2", d.root.isin(["synth_beta_rate", "synth_binary_evidence"]), cfg["b2"]["garas_synth_r2"])):
        cand = d[mask & d.target_basis.isin(["exact_posterior", "optimal_action"])]
        n_post = int(round(n * 0.7))
        for tb_, n_ in (("exact_posterior", n_post), ("optimal_action", n - n_post)):
            p = pick(cand[cand.target_basis == tb_], n_, seed, f"b2|{sub}|{tb_}")
            FIXED2[f"b2|{sub}|{tb_}"] = pick.fixed_second
            if p.n_exp.sum() < n_:
                raise SystemExit(f"bucket 2 {sub} {tb_}: supply {p.n_exp.sum()} < {n_}")
            sel.append(p.assign(m2_bucket="2_exact_bayesian_posteriors_and_eu_actions", m2_sub_allocation=sub))
    # ---------------- bucket 3
    for fam, n in cfg["b3"].items():
        g = gap[gap.root == fam]
        p = pick(g, n, seed, f"b3|{fam}")
        FIXED2[f"b3|{fam}"] = pick.fixed_second
        if p.n_exp.sum() < n:
            raise SystemExit(f"bucket 3 {fam}: supply {p.n_exp.sum()} < {n}")
        sel.append(p.assign(m2_bucket="3_di03_gap_closers_deterministic_oracles", m2_sub_allocation=fam))
    fixed = pd.concat(sel, ignore_index=True)
    # strata of the fixed parts (bucket 2 U3 from its header: all K=3, 513-2048; bucket 4 assumed like discovery, recorded)
    fx = collections.Counter()
    for r in fixed.itertuples():
        fx[(r.len_bucket, r.k_bucket)] += r.n_exp
    fx[("513-2048", "3-5")] += cfg["b2"]["bayes_zen_U3"]
    b4 = cfg["b4_exposures"]
    if cfg.get("b4_assumed_cells"):
        for k, v in cfg["b4_assumed_cells"].items():
            lb_, kb_ = k.split("|")
            fx[(lb_, kb_)] += int(round(v * b4))
    # ---------------- bucket 1 (steered)
    excl_src = set(cfg["b1_exclude_sources"])
    b1 = d[d.label_verification.isin(B1_VERIF) & ~d.di_ancestry & (d.licence_line == "clean")
           & ~d.upstream_source.isin(excl_src) & ~d.canonical_family.isin(excl_src)
           & ~d.label_basis.isin(["model_or_unverified_synthetic", "model_labelled_upstream", "rule_synthetic", "recorded_agent_action"])]
    if "td_workflow_imitation" in b1:
        b1 = b1[b1.td_workflow_imitation.isna()]
    n1 = cfg["b1"]
    want = {}
    kt, lt = cfg["k_bucket_targets"], cfg["length_bucket_targets"]
    for lb_ in LB:
        for kb_ in KB:
            want[(lb_, kb_)] = max(0.0, N * lt[lb_] * kt[kb_] - fx[(lb_, kb_)])
    b1 = b1.assign(_cap=1 + (b1.kind == "choice").astype(int))   # exposure capacity: a second exposure only for choice rows
    sup = b1.groupby(["len_bucket", "k_bucket"])._cap.sum().to_dict()
    cell_target = lr_alloc({c: min(w, sup.get(c, 0)) for c, w in want.items()}, n1)
    cap_src = cfg["b1_source_cap_exposures"]
    used_src = collections.Counter()
    parts = []
    short = 0
    for c in sorted(cell_target, key=lambda c: cell_target[c]):
        tgt = cell_target[c] + short
        g = b1[(b1.len_bucket == c[0]) & (b1.k_bucket == c[1])]
        if g.empty or tgt == 0:
            short = tgt
            continue
        srcs = g.groupby("upstream_source")._cap.sum()
        room = {s: max(0, min(cap_src - used_src[s], int(srcs[s]))) for s in srcs.index}
        alloc = {}
        left, active = tgt, sorted(s for s in room if room[s] > 0)
        while left > 0 and active:
            share = max(1, left // len(active))
            nxt = []
            for s in active:
                gv = min(share, room[s] - alloc.get(s, 0), left)
                alloc[s] = alloc.get(s, 0) + gv
                left -= gv
                if alloc[s] < room[s]:
                    nxt.append(s)
                if left == 0:
                    break
            active = nxt
        got = 0
        for s, n_ in sorted(alloc.items()):
            if n_ <= 0:
                continue
            p = pick(g[g.upstream_source == s], n_, seed, f"b1|{c}|{s}")
            FIXED2[f"b1|{c}|{s}"] = pick.fixed_second
            used_src[s] += int(p.n_exp.sum())
            got += int(p.n_exp.sum())
            parts.append(p)
        short = tgt - got
    b1sel = pd.concat(parts, ignore_index=True) if parts else b1.iloc[:0]
    b1sel = b1sel.drop_duplicates("uid")
    if int(b1sel.n_exp.sum()) < n1:                         # fill any remainder from the largest-room sources, any cell
        rest = b1[~b1.uid.isin(b1sel.uid)]
        need = n1 - int(b1sel.n_exp.sum())
        extra = []
        for s, g in sorted(rest.groupby("upstream_source"), key=lambda x: -len(x[1])):
            r_ = min(cap_src - used_src[s], need, int(g._cap.sum()))
            if r_ <= 0:
                continue
            p = pick(g, r_, seed, f"b1|fill|{s}")
            FIXED2[f"b1|fill|{s}"] = pick.fixed_second
            used_src[s] += int(p.n_exp.sum())
            need -= int(p.n_exp.sum())
            extra.append(p)
            if need <= 0:
                break
        b1sel = pd.concat([b1sel] + extra, ignore_index=True)
    b1sel = b1sel.drop(columns=["_cap"], errors="ignore").assign(m2_bucket="1_core_real_decision_and_disagreement", m2_sub_allocation="b1")
    rep["b1_exposures"] = int(b1sel.n_exp.sum())
    rep["b1_max_source_exposures"] = int(b1sel.groupby("upstream_source").n_exp.sum().max())
    rep["b1_sources"] = b1sel.groupby("upstream_source").n_exp.sum().sort_values(ascending=False).to_dict()
    rep["b1_cell_target"] = {f"{k[0]}|{k[1]}": v for k, v in cell_target.items()}
    allsel = pd.concat([b1sel, fixed], ignore_index=True)
    allsel["parent_route"] = [route(r) for r in allsel.itertuples()]
    if allsel.uid.duplicated().any():
        raise SystemExit("a decision selected twice across buckets")
    rows = []
    for r in allsel.itertuples():
        for k in range(int(r.n_exp)):
            rows.append((r.uid, k))
    ex = pd.DataFrame(rows, columns=["uid", "exposure_k"])
    ex = ex.merge(allsel.drop(columns=["n_exp"]), on="uid")
    allsel.to_parquet(os.path.join(a.out, "m2_selected_decisions_b123.parquet"), index=False)
    ex.to_parquet(os.path.join(a.out, "m2_exposures_b123.parquet"), index=False)
    tot = collections.Counter()
    for r in ex.itertuples():
        tot[(r.len_bucket, r.k_bucket)] += 1
    for k, v in fx.items():
        pass
    rep["fixed_order_second_exposures_unavoidable"] = {k: v for k, v in FIXED2.items() if v}
    rep["fixed_order_second_exposures_total"] = int(sum(FIXED2.values()))
    sec = allsel[allsel.n_exp == 2]
    rep["second_exposures_by_kind"] = sec.assign(kk=sec.kind + "|K" + np.where(sec.K >= 3, ">=3", sec.K.astype(str))).groupby("kk").size().to_dict()
    rep.update(exposures_b123_rendered_here=int(len(ex)), unique_b123=int(allsel.uid.nunique()),
               by_bucket=ex.groupby(["m2_bucket", "m2_sub_allocation"]).size().to_dict(),
               unique_by_bucket=allsel.groupby(["m2_bucket", "m2_sub_allocation"]).size().to_dict(),
               target_basis=ex.groupby(["m2_bucket", "target_basis"]).size().to_dict(),
               parent_route=ex.groupby("parent_route").size().to_dict(),
               td_workflow_imitation_exposures=int(ex.td_workflow_imitation.notna().sum()) if "td_workflow_imitation" in ex else 0,
               cells_b123=dict((f"{k[0]}|{k[1]}", v) for k, v in tot.items()),
               note_alpha="parent_preservation_alpha 0.15 with target-basis routing is the user's spec; the 12B screen ran at alpha 0 and says nothing about alpha 0.15",
               excluded_from_b1=sorted(excl_src))
    def sk(o):
        if isinstance(o, dict):
            return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): sk(v) for k, v in o.items()}
        return o
    rep = sk(rep)
    json.dump(rep, open(os.path.join(a.out, "m2_report_b123.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("exposures_b123_rendered_here", "unique_b123", "b1_exposures", "b1_max_source_exposures", "parent_route",
                                          "fixed_order_second_exposures_total", "second_exposures_by_kind")}, default=str))


if __name__ == "__main__":
    main()
