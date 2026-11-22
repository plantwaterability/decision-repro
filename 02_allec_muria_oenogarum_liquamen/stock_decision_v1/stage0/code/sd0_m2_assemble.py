#!/usr/bin/env python3
"""Assemble Mixture M2 (stock_decision_v1) from its rendered parts and run the publication checks.

  parts:  buckets 1-3  sd0_m2.py + sd0_render.py (this agent)            decisions_m2_b123*.jsonl.gz / exposures_m2_b123*.jsonl.gz
          bucket 2 U3  BAYES-DATA bayes_zen_m2_u3_v1 (as delivered)      decisions_bzen_m2_u3.jsonl.gz / exposures_bzen_m2_u3.jsonl.gz
          bucket 4     TEACHER sdv1_m2_export.py export rendered here    decisions_m2_b4.jsonl.gz / exposures_m2_b4.jsonl.gz
  checks: 48,000 exposures; every decision uid unique across parts; renderer sha256 identical everywhere; no two exposures of one
          decision share a prompt_sha256; ancestry components of every Stage 0 decision in M2 disjoint from dev_select, confirm_seal
          (+ reserve and unused family records), head_fit and cal_select; gap rows' components touch no Stage 0 record; TD flag on
          every synth_tool_policy row; parent_route on every row.
  output: decisions_M2.jsonl.gz, exposures_M2.jsonl.gz (pos 0..47,999 in a seeded hash order; exposure_id "M2:<pos>"), M2_report.json

  python3 sd0_m2_assemble.py --part NAME=DECISIONS,EXPOSURES ... --units units_pools_r2.parquet --gap-comp comp10/components.parquet
        --gap-units units_gap.parquet --out DIR [--order-from PREV_EXPOSURES_M2 --uid-map OLD_PREFIX=NEW_PREFIX ...]
--order-from (M2 v2+): every exposure inherits its position from the previous release (matched on part, decision uid mapped back
  through --uid-map, exposure_k), so a re-render that changes a few rows keeps every other row at the same position; all exposures
  must match.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os

import pandas as pd

GAP_POOLS = ("gap_supply_v1", "gap_supply_v2")
PANEL_GROUPS = {"dev_select", "dev_select_unused", "dev_select_family_unused", "confirm_seal", "confirm_seal_unused", "confirm_seal_reserve",
                "confirm_seal_reserve_unused", "confirm_seal_family_unused", "head_fit", "head_fit_unused", "cal_select", "cal_select_unused"}


def rows(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--gap-comp", required=True)
    ap.add_argument("--gap-units", required=True)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--total", type=int, default=48000)
    ap.add_argument("--order-from", default="")
    ap.add_argument("--uid-map", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fails, rep = [], {"parts": {}}
    dec, exps = {}, []
    for spec in a.part:
        name, paths = spec.split("=", 1)
        dp, ep = paths.split(",")
        n_d = n_e = 0
        for r in rows(dp):
            if r["decision_uid"] in dec:
                fails.append(f"decision {r['decision_uid']} appears in two parts")
            r["m2_part"] = name
            dec[r["decision_uid"]] = r
            n_d += 1
        for e in rows(ep):
            e["m2_part"] = name
            exps.append(e)
            n_e += 1
        rep["parts"][name] = {"decisions": n_d, "exposures": n_e, "decisions_md5": hashlib.md5(open(dp, "rb").read()).hexdigest(),
                              "exposures_md5": hashlib.md5(open(ep, "rb").read()).hexdigest()}
    if len(exps) != a.total:
        fails.append(f"{len(exps)} exposures != {a.total}")
    ren = {e["renderer_sha256"] for e in exps} | {d["renderer_sha256"] for d in dec.values()}
    if len(ren) != 1:
        fails.append(f"renderer sha256 differs across parts: {sorted(ren)}")
    missing = [e["decision_uid"] for e in exps if e["decision_uid"] not in dec]
    if missing:
        fails.append(f"{len(missing)} exposures without a decision row")
    by = collections.defaultdict(set)
    nexp = collections.Counter()
    for e in exps:
        by[e["decision_uid"]].add(e["prompt_sha256"])
        nexp[e["decision_uid"]] += 1
    dup = [u for u in by if len(by[u]) < nexp[u]]
    rep["distinct_prompt_check"] = {"decisions": len(by), "multi_exposure_decisions": sum(1 for u in nexp if nexp[u] > 1),
                                    "decisions_with_repeated_prompt": len(dup)}
    if dup:
        fails.append(f"{len(dup)} decisions repeat a prompt across exposures, e.g. {dup[:3]}")
    # components
    u = pd.read_parquet(a.units, columns=["uid", "comp", "pool_assign"])
    panel_comps = set(u.loc[u.pool_assign.isin(PANEL_GROUPS), "comp"])
    st0 = u.set_index("uid")
    stage0_uids = [x for x in dec if x in st0.index]
    m2_comps = set(st0.loc[stage0_uids, "comp"])
    inter = m2_comps & panel_comps
    rep["component_check"] = {"stage0_decisions": len(stage0_uids), "stage0_components": len(m2_comps), "panel_components": len(panel_comps),
                              "intersection": len(inter)}
    if inter:
        fails.append(f"{len(inter)} M2 components shared with dev / confirmation / head-fit / calibration")
    gc = pd.read_parquet(a.gap_comp, columns=["rk", "comp", "pool"])
    gap_rk = {x.rsplit("|", 1)[0] for x in dec if x.split("|", 1)[0] in GAP_POOLS}
    gcomp = set(gc.loc[gc.rk.isin(gap_rk), "comp"])
    gap_touch = gc[gc.comp.isin(gcomp) & ~gc.pool.isin(GAP_POOLS)]
    rep["component_check"]["gap_records"] = len(gap_rk)
    rep["component_check"]["gap_components_touching_stage0"] = int(gap_touch.comp.nunique())
    if len(gap_touch):
        fails.append("gap rows share components with Stage 0 records")
    gap_train = set(pd.read_parquet(a.gap_units, columns=["uid", "split"]).query("split == 'train'").uid)
    off = [x for x in dec if x.split("|", 1)[0] in GAP_POOLS and x not in gap_train]
    if off:
        fails.append(f"{len(off)} gap rows outside the gap train split")
    rep["component_check"]["u3_check"] = "BAYES-DATA render_bzen_m2_u3.json component check PASS (0 links to any Stage 0 group)"
    # TD flag / parent route
    tool = [d for d in dec.values() if str(d.get("source_family", "")).endswith("synth_tool_policy")]
    if any(d.get("td_workflow_imitation") != "agent_tool_routing" for d in tool):
        fails.append("a synth_tool_policy row without td_workflow_imitation")
    nroute = [u_ for u_, d in dec.items() if d.get("parent_route") not in ("none", "stock", "garas")]
    if nroute:
        fails.append(f"{len(nroute)} decisions without a valid parent_route")
    # positions
    if a.order_from:
        back = [tuple(x.split("=", 1)) for x in a.uid_map]
        def old_uid(u):
            for o, n in back:
                if u.startswith(n):
                    return o + u[len(n):]
            return u
        prev = {}
        for r in rows(a.order_from):
            prev[(r["m2_part"], r["decision_uid"], int(r["exposure_k"]))] = int(r["pos"])
        miss = 0
        for e in exps:
            k = (e["m2_part"], old_uid(e["decision_uid"]), int(e["exposure_k"]))
            if k in prev:
                e["_k"] = prev[k]
            else:
                miss += 1
                e["_k"] = 10 ** 9
        rep["order_from"] = {"file": a.order_from, "md5": hashlib.md5(open(a.order_from, "rb").read()).hexdigest(), "uid_map": a.uid_map,
                             "matched": len(exps) - miss, "unmatched": miss}
        if miss or len(set(e["_k"] for e in exps)) != len(exps):
            fails.append(f"--order-from: {miss} exposures without a previous position (or duplicate positions)")
    else:
        for e in exps:
            e["_k"] = hashlib.sha256(f"{a.seed}|M2final|{e['m2_part']}|{e['decision_uid']}|{e['exposure_k']}".encode()).hexdigest()
    exps.sort(key=lambda e: e["_k"])
    route_x = collections.Counter()
    bucket_x = collections.Counter()
    td_x = 0
    tok = 0
    lb = collections.Counter()
    kb = collections.Counter()
    with gzip.open(os.path.join(a.out, "exposures_M2.jsonl.gz.partial"), "wt", encoding="utf-8") as f:
        for pos, e in enumerate(exps):
            d = dec[e["decision_uid"]]
            row = {"pos": pos, "exposure_id": f"M2:{pos}", "decision_uid": e["decision_uid"], "exposure_k": int(e["exposure_k"]),
                   "option_order": e["option_order"], "prompt_sha256": e["prompt_sha256"], "n_tokens": int(e["n_tokens"]),
                   "renderer_sha256": e["renderer_sha256"], "m2_bucket": d.get("m2_bucket"), "m2_sub_allocation": d.get("m2_sub_allocation"),
                   "parent_route": d.get("parent_route"), "td_workflow_imitation": d.get("td_workflow_imitation"), "m2_part": e["m2_part"],
                   "part_exposure_id": e.get("exposure_id")}
            if "slot_map" in e:
                row["slot_map"] = e["slot_map"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            route_x[row["parent_route"]] += 1
            bucket_x[row["m2_bucket"]] += 1
            td_x += row["td_workflow_imitation"] is not None
            tok += row["n_tokens"]
            n = row["n_tokens"]
            lb["<=512" if n <= 512 else "513-2048" if n <= 2048 else "2049-8192"] += 1
            kb[str(d.get("K"))] += 1
    os.replace(os.path.join(a.out, "exposures_M2.jsonl.gz.partial"), os.path.join(a.out, "exposures_M2.jsonl.gz"))
    with gzip.open(os.path.join(a.out, "decisions_M2.jsonl.gz.partial"), "wt", encoding="utf-8") as f:
        for uid_ in sorted(dec):
            d = dict(dec[uid_])
            d.pop("m2_part", None)
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    os.replace(os.path.join(a.out, "decisions_M2.jsonl.gz.partial"), os.path.join(a.out, "decisions_M2.jsonl.gz"))
    K = collections.Counter()
    for k, v in kb.items():
        k_ = int(k) if k not in ("None", "nan") else 0
        K["2" if k_ == 2 else "3-5" if k_ <= 5 else "6-16" if k_ <= 16 else "17-32" if k_ <= 32 else "33-52"] += v
    rep.update(exposures=len(exps), unique_decisions=len(dec), renderer_sha256=sorted(ren), tokens_r2=tok,
               exposures_by_bucket=dict(bucket_x), parent_route_exposures=dict(route_x),
               parent_route_decisions=dict(collections.Counter(d.get("parent_route") for d in dec.values())),
               td_workflow_imitation_exposures=td_x, td_workflow_imitation_decisions=len(tool),
               length_buckets=dict(lb), k_buckets=dict(K),
               target_basis_exposures=dict(collections.Counter(dec[e["decision_uid"]].get("target_basis") for e in exps)),
               label_semantics_schema=sorted({str(d.get("label_semantics_schema_sha256"))[:16] for d in dec.values()}),
               checks="passed" if not fails else "FAILED", failures=fails)
    for fn in ("exposures_M2.jsonl.gz", "decisions_M2.jsonl.gz"):
        p = os.path.join(a.out, fn)
        rep[fn] = {"md5": hashlib.md5(open(p, "rb").read()).hexdigest(), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    json.dump(rep, open(os.path.join(a.out, "M2_report.json"), "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps({k: rep[k] for k in ("exposures", "unique_decisions", "checks", "failures", "distinct_prompt_check", "component_check",
                                          "parent_route_exposures", "td_workflow_imitation_exposures", "exposures_by_bucket")}, default=str))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
