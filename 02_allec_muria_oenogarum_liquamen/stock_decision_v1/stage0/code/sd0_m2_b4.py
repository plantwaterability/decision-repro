#!/usr/bin/env python3
"""M2 Bucket 4 selection from the TEACHER export (sdv1_m2_export.py): trainable rows only, one selection row per exposure with the
export's own per-exposure option order (option_order_keys), joined to the Stage 0 r2 discovery units; parent_route by the M2 rule.
Afterwards check_gold(): the rendered gold equals the export's gold for every decision.
  python3 sd0_m2_b4.py select --export E.jsonl.gz --units units_pools_r2.parquet --schema S.json --out sel_b4.parquet
  python3 sd0_m2_b4.py check  --export E.jsonl.gz --decisions decisions_m2_b4.jsonl.gz"""
import argparse
import gzip
import importlib.util
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sd0_m2", os.path.join(HERE, "sd0_m2.py"))
M2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M2)


def rows(p):
    with gzip.open(p, "rt") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def select(a):
    S = json.load(open(a.schema))
    ex = [r for r in rows(a.export) if r.get("trainable") and int(r.get("exposures") or 0) > 0]
    u = M2.semantics(pd.read_parquet(a.units), S).set_index("uid")
    out = []
    for r in ex:
        uid = r["decision_uid"]
        if uid not in u.index:
            raise SystemExit(f"{uid}: not in the Stage 0 r2 units")
        base = u.loc[uid]
        if base.pool_assign != "discovery":
            raise SystemExit(f"{uid}: pool {base.pool_assign}, not discovery")
        orders = r["exposure_option_orders"]
        if len(orders) != int(r["exposures"]):
            raise SystemExit(f"{uid}: {len(orders)} orders for {r['exposures']} exposures")
        if base.kind == "noul":                           # the teacher tools key noul options true / false; our rows key them yes / no
            orders = [[{"true": "yes", "false": "no"}.get(str(k).lower(), k) for k in o] for o in orders]
        for j, order in enumerate(orders):
            d = base.to_dict()
            d.update(uid=uid, exposure_k=j, option_order_keys=list(order), m2_bucket="4_stage3_teacher_failure_mined_rows",
                     m2_sub_allocation=r.get("ledger_category"))
            out.append(d)
    df = pd.DataFrame(out)
    df["parent_route"] = [M2.route(x) for x in df.itertuples()]
    df["pos"] = range(len(df))
    df.to_parquet(a.out, index=False)
    print(json.dumps({"exposures": len(df), "decisions": int(df.uid.nunique()), "parent_route": df.parent_route.value_counts().to_dict()}))


def check(a):
    ex = {r["decision_uid"]: r for r in rows(a.export) if r.get("trainable") and int(r.get("exposures") or 0) > 0}
    bad = 0
    for d in rows(a.decisions):
        g, h = d["gold_by_option_key"], ex[d["decision_uid"]]["gold_by_option_key"]
        if set(g) != set(h) or any(abs(float(g[k]) - float(h[k])) > 1e-12 for k in g):
            bad += 1
    print(json.dumps({"decisions": len(ex), "gold_mismatch": bad}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "check"])
    ap.add_argument("--export", required=True)
    ap.add_argument("--units", default="")
    ap.add_argument("--schema", default="")
    ap.add_argument("--decisions", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    select(a) if a.cmd == "select" else check(a)
