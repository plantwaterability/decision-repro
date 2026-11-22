#!/usr/bin/env python3
"""r2 regression (outside review: bf_ret_fact 'answer' posteriors were stamped hard_fact; garas actions had no loss table).
From the REAL pools + the sd0_qsemantics output: (1) a bayes_families ret_fact 'answer' row whose gold is the UNIFORM prior and
(2) one whose gold is a non-uniform posterior both resolve to target_basis exact_posterior and land in soft_target_metric_only in
the failure ledger (fabricated all-wrong predictions); (3) a garas_synth action row carries loss_by_option whose argmin is the gold,
scores with action_regret and lands in common_teacher_failure; (4) a bayes_families action row likewise. Also, every rendered
decision row given on the command line is checked: no exact_posterior-declared row is hard_fact, every optimal_action row of a
generator that states costs has a loss table.
  python3 test_qsem_r2.py QSEM.parquet POOLS_DIR TEACHER_DIR SCHEMA [decisions_*.jsonl.gz ...]"""
import gzip
import json
import sys

import pandas as pd

qsem_p, pools, teacher, schema = sys.argv[1:5]
sys.path.insert(0, teacher)
import sdv1_failure_ledger as fl  # noqa: E402

S = json.load(open(schema))
q = pd.read_parquet(qsem_p).set_index("uid")
FAIL = []


def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c:
        FAIL.append(m)


def record(uid):
    pool, mem, line, qid = uid.split("|")
    with open(f"{pools}/{pool}/{mem}") as f:
        for i, l in enumerate(f, 1):
            if i == int(line):
                return json.loads(l), qid


def row_of(uid):
    rec, qid = record(uid)
    qq = rec["questions"][qid]
    if qq["type"] == "choice":
        opts = [{"key": k, "text": v} for k, v in qq["criteria"].items()]
    elif qq["type"] == "noul":
        opts = [{"key": "yes", "text": "Yes"}, {"key": "no", "text": "No"}]
    else:
        opts = [{"key": k, "text": k} for k in qq["levels"]]
    g = rec["labels"][qid]
    tot = sum(g.values())
    qr = q.loc[uid]
    prov = "exact_posterior" if qr.target_basis_q == "exact_posterior" else "rule_oracle"
    return {"decision_uid": uid, "parent_uid": "p|" + uid, "split": "training", "kind": qq["type"], "question": qq.get("instructions", ""),
            "ordered_options": opts, "gold_by_option_key": {k: v / tot for k, v in g.items()}, "target_basis": qr.target_basis_q,
            "label_verification": "exact_oracle", "label_provenance": prov, "label_uncertainty": {"hard": max(g.values()) / tot >= 0.999999},
            "loss_by_option": json.loads(qr.loss_by_option) if isinstance(qr.loss_by_option, str) else None}


def ledger(r):
    g = fl.gold_of(r)
    wrong = min(g["gold"], key=lambda k: (g["gold"][k], k)) if len(set(g["gold"].values())) > 1 else sorted(g["keys"])[-1]
    preds = [dict(decision_uid=g["decision_uid"], teacher_id=m, transform="base", status="ok",
                  probabilities={k: (0.9 if k == wrong else 0.1 / (len(g["keys"]) - 1)) for k in g["keys"]},
                  prediction_sha256="fabricated", n_input_tokens=5) for m in ["stock", "garas", "d3", "pplx", "olmo"]]
    return fl.ledger_row(g, preds)


bfa = q[q.index.str.startswith("bayes_families_v1|") & q.target_semantics_declared.str.contains("format=answer")]
uni = nonuni = None
for uid in bfa.index[:4000]:
    rec, qid = record(uid)
    v = list(rec["labels"][qid].values())
    if max(v) - min(v) < 1e-12 and uni is None:
        uni = uid
    elif max(v) / sum(v) < 0.999999 and max(v) - min(v) > 1e-6 and nonuni is None:
        nonuni = uid
    if uni and nonuni:
        break
for name, uid in (("uniform-prior answer", uni), ("non-uniform posterior answer", nonuni)):
    check(uid is not None, f"bf_ret_fact {name}: found in the pool")
    if uid:
        r = row_of(uid)
        out = ledger(r)
        check(r["target_basis"] == "exact_posterior", f"bf_ret_fact {name}: target_basis {r['target_basis']}")
        check(out["category"] == "soft_target_metric_only", f"bf_ret_fact {name}: ledger category {out['category']}")
for prefix, label in (("synth_supply_v1|", "garas_synth action"), ("bayes_families_v1|", "bayes_families action")):
    act = q[q.index.str.startswith(prefix) & (q.target_basis_q == "optimal_action") & q.loss_by_option.notna()]
    check(len(act) > 0, f"{label}: rows with loss_by_option ({len(act)})")
    if len(act):
        uid = act.index[0]
        r = row_of(uid)
        loss = r["loss_by_option"]
        gk = max(r["gold_by_option_key"], key=r["gold_by_option_key"].get)
        check(min(loss, key=loss.get) == gk, f"{label}: argmin loss == gold ({loss})")
        out = ledger(r)
        check(out["category"] == "common_teacher_failure", f"{label}: ledger category {out['category']}")
gs = q[q.index.str.startswith("synth_supply_v1|") & q.target_semantics_declared.str.contains("exact_minimum_expected_loss")]
check(len(gs) > 0 and gs.loss_by_option.notna().all(), f"garas: every exact_minimum_expected_loss row has a loss table ({len(gs)})")
for path in sys.argv[5:]:
    bad_soft, bad_loss, n = 0, 0, 0
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        src = r.get("target_basis_source") or ""
        if "quantity=posterior" in src or "exact_outcome_predictive_probability" in src:
            n += 1
            bad_soft += r["target_basis"] != "exact_posterior"
        if r["target_basis"] == "optimal_action" and ("exact_minimum_expected_loss" in src or "meta.format=action" in src):
            bad_loss += not r.get("loss_by_option")
    check(bad_soft == 0 and bad_loss == 0, f"{path.split('/')[-1]}: {n} posterior-declared rows all exact_posterior; actions without losses {bad_loss}")
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
