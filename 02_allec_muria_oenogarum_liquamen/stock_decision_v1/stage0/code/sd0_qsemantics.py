#!/usr/bin/env python3
"""Stage 0 per-QUESTION target semantics declared by the generators (REVIEW_2026-10-13_2300 finding 2). One row per unit whose
record declares what its question's distribution means, mapped onto the shared schema's target_basis (label_semantics_v1.json):
  garas synth (meta.target_semantics[question suffix]):
      exact_outcome_predictive_probability -> exact_posterior     exact_minimum_expected_loss     -> optimal_action
      determinate_comparative_decision     -> optimal_action      determinate_evidence_selection  -> hard_fact
      determinate_answerability            -> hard_fact           determinate_grounded_composition -> hard_fact
  bayes_families (declared per record): format=action WITH meta.decision -> optimal_action (loss_by_option = expected cost of each
      plan: p_event * cost_if_event + (1 - p_event) * cost_if_not); otherwise meta.quantity: posterior -> exact_posterior (incl. the
      ret_fact 'answer' posteriors, uniform priors included), label -> hard_fact. (r2: 'answer' was wrongly hard_fact in r1.)
  garas synth exact_minimum_expected_loss rows carry loss_by_option {accept: (1-p)*fp, reject: p*fn} from the stated costs and the
      exact probability; the gold must be the minimum-loss action or the run is refused.
  A hard basis (hard_fact / optimal_action) declared by a generator on a row whose own label_kind is soft is refused.
  An explicit meta.loss_by_option[question] (gap generators) is carried for optimal_action questions; it must cover the options and
      its argmin must be the gold.
  meta.label_source[question] (bayes_pref and the curated rd3 real-data builders), mapped to (target_basis, label_verification):
      exact:posterior_predictive -> exact_posterior / exact_oracle      outcome -> realised_outcome / outcome_record
      expert, expert:* -> hard_fact / source_human_gold                  votes:N -> human_distribution / retained_human_distribution
      expert_probability:* -> outcome_probability / source_human_gold   published_probability:* -> outcome_probability / model_label
      rule_exact -> hard_fact / exact_oracle
A declared value with no mapping refuses the run. Meaning is never inferred from one-hotness (an exact predictive can be one-hot).
Units without a declaration keep their source-level mapping (stage0_provenance_map) downstream.

  python3 sd0_qsemantics.py --pools NAME=DIR ... --unittable unittable_v2.py --out qsem.parquet
"""
import argparse
import importlib.util
import re
from fractions import Fraction
import json
import os

import pandas as pd

GARAS = {"exact_outcome_predictive_probability": "exact_posterior", "exact_minimum_expected_loss": "optimal_action",
         "determinate_comparative_decision": "optimal_action", "determinate_evidence_selection": "hard_fact",
         "determinate_answerability": "hard_fact", "determinate_grounded_composition": "hard_fact"}
# bayes_families: the generator DECLARES the quantity per record (meta.quantity) and the decision for actions (meta.decision);
# the format name alone never decides the meaning: format=action WITH a declared decision/loss -> optimal_action;
# otherwise quantity=posterior -> exact_posterior (an exact posterior, soft or uniform or one-hot), quantity=label -> hard_fact.
BF_QUANTITY = {"posterior": "exact_posterior", "label": "hard_fact"}
COST_RE = re.compile(r"Costs: accepting when the event is absent costs ([0-9/.]+); rejecting when the event is present costs ([0-9/.]+);")
# meta.label_source[question] (curated rd3 real-data builders and bayes_pref): (target_basis, label_verification), by prefix
LSRC = [("exact:posterior_predictive", ("exact_posterior", "exact_oracle")),
        ("outcome", ("realised_outcome", "outcome_record")),
        ("expert_probability:", ("outcome_probability", "source_human_gold")),       # a human expert's published probability
        ("published_probability:", ("outcome_probability", "model_label")),         # a pipeline / model published probability
        ("expert", ("hard_fact", "source_human_gold")),                              # expert / analyst determination
        ("votes:", ("human_distribution", "retained_human_distribution")),           # retained vote / annotator distribution
        ("rule_exact", ("hard_fact", "exact_oracle"))]


def lsrc_map(v):
    for pre, out in LSRC:
        if v == pre or v.startswith(pre if pre.endswith(":") else pre + ":") or (pre == "expert" and v.startswith("expert:")):
            return out
    return None


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jl(x):
    return json.loads(x) if isinstance(x, str) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", action="append", required=True)
    ap.add_argument("--unittable", required=True)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    UT = load(a.unittable, "unittable_v2")
    rows = []
    for spec in a.pools:
        tag, d = spec.split("=", 1)
        for mem, f in UT.iter_files(d, set(a.splits.split(","))):
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                m = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
                for qid in rec.get("questions") or {}:
                    decl, tb, loss, lv = None, None, None, None
                    ts = m.get("target_semantics")
                    if isinstance(ts, dict):
                        k = qid.split("/")[-1]
                        if k in ts:
                            decl = f"meta.target_semantics[{k}]={ts[k]}"
                            tb = GARAS.get(ts[k])
                            if tb is None:
                                raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: unmapped declared semantics {ts[k]!r}")
                            if ts[k] == "exact_minimum_expected_loss":     # loss table from the stated costs and the exact probability
                                tm = jl(m.get("truth_model")) or {}
                                cm = COST_RE.search(str(rec.get("state") or ""))
                                if not cm or "exact_probability" not in tm:
                                    raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: action row without stated costs / exact probability")
                                pi, fp, fn = Fraction(tm["exact_probability"]), Fraction(cm.group(1)), Fraction(cm.group(2))
                                loss = {"accept": float((1 - pi) * fp), "reject": float(pi * fn)}
                                gold = rec["labels"][qid]
                                if max(gold, key=gold.get) != min(loss, key=loss.get):
                                    raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: gold action is not the minimum-expected-loss action")
                    elif m.get("format") and str(m.get("generator", "")).startswith("build_bayes_families"):
                        fmt, qty = m["format"], m.get("quantity")
                        dec = jl(m.get("decision")) if fmt == "action" else None
                        if fmt == "action" and dec and dec.get("plans"):
                            decl = f"meta.format=action+meta.decision(plans={len(dec['plans'])})"
                            tb = "optimal_action"
                            p = float(dec["p_event"])
                            loss = {pl["key"]: p * float(pl["cost_if_event"]) + (1 - p) * float(pl["cost_if_not"]) for pl in dec["plans"]}
                        else:
                            decl = f"meta.quantity={qty} (format={fmt})"
                            tb = BF_QUANTITY.get(qty)
                        if tb is None:
                            raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: undeclared bayes_families quantity {qty!r} (format {fmt!r})")
                    elif isinstance(m.get("label_source"), dict) and qid in m["label_source"]:
                        v = str(m["label_source"][qid])
                        decl = f"meta.label_source[{qid}]={v}"
                        mp = lsrc_map(v)
                        if mp is None:
                            raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: unmapped label_source {v!r}")
                        tb, lv = mp
                    elo = (m.get("loss_by_option") or {}).get(qid) if isinstance(m.get("loss_by_option"), dict) else None
                    if decl and elo is not None:            # an explicitly declared per-question loss table (gap generators)
                        if tb != "optimal_action":
                            raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: loss_by_option declared on a non-action question ({tb})")
                        loss = {str(k): float(v) for k, v in elo.items()}
                        gold = rec["labels"][qid]
                        if set(loss) != set(gold) or max(gold, key=gold.get) != min(loss, key=loss.get):
                            raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: declared loss table does not cover the options or its argmin is not the gold")
                    lk = (rec.get("label_kind") or {}).get(qid)
                    if decl and tb in ("hard_fact", "optimal_action") and lk == "soft" and not decl.startswith("meta.label_source"):
                        raise SystemExit(f"{tag}|{mem}|{line_no}|{qid}: declared {decl} -> {tb} but the generator's label_kind is soft")
                    if decl:
                        rows.append({"uid": f"{tag}|{mem}|{line_no}|{qid}", "target_semantics_declared": decl, "target_basis_q": tb,
                                     "label_verification_q": lv, "label_kind_declared": lk,
                                     "loss_by_option": json.dumps(loss, sort_keys=True) if loss else None})
    df = pd.DataFrame(rows)
    df.to_parquet(a.out + ".partial", index=False)
    os.replace(a.out + ".partial", a.out)
    print(json.dumps({"rows": len(df), "by_basis": df.groupby("target_basis_q").size().to_dict(),
                      "by_declared": df.target_semantics_declared.str.replace(r"\[.*?\]", "", regex=True).value_counts().head(20).to_dict()}))


if __name__ == "__main__":
    main()
