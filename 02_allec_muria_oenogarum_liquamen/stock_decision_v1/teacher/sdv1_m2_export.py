#!/usr/bin/env python3
"""stock_decision_v1 -> M2 Bucket 4 ("4_stage3_teacher_failure_mined_rows", SOTA_EXECUTION_PACK_2026-10-14/mixture_M2_spec.json):
teacher-mined discovery rows with VERIFIED ground-truth targets, plus the Phase C ledger summary. CPU only.

SELECTION (per decision, from the failure ledger + the merged teacher predictions + the discovery manifest's gold):
  * verification tier in {independent, source_gold} (label_semantics_v1 r2), and not ambiguous_or_defective_label;
  * at least 2 of the 5 models (stock, garas, d3, pplx, olmo; only those that scored the row) are BAD, where a model is bad when
    its native-metric correctness is False, or it puts > 0.35 probability on a single HIGH-LOSS option:
      loss_by_option given  -> options whose regret exceeds half the loss range (loss - min > 0.5 (max - min));
      hard gold (one-hot / tie set) -> options with zero gold mass;
      soft gold (exact posterior, human distribution, ...) -> options with gold mass <= 0.05.
TARGETS: the manifest's gold_by_option_key only (verified ground truth). Teacher probabilities are metadata, never targets
(teacher_supervision.eligible_for_distillation = false).
INDEPENDENT CHECK (the ledger's rule for common teacher failures): a common_teacher_failure whose label comes from an executable
oracle (exact_oracle / execution_check / construction_rule) is satisfied by that oracle; any other common failure is
'pending_two_independent_reviews' and trainable = false until reviewed. Other categories: not_required.
ALLOCATION to --exposures (7,200): trainable rows only; round-robin over source families in a seeded hash order; <= 4 decisions per
parent; <= --family-cap (6%) of the bucket per family; a second exposure per decision only if the first pass falls short, and
(v2, review 2026-10-14) ONLY on choice rows -- K >= 3 first, then K = 2 -- never on score / yes-no rows, whose fixed semantic order
would render the same prompt twice. Every exposure gets an explicit option order (exposure_option_orders; choice orders pairwise
distinct, score / yes-no the stored order) and a prompt identity (sha256 of state, question, kind and the presented options); the
export REFUSES to write if two exposures of a decision share one. The shortfall is reported, never filled with weaker rows.

    python3 sdv1_m2_export.py export  --manifest D.jsonl.gz --ledger L.jsonl.gz --predictions stock.jsonl.gz ... --out E.jsonl.gz
    python3 sdv1_m2_export.py summary --manifest D.jsonl.gz --ledger L.jsonl.gz --report R.json --out S.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sdv1_teacher as st  # noqa: E402
import sdv1_failure_ledger as fl  # noqa: E402

VERSION = "sdv1_m2_export v2 (2026-10-14: second exposures only on re-orderable choice rows; distinct prompts asserted)"
MODELS = ("stock", "garas", "d3", "pplx", "olmo")
ORACLE_CLASSES = ("exact_oracle", "execution_check", "construction_rule")
BAD_MASS = 0.35


def high_loss_keys(gold, loss, soft):
    if loss:
        lo, hi = min(float(v) for v in loss.values()), max(float(v) for v in loss.values())
        return {k for k, v in loss.items() if hi > lo and float(v) - lo > 0.5 * (hi - lo)}
    if not soft:
        return {k for k, g in gold.items() if g <= 1e-12}
    return {k for k, g in gold.items() if g <= 0.05}


def judge(led, preds_by_model, gold, loss):
    soft = led["gold_scoring"] == "soft"
    high = high_loss_keys(gold, loss, soft)
    out = {}
    for m in led["teachers_scored"] + ([led["student"]] if led["student"] in led["native_metric_by_model"] else []):
        p = preds_by_model.get(m)
        if p is None:
            continue
        fails = led["native_metric_by_model"][m]["correct"] is False
        mass = max((p[k] for k in high), default=0.0)
        if fails or mass > BAD_MASS:
            out[m] = {"fails_native_metric": fails, "max_mass_on_one_high_loss_option": round(mass, 4)}
    return out, sorted(high)


def check_status(led):
    if led["category"] != "common_teacher_failure":
        return "not_required"
    return "satisfied_by_executable_oracle" if led["verification_class"] in ORACLE_CLASSES else "pending_two_independent_reviews"


def allocate(rows, exposures, family_cap, seed=20261014):
    """round-robin over families, seeded hash order; <= 4 decisions per parent; <= family_cap * exposures per family."""
    cap = int(family_cap * exposures)
    fam = {}
    for r in rows:
        fam.setdefault(r["source_family"] or "?", []).append(r)
    for f in fam:
        fam[f].sort(key=lambda r: st.sha256_text(f"{seed}|m2b4|{r['decision_uid']}"))
    alloc, per_fam, per_parent, total = {}, {}, {}, 0
    second = {"2a": lambda r: r["kind"] == "choice" and r["K"] >= 3, "2b": lambda r: r["kind"] == "choice" and r["K"] == 2}
    for phase in ("1", "2a", "2b"):
        rnd = 1 if phase == "1" else 2
        ok = (lambda r: True) if phase == "1" else second[phase]
        progress = True
        idx = {f: 0 for f in fam}
        while total < exposures and progress:
            progress = False
            for f in sorted(fam, key=lambda x: st.sha256_text(f"{seed}|fam|{x}")):
                while idx[f] < len(fam[f]):
                    r = fam[f][idx[f]]
                    idx[f] += 1
                    u, p = r["decision_uid"], r["parent_uid"]
                    if alloc.get(u, 0) != rnd - 1 or per_fam.get(f, 0) >= cap or not ok(r):
                        continue
                    if rnd == 1 and per_parent.get(p, 0) >= 4:
                        continue
                    alloc[u] = rnd
                    per_fam[f] = per_fam.get(f, 0) + 1
                    per_parent[p] = per_parent.get(p, 0) + (rnd == 1)
                    total += 1
                    progress = True
                    break
                if total >= exposures:
                    break
        if total >= exposures:
            break
    return alloc, total, per_fam


def exposure_orders(d, n, seed=20261014):
    """n option orders (semantic keys) for a decision's exposures: choice -> pairwise distinct seeded permutations; score / yes-no
    -> the stored semantic order (n must be 1)."""
    keys = st.keys_of(d)
    if d["kind"] != "choice":
        if n > 1:
            raise SystemExit(f"{d['decision_uid']}: {n} exposures of a fixed-order {d['kind']} row would repeat one prompt")
        return [keys]
    orders, k = [], 0
    while len(orders) < n:
        o = keys[:]
        random.Random(int(st.sha256_text(f"{seed}|m2b4-order|{d['decision_uid']}|{k}")[:16], 16)).shuffle(o)
        k += 1
        if o not in orders:
            orders.append(o)
    return orders


def prompt_identity(d, order):
    """what a fixed renderer shows for one exposure: state, question, kind and the options in presented order."""
    opts = dict(d["options"])
    return st.sha256_canon({"state": d["state"], "question": d["instructions"], "kind": d["kind"],
                            "options": [[k, opts[k]] for k in order], "noul_defs": d.get("noul_defs")})


def load_preds(paths):
    by = {}
    for p in paths:
        for _, r in st.iter_jsonl(p):
            if r["transform"] == "base" and r["status"] == "ok":
                by.setdefault(r["decision_uid"], {})[r["teacher_id"]] = r["probabilities"]
    return by


def cmd_export(a):
    man = {r["decision_uid"]: r for _, r in st.iter_jsonl(a.manifest) if not r.get("manifest_header")}
    preds = load_preds(a.predictions)
    allowed = set(a.tiers.split(","))
    cand, seen = [], {"rows": 0, "tier_ok": 0, "bad_ge2": 0}
    for _, led in st.iter_jsonl(a.ledger):
        seen["rows"] += 1
        if led["verification_tier"] not in allowed or led["category"] == "ambiguous_or_defective_label":
            continue
        seen["tier_ok"] += 1
        row = man[led["decision_uid"]]
        g = fl.gold_of(row)
        bad, high = judge(led, preds.get(led["decision_uid"], {}), g["gold"], g.get("loss"))
        if len(bad) < 2:
            continue
        seen["bad_ge2"] += 1
        cs = check_status(led)
        cand.append({"schema": "sdv1-m2-bucket4-row-v1", "decision_uid": led["decision_uid"], "parent_uid": led["parent_uid"],
                     "record_uid": row.get("record_uid"), "source_family": led["source_family"], "area": row.get("area"),
                     "skill": led["skill"], "domain": led["domain"], "kind": led["kind"], "K": led["K"],
                     "target_basis": led["gold_basis"], "label_verification": led["verification_class"],
                     "verification_tier": led["verification_tier"], "label_provenance": led["label_provenance"],
                     "gold_scoring": led["gold_scoring"], "gold_by_option_key": row["gold_by_option_key"],
                     "gold_sha256": led["gold_sha256"], "loss_by_option": row.get("loss_by_option"),
                     "evidence_refs": row.get("evidence_refs") or [], "ledger_category": led["category"],
                     "teachers_scored": led["teachers_scored"], "bad_models": bad, "n_bad": len(bad), "high_loss_keys": high,
                     "independent_check": cs, "trainable": cs != "pending_two_independent_reviews",
                     "teacher_supervision": {"eligible_for_distillation": False,
                                             "reason": "M2 Bucket 4: verified ground-truth targets only"}})
    train = [r for r in cand if r["trainable"]]
    alloc, total, per_fam = allocate(train, a.exposures, a.family_cap)
    ndup = 0
    for r in cand:
        r["exposures"] = alloc.get(r["decision_uid"], 0)
        d = st.normalize_decision(man[r["decision_uid"]])
        orders = exposure_orders(d, r["exposures"]) if r["exposures"] else []
        ids = [prompt_identity(d, o) for o in orders]
        if len(set(ids)) != len(ids):
            ndup += 1
        r["exposure_option_orders"], r["exposure_prompt_identity_sha256"] = orders, ids
    if ndup:
        raise SystemExit(f"DUPLICATE_EXPOSURE_PROMPTS: {ndup} decisions would render one prompt twice (refused)")
    with st.open_text(a.out, "wt") as f:
        for r in cand:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    def cnt_kk(rows):
        c = {}
        for r in rows:
            k = f"{r['kind']}|K{'>=3' if r['K'] >= 3 else r['K']}"
            c[k] = c.get(k, 0) + 1
        return c

    def cnt(key, rows):
        c = {}
        for r in rows:
            c[str(r[key])] = c.get(str(r[key]), 0) + 1
        return dict(sorted(c.items()))
    rep = {"version": VERSION, "manifest_sha256": st.sha256_file(a.manifest), "ledger_sha256": st.sha256_file(a.ledger),
           "predictions_sha256": {os.path.basename(p): st.sha256_file(p) for p in a.predictions}, "funnel": seen,
           "candidates": len(cand), "trainable": len(train), "pending_independent_review": len(cand) - len(train),
           "allocated_decisions": len(alloc), "allocated_exposures": total, "target_exposures": a.exposures,
           "shortfall": max(0, a.exposures - total), "second_exposures": sum(1 for v in alloc.values() if v == 2),
           "second_exposures_by_kind_K": cnt_kk([r for r in train if alloc.get(r["decision_uid"]) == 2]),
           "duplicate_exposure_prompts": 0,
           "family_cap_exposures": int(a.family_cap * a.exposures), "per_family_exposures": dict(sorted(per_fam.items())),
           "by_area": cnt("area", train), "by_tier": cnt("verification_tier", train), "by_category": cnt("ledger_category", train),
           "by_check": cnt("independent_check", cand), "by_target_basis": cnt("target_basis", train),
           "export_md5": st.md5_file(a.out), "export_sha256": st.sha256_file(a.out)}
    json.dump(rep, open(a.out + ".report.json", "w"), indent=1, sort_keys=True)
    print("M2B4_EXPORT", json.dumps({k: rep[k] for k in ("candidates", "trainable", "allocated_exposures", "shortfall", "export_md5")}))
    return 0


def boot_rate(by_parent, n_boot=2000, seed=0):
    """parent-clustered rate of a 0/1 list keyed by parent: point and 95% bootstrap interval."""
    ps = [v for v in by_parent.values() if v]
    n = sum(len(v) for v in ps)
    if not n:
        return {"n": 0, "rate": None, "ci95": None}
    k = sum(sum(v) for v in ps)
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        pick = [ps[rng.randrange(len(ps))] for _ in ps]
        bs.append(sum(sum(v) for v in pick) / sum(len(v) for v in pick))
    bs.sort()
    return {"n": n, "parents": len(ps), "rate": k / n, "ci95": [bs[int(0.025 * (n_boot - 1))], bs[int(0.975 * (n_boot - 1))]]}


def cmd_summary(a):
    man = {r["decision_uid"]: r for _, r in st.iter_jsonl(a.manifest) if not r.get("manifest_header")}
    led = [r for _, r in st.iter_jsonl(a.ledger)]
    out = {"rows": len(led), "by_area_tier_category": {}, "teacher_failure_rates": {}, "teacher_failure_by_area": {}}
    for r in led:
        area = man[r["decision_uid"]].get("area") or "?"
        k = f"{area}|{r['verification_tier']}"
        out["by_area_tier_category"].setdefault(k, {})
        out["by_area_tier_category"][k][r["category"]] = out["by_area_tier_category"][k].get(r["category"], 0) + 1
    for m in MODELS:
        allp, byarea = {}, {}
        for r in led:
            nm = r["native_metric_by_model"].get(m)
            if nm is None or nm["correct"] is None or r["verification_tier"] == "unverified":
                continue
            v = int(nm["correct"] is False)
            allp.setdefault(r["parent_uid"], []).append(v)
            area = man[r["decision_uid"]].get("area") or "?"
            byarea.setdefault(area, {}).setdefault(r["parent_uid"], []).append(v)
        out["teacher_failure_rates"][m] = boot_rate(allp, a.n_boot)
        out["teacher_failure_by_area"][m] = {ar: boot_rate(v, a.n_boot) for ar, v in sorted(byarea.items())}
    rep = json.load(open(a.report)) if a.report else {}
    out["ledger_report_rates"] = {c: {k: rep["random_discovery_sample_rates"][c][k] for k in ("count", "rate", "ci95")}
                                  for c in fl.CATEGORIES} if rep else None
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print("SUMMARY_OK", a.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("--manifest", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tiers", default="independent,source_gold")
    p.add_argument("--exposures", type=int, default=7200)
    p.add_argument("--family-cap", type=float, default=0.06)
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("summary")
    p.add_argument("--manifest", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--report", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.set_defaults(fn=cmd_summary)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
