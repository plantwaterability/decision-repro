#!/usr/bin/env python3
"""bayes_zen_report.py: the bayes_pref-style readout (family x rounds) on the stock_decision_v1 bayes_zen sanity sets.

  python3 bayes_zen_report.py --repo JEV --root SANITY [--set bayes_zen|bayes_zen_rec] [--with-refs --work DIR]
                              [--pred NAME=NPZ ...] [--out-json F]

The sets are ordinary jev eval sets (<root>/<set>/eval/<family>.jsonl), read by analysis/gde1_score.py's SetData / align /
score_model (Cromwell floor 1e-3, as gde1_score), with the r-gain statistics of build_bayes_pref.py's _cluster_deltas
(clusters = episodes). Per family x r: KL, LS, Brier, acc vs the gold argmax, acc vs the realised pick, expected regret
(EU_max - sum_j Q_j EU_j, from meta.eu_by_option) and argmax regret. Gains: LS(r0) - LS(r1) and LS(r1) - LS(r4) with
cluster-bootstrap CIs and the share of the exact gain (the belief-update statistic: the exact predictive is a martingale
under the declared model, so a non-updater gains 0 in expectation).

Prediction npz: lmscore format (test_p, test_t, test_ks, test_qid; test_uid = the rowid uid, so rows join by identity).
--with-refs writes and scores uniform / no_update / first_round_only / exact reference predictors.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", default="bayes_zen")
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--with-refs", action="store_true")
    ap.add_argument("--work", default="")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    repo = Path(a.repo)
    gs = load("gde1_score", repo / "analysis" / "gde1_score.py")
    bp = load("build_bayes_pref", repo / "build_bayes_pref.py")
    cf = gs.load_cf()
    sd = gs.SetData(cf, a.root, a.set, "eval", domain="X")
    meta = {}
    for p in sorted((Path(a.root) / a.set / "eval").glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            meta[r["id"]] = r["meta"]
    M = [meta[u["record_id"]] for u in sd.units]
    fam = np.array([m["family"] for m in M])
    # v2.1: never pool the archive's filtered episodes with unfiltered generators or the human panel (review item 4)
    pool = np.array(["pool:filtered_release" if f.startswith("zen_") and f != "zen_human" else
                     "pool:unfiltered_generator" if f.startswith("gen_") else "pool:human_fitted_eps" for f in fam])
    rr = np.array([m["rounds_observed"] for m in M])
    real = np.array([m["realized_choice_index"] for m in M])
    EU = np.array([m["eu_by_option"] for m in M], float)
    preds = dict(p.split("=", 1) for p in a.pred)
    if a.with_refs:
        work = Path(a.work)
        work.mkdir(parents=True, exist_ok=True)
        T = np.array([np.asarray(u["target"], float) / sum(u["target"]) for u in sd.units])
        refs = {"uniform": np.full((sd.n, 3), 1 / 3), "exact": np.array([m["gold_exact"] for m in M], float)}
        for k in ("no_update", "first_round_only"):          # next_choice sets only (recommend rows carry no predictive refs)
            if all(f"ref_{k}" in m for m in M):
                refs[k] = np.array([m[f"ref_{k}"] for m in M])
        if any((m.get("variants") or {}).get("P_tie") for m in M):   # v2: the unfiltered tie-normalised predictive
            refs["unfiltered_tie_normalised"] = np.array([(m.get("variants") or {}).get("P_tie") or m["gold_exact"] for m in M], float)
        for name, P in refs.items():
            p = work / f"pred_{name}_{a.set}.npz"
            np.savez_compressed(p, test_p=P, test_t=T, test_ks=sd.K, test_qid=np.array(sd.qid), test_uid=np.array(sd.uids),
                                test_source=np.array(sd.source), test_soft=np.array([u["label_kind"] == "soft" for u in sd.units]))
            preds = {name: str(p), **preds}
    ex_sc, _ = gs.score_model(sd, np.array([m["gold_exact"] for m in M], float), np.arange(sd.n), 1e-3)
    fams = sorted(set(fam))
    pools = sorted(set(pool))
    out = {}
    for name, path in preds.items():
        z = np.load(path, allow_pickle=True)
        ui, ri = gs.align(sd, z)
        Q = np.asarray(z["test_p"], float)[ri][:, :3]
        Q = Q / Q.sum(1, keepdims=True)
        sc, bad = gs.score_model(sd, Q, ui, 1e-3)
        top = np.isclose(Q, Q.max(1, keepdims=True), rtol=0, atol=1e-12)
        acc_real = top[np.arange(len(ui)), real[ui]] / top.sum(1)
        eu = EU[ui]
        reg_exp = eu.max(1) - (Q * eu).sum(1)
        reg_arg = eu.max(1) - (np.where(top, eu, 0).sum(1) / top.sum(1))
        cols = {"ls": sc["ls"], "kl": sc["kl"], "acc_gold": sc["acc"], "acc_oracle": acc_real, "ls_exact": ex_sc["ls"][ui]}
        tab = {"invalid_rows": bad, "coverage": len(ui) / sd.n}
        for f in fams + pools:
            sel_f = (pool[ui] == f) if f.startswith("pool:") else fam[ui] == f
            cells = {}
            for k in range(5):
                s_ = sel_f & (rr[ui] == k)
                if s_.any():
                    cells[f"r{k}"] = {"n": int(s_.sum()), "kl": float(sc["kl"][s_].mean()), "ls": float(sc["ls"][s_].mean()),
                                      "brier": float(sc["bs"][s_].mean()), "acc_gold": float(sc["acc"][s_].mean()),
                                      "acc_realised": float(acc_real[s_].mean()), "regret_expected": float(reg_exp[s_].mean()),
                                      "regret_argmax": float(reg_arg[s_].mean())}
            for ra, rb in ((0, 1), (1, 4)):
                cells.update(bp._cluster_deltas(sd.cl[ui], rr[ui], sel_f, ra, rb, cols, a.boot, 20261013))
            tab[f] = cells
        out[name] = tab
    for name, t in out.items():
        print(f"\n== {name} ({a.set}; invalid {t['invalid_rows']}, coverage {t['coverage']:.3f})")
        print(f"{'family':38s} {'r':>2s} {'n':>5s} {'KL':>7s} {'LS':>7s} {'Brier':>7s} {'accG':>6s} {'accR':>6s} {'regE':>7s} {'regA':>7s}")
        for f in fams + pools:
            for k in range(5):
                c = t[f].get(f"r{k}")
                if c:
                    print(f"{f:38s} {k:2d} {c['n']:5d} {c['kl']:7.4f} {c['ls']:7.4f} {c['brier']:7.4f} {c['acc_gold']:6.3f} "
                          f"{c['acc_realised']:6.3f} {c['regret_expected']:7.4f} {c['regret_argmax']:7.4f}")
            for m_ in ("ls_gain_r0_to_r1", "ls_gain_r1_to_r4", "share_of_exact_gain_r1_to_r4"):
                v = t[f].get(m_)
                if v:
                    ci = f" [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" if "ci95" in v else ""
                    print(f"{f:38s}    {m_:30s} {v['delta']:+.4f}{ci}")
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
