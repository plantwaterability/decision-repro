#!/usr/bin/env python3
"""(46) SINGLE-PASS DISTILLATION helpers: the stacked, option-order-marginalised, tempered direct ensemble (the teacher) is
distilled into ONE LoRA scorer that is then run with ONE option order (user direction 2026-09-29: single-pass, no reasoning).

    pool     unseen distillation inputs: build_jev_ext2.py rows no checkpoint ever trained on (the build is fingerprinted
             against jev_data, jev_ext_strict AND jev_ext2), qids made globally unique so scorer rows map back to records;
             a seeded record sample, split train / val
    stack    the teacher: linear pool of the teachers' option-marginalised predictives, then one temperature,
                 q ∝ (sum_m w_m p_m)^(1/tau),
             fitted on HELD-OUT val splits (never the eval sets) under the Bayesian bootstrap (Rubin 1981: unit weights
             ~ Dirichlet(1,...,1)); the teacher uses the posterior mean of (w, log tau), intervals are reported beside it
    targets  writes two training corpora over the pool:
             (a) pure teacher: target = q_teacher  (student minimises KL(teacher || student))
             (b) empirical-Bayes posterior: the rows' own labels are counts (n_raters x label; a gold label with no rater
                 count is one pseudo-observation) under a Dirichlet(kappa * q_teacher) prior; kappa by type-II maximum
                 likelihood (Dirichlet-multinomial evidence) on the multi-rater rows; target = (kappa q + c) / (kappa + n)
    tmerge   (58) merges claude_think.py --shard J/M files (reasoning samples) into one npz aligned to a direct teacher's npz on
             the same units, so the reasoning predictive T joins the stack as one more teacher (fleet/task_58t.sh)

    python3 analysis/claude_distill46.py tmerge --ref claude_d46_pool_d34.npz --shards 'shards/claude_t58_pool_s*of32_zeroshot.npz' \
        --nshards 32 --out claude_d58_pool_t58.npz
    python3 analysis/claude_distill46.py pool --src /data/local/jev_dist46pool_raw --out /data/local/jev_dist46pool
    python3 analysis/claude_distill46.py stack --teachers d18,d25,d34 --set jdval=a.npz,b.npz,c.npz ... --out stack.json
    python3 analysis/claude_distill46.py targets --pool DIR --teachers d18,d25,d34 --pool-npz a.npz,b.npz,c.npz \
        --stack stack.json --out-a DIR_A --out-b DIR_B --summary targets.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("cf", os.path.join(os.path.dirname(HERE), "trainers", "cache_features.py"))
cf = importlib.util.module_from_spec(_s); _s.loader.exec_module(cf)
KMAX_LETTERS = 52
PER_Q = ("questions", "labels", "label_kind")                              # the record fields keyed by question id


# --------------------------------------------------------------------------- pool
def cmd_pool(a):
    recs, seen = [], set()
    for split in ("train", "val"):
        for p in sorted(glob.glob(os.path.join(a.src, split, "**", "*.jsonl"), recursive=True)):
            stem = os.path.relpath(p, os.path.join(a.src, split))[: -len(".jsonl")].replace("/", "__")
            for line in open(p):
                r = json.loads(line)
                if r["id"] in seen:
                    sys.exit(f"duplicate record id {r['id']!r} in {p}")
                seen.add(r["id"])
                r.setdefault("source", stem)
                per_q = [f for f in PER_Q if f in r]
                other = [f for f, v in r.items() if f not in per_q and isinstance(v, dict) and v and set(v) == set(r["questions"])]
                if other:
                    sys.exit(f"{p}: record {r['id']} has per-question fields {other} this rewrite does not know")
                new = {f: {} for f in per_q}
                for qid, q in r["questions"].items():
                    keys, _, _ = cf.question_options(q, r["labels"][qid])
                    if len(keys) > KMAX_LETTERS:
                        continue
                    for f in per_q:
                        new[f][f"{r['id']}|{qid}"] = r[f][qid]
                if new["questions"]:
                    r.update(new)
                    recs.append(r)
    random.Random(a.seed).shuffle(recs)
    out, n_units = {"train": [], "val": []}, {"train": 0, "val": 0}
    for r in recs:
        k = len(r["questions"])
        if n_units["val"] < a.n_val:
            out["val"].append(r); n_units["val"] += k
        elif n_units["train"] < a.n_train:
            out["train"].append(r); n_units["train"] += k
    for split, rs in out.items():
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        with open(os.path.join(a.out, split, "pool.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    src = {}
    for r in out["train"]:
        src[r["source"]] = src.get(r["source"], 0) + len(r["questions"])
    print(f"POOL {len(recs)} candidate records; wrote train {n_units['train']} units / {len(out['train'])} records, "
          f"val {n_units['val']} / {len(out['val'])}")
    print("POOL_SOURCES " + ", ".join(f"{k} {v}" for k, v in sorted(src.items(), key=lambda x: -x[1])))


# --------------------------------------------------------------------------- stack
def load_npz(p):
    z = np.load(p, allow_pickle=False)
    return z["test_p"], z["test_t"], z["test_ks"], z["test_qid"]


def load_set(paths, with_qid=False):
    """[M, n, K] predictives, [n, K] targets, [n, K] option mask (+ the n qids with with_qid). A file may carry `test_valid`
    (tmerge writes it: False where the reasoning teacher has no prediction); such units are dropped for EVERY teacher."""
    ps, ref, valid = [], None, None
    for p in paths:
        P, T, ks, qid = load_npz(p)
        if ref is None:
            ref = (T, ks, qid)
        elif not (np.array_equal(ks, ref[1]) and np.array_equal(qid, ref[2])):
            sys.exit(f"{p}: units differ from {paths[0]} (same --data/--sample/--seed/--exclude-states expected)")
        ps.append(P)
        with np.load(p, allow_pickle=False) as z:
            if "test_valid" in z.files:
                v = z["test_valid"].astype(bool)
                valid = v if valid is None else valid & v
    T, ks, qid = ref
    P = np.stack(ps)
    if valid is not None and not valid.all():
        print(f"{paths[0]} ...: {int((~valid).sum())} of {len(valid)} units dropped for every teacher "
              f"(no prediction in a file with test_valid)", flush=True)
        P, T, ks, qid = P[:, valid], T[valid], ks[valid], qid[valid]
    mask = np.arange(T.shape[1])[None, :] < ks[:, None]
    return (P, T, mask, qid) if with_qid else (P, T, mask)                 # [M, n, K], [n, K], [n, K] (, [n])


def pool_logq(logP, mask, w, tau):
    """log q, q ∝ (sum_m w_m p_m)^(1/tau), renormalised over each unit's options."""
    lp = logsumexp(logP + np.log(np.clip(w, 1e-300, None))[:, None, None], axis=0) / tau
    lp = np.where(mask, lp, -np.inf)
    return lp - logsumexp(lp, axis=1, keepdims=True)


def fit_stack(logP, T, mask, uw):
    M = logP.shape[0]

    def nll(th):
        w = np.exp(th[:M] - logsumexp(th[:M])); lq = pool_logq(logP, mask, w, np.exp(th[M]))
        return -(uw * (T * np.where(mask, lq, 0)).sum(1)).sum() / uw.sum()
    th = minimize(nll, np.zeros(M + 1), method="L-BFGS-B").x
    return np.exp(th[:M] - logsumexp(th[:M])), float(th[M])


def metrics(lq, T, mask):
    q = np.exp(lq)
    return {"log_score": float((T * np.where(mask, lq, 0)).sum(1).mean()),
            "acc": float((q.argmax(1) == T.argmax(1)).mean()),
            "brier": float(np.where(mask, (q - T) ** 2, 0).sum(1).mean())}


def cmd_stack(a):
    names = a.teachers.split(",")
    sets = {}
    for spec in a.set:
        nm, paths = spec.split("=", 1)
        sets[nm] = load_set(paths.split(","))
    kmax = max(s[1].shape[1] for s in sets.values())
    pad = lambda x, v: np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, kmax - x.shape[-1])], constant_values=v)
    logP = np.concatenate([np.log(np.clip(pad(s[0], 0), 1e-12, 1)) for s in sets.values()], 1)
    T = np.concatenate([pad(s[1], 0) for s in sets.values()]); mask = np.concatenate([pad(s[2], False) for s in sets.values()])
    tag = np.concatenate([[nm] * s[1].shape[0] for nm, s in sets.items()])
    rng = np.random.default_rng(a.seed)
    w_hat, lt_hat = fit_stack(logP, T, mask, np.ones(len(T)))
    draws = [fit_stack(logP, T, mask, rng.dirichlet(np.ones(len(T)))) for _ in range(a.draws)]
    W = np.array([d[0] for d in draws]); LT = np.array([d[1] for d in draws])
    w_pm, tau_pm = W.mean(0), float(np.exp(LT.mean()))
    rep = {"teachers": names, "units": {nm: int(s[1].shape[0]) for nm, s in sets.items()},
           "w": w_pm.tolist(), "tau": tau_pm, "w_ml": w_hat.tolist(), "tau_ml": float(np.exp(lt_hat)),
           "w_q05": np.quantile(W, .05, 0).tolist(), "w_q95": np.quantile(W, .95, 0).tolist(),
           "tau_q05": float(np.exp(np.quantile(LT, .05))), "tau_q95": float(np.exp(np.quantile(LT, .95))), "rows": {}}
    cands = {f"{n}": (np.eye(len(names))[i], 1.0) for i, n in enumerate(names)}
    cands.update({f"{n}_T": (np.eye(len(names))[i], None) for i, n in enumerate(names)})
    cands["equal"] = (np.ones(len(names)) / len(names), 1.0)
    cands["teacher"] = (w_pm, tau_pm)
    for cn, (w, tau) in cands.items():
        if tau is None:                                                     # the single model with its own held-out temperature
            tau = float(np.exp(fit_stack(logP[[int(np.argmax(w))]], T, mask, np.ones(len(T)))[1]))
        lq = pool_logq(logP, mask, w, tau)
        rep["rows"][cn] = {"tau": tau, "all": metrics(lq, T, mask),
                           **{nm: metrics(lq[tag == nm], T[tag == nm], mask[tag == nm]) for nm in sets}}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"STACK w = {dict(zip(names, np.round(w_pm, 3)))}  (90% {np.round(rep['w_q05'], 3)} .. {np.round(rep['w_q95'], 3)})  "
          f"tau = {tau_pm:.3f} [{rep['tau_q05']:.3f}, {rep['tau_q95']:.3f}]  on {len(T)} held-out units {rep['units']}")
    for cn, r in rep["rows"].items():
        print(f"  {cn:10s} tau {r['tau']:.3f}  log score {r['all']['log_score']:+.4f}  acc {r['all']['acc']:.4f}  "
              f"Brier {r['all']['brier']:.4f}   " + "  ".join(f"{nm} {r[nm]['log_score']:+.4f}" for nm in sets))


# --------------------------------------------------------------------------- targets
def dirmult_ll(kappa, Q, C, n):
    """sum_i log DirMult(c_i | kappa q_i), multinomial coefficient dropped (constant in kappa)."""
    al = kappa * Q
    return float((gammaln(kappa) - gammaln(kappa + n) + (gammaln(al + C) - gammaln(al)).sum(1)).sum())


def cmd_targets(a):
    names = a.teachers.split(","); st = json.load(open(a.stack))
    assert st["teachers"] == names, (st["teachers"], names)
    w, tau = np.array(st["w"]), float(st["tau"])
    teach = {}
    paths = a.pool_npz.split(",")                                           # the pool's train AND val splits, one file per teacher
    P, T, mask, qids = load_set(paths, with_qid=True)                       # units without a reasoning prediction: unscored
    lq = pool_logq(np.log(np.clip(P, 1e-12, 1)), mask, w, tau)
    for i, q in enumerate(qids):
        teach[str(q)] = np.exp(lq[i][mask[i]])
    rows_kappa, n_drop, n_units = [], 0, 0
    recs = {s: [json.loads(l) for l in open(os.path.join(a.pool, s, "pool.jsonl"))] for s in ("train", "val")}
    for r in recs["train"]:
        n = (r.get("meta") or {}).get("n_raters") or (r.get("meta") or {}).get("n_annotators")
        if not n or n < 2:
            continue
        for qid, q in r["questions"].items():
            if qid in teach:
                _, _, v = cf.question_options(q, r["labels"][qid]); v = np.asarray(v, float); v /= max(v.sum(), 1e-12)
                c = np.round(v * n)
                if c.sum() >= 2:
                    rows_kappa.append((teach[qid], c))
    grid = np.exp(np.linspace(np.log(0.05), np.log(1000), 200))
    if rows_kappa:
        K = max(len(t) for t, _ in rows_kappa)
        Q = np.array([np.pad(t, (0, K - len(t)), constant_values=1e-9) for t, _ in rows_kappa])
        C = np.array([np.pad(c, (0, K - len(c))) for _, c in rows_kappa]); nn = C.sum(1)
        ll = np.array([dirmult_ll(k, Q, C, nn) for k in grid])
        kappa = float(grid[int(np.argmax(ll))])
        if int(np.argmax(ll)) in (0, len(grid) - 1):
            print(f"WARNING kappa at the grid boundary ({kappa:.3g}): the evidence is monotone there", flush=True)
    else:
        kappa, ll = float("nan"), np.array([])
    summ = {"kappa": kappa, "kappa_rows": len(rows_kappa), "w": w.tolist(), "tau": tau, "per_source": {}}
    for arm, out in (("a", a.out_a), ("b", a.out_b)):
        for split in ("train", "val"):
            os.makedirs(os.path.join(out, split), exist_ok=True)
            with open(os.path.join(out, split, "pool.jsonl"), "w") as f:
                for r in recs[split]:
                    r2 = dict(r); r2["questions"], r2["labels"], r2["label_kind"] = {}, {}, {}
                    n = (r.get("meta") or {}).get("n_raters") or (r.get("meta") or {}).get("n_annotators") or 1
                    for qid, q in r["questions"].items():
                        if qid not in teach:
                            n_drop += arm == "a" and split == "train"; continue
                        keys, _, v = cf.question_options(q, r["labels"][qid])
                        v = np.asarray(v, float); v /= max(v.sum(), 1e-12); t = teach[qid]
                        if len(t) != len(keys):
                            sys.exit(f"{qid}: teacher has {len(t)} options, record {len(keys)}")
                        tgt = t if arm == "a" else (kappa * t + n * v) / (kappa + n)
                        lab = {str(k): float(x) for k, x in zip(keys, tgt)}
                        _, _, chk = cf.question_options(q, lab)
                        if not np.allclose(chk, tgt):
                            lab = {str(i): float(x) for i, x in enumerate(tgt)}
                            _, _, chk = cf.question_options(q, lab)
                            assert np.allclose(chk, tgt), qid
                        r2["questions"][qid], r2["labels"][qid], r2["label_kind"][qid] = q, lab, "soft"
                        if arm == "a" and split == "train":
                            n_units += 1
                            s = summ["per_source"].setdefault(r["source"], {"units": 0, "agree": 0, "kl_label_teacher": 0.0})
                            s["units"] += 1; s["agree"] += int(np.argmax(t) == np.argmax(v))
                            s["kl_label_teacher"] += float((v * (np.log(np.clip(v, 1e-12, 1)) - np.log(np.clip(t, 1e-12, 1)))).sum())
                    if r2["questions"]:
                        f.write(json.dumps(r2) + "\n")
    for s in summ["per_source"].values():
        s["agree"] /= max(s["units"], 1); s["kl_label_teacher"] /= max(s["units"], 1)
    summ["train_units"], summ["dropped_unscored"] = n_units, int(n_drop)
    json.dump(summ, open(a.summary, "w"), indent=1)
    print(f"TARGETS kappa {kappa:.3g} (type-II ML on {len(rows_kappa)} multi-rater rows)  train units {n_units}  "
          f"dropped unscored {n_drop}")
    for k, s in sorted(summ["per_source"].items(), key=lambda x: -x[1]["units"]):
        print(f"  {k:32s} {s['units']:6d}  teacher-label argmax agreement {s['agree']:.3f}  KL(label||teacher) {s['kl_label_teacher']:.3f}")


# --------------------------------------------------------------------------- tmerge ((58): the reasoning teacher)
def same_target(a, b, atol):
    """Equal option distributions after renormalisation, within atol (label rounding between corpus builds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and float(np.abs(a / max(a.sum(), 1e-12) - b / max(b.sum(), 1e-12)).max()) <= atol


def cmd_tmerge(a):
    """Merge claude_think.py --shard J/M files (all M of them) into ONE npz on the units of a reference teacher npz scored with
    the same --data/--splits/--sample/--seed/--exclude-states (claude_lmscore.py), rows in the reference's order, padded to its
    option width, so load_set stacks it with the direct teachers.

    Alignment is decided PER SHARD: each shard ran in its own job, and its unit selection came from that slot's copy of the
    corpus. (2026-10-01: three jdval shards ran on slots whose jev_data had been rebuilt after ms_marco left the default build;
    the seeded 1,400-unit sample there was a DIFFERENT sample of the same length.) A shard is aligned by qid when qids are
    unique on both sides (the pool's "record|question" ids), else by position when its unit list has the reference's length
    (held-out qids repeat), else it cannot be aligned. An aligned unit counts as the same unit only if qid, option count and
    target (renormalised, within --target-atol: label rounding) all agree. A shard is ACCEPTED when at least --min-shard-agree of
    its units agree, and then only its agreeing units are kept; otherwise it is REJECTED whole -- its selection is not the
    reference's, and the units that agree there by coincidence (one-hot targets under repeated qids) must not be trusted.
    Reference rows without a kept unit get test_valid False (load_set drops them for every teacher). The merge fails if fewer
    than --min-coverage of the reference units are covered."""
    files = sorted(glob.glob(a.shards)) if any(c in a.shards for c in "*?[") else [f for f in a.shards.split(",") if f]
    groups = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if "test_nshards" not in z.files:
                sys.exit(f"{f}: not a claude_think.py --shard file (no test_nshards)")
            groups.setdefault(int(z["test_nshards"]), {}).setdefault(int(z["test_shard"]), []).append(f)
    M = a.nshards
    if M not in groups:
        sys.exit(f"no shard files with M = {M} among {len(files)} files matching {a.shards!r} (M present: {sorted(groups)})")
    g = groups[M]
    dup = {j: fs for j, fs in g.items() if len(fs) > 1}
    miss = sorted(set(range(M)) - set(g))
    if dup or miss:
        sys.exit(f"shards of {M}: missing {miss}, duplicated {dup}")
    if len(groups) > 1:
        print(f"ignoring shard files with M in {sorted(m for m in groups if m != M)}", flush=True)
    rP, rT, rks, rq = load_npz(a.ref)
    n_ref, kmax = len(rq), rP.shape[1]
    rq_s = [str(q) for q in rq]
    by_q = {q: r for r, q in enumerate(rq_s)} if len(set(rq_s)) == n_ref else None
    kept, K, hows, rejected, n_not_in_ref, n_dropped_in_accepted = {}, None, set(), [], 0, 0
    print(f"tmerge {a.out}: {M} shards vs reference {a.ref} ({n_ref} units); a shard is accepted at >= {a.min_shard_agree} "
          f"agreement (qid, option count, target within {a.target_atol:g})", flush=True)
    for j in range(M):
        with np.load(g[j][0], allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}                                # NpzFile re-reads an array on every z[key]
        units = z["test_unit"].astype(np.int64)
        if np.any(units % M != j):
            sys.exit(f"{g[j][0]}: holds units outside shard {j} of {M}")
        n_units = int(z["test_n_units"])
        has_s = "test_p_samples" in z
        if has_s:
            if K is not None and z["test_p_samples"].shape[1] != K:
                sys.exit(f"{g[j][0]}: {z['test_p_samples'].shape[1]} samples per unit, other shards {K}")
            K = z["test_p_samples"].shape[1]
        rows = []
        for r, u in enumerate(units.tolist()):
            k = int(z["test_ks"][r])
            d = {"p": z["test_p"][r, :k], "qid": str(z["test_qid"][r]), "k": k, "t": z["test_t"][r, :k],
                 "think": float(z["test_think_tokens"][r]), "trunc": bool(z["test_truncated"][r])}
            if has_s:
                d["ps"] = z["test_p_samples"][r, :, :k]; d["ns"] = int(z["test_n_samples"][r])
            rows.append((u, d))
        qids = [d["qid"] for _, d in rows]
        if by_q is not None and len(set(qids)) == len(qids):
            how, tgt = "qid", [by_q.get(d["qid"]) for _, d in rows]
        elif n_units == n_ref:
            how, tgt = "position", [u for u, _ in rows]
        else:
            rejected.append(j)
            print(f"  shard {j:3d}  REJECT  cannot align: its unit list has {n_units} units, the reference {n_ref}, and qids "
                  f"repeat ({g[j][0]})", flush=True)
            continue
        n_in = n_qk = n_exact = n_arg = 0; ok = []
        for (u, d), r in zip(rows, tgt):
            if r is None:
                ok.append(False); continue
            n_in += 1
            qk = d["qid"] == rq_s[r] and d["k"] == int(rks[r])
            n_qk += qk
            n_exact += bool(qk and np.allclose(d["t"], rT[r, : d["k"]], atol=1e-6))
            n_arg += bool(qk and np.argmax(d["t"]) == np.argmax(rT[r, : d["k"]]))
            ok.append(bool(qk and same_target(d["t"], rT[r, : d["k"]], a.target_atol)))
        n_not_in_ref += len(rows) - n_in
        agree = sum(ok) / n_in if n_in else 0.0
        accept = n_in > 0 and agree >= a.min_shard_agree
        print(f"  shard {j:3d}  {'accept' if accept else 'REJECT'}  by {how:8s} units {len(rows):5d}  in reference {n_in:5d}  "
              f"same qid+K {n_qk / max(n_in, 1):.3f}  target exact {n_exact / max(n_in, 1):.3f}  within atol {agree:.3f}  "
              f"argmax {n_arg / max(n_in, 1):.3f}", flush=True)
        if not accept:
            rejected.append(j); continue
        hows.add(how)
        for ((u, d), r), o in zip(zip(rows, tgt), ok):
            if r is None:
                continue
            if not o:
                n_dropped_in_accepted += 1; continue
            if r in kept:
                sys.exit(f"reference row {r} ({rq_s[r]}) is matched by two shard units")
            kept[r] = d
    cov = len(kept) / max(n_ref, 1)
    if rejected:
        print(f"REJECTED shards {rejected} of {M}: their units are not the reference's selection (see the table); those "
              f"reference rows get no reasoning prediction. To recover them, delete those shard files and rerun the shard on a "
              f"node whose corpus reproduces the reference (task_58s.sh pre-checks this).", flush=True)
    if cov < a.min_coverage:
        sys.exit(f"only {len(kept)} of {n_ref} reference units ({cov:.1%}) have an aligned reasoning prediction, below "
                 f"--min-coverage {a.min_coverage}; rejected shards {rejected}")
    P = np.zeros((n_ref, kmax)); valid = np.zeros(n_ref, dtype=bool); think = np.full(n_ref, np.nan)
    trunc = np.zeros(n_ref, dtype=bool); nsamp = np.zeros(n_ref, dtype=np.int64)
    PS = np.full((n_ref, K, kmax), np.nan) if K else None
    for r in range(n_ref):                                                  # placeholder for units without reasoning: uniform
        P[r, : int(rks[r])] = 1.0 / int(rks[r])
    for r, d in kept.items():
        k = d["k"]
        P[r] = 0.0; P[r, :k] = d["p"]; valid[r] = True; think[r] = d["think"]; trunc[r] = d["trunc"]
        if PS is not None:
            nsamp[r] = d["ns"]; ok = ~np.isnan(d["ps"][:, 0])
            PS[r, ok] = 0.0; PS[r, :, :k] = np.where(ok[:, None], d["ps"], np.nan)
    mask = np.arange(kmax)[None, :] < rks[:, None]
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    extra = {}
    with np.load(a.ref, allow_pickle=False) as z:
        for f in ("test_soft", "test_source"):
            if f in z.files:
                extra[f] = z[f]
    if PS is not None:
        extra.update(test_p_samples=PS, test_n_samples=nsamp)
    np.savez_compressed(a.out, test_p=P, test_t=rT, test_ks=rks, test_qid=rq, test_valid=valid, test_mx=P.max(1),
                        test_tot=ent, test_pred_ent=ent, test_think_tokens=think, test_truncated=trunc, **extra)
    v = valid
    ls_ = float((rT[v] * np.where(mask[v], np.log(np.clip(P[v], 1e-12, 1)), 0)).sum(1).mean()) if v.any() else float("nan")
    print(f"TMERGE {a.out}: {int(v.sum())} of {n_ref} reference units have reasoning (accepted {M - len(rejected)} of {M} "
          f"shards, rejected {rejected}; aligned by {', '.join(sorted(hows)) or '-'}; {n_not_in_ref} shard units not in the "
          f"reference; {n_dropped_in_accepted} disagreeing units dropped from accepted shards); "
          + (f"mean samples {nsamp[v].mean():.2f} of {K}; " if PS is not None else "")
          + f"mean think {np.nanmean(think):.0f} tok, truncated {trunc[v].mean():.3f}; acc {(P[v].argmax(1) == rT[v].argmax(1)).mean():.4f}, "
          f"log score {ls_:+.4f}, argmax agreement with the reference teacher {(P[v].argmax(1) == rP[v].argmax(1)).mean():.4f}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool"); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=11000); p.add_argument("--n-val", type=int, default=800)
    p.add_argument("--seed", type=int, default=46)
    p = sub.add_parser("stack"); p.add_argument("--teachers", required=True)
    p.add_argument("--set", action="append", required=True, help="NAME=t1.npz,t2.npz,... (teacher order)")
    p.add_argument("--draws", type=int, default=200); p.add_argument("--seed", type=int, default=46)
    p.add_argument("--out", required=True)
    p = sub.add_parser("targets"); p.add_argument("--pool", required=True); p.add_argument("--teachers", required=True)
    p.add_argument("--pool-npz", required=True, help="t1.npz,t2.npz,... scored with --splits train,val (teacher order)")
    p.add_argument("--stack", required=True); p.add_argument("--out-a", required=True); p.add_argument("--out-b", required=True)
    p.add_argument("--summary", required=True)
    p = sub.add_parser("tmerge"); p.add_argument("--ref", required=True, help="a teacher npz on the same units (row order)")
    p.add_argument("--shards", required=True, help="glob (quoted) or comma list of claude_think.py --shard files")
    p.add_argument("--nshards", type=int, required=True, help="M of the J/M shards to merge (all M must be present)")
    p.add_argument("--target-atol", type=float, default=1e-3, help="max |target difference| after renormalisation for a unit "
                   "to count as the same unit (label rounding between corpus builds)")
    p.add_argument("--min-shard-agree", type=float, default=0.99, help="a shard is accepted only if this share of its units "
                   "agree with the reference; otherwise it ran on another unit selection and is dropped whole")
    p.add_argument("--min-coverage", type=float, default=0.5, help="fail unless this share of reference units is covered")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pool": cmd_pool, "stack": cmd_stack, "targets": cmd_targets, "tmerge": cmd_tmerge}[a.cmd](a)


if __name__ == "__main__":
    main()
