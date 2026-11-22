#!/usr/bin/env python3
"""(SCREEN4) PRESERVATION targets for the Garas / (99) continuation screen (design tmp/screen4/design.md section 6).

The screen trains with the PRODUCTION trainer (claude_lmtune.py e2c6100a, pinned, unchanged: soft-target CE + RPS on score rows), so
preservation is written into the DATA. For a unit with gold target t, a teacher prediction q (stored option order) and weight lam:

    t' = (t + lam * q) / (1 + lam)                                                   (mix)

Identity (tested):  CE(t, p) + lam * KL(q || p) = (1 + lam) * CE(t', p) - lam * H(q)
and d/dz CE(t', softmax z) = p - t' = [ (1 + lam) p - t - lam q ] / (1 + lam), i.e. training on t' IS the objective
"CE to gold + lam x KL to the teacher" on that unit, scaled by 1 / (1 + lam) (a per-unit loss weight of 0.77 at lam = 0.3; disclosed,
not compensated: the production trainer has no per-unit weights). lam = 0.3 is the AWS strands-decider KL-to-backbone weight
(research/competitor_intel/SUMMARY.md section 9); mixing weight lam / (1 + lam) = 0.23.

Two teachers (user direction 2026-10-12: keep Garas's calibration gains, recover Gemma's evidence-grounded capability):
  replay units (Garas's own training distribution, X99)  q = Garas / (99)'s ORDER-MARGINAL prediction (claude_lmscore, n-perm 4, the
                                                          trained checkpoint a4ee3b50): every unit;
  evidence-grounded units (EG pool)                       q = STOCK Gemma-4-31B-it's order-marginal prediction, TEMPERED by one T fitted
                                                          by NLL on the EG dev split (Gemma's raw probabilities are over-confident:
                                                          DI ECE 0.204 vs Garas 0.049), used only where Gemma's argmax is a gold argmax
                                                          ("teacher agrees with gold"); elsewhere the gold target alone;
  Bayesian-pair units                                     no teacher (exact targets; both parents are known to be wrong there).
Fail-closed: a teacher row with another K, non-finite / non-normalised probabilities, a missing unit, or a target mismatch raise.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "screen4_loss v2 (2026-10-12; Garas line)"
LAM_PARENT = 0.3
LAM_TEACHER = 0.3
EPS = 1e-12
T_BOUNDS = (0.05, 20.0)


def _dist(v, what):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError(f"screen4_loss: {what} is not a finite non-negative vector of length >= 2")
    s = v.sum()
    if s <= 0 or abs(s - 1) > 1e-3:
        raise ValueError(f"screen4_loss: {what} sums to {s:.6f}, not 1")
    return v / s


def mix(t, q, lam):
    """(t + lam q) / (1 + lam); both normalised distributions over the same options."""
    if lam < 0:
        raise ValueError("screen4_loss: lam < 0")
    t, q = _dist(t, "target"), _dist(q, "teacher")
    if len(t) != len(q):
        raise ValueError(f"screen4_loss: target K {len(t)} != teacher K {len(q)}")
    return (t + lam * q) / (1 + lam)


def ce(t, p):
    return float(-(np.asarray(t) * np.log(np.clip(p, EPS, 1))).sum())


def kl(q, p):
    q = np.asarray(q, dtype=np.float64)
    m = q > 0
    return float((q[m] * (np.log(q[m]) - np.log(np.clip(np.asarray(p)[m], EPS, 1)))).sum())


def entropy(q):
    q = np.asarray(q, dtype=np.float64)
    m = q > 0
    return float(-(q[m] * np.log(q[m])).sum())


def temper(p, T):
    """p^(1/T), renormalised (log-space; T > 0)"""
    if not (T > 0 and math.isfinite(T)):
        raise ValueError("screen4_loss: temperature must be positive and finite")
    lp = np.log(np.clip(np.asarray(p, dtype=np.float64), EPS, 1)) / T
    lp -= lp.max()
    e = np.exp(lp)
    return e / e.sum()


def nll_at(P, T_rows, T):
    return float(np.mean([ce(t, temper(p, T)) for p, t in zip(P, T_rows)]))


def fit_temperature(P, T_rows, bounds=T_BOUNDS, iters=80):
    """One temperature minimising the mean soft NLL of tempered P against the targets: golden-section search on log T (the release's own
    procedure, pplx autojev evaluate.py). P / T_rows: lists of per-unit vectors (stored order)."""
    if not P or len(P) != len(T_rows):
        raise ValueError("screen4_loss: no rows to fit a temperature on")
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll_at(P, T_rows, math.exp(a)), nll_at(P, T_rows, math.exp(b))
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll_at(P, T_rows, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll_at(P, T_rows, math.exp(b))
    return math.exp((lo + hi) / 2)


def agrees(q, t, tol=1e-9):
    """the teacher's argmax is one of the gold target's argmaxes"""
    t = np.asarray(t, dtype=np.float64)
    return bool(t[int(np.argmax(q))] >= t.max() - tol)


def teacher_rows(npz_path, uids, targets, ks):
    """{uid: stored-order teacher vector} from a claude_lmscore npz, exact-joined by test_uid with K and target checks."""
    z = np.load(npz_path, allow_pickle=False)
    U = [str(u) for u in z["test_uid"]]
    pos = {u: i for i, u in enumerate(U)}
    if len(pos) != len(U):
        raise ValueError(f"screen4_loss: {npz_path}: duplicated test_uid")
    out = {}
    for u, t, k in zip(uids, targets, ks):
        i = pos.get(u)
        if i is None:
            raise ValueError(f"screen4_loss: {npz_path}: no teacher row for unit {u}")
        if int(z["test_ks"][i]) != k:
            raise ValueError(f"screen4_loss: {npz_path}: unit {u} has K {int(z['test_ks'][i])}, the data {k}")
        tt = np.asarray(z["test_t"][i, :k], dtype=np.float64)
        if not np.allclose(tt, np.asarray(t, dtype=np.float64) / max(float(np.sum(t)), EPS), atol=1e-5):
            raise ValueError(f"screen4_loss: {npz_path}: unit {u}'s target differs from the data")
        out[u] = _dist(z["test_p"][i, :k], f"teacher row {u}")
    return out


def plan_targets(units, roles, parent=None, teacher=None, T_teacher=1.0, lam_parent=LAM_PARENT, lam_teacher=LAM_TEACHER):
    """-> ({uid: new stored-order target}, stats). units: dicts with _uid, target; roles {uid: 'replay'|'eg'|'bayes'}.
    replay -> mix toward parent (every unit); eg -> mix toward tempered teacher where it agrees with gold; bayes -> unchanged."""
    new, st = {}, {"replay": 0, "eg": 0, "eg_teacher_agrees": 0, "eg_gold_only": 0, "bayes": 0}
    for u in units:
        uid, t = u["_uid"], _dist(u["target"], f"gold {u['_uid']}")
        r = roles[uid]
        st[r] += 1
        if r == "replay":
            if parent is None:
                raise ValueError("screen4_loss: replay units need the parent predictions")
            new[uid] = mix(t, parent[uid], lam_parent)
        elif r == "eg":
            if teacher is None:
                raise ValueError("screen4_loss: evidence-grounded units need the teacher predictions")
            q = temper(teacher[uid], T_teacher)
            if agrees(q, t):
                new[uid] = mix(t, q, lam_teacher)
                st["eg_teacher_agrees"] += 1
            else:
                new[uid] = t
                st["eg_gold_only"] += 1
        elif r == "bayes":
            new[uid] = t
        else:
            raise ValueError(f"screen4_loss: role {r!r}")
    return new, st
