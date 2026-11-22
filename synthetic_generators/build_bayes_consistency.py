#!/usr/bin/env python3
"""build_bayes_consistency.py -- the BAYESIAN-CONSISTENCY suite (research/bayes_consistency.md): grouped INTERVENTION items on the
(94) numeric corpus's families, so a model that already reads build_numeric_jev.py's 50-bin questions can be tested for Bayesian
consistency and for decisions under declared losses without any new training.

Every group shares one underlying problem (context, stated prior, likelihood, data, wording choices). Its members differ by ONE
intervention, and each member carries its own exact target. The renderers, axes, bins and exact distributions are
build_numeric_jev.py's (imported, not copied); the few sentences a new intervention needs are listed under NEW PHRASING and flagged
per record (meta.bc.novel).

Group types (meta.bc.type; ~240-300 groups each, split evenly over coin / rate / normal_mean)
  prior_only             nodata (zero observations -> the stated prior) + withdata (same prior, data)
  prior_swap             base / strength (same mean, prior strength x k) / mean (same strength, mean moved); same data, one SHARED axis
  exchangeable_reorder   the same raw observations in three orders (coin: H/T sequences 5-160 long; rate: counts; normal: readings)
  raw_vs_sufficient      raw observations vs their sufficient statistics stated directly (sequence vs counts; list vs total; list vs mean)
  sequential_vs_batch    joint (all data at once) / two_batch (batch 1 then batch 2) / stated_post (prior, batch 1, "the posterior is
                         now X", batch 2) / updated_prior (X stated as the prior, then batch 2); plus the CHAIN: b1 (posterior after
                         batch 1) and xchain_mm / xchain_hist (the EXACT batch-1 bins passed on as a moment-matched prior or as a
                         histogram prior, then batch 2). Pass 2 (`chain2`) builds mchain_mm / mchain_hist from a MODEL's own b1 output.
  reparam_units          the same question on an equivalent axis: decimal (proportion / percent, g / kg, mm / cm, ms / s, V / mV,
                         per page / per 100 pages, ...: bins aligned exactly) or non-decimal (per day / per week, C / F, minutes / hours)
  posterior_vs_predictive the parameter posterior + the posterior predictive of the next observation / of the next k
  event_coherence        full distribution + P(> t1), P(<= t1), P(<= t2), P(t1 < . <= t2) as yes/no questions + a 3-way choice
  loss_invariance        no loss + squared / absolute / pinball 0.1 / pinball 0.9 / "underestimating costs 5x" stated in the prompt;
                         the BELIEF must not move; exact Bayes actions in meta.bc.actions

Truth: one parameter per group, drawn from the anchor member's STATED prior (meta.truth.source == "prior") except an OFF-PRIOR slice
(P_OFF = 20% of groups, source "broad": build_numeric_jev.py's broad draws) reported separately; calibration / SBC / realised-value
claims use source == "prior" only. Members whose stated prior differs from the one the truth was drawn from say "other_prior".

AXES v2 (`build --axes v2`; research/numeric_v2_axes.md). v1's declared axes are placed around each member's exact answer (or the
union of a group's answers), so the bin labels carry the answer; v1 stays byte-identical as the "assisted-range" suite. In v2 no axis
reads a prior or an exact target: rate / normal axes come from the VISIBLE DATA only (build_numeric_jev.data_interval / evidence_axis,
zoom / offset from the group's own axis stream), or from a fixed per-context range where the group shares no data; coin is the
natural [0, 1] axis as before. Every member that answers the same question shares ONE axis, chosen from the evidence ALL of them see:
  prior_only            no shared evidence (nodata has none) -> the context's fixed range, shared by nodata and withdata
  prior_swap            the shared data -> identical bins for base / strength / mean (and identical evidence-only prompts)
  reorder / raw_vs_suff the (common) sufficient statistics -> one axis for every order / form
  sequential_vs_batch   batch 2 (the only data updated_prior shows) -> one axis for joint / two_batch / stated_post / updated_prior
                        and the chain members (xchain / mchain); b1 (another question) gets batch 1's axis
  reparam_units         the data, in each unit (decimal: the exact rescaled bins; otherwise the same draws on the converted interval)
  posterior_vs_predictive  the data, with the predictive's spread (sigma^2 / n + sigma^2 [/ k]) for the binned predictives
  event_coherence       the data; the thresholds are bin edges nearest two random quantiles of the data's likelihood (not the
                        posterior's)
  loss_invariance       the data
The exact mass folded into end bins is recorded (meta.exact.folded_*) and summarised in the manifest, never used to redraw.

  .venv/bin/python build_bayes_consistency.py build --out tmp/bc/bayes_consistency_v1            # the suite (~9.4k items)
  .venv/bin/python build_bayes_consistency.py build --out tmp/numv2/bayes_consistency_v2 --axes v2 \
      --numeric-train tmp/numv2/numeric_v2                                                       # v2 axes
  .venv/bin/python build_bayes_consistency.py build --out DIR --scale 0.1                        # smoke
  .venv/bin/python build_bayes_consistency.py check DIR                                          # schema / labels / exactness
  .venv/bin/python build_bayes_consistency.py chain2 --data DIR --npz pass1.npz --out DIR2       # pass 2 from a model's b1 output
  .venv/bin/python build_bayes_consistency.py synth --data DIR --pred prior_only --out X.npz     # a synthetic (broken) predictor
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import special as sp

HERE = Path(__file__).resolve().parent


def _import_bn():
    """build_numeric_jev.py from beside this file (repo root, or the tarball's tools/), else one level up."""
    for d in (HERE, HERE.parent):
        if (d / "build_numeric_jev.py").exists():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import build_numeric_jev
            return build_numeric_jev
    raise ImportError("build_numeric_jev.py not found beside build_bayes_consistency.py")


bn = _import_bn()
VERSION = "build_bayes_consistency.py v1 (2026-10-04)"
VERSION_V2 = "build_bayes_consistency.py v2-axes (2026-10-05)"
AXES = ("v1", "v2")
DEFAULT_AXES = "v1"                 # build(axes=None) uses this (wrappers that call build() without the argument can set it)
NS = "bayes_consistency/v1"
K = bn.K
LICENSE = ("synthetic (ours), commercial-OK: generated by build_bayes_consistency.py with build_numeric_jev.py's renderers, "
           "no third-party data or text")
FAMILIES = ("coin", "rate", "normal_mean")
TYPES = ("prior_only", "prior_swap", "exchangeable_reorder", "raw_vs_sufficient", "sequential_vs_batch", "reparam_units",
         "posterior_vs_predictive", "event_coherence", "loss_invariance")
GROUPS = {"prior_only": 300, "prior_swap": 300, "exchangeable_reorder": 300, "raw_vs_sufficient": 300,
          "sequential_vs_batch": 240, "reparam_units": 300, "posterior_vs_predictive": 300, "event_coherence": 240,
          "loss_invariance": 240}
P_OFF = 0.2
SEQ_BUCKETS = ((5, 10), (11, 20), (21, 40), (41, 80), (81, 160))
LONG_SEQ = 40                       # order_long / order_short synthetic predictors switch here; eval reports reorder by it
LABEL_DP = bn.LABEL_DP
HIST_DP = 4                         # histogram priors are stated to this many decimals
# loss name -> (quantile of the Bayes action or None for the mean, the sentence's cost wording)
LOSSES = {"squared": None, "absolute": 0.5, "pinball_0.1": 0.1, "pinball_0.9": 0.9, "asym_5x": 5 / 6}
COUNT_MAX = 10                      # rate count predictive: levels 0..9 and "10 or more"


# ============================================================================ small helpers
class Pick:
    """A stand-in for random.Random that makes build_numeric_jev's renderers (beta_prior_text, coin_question) pick FIXED templates,
    so every member of a group is worded alike."""

    def __init__(self, i=0, u=0.0):
        self.i, self.u = int(i), float(u)

    def choice(self, seq):
        return seq[self.i % len(seq)]

    def random(self):
        return self.u


def Fr(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    return Fraction(str(x))


def dec_str(x, max_dp=12):
    """Exact finite-decimal text of a Fraction / str / float, no trailing zeros; ValueError if it needs more than max_dp places."""
    f = Fr(x)
    for dp in range(max_dp + 1):
        v = f * 10 ** dp
        if v.denominator == 1:
            s = str(abs(v.numerator)).rjust(dp + 1, "0")
            txt = s if dp == 0 else s[:-dp] + "." + s[-dp:]
            return ("-" if f < 0 else "") + txt
    raise ValueError(f"{f} is not a decimal with <= {max_dp} places")


def n_sig(x):
    """Significant digits of an exact decimal (99 if it is not a finite decimal)."""
    try:
        t = dec_str(x).replace("-", "").replace(".", "").lstrip("0").rstrip("0")
    except ValueError:
        return 99
    return max(1, len(t))


def pl(n, pair):
    return bn._pl(n, pair)


def seed(*parts):
    return bn.seed_of(*parts)


def logu(rng, lo, hi):
    return bn.logu(rng, lo, hi)


def rnd(v):
    return [round(float(x), LABEL_DP) for x in v]


# ============================================================================ extra distributions (natural units)
def ppf_bisect(cdf, q, lo, hi, it=200):
    q = np.atleast_1d(np.asarray(q, float))
    a, b = np.full(q.shape, float(lo)), np.full(q.shape, float(hi))
    for _ in range(it):
        m = (a + b) / 2
        c = np.asarray(cdf(m)) < q
        a, b = np.where(c, m, a), np.where(c, b, m)
    return (a + b) / 2


class UniformD:
    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)
        self.mean, self.var = (self.lo + self.hi) / 2, (self.hi - self.lo) ** 2 / 12
        self.support = (self.lo, self.hi)
        self.pieces = (np.array([self.lo, self.hi]), np.array([1.0]))

    def cdf(self, x):
        return np.clip((np.asarray(x, float) - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def ppf(self, q):
        return self.lo + np.asarray(q, float) * (self.hi - self.lo)

    def params(self):
        return {"dist": "uniform", "params": {"lo": self.lo, "hi": self.hi}}


class HistD:
    """Piecewise-uniform density on `edges` with bin masses `masses` (a histogram prior, or a model's binned answer read as
    uniform within each bin; folded end bins are read as ordinary finite bins)."""

    def __init__(self, edges, masses):
        e = np.asarray(edges, float)
        m = np.clip(np.asarray(masses, float), 0.0, None)
        m = m / m.sum() if m.sum() > 0 else np.full(len(e) - 1, 1.0 / (len(e) - 1))
        self.e, self.m = e, m
        self.Fe = np.concatenate([[0.0], np.cumsum(m)])
        self.Fe[-1] = 1.0
        mid, w = (e[:-1] + e[1:]) / 2, np.diff(e)
        self.mean = float((m * mid).sum())
        self.var = float((m * ((mid - self.mean) ** 2 + w ** 2 / 12)).sum())
        self.support = (float(e[0]), float(e[-1]))
        self.pieces = (e, m)

    def cdf(self, x):
        return np.interp(np.asarray(x, float), self.e, self.Fe)

    def ppf(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        k = np.clip(np.searchsorted(self.Fe, q, side="left") - 1, 0, len(self.m) - 1)
        pk = self.m[k]
        frac = np.where(pk > 0, (q - self.Fe[k]) / np.maximum(pk, 1e-300), 0.0)
        return self.e[k] + np.clip(frac, 0, 1) * (self.e[k + 1] - self.e[k])

    def params(self):
        return {"dist": "hist", "params": {"edges": [float(x) for x in self.e], "masses": [float(x) for x in self.m]}}


class ShiftD:
    def __init__(self, base, delta):
        self.b, self.d = base, float(delta)
        self.mean, self.var = base.mean + self.d, base.var
        self.support = (base.support[0] + self.d, base.support[1] + self.d)

    def cdf(self, x):
        return self.b.cdf(np.asarray(x, float) - self.d)

    def ppf(self, q):
        return np.asarray(self.b.ppf(q)) + self.d

    def params(self):
        return {"dist": "shift", "params": {"base": self.b.params(), "delta": self.d}}


def _lik_cdf(family, st, sigma):
    """The normalised likelihood of the data as a distribution on the parameter: its CDF and survival function (for stable
    differences far in a tail). coin: Beta(h + 1, f + 1); rate: Gamma(tot + 1, T); normal: N(xbar, sigma^2 / n)."""
    if family == "coin":
        a, b = st["h"] + 1.0, st["n"] - st["h"] + 1.0
        return (lambda x: sp.betainc(a, b, np.clip(x, 0, 1))), (lambda x: sp.betainc(b, a, np.clip(1 - np.asarray(x), 0, 1)))
    if family == "rate":
        k, T = st["tot"] + 1.0, float(st["T"])
        return (lambda x: sp.gammainc(k, T * np.maximum(x, 0))), (lambda x: sp.gammaincc(k, T * np.maximum(x, 0)))
    s = float(sigma) / math.sqrt(st["n"])
    xb = float(st["S"]) / st["n"]
    return (lambda x: sp.ndtr((np.asarray(x) - xb) / s)), (lambda x: sp.ndtr((xb - np.asarray(x)) / s))


class HistPostD:
    """Posterior of a HISTOGRAM prior (piecewise-uniform density on edges, masses) times a family likelihood: exact (closed-form
    CDF differences of the normalised likelihood, combined in log space so far tails do not underflow). If the likelihood is
    negligible on the whole prior support (only a badly broken pass-1 answer does this) the prior histogram is kept (degenerate)."""

    def __init__(self, edges, masses, family, st, sigma=None):
        self.e = np.asarray(edges, float)
        m = np.clip(np.asarray(masses, float), 0, None)
        self.m = m / m.sum()
        c = self.m / np.diff(self.e)
        with np.errstate(divide="ignore"):
            self.logc = np.where(c > 0, np.log(np.maximum(c, 1e-300)), -np.inf)
        self.family = family
        self.L, self.S = _lik_cdf(family, st, sigma)
        lw = self.logc + self._logdiff(self.e[:-1], self.e[1:])
        self.degenerate = not np.isfinite(lw).any()
        self.logZ = float(sp.logsumexp(lw)) if not self.degenerate else 0.0
        self._prior = HistD(self.e, self.m)
        self.support = (float(self.e[0]), float(self.e[-1]))
        g = np.linspace(self.e[0], self.e[-1], 4001)
        F = self.cdf(g)
        dF, mid = np.diff(F), (g[:-1] + g[1:]) / 2
        self.mean = float((dF * mid).sum())
        self.var = float((dF * (mid - self.mean) ** 2).sum() + ((g[1] - g[0]) ** 2 / 12))

    def _logdiff(self, a, b):
        """log(L(b) - L(a)), elementwise, from the CDF or (in the upper tail) the survival function."""
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        la, lb = self.L(a), self.L(b)
        d1 = lb - la
        d2 = self.S(a) - self.S(b)
        d = np.where((d1 > 1e-280) & (d1 > 1e-9 * lb), d1, np.where(d2 > 0, d2, d1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, np.log(np.maximum(d, 1e-320)), -np.inf)

    def cdf(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        if self.degenerate:
            return self._prior.cdf(x)
        a, b = self.e[:-1][None, :], self.e[1:][None, :]
        X = np.clip(x[:, None], a, b)
        M = self.logc[None, :] + self._logdiff(np.broadcast_to(a, X.shape), X)
        with np.errstate(invalid="ignore"):
            out = np.exp(sp.logsumexp(M, axis=1) - self.logZ)
        return np.clip(np.nan_to_num(out, nan=0.0), 0, 1)

    def ppf(self, q):
        return ppf_bisect(self.cdf, q, self.e[0], self.e[-1], it=60)

    def params(self):
        return {"dist": "hist_posterior", "params": {"edges": [float(x) for x in self.e],
                                                       "prior_masses": [float(x) for x in self.m]},
                "degenerate": bool(self.degenerate)}


class ConvD:
    """belief (+) N(0, tau^2): the predictive of a normal reading (tau = sigma) or of the mean of k readings (sigma / sqrt k)."""

    def __init__(self, belief, tau):
        self.b, self.tau = belief, float(tau)
        self.mean, self.var = belief.mean, belief.var + self.tau ** 2
        lo, hi = belief.support
        self.support = (-math.inf, math.inf)
        self._lo = (lo if math.isfinite(lo) else belief.mean - 12 * math.sqrt(belief.var)) - 12 * self.tau
        self._hi = (hi if math.isfinite(hi) else belief.mean + 12 * math.sqrt(belief.var)) + 12 * self.tau

    def cdf(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        t = self.tau
        if hasattr(self.b, "pieces"):            # piecewise uniform: int_a^b Phi((x - th) / t) dth = t [G((x-a)/t) - G((x-b)/t)]
            e, m = self.b.pieces
            G = lambda z: z * sp.ndtr(z) + np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            a, b, w = e[:-1][None, :], e[1:][None, :], np.diff(e)[None, :]
            xx = x[:, None]
            return np.clip((m[None, :] * t * (G((xx - a) / t) - G((xx - b) / t)) / w).sum(1), 0, 1)
        mid, mass = grid_masses(self.b, *quad_range(self.b), n=6000)
        return np.clip((mass[None, :] * sp.ndtr((x[:, None] - mid[None, :]) / t)).sum(1), 0, 1)

    def ppf(self, q):
        return ppf_bisect(self.cdf, q, self._lo, self._hi, it=80)


def quad_range(belief):
    lo, hi = belief.support
    sd = math.sqrt(max(belief.var, 1e-300))
    lo = lo if math.isfinite(lo) else belief.mean - 14 * sd
    hi = hi if math.isfinite(hi) else belief.mean + 14 * sd
    return lo, hi


def grid_masses(belief, lo, hi, n=4000):
    """Cell midpoints and masses of `belief` on [lo, hi] (mass outside is put in the first / last cell)."""
    x = np.linspace(lo, hi, n + 1)
    c = np.asarray(belief.cdf(x), float)
    m = np.diff(c)
    m[0] += c[0]
    m[-1] += 1 - c[-1]
    return (x[:-1] + x[1:]) / 2, np.clip(m, 0, None)


def conv_normal(belief, tau):
    if isinstance(belief, bn.NormalD):
        return bn.NormalD(belief.m, math.hypot(belief.s, tau))
    if isinstance(belief, ShiftD) and isinstance(belief.b, bn.NormalD):
        return bn.NormalD(belief.b.m + belief.d, math.hypot(belief.b.s, tau))
    return ConvD(belief, tau)


def moment_match(family, mean, var):
    """A family-shaped belief with this mean / variance (coin: Beta, a + b clipped so a, b >= 0.5; rate: Gamma; normal: Normal)."""
    sd = math.sqrt(max(var, 1e-300))
    if family == "coin":
        mu = min(max(mean, 0.005), 0.995)
        t = mu * (1 - mu) / max(var, 1e-12) - 1
        t = max(t, 0.5 / mu, 0.5 / (1 - mu))
        return bn.BetaD(mu * t, (1 - mu) * t)
    if family == "rate":
        mu = max(mean, 1e-9)
        return bn.GammaD(mu * mu / max(var, 1e-300), mu / max(var, 1e-300))
    return bn.NormalD(mean, sd)


# ============================================================================ family arithmetic (exact, from the stated texts)
def stats_of(family, blocks):
    """Sufficient statistics of data blocks (in shown order). coin {n, h}; rate {tot, T (Fraction), m}; normal {n, S (Fraction)}."""
    if family == "coin":
        n = h = 0
        for b in blocks:
            if b["kind"] == "seq":
                n, h = n + len(b["x"]), h + int(sum(b["x"]))
            else:
                n, h = n + int(b["n"]), h + int(b["h"])
        return {"n": n, "h": h}
    if family == "rate":
        tot, T, m = 0, Fraction(0), 0
        for b in blocks:
            if b["kind"] == "list":
                tot, T, m = tot + int(sum(b["counts"])), T + len(b["counts"]) * Fr(b["L"]), m + len(b["counts"])
            else:
                tot, T, m = tot + int(b["tot"]), T + Fr(b["T"]), m + int(b.get("m", 1))
        return {"tot": tot, "T": T, "m": m, "n": m}
    n, S = 0, Fraction(0)
    for b in blocks:
        if b["kind"] == "list":
            n, S = n + len(b["x"]), S + sum(Fr(v) for v in b["x"])
        else:
            n, S = n + int(b["n"]), S + int(b["n"]) * Fr(b["xbar"])
    return {"n": n, "S": S}


def prior_dist(spec):
    d, p = spec["dist"], spec
    if d == "beta":
        return bn.BetaD(float(Fr(p["a"])), float(Fr(p["b"])))
    if d == "gamma":
        return bn.GammaD(float(Fr(p["shape"])), float(Fr(p["rate"])))
    if d == "normal":
        return bn.NormalD(float(Fr(p["mean"])), float(Fr(p["sd"])))
    if d == "hist":
        return HistD(p["edges"], p["masses"])
    raise ValueError(d)


def posterior(family, spec, blocks, sigma=None, st=None):
    """The exact posterior of the parameter given a stated prior spec and data blocks (Fractions until the end, so members whose
    posteriors must agree get bit-identical parameters)."""
    st = st or stats_of(family, blocks)
    if spec["dist"] == "hist":
        if st["n"] == 0:
            return HistD(spec["edges"], spec["masses"])
        return HistPostD(spec["edges"], spec["masses"], family, st, sigma)
    if family == "coin":
        return bn.BetaD(float(Fr(spec["a"]) + st["h"]), float(Fr(spec["b"]) + st["n"] - st["h"]))
    if family == "rate":
        return bn.GammaD(float(Fr(spec["shape"]) + st["tot"]), float(Fr(spec["rate"]) + st["T"]))
    s0, m0, sg = Fr(spec["sd"]), Fr(spec["mean"]), Fr(sigma)
    prec = 1 / s0 ** 2 + st["n"] / sg ** 2
    mean = (m0 / s0 ** 2 + st["S"] / sg ** 2) / prec
    return bn.NormalD(float(mean), fsqrt(1 / prec))


def fsqrt(fr):
    """sqrt of a Fraction: exact when it is a rational square (so a stated sd round-trips), else the float sqrt."""
    n, d = fr.numerator, fr.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    if rn * rn == n and rd * rd == d:
        return float(Fraction(rn, rd))
    return math.sqrt(float(fr))


def posterior_weighted(family, spec, blocks, sigma, gamma=3.0, min_len=0, max_len=10 ** 9):
    """The ORDER-SENSITIVE predictor's posterior: raw observations weighted exp(gamma * position) (mean weight 1, later = more) when
    the raw sequence length is in [min_len, max_len]; summaries count unweighted."""
    raw = []
    for b in blocks:
        if b["kind"] == "seq":
            raw += [("c", float(x)) for x in b["x"]]
        elif b["kind"] == "list" and family == "rate":
            raw += [("r", float(c), float(Fr(b["L"]))) for c in b["counts"]]
        elif b["kind"] == "list":
            raw += [("n", float(Fr(v))) for v in b["x"]]
    N = len(raw)
    if N < 2 or not (min_len <= N <= max_len):
        return posterior(family, spec, blocks, sigma)
    w = np.exp(gamma * np.arange(N) / (N - 1))
    w = w / w.mean()
    rest = [b for b in blocks if not (b["kind"] == "seq" or b["kind"] == "list")]
    st = stats_of(family, rest)
    st = {k: (float(v) if isinstance(v, Fraction) else v) for k, v in st.items()}
    if family == "coin":
        st["n"] += N
        st["h"] += float(sum(wi * r[1] for wi, r in zip(w, raw)))
    elif family == "rate":
        st["tot"] += float(sum(wi * r[1] for wi, r in zip(w, raw)))
        st["T"] += float(sum(r[2] for r in raw))
        st["n"] = st.get("n", 0) + N
    else:
        st["n"] += N
        st["S"] += float(sum(wi * r[1] for wi, r in zip(w, raw)))
    if spec["dist"] == "hist":
        return HistPostD(spec["edges"], spec["masses"], family, st, sigma)
    if family == "coin":
        return bn.BetaD(float(Fr(spec["a"])) + st["h"], float(Fr(spec["b"])) + st["n"] - st["h"])
    if family == "rate":
        return bn.GammaD(float(Fr(spec["shape"])) + st["tot"], float(Fr(spec["rate"])) + st["T"])
    s0, m0, sg = float(Fr(spec["sd"])), float(Fr(spec["mean"])), float(Fr(sigma))
    prec = 1 / s0 ** 2 + st["n"] / sg ** 2
    return bn.NormalD((m0 / s0 ** 2 + st["S"] / sg ** 2) / prec, math.sqrt(1 / prec))


def mle_of(family, st, sigma=None):
    """(estimate, standard error) in natural units; (None, None) without data (build_numeric_jev.py's conventions)."""
    if family == "coin":
        n, h = st["n"], st["h"]
        return (h / n, math.sqrt(h * (n - h) / n ** 3)) if n else (None, None)
    if family == "rate":
        T = float(st["T"])
        return (st["tot"] / T, math.sqrt(st["tot"]) / T) if T > 0 else (None, None)
    n = st["n"]
    return (float(st["S"]) / n, float(Fr(sigma)) / math.sqrt(n)) if n else (None, None)


def loglik(family, st, theta, sigma=None):
    """log p(data | theta) up to a constant, vectorised over theta (natural units)."""
    th = np.asarray(theta, float)
    if family == "coin":
        t = np.clip(th, 1e-300, 1 - 1e-16)
        return st["h"] * np.log(t) + (st["n"] - st["h"]) * np.log1p(-t)
    if family == "rate":
        t = np.maximum(th, 1e-300)
        return st["tot"] * np.log(t) - t * float(st["T"])
    n, xb, s = st["n"], float(st["S"]) / max(st["n"], 1), float(Fr(sigma))
    return -0.5 * n * (th - xb) ** 2 / s ** 2


def logprior(spec, theta):
    th = np.asarray(theta, float)
    if spec["dist"] == "beta":
        a, b = float(Fr(spec["a"])), float(Fr(spec["b"]))
        t = np.clip(th, 1e-300, 1 - 1e-16)
        return (a - 1) * np.log(t) + (b - 1) * np.log1p(-t) - sp.betaln(a, b)
    if spec["dist"] == "gamma":
        k, r = float(Fr(spec["shape"])), float(Fr(spec["rate"]))
        t = np.maximum(th, 1e-300)
        return k * math.log(r) - sp.gammaln(k) + (k - 1) * np.log(t) - r * t
    if spec["dist"] == "normal":
        m, s = float(Fr(spec["mean"])), float(Fr(spec["sd"]))
        return -0.5 * ((th - m) / s) ** 2 - math.log(s * math.sqrt(2 * math.pi))
    h = HistD(spec["edges"], spec["masses"])
    k = np.clip(np.searchsorted(h.e, th, side="right") - 1, 0, len(h.m) - 1)
    return np.log(np.maximum(h.m[k] / np.diff(h.e)[k], 1e-300))


# ============================================================================ answers: a belief over the parameter -> option probabilities
def masses_on_axis(belief, edges, um):
    """Bin masses of a natural-unit belief on display-unit edges (display = natural * scale + offset); outside mass folds into the
    end bins, as build_numeric_jev.bin_masses does."""
    m, _, _ = bn.bin_masses(belief, np.asarray(edges, float) - float(um.get("offset", 0.0)), float(um["scale"]))
    return np.clip(m, 0, None)


def expect_fn(belief, g, lo, hi, n=4000):
    mid, m = grid_masses(belief, lo, hi, n)
    return (m[:, None] * g(mid)).sum(0) if np.ndim(g(mid[:1])) > 1 else float((m * g(mid)).sum())


def coin_next_k(belief, k):
    j = np.arange(k + 1)
    if isinstance(belief, bn.BetaD):
        a, b = belief.a, belief.b
        lp = (sp.gammaln(k + 1) - sp.gammaln(j + 1) - sp.gammaln(k - j + 1) + sp.betaln(j + a, k - j + b) - sp.betaln(a, b))
        p = np.exp(lp)
    elif hasattr(belief, "pieces"):
        e, m = belief.pieces
        e = np.clip(e, 0, 1)
        w = np.maximum(np.diff(belief.pieces[0]), 1e-300)
        I = sp.betainc(j[:, None] + 1.0, (k - j)[:, None] + 1.0, e[None, :])
        p = ((m / w)[None, :] * np.diff(I, axis=1)).sum(1) / (k + 1)
    else:
        mid, mm = grid_masses(belief, 0.0, 1.0)
        lb = sp.gammaln(k + 1) - sp.gammaln(j + 1) - sp.gammaln(k - j + 1)
        pm = np.clip(mid, 1e-300, 1 - 1e-16)
        p = (mm[None, :] * np.exp(lb[:, None] + j[:, None] * np.log(pm)[None, :] + (k - j)[:, None] * np.log1p(-pm)[None, :])).sum(1)
    p = np.clip(p, 0, None)
    return p / p.sum()


def rate_count(belief, L, cmax=COUNT_MAX):
    """P(N = 0..cmax-1) and P(N >= cmax) for N ~ Poisson(lambda L) mixed over the belief."""
    j = np.arange(cmax)
    if isinstance(belief, bn.GammaD):
        k, r = belief.k, belief.r
        p = np.exp(sp.gammaln(j + k) - sp.gammaln(k) - sp.gammaln(j + 1) + k * math.log(r / (r + L)) + j * math.log(L / (r + L)))
    elif hasattr(belief, "pieces"):
        e, m = belief.pieces
        e = np.maximum(e, 0)
        w = np.maximum(np.diff(belief.pieces[0]), 1e-300)
        P = sp.gammainc(j[:, None] + 1.0, L * e[None, :])
        p = ((m / w)[None, :] * np.diff(P, axis=1)).sum(1) / L
    else:
        lo, hi = quad_range(belief)
        mid, mm = grid_masses(belief, max(lo, 0.0), max(hi, 1e-9))
        mu = np.maximum(mid, 0) * L
        p = (mm[None, :] * np.exp(j[:, None] * np.log(np.maximum(mu, 1e-300))[None, :] - mu[None, :] - sp.gammaln(j + 1)[:, None])).sum(1)
    p = np.clip(p, 0, None)
    return np.append(p, max(0.0, 1.0 - p.sum()))


def rate_none(belief, L):
    """E[exp(-lambda L)] = P(no event in an exposure L)."""
    if isinstance(belief, bn.GammaD):
        return float((belief.r / (belief.r + L)) ** belief.k)
    if hasattr(belief, "pieces"):
        e, m = belief.pieces
        e = np.maximum(e, 0)
        w = np.maximum(np.diff(belief.pieces[0]), 1e-300)
        return float(((m / w) * (np.exp(-L * e[:-1]) - np.exp(-L * e[1:]))).sum() / L)
    lo, hi = quad_range(belief)
    mid, mm = grid_masses(belief, max(lo, 0.0), max(hi, 1e-9))
    return float((mm * np.exp(-np.maximum(mid, 0) * L)).sum())


def coin_mean(belief):
    if isinstance(belief, bn.BetaD) or hasattr(belief, "pieces"):
        return float(min(max(belief.mean, 0.0), 1.0))
    mid, mm = grid_masses(belief, 0.0, 1.0)
    return float((mm * mid).sum())


def answer(meta, belief):
    """The option probabilities (record option order) that a belief over the parameter implies for this record's question."""
    bc = meta["bc"]
    q = bc["q"]
    kind = q["kind"]
    if kind == "bins":
        b = belief
        if q["target"] in ("pred", "predmean"):
            b = conv_normal(belief, float(q["tau"]))
        return masses_on_axis(b, meta["bin_edges"], bc["unit_map"])
    if kind == "event":
        t = q["t"]
        c = np.asarray(belief.cdf(np.array(t, float)), float)
        p = {"gt": 1 - c[0], "le": c[0], "in": c[1] - c[0] if len(c) > 1 else 0.0}[q["event"]]
        p = float(min(max(p, 0.0), 1.0))
        return np.array([p, 1 - p])
    if kind == "choice3":
        c = np.asarray(belief.cdf(np.array(q["t"], float)), float)
        v = np.clip(np.array([c[0], c[1] - c[0], 1 - c[1]]), 0, None)
        return v / v.sum()
    if kind == "next1":
        p = coin_mean(belief)
        return np.array([p, 1 - p])
    if kind == "nextk":
        return coin_next_k(belief, int(q["k"]))
    if kind == "count":
        return rate_count(belief, float(Fr(q["L"])))
    if kind == "atleast1":
        p0 = rate_none(belief, float(Fr(q["L"])))
        return np.array([1 - p0, p0])
    raise ValueError(kind)


def option_keys(rec):
    (qid, q), = rec["questions"].items()
    if q["type"] == "score":
        return list(q["levels"])
    if q["type"] == "noul":
        return ["yes", "no"]
    return list(q["criteria"])


def label_vector(rec):
    (qid, q), = rec["questions"].items()
    v = np.array([rec["labels"][qid][k] for k in option_keys(rec)], float)
    return v / v.sum()


# ============================================================================ expected losses and Bayes actions (natural units)
def partial_expectation(dist, a):
    """E[theta 1{theta < a}] under the exact distribution."""
    if isinstance(dist, bn.BetaD):
        return dist.mean * float(sp.betainc(dist.a + 1, dist.b, min(max(a, 0.0), 1.0)))
    if isinstance(dist, bn.GammaD):
        return dist.mean * float(sp.gammainc(dist.k + 1, dist.r * max(a, 0.0)))
    if isinstance(dist, bn.NormalD):
        z = (a - dist.m) / dist.s
        return dist.m * float(sp.ndtr(z)) - dist.s * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    if hasattr(dist, "pieces"):                  # piecewise uniform (a model's binned answer): exact
        e, m = dist.pieces
        c = m / np.maximum(np.diff(e), 1e-300)
        hi = np.clip(a, e[:-1], e[1:])
        return float((c * (hi ** 2 - e[:-1] ** 2) / 2).sum())
    lo, hi = quad_range(dist)
    mid, m = grid_masses(dist, lo, hi, 20000)
    return float((m * mid * (mid < a)).sum())


def expected_loss(dist, a, loss):
    """E[L(theta, a)] under `dist`: squared = var + (mean - a)^2; pinball_q = q (mean - a) - (PE(a) - a F(a)); absolute = 2 pinball_0.5;
    asym_5x = pinball at q = 5/6 scaled so overestimation costs 1 per unit (x 6)."""
    if loss == "squared":
        return float(dist.var + (dist.mean - a) ** 2)
    q = LOSSES[loss]
    F = float(np.asarray(dist.cdf(np.array([a])))[0])
    pin = q * (dist.mean - a) - (partial_expectation(dist, a) - a * F)
    scale = {"absolute": 2.0, "asym_5x": 6.0}.get(loss, 1.0)
    return float(scale * pin)


def point_loss(theta, a, loss):
    """L(a, theta) for realised theta (scalar or array; the same scaling as expected_loss; min over a is 0 for every loss here)."""
    d = np.asarray(theta, float) - float(a)
    if loss == "squared":
        out = d * d
    else:
        out = {"absolute": 2.0, "asym_5x": 6.0}.get(loss, 1.0) * d * (LOSSES[loss] - (d < 0))
    return float(out) if np.ndim(out) == 0 else out


def bayes_action(dist, loss):
    if LOSSES[loss] is None:
        return float(dist.mean)
    return float(np.asarray(dist.ppf(np.array([LOSSES[loss]])))[0])


def to_disp(x, um):
    return float(x) * float(um["scale"]) + float(um.get("offset", 0.0))


def to_nat(x, um):
    return (np.asarray(x, float) - float(um.get("offset", 0.0))) / float(um["scale"])


# ============================================================================ wording (family renderers; build_numeric_jev.py's templates)
COIN_PRED = {
    "coin": ("Will the next flip land heads?", "How many of the next {k} flips will land heads?"),
    "thumbtack": ("Will the next toss land point up?", "How many of the next {k} tosses will land point up?"),
    "free_throw": ("Will she make her next free throw?", "How many of her next {k} free throws will she make?"),
    "germination": ("Will the next seed planted germinate?", "How many of the next {k} seeds planted will germinate?"),
    "defects": ("Will the next part inspected be defective?", "How many of the next {k} parts inspected will be defective?"),
    "email": ("Will the next recipient open it?", "How many of the next {k} recipients will open it?"),
    "survey": ("Will the next respondent say yes?", "How many of the next {k} respondents will say yes?"),
    "clinical": ("Will the next patient respond?", "How many of the next {k} patients will respond?"),
}
# rate context -> (alt unit, factor alt / native as a fraction string, decimal?)
RATE_ALT = {"calls": ("calls per day", "24", False), "typos": ("typos per 100 pages", "100", True),
            "meteors": ("meteors per minute", "1/60", False), "errors": ("errors per week", "7", False),
            "potholes": ("potholes per 100 km", "100", True), "decays": ("counts per minute", "60", False),
            "arrivals": ("customers per hour", "60", False), "emails": ("emails per week", "7", False)}
# normal context -> (alt unit, factor, offset, decimal?)
NORM_ALT = {"mass": ("kg", "1/1000", "0", True), "length": ("cm", "1/10", "0", True), "temperature": ("degrees F", "9/5", "32", False),
            "reaction": ("s", "1/1000", "0", True), "voltage": ("mV", "1000", "0", True), "nitrate": ("mg/mL", "1/1000", "0", True),
            "commute": ("hours", "1/60", "0", False)}
LOSS_TEXT = {
    "squared": "its cost will be the squared difference between the estimate and the true value.",
    "absolute": "its cost will be the absolute difference between the estimate and the true value.",
    "pinball_0.1": ("each unit by which the estimate falls below the true value costs 1, and each unit by which it lies above "
                    "the true value costs 9."),
    "pinball_0.9": ("each unit by which the estimate falls below the true value costs 9, and each unit by which it lies above "
                    "the true value costs 1."),
    "asym_5x": "underestimating the true value costs 5 times as much per unit as overestimating it.",
}


class Fam:
    """One group's family context: the context entry, the wording choices shared by every member, the prior, the truth."""

    def __init__(self, family, rng, py, off):
        self.family, self.rng, self.py, self.off = family, rng, py, off
        self.novel = []

    # ---- generic
    def state(self, prior_line, data_lines, extra=()):
        return "\n".join([self.intro(), prior_line, *data_lines, *extra])

    def loss_line(self, loss):
        return f"After you answer, a single point estimate of {self.sym_short} will be chosen, and " + LOSS_TEXT[loss]

    def shown_block(self, block, fmt=None):
        return block


class Coin(Fam):
    def __init__(self, rng, py, off, prior=None):
        super().__init__("coin", rng, py, off)
        self.ctx = py.choice(bn.COIN_CTX)
        if prior is None:
            a, b = bn.beta_prior_draw(rng, py)
            prior = {"dist": "beta", "a": bn.fparam(a), "b": bn.fparam(b)}   # the STATED values are the prior (see the doc)
        self.prior = prior
        self.st = {"i_prior": py.randrange(4), "u_prior": py.random(), "i_data": py.randrange(3), "i_seq": py.randrange(2),
                   "i_q": py.randrange(3), "pct": bool(rng.uniform() < 0.25)}
        self.truth = float(rng.uniform()) if off else float(prior_dist(prior).sample(rng, 1)[0])
        self.sym_short = "p"

    def intro(self):
        return self.ctx["intro"] + f" Let p be {self.ctx['p']}."

    def prior_line(self, spec):
        if spec["dist"] == "hist":
            return self.hist_line(spec)
        a, b = Fr(spec["a"]), Fr(spec["b"])
        txt = bn.beta_prior_text(Pick(self.st["i_prior"], self.st["u_prior"]), float(a), float(b))
        assert bn.fparam(float(a)) == spec["a"] and bn.fparam(float(b)) == spec["b"], spec   # stated == used
        return txt

    def draw_seq(self, n, p=None):
        return [int(x) for x in (self.rng.uniform(size=n) < (self.truth if p is None else p))]

    def draw_counts(self, n):
        return {"kind": "counts", "n": int(n), "h": int(self.rng.binomial(n, self.truth)) if n else 0}

    def data_line(self, block):
        ctx = self.ctx
        if block["kind"] == "seq":
            seq = block["x"]
            n = len(seq)
            s1, s0 = ctx["sym"]
            legend = f"{s1} = {ctx['succ']}, {s0} = {ctx['fail']}"
            seq_txt = " ".join(s1 if x else s0 for x in seq)
            return [f"The {n} {pl(n, ctx['trial'])} so far, in order ({legend}): {seq_txt}",
                    f"Outcomes of {n} {pl(n, ctx['trial'])}, in order ({legend}): {seq_txt}"][self.st["i_seq"] % 2]
        n, h = block["n"], block["h"]
        if n == 0:
            return f"No {ctx['trial'][1]} have been observed yet."
        f = n - h
        return [f"Data: {n} {pl(n, ctx['trial'])}: {bn.coin_outcome(ctx, h, f)}.",
                f"Observed so far: {bn.coin_outcome(ctx, h, f)} ({n} {pl(n, ctx['trial'])} in total).",
                f"Results of {n} {pl(n, ctx['trial'])}: {bn.coin_outcome(ctx, h, f)}."][self.st["i_data"] % 3]

    def axis(self, dist=None, seed_parts=None, pct=None):
        ax, ed = bn.coin_axis(self.st["pct"] if pct is None else pct)
        return ax, ed, {"scale": ax["scale"], "offset": 0.0}

    def question(self, axis, edges, kind="post", pct=None):
        return bn.coin_question(Pick(self.st["i_q"]), axis, edges, self.st["pct"] if pct is None else pct)

    def outcome(self, n, h):
        return bn.coin_outcome(self.ctx, h, n - h)

    def two_batch_line(self, b1, b2):
        T = self.ctx["trial"]
        return (f"Data from two batches of {T[1]}. Batch 1: {b1['n']} {pl(b1['n'], T)}: {self.outcome(b1['n'], b1['h'])}. "
                f"Batch 2: {b2['n']} {pl(b2['n'], T)}: {self.outcome(b2['n'], b2['h'])}.")

    def stated_post_lines(self, b1, post1, b2):
        T = self.ctx["trial"]
        return [f"After a first batch of {b1['n']} {pl(b1['n'], T)} ({self.outcome(b1['n'], b1['h'])}), the posterior for p is "
                f"Beta({post1['a']}, {post1['b']}).",
                f"A second batch then gave {b2['n']} {pl(b2['n'], T)}: {self.outcome(b2['n'], b2['h'])}."]

    def event_sym(self, pct):
        return ("p (in percent)" if pct else "p"), ""

    def hist_line(self, spec):
        return hist_sentence(spec, "p (in percent)" if spec.get("pct") else "p", "")


class Rate(Fam):
    def __init__(self, rng, py, off, prior=None):
        super().__init__("rate", rng, py, off)
        self.ctx = py.choice(bn.RATE_CTX)
        if prior is None:
            shape = float(py.choice([0.5, 1, 1.5, 2, 3, 4, 5, 8, 10, 20]))
            mu0 = float(bn.fsig(logu(rng, 0.05, 50), 2))
            rate = float(bn.fsig(shape / mu0, 3))
            prior = {"dist": "gamma", "shape": bn.fparam(shape), "rate": bn.fsig(rate, 3)}
        self.prior = prior
        self.iv_name, L = py.choice(self.ctx["iv"])
        self.L = dec_str(Fr(L))
        self.st = {"i_prior": py.randrange(3), "i_sum": py.randrange(2), "i_q": py.randrange(3), "u_list": py.random()}
        pd = prior_dist(prior)
        self.truth = pd.mean * logu(rng, 1 / 8, 8) if off else float(pd.sample(rng, 1)[0])
        self.unit = self.ctx["unit"]
        self.sym_short = "λ"

    def intro(self):
        return f"{self.ctx['intro']} λ ({self.unit})."

    def prior_line(self, spec):
        if spec["dist"] == "hist":
            return self.hist_line(spec)
        S, R = spec["shape"], spec["rate"]
        mean = bn.fsig(float(Fr(S)) / float(Fr(R)), 3)
        u = self.unit
        return [f"Prior: λ ~ Gamma(shape {S}, rate {R}), so the prior mean is {mean} {u}.",
                f"Prior: a Gamma distribution for λ with shape α = {S} and rate β = {R} (mean α/β = {mean} {u}).",
                f"Before any data, λ ~ Gamma(α = {S}, β = {R}) in {u}."][self.st["i_prior"] % 3]

    def draw_list(self, m):
        return {"kind": "list", "counts": [int(c) for c in self.rng.poisson(self.truth * float(Fr(self.L)), m)], "L": self.L}

    def as_total(self, block):
        c = block["counts"]
        return {"kind": "total", "tot": int(sum(c)), "T": dec_str(len(c) * Fr(block["L"])), "m": len(c)}

    def shown_block(self, block, fmt=None):
        """The block as data_line(block, fmt) shows it: a list stays a list only where the corpus's rule shows the counts."""
        if block["kind"] == "list":
            m = len(block["counts"])
            if fmt is None:
                fmt = "list" if (1 < m <= 12 and self.st["u_list"] < 0.6) else "total"
            if fmt != "list":
                return self.as_total(block)
        return block

    def _Ttxt(self, T):
        t = bn.fsig(float(Fr(T)), 4)
        assert Fr(t) == Fr(T), (t, T)
        return t, (self.ctx["base"][0] if Fr(T) == 1 else self.ctx["base"][1])

    def data_line(self, block, fmt=None):
        """fmt None = build_numeric_jev's rule (list if 1 < m <= 12 and u_list < 0.6; one interval -> 'Observed'; else a summary)."""
        if block["kind"] == "list":
            m = len(block["counts"])
            if fmt is None:
                fmt = "list" if (1 < m <= 12 and self.st["u_list"] < 0.6) else "total"
            if fmt == "list":
                return f"Counts in {m} consecutive {self.iv_name}: {', '.join(map(str, block['counts']))}."
            block = self.as_total(block)
        tot, m = block["tot"], block.get("m", 1)
        Ttxt, Tu = self._Ttxt(block["T"])
        ev = pl(tot, self.ctx["ev"])
        if m == 1:
            return f"Observed: {tot} {ev} in {Ttxt} {Tu}."
        return [f"In total {tot} {ev} were recorded over {m} consecutive {self.iv_name} ({Ttxt} {Tu} in all).",
                f"Observed: {tot} {ev} in {Ttxt} {Tu}."][self.st["i_sum"] % 2]

    def nodata_line(self):
        return f"No {self.ctx['ev'][1]} have been counted yet."

    def axis(self, dist, seed_parts, scale="1", unit=None):
        rng = np.random.default_rng(seed(*seed_parts))
        ax, ed = bn.declared_axis(rng, dist, float(Fr(scale)), unit or self.unit, support_lo=0.0)
        return ax, ed, {"scale": float(Fr(scale)), "offset": 0.0}

    def question(self, axis, edges, kind="post", unit=None):
        u = unit or self.unit
        q = [f"Which bin contains λ, in {u}?", f"In which interval does the rate λ ({u}) lie?",
             f"Given the data, which bin holds the true rate λ ({u})?"][self.st["i_q"] % 3]
        return q + " " + bn.axis_sentence(axis, edges, u)

    def batch_txt(self, b):
        b = self.as_total(b) if b["kind"] == "list" else b
        Ttxt, Tu = self._Ttxt(b["T"])
        return f"{b['tot']} {pl(b['tot'], self.ctx['ev'])} in {Ttxt} {Tu}"

    def two_batch_line(self, b1, b2):
        return f"Data from two batches. Batch 1: {self.batch_txt(b1)}. Batch 2: {self.batch_txt(b2)}."

    def stated_post_lines(self, b1, post1, b2):
        return [f"After a first batch ({self.batch_txt(b1)}), the posterior for λ is Gamma(shape {post1['shape']}, rate "
                f"{post1['rate']}).", f"A second batch then gave {self.batch_txt(b2)}."]

    def event_sym(self, pct=False):
        return "λ", f" {self.unit}"

    def hist_line(self, spec):
        return hist_sentence(spec, "λ", f" {self.unit}")


class Normal(Fam):
    def __init__(self, rng, py, off, prior=None, sigma=None, alt_ok=True):
        super().__init__("normal_mean", rng, py, off)
        self.ctx = ctx = py.choice(bn.NORM_CTX)
        S = logu(rng, *ctx["S"])
        self.S = S
        if prior is None:
            s0 = float(bn.fsig(S * logu(rng, 0.01, 0.3), 2))
            r0 = bn.pow10_floor(s0 / 3)
            m0 = round(round(S * rng.uniform(0.6, 1.6) / r0) * r0, bn.dec_for_res(r0))
            prior = {"dist": "normal", "mean": bn.fnum(m0, bn.dec_for_res(r0)), "sd": bn.fsig(s0, 2)}
        self.prior = prior
        if sigma is None:
            sigma = bn.fsig(float(bn.fsig(S * logu(rng, 0.002, 0.2), 2)), 2)
        self.sigma = sigma
        self.st = {"i_noise": py.randrange(2), "i_prior": py.randrange(2), "i_q": py.randrange(3), "i_pred": py.randrange(2),
                   "u_list": py.random()}
        self.unit = ctx["unit"]
        alt = ctx["alt"] if (alt_ok and ctx["alt"] and rng.uniform() < 0.2) else None
        self.dunit, self.scale = (alt[0], alt[1]) if alt else (self.unit, 1.0)
        pd = prior_dist(prior)
        self.truth = pd.mean + pd.s * float(rng.uniform(-5, 5)) if off else float(pd.sample(rng, 1)[0])
        self.r = bn.pow10_floor(float(Fr(sigma)) / 3)
        self.sym_short = "μ"

    def intro(self):
        c, u, sg = self.ctx, self.unit, self.sigma
        noise = [f"Each {c['reading'][0]} equals {{mu}} plus independent normal error with standard deviation {sg} {u}.",
                 f"{c['reading'][1].capitalize()} scatter around {{mu}} with independent normal noise, sd {sg} {u}."][self.st["i_noise"] % 2]
        return c["intro"].format(mu="μ") + " " + noise.format(mu="μ")

    def prior_line(self, spec):
        if spec["dist"] == "hist":
            return self.hist_line(spec)
        m, s, u = spec["mean"], spec["sd"], self.unit
        return [f"Prior: μ ~ Normal(mean {m}, sd {s}) in {u}.",
                f"Before any data, your belief about μ is normal with mean {m} {u} and standard deviation {s} {u}."][self.st["i_prior"] % 2]

    def draw_list(self, n, mu=None):
        r, dr = self.r, bn.dec_for_res(self.r)
        xs = self.rng.normal(self.truth if mu is None else mu, float(Fr(self.sigma)), n)
        return {"kind": "list", "x": [bn.fnum(round(round(x / r) * r, dr), dr) for x in xs]}

    def nudge_exact_mean(self, block):
        """Move as few readings as possible by one resolution step so the mean is exact at resolution r / 10 (the summary format)."""
        r, dr = Fr(dec_str(Fr(str(self.r)))), bn.dec_for_res(self.r)
        xi = [int(Fr(v) / r) for v in block["x"]]
        n = len(xi)
        d = n // math.gcd(n, 10)
        delta = (-sum(xi)) % d
        if delta > d / 2:
            delta -= d
        idx = list(self.rng.permutation(n))[: abs(delta)]
        for i in idx:
            xi[i] += 1 if delta > 0 else -1
        return {"kind": "list", "x": [bn.fnum(float(v * r), dr) for v in xi]}

    def as_summary(self, block):
        rb = self.r / 10
        n = len(block["x"])
        xbar = sum(Fr(v) for v in block["x"]) / n
        t = bn.fnum(float(xbar), bn.dec_for_res(rb))
        assert Fr(t) == xbar, (t, xbar)
        return {"kind": "summary", "n": n, "xbar": t}

    def data_line(self, block, fmt=None):
        c, u = self.ctx, self.unit
        if block["kind"] == "summary":
            return f"{block['n']} {pl(block['n'], c['reading'])} were taken; their mean is {block['xbar']} {u}."
        vals = block["x"]
        n = len(vals)
        if n == 0:
            return f"No {c['reading'][1]} have been taken yet."
        if fmt is None:
            fmt = "list" if (n <= 12 and self.st["u_list"] < 0.7) else "summary"
        if fmt == "summary":
            rb = self.r / 10
            xbar_t = bn.fnum(round(float(sum(Fr(v) for v in vals) / n) / rb) * rb, bn.dec_for_res(rb))
            return self.data_line({"kind": "summary", "n": n, "xbar": xbar_t})
        return f"The {n} {pl(n, c['reading'])} ({u}): {', '.join(vals)}." if n > 1 else f"One {c['reading'][0]} so far: {vals[0]} {u}."

    def shown_block(self, block, fmt=None):
        """The block as the model sees it (the corpus's summary format rounds the mean; the posterior uses the shown mean)."""
        if block["kind"] == "list" and len(block["x"]):
            n = len(block["x"])
            if fmt is None:
                fmt = "list" if (n <= 12 and self.st["u_list"] < 0.7) else "summary"
            if fmt == "summary":
                rb = self.r / 10
                return {"kind": "summary", "n": n,
                        "xbar": bn.fnum(round(float(sum(Fr(v) for v in block["x"]) / n) / rb) * rb, bn.dec_for_res(rb))}
        return block

    def axis(self, dist, seed_parts, scale=None, unit=None, um=None):
        rng = np.random.default_rng(seed(*seed_parts))
        sc = self.scale if scale is None else float(Fr(scale))
        ax, ed = bn.declared_axis(rng, dist, sc, unit or self.dunit)
        return ax, ed, (um or {"scale": sc, "offset": 0.0})

    def question(self, axis, edges, kind="post", unit=None):
        c, du = self.ctx, unit or self.dunit
        if kind == "pred":
            q = [f"{c['next']} Which bin will its value fall in, in {du}?",
                 f"{c['next']} In which interval will the new value lie ({du})?"][self.st["i_pred"] % 2]
        else:
            q = [f"Which bin contains μ, {c['what']}, in {du}?", f"In which interval does μ ({c['what']}) lie, in {du}?",
                 f"Given the data, which bin holds μ, in {du}?"][self.st["i_q"] % 3]
        return q + " " + bn.axis_sentence(axis, edges, du)

    def list_txt(self, b):
        return ", ".join(b["x"])

    def two_batch_line(self, b1, b2):
        r = self.ctx["reading"][1]
        return f"The {r} came in two batches. Batch 1 ({self.unit}): {self.list_txt(b1)}. Batch 2 ({self.unit}): {self.list_txt(b2)}."

    def stated_post_lines(self, b1, post1, b2):
        c, u = self.ctx, self.unit
        n1, n2 = len(b1["x"]), len(b2["x"])
        return [f"After a first batch of {n1} {pl(n1, c['reading'])} ({u}: {self.list_txt(b1)}), the posterior for μ is "
                f"Normal(mean {post1['mean']}, sd {post1['sd']}) in {u}.",
                f"A second batch then gave ({u}): {self.list_txt(b2)}." if n2 > 1 else
                f"A second batch then gave one {c['reading'][0]}: {b2['x'][0]} {u}."]

    def event_sym(self, pct=False):
        return f"μ ({self.ctx['what']})", f" {self.dunit}"

    def hist_line(self, spec):
        return hist_sentence(spec, "μ", f" {spec.get('dunit', self.unit)}")


FAM_CLS = {"coin": Coin, "rate": Rate, "normal_mean": Normal}


def hist_sentence(spec, sym, u):
    """NEW PHRASING (chain members): a histogram prior over the batch-1 axis, uniform inside bins, probabilities to HIST_DP decimals."""
    lab = spec["labels"]
    items = "; ".join(f"{lab[i]}: {p}" for i, p in enumerate(spec["masses_txt"]) if Fr(p) > 0)
    return (f"After a first batch of data, your belief about {sym} is a histogram over {len(lab)} equal-width bins of "
            f"{spec['width_txt']}{u} from {spec['lo_txt']} to {spec['hi_txt']}{u}: within a bin every value is equally likely, and "
            f"values outside that range are impossible. The bin probabilities (bins not listed have probability 0) are: {items}.")


# ============================================================================ axes shared by several members
class UnionProxy:
    """declared_axis on several distributions at once: ppf([.005, .995]) -> the union of their central 99% intervals; cdf([lo, hi])
    -> the WORST member's folded tails, so the fold check holds for every member."""

    def __init__(self, dists):
        self.d = dists

    def ppf(self, q):
        q = np.asarray(q, float)
        return np.array([min(float(np.asarray(d.ppf(q[:1]))[0]) for d in self.d),
                         max(float(np.asarray(d.ppf(q[1:]))[0]) for d in self.d)])

    def cdf(self, x):
        x = np.asarray(x, float)
        return np.array([max(float(np.asarray(d.cdf(x[:1]))[0]) for d in self.d),
                         min(float(np.asarray(d.cdf(x[1:]))[0]) for d in self.d)])


def scaled_axis(axis, edges, factor, unit):
    """The SAME bins on a decimal-rescaled unit (factor a power of ten): edges x factor exactly, nice width kept."""
    f = Fr(factor)
    w = Fr(dec_str(Fr(str(axis["width"])))) * f
    dec = bn.decimals_of(float(w))
    ed = [float(bn.fnum(float(Fr(str(e)) * f), dec)) for e in edges]
    for a, b in zip(edges, ed):
        assert Fr(str(b)) == Fr(str(a)) * f, (a, b, factor)
    ax = dict(axis, lo=ed[0], hi=ed[-1], width=float(w), decimals=dec, scale=float(axis["scale"] * f), unit=unit,
              aligned_from={"factor": str(factor)})
    return ax, ed


# ============================================================================ v2 axes (from the visible data only)
def flat_lik(fam, blocks):
    """The normalised likelihood of the visible data blocks as a distribution on the parameter (natural units; no prior):
    coin Beta(h + 1, n - h + 1), rate Gamma(tot + 1, T), normal N(xbar, sigma / sqrt(n)); None without data."""
    f = fam.family
    st = stats_of(f, blocks)
    if f == "coin":
        return bn.BetaD(st["h"] + 1.0, st["n"] - st["h"] + 1.0) if st["n"] else None
    if f == "rate":
        return bn.GammaD(st["tot"] + 1.0, float(st["T"])) if st["T"] > 0 else None
    return bn.NormalD(float(st["S"]) / st["n"], float(Fr(fam.sigma)) / math.sqrt(st["n"])) if st["n"] else None


def axis_v2(fam, blocks, seed_parts, kind="param", k=None, um=None, unit=None):
    """(v2) The axis of a rate / normal question from the VISIBLE data `blocks` only (build_numeric_jev.data_interval: the data's
    likelihood interval, or the plug-in predictive spread for kind pred / predmean), zoom / offset from the stream
    seed(*seed_parts, "axis_v2"); no data -> the context's fixed range (build_numeric_jev.context_axis). coin: the natural axis.
    um {scale, offset}: display = natural x scale + offset (default: the family's display scale). -> (axis, edges, um)."""
    f = fam.family
    if f == "coin":
        return fam.axis()
    if um is None:
        um = {"scale": float(fam.scale) if f == "normal_mean" else 1.0, "offset": 0.0}
    sc, off = float(um["scale"]), float(um.get("offset", 0.0))
    u = unit or (fam.dunit if f == "normal_mean" else fam.unit)
    st = stats_of(f, blocks)
    if f == "rate":
        iv = bn.data_interval("rate", tot=st["tot"], T=float(st["T"]))
    else:
        kk = {"param": "normal_mean", "pred": "normal_pred", "predmean": "normal_predmean"}[kind]
        iv = bn.data_interval(kk, n=st["n"], xbar=float(st["S"]) / st["n"] if st["n"] else None, sigma=float(Fr(fam.sigma)), k=k)
    if iv is None:
        ax, ed = bn.context_axis(f, fam.ctx["id"], sc, u)
    elif off:                                   # affine (C -> F): the interval in display units, axis scale 1 (as v1)
        ax, ed = bn.evidence_axis(np.random.default_rng(seed(*seed_parts, "axis_v2")), iv[0] * sc + off, iv[1] * sc + off, 1.0, u)
        ax = dict(ax, unit_map=dict(um))
    else:
        ax, ed = bn.evidence_axis(np.random.default_rng(seed(*seed_parts, "axis_v2")), iv[0] * sc, iv[1] * sc, sc, u,
                                  support_lo=0.0 if f == "rate" else None)
    return ax, ed, {"scale": sc, "offset": off}


# ============================================================================ records
def summarise_blocks(family, blocks, sigma=None):
    st = stats_of(family, blocks)
    out = {k: (dec_str(v) if isinstance(v, Fraction) else v) for k, v in st.items()}
    if family == "normal_mean":
        out["sigma"] = sigma
        out["xbar"] = float(st["S"]) / st["n"] if st["n"] else None
    return out


class Builder:
    axes = "v1"                                      # build() sets the instance's (subclasses keep the old constructor)

    def __init__(self, ns=NS, forbidden=None):
        self.ns = ns
        self.records = defaultdict(list)
        self.seen = set()
        self.forbidden = forbidden or set()          # question texts of numeric_v1 train: never reused verbatim

    # ---------------------------------------------------------------- one member
    def member(self, G, name, state, instructions, qtype, q, belief_exact, bc_extra, axis=None, edges=None, um=None, levels=None,
               criteria=None, truth=None, quantity="posterior", fam_label=None, novel=None, problem=None, prior_spec=None,
               blocks=None, shown=None, ref=None):
        fam = G["F"]
        um = um or {"scale": 1.0, "offset": 0.0}
        problem = problem or {"prior": prior_spec or fam.prior, "blocks": blocks or [], **({"sigma": fam.sigma} if fam.family == "normal_mean" else {})}
        if shown is not None:
            problem["shown_blocks"] = shown
        bc = {"group": G["gid"], "type": G["type"], "family": fam.family, "member": name, "context": fam.ctx["id"],
              "q": q, "unit_map": um, "problem": problem, "novel": novel or [], "anchor": False, "sbc": False,
              "ref": ref or G.get("ref"), **bc_extra}
        if self.axes == "v2":                         # (v2) only: v1 records are byte-identical
            bc["axes"] = "v2"
        meta = {"family": fam_label or fam.family, "split": "test", "context": fam.ctx["id"], "quantity": quantity,
                "generator": VERSION_V2 if self.axes == "v2" else VERSION, "seed_namespace": self.ns, "bc": bc}
        if qtype == "score" and q["kind"] == "bins":
            meta["axis"], meta["bin_edges"] = axis, edges
        target = answer(meta, belief_exact)
        # the member's own stated problem must reproduce the target it is labelled with (catches any builder slip)
        chk = posterior(fam.family, problem["prior"], problem["blocks"], problem.get("sigma"))
        if not isinstance(chk, (HistD, HistPostD)):
            assert chk.params() == belief_exact.params(), (G["gid"], name, chk.params(), belief_exact.params())
        st = stats_of(fam.family, problem["blocks"])
        pd = prior_dist(problem["prior"])
        meta["prior"] = {**pd.params(), "stated": {k: v for k, v in problem["prior"].items()
                                                   if k not in ("edges", "masses", "labels", "masses_txt")}}
        meta["data"] = summarise_blocks(fam.family, problem.get("shown_blocks", problem["blocks"]), problem.get("sigma"))
        ex = {**belief_exact.params()} if hasattr(belief_exact, "params") else {}
        if q["kind"] == "bins":
            tb = conv_normal(belief_exact, float(q["tau"])) if q["target"] in ("pred", "predmean") else belief_exact
            if hasattr(tb, "params") and tb is not belief_exact:
                ex = {**tb.params(), "posterior": belief_exact.params()}
            bm, bs = bn.binned_moments(target, edges)
            ex.update(mean=to_disp(tb.mean, um), sd=math.sqrt(tb.var) * um["scale"], mean_binned=bm, sd_binned=bs)
            c = np.asarray(tb.cdf(to_nat(np.array([edges[0], edges[-1]]), um)), float)
            ex.update(folded_low=float(c[0]), folded_high=float(1 - c[1]))
        else:
            ex.update(mean=float(belief_exact.mean), sd=math.sqrt(belief_exact.var))
        meta["exact"] = ex
        meta["truth"] = truth
        # baselines: the stated prior's answer, and the plug-in MLE (Wald normal for the parameter, plug-in predictive otherwise)
        shown_blocks = problem.get("shown_blocks", problem["blocks"])
        mst = stats_of(fam.family, shown_blocks)
        mle, se = mle_of(fam.family, mst, problem.get("sigma"))
        bl = {"prior": rnd(answer(meta, pd))}
        nopt = len(target)
        if mle is None:
            bl["mle_wald"] = rnd(np.full(nopt, 1.0 / nopt))
            if q["kind"] == "bins":
                bl["mle_smooth"] = rnd(np.full(nopt, 1.0 / nopt))
        else:
            if q["kind"] == "bins" and q["target"] == "param":
                if se and se > 0:
                    w = masses_on_axis(bn.NormalD(mle, se), edges, um)
                    bl["mle_wald"] = rnd(0.99 * w + 0.01 / nopt)
                else:
                    bl["mle_wald"] = rnd(bn._smooth_point(axis, edges, to_disp(mle, um)))
                bl["mle_smooth"] = rnd(bn._smooth_point(axis, edges, to_disp(mle, um)))
            else:
                pt = bn.NormalD(mle, max(abs(mle), 1.0) * 1e-9)
                if q["kind"] == "event" or q["kind"] == "choice3":
                    pt = bn.NormalD(mle, se) if se and se > 0 else pt
                w = answer(meta, pt)
                bl["mle_wald"] = rnd(0.99 * w + 0.01 / nopt)
                if q["kind"] == "bins":
                    bl["mle_smooth"] = rnd(bn._smooth_point(axis, edges, to_disp(mle, um)))
        meta["baselines"] = bl
        meta["mle"] = {"value": None if mle is None else to_disp(mle, um) if q["kind"] == "bins" and q["target"] == "param" else mle,
                       "se": None if se is None else se}
        meta["fingerprint"] = f"{fam.family}|{json.dumps(problem['prior'], sort_keys=True)}|{json.dumps(mst, default=str, sort_keys=True)}"
        bc["n_data"] = int(st.get("n", 0))
        bc["seq_len"] = max([len(b["x"]) for b in problem["blocks"] if b["kind"] == "seq"] or [0])
        bc["truth_nat"] = G.get("truth_nat")
        if q["kind"] == "bins" and q["target"] == "param":
            bc["actions"] = {l: to_disp(bayes_action(belief_exact, l), um) for l in LOSSES}
        if qtype == "score":
            labels = levels if levels is not None else bn.bin_labels(axis, edges)
            question = {"type": "score", "instructions": instructions, "levels": labels}
            keys = labels
        elif qtype == "noul":
            question = {"type": "noul", "instructions": instructions}
            keys = ["yes", "no"]
        else:
            question = {"type": "choice", "instructions": instructions, "criteria": criteria}
            keys = list(criteria)
        assert len(keys) == len(target), (len(keys), len(target))
        assert np.all(target >= -1e-12) and abs(target.sum() - 1) < 1e-8, (name, target.sum())
        key = state + "\n" + instructions
        if key in self.seen or key in self.forbidden or any(key == m["state"] + "\n" + next(iter(m["questions"].values()))["instructions"]
                                   for m in G["members"]):
            raise DuplicateItem(key)
        rid = bn.sha16(f"{self.ns}|{G['gid']}|{name}|{state}|{instructions}")
        qid = f"bc_{G['type']}/{rid}"
        rec = {"id": f"bc-{G['type']}-{rid}", "source": f"bayes_consistency/{G['type']}", "license": LICENSE,
               "base_id": f"bc/{G['gid']}", "state": state, "questions": {qid: question},
               "labels": {qid: dict(zip(keys, rnd(target)))}, "label_kind": {qid: "soft"}, "meta": meta}
        G["members"].append(rec)
        return rec


class DuplicateItem(Exception):
    pass


def truth_rec(G, value_disp=None, source=None):
    return {"value": value_disp, "source": source or G["truth_source"]}


# ============================================================================ group builders
def new_group(gtype, family, gi, attempt, ns):
    s = seed(ns, gtype, family, gi, attempt)
    rng, py = np.random.default_rng(s), random.Random(s)
    off = bool(rng.uniform() < P_OFF)
    gid = f"{gtype}/{family}/{gi:04d}"
    return {"gid": gid, "type": gtype, "rng": rng, "py": py, "off": off, "members": [],
            "truth_source": "broad" if off else "prior", "ns": ns}


def _post_q(target="param", tau=None, k=None):
    q = {"kind": "bins", "target": target}
    if tau is not None:
        q["tau"] = tau
    if k is not None:
        q["k"] = k
    return q


def ref_range(edges, um):
    lo, hi = to_nat(np.array([edges[0], edges[-1]]), um)
    return {"lo": float(lo), "hi": float(hi)}


def g_prior_only(B, G):
    fam, rng = G["F"], G["rng"]
    f = fam.family
    pd = prior_dist(fam.prior)
    if f == "coin":
        n = int(round(logu(rng, 1, 200)))
        wb = fam.draw_counts(n)
        nod = {"kind": "counts", "n": 0, "h": 0}
        ax, ed, um = fam.axis()
        G["ref"] = ref_range(ed, um)
        nodata_line, novel = fam.data_line(nod), []
        ax0, ed0, um0 = ax, ed, um
    elif f == "rate":
        wb = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 30)))))
        nod = None
        ax0, ed0, um0 = fam.axis(pd, (G["ns"], G["gid"], "ax_nodata"))
        ax, ed, um = fam.axis(posterior(f, fam.prior, [wb]), (G["ns"], G["gid"], "ax_withdata"))
        G["ref"] = ref_range(ed0, um0)
        nodata_line, novel = fam.nodata_line(), ["state:3"]
    else:
        wb = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 40)))))
        nod = {"kind": "list", "x": []}
        sc = fam.scale
        ax0, ed0, um0 = fam.axis(pd, (G["ns"], G["gid"], "ax_nodata"))
        ax, ed, um = fam.axis(posterior(f, fam.prior, [wb], fam.sigma), (G["ns"], G["gid"], "ax_withdata"))
        G["ref"] = ref_range(ed0, um0)
        nodata_line, novel = fam.data_line(nod), []
    if B.axes == "v2" and f != "coin":            # (v2) no evidence is shared (nodata has none): the context's fixed range, shared
        ax, ed, um = axis_v2(fam, [], (G["ns"], G["gid"], "ax_group"))
        ax0, ed0, um0 = ax, ed, um
        G["ref"] = ref_range(ed, um)
    lab = "coin" if f == "coin" else f
    B.member(G, "nodata", fam.state(fam.prior_line(fam.prior), [nodata_line]), fam.question(ax0, ed0), "score", _post_q(), pd,
             {}, axis=ax0, edges=ed0, um=um0, truth=truth_rec(G, None), fam_label=lab, novel=novel, blocks=[])
    post = posterior(f, fam.prior, [wb], getattr(fam, "sigma", None))
    rec = B.member(G, "withdata", fam.state(fam.prior_line(fam.prior), [fam.data_line(wb)]), fam.question(ax, ed), "score",
                   _post_q(), post, {}, axis=ax, edges=ed, um=um, truth=truth_rec(G, to_disp(G["truth_nat"], um)),
                   fam_label=lab, blocks=[wb])
    rec["meta"]["bc"]["anchor"] = True


def g_prior_swap(B, G):
    fam, rng, py = G["F"], G["rng"], G["py"]
    f = fam.family
    if f == "coin":
        m = round(float(rng.uniform(0.1, 0.9)) / 0.05) * 0.05
        s0 = logu(rng, 2, 40)
        a, b = max(1, round(m * s0)), max(1, round((1 - m) * s0))
        k = int(py.choice([3, 4]))
        weak, strong = (a, b), (k * a, k * b)
        base, S = (strong, weak) if rng.uniform() < 0.5 else (weak, strong)
        tb = base[0] + base[1]
        d = (1 if rng.uniform() < 0.5 else -1) * max(1, round(float(rng.uniform(0.15, 0.35)) * tb))
        if not (base[0] + d >= 1 and base[1] - d >= 1):
            d = -d
        if not (base[0] + d >= 1 and base[1] - d >= 1):
            raise DuplicateItem("no room to move the prior mean")
        M = (base[0] + d, base[1] - d)
        specs = {"base": {"dist": "beta", "a": str(base[0]), "b": str(base[1])},
                 "strength": {"dist": "beta", "a": str(S[0]), "b": str(S[1])},
                 "mean": {"dist": "beta", "a": str(M[0]), "b": str(M[1])}}
        fam.prior = specs["base"]
        fam.truth = float(rng.uniform()) if G["off"] else float(rng.beta(base[0], base[1]))
        G["truth_nat"] = fam.truth
        data = fam.draw_counts(int(round(logu(rng, 3, 120))))
    elif f == "rate":
        shape = Fr(py.choice(["1", "1.5", "2", "3", "4", "5", "8", "10"]))
        mu0 = float(bn.fsig(logu(rng, 0.05, 50), 2))
        rate = Fr(bn.fsig(float(shape) / mu0, 3))
        k = int(py.choice([3, 4]))
        weak, strong = (shape, rate), (shape * k, rate * k)
        base, S = (strong, weak) if rng.uniform() < 0.5 else (weak, strong)
        g = float(rng.uniform(1.8, 3.0)) ** (1 if rng.uniform() < 0.5 else -1)
        M = (base[0], Fr(bn.fsig(float(base[1]) / g, 3)))
        specs = {n_: {"dist": "gamma", "shape": dec_str(p[0]), "rate": dec_str(p[1])} for n_, p in
                 (("base", base), ("strength", S), ("mean", M))}
        fam.prior = specs["base"]
        pdb = prior_dist(fam.prior)
        fam.truth = pdb.mean * logu(rng, 1 / 8, 8) if G["off"] else float(pdb.sample(rng, 1)[0])
        G["truth_nat"] = fam.truth
        data = fam.shown_block(fam.draw_list(int(round(logu(rng, 2, 12)))))
    else:
        pr = fam.prior
        s0 = Fr(pr["sd"])
        n = int(round(logu(rng, 1, 12)))
        fam.sigma = bn.fsig(float(s0) * math.sqrt(n) * logu(rng, 0.4, 2.5), 2)
        fam.r = bn.pow10_floor(float(Fr(fam.sigma)) / 3)
        cand = [s0 * k_ for k_ in (2, 3)] + [s0 / k_ for k_ in (2, 3)]
        ok = [c for c in cand if n_sig(c) <= 2]                                            # stated with <= 2 significant digits
        sS = ok[int(rng.integers(len(ok)))] if ok else s0 * 2
        r0 = Fr(dec_str(Fr(str(bn.pow10_floor(float(s0) / 3)))))
        delta = Fr(float(rng.uniform(1, 2.5)) * (1 if rng.uniform() < 0.5 else -1)) * s0
        mM = Fr(pr["mean"]) + (delta / r0).__round__() * r0
        specs = {"base": pr, "strength": {"dist": "normal", "mean": pr["mean"], "sd": dec_str(sS)},
                 "mean": {"dist": "normal", "mean": dec_str(mM), "sd": pr["sd"]}}
        pdb = prior_dist(pr)
        fam.truth = pdb.mean + pdb.s * float(rng.uniform(-5, 5)) if G["off"] else float(pdb.sample(rng, 1)[0])
        G["truth_nat"] = fam.truth
        data = fam.shown_block(fam.draw_list(n))
    sig = getattr(fam, "sigma", None)
    posts = {n_: posterior(f, s, [data], sig) for n_, s in specs.items()}
    if B.axes == "v2":                             # (v2) the shared data only: identical bins whatever the prior
        ax, ed, um = axis_v2(fam, [data], (G["ns"], G["gid"], "ax_shared"))
    elif f == "coin":
        ax, ed, um = fam.axis()
    elif f == "rate":
        ax, ed, um = fam.axis(UnionProxy(list(posts.values())), (G["ns"], G["gid"], "ax_shared"))
    else:
        ax, ed, um = fam.axis(UnionProxy(list(posts.values())), (G["ns"], G["gid"], "ax_shared"))
    G["ref"] = ref_range(ed, um)
    instr = fam.question(ax, ed)
    dl = fam.data_line(data)
    for n_ in ("base", "strength", "mean"):
        src = G["truth_source"] if n_ == "base" else "other_prior"
        rec = B.member(G, n_, fam.state(fam.prior_line(specs[n_]), [dl]), instr, "score", _post_q(), posts[n_],
                       {"swap": n_}, axis=ax, edges=ed, um=um, truth=truth_rec(G, to_disp(G["truth_nat"], um), src),
                       prior_spec=specs[n_], blocks=[data])
        if n_ == "base":
            rec["meta"]["bc"]["anchor"] = True


def g_reorder(B, G, gi):
    fam, rng = G["F"], G["rng"]
    f = fam.family
    if f == "coin":
        lo, hi = SEQ_BUCKETS[gi % len(SEQ_BUCKETS)]
        n = int(rng.integers(lo, hi + 1))
        for _ in range(50):
            seq = fam.draw_seq(n)
            if 0 < sum(seq) < n:
                break
        else:
            raise DuplicateItem("degenerate sequence")
        perm = [seq[i] for i in rng.permutation(n)]
        tries = 0
        while perm == seq and tries < 20:
            perm = [seq[i] for i in rng.permutation(n)]
            tries += 1
        first = 1 if rng.uniform() < 0.5 else 0
        blocked = sorted(seq, key=lambda x: x != first)
        orders = {"orig": seq, "perm": perm, "sorted": blocked}
        blocks = {k: {"kind": "seq", "x": v} for k, v in orders.items()}
        ax, ed, um = fam.axis()
        lab = "coin_seq"
    elif f == "rate":
        m = int(rng.integers(4, 13))
        for _ in range(50):
            bl = fam.draw_list(m)
            if len(set(bl["counts"])) > 1:
                break
        else:
            raise DuplicateItem("constant counts")
        c = bl["counts"]
        perm = [c[i] for i in rng.permutation(m)]
        while perm == c:
            perm = [c[i] for i in rng.permutation(m)]
        srt = sorted(c, reverse=bool(rng.uniform() < 0.5))
        blocks = {"orig": bl, "perm": {**bl, "counts": perm}, "sorted": {**bl, "counts": srt}}
        ax, ed, um = fam.axis(posterior(f, fam.prior, [bl]), (G["ns"], G["gid"], "ax")) if B.axes == "v1" else \
            axis_v2(fam, [bl], (G["ns"], G["gid"], "ax"))
        lab = "rate"
    else:
        n = int(rng.integers(4, 13))
        for _ in range(50):
            bl = fam.draw_list(n)
            if len(set(bl["x"])) > 1:
                break
        x = bl["x"]
        perm = [x[i] for i in rng.permutation(n)]
        while perm == x:
            perm = [x[i] for i in rng.permutation(n)]
        srt = sorted(x, key=lambda v: Fr(v), reverse=bool(rng.uniform() < 0.5))
        blocks = {"orig": bl, "perm": {"kind": "list", "x": perm}, "sorted": {"kind": "list", "x": srt}}
        ax, ed, um = fam.axis(posterior(f, fam.prior, [bl], fam.sigma), (G["ns"], G["gid"], "ax")) if B.axes == "v1" else \
            axis_v2(fam, [bl], (G["ns"], G["gid"], "ax"))
        lab = "normal_mean"
    G["ref"] = ref_range(ed, um)
    instr = fam.question(ax, ed)
    post = posterior(f, fam.prior, [blocks["orig"]], getattr(fam, "sigma", None))
    for k_, b in blocks.items():
        line = fam.data_line(b, "list") if f != "coin" else fam.data_line(b)
        rec = B.member(G, k_, fam.state(fam.prior_line(fam.prior), [line]), instr, "score", _post_q(), post, {"order": k_},
                       axis=ax, edges=ed, um=um, truth=truth_rec(G, to_disp(G["truth_nat"], um)), fam_label=lab, blocks=[b])
        if k_ == "orig":
            rec["meta"]["bc"]["anchor"] = True


def g_raw_suff(B, G, gi):
    fam, rng = G["F"], G["rng"]
    f = fam.family
    if f == "coin":
        lo, hi = SEQ_BUCKETS[gi % len(SEQ_BUCKETS)]
        n = int(rng.integers(lo, hi + 1))
        raw = {"kind": "seq", "x": fam.draw_seq(n)}
        suff = {"kind": "counts", "n": n, "h": int(sum(raw["x"]))}
        ax, ed, um = fam.axis()
        lines = {"raw": fam.data_line(raw), "suff": fam.data_line(suff)}
        labs = {"raw": "coin_seq", "suff": "coin"}
    elif f == "rate":
        raw = fam.draw_list(int(rng.integers(2, 13)))
        suff = fam.as_total(raw)
        ax, ed, um = fam.axis(posterior(f, fam.prior, [raw]), (G["ns"], G["gid"], "ax")) if B.axes == "v1" else \
            axis_v2(fam, [suff], (G["ns"], G["gid"], "ax"))
        lines = {"raw": fam.data_line(raw, "list"), "suff": fam.data_line(suff)}
        labs = {"raw": "rate", "suff": "rate"}
    else:
        raw = fam.nudge_exact_mean(fam.draw_list(int(rng.integers(2, 13))))
        suff = fam.as_summary(raw)
        ax, ed, um = fam.axis(posterior(f, fam.prior, [raw], fam.sigma), (G["ns"], G["gid"], "ax")) if B.axes == "v1" else \
            axis_v2(fam, [suff], (G["ns"], G["gid"], "ax"))
        lines = {"raw": fam.data_line(raw, "list"), "suff": fam.data_line(suff)}
        labs = {"raw": "normal_mean", "suff": "normal_mean"}
    G["ref"] = ref_range(ed, um)
    instr = fam.question(ax, ed)
    post = posterior(f, fam.prior, [suff], getattr(fam, "sigma", None))
    assert post.params() == posterior(f, fam.prior, [raw], getattr(fam, "sigma", None)).params()
    for k_, b in (("raw", raw), ("suff", suff)):
        rec = B.member(G, k_, fam.state(fam.prior_line(fam.prior), [lines[k_]]), instr, "score", _post_q(), post, {"form": k_},
                       axis=ax, edges=ed, um=um, truth=truth_rec(G, to_disp(G["truth_nat"], um)), fam_label=labs[k_], blocks=[b])
        if k_ == "suff":
            rec["meta"]["bc"]["anchor"] = True


def normal_exact_sequential(fam, rng, py):
    """A normal problem whose posterior after batch 1 has EXACT finite-decimal parameters: sigma = i (i + 1) u, prior sd = (i + 1) u,
    n1 = 2 i + 1 readings -> posterior sd = i u and mean (i^2 m0 + sum1) / (i + 1)^2 (i in {1, 3, 4}; u a nice number)."""
    i = int(py.choice([1, 3, 4]))
    target = fam.S * logu(rng, 0.01, 0.3) / (i + 1)
    e = math.floor(math.log10(target))
    u = Fr(py.choice([1, 2, 5])) * Fr(10) ** e
    s0, sg, s1 = (i + 1) * u, i * (i + 1) * u, i * u
    r0 = Fr(dec_str(Fr(str(bn.pow10_floor(float(s0) / 3)))))
    m0 = (Fr(fam.S * float(rng.uniform(0.6, 1.6))) / r0).__round__() * r0
    fam.prior = {"dist": "normal", "mean": dec_str(m0), "sd": dec_str(s0)}
    fam.sigma = dec_str(sg)
    fam.r = bn.pow10_floor(float(sg) / 3)
    pd = prior_dist(fam.prior)
    fam.truth = pd.mean + pd.s * float(rng.uniform(-5, 5)) if fam.off else float(pd.sample(rng, 1)[0])
    n1 = 2 * i + 1
    n2 = int(rng.integers(1, 12 - n1 + 1))
    b1, b2 = fam.draw_list(n1), fam.draw_list(n2)
    S1 = sum(Fr(v) for v in b1["x"])
    m1 = (i * i * m0 + S1) / (i + 1) ** 2
    post1 = {"dist": "normal", "mean": dec_str(m1), "sd": dec_str(s1)}
    p1 = posterior("normal_mean", fam.prior, [b1], fam.sigma)
    assert abs(p1.m - float(m1)) < 1e-9 * max(1, abs(p1.m)) and abs(p1.s - float(s1)) < 1e-12 * max(1, p1.s), (p1.m, m1, p1.s, s1)
    return b1, b2, post1


def g_sequential(B, G):
    fam, rng, py = G["F"], G["rng"], G["py"]
    f = fam.family
    sig = None
    if f == "coin":
        b1, b2 = fam.draw_counts(int(round(logu(rng, 2, 60)))), fam.draw_counts(int(round(logu(rng, 2, 60))))
        joint = {"kind": "counts", "n": b1["n"] + b2["n"], "h": b1["h"] + b2["h"]}
        a, b = Fr(fam.prior["a"]), Fr(fam.prior["b"])
        post1 = {"dist": "beta", "a": dec_str(a + b1["h"]), "b": dec_str(b + b1["n"] - b1["h"])}
        if bn.fparam(float(Fr(post1["a"]))) != post1["a"] or bn.fparam(float(Fr(post1["b"]))) != post1["b"]:
            raise DuplicateItem("intermediate posterior not representable in the prior template")
        jl = [fam.data_line(joint)]
    elif f == "rate":
        b1 = fam.as_total(fam.draw_list(int(round(logu(rng, 1, 10)))))
        b2 = fam.as_total(fam.draw_list(int(round(logu(rng, 1, 10)))))
        joint = {"kind": "total", "tot": b1["tot"] + b2["tot"], "T": dec_str(Fr(b1["T"]) + Fr(b2["T"])), "m": b1["m"] + b2["m"]}
        post1 = {"dist": "gamma", "shape": dec_str(Fr(fam.prior["shape"]) + b1["tot"]),
                 "rate": dec_str(Fr(fam.prior["rate"]) + Fr(b1["T"]))}
        jl = [fam.data_line(joint)]
    else:
        b1, b2, post1 = normal_exact_sequential(fam, rng, py)
        sig = fam.sigma
        joint = {"kind": "list", "x": b1["x"] + b2["x"]}
        jl = [fam.data_line(joint, "list")]
    G["truth_nat"] = fam.truth
    post = posterior(f, fam.prior, [joint], sig)
    post_b1 = posterior(f, fam.prior, [b1], sig)
    if B.axes == "v2":                             # (v2) batch 2 is the only data every presentation shows (updated_prior)
        ax, ed, um = axis_v2(fam, [b2], (G["ns"], G["gid"], "ax_seq"))
        ax1, ed1, um1 = axis_v2(fam, [b1], (G["ns"], G["gid"], "ax_b1"))
    elif f == "coin":
        ax, ed, um = fam.axis()
        ax1, ed1, um1 = ax, ed, um
    else:
        ax, ed, um = fam.axis(post, (G["ns"], G["gid"], "ax_joint"))
        ax1, ed1, um1 = fam.axis(post_b1, (G["ns"], G["gid"], "ax_b1"))
    G["ref"] = ref_range(ed, um)
    instr = fam.question(ax, ed)
    tv = to_disp(G["truth_nat"], um)
    pl_ = fam.prior_line(fam.prior)
    d2 = fam.data_line(b2, "list") if f == "normal_mean" else fam.data_line(b2)
    mem = [("joint", fam.state(pl_, jl), [joint], fam.prior, [], None),
           ("two_batch", fam.state(pl_, [fam.two_batch_line(b1, b2)]), [b1, b2], fam.prior, ["state:3"], None),
           ("stated_post", fam.state(pl_, fam.stated_post_lines(b1, post1, b2)), [b2], post1, ["state:3", "state:4"],
            {"prior": fam.prior, "blocks": [b1]}),
           ("updated_prior", fam.state(fam.prior_line(post1), [d2]), [b2], post1, [], None)]
    for name, state, blocks, pspec, novel, restated in mem:
        problem = {"prior": pspec, "blocks": blocks, **({"sigma": sig} if f == "normal_mean" else {})}
        if restated:
            problem["restated"] = restated
            problem["shown_blocks"] = [b1, b2]
        src = G["truth_source"] if pspec is fam.prior else "other_prior"
        rec = B.member(G, name, state, instr, "score", _post_q(), post, {"presentation": name}, axis=ax, edges=ed, um=um,
                       truth=truth_rec(G, tv, src), novel=novel, problem=problem, fam_label=f)
        assert rec["meta"]["exact"]["params"] == post.params()["params"], (name, rec["meta"]["exact"], post.params())
        if name == "joint":
            rec["meta"]["bc"]["anchor"] = True
    # the CHAIN: b1, and the exact batch-1 bins passed on (moment-matched / histogram) with batch 2
    rec_b1 = B.member(G, "b1", fam.state(pl_, [fam.data_line(b1, "list") if f == "normal_mean" else fam.data_line(b1)]),
                      fam.question(ax1, ed1), "score", _post_q(), post_b1, {"chain_role": "b1"}, axis=ax1, edges=ed1, um=um1,
                      truth=truth_rec(G, to_disp(G["truth_nat"], um1)), blocks=[b1], fam_label=f)
    spec = chain_spec(G, fam, rec_b1, b2)
    if B.axes == "v2":                             # (v2) pass 2 rebuilds the group's shared axis from batch 2 (asserted equal)
        spec["axes"], spec["axis_seed"], spec["shared_edges"] = "v2", [G["ns"], G["gid"], "ax_seq"], list(ed)
    rec_b1["meta"]["bc"]["chain_spec"] = spec
    p_b1 = label_vector(rec_b1)
    for rep in ("mm", "hist"):
        chain_member(B, G, spec, p_b1, "x", rep)


def chain_spec(G, fam, rec_b1, b2):
    m = rec_b1["meta"]
    return {"family": fam.family, "ctx": fam.ctx["id"], "style": fam.st, "prior0": fam.prior, "batch2": [b2],
            "sigma": getattr(fam, "sigma", None), "b1_edges": m["bin_edges"], "b1_axis": m["axis"], "unit_map": m["bc"]["unit_map"],
            "b1_qid": next(iter(rec_b1["questions"])), "dunit": getattr(fam, "dunit", None), "scale": getattr(fam, "scale", None),
            "iv": [getattr(fam, "iv_name", None), getattr(fam, "L", None)], "pct": fam.st.get("pct"), "truth_nat": G["truth_nat"],
            "truth_source": G["truth_source"], "S": getattr(fam, "S", None), "r": getattr(fam, "r", None)}


def fam_from_spec(spec, off=False):
    """Rebuild the family renderer of a group from its chain_spec (pass 2 runs from the corpus files alone)."""
    rng, py = np.random.default_rng(0), random.Random(0)
    cls = FAM_CLS[spec["family"]]
    fam = cls.__new__(cls)
    Fam.__init__(fam, spec["family"], rng, py, off)
    ctxs = {"coin": bn.COIN_CTX, "rate": bn.RATE_CTX, "normal_mean": bn.NORM_CTX}[spec["family"]]
    fam.ctx = next(c for c in ctxs if c["id"] == spec["ctx"])
    fam.st, fam.prior, fam.truth = dict(spec["style"]), spec["prior0"], spec["truth_nat"]
    if spec["family"] == "coin":
        fam.sym_short = "p"
    elif spec["family"] == "rate":
        fam.iv_name, fam.L = spec["iv"]
        fam.unit, fam.sym_short = fam.ctx["unit"], "λ"
    else:
        fam.sigma, fam.unit, fam.dunit, fam.scale = spec["sigma"], fam.ctx["unit"], spec["dunit"], spec["scale"]
        fam.S, fam.r, fam.sym_short = spec["S"], spec["r"], "μ"
    return fam


def chain_prior(spec, p_b1, rep):
    """The batch-1 answer p_b1 (bins of the b1 axis) as the stated prior of a second prompt.
    mm   : Sheppard-corrected binned moments (variance floored at a quarter of the uncorrected one) -> Beta / Gamma / Normal at the
           corpus's stated precision (fparam / fsig 3 / fsig 2), so the prompt uses the in-distribution prior template;
    hist : the bins themselves, probabilities rounded to HIST_DP decimals (zeros dropped), renormalised, uniform within bins."""
    f, um = spec["family"], spec["unit_map"]
    edges = np.asarray(spec["b1_edges"], float)
    en = to_nat(edges, um)
    p = np.clip(np.asarray(p_b1, float), 0, None)
    p = p / p.sum()
    info = {}
    if rep == "mm":
        mid, w = (en[:-1] + en[1:]) / 2, np.diff(en)
        mu = float((p * mid).sum())
        v0 = float((p * (mid - mu) ** 2).sum())
        var = max(v0 - float(np.mean(w)) ** 2 / 12, v0 / 4, 1e-30)
        info = {"mean": mu, "var": var, "var_uncorrected": v0}
        d = moment_match(f, mu, var)
        if f == "coin":
            out = {"dist": "beta", "a": bn.fparam(d.a), "b": bn.fparam(d.b)}
            if float(Fr(out["a"])) <= 0 or float(Fr(out["b"])) <= 0:
                out = {"dist": "beta", "a": "0.5", "b": "0.5"}
        elif f == "rate":
            shape = max(d.k, 0.5)
            out = {"dist": "gamma", "shape": bn.fparam(shape), "rate": bn.fsig(float(Fr(bn.fparam(shape))) / max(mu, 1e-12), 3)}
        else:
            sd = bn.fsig(max(math.sqrt(var), 1e-12), 2)
            r0 = bn.pow10_floor(float(Fr(sd)) / 3)
            out = {"dist": "normal", "mean": bn.fnum(round(mu / r0) * r0, bn.dec_for_res(r0)), "sd": sd}
        info["clipped"] = (f == "coin" and (min(mu, 1 - mu) < 0.005)) or (f == "rate" and d.k < 0.5)
        return out, info
    q = np.round(p, HIST_DP)
    if q.sum() <= 0:
        q[int(np.argmax(p))] = 1.0
    txt = [bn.fnum(x, HIST_DP) for x in q]
    masses = np.array([float(Fr(t)) for t in txt])
    ax = dict(spec["b1_axis"], fold_low=False, fold_high=False)
    labels = bn.bin_labels(ax, list(edges))
    dec = ax["decimals"]
    out = {"dist": "hist", "edges": [float(x) for x in en], "masses": [float(x) for x in masses / masses.sum()],
           "masses_txt": txt, "labels": labels, "width_txt": bn.fnum(ax["width"], dec), "lo_txt": bn.fnum(edges[0], dec),
           "hi_txt": bn.fnum(edges[-1], dec), "pct": bool(spec.get("pct")) and f == "coin", "dunit": spec.get("dunit")}
    info = {"stated_sum": float(Fr(sum(Fr(t) for t in txt))), "n_bins_listed": int((q > 0).sum())}
    return out, info


def chain_member(B, G, spec, p_b1, source, rep, model=None):
    """xchain (source 'x': the EXACT b1 bins) or mchain (source 'm': a MODEL's b1 output) for one representation."""
    fam = G["F"]
    f = fam.family
    pspec, info = chain_prior(spec, p_b1, rep)
    sig = spec["sigma"]
    blocks = spec["batch2"]
    post = posterior(f, pspec, blocks, sig)
    if spec.get("axes") == "v2":                   # (v2) the group's shared axis, from batch 2 alone
        ax, ed, um = axis_v2(fam, blocks, tuple(spec["axis_seed"]))
        assert [float(x) for x in ed] == [float(x) for x in spec["shared_edges"]], (G["gid"], "chain axis != the group's")
    elif f == "coin":
        ax, ed, um = fam.axis()
    elif f == "rate":
        ax, ed, um = fam.axis(post, (G["ns"], G["gid"], "chain", source, rep))
    else:
        ax, ed, um = fam.axis(post, (G["ns"], G["gid"], "chain", source, rep))
    d2 = fam.data_line(blocks[0], "list") if f == "normal_mean" else fam.data_line(blocks[0])
    state = fam.state(fam.prior_line(pspec), [d2])
    role = {"x": "xchain", "m": "mchain"}[source]
    rec = B.member(G, f"{role}_{rep}", state, fam.question(ax, ed), "score", _post_q(), post,
                   {"chain_role": role, "chain": {"rep": rep, "source": "exact" if source == "x" else f"model:{model}", **info}},
                   axis=ax, edges=ed, um=um, truth=truth_rec(G, to_disp(G["truth_nat"], um), "other_prior"),
                   novel=["state:2"] if rep == "hist" else [], prior_spec=pspec, blocks=blocks, fam_label=f)
    if isinstance(post, HistPostD) and post.degenerate:
        rec["meta"]["bc"]["chain"]["degenerate"] = True
    return rec


def g_reparam(B, G):
    fam, rng = G["F"], G["rng"]
    f = fam.family
    if f == "coin":
        data = fam.draw_counts(int(round(logu(rng, 1, 300))))
        post = posterior(f, fam.prior, [data])
        ax, ed, um = fam.axis(pct=False)
        ax2, ed2, um2 = fam.axis(pct=True)
        q1, q2 = fam.question(ax, ed, pct=False), fam.question(ax2, ed2, pct=True)
        kind, alt = "decimal", "percent"
    elif f == "rate":
        data = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 30)))))
        post = posterior(f, fam.prior, [data])
        alt, fac, dec = RATE_ALT[fam.ctx["id"]]
        v2 = B.axes == "v2"                        # (v2) native and alt axes from the same data and the same draws
        ax, ed, um = fam.axis(post, (G["ns"], G["gid"], "ax")) if not v2 else axis_v2(fam, [data], (G["ns"], G["gid"], "ax"))
        if dec:
            ax2, ed2 = scaled_axis(ax, ed, fac, alt)
            um2 = {"scale": float(Fr(fac)), "offset": 0.0}
        elif v2:
            ax2, ed2, um2 = axis_v2(fam, [data], (G["ns"], G["gid"], "ax"), um={"scale": float(Fr(fac)), "offset": 0.0}, unit=alt)
        else:
            ax2, ed2, um2 = fam.axis(post, (G["ns"], G["gid"], "ax_alt"), scale=fac, unit=alt)
        q1, q2 = fam.question(ax, ed), fam.question(ax2, ed2, unit=alt)
        kind = "decimal" if dec else "nondecimal"
    else:
        fam.dunit, fam.scale = fam.unit, 1.0
        data = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 40)))))
        post = posterior(f, fam.prior, [data], fam.sigma)
        alt, fac, off, dec = NORM_ALT[fam.ctx["id"]]
        v2 = B.axes == "v2"                        # (v2) native and alt axes from the same data and the same draws
        ax, ed, um = fam.axis(post, (G["ns"], G["gid"], "ax"), scale="1", unit=fam.unit) if not v2 else \
            axis_v2(fam, [data], (G["ns"], G["gid"], "ax"), um={"scale": 1.0, "offset": 0.0}, unit=fam.unit)
        if dec:
            ax2, ed2 = scaled_axis(ax, ed, fac, alt)
            um2 = {"scale": float(Fr(fac)), "offset": 0.0}
        elif v2:
            ax2, ed2, um2 = axis_v2(fam, [data], (G["ns"], G["gid"], "ax"), um={"scale": float(Fr(fac)), "offset": float(Fr(off))},
                                    unit=alt)
        elif Fr(off) != 0:                      # affine (Celsius -> Fahrenheit): the target in F, mapped back by the unit map
            dF = bn.NormalD(post.m * float(Fr(fac)) + float(Fr(off)), post.s * float(Fr(fac)))
            ax2, ed2, _ = fam.axis(dF, (G["ns"], G["gid"], "ax_alt"), scale="1", unit=alt)
            um2 = {"scale": float(Fr(fac)), "offset": float(Fr(off))}
            ax2 = dict(ax2, unit_map=um2)
        else:
            ax2, ed2, um2 = fam.axis(post, (G["ns"], G["gid"], "ax_alt"), scale=fac, unit=alt)
        q1, q2 = fam.question(ax, ed, unit=fam.unit), fam.question(ax2, ed2, unit=alt)
        kind = "decimal" if dec else ("affine" if Fr(off) != 0 else "nondecimal")
    G["ref"] = ref_range(ed, um)
    state = fam.state(fam.prior_line(fam.prior), [fam.data_line(data)])
    sig = getattr(fam, "sigma", None)
    rec = B.member(G, "native", state, q1, "score", _post_q(), post, {"reparam": kind, "unit_alt": None}, axis=ax, edges=ed, um=um,
                   truth=truth_rec(G, to_disp(G["truth_nat"], um)), blocks=[data], fam_label=f)
    rec["meta"]["bc"]["anchor"] = True
    B.member(G, "alt", state, q2, "score", _post_q(), post, {"reparam": kind, "unit_alt": alt}, axis=ax2, edges=ed2, um=um2,
             truth=truth_rec(G, to_disp(G["truth_nat"], um2)), blocks=[data], fam_label=f,
             novel=["instructions"] if f == "rate" or kind == "affine" or alt in ("mg/mL", "hours") else [])


def g_pvp(B, G):
    fam, rng, py = G["F"], G["rng"], G["py"]
    f = fam.family
    t = G["truth_nat"]
    if f == "coin":
        data = fam.draw_counts(int(round(logu(rng, 1, 100))))
        post = posterior(f, fam.prior, [data])
        ax, ed, um = fam.axis()
        G["ref"] = ref_range(ed, um)
        state = fam.state(fam.prior_line(fam.prior), [fam.data_line(data)])
        rec = B.member(G, "post", state, fam.question(ax, ed), "score", _post_q(), post, {}, axis=ax, edges=ed, um=um,
                       truth=truth_rec(G, to_disp(t, um)), blocks=[data])
        rec["meta"]["bc"]["anchor"] = True
        n1, nk = COIN_PRED[fam.ctx["id"]]
        y1 = int(rng.uniform() < t)
        B.member(G, "next1", state, n1, "noul", {"kind": "next1"}, post, {}, truth=truth_rec(G, y1), quantity="predictive",
                 fam_label="pred_next", novel=["instructions"], blocks=[data])
        k = int(rng.integers(4, 11))
        yk = int(rng.binomial(k, t))
        B.member(G, "nextk", state, nk.format(k=k), "score", {"kind": "nextk", "k": k}, post, {}, levels=[str(j) for j in range(k + 1)],
                 truth=truth_rec(G, yk), quantity="predictive", fam_label="pred_count", novel=["instructions"], blocks=[data])
    elif f == "rate":
        data = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 30)))))
        post = posterior(f, fam.prior, [data])
        ax, ed, um = fam.axis(post, (G["ns"], G["gid"], "ax")) if B.axes == "v1" else axis_v2(fam, [data], (G["ns"], G["gid"], "ax"))
        G["ref"] = ref_range(ed, um)
        state = fam.state(fam.prior_line(fam.prior), [fam.data_line(data)])
        rec = B.member(G, "post", state, fam.question(ax, ed), "score", _post_q(), post, {}, axis=ax, edges=ed, um=um,
                       truth=truth_rec(G, to_disp(t, um)), blocks=[data])
        rec["meta"]["bc"]["anchor"] = True
        target = float(rng.uniform(0.7, 5.0))
        cands = [Fr(x) for x in ("0.1", "0.2", "0.25", "0.5", "1", "2", "3", "4", "5", "10", "20", "50", "100", "200", "500", "1000")]
        Ln = min(cands, key=lambda c: abs(math.log(float(c) * post.mean / target)))
        Lt, base = dec_str(Ln), fam.ctx["base"]
        bu = base[0] if Ln == 1 else base[1]
        ev = fam.ctx["ev"]
        y = int(rng.poisson(t * float(Ln)))
        B.member(G, "count", state, f"How many {ev[1]} will be recorded in the next {Lt} {bu}?", "score",
                 {"kind": "count", "L": Lt}, post, {}, levels=[str(j) for j in range(COUNT_MAX)] + [f"{COUNT_MAX} or more"],
                 truth=truth_rec(G, y), quantity="predictive", fam_label="pred_count", novel=["instructions"], blocks=[data])
        B.member(G, "atleast1", state, f"Will at least one {ev[0]} be recorded in the next {Lt} {bu}?", "noul",
                 {"kind": "atleast1", "L": Lt}, post, {}, truth=truth_rec(G, int(y >= 1)), quantity="predictive",
                 fam_label="pred_next", novel=["instructions"], blocks=[data])
    else:
        data = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 40)))))
        sig = float(Fr(fam.sigma))
        post = posterior(f, fam.prior, [data], fam.sigma)
        v2 = B.axes == "v2"                        # (v2) every axis from the data (+ the predictive's own spread)
        ax, ed, um = fam.axis(post, (G["ns"], G["gid"], "ax")) if not v2 else axis_v2(fam, [data], (G["ns"], G["gid"], "ax"))
        G["ref"] = ref_range(ed, um)
        state = fam.state(fam.prior_line(fam.prior), [fam.data_line(data)])
        rec = B.member(G, "post", state, fam.question(ax, ed), "score", _post_q(), post, {}, axis=ax, edges=ed, um=um,
                       truth=truth_rec(G, to_disp(t, um)), blocks=[data])
        rec["meta"]["bc"]["anchor"] = True
        pr = bn.NormalD(post.m, math.hypot(post.s, sig))
        axp, edp, ump = fam.axis(pr, (G["ns"], G["gid"], "ax_pred")) if not v2 else \
            axis_v2(fam, [data], (G["ns"], G["gid"], "ax_pred"), kind="pred")
        y = float(rng.normal(t, sig))
        B.member(G, "pred", state, fam.question(axp, edp, kind="pred"), "score", _post_q("pred", tau=sig), post, {},
                 axis=axp, edges=edp, um=ump, truth=truth_rec(G, to_disp(y, ump)), quantity="predictive", fam_label="normal_pred",
                 blocks=[data])
        k = int(rng.integers(2, 11))
        pk = bn.NormalD(post.m, math.hypot(post.s, sig / math.sqrt(k)))
        axk, edk, umk = fam.axis(pk, (G["ns"], G["gid"], "ax_predk")) if not v2 else \
            axis_v2(fam, [data], (G["ns"], G["gid"], "ax_predk"), kind="predmean", k=k)
        yk = float(rng.normal(t, sig / math.sqrt(k)))
        r = fam.ctx["reading"]
        qk = (f"{k} more {r[1]} are about to be taken. Which bin will their mean fall in, in {fam.dunit}? "
              + bn.axis_sentence(axk, edk, fam.dunit))
        B.member(G, "predmean", state, qk, "score", _post_q("predmean", tau=sig / math.sqrt(k), k=k), post, {},
                 axis=axk, edges=edk, um=umk, truth=truth_rec(G, to_disp(yk, umk)), quantity="predictive",
                 fam_label="normal_predmean", novel=["instructions"], blocks=[data])


def g_events(B, G):
    fam, rng = G["F"], G["rng"]
    f = fam.family
    if f == "coin":
        data = fam.draw_counts(int(round(logu(rng, 2, 200))))
        sig = None
    elif f == "rate":
        data = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 30)))))
        sig = None
    else:
        data = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 40)))))
        sig = fam.sigma
    post = posterior(f, fam.prior, [data], sig)
    if B.axes == "v2":                             # (v2) axis AND thresholds from the data: two random quantiles of its likelihood
        ax, ed, um = axis_v2(fam, [data], (G["ns"], G["gid"], "ax"))
        G["ref"] = ref_range(ed, um)
        en = to_nat(np.asarray(ed, float), um)
        q1, q2 = float(rng.uniform(0.15, 0.45)), float(rng.uniform(0.55, 0.85))
        lq = np.asarray(flat_lik(fam, [data]).ppf(np.array([q1, q2])), float)
        j1 = int(np.argmin(np.abs(en[1:-1] - lq[0]))) + 1
        j2 = int(np.argmin(np.abs(en[1:-1] - lq[1]))) + 1
        if j2 <= j1:
            j1, j2 = (j1, j1 + 1) if j1 < K - 1 else (K - 2, K - 1)
    else:
        ax, ed, um = fam.axis() if f == "coin" else fam.axis(post, (G["ns"], G["gid"], "ax"))
        G["ref"] = ref_range(ed, um)
        en = to_nat(np.asarray(ed, float), um)
        F_e = np.asarray(post.cdf(en), float)
        for _ in range(40):
            q1, q2 = float(rng.uniform(0.15, 0.45)), float(rng.uniform(0.55, 0.85))
            j1 = int(np.argmin(np.abs(F_e[1:-1] - q1))) + 1
            j2 = int(np.argmin(np.abs(F_e[1:-1] - q2))) + 1
            if j1 < j2 and 0.02 < F_e[j1] < 0.7 and 0.3 < F_e[j2] < 0.98:
                break
        else:
            raise DuplicateItem("no informative thresholds")
    t1, t2 = float(en[j1]), float(en[j2])
    dec = ax["decimals"]
    s1, s2 = bn.fnum(ed[j1], dec), bn.fnum(ed[j2], dec)
    sym, u = fam.event_sym(fam.st.get("pct", False)) if f == "coin" else fam.event_sym()
    state = fam.state(fam.prior_line(fam.prior), [fam.data_line(data)])
    lab = f
    rec = B.member(G, "full", state, fam.question(ax, ed), "score", _post_q(), post, {"t_idx": [j1, j2]}, axis=ax, edges=ed, um=um,
                   truth=truth_rec(G, to_disp(G["truth_nat"], um)), blocks=[data], fam_label=lab)
    rec["meta"]["bc"]["anchor"] = True
    tv = G["truth_nat"]
    ev = {"gt_t1": ("gt", [t1], f"Is {sym} greater than {s1}{u}?", tv > t1),
          "le_t1": ("le", [t1], f"Is {sym} at most {s1}{u}?", tv <= t1),
          "le_t2": ("le", [t2], f"Is {sym} at most {s2}{u}?", tv <= t2),
          "in_t1_t2": ("in", [t1, t2], f"Is {sym} greater than {s1}{u} and at most {s2}{u}?", t1 < tv <= t2)}
    common = {"t_idx": [j1, j2], "t_disp": [s1, s2]}
    for name, (e, t, instr, y) in ev.items():
        B.member(G, name, state, instr, "noul", {"kind": "event", "event": e, "t": t}, post, common, truth=truth_rec(G, int(y)),
                 fam_label="event", novel=["instructions"], blocks=[data])
    crit = {"low": f"At most {s1}{u}", "mid": f"Greater than {s1}{u} and at most {s2}{u}", "high": f"Greater than {s2}{u}"}
    B.member(G, "choice3", state, f"Which range contains {sym}?", "choice", {"kind": "choice3", "t": [t1, t2]}, post, common,
             criteria=crit, truth=truth_rec(G, "low" if tv <= t1 else ("mid" if tv <= t2 else "high")), fam_label="event",
             novel=["instructions"], blocks=[data])


def g_loss(B, G):
    fam, rng = G["F"], G["rng"]
    f = fam.family
    if f == "coin":
        data, sig = fam.draw_counts(int(round(logu(rng, 1, 200)))), None
    elif f == "rate":
        data, sig = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 30))))), None
    else:
        data, sig = fam.shown_block(fam.draw_list(int(round(logu(rng, 1, 40))))), fam.sigma
    post = posterior(f, fam.prior, [data], sig)
    if B.axes == "v2":                             # (v2) the data's axis, shared by every loss member
        ax, ed, um = axis_v2(fam, [data], (G["ns"], G["gid"], "ax"))
    else:
        ax, ed, um = fam.axis() if f == "coin" else fam.axis(post, (G["ns"], G["gid"], "ax"))
    G["ref"] = ref_range(ed, um)
    instr = fam.question(ax, ed)
    base = [fam.prior_line(fam.prior), [fam.data_line(data)]]
    tv = to_disp(G["truth_nat"], um)
    rec = B.member(G, "none", fam.state(*base), instr, "score", _post_q(), post, {"loss": None}, axis=ax, edges=ed, um=um,
                   truth=truth_rec(G, tv), blocks=[data], fam_label=f)
    rec["meta"]["bc"]["anchor"] = True
    for loss in LOSSES:
        B.member(G, loss, fam.state(*base, extra=[fam.loss_line(loss)]), instr, "score", _post_q(), post, {"loss": loss},
                 axis=ax, edges=ed, um=um, truth=truth_rec(G, tv), blocks=[data], fam_label=f, novel=["state:4"])


BUILDERS = {"prior_only": g_prior_only, "prior_swap": g_prior_swap, "exchangeable_reorder": g_reorder,
            "raw_vs_sufficient": g_raw_suff, "sequential_vs_batch": g_sequential, "reparam_units": g_reparam,
            "posterior_vs_predictive": g_pvp, "event_coherence": g_events, "loss_invariance": g_loss}


def build_group(B, gtype, family, gi, ns, max_attempts=50):
    for att in range(max_attempts):
        G = new_group(gtype, family, gi, att, ns)
        G["F"] = FAM_CLS[family](G["rng"], G["py"], G["off"])
        G["truth_nat"] = G["F"].truth
        try:
            fn = BUILDERS[gtype]
            fn(B, G, gi) if gtype in ("exchangeable_reorder", "raw_vs_sufficient") else fn(B, G)
        except DuplicateItem:
            continue
        for r in G["members"]:
            B.seen.add(r["state"] + "\n" + next(iter(r["questions"].values()))["instructions"])
            b = r["meta"]["bc"]
            b["attempt"] = att
            b["off_prior"] = G["off"]
            b["sbc"] = bool(b["anchor"] and r["meta"]["truth"]["source"] == "prior" and b["n_data"] > 0)
        return G
    raise RuntimeError(f"cannot build {gtype}/{family}/{gi}")


# ============================================================================ build / chain2 / check
README = """# bayes_consistency_v1 -- item schema

Built by `build_bayes_consistency.py` (VERSION below) on `build_numeric_jev.py`'s renderers, axes and exact distributions. Every
record is a jev record with ONE question (as the numeric corpus): `id`, `source` (bayes_consistency/<type>), `license`, `base_id`
(= bc/<group id>: the bootstrap cluster), `state`, `questions {qid: question}`, `labels {qid: {option: exact probability}}`,
`label_kind` soft, `meta`. Files: `test/bc_<type>.jsonl`. Score with
`claude_lmscore.py --data DIR --splits test --n-perm 1 --exclude-identity` (Gemma: `--model google/gemma-4-31B-it --exact-path`;
(95) head checkpoints: `--num-head`, which reads only the 50-bin numeric members).

Question types
- `score` with 50 levels + `meta.axis` / `meta.bin_edges` (build_numeric_jev.py's numeric format; read by the (95) head): posterior
  of the parameter (`meta.bc.q.target` param), posterior predictive of the next normal reading (pred, the normal_pred format) or of
  the mean of the next k (predmean).
- `score` with k + 1 levels "0".."k" (coin: heads in the next k trials) or "0".."9", "10 or more" (rate: events in the next
  exposure). No `meta.bin_edges`, so the head never reads them.
- `noul` (Yes / No; labels {yes, no}): event queries P(theta > t), P(theta <= t), P(t1 < theta <= t2); the next coin trial; at
  least one rate event.
- `choice` (3 options low / mid / high): which range contains theta.

`meta` (as the numeric corpus, plus `bc`)
- family (coin, coin_seq, rate, normal_mean, normal_pred, normal_predmean, pred_next, pred_count, event), split test, context,
  quantity, prior {dist, params, stated}, exact {dist, params, mean, sd [, mean_binned, sd_binned, folded_low, folded_high]},
  truth {value (display units for 50-bin members; the realised outcome for predictive members), source prior | broad |
  other_prior}, data (sufficient statistics), baselines {prior, mle_wald [, mle_smooth]} (option order), mle, fingerprint,
  fp_in_train (problem also in numeric_v1 train), state_in_train.
- `bc`:
  - group, type, member, family (coin / rate / normal_mean), context, anchor (the group's in-distribution parameter question), sbc
    (anchor with data and a truth drawn from its stated prior), off_prior (the group's truth is a broad draw), attempt;
  - q: {kind: bins (target param / pred / predmean, tau) | event (event gt / le / in, t natural units) | choice3 (t) | next1 |
    nextk (k) | count (L) | atleast1 (L)};
  - unit_map {scale, offset}: display = natural x scale + offset (reparam_units alt members: percent, per week, Fahrenheit ...);
  - problem: {prior (stated spec, strings), blocks (data as shown, in order: coin seq / counts, rate list / total, normal list /
    summary), sigma, [shown_blocks, restated]}: the minimal correct reading, from which the exact target is recomputed;
  - ref {lo, hi}: the group's reference axis range (natural units); truth_nat; n_data; seq_len;
  - actions {loss: Bayes action, display units} for parameter members (squared -> mean, absolute -> median, pinball q -> q-quantile,
    asym_5x -> 5/6-quantile), from the exact continuous posterior;
  - novel: which parts use NEW PHRASING (`state:i` = line i of the state, 1 = context, 2 = prior, 3 = data, 4+ = extra lines;
    `instructions`); empty = in-distribution wording;
  - type-specific: swap (prior_swap), order (exchangeable_reorder), form (raw_vs_sufficient), presentation / chain_role / chain /
    chain_spec (sequential_vs_batch), reparam / unit_alt, t_idx / t_disp (event_coherence), loss (loss_invariance).

NEW PHRASING (minimal; everything else is build_numeric_jev.py's templates)
- rate with no data: "No calls have been counted yet." (the numeric corpus never has zero rate data)
- two_batch: "Data from two batches of flips. Batch 1: ... Batch 2: ..." (rate / normal alike)
- stated_post: "After a first batch of N flips (...), the posterior for p is Beta(a', b')." + "A second batch then gave ..."
- histogram prior (xchain_hist / mchain_hist): "After a first batch of data, your belief about p is a histogram over 50 equal-width
  bins ... The bin probabilities (bins not listed have probability 0) are: 0.40-0.42: 0.0123; ..."
- reparam alt units for rate (per day -> per week etc.), Celsius -> Fahrenheit, nitrate mg/mL, commute hours (the corpus's own
  alt units g -> kg, mm -> cm, ms -> s, V -> mV and percent axes are in-distribution)
- predictive questions: "Will the next flip land heads?", "How many of the next k flips will land heads?", "How many calls will be
  recorded in the next 2 hours?", "Will at least one call be recorded ...?", "k more readings are about to be taken. Which bin
  will their mean fall in ...?"
- event questions: "Is p greater than 0.36?", "Is p at most 0.36?", "Is p greater than 0.36 and at most 0.52?", "Which range
  contains p?"
- loss statements (a line after the data): "After you answer, a single point estimate of p will be chosen, and its cost will be the
  squared difference ..." (absolute; 1 below / 9 above; 9 below / 1 above; "underestimating ... costs 5 times as much per unit")

Stated numbers are exactly the numbers the labels use (every prior / intermediate posterior is formatted first and parsed back;
normal posteriors are computed in exact rationals so members that must agree have bit-identical labels).
"""


README_V2 = README.replace("# bayes_consistency_v1 -- item schema", "# bayes_consistency_v2 -- item schema (answer-independent axes)") \
    .replace("Stated numbers are exactly the numbers the labels use", """AXES (v2; research/numeric_v2_axes.md). No axis reads a prior or
an exact target. rate / normal axes come from the VISIBLE DATA only: the data's likelihood interval (rate Gamma(total + 1, exposure);
normal xbar +/- 2.576 sigma / sqrt(n); predictive members add the predictive's own spread), times a random zoom (0.8-6) with a random
offset from the group's own axis stream; groups that share no data (prior_only) use the context's fixed range. Members answering
the same question share ONE axis (identical bins: prior_swap, reorder, raw_vs_sufficient, sequential presentations + chain,
loss_invariance, prior_only); reparam alt axes use the same draws on the converted interval; event thresholds are bin edges near
random quantiles of the data's likelihood. Folded end-bin masses are recorded (meta.exact.folded_*, manifest axis_audit), never
used to redraw. `meta.bc.axes` = "v2".

Stated numbers are exactly the numbers the labels use""")


def load_numeric_train(path):
    fps, states = set(), set()
    p = Path(path) / "train"
    if not p.is_dir():
        return None
    for f in sorted(p.glob("*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            fps.add(r["meta"]["fingerprint"])
            (qid, q), = r["questions"].items()
            states.add(r["state"] + "\n" + q["instructions"])
    return fps, states


def numeric_fp(rec):
    """build_numeric_jev.py's fingerprint of the same inference problem, where one exists (prior + sufficient statistics)."""
    m = rec["meta"]
    bc = m["bc"]
    pr, st = bc["problem"]["prior"], stats_of(bc["family"], bc["problem"]["blocks"])
    if pr["dist"] == "beta":
        return f"coin|{bn.fparam(float(Fr(pr['a'])))}|{bn.fparam(float(Fr(pr['b'])))}|{st['n']}|{st['h']}"
    if pr["dist"] == "gamma":
        return f"rate|{pr['shape']}|{bn.fsig(float(Fr(pr['rate'])), 3)}|{st['tot']}|{bn.fsig(float(st['T']), 6)}"
    return None


# members that answer the same question on ONE axis (v2: chosen from the evidence all of them see); check() verifies identical bins
SHARED_AXIS = {"prior_only": ("nodata", "withdata"), "prior_swap": ("base", "strength", "mean"),
               "exchangeable_reorder": ("orig", "perm", "sorted"), "raw_vs_sufficient": ("raw", "suff"),
               "sequential_vs_batch": ("joint", "two_batch", "stated_post", "updated_prior", "xchain_mm", "xchain_hist", "mchain_mm",
                                       "mchain_hist"),
               "loss_invariance": ("none",) + tuple(LOSSES)}


def axis_audit(recs):
    """(v2) Per type: groups whose shared members (SHARED_AXIS) have identical bin edges, and the exact mass folded into the end bins
    of every 50-bin member (mean, quantiles, share > 5% / 25% / 50%), by type / member."""
    by_g = defaultdict(dict)
    fold = defaultdict(list)
    for r in recs:
        m, bc = r["meta"], r["meta"]["bc"]
        if "bin_edges" in m:
            by_g[bc["group"]][bc["member"]] = [float(x) for x in m["bin_edges"]]
            fold[f"{bc['type']}/{bc['member']}"].append(m["exact"]["folded_low"] + m["exact"]["folded_high"])
    shared = defaultdict(lambda: {"groups": 0, "identical": 0})
    for g, mem in by_g.items():
        t = g.split("/")[0]
        names = [n for n in SHARED_AXIS.get(t, ()) if n in mem]
        if len(names) > 1:
            shared[t]["groups"] += 1
            shared[t]["identical"] += all(mem[n] == mem[names[0]] for n in names)
    fs = {}
    for k, v in sorted(fold.items()):
        v = np.asarray(v)
        fs[k] = {"n": len(v), "mean": round(float(v.mean()), 5), "p50": round(float(np.median(v)), 6),
                 "p95": round(float(np.percentile(v, 95)), 5), "max": round(float(v.max()), 5),
                 "gt": {t: round(float((v > t).mean()), 4) for t in (0.05, 0.25, 0.5)}}
    return {"shared_axis": dict(shared), "folded": fs}


def build(out, scale=1.0, ns=NS, numeric_train=None, log=print, axes=None):
    """axes None -> DEFAULT_AXES ('v1', byte-identical); 'v2': the answer-independent axes (module docstring)."""
    axes = axes or DEFAULT_AXES
    if axes not in AXES:
        raise ValueError(f"axes {axes!r}")
    out = Path(out)
    t0 = time.time()
    tr = load_numeric_train(numeric_train) if numeric_train else None
    B = Builder(ns, forbidden=tr[1] if tr else None)
    B.axes = axes
    counts = {}
    for gtype in TYPES:
        n_fam = max(1, int(round(GROUPS[gtype] * scale / len(FAMILIES))))
        for fam in FAMILIES:
            for gi in range(n_fam):
                G = build_group(B, gtype, fam, gi, ns)
                B.records[gtype].extend(G["members"])
        counts[gtype] = {"groups": n_fam * len(FAMILIES), "items": len(B.records[gtype])}
        log(f"  {gtype}: {counts[gtype]} ({time.time() - t0:.0f} s)")
    flags = Counter()
    for gtype, recs in B.records.items():
        for r in recs:
            fp = numeric_fp(r)
            r["meta"]["fp_in_train"] = bool(tr and fp and fp in tr[0])
            r["meta"]["state_in_train"] = bool(tr and (r["state"] + "\n" + next(iter(r["questions"].values()))["instructions"]) in tr[1])
            flags["fp_in_train"] += r["meta"]["fp_in_train"]
            flags["state_in_train"] += r["meta"]["state_in_train"]
    (out / "test").mkdir(parents=True, exist_ok=True)
    for gtype, recs in B.records.items():
        with open(out / "test" / f"bc_{gtype}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    allr = [r for recs in B.records.values() for r in recs]
    qkinds = Counter(f"{r['meta']['bc']['type']}/{r['meta']['bc']['member']}" for r in allr)
    man = {"version": VERSION, "seed_namespace": ns, "scale": scale, "K": K, "license": LICENSE, "counts": counts,
           "items": len(allr), "groups": sum(c["groups"] for c in counts.values()),
           "members": dict(sorted(qkinds.items())), "p_off": P_OFF,
           "off_prior_groups": len({r["meta"]["bc"]["group"] for r in allr if r["meta"]["bc"]["off_prior"]}),
           "sbc_items": int(sum(r["meta"]["bc"]["sbc"] for r in allr)),
           "novel_items": int(sum(bool(r["meta"]["bc"]["novel"]) for r in allr)),
           "question_types": dict(Counter(next(iter(r["questions"].values()))["type"] for r in allr)),
           "numeric_train_checked": bool(tr), "train_overlap": dict(flags), "build_secs": round(time.time() - t0, 1)}
    if axes == "v2":                                 # (v2) only: the v1 manifest / README are unchanged
        man["version"], man["axes"] = VERSION_V2, "v2"
        man["numeric_train"] = str(numeric_train) if numeric_train else None
        man["axis_audit"] = axis_audit(allr)
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        (out / "README.md").write_text(README_V2.replace("VERSION below", VERSION_V2))
        return man
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    (out / "README.md").write_text(README.replace("VERSION below", VERSION))
    return man


def load_records(data, splits=("test",), pattern="*.jsonl"):
    recs = []
    for s in splits:
        for p in sorted((Path(data) / s).glob(pattern)):
            for ln in open(p, encoding="utf-8"):
                recs.append(json.loads(ln))
    return recs


def load_npz_rows(path):
    z = np.load(path, allow_pickle=False)
    P, qids = z["test_p"], [str(q) for q in z["test_qid"]]
    ks = z["test_ks"] if "test_ks" in z.files else np.full(len(qids), P.shape[1])
    out = {}
    for i, q in enumerate(qids):
        p = np.clip(P[i, : int(ks[i])].astype(np.float64), 0, None)
        out[q] = p / p.sum() if p.sum() > 0 else np.full(int(ks[i]), 1.0 / int(ks[i]))
    return out


def chain2(data, npz, out, model="model", ns=NS, log=print):
    """PASS 2 of the chained sequential update: each sequential_vs_batch group's b1 member as the MODEL answered it (pass-1 npz) ->
    a moment-matched and a histogram stated prior -> the second prompt with batch 2 (mchain_mm / mchain_hist)."""
    recs = [r for r in load_records(data) if r["meta"]["bc"]["type"] == "sequential_vs_batch"]
    preds = load_npz_rows(npz)
    B = Builder(ns)
    n_miss, outr = 0, []
    for r in recs:
        bc = r["meta"]["bc"]
        if bc["member"] != "b1":
            continue
        qid = next(iter(r["questions"]))
        if qid not in preds or len(preds[qid]) != K:
            n_miss += 1
            continue
        spec = bc["chain_spec"]
        B.axes = spec.get("axes", "v1")              # (v2) the suite's axis rule (pass 2 rebuilds the group's shared axis)
        fam = fam_from_spec(spec)
        G = {"gid": bc["group"], "type": "sequential_vs_batch", "members": [], "ns": ns, "F": fam, "truth_nat": spec["truth_nat"],
             "truth_source": spec["truth_source"], "ref": bc["ref"]}
        for rep in ("mm", "hist"):
            rec = chain_member(B, G, spec, preds[qid], "m", rep, model=model)
            rec["meta"]["bc"]["b1_qid"] = qid
            rec["meta"]["bc"]["chain_pass"] = 2
            rec["meta"]["fp_in_train"] = False
            rec["meta"]["state_in_train"] = False
        outr += G["members"]
    (Path(out) / "test").mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "test" / "bc_chain2.jsonl", "w", encoding="utf-8") as f:
        for r in outr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"version": VERSION, "pass": 2, "model": model, "from_npz": str(npz), "items": len(outr), "b1_missing": n_miss,
           "degenerate": int(sum(bool(r["meta"]["bc"]["chain"].get("degenerate")) for r in outr))}
    (Path(out) / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"chain2: {len(outr)} pass-2 items for {model} ({n_miss} b1 rows missing)")
    return man


def check(root):
    """Schema (build_jev_data.validate when importable; cache_features.iter_units / question_options), one question per record,
    qids / states unique, labels = exact targets recomputed from meta.bc.problem, members that must agree have identical labels."""
    root = Path(root)
    rep = {"errors": [], "items": 0, "recompute_maxdiff": 0.0, "agree_maxdiff": {}}
    try:
        bjd = bn.load_module("bjd_bc", HERE / "build_jev_data.py", stub_torch=False) if (HERE / "build_jev_data.py").exists() else None
    except Exception:
        bjd = None
    cf = None
    if (HERE / "trainers" / "cache_features.py").exists():
        cf = bn.load_module("cf_bc", HERE / "trainers" / "cache_features.py")
    recs = load_records(root)
    qids, keys = set(), set()
    by_group = defaultdict(dict)
    for r in recs:
        rep["items"] += 1
        if bjd:
            try:
                bjd.validate(r)
            except AssertionError as e:
                rep["errors"].append(f"{r['id']}: {e}")
        if len(r["questions"]) != 1:
            rep["errors"].append(f"{r['id']}: {len(r['questions'])} questions")
        (qid, q), = r["questions"].items()
        if qid in qids:
            rep["errors"].append(f"duplicate qid {qid}")
        qids.add(qid)
        k = r["state"] + "\n" + q["instructions"]
        if k in keys:
            rep["errors"].append(f"duplicate question text {qid}")
        keys.add(k)
        bc = r["meta"]["bc"]
        pr = bc["problem"]
        ex = posterior(bc["family"], pr["prior"], pr["blocks"], pr.get("sigma"))
        t = answer(r["meta"], ex)
        lab = label_vector(r)
        rep["recompute_maxdiff"] = max(rep["recompute_maxdiff"], float(np.abs(t - lab).max()))
        by_group[bc["group"]][bc["member"]] = lab
        if q["type"] == "score" and "bin_edges" in r["meta"] and len(q["levels"]) != K:
            rep["errors"].append(f"{qid}: K")
    if cf:
        units = list(cf.iter_units(str(root), ["test"], set()))
        rep["units"] = len(units)
        if len(units) != rep["items"]:
            rep["errors"].append(f"iter_units {len(units)} != records {rep['items']}")
        if any(cf.is_noncommercial(u) for u in units):
            rep["errors"].append("NC-flagged units")
    agree = {"exchangeable_reorder": ("orig", "perm", "sorted"), "raw_vs_sufficient": ("raw", "suff"),
             "sequential_vs_batch": ("joint", "two_batch", "stated_post", "updated_prior"),
             "loss_invariance": ("none",) + tuple(LOSSES)}
    for g, mem in by_group.items():
        t = g.split("/")[0]
        if t in agree:
            vs = [mem[m] for m in agree[t] if m in mem]
            d = max(float(np.abs(v - vs[0]).max()) for v in vs)
            rep["agree_maxdiff"][t] = max(rep["agree_maxdiff"].get(t, 0.0), d)
    if rep["recompute_maxdiff"] > 1e-7:
        rep["errors"].append(f"labels differ from recomputed targets by {rep['recompute_maxdiff']}")
    for t, d in rep["agree_maxdiff"].items():
        if d > 0:
            rep["errors"].append(f"{t}: members that must agree differ by {d}")
    if any((r["meta"]["bc"].get("axes") == "v2") for r in recs):   # (v2) shared members must have identical bins
        aud = axis_audit(recs)["shared_axis"]
        rep["v2_shared_axis"] = aud
        for t, d in aud.items():
            if d["identical"] != d["groups"]:
                rep["errors"].append(f"{t}: {d['groups'] - d['identical']} groups whose shared members have different axes")
    rep["n_errors"] = len(rep["errors"])
    rep["errors"] = rep["errors"][:40]
    return rep


# ============================================================================ synthetic predictors (suite validation)
PREDICTORS = {
    "exact": "the exact labels",
    "prior_only": "returns the stated prior (prior predictive for predictive questions)",
    "likelihood_only": "flat prior: the normalised likelihood (uniform over the reference axis without data)",
    "fixed_var": "right mean, constant sd (a tenth of the group's reference axis)",
    "order_sensitive": "weights later raw observations more (exp(3 * position), mean weight 1)",
    "order_long": "order-sensitive only for raw sequences longer than 40",
    "order_short": "ignores order only for long sequences: order-sensitive for raw sequences of length <= 40",
    "overconfident": "exact mean, sd x 0.6 (moment-matched in the family)",
    "underconfident": "exact mean, sd x 1.6",
    "mle_point": "a point mass at the MLE (1% uniform), plug-in predictive; uniform without data",
    "double_count": "a stated intermediate posterior is updated with the restated batch 1 again",
    "unit_blind": "reads the native-unit belief on an alternative-unit axis without converting",
    "event_bias": "adds +0.15 to the yes-probability of every event question",
    "loss_shift": "moves its belief to the stated loss's Bayes action (belief changes with the loss)",
    "plugin_pred": "posterior predictive by plugging in the posterior mean (ignores parameter uncertainty)",
}


def pred_belief(meta, pred):
    bc = meta["bc"]
    f, pr = bc["family"], bc["problem"]
    sig = pr.get("sigma")
    exact = posterior(f, pr["prior"], pr["blocks"], sig)
    ref = bc["ref"]
    if pred in ("exact", "event_bias", "unit_blind", "plugin_pred"):
        return exact
    if pred == "prior_only":
        return prior_dist(pr["prior"])
    if pred == "likelihood_only":
        st = stats_of(f, pr["blocks"])
        if st["n"] == 0:
            return UniformD(ref["lo"], ref["hi"])
        if f == "coin":
            return bn.BetaD(1.0 + st["h"], 1.0 + st["n"] - st["h"])
        if f == "rate":
            return bn.GammaD(1.0 + st["tot"], float(st["T"]))
        return bn.NormalD(float(st["S"]) / st["n"], float(Fr(sig)) / math.sqrt(st["n"]))
    if pred == "fixed_var":
        return bn.NormalD(exact.mean, (ref["hi"] - ref["lo"]) / 10)
    if pred == "order_sensitive":
        return posterior_weighted(f, pr["prior"], pr["blocks"], sig)
    if pred == "order_long":
        return posterior_weighted(f, pr["prior"], pr["blocks"], sig, min_len=LONG_SEQ + 1)
    if pred == "order_short":
        return posterior_weighted(f, pr["prior"], pr["blocks"], sig, max_len=LONG_SEQ)
    if pred in ("overconfident", "underconfident"):
        c = 0.6 if pred == "overconfident" else 1.6
        return moment_match(f, exact.mean, exact.var * c * c)
    if pred == "mle_point":
        mle, _ = mle_of(f, stats_of(f, pr.get("shown_blocks", pr["blocks"])), sig)
        if mle is None:
            return UniformD(ref["lo"], ref["hi"])
        return bn.NormalD(mle, (ref["hi"] - ref["lo"]) * 1e-5)
    if pred == "double_count":
        if "restated" in pr:
            return posterior(f, pr["prior"], pr["restated"]["blocks"] + pr["blocks"], sig)
        return exact
    if pred == "loss_shift":
        if bc.get("loss"):
            return ShiftD(exact, bayes_action(exact, bc["loss"]) - exact.mean)
        return exact
    raise ValueError(pred)


def synth_predict(rec, pred):
    m = rec["meta"]
    bc = m["bc"]
    if pred == "exact":
        return label_vector(rec)
    b = pred_belief(m, pred)
    q = bc["q"]
    if pred == "unit_blind" and q["kind"] == "bins" and bc.get("unit_alt"):
        p = masses_on_axis(b, m["bin_edges"], {"scale": 1.0, "offset": 0.0})
    elif pred == "plugin_pred" and m.get("quantity") == "predictive":
        p = answer(m, bn.NormalD(b.mean, max(abs(b.mean), 1e-3) * 1e-7))
    else:
        p = answer(m, b)
    if pred == "event_bias" and q["kind"] == "event":
        y = float(np.clip(p[0] + 0.15, 0.005, 0.995))
        p = np.array([y, 1 - y])
    if pred == "mle_point":
        p = 0.99 * p + 0.01 / len(p)
    p = np.clip(p, 0, None)
    return p / p.sum()


def write_npz(recs, rows, out):
    """rows {qid: p (option order)} -> a claude_lmscore-format npz (test_p, test_t, test_ks, test_soft, test_qid, test_source)."""
    qs = [r for r in recs if next(iter(r["questions"])) in rows]
    kmax = max(len(option_keys(r)) for r in qs)
    n = len(qs)
    P, T = np.zeros((n, kmax)), np.zeros((n, kmax))
    ks = np.zeros(n, dtype=np.int64)
    qid, src = [], []
    for i, r in enumerate(qs):
        q = next(iter(r["questions"]))
        p, t = rows[q], label_vector(r)
        P[i, : len(p)], T[i, : len(t)], ks[i] = p, t, len(p)
        qid.append(q)
        src.append(r["source"])
    np.savez_compressed(out, test_p=P, test_t=T, test_ks=ks, test_soft=np.ones(n, dtype=bool), test_qid=np.array(qid),
                        test_source=np.array(src))
    return n


def synth(data, pred, out, pattern="*.jsonl"):
    recs = load_records(data, pattern=pattern)
    rows = {next(iter(r["questions"])): synth_predict(r, pred) for r in recs}
    return write_npz(recs, rows, out)


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--ns", default=NS)
    b.add_argument("--numeric-train", default=str(HERE / "tmp" / "numeric_v1" / "numeric_v1"),
                   help="numeric_v1 corpus root, to flag problems / questions that also occur in its train split ('' = skip)")
    b.add_argument("--axes", choices=AXES, default=None,
                   help="v1 (default, byte-identical): axes around the exact answers; v2: answer- and prior-independent axes")
    c = sub.add_parser("check")
    c.add_argument("root")
    c.add_argument("--json", default="")
    c2 = sub.add_parser("chain2")
    c2.add_argument("--data", required=True)
    c2.add_argument("--npz", required=True)
    c2.add_argument("--out", required=True)
    c2.add_argument("--model", default="model")
    s = sub.add_parser("synth")
    s.add_argument("--data", required=True)
    s.add_argument("--pred", required=True, choices=sorted(PREDICTORS))
    s.add_argument("--out", required=True)
    s.add_argument("--pattern", default="*.jsonl")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        man = build(a.out, a.scale, a.ns, a.numeric_train or None, axes=a.axes)
        print(json.dumps(man, indent=1))
        return 0
    if a.cmd == "check":
        rep = check(a.root)
        txt = json.dumps(rep, indent=1)
        print(txt)
        if a.json:
            Path(a.json).write_text(txt)
        return 0 if rep["n_errors"] == 0 else 1
    if a.cmd == "chain2":
        chain2(a.data, a.npz, a.out, a.model)
        return 0
    if a.cmd == "synth":
        n = synth(a.data, a.pred, a.out, a.pattern)
        print(f"wrote {a.out}: {n} rows ({a.pred}: {PREDICTORS[a.pred]})")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
