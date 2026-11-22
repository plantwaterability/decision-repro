#!/usr/bin/env python3
"""build_numeric_jev.py -- the (94) NUMERIC-ANSWER PILOT corpus (research/numeric_pilot.md, research/probability_native.md item 1).

A numeric answer is asked as a `score` question whose K = 50 ordered options are equal-width BINS on a declared axis, so one
forward pass of the existing letter readout (claude_lmscore.py; letters A..Z a..x) returns a distribution over the number. The
label of every question is the EXACT posterior (or posterior-predictive) probability mass in each bin, from a conjugate model
that the state states in full: prior, likelihood and data. Nothing about the architecture changes; claude_lmtune.py trains on
these records like any other `score` question (soft CE, optionally + RPS in level order).

Families
  in-family (train / val / test)
    coin         Beta-Binomial: prior Beta(a, b) on a success probability p, n trials with h successes -> p | data
    rate         Gamma-Poisson: prior Gamma(shape, rate) on a Poisson rate, counts over an exposure -> rate | data
    normal_mean  Normal mean, known noise sd: prior N(m0, s0^2), n readings -> mu | data
  held out (test only; never trained)
    normal_pred  the posterior PREDICTIVE of the next reading in the normal_mean setting (realised value recorded)
    ab_diff      A/B test: posterior of p_A - p_B for two independent Beta-Binomial rates (numeric convolution)
    coin_seq     coin with the evidence as a raw outcome sequence (H T T H ...) instead of counts; seq_len recorded, 7 length
                 buckets 1-5 .. 161-320, to measure single-pass aggregation against evidence length
  probe          the user's example: prior Beta(10, 10), 6 heads in 10 flips -> exact Beta(16, 14), mean 0.5333, sd 0.0896
                 (the Pyro SVI tutorial reports 0.532 +/- 0.090); a canonical record + 3 template renderings, one base_id

Axis conventions (meta.axis, meta.bin_edges; display units)
  natural   coin / coin_seq / probe: [0, 1] (or 0..100 percent), width 0.02 (2); ab_diff: [-1, 1] (or -100..100 percentage
            points), width 0.04 (4). No folding.
  declared  rate / normal_mean / normal_pred: a per-question range stated in the question with its unit; equal bins of a
            "nice" width (1, 2, 2.5, 5 x 10^k). The range is placed around the answer's own central 99% interval with a random
            ZOOM (axis width / interval width, log-uniform 0.8..6 before rounding the width up to a nice number) and a random
            OFFSET (the interval's position inside the axis is uniform), so the axis carries no location hint and only a loose
            scale hint. Probability outside the range is FOLDED into the end bins (labels "under X" / "X or more"; the
            question says so); the folded masses are recorded (meta.exact.folded_low / folded_high), and an axis folding more
            than 25% is redrawn. A rate axis never starts below 0.

  ** v1's declared axes ENCODE THE ANSWER (an external audit, 2026-10-05; research/numeric_v2_axes.md): the range is a function of
  the exact posterior's 0.5% / 99.5% quantiles, so the bin labels give away its location and spread, and an evidence-only prompt
  still carries the prior through them. v1 is kept, byte-identical, as the "assisted-range" benchmark. **

Axis rule v2 (`build --axes v2`; research/numeric_v2_axes.md). The axis NEVER reads the prior, the posterior or the predictive:
  natural   coin / coin_seq / probe / ab_diff: as v1 (fixed per family).
  evidence  rate / normal_mean / normal_pred with data: a function of the VISIBLE EVIDENCE only. The data interval is the central
            99% interval of the normalised likelihood (the flat-prior posterior): rate Gamma(total + 1, exposure); normal_mean
            xbar +/- 2.576 sigma / sqrt(n); normal_pred xbar +/- 2.576 sqrt(sigma^2 / n + sigma^2) (sigma is the stated noise sd).
            Random ZOOM (axis width / data-interval width, log-uniform ZOOM_V2, chosen on TRAIN folded-mass statistics) and
            OFFSET (the data interval's position inside the axis is uniform), nice width, folded ends, a rate axis never below 0.
            The zoom / offset come from their own RNG stream (seed + "axis_v2"), so nothing about the item changes them. NO
            rejection step: the exact mass folded into the end bins is recorded (meta.exact.folded_*) and reported, never used.
  context   no data (normal_mean n = 0; the consistency suite's no-data members): a fixed physical range per context, from 0 to a
            round number above the context's largest plausible value (CTX_RANGE), in the question's unit.
  v2 also fixes v1's rounded-prior label bug (a Beta prior's label uses exactly the printed parameters), draws the prior sentence's
  template from its own stream (so a different prior changes the prior line only), keys the evidence-only sentence by the item
  (not the prior-dependent record id), and adds two test-only STRESS slices on the same evidence rule (stress_offc: the data
  interval at one end of a 4-10x zoomed axis; stress_wide: a 20-60x zoom), base families rate / normal_mean alternating.

Record schema (jev): id, source (numeric/<family>), license, base_id, state, questions {qid: {type: score, instructions,
levels [50 bin labels, lowest first]}}, labels {qid: {label: mass}}, label_kind {qid: soft}, meta (family, split, format,
context, axis, bin_edges, prior / exact posterior parameters, exact mean / sd (continuous and binned), folded masses, the
true / realised value, sufficient statistics, fingerprint, and in val / test the baselines: prior-only, plug-in MLE point mass
(smoothed), plug-in MLE Wald normal). qids are unique corpus-wide.

Truth: the parameter is drawn from the stated prior for 80% of questions and from a broad distribution for 20% (prior-data
conflict); meta.truth.source says which. Labels are the exact Bayes posterior under the STATED prior either way.

Determinism: every record is seeded from sha1(namespace | split | family | index | attempt). A question (state + instructions)
never repeats across splits or within val / test (redrawn). Identical inference problems (fingerprint = family, stated prior,
sufficient statistics) are allowed in train -- rejecting them would remove the small-n / uniform-prior corner from training --
except the probe's; every val / test record carries meta.fp_in_train, and eval_numeric.py reports the fresh subset too.

  .venv/bin/python build_numeric_jev.py build --out tmp/numeric_v1/numeric_v1            # full corpus (~70k questions)
  .venv/bin/python build_numeric_jev.py build --out tmp/numv2/numeric_v2 --axes v2       # v2: answer-independent axes
  .venv/bin/python build_numeric_jev.py build --out DIR --scale 0.01                     # smoke
  .venv/bin/python build_numeric_jev.py check DIR [--prompt-lengths]                     # schema / disjointness / labels
  .venv/bin/python build_numeric_jev.py self-test                                       # closed forms vs Monte Carlo
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
from pathlib import Path

import numpy as np
from scipy import special as sp

HERE = Path(__file__).resolve().parent
VERSION = "build_numeric_jev.py v1 (2026-10-04)"
NS = "numeric/v1"
K = 50
LICENSE = "synthetic (ours), commercial-OK: generated by build_numeric_jev.py, no third-party data or text"
IN_FAMILY = ("coin", "rate", "normal_mean")
HELD_OUT = ("normal_pred", "ab_diff", "coin_seq")
SIZES = {
    "train": {"coin": 20000, "rate": 20000, "normal_mean": 20000},
    "val": {"coin": 1000, "rate": 1000, "normal_mean": 1000},
    "test": {"coin": 1000, "rate": 1000, "normal_mean": 1000, "normal_pred": 1500, "ab_diff": 1500, "coin_seq": 2100},
}
SEQ_BUCKETS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 160), (161, 320))
P_BROAD = 0.2                      # share of questions whose truth is NOT drawn from the stated prior
ZOOM = (0.8, 6.0)                  # declared axes: axis width / central-99% width, log-uniform, before nice rounding
MAX_FOLD = 0.25                    # declared axes: at most this much of the answer's mass folded into the end bins
LABEL_DP = 8                       # labels / baselines rounded to this many decimals
PROBE = {"a": 10, "b": 10, "n": 10, "h": 6}
# ---------------------------------------------------------------------------- axis rule v2 (answer- and prior-independent)
VERSION_V2 = "build_numeric_jev.py v2-axes (2026-10-05)"
AXES = ("v1", "v2")
ZOOM_V2 = (0.8, 6.0)               # v2: axis width / DATA-interval width, log-uniform. zoom_lo by a rule fixed in advance on TRAIN only
                                   # (tmp/numv2/zoom_sweep.py: the smallest of 0.8 .. 3 with mean folded <= 1% and p95 <= 5%; v1's range)
Z99 = 2.5758293035489004           # half-width of a central 99% normal interval, in sds
STRESS = {"stress_offc": {"zoom": (4.0, 10.0), "place": "end"},   # test only: the data interval at one end of the axis
          "stress_wide": {"zoom": (20.0, 60.0), "place": None}}  # test only: the posterior occupies a few bins
STRESS_BASE = ("rate", "normal_mean")                            # alternating by item index
SIZES_V2_EXTRA = {"test": {"stress_offc": 1000, "stress_wide": 1000}}
RATE_CTX_RANGE = 50.0              # v2 context axis, rate (no data): 0 .. 50 per context unit (the prior means' upper end)
# CTX_RANGE (normal contexts, no data: 0 .. 1.6 x the context's largest scale S) is defined below NORM_CTX


# ============================================================================ seeding / formatting
def seed_of(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def sha16(s):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def nice_ceil(x):
    """Smallest of {1, 2, 2.5, 5} x 10^e that is >= x."""
    e = math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        v = m * 10.0 ** e
        if v >= x * (1 - 1e-12):
            return float(f"{v:.12g}")
    return float(f"{10.0 ** (e + 1):.12g}")


def decimals_of(w):
    """Decimals that represent every multiple of a nice width w exactly."""
    e = math.floor(math.log10(w) + 1e-12)
    mant = w / 10.0 ** e
    return max(0, -e + (1 if abs(mant - 2.5) < 1e-9 else 0))


def fnum(x, dec):
    """Fixed-point text with `dec` decimals and no negative zero."""
    s = f"{x:.{dec}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def fsig(x, sig=2):
    """`sig` significant digits (integer part never rounded away), trailing zeros stripped."""
    if x == 0:
        return "0"
    dec = max(0, sig - 1 - math.floor(math.log10(abs(x))))
    s = f"{x:.{dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fparam(x):
    """A prior parameter as text: integers without a decimal point, else up to 3 decimals."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else fsig(x, 3)


def pow10_floor(x):
    return 10.0 ** math.floor(math.log10(x))


def dec_for_res(r):
    return max(0, -int(math.floor(math.log10(r) + 1e-12)))


# ============================================================================ distributions (natural units)
class BetaD:
    def __init__(self, a, b):
        self.a, self.b = float(a), float(b)
        self.mean = self.a / (self.a + self.b)
        self.var = self.a * self.b / ((self.a + self.b) ** 2 * (self.a + self.b + 1))
        self.support = (0.0, 1.0)

    def cdf(self, x):
        return sp.betainc(self.a, self.b, np.clip(np.asarray(x, float), 0.0, 1.0))

    def ppf(self, q):
        return sp.betaincinv(self.a, self.b, np.asarray(q, float))

    def params(self):
        return {"dist": "beta", "params": {"a": self.a, "b": self.b}}

    def sample(self, rng, n):
        return rng.beta(self.a, self.b, n)


class GammaD:
    def __init__(self, shape, rate):
        self.k, self.r = float(shape), float(rate)
        self.mean = self.k / self.r
        self.var = self.k / self.r ** 2
        self.support = (0.0, math.inf)

    def cdf(self, x):
        return sp.gammainc(self.k, self.r * np.maximum(np.asarray(x, float), 0.0))

    def ppf(self, q):
        return sp.gammaincinv(self.k, np.asarray(q, float)) / self.r

    def params(self):
        return {"dist": "gamma", "params": {"shape": self.k, "rate": self.r}}

    def sample(self, rng, n):
        return rng.gamma(self.k, 1.0 / self.r, n)


class NormalD:
    def __init__(self, m, s):
        self.m, self.s = float(m), float(s)
        self.mean, self.var = self.m, self.s ** 2
        self.support = (-math.inf, math.inf)

    def cdf(self, x):
        return sp.ndtr((np.asarray(x, float) - self.m) / self.s)

    def ppf(self, q):
        return self.m + self.s * sp.ndtri(np.asarray(q, float))

    def params(self):
        return {"dist": "normal", "params": {"mean": self.m, "sd": self.s}}

    def sample(self, rng, n):
        return rng.normal(self.m, self.s, n)


_GL = {}


def _gl_graded(n_uniform=128, n_graded=48, n_nodes=16):
    """Composite Gauss-Legendre nodes / weights on (0, 1): uniform panels plus panels graded geometrically towards both
    ends (breakpoints 2^-k and 1 - 2^-k), for integrands with algebraic end singularities (Beta quantiles, sqrt kinks)."""
    key = (n_uniform, n_graded, n_nodes)
    if key not in _GL:
        br = sorted(set([0.5 ** k for k in range(2, n_graded)] + [1 - 0.5 ** k for k in range(2, n_graded)]
                        + list(np.linspace(0.0, 1.0, n_uniform + 1))))
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        xs, ws = [], []
        for lo, hi in zip(br[:-1], br[1:]):
            xs.append(lo + (hi - lo) * (x + 1) / 2)
            ws.append(w * (hi - lo) / 2)
        _GL[key] = (np.concatenate(xs), np.concatenate(ws))
    return _GL[key]


class DiffBetaD:
    """D = p_A - p_B, p_A ~ Beta, p_B ~ Beta independent. CDF by quadrature over the SHARPER variable's quantile u, split
    exactly where the other variable's CDF leaves (0, 1) (its kinks), graded Gauss-Legendre on the remaining interval:
      sharper = B:  P(D <= t) = (1 - F_B(1 - t)) + int_{F_B(-t)}^{F_B(1 - t)} F_A(Q_B(u) + t) du
      sharper = A:  P(D <= t) = F_A(t) + int_{F_A(t)}^{F_A(1 + t)} (1 - F_B(Q_A(u) - t)) du
    Accuracy ~1e-12 on smooth cases, <~1e-9 with Jeffreys-type end singularities (self_test compares both orderings)."""

    def __init__(self, A, B, order="auto"):
        self.A, self.B = A, B
        self.mean = A.mean - B.mean
        self.var = A.var + B.var
        self.support = (-1.0, 1.0)
        self.order = order if order != "auto" else ("B" if B.var <= A.var else "A")

    def cdf(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        s, w = _gl_graded()
        if self.order == "B":
            u0, u1 = self.B.cdf(-t), self.B.cdf(1 - t)
            u = u0[:, None] + (u1 - u0)[:, None] * s[None, :]
            g = self.A.cdf(self.B.ppf(u) + t[:, None])
            out = (1 - u1) + (u1 - u0) * (g * w[None, :]).sum(1)
        else:
            u0, u1 = self.A.cdf(t), self.A.cdf(1 + t)
            u = u0[:, None] + (u1 - u0)[:, None] * s[None, :]
            g = 1.0 - self.B.cdf(self.A.ppf(u) - t[:, None])
            out = u0 + (u1 - u0) * (g * w[None, :]).sum(1)
        return np.clip(out, 0.0, 1.0)

    def params(self):
        return {"dist": "beta_diff", "params": {"A": self.A.params(), "B": self.B.params()}}

    def sample(self, rng, n):
        return self.A.sample(rng, n) - self.B.sample(rng, n)


def bin_masses(dist, edges_disp, scale):
    """Exact mass per bin; mass outside the axis is folded into the end bins. Returns (masses, folded_low, folded_high)."""
    e = np.asarray(edges_disp, float) / scale
    c = np.asarray(dist.cdf(e), float)
    inner = np.concatenate([[0.0], c[1:-1], [1.0]])
    inner = np.maximum.accumulate(np.clip(inner, 0.0, 1.0))      # monotone up to quadrature rounding
    m = np.diff(inner)
    return m, float(c[0]), float(1.0 - c[-1])


def binned_moments(p, edges):
    mid = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    p = np.asarray(p, float) / max(float(np.sum(p)), 1e-300)
    mu = float((p * mid).sum())
    return mu, float(math.sqrt(max((p * (mid - mu) ** 2).sum(), 0.0)))


# ============================================================================ axes
def natural_axis(lo, hi, width, scale, unit):
    dec = decimals_of(width)
    edges = [float(fnum(lo + k * width, dec)) for k in range(K + 1)]
    assert abs(edges[-1] - hi) < 1e-9, (edges[-1], hi)
    return {"rule": "natural", "lo": edges[0], "hi": edges[-1], "K": K, "width": width, "decimals": dec, "scale": scale,
            "unit": unit, "fold_low": False, "fold_high": False}, edges


def declared_axis(rng, dist, scale, unit, support_lo=None):
    """A range around the answer's central 99% interval: random zoom and offset, nice width, folded tails. A draw that would
    fold more than MAX_FOLD of the answer's mass into the end bins is redrawn (zoom and offset; after 20 tries zoom 2,
    centred), so no question is mostly 'under X' / 'X or more'."""
    q0, q1 = (float(x) * scale for x in dist.ppf([0.005, 0.995]))
    span = max(q1 - q0, 1e-12)
    for tries in range(21):
        zoom, u = (logu(rng, *ZOOM), float(rng.uniform())) if tries < 20 else (2.0, 0.5)
        w = nice_ceil(zoom * span / K)
        dec = decimals_of(w)
        lo = math.floor((q0 - u * (K * w - span)) / w + 1e-9) * w
        if support_lo is not None and lo < support_lo * scale:
            lo = support_lo * scale
        edges = [float(fnum(lo + k * w, dec)) for k in range(K + 1)]
        c = dist.cdf(np.array([edges[0], edges[-1]]) / scale)
        if float(c[0]) + 1 - float(c[1]) <= MAX_FOLD:
            break
    fold_low = support_lo is None or edges[0] > support_lo * scale
    return {"rule": "declared", "lo": edges[0], "hi": edges[-1], "K": K, "width": w, "decimals": dec, "scale": scale,
            "unit": unit, "fold_low": bool(fold_low), "fold_high": True, "zoom_drawn": round(zoom, 4),
            "offset_u": round(u, 4), "axis_tries": tries + 1, "q005": q0, "q995": q1}, edges


def evidence_axis(rng, q0, q1, scale, unit, support_lo=None, zoom=ZOOM_V2, place=None, rule="evidence_v2"):
    """(v2) An axis around a DATA interval [q0, q1] (display units) that the caller derived from the visible evidence alone: random
    zoom (axis width / interval width, log-uniform `zoom`) and offset (the interval's position inside the axis is uniform), nice
    width, folded tails, never below `support_lo`. Two draws from `rng` (zoom, offset; `place` "end" adds a third, the side, and
    puts the interval within the outer 3% of the free room at that end). No target distribution is read, so nothing is redrawn:
    the folded masses are only measured (meta.exact.folded_low / folded_high)."""
    q0, q1 = float(q0), float(q1)
    span = max(q1 - q0, 1e-12)
    z = logu(rng, *zoom)
    u = float(rng.uniform())
    if place == "end":
        u = 0.03 * u if float(rng.uniform()) < 0.5 else 1.0 - 0.03 * u
    w = nice_ceil(z * span / K)
    dec = decimals_of(w)
    lo = math.floor((q0 - u * (K * w - span)) / w + 1e-9) * w
    if support_lo is not None and lo < support_lo * scale:
        lo = support_lo * scale
    edges = [float(fnum(lo + k * w, dec)) for k in range(K + 1)]
    fold_low = support_lo is None or edges[0] > support_lo * scale
    return {"rule": rule, "lo": edges[0], "hi": edges[-1], "K": K, "width": w, "decimals": dec, "scale": scale, "unit": unit,
            "fold_low": bool(fold_low), "fold_high": True, "zoom_drawn": z, "offset_u": u,
            "ev_lo": q0, "ev_hi": q1, "zoom_range": list(zoom), "place": place}, edges


def axis_from_record(meta):
    """(v2 check) Recompute a v2 record's axis from its VISIBLE data (meta.data), the stated noise sd, the context and the recorded
    zoom / offset draws alone -- no prior, no exact target. -> (axis, edges), or None for natural axes."""
    ax = meta["axis"]
    if ax["rule"] == "natural":
        return None
    fam = (meta.get("stress") or {}).get("base_family") or meta["family"]
    if ax["rule"] == "context_v2":
        return context_axis(fam, meta["context"], ax["scale"], ax["unit"])
    d = meta["data"]
    if fam == "rate":
        iv = data_interval("rate", tot=d["total"], T=d["exposure"])
        sup = 0.0
    else:
        kind = "normal_pred" if meta.get("quantity") == "predictive" else "normal_mean"
        iv = data_interval(kind, n=d["n"], xbar=d["xbar"], sigma=d["sigma"])
        sup = None
    sc = ax["scale"]

    class _Fixed:                                      # replays the recorded draws
        def __init__(self, vals):
            self.v = list(vals)

        def uniform(self, lo=0.0, hi=1.0):
            return self.v.pop(0)

    # logu(rng, lo, hi) = exp(uniform(log lo, log hi)): feed back log(zoom); the offset (and, for "end", the side) as recorded
    u = ax["offset_u"]
    vals = [math.log(ax["zoom_drawn"])]
    if ax.get("place") == "end":
        vals += [u / 0.03, 0.0] if u <= 0.03 else [(1.0 - u) / 0.03, 1.0]
    else:
        vals += [u]
    return evidence_axis(_Fixed(vals), iv[0] * sc, iv[1] * sc, sc, ax["unit"], support_lo=sup, zoom=tuple(ax["zoom_range"]),
                         place=ax.get("place"))


def data_interval(kind, n=0, h=0, tot=0, T=0.0, xbar=None, sigma=None, k=None):
    """(v2) The central 99% interval (natural units) of the normalised likelihood of the VISIBLE data -- the flat-prior posterior --
    or of the matching plug-in predictive spread; None without data. No prior enters.
      coin            Beta(h + 1, n - h + 1)
      rate            Gamma(tot + 1, T)
      normal_mean     xbar +/- Z99 sigma / sqrt(n)
      normal_pred     xbar +/- Z99 sqrt(sigma^2 / n + sigma^2)          (the next reading)
      normal_predmean xbar +/- Z99 sqrt(sigma^2 / n + sigma^2 / k)      (the mean of the next k)"""
    if kind == "coin":
        return None if not n else tuple(float(x) for x in BetaD(h + 1.0, n - h + 1.0).ppf([0.005, 0.995]))
    if kind == "rate":
        return None if not T or T <= 0 else tuple(float(x) for x in GammaD(tot + 1.0, float(T)).ppf([0.005, 0.995]))
    if not n:
        return None
    s = float(sigma) / math.sqrt(n)
    if kind == "normal_pred":
        s = math.hypot(s, float(sigma))
    elif kind == "normal_predmean":
        s = math.hypot(s, float(sigma) / math.sqrt(k))
    return float(xbar) - Z99 * s, float(xbar) + Z99 * s


def context_axis(family, ctx_id, scale, unit):
    """(v2) No data: a FIXED physical range per context, 0 .. RATE_CTX_RANGE (rate) or 0 .. CTX_RANGE[ctx] (normal) in natural units,
    times the display scale, in K nice-width bins. Reads neither the prior nor anything drawn for the item."""
    R = (RATE_CTX_RANGE if family == "rate" else CTX_RANGE[ctx_id]) * scale
    w = nice_ceil(R / K)
    dec = decimals_of(w)
    edges = [float(fnum(k * w, dec)) for k in range(K + 1)]
    return {"rule": "context_v2", "lo": edges[0], "hi": edges[-1], "K": K, "width": w, "decimals": dec, "scale": scale,
            "unit": unit, "fold_low": family != "rate", "fold_high": True, "ctx_range": R}, edges


def bin_labels(axis, edges):
    dec = axis["decimals"]
    neg = edges[0] < 0
    out = []
    for k in range(K):
        a, b = fnum(edges[k], dec), fnum(edges[k + 1], dec)
        if k == 0 and axis["fold_low"]:
            out.append(f"under {b}")
        elif k == K - 1 and axis["fold_high"]:
            out.append(f"{a} or more")
        else:
            out.append(f"{a} to {b}" if neg else f"{a}-{b}")
    assert len(set(out)) == K
    return out


def axis_sentence(axis, edges, unit_txt):
    dec = axis["decimals"]
    lo, hi, w = fnum(edges[0], dec), fnum(edges[-1], dec), fnum(axis["width"], dec)
    u = f" {unit_txt}" if unit_txt else ""
    s = f"The {K} options are equal-width bins of {w}{u} from {lo} to {hi}{u}."
    if axis["fold_low"] and axis["fold_high"]:
        s += f" Values below {lo} count in the first bin and values above {hi} in the last."
    elif axis["fold_high"]:
        s += f" Values above {hi} count in the last bin."
    elif axis["fold_low"]:
        s += f" Values below {lo} count in the first bin."
    return s


# ============================================================================ baselines (val / test only)
def _smooth_point(axis, edges, x_disp):
    if x_disp is None or not math.isfinite(x_disp):
        return np.full(K, 1.0 / K)
    j = int(np.clip(math.floor((x_disp - edges[0]) / axis["width"]), 0, K - 1))
    ker = np.exp(-0.5 * (np.arange(K) - j) ** 2)                  # discrete Gaussian, sd = 1 bin, truncated at the ends
    ker /= ker.sum()
    return 0.99 * ker + 0.01 / K


def _wald(axis, edges, mean_nat, se_nat):
    if mean_nat is None:
        return np.full(K, 1.0 / K)
    if not se_nat or se_nat <= 0:
        return _smooth_point(axis, edges, mean_nat * axis["scale"])
    m, _, _ = bin_masses(NormalD(mean_nat, se_nat), edges, axis["scale"])
    return 0.99 * m + 0.01 / K


def rnd(v):
    return [round(float(x), LABEL_DP) for x in v]


# ============================================================================ contexts and wording
COIN_CTX = [
    {"id": "coin", "intro": "A coin of unknown bias is flipped repeatedly; flips are independent given its bias.",
     "p": "the probability that a flip lands heads", "trial": ("flip", "flips"), "succ": "heads", "fail": "tails",
     "sym": ("H", "T")},
    {"id": "thumbtack", "intro": "A thumbtack is tossed onto a table again and again; tosses are independent given how it is shaped.",
     "p": "the probability that it lands point up", "trial": ("toss", "tosses"), "succ": "point up", "fail": "point down",
     "sym": ("U", "D")},
    {"id": "free_throw", "intro": "A basketball player shoots free throws; shots are independent given her success rate.",
     "p": "her probability of making a free throw", "trial": ("shot", "shots"), "succ": "made", "fail": "missed",
     "sym": ("M", "X")},
    {"id": "germination", "intro": "Seeds from a new batch are planted; each seed germinates independently with the same probability.",
     "p": "the germination probability", "trial": ("seed", "seeds"), "succ": "germinated", "fail": "did not",
     "sym": ("G", "N")},
    {"id": "defects", "intro": "Parts from a production line are inspected; each part is defective independently with the same probability.",
     "p": "the defect rate", "trial": ("part", "parts"), "succ": "defective", "fail": "fine", "sym": ("D", "F")},
    {"id": "email", "intro": "A marketing email goes out; each recipient opens it independently with the same probability.",
     "p": "the open rate", "trial": ("recipient", "recipients"), "succ": "opened it", "fail": "did not", "sym": ("O", "N")},
    {"id": "survey", "intro": "People drawn at random from a large population are asked whether they support a proposal.",
     "p": "the share of the population that supports it", "trial": ("respondent", "respondents"), "succ": "said yes",
     "fail": "said no", "sym": ("Y", "N")},
    {"id": "clinical", "intro": "Patients receive a new treatment; each responds independently with the same probability.",
     "p": "the response rate", "trial": ("patient", "patients"), "succ": "responded", "fail": "did not", "sym": ("R", "N")},
]

RATE_CTX = [
    {"id": "calls", "intro": "Calls reach a help desk as a Poisson process with an unknown rate", "ev": ("call", "calls"),
     "unit": "calls per hour", "per": "hour", "iv": [("one-hour periods", 1.0), ("30-minute periods", 0.5),
                                                     ("two-hour periods", 2.0), ("15-minute periods", 0.25)], "base": ("hour", "hours")},
    {"id": "typos", "intro": "Typos occur in a long manuscript as a Poisson process with an unknown rate", "ev": ("typo", "typos"),
     "unit": "typos per page", "per": "page", "iv": [("single pages", 1.0), ("five-page sections", 5.0),
                                                     ("ten-page chapters", 10.0)], "base": ("page", "pages")},
    {"id": "meteors", "intro": "Meteors are seen during a shower as a Poisson process with an unknown rate", "ev": ("meteor", "meteors"),
     "unit": "meteors per hour", "per": "hour", "iv": [("one-hour watches", 1.0), ("30-minute watches", 0.5),
                                                       ("15-minute watches", 0.25)], "base": ("hour", "hours")},
    {"id": "errors", "intro": "A server logs errors as a Poisson process with an unknown rate", "ev": ("error", "errors"),
     "unit": "errors per day", "per": "day", "iv": [("days", 1.0), ("weeks", 7.0), ("12-hour shifts", 0.5)], "base": ("day", "days")},
    {"id": "potholes", "intro": "Potholes along a highway occur as a Poisson process with an unknown rate", "ev": ("pothole", "potholes"),
     "unit": "potholes per km", "per": "km", "iv": [("1-km stretches", 1.0), ("500-m stretches", 0.5), ("2-km stretches", 2.0)],
     "base": ("km", "km")},
    {"id": "decays", "intro": "A detector counts decays from a radioactive sample as a Poisson process with an unknown rate",
     "ev": ("count", "counts"), "unit": "counts per second", "per": "second",
     "iv": [("one-second windows", 1.0), ("ten-second windows", 10.0), ("0.1-second windows", 0.1)], "base": ("second", "seconds")},
    {"id": "arrivals", "intro": "Customers arrive at a shop as a Poisson process with an unknown rate", "ev": ("customer", "customers"),
     "unit": "customers per minute", "per": "minute", "iv": [("one-minute intervals", 1.0), ("five-minute intervals", 5.0)],
     "base": ("minute", "minutes")},
    {"id": "emails", "intro": "Support emails arrive as a Poisson process with an unknown rate", "ev": ("email", "emails"),
     "unit": "emails per day", "per": "day", "iv": [("days", 1.0), ("weeks", 7.0)], "base": ("day", "days")},
]

# typical magnitude S, unit, optional alternative display unit (name, factor: display = natural x factor)
NORM_CTX = [
    {"id": "mass", "S": (20, 2000), "unit": "g", "alt": ("kg", 0.001),
     "intro": "An object of unknown true mass {mu} is weighed several times on a lab scale.", "what": "the object's true mass",
     "reading": ("weighing", "weighings"), "next": "One more weighing is about to be made."},
    {"id": "length", "S": (10, 800), "unit": "mm", "alt": ("cm", 0.1),
     "intro": "A metal rod of unknown true length {mu} is measured several times with a caliper.", "what": "the rod's true length",
     "reading": ("measurement", "measurements"), "next": "The rod is about to be measured once more."},
    {"id": "temperature", "S": (5, 200), "unit": "degrees C", "alt": None,
     "intro": "A sensor reads the temperature of a sealed chamber held at an unknown constant temperature {mu}.",
     "what": "the chamber's true temperature", "reading": ("reading", "readings"), "next": "The sensor is about to take one more reading."},
    {"id": "reaction", "S": (150, 900), "unit": "ms", "alt": ("s", 0.001),
     "intro": "A participant's reaction time is measured over repeated trials; their true mean reaction time is {mu}.",
     "what": "the participant's true mean reaction time", "reading": ("trial", "trials"), "next": "One more trial is about to be run."},
    {"id": "voltage", "S": (0.5, 15), "unit": "V", "alt": ("mV", 1000.0),
     "intro": "The voltage {mu} of a reference cell is measured several times with a noisy meter.", "what": "the cell's true voltage",
     "reading": ("reading", "readings"), "next": "The meter is about to take one more reading."},
    {"id": "nitrate", "S": (0.5, 60), "unit": "mg/L", "alt": None,
     "intro": "A water sample's nitrate concentration {mu} is assayed several times.", "what": "the sample's true nitrate concentration",
     "reading": ("assay", "assays"), "next": "One more assay is about to be run."},
    {"id": "commute", "S": (10, 120), "unit": "minutes", "alt": None,
     "intro": "Your commute on a fixed route takes {mu} minutes on average; days vary independently around that mean.",
     "what": "the true mean commute time", "reading": ("trip", "trips"), "next": "Tomorrow's trip is next."},
]
CTX_RANGE = {c["id"]: 1.6 * c["S"][1] for c in NORM_CTX}   # v2 context axis (no data), natural units: prior means lie in S x [0.6, 1.6]

AB_CTX = [
    {"id": "web", "intro": "Two versions of a web page are shown to random visitors.", "rate": "conversion rates",
     "data": "Version A: {hA} conversions from {nA} visitors. Version B: {hB} conversions from {nB} visitors."},
    {"id": "drug", "intro": "Patients are randomised to a new drug (A) or the standard drug (B).", "rate": "response rates",
     "data": "Drug A: {hA} of {nA} patients responded. Drug B: {hB} of {nB} patients responded."},
    {"id": "shooters", "intro": "Two basketball players, A and B, shoot free throws.", "rate": "probabilities of making a shot",
     "data": "Player A made {hA} of {nA} shots. Player B made {hB} of {nB} shots."},
    {"id": "coins", "intro": "Two coins of unknown bias, A and B, are flipped.", "rate": "probabilities of heads",
     "data": "Coin A: {hA} heads in {nA} flips. Coin B: {hB} heads in {nB} flips."},
    {"id": "ads", "intro": "Two ad designs, A and B, are shown to random users.", "rate": "click-through rates",
     "data": "Design A: {hA} clicks from {nA} views. Design B: {hB} clicks from {nB} views."},
]


def _pl(n, pair):
    return pair[0] if n == 1 else pair[1]


def beta_prior_draw(rng, py, strength=(2, 200)):
    """(a, b) as stated in the text: uniform 25%, Jeffreys 8%, else mean x strength with integer (70%) or 1-dp values."""
    u = rng.uniform()
    if u < 0.25:
        return 1.0, 1.0
    if u < 0.33:
        return 0.5, 0.5
    mean = round(float(rng.uniform(0.05, 0.95)) / 0.05) * 0.05
    s = logu(rng, *strength)
    if rng.uniform() < 0.7:
        a, b = max(1, round(mean * s)), max(1, round((1 - mean) * s))
    else:
        a, b = max(0.5, round(mean * s, 1)), max(0.5, round((1 - mean) * s, 1))
    return float(a), float(b)


def beta_prior_text(py, a, b, sym="p"):
    if a == 1 and b == 1 and py.random() < 0.6:
        return py.choice([f"Prior: {sym} is uniform on [0, 1] (Beta(1, 1)).",
                          f"Before any data, every value of {sym} between 0 and 1 is equally likely (a Beta(1, 1) prior)."])
    A, B = fparam(a), fparam(b)
    return py.choice([f"Prior: {sym} ~ Beta({A}, {B}).",
                      f"Before any data, your belief about {sym} is a Beta({A}, {B}) distribution.",
                      f"Prior: a Beta distribution with a = {A} and b = {B} (prior mean {fsig(a / (a + b), 3)}).",
                      f"Prior for {sym}: Beta({A}, {B})."])


def coin_outcome(ctx, h, f):
    return f"{h} {ctx['succ']} and {f} {ctx['fail']}"


def coin_question(py, axis, edges, pct):
    sym = "p (in percent)" if pct else "p"
    lead = py.choice([f"Which bin contains {sym}?", f"In which interval does {sym} lie?",
                      f"Given the data, which bin holds the true value of {sym}?"])
    return lead + " " + axis_sentence(axis, edges, "percentage points" if pct else "")


def coin_axis(pct):
    return natural_axis(0.0, 100.0, 2.0, 100.0, "percent") if pct else natural_axis(0.0, 1.0, 0.02, 1.0, "")


# ============================================================================ family generators
# each returns a dict: state, instructions, axis, edges, exact (dist), prior_dist, mle (mean, se) natural, truth, fp, extra meta
def _v2_override(v2):
    """(v2, tests) the stated prior that replaces the drawn one AFTER truth and data are drawn (the data stay fixed), else None."""
    return None if v2 is None else v2.get("override")


def gen_coin(rng, py, fmt="counts", seq_len=None, fixed=None, v2=None):
    """v2 (None = the v1 path, unchanged): {"pp": random.Random for the prior sentence, "override": {a, b} or None}. v2 states
    and uses the same (printed) prior parameters."""
    ctx = py.choice(COIN_CTX) if fixed is None else fixed["ctx"]
    if fixed is None:
        a, b = beta_prior_draw(rng, py)
        if v2 is not None:
            a, b = float(fparam(a)), float(fparam(b))          # (v2) the label uses exactly the printed prior
    else:
        a, b = fixed["a"], fixed["b"]
    prior = BetaD(a, b)
    broad = fixed is None and rng.uniform() < P_BROAD
    p_true = float(rng.uniform()) if broad else float(prior.sample(rng, 1)[0])
    if fixed is not None:
        n, h, p_true, broad = fixed["n"], fixed["h"], None, False
        seq = None
    elif fmt == "sequence":
        n = seq_len
        seq = (rng.uniform(size=n) < p_true).astype(int)
        h = int(seq.sum())
    else:
        u = rng.uniform()
        n = 0 if u < 0.05 else int(round(logu(rng, 1, 500)))
        h = int(rng.binomial(n, p_true)) if n else 0
        seq = None
    ov = _v2_override(v2)
    if ov is not None:
        a, b = float(ov["a"]), float(ov["b"])
        prior = BetaD(a, b)
    f = n - h
    post = BetaD(a + h, b + f)
    pct = fixed is None and rng.uniform() < 0.25
    axis, edges = coin_axis(pct)
    intro = ctx["intro"] + f" Let p be {ctx['p']}."
    pri = beta_prior_text(py if v2 is None else v2["pp"], a, b)
    T1, Tn = ctx["trial"]
    if fmt == "sequence":
        s1, s0 = ctx["sym"]
        legend = f"{s1} = {ctx['succ']}, {s0} = {ctx['fail']}"
        seq_txt = " ".join(s1 if x else s0 for x in seq)
        data = py.choice([f"The {n} {_pl(n, ctx['trial'])} so far, in order ({legend}): {seq_txt}",
                          f"Outcomes of {n} {_pl(n, ctx['trial'])}, in order ({legend}): {seq_txt}"])
    elif n == 0:
        data = f"No {Tn} have been observed yet."
    else:
        data = py.choice([f"Data: {n} {_pl(n, ctx['trial'])}: {coin_outcome(ctx, h, f)}.",
                          f"Observed so far: {coin_outcome(ctx, h, f)} ({n} {_pl(n, ctx['trial'])} in total).",
                          f"Results of {n} {_pl(n, ctx['trial'])}: {coin_outcome(ctx, h, f)}."])
    state = "\n".join([intro, pri, data])
    mle = (h / n, math.sqrt(h * f / n ** 3) if n else None) if n else (None, None)
    return {"family": "coin_seq" if fmt == "sequence" else "coin", "context": ctx["id"], "format": fmt, "state": state,
            "instructions": coin_question(py, axis, edges, pct), "axis": axis, "edges": edges, "exact": post,
            "prior": prior, "mle": mle, "quantity": "posterior",
            "truth": None if p_true is None else {"value": p_true * axis["scale"],
                                                  "source": "other_prior" if ov else ("broad" if broad else "prior")},
            "fp": f"coin|{fparam(a)}|{fparam(b)}|{n}|{h}",
            "data": {"n": n, "h": h, "seq_len": seq_len} if fmt == "sequence" else {"n": n, "h": h},
            "prior_params": {"a": a, "b": b}, "prior_line": pri}


def gen_rate(rng, py, v2=None):
    """v2 (None = the v1 path, unchanged): {"arng": the axis stream, "pp": the prior-sentence stream, "override": {shape, rate}
    or None, "stress": None or a STRESS key}: the axis comes from the counts alone (data_interval / evidence_axis)."""
    ctx = py.choice(RATE_CTX)
    shape = float(py.choice([0.5, 1, 1.5, 2, 3, 4, 5, 8, 10, 20]))
    mu0 = float(fsig(logu(rng, 0.05, 50), 2))
    rate = float(fsig(shape / mu0, 3))
    prior = GammaD(shape, rate)
    broad = rng.uniform() < P_BROAD
    lam = prior.mean * logu(rng, 1 / 8, 8) if broad else float(prior.sample(rng, 1)[0])
    iv_name, L = py.choice(ctx["iv"])
    m = int(round(logu(rng, 1, 30)))
    counts = rng.poisson(lam * L, m).astype(int)
    tot, T = int(counts.sum()), round(m * L, 10)
    ov = _v2_override(v2)
    if ov is not None:
        shape, rate = float(ov["shape"]), float(ov["rate"])
        prior = GammaD(shape, rate)
    post = GammaD(shape + tot, rate + T)
    if v2 is None:
        axis, edges = declared_axis(rng, post, 1.0, ctx["unit"], support_lo=0.0)
    else:
        st = STRESS.get(v2.get("stress")) or {}
        q0, q1 = data_interval("rate", tot=tot, T=T)
        axis, edges = evidence_axis(v2["arng"], q0, q1, 1.0, ctx["unit"], support_lo=0.0, zoom=st.get("zoom", ZOOM_V2),
                                    place=st.get("place"))
    unit = ctx["unit"]
    intro = f"{ctx['intro']} λ ({unit})."
    S, R = fparam(shape), fsig(rate, 3)
    pri = (py if v2 is None else v2["pp"]).choice([
                     f"Prior: λ ~ Gamma(shape {S}, rate {R}), so the prior mean is {fsig(shape / rate, 3)} {unit}.",
                     f"Prior: a Gamma distribution for λ with shape α = {S} and rate β = {R} (mean α/β = "
                     f"{fsig(shape / rate, 3)} {unit}).",
                     f"Before any data, λ ~ Gamma(α = {S}, β = {R}) in {unit}."])
    Ttxt = fsig(T, 4)
    Tu = ctx["base"][0] if T == 1 else ctx["base"][1]
    ev = _pl(tot, ctx["ev"])
    if 1 < m <= 12 and py.random() < 0.6:
        data = f"Counts in {m} consecutive {iv_name}: {', '.join(map(str, counts))}."
        fmt = "list"
    elif m == 1:
        data = f"Observed: {tot} {ev} in {Ttxt} {Tu}."
        fmt = "summary"
    else:
        data = py.choice([f"In total {tot} {ev} were recorded over {m} consecutive {iv_name} ({Ttxt} {Tu} in all).",
                          f"Observed: {tot} {ev} in {Ttxt} {Tu}."])
        fmt = "summary"
    q = py.choice([f"Which bin contains λ, in {unit}?", f"In which interval does the rate λ ({unit}) lie?",
                   f"Given the data, which bin holds the true rate λ ({unit})?"])
    return {"family": "rate", "context": ctx["id"], "format": fmt, "state": "\n".join([intro, pri, data]),
            "instructions": q + " " + axis_sentence(axis, edges, unit), "axis": axis, "edges": edges, "exact": post,
            "prior": prior, "mle": (tot / T, math.sqrt(tot) / T), "quantity": "posterior",
            "truth": {"value": lam, "source": "other_prior" if ov else ("broad" if broad else "prior")},
            "fp": f"rate|{S}|{R}|{tot}|{fsig(T, 6)}", "data": {"m": m, "interval": L, "exposure": T, "total": tot},
            "prior_params": {"shape": shape, "rate": rate}, "prior_line": pri}


def gen_normal(rng, py, predictive=False, v2=None):
    """v2 (None = the v1 path, unchanged): as gen_rate's ({"override": {mean, sd}}); the axis comes from the readings and the
    stated noise sd alone (data_interval), or the context's fixed range without readings (context_axis)."""
    ctx = py.choice(NORM_CTX)
    S = logu(rng, *ctx["S"])
    s0 = float(fsig(S * logu(rng, 0.01, 0.3), 2))
    sig = float(fsig(S * logu(rng, 0.002, 0.2), 2))
    r0 = pow10_floor(s0 / 3)
    m0 = round(round(S * rng.uniform(0.6, 1.6) / r0) * r0, dec_for_res(r0))
    prior = NormalD(m0, s0)
    broad = rng.uniform() < P_BROAD
    mu = m0 + s0 * float(rng.uniform(-5, 5)) if broad else float(prior.sample(rng, 1)[0])
    u = rng.uniform()
    n = 0 if (u < 0.03 and not predictive) else int(round(logu(rng, 1, 40)))
    r = pow10_floor(sig / 3)
    dr = dec_for_res(r)
    xs = [round(round(x / r) * r, dr) for x in rng.normal(mu, sig, n)]
    unit = ctx["unit"]
    if n and n <= 12 and py.random() < 0.7:
        vals = [fnum(x, dr) for x in xs]
        xbar, fmt = sum(float(v) for v in vals) / n, "list"
        data = f"The {n} {_pl(n, ctx['reading'])} ({unit}): {', '.join(vals)}." if n > 1 else \
            f"One {ctx['reading'][0]} so far: {vals[0]} {unit}."
    elif n:
        rb = r / 10
        xbar_t = fnum(round(np.mean(xs) / rb) * rb, dec_for_res(rb))
        xbar, fmt = float(xbar_t), "summary"
        data = f"{n} {_pl(n, ctx['reading'])} were taken; their mean is {xbar_t} {unit}."
    else:
        xbar, fmt = None, "none"
        data = f"No {ctx['reading'][1]} have been taken yet."
    ov = _v2_override(v2)
    if ov is not None:
        m0, s0 = float(ov["mean"]), float(ov["sd"])
        r0 = pow10_floor(s0 / 3)
        prior = NormalD(m0, s0)
    prec = 1 / s0 ** 2 + (n / sig ** 2 if n else 0.0)
    mn = (m0 / s0 ** 2 + (n * xbar / sig ** 2 if n else 0.0)) / prec
    post = NormalD(mn, math.sqrt(1 / prec))
    alt = ctx["alt"] if ctx["alt"] and rng.uniform() < 0.2 else None
    dunit, scale = (alt[0], alt[1]) if alt else (unit, 1.0)
    noise = py.choice([f"Each {ctx['reading'][0]} equals {{mu}} plus independent normal error with standard deviation {fsig(sig, 2)} {unit}.",
                       f"{ctx['reading'][1].capitalize()} scatter around {{mu}} with independent normal noise, sd {fsig(sig, 2)} {unit}."])
    pri = (py if v2 is None else v2["pp"]).choice([
                     f"Prior: μ ~ Normal(mean {fnum(m0, dec_for_res(r0))}, sd {fsig(s0, 2)}) in {unit}.",
                     f"Before any data, your belief about μ is normal with mean {fnum(m0, dec_for_res(r0))} {unit} and "
                     f"standard deviation {fsig(s0, 2)} {unit}."])
    intro = ctx["intro"].format(mu="μ") + " " + noise.format(mu="μ")
    if predictive:
        target = NormalD(post.m, math.sqrt(post.s ** 2 + sig ** 2))
        prior_t = NormalD(m0, math.sqrt(s0 ** 2 + sig ** 2))
        x_next = float(rng.normal(mu, sig))
        q = py.choice([f"{ctx['next']} Which bin will its value fall in, in {dunit}?",
                       f"{ctx['next']} In which interval will the new value lie ({dunit})?"])
        truth = {"value": x_next * scale, "source": "other_prior" if ov else ("broad" if broad else "prior"), "mu_true": mu * scale}
        mle = (xbar, sig) if n else (None, None)
    else:
        target, prior_t = post, prior
        q = py.choice([f"Which bin contains μ, {ctx['what']}, in {dunit}?",
                       f"In which interval does μ ({ctx['what']}) lie, in {dunit}?",
                       f"Given the data, which bin holds μ, in {dunit}?"])
        truth = {"value": mu * scale, "source": "other_prior" if ov else ("broad" if broad else "prior")}
        mle = (xbar, sig / math.sqrt(n)) if n else (None, None)
    if v2 is None:
        axis, edges = declared_axis(rng, target, scale, dunit)
    else:
        iv = data_interval("normal_pred" if predictive else "normal_mean", n=n, xbar=xbar, sigma=sig)
        if iv is None:
            axis, edges = context_axis("normal_mean", ctx["id"], scale, dunit)
        else:
            st = STRESS.get(v2.get("stress")) or {}
            axis, edges = evidence_axis(v2["arng"], iv[0] * scale, iv[1] * scale, scale, dunit, zoom=st.get("zoom", ZOOM_V2),
                                        place=st.get("place"))
    return {"family": "normal_pred" if predictive else "normal_mean", "context": ctx["id"], "format": fmt,
            "state": "\n".join([intro, pri, data]), "instructions": q + " " + axis_sentence(axis, edges, dunit),
            "axis": axis, "edges": edges, "exact": target, "prior": prior_t, "mle": mle,
            "quantity": "predictive" if predictive else "posterior", "truth": truth,
            "fp": f"normal|{m0}|{s0}|{sig}|{n}|{xbar}", "data": {"n": n, "xbar": xbar, "sigma": sig, "unit_data": unit},
            "prior_params": {"mean": m0, "sd": s0, "sigma": sig}, "posterior_mu": post.params(), "prior_line": pri}


def gen_ab(rng, py, v2=None):
    """v2 (None = the v1 path): the printed prior parameters are the ones used; the prior sentence has its own stream."""
    ctx = py.choice(AB_CTX)
    aA, bA = beta_prior_draw(rng, py, strength=(2, 60))
    aB, bB = (aA, bA) if rng.uniform() < 0.7 else beta_prior_draw(rng, py, strength=(2, 60))
    if v2 is not None:
        aA, bA, aB, bB = (float(fparam(x)) for x in (aA, bA, aB, bB))
    PA, PB = BetaD(aA, bA), BetaD(aB, bB)
    broad = rng.uniform() < P_BROAD
    pA, pB = (float(rng.uniform()), float(rng.uniform())) if broad else (float(PA.sample(rng, 1)[0]), float(PB.sample(rng, 1)[0]))
    nA, nB = int(round(logu(rng, 3, 400))), int(round(logu(rng, 3, 400)))
    hA, hB = int(rng.binomial(nA, pA)), int(rng.binomial(nB, pB))
    post = DiffBetaD(BetaD(aA + hA, bA + nA - hA), BetaD(aB + hB, bB + nB - hB))
    pct = rng.uniform() < 0.25
    axis, edges = natural_axis(-100.0, 100.0, 4.0, 100.0, "percentage points") if pct else natural_axis(-1.0, 1.0, 0.04, 1.0, "")
    if (aA, bA) == (aB, bB):
        pri = (py if v2 is None else v2["pp"]).choice([f"Prior: p_A and p_B are independent, each Beta({fparam(aA)}, {fparam(bA)}).",
                         f"Before any data, p_A ~ Beta({fparam(aA)}, {fparam(bA)}) and p_B ~ Beta({fparam(aB)}, {fparam(bB)}), "
                         "independent."])
    else:
        pri = f"Prior: p_A ~ Beta({fparam(aA)}, {fparam(bA)}) and p_B ~ Beta({fparam(aB)}, {fparam(bB)}), independent."
    intro = ctx["intro"] + f" Let p_A and p_B be their {ctx['rate']}; outcomes are independent given these rates."
    data = ctx["data"].format(hA=hA, nA=nA, hB=hB, nB=nB)
    sym = "d (in percentage points)" if pct else "d"
    q = f"Let d = p_A - p_B. " + py.choice([f"Which bin contains {sym}?", f"In which interval does {sym} lie?"]) + " " + \
        axis_sentence(axis, edges, "percentage points" if pct else "")
    mA, mB = hA / nA, hB / nB
    se = math.sqrt(mA * (1 - mA) / nA + mB * (1 - mB) / nB)
    return {"family": "ab_diff", "context": ctx["id"], "format": "counts", "state": "\n".join([intro, pri, data]),
            "instructions": q, "axis": axis, "edges": edges, "exact": post, "prior": DiffBetaD(PA, PB), "mle": (mA - mB, se),
            "quantity": "posterior", "truth": {"value": (pA - pB) * axis["scale"], "source": "broad" if broad else "prior"},
            "fp": f"ab|{aA}|{bA}|{aB}|{bB}|{nA}|{hA}|{nB}|{hB}",
            "data": {"nA": nA, "hA": hA, "nB": nB, "hB": hB}, "prior_params": {"A": [aA, bA], "B": [aB, bB]},
            "prior_line": pri}


PROBE_CANON = ("A coin is flipped 10 times and comes up heads 6 times and tails 4 times. Before seeing the flips, the belief about "
               "the coin's fairness, its probability of heads p, is Beta(10, 10). Flips are independent given p.")
PROBE_CANON_EV = ("A coin is flipped 10 times and comes up heads 6 times and tails 4 times. Flips are independent given p, the coin's "
                  "probability of heads. A prior over p is not stated here; it will be applied separately.")


def gen_probe(variant, axes="v1"):
    """variant 0 = the canonical (pre-registered) wording; 1..3 = template renderings of the same numbers."""
    py = random.Random(seed_of(NS, "probe", variant))
    rng = np.random.default_rng(seed_of(NS, "probe", variant))
    fixed = {"ctx": COIN_CTX[0], **PROBE}
    v2 = None if axes == "v1" else {"pp": random.Random(seed_of(NS, "probe", variant, "prior_text_v2")), "override": None}
    g = gen_coin(rng, py, fixed=fixed, v2=v2)
    if v2 is not None:
        g["axes"], g["ev_key"] = "v2", f"{NS}|test|probe|{variant}"
    if variant == 0:
        g["state"] = PROBE_CANON
        g["instructions"] = "Which bin contains p? " + axis_sentence(g["axis"], g["edges"], "")
        g["prior_line"] = None                        # (p101) the prior sits inside the canonical paragraph
    g["family"], g["context"] = "probe", "coin"
    return g


# ============================================================================ (p101) prompt variants and exact log prior masses
# research/p101_likelihood_ratio_head.md. The PRIOR-OFFSET readout q_i = softmax_i(log pi_i + l_i(D)) needs, per question, the stated
# prior's log mass in every bin (meta.log_prior_bins, level order, folded end bins as the labels fold them), and an EVIDENCE-ONLY
# rendering of the same item: the prompt states the model and the data, and the prior sentence is replaced by one saying a prior
# will be applied separately. Items, axes, labels (the exact posterior under the stated prior) and ids are those of the default
# rendering: the generators draw exactly as before and the record id is computed from the default text. Defaults (variant
# "prior_text", no log prior) write byte-identical records.
PROMPT_VARIANTS = ("prior_text", "evidence_only")
# Two offset conventions for a FOLDED end bin (both written; the trainer / scorer pick one by meta key): log_prior_bins = the prior's
# mass over the whole folded tail (exact with no data: l = 0 gives the prior), log_prior_bins_nominal = its mass over the bin's
# nominal interval only (with data, the posterior's tail mass sits next to the edge, so the tail's evidence is then nearly
# prior-free: build_numeric_evidence.py floor measures both). Interior bins and unfolded axes: identical.
LOGP_DP = 8                         # meta.log_prior_bins decimals
PARAM_FAMILIES = ("coin", "coin_seq", "probe", "rate", "normal_mean")   # the target IS the parameter the prior is stated on
EV_SYM = {"coin": "p", "coin_seq": "p", "probe": "p", "rate": "λ", "normal_mean": "μ", "normal_pred": "μ", "ab_diff": "p_A and p_B"}
EV_TEMPLATES = ("A prior for {s} is not stated here; it will be applied separately.",
                "No prior for {s} is given in this question: a prior will be applied separately.",
                "Prior: not stated here (a prior over {s} will be applied separately).")
EV_TEMPLATES_AB = ("Priors for p_A and p_B are not stated here; they will be applied separately.",
                   "No priors for p_A and p_B are given in this question: priors will be applied separately.",
                   "Priors: not stated here (priors over p_A and p_B will be applied separately).")
OFFSET_NOTE = {"normal_pred": "the target is the next reading, not the parameter the prior is on: softmax(log prior-predictive + l) "
                              "is exact only if l depends on the prior (p(D | y) integrates mu under the prior given y)",
               "ab_diff": "the target d = p_A - p_B is a function of two parameters: p(D | d) integrates the nuisance under the "
                          "prior given d, so a prior-free l over d-bins is an approximation"}
_LOG2 = math.log(2.0)
_TINY = 1e-280                      # below this a CDF / survival value is recomputed in log space (Beta / Gamma tails)


def evidence_line(family, key):
    """The sentence that replaces the prior in an evidence-only prompt; `key` (any string) picks the template deterministically."""
    t = EV_TEMPLATES_AB if family == "ab_diff" else EV_TEMPLATES
    return t[seed_of("p101-ev", key) % len(t)].format(s=EV_SYM.get(family, "the unknown quantity"))


def evidence_state(state, prior_line, family, key):
    """The state with its prior sentence (one whole line, `prior_line`) replaced by evidence_line(...). ValueError if the line is not
    there exactly once."""
    lines = state.split("\n")
    hits = [i for i, x in enumerate(lines) if x == prior_line]
    if prior_line is None or len(hits) != 1:
        raise ValueError(f"the prior sentence is not one whole line of the state: {prior_line!r}")
    lines[hits[0]] = evidence_line(family, key)
    return "\n".join(lines)


def _log1mexp(d):
    """log(1 - exp(-d)) for d >= 0 (Maechler 2012); -inf at d = 0."""
    d = np.asarray(d, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d < _LOG2, np.log(-np.expm1(-d)), np.log1p(-np.exp(-d)))


def _logdiff(lo, hi):
    """log(exp(hi) - exp(lo)) for hi >= lo (both logs; lo may be -inf)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    with np.errstate(invalid="ignore"):
        d = np.where(np.isneginf(lo), np.inf, hi - lo)
    return np.where(np.isneginf(hi), -np.inf, hi + _log1mexp(np.maximum(d, 0.0)))


def _log_beta_lower(a, b, x, terms=20000):
    """log I_x(a, b) by the series I_x = x^a (1-x)^b / (a B(a, b)) 2F1(a + b, 1; a + 1; x), summed in log space; for the deep LOWER
    tail (x below the mean), where the terms fall geometrically."""
    x = np.asarray(x, float)
    out = np.full(x.shape, -np.inf)
    for i, xi in np.ndenumerate(x):
        if xi <= 0:
            continue
        t, s, n = 1.0, 1.0, 0
        while n < terms:
            t *= (a + b + n) * xi / (a + 1 + n)
            s += t
            n += 1
            if t < 1e-17 * s:
                break
        out[i] = a * math.log(xi) + b * math.log1p(-xi) - math.log(a) - float(sp.betaln(a, b)) + math.log(s)
    return out


def _log_gamma_lower(k, y, terms=20000):
    """log P(k, y) by the series P = y^k e^-y / Gamma(k + 1) sum_n y^n / ((k+1)..(k+n)), in log space (deep lower tail)."""
    y = np.asarray(y, float)
    out = np.full(y.shape, -np.inf)
    for i, yi in np.ndenumerate(y):
        if yi <= 0:
            continue
        t, s, n = 1.0, 1.0, 0
        while n < terms:
            n += 1
            t *= yi / (k + n)
            s += t
            if t < 1e-17 * s:
                break
        out[i] = k * math.log(yi) - yi - float(sp.gammaln(k + 1)) + math.log(s)
    return out


def _log_gamma_upper(k, y, iters=2000):
    """log Q(k, y) by the continued fraction (modified Lentz, Numerical Recipes gcf), for y > k + 1 (deep UPPER tail)."""
    y = np.asarray(y, float)
    out = np.full(y.shape, -np.inf)
    for i, yi in np.ndenumerate(y):
        if not np.isfinite(yi):
            continue
        b = yi + 1.0 - k
        c, d = 1e300, 1.0 / b
        h = d
        for n in range(1, iters):
            an = -n * (n - k)
            b += 2.0
            d = an * d + b
            d = 1e-300 if abs(d) < 1e-300 else d
            c = b + an / c
            c = 1e-300 if abs(c) < 1e-300 else c
            d = 1.0 / d
            h *= d * c
            if abs(d * c - 1.0) < 1e-16:
                break
        out[i] = -yi + k * math.log(yi) - float(sp.gammaln(k)) + math.log(h)
    return out


def log_cdf_sf(dist, x):
    """(log F(x), log S(x)) of a distribution at natural-unit points x, accurate in both tails: Normal by log_ndtr; Beta / Gamma by
    scipy where the value is above 1e-280, else the log-space series / continued fraction above; the A/B difference by its
    quadrature CDF (the survival function from the mirrored difference), floored at 1e-300 (its tails are not exact)."""
    x = np.asarray(x, float)
    kind = type(dist).__name__                            # by name: the module may be loaded twice (tests, a tarball's tools/)
    if kind == "NormalD":
        z = (x - dist.m) / dist.s
        return sp.log_ndtr(z), sp.log_ndtr(-z)
    with np.errstate(divide="ignore"):
        if kind == "BetaD":
            xc = np.clip(x, 0.0, 1.0)
            F, S = sp.betainc(dist.a, dist.b, xc), sp.betainc(dist.b, dist.a, 1.0 - xc)
            lF, lS = np.log(F), np.log(S)
            lo, hi = (F < _TINY) & (xc > 0), (S < _TINY) & (xc < 1)
            if lo.any():
                lF[lo] = _log_beta_lower(dist.a, dist.b, xc[lo])
            if hi.any():
                lS[hi] = _log_beta_lower(dist.b, dist.a, 1.0 - xc[hi])
            return lF, lS
        if kind == "GammaD":
            y = dist.r * np.maximum(x, 0.0)
            F, S = sp.gammainc(dist.k, y), sp.gammaincc(dist.k, y)
            lF, lS = np.log(F), np.log(S)
            lo, hi = (F < _TINY) & (y > 0), (S < _TINY) & np.isfinite(y)
            if lo.any():
                lF[lo] = _log_gamma_lower(dist.k, y[lo])
            if hi.any():
                big = hi & (y > dist.k + 1)
                lS[big] = _log_gamma_upper(dist.k, y[big])
            return lF, lS
        if kind == "DiffBetaD":
            F = np.asarray(dist.cdf(x), float)
            S = np.ones_like(F)
            up = F > 0.5
            if up.any():
                S[up] = type(dist)(dist.B, dist.A, dist.order).cdf(-x[up])
            S[~up] = 1.0 - F[~up]
            return np.log(np.maximum(F, 1e-300)), np.log(np.maximum(S, 1e-300))
    raise TypeError(f"log_cdf_sf: {type(dist).__name__}")


def log_bin_masses(dist, edges_disp, scale, fold=True):
    """log of bin_masses(dist, edges, scale) -- the same bins, mass outside the axis folded into the end bins -- computed in log space
    (each interior bin from whichever tail is smaller), so a bin with mass 1e-300 still has its exact log. [K] float64.
    fold=False: every bin's NOMINAL interval [e_k, e_k+1] instead (the end bins without the folded tails; equal on unfolded axes)."""
    e = np.asarray(edges_disp, float) / scale
    if not fold:
        lF, lS = log_cdf_sf(dist, e)
        lower = lF[1:] <= math.log(0.5)
        return np.where(lower, _logdiff(lF[:-1], lF[1:]), _logdiff(lS[1:], lS[:-1]))
    lF, lS = log_cdf_sf(dist, e[1:-1])                    # interior edges e_1 .. e_{K-1}
    out = np.empty(len(e) - 1)
    out[0], out[-1] = lF[0], lS[-1]
    lower = lF[1:] <= math.log(0.5)                       # interior bin (e_k, e_{k+1}], k = 1..K-2: both edges' F / S
    out[1:-1] = np.where(lower, _logdiff(lF[:-1], lF[1:]), _logdiff(lS[1:], lS[:-1]))
    return out


def log_prior_bins(prior_dist, edges_disp, scale, fold=True):
    """meta.log_prior_bins (fold=True): the stated prior's exact log mass per bin (level order), the folded end bins holding the
    whole tails, as the labels do; meta.log_prior_bins_nominal (fold=False): the end bins' NOMINAL intervals only. Rounded to
    LOGP_DP decimals; never -inf here (every prior has positive density on its support), floored at -1e4 if it were."""
    v = np.maximum(log_bin_masses(prior_dist, edges_disp, scale, fold), -1e4)
    return [round(float(x), LOGP_DP) for x in v]


# ============================================================================ records
def make_record(g, split, idx, ns, eval_split, variant="prior_text", log_prior=False):
    fam = g["family"]
    axis, edges = g["axis"], g["edges"]
    exact, fl, fh = bin_masses(g["exact"], edges, axis["scale"])
    assert np.all(exact >= -1e-12) and abs(exact.sum() - 1) < 1e-9, (fam, exact.sum())
    levels = bin_labels(axis, edges)
    lab = rnd(exact)
    rid = sha16(f"{ns}|{split}|{fam}|{idx}|{g['state']}|{g['instructions']}")
    qid = f"numeric_{fam}/{rid}"
    scale = axis["scale"]
    bm, bs = binned_moments(exact, edges)
    meta = {"family": fam, "split": split, "format": g["format"], "context": g["context"], "quantity": g["quantity"],
            "axis": axis, "bin_edges": edges,
            "prior": {**g["prior"].params(), "stated": g["prior_params"]},
            "exact": {**g["exact"].params(), "mean": g["exact"].mean * scale, "sd": math.sqrt(g["exact"].var) * scale,
                      "mean_binned": bm, "sd_binned": bs, "folded_low": fl, "folded_high": fh},
            "truth": g["truth"], "data": g["data"], "fingerprint": g["fp"],
            "generator": VERSION_V2 if g.get("axes") == "v2" else VERSION, "seed_namespace": ns}
    if g.get("axes") == "v2":                          # (v2) only: v1 records are byte-identical
        meta["axes"] = "v2"
        if "stress" in g:
            meta["stress"] = g["stress"]
    if "posterior_mu" in g:
        meta["posterior_mu"] = g["posterior_mu"]
    if g["family"] == "coin_seq":
        meta["seq_len"] = g["data"]["seq_len"]
    if fam == "probe":
        meta["probe_variant"] = idx                      # 0 = the canonical, pre-registered wording
    if eval_split:
        pm, _, _ = bin_masses(g["prior"], edges, scale)
        mean, se = g["mle"]
        meta["baselines"] = {"prior": rnd(pm),
                             "mle_smooth": rnd(_smooth_point(axis, edges, None if mean is None else mean * scale)),
                             "mle_wald": rnd(_wald(axis, edges, mean, se))}
        meta["mle"] = {"value": None if mean is None else mean * scale, "se": None if se is None else se * scale}
    base = f"numeric/probe/beta10_10_6of10" if fam == "probe" else f"numeric/{fam}/{rid}"
    state = g["state"]
    if variant != "prior_text" or log_prior:          # (p101) only then: the default record is untouched
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"variant {variant!r}")
        meta["prompt_variant"] = variant
        meta["log_prior_bins"] = log_prior_bins(g["prior"], edges, scale)
        meta["log_prior_bins_nominal"] = log_prior_bins(g["prior"], edges, scale, fold=False)
        pfam = g.get("base_family", fam)              # (v2) a stress item's parameter family
        meta["offset_exact"] = pfam in PARAM_FAMILIES
        if pfam in OFFSET_NOTE:
            meta["offset_note"] = OFFSET_NOTE[pfam]
        if variant == "evidence_only":
            state = PROBE_CANON_EV if (fam == "probe" and g.get("prior_line") is None) else \
                evidence_state(g["state"], g["prior_line"], pfam, g.get("ev_key", rid))
    return {"id": f"numeric-{fam}-{split}-{rid}", "source": f"numeric/{fam}", "license": LICENSE, "base_id": base,
            "state": state, "questions": {qid: {"type": "score", "instructions": g["instructions"], "levels": levels}},
            "labels": {qid: dict(zip(levels, lab))}, "label_kind": {qid: "soft"}, "meta": meta}


def draw(family, split, i, ns, attempt=0, axes="v1", prior_override=None):
    if axes != "v1":
        return draw_v2(family, split, i, ns, attempt, prior_override)
    s = seed_of(ns, split, family, i, attempt)
    rng, py = np.random.default_rng(s), random.Random(s)
    if family == "coin":
        return gen_coin(rng, py)
    if family == "rate":
        return gen_rate(rng, py)
    if family == "normal_mean":
        return gen_normal(rng, py)
    if family == "normal_pred":
        return gen_normal(rng, py, predictive=True)
    if family == "ab_diff":
        return gen_ab(rng, py)
    if family == "coin_seq":
        lo, hi = SEQ_BUCKETS[i % len(SEQ_BUCKETS)]
        return gen_coin(rng, py, fmt="sequence", seq_len=int(rng.integers(lo, hi + 1)))
    raise ValueError(family)


def draw_v2(family, split, i, ns, attempt=0, prior_override=None):
    """(v2) The same item draws as v1 (same seeds: priors, truths and data are v1's, except the Beta priors v2 rounds to their
    printed values first), with the axis from its own stream (seed + "axis_v2"), the prior sentence from its own stream (seed +
    "prior_text_v2"), and the evidence-only sentence keyed by the item. `prior_override` (tests): replace the STATED prior after
    the truth and the data are drawn, leaving everything else as it was. Stress families draw a rate / normal_mean item from
    their own seeds and use the STRESS zoom / placement."""
    s = seed_of(ns, split, family, i, attempt)
    rng, py = np.random.default_rng(s), random.Random(s)
    v2 = {"arng": np.random.default_rng(seed_of(ns, split, family, i, attempt, "axis_v2")),
          "pp": random.Random(seed_of(ns, split, family, i, attempt, "prior_text_v2")), "override": prior_override,
          "stress": family if family in STRESS else None}
    base = STRESS_BASE[i % len(STRESS_BASE)] if family in STRESS else family
    if base == "coin":
        g = gen_coin(rng, py, v2=v2)
    elif base == "rate":
        g = gen_rate(rng, py, v2=v2)
    elif base == "normal_mean":
        g = gen_normal(rng, py, v2=v2)
    elif base == "normal_pred":
        g = gen_normal(rng, py, predictive=True, v2=v2)
    elif base == "ab_diff":
        g = gen_ab(rng, py, v2=v2)
    elif base == "coin_seq":
        lo, hi = SEQ_BUCKETS[i % len(SEQ_BUCKETS)]
        g = gen_coin(rng, py, fmt="sequence", seq_len=int(rng.integers(lo, hi + 1)), v2=v2)
    else:
        raise ValueError(family)
    g["axes"], g["ev_key"] = "v2", f"{ns}|{split}|{family}|{i}"
    if family in STRESS:
        g["stress"] = {"kind": family, "base_family": base, **STRESS[family]}
        g["base_family"], g["family"] = base, family
        g["fp"] = f"{family}|{g['fp']}"
    return g


PROBE_FP = f"coin|{PROBE['a']}|{PROBE['b']}|{PROBE['n']}|{PROBE['h']}"


def build(out, scale=1.0, ns=NS, log=print, variant="prior_text", log_prior=False, axes="v1"):
    """Fingerprint policy: identical QUESTIONS (state + instructions) never repeat across splits, nor within val / test.
    Identical inference problems (fingerprint = family, stated prior, sufficient statistics) are NOT rejected from train, except
    the probe's: rejecting them would carve the small-n / uniform-prior corner out of the training distribution. Instead every
    val / test record carries meta.fp_in_train, and eval_numeric.py reports the fresh subset separately.
    (p101) `variant` / `log_prior`: the same items rendered as PROMPT_VARIANTS[variant] and/or with meta.log_prior_bins (see
    make_record); the draws and the de-duplication run on the default text, so the items are the default build's. An
    evidence-only val / test record also carries meta.ev_prompt_in_train: its prompt text occurs in train (with another prior).
    axes "v2": the answer-independent axis rule (module docstring; draw_v2), the same sizes / seeds plus the test-only stress
    slices; the manifest adds the folded-mass and bin-occupancy statistics of the rule."""
    if axes not in AXES:
        raise ValueError(f"axes {axes!r}")
    out = Path(out)
    t0 = time.time()
    sizes = {s: {f: max(1, int(round(n * scale))) for f, n in d.items()} for s, d in SIZES.items()}
    if axes == "v2":
        for s_, d in SIZES_V2_EXTRA.items():
            sizes[s_].update({f: max(1, int(round(n * scale))) for f, n in d.items()})
    seen_q = set()
    stats = {"counts": {}, "redrawn": Counter(), "folded": {}, "label_entropy": {}, "fp_in_train": {}}
    by_split = defaultdict(lambda: defaultdict(list))
    for v in range(4):                                   # the probe first (fixed numbers), then val / test, then train
        g = gen_probe(v, axes)
        by_split["test"]["probe"].append(make_record(g, "test", v, ns, True, variant, log_prior))
        seen_q.add(g["state"] + "\n" + g["instructions"])
    for split in ("test", "val", "train"):
        ev = split != "train"
        for fam, n in sizes[split].items():
            recs = by_split[split][fam]
            for i in range(n):
                att = 0
                while True:
                    g = draw(fam, split, i, ns, att, axes)
                    key = g["state"] + "\n" + g["instructions"]
                    if key in seen_q or (not ev and g["fp"] == PROBE_FP) or \
                            (fam in STRESS and g["axis"]["rule"] != "evidence_v2"):   # (v2) a stress item needs data
                        att += 1
                        stats["redrawn"][f"{split}/{fam}"] += 1
                        if att > 200:
                            raise RuntimeError(f"cannot draw a fresh {fam} item {i}")
                        continue
                    break
                if ev:
                    seen_q.add(key)
                recs.append(make_record(g, split, i, ns, ev, variant, log_prior))
            log(f"  {split}/{fam}: {len(recs)} ({time.time() - t0:.0f} s)")
    train_fp = {r["meta"]["fingerprint"] for recs in by_split["train"].values() for r in recs}
    for split in ("val", "test"):
        for fam, recs in by_split[split].items():
            for r in recs:
                r["meta"]["fp_in_train"] = r["meta"]["fingerprint"] in train_fp
            stats["fp_in_train"][f"{split}/{fam}"] = int(sum(r["meta"]["fp_in_train"] for r in recs))
    if variant == "evidence_only":                   # (p101) the evidence-only text alone can repeat across splits
        stats["ev_prompt_in_train"] = {}
        train_q = {r["state"] + "\n" + next(iter(r["questions"].values()))["instructions"]
                   for recs in by_split["train"].values() for r in recs}
        for split in ("val", "test"):
            for fam, recs in by_split[split].items():
                for r in recs:
                    r["meta"]["ev_prompt_in_train"] = r["state"] + "\n" + next(iter(r["questions"].values()))["instructions"] in train_q
                stats["ev_prompt_in_train"][f"{split}/{fam}"] = int(sum(r["meta"]["ev_prompt_in_train"] for r in recs))
    for split, fams in by_split.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        for fam, recs in fams.items():
            with open(out / split / f"numeric_{fam}.jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["counts"][f"{split}/{fam}"] = len(recs)
            fo = [r["meta"]["exact"]["folded_low"] + r["meta"]["exact"]["folded_high"] for r in recs]
            stats["folded"][f"{split}/{fam}"] = {"mean": round(float(np.mean(fo)), 5), "p95": round(float(np.percentile(fo, 95)), 5),
                                                  "max": round(float(np.max(fo)), 5)}
            ents = []
            for r in recs:
                v = np.array(list(next(iter(r["labels"].values())).values()))
                v = v[v > 0]
                ents.append(float(-(v * np.log(v)).sum()))
            stats["label_entropy"][f"{split}/{fam}"] = {"mean": round(float(np.mean(ents)), 4),
                                                        "log_skill_room_mean": round(math.log(K) - float(np.mean(ents)), 4)}
    stats["redrawn"] = dict(stats["redrawn"])
    man = {"version": VERSION, "seed_namespace": ns, "scale": scale, "K": K, "license": LICENSE, "sizes": sizes,
           "families": {"in_family": list(IN_FAMILY), "held_out": list(HELD_OUT), "probe": "test/numeric_probe.jsonl"},
           "seq_buckets": SEQ_BUCKETS, "p_broad": P_BROAD, "zoom": ZOOM, "stats": stats,
           "probe": {"exact": "Beta(16, 14)", "mean": 16 / 30, "sd": math.sqrt(16 * 14 / (30 ** 2 * 31)),
                     "pyro_svi_tutorial": "0.532 +/- 0.090"},
           "build_secs": round(time.time() - t0, 1)}
    if variant != "prior_text" or log_prior:         # (p101) only then
        man["p101"] = {"variant": variant, "log_prior_bins": True, "offset_exact_families": list(PARAM_FAMILIES),
                       "offset_notes": OFFSET_NOTE}
    if axes == "v2":                                 # (v2) only: the v1 manifest is unchanged
        man["version"], man["axes"] = VERSION_V2, "v2"
        man["axis_rule_v2"] = {"zoom": ZOOM_V2, "z99": Z99, "stress": STRESS, "stress_base": STRESS_BASE,
                               "rate_ctx_range": RATE_CTX_RANGE, "ctx_range": CTX_RANGE, "label_fix": "Beta priors: printed = used",
                               "reads": "visible data (counts / exposure, readings / mean, stated noise sd) and fixed per-family "
                                        "or per-context constants only; never the prior, the posterior or the predictive"}
        man["axis_stats_v2"] = axis_stats({sp: dict(f) for sp, f in by_split.items()})
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def axis_stats(by_split):
    """(v2) Per split / family with declared axes: the exact mass folded into the end bins (mean, quantiles, share above 5% / 25% /
    50%), and how many bins carry the exact target's central 98% (median, p10, share <= 3 bins): the price of an answer-blind axis."""
    out = {}
    for split, fams in by_split.items():
        for fam, recs in fams.items():
            rows = [r for r in recs if r["meta"]["axis"]["rule"] != "natural"]
            if not rows:
                continue
            fo = np.array([r["meta"]["exact"]["folded_low"] + r["meta"]["exact"]["folded_high"] for r in rows])
            occ = []
            for r in rows:
                (qid, q), = r["questions"].items()
                c = np.cumsum([r["labels"][qid][k] for k in q["levels"]])
                occ.append(int(np.searchsorted(c, 0.99, side="left") - np.searchsorted(c, 0.01, side="left") + 1))
            occ = np.array(occ)
            rules = Counter(r["meta"]["axis"]["rule"] for r in rows)
            out[f"{split}/{fam}"] = {"n": len(rows), "rules": dict(rules), "folded_mean": round(float(fo.mean()), 5),
                                     "folded_q": {q: round(float(np.percentile(fo, q)), 5) for q in (50, 90, 95, 99)},
                                     "folded_max": round(float(fo.max()), 5),
                                     "share_folded_gt": {t: round(float((fo > t).mean()), 4) for t in (0.05, 0.25, 0.5)},
                                     "bins98_median": float(np.median(occ)), "bins98_p10": float(np.percentile(occ, 10)),
                                     "share_bins98_le3": round(float((occ <= 3).mean()), 4)}
    return out


# ============================================================================ checks
def load_module(name, path, stub_torch=True):
    import importlib.util
    import tempfile
    stub = None
    if stub_torch:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            stub = tempfile.mkdtemp(prefix="numeric_cf_stubs_", dir=str(HERE / "tmp") if (HERE / "tmp").is_dir() else None)
            Path(stub, "transformers.py").write_text("AutoTokenizer = None\n")
            Path(stub, "torch.py").write_text("class Tensor:\n    pass\n")
            sys.path.insert(0, stub)
    dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = dwb
        if stub:
            for m in ("torch", "transformers"):        # never leave a stub where a later real import would find it
                f = str(getattr(sys.modules.get(m), "__file__", "") or "")
                if f.startswith(stub):
                    del sys.modules[m]
            sys.path.remove(stub)
            for p in Path(stub).iterdir():
                p.unlink()
            Path(stub).rmdir()


def dist_from_params(d):
    """meta.exact / meta.prior {"dist", "params" (natural units)} -> the distribution object."""
    p = d["params"]
    if d["dist"] == "beta":
        return BetaD(p["a"], p["b"])
    if d["dist"] == "gamma":
        return GammaD(p["shape"], p["rate"])
    if d["dist"] == "normal":
        return NormalD(p["mean"], p["sd"])
    if d["dist"] == "beta_diff":
        return DiffBetaD(dist_from_params(p["A"]), dist_from_params(p["B"]))
    raise ValueError(d["dist"])


def check(root, sample_recompute=300, prompt_lengths=False, tokenizer=None):
    """Schema (build_jev_data.validate), cache_features.iter_units / question_options / is_noncommercial, K = 50, qids unique,
    train / eval fingerprint and state disjointness, labels = exact masses recomputed from meta for a sample."""
    root = Path(root)
    cf = load_module("cf_numeric", HERE / "trainers" / "cache_features.py")
    bjd = load_module("bjd_numeric", HERE / "build_jev_data.py", stub_torch=False)
    rep = {"errors": [], "units": {}, "label_sum_maxdev": 0.0, "recompute_maxdiff": 0.0}
    lp_diff, n_lp = 0.0, 0                              # (p101) meta.log_prior_bins, when the records carry it
    ev_variant, fpflag = False, defaultdict(dict)       # (p101) evidence-only: identical inputs allowed only as identical problems
    qids, fps, states, flags = set(), defaultdict(set), defaultdict(set), defaultdict(set)
    rng = random.Random(0)
    for split in ("train", "val", "test"):
        for p in sorted((root / split).glob("*.jsonl")):
            recs = [json.loads(ln) for ln in open(p, encoding="utf-8")]
            for r in recs:
                bjd.validate(r)
                (qid, q), = r["questions"].items()
                if qid in qids:
                    rep["errors"].append(f"duplicate qid {qid}")
                qids.add(qid)
                if len(q["levels"]) != K:
                    rep["errors"].append(f"{qid}: K = {len(q['levels'])}")
                m = r["meta"]
                lab = np.array([r["labels"][qid][k] for k in q["levels"]])
                rep["label_sum_maxdev"] = max(rep["label_sum_maxdev"], abs(lab.sum() - 1))
                fps[split].add(m["fingerprint"])
                if m.get("fp_in_train"):
                    flags[split].add(m["fingerprint"])
                qkey = r["state"] + "\n" + q["instructions"]
                if m.get("prompt_variant") == "evidence_only":   # (p101) the same text with another prior is another input
                    qkey += "\n" + json.dumps(m.get("log_prior_bins"))
                    ev_variant = True
                states[split].add(qkey)
                fpflag[split][qkey] = fpflag[split].get(qkey, True) and bool(m.get("fp_in_train"))
                if "log_prior_bins" in m:
                    n_lp += 1
                    lpb = np.asarray(m["log_prior_bins"], float)
                    if lpb.shape != (K,) or not np.all(np.isfinite(lpb)):
                        rep["errors"].append(f"{qid}: log_prior_bins shape / finite")
                    elif rng.random() < sample_recompute / 20000:
                        ref = log_prior_bins(dist_from_params(m["prior"]), m["bin_edges"], m["axis"]["scale"])
                        lp_diff = max(lp_diff, float(np.abs(np.asarray(ref) - lpb).max()))
                if r["license"] != LICENSE:
                    rep["errors"].append(f"{qid}: licence")
                if m.get("axes") == "v2":                # (v2) the axis from visible data alone; printed prior = used prior
                    rep.setdefault("v2_checked", 0)
                    rep["v2_checked"] += 1
                    rep["errors"].extend(f"{qid}: {e}" for e in check_v2_record(r))
                if rng.random() < sample_recompute / 20000:
                    ex, _, _ = bin_masses(dist_from_params(m["exact"]), m["bin_edges"], m["axis"]["scale"])
                    rep["recompute_maxdiff"] = max(rep["recompute_maxdiff"], float(np.abs(ex - lab).max()))
            units = list(cf.iter_units(str(root), [split], {p.stem}))
            for u in units:
                if cf.is_noncommercial(u):
                    rep["errors"].append(f"{u['qid']}: flagged non-commercial")
                if len(u["option_keys"]) != K or u["qtype"] != "score" or u["label_kind"] != "soft":
                    rep["errors"].append(f"{u['qid']}: unit shape")
            rep["units"][f"{split}/{p.stem}"] = len(units)
    rep["fp_overlap"] = {}
    for ev in ("val", "test"):
        rep["fp_overlap"][ev] = len(fps["train"] & fps[ev])
        ov = states["train"] & states[ev]
        if ov and ev_variant and all(fpflag[ev][k] for k in ov):
            # (p101) the default build never repeats a question text across splits; two of its questions that differed only in
            # the WORDING of the same prior become one input once the prior sentence is gone -- allowed when it is the same
            # inference problem (meta.fp_in_train: eval_numeric.py's fresh subset excludes it), reported
            rep.setdefault("ev_identical_inputs", {})[f"train&{ev}"] = len(ov)
        elif ov:
            rep["errors"].append(f"{len(ov)} train questions also in {ev}")
        if flags[ev] != {f for f in fps[ev] if f in fps["train"]}:
            rep["errors"].append(f"meta.fp_in_train flags in {ev} do not match the train fingerprints")
    if states["val"] & states["test"]:
        if ev_variant:                                  # (p101) as above: reported, not an error, for the evidence-only rendering
            rep.setdefault("ev_identical_inputs", {})["val&test"] = len(states["val"] & states["test"])
        else:
            rep["errors"].append("val / test questions overlap")
    if PROBE_FP in fps["train"]:
        rep["errors"].append("the probe's fingerprint is in train")
    if rep["label_sum_maxdev"] > 1e-6:
        rep["errors"].append(f"label sums off by {rep['label_sum_maxdev']}")
    if rep["recompute_maxdiff"] > 1e-7:
        rep["errors"].append(f"labels differ from recomputed exact masses by {rep['recompute_maxdiff']}")
    if prompt_lengths:
        rep["prompt_tokens"] = prompt_length_report(root, tokenizer)
    if n_lp:
        rep["log_prior_records"], rep["log_prior_recompute_maxdiff"] = n_lp, lp_diff
        if lp_diff > 1e-6:
            rep["errors"].append(f"log_prior_bins differ from recomputed values by {lp_diff}")
    rep["n_errors"] = len(rep["errors"])
    rep["errors"] = rep["errors"][:50]
    return rep


def check_v2_record(r):
    """(v2) Errors of one record: its axis is not reproduced from the visible data alone (axis_from_record), or a Beta prior's
    printed parameters (fparam) are not exactly the ones its label used (v1's rounded-prior bug)."""
    m, errs = r["meta"], []
    rec = axis_from_record(m)
    if rec is not None:
        ax, ed = rec
        if [float(x) for x in ed] != [float(x) for x in m["bin_edges"]]:
            errs.append(f"axis not reproduced from the visible data ({ax['lo']}..{ax['hi']} vs {m['bin_edges'][0]}..{m['bin_edges'][-1]})")
    st = (m.get("prior") or {}).get("stated") or {}
    pairs = [(st.get("a"), st.get("b"))] if "a" in st else [tuple(v) for k, v in st.items() if k in ("A", "B")]
    for a, b in pairs:
        for x in (a, b):
            if float(fparam(float(x))) != float(x):
                errs.append(f"Beta prior parameter {x} is printed as {fparam(float(x))}")
            elif f"{fparam(float(x))}" not in r["state"] and m["family"] != "probe" and m.get("prompt_variant") != "evidence_only":
                errs.append(f"Beta prior parameter {fparam(float(x))} not in the state")
    return errs


def prompt_length_report(root, tokenizer=None, per_file=200):
    """Render units through claude_lmscore.build_prompt with a local Qwen tokenizer (chat template) and count tokens."""
    from transformers import AutoTokenizer
    ls = load_module("ls_numeric", HERE / "trainers" / "claude_lmscore.py", stub_torch=False)
    tok = AutoTokenizer.from_pretrained(tokenizer or "Qwen/Qwen2.5-0.5B-Instruct")
    out = {}
    for split in ("train", "val", "test"):
        for p in sorted((Path(root) / split).glob("*.jsonl")):
            units = list(ls.cf.iter_units(str(root), [split], {p.stem}))
            step = max(1, len(units) // per_file)
            n = []
            for u in units[::step]:
                t = ls.build_prompt(tok, u, list(range(len(u["option_keys"]))), 4096)
                n.append(len(tok(t, add_special_tokens=False)["input_ids"]))
            out[f"{split}/{p.stem}"] = {"n": len(n), "median": int(np.median(n)), "p95": int(np.percentile(n, 95)),
                                        "max": int(max(n))}
    return out


# ============================================================================ self-test (closed forms vs Monte Carlo)
def self_test(n_mc=400000, seed=0):
    """Importance-sampled posteriors (prior draws weighted by the likelihood) and direct predictive / difference draws
    against the closed-form bin masses. Returns the max |closed - MC| per family and the 5-sigma bound used."""
    rng = np.random.default_rng(seed)
    res = {}

    def cmp(name, exact, w_samples, weights, edges, scale):
        e = np.asarray(edges) / scale
        idx = np.clip(np.searchsorted(e, w_samples, side="right") - 1, 0, K - 1)
        mc = np.bincount(idx, weights=weights, minlength=K)
        mc /= mc.sum()
        ess = weights.sum() ** 2 / (weights ** 2).sum()
        bound = 5 * np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / ess) + 2e-4
        d = np.abs(mc - exact)
        res[name] = {"max_abs": float(d.max()), "ok": bool(np.all(d <= bound)), "ess": float(ess)}

    # coin: prior Beta(2, 3), 7 of 12
    a, b, n, h = 2.0, 3.0, 12, 7
    ax, ed = coin_axis(False)
    ex, _, _ = bin_masses(BetaD(a + h, b + n - h), ed, 1.0)
    th = rng.beta(a, b, n_mc)
    cmp("coin", ex, th, th ** h * (1 - th) ** (n - h), ed, 1.0)
    # rate: Gamma(3, 0.5) prior, 17 events in exposure 4
    sh, rt, k, T = 3.0, 0.5, 17, 4.0
    post = GammaD(sh + k, rt + T)
    ax, ed = declared_axis(np.random.default_rng(1), post, 1.0, "per hour", support_lo=0.0)
    ex, _, _ = bin_masses(post, ed, 1.0)
    lam = rng.gamma(sh, 1 / rt, n_mc)
    cmp("rate", ex, lam, np.exp(k * np.log(lam) - lam * T - (k * np.log(k / T) - k)), ed, 1.0)
    # normal mean: N(100, 10^2) prior, sigma 4, readings
    m0, s0, sig = 100.0, 10.0, 4.0
    xs = np.array([104.2, 97.9, 101.5, 108.0, 99.3])
    prec = 1 / s0 ** 2 + len(xs) / sig ** 2
    post = NormalD((m0 / s0 ** 2 + xs.sum() / sig ** 2) / prec, math.sqrt(1 / prec))
    ax, ed = declared_axis(np.random.default_rng(2), post, 1.0, "g")
    ex, _, _ = bin_masses(post, ed, 1.0)
    mu = rng.normal(m0, s0, n_mc)
    ll = -0.5 * ((xs[None, :] - mu[:, None]) ** 2).sum(1) / sig ** 2
    cmp("normal_mean", ex, mu, np.exp(ll - ll.max()), ed, 1.0)
    # predictive: mu from the IS posterior, then a new reading
    pred = NormalD(post.m, math.sqrt(post.s ** 2 + sig ** 2))
    ax, ed = declared_axis(np.random.default_rng(3), pred, 1.0, "g")
    ex, _, _ = bin_masses(pred, ed, 1.0)
    xn = mu + sig * rng.standard_normal(n_mc)
    cmp("normal_pred", ex, xn, np.exp(ll - ll.max()), ed, 1.0)
    # A/B difference: direct posterior draws
    A, B = BetaD(1 + 30, 1 + 200 - 30), BetaD(2 + 41, 2 + 210 - 41)
    dd = DiffBetaD(A, B)
    ax, ed = natural_axis(-1.0, 1.0, 0.04, 1.0, "")
    ex, _, _ = bin_masses(dd, ed, 1.0)
    d = A.sample(rng, 4 * n_mc) - B.sample(rng, 4 * n_mc)
    cmp("ab_diff", ex, d, np.ones_like(d), ed, 1.0)
    # the two quadrature orderings agree
    t = np.linspace(-0.5, 0.5, 41)
    res["ab_diff_orderings_maxdiff"] = float(np.abs(DiffBetaD(A, B, "A").cdf(t) - DiffBetaD(A, B, "B").cdf(t)).max())
    # probe numbers
    pr = BetaD(16, 14)
    res["probe"] = {"mean": pr.mean, "sd": math.sqrt(pr.var)}
    return res


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--ns", default=NS)
    b.add_argument("--variant", choices=PROMPT_VARIANTS, default="prior_text",
                   help="(p101) evidence_only: the prior sentence replaced (the same items, labels and ids); default unchanged")
    b.add_argument("--log-prior", action="store_true", help="(p101) also write meta.log_prior_bins (implied by --variant evidence_only)")
    b.add_argument("--axes", choices=AXES, default="v1",
                   help="v1 (default, byte-identical): axes around the exact answer (the assisted-range benchmark); v2: answer- and "
                        "prior-independent axes from the visible evidence (research/numeric_v2_axes.md)")
    c = sub.add_parser("check")
    c.add_argument("root")
    c.add_argument("--prompt-lengths", action="store_true")
    c.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    c.add_argument("--json", default="")
    s = sub.add_parser("self-test")
    s.add_argument("--n-mc", type=int, default=400000)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        man = build(a.out, a.scale, a.ns, variant=a.variant, log_prior=a.log_prior, axes=a.axes)
        print(json.dumps({k: man[k] for k in ("sizes", "build_secs")}, indent=1))
        print(json.dumps(man["stats"], indent=1))
        return 0
    if a.cmd == "check":
        rep = check(a.root, prompt_lengths=a.prompt_lengths, tokenizer=a.tokenizer)
        txt = json.dumps(rep, indent=1)
        print(txt)
        if a.json:
            Path(a.json).write_text(txt)
        return 0 if rep["n_errors"] == 0 else 1
    if a.cmd == "self-test":
        res = self_test(a.n_mc)
        print(json.dumps(res, indent=1))
        ok = all(v.get("ok", True) for v in res.values() if isinstance(v, dict)) and res["ab_diff_orderings_maxdiff"] < 1e-9
        return 0 if ok else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
