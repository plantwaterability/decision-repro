#!/usr/bin/env python3
"""
build_bayes_pref.py -- GDE-1 X-domain candidate set `bayes_pref`: an EXACT-POSTERIOR belief-update probe, plus a
training-corpus mode for the Bayesian-teaching A/B. Design and validation: research/bayes_pref.md.

DESIGN SOURCE. The task design is reimplemented from the Methods of Qiu, Sha, Allen, Kim, Linzen & van Steenkiste,
"Bayesian teaching enables probabilistic reasoning in large language models", Nature Communications 17:1238 (2026),
doi:10.1038/s41467-025-67998-6. The article is CC BY-NC-ND 4.0, so nothing here copies its text, prompts or data: the
generator, the item renderings and every sentence of the states are ours, and the paper's Zenodo data is not used.

THE TASK
  * Simulated users: a reward weight per feature, theta in {-1, -0.5, 0, +0.5, +1}^4 minus the all-zero vector (624).
  * Items: 4 features, each on an evenly spaced grid in [0, 1] (11 levels; flights' stops 3 levels; hotels' rating
    and amenities 5 levels, as in the paper's hotel task). Each round shows 3 distinct items; the user picks
    argmax_o theta . phi(o), ties broken uniformly at random.
  * A unit = one round of one trajectory: state = the family's description of the items and of the user model, the r
    earlier rounds (the 3 items shown and the user's pick), and the 3 current items; ONE question, "which item will this
    user pick now?", a `choice` over A/B/C (= items 1/2/3 of the current round; each option text carries the item).
  * Gold = the EXACT posterior predictive under the uniform prior over the 624 users, by enumeration:
        L_t(theta)  = 1[c_t in A_t(theta)] / |A_t(theta)|,  A_t(theta) = argmax set of round t under theta
        q(theta|h)  ∝ prod_{t<=r} L_t(theta)
        P(k | h, O) = sum_theta q(theta|h) 1[k in A(theta, O)] / |A(theta, O)|
    The rewards are exact integers ((2 theta) . (20 phi)), so ties are detected exactly.
  * Metadata (never in the state): the user's weights, the oracle answer (the simulated user's actual pick this round),
    the Bayes-optimal pick (argmax_k P; ties broken by a seeded draw), the paper's Assistant pick (posterior-mean plug-in
    with the indicator likelihood), the full-precision gold, the item levels of every round.
  * Families (surface forms): flights (the paper's features: departure time, travel time, stops, fare), hotels (the
    paper's hotel features: distance to the centre, price, rating, amenities), trails (NEW: day-hike suggestions --
    length, climb, drive time, shade; not a typed-decisions workflow and not shopping).
  * EVAL: the 624 users are split at random into three families of 208; one 5-round trajectory per user; the 5 units of
    a trajectory (r = 0..4 earlier rounds) share base_id (the bootstrap cluster). 3 x 208 x 5 = 3,120 units, 624 clusters.
  * TRAIN mode: a disjoint seed namespace; users drawn uniformly from the 624 (the same prior the posterior uses, so the
    oracle one-hot is an unbiased draw of the exact target); every option set that occurs anywhere in the eval set is
    rejected (checked by fingerprint), so no (user, option set) pair is shared. Three target variants per unit: `exact`
    (posterior predictive), `oracle` (one-hot actual pick), `bayes_argmax` (one-hot of P's mode).

Usage (.venv python: numpy; scipy for validate / report):
  .venv/bin/python3 build_bayes_pref.py eval     --out /tmp/bayes_pref_set                 # build + check
  .venv/bin/python3 build_bayes_pref.py validate --out /tmp/bayes_pref_set --work /tmp/bayes_pref_work
  .venv/bin/python3 build_bayes_pref.py train    --n-units 200 --out /tmp/bayes_pref_train_smoke.jsonl
  .venv/bin/python3 build_bayes_pref.py train    --n-units 8000 --families flights --arms-dir /tmp/bayes_pref_train
  .venv/bin/python3 build_bayes_pref.py report   --root /tmp/bayes_pref_set --pred d34c=res/claude_gde1_d34c_bayes_pref.npz
  .venv/bin/python3 build_bayes_pref.py --self-test
Scoring (GPU, not run here): claude_lmscore.py --data /tmp/bayes_pref_set/bayes_pref --splits eval --out ..._bayes_pref.npz, then
analysis/gde1_score.py --sets bayes_pref (domain X) and `report` for the family x rounds table.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

from build_jev_data import choice_q, validate  # noqa: E402

SET = "bayes_pref"
VERSION = "bayes_pref v1 (2026-10-01)"
BUILT_DATE = "2026-10-01"
OUT = "/tmp/bayes_pref_set"   # 2026-10-02: kept OUT of /tmp/gde1 (a candidate probe, not a GDE-1 set until decided)
WORK = "/tmp/bayes_pref_work"
EVAL_NS = "bayes_pref/eval/v1"
TRAIN_NS = "bayes_pref/train"
QID = "next_choice"
KEYS = ("A", "B", "C")
N_ROUNDS = 5                         # rounds per trajectory; unit r (0..4) predicts round r + 1 from r earlier rounds
FAMILY_ORDER = ("flights", "hotels", "trails")
PAPER = ("Qiu, Sha, Allen, Kim, Linzen & van Steenkiste, 'Bayesian teaching enables probabilistic reasoning in large "
         "language models', Nature Communications 17:1238 (2026), doi:10.1038/s41467-025-67998-6")
# The record licence: OUR generated data. The paper's own licence (CC BY-NC-ND 4.0, the article) is stated in the
# manifest, NOT here: cache_features.is_noncommercial reads this string, and a TRAINING record must never trip it.
LICENSE = ("generated by build_bayes_pref.py (ours): synthetic users and items, no third-party data or text; the task "
           "design is reimplemented from the Methods of Qiu et al. 2026 (Nature Communications 17:1238)")
LICENSE_TRAIN = LICENSE

# --------------------------------------------------------------------------- users
W_HALF = (-2, -1, 0, 1, 2)                                  # 2 * theta, so rewards are exact integers
THETA2 = np.array([t for t in itertools.product(W_HALF, repeat=4) if any(t)], dtype=np.int64)   # (624, 4)
N_USERS = len(THETA2)
assert N_USERS == 624
GRID = 20                                                   # features stored as GRID * phi (11 levels -> 0, 2, .., 20)


def phi_int(levels, nlev):
    """Level indices -> GRID * phi, exact integers (phi = j / (L - 1))."""
    return np.array([j * GRID // (L - 1) for j, L in zip(levels, nlev)], dtype=np.int64)


def theta_of(u):
    return [float(x) / 2 for x in THETA2[u]]


# --------------------------------------------------------------------------- families (surface forms; all text ours)
def _clock(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _hm(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h} h {m:02d} min"


AMENITIES = ("free Wi-Fi", "breakfast", "gym", "pool")


def _amen(j):
    return "no amenities" if j == 0 else f"{j} amenit{'y' if j == 1 else 'ies'} ({', '.join(AMENITIES[:j])})"


FAMILIES = {
    "flights": {
        "intro": "A booking assistant works with one traveller who takes the same kind of trip again and again. In every "
                 "round the traveller is offered three flights and books exactly one of them.",
        "person": "traveller", "item": "flight", "Item": "Flight", "did": "booked", "does": "books",
        "question": "Which of the three flights offered in the current round will this traveller book?",
        "features": [("departure time", 11, lambda j: _clock(300 + 90 * j)),
                     ("total travel time", 11, lambda j: _hm(80 + 55 * j)),
                     ("number of stops", 3, lambda j: ("nonstop", "1 stop", "2 stops")[j]),
                     ("fare", 11, lambda j: f"${90 + 65 * j}")],
        "line": lambda t: f"departs {t[0]}, {t[1]} total travel time, {t[2]}, fare {t[3]}",
    },
    "hotels": {
        "intro": "A travel assistant works with one guest who needs a hotel for a string of separate city trips. In every "
                 "round the guest is offered three hotels and books exactly one of them.",
        "person": "guest", "item": "hotel", "Item": "Hotel", "did": "booked", "does": "books",
        "question": "Which of the three hotels offered in the current round will this guest book?",
        "features": [("distance from the city centre", 11, lambda j: f"{0.4 + 0.8 * j:.1f} km"),
                     ("price per night", 11, lambda j: f"${70 + 28 * j}"),
                     ("guest rating", 5, lambda j: f"{j + 1} out of 5"),
                     ("amenities", 5, _amen)],
        "line": lambda t: f"{t[0]} from the city centre, {t[1]} per night, guest rating {t[2]}, {t[3]}",
    },
    "trails": {
        "intro": "A hiking app suggests day hikes to one hiker who heads out most weekends. In every round the hiker is "
                 "offered three trails and picks exactly one of them to walk.",
        "person": "hiker", "item": "trail", "Item": "Trail", "did": "picked", "does": "picks",
        "question": "Which of the three trails offered in the current round will this hiker pick?",
        "features": [("round-trip length", 11, lambda j: f"{3 + 1.5 * j:.1f} km"),
                     ("total climb", 11, lambda j: f"{120 * j:,} m"),
                     ("drive to the trailhead", 11, lambda j: f"{15 + 9 * j} min"),
                     ("shade", 3, lambda j: ("mostly exposed", "partly shaded", "mostly shaded")[j])],
        "line": lambda t: f"{t[0]} round trip, {t[1]} of climbing, {t[2]} drive to the trailhead, {t[3]}",
    },
}


def nlev(fam):
    return [f[1] for f in FAMILIES[fam]["features"]]


def item_text(fam, levels):
    F = FAMILIES[fam]
    return F["line"]([f[2](j) for f, j in zip(F["features"], levels)])


def _scale_line(name, L, rend):
    if L <= 5:
        vals = ", ".join(f"{rend(j)} = {j / (L - 1):g}" for j in range(L))
        return f"- {name}: {vals}"
    return f"- {name}: {rend(0)} = 0 up to {rend(L - 1)} = 1, in {L - 1} equal steps of {1 / (L - 1):g}"


def model_text(fam):
    F = FAMILIES[fam]
    p, it, did = F["person"], F["item"], F["did"]
    scales = "\n".join(_scale_line(n, L, r) for n, L, r in F["features"])
    return (f"How the {p} chooses. Every {it} is scored on four features, each placed on a scale from 0 to 1:\n{scales}\n"
            f"The {p} has a private weight for each feature, one of -1, -0.5, 0, +0.5 or +1: a negative weight means lower "
            f"values suit them better, a positive weight means higher values suit them better, and 0 means the feature "
            f"does not matter to them; a weight of 1 counts twice as much as 0.5. Before any of their choices are seen, every "
            f"combination of the four weights is equally likely, except that the weights are never all 0 (so there are "
            f"624 possible {p}s). In each round the {p} {F['does']} the {it} with the largest weighted sum of its four scaled "
            f"features; if two or three {it}s share the largest sum, each of them is equally likely to be {did}. The "
            f"weights stay the same in every round.")


def round_block(fam, opts):
    F = FAMILIES[fam]
    return "\n".join(f"  {F['Item']} {i + 1}: {item_text(fam, o)}" for i, o in enumerate(opts))


def render_state(fam, hist, cur):
    """hist: [(opts, choice_index)], cur: opts (3 level vectors). The pick of the CURRENT round is never rendered."""
    F = FAMILIES[fam]
    parts = [F["intro"], "", model_text(fam), ""]
    if not hist:
        parts.append(f"Earlier rounds with this {F['person']}: none; the current round is their first.")
    else:
        parts.append(f"Earlier rounds with this {F['person']}:")
        for t, (opts, c) in enumerate(hist):
            parts.append(f"Round {t + 1}, offered:\n{round_block(fam, opts)}\n  The {F['person']} {F['did']} "
                         f"{F['Item']} {c + 1}.")
    parts.append("")
    parts.append(f"Round {len(hist) + 1} (the current round), offered:\n{round_block(fam, cur)}")
    return "\n".join(parts)


def question(fam, cur):
    F = FAMILIES[fam]
    return choice_q(F["question"], {k: f"{F['Item']} {i + 1}: {item_text(fam, o)}" for i, (k, o) in enumerate(zip(KEYS, cur))})


# --------------------------------------------------------------------------- exact inference (vectorised)
def choice_matrix(F):
    """F: (3, 4) GRID-scaled integer features. -> (C, A): C[theta, k] = P(pick k | theta) with uniform tie-breaking,
    A = the argmax indicator (624, 3)."""
    R = THETA2 @ F.T
    A = R == R.max(1, keepdims=True)
    return A / A.sum(1, keepdims=True), A


def posterior(hist_F, hist_c, likelihood="exact"):
    """Normalised posterior over the 624 users. likelihood='exact': 1[c in A] / |A| (the generator's own); 'indicator':
    1[c in A] (the paper's Eq. 3, which ignores the tie-breaking probability)."""
    w = np.ones(N_USERS)
    for F, c in zip(hist_F, hist_c):
        C, A = choice_matrix(F)
        w = w * (C[:, c] if likelihood == "exact" else A[:, c])
    s = w.sum()
    assert s > 0
    return w / s


def predictive(post, F):
    C, _ = choice_matrix(F)
    return post @ C


def plugin_pick_set(post, F):
    """The paper's Assistant: argmax_o E_q[theta] . phi(o) (posterior-mean plug-in); the set of tied maxima."""
    s = (post @ THETA2.astype(float)) @ F.T.astype(float)
    return np.flatnonzero(np.isclose(s, s.max(), rtol=0, atol=1e-12))


def top_set(P):
    return np.flatnonzero(np.isclose(P, P.max(), rtol=0, atol=1e-12))


# --------------------------------------------------------------------------- exact inference (independent, rational)
def exact_predictive_rational(fam, hist_levels, hist_c, cur_levels):
    """A second, independent implementation for validation: pure Python, exact Fractions, its own enumeration of the
    users and its own phi = j / (L - 1); shares nothing with the numpy path but the definitions."""
    L = nlev(fam)
    wv = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    users = []
    for a in wv:
        for b in wv:
            for c in wv:
                for d in wv:
                    if a or b or c or d:
                        users.append((a, b, c, d))

    def phi(levels):
        return [Fraction(j, n - 1) for j, n in zip(levels, L)]

    def pick_probs(theta, opts):
        r = [sum(t * x for t, x in zip(theta, phi(o))) for o in opts]
        m = max(r)
        win = [i for i, x in enumerate(r) if x == m]
        return [Fraction(1, len(win)) if i in win else Fraction(0) for i in range(len(opts))]

    weights = []
    for theta in users:
        w = Fraction(1)
        for opts, ch in zip(hist_levels, hist_c):
            w *= pick_probs(theta, opts)[ch]
            if w == 0:
                break
        weights.append(w)
    Z = sum(weights)
    out = [Fraction(0)] * len(cur_levels)
    for theta, w in zip(users, weights):
        if w:
            for k, p in enumerate(pick_probs(theta, cur_levels)):
                out[k] += w * p
    return [x / Z for x in out]


# --------------------------------------------------------------------------- generation
def seed_of(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def sha16(s):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def draw_options(rng, fam, forbid=None, stats=None):
    """3 distinct items, every feature level uniform; `forbid`: option-set fingerprints to reject (train mode)."""
    L = nlev(fam)
    while True:
        opts = [tuple(int(rng.integers(0, n)) for n in L) for _ in range(3)]
        if len(set(opts)) < 3:
            if stats is not None:
                stats["redraw/duplicate_item"] += 1
            continue
        if forbid is not None and fingerprint(fam, opts) in forbid:
            if stats is not None:
                stats["redraw/eval_option_set"] += 1
            continue
        return opts


def fingerprint(fam, opts):
    """An option set as an unordered set of GRID-scaled feature vectors (family-agnostic: two families with the same
    grid and the same numbers are the same inference problem)."""
    L = nlev(fam)
    return tuple(sorted(tuple(phi_int(o, L).tolist()) for o in opts))


def gen_trajectory(fam, user, rng, forbid=None, stats=None):
    """5 rounds: [(opts, F, choice)], the user's pick drawn with uniform tie-breaking."""
    L = nlev(fam)
    out = []
    for _ in range(N_ROUNDS):
        opts = draw_options(rng, fam, forbid, stats)
        F = np.stack([phi_int(o, L) for o in opts])
        C, _ = choice_matrix(F)
        c = int(rng.choice(3, p=C[user]))
        out.append((opts, F, c))
    return out


def unit_payload(fam, user, traj, r, rng_tie):
    """Everything about unit r of a trajectory: state, question, gold, references."""
    hist = traj[:r]
    opts, F, c_now = traj[r]
    post = posterior([h[1] for h in hist], [h[2] for h in hist])
    P = predictive(post, F)
    post_ind = posterior([h[1] for h in hist], [h[2] for h in hist], likelihood="indicator")
    ts = top_set(P)
    bayes = int(ts[0] if len(ts) == 1 else rng_tie.choice(ts))
    ps = plugin_pick_set(post_ind, F)
    state = render_state(fam, [(h[0], h[2]) for h in hist], opts)
    return {
        "state": state, "q": question(fam, opts), "P": P, "post": post,
        "oracle": c_now, "bayes": bayes, "bayes_set": ts.tolist(), "plugin_set": ps.tolist(),
        "support": int((post > 0).sum()), "true_post": float(post[user]),
        "post_entropy": float(-(post[post > 0] * np.log(post[post > 0])).sum()),
    }


def lab_dict(vals):
    """Rounded to 6 dp, summing to exactly 1 (the largest entry absorbs the rounding), as build_gde1_b.lab_dict."""
    v = [max(0.0, float(x)) for x in vals]
    s = sum(v)
    v = [round(x / s, 6) for x in v]
    j = max(range(len(v)), key=lambda i: v[i])
    v[j] = round(v[j] + (1.0 - sum(v)), 6)
    return dict(zip(KEYS, v))


def one_hot(k):
    return {kk: (1.0 if i == k else 0.0) for i, kk in enumerate(KEYS)}


def kind_of(lab):
    return "hard" if sum(1 for v in lab.values() if v > 0) == 1 else "soft"


def entropy(vals):
    return -sum(x * math.log(x) for x in vals if x > 0)


def eval_assignment():
    """The 624 users in a seeded random order, split into three families of 208."""
    rng = np.random.default_rng(seed_of(EVAL_NS, "users"))
    perm = rng.permutation(N_USERS)
    n = N_USERS // len(FAMILY_ORDER)
    return {fam: [int(u) for u in perm[i * n:(i + 1) * n]] for i, fam in enumerate(FAMILY_ORDER)}


def eval_trajectories():
    out = []
    for fam, users in eval_assignment().items():
        for t, u in enumerate(users):
            rng = np.random.default_rng(seed_of(EVAL_NS, fam, t, u))
            out.append((fam, t, u, gen_trajectory(fam, u, rng)))
    return out


def eval_fingerprints():
    return {fingerprint(fam, opts) for fam, _, _, traj in eval_trajectories() for opts, _, _ in traj}


def make_record(fam, t, u, traj, r, ns, train=False):
    rng_tie = np.random.default_rng(seed_of(ns, fam, t, u, r, "tie"))
    pl = unit_payload(fam, u, traj, r, rng_tie)
    gold = lab_dict(pl["P"])
    traj_id = f"{fam}/t{t:04d}"
    hist_meta = [{"options_levels": [list(o) for o in traj[i][0]], "pick": KEYS[traj[i][2]]} for i in range(r)]
    meta = {
        "family": fam, "rounds_observed": r, "round_now": r + 1, "trajectory_id": traj_id,
        "user_index": u, "user_weights": theta_of(u),
        "oracle_choice": KEYS[pl["oracle"]],
        "bayes_argmax": KEYS[pl["bayes"]], "bayes_argmax_set": [KEYS[i] for i in pl["bayes_set"]],
        "paper_assistant_pick_set": [KEYS[i] for i in pl["plugin_set"]],
        "gold_exact": [float(x) for x in pl["P"]],
        "posterior_support": pl["support"], "posterior_entropy_nats": round(pl["post_entropy"], 6),
        "true_user_posterior": pl["true_post"],
        "options_levels": [list(o) for o in traj[r][0]], "history": hist_meta,
        "feature_levels": nlev(fam), "feature_names": [f[0] for f in FAMILIES[fam]["features"]],
        "label_entropy_nats": round(entropy(pl["P"]), 6),
        "generator": VERSION, "seed_namespace": ns,
    }
    qs = {QID: pl["q"]}
    if not train:
        labs = {QID: gold}
        rec = {"id": f"gde1-{SET}-{sha16(ns + '|' + traj_id + '|' + str(r))}", "source": f"{SET}/{fam}",
               "license": LICENSE, "state": pl["state"], "questions": qs, "labels": labs,
               "label_kind": {QID: kind_of(gold)}, "base_id": f"{SET}/{traj_id}",
               "meta": {"gde1_set": SET, "domain": "X", "gde1_part": "bayes_pref", "gde1_version": VERSION,
                        "eval_only": True,
                        "label_provenance": "exact (posterior predictive by enumeration over the 624 users; uniform "
                                            "prior; likelihood 1[pick in argmax] / |argmax|; build_bayes_pref.py)",
                        "label_source": {QID: "exact:posterior_predictive"},
                        "decision_date": BUILT_DATE, "info_date": BUILT_DATE, "time_slice": "n/a (exact labels)",
                        "td_workflow": None, **meta}}
    else:
        targets = {"exact": gold, "oracle": one_hot(pl["oracle"]), "bayes_argmax": one_hot(pl["bayes"])}
        rec = {"id": f"{SET}_train-{sha16(ns + '|' + traj_id + '|' + str(r))}", "source": f"{SET}_train/{fam}",
               "license": LICENSE_TRAIN, "state": pl["state"], "questions": qs, "labels": {QID: gold},
               "label_kind": {QID: kind_of(gold)}, "base_id": f"{SET}_train/{ns.split('/')[-1]}/{traj_id}",
               "meta": {"label_source": {QID: "exact:posterior_predictive"}, "target_variant": "exact",
                        "targets": targets, "td_workflow": None, "eval_only": False, **meta}}
    validate(rec)
    return rec


def with_variant(rec, variant):
    """The same record with `labels` = one target variant (identical state, question, option order and id)."""
    r = json.loads(json.dumps(rec))
    lab = r["meta"]["targets"][variant]
    r["labels"] = {QID: lab}
    r["label_kind"] = {QID: kind_of(lab)}
    r["meta"]["target_variant"] = variant
    r["meta"]["label_source"] = {QID: {"exact": "exact:posterior_predictive", "oracle": "simulated:actual_pick",
                                       "bayes_argmax": "exact:predictive_mode_one_hot"}[variant]}
    validate(r)
    return r


# --------------------------------------------------------------------------- shared helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def nc_regex():
    """cache_features.NC_LICENCE, read from the file (cache_features imports torch at module level)."""
    src = (HERE / "trainers" / "cache_features.py").read_text()
    m = re.search(r'NC_LICENCE = re\.compile\((r".*?")\s*,\s*re\.I\)', src)
    return re.compile(eval(m.group(1)), re.I)   # noqa: S307  (a raw string literal from our own file)


def read_set(root):
    recs = []
    for p in sorted((Path(root) / SET / "eval").glob("*.jsonl")):
        recs += [json.loads(line) for line in p.open(encoding="utf-8")]
    return recs


# --------------------------------------------------------------------------- eval build
def cmd_eval(a):
    t0 = time.time()
    b = load_module("build_gde1_b", HERE / "build_gde1_b.py")
    assert not nc_regex().search(LICENSE), "the record licence string trips the NC filter"
    trajs = eval_trajectories()
    recs = [make_record(fam, t, u, traj, r, EVAL_NS) for fam, t, u, traj in trajs for r in range(N_ROUNDS)]
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids))
    n_states = len({r["state"] for r in recs})
    rep = {"records": len(recs), "units": len(recs), "trajectories": len(trajs), "distinct_states": n_states,
           "units_by_family": dict(Counter(r["meta"]["family"] for r in recs)),
           "units_by_rounds_observed": dict(sorted(Counter(r["meta"]["rounds_observed"] for r in recs).items())),
           "label_kind": dict(Counter(r["label_kind"][QID] for r in recs)), "seconds": round(time.time() - t0, 1)}
    by_fr = defaultdict(list)
    for r in recs:
        by_fr[(r["meta"]["family"], r["meta"]["rounds_observed"])].append(r)
    cell = {}
    for (fam, rr), v in sorted(by_fr.items()):
        cell[f"{fam}/r{rr}"] = {
            "n": len(v), "mean_label_entropy_nats": round(float(np.mean([r["meta"]["label_entropy_nats"] for r in v])), 4),
            "mean_max_p": round(float(np.mean([max(r["meta"]["gold_exact"]) for r in v])), 4),
            "hard_labels": sum(r["label_kind"][QID] == "hard" for r in v),
            "mean_posterior_support": round(float(np.mean([r["meta"]["posterior_support"] for r in v])), 1)}
    manifest = {
        "set": SET, "domain": "X", "builder": VERSION, "built": datetime.now().isoformat(timespec="seconds"),
        "license": LICENSE,
        "license_evidence": {
            "data": "generated here: no third-party data, text or prompts; the items, users, renderings and sentences are ours",
            "design_source": PAPER,
            "design_source_licence": "the article is CC BY-NC-ND 4.0 (Open Access statement, p. 11): the DESIGN (task "
                                     "structure in its Methods) is reimplemented; no text, figure, prompt or data is "
                                     "copied, and the Zenodo data (record 17677329, CC BY 4.0) was not downloaded"},
        "eval_only": True, "build_report": rep, "cells": cell,
        "source": {"generator": "build_bayes_pref.py", "seed_namespace": EVAL_NS, "users": N_USERS,
                   "families": {f: {"features": [(n, L) for n, L, _ in FAMILIES[f]["features"]]} for f in FAMILY_ORDER}},
        "time_split": "none: labels are exact by construction (regenerate with a new seed namespace per release)",
        "label": "exact posterior predictive P(pick | earlier picks, current options) under the uniform prior over 624 "
                 "users; tie-aware likelihood; meta.gold_exact = full precision, labels rounded to 6 dp",
        "clusters": "base_id = one user's 5-round trajectory (5 units: r = 0..4 earlier rounds)",
        "splits": {"eval": {"stats": None}},
        "dedup_vs_training": {
            "status": "text dedup not meaningful: every state shares this generator's template and the item lines repeat "
                      "by construction (3 x 11^3 flights). Disjointness from the bayes_pref TRAINING corpus is enforced "
                      "by option-set fingerprint (train mode rejects every option set used anywhere in this eval set). "
                      "The usual node gate (build_gde1_b.py dedup --kind built) can still be run with the other sets."},
    }
    out = {"eval": recs}
    root = Path(a.out) / SET
    b.write_set(a.out, SET, out, manifest)
    manifest["splits"]["eval"]["stats"] = b.split_stats(recs)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, ensure_ascii=False))
    cmd_check(a, b)


def cmd_check(a, b=None):
    b = b or load_module("build_gde1_b", HERE / "build_gde1_b.py")
    b.SCAN[SET] = re.compile(r"\boracle\b|posterior|\bgold\b|\banswer\b|bayes|theta", re.I)
    cf = b.load_cf()
    rep = b.check_set(a.out, SET, cf)
    # structure: the current pick is never in the state; exactly r earlier picks are; balanced rounds per trajectory
    recs = read_set(a.out)
    bad = Counter()
    per_traj = defaultdict(set)
    for r in recs:
        m, fam = r["meta"], r["meta"]["family"]
        F = FAMILIES[fam]
        n_picks = len(re.findall(rf"^  The {F['person']} {F['did']} {F['Item']} [123]\.$", r["state"], re.M))
        if n_picks != m["rounds_observed"]:
            bad["picks_in_state != rounds_observed"] += 1
        if f"Round {m['rounds_observed'] + 1} (the current round)" not in r["state"]:
            bad["current_round_header"] += 1
        tail = r["state"].split("(the current round), offered:\n")[-1].split("\n")
        if len(tail) != 3 or not all(ln.startswith(f"  {F['Item']} {i + 1}: ") for i, ln in enumerate(tail)):
            bad["text_after_current_options"] += 1
        if abs(sum(m["gold_exact"]) - 1) > 1e-12 or max(abs(x - r["labels"][QID][k]) for x, k in zip(m["gold_exact"], KEYS)) > 1e-6:
            bad["gold_vs_labels"] += 1
        if m["oracle_choice"] not in KEYS or m["gold_exact"][KEYS.index(m["oracle_choice"])] <= 0:
            bad["oracle_has_zero_gold"] += 1
        if r["base_id"] != f"{SET}/{m['trajectory_id']}":
            bad["base_id"] += 1
        per_traj[r["base_id"]].add(m["rounds_observed"])
    unbalanced = sum(1 for v in per_traj.values() if v != set(range(N_ROUNDS)))
    rep["bayes_pref_structure"] = {"problems": dict(bad) or "none", "trajectories": len(per_traj),
                                   "trajectories_without_all_5_rounds": unbalanced,
                                   "nc_filter_on_licence": bool(nc_regex().search(LICENSE))}
    (Path(a.out) / SET / "check_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"check: {rep['total_units']} units, schema {rep['schema_problems']}, structure {rep['bayes_pref_structure']}")
    if bad or unbalanced or rep["schema_problems"] != "none":
        raise SystemExit("check FAILED")
    return rep


# --------------------------------------------------------------------------- train mode
def cmd_train(a):
    t0 = time.time()
    fams = [f for f in a.families.split(",") if f]
    for f in fams:
        if f not in FAMILIES:
            raise SystemExit(f"unknown family {f!r}")
    ns = f"{TRAIN_NS}/{a.seed}"
    if ns == EVAL_NS:
        raise SystemExit("the train namespace must differ from the eval namespace")
    forbid = eval_fingerprints()
    if Path(a.eval_root, SET, "eval").exists():          # the built files must be what the generator regenerates
        built = {fingerprint(r["meta"]["family"], [tuple(o) for o in r["meta"]["options_levels"]]) for r in read_set(a.eval_root)}
        if built != forbid:
            raise SystemExit(f"the built eval set at {a.eval_root} differs from this generator's eval draws: rebuild it")
    stats = Counter()
    n_traj = -(-a.n_units // N_ROUNDS)
    recs, fps = [], set()
    for t in range(n_traj):
        fam = fams[t % len(fams)]
        rng = np.random.default_rng(seed_of(ns, fam, t))
        u = int(rng.integers(0, N_USERS))                 # users ~ the uniform prior (oracle = an unbiased draw of P)
        traj = gen_trajectory(fam, u, rng, forbid, stats)
        fps |= {fingerprint(fam, o) for o, _, _ in traj}
        for r in range(N_ROUNDS):
            recs.append(make_record(fam, t, u, traj, r, ns, train=True))
    recs = recs[: a.n_units]
    overlap = len(fps & forbid)
    assert overlap == 0
    eval_states = set()
    if Path(a.eval_root, SET, "eval").exists():
        eval_states = {r["state"] for r in read_set(a.eval_root)}
    same_state = sum(r["state"] in eval_states for r in recs)
    assert same_state == 0
    assert not nc_regex().search(LICENSE_TRAIN)
    rep = {"generator": VERSION, "seed_namespace": ns, "units": len(recs), "trajectories": n_traj,
           "families": dict(Counter(r["meta"]["family"] for r in recs)),
           "users_distinct": len({r["meta"]["user_index"] for r in recs}),
           "rounds_observed": dict(sorted(Counter(r["meta"]["rounds_observed"] for r in recs).items())),
           "option_sets_shared_with_eval": overlap, "states_identical_to_eval": same_state,
           "eval_option_sets_forbidden": len(forbid), "redraws": dict(stats),
           "exact_label_kind": dict(Counter(r["label_kind"][QID] for r in recs)),
           "variant_agreement": {
               "oracle == bayes_argmax": round(float(np.mean([r["meta"]["oracle_choice"] == r["meta"]["bayes_argmax"]
                                                              for r in recs])), 4),
               "mean exact mass on oracle": round(float(np.mean([r["meta"]["gold_exact"][KEYS.index(r["meta"]["oracle_choice"])]
                                                                 for r in recs])), 4)},
           "seconds": round(time.time() - t0, 1)}
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(with_variant(r, a.variant), ensure_ascii=False) + "\n")
        p.with_suffix(".manifest.json").write_text(json.dumps({**rep, "labels_variant": a.variant}, indent=1))
        log(f"train: wrote {len(recs)} units ({a.variant} labels; all three variants in meta.targets) to {p}")
    if a.arms_dir:
        for v in ("exact", "oracle", "bayes_argmax"):
            d = Path(a.arms_dir) / v / "train"
            d.mkdir(parents=True, exist_ok=True)
            files = {}
            for r in recs:
                fn = r["source"].replace("/", "__") + ".jsonl"
                if fn not in files:
                    files[fn] = (d / fn).open("w", encoding="utf-8")
                files[fn].write(json.dumps(with_variant(r, v), ensure_ascii=False) + "\n")
            for fh in files.values():
                fh.close()
        (Path(a.arms_dir) / "manifest.json").write_text(json.dumps(rep, indent=1))
        log(f"train: wrote arms exact / oracle / bayes_argmax under {a.arms_dir}")
    print(json.dumps(rep, indent=1))
    return rep


# --------------------------------------------------------------------------- validation
def paper_protocol(fam="flights", seeds=(0, 1, 2), n_eval=100, users=None):
    """The paper's simulated-user protocol: every user, 5 rounds of feedback, and after each round (0..5) the accuracy
    of a recommendation rule on `n_eval` fresh option sets (no feedback on those), against the user's own pick with
    uniform tie-breaking. Rules: the paper's Assistant (indicator likelihood + posterior-mean plug-in), the same with the
    tie-aware likelihood, and the Bayes action (mode of the exact posterior predictive). Expected accuracies, exact in
    the user's and the rule's tie-breaking."""
    L = nlev(fam)
    users = range(N_USERS) if users is None else users
    TH = THETA2.astype(float)
    acc = {k: np.zeros(N_ROUNDS + 1) for k in ("paper_assistant", "plugin_tieaware", "bayes_mode")}
    n = 0
    for s in seeds:
        for u in users:
            rng = np.random.default_rng(seed_of("paper_protocol", fam, s, u))
            traj = gen_trajectory(fam, u, rng)
            EV = np.stack([np.stack([phi_int(o, L) for o in draw_options(rng, fam)]) for _ in range(n_eval)])  # (n,3,4)
            R = np.einsum("td,nkd->tnk", THETA2, EV)
            A = R == R.max(2, keepdims=True)
            C = A / A.sum(2, keepdims=True)                                        # (624, n, 3)
            truth = C[u]
            w_t, w_i = np.ones(N_USERS), np.ones(N_USERS)
            for r in range(N_ROUNDS + 1):
                if r:
                    Ch, Ah = choice_matrix(traj[r - 1][1])
                    c = traj[r - 1][2]
                    w_t, w_i = w_t * Ch[:, c], w_i * Ah[:, c]
                for name, w, rule in (("paper_assistant", w_i, "plug"), ("plugin_tieaware", w_t, "plug"),
                                      ("bayes_mode", w_t, "mode")):
                    q = w / w.sum()
                    S = np.einsum("nkd,d->nk", EV.astype(float), q @ TH) if rule == "plug" else np.einsum("t,tnk->nk", q, C)
                    top = np.isclose(S, S.max(1, keepdims=True), rtol=0, atol=1e-12)
                    acc[name][r] += float(((top / top.sum(1, keepdims=True)) * truth).sum(1).mean())
            n += 1
    return {k: [round(float(100 * x / n), 2) for x in v] for k, v in acc.items()}, n


def write_npz(path, units, P):
    k = np.array([len(u["target"]) for u in units])
    T = np.zeros((len(units), 3))
    for i, u in enumerate(units):
        t = np.asarray(u["target"], float)
        T[i, : len(t)] = t / t.sum()
    np.savez_compressed(path, test_p=P, test_t=T, test_ks=k, test_soft=np.array([u["label_kind"] == "soft" for u in units]),
                        test_qid=np.array([str(u["qid"]) for u in units]),
                        test_source=np.array([str(u["source"]) for u in units]))


def ref_predictions(units, recs_by_id):
    """Reference predictors (no model): `exact` (the ceiling), `first_round_only` (the exact predictive from round 1's
    pick alone: a reader that updates once and then plateaus, the paper's description of untuned LLMs), `no_update` (the
    prior predictive of the current options: history ignored) and `uniform`."""
    exact, first, none = [], [], []
    for u in units:
        m = recs_by_id[u["record_id"]]["meta"]
        L = m["feature_levels"]
        F = np.stack([phi_int(o, L) for o in m["options_levels"]])
        Fh = [np.stack([phi_int(o, L) for o in h["options_levels"]]) for h in m["history"]]
        ch = [KEYS.index(h["pick"]) for h in m["history"]]
        exact.append(m["gold_exact"])
        none.append(predictive(posterior([], []), F))
        first.append(predictive(posterior(Fh[:1], ch[:1]), F))
    return {"uniform": np.full((len(units), 3), 1 / 3), "no_update": np.array(none),
            "first_round_only": np.array(first), "exact": np.array(exact)}


LOADER_CHECK = r'''
import importlib.util, json, os, sys, collections
repo, data = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("ls", os.path.join(repo, "trainers", "claude_lmscore.py"))
ls = importlib.util.module_from_spec(spec); spec.loader.exec_module(ls)
import torch, transformers
units = list(ls.cf.iter_units(data, ["eval"], set(), None))
out = {"torch": torch.__version__, "transformers": transformers.__version__, "transformers_path": transformers.__file__,
       "units": len(units), "by_source": dict(collections.Counter(u["source"] for u in units)),
       "k": dict(collections.Counter(len(u["option_keys"]) for u in units)),
       "label_kind": dict(collections.Counter(u["label_kind"] for u in units)),
       "nc_units": sum(ls.cf.is_noncommercial(u) for u in units),
       "base_ids": len({u["base_id"] for u in units})}
tok_path = sys.argv[3] if len(sys.argv) > 3 else ""
if tok_path:
    tok = transformers.AutoTokenizer.from_pretrained(tok_path)
    lens, none = [], 0
    for i, u in enumerate(units):
        for order in ([0, 1, 2], [2, 0, 1]):
            t = ls.build_prompt(tok, u, order, 4096)
            if t is None:
                none += 1; continue
            lens.append(len(tok(t, add_special_tokens=False)["input_ids"]))
            if i == 4 and order == [2, 0, 1]:
                out["example_prompt_r4_permuted"] = t
    lens.sort()
    out["prompt_tokens"] = {"tokenizer": os.path.basename(os.path.dirname(os.path.dirname(tok_path.rstrip("/")))),
                            "median": lens[len(lens) // 2], "max": lens[-1], "over_4096": sum(x > 4096 for x in lens),
                            "prompts": len(lens), "state_cut_none": none}
print("LOADER_JSON " + json.dumps(out))
'''


def cmd_validate(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    recs = read_set(a.out)
    by_id = {r["id"]: r for r in recs}
    res = {"when": datetime.now().isoformat(timespec="seconds")}

    # (1) exact posterior: the stored numpy gold vs the independent rational implementation, on 200 units
    t0 = time.time()
    rng = np.random.default_rng(seed_of("validate", "sample"))
    by_cell = defaultdict(list)
    for r in recs:
        by_cell[(r["meta"]["family"], r["meta"]["rounds_observed"])].append(r)
    sample = []
    for ci, key in enumerate(sorted(by_cell)):                       # 15 cells: 5 x 14 + 10 x 13 = 200, stratified
        v = by_cell[key]
        sample += [v[i] for i in rng.choice(len(v), 14 if ci < 5 else 13, replace=False)]
    assert len(sample) == 200
    dmax, dmax_lab = 0.0, 0.0
    for r in sample:
        m = r["meta"]
        hist_l = [[tuple(o) for o in h["options_levels"]] for h in m["history"]]
        hist_c = [KEYS.index(h["pick"]) for h in m["history"]]
        ex = exact_predictive_rational(m["family"], hist_l, hist_c, [tuple(o) for o in m["options_levels"]])
        exf = [x.numerator / x.denominator for x in ex]
        dmax = max(dmax, max(abs(x - y) for x, y in zip(exf, m["gold_exact"])))
        dmax_lab = max(dmax_lab, max(abs(x - r["labels"][QID][k]) for x, k in zip(exf, KEYS)))
    res["exact_posterior_crosscheck"] = {
        "units": len(sample), "cells": len(by_cell),
        "max_abs_diff_numpy_vs_rational": dmax, "pass_lt_1e-9": dmax < 1e-9,
        "max_abs_diff_rounded_labels_vs_rational": dmax_lab, "seconds": round(time.time() - t0, 1)}
    log(f"(1) exact cross-check: max |numpy - rational| = {dmax:.3g} on {len(sample)} units; rounded labels {dmax_lab:.3g}")
    assert dmax < 1e-9

    # (2) Bayes accuracy vs rounds: on the eval set, and under the paper's protocol
    curve = {}
    for fam in FAMILY_ORDER:
        rows = {}
        for rr in range(N_ROUNDS):
            v = by_cell[(fam, rr)]
            o = [KEYS.index(r["meta"]["oracle_choice"]) for r in v]
            bset = [[KEYS.index(k) for k in r["meta"]["bayes_argmax_set"]] for r in v]
            pset = [[KEYS.index(k) for k in r["meta"]["paper_assistant_pick_set"]] for r in v]
            rows[f"r{rr}"] = {
                "n": len(v),
                "bayes_mode_acc_vs_oracle": round(100 * float(np.mean([(oo in s) / len(s) for oo, s in zip(o, bset)])), 1),
                "bayes_mode_expected_acc": round(100 * float(np.mean([max(r["meta"]["gold_exact"]) for r in v])), 1),
                "paper_assistant_acc_vs_oracle": round(100 * float(np.mean([(oo in s) / len(s) for oo, s in zip(o, pset)])), 1),
                "paper_assistant_expected_acc": round(100 * float(np.mean(
                    [np.mean([r["meta"]["gold_exact"][i] for i in s]) for r, s in zip(v, pset)])), 1),
                "mean_label_entropy": round(float(np.mean([r["meta"]["label_entropy_nats"] for r in v])), 4)}
        curve[fam] = rows
    t0 = time.time()
    proto, n_runs = paper_protocol("flights")
    proto_h, n_runs_h = paper_protocol("hotels")
    res["accuracy_vs_rounds"] = {
        "eval_set": curve,
        "paper_protocol_flights": {"runs": n_runs, "eval_sets_per_round": 100, "rounds_0_to_5": proto,
                                   "paper_reported_bayesian_assistant": {"after_round_1": 58, "after_round_5": 81}},
        "paper_protocol_hotels": {"runs": n_runs_h, "rounds_0_to_5": proto_h},
        "seconds_protocol": round(time.time() - t0, 1)}
    log(f"(2) flights paper protocol (rounds 0..5): {proto}")
    log(f"    eval-set Bayes mode vs oracle, flights r0..r4: "
        f"{[curve['flights'][f'r{i}']['bayes_mode_acc_vs_oracle'] for i in range(N_ROUNDS)]}")

    # (3) gde1_score.py with a uniform predictor and the exact posterior as predictor
    gs = load_module("gde1_score", HERE / "analysis" / "gde1_score.py")
    cf = gs.load_cf()
    units = gs._quiet(lambda: list(cf.iter_units(str(Path(a.out) / SET), ["eval"], set(), None)))
    refs = ref_predictions(units, by_id)
    paths = {}
    for name, P in refs.items():
        paths[name] = work / f"pred_{name}_{{set}}.npz"
        write_npz(str(paths[name]).replace("{set}", SET), units, P)
    rep_path = work / "gde1_score_report.json"
    cmd = [sys.executable, str(HERE / "analysis" / "gde1_score.py"), "--root", a.out, "--sets", SET,
           "--pred", f"uniform_npz={paths['uniform']}", "--pred", f"exact={paths['exact']}",
           "--pred", f"no_update={paths['no_update']}", "--pred", f"first_round_only={paths['first_round_only']}",
           "--ref", "uniform_npz",
           "--out", str(rep_path), "--boot", str(a.boot), "--floor-draws", str(a.floor_draws)]
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True)
    (work / "gde1_score_stdout.txt").write_text(pr.stdout + pr.stderr)
    if pr.returncode:
        raise SystemExit(f"gde1_score failed:\n{pr.stderr[-3000:]}")
    sr = json.loads(rep_path.read_text())
    ps = sr["per_set"]
    floor = float(np.mean([1 - sum(v * v for v in r["labels"][QID].values()) for r in recs]))
    res["gde1_score"] = {
        "command": " ".join(cmd), "seconds": round(time.time() - t0, 1),
        "notes": sr["notes"],
        **{m: {k: ps[m][SET][k] for k in ("n", "ls", "lss", "bs", "bss", "kl", "acc", "invalid_rows", "coverage")}
           for m in ("uniform", "prior", "uniform_npz", "exact", "no_update", "first_round_only")},
        "brier_floor_mean_1_minus_sum_r2": floor,
        "exact_brier_minus_floor": ps["exact"][SET]["bs"] - floor,
        "exact_lss_gt_0": ps["exact"][SET]["lss"] > 0,
        "exact_calibration_toplabel_smece": ps["exact"][SET]["calibration_toplabel"]["smece"],
        "paired_exact_minus_uniform_npz": sr["paired"].get("exact - uniform_npz", {}).get("per_set", {}).get(SET)}
    log(f"(3) gde1_score: exact LSS {ps['exact'][SET]['lss']:+.4f}, Brier {ps['exact'][SET]['bs']:.6f} vs floor "
        f"{floor:.6f}; uniform LSS {ps['uniform'][SET]['lss']:+.4f}")
    assert ps["exact"][SET]["lss"] > 0 and abs(ps["exact"][SET]["bs"] - floor) < 1e-5
    assert abs(ps["uniform_npz"][SET]["ls"] - ps["uniform"][SET]["ls"]) < 1e-12

    # family x rounds table for the two reference predictors (the ceiling and the floor of every later readout)
    res["family_by_rounds_reference"] = family_round_table(gs, a.out, paths, boot=a.boot)

    # (4) claude_lmscore's unit loader on CPU (real torch, transformers 5.12.1 from /tmp/tf512)
    tok = a.tokenizer if a.tokenizer and Path(a.tokenizer).exists() else ""
    env = dict(os.environ, PYTHONPATH=f"{a.tf_path}:{HERE}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    py = a.lmscore_python
    pr = subprocess.run([py, "-c", LOADER_CHECK, str(HERE), str(Path(a.out) / SET), tok], capture_output=True, text=True,
                        env=env)
    line = next((ln for ln in pr.stdout.splitlines() if ln.startswith("LOADER_JSON ")), None)
    if pr.returncode or not line:
        raise SystemExit(f"loader check failed:\n{pr.stdout[-2000:]}\n{pr.stderr[-3000:]}")
    lj = json.loads(line[len("LOADER_JSON "):])
    ex = lj.pop("example_prompt_r4_permuted", None)
    if ex:
        (work / "example_prompt_r4_permuted.txt").write_text(ex)
    res["lmscore_loader"] = {"python": py, "PYTHONPATH": env["PYTHONPATH"], **lj}
    log(f"(4) lmscore loader: {lj['units']} units, K {lj['k']}, transformers {lj['transformers']}, "
        f"tokens {lj.get('prompt_tokens')}")
    assert lj["units"] == len(recs) and lj["nc_units"] == 0

    (work / "validation.json").write_text(json.dumps(res, indent=1, default=float))
    root = Path(a.out) / SET
    man = json.loads((root / "manifest.json").read_text())
    man["validation"] = res
    (root / "manifest.json").write_text(json.dumps(man, indent=1, default=float, ensure_ascii=False))
    cr = json.loads((root / "check_report.json").read_text())
    cr["validation_summary"] = {
        "exact_posterior_max_abs_diff": dmax,
        "flights_paper_protocol_rounds_0_to_5": proto,
        "exact_lss": ps["exact"][SET]["lss"], "exact_brier_minus_floor": ps["exact"][SET]["bs"] - floor,
        "uniform_lss": ps["uniform"][SET]["lss"], "lmscore_loader_units": lj["units"]}
    (root / "check_report.json").write_text(json.dumps(cr, indent=1, default=float))
    log(f"validation written to {work / 'validation.json'} and the set's manifest / check_report")
    return res


# --------------------------------------------------------------------------- family x rounds readout
GAIN_PAIRS = ((0, 1), (1, 4))


def family_round_table(gs, root, preds, boot=2000, seed=20261001, floor=1e-3):
    """{model: {family: {rN: metrics, deltas}}} from lmscore-format npz files, aligned and scored exactly as
    gde1_score.py does (same Cromwell floor). Deltas, each with a cluster-bootstrap 95% CI (clusters = trajectories, so
    the r_a and r_b units of one user move together):
      ls_gain_rA_to_rB = LS(r_a) - LS(r_b)   the BELIEF-UPDATE statistic. The exact predictive is a martingale
                         (E[P_rb | first r_a rounds, current options] = P_ra), so ANY predictor that ignores rounds
                         r_a+1..r_b has the same EXPECTED log score at r_a and r_b: its gain is 0 in expectation, whatever
                         it does with the current options. The exact posterior's gain is the ceiling (about +0.20 for
                         0->1 and +0.40 for 1->4).
      share_of_exact_gain = gain(model) / gain(exact), on the same resampled clusters.
      kl_rB_minus_rA, acc_gold_rB_minus_rA, acc_oracle_rB_minus_rA (plain differences)."""
    cf = gs.load_cf()
    sd = gs.SetData(cf, root, SET, "eval")
    meta = {}
    for p in sorted((Path(root) / SET / "eval").glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            meta[r["id"]] = r["meta"]
    fam = np.array([meta[u["record_id"]]["family"] for u in sd.units])
    rr = np.array([meta[u["record_id"]]["rounds_observed"] for u in sd.units])
    orc = np.array([KEYS.index(meta[u["record_id"]]["oracle_choice"]) for u in sd.units])
    ex_sc, _ = gs.score_model(sd, np.array([meta[u["record_id"]]["gold_exact"] for u in sd.units]), np.arange(sd.n), floor)
    out = {}
    for name, path in preds.items():
        z = np.load(str(path).replace("{set}", SET), allow_pickle=True)
        ui, ri = gs.align(sd, z)
        Q = np.asarray(z["test_p"], float)[ri]
        sc, bad = gs.score_model(sd, Q, ui, floor)
        top = np.isclose(Q, Q.max(1, keepdims=True), rtol=0, atol=1e-12)
        acc_or = top[np.arange(len(ui)), orc[ui]] / top.sum(1)
        cols = {"ls": sc["ls"], "kl": sc["kl"], "acc_gold": sc["acc"], "acc_oracle": acc_or, "ls_exact": ex_sc["ls"][ui]}
        tab = {"invalid_rows": bad, "coverage": len(ui) / sd.n}
        for f in FAMILY_ORDER + ("all",):
            sel_f = np.ones(len(ui), bool) if f == "all" else fam[ui] == f
            cells = {}
            for k in range(N_ROUNDS):
                s_ = sel_f & (rr[ui] == k)
                if s_.any():
                    cells[f"r{k}"] = {"n": int(s_.sum()), "kl": float(sc["kl"][s_].mean()), "ls": float(sc["ls"][s_].mean()),
                                      "brier": float(sc["bs"][s_].mean()), "acc_gold": float(sc["acc"][s_].mean()),
                                      "acc_oracle": float(acc_or[s_].mean()), "max_p": float(sc["s"][s_].mean())}
            for ra, rb in GAIN_PAIRS:
                cells.update(_cluster_deltas(sd.cl[ui], rr[ui], sel_f, ra, rb, cols, boot, seed))
            tab[f] = cells
        out[name] = tab
    return out


def _cluster_deltas(cl, rr, sel, ra, rb, cols, boot, seed):
    """Paired r_a -> r_b statistics over the clusters (trajectories) that have both units; bootstrap over clusters."""
    ia = {c: i for i, c in zip(np.flatnonzero(sel & (rr == ra)), cl[sel & (rr == ra)])}
    ib = {c: i for i, c in zip(np.flatnonzero(sel & (rr == rb)), cl[sel & (rr == rb)])}
    both = sorted(set(ia) & set(ib))
    if not both:
        return {}
    A = np.array([ia[c] for c in both])
    B = np.array([ib[c] for c in both])
    n = len(both)
    idx = np.random.default_rng([seed, ra, rb, n]).integers(0, n, (boot, n)) if boot else None

    def est(fn):
        pt = float(fn(np.arange(n)))
        if idx is None:
            return {"delta": pt, "n_clusters": n}
        draws = np.array([fn(j) for j in idx])
        return {"delta": pt, "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                "n_clusters": n}

    ls, lse = cols["ls"], cols["ls_exact"]
    out = {f"ls_gain_r{ra}_to_r{rb}": est(lambda j: ls[A[j]].mean() - ls[B[j]].mean()),
           f"share_of_exact_gain_r{ra}_to_r{rb}": est(
               lambda j: (ls[A[j]].mean() - ls[B[j]].mean()) / max(lse[A[j]].mean() - lse[B[j]].mean(), 1e-9))}
    for m in ("kl", "acc_gold", "acc_oracle"):
        v = cols[m]
        out[f"{m}_r{rb}_minus_r{ra}"] = est(lambda j, v=v: v[B[j]].mean() - v[A[j]].mean())
    return out


def print_table(tab):
    for name, t in tab.items():
        print(f"\n== {name}  (invalid rows {t['invalid_rows']}, coverage {t['coverage']:.3f})")
        print(f"{'family':8s} {'r':>2s} {'n':>5s} {'KL':>7s} {'LS':>7s} {'Brier':>7s} {'acc_gold':>8s} {'acc_orc':>8s} {'max_p':>6s}")
        for f in FAMILY_ORDER + ("all",):
            for k in range(N_ROUNDS):
                c = t[f].get(f"r{k}")
                if c:
                    print(f"{f:8s} {k:2d} {c['n']:5d} {c['kl']:7.4f} {c['ls']:7.4f} {c['brier']:7.4f} {c['acc_gold']:8.3f} "
                          f"{c['acc_oracle']:8.3f} {c['max_p']:6.3f}")
            for m, v in t[f].items():
                if isinstance(v, dict) and "delta" in v:
                    ci = f" [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" if "ci95" in v else ""
                    print(f"{f:8s}    {m:34s} {v['delta']:+.4f}{ci}")


def cmd_report(a):
    gs = load_module("gde1_score", HERE / "analysis" / "gde1_score.py")
    preds = dict(p.split("=", 1) for p in a.pred)
    if a.with_refs:
        cf = gs.load_cf()
        units = gs._quiet(lambda: list(cf.iter_units(str(Path(a.root) / SET), ["eval"], set(), None)))
        by_id = {r["id"]: r for r in read_set(a.root)}
        Path(a.work).mkdir(parents=True, exist_ok=True)
        for name, P in ref_predictions(units, by_id).items():
            p = Path(a.work) / f"pred_{name}_{SET}.npz"
            write_npz(p, units, P)
            preds = {name: str(p), **preds}
    tab = family_round_table(gs, a.root, preds, boot=a.boot)
    print_table(tab)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(tab, indent=1))


# --------------------------------------------------------------------------- self-test
def self_test():
    # users and the scale
    assert N_USERS == 624 and len({tuple(t) for t in THETA2}) == 624 and not (THETA2 == 0).all(1).any()
    for fam in FAMILY_ORDER:
        for name, L, rend in FAMILIES[fam]["features"]:
            assert len({rend(j) for j in range(L)}) == L, (fam, name)
        assert phi_int([L - 1 for L in nlev(fam)], nlev(fam)).tolist() == [GRID] * 4
    # ties: three items equal on every feature but the first; a user who only weighs feature 2 picks uniformly
    F = np.array([[0, 4, 10, 6], [10, 4, 10, 6], [20, 4, 10, 6]])
    C, A = choice_matrix(F)
    u = int(np.flatnonzero((THETA2 == [0, 1, 0, 0]).all(1))[0])
    assert np.allclose(C[u], 1 / 3) and A[u].all()
    u2 = int(np.flatnonzero((THETA2 == [2, 0, 0, 0]).all(1))[0])
    assert np.allclose(C[u2], [0, 0, 1])
    # the true user is never ruled out; numpy == rational on random units of every family
    rng = np.random.default_rng(7)
    for fam in FAMILY_ORDER:
        for _ in range(4):
            u = int(rng.integers(N_USERS))
            traj = gen_trajectory(fam, u, rng)
            for r in range(N_ROUNDS):
                post = posterior([t[1] for t in traj[:r]], [t[2] for t in traj[:r]])
                assert post[u] > 0
                P = predictive(post, traj[r][1])
                ex = exact_predictive_rational(fam, [t[0] for t in traj[:r]], [t[2] for t in traj[:r]], traj[r][0])
                assert max(abs(x - float(y)) for x, y in zip(P, ex)) < 1e-12
                assert P[traj[r][2]] > 0
    # the state never shows the current pick, and shows every earlier one
    fam, u = "trails", 5
    traj = gen_trajectory(fam, u, np.random.default_rng(3))
    rec = make_record(fam, 0, u, traj, 3, "selftest")
    assert rec["state"].count("The hiker picked Trail") == 3 and "Round 4 (the current round)" in rec["state"]
    assert abs(sum(rec["labels"][QID].values()) - 1) < 1e-9
    trec = make_record(fam, 0, u, traj, 3, "selftest/train", train=True)
    for v in ("exact", "oracle", "bayes_argmax"):
        vr = with_variant(trec, v)
        assert vr["state"] == trec["state"] and vr["id"] == trec["id"]
    assert not nc_regex().search(LICENSE)
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("eval")
    p.add_argument("--out", default=OUT)
    p = sub.add_parser("check")
    p.add_argument("--out", default=OUT)
    p = sub.add_parser("validate")
    p.add_argument("--out", default=OUT)
    p.add_argument("--work", default=WORK)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--floor-draws", type=int, default=200)
    p.add_argument("--tf-path", default="/tmp/tf512")
    p.add_argument("--lmscore-python", default=str(HERE / ".venv" / "bin" / "python3"))
    p.add_argument("--tokenizer", default=next(iter(sorted(Path.home().glob(
        ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*"))), Path("/nonexistent")).as_posix(),
        help="a LOCAL tokenizer dir to render prompts through claude_lmscore.build_prompt (length check only)")
    p = sub.add_parser("train")
    p.add_argument("--n-units", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--families", default="flights,hotels,trails")
    p.add_argument("--variant", choices=("exact", "oracle", "bayes_argmax"), default="exact",
                   help="which target goes into `labels` of --out (all three are always in meta.targets)")
    p.add_argument("--out", default="", help="one jsonl")
    p.add_argument("--arms-dir", default="", help="DIR/{exact,oracle,bayes_argmax}/train/<source>.jsonl: identical records")
    p.add_argument("--eval-root", default=OUT, help="the built eval set, for the identical-state check")
    p = sub.add_parser("report")
    p.add_argument("--root", default=OUT)
    p.add_argument("--pred", action="append", default=[], metavar="NAME=NPZ")
    p.add_argument("--with-refs", action="store_true", help="also print the uniform and exact reference rows")
    p.add_argument("--work", default=WORK)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out-json", default="")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.cmd == "eval":
        cmd_eval(a)
    elif a.cmd == "check":
        cmd_check(a)
    elif a.cmd == "validate":
        cmd_validate(a)
    elif a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "report":
        cmd_report(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
