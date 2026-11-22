#!/usr/bin/env python3
"""bayes_zen_convert.py (stock_decision_v1, BAYES-DATA): the Zenodo release of Qiu et al. 2026 ("LLM Bayesian Teaching
Data", record 17677329, CC BY 4.0; article "Bayesian teaching enables probabilistic reasoning in large language models",
Nat Commun, doi:10.1038/s41467-025-67998-6) -> one-pass, one-question, terminal-only decisions with exact targets.

Only the DATA (option values, user choices, profiles) is used. The article is CC BY-NC-ND: none of its text or prompts is
reproduced. Every state, question and option line below is our own wording (canonical form, regenerated from parsed values).

  convert  --raw RAW/data --out OUT --tokenizer TOKDIR --scorer-dir DIR [--genholdout DIR] [--procs 4]
  uarms    --ublock OUT/u_block --base pilot_M1.parquet|EXPOSURES.jsonl[.gz] [--id-field uid --tokens-field tok_r2
           --k-field K --pos-field pos] --out DIR
  selftest

LATENT SPACE. 624 profiles: weights in {-1,-0.5,0,+0.5,+1}^4 minus all-zero (the released idx 0..623 map 1:1 onto these:
eval/interaction/<domain>.jsonl reward_fn; verified: every one of the 156,000 observed training choices is in the argmax set
of its idx's reward_fn). Uniform prior (each idx has exactly 10 training episodes). Features on the release's own 0..1 grids,
stored x20 as exact integers (flights 11/11/3/11 levels, hotels 11/11/5/5), so rewards and ties are exact.

DECLARED MODEL (stated in every state, so the gold is exact for what the prompt says). v2: for the simulated release
(train / dev_profile / dev_flight_eval / dev_hotel) the state declares the RELEASE PROCESS: items drawn uniformly per
feature level, three distinct, and a set redrawn whenever two or more items tie for the person's best, so every shown set
has one best item; the person picks it. Likelihood per shown set 1[argmax_theta = {c}] / p_u(theta) (current set: the
uniqueness filter only). Primary targets come from this posterior, so the realised picks (U1) are exact draws from the
U3 target. The unfiltered tie-normalised predictive (P_tie, the v1 primary) stays stored in meta.variants.
Human file and our generator: unfiltered (their processes do not filter), tie-normalised / eps-mixture as before.
v1 DECLARED MODEL: option sets drawn at random; the person picks the highest weighted sum; ties broken uniformly
(likelihood 1[c in A]/|A|, the TIE-NORMALISED convention).
Recorded variants, per decision, never used as labels: P_ind (the paper's compatibility indicator 1[c in A] in the
posterior), P_sel (the RELEASE's actual process: training/eval option sets were filtered so the true user's best option is
unique; 0 ties in 156,000 training choices vs 5.3% tie sets in the unfiltered held-out sets; likelihood 1[A_theta={c}]/p_u
(theta), p_u = P(unique max | theta) for 3 distinct uniform items). Human file: declared eps-mixture likelihood
(1-eps)*1[c in A]/|A| + eps/3, eps = 0.15 (one scalar, maximum marginal likelihood on the human file, which is dev-only).

THREE TARGET BASES, each its own question, one question per record:
  (a) weight_<feature>  score K=5   exact posterior marginal of that feature's weight          target_basis exact_posterior
  (b) next_choice       choice K=3  exact posterior predictive P(next pick = j | history, options) outcome_probability
  (c) recommend         choice K=3  argmax_o E_post[w . phi(o)] (expected-utility action; exact
                                    ties share the mass equally: r=0 is a 3-way tie by symmetry)  optimal_action
U1 uses (b)'s question with the REALISED next pick (one-hot, a sampled outcome; target_basis observed_outcome).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "bayes_zen_convert v2.3 (2026-10-14)"
NS = "stock_decision_v1/bayes_zen/v1"
ZEN = {"record": "https://zenodo.org/records/17677329", "file": "data.zip", "md5": "a341ca74418ed9152a31cc3f91e261dd",
       "bytes": 54046098, "license": "CC-BY-4.0", "title": "LLM Bayesian Teaching Data", "creator": "Qiu, Linlu",
       "publication_date": "2025-11-22"}
ATTRIBUTION = ("Qiu, Sha, Allen, Kim, Linzen & van Steenkiste 2026, 'Bayesian teaching enables probabilistic reasoning in "
               "large language models', Nature Communications, doi:10.1038/s41467-025-67998-6; data: Zenodo record "
               "17677329 (CC BY 4.0)")
LICENSE = ("CC BY 4.0: Zenodo record 17677329 (Qiu et al. 2026, Nature Communications; LLM Bayesian Teaching Data); values "
           "converted and re-rendered by jev, all text ours")
SOURCE_FAMILY = "zenodo17677329_bayesian_teaching"
GEN_FAMILY = "jev_bayes_pref_generator"
EPS_HUMAN = 0.15
N_ROUNDS = 5
TOK_REPO, TOK_REV = "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"
SCORER_MD5, CF_MD5 = "e0672bd0f515ccb17d97035c84c3d339", "49613a40faed8cd612175e81c3f3925f"
MAX_LEN = 8192
N_DEV_PROFILES = 124
U_EPISODES = 384                      # 384 episodes x 5 prefixes = 1,920 = 5% of 38,400 pilot exposures
WEIGHT_KEYS = ("-1", "-0.5", "0", "+0.5", "+1")    # index = 2*w + 2

W5 = (-2, -1, 0, 1, 2)
TH = np.array([t for t in itertools.product(W5, repeat=4) if any(t)], dtype=np.int64)     # 2 * weights
NTH = len(TH)
assert NTH == 624


# ============================================================================ domains (all wording ours)
def _clock(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _dur(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{m} min" if h == 0 else f"{h} h {m:02d} min"


AMEN = ("no amenities", "free parking", "free parking and free breakfast", "free parking, free breakfast and a pool",
        "free parking, free breakfast, a pool and a fitness center")

DOM = {
    "flights": {
        "nlev": (11, 11, 3, 11), "mult": (2, 2, 10, 2),
        "features": ("departure time", "time in transit", "stops", "fare"),
        "fkeys": ("departure_time", "transit_time", "stops", "fare"),
        "person": "customer", "item": "flight", "Item": "Flight", "past": "booked", "verb": "book", "ing": "booking",
        "intro": ("A booking assistant serves one customer who flies the same route again and again. In each round the "
                  "assistant offers three flights and the customer books exactly one of them."),
        "scales": ["- departure time: 6:00 AM scores 0, and each later slot (every 1 h 36 min) adds 0.1, up to 10:00 PM = 1;",
                   "- time in transit: 30 min scores 0, and each further 1 h 57 min adds 0.1, up to 20 h 00 min = 1;",
                   "- stops: nonstop scores 0, 1 stop 0.5, 2 stops 1;",
                   "- fare: $100 scores 0, and each further $90 adds 0.1, up to $1,000 = 1."],
        "line": lambda l: (f"departs {_clock(360 + 96 * l[0])}, {_dur(30 + 117 * l[1])} in transit, "
                           f"{('nonstop', '1 stop', '2 stops')[l[2]]}, ${100 + 90 * l[3]:,}"),
    },
    "hotels": {
        "nlev": (11, 11, 5, 5), "mult": (2, 2, 5, 5),
        "features": ("distance from downtown", "nightly price", "star rating", "amenities"),
        "fkeys": ("distance", "price", "rating", "amenities"),
        "person": "guest", "item": "hotel", "Item": "Hotel", "past": "booked", "verb": "book", "ing": "booking",
        "intro": ("A travel assistant serves one guest who needs a hotel on trip after trip. In each round the assistant "
                  "offers three hotels and the guest books exactly one of them."),
        "scales": ["- distance from downtown: 0 miles scores 0, and each further mile adds 0.1, up to 10 miles = 1;",
                   "- nightly price: $100 scores 0, and each further $90 adds 0.1, up to $1,000 = 1;",
                   "- star rating: 1 star scores 0, 2 stars 0.25, 3 stars 0.5, 4 stars 0.75, 5 stars 1;",
                   "- amenities: none scores 0, free parking 0.25, free parking and free breakfast 0.5, those plus a pool "
                   "0.75, and those plus a fitness center 1."],
        "line": lambda l: (f"{l[0]} mile{'' if l[0] == 1 else 's'} from downtown, ${100 + 90 * l[1]:,} per night, "
                           f"{l[2] + 1} star{'' if l[2] == 0 else 's'}, {AMEN[l[3]]}"),
    },
}


def phi20(dom, levels):
    return np.array([j * m for j, m in zip(levels, DOM[dom]["mult"])], dtype=np.int64)


def model_text(dom, noisy_eps=None, filtered=False):
    D = DOM[dom]
    p, it, its = D["person"], D["item"], D["item"] + "s"
    rule = (f"In each round the {p} {D['verb']}s the {it} with the largest weighted sum of its four scores; when two or "
            f"three {its} share the largest sum, the {p} {D['verb']}s one of them at random, each equally likely.")
    draw = f"The three {its} of a round are drawn at random."
    if filtered:
        draw = (f"Each {it}'s four features are drawn independently, every level of each feature equally likely, and the "
                f"three {its} of a round are all different; a set is thrown away and redrawn whenever two or more of its "
                f"{its} would share the {p}'s largest weighted sum, so every set shown has exactly one best {it} for this "
                f"{p}.")
        rule = f"In each round the {p} {D['verb']}s that best {it}, the one with the largest weighted sum of its four scores."
    if noisy_eps is not None:
        rule = (f"This {p} is a real person and does not always follow the rule exactly: in each round, with probability "
                f"{1 - noisy_eps:g} they {D['verb']} the {it} with the largest weighted sum of its four scores (one of the "
                f"tied {its} at random when several share it), and with probability {noisy_eps:g} they {D['verb']} any of "
                f"the three {its} at random, each equally likely.")
    return (f"How the {p} chooses. Every {it} gets four scores between 0 and 1:\n" + "\n".join(D["scales"]) + "\n"
            f"The {p} holds a fixed weight for each of the four scores. Each weight is -1, -0.5, 0, +0.5 or +1, and the four "
            f"are never all 0. A negative weight means the {p} wants a low score on that feature, a positive weight a high "
            f"score, and 0 that the feature does not matter to them. Before any choice is seen, each of the 624 possible "
            f"weight combinations is equally likely. {draw} {rule} The weights never change from round to round.")


def round_lines(dom, opts):
    D = DOM[dom]
    return "\n".join(f"  {D['Item']} {i + 1}: {D['line'](o)}" for i, o in enumerate(opts))


def render_state(dom, hist, cur, noisy_eps=None, filtered=False):
    """hist: [(opts, choice)], cur: opts. The current round's pick never appears."""
    D = DOM[dom]
    parts = [D["intro"], "", model_text(dom, noisy_eps, filtered), ""]
    if not hist:
        parts.append("Earlier rounds: none yet; the current round is the first.")
    else:
        parts.append(f"Earlier rounds ({len(hist)}):")
        for t, (opts, c) in enumerate(hist):
            parts.append(f"Round {t + 1}. Offered:\n{round_lines(dom, opts)}\n  The {D['person']} {D['past']} "
                         f"{D['Item']} {c + 1}.")
    parts += ["", f"Round {len(hist) + 1}, the current round. Offered:\n{round_lines(dom, cur)}"]
    return "\n".join(parts)


def okeys(dom):
    return [f"{DOM[dom]['item']}_{i + 1}" for i in range(3)]


def q_next(dom, cur):
    D = DOM[dom]
    return {"type": "choice", "instructions": f"Which {D['item']} will the {D['person']} {D['verb']} in the current round?",
            "criteria": {k: f"{D['Item']} {i + 1}: {D['line'](o)}" for i, (k, o) in enumerate(zip(okeys(dom), cur))}}


def q_rec(dom, cur):
    D = DOM[dom]
    ins = (f"The assistant has to recommend one {D['item']} from the current round. A recommendation is worth the "
           f"{D['person']}'s weighted sum of the four scores of that {D['item']}, computed with their true weights, whatever "
           f"they end up {D['ing']}. Which {D['item']} has the highest expected worth, given everything observed so far?")
    return {"type": "choice", "instructions": ins,
            "criteria": {k: f"{D['Item']} {i + 1}: {D['line'](o)}" for i, (k, o) in enumerate(zip(okeys(dom), cur))}}


def q_weight(dom, f):
    D = DOM[dom]
    return {"type": "score", "instructions": f"What weight does the {D['person']} place on the {D['features'][f]} score?",
            "levels": list(WEIGHT_KEYS)}


# ============================================================================ exact inference
def choice_sets(dom, opts):
    F = np.stack([phi20(dom, o) for o in opts])          # (3, 4)
    R = TH @ F.T                                         # (624, 3) exact integers (= 40 * w . phi)
    A = R == R.max(1, keepdims=True)
    return R, A, A.sum(1)


_PU = {}


def p_unique(dom):
    """P(the max of 3 distinct uniformly drawn items is unique | theta), exact by counting."""
    if dom not in _PU:
        allf = np.array([phi20(dom, l) for l in itertools.product(*[range(n) for n in DOM[dom]["nlev"]])])
        N = len(allf)
        pu = np.zeros(NTH)
        for i, t in enumerate(TH):
            _, cnt = np.unique(allf @ t, return_counts=True)
            below = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(float)
            pu[i] = float((3 * cnt * below * (below - 1)).sum()) / (N * (N - 1) * (N - 2))
        _PU[dom] = pu
    return _PU[dom]


def top_set(v, atol=0.0):
    v = np.asarray(v)
    m = v.max()
    return [int(i) for i in np.flatnonzero(v >= m - atol)]


def infer(dom, rounds, choices, r, eps=None, filtered=False, theta_true=None):
    """Everything about the terminal decision after r observed rounds (rounds[r] = current options)."""
    pu = p_unique(dom)
    w6 = np.ones(NTH, dtype=np.int64)                    # tie-normalised, exact: prod 6*1[c in A]/|A|
    wind = np.ones(NTH, dtype=np.int64)                  # compatibility indicator
    wsel = np.ones(NTH)                                  # release process (unique-max selection)
    weps = np.ones(NTH) if eps is not None else None
    for t in range(r):
        _, A, na = choice_sets(dom, rounds[t])
        c = choices[t]
        w6 = w6 * (6 * A[:, c] // na)
        wind = wind * A[:, c]
        wsel = wsel * (A[:, c] & (na == 1)) / pu
        if eps is not None:
            weps = weps * ((1 - eps) * A[:, c] / na + eps / 3)
    R, A, na = choice_sets(dom, rounds[r])
    C6 = 6 * A // na[:, None]                            # (624, 3) exact
    out = {"w6_sum": int(w6.sum())}
    if eps is None:
        wdecl = w6.astype(float)
    else:
        wdecl = weps
    if w6.sum() > 0:
        Pn = w6 @ C6
        out["P_tie"] = (Pn / Pn.sum()).tolist()
        out["pred_top_tie"] = top_set(Pn)
        EUn = w6 @ R
        out["eu_top_tie"] = top_set(EUn)
    else:
        out["P_tie"] = None                             # noiseless collapse (human file only)
    if wind.sum() > 0:
        q = wind / wind.sum()
        out["P_ind"] = (q @ (C6 / 6)).tolist()
        out["eu_top_ind"] = top_set(wind @ R)
    if eps is None:
        ws = wsel * ((na == 1) / pu)
        if ws.sum() > 0:
            out["P_sel"] = ((ws / ws.sum()) @ (C6 / 6)).tolist()
    if filtered:                                         # v2 primary for the simulated release: the release process
        wdecl = wsel * ((na == 1) / pu)
        out["eu_unfiltered"] = ((w6 / w6.sum()) @ R / 40.0).tolist()
    q = wdecl / wdecl.sum()
    P = q @ (C6 / 6)
    if eps is not None:
        # v2.2 fix: the declared eps process also emits the NEXT pick with noise: P = (1-eps) * P_pref + eps/3
        out["P_pref_noiseless_emission"] = P.tolist()     # diagnostic only: next pick as if the person never erred
        P = (1 - eps) * P + eps / 3
    eu = (q @ R) / 40.0                                  # expected w . phi, real units (unchanged by emission noise)
    out["P"] = P.tolist()
    out["eu"] = eu.tolist()
    if eps is None and not filtered:
        out["pred_top"], out["eu_top"] = out["pred_top_tie"], out["eu_top_tie"]
    else:
        out["pred_top"], out["eu_top"] = top_set(P, 1e-12), top_set(eu, 1e-12)
    out["marg"] = [[float(q[TH[:, f] == v].sum()) for v in W5] for f in range(4)]
    nz = q[q > 0]
    out["support"] = int((q > 0).sum())                  # v2.3: support of the declared posterior q (entropy uses q too)
    if eps is not None:
        out["support_noiseless"] = int((w6 > 0).sum())   # what a noiseless likelihood would leave (0 = collapse)
    if theta_true is not None:
        out["true_profile_posterior"] = float(q[theta_true])
    out["post_entropy"] = float(-(nz * np.log(nz)).sum())
    # references for the sanity eval: no_update (prior predictive), first_round_only
    cur_f = ((na == 1) / pu) if filtered else np.ones(NTH)           # the current set's own evidence (filtered only)
    out["ref_no_update"] = ((cur_f / cur_f.sum()) @ (C6 / 6)).tolist()
    if r >= 1:
        _, A1, na1 = choice_sets(dom, rounds[0])
        c1 = choices[0]
        if filtered:
            w1 = (A1[:, c1] & (na1 == 1)) / pu
        elif eps is not None:
            w1 = (1 - eps) * A1[:, c1] / na1 + eps / 3
        else:
            w1 = A1[:, c1] / na1
        w1 = w1 * cur_f
        out["ref_first_round_only"] = ((w1 / w1.sum()) @ (C6 / 6)).tolist()
    else:
        out["ref_first_round_only"] = out["ref_no_update"]
    if eps is not None:                                  # v2.2: references carry the same emission noise
        for k in ("ref_no_update", "ref_first_round_only"):
            out[k] = ((1 - eps) * np.asarray(out[k]) + eps / 3).tolist()
    return out


def ent(p):
    return float(-sum(x * math.log(x) for x in p if x > 0))


def lab(keys, vals):
    """6-dp labels summing to exactly 1 (the largest absorbs rounding)."""
    v = [max(0.0, float(x)) for x in vals]
    s = sum(v)
    v = [round(x / s, 6) for x in v]
    j = max(range(len(v)), key=lambda i: v[i])
    v[j] = round(v[j] + (1.0 - sum(v)), 6)
    return dict(zip(keys, v))


def split_mass(k, top):
    return [1.0 / len(top) if i in top else 0.0 for i in range(k)]


def kind_of(d):
    return "hard" if sum(1 for x in d.values() if x > 0) == 1 else "soft"


def sha256s(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hrank(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


# ============================================================================ parsing the release
FL = re.compile(r"departure time: (\d\d:\d\d [AP]M), duration: ([^,]+), number of stops: (\d), price: \$(\d+)")
ANS = re.compile(r"The best option is Flight (\d)\.")
FB = re.compile(r"^Your option Flight (\d) is (?:correct\.|incorrect\. I prefer Flight (\d)\.)")


def _hm(s):
    h = re.search(r"(\d+) hr", s)
    m = re.search(r"(\d+) min", s)
    return 60 * (int(h.group(1)) if h else 0) + (int(m.group(1)) if m else 0)


def _ck(s):
    t, ap = s.split(" ")
    h, m = map(int, t.split(":"))
    return 60 * (h % 12 + (12 if ap == "PM" else 0)) + m


def flight_levels(m):
    dep, dur, pr = _ck(m[0]) - 360, _hm(m[1]) - 30, int(m[3]) - 100
    assert dep % 96 == 0 and dur % 117 == 0 and pr % 90 == 0, m
    return (dep // 96, dur // 117, int(m[2]), pr // 90)


def parse_train_row(r):
    ms = r["messages"]
    assert [m["role"] for m in ms] == ["user", "assistant"] * (len(ms) // 2)
    us, As = [m["content"] for m in ms[0::2]], [m["content"] for m in ms[1::2]]
    opts = [tuple(flight_levels(m) for m in FL.findall(u)) for u in us]
    assert all(len(o) == 3 for o in opts)
    ans = [int(ANS.search(a).group(1)) - 1 for a in As]
    ch = []
    for t, u in enumerate(us[1:]):
        m = FB.match(u)
        assert m and int(m.group(1)) - 1 == ans[t], u[:120]
        ch.append(int(m.group(2) or m.group(1)) - 1)
    return tuple(opts), ch, ans


HL = re.compile(r"distance to downtown: (\d+) miles?, price: \$(\d+), rating: (\d) stars?, amenities: (.+)$")
AMEN_SRC = {"no amenities": 0, "free parking": 1, "free parking and free breakfast": 2,
            "free parking, free breakfast, and pool": 3, "free parking, free breakfast, pool, and fitness center": 4}


def levels_from_numpy(dom, num):
    out = []
    for o in num:
        lv = []
        for x, n in zip(o, DOM[dom]["nlev"]):
            j = round(x * (n - 1))
            assert abs(j - x * (n - 1)) < 1e-6, (dom, o)
            lv.append(int(j))
        out.append(tuple(lv))
    return tuple(out)


def load_episodes(raw):
    """-> list of canonical episodes: {src, dom, idx, rounds (5 option sets), choices (5), ...}"""
    rf = {}
    for l in open(f"{raw}/eval/interaction/flight.jsonl"):
        r = json.loads(l)
        rf[r["idx"]] = tuple(int(round(2 * x)) for x in r["reward_fn"])
    thidx = {tuple(t): i for i, t in enumerate(TH.tolist())}
    eps = []
    # ---- training file: bayesian + oracle, episodes keyed by (idx, 5 option sets)
    B, O = {}, {}
    pref = {"bayesian": defaultdict(list), "oracle": defaultdict(list)}
    for name, store in (("bayesian", B), ("oracle", O)):
        for ln, l in enumerate(open(f"{raw}/train/{name}.jsonl")):
            r = json.loads(l)
            opts, ch, ans = parse_train_row(r)
            pref[name][r["idx"]].append((opts, ch, ans, ln + 1))
            if len(opts) == 5:
                store[(r["idx"], opts)] = (ch, ans, ln + 1)
    assert len(B) == len(O) == 6240 and set(B) == set(O)
    for (idx, opts), (chB, ansB, lnB) in sorted(B.items()):
        chO, ansO, lnO = O[(idx, opts)]
        assert chB == chO[:4] == ansO[:4]
        choices = list(chO) + [ansO[4]]
        # prefix rows of this episode in both files (line numbers) and the paper's Bayesian-assistant terminal answers
        lines, paper = {}, {}
        for name in ("bayesian", "oracle"):
            for (popts, pch, pans, pln) in pref[name][idx]:
                rr = len(popts)
                if popts == opts[:rr] and pch == choices[:rr - 1]:
                    assert (name, rr) not in lines
                    lines[(name, rr)] = pln
                    if name == "bayesian":
                        paper[rr - 1] = pans[-1]
        assert len(lines) == 10 and len(paper) == 5
        eps.append({"src": "train", "dom": "flights", "idx": idx, "theta_index": thidx[rf[idx]], "rounds": opts,
                    "choices": choices, "paper_bayes_answer": [paper[r] for r in range(5)],
                    "source_lines": {f"{n}.jsonl:prefix{rr}": ln for (n, rr), ln in sorted(lines.items())}})
    # ---- eval interaction files (rounds_numpy, reward_fn)
    for src, fn, dom in (("flight_eval", "flight", "flights"), ("hotel_eval", "hotel", "hotels"),
                         ("human", "flight_human", "flights")):
        for ln, l in enumerate(open(f"{raw}/eval/interaction/{fn}.jsonl")):
            r = json.loads(l)
            rounds = tuple(levels_from_numpy(dom, num) for num in r["rounds_numpy"])
            if dom == "flights":                     # the text and the numbers agree
                for rd, lv in zip(r["rounds"], rounds):
                    assert tuple(flight_levels(m) for m in FL.findall(" ".join(rd["options"]))) == lv
            else:
                for rd, lv in zip(r["rounds"], rounds):
                    for o, l in zip(rd["options"], lv):
                        mm = HL.search(o)
                        assert mm and (int(mm.group(1)), (int(mm.group(2)) - 100) // 90, int(mm.group(3)) - 1) == l[:3], (o, l)
                        assert AMEN_SRC[mm.group(4)] == l[3], (o, l)
            t2 = tuple(int(round(2 * x)) for x in r["reward_fn"])
            e = {"src": src, "dom": dom, "idx": r["idx"], "rounds": rounds, "choices": [rd["user_idx"] for rd in r["rounds"]],
                 "source_lines": {f"eval/interaction/{fn}.jsonl": ln + 1}}
            if "seed" in r:
                e["release_seed"] = r["seed"]
            if src != "human":
                assert rf[r["idx"]] == t2
                e["theta_index"] = thidx[t2]
            else:
                e["human_reward_fn_index"] = thidx.get(t2)
                e["human_episode"] = ln
            eps.append(e)
    return eps


def episode_uid(e):
    fp = canon([e["src"], e["dom"], e["idx"], [list(map(list, o)) for o in e["rounds"]], e["choices"], e["source_lines"]])
    return f"bzen:{e['src']}:{e['dom']}:i{e['idx']:04d}:{sha256s(fp)[:12]}"


# ============================================================================ records
def split_of(e, dev_idx):
    if e["src"] == "train":
        return "dev_profile" if e["idx"] in dev_idx else "train"
    return {"flight_eval": "dev_flight_eval", "hotel_eval": "dev_hotel", "human": "dev_human"}[e["src"]]


def release_tie_round(e):
    """v2.3: the first shown round in which the TRUE profile's maximum is tied (simulated release, theta* known), else None.
    The declared unique-best filter forbids it, and the filtered likelihood then gives theta* zero mass from that round on, so
    every decision whose current set or history includes the round is quarantined before inference (dev/eval only)."""
    if "theta_index" not in e:
        return None
    for t, o in enumerate(e["rounds"]):
        if choice_sets(e["dom"], o)[2][e["theta_index"]] > 1:
            return t
    return None


def build_decisions(e, dev_idx, code_sha):
    dom, rounds, ch = e["dom"], e["rounds"], e["choices"]
    noisy = EPS_HUMAN if e["src"] == "human" else None
    filtered = e["src"] in ("train", "flight_eval", "hotel_eval")
    euid = episode_uid(e)
    split = split_of(e, dev_idx)
    # human file: idx = one option-set sequence shown to 10 different participants (rows), so identical histories recur
    # across people; the whole sequence is one ancestry component
    comp = (f"bzen:human:optseq{e['idx']:03d}" if e["src"] == "human" else f"bzen:profile:i{e['idx']:04d}")
    gen_fam = {"train": "qiu2026_flight_sim_train", "flight_eval": "qiu2026_flight_sim_eval",
               "hotel_eval": "qiu2026_hotel_sim_eval", "human": "qiu2026_flight_human"}[e["src"]]
    t_tie = release_tie_round(e) if filtered else None
    if t_tie is not None and split == "train":
        raise SystemExit(f"{euid}: a training episode violates the declared filter (round {t_tie + 1}); refuse")
    recs, diag = [], []
    for r in range(N_ROUNDS):
        quar = t_tie is not None and r >= t_tie
        inf = infer(dom, rounds, ch, r, noisy, filtered, e.get("theta_index"))
        state = render_state(dom, [(rounds[t], ch[t]) for t in range(r)], rounds[r], noisy, filtered)
        keys = okeys(dom)
        realized = ch[r]
        dkey = f"{euid}/r{r}"
        base_meta = {
            "family": f"zen_{e['src']}", "domain": dom, "rounds_observed": r, "round_now": r + 1, "trajectory_id": euid,
            "episode_uid": euid, "profile_idx": e["idx"], "profile_heldout": e["idx"] in dev_idx if e["src"] != "human" else None,
            "split": split, "realized_choice_index": realized, "oracle_choice": keys[realized],
            "options_levels": [list(o) for o in rounds[r]],
            "noise_model": None if noisy is None else {"kind": "eps_mixture", "eps": noisy, "fit": "max marginal likelihood "
                                                       "on the human file (dev-only), grid 0.01"},
            "declared_model": ("release_process_unique_best (filtered)" if filtered else
                               "eps_mixture" if noisy else "tie_normalised (unfiltered)"),
            "quarantine": ({"status": "QUARANTINED_EVAL", "tie_round_index": t_tie,
                            "reason": (f"the release shows round {t_tie + 1} with this profile's maximum tied, which the declared "
                                       "unique-best filter forbids; the filtered likelihood gives the true profile zero mass from "
                                       "then on, so this target is not exact for the release. Excluded from reads and audit.")}
                           if quar else None),
            "true_profile_posterior": inf.get("true_profile_posterior"),
        }
        variants = {k: inf.get(k) for k in ("P_tie", "P_ind", "P_sel") if inf.get(k) is not None}
        diag.append({"decision": dkey, "split": split, "src": e["src"], "dom": dom, "idx": e["idx"], "r": r,
                     "P": inf["P"], "eu": inf["eu"], "pred_top": inf["pred_top"], "eu_top": inf["eu_top"],
                     "eu_top_ind": inf.get("eu_top_ind"), "eu_top_tie": inf.get("eu_top_tie"), "realized": realized,
                     "support": inf["support"], "filtered": filtered, "quarantined": quar,
                     "true_post": inf.get("true_profile_posterior"),
                     "P_tie": inf.get("P_tie"), "P_sel": inf.get("P_sel"), "P_ind": inf.get("P_ind"),
                     "paper": e.get("paper_bayes_answer", [None] * 5)[r], "marg": inf["marg"],
                     "true_tie_set_size": int(choice_sets(dom, rounds[r])[2][e["theta_index"]]) if "theta_index" in e else None})
        common_sd = {"parent_uid": euid, "ancestry_component_id": comp, "source_repo": "zenodo:17677329",
                     "source_revision": f"md5:{ZEN['md5']}", "source_record_id": canon(e["source_lines"]),
                     "source_family": SOURCE_FAMILY, "skill": "bayesian_preference_inference", "domain": dom,
                     "generator_family": gen_fam, "generator_revision": f"zenodo:17677329@md5:{ZEN['md5']}", "split": split,
                     "license_use": "CC-BY-4.0: commercial use permitted with attribution; attribution: " + ATTRIBUTION,
                     "teacher_runs": [], "evidence_refs": [], "counterfactual_parent_uid": None, "weight": 1.0,
                     "tokenizer_revision": f"{TOK_REPO}@{TOK_REV}"}
        verif = {"status": "ACCEPTED", "class": "exact_oracle", "oracle_id": "bayes_zen_exact_enumeration_624",
                 "oracle_revision": code_sha, "prior": "uniform over the 624 weight profiles",
                 "likelihood": ("eps_mixture(eps=0.15)" if noisy else
                                "release_process 1[argmax_theta={c}]/p_u(theta) per shown set; current set 1[unique]/p_u"
                                if filtered else "tie_normalised 1[c in argmax]/|argmax|"),
                 "visible_input_sha256": sha256s(state)}
        if quar:
            verif = {**verif, "status": "QUARANTINED_EVAL"}
        # (b) next choice
        Pb = inf["P"]
        qb = q_next(dom, rounds[r])
        lb = lab(keys, Pb)
        meta_b = {**base_meta, "question_basis": "next_choice_predictive", "gold_exact": Pb, "eu_by_option": inf["eu"],
                  "eu_argmax_set": inf["eu_top"], "pred_argmax_set": inf["pred_top"], "label_entropy_nats": ent(Pb),
                  "eu_by_option_unfiltered": inf.get("eu_unfiltered"), "eu_argmax_set_unfiltered": inf.get("eu_top_tie"),
                  "posterior_support": inf["support"], "posterior_entropy_nats": inf["post_entropy"],
                  "posterior_support_noiseless": inf.get("support_noiseless"),
                  "variants": variants, "ref_no_update": inf["ref_no_update"],
                  "ref_first_round_only": inf["ref_first_round_only"],
                  "paper_bayes_answer_index": e.get("paper_bayes_answer", [None] * 5)[r],
                  "P_pref_noiseless_emission": inf.get("P_pref_noiseless_emission")}
        recs.append(make_rec(f"{dkey}/next_choice", f"bayes_zen/{dom}/next_choice", state, "next_choice", qb, lb, euid,
                             meta_b, {**common_sd, "target_basis": "outcome_probability", "kind": "choice", "K": 3,
                                      "label_verification": verif, "decision_group": dkey,
                                      "label_uncertainty": "none under the declared model (exact enumeration)"}))
        # (c) recommendation (expected-utility action)
        qc = q_rec(dom, rounds[r])
        lc = lab(keys, split_mass(3, inf["eu_top"]))
        meta_c = {**base_meta, "question_basis": "eu_recommendation", "gold_exact": split_mass(3, inf["eu_top"]),
                  "eu_by_option": inf["eu"], "eu_argmax_set": inf["eu_top"], "pred_argmax_set": inf["pred_top"],
                  "eu_argmax_set_indicator_variant": inf.get("eu_top_ind"), "eu_argmax_set_unfiltered": inf.get("eu_top_tie"),
                  "utility": "w . phi(option), real units"}
        recs.append(make_rec(f"{dkey}/recommend", f"bayes_zen/{dom}/recommend", state, "recommend", qc, lc, euid, meta_c,
                             {**common_sd, "target_basis": "optimal_action", "kind": "choice", "K": 3,
                              "label_verification": {**verif, "loss": "negative utility w . phi(recommended option)"},
                              "decision_group": dkey,
                              "label_uncertainty": "none: argmax of exact expected utility; exact ties split evenly"}))
        # (a) feature-weight beliefs
        for f in range(4):
            qa = q_weight(dom, f)
            la = lab(list(WEIGHT_KEYS), inf["marg"][f])
            meta_a = {**base_meta, "question_basis": "belief_marginal", "feature": DOM[dom]["fkeys"][f],
                      "gold_exact": inf["marg"][f], "label_entropy_nats": ent(inf["marg"][f])}
            recs.append(make_rec(f"{dkey}/weight_{DOM[dom]['fkeys'][f]}", f"bayes_zen/{dom}/weight", state,
                                 f"weight_{DOM[dom]['fkeys'][f]}", qa, la, euid, meta_a,
                                 {**common_sd, "target_basis": "exact_posterior", "kind": "score", "K": 5,
                                  "label_verification": verif, "decision_group": dkey,
                                  "label_uncertainty": "none under the declared model (exact enumeration)"}))
    return recs, diag


def make_rec(rid, source, state, qid, q, labels, base_id, meta, sd):
    keys = list(q["criteria"]) if q["type"] == "choice" else list(q["levels"])
    sd = {**sd, "decision_uid": rid, "record_uid": rid, "ordered_options": keys, "gold_by_option_key": labels,
          "option_sha256": sha256s(canon(keys)), "gold_sha256": sha256s(canon(labels))}
    rec = {"id": rid, "source": source, "license": LICENSE, "state": state, "questions": {qid: q}, "labels": {qid: labels},
           "label_kind": {qid: kind_of(labels)}, "base_id": base_id, "meta": {**meta, "sd": sd}}
    validate(rec)
    return rec


def validate(r):
    assert isinstance(r["state"], str) and r["state"].strip()
    for qid, q in r["questions"].items():
        keys = list(q["criteria"]) if q["type"] == "choice" else list(q["levels"])
        assert 2 <= len(keys) <= 52 and len(set(keys)) == len(keys)
        if q["type"] == "choice":
            texts = list(q["criteria"].values())
            assert len(set(texts)) == len(texts), "option text collision"
        lab_ = r["labels"][qid]
        assert set(lab_) == set(keys) and all(0 <= v <= 1 for v in lab_.values()) and abs(sum(lab_.values()) - 1) < 1e-6
        assert r["label_kind"][qid] in ("hard", "soft")


# ============================================================================ generator holdout (our own generator)
def genholdout_records(gen_dir, bp_path):
    """Our build_bayes_pref.py eval set (3,120 units; flights / hotels / trails on OUR grids and OUR older template) as a
    source-family + generator + template holdout. Labels unchanged (its exact tie-normalised predictive, correct for its own
    unfiltered generator); EU per option added from its own meta."""
    spec = importlib.util.spec_from_file_location("bbp", bp_path)
    bp = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(bp_path))
    spec.loader.exec_module(bp)
    out = []
    for p in sorted(Path(gen_dir).glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            m = r["meta"]
            L = m["feature_levels"]
            F = np.stack([bp.phi_int(o, L) for o in m["options_levels"]])
            Fh = [np.stack([bp.phi_int(o, L) for o in h["options_levels"]]) for h in m["history"]]
            chh = [bp.KEYS.index(h["pick"]) for h in m["history"]]
            post = bp.posterior(Fh, chh)
            eu = ((post @ bp.THETA2.astype(float)) @ F.T.astype(float)) / (2 * bp.GRID)
            P0 = bp.predictive(bp.posterior([], []), F)
            P1 = bp.predictive(bp.posterior(Fh[:1], chh[:1]), F) if chh else P0
            rid = f"bzen:genholdout:{r['id']}"
            keys = list(r["questions"]["next_choice"]["criteria"])
            m2 = {"family": f"gen_{m['family']}", "domain": m["family"], "rounds_observed": m["rounds_observed"],
                  "round_now": m["round_now"], "trajectory_id": f"bzen:gen:{m['trajectory_id']}",
                  "episode_uid": f"bzen:gen:{m['trajectory_id']}", "profile_idx": m["user_index"], "split": "dev_generator_holdout",
                  "realized_choice_index": keys.index(m["oracle_choice"]), "oracle_choice": m["oracle_choice"],
                  "question_basis": "next_choice_predictive", "gold_exact": m["gold_exact"], "eu_by_option": eu.tolist(),
                  "eu_argmax_set": top_set(eu, 1e-9), "pred_argmax_set": top_set(np.array(m["gold_exact"]), 1e-12),
                  "ref_no_update": P0.tolist(), "ref_first_round_only": P1.tolist(),
                  "generator_record_id": r["id"], "generator_meta_keys_dropped": "history/options kept in the source file"}
            sd = {"decision_uid": rid, "record_uid": rid, "parent_uid": m2["episode_uid"],
                  "ancestry_component_id": f"bzen:gen:{m['trajectory_id']}", "source_repo": "jev:build_bayes_pref.py",
                  "source_revision": f"md5:{hashlib.md5(open(bp_path, 'rb').read()).hexdigest()}",
                  "source_record_id": r["id"], "source_family": GEN_FAMILY, "skill": "bayesian_preference_inference",
                  "domain": m["family"], "generator_family": "jev_bayes_pref_eval_v1", "generator_revision": m["generator"],
                  "split": "dev_generator_holdout", "target_basis": "outcome_probability", "kind": "choice", "K": 3,
                  "ordered_options": keys, "gold_by_option_key": r["labels"]["next_choice"],
                  "option_sha256": sha256s(canon(keys)), "gold_sha256": sha256s(canon(r["labels"]["next_choice"])),
                  "label_verification": {"status": "ACCEPTED", "class": "exact_oracle",
                                         "oracle_id": "build_bayes_pref exact predictive (two implementations, max diff 2.9e-15)",
                                         "likelihood": "tie_normalised", "visible_input_sha256": sha256s(r["state"])},
                  "label_uncertainty": "none (exact; generator unfiltered, so tie-normalised is its true process)",
                  "license_use": "ours (generated by build_bayes_pref.py)", "teacher_runs": [], "evidence_refs": [],
                  "counterfactual_parent_uid": None, "weight": 1.0, "tokenizer_revision": f"{TOK_REPO}@{TOK_REV}",
                  "decision_group": f"{m2['episode_uid']}/r{m['rounds_observed']}"}
            rec = {"id": rid, "source": f"bayes_zen_genholdout/{m['family']}", "license": r["license"], "state": r["state"],
                   "questions": r["questions"], "labels": r["labels"], "label_kind": r["label_kind"],
                   "base_id": m2["episode_uid"], "meta": {**m2, "sd": sd}}
            validate(rec)
            out.append(rec)
    return out


# ============================================================================ rendering + tokens (pinned production renderer)
_G = {}


def _init_worker(tokdir, scorer_dir):
    sys.path.insert(0, scorer_dir)
    spec = importlib.util.spec_from_file_location("ls_pinned_bzen", os.path.join(scorer_dir, "claude_lmscore.py"))
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    from transformers import AutoTokenizer
    _G["ls"], _G["tok"] = ls, AutoTokenizer.from_pretrained(tokdir)


def unit_of(rec):
    qid, q = next(iter(rec["questions"].items()))
    keys, texts, target = _G["ls"].cf.question_options(q, rec["labels"][qid])
    return {"qid": qid, "qtype": q["type"], "state": rec["state"], "instructions": q["instructions"], "option_keys": keys,
            "option_texts": texts, "noul_criteria": None, "target": target}


def _render_one(args):
    rec, order = args
    u = unit_of(rec)
    order = list(range(len(u["option_keys"]))) if order is None else order
    tok = _G["tok"]
    text = _G["ls"].build_prompt(tok, u, order, MAX_LEN, noul_defs=True)
    if text is None:
        return None
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    cut = rec["state"] not in text
    return {"n_tokens": n, "prompt_sha256": sha256s(text), "cut": cut,
            "rendered_options": [u["option_keys"][i] for i in order]}


def renderer_spec(tokdir, scorer_dir):
    files = {}
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        p = os.path.join(tokdir, f)
        if os.path.exists(p):
            files[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()   # noqa: E731
    spec = {"schema": "fullft-render-manifest-v1", "builder": "claude_lmscore.build_prompt",
            "scorer_md5": md5(os.path.join(scorer_dir, "claude_lmscore.py")),
            "cache_features_md5": md5(os.path.join(scorer_dir, "cache_features.py")), "tokenizer_files": files,
            "tokenizer_revision": TOK_REV, "max_len": MAX_LEN, "margin": 16, "noul_defs": True, "whole_case": False,
            "plain": False, "chat_preset": None, "labels": "A-Z a-z", "tokenize": "add_special_tokens=False",
            "renderer_id": "R2 (garas-render-2: noul definitions shown + exact-collision key suffix); on these rows R2 == R1 "
                           "byte-for-byte (no noul questions, no option-text collisions: asserted)"}
    assert spec["scorer_md5"] == SCORER_MD5 and spec["cache_features_md5"] == CF_MD5, spec
    return spec, hashlib.sha256(canon(spec).encode()).hexdigest()


# ============================================================================ convert
def option_order(*parts):
    s = int(hrank(*parts)[:16], 16)
    return [int(x) for x in np.random.default_rng(s).permutation(3)]


def qstats(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "p05": round(float(np.quantile(v, .05)), 4),
            "p50": round(float(np.quantile(v, .5)), 4), "p95": round(float(np.quantile(v, .95)), 4),
            "max": round(float(v.max()), 4), "zero_share": round(float((v < 1e-12).mean()), 4)}


def cmd_convert(a):
    t0 = time.time()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: immutable output")
    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    rspec, rsha = renderer_spec(a.tokenizer, a.scorer_dir)
    eps_ = load_episodes(a.raw)
    print(f"[{time.time() - t0:.0f}s] episodes: {Counter(e['src'] for e in eps_)}", flush=True)
    # ---- profile split (seeded, before any target is computed)
    ranked = sorted(range(NTH), key=lambda i: hrank(NS, "profile_split", i))
    dev_idx = set(ranked[:N_DEV_PROFILES])
    recs, diags = [], []
    for e in eps_:
        rr, dd = build_decisions(e, dev_idx, code_sha)
        recs += rr
        diags += dd
    print(f"[{time.time() - t0:.0f}s] zenodo records {len(recs)}", flush=True)
    if a.genholdout:
        g = genholdout_records(a.genholdout, a.bp_path)
        recs += g
        print(f"[{time.time() - t0:.0f}s] generator-holdout records {len(g)}", flush=True)
    # ---- U block: 384 training episodes from distinct profiles, all five prefixes
    train_eps = defaultdict(list)
    for e in eps_:
        if e["src"] == "train" and e["idx"] not in dev_idx:
            train_eps[e["idx"]].append(episode_uid(e))
    u_idx = sorted(train_eps, key=lambda i: hrank(NS, "ublock_profile", i))[:U_EPISODES]
    u_eps = sorted(min(train_eps[i], key=lambda x: hrank(NS, "ublock_episode", x)) for i in u_idx)
    u_set = set(u_eps)
    # ---- render + count every record (identity order) and every U exposure (frozen order)
    by_id = {r["id"]: r for r in recs}
    assert len(by_id) == len(recs), Counter(Counter(r["id"] for r in recs).values())
    # human: participants sharing an option sequence and an identical history -> empirical next-pick counts (dev diagnostic)
    same = defaultdict(Counter)
    for r in recs:
        m = r["meta"]
        if m["family"] == "zen_human" and next(iter(r["questions"])) == "next_choice":
            same[sha256s(r["state"])][m["realized_choice_index"]] += 1
    for r in recs:
        m = r["meta"]
        if m["family"] == "zen_human" and next(iter(r["questions"])) == "next_choice":
            c = same[sha256s(r["state"])]
            m["same_state_participants"] = int(sum(c.values()))
            m["same_state_realized_counts"] = [int(c[j]) for j in range(3)]
    ublock = []
    for eu in u_eps:
        for r in range(N_ROUNDS):
            dkey = f"{eu}/r{r}"
            ublock.append({"decision_group": dkey, "episode_uid": eu, "r": r, "option_order": option_order(NS, "uorder", dkey)})
    jobs = [(rec, None) for rec in recs]
    for u in ublock:
        for q in ("next_choice", "recommend"):
            jobs.append((by_id[f"{u['decision_group']}/{q}"], u["option_order"]))
    with mp.Pool(a.procs, initializer=_init_worker, initargs=(a.tokenizer, a.scorer_dir)) as pool:
        res = pool.map(_render_one, jobs, chunksize=256)
    print(f"[{time.time() - t0:.0f}s] rendered {len(res)} prompts", flush=True)
    too_long = [j[0]["id"] for j, x in zip(jobs, res) if x is None or x["cut"]]
    assert not too_long, too_long[:5]
    for (rec, order), x in zip(jobs[:len(recs)], res[:len(recs)]):
        rec["meta"]["sd"].update({"renderer_sha256": rsha, "prompt_sha256": x["prompt_sha256"],
                                  "rendered_token_count": x["n_tokens"], "render_option_order": "identity (stored order)"})
    uexp = {}
    for (rec, order), x in zip(jobs[len(recs):], res[len(recs):]):
        uexp[(rec["id"], tuple(order))] = x
    # ---- write data/<split>/<stem>.jsonl
    data = out / "data"
    fh = {}
    counts = Counter()
    for rec in recs:
        split = rec["meta"]["sd"]["split"]
        stem = rec["source"].replace("/", "__")
        p = data / split / f"{stem}.jsonl"
        if p not in fh:
            p.parent.mkdir(parents=True, exist_ok=True)
            fh[p] = open(p, "w", encoding="utf-8")
        fh[p].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[(split, rec["source"])] += 1
    for f in fh.values():
        f.close()
    # ---- canonical episode table
    (out / "episodes").mkdir(parents=True)
    with gzip.open(out / "episodes" / "episodes.jsonl.gz", "wt", encoding="utf-8") as f:
        for e in eps_:
            f.write(json.dumps({"episode_uid": episode_uid(e), "split": split_of(e, dev_idx), "u_block": episode_uid(e) in u_set,
                                **{k: v for k, v in e.items() if k != "rounds"},
                                "rounds": [[list(o) for o in rd] for rd in e["rounds"]]}) + "\n")
    # ---- U block (records per arm + frozen exposure lists)
    ub = out / "u_block"
    arms = {"U1": ("next_choice", "observed_outcome"), "U1p": ("next_choice", "sampled_predictive_paper"),
            "U2": ("recommend", "optimal_action"), "U3": ("next_choice", "outcome_probability")}
    ublock_rows = {}
    for arm, (q, basis) in arms.items():
        d = ub / arm / "train"
        d.mkdir(parents=True)
        rows = []
        with open(d / f"bayes_zen_ublock__{arm}.jsonl", "w", encoding="utf-8") as f:
            for pos, u in enumerate(ublock):
                src = by_id[f"{u['decision_group']}/{q}"]
                rec = json.loads(json.dumps(src))
                rid = f"{u['decision_group']}/{q}/{arm}"
                rec["id"] = rid
                rec["source"] = f"bayes_zen_ublock/{arm}"
                if arm in ("U1", "U1p"):
                    j = rec["meta"]["realized_choice_index"] if arm == "U1" else rec["meta"]["paper_bayes_answer_index"]
                    k = okeys("flights")[j]
                    rec["labels"][q] = {kk: (1.0 if kk == k else 0.0) for kk in okeys("flights")}
                    rec["label_kind"][q] = "hard"
                    rec["meta"]["gold_exact"] = [1.0 if kk == k else 0.0 for kk in okeys("flights")]
                    rec["meta"]["question_basis"] = ("next_choice_realized_outcome" if arm == "U1" else
                                                     "next_choice_paper_bayesian_sample")
                rec["meta"]["sd"].update({"decision_uid": rid, "record_uid": rid, "target_basis": basis, "split": "train",
                                          "gold_by_option_key": rec["labels"][q],
                                          "gold_sha256": sha256s(canon(rec["labels"][q])), "u_arm": arm,
                                          "derived_from": src["id"]})
                if arm == "U1":
                    rec["meta"]["sd"]["label_verification"] = {"status": "ACCEPTED", "class": "observed_outcome",
                                                               "evidence": "realised pick in the release (oracle.jsonl terminal "
                                                                           "answer / next-round feedback); a draw from the "
                                                                           "release process = the U3 target"}
                    rec["meta"]["sd"]["label_uncertainty"] = "a single sampled outcome (one-hot), not a distribution"
                if arm == "U1p":
                    rec["meta"]["sd"]["label_verification"] = {"status": "ACCEPTED", "class": "paper_teacher_sample",
                                                               "evidence": "the release's bayesian.jsonl terminal recommendation "
                                                                           "(the paper's Bayesian-teaching label); empirically a "
                                                                           "calibrated draw from the UNFILTERED tie-normalised "
                                                                           "predictive (meta.variants.P_tie), not from U3's target"}
                    rec["meta"]["sd"]["label_uncertainty"] = "a single sampled label (one-hot), not a distribution"
                validate(rec)
                x = uexp[(src["id"], tuple(u["option_order"]))]
                rec["meta"]["sd"].update({"renderer_sha256": rsha, "prompt_sha256": x["prompt_sha256"],
                                          "rendered_token_count": x["n_tokens"]})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append({"block_pos": pos, "decision_uid": rid, "decision_group": u["decision_group"],
                             "episode_uid": u["episode_uid"], "rounds_observed": u["r"],
                             "option_order": x["rendered_options"], "prompt_sha256": x["prompt_sha256"],
                             "n_tokens": x["n_tokens"], "weight": 1.0, "parent_route": "none",
                             "mixture_family": "exact_bayesian_numeric", "target_basis_kind": basis,
                             "gold": rec["labels"][q], "renderer_sha256": rsha})
        with open(ub / f"exposures_{arm}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        ublock_rows[arm] = rows
    # ---- sanity eval sets (next_choice + recommend), scorer-readable layout <root>/<set>/eval/<family>.jsonl
    for setname, q in (("bayes_zen", "next_choice"), ("bayes_zen_rec", "recommend")):
        fams = defaultdict(list)
        for rec in recs:
            m = rec["meta"]
            if m["sd"]["split"] == "train" or next(iter(rec["questions"])) != q or m.get("quarantine"):
                continue
            fam = m["family"]
            if m["sd"]["split"] == "dev_flight_eval":
                fam += "_unseen_profile" if m["profile_heldout"] else "_seen_profile"
            if m["sd"]["split"] == "dev_profile":
                fam = "zen_train_file_heldout_profiles"
            fams[fam].append({**rec, "meta": {**m, "family": fam}})
        d = out / "sanity" / setname / "eval"
        d.mkdir(parents=True)
        for fam, rr in sorted(fams.items()):
            with open(d / f"{fam}.jsonl", "w", encoding="utf-8") as f:
                for rec in rr:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # ---- audit
    audit = make_audit(recs, diags, counts, dev_idx, u_eps, ublock_rows, eps_)
    audit["seconds"] = round(time.time() - t0, 1)
    (out / "audit.json").write_text(json.dumps(audit, indent=1))
    man = {"schema": "stock_decision_v1-bayes_zen-v1", "builder": VERSION, "builder_sha256": code_sha, "namespace": NS,
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "zenodo": ZEN, "attribution": ATTRIBUTION,
           "record_license_string": LICENSE,
           "article_licence_note": ("the article is CC BY-NC-ND 4.0: no article text, figure or prompt is copied; the "
                                    "release's own prompt/answer strings are parsed for values only and never reproduced"),
           "renderer_sha256": rsha, "renderer_spec": rspec, "conventions": CONVENTIONS,
           "splits": {s: sorted({k[1] for k in counts if k[0] == s}) for s in sorted({k[0] for k in counts})},
           "dev_profile_idx": sorted(dev_idx), "u_block_episodes": u_eps,
           "registered_not_converted": REGISTERED_NOT_CONVERTED,
           "files": {str(p.relative_to(out)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                               "md5": hashlib.md5(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                     for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(audit["headline"], indent=1))
    print(f"[{time.time() - t0:.0f}s] done", flush=True)


CONVENTIONS = {
    "latent_space": "624 weight profiles {-1,-0.5,0,+0.5,+1}^4 minus all-zero; uniform prior; idx -> profile from "
                    "eval/interaction/*.jsonl reward_fn (verified on every training choice)",
    "declared_likelihood": "v2: simulated release (train, dev_profile, dev_flight_eval, dev_hotel) declares the RELEASE PROCESS "
                           "(uniform independent feature levels, 3 distinct items, set redrawn unless the person's best is "
                           "unique): per shown set 1[argmax_theta = {c}] / p_u(theta), current set 1[unique] / p_u(theta); "
                           "primary targets (a)/(b)/(c) from this posterior, so realised picks are exact draws from the "
                           "next-choice target (E[U1] = U3). Human: eps-mixture; generator holdout: tie_normalised (unfiltered "
                           "processes)",
    "tie_normalised_variant": "meta.variants.P_tie (the v1 primary): unfiltered, P(c|theta,O) = 1[c in argmax]/|argmax|; "
                              "meta.eu_by_option_unfiltered / eu_argmax_set_unfiltered for the action",
    "indicator_variant": "meta.variants.P_ind: the paper's compatibility indicator 1[c in argmax] in the posterior; current "
                         "round predicted with 1/|argmax|",
    "release_process_variant": "meta.variants.P_sel (= the v2 primary for the simulated release): the released simulated data were generated with option sets filtered so the "
                               "true user's best option is unique (0 ties in 156,000 training choices, 0 in 3,120 flight-eval, 2 "
                               "in 3,120 hotel-eval, vs 5.3% / 3.9% tie sets in the unfiltered held-out flight / hotel sets). "
                               "P_sel = posterior with 1[argmax_theta = {c}] / p_u(theta) per shown set, p_u = P(unique max | "
                               "theta) for 3 distinct uniformly drawn items.",
    "human_noise": "dev_human: eps-mixture (1-eps)*1[c in argmax]/|argmax| + eps/3, eps = 0.15 (MLE on the human file, grid "
                   "0.01, uniform prior over 624); declared in the state. The noiseless posterior is EMPTY for 76 of 500 human "
                   "episodes, so a noiseless label would be undefined or falsely confident",
    "eu_recommendation": "argmax_o E_post[w . phi(o)] (= the plug-in posterior mean, utility linear); ties (within 1e-12) "
                         "split the label mass. Unfiltered model: every r=0 decision is a 3-way tie (prior mean weight 0). "
                         "Filtered (v2 primary): the shown set is itself evidence (only profiles with a unique best survive), "
                         "so r=0 usually has a unique, small-margin EU argmax; see audit eu_best_minus_second_by_r_train",
    "terminal_only": "every record is one terminal decision after r observed rounds (r = 0..4); earlier answers are history, "
                     "never targets; the assistant's own earlier recommendations are NOT shown (canonical history = option sets "
                     "+ actual user choices)",
    "grouping": "parent_uid = episode (all five prefixes and every derived question share it); ancestry_component_id = "
                "profile idx (all ten episodes of an idx, and the same idx's eval/hotel episodes); splits by profile",
    "paper_bayes_answers": "meta.paper_bayes_answer_index (train/dev_profile next_choice rows): the release's bayesian.jsonl "
                           "terminal recommendation; empirically these are calibrated DRAWS from the exact tie-normalised "
                           "predictive (posterior sampling), not the expected-utility argmax",
}
REGISTERED_NOT_CONVERTED = {
    "eval/interaction/flight_{2,3,5,6,7,8}features.jsonl + eval/heldout/flight_*features.json": "dev transfer candidates "
        "(rounds_numpy + reward_fn present, 0 ties: filtered like the rest); 5-8 features need 5^n-1 profile spaces and new "
        "feature wordings; not converted in v1",
    "eval/interaction/webshop/*.jsonl + eval/heldout/webshop/*.jsonl (100 categories x 10 users)": "dev-only; latent goal "
        "space not enumerable from the release, so no exact target; outcome-label conversion possible later",
    "eval/heldout/{flight,hotel}.json (100 option sets x 624 profiles)": "paper's held-out-set protocol; usable as extra "
        "dev probes (exact targets computable); not materialised in v1",
}


def make_audit(recs, diags, counts, dev_idx, u_eps, ublock_rows, eps_):
    A = {}
    diags_all = diags
    diags = [d for d in diags_all if not d.get("quarantined")]
    A["quarantine"] = {"decisions": sum(1 for d in diags_all if d.get("quarantined")),
                       "records": sum(1 for r in recs if (r.get("meta") or {}).get("quarantine")),
                       "episodes": sorted({d["decision"].rsplit("/", 1)[0] for d in diags_all if d.get("quarantined")}),
                       "true_profile_posterior_on_quarantined": sorted({round(d["true_post"], 6) for d in diags_all
                                                                         if d.get("quarantined") and d.get("true_post") is not None}),
                       "rule": "release_tie_round(): decisions at or after a shown round where the true profile's maximum is tied; "
                               "excluded from every audit statistic below except the release tie rate, and from the sanity sets"}
    A["counts_by_split_source"] = {f"{s} | {src}": n for (s, src), n in sorted(counts.items())}
    by_split = Counter()
    for (s, src), n in counts.items():
        by_split[s] += n
    A["records_by_split"] = dict(by_split)
    A["decisions_by_split"] = dict(Counter(d["split"] for d in diags_all))
    A["decisions_by_split_excluding_quarantine"] = dict(Counter(d["split"] for d in diags))
    # leakage
    split_of_idx = defaultdict(set)
    split_of_ep = defaultdict(set)
    for rec in recs:
        m = rec["meta"]
        split_of_ep[m["episode_uid"]].add(m["sd"]["split"])
        if m["sd"]["source_family"] == SOURCE_FAMILY and m["family"].startswith("zen_train"):
            split_of_idx[m["profile_idx"]].add(m["sd"]["split"])
    states_by_split = defaultdict(set)
    for rec in recs:
        states_by_split[rec["meta"]["sd"]["split"]].add(sha256s(rec["state"]))
    fps = defaultdict(set)
    for e in eps_:
        s = "train" if (e["src"] == "train" and e["idx"] not in dev_idx) else "dev"
        for o in e["rounds"]:
            fps[(s, e["dom"])].add(tuple(sorted(o)))
    A["leakage"] = {
        "train_file_profiles_in_more_than_one_split": sum(1 for v in split_of_idx.values() if len(v) > 1),
        "episodes_in_more_than_one_split": sum(1 for v in split_of_ep.values() if len(v) > 1),
        "train_profiles": len([i for i, v in split_of_idx.items() if v == {"train"}]),
        "dev_profile_profiles": len([i for i, v in split_of_idx.items() if v == {"dev_profile"}]),
        "identical_states_train_vs_any_dev": len(states_by_split["train"] & set().union(
            *[v for k, v in states_by_split.items() if k != "train"])),
        "flight_option_sets_shared_train_vs_dev": len(fps[("train", "flights")] & fps[("dev", "flights")]),
        "u_block_episodes_all_train": all(split_of_ep[e] == {"train"} for e in u_eps),
        "note": ("dev_flight_eval (seen-profile part) and dev_hotel share PROFILES with train by construction of the release "
                 "(idx i = the same weights in every file); they are dev-only. Profile-held-out reads: dev_profile, "
                 "dev_flight_eval_unseen_profile, and dev_hotel rows with profile_heldout=true"),
    }
    # ties
    tr = [d for d in diags if d["src"] in ("train",)]
    A["ties"] = {
        "true_user_tie_rate_current_set": {src: round(float(np.mean([d["true_tie_set_size"] > 1 for d in diags_all
                                                                       if d["src"] == src])), 5)
                                           for src in ("train", "flight_eval", "hotel_eval")},
        "eu_action_exact_tie_rate_by_r": {r: round(float(np.mean([len(d["eu_top"]) > 1 for d in tr if d["r"] == r])), 4)
                                          for r in range(5)},
        "predictive_argmax_tie_rate_by_r": {r: round(float(np.mean([len(d["pred_top"]) > 1 for d in tr if d["r"] == r])), 4)
                                            for r in range(5)},
        "eu_set_differs_indicator_vs_tie_normalised": int(sum(1 for d in diags if d.get("eu_top_ind") is not None
                                                              and d.get("eu_top_tie") is not None
                                                              and set(d["eu_top_ind"]) != set(d["eu_top_tie"]))),
        "eu_action_exact_tie_rate_by_r_unfiltered": {r: round(float(np.mean([len(d["eu_top_tie"]) > 1 for d in tr
                                                                              if d["r"] == r])), 4) for r in range(5)},
    }
    # entropy distributions
    ent_b = defaultdict(list)
    ent_a = defaultdict(list)
    for d in diags:
        ent_b[(d["split"], d["r"])].append(ent(d["P"]))
        for f in range(4):
            ent_a[(d["split"], d["r"])].append(ent(d["marg"][f]))
    A["target_entropy_nats"] = {
        "next_choice_by_split_r": {f"{s}/r{r}": qstats(v) for (s, r), v in sorted(ent_b.items())},
        "weight_marginal_by_split_r": {f"{s}/r{r}": qstats(v) for (s, r), v in sorted(ent_a.items())},
        "recommend": "one-hot except EU ties: see ties.eu_action_exact_tie_rate_by_r (filtered primary) and _unfiltered",
        "u_arms": {"U1": 0.0, "U1p": 0.0, "U2": "0 except EU ties", "U3": "= next_choice train rows"},
    }
    # EU vs predictive
    dis = defaultdict(lambda: [0, 0, 0])
    prof = defaultdict(lambda: [0, 0])
    for d in diags:
        k = (d["split"], d["r"])
        disjoint = not (set(d["eu_top"]) & set(d["pred_top"]))
        differ = set(d["eu_top"]) != set(d["pred_top"])
        dis[k][0] += disjoint
        dis[k][1] += differ
        dis[k][2] += 1
        if d["src"] == "train":
            prof[d["idx"]][0] += disjoint
            prof[d["idx"]][1] += 1
    A["eu_vs_predictive_argmax"] = {
        "definition": "disjoint = the EU-optimal set and the predictive-argmax set share no option; differ = sets not equal",
        "by_split_r": {f"{s}/r{r}": {"disjoint": v[0], "differ": v[1], "n": v[2]} for (s, r), v in sorted(dis.items())},
        "train_file_total_disjoint": int(sum(v[0] for (s, r), v in dis.items() if s in ("train", "dev_profile"))),
        "train_file_total": int(sum(v[2] for (s, r), v in dis.items() if s in ("train", "dev_profile"))),
        "train_file_r_ge_1_disjoint_share": round(sum(v[0] for (s, r), v in dis.items() if s in ("train", "dev_profile") and r >= 1)
                                                  / max(1, sum(v[2] for (s, r), v in dis.items() if s in ("train", "dev_profile")
                                                               and r >= 1)), 4),
        "profiles_with_any_disjoint": sum(1 for v in prof.values() if v[0] > 0),
        "top_profiles": [{"idx": i, "rate": round(v[0] / v[1], 3), "n": v[1], "weights": (TH[[e["theta_index"] for e in eps_
                          if e["src"] == "train" and e["idx"] == i][0]] / 2).tolist()}
                         for i, v in sorted(prof.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:10]],
    }
    # model variants vs realised outcomes (release process check)
    vs = {}
    for src in ("train", "flight_eval", "hotel_eval"):
        dd = [d for d in diags if d["src"] == src]
        for nm in ("P_tie", "P_ind", "P_sel"):
            ll = [-math.log(max(d[nm][d["realized"]], 1e-12)) for d in dd if d.get(nm) is not None]
            vs[f"{src}/{nm}_logscore_vs_realised"] = round(float(np.mean(ll)), 5)
        kl = defaultdict(list)
        for d in dd:
            if d.get("P_sel") is not None:
                ps, pt = np.array(d["P_sel"]), np.array(d["P_tie"])
                kl[d["r"]].append(float(np.sum(np.where(ps > 0, ps * np.log(np.maximum(ps, 1e-300) / np.maximum(pt, 1e-300)), 0))))
        vs[f"{src}/KL(P_sel||P_tie)_by_r"] = {r: qstats(v) for r, v in sorted(kl.items())}
    A["declared_vs_release_process"] = vs
    # paper's bayesian.jsonl answers
    pb = [d for d in diags if d["src"] == "train" and d["paper"] is not None]
    cal = defaultdict(lambda: [0, 0])
    for d in pb:
        for j in range(3):
            b = min(int(d["P_tie"][j] * 10), 9)
            cal[b][0] += d["paper"] == j
            cal[b][1] += 1
    A["paper_bayesian_answers"] = {
        "n": len(pb),
        "agree_with_eu_argmax_when_unique": round(float(np.mean([d["paper"] in d["eu_top"] for d in pb if len(d["eu_top"]) == 1])), 4),
        "agree_with_predictive_argmax_when_unique": round(float(np.mean([d["paper"] in d["pred_top"] for d in pb
                                                                          if len(d["pred_top"]) == 1])), 4),
        "mean_P_tie_of_answer_vs_E_sum_P2_by_r": {r: [round(float(np.mean([d["P_tie"][d["paper"]] for d in pb if d["r"] == r])), 4),
                                                      round(float(np.mean([sum(x * x for x in d["P_tie"]) for d in pb
                                                                           if d["r"] == r])), 4)] for r in range(5)},
        "calibration_P_tie_bin_vs_answer_freq": {f"{b / 10:.1f}": [v[1], round(v[0] / v[1], 3)] for b, v in sorted(cal.items())},
        "reading": "answers behave as draws from the exact tie-normalised predictive (posterior sampling), not EU argmax",
    }
    # human
    hd = [d for d in diags if d["src"] == "human"]
    A["human"] = {
        "decisions": len(hd),
        "structure": "50 option-set sequences (idx) x 10 participants (rows, each with its own reward_fn); identical "
                     "histories recur across participants",
        "noiseless_posterior_empty": sum(1 for d in hd if d["P_tie"] is None),
        "noiseless_one_hot_where_eps_model_soft": sum(1 for d in hd if d["P_tie"] is not None
                                                      and max(d["P_tie"]) > 1 - 1e-12 and max(d["P"]) < 1 - 1e-9),
        "eps": EPS_HUMAN,
        "mean_logscore_realised_eps_model": round(float(np.mean([-math.log(d["P"][d["realized"]]) for d in hd])), 4),
        "mean_max_p_eps_model_by_r": {r: round(float(np.mean([max(d["P"]) for d in hd if d["r"] == r])), 4) for r in range(5)},
    }
    # tokens
    toks = defaultdict(list)
    for rec in recs:
        toks[(rec["meta"]["sd"]["split"], next(iter(rec["questions"])).split("_")[0])].append(rec["meta"]["sd"]["rendered_token_count"])
    A["rendered_tokens_R2_gemma4_31B_pinned"] = {f"{s}/{q}": qstats(v) for (s, q), v in sorted(toks.items())}
    A["u_arms"] = {arm: {"exposures": len(rows), "unique_decisions": len({r["decision_uid"] for r in rows}),
                         "episodes": len({r["episode_uid"] for r in rows}),
                         "profiles": len({r["episode_uid"].split(":")[3] for r in rows}),
                         "by_r": dict(Counter(r["rounds_observed"] for r in rows)),
                         "tokens_total": int(sum(r["n_tokens"] for r in rows)), "tokens": qstats([r["n_tokens"] for r in rows]),
                         "length_strata": dict(Counter("1-512" if r["n_tokens"] <= 512 else "513-2048" if r["n_tokens"] <= 2048
                                                       else "2049-8192" for r in rows)),
                         "hard_labels": sum(1 for r in rows if sum(1 for v in r["gold"].values() if v > 0) == 1),
                         "hard_labels_by_r": {str(k): sum(1 for r in rows if r["rounds_observed"] == k and
                                                         sum(1 for v in r["gold"].values() if v > 0) == 1) for k in range(5)}}
                   for arm, rows in ublock_rows.items()}
    arms_ = list(ublock_rows)
    same = all(len({ublock_rows[x][i]["decision_group"] for x in arms_}) == 1 and
               len({tuple(ublock_rows[x][i]["option_order"]) for x in arms_}) == 1 for i in range(len(ublock_rows["U1"])))
    A["u_arms_matched"] = {"same_decisions_same_order_same_option_order": same,
                           "U1_U1p_U3_identical_prompts": all(a_["prompt_sha256"] == b_["prompt_sha256"] == c_["prompt_sha256"]
                                                              for a_, b_, c_ in zip(ublock_rows["U1"], ublock_rows["U1p"],
                                                                                    ublock_rows["U3"])),
                           "U2_prompt_differs_only_by_question": True}
    # E[U1] = U3: realised picks against the primary (filtered) predictive, calibration on train; U1p against P_tie
    trd = [d for d in diags if d["src"] == "train"]
    calr = defaultdict(lambda: [0, 0])
    for d in trd:
        for j in range(3):
            b = min(int(d["P"][j] * 10), 9)
            calr[b][0] += d["realized"] == j
            calr[b][1] += 1
    A["realised_vs_primary_predictive_calibration_train"] = {f"{b / 10:.1f}": [v[1], round(v[0] / v[1], 3)]
                                                            for b, v in sorted(calr.items())}
    A["realised_mean_P_by_r_vs_E_sum_P2"] = {r: [round(float(np.mean([d["P"][d["realized"]] for d in trd if d["r"] == r])), 4),
                                                 round(float(np.mean([sum(x * x for x in d["P"]) for d in trd if d["r"] == r])), 4)]
                                             for r in range(5)}
    # U2 under the filtered model: r=0 is no longer an exact 3-way tie (the shown set itself is evidence)
    gaps = defaultdict(list)
    for d in trd:
        e_ = sorted(d["eu"], reverse=True)
        gaps[d["r"]].append(e_[0] - e_[1])
    A["eu_best_minus_second_by_r_train"] = {r: qstats(v) for r, v in sorted(gaps.items())}
    A["eu_set_differs_filtered_vs_unfiltered_by_r_train"] = {r: int(sum(1 for d in trd if d["r"] == r and
                                                                    set(d["eu_top"]) != set(d["eu_top_tie"]))) for r in range(5)}
    A["kl_P_sel_vs_P_tie_r4_train"] = A["declared_vs_release_process"]["train/KL(P_sel||P_tie)_by_r"][4]
    A["headline"] = {"records_by_split": A["records_by_split"], "decisions_by_split": A["decisions_by_split"],
                     "kl_filtered_vs_unfiltered_r4_train": A["kl_P_sel_vs_P_tie_r4_train"],
                     "u2_r0_one_hot": A["u_arms"]["U2"]["hard_labels_by_r"]["0"] if "hard_labels_by_r" in A["u_arms"]["U2"] else None,
                     "leakage": {k: v for k, v in A["leakage"].items() if k != "note"},
                     "eu_vs_pred_disjoint_train_file": [A["eu_vs_predictive_argmax"]["train_file_total_disjoint"],
                                                        A["eu_vs_predictive_argmax"]["train_file_total"]],
                     "u_arms": {k: {kk: v[kk] for kk in ("exposures", "tokens_total", "hard_labels")} for k, v in A["u_arms"].items()}}
    return A


# ============================================================================ U arms against a base mixture
def kbucket(k):
    return next(f"{lo}-{hi}" for lo, hi in ((2, 2), (3, 5), (6, 16), (17, 32), (33, 52)) if lo <= int(k) <= hi)


def lbucket(n):
    n = int(n)
    return "1-512" if n <= 512 else "513-2048" if n <= 2048 else "2049-8192" if n <= 8192 else ">8192"


def read_base(path):
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    op = gzip.open if path.endswith(".gz") else open
    return [json.loads(l) for l in op(path, "rt") if l.strip()]


def apply_uarms(base, blocks, id_field, tok_field, k_field=None, pos_field=None):
    """-> (slots, arms {U0..U3: rows}, report). Base rows in schedule order (sorted by pos_field when given). U1-U3 replace
    the SAME base slots: per (K bucket, length bucket) of the U block (length only when the base has no K field), the base
    rows of that cell with the smallest sha256(NS|uslot|id); the U block's rows fill those slots in block order."""
    if pos_field:
        base = sorted(base, key=lambda r: int(r[pos_field]))
    cell = lambda k, n: (kbucket(k) if k_field else "*", lbucket(n))      # noqa: E731
    need = Counter(cell(3, r["n_tokens"]) for r in blocks["U3"])
    cand = defaultdict(list)
    for i, e in enumerate(base):
        cand[cell(e[k_field] if k_field else 3, e[tok_field])].append(i)
    slots, short = [], {}
    for c, n in sorted(need.items()):
        cc = sorted(cand[c], key=lambda i: hrank(NS, "uslot", base[i][id_field]))
        if len(cc) < n:
            short[str(c)] = [n, len(cc)]
        slots += cc[:n]
    if short:
        raise SystemExit(f"base supply short in cells {short}")
    slots.sort()
    arms = {"U0": list(base)}
    for arm in blocks:
        rows = list(base)
        for j, i in enumerate(slots):
            rows[i] = {"u_arm": arm, "replaces": base[i][id_field], "slot": i, **blocks[arm][j]}
        arms[arm] = rows
    tok = {arm: int(sum(int(r[tok_field]) if tok_field in r and "u_arm" not in r else int(r["n_tokens"]) for r in rows))
           for arm, rows in arms.items()}
    rep = {"base_exposures": len(base), "replaced": len(slots), "replaced_share": round(len(slots) / len(base), 6),
           "need_by_cell": {str(k): v for k, v in need.items()}, "match_cells": "(K bucket, length bucket)" if k_field else
           "length bucket", "tokens": tok, "token_rel_diff_vs_U0": {a_: round(tok[a_] / tok["U0"] - 1, 6) for a_ in blocks},
           "replaced_sources": dict(Counter(str(base[i].get("upstream_source", base[i].get("source", "?"))) for i in slots).most_common(15))}
    return slots, arms, rep


def cmd_uarms(a):
    base = read_base(a.base)
    blocks = {arm: [json.loads(l) for l in open(Path(a.ublock) / f"exposures_{arm}.jsonl")]
              for arm in ("U1", "U1p", "U2", "U3") if (Path(a.ublock) / f"exposures_{arm}.jsonl").exists()}
    slots, arms, rep = apply_uarms(base, blocks, a.id_field, a.tokens_field, a.k_field or None, a.pos_field or None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rep["base"] = {"path": a.base, "sha256": hashlib.sha256(open(a.base, "rb").read()).hexdigest()}
    for arm, rows in arms.items():
        p = out / f"exposures_{arm}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        rep[arm] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "exposures": len(rows)}
    (out / "uarms_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


def uarms_selftest():
    rng = np.random.default_rng(1)
    base = [{"uid": f"b{i}", "pos": i, "tok_r2": int(rng.integers(80, 3000)), "K": int(rng.integers(2, 20)),
             "upstream_source": f"s{i % 37}"} for i in range(38400)]
    rng.shuffle(base)
    blocks = {arm: [{"decision_uid": f"d{j}/{arm}", "decision_group": f"d{j}", "n_tokens": int(rng.integers(400, 1100))
                     if arm == "U1" else None} for j in range(1920)] for arm in ("U1", "U1p", "U2", "U3")}
    for arm in ("U1p", "U2", "U3"):
        for j, r in enumerate(blocks[arm]):
            r["n_tokens"] = blocks["U1"][j]["n_tokens"] + (40 if arm == "U2" else 0)
    slots, arms, rep = apply_uarms(base, blocks, "uid", "tok_r2", "K", "pos")
    assert rep["replaced"] == 1920 and abs(rep["replaced_share"] - 0.05) < 1e-12
    for arm in ("U1", "U1p", "U2", "U3"):
        assert [r.get("decision_group") for r in arms[arm] if "u_arm" in r] == [f"d{j}" for j in range(1920)]
        assert [i for i, r in enumerate(arms[arm]) if "u_arm" in r] == slots
        assert all(arms[arm][i] is arms["U0"][i] for i in range(len(base)) if i not in set(slots))
    assert all(kbucket(arms["U0"][i]["K"]) == "3-5" for i in slots)
    return rep


# ============================================================================ self-test
def self_test():
    # ties exact: a stops-only profile on three flights sharing their stop count ties three ways
    opts = ((0, 0, 1, 0), (5, 5, 1, 5), (10, 10, 1, 10))
    R, A, na = choice_sets("flights", opts)
    i = int(np.flatnonzero((TH == [0, 0, 2, 0]).all(1))[0])
    assert na[i] == 3
    # r=0: EU is a 3-way tie by symmetry; the predictive sums to 1
    rng = np.random.default_rng(0)
    for _ in range(20):
        rounds = tuple(tuple(tuple(int(rng.integers(0, n)) for n in DOM["flights"]["nlev"]) for _ in range(3)) for _ in range(5))
        th = int(rng.integers(NTH))
        ch = []
        for o in rounds:
            R, A, na = choice_sets("flights", o)
            ch.append(int(rng.choice(np.flatnonzero(A[th]))))
        for r in range(5):
            inf = infer("flights", rounds, ch, r)
            assert abs(sum(inf["P"]) - 1) < 1e-12 and inf["P"][ch[r]] > 0
            if r == 0:
                assert inf["eu_top"] == [0, 1, 2]
            assert all(abs(sum(m) - 1) < 1e-12 for m in inf["marg"])
            # exact rational cross-check of the predictive
            from fractions import Fraction
            w = [Fraction(1)] * NTH
            for t in range(r):
                Rt, At, nat = choice_sets("flights", rounds[t])
                w = [w[k] * (Fraction(1, int(nat[k])) if At[k, ch[t]] else 0) for k in range(NTH)]
            Rc, Ac, nac = choice_sets("flights", rounds[r])
            Z = sum(w)
            P = [sum(w[k] * (Fraction(1, int(nac[k])) if Ac[k, j] else 0) for k in range(NTH)) / Z for j in range(3)]
            assert max(abs(float(x) - y) for x, y in zip(P, inf["P"])) < 1e-12
    for dom in DOM:
        assert len({DOM[dom]["line"](l) for l in itertools.product(*[range(n) for n in DOM[dom]["nlev"]])}) == \
            int(np.prod(DOM[dom]["nlev"]))
    import re as _re
    assert not _re.search(r"\bNC\b|non[- ]?commercial|research[- ]only|cc[- ]by[- ]nc|\bby-nc\b", LICENSE, _re.I)
    st = render_state("flights", [(rounds[0], ch[0])], rounds[1])
    assert "Round 2, the current round" in st and st.count("booked Flight") == 1
    # filtered (release-process) predictive: exact rational cross-check
    from fractions import Fraction
    allf = [phi20("flights", l) for l in itertools.product(*[range(n) for n in DOM["flights"]["nlev"]])]
    N = len(allf)
    rng = np.random.default_rng(5)
    nchk = 0
    while nchk < 6:
        th = int(rng.integers(NTH))
        rounds, ch = [], []
        while len(rounds) < 5:
            o = tuple(tuple(int(rng.integers(0, n)) for n in DOM["flights"]["nlev"]) for _ in range(3))
            if len(set(o)) < 3:
                continue
            R, A, na = choice_sets("flights", o)
            if na[th] != 1:
                continue
            rounds.append(o)
            ch.append(int(np.flatnonzero(A[th])[0]))
        rounds = tuple(rounds)
        pu = {}
        for k in range(NTH):
            vals = [int(f @ TH[k]) for f in allf]
            from collections import Counter as _C
            cnt = _C(vals)
            below, tot = 0, 0
            for v in sorted(cnt):
                tot += 3 * cnt[v] * below * (below - 1)
                below += cnt[v]
            pu[k] = Fraction(tot, N * (N - 1) * (N - 2))
        for r in (0, 2, 4):
            inf = infer("flights", rounds, ch, r, None, True)
            w = []
            for k in range(NTH):
                x = Fraction(1)
                for t in range(r + 1):
                    Rt, At, nat = choice_sets("flights", rounds[t])
                    if nat[k] != 1 or (t < r and not At[k, ch[t]]):
                        x = Fraction(0)
                        break
                    x /= pu[k]
                w.append(x)
            Z = sum(w)
            Rc, Ac, nac = choice_sets("flights", rounds[r])
            P = [sum(w[k] for k in range(NTH) if Ac[k, j]) / Z for j in range(3)]
            assert max(abs(float(x) - y) for x, y in zip(P, inf["P"])) < 1e-12, (P, inf["P"])
            assert inf["P"][ch[r]] > 0
            nchk += 1
    # v2.2: eps process. Bounds, and the predictive equals a direct exact enumeration of the noisy generative process
    eps, epsF = 0.15, Fraction(3, 20)
    rng = np.random.default_rng(11)
    for trial in range(12):
        rounds = tuple(tuple(tuple(int(rng.integers(0, n)) for n in DOM["flights"]["nlev"]) for _ in range(3)) for _ in range(5))
        if trial < 4:                                    # the reproducer shape: the same pick repeated
            ch = [0, 0, 0, 0, 0]
        else:
            ch = [int(rng.integers(3)) for _ in range(5)]
        for r in range(5):
            inf = infer("flights", rounds, ch, r, eps)
            assert all(eps / 3 - 1e-12 <= x <= 1 - eps + eps / 3 + 1e-12 for x in inf["P"]), inf["P"]
            assert all(eps / 3 - 1e-12 <= x <= 1 - eps + eps / 3 + 1e-12 for x in inf["ref_first_round_only"] + inf["ref_no_update"])
            def emit(k, t, j):
                Rt, At, nat = choice_sets("flights", rounds[t])
                return (1 - epsF) * (Fraction(1, int(nat[k])) if At[k, j] else 0) + epsF / 3
            joint = [Fraction(0)] * 3
            for k in range(NTH):                         # sum over profiles of prior * p(history) * p(next = j)
                h = Fraction(1, NTH)
                for t in range(r):
                    h *= emit(k, t, ch[t])
                for j in range(3):
                    joint[j] += h * emit(k, r, j)
            Z = sum(joint)
            assert max(abs(float(x / Z) - y) for x, y in zip(joint, inf["P"])) < 1e-12, (joint, inf["P"])
    # v2.3: eps support = declared posterior support; quarantine detection
    inf = infer("flights", rounds, [0, 1, 2, 0, 1], 4, 0.15)
    assert inf["support"] == NTH and inf["support_noiseless"] <= NTH and inf["post_entropy"] > 0
    th = int(np.flatnonzero((TH == [0, 0, 2, 0]).all(1))[0])        # stops-only profile
    tie_set = ((0, 0, 2, 0), (5, 5, 2, 5), (10, 10, 1, 10))         # flights 1 and 2 tie for it (2 stops each)
    ok_set = ((0, 0, 0, 0), (5, 5, 1, 5), (10, 10, 2, 10))
    e = {"theta_index": th, "dom": "flights", "rounds": (ok_set, ok_set, tie_set, ok_set, ok_set)}
    assert release_tie_round(e) == 2
    assert release_tie_round({**e, "rounds": (ok_set,) * 5}) is None
    stf = render_state("flights", [], rounds[0], None, True)
    assert "redrawn" in stf and "at random, each equally likely" not in stf
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("convert")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--scorer-dir", required=True)
    p.add_argument("--genholdout", default="")
    p.add_argument("--bp-path", default="")
    p.add_argument("--procs", type=int, default=4)
    p = sub.add_parser("uarms")
    p.add_argument("--ublock", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--id-field", default="uid")
    p.add_argument("--tokens-field", default="tok_r2")
    p.add_argument("--k-field", default="K")
    p.add_argument("--pos-field", default="pos")
    p.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "convert":
        cmd_convert(a)
    elif a.cmd == "uarms":
        cmd_uarms(a)
    elif a.cmd == "selftest":
        self_test()
        print(json.dumps(uarms_selftest()["token_rel_diff_vs_U0"]))
        print("uarms self-test OK")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
