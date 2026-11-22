#!/usr/bin/env python3
"""(PC) prior-conditioning continuation data: numeric_v2 train items, unpaired (arm N) and as prior-swap PAIRS (arm P).

Design: tmp/prior_cond/design.md; pre-registration draft: tmp/prior_cond/prereg.md. The registered BC99 next step (AGENT_NOTES
2026-10-10 19:40Z, row A): a (99)-init continuation adding numeric_v2 train WITH prior-swap augmentation vs matched controls.

Pools (numeric_v2 = fleetq/data/numeric_v2.tgz 5b7c4cb5, answer-independent v2 axes; families coin / rate / normal_mean):
  B1, B2   two DISJOINT seeded samples of numeric_v2 TRAIN items, N_PER_FAM per family each. Their records are copied VERBATIM
           (byte-identical lines; `check` verifies against the tarball's own train files).
  N        = B1 + B2 (6,000 units): numeric_v2 as (94 g v2) trained on it, unpaired.
  P        = B1 + SW(B1) (6,000 units): every B1 item AND its swap twin. SW(b) is item b redrawn by build_numeric_jev.draw(...,
           prior_override=...) -- the SAME truth, data, context, axis, wording streams and question; only the stated prior (the
           state's prior line) and hence the exact label differ.
  So N and P share B1 byte-for-byte and differ ONLY in B2 (3,000 fresh problems, each under its own prior) vs SW(B1) (the B1 problems
  again under a second prior). Units, families, the 50-bin format, the label kind and the trainer flags are identical.

The swap prior (exact marginal matching). For a base item with truth theta drawn by the generator's own truth model T (on-prior:
theta ~ prior; "broad": coin theta ~ U(0, 1), rate lam = mean x logU(1/8, 8), normal mu = m0 + s0 U(-5, 5)), the swap prior is drawn
from p(prior' | theta, T, context) proportional to p_gen(prior' | context) p_T(theta | prior') by sampling-importance-resampling
over SIR_L candidates from the generator's OWN prior draw (coin: beta_prior_draw + the v2 printed rounding; rate: shape from the
list, mean logU(0.05, 50) at 2 sig. digits, rate at 3; normal: given the item's latent scale S, sd S logU(0.01, 0.3) at 2 sig.
digits and mean S U(0.6, 1.6) on the r0 grid). Then (prior', theta, data) has the generator's joint distribution: a swap member is
marginally a numeric_v2 train item (up to SIR's finite-L approximation), so P's items are distributed as N's and the arms differ
in the COUPLING alone (pairs share theta / data / wording). A candidate equal to the base prior is excluded (no identical twins).
The normal latent S is replayed from the item's own seed (draw_v2's first two draws) and checked against the stated prior sd.

    python3 build_prior_cond.py build --numeric DIR/numeric_v2 --out DIR/pc_numeric_v1 [--suite DIR/bayes_consistency_v2]
    python3 build_prior_cond.py check DIR/pc_numeric_v1 --numeric DIR/numeric_v2 [--suite DIR/bayes_consistency_v2]
    python3 build_prior_cond.py self-test
build_numeric_jev.py is loaded from DIR/numeric_v2/tools/ (the tarball's own generator) unless --bn names another copy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

VERSION = "build_prior_cond.py v1 (2026-10-11)"
NS_PC = "prior_cond/v1"
FAMS = ("coin", "rate", "normal_mean")
N_PER_FAM = 1000                    # per family, in each of B1 and B2
N_TRAIN = 20000                     # numeric_v2 train items per family
SIR_L = 1024                        # SIR candidates per swap (a further block of SIR_L if no candidate has positive weight)
SIR_MAX_BLOCKS = 16
RATE_SHAPES = (0.5, 1, 1.5, 2, 3, 4, 5, 8, 10, 20)     # gen_rate's list
LABEL_TOL = 2e-8                    # labels are rounded to 8 decimals by the generator
HERE = Path(__file__).resolve().parent


# ============================================================================ the generator
def load_bn(path):
    spec = importlib.util.spec_from_file_location("build_numeric_jev_pc", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("draw", "make_record", "beta_prior_draw", "fparam", "fsig", "logu", "pow10_floor", "dec_for_res", "seed_of",
                 "NORM_CTX", "NS", "check_v2_record"):
        if not hasattr(mod, name):
            raise SystemExit(f"{path}: build_numeric_jev has no {name} (not the v2 generator)")
    return mod


def bn_path(numeric, explicit=None):
    p = Path(explicit) if explicit else Path(numeric) / "tools" / "build_numeric_jev.py"
    if not p.exists():
        raise SystemExit(f"no generator at {p}")
    return p


def rec_line(rec):
    return json.dumps(rec, ensure_ascii=False)


# The generator's records are NOT byte-reproducible across platforms (2026-10-11, x7 node, Linux x86_64 / Python 3.12.13 / numpy 2.2.6 /
# scipy 1.17.1, against the arm64 laptop build / Python 3.13.15 / numpy 2.5.3 / scipy 1.18.1): numeric_v2's own train records differ
# there too, ONLY in unrounded float diagnostics under meta (binned moments, truth value, axis zoom / data interval, folded masses), by
# <= 3.6e-12 relative / 1.8e-12 absolute (libm / numpy / scipy last-ulp). Everything the trainer reads -- state, questions, labels --
# and every non-float field was identical on all 3,000 swaps and 3,000 bases. So regeneration is compared with record_diffs: every
# leaf identical (keys, strings, ints, bools, labels, the structure), except float leaves under meta, which must agree to FLOAT_REL /
# FLOAT_ABS. Labels are training targets (rounded to 8 decimals by the generator) and stay EXACT.
FLOAT_REL, FLOAT_ABS = 1e-9, 1e-12


def record_diffs(a, b, path=""):
    """[(path, a, b)] where two records differ beyond platform float noise (see above)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                out += record_diffs(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(path, f"len {len(a)}", f"len {len(b)}")]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += record_diffs(x, y, f"{path}[{i}]")
        return out
    if type(a) is float and type(b) is float and path.startswith("/meta/"):
        return [] if (a == b or math.isclose(a, b, rel_tol=FLOAT_REL, abs_tol=FLOAT_ABS)) else [(path, a, b)]
    return [] if (type(a) is type(b) and a == b) else [(path, a, b)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select_indices(fam, n_per=N_PER_FAM, n_train=N_TRAIN):
    """B1, B2: disjoint seeded samples of item indices (one fixed RNG per family; never depends on the data)."""
    rng = random.Random(f"{NS_PC}|select|{fam}")
    pick = rng.sample(range(n_train), 2 * n_per)
    return sorted(pick[:n_per]), sorted(pick[n_per:])


def regenerate(bn, fam, i, line, max_att=200):
    """(attempt, g) such that make_record(draw(fam, train, i, attempt)) reproduces `line` byte for byte; else SystemExit."""
    for att in range(max_att + 1):
        g = bn.draw(fam, "train", i, bn.NS, att, axes="v2")
        if rec_line(bn.make_record(g, "train", i, bn.NS, False)) == line:
            return att, g
    raise SystemExit(f"{fam} train item {i}: the generator does not reproduce numeric_v2's record (not the same build)")


def normal_latent_scale(bn, i, att, g):
    """The normal item's latent scale S, replayed from draw_v2's own streams (ctx = py.choice, S = logU over ctx['S']); checked:
    the context id, and the next rng draw reproduces the stated prior sd."""
    s = bn.seed_of(bn.NS, "train", "normal_mean", i, att)
    py, rng = random.Random(s), np.random.default_rng(s)
    ctx = py.choice(bn.NORM_CTX)
    S = bn.logu(rng, *ctx["S"])
    s0 = float(bn.fsig(S * bn.logu(rng, 0.01, 0.3), 2))
    if ctx["id"] != g["context"] or s0 != float(g["prior_params"]["sd"]):
        raise SystemExit(f"normal item {i}: latent-scale replay failed ({ctx['id']} / {g['context']}, sd {s0} / {g['prior_params']['sd']})")
    return float(S), ctx


def truth_nat(g):
    """(theta in natural units, truth model 'prior' | 'broad')."""
    t = g["truth"]
    return float(t["value"]) / float(g["axis"]["scale"]), ("broad" if t["source"] == "broad" else "prior")


# ============================================================================ swap priors (SIR from p_gen(prior') p_T(theta | prior'))
def candidates(bn, fam, rng, py, n, S=None):
    out = []
    for _ in range(n):
        if fam == "coin":
            a, b = bn.beta_prior_draw(rng, py)
            out.append({"a": float(bn.fparam(a)), "b": float(bn.fparam(b))})
        elif fam == "rate":
            shape = float(py.choice(RATE_SHAPES))
            mu0 = float(bn.fsig(bn.logu(rng, 0.05, 50), 2))
            out.append({"shape": shape, "rate": float(bn.fsig(shape / mu0, 3))})
        else:
            s0 = float(bn.fsig(S * bn.logu(rng, 0.01, 0.3), 2))
            r0 = bn.pow10_floor(s0 / 3)
            m0 = round(round(S * rng.uniform(0.6, 1.6) / r0) * r0, bn.dec_for_res(r0))
            out.append({"mean": float(m0), "sd": s0})
    return out


def log_weights(fam, cands, theta, model):
    """log p_T(theta | prior c) for every candidate c (vectorised), up to a constant shared by all candidates of one item."""
    if fam == "coin":
        a = np.array([c["a"] for c in cands]); b = np.array([c["b"] for c in cands])
        return np.zeros(len(cands)) if model == "broad" else stats.beta.logpdf(theta, a, b)
    if fam == "rate":
        k = np.array([c["shape"] for c in cands]); r = np.array([c["rate"] for c in cands])
        if model == "broad":
            m = k / r
            return np.where((m / 8 <= theta) & (theta <= 8 * m), 0.0, -np.inf)
        return stats.gamma.logpdf(theta, k, scale=1.0 / r)
    m = np.array([c["mean"] for c in cands]); s = np.array([c["sd"] for c in cands])
    if model == "broad":
        return np.where(np.abs(theta - m) <= 5 * s, -np.log(10 * s), -np.inf)
    return stats.norm.logpdf(theta, m, s)


def log_weight(fam, c, theta, model):
    return float(log_weights(fam, [c], theta, model)[0])


def same_prior(fam, c, base):
    keys = {"coin": ("a", "b"), "rate": ("shape", "rate"), "normal_mean": ("mean", "sd")}[fam]
    return all(float(c[k]) == float(base[k]) for k in keys)


def swap_prior(bn, fam, i, att, g, L=SIR_L, accept=None, max_picks=64):
    """-> (override dict for draw(prior_override=...), info). Deterministic in (fam, i, att). Picks are drawn in sequence from the
    SIR weights; `accept(override)` (the builder's duplicate-text rule, as build_numeric_jev redraws a question that repeats an
    eval question) may refuse a pick, and the next pick is taken (pick_rank records how many were refused)."""
    theta, model = truth_nat(g)
    S = normal_latent_scale(bn, i, att, g)[0] if fam == "normal_mean" else None
    seed = bn.seed_of(NS_PC, fam, i, att, "cand")
    rng, py = np.random.default_rng(seed), random.Random(seed)
    base = g["prior_params"]
    cands, lw = [], np.zeros(0)
    for blk in range(SIR_MAX_BLOCKS):
        new = candidates(bn, fam, rng, py, L, S)
        w_new = np.asarray(log_weights(fam, new, theta, model), float)
        w_new[[same_prior(fam, c, base) for c in new]] = -np.inf
        cands += new
        lw = np.concatenate([lw, np.nan_to_num(w_new, nan=-np.inf)])
        if np.isfinite(lw.max()):
            break
    if not np.isfinite(lw.max()):
        raise SystemExit(f"{fam} item {i}: no swap candidate with positive weight in {len(cands)} draws")
    w = np.exp(lw - lw.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    prng = np.random.default_rng(bn.seed_of(NS_PC, fam, i, att, "pick"))
    for rank in range(max_picks):
        j = int(prng.choice(len(w), p=w))
        c = cands[j]
        ov = {"a": c["a"], "b": c["b"]} if fam == "coin" else ({"shape": c["shape"], "rate": c["rate"]} if fam == "rate"
                                                               else {"mean": c["mean"], "sd": c["sd"]})
        if accept is None or accept(ov):
            return ov, {"truth_model": model, "theta": theta, "latent_S": S, "n_candidates": len(cands),
                        "n_positive": int(np.isfinite(lw).sum()), "ess": round(ess, 2), "pick": j, "pick_rank": rank}
    raise SystemExit(f"{fam} item {i}: {max_picks} swap picks all refused (duplicate texts)")


# ============================================================================ independent label recomputation (scipy.stats; not the generator's code)
def parse_prior_line(fam, line):
    num = r"(-?[0-9]+(?:\.[0-9]+)?)"
    if fam == "coin":
        if re.search(r"uniform on \[0, 1\] \(Beta\(1, 1\)\)|equally likely \(a Beta\(1, 1\) prior\)", line):
            return {"a": 1.0, "b": 1.0}
        m = re.search(rf"Beta\({num}, {num}\)", line) or re.search(rf"a = {num} and b = {num}", line)
        return {"a": float(m.group(1)), "b": float(m.group(2))}
    if fam == "rate":
        m = (re.search(rf"Gamma\(shape {num}, rate {num}\)", line) or re.search(rf"shape α = {num} and rate β = {num}", line)
             or re.search(rf"Gamma\(α = {num}, β = {num}\)", line))
        return {"shape": float(m.group(1)), "rate": float(m.group(2))}
    m = re.search(rf"Normal\(mean {num}, sd {num}\)", line) or \
        re.search(rf"normal with mean {num} \S.*? and standard deviation {num} ", line + " ")
    return {"mean": float(m.group(1)), "sd": float(m.group(2))}


def exact_labels(fam, prior, data, edges_disp, scale):
    """Bin masses of the exact posterior on the declared bins, tails folded into the end bins."""
    e = np.asarray(edges_disp, float) / scale
    if fam == "coin":
        d = stats.beta(prior["a"] + data["h"], prior["b"] + data["n"] - data["h"])
    elif fam == "rate":
        d = stats.gamma(prior["shape"] + data["total"], scale=1.0 / (prior["rate"] + data["exposure"]))
    else:
        n, s0, sig = data["n"], prior["sd"], data["sigma"]
        prec = 1 / s0 ** 2 + (n / sig ** 2 if n else 0.0)
        mn = (prior["mean"] / s0 ** 2 + (n * data["xbar"] / sig ** 2 if n else 0.0)) / prec
        d = stats.norm(mn, math.sqrt(1 / prec))
    c = d.cdf(e)
    inner = np.concatenate([[0.0], c[1:-1], [1.0]])
    return np.diff(np.maximum.accumulate(np.clip(inner, 0.0, 1.0)))


def label_vec(rec):
    return np.array(list(next(iter(rec["labels"].values())).values()), float)


def qkey(rec):
    return rec["state"] + "\n" + next(iter(rec["questions"].values()))["instructions"]


def pair_problems(base, swap):
    """Every way a swap record may differ from its base other than the stated prior and the label (empty = a valid pair)."""
    errs = []
    bl, sl = base["state"].split("\n"), swap["state"].split("\n")
    if len(bl) != len(sl) or len(bl) < 3:
        return [f"state line count {len(bl)} / {len(sl)}"]
    diff = [k for k in range(len(bl)) if bl[k] != sl[k]]
    if diff != [1]:
        errs.append(f"state lines differing {diff} (only the prior line, index 1, may)")
    qb, qs = next(iter(base["questions"].values())), next(iter(swap["questions"].values()))
    if qb != qs:
        errs.append("question (instructions / levels) differs")
    mb, ms = base["meta"], swap["meta"]
    for k in ("family", "split", "format", "context", "quantity", "axis", "bin_edges", "data", "axes", "generator", "seed_namespace"):
        if mb.get(k) != ms.get(k):
            errs.append(f"meta.{k} differs")
    if mb["truth"]["value"] != ms["truth"]["value"]:
        errs.append("truth value differs")
    if ms["truth"].get("source") != "other_prior":
        errs.append("swap truth.source is not other_prior")
    fam = mb["family"]
    pb, ps = parse_prior_line(fam, bl[1]), parse_prior_line(fam, sl[1])
    keys = {"coin": ("a", "b"), "rate": ("shape", "rate"), "normal_mean": ("mean", "sd")}[fam]
    if any(float(ps[k]) != float(ms["prior"]["stated"][k]) for k in keys):
        errs.append(f"printed swap prior {ps} != used {ms['prior']['stated']}")
    if any(float(pb[k]) != float(mb["prior"]["stated"][k]) for k in keys):
        errs.append(f"printed base prior {pb} != used {mb['prior']['stated']}")
    if all(float(ps[k]) == float(pb[k]) for k in keys):
        errs.append("swap prior equals the base prior")
    if base["license"] != swap["license"] or base["source"] != swap["source"]:
        errs.append("licence / source differs")
    return errs


# ============================================================================ build
def read_train(numeric):
    out = {}
    for fam in FAMS:
        p = Path(numeric) / "train" / f"numeric_{fam}.jsonl"
        out[fam] = p.read_text(encoding="utf-8").splitlines()
        if len(out[fam]) != N_TRAIN:
            raise SystemExit(f"{p}: {len(out[fam])} lines, expected {N_TRAIN}")
    return out


def numeric_keys(numeric):
    keys = set()
    for p in sorted(Path(numeric).glob("*/*.jsonl")):
        for l in open(p, encoding="utf-8"):
            keys.add(qkey(json.loads(l)))
    return keys


def coin_key(rec):
    """a numeric coin record's problem (stated prior a, b; n, h), comparable with the suite's coin fingerprints"""
    st, d = rec["meta"]["prior"]["stated"], rec["meta"]["data"]
    return (float(st["a"]), float(st["b"]), int(d["n"]), int(d["h"]))


def suite_index(suite):
    """(question texts, coin problem keys) of the Bayesian-consistency suite (the eval set), or (set(), set()). The suite's
    fingerprint is 'coin|{prior json}|{data json}'; rate / normal problems are continuous and practically never coincide."""
    if not suite:
        return set(), set()
    texts, fps = set(), set()
    for p in sorted(Path(suite).glob("test/*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            for q in r["questions"].values():
                texts.add(r["state"] + "\n" + q["instructions"])
            fp = (r.get("meta") or {}).get("fingerprint") or ""
            if fp.startswith("coin|"):
                _, pr, da = fp.split("|", 2)
                pr, da = json.loads(pr), json.loads(da)
                if {"a", "b"} <= set(pr) and {"n", "h"} <= set(da):
                    fps.add((float(pr["a"]), float(pr["b"]), int(da["n"]), int(da["h"])))
    return texts, fps


def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def build(numeric, out, bnp=None, suite=None, n_per=N_PER_FAM, log=print):
    t0 = time.time()
    bn = load_bn(bn_path(numeric, bnp))
    lines = read_train(numeric)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: refusing to overwrite a build")
    for d in ("N/train", "P/train", "tools", "reference"):
        (out / d).mkdir(parents=True, exist_ok=True)
    man = {"version": VERSION, "namespace": NS_PC, "numeric_v2": {"dir": str(numeric), "generator": str(bn_path(numeric, bnp)),
           "generator_md5": md5_file(bn_path(numeric, bnp)), "train_files_sha256": {}}, "n_per_family": n_per, "sir_L": SIR_L,
           "families": {}, "files": {}}
    for fam in FAMS:
        man["numeric_v2"]["train_files_sha256"][fam] = hashlib.sha256(("\n".join(lines[fam]) + "\n").encode()).hexdigest()
    diag = {}
    forbidden = numeric_keys(numeric) | suite_index(suite)[0]     # no swap may repeat any numeric_v2 / suite question, or another swap
    for fam in FAMS:
        b1, b2 = select_indices(fam, n_per)
        base_b1, base_b2, swaps, pairs = [], [], [], []
        for i in b1:
            line = lines[fam][i]
            att, g = regenerate(bn, fam, i, line)

            def accept(ov, fam=fam, i=i, att=att):
                k = qkey(bn.make_record(bn.draw(fam, "train", i, bn.NS, att, axes="v2", prior_override=ov), "train", i, bn.NS, False))
                return k not in forbidden
            ov, info = swap_prior(bn, fam, i, att, g, accept=accept)
            g2 = bn.draw(fam, "train", i, bn.NS, att, axes="v2", prior_override=ov)
            rec2 = bn.make_record(g2, "train", i, bn.NS, False)
            forbidden.add(qkey(rec2))
            base = json.loads(line)
            t = tv(label_vec(base), label_vec(rec2))
            rec2["meta"]["prior_cond"] = {"version": VERSION, "member": "swap", "pair_of": base["id"], "item_index": i, "attempt": att,
                                          "base_prior": g["prior_params"], "label_tv_vs_base": round(t, 6),
                                          **{k: v for k, v in info.items() if k != "theta"}}
            errs = pair_problems(base, rec2)
            if errs:
                raise SystemExit(f"{fam} item {i}: invalid pair: {errs}")
            base_b1.append(line); swaps.append(rec_line(rec2))
            pairs.append({"i": i, "attempt": att, "base_id": base["id"], "swap_id": rec2["id"], "tv": round(t, 6),
                          "truth_model": info["truth_model"], "ess": info["ess"], "n_candidates": info["n_candidates"],
                          "pick_rank": info["pick_rank"]})
        for i in b2:
            base_b2.append(lines[fam][i])
        (out / "N/train" / f"numeric_{fam}.jsonl").write_text("\n".join(base_b1 + base_b2) + "\n", encoding="utf-8")
        (out / "P/train" / f"numeric_{fam}.jsonl").write_text("\n".join(base_b1) + "\n", encoding="utf-8")
        (out / "P/train" / f"numeric_{fam}__pcswap.jsonl").write_text("\n".join(swaps) + "\n", encoding="utf-8")
        man["families"][fam] = {"B1": b1, "B2": b2, "pairs": pairs}
        diag[fam] = diagnostics(fam, [json.loads(x) for x in base_b1], [json.loads(x) for x in swaps],
                                [json.loads(x) for x in base_b2], pairs)
        log(f"  {fam}: B1 {len(b1)} + swaps {len(swaps)} (P), B1 + B2 {len(b1) + len(b2)} (N); swap TV median "
            f"{diag[fam]['swap_tv']['median']:.3f}; ESS min {diag[fam]['sir']['ess_min']:.1f} ({time.time() - t0:.0f} s)")
    for f in sorted(out.glob("*/train/*.jsonl")):
        man["files"][str(f.relative_to(out))] = {"lines": sum(1 for _ in open(f, encoding="utf-8")), "sha256": sha256_file(f)}
    man["diagnostics"] = diag
    man["build_secs"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    rep = check(out, numeric, bnp, suite, log=log)
    (out / "reference" / "check.json").write_text(json.dumps(rep, indent=1))
    for src in (Path(__file__).resolve(), bn_path(numeric, bnp)):
        (out / "tools" / src.name).write_bytes(src.read_bytes())
    return man, rep


def diagnostics(fam, b1, sw, b2, pairs):
    """Descriptive (not gates): how different the twins' labels are, prior-data conflict, folding, and the marginal of the
    stated priors of the swaps against B2's (independent problems from the generator): they should agree (exact SIR)."""
    def q(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "max": float(v.max())}
    tvs = [p["tv"] for p in pairs]
    dm = []      # exact posterior mean shift base -> swap, in base posterior sds (the suite's dm_ex)
    for b, s in zip(b1, sw):
        dm.append(abs(s["meta"]["exact"]["mean"] - b["meta"]["exact"]["mean"]) / max(b["meta"]["exact"]["sd"], 1e-300))
    fold = lambda rs: [r["meta"]["exact"]["folded_low"] + r["meta"]["exact"]["folded_high"] for r in rs]
    def feats(rs):
        st = [r["meta"]["prior"]["stated"] for r in rs]
        if fam == "coin":
            return {"log_strength": [math.log(x["a"] + x["b"]) for x in st], "mean": [x["a"] / (x["a"] + x["b"]) for x in st]}
        if fam == "rate":
            return {"shape": [x["shape"] for x in st], "log_mean": [math.log(x["shape"] / x["rate"]) for x in st]}
        return {"log_sd_over_axis": [math.log(x["sd"]) for x in st]}
    ks = {}
    fs, f2 = feats(sw), feats(b2)
    for k in fs:
        r = stats.ks_2samp(fs[k], f2[k])
        ks[k] = {"ks": round(float(r.statistic), 4), "p": round(float(r.pvalue), 4)}
    ess = [p["ess"] for p in pairs]
    return {"swap_tv": q(tvs), "share_tv_gt_0.05": float(np.mean(np.asarray(tvs) > 0.05)),
            "share_tv_lt_0.01": float(np.mean(np.asarray(tvs) < 0.01)), "dm_exact_sds": q(dm),
            "folded": {"B1": q(fold(b1)), "swap": q(fold(sw)), "B2": q(fold(b2))},
            "share_folded_gt_0.25": {"B1": float(np.mean(np.asarray(fold(b1)) > 0.25)), "swap": float(np.mean(np.asarray(fold(sw)) > 0.25)),
                                     "B2": float(np.mean(np.asarray(fold(b2)) > 0.25))},
            "truth_model": dict(Counter(p["truth_model"] for p in pairs)),
            "prior_marginal_swap_vs_B2_ks": ks,
            "sir": {"ess_min": float(min(ess)), "ess_median": float(np.median(ess)),
                    "extra_blocks": int(sum(p["n_candidates"] > SIR_L for p in pairs)),
                    "refused_picks": int(sum(p["pick_rank"] for p in pairs)),
                    "pairs_with_a_refused_pick": int(sum(p["pick_rank"] > 0 for p in pairs))}}


# ============================================================================ check
def check(out, numeric, bnp=None, suite=None, log=print):
    """Every invariant the design relies on; raises SystemExit on the first class of failure, else returns a report."""
    out = Path(out)
    man = json.loads((out / "manifest.json").read_text())
    lines = read_train(numeric)
    by_id = {fam: {json.loads(l)["id"]: l for l in lines[fam]} for fam in FAMS}
    nkeys = numeric_keys(numeric)
    s_texts, s_fps = suite_index(suite)
    rep = {"version": VERSION, "errors": [], "counts": {}, "suite_checked": bool(suite)}
    E = rep["errors"]
    bn = load_bn(bn_path(numeric, bnp))
    keys_of = {"coin": ("a", "b"), "rate": ("shape", "rate"), "normal_mean": ("mean", "sd")}
    n_regen = Counter()
    seen_keys = set()
    fp_suite = Counter()
    max_lab = 0.0
    for fam in FAMS:
        fm = man["families"][fam]
        b1, b2 = fm["B1"], fm["B2"]
        exp1, exp2 = select_indices(fam, man["n_per_family"])
        if (b1, b2) != (exp1, exp2):
            E.append(f"{fam}: B1 / B2 indices are not the registered seeded selection")
        if set(b1) & set(b2) or len(set(b1)) != len(b1) or len(set(b2)) != len(b2):
            E.append(f"{fam}: B1 and B2 overlap or repeat")
        n_lines = (out / "N/train" / f"numeric_{fam}.jsonl").read_text(encoding="utf-8").splitlines()
        p_lines = (out / "P/train" / f"numeric_{fam}.jsonl").read_text(encoding="utf-8").splitlines()
        s_lines = (out / "P/train" / f"numeric_{fam}__pcswap.jsonl").read_text(encoding="utf-8").splitlines()
        if n_lines != [lines[fam][i] for i in b1] + [lines[fam][i] for i in b2]:
            E.append(f"{fam}: N is not B1 + B2 verbatim (numeric_v2 train lines)")
        if p_lines != [lines[fam][i] for i in b1]:
            E.append(f"{fam}: P's base members are not B1 verbatim")
        if len(s_lines) != len(b1):
            E.append(f"{fam}: {len(s_lines)} swaps for {len(b1)} B1 items")
        for k, (i, sl) in enumerate(zip(b1, s_lines)):
            s = json.loads(sl)
            pc = s["meta"].get("prior_cond") or {}
            base_line = by_id[fam].get(pc.get("pair_of"))
            if base_line is None or base_line != lines[fam][i] or pc.get("item_index") != i:
                E.append(f"{fam} swap {k}: pair_of / item_index does not name B1 item {i}")
                continue
            base = json.loads(base_line)
            for e in pair_problems(base, s):
                E.append(f"{fam} item {i}: {e}")
            # the swap is the generator's record for (item, attempt) with only the stated prior overridden: every field identical,
            # unrounded meta floats to platform noise (record_diffs); byte-identical on the build platform (n_byte_identical)
            st0 = s["meta"]["prior"]["stated"]
            att = int(pc.get("attempt", -1))
            regen = bn.make_record(bn.draw(fam, "train", i, bn.NS, att, axes="v2", prior_override={k: st0[k] for k in keys_of[fam]}),
                                   "train", i, bn.NS, False)
            regen["meta"]["prior_cond"] = pc
            regen = json.loads(rec_line(regen))                   # the shipped representation (JSON types), then compare
            d = record_diffs(s, regen)
            if d:
                E.append(f"{fam} item {i}: the swap is not the generator's record for this item with its stated prior: {d[:2]}")
            n_regen["swap_byte_identical"] += rec_line(regen) == sl
            # and the base: the same generator, attempt and item give numeric_v2's record (the attempt is the one the swap used)
            db = record_diffs(base, json.loads(rec_line(bn.make_record(bn.draw(fam, "train", i, bn.NS, att, axes="v2"), "train", i, bn.NS,
                                                                        False))))
            if db:
                E.append(f"{fam} item {i}: the generator (attempt {att}) does not regenerate numeric_v2's base record: {db[:2]}")
            n_regen["checked"] += 1
            st = s["meta"]["prior"]["stated"]
            lab = exact_labels(fam, st, s["meta"]["data"], s["meta"]["bin_edges"], s["meta"]["axis"]["scale"])
            got = label_vec(s)
            d = float(np.abs(lab - got).max())
            max_lab = max(max_lab, d)
            if d > LABEL_TOL or abs(got.sum() - 1) > 1e-6 or len(got) != 50:
                E.append(f"{fam} item {i}: swap label is not the exact posterior under the stated prior (max |d| {d:.2e})")
            bl = exact_labels(fam, base["meta"]["prior"]["stated"], base["meta"]["data"], base["meta"]["bin_edges"],
                              base["meta"]["axis"]["scale"])
            if float(np.abs(bl - label_vec(base)).max()) > LABEL_TOL:
                E.append(f"{fam} item {i}: base label not reproduced by the independent recomputation")
            key = qkey(s)
            if key in nkeys:
                E.append(f"{fam} item {i}: swap question text occurs in numeric_v2")
            if key in s_texts:
                E.append(f"{fam} item {i}: swap question text occurs in the consistency suite")
            if key in seen_keys:
                E.append(f"{fam} item {i}: duplicate swap question text")
            seen_keys.add(key)
            if fam == "coin":
                fp_suite["swap"] += coin_key(s) in s_fps
                fp_suite["B1"] += coin_key(base) in s_fps
        if fam == "coin":
            for i in b2:
                fp_suite["B2"] += coin_key(json.loads(lines[fam][i])) in s_fps
        rep["counts"][fam] = {"N": len(n_lines), "P_base": len(p_lines), "P_swap": len(s_lines)}
    for rel, f in man["files"].items():
        p = out / rel
        if not p.exists() or sha256_file(p) != f["sha256"]:
            E.append(f"{rel}: missing or sha256 differs from the manifest")
    units = {arm: sum(sum(1 for _ in open(p, encoding="utf-8")) for p in (out / arm / "train").glob("*.jsonl")) for arm in ("N", "P")}
    rep["units"] = units
    if units["N"] != units["P"] or units["N"] != 2 * len(FAMS) * man["n_per_family"]:
        E.append(f"unit counts N {units['N']} / P {units['P']} (expected {2 * len(FAMS) * man['n_per_family']} each)")
    rep["max_label_dev"] = max_lab
    rep["regeneration"] = {**dict(n_regen), "float_rel_tol": FLOAT_REL, "float_abs_tol": FLOAT_ABS,
                           "rule": "every field identical; only unrounded float leaves under /meta/ may differ, by platform noise"}
    rep["coin_problems_in_suite"] = dict(fp_suite)       # descriptive: small-n coin problems recur by chance (numeric_v2 train too)
    if suite and fp_suite["swap"] > fp_suite["B2"] + 0.01 * man["n_per_family"]:
        E.append(f"swaps share more suite coin problems ({fp_suite['swap']}) than fresh items ({fp_suite['B2']}): P would gain test exposure")
    rep["n_errors"] = len(E)
    log(f"check: {len(E)} errors; units {units}; max label deviation {max_lab:.2e}; regenerated {n_regen['checked']} swaps + bases "
        f"({n_regen['swap_byte_identical']} swaps byte-identical on this platform); coin problems also in the suite {dict(fp_suite)} "
        f"({'suite checked' if suite else 'NO suite given'})")
    return rep


# ============================================================================ replay: R / R2 from (99)'s training set X99, and mix2 val
REPLAY_UNITS = 6000                 # units in R and in R2 (= the numeric pool's units: every arm has 12,000 = 1,500 steps x 8)
K_MAX = 52                          # the trainer drops units with more options than letters; such records are never sampled


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def x99_files(src):
    """(99)'s training set as fleet/task_ddp.sh 99 assembles it, as an ordered list of (X99 file name, path, filter_real): jev_mix2
    train (jev_data train minus jev_real test states, + ext_strict / ext2 under their mix2 names), rd3_ / bp_ / gdelt_ prefixed train
    files. Sorted by the X99 name (the order the trainer lists the directory). And mix2 VAL the same way (X99/val = mix2 val)."""
    def mix2(split):
        out = [(Path(p).name, Path(p), True) for p in sorted(glob_(src["jev_data"], split, "*.jsonl"))]
        for key, pre in (("jev_ext_strict", "ext_"), ("jev_ext2", "ext2_")):
            base = Path(src[key]) / split
            for p in sorted(base.glob("**/*.jsonl")):
                out.append((pre + str(p.relative_to(base)).replace("/", "__"), p, False))
        return out
    train = mix2("train")
    for key, pre, sub in (("jev_rd3", "rd3_", "train"), ("bp_train_v1", "bp_", "exact/train"), ("jev_gdelt_v1", "gdelt_", "train")):
        train += [(pre + p.name, p, False) for p in sorted((Path(src[key]) / sub).glob("*.jsonl"))]
    return sorted(train, key=lambda x: x[0]), sorted(mix2("val"), key=lambda x: x[0])


def glob_(root, split, pat):
    return [str(p) for p in (Path(root) / split).glob(pat)]


def replay(src, out, jev_real, cf_path, ov_path, units=REPLAY_UNITS, inputs=None, log=print):
    """Two DISJOINT seeded record samples of X99 train with EXACTLY `units` units each (R, R2), and mix2 val; verbatim lines.
    Eligible records: state not in the rd3 stack pool (never trained on) nor in jev_real test; not non-commercial (cache_features
    rule, judged with the record's ORIGINAL file stem); every question <= 52 options."""
    t0 = time.time()
    cf, ov = load_path("cf_pc", cf_path), load_path("ov_pc", ov_path)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: refusing to overwrite a build")
    real = Path(jev_real) / "test" / "real.jsonl"
    real_states = {ov.norm(json.loads(l)["state"]) for l in open(real, encoding="utf-8") if l.strip()}
    n_real = sum(len(json.loads(l)["questions"]) for l in open(real, encoding="utf-8") if l.strip())
    pool_states = set()
    for p in sorted(Path(src["jev_rd3_stackpool"]).glob("**/*.jsonl")):
        pool_states.update(ov.norm(json.loads(l)["state"]) for l in open(p, encoding="utf-8") if l.strip())
    if not pool_states:
        raise SystemExit("empty stack pool: wrong directory")
    train_files, val_files = x99_files(src)
    recs, drop = [], Counter()          # (file index, line); eligibility
    for fi, (name, p, filt) in enumerate(train_files):
        stem = Path(name).stem
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            st = ov.norm(r["state"])
            if st in real_states:
                drop["jev_real_test_state" + ("" if filt else "_nonmix2")] += 1; continue
            if st in pool_states:
                drop["stackpool_state"] += 1; continue
            lic = str(r.get("license") or (r.get("meta") or {}).get("license") or "")
            if cf.is_noncommercial({"source": r.get("source") or stem, "license": lic}):
                drop["noncommercial"] += 1; continue
            ks = [len(cf.question_options(q, r["labels"][qid])[0]) for qid, q in r["questions"].items()]
            if not ks or max(ks) > K_MAX:
                drop["k_gt_52_or_empty"] += 1; continue
            recs.append((fi, l if l.endswith("\n") else l + "\n", len(ks)))
    order = list(range(len(recs)))
    random.Random(f"{NS_PC}|replay").shuffle(order)
    picks, got = ([], []), [0, 0]
    for k in order:
        n = recs[k][2]
        for j in (0, 1):
            if got[j] < units:
                if got[j] + n <= units:
                    picks[j].append(k); got[j] += n
                break
        if got[0] == units and got[1] == units:
            break
    if got != [units, units]:
        raise SystemExit(f"could not fill R / R2 exactly: {got}")
    comp = {}
    for tag, sel in (("R", picks[0]), ("R2", picks[1])):
        d = out / tag / "train"; d.mkdir(parents=True)
        by = defaultdict(list)
        for k in sorted(sel):
            by[recs[k][0]].append(recs[k][1])
        pre = "r_" if tag == "R" else "r2_"
        for fi, ls in sorted(by.items()):
            (d / (pre + train_files[fi][0])).write_text("".join(ls), encoding="utf-8")
        cls = lambda name: {"rd3_": "rd3", "bp_": "flights", "gdelt_": "gdelt"}.get(next((x for x in ("rd3_", "bp_", "gdelt_")
                                                                                       if name.startswith(x)), ""), "mix2")
        comp[tag] = dict(Counter(cls(train_files[recs[k][0]][0]) for k in sel for _ in range(recs[k][2])))
    vd = out / "val"; vd.mkdir(parents=True)
    for name, p, filt in val_files:
        ls = [l if l.endswith("\n") else l + "\n" for l in open(p, encoding="utf-8")
              if l.strip() and not (filt and ov.norm(json.loads(l)["state"]) in real_states)]
        (vd / name).write_text("".join(ls), encoding="utf-8")
    # verification through the TRAINER's own unit reader and NC rule on the written files (the stems are now r_ / r2_ names)
    ver = {}
    for tag in ("R", "R2"):
        us = list(cf.iter_units(str(out / tag), ["train"], set(), None))
        ver[tag] = {"units": len(us), "nc_by_trainer_rule": sum(bool(cf.is_noncommercial(u)) for u in us),
                    "k_gt_52": sum(len(u["option_keys"]) > K_MAX for u in us),
                    "stackpool_or_real_states": sum(ov.norm(u["state"]) in pool_states or ov.norm(u["state"]) in real_states for u in us)}
        if ver[tag] != {"units": units, "nc_by_trainer_rule": 0, "k_gt_52": 0, "stackpool_or_real_states": 0}:
            raise SystemExit(f"replay {tag} failed verification: {ver[tag]}")
    ids = [set(json.loads(l)["id"] for p in (out / tag / "train").glob("*.jsonl") for l in open(p, encoding="utf-8")) for tag in ("R", "R2")]
    if ids[0] & ids[1]:
        raise SystemExit(f"R and R2 share {len(ids[0] & ids[1])} record ids")
    vu = list(cf.iter_units(str(out), ["val"], set(), None))
    man = {"version": VERSION, "namespace": NS_PC, "units_each": units, "inputs": inputs or {}, "jev_real": {"units": n_real,
           "sha256": sha256_file(real)}, "stackpool_states": len(pool_states), "pool_records": len(recs), "dropped": dict(drop),
           "x99_train_files": len(train_files), "val_files": len(val_files), "val_units": len(vu),
           "val_units_after_nc": sum(not cf.is_noncommercial(u) for u in vu), "composition_units": comp, "verify": ver,
           "records": {"R": len(picks[0]), "R2": len(picks[1])}, "files": {}, "build_secs": round(time.time() - t0, 1)}
    for f in sorted(out.glob("**/*.jsonl")):
        man["files"][str(f.relative_to(out))] = sha256_file(f)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"replay: R {got[0]} / R2 {got[1]} units ({len(picks[0])} / {len(picks[1])} records) from {len(recs)} eligible records; dropped "
        f"{dict(drop)}; composition {comp}; val {len(vu)} units; {time.time() - t0:.0f} s")
    return man


# ============================================================================ self-test (no data needed)
def self_test(bnp=None):
    bn = load_bn(Path(bnp) if bnp else HERE / "build_numeric_jev.py")
    # 1. SIR marginal: swap priors drawn given on-prior thetas have the generator's prior marginal (coin, rate, normal at fixed S)
    for fam in FAMS:
        rng, py = np.random.default_rng(7), random.Random(7)
        S = 100.0
        base = candidates(bn, fam, rng, py, 3000, S)
        ref = candidates(bn, fam, rng, py, 3000, S)
        sw, lsel, lrand = [], [], []
        for j, c in enumerate(base):
            th = (float(rng.beta(c["a"], c["b"])) if fam == "coin" else
                  float(rng.gamma(c["shape"], 1 / c["rate"])) if fam == "rate" else float(rng.normal(c["mean"], c["sd"])))
            cs = candidates(bn, fam, rng, py, 400, S)
            lw = np.nan_to_num(np.asarray(log_weights(fam, cs, th, "prior"), float), nan=-np.inf)
            w = np.exp(lw - lw.max()); w /= w.sum()
            k = int(rng.choice(len(cs), p=w))
            sw.append(cs[k]); lsel.append(lw[k]); lrand.append(lw[int(rng.integers(len(cs)))])
        f = {"coin": lambda x: math.log(x["a"] + x["b"]), "rate": lambda x: math.log(x["shape"] / x["rate"]),
             "normal_mean": lambda x: math.log(x["sd"])}[fam]
        r = stats.ks_2samp([f(x) for x in sw], [f(x) for x in ref])
        assert r.pvalue > 1e-3, (fam, r)          # the swap priors' marginal = the generator's
        # the coupling: a selected prior makes theta far more likely than a random generator prior (a SIR that ignored theta,
        # i.e. uniform weights, has the same marginal and would pass the KS check alone)
        lr = np.asarray(lrand); lr = lr[np.isfinite(lr)]
        assert float(np.mean(lsel)) > float(np.mean(lr)) + 0.3, (fam, np.mean(lsel), np.mean(lr))
    # 2. parse_prior_line covers every template the generator prints
    for fam in FAMS:
        for i in range(60):
            g = bn.draw(fam, "train", i, bn.NS, 0, axes="v2")
            p = parse_prior_line(fam, g["state"].split("\n")[1])
            keys = {"coin": ("a", "b"), "rate": ("shape", "rate"), "normal_mean": ("mean", "sd")}[fam]
            assert all(float(p[k]) == float(g["prior_params"][k]) for k in keys), (fam, i, p, g["prior_params"], g["state"])
    # 3. pairs from the generator are valid; labels recompute; a corrupted pair is caught
    n = 0
    for fam in FAMS:
        for i in range(25):
            g = bn.draw(fam, "train", i, bn.NS, 0, axes="v2")
            base = bn.make_record(g, "train", i, bn.NS, False)
            ov, info = swap_prior(bn, fam, i, 0, g, L=256)
            s = bn.make_record(bn.draw(fam, "train", i, bn.NS, 0, axes="v2", prior_override=ov), "train", i, bn.NS, False)
            assert not pair_problems(base, s), (fam, i, pair_problems(base, s))
            lab = exact_labels(fam, s["meta"]["prior"]["stated"], s["meta"]["data"], s["meta"]["bin_edges"], s["meta"]["axis"]["scale"])
            assert float(np.abs(lab - label_vec(s)).max()) <= LABEL_TOL
            bad = json.loads(json.dumps(s)); bad["state"] = bad["state"].replace("\n", "\nExtra.\n", 1)
            assert pair_problems(base, bad)
            bad = json.loads(json.dumps(s)); ls = bad["state"].split("\n"); ls[2] = ls[2] + " (edited)"; bad["state"] = "\n".join(ls)
            assert any("state lines differing" in e for e in pair_problems(base, bad))     # same line count, a non-prior line edited
            bad = json.loads(json.dumps(s)); bad["meta"]["data"] = {**bad["meta"]["data"], "n": -1}
            assert pair_problems(base, bad)
            same = bn.make_record(bn.draw(fam, "train", i, bn.NS, 0, axes="v2", prior_override=dict(g["prior_params"]) if fam != "normal_mean"
                                          else {"mean": g["prior_params"]["mean"], "sd": g["prior_params"]["sd"]}), "train", i, bn.NS, False)
            assert "swap prior equals the base prior" in pair_problems(base, same)
            n += 1
    # 4. the selection is fixed, disjoint and data-independent
    for fam in FAMS:
        a1, a2 = select_indices(fam)
        assert select_indices(fam) == (a1, a2) and not set(a1) & set(a2) and len(a1) == len(a2) == N_PER_FAM
    print(f"self-test passed ({n} generator pairs; SIR marginals; prior-line parser; selection)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--numeric", required=True); b.add_argument("--out", required=True)
    b.add_argument("--suite", default=""); b.add_argument("--bn", default=""); b.add_argument("--n-per-family", type=int, default=N_PER_FAM)
    c = sub.add_parser("check"); c.add_argument("root"); c.add_argument("--numeric", required=True); c.add_argument("--suite", default="")
    c.add_argument("--bn", default=""); c.add_argument("--json", default="")
    s = sub.add_parser("self-test"); s.add_argument("--bn", default="")
    r = sub.add_parser("replay")
    for k in ("jev_data", "jev_ext_strict", "jev_ext2", "jev_rd3", "jev_rd3_stackpool", "jev_gdelt_v1", "bp_train_v1"):
        r.add_argument("--" + k.replace("_", "-"), required=True)
    r.add_argument("--jev-real", required=True); r.add_argument("--out", required=True); r.add_argument("--units", type=int, default=REPLAY_UNITS)
    r.add_argument("--cf", required=True, help="the pinned trainers/cache_features.py"); r.add_argument("--overlap-py", required=True,
                   help="analysis/claude_overlap.py (norm)"); r.add_argument("--inputs-json", default="", help="input tarball md5s, recorded")
    a = ap.parse_args(argv)
    if a.cmd == "self-test":
        return self_test(a.bn or None)
    if a.cmd == "replay":
        src = {k: getattr(a, k) for k in ("jev_data", "jev_ext_strict", "jev_ext2", "jev_rd3", "jev_rd3_stackpool", "jev_gdelt_v1",
                                          "bp_train_v1")}
        replay(src, a.out, a.jev_real, a.cf, a.overlap_py, a.units, json.loads(a.inputs_json) if a.inputs_json else None)
        return 0
    if a.cmd == "build":
        man, rep = build(a.numeric, a.out, a.bn or None, a.suite or None, a.n_per_family)
        return 1 if rep["n_errors"] else 0
    rep = check(a.root, a.numeric, a.bn or None, a.suite or None)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    for e in rep["errors"][:20]:
        print("ERROR", e)
    return 1 if rep["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
