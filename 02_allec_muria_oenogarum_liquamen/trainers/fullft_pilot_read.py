#!/usr/bin/env python3
"""(PILOT) Registered reader of the full-FT pilot (tmp/fullft/pilot_prereg_draft.md sections 2.1, 4-8; draft, not registered yet).

Every number comes from devP (DEV only): pilot-val (one panel over its pools), caldev, egval, didev (blocks; hover / esci / anli =
TARGET), bfdev. Models: each recipe X in {A, B, C, D} at seeds 0, 1 (2 of 3 seeds under the 6.4 extension), the S-a run (recipe B,
seed 0, LR_hi), and two references scored on devP's exact prompts: Garas / (99) NATIVE (the gate reference) and stock Gemma NATIVE.

READOUTS per run, from its step-1,875 DEV evaluation: OUTPUT = the posterior predictive of the HEAD_RULE head (kl > 0, the lowest
panel-macro NLL over the five panels; ties within 1e-4 -> smaller kl, then diag); MEAN = that head's mean head; DET = the kl-0 full head
read as its mean head; NATIVE = the native letter readout.

UNIT METRICS (SCREEN4's definitions, so its bars carry over): accuracy = argmax p is an argmax of the target; soft NLL; Brier; expected
0-1 regret = max t - t[argmax p]; ECE = 15 equal-width bins of max p against that accuracy; exact-posterior KL(t || p) on bfdev.
STATISTICS: per-unit paired deltas at matched prompts; the 2-seed mean of seed-matched deltas; cluster bootstrap over source groups
(base id), 2,000 draws, seed 0, row 0 = the point.

  CAP(X)        didev HELD-OUT target-block accuracy, block mean, X OUTPUT - Garas NATIVE: point >= 0.020, CI lower > 0, per-seed > 0
  GATES (X OUTPUT - Garas NATIVE; HOLDS / HARM / FAILS; adoption-grade = HOLDS and, where the recipe's seed noise floor (max pairwise
        |s_i - s_j| of the point) exceeds the bar, the per-seed rule):
        nontarget_acc >= -0.010 (didev non-target block mean) | dev_nll <= 0.010 | dev_brier <= 0.005 | dev_ece <= 0.010 |
        dev_regret <= 0.005 (didev + egval pooled) | cal_nll <= 0.005 | cal_acc >= -0.005 | cal_ece <= 0.005 (caldev)
  HEAD          OUTPUT - the run's own NATIVE, NLL over every devP unit, 2-seed point <= +0.005
  PER-SEED RULE every seed (2 seeds) or at least 2 of 3 (the 6.4 extension)
  B - A         PRIMARY CONTRAST on source-held-out evidence / relevance (held-out target blocks + egval as one block, block macro):
                GAIN_SPEC = NLL point <= -0.02 with CI upper < 0 AND accuracy point >= +0.010, per-seed NLL < 0 and accuracy > 0
  CONTRASTS     GAIN / COST = the CI excludes 0 AND the per-seed rule agrees in sign; NOT_RESOLVED otherwise (never "no effect")
  FITTED        (A1: canonical source ancestry) a didev block is FITTED when its name (without "didev_") or any '/ : |' segment (>= 4
                characters) of any of its units' sources, lowercase alphanumerics, equals a training root of ANY recipe (same
                normalisation; plus registered aliases); an EG-val UNIT is source-fitted by the same rule on its own source. Fitted
                TARGET blocks and source-fitted EG-val units leave the primary for every recipe (reported separately, as is ESCI)
  DECISION      INVALID | ADVANCE_X | HEAD_COSTS | GUARD_BLOCKED | NO_CAPABILITY_GAIN (first match); eligible = CAP and every gate
                adoption-grade and HEAD and not UNSTABLE; the highest CAP point, else (lead not a resolved GAIN) the order B, D, A, C
  S-a           d = pilot-val NATIVE NLL (LR_hi run - pilot_B_s0) at 1,875: LR_OK (CI lower > -0.010) | LR_LIMITED (CI upper < -0.010)
                | LR_UNRESOLVED; labels every row, decides none
  S-b           STILL_LEARNING: pilot-val NATIVE NLL improves by > 0.02 from step 1,250 to 1,875 in every recipe at every seed
  G3'           post-warmup steps >= 50; m / l = the medians of pre-clip grad norm / train loss there; UNSTABLE = 3 consecutive steps with
                grad norm > 20 m, or a step with grad norm > 20 m AND loss > 2 l, or any non-finite loss / grad norm. Attempt 1
                UNSTABLE -> INVALID (rerun once); attempt 2 UNSTABLE -> the recipe is UNSTABLE (cannot advance)

  python3 trainers/fullft_pilot_read.py --self-test
  python3 trainers/fullft_pilot_read.py --inputs IN/inputs.json --out V.json --text V.txt
  python3 trainers/fullft_pilot_read.py --list-keys --pins PINS.json --dev devP          (the results keys the read needs)
  python3 trainers/fullft_pilot_read.py --mirror DIR --pins PINS.json --dev devP --out V.json --text V.txt
            (DIR = a local mirror of results_fullft/: inputs are built from it, with the overlap / roots / sha256 checks)

PINS.json (written at registration): {"registered": {lr, lr_hi, alpha {A..D}, args {...}, trainer_version, dev_rows_sha256,
manifest_rows_sha256 {A..D}, refs {garas {...}, stock {...}}, fitted_aliases {}}, "tags": {A: render / manifest tag, ...},
"runs": {key: {run, recipe, seed, role main | lrhi, attempt}}}
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

VERSION = "fullft_pilot_read v2 (2026-10-13; amendment A1: disclosed corrections before any read)"
# A1: (1) the run log is assembled from immutable numbered attempt pieces when present (completeness check unchanged); (2) the joined
# training BODY is fetched and hashed against the registered digest, roots / overlap come from that body, and each run's recorded data
# digest must equal its recipe's (pre-A1 runs listed in registered.legacy_unbound_runs are bound through the body digest only);
# (3) contrast wording: equivalence / non-inferiority claims only within the declared margins, else "not resolved"; (4) canonical source
# ancestry across EG-val as well as didev: the primary is source-held-out; source-fitted EG-val units and ESCI are reported separately;
# the head reading "adds nothing measurable" also needs every panel's OUTPUT - DET NLL CI inside +-NOTHING_EPS (else "not resolved");
# (5) the S-a LR_hi run UNSTABLE on its second attempt is reported, never marks recipe B UNSTABLE (S-a decides no row, prereg 2.1).
EQUIV = {"cal_nll": 0.005, "dev_nll": 0.010, "target_acc": 0.010}     # C - B equivalence margins (the gate bars' sizes)
NONINF_BF_KL = 0.005                                                   # D - B bfdev KL non-inferiority margin (nats)
MIN_SEG = 4                                                            # source segments shorter than this never establish ancestry
B = 2000
RECIPES = ("A", "B", "C", "D")
PANELS = ("pilotval", "panel:caldev", "panel:egval", "panel:didev", "panel:bfdev")
TARGET = ("hover", "esci", "anli")
CAP_MIN = 0.020
GATES = {  # name: (statistic, bar, direction)
    "nontarget_acc": ("nontarget_acc_block", -0.010, "ge"), "dev_nll": ("dev_nll", 0.010, "le"), "dev_brier": ("dev_brier", 0.005, "le"),
    "dev_ece": ("dev_ece", 0.010, "le"), "dev_regret": ("dev_regret", 0.005, "le"), "cal_nll": ("cal_nll", 0.005, "le"),
    "cal_acc": ("cal_acc", -0.005, "ge"), "cal_ece": ("cal_ece", 0.005, "le"),
}
LOWER_BETTER = ("nll", "brier", "ece", "regret", "kl")
HEAD_BAR = 0.005
PREF = ("B", "D", "A", "C")
SPEC_NLL, SPEC_ACC = -0.02, 0.010
LR_TOL = 0.010
STILL = 0.02
G3_K, G3_LOSS, G3_RUN, WARMUP = 20.0, 2.0, 3, 50
STEPS = 1875
EVAL_STEPS = (625, 1250, 1875)
NOTHING_EPS = 0.002
AUROC_MIN = 0.55
GRID = [{"prior": 1.0, "kl": kl, "cov": cov} for kl in (0.0, 1.0, 10000.0, 1000000.0) for cov in ("diag", "full")]


def open_any(p):
    return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, encoding="utf-8")


def jsonl(p):
    with open_any(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def ci_of(v):
    v = np.asarray(v, float)
    return [float(v[0]), float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))]


def panel_name(v):
    parts = str(v).split(":")
    return parts[0] if parts[0] == "pilotval" else ":".join(parts[:2])


def block_of(v):
    parts = str(v).split(":")
    if len(parts) >= 3 and parts[1] == "didev":
        b = parts[2]
        return b[len("didev_"):] if b.startswith("didev_") else b
    return None


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ============================================================================ unit metrics, bootstrap, statuses
def unit_metrics(P, T, K):
    idx = np.arange(P.shape[1])[None, :] < K[:, None]
    P = np.where(idx, P, 0.0)
    T = np.where(idx, T, 0.0)
    T = T / np.clip(T.sum(1, keepdims=True), 1e-12, None)
    lp = np.log(np.clip(P, 1e-12, 1))
    am = np.where(idx, P, -1.0).argmax(1)
    tmax = T.max(1)
    acc = (T[np.arange(len(T)), am] >= tmax - 1e-12).astype(float)
    ent = -(T * np.log(np.clip(T, 1e-300, 1))).sum(1)
    nll = -(T * lp).sum(1)
    return {"acc": acc, "nll": nll, "brier": ((P - T) ** 2).sum(1), "regret": tmax - T[np.arange(len(T)), am], "mx": P.max(1),
            "kl": nll - ent, "accm": T[np.arange(len(T)), am]}


def ece_w(correct, mx, w):
    edges = np.linspace(0, 1, 16)
    tot, e = w.sum(), 0.0
    for i in range(15):
        s = (mx > edges[i]) & (mx <= edges[i + 1])
        ws = w[s].sum()
        if ws > 0:
            e += abs((w[s] * correct[s]).sum() / ws - (w[s] * mx[s]).sum() / ws) * ws / tot
    return e


class Draws:
    """one cluster bootstrap over a fixed unit set; row 0 = the point (weights 1)"""

    def __init__(self, clusters, nboot=B, seed=0):
        uc, inv = np.unique(np.asarray(clusters), return_inverse=True)
        rng = np.random.default_rng(seed)
        cw = np.vstack([np.ones(len(uc))] + [np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc)) for _ in range(nboot)])
        self.W = cw[:, inv]

    def mean(self, d):
        return (self.W @ d) / np.clip(self.W.sum(1), 1e-12, None)

    def block_mean(self, d, blocks):
        blocks = np.asarray(blocks)
        out = []
        for b in sorted(set(blocks.tolist())):
            m = blocks == b
            out.append((self.W[:, m] @ d[m]) / np.clip(self.W[:, m].sum(1), 1e-12, None))
        return np.mean(out, axis=0)

    def ece(self, correct, mx):
        return np.array([ece_w(correct, mx, w) for w in self.W])


def status(est, bar, direction):
    v, lo, hi = est
    if direction == "ge":
        return "HOLDS" if v >= bar else ("HARM" if hi < bar else "FAILS")
    return "HOLDS" if v <= bar else ("HARM" if lo > bar else "FAILS")


def per_seed_ok(flags):
    """every seed (2 seeds) or at least 2 of 3 (the 6.4 extension)"""
    flags = list(flags)
    return all(flags) if len(flags) <= 2 else sum(bool(x) for x in flags) >= 2


def cap_rule(c, per):
    """CAP: the 2-seed point >= CAP_MIN, its CI lower bound > 0, and the per-seed rule on positive points"""
    return bool(c[0] >= CAP_MIN and c[1] > 0 and per_seed_ok(p > 0 for p in per))


def adoption_grade(st, per_seed, noisy):
    return st == "HOLDS" and (not noisy or per_seed_ok(x == "HOLDS" for x in per_seed))


def sign_label(c, per, lower_better=False):
    """GAIN / COST with the per-seed rule; c = [point, lo, hi] of X - Y; per = per-seed points"""
    pos = c[1] > 0 and per_seed_ok(p > 0 for p in per)
    neg = c[2] < 0 and per_seed_ok(p < 0 for p in per)
    if lower_better:
        return "GAIN" if neg else ("COST" if pos else "NOT_RESOLVED")
    return "GAIN" if pos else ("COST" if neg else "NOT_RESOLVED")


def auroc(score, err, w=None):
    """weighted AUROC of `score` for err == 1 (ties count 1/2)"""
    w = np.ones(len(score)) if w is None else np.asarray(w, float)
    pos, neg = err == 1, err == 0
    if w[pos].sum() == 0 or w[neg].sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s, e, ww = score[order], err[order], w[order]
    tot_pos, tot_neg = ww[e == 1].sum(), ww[e == 0].sum()
    auc, cum_neg, i = 0.0, 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        wp, wn = ww[i:j][e[i:j] == 1].sum(), ww[i:j][e[i:j] == 0].sum()
        auc += wp * (cum_neg + 0.5 * wn)
        cum_neg += wn
        i = j
    return float(auc / (tot_pos * tot_neg))


def auroc_draws(score, err, W):
    """auroc() for every bootstrap weight row of W at once (vectorised; ties count 1/2) -> [n_draws]"""
    order = np.argsort(score, kind="mergesort")
    s, e = score[order], err[order].astype(bool)
    Wd = W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    wp = np.add.reduceat(np.where(e[None, :], Wd, 0.0), starts, axis=1)
    wn = np.add.reduceat(np.where(~e[None, :], Wd, 0.0), starts, axis=1)
    cum_neg = np.cumsum(wn, axis=1) - wn
    tp, tn = wp.sum(1), wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wp * (cum_neg + 0.5 * wn)).sum(1) / (tp * tn)


# ============================================================================ G3' (stated in advance)
def g3_unstable(steps):
    """steps: {step: (loss, gnorm)} (the LAST logged occurrence of each step). -> (unstable, reasons)"""
    why = []
    for s, (l, g) in steps.items():
        if not (math.isfinite(l) and math.isfinite(g)):
            why.append(f"(c) non-finite at step {s}")
            break
    post = sorted(s for s in steps if s >= WARMUP)
    if not post:
        return bool(why), why
    gs = np.array([steps[s][1] for s in post])
    ls = np.array([steps[s][0] for s in post])
    fin = np.isfinite(gs) & np.isfinite(ls)
    m, l = float(np.median(gs[fin])), float(np.median(ls[fin]))
    hot = gs > G3_K * m
    run = 0
    for i, h in enumerate(hot):
        run = run + 1 if h and (i == 0 or post[i] == post[i - 1] + 1) else (1 if h else 0)
        if run >= G3_RUN:
            why.append(f"(a) {G3_RUN} consecutive steps over {G3_K:g} x median {m:.3g} ending at step {post[i]}")
            break
    lossy = [post[i] for i in range(len(post)) if hot[i] and ls[i] > G3_LOSS * l]
    if lossy:
        why.append(f"(b) spike with loss > {G3_LOSS:g} x median at steps {lossy[:5]}")
    return bool(why), why


def parse_log(path):
    """STEP rows of a train.log (resumed runs repeat steps: the last occurrence wins) -> ({step: (loss, gnorm)}, mem summary, resumed)"""
    steps, mem_warn, retries, resumed = {}, 0, 0, []
    with open_any(path) as f:
        for line in f:
            if line.startswith("STEP "):
                try:
                    r = json.loads(line[5:])
                except ValueError:
                    continue                          # the truncated last line of an interrupted attempt
                steps[int(r["step"])] = (float(r["loss"]), float(r["gnorm"]))
                retries += int((r.get("mem") or {}).get("retries", 0))
            elif line.startswith("MEM_WARN"):
                mem_warn += 1
            elif line.startswith("RESUMED at step"):
                resumed.append(int(line.split()[3]))
    return steps, {"mem_warn_steps": mem_warn, "malloc_retries": retries}, resumed


# ============================================================================ inputs
class Dev:
    def __init__(self, rows_path, panels_path):
        rows = [r for r in jsonl(rows_path) if not r.get("manifest_header")]
        pm = json.load(open(panels_path))["panel_of"]
        self.rows = rows
        self.uids = [r["decision_uid"] for r in rows]
        if len(set(self.uids)) != len(self.uids):
            raise ValueError("devP: duplicate decision_uid")
        self.sha = {r["decision_uid"]: r["prompt_sha256"] for r in rows}
        self.pos = {u: i for i, u in enumerate(self.uids)}
        n = len(rows)
        self.T = np.zeros((n, 52))
        self.K = np.zeros(n, dtype=int)
        self.order = []
        for i, r in enumerate(rows):
            keys = list(r["option_order"])
            self.T[i, :len(keys)] = [float(r["gold"][k]) for k in keys]
            self.K[i] = len(keys)
            self.order.append(keys)
        missing = [u for u in self.uids if u not in pm]
        if missing:
            raise ValueError(f"devP: {len(missing)} units have no panel, e.g. {missing[:2]}")
        self.raw_panel = [pm[u] for u in self.uids]
        self.panel = np.array([panel_name(v) for v in self.raw_panel])
        bad = sorted(set(self.panel.tolist()) - set(PANELS))
        if bad:
            raise ValueError(f"devP: unknown panels {bad}")
        self.block = np.array([block_of(v) or "" for v in self.raw_panel])
        self.cluster = np.array([str(r.get("base_id") or r.get("record_id") or r["decision_uid"]) for r in rows])
        self.record = [str(r.get("record_id") or "") for r in rows]
        self.source = [str(r.get("source") or "") for r in rows]


def probs_from_eval(dev, rows, kind, g=None):
    """eval rows (fullft_train do_eval) -> P [n, 52] in devP order; kind native | mean | predictive | epi"""
    by = {}
    for r in rows:
        by[r["decision_uid"]] = r
    n = len(dev.uids)
    if kind == "epi":
        out = np.zeros(n)
        for i, u in enumerate(dev.uids):
            out[i] = float(by[u]["epi"][g])
        return out
    P = np.zeros((n, 52))
    for i, u in enumerate(dev.uids):
        r = by[u]
        v = r["native"] if kind == "native" else r[kind][g]
        P[i, :len(v)] = v
    return P


def probs_from_ref(dev, rows):
    by = {(r["decision_uid"], r["prompt_sha256"]): r for r in rows}
    P = np.zeros((len(dev.uids), 52))
    for i, u in enumerate(dev.uids):
        r = by[(u, dev.sha[u])]
        P[i, :dev.K[i]] = [r["probabilities"][k] for k in dev.order[i]]
    return P


def check_eval_rows(dev, rows, where):
    probs = []
    seen = {}
    for r in rows:
        u = r.get("decision_uid")
        if u in seen:
            probs.append(f"{where}: {u} twice")
        seen[u] = r
    if set(seen) != set(dev.uids):
        probs.append(f"{where}: units {len(seen)} != devP {len(dev.uids)} (or other units)")
        return probs
    for u, r in seen.items():
        if r.get("prompt_sha256") != dev.sha[u]:
            probs.append(f"{where}: {u} prompt sha256 differs from devP")
            break
        i = dev.pos[u]
        if int(r.get("k", len(r["native"]))) != dev.K[i]:
            probs.append(f"{where}: {u} K differs")
            break
        if "t" in r and np.abs(np.asarray(r["t"], float) - dev.T[i, :dev.K[i]]).max() > 1e-9:
            probs.append(f"{where}: {u} gold differs from devP")
            break
    return probs


def head_rule(dev, rows, heads):
    """the panel-macro HEAD_RULE -> (selected index, per-head panel-macro NLL)"""
    nll = []
    for g in range(len(heads)):
        um = unit_metrics(probs_from_eval(dev, rows, "predictive", g), dev.T, dev.K)["nll"]
        nll.append(float(np.mean([um[dev.panel == p].mean() for p in PANELS if (dev.panel == p).any()])))
    cand = [g for g, h in enumerate(heads) if float(h["kl"]) > 0]
    best = min(nll[g] for g in cand)
    tied = [g for g in cand if nll[g] - best < 1e-4]
    return min(tied, key=lambda g: (float(heads[g]["kl"]), heads[g]["cov"] != "diag")), nll


def det_index(heads):
    for g, h in enumerate(heads):
        if float(h["kl"]) == 0 and h["cov"] == "full":
            return g
    raise ValueError("no kl-0 full head in the grid")


# ============================================================================ inputs from a results mirror
def needed_keys(pins, dev):
    keys = [f"render/{dev}/rows.jsonl.gz", f"render/{dev}/rows.jsonl.header.json", f"render/{dev}/panels.json"]
    for r in ("garas", "stock"):
        keys += [f"ref/{dev}/{r}/ref.jsonl.gz", f"ref/{dev}/{r}/summary_0.json"]
    for X, tag in sorted(pins["tags"].items()):
        keys += [f"manifest/{tag}/train.jsonl.gz", f"manifest/{tag}/train.jsonl.header.json"]
    for k, v in sorted(pins["runs"].items()):
        keys += [f"runs/{v['run']}/eval_step{s}.jsonl" for s in EVAL_STEPS]
        keys += [f"runs/{v['run']}/train.log", f"runs/{v['run']}/done.json", f"runs/{v['run']}/export/vbll_head.json"]
    return keys


def sha256_stream(path):
    h = hashlib.sha256()
    with (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")) as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_log(d):
    """A1: a run's log = its immutable numbered attempt logs train.log.attempt.N in attempt order (parse_log: the last STEP record per
    step wins) when any exist, else the (legacy, task_fullft <= 10) train.log -> (path, number of pieces)"""
    pieces = sorted((int(f.rsplit(".", 1)[1]), f) for f in (os.listdir(d) if os.path.isdir(d) else [])
                    if re.match(r"^train\.log\.attempt\.\d+$", f))
    if not pieces:
        return f"{d}/train.log", 0
    log = f"{d}/train.log.assembled_by_reader"
    with open(log, "w", encoding="utf-8") as g:
        for _, f in pieces:
            with open(f"{d}/{f}", encoding="utf-8", errors="replace") as h_:
                for line in h_:
                    g.write(line if line.endswith("\n") else line + "\n")
    return log, len(pieces)


def build_inputs(mir, pins, dev):
    """a results_fullft/ mirror + the registered pins -> the read's inputs (missing files are left for the read to refuse)"""
    p = lambda k: os.path.join(mir, k)  # noqa: E731
    ex = os.path.exists
    hdr = json.load(open(p(f"render/{dev}/rows.jsonl.header.json"))) if ex(p(f"render/{dev}/rows.jsonl.header.json")) else {}
    rows_sha = sha256_stream(p(f"render/{dev}/rows.jsonl.gz")) if ex(p(f"render/{dev}/rows.jsonl.gz")) else None
    inp = {"dev": {"rows": p(f"render/{dev}/rows.jsonl.gz"), "panels": p(f"render/{dev}/panels.json"), "rows_sha256": rows_sha,
                   "header_rows_sha256": hdr.get("rows_sha256")},
           "registered": pins.get("registered") or {}, "refs": {}, "runs": {}, "manifests": {}, "training_roots": {}, "overlap": {}}
    for r in ("garas", "stock"):
        sp = p(f"ref/{dev}/{r}/summary_0.json")
        inp["refs"][r] = {"rows": p(f"ref/{dev}/{r}/ref.jsonl.gz"), "scored_by": json.load(open(sp)).get("scored_by", {}) if ex(sp) else {}}
    dev_rows = [x for x in jsonl(p(f"render/{dev}/rows.jsonl.gz")) if not x.get("manifest_header")] if rows_sha else []
    dsha = {x["prompt_sha256"] for x in dev_rows}
    drec = {str(x.get("record_id")) for x in dev_rows if x.get("record_id")}
    for X, tag in pins["tags"].items():
        mh = p(f"manifest/{tag}/train.jsonl.header.json")
        h = json.load(open(mh)) if ex(mh) else {}
        bp = p(f"manifest/{tag}/train.jsonl.gz")
        inp["manifests"][X] = {"rows_sha256": h.get("rows_sha256"), "total_decision_weight": h.get("total_decision_weight"),
                               "body_rows_sha256": sha256_stream(bp) if ex(bp) else None}
        rp = bp
        if ex(rp):
            roots, sh, rc = set(), set(), set()
            for x in jsonl(rp):
                if x.get("manifest_header"):
                    continue
                if x.get("root"):
                    roots.add(x["root"])
                sh.add(x["prompt_sha256"])
                rc.add(str(x.get("record_id")))
            inp["training_roots"][X] = sorted(roots)
            inp["overlap"][X] = {"prompt_sha256": len(dsha & sh), "record_id": len(drec & rc)}
    for k, v in pins["runs"].items():
        d = p(f"runs/{v['run']}")
        log, pieces = run_log(d)
        done = json.load(open(f"{d}/done.json")) if ex(f"{d}/done.json") else None
        vh = json.load(open(f"{d}/export/vbll_head.json")) if ex(f"{d}/export/vbll_head.json") else {}
        inp["runs"][k] = {"recipe": v["recipe"], "seed": v["seed"], "role": v.get("role", "main"), "attempt": v.get("attempt", 1),
                          "evals": {str(s): f"{d}/eval_step{s}.jsonl" for s in EVAL_STEPS}, "log": log, "attempt_pieces": pieces,
                          "done": done is not None, "record": (done or {}).get("record") or {},
                          "exported_selected": vh.get("selected_from_grid")}
    return inp


# ============================================================================ the read
def read(inp, nboot=B):
    if not (os.path.exists(inp["dev"]["rows"]) and os.path.exists(inp["dev"]["panels"])):
        return {"version": VERSION, "row": "INVALID", "problems": ["devP rows / panels missing"], "text": "INVALID: devP rows / panels missing"}
    dev = Dev(inp["dev"]["rows"], inp["dev"]["panels"])
    reg = inp.get("registered") or {}
    problems = []
    if reg.get("dev_rows_sha256") and inp["dev"].get("rows_sha256") != reg["dev_rows_sha256"]:
        problems.append(f"devP rows_sha256 {inp['dev'].get('rows_sha256')} != registered {reg['dev_rows_sha256']}")
    if inp["dev"].get("header_rows_sha256") and inp["dev"].get("rows_sha256") != inp["dev"]["header_rows_sha256"]:
        problems.append("devP rows differ from their own header's rows_sha256")
    if reg.get("renderer_sha256"):
        bad_r = sum(1 for r in dev.rows if r.get("renderer_sha256") != reg["renderer_sha256"])
        if bad_r:
            problems.append(f"devP: {bad_r} rows not rendered by the registered renderer {reg['renderer_sha256'][:12]}")
    # ---- references
    refs, ref_meta = {}, {}
    for name in ("garas", "stock"):
        if not os.path.exists((inp["refs"].get(name) or {}).get("rows", "")):
            problems.append(f"reference {name}: missing")
            continue
        rr = jsonl(inp["refs"][name]["rows"])
        keys = {(r["decision_uid"], r["prompt_sha256"]) for r in rr}
        if keys != {(u, dev.sha[u]) for u in dev.uids} or len(rr) != len(dev.uids):
            problems.append(f"reference {name}: rows do not match devP's prompts 1:1")
            continue
        refs[name] = probs_from_ref(dev, rr)
        sb = inp["refs"][name].get("scored_by") or {}
        ref_meta[name] = sb
        want = (reg.get("refs") or {}).get(name) or {}
        for k, v in want.items():
            if sb.get(k) != v:
                problems.append(f"reference {name}: {k} = {sb.get(k)} != registered {v}")
    # ---- A1: the joined training bodies are the registered bytes
    for X in RECIPES:
        mv = (inp.get("manifests") or {}).get(X) or {}
        want = (reg.get("manifest_rows_sha256") or {}).get(X)
        if not mv.get("body_rows_sha256"):
            problems.append(f"manifest {X}: the joined training body was not fetched / hashed")
        elif want and mv["body_rows_sha256"] != want:
            problems.append(f"manifest {X}: training body sha256 {mv['body_rows_sha256']} != registered {want}")
    # ---- fitted blocks and units: canonical source ancestry (block names AND every unit's source segments) across didev and EG-val
    roots = {X: set(norm_name(r) for r in v) for X, v in (inp.get("training_roots") or {}).items()}
    aliases = {norm_name(k): [norm_name(x) for x in v] for k, v in (reg.get("fitted_aliases") or {}).items()}
    all_roots = set().union(*roots.values()) if roots else set()

    def anc(segs, exact=()):                         # source segments need >= MIN_SEG characters; a block's own name always counts
        names = {x for x in segs if len(x) >= MIN_SEG} | set(exact)
        names |= {y for x in list(names) for y in aliases.get(x, [])}
        return sorted(X for X, rs in roots.items() if names & rs)
    seg = [{norm_name(x) for x in re.split(r"[/:|]", src) if x} for src in dev.source]
    blocks = sorted(set(b for b in dev.block.tolist() if b))
    fitted = {}
    for b in blocks:
        fitted[b] = anc(set().union(*[seg[i] for i in range(len(seg)) if dev.block[i] == b]), exact={norm_name(b)})
    eg_fitted = np.array([dev.panel[i] == "panel:egval" and bool(anc(seg[i])) for i in range(len(seg))])
    eg_fitted_sources = sorted({dev.source[i] for i in np.where(eg_fitted)[0]})
    if set(roots) != set(RECIPES):
        problems.append(f"training roots given for {sorted(roots)}, need {list(RECIPES)}")
    held_targets = [b for b in TARGET if b in blocks and not fitted.get(b)]
    if not held_targets:
        problems.append("every target block is FITTED (or absent): the primary cannot be read")
    # ---- disjointness of devP from every training manifest (exact prompt and record identity)
    for X, ov in (inp.get("overlap") or {}).items():
        if ov.get("prompt_sha256") or ov.get("record_id"):
            problems.append(f"devP overlaps recipe {X}'s training manifest: {ov}")
    for X in RECIPES:
        if X not in (inp.get("overlap") or {}):
            problems.append(f"no devP / training-manifest overlap check for recipe {X}")
    # ---- runs
    runs = inp["runs"]
    by_recipe = {X: sorted([k for k, v in runs.items() if v.get("recipe") == X and v.get("role", "main") == "main"],
                           key=lambda k: runs[k]["seed"]) for X in RECIPES}
    seeds = {X: [runs[k]["seed"] for k in by_recipe[X]] for X in RECIPES}
    if len({tuple(v) for v in seeds.values()}) != 1 or len(seeds["A"]) not in (2, 3):
        problems.append(f"recipes must share 2 (or 3) seeds: {seeds}")
    lrhi = [k for k, v in runs.items() if v.get("role") == "lrhi"]
    if len(lrhi) != 1:
        problems.append(f"the S-a LR_hi run is required exactly once (found {lrhi})")
    model = {}
    unstable_recipe = set()
    lrhi_unstable = False
    run_info = {}
    for k, v in runs.items():
        info = {"recipe": v.get("recipe"), "seed": v.get("seed"), "role": v.get("role", "main")}
        evals = {}
        for s in EVAL_STEPS:
            p = (v.get("evals") or {}).get(str(s))
            if not p or not os.path.exists(p):
                problems.append(f"{k}: no eval at step {s}")
                continue
            evals[s] = jsonl(p)
            problems += check_eval_rows(dev, evals[s], f"{k}@{s}")
        rec = v.get("record") or {}
        heads = (rec.get("head") or {}).get("heads")
        if heads is None or len(heads) != 8 or [(float(h["prior"]), float(h["kl"]), h["cov"]) for h in heads] != \
                [(h["prior"], h["kl"], h["cov"]) for h in GRID]:
            problems.append(f"{k}: the head grid is not the registered 8-head grid")
            heads = GRID
        args = rec.get("args") or {}
        want_args = dict(reg.get("args") or {})
        if info["role"] == "lrhi":
            want_args["lr"] = reg.get("lr_hi")
        elif reg.get("lr") is not None:
            want_args["lr"] = reg.get("lr")
        if v.get("recipe") in (reg.get("alpha") or {}):
            want_args["alpha"] = reg["alpha"][v["recipe"]]
        want_args["seed"] = v.get("seed")
        for a_k, a_v in want_args.items():
            if a_v is None:
                continue
            got = args.get(a_k)
            if (isinstance(a_v, float) and (got is None or abs(float(got) - a_v) > 1e-12)) or (not isinstance(a_v, float) and got != a_v):
                problems.append(f"{k}: arg {a_k} = {got} != registered {a_v}")
        if reg.get("trainer_version") and rec.get("version") != reg["trainer_version"]:
            problems.append(f"{k}: trainer {rec.get('version')} != registered {reg['trainer_version']}")
        man = (inp.get("manifests") or {}).get(v.get("recipe")) or {}
        wt = (rec.get("head") or {}).get("W_total")
        if man.get("total_decision_weight") is not None and (wt is None or abs(float(wt) - float(man["total_decision_weight"])) > 1e-6):
            problems.append(f"{k}: head W_total {wt} != manifest total {man.get('total_decision_weight')}")
        steps, memsum, resumed = parse_log(v["log"]) if v.get("log") and os.path.exists(v["log"]) else ({}, {}, [])
        if set(steps) != set(range(STEPS)):
            problems.append(f"{k}: logged steps {len(steps)} != 0..{STEPS - 1} (incomplete or a gap)")
        if not v.get("done"):
            problems.append(f"{k}: not done (no done.json / export)")
        data = rec.get("data") or {}
        want_d = (reg.get("manifest_rows_sha256") or {}).get(v.get("recipe"))
        if data.get("rows_sha256"):
            if want_d and data["rows_sha256"] != want_d:
                problems.append(f"{k}: trained on data {data['rows_sha256']} != registered {want_d} for recipe {v.get('recipe')}")
        elif k not in (reg.get("legacy_unbound_runs") or []):
            problems.append(f"{k}: no recorded training-data digest (only registered pre-A1 runs may lack it)")
        info["data_rows_sha256"] = data.get("rows_sha256")
        info["code"] = rec.get("code")
        if rec.get("code") and reg.get("trainer_code_md5") and rec["code"].get("fullft_train.py") not in reg["trainer_code_md5"]:
            problems.append(f"{k}: trainer code {rec['code'].get('fullft_train.py')} not in the registered {reg['trainer_code_md5']}")
        uns, why = g3_unstable(steps) if steps else (False, [])
        info.update({"mem": memsum, "resumed_at": resumed, "g3_unstable": uns, "g3_why": why, "attempt": int(v.get("attempt", 1))})
        if uns:
            if info["attempt"] >= 2 and info["role"] == "lrhi":
                lrhi_unstable = True                 # A1 (5): S-a decides no row; a twice-UNSTABLE LR_hi run is reported, recipe B is not
            elif info["attempt"] >= 2:
                unstable_recipe.add(v.get("recipe"))
            else:
                problems.append(f"{k}: UNSTABLE under G3' ({why}); rerun once with the same seed")
        if 1875 in evals and len(heads) == 8:
            sel, hn = head_rule(dev, evals[1875], heads)
            info["head"] = {"selected": sel, "spec": heads[sel], "panel_macro_nll": hn}
            exp_sel = v.get("exported_selected")
            if exp_sel is None:
                problems.append(f"{k}: no exported head (task head) to check")
            elif int(exp_sel) != sel:
                problems.append(f"{k}: exported head {exp_sel} != the read's HEAD_RULE {sel}")
            d = det_index(heads)
            model[k] = {"out": probs_from_eval(dev, evals[1875], "predictive", sel), "mean": probs_from_eval(dev, evals[1875], "mean", sel),
                        "det": probs_from_eval(dev, evals[1875], "mean", d), "native": probs_from_eval(dev, evals[1875], "native"),
                        "epi": probs_from_eval(dev, evals[1875], "epi", sel),
                        "curve_native": {s: probs_from_eval(dev, evals[s], "native") for s in evals}}
        run_info[k] = info
    for X, ov in (inp.get("manifests") or {}).items():
        want = (reg.get("manifest_rows_sha256") or {}).get(X)
        if want and ov.get("rows_sha256") != want:
            problems.append(f"manifest {X}: rows_sha256 {ov.get('rows_sha256')} != registered {want}")
    res = {"version": VERSION, "runs": run_info, "fitted_blocks": fitted, "held_out_targets": held_targets, "problems": problems,
           "egval_fitted_units": int(eg_fitted.sum()), "egval_fitted_sources": eg_fitted_sources,
           "unstable_recipes": sorted(unstable_recipe), "reference_scored_by": ref_meta}
    if problems or len(refs) != 2 or any(k not in model for X in RECIPES for k in by_recipe[X]) or (lrhi and lrhi[0] not in model):
        res["row"] = "INVALID"
        res["text"] = "INVALID: " + "; ".join(problems[:12] or ["a model or reference could not be read"])
        return res
    # ---- metrics and bootstrap draws
    def mets(P):
        return unit_metrics(P, dev.T, dev.K)
    M = {k: {r: mets(m[r]) for r in ("out", "mean", "det", "native")} for k, m in model.items()}
    R = {name: mets(P) for name, P in refs.items()}
    isp = lambda p: dev.panel == p  # noqa: E731
    dd = isp("panel:didev")
    tmask = dd & np.isin(dev.block, held_targets)
    nmask = dd & ~np.isin(dev.block, list(TARGET))
    devmask = dd | isp("panel:egval")
    calmask, bfmask, pvmask = isp("panel:caldev"), isp("panel:bfdev"), isp("pilotval")
    egmask_held = isp("panel:egval") & ~eg_fitted
    egmask_fit = isp("panel:egval") & eg_fitted
    primask = tmask | egmask_held
    D = {"t": Draws(dev.cluster[tmask], nboot), "n": Draws(dev.cluster[nmask], nboot), "dev": Draws(dev.cluster[devmask], nboot),
         "cal": Draws(dev.cluster[calmask], nboot), "all": Draws(dev.cluster, nboot), "bf": Draws(dev.cluster[bfmask], nboot),
         "pv": Draws(dev.cluster[pvmask], nboot), "pri": Draws(dev.cluster[primask], nboot)}
    esci_mask = dd & (dev.block == "esci")
    pri_blocks = np.where(dev.block[primask] != "", dev.block[primask], "egval")

    def stat(mx, kind):
        if kind == "target_acc":
            return D["t"].block_mean(mx["acc"][tmask], dev.block[tmask])
        if kind == "nontarget_acc_block":
            return D["n"].block_mean(mx["acc"][nmask], dev.block[nmask])
        if kind == "primary_nll":
            return D["pri"].block_mean(mx["nll"][primask], pri_blocks)
        if kind == "primary_acc":
            return D["pri"].block_mean(mx["acc"][primask], pri_blocks)
        scope, k = kind.split("_", 1)
        key, mask = {"dev": ("dev", devmask), "cal": ("cal", calmask), "all": ("all", np.ones(len(dev.uids), bool)),
                     "bf": ("bf", bfmask), "pv": ("pv", pvmask)}[scope]
        if k == "ece":
            return D[key].ece(mx["acc"][mask], mx["mx"][mask])
        return D[key].mean(mx[k][mask])
    keys = {X: by_recipe[X] for X in RECIPES}
    ref_g = R["garas"]
    # ---- CAP, gates, HEAD per recipe
    rec_out = {}
    for X in RECIPES:
        ms = keys[X]
        prim = [stat(M[m]["out"], "target_acc") - stat(ref_g, "target_acc") for m in ms]
        c = ci_of(np.mean(prim, axis=0))
        per = [float(p[0]) for p in prim]
        cap = cap_rule(c, per)
        gates = {}
        for g, (k, bar, direction) in GATES.items():
            ds = [stat(M[m]["out"], k) - stat(ref_g, k) for m in ms]
            est, per_s = ci_of(np.mean(ds, axis=0)), [ci_of(d) for d in ds]
            pts = [float(d[0]) for d in ds]
            floor = max(abs(a - b) for i, a in enumerate(pts) for b in pts[i + 1:])
            st, pst = status(est, bar, direction), [status(p, bar, direction) for p in per_s]
            noisy = floor > abs(bar)
            gates[g] = {"estimate": est, "per_seed": per_s, "bar": bar, "direction": direction, "status": st, "per_seed_status": pst,
                        "noise_floor": floor, "noise_exceeds_bar": noisy, "adoption_grade": adoption_grade(st, pst, noisy)}
        hd = ci_of(np.mean([stat(M[m]["out"], "all_nll") - stat(M[m]["native"], "all_nll") for m in ms], axis=0))
        rec_out[X] = {"cap": c, "cap_per_seed": per, "cap_ok": bool(cap), "gates": gates, "head_minus_native_nll": hd,
                      "head_ok": hd[0] <= HEAD_BAR, "unstable": X in unstable_recipe}
    # ---- contrasts
    def contrast(X, Y, k, readout="out"):
        ds = [stat(M[mx][readout], k) - stat(M[my][readout], k) for mx, my in zip(keys[X], keys[Y])]
        c = ci_of(np.mean(ds, axis=0))
        per = [float(d[0]) for d in ds]
        return {"estimate": c, "per_seed": per, "label": sign_label(c, per, lower_better=k.split("_")[-1] in LOWER_BETTER)}
    cons = {}
    ba_n, ba_a = contrast("B", "A", "primary_nll"), contrast("B", "A", "primary_acc")
    spec = (ba_n["estimate"][0] <= SPEC_NLL and ba_n["estimate"][2] < 0 and ba_a["estimate"][0] >= SPEC_ACC
            and per_seed_ok(p < 0 for p in ba_n["per_seed"]) and per_seed_ok(p > 0 for p in ba_a["per_seed"]))
    cons["B-A"] = {"primary_nll": ba_n, "primary_acc": ba_a, "GAIN_SPEC": bool(spec), "label": "GAIN_SPEC" if spec else ba_n["label"]}
    cons["C-B"] = {k: contrast("C", "B", k) for k in ("cal_nll", "cal_acc", "dev_nll", "target_acc")}
    cons["D-B"] = {k: contrast("D", "B", k) for k in ("bf_kl", "bf_nll") + tuple(v[0] for v in GATES.values())}
    pair = {}
    for X in RECIPES:
        for Y in RECIPES:
            if X != Y:
                pair[f"{X}-{Y}"] = contrast(X, Y, "target_acc")
    cons["pairwise_target_acc"] = pair
    # fitted-only gain check (B - A on FITTED didev blocks, non-target and target): the transfer falsification
    fit_blocks = [b for b in blocks if fitted.get(b)]
    if fit_blocks:
        fmask = dd & np.isin(dev.block, fit_blocks)
        Df = Draws(dev.cluster[fmask], nboot)
        ds = [Df.block_mean(M[mx]["out"]["acc"][fmask], dev.block[fmask]) - Df.block_mean(M[my]["out"]["acc"][fmask], dev.block[fmask])
              for mx, my in zip(keys["B"], keys["A"])]
        c = ci_of(np.mean(ds, axis=0))
        cons["B-A_fitted_blocks_acc"] = {"estimate": c, "per_seed": [float(d[0]) for d in ds],
                                         "label": sign_label(c, [float(d[0]) for d in ds]), "blocks": fit_blocks}
    def sub(mask, k):
        Dm = Draws(dev.cluster[mask], nboot)
        out = {}
        for X in RECIPES:
            ds = [Dm.mean(M[m]["out"][k][mask]) - Dm.mean(ref_g[k][mask]) for m in keys[X]]
            out[X] = ci_of(np.mean(ds, axis=0))
        return out
    descriptive = {}
    if egmask_fit.any():
        descriptive["egval_source_fitted_vs_garas"] = {"units": int(egmask_fit.sum()), "acc": sub(egmask_fit, "acc"), "nll": sub(egmask_fit, "nll")}
    if esci_mask.any():
        descriptive["esci_vs_garas"] = {"units": int(esci_mask.sum()), "fitted_by": fitted.get("esci", []), "acc": sub(esci_mask, "acc"),
                                        "nll": sub(esci_mask, "nll")}
    for nm, mask in (("dev", devmask), ("cal", calmask)):
        descriptive[f"{nm}_ece_target_mass_vs_garas"] = {X: float(np.mean([ece_w(M[m]["out"]["accm"][mask], M[m]["out"]["mx"][mask],
                                                                                np.ones(int(mask.sum()))) for m in keys[X]])
                                                                   - ece_w(ref_g["accm"][mask], ref_g["mx"][mask], np.ones(int(mask.sum()))))
                                                          for X in RECIPES}
    stock = {X: {k: ci_of(np.mean([stat(M[m]["out"], k) - stat(R["stock"], k) for m in keys[X]], axis=0))
                 for k in ("cal_nll", "dev_nll", "target_acc")} for X in RECIPES}
    # ---- decision
    gok = {X: all(g["adoption_grade"] for g in v["gates"].values()) for X, v in rec_out.items()}
    elig = [X for X, v in rec_out.items() if v["cap_ok"] and gok[X] and v["head_ok"] and not v["unstable"]]
    note = ""
    if elig:
        best = max(elig, key=lambda X: rec_out[X]["cap"][0])
        rest = [X for X in elig if X != best]
        if rest:
            ru = max(rest, key=lambda X: rec_out[X]["cap"][0])
            if pair[f"{best}-{ru}"]["label"] != "GAIN":
                best = next(X for X in PREF if X in elig)
                note = f"; the CAP lead over the runner-up is NOT a resolved GAIN, so the registered order B, D, A, C picks {best}"
        row = f"ADVANCE_{best}"
        text = f"recipe {best} is eligible (CAP {rec_out[best]['cap'][0]:+.4f}){note}. Next: ONE sealed confirmation read (CONFIRM-P1)."
    elif any(v["cap_ok"] and gok[X] and not v["head_ok"] and not v["unstable"] for X, v in rec_out.items()):
        row, text = "HEAD_COSTS", "a recipe passes CAP and every gate but its posterior predictive costs NLL vs its own native readout."
    elif any(v["cap_ok"] for v in rec_out.values()):
        blk = {X: [g for g, v in rec_out[X]["gates"].items() if not v["adoption_grade"]] for X in RECIPES if rec_out[X]["cap_ok"]}
        row, text = "GUARD_BLOCKED", f"CAP holds for {sorted(blk)} but gates are not adoption-grade: {blk}; nothing advances."
    else:
        row, text = "NO_CAPABILITY_GAIN", "no recipe has CAP; nothing advances (not resolved, never 'no effect')."
    # ---- S-a and S-b
    lk = lrhi[0]
    b0 = next(k for k in keys["B"] if runs[k]["seed"] == 0)
    d_lr = ci_of(stat(M[lk]["native"], "pv_nll") - stat(M[b0]["native"], "pv_nll"))
    lr_label = "LR_OK" if d_lr[1] > -LR_TOL else ("LR_LIMITED" if d_lr[2] < -LR_TOL else "LR_UNRESOLVED")
    curves = {k: {s: float(unit_metrics(model[k]["curve_native"][s], dev.T, dev.K)["nll"][pvmask].mean()) for s in model[k]["curve_native"]}
              for k in model}
    still = all(curves[m][1250] - curves[m][1875] > STILL for X in RECIPES for m in keys[X])
    sa = {"label": lr_label, "lrhi_unstable_twice": lrhi_unstable, "pv_native_nll_lrhi_minus_lrstar": d_lr,
          "also": {k: ci_of(stat(M[lk]["out"], k) - stat(M[b0]["out"], k)) for k in ("cal_nll", "target_acc")}, "curves": curves}
    if lr_label == "LR_LIMITED" and row in ("GUARD_BLOCKED", "NO_CAPABILITY_GAIN"):
        text += " At LR*, possibly LR-limited (S-a): this is NOT evidence that full fine-tuning does not help."
    if lrhi_unstable:
        text += " The S-a LR_hi run was UNSTABLE under G3' on both attempts (reported with the LR label; it decides no row)."
    # ---- head report (7)
    head = {}
    PD = {p: Draws(dev.cluster[isp(p)], nboot) for p in PANELS}
    for X in RECIPES:
        per_m = {}
        for m in keys[X]:
            row_p = {}
            for p in PANELS:
                mask = isp(p)
                row_p[p] = {r: {"nll": float(M[m][r]["nll"][mask].mean()), "brier": float(M[m][r]["brier"][mask].mean()),
                                "acc": float(M[m][r]["acc"][mask].mean()),
                                "ece": float(ece_w(M[m][r]["acc"][mask], M[m][r]["mx"][mask], np.ones(int(mask.sum()))))}
                            for r in ("out", "mean", "det", "native")}
            per_m[m] = row_p
        nothing_pt = all(abs(per_m[m][p]["out"]["nll"] - per_m[m][p]["det"]["nll"]) < NOTHING_EPS for m in keys[X] for p in PANELS)
        od_ci = {p: ci_of(np.mean([PD[p].mean(M[m]["out"]["nll"][isp(p)] - M[m]["det"]["nll"][isp(p)]) for m in keys[X]], axis=0))
                 for p in PANELS}                    # A1: "nothing measurable" = an equivalence claim -> every panel's CI inside the margin
        nothing = nothing_pt and all(-NOTHING_EPS < c[1] and c[2] < NOTHING_EPS for c in od_ci.values())
        recal = nothing and float(np.mean([M[m]["out"]["nll"].mean() - M[m]["native"]["nll"].mean() for m in keys[X]])) <= -NOTHING_EPS
        Dt_idx = np.where(tmask)[0]
        au = []
        for m in keys[X]:
            err = (M[m]["out"]["acc"][Dt_idx] == 0).astype(int)
            au.append(auroc_draws(model[m]["epi"][Dt_idx], err, D["t"].W))
        a_ci = ci_of(np.mean(au, axis=0))
        beats = any(all(per_m[m][p]["det"]["nll"] - per_m[m][p]["out"]["nll"] > NOTHING_EPS for m in keys[X]) for p in PANELS)
        informative = a_ci[0] >= AUROC_MIN and a_ci[1] > 0.5 and beats
        head[X] = {"per_model": per_m, "epi_error_auroc_targets": a_ci, "out_minus_det_nll": od_ci, "selected": {m: run_info[m]["head"]["spec"] for m in keys[X]},
                   "reading": "head-weight uncertainty is informative" if informative else
                   ("the head works as a recalibration layer; the Bayesian part adds nothing measurable" if recal else
                    ("the Bayesian part adds nothing measurable" if nothing else
                     (f"OUTPUT - DET NLL is inside +-{NOTHING_EPS} at the point on every panel, but equivalence is NOT RESOLVED (a CI "
                      "extends beyond the margin)" if nothing_pt else "no pre-declared reading applies")))}
    # ---- recipe-level conclusions (6.1)
    concl = []
    if cons["B-A"]["GAIN_SPEC"]:
        concl.append("B - A = GAIN_SPEC: the targeted-data hypothesis is supported on source-held-out evidence / relevance.")
    fb = cons.get("B-A_fitted_blocks_acc")
    if fb and fb["label"] == "GAIN" and not cons["B-A"]["GAIN_SPEC"]:
        pa, pn = cons["B-A"]["primary_acc"]["estimate"], cons["B-A"]["primary_nll"]["estimate"]
        if pa[2] < SPEC_ACC and pn[1] > SPEC_NLL:
            concl.append("B - A gains on FITTED blocks while the source-held-out gain is bounded below the spec's bars (accuracy CI upper "
                         f"{pa[2]:+.4f} < +{SPEC_ACC}, NLL CI lower {pn[1]:+.4f} > {SPEC_NLL}): the transfer hypothesis is NOT SUPPORTED.")
        else:
            concl.append("B - A gains on FITTED blocks; the source-held-out transfer is NOT RESOLVED (no rejection claim).")
    cb = cons["C-B"]
    equiv = all(-EQUIV[k] <= cb[k]["estimate"][1] and cb[k]["estimate"][2] <= EQUIV[k] for k in EQUIV)
    if equiv:
        concl.append("C - B is EQUIVALENT within the declared margins (every CI inside +-" + ", +-".join(f"{EQUIV[k]} {k}" for k in EQUIV)
                     + "): preservation shows no effect larger than these margins.")
    elif cb["cal_nll"]["label"] == "COST":
        concl.append("C - B caldev NLL = COST for C (preservation helps calibration): keep alpha 0.15.")
    else:
        other = {k: cb[k]["label"] for k in EQUIV if cb[k]["label"] in ("GAIN", "COST")}
        concl.append("C - B: equivalence NOT RESOLVED (a CI extends beyond the declared margins); caldev NLL " + cb["cal_nll"]["label"]
                     + (f"; resolved for C - B: {other}" if other else "") + ".")
    db = cons["D-B"]
    if gok["D"] and db["bf_kl"]["estimate"][2] <= NONINF_BF_KL:
        concl.append(f"D is NON-INFERIOR on bfdev KL (CI upper {db['bf_kl']['estimate'][2]:+.4f} <= +{NONINF_BF_KL}) and holds every gate: "
                     "six percent numeric is not shown to be necessary at this margin.")
    elif db["bf_kl"]["label"] == "COST":
        concl.append("D - B bfdev KL = COST: keep six percent numeric.")
    else:
        concl.append("D - B: NOT RESOLVED (no non-inferiority claim at the declared margin).")
    trig = seed2_triggers(rec_out, cons["B-A"])
    res.update({"row": row, "text": text, "eligible": elig, "recipes": rec_out, "contrasts": cons, "vs_stock": stock, "lr_check": sa,
                "still_learning": bool(still), "head_report": head, "conclusions": concl, "seed2_trigger": trig, "descriptive": descriptive,
                "primary_units": {"held_out_target": int(tmask.sum()), "egval_held_out": int(egmask_held.sum())},
                "lr_label": lr_label})
    return res


def seed2_triggers(rec_out, ba):
    """the 6.4 trigger: a recipe whose ONLY obstacle to eligibility is a seed disagreement (CAP met at the 2-seed point and CI but not
    per seed; or a gate that HOLDS at the point but is noisy and split across seeds), or B - A GAIN_SPEC failing only per seed"""
    trig = []
    for X, v in rec_out.items():
        reasons = []
        if not v["cap_ok"]:
            split = v["cap"][0] >= CAP_MIN and v["cap"][1] > 0 and not per_seed_ok(p > 0 for p in v["cap_per_seed"])
            reasons.append(("cap", split))
        for g, gv in v["gates"].items():
            if not gv["adoption_grade"]:
                reasons.append((g, gv["status"] == "HOLDS" and gv["noise_exceeds_bar"]))
        if not v["head_ok"]:
            reasons.append(("head", False))
        if v.get("unstable"):
            reasons.append(("unstable", False))
        if len(reasons) == 1 and reasons[0][1]:
            trig.append({"recipe": X, "item": reasons[0][0]})
    n, a = ba["primary_nll"], ba["primary_acc"]
    if not ba["GAIN_SPEC"] and n["estimate"][0] <= SPEC_NLL and n["estimate"][2] < 0 and a["estimate"][0] >= SPEC_ACC:
        trig.append({"recipe": "B,A", "item": "B-A GAIN_SPEC fails only on the per-seed rule"})
    return trig


def report(res):
    L = [f"{VERSION}", f"ROW: {res['row']}  [{res.get('lr_label', 'LR check not read')}]", res["text"]]
    if res["row"] == "INVALID":
        L += [f"  problem: {p}" for p in res["problems"]]
        return "\n".join(L) + "\n"
    f = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]"  # noqa: E731
    L.append(f"held-out target blocks: {res['held_out_targets']}; fitted blocks: { {b: v for b, v in res['fitted_blocks'].items() if v} }")
    L.append(f"source-fitted EG-val units (outside the source-held-out primary, reported below): {res.get('egval_fitted_units', 0)} "
             f"from {res.get('egval_fitted_sources', [])}")
    L.append("training data per run (sha256 of the bytes trained on): " + ", ".join(
        f"{k} {(v.get('data_rows_sha256') or '')[:16] or 'legacy-unbound (body-verified only)'}" for k, v in sorted(res["runs"].items())))
    for X, v in res["recipes"].items():
        L.append(f"recipe {X}: CAP {f(v['cap'])} per-seed {[round(p, 4) for p in v['cap_per_seed']]} -> {'CAP' if v['cap_ok'] else 'no CAP'}; "
                 f"HEAD {f(v['head_minus_native_nll'])} {'ok' if v['head_ok'] else 'FAILS'}{'; UNSTABLE' if v['unstable'] else ''}")
        for g, gv in v["gates"].items():
            L.append(f"    {g:14s} {f(gv['estimate'])} bar {gv['bar']:+.3f} {gv['status']:6s} per-seed {gv['per_seed_status']} "
                     f"floor {gv['noise_floor']:.4f} -> {'adoption-grade' if gv['adoption_grade'] else 'NOT adoption-grade'}")
    c = res["contrasts"]
    L.append(f"B - A (PRIMARY CONTRAST): NLL {f(c['B-A']['primary_nll']['estimate'])} acc {f(c['B-A']['primary_acc']['estimate'])} -> "
             f"{c['B-A']['label']}")
    for name in ("C-B", "D-B"):
        for k, v in c[name].items():
            L.append(f"{name} {k}: {f(v['estimate'])} {v['label']}")
    L.append(f"S-a: {res['lr_check']['label']}{' [LR_hi run UNSTABLE on both attempts]' if res['lr_check'].get('lrhi_unstable_twice') else ''} (LR_hi - LR* pilot-val native NLL {f(res['lr_check']['pv_native_nll_lrhi_minus_lrstar'])}); "
             f"S-b STILL_LEARNING: {res['still_learning']}")
    for X, h in res["head_report"].items():
        L.append(f"head {X}: {h['reading']}; epi error-AUROC (targets) {f(h['epi_error_auroc_targets'])}")
    L += [f"conclusion: {x}" for x in res["conclusions"]]
    for k, v in (res.get("descriptive") or {}).items():
        if isinstance(v, dict) and "acc" in v:
            L.append(f"descriptive {k} ({v['units']} units{'; fitted by ' + str(v['fitted_by']) if 'fitted_by' in v else ''}), OUTPUT - Garas: "
                     + "; ".join(f"{X} acc {f(v['acc'][X])} nll {f(v['nll'][X])}" for X in RECIPES))
        else:
            L.append(f"descriptive {k} (OUTPUT - Garas, point): " + ", ".join(f"{X} {x:+.4f}" for X, x in v.items()))
    if res["seed2_trigger"]:
        L.append(f"6.4 seed-2 trigger: {res['seed2_trigger']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs")
    ap.add_argument("--out")
    ap.add_argument("--text")
    ap.add_argument("--nboot", type=int, default=B)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list-keys", action="store_true")
    ap.add_argument("--mirror")
    ap.add_argument("--pins")
    ap.add_argument("--dev", default="devP")
    ap.add_argument("--inputs-out", help="--mirror: also write the built inputs here")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.list_keys:
        print("\n".join(needed_keys(json.load(open(a.pins)), a.dev)))
        return 0
    if a.mirror:
        inp = build_inputs(a.mirror, json.load(open(a.pins)), a.dev)
        if a.inputs_out:
            json.dump(inp, open(a.inputs_out, "w"), indent=1)
    else:
        inp = json.load(open(a.inputs))
    res = read(inp, a.nboot)
    txt = report(res)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)
    if a.text:
        open(a.text, "w").write(txt)
    sys.stdout.write(txt)
    print("PILOT_READ " + json.dumps({"row": res["row"], "lr_label": res.get("lr_label"), "eligible": res.get("eligible")}))
    return 0


def self_test():
    P = np.array([[0.7, 0.3, 0.0], [0.2, 0.8, 0.0]]); T = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]); K = np.array([2, 2])
    um = unit_metrics(np.pad(P, ((0, 0), (0, 49))), np.pad(T, ((0, 0), (0, 49))), K)
    assert um["acc"].tolist() == [1.0, 1.0] and abs(um["nll"][0] + math.log(0.7)) < 1e-12 and abs(um["regret"][1]) < 1e-12
    assert abs(um["kl"][0] + math.log(0.7)) < 1e-12 and abs(um["kl"][1] - (0.5 * math.log(0.5 / 0.2) + 0.5 * math.log(0.5 / 0.8))) < 1e-12
    assert status([0.004, -0.001, 0.009], 0.005, "le") == "HOLDS" and status([0.009, 0.006, 0.012], 0.005, "le") == "HARM"
    assert status([-0.006, -0.01, -0.001], -0.005, "ge") == "FAILS" and status([-0.02, -0.03, -0.012], -0.010, "ge") == "HARM"
    assert per_seed_ok([True, False, True]) and not per_seed_ok([True, False]) and not per_seed_ok([True, False, False])
    assert cap_rule([0.025, 0.005, 0.04], [0.02, 0.03]) and not cap_rule([0.015, 0.005, 0.03], [0.01, 0.02])
    assert not cap_rule([0.05, -0.001, 0.1], [0.05, 0.05]) and not cap_rule([0.05, 0.01, 0.1], [0.11, -0.01])
    assert adoption_grade("HOLDS", ["HOLDS", "FAILS"], False) and not adoption_grade("HOLDS", ["HOLDS", "FAILS"], True)
    assert adoption_grade("HOLDS", ["HOLDS", "HOLDS"], True) and not adoption_grade("FAILS", ["HOLDS", "HOLDS"], False)
    assert adoption_grade("HOLDS", ["HOLDS", "FAILS", "HOLDS"], True)
    assert sign_label([-0.02, -0.03, -0.01], [-0.01, -0.03], lower_better=True) == "GAIN"
    assert sign_label([-0.02, -0.03, -0.01], [0.01, -0.03]) == "NOT_RESOLVED"
    assert abs(auroc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1])) - 1.0) < 1e-12
    assert abs(auroc(np.array([0.5, 0.5]), np.array([0, 1])) - 0.5) < 1e-12
    rng = np.random.default_rng(1)
    sc, er = rng.integers(0, 5, 60).astype(float), (rng.random(60) < 0.4).astype(int)
    Wt = np.vstack([np.ones(60), rng.integers(0, 3, 60)])
    v = auroc_draws(sc, er, Wt)
    assert abs(v[0] - auroc(sc, er)) < 1e-12 and abs(v[1] - auroc(sc, er, Wt[1])) < 1e-12
    st = {s: (1.0, 10.0) for s in range(200)}
    assert not g3_unstable(st)[0]
    assert not g3_unstable({s: (1.0 + 0.01 * (s % 5), 10.0 + (s % 7)) for s in range(200)})[0]
    st[120] = (1.0, 5000.0)
    assert not g3_unstable(st)[0]
    st[121], st[122] = (1.0, 5000.0), (1.0, 5000.0)
    assert g3_unstable(st)[0]
    st = {s: (1.0, 10.0) for s in range(200)}
    st[150] = (3.0, 5000.0)
    assert g3_unstable(st)[0]
    st = {s: (1.0, 10.0) for s in range(200)}
    st[10] = (float("nan"), 1.0)
    assert g3_unstable(st)[0]
    assert panel_name("pilotval:hans_v1") == "pilotval" and block_of("panel:didev:didev_hover") == "hover" and norm_name("Comm-QA_x") == "commqax"
    g_ok = {"adoption_grade": True, "status": "HOLDS", "noise_exceeds_bar": False}
    g_split = {"adoption_grade": False, "status": "HOLDS", "noise_exceeds_bar": True}
    g_fail = {"adoption_grade": False, "status": "FAILS", "noise_exceeds_bar": True}
    base = {"cap_ok": True, "cap": [0.05, 0.01, 0.09], "cap_per_seed": [0.05, 0.05], "head_ok": True, "unstable": False}
    ba0 = {"GAIN_SPEC": False, "primary_nll": {"estimate": [-0.01, -0.02, 0.0]}, "primary_acc": {"estimate": [0.0, -0.01, 0.01]}}
    t = seed2_triggers({"A": {**base, "gates": {"x": g_split, "y": g_ok}}, "B": {**base, "gates": {"x": g_split, "y": g_fail}},
                        "C": {**base, "cap_ok": False, "cap_per_seed": [0.09, -0.01], "gates": {"x": g_ok}},
                        "D": {**base, "gates": {"x": g_split}, "unstable": True}}, ba0)
    assert t == [{"recipe": "A", "item": "x"}, {"recipe": "C", "item": "cap"}], t
    ba1 = {"GAIN_SPEC": False, "primary_nll": {"estimate": [-0.03, -0.05, -0.01]}, "primary_acc": {"estimate": [0.02, 0.0, 0.04]}}
    assert seed2_triggers({}, ba1) == [{"recipe": "B,A", "item": "B-A GAIN_SPEC fails only on the per-seed rule"}]
    print("SELF_TEST_OK fullft_pilot_read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
