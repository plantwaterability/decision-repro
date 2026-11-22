#!/usr/bin/env python3
"""(stock_decision_v1) Final-feature head refit, the TRAINED deterministic residual control and the scalar temperatures, on CACHED
features of a frozen backbone (fullft_train.py --extract-features: h, ell, K, t, w per decision).

HEAD (plan.json bayesian_head). The production AuxVBLL (the pinned claude_lmtune.py, md5-checked by fullft_head.pinned_trainer), ONE
head {prior 1, kl 1, cov full}, dim 128, seed 0. Likelihood contract:
    z_j = tau ell_j + phi_j(h)^T beta,   q(beta) = N(mu, S) (S = L L^T, full),   prior N(0, I)
phi = the centred production feature map (P learned, a point estimate; fixed seeded letter codes); the closed-form classification
bound (AuxVBLL.bound). Per optimizer step over a global batch b of decisions:
    loss = -sum_{i in b} w_i bound_i / W_step  +  KL(q || p) / W_unique_headfit
W_step = sum_{i in b} w_i over the WHOLE global batch (micro-batches / ranks each divide by it; their gradients are SUMMED);
W_unique_headfit = the sum of the head-fit pool's UNIQUE decision weights (every decision once; duplicates refused). The KL term is
added once per step. The learned point-estimate feature map makes this a CONDITIONAL variational last-layer posterior (given the
backbone's features and P), not a posterior over the Gemma weights.
SAMPLING: epochs of a seeded permutation of the unique decisions (uniform, without replacement within an epoch); the decision weight
enters the likelihood only, never the sampling, so it is applied exactly once. 1,500 steps, global batch 64, AdamW (0.9, 0.95,
eps 1e-8, wd 0), LR 1e-3, linear warmup 50 then cosine to 0, per-head gradient clip 1.0 (AuxVBLL.clip_).

DETERMINISTIC RESIDUAL CONTROL: the SAME feature map and initialisation (an AuxVBLL built with the same seed), z_j = tau ell_j +
phi_j(h)^T mu, trained with the deterministic weighted cross-entropy (no variance, no KL; the covariance parameters are frozen and
unused), same sampler / optimizer / steps. It is NOT a KL = 0 VBLL (that would remain stochastic).

TEMPERATURE: one scalar T per readout, fitted on CALIBRATION selection only: minimise the source-balanced NLL (the mean over sources of
each source's mean decision NLL) over log T in [log 0.1, log 10] by golden-section search to tolerance 1e-6, with the fixed draws:
    native+T   softmax(ell / T)
    det+T      softmax(z_det / T)
    mean+T     softmax((tau ell + phi^T mu) / T)             (posterior-mean residual)
    vbll+T     mean_s softmax(z_s / T)                       (64 fixed draws, seed 0)
The VBLL semantics divide each draw's logits by T before the softmax; they differ from softmax(mean_s z_s / T) and from powering the
averaged probabilities. predict_T is the one function used by the fit, calibration, scoring and the export (vbll_head.json
"temperature"; release/garas_vbll/vbll_core.predict(temperature=T) serves it the same way).

  python3 sdv1_head.py refit --features HEADFIT.pt --pinned-code DIR --out HEADDIR [--steps 1500 --batch 64 --lr 1e-3]
  python3 sdv1_head.py fit-t --features CAL.pt --heads HEADDIR             (writes HEADDIR/temperature.json + the export's T)
  python3 sdv1_head.py score --features DEV.pt --heads HEADDIR --out PRED.pt   (the four readouts, pre / post T)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fullft_head as fh  # noqa: E402

VERSION = "sdv1_head v4.1 (2026-10-15; v4.0 + the Latent01 readout device-safe for CUDA serving; rules select | sdh | beta0 | p0beta | fixedE | lloc | l01post | l01post_di)"
# v4.1 (Gemini review of Codex, 10-15): l01_extract / l01_probs allocate every buffer on the states' device (they built CPU zeros /
# ones; CUDA serving crashed) and the plain-softmax fast path is taken only when no noul variance applies (infl 0 or no noul row).
# v4.0 (user decision 10-15 via the coordinator: serve the full Latent01-post): rule l01post = the reference Latent01 (research/claude_bayes_
# handoff/latent01_src/run_latent01_all8.py e80a40ed) with MACKAY-HEAD's frozen constants (L01, predict2 parametrisation): refit fits the
# FastLatent01 state on the head-fit pool (V_E, mu_hp, V_perp, dbar_H, the noul Laplace, per-kind s), fit-t its scalar T on the cal pool,
# score / export the readout (lloc shift + [0, 1] one-vs-rest S-curve sigma integral, 16-node Gauss-Hermite + the score-interval integral).
# Kinds: the manifest's kind at training; the serving engine passes the request's question type (choice / noul; the kit has no score type).
# Until L01 is frozen, l01post is neither fitted nor selectable.
# v3.9 (coordinator 10-15, MACKAY-HEAD AGENT_NOTES 10:59Z; tmp/mackay_head/lloc_fixed.py bafa3f07 on FastLatent01 e80a40ed): the L-loc head
# (rule lloc), MACKAY-HEAD's frozen constants (noul_dim 128, lam_noul 1e3, alpha 0, beta 0.25):
#   flag   the Yes / No rows come from the RENDERED prompt only (yes_no_index: K = 2, options exactly Yes / No, plain or "Yes: <definition>",
#          either order; the same code as vbll_core.yes_no_index, so the engine derives the same flag from the prompt it renders); the
#          manifest kind is never used (its agreement is reported: lloc_flags)
#   refit  on the head-fit features (+ their manifest's prompts): V_E (E's right singular vectors), V_perp = top-128 PCA (pca_lowrank
#          seed 0) of the centred rms(h - (h V_E) V_E^T) of ALL head-fit rows, mu / S = the unweighted Laplace MAP / covariance of the Yes /
#          No logit-difference residual (prior N(0, I / 1e3)) -> lloc_fit.pt
#   serve  z = ell; on Yes / No rows z_yes += m, z_no -= m, m = (beta / 2) u . (mu + L eps_s), u = rms(h - (h V_E) V_E^T) V_perp;
#          p = mean_s softmax(z_s / T) over 32 antithetic pairs (the Bayesian head requirement: the posterior predictive is served);
#          lloc0 = the MAP readout (MACKAY-HEAD's frozen L-loc exactly) is fitted (its own T) and reported, never served
#   T      one scalar T per readout on the CAL pool (source-balanced NLL); export = the vbll_core L-loc schema (jev-lloc-head-v1)
# v3.8 (Codex review 2026-10-14 2050): (a) the fixedE grid includes kappa = 0 (mean only), each cell with its own T; the served fixedE
# head stays kappa > 0 (the Bayesian head is a hard requirement, user 10-12), the best kappa = 0 cell is reported as candidate 'fe0'
# (and whether it would have won); (b) honest naming: the covariance is a FISHER GAUSSIAN around the isotropic blend mu = alpha 1 -- NOT a
# Laplace at a MAP: alpha is chosen on cal, not by maximising the head-fit posterior; the head-fit gradient at alpha 1 is recorded
# (fixede_fisher.pt; v3.7's fixede_laplace.pt still loads); (c) the export names its draw spec as draw_seed / samples, and
# vbll_core.FixedEHead honours n_samples / seed (antithetic pairs) and refuses a draw grid.
# v3.7 (coordinator 10-15, MACKAY-HEAD tmp/mackay_head/sd12_fixedE.py, verified on the 8 12B runs): the FIXED-E SUBSPACE head 'fe'.
#   V      the thin SVD basis of the letter rows E [52, D] = U S V^T: V [D, 52]; codes = E V [52, 52]
#   phi    phi_bk = (h V) * ((E_k - mean_{j < K_b} E_j) V)   [n, K, 52]; NO RMS norms, NO 1 / sqrt(F); sum_f phi_bk = centred h . E_k
#   mean   z_bk = ell_bk + phi_bk . mu,  mu = alpha * 1 (isotropic: alpha * the centred raw letter logit)
#   cov    S = kappa (lam I + sum_b w_b [sum_k p_bk phi_bk phi_bk^T - phibar_b phibar_b^T])^-1, the centred Fisher (Laplace) on the HEAD-FIT
#          pool at p = softmax(ell + alpha z~), lam = 1; L = chol(S)
#   draws  32 seed-0 normal draws [32, 52] (fp64) and their negatives (64 draws, sample mean exactly 0); p = mean_s softmax((z + phi L eps_s) / T)
#   select alpha in FIXEDE alphas, kappa in FIXEDE kappas and ONE scalar T on the CAL pool (source-balanced NLL); NO noul gate
# refit stores fixede_laplace.pt (V, codes, L per alpha); fit-t fits 'fe' under every rule (reported) and rule fixedE serves it: the
# export is the vbll_core 'fixedE' schema (V, codes, mu, L, T, the draw spec), served unchanged by release/garas_vbll/vbll_core.FixedEHead.
# v3.6: the features' letter readout (fullft_head $JEV_LETTER_READOUT: stock | blend:A) is part of their binding: refit / fit-t / score
# refuse unless the --expect letter_readout (absent = stock) equals this process's readout, so the export's vbll_head.json
# (fullft_head.export_head records a non-stock readout) names the readout the head was trained on.
# v3.5 (coordinator 10-14, MACKAY-HEAD fallback): refit also fits 'p0' = the AuxVBLL with its feature map P FROZEN at the seeded init
# (mu, S, tau trained by the same bound / sampler / optimizer / steps; P untouched, so phi is the init map). fit-t builds 'p0b' = p0
# with mu x beta*, beta* chosen ON THE CAL POOL from BETA_GRID by the minimum source-balanced vbll+T NLL (each beta with its own scalar
# T; ties to the smaller beta); every beta's NLL is recorded. Rule p0beta serves p0b (stored as heads/vbll_p0b.pt; the export carries
# mu x beta*, the engine serves it unchanged). p0 / p0b are reported under every rule; 'select' still chooses among {sdh, iso, a1} only.
# v3.4 (coordinator 10-14, the 12B head analysis): HEAD RULES (fit-t --rule). The candidate b0 = the sdh head with its posterior MEAN
# scaled by beta = 0 (--mean-scale; covariance S, the feature map P and tau unchanged): the beta = 0 posterior predictive
# mean_s softmax((tau ell + phi^T eps_s) / T), eps_s ~ N(0, S) -- the VBLL covariance kept, the mean shift off. It is a stored head
# (heads/vbll_b0.pt, mu x beta), so the serving export carries the scaled mu and the engine serves it UNCHANGED. Rules:
#   select  (default; = v3.3) min cal source-balanced vbll+T NLL over the available {sdh, iso, a1}
#   sdh     the AuxVBLL refit as fitted (= v2.x's served head)
#   beta0   b0, with ONE scalar T fitted on the cal features given (the bridge: cal_famheld_clean2)
# Every candidate (b0 included) is fitted (T), scored and reported under every rule; native+T stays the control readout.
# v3.3 (review 2026-10-14 17:00): (P1) the serving engine applies ONE scalar temperature, so selection and export use the scalar T only;
# the two-bucket fit is REPORTED (would_improve) but never adopted, and export_serving_head refuses per-kind temperatures; score's
# default readouts are the SELECTED candidate (the sdh-only legacy set is readouts_sdh). (P2) the evidence finalizer: gamma from the
# eigenvalues of the data Hessian (sum lam / (lam + alpha), never negative), alpha clamped to [ALPHA_MIN, ALPHA_MAX], a vanishing
# residual (|mu|^2 -> 0) is the explicit limit alpha = ALPHA_MAX; convergence is recorded (hitting the loop cap is NOT converged); an
# unconverged or failed candidate is marked unavailable and the run continues without it.
# v3.2: = v3.1 + the fullft29 fix (v2.2): the feature binding rounds targets from float32 on both sides (the extractor's precision).
# v3.1 (outside review): score rows (e.g. K = 50 bayes_families grids) can pull a single global T. Each candidate's vbll+T is also fitted
# with TWO buckets on cal_select (T_choice for choice / noul rows, T_score for score rows; each bucket's T by golden section on its own
# source-balanced NLL); the two-bucket form is ADOPTED for a candidate only if it lowers cal_select's overall source-balanced NLL, and the
# candidate choice uses the adopted form. The bucket of a row = its manifest 'kind' (or 'qtype'), read from --manifest in the feature
# order (check-features guarantees the order). dev is scored with every form for reporting; it is never used to choose.
# v3 (2026-10-14, coordinator item 3 after the MACKAY-HEAD A/B): after the AuxVBLL.bound refit ('sdh'), a STAGE-B FINALIZER builds two
# more candidate heads on the SAME head-fit features: P and tau frozen (the sdh feature map), mu re-solved by exact (damped) Newton on
# sum_i w_i CE(t_i, softmax(tau ell_i + phi_i mu)) + alpha / 2 |mu|^2, and the ISOTROPIC Laplace covariance S = (H + alpha I)^-1 at the
# mode (NO ARD; no ell-dropout -- both hurt in the A/B):
#   iso   alpha by MacKay's evidence fixed point (gamma = F - alpha tr S, alpha = gamma / |mu|^2), alternated with the Newton solve
#   a1    alpha = 1 (the refit's own N(0, I) prior on the head-fit pool's total weight)
# fit-t fits T per readout as before and the vbll+T temperature of every candidate on cal_select, and SELECTS among {sdh, iso, a1}
# by cal_select's source-balanced vbll+T NLL (ties: sdh first). dev is never read for the choice; the serving export is the selected
# candidate with its T. score reports every candidate and the selection.
HEAD_SPEC = {"prior": 1.0, "kl": 1.0, "cov": "full"}
REFIT = {"steps": 1500, "batch": 64, "lr": 1e-3, "warmup": 50, "dim": 128, "seed": 0, "samples": 64, "draw_seed": 0, "clip": 1.0}
T_BOUNDS, T_TOL = (0.1, 10.0), 1e-6
READOUTS = ("native", "det", "mean", "vbll")


# ------------------------------------------------------------------------------------------------------------------ features
def load_features(path):
    f = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("h", "ell", "k", "t", "w", "decision_uid", "E_letters"):
        if k not in f:
            raise SystemExit(f"{path}: features lack {k!r}")
    return f


def loss_of(r):
    """(v2.1) = fullft_train.loss_of (tested equal): the row's per-option losses in option_order and a status ok / absent / partial"""
    lo = r.get("loss_by_option")
    if lo is None:
        lo = (r.get("meta") or {}).get("loss_by_option")
    if lo is None:
        return None, "absent"
    if isinstance(lo, str):
        lo = json.loads(lo)
    keys = list(r["option_order"])
    if isinstance(lo, dict):
        if any(k not in lo for k in keys):
            return None, "partial"
        vals = [lo[k] for k in keys]
    elif isinstance(lo, (list, tuple)) and len(lo) == len(keys):
        vals = list(lo)
    else:
        raise ValueError(f"{r.get('decision_uid')}: loss_by_option is neither a dict over the options nor a list in option order")
    vals = [float(v) for v in vals]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{r.get('decision_uid')}: non-finite loss_by_option")
    return vals, "ok"


def f32(x):
    """the float32 value of x as a Python float (the precision the extractor stores targets in)"""
    import struct
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def first_difference(a, b):
    """-> a short description of the first differing row between two id lists (None when equal)"""
    if len(a) != len(b):
        return f"{len(a)} rows vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            for j, nm in enumerate(("decision_uid", "prompt_sha256", "K", "targets", "losses")):
                xv, yv = (x[j] if j < len(x) else None), (y[j] if j < len(y) else None)
                if xv != yv:
                    return f"row {i} ({x[0]}): {nm} {str(xv)[:120]} != {str(yv)[:120]}"
    return None


def manifest_ids(path):
    """the ordered decision identities of a trainer manifest: decision_uid, prompt_sha256, K and the gold vector in letter order
    (v2.1: + the loss vector in letter order when the row has a complete loss_by_option)"""
    out = []
    with open(path) as fh_:
        for line in fh_:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("manifest_header"):
                continue
            keys = list(r["option_order"])
            row = [r["decision_uid"], r["prompt_sha256"], len(keys), [round(f32(r["gold"][k]), 6) for k in keys]]
            vals, st = loss_of(r)
            if st == "ok":
                row.append([round(v, 6) for v in vals])
            out.append(row)
    return out


def features_ids(f):
    k = f["k"].tolist()
    t = f["t"].tolist()
    out = [[u, p, int(kk), [round(f32(x), 6) for x in tt[:kk]]] for u, p, kk, tt in zip(f["decision_uid"], f["prompt_sha256"], k, t)]
    if "has_loss" in f:                              # (v2.1) the cache's loss vectors are part of its identity
        L, has = f["loss_by_option"].tolist(), f["has_loss"].tolist()
        for i, row in enumerate(out):
            if has[i]:
                row.append([round(float(x), 6) for x in L[i][:row[2]]])
    return out


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_stamp(f, expect):
    """the binding of a feature cache: the caller's expected identity (identity / export / policy / role / manifest digest / extractor
    code) plus the file's own ordered decision / prompt / target / K digest and its recorded extraction policy"""
    return {**expect, "decisions_digest": digest(features_ids(f)), "n": len(f["decision_uid"]),
            "recorded_policy": (f.get("meta") or {}).get("attention_policy")}


def check_features(path, meta_path, manifest, expect):
    """-> list of reasons the cache must NOT be reused (empty = valid): the stamp must equal the expected binding, the file must hold
    exactly the manifest's ordered decision / prompt / target / K identities, and its recorded policy must be the expected one"""
    why = []
    if not (os.path.exists(path) and os.path.exists(meta_path)):
        return ["missing features or stamp"]
    st = json.load(open(meta_path))
    for k, v in expect.items():
        if st.get(k) != v:
            why.append(f"stamp {k}: {st.get(k)!r} != expected {v!r}")
    f = torch.load(path, map_location="cpu", weights_only=False)
    ids = features_ids(f)
    if digest(ids) != st.get("decisions_digest"):
        why.append("the file's decision / prompt / target / K digest != its stamp")
    mids = manifest_ids(manifest)
    if ids != mids:
        why.append("the file's ordered decision / prompt / target / K ids != the manifest's: " + str(first_difference(ids, mids)))
    if (f.get("meta") or {}).get("attention_policy") != expect.get("policy"):
        why.append(f"extracted under policy {(f.get('meta') or {}).get('attention_policy')} != {expect.get('policy')}")
    return why


def unique_weight(f):
    """W_unique_headfit: the pool's unique decisions, each once (a duplicate decision_uid is refused: the pool must be unique)"""
    uids = list(f["decision_uid"])
    if len(set(uids)) != len(uids):
        raise SystemExit(f"head-fit pool: {len(uids) - len(set(uids))} duplicate decisions (the pool must hold each decision once)")
    w = f["w"].double()
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or float(w.sum()) <= 0:
        raise SystemExit("head-fit pool: weights must be finite, nonnegative, with a positive sum")
    return float(w.sum())


# ------------------------------------------------------------------------------------------------------------------- heads
def make_head(lt, E_letters, kind="vbll", dim=REFIT["dim"], seed=REFIT["seed"]):
    """the production AuxVBLL with the fixed head spec; kind 'det' = the same module (same feature map and init) whose covariance
    parameters are frozen and unused (deterministic residual control)"""
    head = lt.AuxVBLL(E_letters.float(), [dict(HEAD_SPEC)], dim, seed=seed)
    if kind == "vbll_p0":                            # (v3.5) the feature map P frozen at its seeded init
        head.P.requires_grad_(False)
        kind = "vbll"
    elif kind == "det":
        head.L_diag.requires_grad_(False)
        if head.L_off is not None:
            head.L_off.requires_grad_(False)
    elif kind != "vbll":
        raise ValueError(kind)
    head.jev_kind = kind
    return head


def det_logits(head, h, ell, ks):
    mean, _, m = head.moments(h, ell, ks)
    return mean[0], m


def step_loss(head, batches, W_unique):
    """one optimizer step's loss over a GLOBAL batch given as micro-batches [(h, ell, t, k, w)]: each micro-batch's data term is
    divided by the global W_step (their sum = the whole batch's), the KL term (VBLL) is added once. -> (loss, data, kl)"""
    W_step = float(sum(float(b[4].sum()) for b in batches))
    if W_step <= 0:
        raise ValueError("a global batch with zero total weight")
    data = 0.0
    for h, ell, t, k, w in batches:
        if head.jev_kind == "vbll":
            bnd = head.bound(h, ell, t, k)[0]
            data = data + (-(w.float() * bnd).sum() / W_step)
        else:
            z, m = det_logits(head, h, ell, k)
            logp = torch.log_softmax(z.masked_fill(~m, -1e9), -1).masked_fill(~m, 0.0)
            data = data + (-(w.float() * (t.float()[:, :z.shape[1]] * logp).sum(-1)).sum() / W_step)
    kl = (head.kl_w * head.kl()).sum() / W_unique if head.jev_kind == "vbll" else torch.zeros((), dtype=torch.float32)
    return data + kl, data, kl


def sampler(n, steps, batch, seed):
    """index batches: consecutive epochs of numpy default_rng([seed, epoch]) permutations of range(n), cut into `batch`-sized slices
    (uniform without replacement inside an epoch; an epoch's tail continues into the next epoch's permutation)"""
    import numpy as np
    out, buf, e = [], [], 0
    while len(out) < steps:
        if len(buf) < batch:
            buf = buf + [int(x) for x in np.random.default_rng([int(seed), e]).permutation(n)]
            e += 1
            continue
        out.append(buf[:batch])
        buf = buf[batch:]
    return out


def fit(head, f, steps=REFIT["steps"], batch=REFIT["batch"], lr=REFIT["lr"], warmup=REFIT["warmup"], seed=REFIT["seed"],
        micro=1, log_every=0):
    """train `head` (vbll or det) on the cached features f -> history. `micro` splits each global batch into that many micro-batches
    (the gradient is identical: the normalisation is global)."""
    W_unique = unique_weight(f)
    h, ell, ks, t, w = f["h"].float(), f["ell"].float(), f["k"].long(), f["t"].float(), f["w"].float()
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(1, steps))))
    hist = []
    for si, idx in enumerate(sampler(len(ks), steps, batch, seed)):
        ix = torch.tensor(idx)
        parts = [ix[j::micro] for j in range(micro)] if micro > 1 else [ix]
        loss, data, kl = step_loss(head, [(h[p], ell[p], t[p], ks[p], w[p]) for p in parts], W_unique)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norms = head.clip_(REFIT["clip"])
        opt.step()
        sched.step()
        if log_every and (si % log_every == 0 or si == steps - 1):
            hist.append({"step": si, "loss": float(loss), "data": float(data), "kl": float(kl), "gnorm": float(norms[0])})
    return {"W_unique_headfit": W_unique, "steps": steps, "batch": batch, "lr": lr, "warmup": warmup, "seed": seed, "history": hist}


# ------------------------------------------------------------------------------------------------------------- readouts + T
def draw_logits(head, h, ell, ks, samples=REFIT["samples"], seed=REFIT["draw_seed"]):
    """the VBLL predictive's fixed draws exactly as AuxVBLL.predict / vbll_core.predict draw them: z [S, n, K], mask [n, K]"""
    mean, var, mask, phi = head.moments(h.float(), ell.float(), ks, want_phi=True)
    eps = torch.randn(head.mu.shape[0], int(samples), head.F, generator=torch.Generator().manual_seed(int(seed))).to(phi.device)
    beta = head.mu[:, None] + (torch.einsum("gef,gsf->gse", head.chol(), eps) if head.L_off is not None
                               else eps * head.sigma()[:, None])
    z = head.log_tau.exp()[:, None, None, None] * ell.float()[None, None] + torch.einsum("gbkf,gsf->gsbk", phi, beta)
    return z[0], mask


def base_logits(kind, head, h, ell, ks):
    """the logits a readout tempers: native ell; det / mean: tau ell + phi^T mu of `head`; vbll: the draws [S, n, K]"""
    kmax = ell.shape[1]
    mask = torch.arange(kmax)[None, :] < ks[:, None]
    if kind == "native":
        return ell.float(), mask
    if kind in ("det", "mean"):
        return det_logits(head, h, ell, ks)
    if kind == "vbll":
        return draw_logits(head, h, ell, ks)
    raise ValueError(kind)


def probs_T(kind, z, mask, T):
    """THE temperature semantics: native / det / mean softmax(z / T); vbll mean_s softmax(z_s / T)"""
    if kind == "vbll":
        return torch.softmax((z / T).masked_fill(~mask[None], -1e9), -1).mean(0).masked_fill(~mask, 0.0)
    return torch.softmax((z / T).masked_fill(~mask, -1e9), -1).masked_fill(~mask, 0.0)


def predict_T(kind, head, h, ell, ks, T=1.0):
    with torch.no_grad():
        z, m = base_logits(kind, head, h, ell, ks)
        return probs_T(kind, z, m, float(T))


def source_balanced_nll(p, t, sources):
    nll = -(t.double()[:, :p.shape[1]] * p.double().clamp_min(1e-12).log()).sum(-1)
    groups = {}
    for i, s in enumerate(sources):
        groups.setdefault(str(s), []).append(i)
    return float(sum(nll[g].mean() for g in groups.values()) / len(groups))


def golden(fun, lo, hi, tol=T_TOL):
    """golden-section minimisation of fun on [lo, hi] to interval width <= tol (deterministic) -> (x, f(x))"""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fun(c), fun(d)
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = fun(d)
    x = (a + b) / 2
    return x, fun(x)


def manifest_kinds(path):
    """per manifest row (feature order): 'score' or 'choice' (choice / noul / anything else); None when no manifest"""
    if not path:
        return None
    out = []
    with open(path) as fh_:
        for line in fh_:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("manifest_header"):
                continue
            out.append("score" if (r.get("kind") or r.get("qtype")) == "score" else "choice")
    return out


def probs_Tvec(kind, z, mask, Tv):
    """probs_T with one temperature per row (Tv [n])"""
    if kind == "vbll":
        return torch.softmax((z / Tv[None, :, None]).masked_fill(~mask[None], -1e9), -1).mean(0).masked_fill(~mask, 0.0)
    return torch.softmax((z / Tv[:, None]).masked_fill(~mask, -1e9), -1).masked_fill(~mask, 0.0)


def fit_T2(kind, head, f, kinds, single, bounds=T_BOUNDS, tol=T_TOL):
    """(v3.1) the two-bucket temperatures on calibration features and the overall source-balanced NLL with them; adopted iff it beats
    the single T's (`single` = fit_T's result)"""
    h, ell, ks, t = f["h"].float(), f["ell"].float(), f["k"].long(), f["t"].float()
    src = [s_ if s_ is not None else "unknown" for s_ in (f.get("source") or ["all"] * len(ks))]
    if kinds is None or len(kinds) != len(ks):
        return {"available": False, "adopted": False, "why": "no per-row kinds (no --manifest or a length mismatch)"}
    out = {"available": True, "n": {b: kinds.count(b) for b in ("choice", "score")}, "T": {}}
    with torch.no_grad():
        z, m = base_logits(kind, head, h, ell, ks)
        for b in ("choice", "score"):
            ix = [i for i, k in enumerate(kinds) if k == b]
            if not ix:
                out["T"][b] = single["T"]
                continue
            ixt = torch.tensor(ix)
            zb = z[:, ixt] if kind == "vbll" else z[ixt]
            obj = lambda lt_, zb=zb, ixt=ixt, ix=ix: source_balanced_nll(probs_T(kind, zb, m[ixt], math.exp(lt_)), t[ixt], [src[i] for i in ix])  # noqa: E731
            x, _ = golden(obj, math.log(bounds[0]), math.log(bounds[1]), tol)
            out["T"][b] = math.exp(x)
        Tv = torch.tensor([out["T"][k] for k in kinds], dtype=torch.float32)
        out["nll_post"] = source_balanced_nll(probs_Tvec(kind, z, m, Tv), t, src)
    out["nll_single"] = single["nll_post"]
    out["adopted"] = out["nll_post"] < single["nll_post"]
    return out


def fit_T(kind, head, f, bounds=T_BOUNDS, tol=T_TOL):
    """T for one readout on calibration features: golden section over log T in [log lo, log hi], source-balanced NLL, fixed draws"""
    h, ell, ks, t = f["h"].float(), f["ell"].float(), f["k"].long(), f["t"].float()
    src = f.get("source") or ["all"] * len(ks)
    src = [s if s is not None else "unknown" for s in src]
    with torch.no_grad():
        z, m = base_logits(kind, head, h, ell, ks)
        obj = lambda lt_: source_balanced_nll(probs_T(kind, z, m, math.exp(lt_)), t, src)  # noqa: E731
        x, fx = golden(obj, math.log(bounds[0]), math.log(bounds[1]), tol)
        pre = obj(0.0)
    return {"T": math.exp(x), "log_T": x, "nll_pre": pre, "nll_post": fx, "bounds": list(bounds), "tol": tol,
            "objective": "source-balanced NLL (mean over sources of the mean decision NLL)", "n": int(len(ks)),
            "sources": len(set(map(str, src))), "at_bound": abs(x - math.log(bounds[0])) < 1e-3 or abs(x - math.log(bounds[1])) < 1e-3}


# ------------------------------------------------------------------------------------------------------------------- export
# ------------------------------------------------------------------------------------------------------------ v3 finalizer
FINAL = {"newton_tol": 1e-12, "newton_max": 60, "alpha_tol": 1e-6, "alpha_max": 200, "chunk": 1024, "alpha_lo": 1e-8, "alpha_hi": 1e8}
CANDIDATES = ("sdh", "iso", "a1")
RULES = ("select", "sdh", "beta0", "p0beta", "fixedE", "lloc", "l01post", "l01post_di")
FIXEDE = {"alphas": (0.0, 0.005, 0.01, 0.02, 0.03, 0.05), "kappas": (0.0, 1.0, 10.0, 100.0), "pairs": 32, "seed": 0, "lam": 1.0, "chunk": 1024}
FIXEDE_SCHEMA = "jev-fixede-head-v1"


def fe_basis(E):
    """(v3.7) V [D, 52] orthonormal (the thin SVD of the letter rows E [52, D] = U S V^T) and codes = E V [52, 52], fp64"""
    _, _, Vt = torch.linalg.svd(E.double(), full_matrices=False)
    V = Vt.T.contiguous()
    return V, E.double() @ V


def fe_phi(h, ks, K, V, codes):
    """phi [n, K, 52] fp64 (zero on unshown options) and the shown-option mask [n, K]. The SAME code as vbll_core.FixedEHead.phi."""
    n = int(ks.shape[0])
    mask = torch.arange(K, device=h.device)[None, :] < ks[:, None]
    m = mask.double()
    hE = h.double() @ V
    ek = codes[:K][None].expand(n, K, -1)
    ebar = (ek * m[..., None]).sum(1, keepdim=True) / m.sum(1)[:, None, None]
    return hE[:, None, :] * (ek - ebar) * m[..., None], mask


def fe_eps(F, pairs=FIXEDE["pairs"], seed=FIXEDE["seed"]):
    """the fixed antithetic draws [2 * pairs, F] fp64: seed-0 normals, then their negatives (vbll_core.fe_eps is the same code)"""
    e = torch.randn(int(pairs), int(F), generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64)
    return torch.cat([e, -e])


def fe_probs(ell, phi, mask, mu, L, T, eps):
    """the served predictive p [n, K] = mean_s softmax((ell + phi mu + phi L eps_s) / T) (vbll_core.FixedEHead.predict: same ops)"""
    mean = ell.double() + phi @ mu
    noise = torch.einsum("bkf,sf->sbk", phi, eps.to(phi.device) @ L.T).float()
    z = (mean.float()[None] + noise) / float(T)
    return torch.softmax(z.masked_fill(~mask[None], -1e9), -1).mean(0).masked_fill(~mask, 0.0)


def fe_fisher_gaussian(f, V, codes, alphas=FIXEDE["alphas"], lam=FIXEDE["lam"], chunk=FIXEDE["chunk"]):
    """(v3.8 name) the FISHER GAUSSIAN around the isotropic blend: per alpha, L = chol((lam I + sum_b w_b [sum_k p phi phi^T - phibar
    phibar^T])^-1) on these (head-fit) features at p = softmax(ell + alpha z~). NOT a Laplace at a MAP (alpha 1 is not the head-fit
    posterior mode); the head-fit gradient of sum_b w_b CE + lam / 2 |mu|^2 at mu = alpha 1 is recorded per alpha (grad_norm).
    -> {alpha: L [F, F] fp64}, info"""
    h, ell, ks, w = f["h"], f["ell"], f["k"].long(), f["w"].double()
    K = int(ks.max())
    F = int(V.shape[1])
    Hs = {a: torch.zeros(F, F, dtype=torch.float64) for a in alphas}
    G = {a: torch.zeros(F, dtype=torch.float64) for a in alphas}
    for s0 in range(0, len(ks), chunk):
        sl = slice(s0, s0 + chunk)
        phi, mask = fe_phi(h[sl], ks[sl], K, V, codes)
        zt = phi.sum(-1)
        for a in alphas:
            p = torch.softmax((ell[sl, :K].double() + a * zt).masked_fill(~mask, -1e9), -1) * mask
            pbar = torch.einsum("bk,bkf->bf", p, phi)
            pw = p * w[sl, None]
            Hs[a] += torch.einsum("bk,bkf,bkg->fg", pw, phi, phi) - (pbar * w[sl, None]).T @ pbar
            tt = f["t"][sl, :K].double() * mask
            tt = tt / tt.sum(1, keepdim=True).clamp_min(1e-12)
            G[a] += torch.einsum("bk,bkf->f", (p - tt) * w[sl, None], phi)
    out, info = {}, {}
    for a in alphas:
        Hm = 0.5 * (Hs[a] + Hs[a].T) + lam * torch.eye(F, dtype=torch.float64)
        S = torch.linalg.inv(Hm)
        out[a] = torch.linalg.cholesky(0.5 * (S + S.T))
        gfull = G[a] + lam * float(a) * torch.ones(F, dtype=torch.float64)
        info[str(a)] = {"trace_S": float(S.trace()), "min_eig_H": float(torch.linalg.eigvalsh(Hm).min()), "grad_norm_at_alpha1": float(gfull.norm()),
                        "newton_step_norm": float((S @ gfull).norm())}
    return out, {"n": int(len(ks)), "K": K, "lam": lam, "weights": "w (the pool's decision weights)", "per_alpha": info,
                 "kind": "Fisher Gaussian around the isotropic blend (not a MAP Laplace)"}


fe_laplace = fe_fisher_gaussian                      # v3.7 name (kept for callers)


def fe_fit(f, lap, kappas=FIXEDE["kappas"], bounds=T_BOUNDS, tol=T_TOL):
    """(alpha, kappa, T) on CAL features: every alpha / kappa (> 0) pair with its own scalar T (golden section, source-balanced NLL);
    the minimum NLL wins (ties: smaller alpha, then smaller kappa). -> chosen dict, grid"""
    V, codes, Ls = lap["V"], lap["codes"], lap["L"]
    ks, t = f["k"].long(), f["t"].float()
    K = int(ks.max())
    src = [s_ if s_ is not None else "unknown" for s_ in (f.get("source") or ["all"] * len(ks))]
    phi, mask = fe_phi(f["h"], ks, K, V, codes)
    ell = f["ell"][:, :K]
    eps = fe_eps(V.shape[1])
    grid = {}
    with torch.no_grad():
        for a in sorted(Ls):
            mu = torch.full((V.shape[1],), float(a), dtype=torch.float64)
            for kap in kappas:
                Le = math.sqrt(kap) * Ls[a]
                obj = lambda lt_, mu=mu, Le=Le: source_balanced_nll(fe_probs(ell, phi, mask, mu, Le, math.exp(lt_), eps), t[:, :K], src)  # noqa: E731
                x, fx = golden(obj, math.log(bounds[0]), math.log(bounds[1]), tol)
                grid[(a, kap)] = {"T": math.exp(x), "log_T": x, "nll_post": fx, "nll_pre": obj(0.0),
                                  "at_bound": abs(x - math.log(bounds[0])) < 1e-3 or abs(x - math.log(bounds[1])) < 1e-3}
    key = lambda k_: (round(grid[k_]["nll_post"], 12), k_[0], k_[1])  # noqa: E731
    best = min((k_ for k_ in grid if k_[1] > 0), key=key)            # (v3.8) the served head is Bayesian: kappa > 0
    ch = dict(grid[best], alpha=best[0], kappa=best[1], n=int(len(ks)), sources=len(set(map(str, src))), bounds=list(bounds), tol=tol)
    k0 = [k_ for k_ in grid if k_[1] == 0]
    if k0:
        b0 = min(k0, key=key)
        ch["mean_only"] = dict(grid[b0], alpha=b0[0], kappa=0.0, would_win=grid[b0]["nll_post"] < grid[best]["nll_post"])
    return ch, {f"{a:g}|{k:g}": v for (a, k), v in grid.items()}


def fe_head(lap, alpha, kappa):
    """the served fixedE parameters: V, codes, mu = alpha 1, L = sqrt(kappa) chol(S_alpha)"""
    F = int(lap["V"].shape[1])
    return {"V": lap["V"], "codes": lap["codes"], "mu": torch.full((F,), float(alpha), dtype=torch.float64),
            "L": math.sqrt(float(kappa)) * lap["L"][float(alpha)]}


def fe_predict(f, hd, T, K=None):
    """p [n, K] for features f under the fixedE head dict hd at temperature T"""
    ks = f["k"].long()
    K = int(K or ks.max())
    phi, mask = fe_phi(f["h"], ks, K, hd["V"], hd["codes"])
    with torch.no_grad():
        return fe_probs(f["ell"][:, :K], phi, mask, hd["mu"], hd["L"], T, fe_eps(hd["V"].shape[1]))


def export_fixede(d, hd, label_ids, T, extra=None):
    """the fixedE serving head (vbll_core FIXEDE schema): vbll_head.safetensors {V, codes, mu, L} (fp64) + vbll_head.json"""
    from safetensors.torch import save_file
    os.makedirs(d, exist_ok=True)
    save_file({k: hd[k].detach().cpu().double().contiguous() for k in ("V", "codes", "mu", "L")}, os.path.join(d, fh.HEAD_FILE))
    meta = {"schema": FIXEDE_SCHEMA, "mode": "fixedE", "norm": "none", "dim": int(hd["V"].shape[1]), "hidden": int(hd["V"].shape[0]),
            "heads": [{"rule": "fixedE", "alpha": float(hd["mu"][0]), "cov": "full", "prior": "laplace (centred Fisher, lam 1)"}],
            "label_ids": [int(x) for x in label_ids], "draws": {"seed": FIXEDE["seed"], "pairs": FIXEDE["pairs"], "antithetic": True},
            "samples": 2 * FIXEDE["pairs"], "draw_seed": FIXEDE["seed"], "temperature": float(T), "temperature_semantics": "mean_s softmax(z_s / T)", "refit": VERSION,
            "readout": "posterior_predictive", "letters": "A-Z a-z (52, one token each)",
            "native_logits": "fp32 h @ E[letters]^T with final_logit_softcapping (tied embedding)",
            "tensor_sha256": hashlib.sha256(open(os.path.join(d, fh.HEAD_FILE), "rb").read()).hexdigest(), **(extra or {})}
    ro_env = getattr(fh, "READOUT_ENV", None)
    if ro_env and os.environ.get(ro_env) and os.environ.get(ro_env) != "stock":
        meta["letter_readout"] = os.environ[ro_env]
    json.dump(meta, open(os.path.join(d, fh.HEAD_META), "w"), indent=1, sort_keys=True)
    return meta


# ------------------------------------------------------------------------------------------------------------ v3.9 L-loc
LLOC = {"noul_dim": 128, "lam_noul": 1e3, "alpha": 0.0, "beta_noul": 0.25, "pca_seed": 0, "pca_niter": 4, "pairs": 32, "seed": 0,
        "newton_tol": 1e-10, "newton_max": 100, "chunk": 4096}
LLOC_SCHEMA = "jev-lloc-head-v1"
OPT_HEAD = "\n\nOptions:\n"
OPT_TAIL = "\n\nAnswer with the letter of the single best option and nothing else."
_YES, _NO = r"Yes(?:: .*)?", r"No(?:: .*)?"


def yes_no_index(prompt, k):
    """(v3.9) THE Yes / No flag of the L-loc readout, from the RENDERED prompt only (no dataset metadata): a K = 2 question whose rendered
    options are exactly Yes and No ('Yes' or 'Yes: <definition>', 'No' or 'No: <definition>', either order) -> (yes position, no
    position); anything else -> None. The options block is the text between the LAST '\\n\\nOptions:\\n' and the LAST '\\n\\nAnswer with
    the letter of the single best option and nothing else.' (garas_core.PROMPT / PROMPT_CASE and the stage-2 renderer).
    release/garas_vbll/vbll_core.yes_no_index is the same code (test_sdv1 checks the text)."""
    if int(k) != 2:
        return None
    i, j = prompt.rfind(OPT_HEAD), prompt.rfind(OPT_TAIL)
    if i < 0 or j < i:
        return None
    block = prompt[i + len(OPT_HEAD):j]
    if re.fullmatch(rf"A\) {_YES}\nB\) {_NO}", block, flags=re.S):
        return (0, 1)
    if re.fullmatch(rf"A\) {_NO}\nB\) {_YES}", block, flags=re.S):
        return (1, 0)
    return None


def manifest_yes_no(path, f):
    """(v3.9) the Yes / No positions [n, 2] (long; -1 = not a Yes / No row) of the manifest's rendered prompts, in the features' order
    (decision_uid checked row by row), and how they compare with the manifest's 'kind' (reported only; never used). A manifest without
    rendered prompts -> (None, why)"""
    import gzip
    op = gzip.open if open(path, "rb").read(2) == b"\x1f\x8b" else open
    uids, ks = list(f["decision_uid"]), f["k"].long().tolist()
    yn, kinds, srcs, i = [], [], [], 0
    with op(path, "rt") as fh_:
        for line in fh_:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("manifest_header"):
                continue
            if i >= len(uids) or r["decision_uid"] != uids[i]:
                raise SystemExit(f"manifest_yes_no: manifest row {i} is {r['decision_uid']!r}, the features have "
                                 f"{uids[i] if i < len(uids) else 'nothing'!r} (the flags must come from the features' own prompts)")
            if not isinstance(r.get("prompt"), str):
                return None, {"available": False, "why": f"manifest row {i} has no rendered 'prompt'"}
            yn.append(yes_no_index(r["prompt"], ks[i]) or (-1, -1))
            kinds.append(r.get("kind") or r.get("qtype"))
            srcs.append(r.get("source_family") or r.get("source"))
            i += 1
    if i != len(uids):
        raise SystemExit(f"manifest_yes_no: the manifest has {i} rows, the features {len(uids)}")
    t = torch.tensor(yn, dtype=torch.long).reshape(-1, 2)
    fl = (t[:, 0] >= 0).tolist()
    extra = {}
    for b, k_, s_ in zip(fl, kinds, srcs):
        if b and k_ != "noul":
            extra[str(s_)] = extra.get(str(s_), 0) + 1
    st = {"n": len(fl), "yes_no_rows": sum(fl), "yes_first": int((t[:, 0] == 0).sum()), "no_first": int((t[:, 0] == 1).sum()),
          "kind_noul": sum(k_ == "noul" for k_ in kinds), "flagged_and_noul": sum(b and k_ == "noul" for b, k_ in zip(fl, kinds)),
          "flagged_not_noul": sum(b and k_ != "noul" for b, k_ in zip(fl, kinds)), "noul_not_flagged": sum((not b) and k_ == "noul" for b, k_ in zip(fl, kinds)),
          "flagged_not_noul_by_source": extra, "rule": "yes_no_index(rendered prompt, K) (the serving rule); the manifest kind is reported only",
          "available": True}
    return t, st


def lloc_hperp(h, VE):
    """rms(h - (h V_E) V_E^T) fp64 (the letter-orthogonal state, MACKAY-HEAD FastLatent01._h_perp)"""
    hd = h.double()
    hp = hd - (hd @ VE) @ VE.T
    return hp * torch.rsqrt(hp.pow(2).mean(-1, keepdim=True) + 1e-12)


def lloc_u(h, VE, Vp, chunk=LLOC["chunk"]):
    """u = lloc_hperp(h) V_perp [n, q] fp64 (NOT centred: the readout's u_noul)"""
    return torch.cat([lloc_hperp(h[s:s + chunk], VE) @ Vp for s in range(0, h.shape[0], chunk)]) if h.shape[0] else torch.zeros(0, Vp.shape[1], dtype=torch.float64)


def lloc_fit(f, yn, spec=LLOC):
    """(v3.9) the L-loc fit on HEAD-FIT features (MACKAY-HEAD tmp/mackay_head/lloc_fixed.py bafa3f07 / FastLatent01 e80a40ed):
    V_E = the right singular vectors of E_letters [D, 52]; V_perp = top-q PCA (torch.pca_lowrank, seed 0, niter 4, fp32) of the centred
    letter-orthogonal states of ALL head-fit rows; on the Yes / No rows (yes_no_index) the Laplace MAP of the logit-difference residual
    z_yes = ell_yes + u.mu / 2, z_no = ell_no - u.mu / 2 (unweighted, prior N(0, I / lam_noul); exact Newton) and its Laplace covariance
    S = (sum_b p_b (1 - p_b) u_b u_b^T + lam I)^-1 at the mode. -> tensors, info"""
    E = f["E_letters"].double()
    VE = torch.linalg.svd(E, full_matrices=False).Vh.T.contiguous()
    n, D = int(f["h"].shape[0]), int(f["h"].shape[1])
    hp = torch.cat([lloc_hperp(f["h"][s:s + spec["chunk"]], VE) for s in range(0, n, spec["chunk"])])
    mu_hp = hp.mean(0, keepdim=True)
    q = min(int(spec["noul_dim"]), n, D)
    Xc = (hp - mu_hp).float()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(spec["pca_seed"]))
        _, _, Vf = torch.pca_lowrank(Xc, q=q, center=False, niter=int(spec["pca_niter"]))
    del Xc
    Vp = Vf.double().contiguous()
    rows = (yn[:, 0] >= 0).nonzero().flatten()
    u = hp[rows] @ Vp
    del hp
    lam = float(spec["lam_noul"])
    ar = torch.arange(len(rows))
    ell, t = f["ell"].double()[rows], f["t"].double()[rows]
    iy, ino = yn[rows, 0], yn[rows, 1]
    ly, ln_ = ell[ar, iy], ell[ar, ino]
    ty, tn = t[ar, iy], t[ar, ino]
    s_ = (ty + tn).clamp_min(1e-12)
    ty, tn = ty / s_, tn / s_

    def parts(mu):
        m = 0.5 * (u @ mu)
        zy, zn = ly + m, ln_ - m
        lse = torch.logaddexp(zy, zn)
        py, pn = torch.exp(zy - lse), torch.exp(zn - lse)
        obj = float((lse - ty * zy - tn * zn).sum() + 0.5 * lam * (mu @ mu))
        g = u.T @ (0.5 * ((py - ty) - (pn - tn))) + lam * mu
        Hm = (u * (py * pn)[:, None]).T @ u + lam * torch.eye(q, dtype=torch.float64)
        return obj, g, 0.5 * (Hm + Hm.T)
    mu = torch.zeros(q, dtype=torch.float64)
    f0, g, Hm = parts(mu)
    it, converged = 0, float(g.abs().max()) < spec["newton_tol"]
    while not converged and it < int(spec["newton_max"]):
        step = torch.linalg.solve(Hm, g)
        a_ = 1.0
        while True:
            new = mu - a_ * step
            f1 = parts(new)[0]
            if f1 <= f0 - 1e-4 * a_ * float(g @ step) or a_ < 1e-10:
                break
            a_ *= 0.5
        mu = new
        f0, g, Hm = parts(mu)
        it += 1
        converged = float(g.abs().max()) < spec["newton_tol"]
    S = torch.linalg.inv(Hm)
    S = 0.5 * (S + S.T)
    L = torch.linalg.cholesky(S)
    info = {"n": n, "D": D, "noul_dim": q, "yes_no_rows": int(len(rows)), "lam_noul": lam, "newton_iters": it, "converged": bool(converged),
            "grad_inf": float(g.abs().max()), "objective": f0, "mu_norm": float(mu.norm()), "trace_S": float(S.trace()),
            "shift_sd_headfit": float((0.5 * float(spec["beta_noul"]) * (u @ mu)).std()) if len(rows) > 1 else 0.0,
            "pca": {"seed": spec["pca_seed"], "niter": spec["pca_niter"], "dtype": "float32", "rows": "all head-fit rows, centred"},
            "weights": "unweighted (MACKAY-HEAD lap_fit)", "spec": {k: spec[k] for k in ("noul_dim", "lam_noul", "alpha", "beta_noul")}}
    return {"V_E": VE, "V_perp": Vp, "mu_hp": mu_hp[0], "mu": mu, "S": S, "L": L, "beta": float(spec["beta_noul"])}, info


def lloc_prep(h, yn, hd):
    """the per-row parts of the readout that do not depend on T: the Yes / No rows, (beta / 2) u.mu [r] and (beta / 2) u L [r, q]"""
    rows = (yn[:, 0] >= 0).nonzero().flatten()
    u = lloc_u(h[rows], hd["V_E"], hd["V_perp"])
    b2 = 0.5 * float(hd["beta"])
    return rows, b2 * (u @ hd["mu"]), b2 * (u @ hd["L"])


def lloc_probs(ell, ks, yn, prep, T, eps=None):
    """(v3.9) THE L-loc readout p [n, K] fp64: z = ell; on the Yes / No rows z_yes += m_s, z_no -= m_s with m_s = (beta / 2) u.(mu + L eps_s);
    p = mean_s softmax(z_s / T) (eps None: the MAP readout, m = (beta / 2) u.mu); other rows softmax(ell / T).
    vbll_core.LlocHead.predict runs the same operations in the same order."""
    K = ell.shape[1]
    mask = torch.arange(K, device=ell.device)[None, :] < ks[:, None]
    e = ell.double()
    p = torch.softmax((e / float(T)).masked_fill(~mask, float("-inf")), -1)
    rows, m, uL = prep
    if len(rows):
        ms = m[None] if eps is None else m[None] + (uL @ eps.to(uL.device).T).T            # [S, r]
        iy, ino = yn[rows, 0].to(e.device), yn[rows, 1].to(e.device)
        ar = torch.arange(len(rows), device=e.device)
        d = (e[rows, iy] - e[rows, ino])[None] + 2.0 * ms                                   # z_yes - z_no per draw
        py, pn = torch.sigmoid(d / float(T)).mean(0), torch.sigmoid(-d / float(T)).mean(0)
        pr = torch.zeros(len(rows), K, dtype=torch.float64, device=e.device)
        pr[ar, iy], pr[ar, ino] = py, pn
        p[rows] = pr
    return p.masked_fill(~mask, 0.0)


def lloc_eps(F):
    """the served draws: LLOC pairs (32) seed-0 antithetic normals [64, F] (fe_eps)"""
    return fe_eps(F, LLOC["pairs"], LLOC["seed"])


def lloc_fit_T(f, yn, hd, eps, bounds=T_BOUNDS, tol=T_TOL):
    """the scalar T of one L-loc readout on CAL features (golden section over log T, source-balanced NLL, as fit_T)"""
    ks, t = f["k"].long(), f["t"].float()
    K = int(ks.max())
    src = [s_ if s_ is not None else "unknown" for s_ in (f.get("source") or ["all"] * len(ks))]
    prep = lloc_prep(f["h"], yn, hd)
    ell = f["ell"][:, :K]
    with torch.no_grad():
        obj = lambda lt_: source_balanced_nll(lloc_probs(ell, ks, yn, prep, math.exp(lt_), eps), t[:, :K], src)  # noqa: E731
        x, fx = golden(obj, math.log(bounds[0]), math.log(bounds[1]), tol)
        pre = obj(0.0)
    return {"T": math.exp(x), "log_T": x, "nll_pre": pre, "nll_post": fx, "bounds": list(bounds), "tol": tol, "n": int(len(ks)),
            "sources": len(set(map(str, src))), "yes_no_rows": int(len(prep[0])), "draws": 0 if eps is None else int(eps.shape[0]),
            "at_bound": abs(x - math.log(bounds[0])) < 1e-3 or abs(x - math.log(bounds[1])) < 1e-3}


def lloc_predict(f, yn, hd, T, K=None, eps=None):
    ks = f["k"].long()
    K = int(K or ks.max())
    with torch.no_grad():
        return lloc_probs(f["ell"][:, :K], ks, yn, lloc_prep(f["h"], yn, hd), T, eps)


def export_lloc(d, hd, label_ids, T, extra=None):
    """the L-loc serving head (vbll_core LLOC schema): vbll_head.safetensors {V_E, V_perp, mu, L} (fp64) + vbll_head.json"""
    from safetensors.torch import save_file
    os.makedirs(d, exist_ok=True)
    save_file({k: hd[k].detach().cpu().double().contiguous() for k in ("V_E", "V_perp", "mu", "L")}, os.path.join(d, fh.HEAD_FILE))
    meta = {"schema": LLOC_SCHEMA, "mode": "lloc", "dim": int(hd["V_perp"].shape[1]), "hidden": int(hd["V_perp"].shape[0]),
            "letter_dim": int(hd["V_E"].shape[1]), "beta": float(hd["beta"]), "alpha": LLOC["alpha"], "lam_noul": LLOC["lam_noul"],
            "heads": [{"rule": "lloc", "cov": "full", "prior": f"N(0, I / {LLOC['lam_noul']:g}) on the Yes / No residual; Laplace at the MAP"}],
            "label_ids": [int(x) for x in label_ids], "draws": {"seed": LLOC["seed"], "pairs": LLOC["pairs"], "antithetic": True},
            "samples": 2 * LLOC["pairs"], "draw_seed": LLOC["seed"], "temperature": float(T), "temperature_semantics": "mean_s softmax(z_s / T)",
            "refit": VERSION, "readout": "posterior_predictive", "letters": "A-Z a-z (52, one token each)",
            "native_logits": "fp32 h @ E[letters]^T with final_logit_softcapping (tied embedding)",
            "yes_no_flag": {"from": "the rendered prompt only (vbll_core.yes_no_index; sdv1_head.yes_no_index is the same code)", "K": 2,
                            "options_block": [OPT_HEAD, OPT_TAIL], "yes": _YES, "no": _NO, "either_order": True},
            "tensor_sha256": hashlib.sha256(open(os.path.join(d, fh.HEAD_FILE), "rb").read()).hexdigest(), **(extra or {})}
    ro_env = getattr(fh, "READOUT_ENV", None)
    if ro_env and os.environ.get(ro_env) and os.environ.get(ro_env) != "stock":
        meta["letter_readout"] = os.environ[ro_env]
    json.dump(meta, open(os.path.join(d, fh.HEAD_META), "w"), indent=1, sort_keys=True)
    return meta


# ------------------------------------------------------------------------------------------------------------ v4.0 Latent01-post
# The constants in MACKAY-HEAD's predict2 parametrisation (tmp/mackay_head/latent01_freeze.py): nd = noul_dim, lam = lam_noul, alpha (softcap
# blend), beta (noul shift), sig (sigma_epi), tau (per-kind T), infl (noul variance inflation), kern / lam_uni / sig_uni (score interval),
# dnorm ('ratio' = d_epi / dbar_H, 'const' = 1). 'frozen' False: rule l01post refuses and no l01post candidate is fitted.
L01 = {"frozen": True, "source": "MACKAY-HEAD latent01_freeze.py 673be743, C-selected (pooled C 0.589917; tmp/mackay_head/sd12/latent01_frozen)",
       "nd": 512, "lam": 1e3, "alpha": 0.0, "beta": 0.25, "sig": 0.0, "tau": 0.3, "infl": 10.0, "kern": 0.2, "lam_uni": 0.05, "sig_uni": 0.2,
       "dnorm": "ratio"}
# (coordinator 10-15) the BOARD variant l01post_di: score rows are choice rows everywhere (fit, T, scoring; the Decision Index kit has only
# choice / noul), so its per-kind s / tau are choice / noul only; its kern / lam_uni / sig_uni act only on the optional ordinal path (a
# request question with ordinal: true, served by the engine only when constructed with ordinal=True; never on the board)
L01_DI = {"frozen": True, "source": "MACKAY-HEAD latent01_freeze.py e5d3e5be MK_DI=1, C-selected (pooled C 0.590183; tmp/mackay_head/sd12/"
                                     "latent01_frozen_di); kern / lam_uni / sig_uni = the full freeze's (they act only on the optional ordinal path)",
          "nd": 512, "lam": 1e3, "alpha": 0.0, "beta": 0.25, "sig": 0.0, "tau": 0.3, "infl": 10.0, "kern": 0.2, "lam_uni": 0.05, "sig_uni": 0.2,
          "dnorm": "ratio"}
L01_VARIANTS = {"l01post": ("full", "l01_fit.pt"), "l01post_di": ("di", "l01di_fit.pt")}       # candidate -> (variant, refit file)


def l01_spec(name):
    return L01_DI if L01_VARIANTS[name][0] == "di" else L01
L01_FIT = {"pca_seed": 0, "pca_niter": 4, "skind_bounds": (0.5, 4.0), "skind_tol": 1e-4, "gh_nodes": 16, "chunk": 4096}
L01_SCHEMA = "jev-l01post-head-v1"
KIND_CODE = {"choice": 0, "noul": 1, "score": 2}
_GHX, _GHW = __import__("numpy").polynomial.hermite.hermgauss(L01_FIT["gh_nodes"])
GH_Z = torch.from_numpy(_GHX * math.sqrt(2.0)).to(torch.float64)
GH_W = torch.from_numpy(_GHW / math.sqrt(math.pi)).to(torch.float64)


def di_kinds(kinds):
    """(v4.0) the board variant's kinds: score -> choice"""
    k = kinds.clone()
    k[k == KIND_CODE["score"]] = KIND_CODE["choice"]
    return k


def manifest_kind_codes(path, f):
    """(v4.0) per feature row the question kind code (choice 0 / noul 1 / score 2; anything else = choice) from the manifest's 'kind'
    (or 'qtype'), in the features' order (decision_uid checked). The serving engine passes the REQUEST's question type."""
    import gzip
    op = gzip.open if open(path, "rb").read(2) == b"\x1f\x8b" else open
    uids, out, i = list(f["decision_uid"]), [], 0
    with op(path, "rt") as fh_:
        for line in fh_:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("manifest_header"):
                continue
            if i >= len(uids) or r["decision_uid"] != uids[i]:
                raise SystemExit(f"manifest_kind_codes: manifest row {i} is {r['decision_uid']!r}, the features differ")
            out.append(KIND_CODE.get(r.get("kind") or r.get("qtype"), 0))
            i += 1
    if i != len(uids):
        raise SystemExit(f"manifest_kind_codes: the manifest has {i} rows, the features {len(uids)}")
    return torch.tensor(out, dtype=torch.long)


def l01_ovr(mu_scaled, sigma_01, mask):
    """the reference ovr_scurve_integral (run_latent01_all8.py e80a40ed), same operations: -> p [n, K], mi [n]"""
    m = mask
    z = mu_scaled.masked_fill(~m, -1e9)
    lse_all = torch.logsumexp(z, dim=-1, keepdim=True)
    p0 = torch.softmax(z, dim=-1).masked_fill(~m, 0.0)
    margin = (z - lse_all) - torch.log1p(-p0.clamp_max(1.0 - 1e-12))
    kb = m.double().sum(dim=-1, keepdim=True).clamp_min(2.0)
    sig_eff = sigma_01 * (2.0 / kb).sqrt() * m.double()
    gz, gw = GH_Z.to(z.device), GH_W.to(z.device)
    arg = margin[None, :, :] + sig_eff[None, :, :] * gz[:, None, None]
    f_u = torch.sigmoid(arg).masked_fill(~m[None, :, :], 0.0)
    I_k = torch.einsum("q,qbk->bk", gw, f_u).masked_fill(~m, 0.0)
    p = I_k / I_k.sum(dim=-1, keepdim=True).clamp_min(1e-15)

    def bern(q):
        q_c = q.clamp(1e-15, 1.0 - 1e-15)
        return -(q_c * q_c.log() + (1.0 - q_c) * (1.0 - q_c).log())
    mi_k = (bern(I_k) - torch.einsum("q,qbk->bk", gw, bern(f_u))).clamp_min(0.0) * m.double()
    return p, mi_k.sum(dim=-1) / kb.squeeze(-1)


def l01_score(p_init, k_vec, mask, sig_kern, lam_uni, sig_uni):
    """the reference score_interval_integral (e80a40ed), same operations"""
    if sig_kern <= 0.0 and lam_uni <= 0.0:
        return p_init
    Kmax = p_init.shape[1]
    K_b = k_vec.to(p_init.dtype)[:, None]
    grid = torch.arange(Kmax, dtype=p_init.dtype, device=p_init.device)[None, :]
    u_j = (grid + 0.5) / K_b
    c_lo = grid / K_b
    c_hi = (grid + 1.0) / K_b
    p_out = p_init
    if sig_kern > 0.0:
        bw = (sig_kern / K_b)[:, :, None]
        z_hi = (c_hi[:, None, :] - u_j[:, :, None]) / bw
        z_lo = (c_lo[:, None, :] - u_j[:, :, None]) / bw
        cdf_diff = 0.5 * (torch.erf(z_hi / math.sqrt(2.0)) - torch.erf(z_lo / math.sqrt(2.0)))
        cdf_diff = cdf_diff.masked_fill(~mask[:, None, :], 0.0)
        cdf_diff = cdf_diff / cdf_diff.sum(dim=-1, keepdim=True).clamp_min(1e-15)
        p_out = torch.einsum("bj,bjk->bk", p_out, cdf_diff)
    if lam_uni > 0.0:
        mu_s = (p_init * u_j).sum(dim=-1, keepdim=True)
        var_s = (p_init * (u_j - mu_s).pow(2)).sum(dim=-1, keepdim=True)
        sd_s = (var_s + (sig_uni / K_b).pow(2)).sqrt()
        zhi = (c_hi - mu_s) / sd_s
        zlo = (c_lo - mu_s) / sd_s
        p_uni = (0.5 * (torch.erf(zhi / math.sqrt(2.0)) - torch.erf(zlo / math.sqrt(2.0)))).masked_fill(~mask, 0.0)
        p_uni = p_uni / p_uni.sum(dim=-1, keepdim=True).clamp_min(1e-15)
        p_out = (1.0 - lam_uni) * p_out + lam_uni * p_uni
    return p_out


def l01_extract(h, ell, ks, kinds, st, chunk=L01_FIT["chunk"]):
    """the reference FastLatent01.extract (e80a40ed) on the stored state st: ell_c, dz_c, mask, res, dm_noul, var_noul, row_rms (fp64)"""
    VE, Vp, mu_hp, E = st["V_E"], st["V_perp"], st["mu_hp"].reshape(1, -1), st["E"]
    dv = h.device                                    # (v4.1) every buffer on the states' device (CUDA serving)
    kinds = kinds.to(dv)
    res_l, un_l = [], []
    for s0 in range(0, h.shape[0], chunk):
        hp_raw = lloc_hperp(h[s0:s0 + chunk], VE)
        hp = hp_raw - mu_hp
        u = hp @ Vp
        res_l.append((hp - u @ Vp.T).pow(2).sum(-1).sqrt())
        un_l.append(hp_raw @ Vp)
    res = torch.cat(res_l) if res_l else torch.zeros(0, dtype=torch.float64, device=dv)
    u_noul = torch.cat(un_l) if un_l else torch.zeros(0, Vp.shape[1], dtype=torch.float64, device=dv)
    K = ell.shape[1]
    mask = torch.arange(K, device=ell.device)[None] < ks[:, None]
    m = mask.double()
    e = ell.double()
    z = (h.double() @ E.T)[:, :K]
    if K > z.shape[1]:                                   # options past the reference basis rows: dz = 0 there
        z = torch.cat([z, e[:, z.shape[1]:K]], 1)
    ell_c = (e - (e * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    z_c = (z - (z * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    n = int(ks.shape[0])
    idx_n = (kinds == KIND_CODE["noul"]).nonzero().flatten()
    dm_noul = torch.zeros(n, dtype=torch.float64, device=dv)
    var_noul = torch.zeros(n, dtype=torch.float64, device=dv)
    if len(idx_n) > 0:
        un = u_noul[idx_n]
        dm_noul[idx_n] = (un @ st["mu"]) / 2.0
        var_noul[idx_n] = ((un @ st["S"]) * un).sum(-1).clamp_min(0.0) / 4.0
    row_rms = ((ell_c.pow(2) * m).sum(1) / m.sum(1).clamp_min(1.0) + 1.0).sqrt()
    return {"ell_c": ell_c, "dz_c": z_c - ell_c, "mask": mask, "res": res, "dm_noul": dm_noul, "var_noul": var_noul, "row_rms": row_rms,
            "kinds": kinds, "k": ks}


def l01_probs(d, st, T, c):
    """(v4.0) THE Latent01-post readout = MACKAY-HEAD predict2 (the reference predict with its constants exposed), same operations:
    -> p [n, K] fp64, mi [n] (the S-curve mutual information; zeros on the plain-softmax path)"""
    m = d["mask"]
    kinds = d["kinds"]
    mu = (d["ell_c"] + c["alpha"] * d["dz_c"]).clone()
    is_noul = kinds == KIND_CODE["noul"]
    if c["beta"] > 0.0 and bool(is_noul.any()):
        dm = c["beta"] * d["dm_noul"][is_noul]
        mu[is_noul, 0] += dm
        mu[is_noul, 1] -= dm
    if c["tau"] > 0.0:
        r_scale = torch.ones(len(mu), dtype=torch.float64, device=mu.device)
        for kd in ("noul", "choice", "score"):
            r_scale[kinds == KIND_CODE[kd]] = 1.0 + c["tau"] * (st["rel_s"][kd] - 1.0)
        T_row = T * r_scale[:, None]
    else:
        T_row = T
    score_on = c["kern"] > 0.0 or c["lam_uni"] > 0.0
    if c["sig"] <= 0.0 and not score_on and c["tau"] <= 0.0 and (c["infl"] <= 0.0 or not bool(is_noul.any())):   # (v4.1) noul variance never skipped
        return torch.softmax((mu / T).masked_fill(~m, -1e9), -1).masked_fill(~m, 0.0), torch.zeros(len(mu), dtype=torch.float64, device=mu.device)
    mu_scaled = mu / T_row
    rr = d["res"] / st["res_mean"] if c["dnorm"] == "ratio" else torch.ones_like(d["res"])
    sd_row = c["sig"] * rr * d["row_rms"]
    sigma = sd_row[:, None].expand_as(mu).clone()
    if bool(is_noul.any()):
        sigma[is_noul, :2] = (sigma[is_noul, :2].pow(2) + c["infl"] * d["var_noul"][is_noul, None]).sqrt()
    p, mi = l01_ovr(mu_scaled, (sigma / T_row) * m.double(), m)
    is_score = kinds == KIND_CODE["score"]
    if score_on and bool(is_score.any()):
        idx = is_score.nonzero().flatten()
        p[idx] = l01_score(p[idx], d["k"][idx], m[idx], c["kern"], c["lam_uni"], c["sig_uni"])
    return p, mi


def _noul_newton(u, ly, ln_, ty, tn, lam, tol, itmax):
    """the unweighted Laplace MAP / covariance of the Yes / No logit-difference residual (z_yes = ly + u.mu / 2, z_no = ln - u.mu / 2,
    prior N(0, I / lam)) by damped exact Newton (lap_fit's objective and step rule) -> mu, S, info"""
    q = u.shape[1]

    def parts(mu):
        mm = 0.5 * (u @ mu)
        zy, zn = ly + mm, ln_ - mm
        lse = torch.logaddexp(zy, zn)
        py, pn = torch.exp(zy - lse), torch.exp(zn - lse)
        obj = float((lse - ty * zy - tn * zn).sum() + 0.5 * lam * (mu @ mu))
        g = u.T @ (0.5 * ((py - ty) - (pn - tn))) + lam * mu
        Hm = (u * (py * pn)[:, None]).T @ u + lam * torch.eye(q, dtype=torch.float64)
        return obj, g, 0.5 * (Hm + Hm.T)
    mu = torch.zeros(q, dtype=torch.float64)
    f0, g, Hm = parts(mu)
    it, conv = 0, float(g.abs().max()) < tol
    while not conv and it < itmax:
        step = torch.linalg.solve(Hm, g)
        a_ = 1.0
        while True:
            new = mu - a_ * step
            f1 = parts(new)[0]
            if f1 <= f0 - 1e-4 * a_ * float(g @ step) or a_ < 1e-10:
                break
            a_ *= 0.5
        mu = new
        f0, g, Hm = parts(mu)
        it += 1
        conv = float(g.abs().max()) < tol
    S = torch.linalg.inv(Hm)
    return mu, 0.5 * (S + S.T), {"newton_iters": it, "converged": bool(conv), "grad_inf": float(g.abs().max()), "objective": f0}


def l01_fit(f, kinds, c=None, fitc=L01_FIT, variant="full"):
    """(v4.0) the Latent01 state on HEAD-FIT features = the reference FastLatent01.__init__ (e80a40ed) with noul_dim c['nd'] and lam_noul
    c['lam']: V_E (E's right singular vectors), mu_hp + V_perp (top-nd PCA, pca_lowrank seed 0 niter 4, fp32, of the centred letter-
    orthogonal states of ALL head-fit rows), res_mean (dbar_H), the noul Laplace (mu, S) on the kind-noul rows (positions 0 / 1), and the
    per-kind spread s_kind (golden section on each kind's family-balanced NLL of softmax(ell_c / s), log s in [log 0.5, log 4], tol 1e-4)
    -> state, info"""
    c = c or (L01_DI if variant == "di" else L01)
    if variant == "di":                              # the board variant: score rows are choice rows (also for s_kind)
        kinds = di_kinds(kinds)
    E = f["E_letters"].double()
    VE = torch.linalg.svd(E, full_matrices=False).Vh.T.contiguous()
    n, D = int(f["h"].shape[0]), int(f["h"].shape[1])
    ch = fitc["chunk"]
    hp = torch.cat([lloc_hperp(f["h"][s:s + ch], VE) for s in range(0, n, ch)])
    mu_hp = hp.mean(0, keepdim=True)
    q = min(int(c["nd"]), n, D)
    Xc = (hp - mu_hp).float()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(fitc["pca_seed"]))
        _, _, Vf = torch.pca_lowrank(Xc, q=q, center=False, niter=int(fitc["pca_niter"]))
    del Xc
    Vp = Vf.double().contiguous()
    res = []
    for s in range(0, n, ch):
        hc = hp[s:s + ch] - mu_hp
        uc = hc @ Vp
        res.append((hc - uc @ Vp.T).pow(2).sum(-1).sqrt())
    res_mean = float(torch.cat(res).mean())
    del hp
    st = {"E": E, "V_E": VE, "V_perp": Vp, "mu_hp": mu_hp[0], "res_mean": res_mean, "mu": torch.zeros(q, dtype=torch.float64),
          "S": torch.zeros(q, q, dtype=torch.float64)}
    ks = f["k"].long()
    K = int(ks.max())
    d = l01_extract(f["h"], f["ell"][:, :K], ks, kinds, st)
    tH = f["t"].double()[:, :K]
    tH = tH / tH.sum(1, keepdim=True).clamp_min(1e-12)
    idx_n = (kinds == KIND_CODE["noul"]).nonzero().flatten()
    lam = float(c["lam"])
    if len(idx_n):
        un = torch.cat([lloc_hperp(f["h"][idx_n[s:s + ch]], VE) @ Vp for s in range(0, len(idx_n), ch)])
        o = d["ell_c"][idx_n, :2]
        tt = tH[idx_n, :2]
        tt = tt / tt.sum(1, keepdim=True).clamp_min(1e-12)
        mu, S, ninfo = _noul_newton(un, o[:, 0], o[:, 1], tt[:, 0], tt[:, 1], lam, 1e-10, 100)
    else:
        mu, S, ninfo = torch.zeros(q, dtype=torch.float64), torch.eye(q, dtype=torch.float64) / lam, {"newton_iters": 0, "converged": True}
    st["mu"], st["S"] = mu, S
    src = list(f.get("source") or ["all"] * n)
    s_kind, sk_info = {}, {}
    for kd in ("noul", "choice", "score"):
        km = (kinds == KIND_CODE[kd]).nonzero().flatten()
        if len(km) == 0:
            s_kind[kd], sk_info[kd] = 1.0, "no rows: 1"
            continue
        sub_m, sub_ell, sub_t = d["mask"][km], d["ell_c"][km], tH[km]
        fams = [str(src[i]) for i in km.tolist()]
        fu = sorted(set(fams))
        fi = torch.tensor([fu.index(x) for x in fams])

        def fam_loss(lt, sub_m=sub_m, sub_ell=sub_ell, sub_t=sub_t, fi=fi, nf=len(fu)):
            T_k = math.exp(lt)
            p_k = torch.softmax((sub_ell / T_k).masked_fill(~sub_m, -1e9), -1).masked_fill(~sub_m, 0.0)
            nl = -(sub_t * p_k.clamp_min(1e-12).log()).sum(-1)
            s_ = torch.zeros(nf, dtype=torch.float64).index_add_(0, fi, nl)
            c_ = torch.zeros(nf, dtype=torch.float64).index_add_(0, fi, torch.ones_like(nl))
            return float((s_[c_ > 0] / c_[c_ > 0]).mean())
        x_k, _ = golden(fam_loss, math.log(fitc["skind_bounds"][0]), math.log(fitc["skind_bounds"][1]), fitc["skind_tol"])
        s_kind[kd], sk_info[kd] = math.exp(x_k), int(len(km))
    st["rel_s"] = {k: s_kind[k] / s_kind["choice"] for k in ("noul", "choice", "score")}
    if variant == "di":                              # an ordinal-path score row is tempered as a choice row
        st["rel_s"]["score"] = 1.0
    st["s_kind"], st["variant"] = s_kind, variant
    info = {"variant": variant, "n": n, "D": D, "nd": q, "lam": lam, "res_mean": res_mean, "noul_rows": int(len(idx_n)), "noul": ninfo, "s_kind": s_kind,
            "rel_s": st["rel_s"], "s_kind_rows": sk_info, "kinds": {k: int((kinds == v).sum()) for k, v in KIND_CODE.items()},
            "spec": {k: c[k] for k in ("nd", "lam", "alpha", "beta", "sig", "tau", "infl", "kern", "lam_uni", "sig_uni", "dnorm", "source")}}
    return st, info


def l01_fit_T(f, kinds, st, c=None, bounds=T_BOUNDS, tol=T_TOL):
    """the scalar T of the Latent01-post readout on CAL features (golden section over log T, source-balanced NLL); the di variant maps
    score -> choice"""
    c = c or (L01_DI if st.get("variant") == "di" else L01)
    if st.get("variant") == "di":
        kinds = di_kinds(kinds)
    ks, t = f["k"].long(), f["t"].float()
    K = int(ks.max())
    src = [s_ if s_ is not None else "unknown" for s_ in (f.get("source") or ["all"] * len(ks))]
    d = l01_extract(f["h"], f["ell"][:, :K], ks, kinds, st)
    with torch.no_grad():
        obj = lambda lt_: source_balanced_nll(l01_probs(d, st, math.exp(lt_), c)[0], t[:, :K], src)  # noqa: E731
        x, fx = golden(obj, math.log(bounds[0]), math.log(bounds[1]), tol)
        pre = obj(0.0)
    return {"T": math.exp(x), "log_T": x, "nll_pre": pre, "nll_post": fx, "bounds": list(bounds), "tol": tol, "n": int(len(ks)),
            "sources": len(set(map(str, src))), "kinds": {k: int((kinds == v).sum()) for k, v in KIND_CODE.items()},
            "at_bound": abs(x - math.log(bounds[0])) < 1e-3 or abs(x - math.log(bounds[1])) < 1e-3}


def l01_predict(f, kinds, st, T, K=None, c=None):
    """p for features f (manifest kinds; the di variant maps score -> choice, as the board serves them)"""
    if st.get("variant") == "di":
        kinds = di_kinds(kinds)
    c = c or (L01_DI if st.get("variant") == "di" else L01)
    ks = f["k"].long()
    K = int(K or ks.max())
    with torch.no_grad():
        return l01_probs(l01_extract(f["h"], f["ell"][:, :K], ks, kinds, st), st, T, c)[0]


def export_l01(d, st, label_ids, T, c=None, extra=None):
    """the Latent01-post serving head (vbll_core L01 schema): vbll_head.safetensors {E, V_E, V_perp, mu_hp, mu, S} (fp64) + vbll_head.json"""
    from safetensors.torch import save_file
    c = c if c is not None else (L01_DI if st.get("variant") == "di" else L01)
    os.makedirs(d, exist_ok=True)
    save_file({k: st[k].detach().cpu().double().contiguous() for k in ("E", "V_E", "V_perp", "mu_hp", "mu", "S")}, os.path.join(d, fh.HEAD_FILE))
    meta = {"schema": L01_SCHEMA, "mode": "l01post_di" if st.get("variant") == "di" else "l01post", "variant": st.get("variant", "full"), "dim": int(st["V_perp"].shape[1]), "hidden": int(st["V_perp"].shape[0]),
            "constants": {k: c[k] for k in ("nd", "lam", "alpha", "beta", "sig", "tau", "infl", "kern", "lam_uni", "sig_uni", "dnorm")},
            "constants_source": c.get("source"), "res_mean": float(st["res_mean"]), "rel_s": {k: float(v) for k, v in st["rel_s"].items()},
            "gh_nodes": L01_FIT["gh_nodes"], "heads": [{"rule": "l01post", "cov": "laplace (noul)", "prior": f"N(0, I / {float(c['lam']):g}) on the noul residual"}],
            "label_ids": [int(x) for x in label_ids], "temperature": float(T), "samples": 0, "draw_seed": 0,
            "temperature_semantics": "the Latent01-post readout at T (per-kind T_row = T (1 + tau (rel_s[kind] - 1)))", "refit": VERSION,
            "readout": "latent01_post: [0, 1] one-vs-rest S-curve sigma integral (16-node Gauss-Hermite) + score-interval integral",
            "letters": "A-Z a-z (52, one token each)", "native_logits": "fp32 h @ E[letters]^T with final_logit_softcapping (tied embedding)",
            "kinds": {"codes": KIND_CODE, "noul": "positions 0 / 1 = yes / no (the request's noul order)",
                      "from": "training: the manifest's kind (variant di: score -> choice); serving: the request's question type (choice / noul); "
                              "score only on the engine's optional ordinal path (ordinal=True and q['ordinal'] true; never on the board)"},
            "tensor_sha256": hashlib.sha256(open(os.path.join(d, fh.HEAD_FILE), "rb").read()).hexdigest(), **(extra or {})}
    ro_env = getattr(fh, "READOUT_ENV", None)
    if ro_env and os.environ.get(ro_env) and os.environ.get(ro_env) != "stock":
        meta["letter_readout"] = os.environ[ro_env]
    json.dump(meta, open(os.path.join(d, fh.HEAD_META), "w"), indent=1, sort_keys=True)
    return meta


BETA_GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def mean_scaled(lt, E_letters, src, beta):
    """(v3.4) a copy of the stored head `src` with its posterior mean mu scaled by beta (S / P / tau unchanged)"""
    hd = make_head(lt, E_letters, "vbll", src.meta()["dim"], src.meta()["seed"])
    hd.load_state_dict(src.state_dict())
    with torch.no_grad():
        hd.mu.mul_(float(beta))
    hd.jev_kind = "vbll"
    return hd


def _phi_all(head, f):
    """the frozen feature map on every head-fit row, in chunks of rows sorted by K (each chunk cut to its own largest K):
    [(phi [n, k, F] fp64, base tau * ell [n, k] fp64, mask [n, k], t [n, k] fp64, w [n] fp64)]"""
    h, ell, ks = f["h"].float(), f["ell"].float(), f["k"].long()
    order = torch.argsort(ks, stable=True)
    tau = float(head.log_tau.exp()[0])
    chunks = []
    with torch.no_grad():
        for i in range(0, len(ks), FINAL["chunk"]):
            ix = order[i:i + FINAL["chunk"]]
            kc = int(ks[ix].max())
            _, _, m, ph = head.moments(h[ix], ell[ix, :kc], ks[ix], want_phi=True)
            chunks.append((ph[0].double(), tau * ell[ix, :kc].double(), m, f["t"][ix, :kc].double(), f["w"][ix].double()))
    return chunks


def _objective(mu, chunks, alpha, want_grad=True):
    """J = sum_i w_i CE_i + alpha / 2 |mu|^2 (+ gradient and the exact Hessian of softmax CE: sum_i w_i phi^T (diag p - p p^T) phi)"""
    F = mu.shape[0]
    J = 0.5 * alpha * float(mu @ mu)
    g = alpha * mu.clone() if want_grad else None
    H = alpha * torch.eye(F, dtype=torch.float64) if want_grad else None
    for ph, b, m, tt, ww in chunks:
        z = (b + torch.einsum("nkf,f->nk", ph, mu)).masked_fill(~m, float("-inf"))
        lp = torch.log_softmax(z, -1).masked_fill(~m, 0.0)
        J += float(-(ww * (tt * lp).sum(-1)).sum())
        if want_grad:
            p = lp.exp().masked_fill(~m, 0.0)
            g += torch.einsum("n,nkf,nk->f", ww, ph, p - tt)
            v = torch.einsum("nkf,nk->nf", ph, p)
            A = ph * p.sqrt()[..., None]
            H += torch.einsum("n,nkf,nke->fe", ww, A, A) - torch.einsum("n,nf,ne->fe", ww, v, v)
    return J, g, H


def newton(mu0, chunks, alpha):
    """damped Newton to the MAP mean -> (mu, H at mu (incl. alpha I), iterations, final Newton decrement, J)"""
    mu = mu0.clone()
    J, g, H = _objective(mu, chunks, alpha)
    it, dec = 0, float("inf")
    for it in range(1, FINAL["newton_max"] + 1):
        d = torch.linalg.solve(H, g)
        dec = float(g @ d)
        if dec / 2 <= FINAL["newton_tol"]:
            break
        s_ = 1.0
        while True:
            J2 = _objective(mu - s_ * d, chunks, alpha, want_grad=False)[0]
            if J2 <= J - 0.25 * s_ * dec or s_ < 1e-8:
                break
            s_ *= 0.5
        mu = mu - s_ * d
        J, g, H = _objective(mu, chunks, alpha)
    return mu, H, it, dec, J


def finalize(head, f, alpha_mode):
    """one v3 candidate: the sdh feature map / tau frozen, mu by Newton, S = H^-1 (isotropic prior, no ARD). alpha_mode 'evidence'
    (MacKay fixed point, stable form) or a float -> (candidate head or None, info). info['available'] is False when the Newton solve
    or the evidence iteration did not converge (the loop cap is not convergence) or a step failed."""
    import copy
    if head.L_off is None:
        raise SystemExit("finalize: the head spec must carry a full covariance")
    info = {"alpha_mode": alpha_mode, "available": False}
    try:
        chunks = _phi_all(head, f)
        F = chunks[0][0].shape[-1]
        mu = head.mu.detach()[0].double()
        hist, conv_alpha, limit = [], True, None
        lo, hi = FINAL["alpha_lo"], FINAL["alpha_hi"]
        if alpha_mode == "evidence":
            alpha, conv_alpha = 1.0, False
            for _ in range(FINAL["alpha_max"]):
                mu, H, it, dec, J = newton(mu, chunks, alpha)
                lam = torch.linalg.eigvalsh(H - alpha * torch.eye(F, dtype=torch.float64)).clamp_min(0.0)
                gamma = float((lam / (lam + alpha)).sum())         # effective number of well-determined directions, in [0, F]
                m2 = float(mu @ mu)
                hist.append({"alpha": alpha, "gamma": gamma, "mu2": m2, "newton_iters": it, "J": J})
                if m2 <= 1e-12 * F or gamma <= 1e-12:              # vanishing residual: the evidence optimum is alpha -> infinity
                    new, limit = hi, "vanishing_residual"
                else:
                    new = min(max(gamma / m2, lo), hi)
                if abs(math.log(new) - math.log(alpha)) < FINAL["alpha_tol"] or (limit and alpha == hi):
                    alpha, conv_alpha = new, True
                    break
                alpha = new
            info["alpha_iterations"] = len(hist)
        else:
            alpha = float(alpha_mode)
        mu, H, it, dec, J = newton(mu, chunks, alpha)
        conv_newton = dec / 2 <= FINAL["newton_tol"] * 10
        S = torch.linalg.inv(H)
        S = 0.5 * (S + S.T)
        L = torch.linalg.cholesky(S)
        logdetH = float(torch.linalg.slogdet(H)[1])
        lam = torch.linalg.eigvalsh(H - alpha * torch.eye(F, dtype=torch.float64)).clamp_min(0.0)
        cand = copy.deepcopy(head)
        with torch.no_grad():
            cand.mu[0] = mu.float()
            d = torch.diagonal(L).clamp_min(2e-6) - 1e-6
            cand.L_diag[0] = torch.log(torch.expm1(d)).float()
            cand.L_off[0] = torch.tril(L, -1).float()
        info.update({"alpha": alpha, "gamma": float((lam / (lam + alpha)).sum()), "newton_iters": it, "newton_decrement": dec, "J": J,
                     "grad_norm": float(_objective(mu, chunks, alpha)[1].norm()), "limit": limit,
                     "log_evidence_laplace": -J + 0.5 * F * math.log(alpha) - 0.5 * logdetH, "alpha_history": hist[-5:],
                     "chol_max_abs_err": float((cand.chol()[0].double() - L).abs().max()),
                     "converged": {"alpha": conv_alpha, "newton": conv_newton}})
        info["available"] = bool(conv_alpha and conv_newton and math.isfinite(J))
        return (cand if info["available"] else None), info
    except Exception as e:      # noqa: BLE001  -- a failed candidate is unavailable; the run continues
        info["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return None, info


def save_heads(d, vbll, det, info):
    os.makedirs(d, exist_ok=True)
    torch.save({"state": vbll.state_dict(), **vbll.meta(), "kind": "vbll"}, os.path.join(d, "vbll_refit.pt"))
    torch.save({"state": det.state_dict(), **det.meta(), "kind": "det"}, os.path.join(d, "det_residual.pt"))
    json.dump(info, open(os.path.join(d, "refit.json"), "w"), indent=1, default=str)


def load_heads(d, lt, E_letters):
    out = {}
    for kind, fn in (("vbll", "vbll_refit.pt"), ("det", "det_residual.pt"), ("iso", "vbll_iso.pt"), ("a1", "vbll_a1.pt"), ("b0", "vbll_b0.pt"),
                     ("p0", "vbll_p0.pt"), ("p0b", "vbll_p0b.pt")):
        if not os.path.exists(os.path.join(d, fn)):
            continue
        st = torch.load(os.path.join(d, fn), map_location="cpu", weights_only=False)
        hd = make_head(lt, E_letters, "det" if kind == "det" else "vbll", st["dim"], st["seed"])
        hd.load_state_dict(st["state"])
        out[kind] = hd
    return out


def export_serving_head(d, vbll, label_ids, T_vbll, W_unique, extra=None):
    """the deployable head in the production export format (fullft_head.export_head: vbll_head.safetensors + vbll_head.json) with
    the fitted VBLL temperature; the serving engine applies mean_s softmax(z_s / T)"""
    if extra and (extra.get("temperature_by_kind") or extra.get("temperature_mode") not in (None, "single")):
        raise SystemExit("export_serving_head: the serving engine applies ONE scalar temperature; per-kind temperatures are not served")
    meta = fh.export_head(d, vbll, label_ids, 0, REFIT["samples"], W_unique,
                          {"temperature": float(T_vbll), "temperature_semantics": "mean_s softmax(z_s / T)", "refit": VERSION,
                           **(extra or {})})
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("refit", "fit-t", "score", "stamp-features", "check-features"))
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", default="", help="stamp / check-features: the manifest the features must come from")
    ap.add_argument("--expect", default="", help="stamp / check-features: JSON of the expected binding (identity, export, policy, ...)")
    ap.add_argument("--meta", default="", help="stamp / check-features: the stamp file")
    ap.add_argument("--pinned-code", default=HERE)
    ap.add_argument("--heads")
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=REFIT["steps"])
    ap.add_argument("--batch", type=int, default=REFIT["batch"])
    ap.add_argument("--lr", type=float, default=REFIT["lr"])
    ap.add_argument("--no-feature-check", action="store_true", help="refit / fit-t / score: skip the feature check (tests only; recorded)")
    ap.add_argument("--no-finalize", action="store_true", help="(v3) refit: skip the Stage-B finalizer candidates (= v2 behaviour)")
    ap.add_argument("--rule", choices=RULES, default="select", help="(v3.4) fit-t: which candidate is served (see the header)")
    ap.add_argument("--mean-scale", type=float, default=0.0, help="(v3.4) fit-t: beta of the b0 candidate (mu x beta); 0 = mean shift off")
    a = ap.parse_args(argv)
    if a.cmd in ("stamp-features", "check-features"):
        expect = json.loads(a.expect)
        if a.cmd == "check-features":
            why = check_features(a.features, a.meta, a.manifest, expect)
            print("FEATURES_VALID" if not why else "FEATURES_STALE " + "; ".join(why))
            return 0 if not why else 1
        f = torch.load(a.features, map_location="cpu", weights_only=False)
        fi, mi = features_ids(f), manifest_ids(a.manifest)
        if fi != mi:
            raise SystemExit("stamp-features: the features do not hold the manifest's ordered decisions: " + str(first_difference(fi, mi)))
        json.dump(feature_stamp(f, expect), open(a.meta, "w"), indent=1, sort_keys=True)
        print("FEATURES_STAMPED " + a.meta)
        return 0
    # (2026-10-14) refit / fit-t / score run standalone too: they check their features themselves (the stamp beside the file by
    # default) against --manifest / --expect, and refuse a stale cache. --no-feature-check is the recorded exception.
    if a.no_feature_check:
        fcheck = "skipped (--no-feature-check)"
    else:
        if not (a.manifest and a.expect):
            raise SystemExit(f"{a.cmd}: give --manifest and --expect (the features are checked before use), or --no-feature-check")
        meta = a.meta or (a.features[:-3] if a.features.endswith(".pt") else a.features) + ".stamp.json"
        why = check_features(a.features, meta, a.manifest, json.loads(a.expect))
        if why:
            raise SystemExit(f"FEATURES_STALE {a.cmd} {a.features}: " + "; ".join(why))
        fcheck = "valid"
        print(f"FEATURES_VALID {a.cmd} {a.features}")
        want_ro = json.loads(a.expect).get("letter_readout") or "stock"     # (v3.6)
        have_ro = (os.environ.get(fh.READOUT_ENV) if hasattr(fh, "READOUT_ENV") else None) or "stock"   # (v3.7) fullft_head without readouts = stock
        if want_ro != have_ro:
            raise SystemExit(f"READOUT_MISMATCH {a.cmd}: the features were made with letter readout {want_ro}, this process has {have_ro}")
    lt = fh.pinned_trainer(a.pinned_code)
    f = load_features(a.features)
    E = f["E_letters"].float()
    if a.cmd == "refit":
        out = {}
        heads = {}
        for kind in ("vbll", "det", "vbll_p0"):
            hd = make_head(lt, E, kind)
            out[kind] = fit(hd, f, a.steps, a.batch, a.lr, log_every=100)
            heads[kind] = hd
        p0 = heads.pop("vbll_p0")                    # (v3.5) saved beside the others; never a refit 'kind' downstream
        os.makedirs(a.out, exist_ok=True)
        torch.save({"state": p0.state_dict(), **p0.meta(), "kind": "vbll", "candidate": "p0", "frozen": ["P"]}, os.path.join(a.out, "vbll_p0.pt"))
        info = {"version": VERSION, "spec": HEAD_SPEC, "refit": {**REFIT, "steps": a.steps, "batch": a.batch, "lr": a.lr},
                "features": {"file": os.path.basename(a.features), "sha256": hashlib.sha256(open(a.features, "rb").read()).hexdigest(),
                             "meta": f.get("meta"), "check": fcheck}, "fits": out}
        if not a.no_finalize:                        # (v3) the Stage-B candidates on the same head-fit features
            info["finalizer"] = {"spec": FINAL, "candidates": {}}
            for name, mode in (("iso", "evidence"), ("a1", 1.0)):
                cand, ci = finalize(heads["vbll"], f, mode)
                os.makedirs(a.out, exist_ok=True)
                stale = os.path.join(a.out, f"vbll_{name}.pt")
                if os.path.exists(stale):
                    os.remove(stale)
                if cand is not None:
                    cand.jev_kind = "vbll"
                    torch.save({"state": cand.state_dict(), **cand.meta(), "kind": "vbll", "candidate": name}, stale)
                info["finalizer"]["candidates"][name] = ci
                print(f"FINALIZE {name} " + json.dumps({k: ci.get(k) for k in ("available", "alpha", "gamma", "newton_iters", "converged",
                                                                                "limit", "error")}))
        V, codes = fe_basis(E)                       # (v3.7) the fixedE Laplace on the head-fit pool, every alpha of the grid
        Ls, li = fe_fisher_gaussian(f, V, codes)
        torch.save({"V": V, "codes": codes, "L": Ls, "info": li, "version": VERSION}, os.path.join(a.out, "fixede_fisher.pt"))
        zc_err = 0.0
        ks_ = f["k"].long()
        for s0 in range(0, min(len(ks_), 4096), 1024):           # the identity sum_f phi == centred h . E_k (recorded)
            phi_, m_ = fe_phi(f["h"][s0:s0 + 1024], ks_[s0:s0 + 1024], int(ks_.max()), V, codes)
            zr = f["h"][s0:s0 + 1024].double() @ E.double().T[:, :int(ks_.max())]
            md = m_.double()
            zc = (zr - (zr * md).sum(1, keepdim=True) / md.sum(1, keepdim=True)) * md
            zc_err = max(zc_err, float((phi_.sum(-1) - zc).abs().max()))
        info["fixede"] = dict(li, identity_max_abs=zc_err)
        print("FIXEDE_FISHER " + json.dumps({"n": li["n"], "identity_max_abs": zc_err, "trace_S": {k: round(v["trace_S"], 4) for k, v in li["per_alpha"].items()}}))
        stale = os.path.join(a.out, "lloc_fit.pt")
        if os.path.exists(stale):
            os.remove(stale)
        yn, fl = manifest_yes_no(a.manifest, f) if a.manifest else (None, {"why": "no --manifest"})
        if yn is not None:                           # (v3.9) the L-loc fit on the head-fit pool; Yes / No rows from the rendered prompts
            lf, lli = lloc_fit(f, yn)
            torch.save({**lf, "info": lli, "flags": fl, "version": VERSION}, stale)
            info["lloc"] = dict(lli, flags=fl, available=True)
            print("LLOC_FIT " + json.dumps({k: lli[k] for k in ("n", "noul_dim", "yes_no_rows", "newton_iters", "converged", "grad_inf", "mu_norm", "shift_sd_headfit")})
                  + " FLAGS " + json.dumps({k: fl[k] for k in ("yes_no_rows", "kind_noul", "flagged_not_noul", "noul_not_flagged")}))
        else:
            info["lloc"] = {"available": False, "why": fl["why"] + ": the Yes / No rows come from the rendered prompts"}
        kc01 = manifest_kind_codes(a.manifest, f) if a.manifest else None
        for name01, (var01, fn01) in L01_VARIANTS.items():   # (v4.0) Latent01-post (full; di = the board variant) on the head-fit pool
            stale = os.path.join(a.out, fn01)
            if os.path.exists(stale):
                os.remove(stale)
            c01 = l01_spec(name01)
            if not c01["frozen"]:
                info[name01] = {"available": False, "why": f"{name01} constants not frozen ({c01.get('source')})"}
            elif kc01 is None:
                info[name01] = {"available": False, "why": "no --manifest: the kinds come from it"}
            else:
                st01, i01 = l01_fit(f, kc01, c01, variant=var01)
                torch.save({**st01, "info": i01, "version": VERSION}, stale)
                info[name01] = dict(i01, available=True)
                print(f"L01_FIT {name01} " + json.dumps({k: i01[k] for k in ("n", "nd", "noul_rows", "res_mean", "rel_s", "kinds")}))
        save_heads(a.out, heads["vbll"], heads["det"], info)
        print("REFIT " + json.dumps({k: v["history"][-1] for k, v in out.items()}))
        return 0
    hs = load_heads(a.heads, lt, E)
    pick = {"native": None, "det": hs["det"], "mean": hs["vbll"], "vbll": hs["vbll"]}
    if a.cmd == "fit-t":
        ts = {k: fit_T(k, pick[k], f) for k in READOUTS}
        hs["b0"] = mean_scaled(lt, E, hs["vbll"], a.mean_scale)     # (v3.4) stored, so score and the export use these exact values
        torch.save({"state": hs["b0"].state_dict(), **hs["b0"].meta(), "kind": "vbll", "candidate": "b0", "mean_scale": a.mean_scale,
                    "from": "vbll_refit.pt"}, os.path.join(a.heads, "vbll_b0.pt"))
        cands = {"sdh": ts["vbll"]}                  # (v3) every candidate's vbll+T on cal_select; the choice is made HERE, on cal only
        for name in CANDIDATES[1:] + ("b0",):
            if name in hs:
                cands[name] = fit_T("vbll", hs[name], f)
        beta_scan = None
        if "p0" in hs:                               # (v3.5) beta* on THIS cal pool, each beta with its own scalar T
            beta_scan = {}
            for b in BETA_GRID:
                beta_scan[b] = fit_T("vbll", mean_scaled(lt, E, hs["p0"], b), f)
            bstar = min(BETA_GRID, key=lambda b: (round(beta_scan[b]["nll_post"], 12), b))
            hs["p0b"] = mean_scaled(lt, E, hs["p0"], bstar)
            torch.save({"state": hs["p0b"].state_dict(), **hs["p0b"].meta(), "kind": "vbll", "candidate": "p0b", "mean_scale": bstar,
                        "from": "vbll_p0.pt"}, os.path.join(a.heads, "vbll_p0b.pt"))
            cands["p0b"] = dict(beta_scan[bstar], beta=bstar)
            beta_scan = {"grid": list(BETA_GRID), "beta": bstar,
                         "nll_post": {str(b): beta_scan[b]["nll_post"] for b in BETA_GRID}, "T": {str(b): beta_scan[b]["T"] for b in BETA_GRID}}
        elif a.rule == "p0beta":
            raise SystemExit("fit-t --rule p0beta: the heads dir has no vbll_p0.pt (refit with sdv1_head v3.5+)")
        fe_grid = None
        lp = next((x for x in (os.path.join(a.heads, "fixede_fisher.pt"), os.path.join(a.heads, "fixede_laplace.pt")) if os.path.exists(x)),
                  os.path.join(a.heads, "fixede_fisher.pt"))
        if os.path.exists(lp):                       # (v3.7) fixedE: alpha / kappa / T on THIS cal pool
            lap = torch.load(lp, map_location="cpu", weights_only=False)
            ch, fe_grid = fe_fit(f, lap)
            hdfe = fe_head(lap, ch["alpha"], ch["kappa"])
            torch.save({**hdfe, "alpha": ch["alpha"], "kappa": ch["kappa"], "T": ch["T"], "version": VERSION}, os.path.join(a.heads, "fixede.pt"))
            cands["fe"] = {k_: v_ for k_, v_ in ch.items() if k_ != "mean_only"}
            if ch.get("mean_only"):                   # (v3.8) the kappa = 0 control, its own alpha / T (reported, never served)
                m0 = ch["mean_only"]
                torch.save({**fe_head(lap, m0["alpha"], 0.0), "alpha": m0["alpha"], "kappa": 0.0, "T": m0["T"], "version": VERSION},
                           os.path.join(a.heads, "fixede0.pt"))
                cands["fe0"] = dict(m0)
        elif a.rule == "fixedE":
            raise SystemExit("fit-t --rule fixedE: the heads dir has no fixede_fisher.pt (refit with sdv1_head v3.7+)")
        lloc_flags = None
        llp = os.path.join(a.heads, "lloc_fit.pt")
        yn, lloc_flags = manifest_yes_no(a.manifest, f) if (a.manifest and os.path.exists(llp)) else (None, None)
        if yn is not None:                           # (v3.9) L-loc: the served posterior predictive 'lloc' and the MAP control 'lloc0', each its own T on THIS cal pool
            lf = torch.load(llp, map_location="cpu", weights_only=False)
            for name_, eps_ in (("lloc", lloc_eps(lf["V_perp"].shape[1])), ("lloc0", None)):
                r_ = lloc_fit_T(f, yn, lf, eps_)
                torch.save({k_: lf[k_] for k_ in ("V_E", "V_perp", "mu", "L", "beta")} | {"T": r_["T"], "draws": eps_ is not None, "version": VERSION},
                           os.path.join(a.heads, f"{name_}.pt"))
                cands[name_] = r_
            cands["lloc0"]["would_win"] = cands["lloc0"]["nll_post"] < cands["lloc"]["nll_post"]
        st01s = {}
        for name01, (var01, fn01) in L01_VARIANTS.items():   # (v4.0) Latent01-post (full / di): its scalar T on THIS cal pool
            l01p, c01 = os.path.join(a.heads, fn01), l01_spec(name01)
            if os.path.exists(l01p) and a.manifest and c01["frozen"]:
                st01 = torch.load(l01p, map_location="cpu", weights_only=False)
                r01 = l01_fit_T(f, manifest_kind_codes(a.manifest, f), st01, c01)
                st01s[name01] = st01
                torch.save({**{k_: v_ for k_, v_ in st01.items() if k_ != "info"}, "T": r01["T"], "constants": dict(c01), "version": VERSION},
                           os.path.join(a.heads, f"{name01}.pt"))
                cands[name01] = r01
            elif a.rule == name01:
                raise SystemExit(f"fit-t --rule {name01}: " + ("its constants are not frozen" if not c01["frozen"] else
                                 f"the heads dir has no {fn01} (refit with sdv1_head v4.0+ and a manifest)" if not os.path.exists(l01p) else "give --manifest (the kinds come from it)"))
        if a.rule == "lloc" and "lloc" not in cands:
            raise SystemExit("fit-t --rule lloc: " + ("the heads dir has no lloc_fit.pt (refit with sdv1_head v3.9+ and a manifest with prompts)" if not os.path.exists(llp)
                                                      else "give --manifest with rendered prompts (the Yes / No rows come from them)"))
        kinds = manifest_kinds(a.manifest)           # (v3.1) one or two temperature buckets, adopted on cal_select only
        for name in list(cands):                     # (v3.3) the two-bucket fit is REPORTED only: the engine serves one scalar T
            if name in ("fe", "fe0", "lloc", "lloc0", "l01post", "l01post_di"):
                cands[name] = dict(cands[name], two_bucket={"available": False, "why": f"{name}: not fitted"}, nll_adopted=cands[name]["nll_post"])
                continue
            t2 = fit_T2("vbll", hs["vbll"] if name == "sdh" else hs[name], f, kinds, cands[name])
            t2["would_improve"] = bool(t2.pop("adopted", False))
            t2["served"] = False
            cands[name] = dict(cands[name], two_bucket=t2, nll_adopted=cands[name]["nll_post"])
        if a.rule == "select":
            sel = min((n_ for n_ in cands if n_ in CANDIDATES), key=lambda n_: (round(cands[n_]["nll_adopted"], 12), CANDIDATES.index(n_)))
        else:
            sel = {"sdh": "sdh", "beta0": "b0", "p0beta": "p0b", "fixedE": "fe", "lloc": "lloc", "l01post": "l01post", "l01post_di": "l01post_di"}[a.rule]
        json.dump({"version": VERSION, "features_sha256": hashlib.sha256(open(a.features, "rb").read()).hexdigest(), "features_check": fcheck,
                   "T": ts, "candidates": {n_: {k: v[k] for k in ("T", "nll_pre", "nll_post", "at_bound", "two_bucket", "nll_adopted", "beta", "alpha", "kappa", "would_win", "yes_no_rows", "draws", "kinds") if k in v}
                                           for n_, v in cands.items()},
                   "selected": sel, "rule": a.rule, "mean_scale": a.mean_scale, "p0_beta_scan": beta_scan, "fixede_grid": fe_grid, "lloc_flags": lloc_flags,
                   "selection_rule": {"select": "min cal source-balanced vbll+T NLL with ONE scalar T (what the engine serves) over the "
                                                "available {sdh, iso, a1}; ties sdh first; dev never read; two-bucket T reported only",
                                      "sdh": "the AuxVBLL refit (sdh) with its scalar vbll+T on cal",
                                      "beta0": f"b0 = sdh with mu x {a.mean_scale:g} (covariance kept, mean shift off) with its scalar vbll+T on cal",
                                      "p0beta": "p0b = the P-frozen-at-init refit with mu x beta*, beta* and its scalar T chosen on cal",
                                      "fixedE": "fe = the fixed-E subspace head (mu = alpha 1, head-fit centred-Fisher Laplace x kappa, 32 antithetic pairs); alpha / kappa / T on cal",
                                      "lloc": "lloc = the L-loc readout (native logits + the Yes / No shift (beta / 2) u_perp . beta_noul, beta_noul ~ its head-fit "
                                              "Laplace N(mu, S), 32 antithetic pairs; MACKAY-HEAD's frozen constants); its scalar T on cal; lloc0 (MAP) reported",
                                      "l01post": "l01post = the Latent01-post readout (the reference FastLatent01 state on head-fit; MACKAY-HEAD's frozen constants "
                                                 "in L01); its scalar T on cal; kinds from the manifest",
                                      "l01post_di": "l01post_di = the BOARD Latent01-post (score rows are choice rows in fit, T and scoring; constants L01_DI); "
                                                    "its scalar T on cal"}[a.rule]},
                  open(os.path.join(a.heads, "temperature.json"), "w"), indent=1)
        info = json.load(open(os.path.join(a.heads, "refit.json")))
        lab = (f.get("meta") or {}).get("label_ids") or list(range(52))
        if sel in ("fe", "lloc", "l01post", "l01post_di"):   # (v3.7) the vbll_core fixedE schema; (v3.9) L-loc; (v4.0) Latent01-post
            sv_dir = os.path.join(a.heads, "serving")
            for fn_ in (fh.HEAD_FILE, fh.HEAD_META):
                if os.path.exists(os.path.join(sv_dir, fn_)):
                    os.remove(os.path.join(sv_dir, fn_))
            if sel == "fe":
                export_fixede(sv_dir, hdfe, lab, cands["fe"]["T"], {"candidate": "fe", "temperature_mode": "single", "head_rule": a.rule,
                                                                    "kappa": cands["fe"]["kappa"]})
            elif sel in ("l01post", "l01post_di"):
                export_l01(sv_dir, st01s[sel], lab, cands[sel]["T"], l01_spec(sel), {"candidate": sel, "temperature_mode": "single", "head_rule": a.rule})
            else:
                export_lloc(sv_dir, lf, lab, cands["lloc"]["T"], {"candidate": "lloc", "temperature_mode": "single", "head_rule": a.rule,
                                                                  "lloc0": {"T": cands["lloc0"]["T"], "nll_post": cands["lloc0"]["nll_post"],
                                                                            "would_win": cands["lloc0"]["would_win"]}})
        else:
            export_serving_head(os.path.join(a.heads, "serving"), hs["vbll"] if sel == "sdh" else hs[sel], lab, cands[sel]["T"],
                            info["fits"]["vbll"]["W_unique_headfit"], {"candidate": sel, "temperature_mode": "single", "head_rule": a.rule,
                                                                        **({"mean_scale": a.mean_scale} if sel == "b0" else {}),
                                                                        **({"mean_scale": cands["p0b"]["beta"], "frozen": ["P"]} if sel == "p0b" else {})})
        print("FIT_T " + json.dumps({k: round(v["T"], 6) for k, v in ts.items()}) + f" RULE {a.rule} SELECTED " + sel + " "
              + json.dumps({n_: [round(v["nll_post"], 6), round(v["two_bucket"].get("nll_post", float("nan")), 6) if v["two_bucket"].get("available")
                                 else None, v["two_bucket"].get("would_improve")] for n_, v in cands.items()}))
        return 0
    TJ = json.load(open(os.path.join(a.heads, "temperature.json")))
    T = TJ["T"]
    h, ell, ks = f["h"].float(), f["ell"].float(), f["k"].long()
    out = {k: {"pre": predict_T(k, pick[k], h, ell, ks, 1.0), "post": predict_T(k, pick[k], h, ell, ks, T[k]["T"])} for k in READOUTS}
    cand = {}
    kinds = manifest_kinds(a.manifest)
    yn_dev = None
    for name, c in (TJ.get("candidates") or {}).items():         # (v3) every candidate, and the cal-selected one
        if name in L01_VARIANTS:                     # (v4.0) Latent01-post: the kinds of THESE features' manifest rows (di: score -> choice)
            if not a.manifest:
                if TJ.get("selected") == name:
                    raise SystemExit(f"score: the selected head is {name}: give --manifest (the kinds come from it)")
                continue
            l0_ = torch.load(os.path.join(a.heads, f"{name}.pt"), map_location="cpu", weights_only=False)
            kc_ = manifest_kind_codes(a.manifest, f)
            K_ = ell.shape[1]
            cand[name] = {"pre": l01_predict(f, kc_, l0_, 1.0, K_, l0_["constants"]), "post": l01_predict(f, kc_, l0_, c["T"], K_, l0_["constants"]),
                          "T": c["T"]}
            continue
        if name in ("lloc", "lloc0"):                # (v3.9) L-loc: the Yes / No rows from THESE features' rendered prompts
            if yn_dev is None and a.manifest:
                yn_dev, _ = manifest_yes_no(a.manifest, f)
            if yn_dev is None:
                if TJ.get("selected") == "lloc":
                    raise SystemExit("score: the selected head is lloc: give --manifest with rendered prompts (the Yes / No rows come from them)")
                continue
            ll_ = torch.load(os.path.join(a.heads, f"{name}.pt"), map_location="cpu", weights_only=False)
            eps_ = lloc_eps(ll_["V_perp"].shape[1]) if ll_["draws"] else None
            K_ = ell.shape[1]
            cand[name] = {"pre": lloc_predict(f, yn_dev, ll_, 1.0, K_, eps_), "post": lloc_predict(f, yn_dev, ll_, c["T"], K_, eps_), "T": c["T"],
                          "yes_no_rows": int((yn_dev[:, 0] >= 0).sum())}
            continue
        if name in ("fe", "fe0"):                    # (v3.7) the fixedE head (its own predictive); (v3.8) fe0 = the kappa 0 control
            fe_ = torch.load(os.path.join(a.heads, "fixede.pt" if name == "fe" else "fixede0.pt"), map_location="cpu", weights_only=False)
            K_ = ell.shape[1]
            cand[name] = {"pre": fe_predict(f, fe_, 1.0, K_), "post": fe_predict(f, fe_, c["T"], K_), "T": c["T"],
                          "alpha": fe_["alpha"], "kappa": fe_["kappa"]}
            continue
        hd = hs["vbll"] if name == "sdh" else hs[name]
        cand[name] = {"pre": predict_T("vbll", hd, h, ell, ks, 1.0), "post": predict_T("vbll", hd, h, ell, ks, c["T"]), "T": c["T"]}
        tb = c.get("two_bucket") or {}
        if tb.get("available") and kinds is not None and len(kinds) == len(ks):    # (v3.1) the two-bucket form, for reporting
            with torch.no_grad():
                z, m = base_logits("vbll", hd, h, ell, ks)
                cand[name]["post_two_bucket"] = probs_Tvec("vbll", z, m, torch.tensor([tb["T"][k] for k in kinds], dtype=torch.float32))
            cand[name]["two_bucket_adopted"] = bool(tb.get("adopted"))
            cand[name]["post_adopted"] = cand[name]["post_two_bucket"] if tb.get("adopted") else cand[name]["post"]
    sel = TJ.get("selected") or "sdh"                # (v3.3) the default readouts are the SELECTED candidate (what is served)
    default = dict(out)
    if sel in cand:
        default["vbll"] = {"pre": cand[sel]["pre"], "post": cand[sel]["post"]}
    torch.save({"decision_uid": f["decision_uid"], "readouts": default, "readouts_vbll_head": sel, "readouts_sdh": out,
                "T": {k: T[k]["T"] for k in READOUTS}, "T_selected": (TJ.get("candidates") or {}).get(sel, {}).get("T", T["vbll"]["T"]),
                "features_check": fcheck, "candidates": cand, "selected": sel}, a.out)
    print(f"SCORED {len(ks)} decisions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
