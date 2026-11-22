#!/usr/bin/env python3
"""(FULLFT) The residual VBLL decision head and the decision loss for FULL fine-tuning of Gemma-4-31B-it (main line, memory note
jev-bayesian-head-required): backbone -> native letter logits ell -> residual variational Bayesian last layer
z_j = tau ell_j + phi_j(h)^T beta, q(beta) = N(mu, S) -> posterior predictive E_q softmax(z), from ONE backbone forward.

The head IS the production AuxVBLL (claude_lmtune.py e2c6100a, loaded by path from the pinned production set; the same class
claude_lmscore.py e0672bd0 --aux-vbll scores), so its math, init draws, codes and Monte Carlo predictive are the ones SCREEN4 measures.

DETACHED features (decision, see module notes in fullft_train.py): the head reads h and ell detached; its gradient never reaches the
backbone, so the backbone's update is exactly that of the declared backbone objective (tested bitwise: with or without the head).

BACKBONE LOSS per decision i (letter order; p = softmax(ell[:k])), spec mixture_spec_v1.json parent_preservation.loss:
    eligible (has a parent cache entry, alpha > 0):  (1 - alpha) CE(t_i, p) + alpha KL(q_i || p)        (alpha = 0.15 in the spec)
    otherwise (numeric / exact units, or alpha = 0): CE(t_i, p)          [--ineligible-gold-weight one]  or (1 - alpha) CE  [1-alpha]
Equivalent to SCREEN4's data-side form: (1 - a) CE(t, p) + a KL(q || p) = CE(t', p) - a H(q) with t' = (1 - a) t + a q, i.e.
screen4_loss.mix(t, q, lam) with lam = a / (1 - a) (t' identical; the same gradient); the screen's lam form CE + lam KL is this times
(1 + lam). The gold target t is never modified; q only enters through the KL term.

GLOBAL NORMALISATION. A step's loss is sum_i w_i loss_i / W_step, W_step = sum of the weights of ALL decisions of the global step
(every rank, every micro-batch), so the gradient does not depend on the world size, the packing or the accumulation split.

HEAD OBJECTIVE (head_likelihood_contract): per step  -sum_i w_i bound_i / W_step  +  sum_g kl_g KL_g(q || p) / W_total, where
W_total = the PINNED sum of training decision weights from the immutable exposure manifest (each decision once; never the number of
optimizer steps or epoch visits). The data term's gradient is SUMMED over ranks (each rank's share already divided by the global
W_step), then the KL term is added ONCE per optimizer step on every rank (same values), then per-head clip and the head's own AdamW.
The head trains on GOLD targets (its likelihood is the declared weighted gold likelihood; parent probabilities never enter it).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os

import numpy as np
import torch

VERSION = "fullft_head v4 (2026-10-13)"
PINNED_TRAINER_MD5 = "e2c6100a1c0651804613da6aaa477fb0"
GRID = {"prior": "1", "kl": "0,1,10000,1000000", "cov": "diag,full", "dim": 128, "samples": 64, "lr": 1e-3}   # SCREEN4's grid
_LT = {}


def pinned_trainer(code_dir):
    """the pinned production claude_lmtune.py (md5-checked) from `code_dir` (holding trainers/claude_lmtune.py, claude_lmscore.py,
    cache_features.py, claude_ckpt.py)."""
    p = os.path.join(code_dir, "claude_lmtune.py")
    if not os.path.exists(p):
        p = os.path.join(code_dir, "trainers", "claude_lmtune.py")
    if p in _LT:
        return _LT[p]
    m5 = hashlib.md5(open(p, "rb").read()).hexdigest()
    if m5 != PINNED_TRAINER_MD5 and not os.environ.get("JEV_ALLOW_UNPINNED_TRAINER"):
        raise SystemExit(f"fullft_head: {p} md5 {m5} != the pinned production trainer {PINNED_TRAINER_MD5}")
    spec = importlib.util.spec_from_file_location("lt_pinned_fullft", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LT[p] = mod
    return mod


def heads_of(prior=GRID["prior"], kl=GRID["kl"], cov=GRID["cov"]):
    out = []
    for s in str(prior).split(","):
        for k in str(kl).split(","):
            for c in str(cov).split(","):
                out.append({"prior": float(s), "kl": float(k), "cov": c.strip()})
    return out


# ---------------------------------------------------------------------------------------------------------------- decision loss
def _f(x):
    """fp32, or fp64 when the input is fp64 (CPU exactness tests)"""
    return x if x.dtype == torch.float64 else x.float()


READOUT_ENV = "JEV_LETTER_READOUT"   # (2026-10-15) letter readout mode; unset or "stock" = the model's own formula (byte-identical)


def letter_readout_spec(spec=None):
    """parse a letter-readout spec -> (mode, alpha): 'stock' (default) | 'blend:A' (0 < A < 1). Anything else is refused."""
    spec = (os.environ.get(READOUT_ENV, "") if spec is None else spec) or "stock"
    if spec == "stock":
        return "stock", 0.0
    mode, _, a = spec.partition(":")
    if mode == "blend":
        try:
            av = float(a)
        except ValueError:
            av = -1.0
        if 0.0 < av < 1.0 and math.isfinite(av):
            return "blend", av
    raise ValueError(f"{READOUT_ENV}={spec!r}: 'stock' or 'blend:A' with 0 < A < 1")


def apply_readout(z, cap, spec=None):
    """z = h E^T (fp32 / fp64) -> the letter logits. stock: cap tanh(z / cap) (no cap: z). blend:A (the 31B / 12B softcap-tail fix
    candidate): (1 - A) cap tanh(z / cap) + A z -- the stock value near z = 0, a minimum slope A in the tail. The serving engine
    (release/garas_vbll/vbll_core.apply_readout) computes the SAME expression in the same order."""
    mode, a = letter_readout_spec(spec)
    if mode == "stock" or not cap:
        if mode != "stock":
            raise ValueError("blend readout needs a softcap")
        return cap * torch.tanh(z / cap) if cap else z
    return (cap * torch.tanh(z / cap)) * (1.0 - a) + z * a


def letter_logits(h, E, cap, spec=None):
    """the model's own head formula on the option letters in fp32 (claude_lmtune.letter_head_fp32 for Gemma-4: tied embedding rows,
    final_logit_softcapping): h [n, H], E [52, H] -> [n, 52]. (2026-10-15) $JEV_LETTER_READOUT (or spec) = blend:A selects the
    blended readout (apply_readout); unset / 'stock' is the unchanged formula."""
    z = _f(h) @ E.to(_f(h).dtype).t()
    return apply_readout(z, cap, spec)


def masked_logp(ell, ks):
    mask = torch.arange(ell.shape[1], device=ell.device)[None, :] < ks[:, None]
    return torch.log_softmax(_f(ell).masked_fill(~mask, -1e9), -1).masked_fill(~mask, 0.0), mask


def decision_losses(ell, ks, tgt, parent, has_parent, alpha, ineligible="one"):
    """per-decision backbone loss [n] (module docstring). ell [n, K] fp32; tgt / parent [n, K] letter order (0 beyond k; parent rows
    of decisions without a parent are ignored); has_parent [n] bool. -> (loss [n], ce [n], kl [n])."""
    logp, mask = masked_logp(ell, ks)
    tgt, parent = tgt.to(logp.dtype), parent.to(logp.dtype)
    ce = -(tgt * logp).sum(-1)
    q = torch.where(has_parent[:, None], parent, torch.zeros_like(parent))
    ql = torch.where(q > 0, q.clamp_min(1e-30).log(), torch.zeros_like(q))
    kl = (q * (ql - logp)).masked_fill(~mask, 0.0).sum(-1)
    a = float(alpha)
    if a < 0 or a >= 1:
        raise ValueError("alpha must be in [0, 1)")
    elig = has_parent & (a > 0)
    g_inel = 1.0 if ineligible == "one" else 1.0 - a
    if ineligible not in ("one", "1-alpha"):
        raise ValueError(f"ineligible gold weight {ineligible!r}")
    loss = torch.where(elig, (1 - a) * ce + a * kl, g_inel * ce)
    return loss, ce.detach(), torch.where(elig, kl, torch.zeros_like(kl)).detach()


# ------------------------------------------------------------------------------------------------------------------------ head
class HeadState:
    """the residual VBLL head (pinned AuxVBLL, G heads), its AdamW and LR schedule, and the contract's normalisation."""

    def __init__(self, lt, letters, heads, dim, seed, lr, total_weight, warmup, steps, device):
        if not (total_weight > 0 and math.isfinite(total_weight)):
            raise ValueError("the pinned total decision weight must be positive (exposure manifest)")
        self.lt, self.total_weight = lt, float(total_weight)
        self.head = lt.AuxVBLL(letters, heads, dim, seed=seed).to(device)
        self.params = list(self.head.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(1, steps))))
        self.lr, self.seed, self.dim = lr, seed, dim

    def data_backward(self, h, ell, tgt, ks, w, W_step):
        """this rank's data term: -sum_i w_i bound_i / W_step (gradient into the head only; h / ell detached). -> per-head sums of
        -bound (float64, for logging)."""
        bnd = self.head.bound(h.detach().float(), ell.detach().float(), tgt, ks)          # [G, n]
        (-(bnd * w[None]).sum(1).sum() / W_step).backward()
        return (-bnd.detach() * w[None]).double().sum(1)

    def kl_term(self):
        return (self.head.kl_w * self.head.kl()).sum() / self.total_weight

    def step(self, reduce_fn=None):
        """after every micro-batch's data_backward: SUM the data-term gradients over ranks (reduce_fn), add the KL term ONCE, clip per
        head at 1.0, AdamW, schedule. -> per-head grad norms before clipping."""
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)     # a rank with only dummy forwards still joins the reduction
        if reduce_fn is not None:
            reduce_fn([p.grad for p in self.params])
        self.kl_term().backward()
        norms = self.head.clip_(1.0)
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        return norms

    def state(self):
        return {"head": self.head.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict()}

    def load(self, st):
        self.head.load_state_dict(st["head"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])


def readouts(head, h, ell, ks, samples=64, seed=0):
    """from ONE forward's h and ell: native p = softmax(ell), mean-head p = softmax(tau ell + phi^T mu) per head, posterior-predictive
    p = E_q softmax(z) per head (fixed seeded draws, AuxVBLL.predict). -> dict of tensors ([n, K], [G, n, K], [G, n, K], epi / alea)."""
    with torch.no_grad():
        logp, mask = masked_logp(ell, ks)
        mean, _, m = head.moments(h.float(), ell.float(), ks)
        pm = torch.softmax(mean.masked_fill(~m[None], -1e9), -1).masked_fill(~m[None], 0.0)
        pp, epi, alea, pvar, lvar = head.predict(h.float(), ell.float(), ks, samples, seed)
    return {"native": logp.exp().masked_fill(~mask, 0.0), "mean": pm, "predictive": pp, "epi": epi, "alea": alea, "lvar": lvar}


# ------------------------------------------------------------------------------------------------------------- export / import
HEAD_FILE, HEAD_META = "vbll_head.safetensors", "vbll_head.json"


def export_head(dirpath, head, label_ids, selected, samples, total_weight, extra=None):
    """write the SELECTED head (G = 1) beside the model weights: vbll_head.safetensors (AuxVBLL state) + vbll_head.json (spec). The
    serving engine refuses to run without these files."""
    from safetensors.torch import save_file
    g = int(selected)
    st = head.state_dict()
    one = {}
    for k, v in st.items():
        if k == "L_off" and head.heads[g]["cov"] != "full":
            continue                                   # a diagonal head has no off-diagonal factor
        one[k] = v[g:g + 1].detach().cpu().contiguous() if v.dim() >= 1 and v.shape[0] == len(head.heads) else v.detach().cpu()
    os.makedirs(dirpath, exist_ok=True)
    save_file(one, os.path.join(dirpath, HEAD_FILE))
    meta = {"schema": "jev-residual-vbll-head-v1", "class": "AuxVBLL (claude_lmtune.py e2c6100a)", "heads": [head.heads[g]],
            "dim": head.F, "hidden": head.H, "seed": head.seed, "label_ids": [int(x) for x in label_ids], "samples": int(samples),
            "draw_seed": 0, "draw_grid_size": len(head.heads), "draw_slot": g, "selected_from_grid": g, "grid": head.heads, "kl_normaliser_total_weight": float(total_weight),
            "readout": "posterior_predictive", "letters": "A-Z a-z (52, one token each)",
            "native_logits": "fp32 h @ E[letters]^T with final_logit_softcapping (tied embedding)",
            "tensor_sha256": hashlib.sha256(open(os.path.join(dirpath, HEAD_FILE), "rb").read()).hexdigest(),
            **({"letter_readout": os.environ[READOUT_ENV]} if letter_readout_spec()[0] != "stock" else {}), **(extra or {})}
    with open(os.path.join(dirpath, HEAD_META), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def import_head(dirpath, lt, letters):
    """-> (AuxVBLL with G = 1, meta). Refuses a missing / altered head (sha256)."""
    from safetensors.torch import load_file
    p, j = os.path.join(dirpath, HEAD_FILE), os.path.join(dirpath, HEAD_META)
    if not (os.path.exists(p) and os.path.exists(j)):
        raise FileNotFoundError(f"{dirpath}: no residual VBLL head ({HEAD_FILE} + {HEAD_META}); this model must not be served without it")
    meta = json.load(open(j))
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != meta["tensor_sha256"]:
        raise ValueError(f"{p}: sha256 differs from {HEAD_META}")
    m = lt.AuxVBLL(letters, meta["heads"], meta["dim"], seed=meta["seed"])
    m.load_state_dict(load_file(p))
    return m, meta


def collapse_covariance(head):
    """the MEAN-HEAD control as a head: covariances collapsed (screen4_head.meanhead), so the Monte Carlo predictive equals
    softmax(tau ell + phi^T mu) to ~1e-6."""
    with torch.no_grad():
        head.L_diag.fill_(-40.0)
        if head.L_off is not None:
            head.L_off.zero_()
    return head


# ------------------------------------------------------------------------------------------------------------ head selection CLI
def select_head(nll_by_head, heads):
    """SCREEN4's HEAD_RULE (screen4_head.select_head): the variational head (kl > 0) with the lowest pooled DEV NLL; ties (< 1e-4) ->
    the smaller kl, then diag before full."""
    cand = [g for g, h in enumerate(heads) if float(h["kl"]) > 0]
    if not cand:
        raise ValueError("no variational head in the grid")
    best = min(nll_by_head[g] for g in cand)
    tied = [g for g in cand if nll_by_head[g] - best < 1e-4]
    return min(tied, key=lambda g: (float(heads[g]["kl"]), heads[g]["cov"] != "diag"))


def panel_name(v):
    """(v7) a devset panel_of value -> its panel for HEAD_RULE / the read: "pilotval:<pool>" -> "pilotval" (ONE panel over its pools);
    "panel:<name>[:<block>]" -> "panel:<name>"."""
    parts = str(v).split(":")
    return parts[0] if parts[0] == "pilotval" else ":".join(parts[:2])


def _unit_metrics(r):
    t = np.asarray(r["t"], dtype=np.float64)
    p = np.clip(np.asarray(r["native"], dtype=np.float64), 1e-12, 1)
    out = {"nll": float(-(t * np.log(p)).sum()), "acc": float(t[int(np.argmax(p))] >= t.max() - 1e-9), "brier": float(((p - t) ** 2).sum())}
    if "predictive" in r:
        out["pred_nll"] = [float(-(t * np.log(np.clip(np.asarray(q), 1e-12, 1))).sum()) for q in r["predictive"]]
    return out


def lr_read(runs, dev_panels, base_eval, default_lr, seed=0, n_boot=2000, forget_tol=0.05, spike=20.0, warmup=50, g3="spike20"):
    """(v3) the LR-probe read, PRE-REGISTERED (fleet/task_fullft4.sh header; the hold lines' report):
      runs: {lr (float): {"evals": {step: [eval rows]}, "steps": [STEP rows]}}; dev_panels: decision_uid -> panel ("pilotval:<pool>"
      or "panel:<name>"); base_eval: the init's eval rows (step 0).
      PRIMARY  pilot-val native NLL at each run's LAST eval step; the lowest wins.
      TIE      paired bootstrap (units resampled, n_boot) of NLL(best) - NLL(default_lr): if the 95% CI includes 0, keep default_lr.
      GUARDS   (disqualify a run) G1 a missing / non-finite eval or train loss; G2 forgetting: decision-quality panels' (panel:*)
               native NLL at the last eval > the init's + forget_tol; G3 instability: a post-warmup pre-clip grad norm > spike x the
               run's post-warmup median.
      FLAGS    the winner at the grid's edge while its pilot-val NLL still falls over its last two evals faster than default_lr's.
      STOP     every run disqualified, or every run's pilot-val NLL at its first eval above the init's + 0.1."""
    def agg(rows, pref):
        sel = [_unit_metrics(r) for r in rows if dev_panels.get(r["decision_uid"], "").startswith(pref)]
        if not sel:
            return None
        o = {"n": len(sel), "nll": float(np.mean([m["nll"] for m in sel])), "acc": float(np.mean([m["acc"] for m in sel])),
             "brier": float(np.mean([m["brier"] for m in sel]))}
        if "pred_nll" in sel[0]:
            o["pred_nll_by_head"] = [float(np.mean([m["pred_nll"][g] for m in sel])) for g in range(len(sel[0]["pred_nll"]))]
        return o
    unk = sorted({r["decision_uid"] for rr in [base_eval] + [ev for v in runs.values() for ev in v["evals"].values()] for r in rr}
                 - set(dev_panels))
    if unk:
        raise ValueError(f"{len(unk)} evaluated units have no panel, e.g. {unk[:2]}")
    init = {"pilotval": agg(base_eval, "pilotval:"), "panels": agg(base_eval, "panel:")}
    table, dq = {}, {}
    for lr, v in sorted(runs.items()):
        steps = sorted(v["evals"])
        rows = {s: {"pilotval": agg(v["evals"][s], "pilotval:"), "panels": agg(v["evals"][s], "panel:")} for s in steps}
        g = [float(x["gnorm"]) for x in v["steps"] if int(x["step"]) >= warmup and x.get("gnorm") is not None]
        losses = [float(x["loss"]) for x in v["steps"]]
        why = []
        if not steps or not all(np.isfinite(losses)) or any(r["pilotval"] is None or r["panels"] is None for r in rows.values()):
            why.append("G1 missing / non-finite")
        else:
            last = rows[steps[-1]]
            if last["panels"]["nll"] > init["panels"]["nll"] + forget_tol:
                why.append(f"G2 forgetting: panels NLL {last['panels']['nll']:.4f} > init {init['panels']['nll']:.4f} + {forget_tol}")
        if g and g3 == "spike20" and max(g) > spike * float(np.median(g)):
            why.append(f"G3 grad-norm spike {max(g):.3g} > {spike} x median {float(np.median(g)):.3g}")
        elif g and g3 == "persistent":                 # DIAGNOSTIC ONLY (not the registered rule): a spike must persist or move the loss
            post = [x for x in v["steps"] if int(x["step"]) >= warmup and x.get("gnorm") is not None]
            med_g = float(np.median(g))
            med_l = float(np.median([float(x["loss"]) for x in post]))
            hot = [float(x["gnorm"]) > spike * med_g for x in post]
            run3 = any(all(hot[i:i + 3]) for i in range(max(0, len(hot) - 2)))
            lossy = [int(x["step"]) for x, h in zip(post, hot) if h and float(x["loss"]) > 2 * med_l]
            if run3 or lossy:
                why.append(f"G3' persistent spikes (3 consecutive: {run3}; with loss > 2 x median at steps {lossy[:5]})")
        table[lr] = {"evals": rows, "post_warmup_gnorm_median": float(np.median(g)) if g else None,
                     "post_warmup_gnorm_max": max(g) if g else None, "last_step": max((int(x["step"]) for x in v["steps"]), default=None)}
        dq[lr] = why
    ok = [lr for lr in table if not dq[lr]]
    out = {"init": init, "table": table, "disqualified": {lr: w for lr, w in dq.items() if w}, "stop": None, "flags": []}
    first_bad = [lr for lr in table if table[lr]["evals"] and (lambda r: r["pilotval"] is not None and
                 r["pilotval"]["nll"] > init["pilotval"]["nll"] + 0.1)(table[lr]["evals"][min(table[lr]["evals"])])]
    if not ok:
        out["stop"] = "every run disqualified"
    elif len(first_bad) == len(table):
        out["stop"] = "every run's pilot-val NLL at its first eval is > init + 0.1"
    if out["stop"]:
        out["chosen_lr"] = None
        return out
    lastv = {lr: table[lr]["evals"][max(table[lr]["evals"])]["pilotval"]["nll"] for lr in ok}
    best = min(ok, key=lambda lr: (lastv[lr], abs(math.log(lr / default_lr))))
    chosen, ci = best, None
    if best != default_lr and default_lr in ok:
        def per_unit(lr):
            ev = runs[lr]["evals"][max(runs[lr]["evals"])]
            return {r["decision_uid"]: _unit_metrics(r)["nll"] for r in ev if dev_panels[r["decision_uid"]].startswith("pilotval:")}
        a_, b_ = per_unit(best), per_unit(default_lr)
        uids = sorted(set(a_) & set(b_))
        d = np.array([a_[u] - b_[u] for u in uids])
        rng = np.random.default_rng(seed)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        if ci[1] >= 0:
            chosen = default_lr
    out["bootstrap_best_minus_default_ci95"] = ci
    out["best_by_point"] = best
    out["chosen_lr"] = chosen
    grid = sorted(table)
    if chosen in (grid[0], grid[-1]) and chosen != default_lr:
        def slope(lr):
            ss = sorted(table[lr]["evals"])
            return table[lr]["evals"][ss[-1]]["pilotval"]["nll"] - table[lr]["evals"][ss[-2]]["pilotval"]["nll"] if len(ss) > 1 else 0.0
        if default_lr in table and slope(chosen) < slope(default_lr):
            out["flags"].append(f"chosen LR {chosen:g} is at the grid's edge and still improving faster than {default_lr:g}: the optimum "
                                "may lie beyond the grid")
    return out


def lr_read_registered_plus_diagnostic(runs, dev_panels, base_eval, default_lr, registered="spike20"):
    """(v3) the REGISTERED read (G3 = any post-warmup pre-clip grad norm > 20 x the run's median), plus a clearly separate
    diagnostic under G3' (a spike disqualifies only if it persists 3 consecutive steps or comes with a train loss > 2 x the median):
    the LR-probe of 2026-10-13 showed isolated, clipped pre-clip spikes (1e-5: 29,389 at step 91 against a median of 28.5, loss
    0.81) that the registered G3 counts. The decision is the registered one; the diagnostic is for the coordinator."""
    other = "persistent" if registered == "spike20" else "spike20"
    reg = lr_read(runs, dev_panels, base_eval, default_lr, g3=registered)
    diag = lr_read(runs, dev_panels, base_eval, default_lr, g3=other)
    reg[f"diagnostic_g3_{other}"] = {k: diag.get(k) for k in ("chosen_lr", "best_by_point", "bootstrap_best_minus_default_ci95",
                                                              "disqualified", "stop", "flags")}
    reg["registered_g3"] = registered
    reg["default_lr"] = default_lr
    reg["note"] = f"chosen_lr follows the REGISTERED rules (G3 = {registered}); diagnostic_g3_{other} is NOT the registered decision"
    return reg


def main(argv=None):
    """select-export --grid EXPORT/vbll_head_grid.pt --eval DEV_eval.jsonl --label-ids-from EXPORT --out EXPORT: pick the output head
    on a DEV evaluation (fullft_train.py eval_step*.jsonl rows) by HEAD_RULE and write vbll_head.safetensors + vbll_head.json."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select-export", "lr-read"))
    ap.add_argument("--grid")
    ap.add_argument("--eval")
    ap.add_argument("--pinned-code")
    ap.add_argument("--tokenizer")
    ap.add_argument("--total-weight", type=float)
    ap.add_argument("--samples", type=int, default=GRID["samples"])
    ap.add_argument("--runs", nargs="*", help="lr-read: LR=RUN_DIR (eval_step*.jsonl + train.log)")
    ap.add_argument("--panels", help="lr-read / select-export: the DEV dir's panels.json (select-export: panel-macro HEAD_RULE)")
    ap.add_argument("--base-eval", help="lr-read: the init's eval rows (eval_step0.jsonl)")
    ap.add_argument("--default-lr", type=float, default=1e-5)
    ap.add_argument("--g3", choices=("spike20", "persistent"), default="spike20",
                    help="lr-read: the REGISTERED spike rule (spike20 = the original probe's G3; persistent = G3', registered for E1 / E2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "lr-read":
        import glob
        pm = json.load(open(a.panels))["panel_of"]
        runs = {}
        for spec in a.runs:
            lr, d = spec.split("=", 1)
            ev = {}
            for f in glob.glob(os.path.join(d, "eval_step*.jsonl")):
                s = int(os.path.basename(f)[len("eval_step"):-len(".jsonl")])
                if s > 0:
                    ev[s] = [json.loads(l) for l in open(f) if l.strip()]
            st = [json.loads(l[5:]) for l in open(os.path.join(d, "train.log"), errors="replace") if l.startswith("STEP ")]
            runs[float(lr)] = {"evals": ev, "steps": st}
        base = [json.loads(l) for l in open(a.base_eval) if l.strip()]
        if a.default_lr not in runs:
            raise SystemExit(f"--default-lr {a.default_lr} is not one of the runs {sorted(runs)}")
        res = lr_read_registered_plus_diagnostic(runs, pm, base, a.default_lr, registered=a.g3)
        res = json.loads(json.dumps(res, default=float))
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("LR_READ " + json.dumps({k: res.get(k) for k in ("registered_g3", "default_lr", "chosen_lr", "best_by_point",
                                                               "bootstrap_best_minus_default_ci95", "stop", "disqualified", "flags",
                                                               "diagnostic_g3_persistent", "diagnostic_g3_spike20")}, default=str))
        return 0
    for req in ("grid", "eval", "pinned_code", "tokenizer", "total_weight"):
        if getattr(a, req) is None:
            raise SystemExit(f"select-export needs --{req.replace('_', '-')}")
    import numpy as np
    from transformers import AutoTokenizer
    lt = pinned_trainer(a.pinned_code)
    g = torch.load(a.grid, map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    lab = [tok(c, add_special_tokens=False)["input_ids"][0] for c in labels]
    head = lt.AuxVBLL(torch.zeros(52, g["hidden"]), g["heads"], g["dim"], seed=g["seed"])
    head.load_state_dict(g["state"])
    rows = [json.loads(l) for l in open(a.eval) if l.strip()]
    G = len(g["heads"])
    unit_nll = [[-(np.asarray(r["t"]) * np.log(np.clip(np.asarray(r["predictive"][k]), 1e-12, 1))).sum() for r in rows] for k in range(G)]
    if a.panels:                                      # (v4) panel-macro: the mean over panels of each panel's unit-pooled NLL
        pm = json.load(open(a.panels))["panel_of"]
        miss = [r["decision_uid"] for r in rows if r["decision_uid"] not in pm]
        if miss:
            raise SystemExit(f"{len(miss)} eval rows have no panel, e.g. {miss[:2]}")
        pan = [panel_name(pm[r["decision_uid"]]) for r in rows]
        names = sorted(set(pan))
        nll = [float(np.mean([np.mean([x for x, p_ in zip(unit_nll[k], pan) if p_ == nm]) for nm in names])) for k in range(G)]
        rule = f"lowest panel-macro DEV NLL over {names} among kl > 0 heads (HEAD_RULE)"
    else:
        nll = [float(np.mean(unit_nll[k])) for k in range(G)]
        rule = "lowest pooled DEV NLL among kl > 0 heads (SCREEN4 HEAD_RULE)"
    sel = select_head(nll, g["heads"])
    meta = export_head(a.out, head, lab, sel, a.samples, a.total_weight,
                       {"selection": {"rule": rule, "dev_nll": nll,
                                      "eval_file_sha256": hashlib.sha256(open(a.eval, "rb").read()).hexdigest(), "n_dev": len(rows)}})
    print("HEAD_SELECTED " + json.dumps({"head": sel, "spec": g["heads"][sel], "dev_nll": nll}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
