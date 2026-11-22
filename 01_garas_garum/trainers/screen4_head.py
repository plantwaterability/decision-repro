#!/usr/bin/env python3
"""(SCREEN4) The RESIDUAL VBLL decision head: checkpoint tools and the pre-declared head rule (design tmp/screen4/design.md section 4).

Architecture (user requirement 2026-10-12): Gemma backbone (LoRA, ordinary fine-tuning) -> its native decision (letter) logits ell ->
a residual variational Bayesian last layer, z_j = tau * ell_j + phi_j(h)^T beta with q(beta) = N(mu, S) -> the POSTERIOR-PREDICTIVE
p(y | x) = E_q softmax(z), integrated at inference by Monte Carlo over q (64 fixed draws). One backbone forward plus a 128-dim head.
The head IS the production trainer's --aux-vbll head (claude_lmtune.py e2c6100a, AuxVBLL; Harrison et al. 2024 discriminative VBLL,
closed-form Jensen bound, KL to N(0, s I) weighted kl / n_train) trained on DETACHED inputs, so the backbone's training is that of
the same run without the head (the trainer's own bit-for-bit test) and every head of the grid sees the same backbone trajectory.
It is scored by the pinned claude_lmscore.py --aux-vbll (test_aux_p = the predictive of the selected head; test_aux_grid_p = every
head's). What the head is NOT: it captures uncertainty in the head weights CONDITIONAL on the backbone's features; it says nothing
about backbone uncertainty.

Grid (every arm, one training run): prior s = 1; kl in {0, 1, 1e4, 1e6} x cov in {diag, full} (the (99) grid + kl 0).
  kl 0     the DETERMINISTIC-HEAD CONTROL: the bound with no pull towards the prior; its variance is free and shrinks, so it is read
           as its MEAN head (beta = mu) -- a point-estimate residual head trained on the same examples and features;
  kl > 0   the variational heads; the arm's OUTPUT head is selected among them by HEAD_RULE on the DEV split only.
MEAN-HEAD readout: `meanhead` writes a copy of a checkpoint whose head covariances are collapsed (sigma -> 1e-6, no off-diagonal), so
the pinned scorer's Monte Carlo predictive equals softmax(tau * ell + phi^T mu) to ~1e-6 with NO scorer change; the LoRA tensors are
copied unchanged (checked by md5 of their bytes).

  python3 trainers/screen4_head.py meanhead --in CK --out CK_MEAN
  python3 trainers/screen4_head.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys

import numpy as np

VERSION = "screen4_head v1 (2026-10-12)"
GRID = {"aux_vbll_prior": "1", "aux_vbll_kl": "0,1,10000,1000000", "aux_vbll_cov": "diag,full", "aux_vbll_dim": 128,
        "aux_vbll_samples": 64, "aux_vbll_lr": 1e-3}
COLLAPSED_L_DIAG = -40.0          # softplus(-40) ~ 4e-18 -> sigma = 1e-6 (the AuxVBLL floor)
HEAD_RULE = ("the variational head (kl > 0) with the lowest unit-pooled mean NLL over the DEV panels (DI-family dev + EG dev + caldev, "
             "each panel weighted equally); ties (|dNLL| < 1e-4) -> the smaller kl, then diag before full")


def trainer_flags():
    """the --aux-vbll flags every arm passes to claude_lmtune.py (one string)"""
    return (f"--aux-vbll --aux-vbll-prior {GRID['aux_vbll_prior']} --aux-vbll-kl {GRID['aux_vbll_kl']} --aux-vbll-cov "
            f"{GRID['aux_vbll_cov']} --aux-vbll-dim {GRID['aux_vbll_dim']} --aux-vbll-samples {GRID['aux_vbll_samples']} "
            f"--aux-vbll-lr {GRID['aux_vbll_lr']:g}")


def heads_of_grid():
    """the head list in the trainer's nesting order (prior, kl, cov) -- claude_lmtune.aux_vbll_grid's order"""
    out = []
    for s in GRID["aux_vbll_prior"].split(","):
        for k in GRID["aux_vbll_kl"].split(","):
            for c in GRID["aux_vbll_cov"].split(","):
                out.append({"prior": float(s), "kl": float(k), "cov": c})
    return out


def lora_digest(lora):
    import torch
    h = hashlib.md5()
    for k in sorted(lora):
        b = io.BytesIO()
        torch.save(lora[k].detach().cpu().contiguous(), b)
        h.update(k.encode())
        h.update(b.getvalue())
    return h.hexdigest()


def meanhead(src, dst):
    """copy checkpoint `src` to `dst` with every head's covariance collapsed (module docstring). -> record"""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "aux_vbll" not in ck:
        raise SystemExit(f"{src}: no aux_vbll head")
    st = ck["aux_vbll"]["state"]
    before = lora_digest(ck["lora"])
    st["L_diag"] = torch.full_like(st["L_diag"], COLLAPSED_L_DIAG)
    if st.get("L_off") is not None:
        st["L_off"] = torch.zeros_like(st["L_off"])
    ck.setdefault("config", {})["screen4_meanhead"] = {"version": VERSION, "source": str(src), "L_diag": COLLAPSED_L_DIAG}
    if lora_digest(ck["lora"]) != before:
        raise SystemExit("the LoRA changed")
    torch.save(ck, dst)
    return {"lora_digest": before, "heads": ck["aux_vbll"].get("heads")}


def select_head(nll_by_head, heads):
    """HEAD_RULE. nll_by_head: [G] pooled dev NLL. -> index of the output head (variational heads only)"""
    cand = [g for g, h in enumerate(heads) if float(h["kl"]) > 0]
    if not cand:
        raise ValueError("no variational head in the grid")
    best = min(nll_by_head[g] for g in cand)
    tied = [g for g in cand if nll_by_head[g] - best < 1e-4]
    return min(tied, key=lambda g: (float(heads[g]["kl"]), heads[g]["cov"] != "diag"))


def control_head(heads, cov="full"):
    """the deterministic-head control: the kl-0 head of the given covariance family"""
    for g, h in enumerate(heads):
        if float(h["kl"]) == 0 and h["cov"] == cov:
            return g
    raise ValueError("no kl-0 head")


def pooled_nll(P_by_panel, T_by_panel, K_by_panel):
    """equal-weight mean over panels of each panel's unit-mean soft NLL; P [n, Kmax]"""
    vals = []
    for P, T, K in zip(P_by_panel, T_by_panel, K_by_panel):
        idx = np.arange(P.shape[1])[None, :] < K[:, None]
        lp = np.log(np.clip(np.where(idx, P, 1.0), 1e-12, 1))
        t = np.where(idx, T, 0.0)
        vals.append(float((-(t * lp).sum(1)).mean()))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("meanhead", "flags"))
    ap.add_argument("--in", dest="src")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "flags":
        print(trainer_flags())
        return 0
    if a.cmd == "meanhead":
        print("MEANHEAD_OK " + json.dumps(meanhead(a.src, a.out)))
        return 0
    ap.error("a command")


def self_test():
    h = heads_of_grid()
    assert len(h) == 8 and h[0] == {"prior": 1.0, "kl": 0.0, "cov": "diag"} and h[-1] == {"prior": 1.0, "kl": 1e6, "cov": "full"}
    nll = [0.50, 0.49, 0.48, 0.47, 0.47, 0.46, 0.47, 0.4600001]
    assert select_head(nll, h) == 5, select_head(nll, h)               # kl 1e4 full beats kl 1e6 full only within the tie -> smaller kl
    assert control_head(h, "full") == 1 and control_head(h, "diag") == 0
    P = [np.array([[0.7, 0.3, 0.0], [0.5, 0.5, 0.0]])]
    T = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
    K = [np.array([2, 2])]
    assert abs(pooled_nll(P, T, K) - (-(math.log(0.7) + math.log(0.5)) / 2)) < 1e-12
    print("SELF_TEST_OK screen4_head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
