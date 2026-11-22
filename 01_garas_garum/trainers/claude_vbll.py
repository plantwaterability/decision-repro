#!/usr/bin/env python3
"""A VARIATIONAL BAYESIAN LAST LAYER THAT CORRECTS THE LM'S OWN ANSWER DISTRIBUTION.

    z_o = tau * ell_o + phi_o^T beta,      q(beta) = N(mu, L L^T),   p(beta) = N(0, s I)

ell_o is the LM's letter logit (claude_features.py), which transfers zero-shot better than any head
trained on cached features. phi_o = OptionReadout([h_o; h_d; h_o * h_d]) centred across the option
set (as JEVHead does: a shared component cancels in the softmax but inflates the variance). beta is
the only random quantity; the readout and tau are point estimates, as in VBLL.

ELBO, per unit, with the soft human target t (Jensen: E[logsumexp] <= log E[sum exp]):

    E_q[ sum_o t_o log softmax(z)_o ] >= sum_o t_o m_o - log sum_o exp(m_o + v_o / 2)
    m_o = tau * ell_o + phi_o^T mu,   v_o = phi_o^T L L^T phi_o
    ELBO = mean_units(bound) - KL(q || p) / T,   T = distinct base records in train

The prior is centred on NO correction, so where the training corpus says little the predictive falls
back to the LM -- the behaviour wanted under the jev_data -> jev_real shift. Predictive, epistemic
(mutual information over beta) and aleatoric (expected entropy) come from Monte Carlo over q(beta);
rows that are option-order permutations of one unit are mapped back and averaged.

    python3 trainers/claude_vbll.py --train feats/jevdata_train.pt --val feats/jevdata_val.pt \
        --test feats/jevreal_test.pt --prior-scales 0.1,1,10 --out analysis/claude_vbll_zeroshot_units.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vbll  # noqa: E402


class LogitVBLL(nn.Module):
    def __init__(self, d_model, n_features=128, prior_scale=1.0):
        super().__init__()
        self.readout = vbll.OptionReadout(d_model, n_features, use_context=True)
        self.mu = nn.Parameter(torch.zeros(n_features))
        self.L_raw = nn.Parameter(torch.zeros(n_features, n_features))
        self.L_diag = nn.Parameter(torch.full((n_features,), math.log(math.expm1(math.sqrt(prior_scale) * 0.1))))
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.s = float(prior_scale)
        self.F = n_features

    def chol(self):
        return torch.tril(self.L_raw, -1) + torch.diag(F.softplus(self.L_diag) + 1e-6)

    def features(self, h_o, h_d, mask):
        phi = self.readout(h_o, context=h_d)
        m = mask.unsqueeze(-1).to(phi.dtype)
        return phi - (phi * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)

    def moments(self, ell, h_o, h_d, mask):
        phi = self.features(h_o, h_d, mask)
        mean = self.log_tau.exp() * ell + phi @ self.mu
        var = ((phi @ self.chol()) ** 2).sum(-1)
        return phi, mean, var

    def kl(self):
        L = self.chol()
        tr = (L ** 2).sum() / self.s
        quad = (self.mu ** 2).sum() / self.s
        logdet_q = 2 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * (tr + quad - self.F + self.F * math.log(self.s) - logdet_q)

    def bound(self, ell, h_o, h_d, t, mask):
        _, mean, var = self.moments(ell, h_o, h_d, mask)
        neg = torch.finfo(mean.dtype).min / 4
        lse = torch.logsumexp((mean + var / 2).masked_fill(~mask, neg), -1)
        return (t * mean.masked_fill(~mask, 0.0)).sum(-1) - lse

    @torch.no_grad()
    def predict(self, ell, h_o, h_d, mask, n_samples=64):
        phi, mean0, _ = self.moments(ell, h_o, h_d, mask)
        base = self.log_tau.exp() * ell
        eps = torch.randn(n_samples, self.F, device=ell.device)
        betas = self.mu + eps @ self.chol().T                               # [S, F]
        z = base.unsqueeze(0) + torch.einsum("bkf,sf->sbk", phi, betas)     # [S, B, K]
        z = z.masked_fill(~mask.unsqueeze(0), -1e9)
        ps = torch.softmax(z, -1)
        p = ps.mean(0)
        ent = lambda q: -(q * q.clamp_min(1e-12).log()).sum(-1)
        alea = ent(ps).mean(0)
        return p, ent(p) - alea, alea


def load(path, device):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d


def batches(d, idx, device):
    kmax = max(d["k"][i] for i in idx)
    D = d["h_d"][idx[0]].shape[-1]
    B = len(idx)
    ell = torch.zeros(B, kmax); t = torch.zeros(B, kmax)
    h_o = torch.zeros(B, kmax, D); h_d = torch.zeros(B, D)
    mask = torch.zeros(B, kmax, dtype=torch.bool)
    for b, i in enumerate(idx):
        k = d["k"][i]
        ell[b, :k] = d["ell"][i]; t[b, :k] = d["t"][i]
        h_o[b, :k] = d["h_o"][i].float(); h_d[b] = d["h_d"][i].float(); mask[b, :k] = True
    return ell.to(device), h_o.to(device), h_d.to(device), t.to(device), mask.to(device)


def fit(train, val, prior_scale, a, device):
    torch.manual_seed(a.seed)
    D = train["h_d"][0].shape[-1]
    T = len(set(train["base_id"]))
    head = LogitVBLL(D, a.n_features, prior_scale).to(device)
    opt = torch.optim.AdamW([
        {"params": head.readout.parameters(), "lr": a.lr_readout, "weight_decay": 0.01},
        {"params": [head.mu, head.L_raw, head.L_diag, head.log_tau], "lr": a.lr_vbll, "weight_decay": 0.0},
    ])
    n = len(train["k"])
    for ep in range(a.epochs):
        head.train()
        perm = torch.randperm(n).tolist()
        tot = 0.0
        for s in range(0, n, a.bs):
            idx = perm[s : s + a.bs]
            ell, h_o, h_d, t, m = batches(train, idx, device)
            loss = -(head.bound(ell, h_o, h_d, t, m).mean() - head.kl() / T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        print(f"    s={prior_scale:g} epoch {ep + 1}: -ELBO/unit {tot / n:.4f}  KL/T {float(head.kl()) / T:.5f}  "
              f"tau {float(head.log_tau.exp()):.3f}", flush=True)
    return head, T


@torch.no_grad()
def evaluate(head, d, device, bs=128, n_samples=64):
    head.eval()
    n = len(d["k"])
    P, EPI, ALEA, BND = [], [], [], []
    for s in range(0, n, bs):
        idx = list(range(s, min(n, s + bs)))
        ell, h_o, h_d, t, m = batches(d, idx, device)
        p, epi, alea = head.predict(ell, h_o, h_d, m, n_samples)
        BND.append(head.bound(ell, h_o, h_d, t, m).cpu())
        for b, i in enumerate(idx):
            P.append(p[b, : d["k"][i]].cpu().numpy())
        EPI.append(epi.cpu()); ALEA.append(alea.cpu())
    return P, torch.cat(EPI).numpy(), torch.cat(ALEA).numpy(), torch.cat(BND).numpy()


def per_unit(d, P, epi, alea):
    """Map each row's distribution back to the unit's original option order and average rows."""
    groups = {}
    for r in range(len(d["k"])):
        groups.setdefault(d["unit"][r], []).append(r)
    units = sorted(groups)
    kmax = max(d["k"][groups[u][0]] for u in units)
    n = len(units)
    p_out = np.zeros((n, kmax)); t_out = np.zeros((n, kmax))
    ks = np.zeros(n, dtype=np.int64); soft = np.zeros(n, dtype=bool)
    e_out = np.zeros(n); a_out = np.zeros(n); spread = np.zeros(n); qid = []
    for row, u in enumerate(units):
        rs = groups[u]
        k = d["k"][rs[0]]
        ps, ts = [], None
        for r in rs:
            o = np.asarray(d["order"][r])
            p = np.zeros(k); p[o] = P[r]; ps.append(p)
            if ts is None:
                ts = np.zeros(k); ts[o] = d["t"][r].numpy()
        ps = np.stack(ps)
        p_out[row, :k], t_out[row, :k] = ps.mean(0), ts
        ks[row], soft[row] = k, d["soft"][rs[0]]
        e_out[row] = np.mean([epi[r] for r in rs]); a_out[row] = np.mean([alea[r] for r in rs])
        spread[row] = ps.std(0).sum() if len(rs) > 1 else 0.0
        qid.append(d["qid"][rs[0]])
    return p_out, t_out, ks, soft, e_out, a_out, spread, np.array(qid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--prior-scales", default="0.1,1,10")
    ap.add_argument("--n-features", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr-readout", type=float, default=3e-4)
    ap.add_argument("--lr-vbll", type=float, default=3e-3)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = load(a.train, device), load(a.val, device), load(a.test, device)
    print(f"rows: train {len(train['k'])}  val {len(val['k'])}  test {len(test['k'])}", flush=True)

    # The like-for-like baseline: the LM's own letter distribution on EXACTLY these test rows and
    # option orders, so any difference below is the head's doing, not a change of rows or orders.
    Pb = [torch.softmax(test["ell"][r], -1).numpy() for r in range(len(test["k"]))]
    zeros = np.zeros(len(Pb))
    pb, tb, ksb, softb, _, alb, spb, qidb = per_unit(test, Pb, zeros, zeros)
    totb = -(pb * np.log(np.clip(pb, 1e-12, 1))).sum(1)
    base_out = a.out.replace(".npz", "_ellbase.npz")
    np.savez_compressed(base_out, test_p=pb, test_t=tb, test_ks=ksb, test_soft=softb, test_qid=qidb,
                        test_tot=totb, test_pred_ent=totb, test_mx=pb.max(1), test_perm_spread=spb)
    print(f"wrote LM-only baseline {base_out}", flush=True)

    # Prior scale by VALIDATION predictive NLL on jev_data -- never by jev_real.
    best = None
    for s in [float(x) for x in a.prior_scales.split(",")]:
        head, T = fit(train, val, s, a, device)
        P, epi, alea, bnd = evaluate(head, val, device, n_samples=a.n_samples)
        nll = float(np.mean([-(val["t"][r].numpy() * np.log(np.clip(P[r], 1e-12, 1))).sum()
                             for r in range(len(P))]))
        acc = float(np.mean([P[r].argmax() == val["t"][r].numpy().argmax() for r in range(len(P))]))
        elbo = float(bnd.mean()) - float(head.kl()) / T
        print(f"  prior s={s:g}: val ELBO/unit {elbo:.4f}  predictive NLL {nll:.4f}  acc {acc:.4f}  "
              f"mean epi {epi.mean():.5f}  mean alea {alea.mean():.4f}", flush=True)
        if best is None or nll < best[0]:
            best = (nll, s, head, elbo)
    nll, s, head, elbo = best
    print(f"selected prior s={s:g} by jev_data val NLL {nll:.4f} (val ELBO/unit {elbo:.4f})", flush=True)

    P, epi, alea, _ = evaluate(head, test, device, n_samples=a.n_samples)
    p, t, ks, soft, e, al, spread, qid = per_unit(test, P, epi, alea)
    tot = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    # OOD signal: is epistemic higher on the unseen benchmark than on the training corpus's val?
    _, epi_val, _, _ = evaluate(head, val, device, n_samples=a.n_samples)
    np.savez_compressed(a.out, test_p=p, test_t=t, test_ks=ks, test_soft=soft, test_qid=qid,
                        test_tot=tot, test_pred_ent=al, test_epi=e, test_mx=p.max(1),
                        test_perm_spread=spread, val_epi=epi_val,
                        prior_scale=s, val_elbo=elbo, val_nll=nll, tau=float(head.log_tau.exp()))
    print(f"wrote {a.out}: {len(p)} units; tau {float(head.log_tau.exp()):.3f}; "
          f"mean epi test {e.mean():.5f} vs jev_data val {epi_val.mean():.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
