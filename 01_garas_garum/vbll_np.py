"""Vectorised NumPy implementation of the JEV VBLL core.

Mirrors `vbll_ref.py` exactly, but batched so the Monte Carlo verification can use
millions of samples instead of tens of thousands. `test_vbll.py` asserts the two agree
elementwise, which is what lets the stdlib oracle stand in for the real thing.

Shapes
    phis      [K, N]   per-option features (K options, N feature dims)
    w_bar     [N]      variational posterior mean of the shared scorer
    P         [N, N]   lower-triangular Cholesky factor of the posterior covariance, S = P P^T
    p_target  [K]      target distribution over options

The head is *shared* across options, which is what makes the listwise softmax exactly
permutation-invariant and lets unseen option text be scored zero-shot.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "lse",
    "softmax",
    "entropy",
    "build_P",
    "covariance",
    "quad_cov",
    "logdet_chol",
    "kl_q_p",
    "sample_w",
    "bound",
    "predict",
    "probit_softmax",
]


# --------------------------------------------------------------------------- numerics
def lse(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy(p, axis=-1, eps=1e-300):
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, eps)), 0.0), axis=axis)


# --------------------------------------------------------------------------- parameters
def build_P(strict_lower, log_diag):
    """S = P P^T with P lower triangular, positive diagonal (paper Eq. 43)."""
    strict_lower = np.asarray(strict_lower, dtype=np.float64)
    log_diag = np.asarray(log_diag, dtype=np.float64)
    n = log_diag.shape[0]
    P = np.tril(strict_lower, -1)
    P[np.arange(n), np.arange(n)] = np.exp(log_diag)
    return P


def covariance(P):
    return P @ P.T


def quad_cov(P, phis):
    """phi^T S phi = ||P^T phi||^2 for each row of `phis`, in O(K N^2) (paper App. C)."""
    F = np.asarray(phis, dtype=np.float64) @ P
    return np.sum(F * F, axis=-1)


def logdet_chol(P):
    return 2.0 * np.sum(np.log(np.diag(P)))


def kl_q_p(w_bar, P, prior_scale=1.0):
    """KL( N(w_bar, S) || N(0, s I) ), paper Eq. 46-49 with N_y = 1."""
    w_bar = np.asarray(w_bar, dtype=np.float64)
    n = w_bar.shape[0]
    s = float(prior_scale)
    trS = float(np.sum(P * P))
    return 0.5 * (
        float(np.dot(w_bar, w_bar)) / s + trS / s - n + n * math.log(s) - logdet_chol(P)
    )


def sample_w(w_bar, P, n_samples, rng):
    """[M, N] reparameterised samples w = w_bar + P eps, eps ~ N(0, I)."""
    w_bar = np.asarray(w_bar, dtype=np.float64)
    n = w_bar.shape[0]
    eps = rng.standard_normal((n_samples, n))
    return w_bar[None, :] + eps @ P.T


# --------------------------------------------------------------------------- model
def bound(phis, p_target, w_bar, P, sigma2=0.0):
    """Soft-target VBLL bound for one example (paper Eq. 14, generalised to p*).

        L = sum_o p*_o zbar_o - LSE_o( zbar_o + 1/2 (phi_o^T S phi_o + sigma^2) )

    A convex combination of per-option lower bounds, so L lower-bounds
    E_{o ~ p*}[ log p(o | x) ] under the variational posterior; reduces to Eq. 14 for
    one-hot p*. Verified against Monte Carlo in test_vbll.py.
    """
    phis = np.asarray(phis, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)
    zbar = phis @ np.asarray(w_bar, dtype=np.float64)
    v = quad_cov(P, phis)
    penalty = 0.5 * (v + float(sigma2))
    denom = lse(zbar + penalty)
    lb = zbar - denom
    return {
        "bound": float(p_target @ lb),
        "zbar": zbar,
        "var": v,
        "lb": lb,
        "lse": float(denom),
    }


def probit_softmax(z, sigma2):
    """E_eps[ softmax(z + eps) ] under eps ~ N(0, sigma^2) via the probit approximation,

        E[softmax] ~ softmax( z / sqrt(1 + pi sigma^2 / 8) )

    Used on the serving hot path; test_vbll.py checks it against an inner Monte Carlo.
    """
    if sigma2 <= 0.0:
        return softmax(z, axis=-1)
    return softmax(np.asarray(z, dtype=np.float64) / math.sqrt(1.0 + math.pi * sigma2 / 8.0), axis=-1)


def predict(phis, w_bar, P, n_samples=2000, seed=0, sigma2=0.0, n_noise=64, chunk=256):
    """Posterior predictive with the two uncertainty sources kept separate.

    Marginalise the logit noise *first*, per weight sample, so the label-noise channel
    does not leak into the concavity gap:

        p_W(o) = E_eps[ softmax(z_W + eps) ]
        Total      = H( E_W[p_W] )
        Aleatoric  = E_W[ H(p_W) ]
        Epistemic  = Total - Aleatoric = I(o ; W | x)     (mutual information)

    Entropy is concave, so Total >= Aleatoric and the epistemic term is non-negative up
    to Monte Carlo noise.

    `n_noise > 0` (the default) is the accurate inner Monte Carlo for the eps expectation.
    `n_noise = 0` selects the probit approximation instead, which is faster and
    deterministic but degrades as sigma^2 grows -- measured max abs error against a
    30k-sample Monte Carlo is ~0.004 at sigma^2=0.1, ~0.02 at 0.5, ~0.03 at 1.0 (test [10]).
    Probit shrinks logits toward zero, so it is also biased toward *higher* entropy in the
    near-uniform regime (~0.1 nats over 64 options at sigma^2=0.5, test [12]); small
    per-element probability error does not imply small entropy error. Use it only on a
    serving hot path with small logit noise; the reported uncertainty decomposition must
    come from the inner Monte Carlo.
    """
    phis = np.asarray(phis, dtype=np.float64)
    w_bar = np.asarray(w_bar, dtype=np.float64)
    K = phis.shape[0]
    rng = np.random.default_rng(seed)

    p_sum = np.zeros(K)
    H_acc = 0.0
    p_samples = np.empty((n_samples, K))
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        W = sample_w(w_bar, P, m, rng)          # [m, N]
        z = W @ phis.T                          # [m, K]
        if sigma2 > 0.0 and n_noise > 0:
            # Inner expectation: independent per-option Gaussian logit noise.
            noise = rng.standard_normal((m, n_noise, K)) * math.sqrt(sigma2)
            pW = softmax(z[:, None, :] + noise, axis=-1).mean(axis=1)   # [m, K]
        else:
            pW = probit_softmax(z, sigma2)
        p_samples[done : done + m] = pW
        p_sum += pW.sum(axis=0)
        H_acc += float(entropy(pW, axis=-1).sum())
        done += m

    p_mean = p_sum / n_samples
    H_total = float(entropy(p_mean))
    H_aleatoric = H_acc / n_samples
    return {
        "p_mean": p_mean,
        "H_total": H_total,
        "H_aleatoric": H_aleatoric,
        "I_epistemic": H_total - H_aleatoric,
        "var": p_samples.var(axis=0),
        "p_samples": p_samples,
    }
