"""PyTorch production module for the JEV decision head.

This is the trained model. It mirrors `vbll_np.py` exactly -- the NumPy path is the
verified reference, so any divergence here is a bug, not a design choice.

Architecture, in one line: a *shared* binary variational-Bayesian last layer scores each
option independently, and a softmax couples the scores into a distribution.

    h_o   = Qwen3.8-27B hidden state at option o's readout marker      [K, d_model]
    phi_o = MLP(h_o)                                                   [K, N_phi]
    z_o   ~ N(w^T phi_o, phi_o^T S phi_o)      q(w) = N(w_bar, P P^T)
    p(o)  = softmax_o( z )_o

Why shared: the option set's size and identity change per example (2..255 options, free
text). A fixed-width softmax head cannot express that. A shared per-option scorer can,
and because the listwise softmax of a shared scorer is a symmetric function, the model is
*exactly* permutation-invariant in option order -- so option reshuffling is a true
invariance rather than an augmentation, and options never seen in training are scored
zero-shot.

Training objective (paper Eq. 14 generalised to distributional targets; see vbll_ref.py
for the derivation and test_vbll.py for the numerical verification):

    L = sum_o p*_o zbar_o - LSE_o( zbar_o + 1/2 (phi_o^T S phi_o + sigma^2) )
    loss = -L + KL(q(w) || N(0, s I)) / T

Uncertainty, reported per decision (mutual information decomposition):

    p_W(o) = E_eps[ softmax(z_W + eps) ]
    Total      = H( E_W[p_W] )
    Aleatoric  = E_W[ H(p_W) ]
    Epistemic  = Total - Aleatoric = I(o ; W | x)

The logit noise is marginalised *first*, per weight sample. Sampling weight and noise
jointly and calling E[H(p)] "aleatoric" inverts the sigma^2 monotonicity, because entropy
is concave -- see the note in vbll_np.predict.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "lower_triangular",
    "VBLLScorer",
    "OptionReadout",
    "AleatoricHead",
    "JEVHead",
]


def lower_triangular(strict_lower: torch.Tensor, log_diag: torch.Tensor) -> torch.Tensor:
    """P = strict_lower + diag(exp(log_diag)), lower triangular, positive diagonal.

    Paper Eq. 43. Parameterising P directly rather than S means the forward pass never
    calls a Cholesky decomposition: logdet(S) = 2 sum(log diag P) and tr(S) = ||P||_F^2.
    """
    n = log_diag.shape[-1]
    P = torch.tril(strict_lower, diagonal=-1).clone()
    idx = torch.arange(n, device=P.device)
    P[..., idx, idx] = torch.exp(log_diag)
    return P


class VBLLScorer(nn.Module):
    """Shared binary discriminative VBLL (Harrison, Willes & Snoek, ICLR 2024, Eq. 14/43).

    One row of weights, applied to every option of every question. The "classes" of the
    paper's multiclass formulation are the options themselves, which is what lets the
    label space vary per example.
    """

    def __init__(
        self,
        n_features: int,
        prior_scale: float = 1.0,
        init_log_cov_diag: float = math.log(0.01),
        init_log_sigma2: float = math.log(0.05),
        learn_noise: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.prior_scale = float(prior_scale)
        self.learn_noise = bool(learn_noise)

        self.w_bar = nn.Parameter(torch.zeros(n_features))
        self.P_strict = nn.Parameter(torch.zeros(n_features, n_features))
        self.log_diag = nn.Parameter(torch.full((n_features,), float(init_log_cov_diag)))
        if self.learn_noise:
            self.log_sigma2 = nn.Parameter(torch.tensor(float(init_log_sigma2)))
        else:
            self.register_buffer("log_sigma2", torch.tensor(float(init_log_sigma2)))

    # ------------------------------------------------------------------ pieces
    def chol(self) -> torch.Tensor:
        return lower_triangular(self.P_strict, self.log_diag)

    @property
    def sigma2(self) -> torch.Tensor:
        """Gaussian logit-noise variance (paper's sigma^2).

        With a SHARED scorer this parameter cannot learn anything, so `JEVHead` passes
        `learn_noise=False` and it is frozen at zero there. **The CLASS default is
        `learn_noise=True`, so `VBLLScorer(...)` on its own has a LIVE channel** (`sigma2` starts
        at `softplus(log 0.05) = 0.0488`). The docstring used to say "frozen at zero by default"
        flatly, which was false for the class and true only for the one caller -- and a reader who
        instantiated `VBLLScorer` directly would have optimised it believing it was inert.
        sigma^2 enters the bound only inside the LSE, once per option with the same
        value, and a constant added to every logit cancels in the softmax -- so the bound
        collapses to exactly `... - sigma^2/2`, with dL/d(sigma^2) = -1/2 and no
        counteracting term. Every gradient step drives it to its floor, and the "learned
        label-noise channel" never learns.

        The paper's multiclass model escapes this because sigma_k^2 is *per class*; our
        classes are the options themselves, so there is no per-class noise to learn. A
        heteroscedastic head would collapse identically under this Jensen bound, because the
        bound makes noise purely punitive -- learning noise that HELPS on hard examples needs
        a Monte Carlo log E[softmax] likelihood instead.

        Nothing is lost by freezing it: soft labels already spread the predictive, and
        AleatoricHead carries the human-disagreement signal with direct supervision.
        """
        if not self.learn_noise:
            return torch.zeros((), dtype=self.w_bar.dtype, device=self.w_bar.device)
        return F.softplus(self.log_sigma2)

    def quad_form(self, phis: torch.Tensor) -> torch.Tensor:
        """phi^T S phi = ||P^T phi||^2, O(K N^2) (paper App. C). phis: [..., K, N]."""
        F_ = phis @ self.chol()
        return (F_ * F_).sum(dim=-1)

    def kl(self) -> torch.Tensor:
        """KL( N(w_bar, S) || N(0, s I) ), paper Eq. 46-49 with N_y = 1."""
        n = self.n_features
        s = self.prior_scale
        P = self.chol()
        trS = (P * P).sum()
        logdet_S = 2.0 * torch.log(torch.diagonal(P)).sum()
        return 0.5 * (
            (self.w_bar * self.w_bar).sum() / s + trS / s - n + n * math.log(s) - logdet_S
        )

    # ------------------------------------------------------------------ training
    def bound(
        self,
        phis: torch.Tensor,
        p_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The soft-target VBLL bound, mean over the batch.

        phis     [B, K, N]  per-option features
        p_target [B, K]     target distribution over options (rows sum to 1)
        mask     [B, K]     bool, True for real (non-padding) options
        """
        zbar = torch.einsum("bkn,n->bk", phis, self.w_bar)
        v = self.quad_form(phis)
        penalty = 0.5 * (v + self.sigma2)

        if mask is not None:
            neg = torch.finfo(zbar.dtype).min
            lse = torch.logsumexp(torch.where(mask, zbar + penalty, torch.full_like(zbar, neg)), dim=-1)
        else:
            lse = torch.logsumexp(zbar + penalty, dim=-1)

        lb = zbar - lse.unsqueeze(-1)
        per_example = (p_target * lb).sum(dim=-1)
        return {"bound": per_example.mean(), "zbar": zbar, "var": v, "lb": lb, "lse": lse}

    def loss(
        self,
        phis: torch.Tensor,
        p_target: torch.Tensor,
        mask: torch.Tensor | None = None,
        n_train: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training loss = -bound + KL/T. `n_train` is T (the dataset size); when omitted
        the KL is scaled by the batch size, which is the usual mini-batch approximation."""
        out = self.bound(phis, p_target, mask)
        b = phis.shape[0]
        T = float(n_train) if n_train else float(b)
        kl = self.kl() / T
        out["kl"] = kl
        out["loss"] = -out["bound"] + kl
        return out

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(
        self,
        phis: torch.Tensor,
        n_samples: int = 64,
        n_noise: int = 0,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Posterior predictive with the mutual-information decomposition.

        `n_noise > 0` uses an inner Monte Carlo for E_eps[softmax]; `n_noise == 0` uses the
        probit approximation (fast, deterministic, biased toward uniformity in entropy --
        see vbll_np.predict). The decomposition should always use the inner Monte Carlo.
        """
        B, K, _ = phis.shape
        P = self.chol()
        # The LOGIT variance phi^T S phi, separately from the predictive. This is the factor
        # the mutual information multiplies the margin by, so measuring it on its own is what
        # separates "the posterior is uninformative" from "the posterior is informative but
        # swamped by the margin term".
        logit_var = self.quad_form(phis)
        # Prior variance for the same features, s * ||phi||^2. The ratio logit_var/prior_var is
        # the posterior-health check: the projector is trained under a bound that PENALISES
        # variance, so it can push phi into directions where S is small. If this ratio is tiny
        # the posterior has effectively collapsed and there is nothing left to measure.
        prior_var = self.prior_scale * (phis.float() ** 2).sum(dim=-1)
        eps = torch.randn(n_samples, self.n_features, device=phis.device, dtype=phis.dtype)
        W = self.w_bar.unsqueeze(0) + eps @ P.T                      # [M, N]
        z = torch.einsum("bkn,mn->bmk", phis, W)                     # [B, M, K]

        if n_noise > 0 and self.sigma2 > 0:
            sd = self.sigma2.sqrt()
            noise = torch.randn(B, n_samples, n_noise, K, device=phis.device, dtype=phis.dtype) * sd
            pW = F.softmax(z.unsqueeze(2) + noise, dim=-1).mean(dim=2)   # [B, M, K]
        else:
            scale = (1.0 + math.pi * float(self.sigma2) / 8.0) ** 0.5
            pW = F.softmax(z / scale, dim=-1)

        if mask is not None:
            pW = pW * mask.unsqueeze(1)
            pW = pW / pW.sum(dim=-1, keepdim=True).clamp_min(1e-30)

        p_mean = pW.mean(dim=1)                                      # [B, K]
        H_total = -(p_mean.clamp_min(1e-30).log() * p_mean).sum(dim=-1)
        H_per_w = -(pW.clamp_min(1e-30).log() * pW).sum(dim=-1)       # [B, M]
        H_aleatoric = H_per_w.mean(dim=1)

        # Epistemic uncertainty as a MEAN KL, not as a difference of entropies:
        #     I(o;W|x) = H(p_mean) - E_m[H(p_W)] = E_m[ KL(p_W=m || p_mean) ]
        # The two are algebraically identical (expand the KL and use E_m[p_m] = p_mean, making
        # the cross term sum to -H(p_mean)). The KL form is what to compute: every term is
        # non-negative and evaluated directly, so the answer is not recovered by subtracting
        # two large, nearly equal entropies.
        #
        # This matters at the magnitudes we actually observe. Mean epistemic is ~3e-4 nats
        # against mean aleatoric ~0.55 nats in the gate run, so the subtraction form loses
        # log2(0.55/3e-4) ~ 11 of float32's 24 significand bits to cancellation. At that ratio
        # the difference form also stops working at all in bf16, whose 7 explicit mantissa bits
        # give an ulp of 2^-8 ~ 3.9e-3 on [0.5, 1): the entire signal is about a thirteenth of
        # one ulp. The KL form has no such floor. H_total/H_aleatoric are retained because the
        # calibration and error-detection metrics are defined on them.
        log_p_mean = p_mean.clamp_min(1e-30).log().unsqueeze(1)                 # [B, 1, K]
        kl_per_w = (pW * (pW.clamp_min(1e-30).log() - log_p_mean)).sum(dim=-1)  # [B, M]
        I_epistemic = kl_per_w.mean(dim=1)

        return {
            "p_mean": p_mean,
            "H_total": H_total,
            "H_aleatoric": H_aleatoric,
            "I_epistemic": I_epistemic,
            # Kept only so the identity above can be asserted in the tests; not used downstream.
            "I_epistemic_subtraction": H_total - H_aleatoric,
            "var": pW.var(dim=1),
            "logit_var": logit_var,
            "prior_var": prior_var,
        }


class OptionReadout(nn.Module):
    """Qwen hidden state -> VBLL feature.

    The paper's LLM-features experiment (App. D.6) puts two linear layers between a frozen
    LM's last-token embedding and the head. Same here, plus a LayerNorm, because the
    Gated DeltaNet stack's final hidden states are not zero-mean across depth.

    `use_context` fuses the decision readout h_d into every option's feature:

        phi_o = MLP([h_o ; h_d ; h_o * h_d])

    This is not cosmetic. h_o sits at option o's `</option>` marker, and the backbone is
    CAUSAL -- so option 1's readout has seen none of its competitors. Without h_d the only
    listwise coupling is the softmax, and the features cannot encode a comparison, which is
    exactly what a question like "which option is best" needs. h_d sits after every option
    and has attended to all of them. This is also the cleanest explanation of the measured
    order sensitivity (worst cosine 0.41): features that had not seen the list cannot be
    invariant to its order.

    N_phi is deliberately small (256-512, not d_model=5120). The VBLL's covariance is
    N_phi x N_phi and the KL carries an N_phi log-determinant; at 5120 that is ~26M
    parameters for a single row, at 256 it is free.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int = 256,
        hidden: int | None = None,
        dropout: float = 0.0,
        use_context: bool = False,
    ):
        super().__init__()
        hidden = hidden or max(n_features, d_model // 8)
        self.use_context = bool(use_context)
        in_dim = 3 * d_model if self.use_context else d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_features),
        )
        self.dropout = nn.Dropout(dropout)

    def _fuse(self, h: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if self.use_context:
            if context is None:
                raise ValueError(
                    "OptionReadout(use_context=True) needs the decision readout h_d; "
                    "pass context=h_decision"
                )
            c = context.unsqueeze(1).expand_as(h)
            return torch.cat([h, c, h * c], dim=-1)
        return h

    def forward(self, h: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.dropout(self.net(self._fuse(h, context)))

    def encode(self, h: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """The trunk output before the final projection: [B, K, hidden].

        This is the representation the readout learns *for the decision*, so an auxiliary head
        consuming it is jointly optimised with the scorer rather than probing a frozen input.
        The aleatoric head previously took the cached `h_decision` straight from the shards,
        which nothing optimises for it -- it reached r = +0.73 against human disagreement
        anyway, on a representation chosen entirely for a different objective.
        """
        return self.net[:3](self._fuse(h, context))     # Linear, GELU, LayerNorm


class AleatoricHead(nn.Module):
    """Predicts the entropy of the *human* label distribution for a (state, question).

    The loader's soft labels are annotator histograms, so they measure irreducible
    disagreement directly. The VBLL posterior gives the epistemic term; this head gives the
    aleatoric term a supervised target instead of leaving it to the logit-noise channel
    to absorb.

    Output is bounded in [0, max_entropy]. **`max_entropy` must match the scale of the target.**
    The loader NORMALISES its entropy target by `log K` so that a binary question and a 5-level
    question sit on the same 0..1 scale, so the target is in [0, 1] and this head must be too.

    This was wrong: the constructor defaulted to `log(255) = 5.54` nats, so the head was asked to
    fit a target 5.5x smaller than its output range. The symptom was unmistakable once looked for
    -- `corr(pred_entropy, target_entropy)` came out at **-0.047**, i.e. the aleatoric head had no
    positive relationship to its own supervised target, while the aleatoric term is the dominant
    one (mean 0.62 nats against epistemic 0.0004). Any "confidence" built on it was therefore not
    tracking human disagreement at all.
    """

    def __init__(self, d_model: int, hidden: int = 512, max_entropy: float = 1.0):
        super().__init__()
        self.max_entropy = float(max_entropy)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_decision: torch.Tensor) -> torch.Tensor:
        """h_decision [B, d_model] -> predicted normalised entropy [B], in [0, max_entropy]."""
        return self.max_entropy * torch.sigmoid(self.net(h_decision).squeeze(-1))


class DirichletAleatoricHead(nn.Module):
    """Predicts the parameters of a Dirichlet over the OPTION simplex -- the whole human vote
    distribution, not just its entropy.

    WHY THIS REPLACES THE SCALAR HEAD. `AleatoricHead` regresses a single number, the normalised
    entropy of the human vote share, through a sigmoid under an MSE loss. On the external benchmark
    that head is degenerate on 11 of 17 configs, with a within-config correlation of exactly
    0.0000 -- it emits a constant per config. That is not a dataset property, it is what MSE on a
    scalar does: the loss is minimised by the conditional MEAN, so where the per-item signal is
    weak the cheapest way to reduce loss is to output the config's average entropy and stop. The
    sigmoid makes it stickier still, because a saturated sigmoid has vanishing gradient and cannot
    move off the constant it settled on.

    The Dirichlet head removes the failure mode by changing the TASK. It predicts one
    concentration per option, `alpha [B, K]`, and is trained on the Dirichlet-multinomial
    log-likelihood of the recovered annotator counts:

        -log p(n | alpha) = -[ lgamma(A) - lgamma(A+N) + sum_i ( lgamma(a_i+n_i) - lgamma(a_i) ) ]

    with a = alpha, A = sum a, N = sum n. That is a strictly PROPER scoring rule: its expectation
    is minimised, uniquely, at the true predictive distribution, so the Bayes-optimal response is
    to report the truth rather than the mean. It is also strictly harder to satisfy with a
    constant, because a constant must account for every option's vote count rather than one
    summary number -- and agreement becomes a DERIVED quantity rather than a regression target.

    Two further properties fall out for free:
      * K varies per unit in this corpus (2 to 28+). A per-option head is K-agnostic; a scalar head
        had to be told K in advance.
      * The full predictive over vote shares is available, so the DISAGREEMENT estimate can be
        scored for calibration, not merely rank-correlated.

    AGREEMENT IS NOT ALPHA0 -- the first draft of this class said it was, and the test suite
    caught the error. `alpha0 = sum alpha` is the Dirichlet's equivalent sample size: it measures
    how CONFIDENT the model is about the vote share, not how much the annotators agree. The two
    are close to opposite for the extremes that matter:

        alpha = [50, 50, 50]      -> votes are certainly EVENLY SPLIT   -> maximum disagreement
        alpha = [0.001, 0.001, .001] -> votes are certainly one option, but not which -> unanimous

    Both have small |E[H(p)] - H(mean)| structure that alpha0 alone gets backwards: the first is
    near-uniform votes, the second near-unanimous ones. The quantity that actually tracks
    agreement is the **expected entropy of the vote distribution**,

        agreement = 1 - E_{p~Dir(alpha)}[H(p)] / log K

    which is dominated by the entropy of the Dirichlet MEAN, alpha/A, with alpha0 entering only as
    a second-order correction to the spread around it. `normalized_agreement` computes exactly
    this, and `expected_entropy` is its closed-form core. The test
    `test_agreement_bounds` pins the two extremes so this cannot silently invert again.

    NOTE ON THE TWO SCALE KNOBS, corrected after the tests caught a wrong first draft:
      * `alpha = alpha_min + alpha_scale * softplus(z)` and **softplus is UNBOUNDED**, so
        `alpha_scale` does NOT cap how sharp the head can get. It sets the gradient scale at
        `z ~ 0`: too small and the likelihood cannot move alpha off the floor in a useful number
        of steps, too large and early updates are violent. It is a conditioning knob, not a
        capacity knob.
      * `alpha_min` is the floor, and it is the knob with a real representational consequence:
        with every alpha >= 1 the head can never express "the votes are concentrated on one
        option, but I do not know which" -- the sparse regime needs alpha < 1. At alpha_min = 1
        the most spread-out predictive available has E[H] = H(uniform), which is fine, but a
        genuinely *unanimous* item is reached by driving the winning alpha large (zero counts
        contribute nothing to the likelihood), so alpha_min = 1 does not block agreement -- only
        the "confidently unpredictable" state. Lower it if that state matters.
    In the tighter corners, only `normalized_agreement` should be compared against the scalar
    head's target, because it is defined on the same 0..1 scale.
    """

    def __init__(self, d_model: int, hidden: int = 512, alpha_scale: float = 8.0,
                 alpha_min: float = 1.0, unit_bias: bool = False):
        super().__init__()
        self.alpha_scale = float(alpha_scale)
        self.alpha_min = float(alpha_min)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # A dedicated PER-UNIT path into `z`. `h_decision` is already broadcast against every
        # option above, so in principle `net` can read it -- but it must re-derive it through a map
        # applied per option, where the option-specific component dominates and the unit-level
        # signal is diluted across K rows.
        #
        # That asymmetry is measurable and it is why this exists: a convex ridge on `h_decision`
        # ALONE predicts per-item human entropy at r = 0.514 with held-out R2 = 0.264
        # (`analysis/representation_ceiling.py`, 40,201 units, base_id-disjoint CV), while the
        # option count alone explains nothing (R2 = 0.003). The ridge solves a UNIT-level task --
        # one number per unit from one unit vector -- whereas the Dirichlet head must decompose the
        # same target into K per-option concentrations. `unit_bias` gives the unit-level signal an
        # un-diluted route to every option's concentration, which is exactly the shape of the
        # quantity `alpha` is being asked to summarise.
        #
        # The last layer is ZERO-INITIALISED, so enabling this reproduces the previous model
        # EXACTLY at step 0 and the comparison is a pure A/B rather than a new random draw.
        self.unit_bias = bool(unit_bias)
        if self.unit_bias:
            self.unit_bias_net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.unit_bias_net[-1].weight)
            nn.init.zeros_(self.unit_bias_net[-1].bias)

    def forward(self, h_options: torch.Tensor, h_decision: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """h_options [B,K,d], h_decision [B,d] -> alpha [B,K], every entry > 0.

        `h_decision` is broadcast against every option deliberately: option o's own readout at its
        `</option>` marker has not seen its competitors under a causal backbone, so the comparison
        signal must enter through the decision readout, which has. This is the same argument
        `OptionReadout.use_context` makes for the VBLL features.
        """
        hd = h_decision.unsqueeze(1).expand_as(h_options)
        z = self.net(torch.cat([h_options, hd], dim=-1)).squeeze(-1)       # [B, K]
        if self.unit_bias:
            # [B,1] broadcast over options: the unit-level entropy signal, added rather than
            # re-derived per option. Zero at init, so step 0 equals the head without it.
            z = z + self.unit_bias_net(h_decision)
        alpha = self.alpha_min + self.alpha_scale * F.softplus(z)
        if mask is not None:
            # A padded slot must not contribute to A = sum alpha, or the agreement statistic
            # would depend on how many phantom slots the batch happened to allocate.
            alpha = alpha.masked_fill(~mask, self.alpha_min)
        return alpha

    @staticmethod
    def nll(alpha: torch.Tensor, counts: torch.Tensor,
            mask: torch.Tensor | None = None) -> torch.Tensor:
        """Dirichlet-multinomial NLL, mean over the rows that actually carry votes.

        A strictly proper scoring rule in alpha, so minimising it is the right objective and not
        merely a convenient one.

        Two masking rules, and getting either wrong makes the loss depend on how the batch was
        padded rather than on the data (`test_nll_padding.py` pins both):

        1. **`A` counts REAL options only.** `forward` already lowers padded slots to
           `alpha_min` for exactly this reason -- its comment says a phantom slot "must not
           contribute to A = sum alpha, or the agreement statistic would depend on how many
           phantom slots the batch happened to allocate". The per-entry `lgamma` terms need a
           sentinel of 1.0 on masked slots to stay finite (`lgamma(0)` is inf), but summing that
           sentinel into `A` undid the guard: a 3-option unit in a batch padded to 18 trained
           against `A + 15`, flattening the likelihood by `A_padded / A_true` -- measured at
           6.50 -> 21.50, tripling the loss, in that test.
        2. **Fully masked rows are excluded, not averaged in as zeros.** A hard-labelled unit
           under the entropy guard has every entry masked, so it observes nothing; averaging its
           0 into the batch scales the objective (and its gradient) by the batch's soft
           fraction. Measured: adding one such row halved the loss.
        """
        a, n = alpha, counts
        if mask is not None:
            a = a.masked_fill(~mask, 1.0)          # sentinel: keeps lgamma(a) finite, n = 0
            n = n.masked_fill(~mask, 0.0)
            A = alpha.masked_fill(~mask, 0.0).sum(-1, keepdim=True)
        else:
            A = a.sum(-1, keepdim=True)
        N = n.sum(-1, keepdim=True)
        # A row with no observed votes has A = 0 and N = 0, so `lgamma(A) - lgamma(A + N)` is
        # inf - inf. Drop such rows from the mean instead of giving them a 0 -- see rule 2.
        obs = N.squeeze(-1) > 0
        if not bool(obs.any()):
            return torch.zeros((), device=alpha.device, dtype=alpha.dtype)
        ll = (torch.lgamma(A) - torch.lgamma(A + N)
              + (torch.lgamma(a + n) - torch.lgamma(a)).sum(-1, keepdim=True))
        return -ll[obs].mean()

    @staticmethod
    def alpha_prior_nll(alpha: torch.Tensor, mask: torch.Tensor | None = None,
                        shape: float = 2.0, theta: float = 2.0) -> torch.Tensor:
        """Negative log of a Gamma(shape, rate=1/theta) prior on each concentration, up to constants.

        WHY THIS EXISTS. `alpha = alpha_min + alpha_scale * softplus(z)` is UNBOUNDED above, and the
        Dirichlet-multinomial likelihood is strictly improved on the TRAINING data by sharpening
        `alpha` without limit -- the maximum-likelihood answer is infinite concentration. Nothing in
        the objective penalises that, so early stopping was doing all the regularisation. Measured
        (TODO R-40): 30 epochs instead of 6 drove val NLL **0.8075 -> 1.8048** and test ECE
        **0.0533 -> 0.3888** while the training loss kept falling. The head was not converging; it
        was overfitting into overconfidence, and the unbounded parameterisation is why.

        A Gamma prior on a Dirichlet's concentration is the standard conjugate choice, and with
        shape `k` and rate `1/theta` its negative log density is `-(k-1)*log a + a/theta`. **Both
        terms earn their place:** `a/theta` alone pulls every concentration toward zero (i.e. toward
        uniform, destroying the sharp predictions the head exists to make), and `-(k-1)*log a`
        alone pushes away from zero. At `k = 2` the mode sits at `theta`, so `theta` is the
        concentration the prior believes in and the `-log a` term is what keeps the head able to be
        confident. `theta = 2.0` matches the corpus scale: counts total ~3 raters over K options, so
        a per-option concentration near 1 is the natural magnitude and one rater's worth of evidence
        is 1.

        Masked slots do not vote on the prior either -- the same rule `nll` follows, for the same
        reason: a phantom slot must not change the objective by existing.
        """
        if mask is None:
            a = alpha.clamp_min(1e-6)
            return (-(shape - 1.0) * a.log() + a / theta).mean()
        # Sentinel 1.0 only so `log` stays finite; those entries are then zeroed before the mean.
        a = alpha.masked_fill(~mask, 1.0)
        t = (-(shape - 1.0) * a.log() + a / theta).masked_fill(~mask, 0.0)
        denom = mask.sum(-1).clamp_min(1).to(t.dtype)
        return (t.sum(-1) / denom).mean()

    @staticmethod
    def mean(alpha: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """The Dirichlet mean, alpha/A -- the predicted vote share."""
        a = alpha if mask is None else alpha.masked_fill(~mask, 0.0)
        return a / a.sum(-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def expected_entropy(alpha: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
        """E[H(p)] under Dirichlet(alpha): psi(A+1) - sum_i (a_i/A) psi(a_i+1). Exact, no sampling.

        The same closed form as `bayes_entropy_nats`, with the model's alpha in place of an
        empirical posterior -- so what is summarised is the head's own uncertainty about the vote
        share, which is what "aleatoric" is supposed to mean here. Cheap enough to use as the
        training-time diagnostic, unlike a Monte Carlo estimate.

        Masked entries are zeroed rather than set to a tiny epsilon: digamma(0+1) = -0.577 is
        finite, so `(0/A) * digamma(1)` is exactly 0, whereas a tiny epsilon makes the ratio 0
        and digamma(~1) finite but the product a 0*inf that would poison the batch with NaN.
        """
        a = alpha if mask is None else alpha.masked_fill(~mask, 0.0)
        A = a.sum(-1)
        # A FULLY MASKED row has A = 0, and `digamma(0 + 1) = -0.5772` would be returned as a
        # NEGATIVE expected entropy -- and `normalized_agreement` is `1 - E[H]/log K`, so it would
        # return `1 + 0.5772/log K`, i.e. an agreement ABOVE 1. Since the Dirichlet branch no
        # longer applies `entropy_mask` (see R-37), a hard-labelled or all-padding row reaches
        # here, so this is reachable rather than theoretical. Zero is the honest value: no real
        # options means no measured spread, not negative spread.
        safe = A.clamp_min(1e-12)
        ent = torch.digamma(A + 1.0) - ((a / safe.unsqueeze(-1))
                                        * torch.digamma(a + 1.0)).sum(-1)
        return torch.where(A > 0, ent, torch.zeros_like(ent))

    @staticmethod
    def normalized_agreement(alpha: torch.Tensor, mask: torch.Tensor | None = None,
                             k: int | None = None) -> torch.Tensor:
        """1 - E[H]/log K, on the same 0..1 scale as the scalar head's target.

        This is the quantity to correlate against human agreement, so that the Dirichlet head is
        scored against EXACTLY the statistic the scalar head was scored against. Comparing a new
        architecture on a new metric would not be a comparison.

        **K is the unit's REAL option count, not the padded width.** `alpha.shape[-1]` is the
        batch's maximum, so using it divides a 3-option unit by `log 18` instead of `log 3` and
        reports an agreement that is wrong by a factor of `log 3 / log 18 = 0.38` -- a scale
        error that silently shrinks every reported agreement for short units while leaving the
        long ones right, which is worse than a uniform bias. With a mask, take `K` per row from
        the mask. An explicit `k` still wins, for callers scoring a fixed-K collection.
        """
        a = alpha if mask is None else alpha.masked_fill(~mask, 0.0)
        if k is not None:
            logk = torch.full((a.shape[0],), float(math.log(max(k, 2))),
                              device=a.device, dtype=a.dtype)
        elif mask is not None:
            logk = mask.sum(-1).clamp_min(2).to(a.dtype).log()
        else:
            logk = torch.full((a.shape[0],), float(math.log(max(int(a.shape[-1]), 2))),
                              device=a.device, dtype=a.dtype)
        return 1.0 - DirichletAleatoricHead.expected_entropy(alpha, mask) / logk


class JEVHead(nn.Module):
    """The full decision head: readout + VBLL scorer + supervised aleatoric term."""

    def __init__(
        self,
        d_model: int = 5120,
        n_features: int = 256,
        prior_scale: float = 1.0,
        max_options: int = 255,
        aleatoric_weight: float = 0.1,
        use_context: bool = True,
        learn_noise: bool = False,
        center_features: bool = True,
        aleatoric_from: str = "readout",
        alea_unit_bias: bool = False,
        alpha_prior_weight: float = 0.0,
        alpha_prior_theta: float = 2.0,
    ):
        super().__init__()
        self.readout = OptionReadout(d_model, n_features, use_context=use_context)
        self.scorer = VBLLScorer(n_features, prior_scale=prior_scale, learn_noise=learn_noise)
        # max_entropy=1.0 because the loader normalises its entropy target by log K, so the target
        # lives in [0, 1]. Passing math.log(max_options) here (as this did) put the head's output
        # range 5.5x above its target and left corr(pred_entropy, target) at -0.047.
        #
        # Input dim is the readout TRUNK width, not d_model: the head now consumes a state
        # representation the readout is optimised to produce (see `_state_repr`), so its gradient
        # reaches the readout. It used to take the cached h_decision directly -- a frozen tensor
        # nothing optimises for it -- which is why it could not be described as jointly trained.
        hidden = getattr(self.readout, "net", None)
        hidden = hidden[0].out_features if hidden is not None else d_model
        if aleatoric_from not in ("readout", "frozen", "dirichlet"):
            raise ValueError("aleatoric_from must be 'readout', 'frozen' or 'dirichlet', "
                             f"got {aleatoric_from!r}")
        self.aleatoric_from = aleatoric_from
        if aleatoric_from == "dirichlet":
            # Per-OPTION, deliberately NOT pooled through `_state_repr`. The Dirichlet likelihood is
            # defined on a concentration per option, so pooling to one vector per unit would throw
            # away exactly the structure the objective is written on. It consumes d_model-wide
            # option readouts, which also means it does not share the aleatoric head's trunk -- by
            # design, since the whole point is to stop it collapsing to the conditional mean.
            self.aleatoric = DirichletAleatoricHead(d_model, unit_bias=alea_unit_bias)
        else:
            self.aleatoric = AleatoricHead(hidden if aleatoric_from == "readout" else d_model,
                                           max_entropy=1.0)
        self.aleatoric_weight = float(aleatoric_weight)
        # A Gamma(2, 1/theta) prior on the Dirichlet concentration. 0.0 = off, so every recorded
        # result is reproducible by passing nothing. See `alpha_prior_nll` for why it is needed:
        # the concentration parameterisation is unbounded, so early stopping was the only thing
        # preventing the head from sharpening into overconfidence (R-40 measured it happening).
        self.alpha_prior_weight = float(alpha_prior_weight)
        self.alpha_prior_theta = float(alpha_prior_theta)
        self.center_features = bool(center_features)

    def _state_repr(self, h_options, h_decision, mask=None):
        """State-level representation for the aleatoric head: the readout trunk, pooled over
        options. [B, hidden].

        Mean-pooling (mask-aware) is a deliberate choice over the last option or the decision
        marker: the question is a property of the whole (state, question) pair, and every option
        shares the state, so pooling keeps all of them in the gradient path. It also makes the
        representation invariant to option ORDER up to pooling, which is one of the two measured
        invariance deficits (worst cosine 0.41 elsewhere).

        `aleatoric_from="frozen"` returns the cached `h_decision` untouched — the pre-reroute
        behaviour. Kept because the two representations answer different questions and the 2x2
        (representation x label-target sharpness) is not complete without it: a frozen input gives
        the head its own trunk and no competition with the cross-entropy objective at all.
        """
        if self.aleatoric_from == "frozen":
            return h_decision
        h = self.readout.encode(h_options, context=h_decision)
        if mask is not None:
            m = mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
        return h.mean(1)

    def _features(self, h_options, h_decision, mask=None):
        """phi_o, optionally centred across the option set.

        Centring is not cosmetic and it is not the same as the softmax's shift invariance.
        Adding a common vector c to every phi_o shifts every logit by w.c, which cancels in
        the softmax -- so c carries NO decision information. But the VBLL variance term is
        phi_o^T S phi_o, which is NOT shift invariant: a shared c contributes c^T S c to
        every option and 2 c^T S u_o across them. The shared part therefore inflates every
        option's variance equally, which dilutes exactly the per-option differences the
        epistemic term is supposed to rank by -- and with `use_context` the fused h_d block
        is shared by construction, so the effect is guaranteed present.

        Subtracting the option-mean makes the features and their variance agree about what
        is decision-relevant.

        `mask=None` means "no padding", not "no centring": the shared-`h_d` argument above is
        unchanged by the absence of a mask, so centring applies either way. It used to be guarded
        on `mask is not None`, which silently skipped it for every caller passing `None` -- measured
        `|option-mean|` 2.4e-8 with a mask versus **0.49** without, i.e. exactly the dilution this
        exists to remove. The mean is taken over the REAL options when a mask is given, and over
        all options otherwise.
        """
        phis = self.readout(h_options, context=h_decision)
        if self.center_features:
            if mask is None:
                mu = phis.mean(dim=1, keepdim=True)
            else:
                m = mask.unsqueeze(-1).to(phis.dtype)
                mu = (phis * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
            phis = phis - mu
        return phis

    def forward(
        self,
        h_options: torch.Tensor,
        h_decision: torch.Tensor,
        p_target: torch.Tensor,
        mask: torch.Tensor | None = None,
        target_entropy: torch.Tensor | None = None,
        n_train: int | None = None,
        entropy_mask: torch.Tensor | None = None,
        target_counts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """h_options [B, K, d_model] readout per option; h_decision [B, d_model] state readout.

        `entropy_mask` must be True only where the label really is an annotator distribution.
        A hard one-hot label has H[p*] = 0 by construction, not because humans agreed, so
        training the head on those zeros teaches it to predict "no disagreement" on every
        GSM8K and Banking77 item -- where disagreement was never measured at all.
        """
        phis = self._features(h_options, h_decision, mask)
        out = self.scorer.loss(phis, p_target, mask=mask, n_train=n_train)
        out["phis"] = phis

        if self.aleatoric_from == "dirichlet":
            # The Dirichlet branch. Trained on the DIRICHLET-MULTINOMIAL LIKELIHOOD of the recovered
            # annotator counts -- a strictly proper scoring rule, so the Bayes-optimal response is
            # the true predictive rather than its conditional mean, which is what let the MSE head
            # collapse to a per-config constant on 11 of 17 configs.
            #
            # HARD LABELS ARE INCLUDED, unlike the MSE branch, and that is not an oversight. The
            # reason `entropy_mask` exists is that an MSE target of 0 on a hard one-hot means "no
            # disagreement anywhere in this batch", which the head would be taught to reproduce --
            # inventing agreement that was never measured. Under a likelihood there is no such
            # distortion: a one-hot with n_raters = 1 enters as ONE observation, appropriately
            # weak, and contributes no more than one rater is worth. Excluding them would throw
            # away most of the corpus.
            if target_counts is None:
                # A NO-OP, not a fallback. Without counts there is no likelihood to evaluate, and
                # silently skipping the branch would train only the scorer while the class
                # docstring promises the head is trained on the Dirichlet-multinomial
                # log-likelihood -- a head that looks wired up and is simply untrained. Fail loudly.
                raise ValueError(
                    "aleatoric_from='dirichlet' needs `target_counts` (the recovered annotator "
                    "counts). Without them the aleatoric branch cannot be evaluated, and skipping "
                    "it would silently leave the Dirichlet head untrained.")
            alpha = self.aleatoric(h_options, h_decision, mask)
            # `entropy_mask` is deliberately NOT applied. The argument is in the comment above and
            # the code used to contradict it: `m = mask & entropy_mask` fully masked every
            # hard-labelled unit out of the loss, so despite "HARD LABELS ARE INCLUDED" the head
            # trained on the 38,092 soft units of 73,618 and threw away 48% of the corpus. It is
            # not a guard that is needed here -- an MSE target of 0 on a one-hot claims "no
            # disagreement anywhere", which is what `entropy_mask` exists to prevent, whereas
            # under the Dirichlet-multinomial likelihood a one-hot with n_raters = 1 contributes
            # `log(A / a_chosen)`, a weak supervised classification gradient that pushes alpha
            # toward the chosen option. That is one rater's worth of evidence, and exactly what the
            # head is starved of. `nll` already excludes fully-masked rows, so an all-hard batch is
            # safe.
            aux = DirichletAleatoricHead.nll(alpha, target_counts, mask=mask)
            if self.alpha_prior_weight > 0.0:
                pri = DirichletAleatoricHead.alpha_prior_nll(
                    alpha, mask=mask, theta=self.alpha_prior_theta)
                out["alpha_prior_loss"] = pri
                aux = aux + self.alpha_prior_weight * pri
            out["aleatoric_loss"] = aux
            out["alpha"] = alpha
            out["pred_agreement"] = DirichletAleatoricHead.normalized_agreement(alpha, mask=mask)
            out["pred_entropy"] = DirichletAleatoricHead.expected_entropy(alpha, mask=mask)
            out["loss"] = out["loss"] + self.aleatoric_weight * aux
        elif target_entropy is not None:
            # The readout trunk, not the cached h_decision: see `_state_repr`. This is what makes
            # the aleatoric head jointly trained rather than a probe of a frozen representation.
            pred_entropy = self.aleatoric(self._state_repr(h_options, h_decision, mask))
            if entropy_mask is None:
                aux = F.mse_loss(pred_entropy, target_entropy)
            elif entropy_mask.any():
                aux = F.mse_loss(pred_entropy[entropy_mask], target_entropy[entropy_mask])
            else:
                # A batch with no soft labels at all must contribute no aleatoric gradient
                # rather than a zero target it would be taught to match.
                aux = torch.zeros((), device=pred_entropy.device, dtype=pred_entropy.dtype)
            out["aleatoric_loss"] = aux
            out["pred_entropy"] = pred_entropy
            out["loss"] = out["loss"] + self.aleatoric_weight * aux
        return out

    @torch.no_grad()
    def predict(self, h_options, h_decision=None, n_samples: int = 64, n_noise: int = 0, mask=None):
        phis = self._features(h_options, h_decision, mask)
        out = self.scorer.predict(phis, n_samples=n_samples, n_noise=n_noise, mask=mask)
        if h_decision is not None and self.aleatoric_from == "dirichlet":
            alpha = self.aleatoric(h_options, h_decision, mask)
            out["alpha"] = alpha
            out["pred_entropy"] = DirichletAleatoricHead.expected_entropy(alpha, mask=mask)
            out["pred_agreement"] = DirichletAleatoricHead.normalized_agreement(alpha, mask=mask)
        elif h_decision is not None:
            # Same path as training. Taking h_decision here instead would have meant the reported
            # pred_entropy came from a different input than the one the head was trained on --
            # a silent train/eval mismatch that would have looked like a bad head.
            out["pred_entropy"] = self.aleatoric(self._state_repr(h_options, h_decision, mask))
            # Supervised aleatoric + posterior epistemic = the reported total confidence
            # decomposition. See ARCHITECTURE.md for what to do with the two.
            out["I_epistemic_supervised_total"] = out["I_epistemic"] + out["pred_entropy"]
        return out

    def param_groups(self, readout_lr: float, vbll_lr: float):
        return [
            {"params": self.readout.parameters(), "lr": readout_lr},
            {"params": self.scorer.parameters(), "lr": vbll_lr},
            {"params": self.aleatoric.parameters(), "lr": readout_lr},
        ]

    def main_parameters(self):
        """The predictive: readout + VBLL scorer. Excludes the supervised aleatoric head.

        Exists so gradient clipping can be applied to the predictive and the auxiliary head
        SEPARATELY. `clip_grad_norm_` over `self.parameters()` computes ONE norm across both and
        scales them by a common factor, which makes `aleatoric_weight` a global learning-rate knob:
        raising the auxiliary loss raises the global norm and shrinks the readout's and scorer's
        steps by the same factor. Measured in `test_grad_clip_coupling.py`, for the Dirichlet head
        the readout's RAW gradient is identical at auxiliary weights 0.0 / 0.3 / 3.0 (0.000380641)
        while its POST-CLIP gradient is 0.000146584 at w = 3.0 -- a factor of exactly the clip
        factor 0.385099.

        **SCOPE, and this is narrower than an earlier version of this docstring claimed.** The
        split fully decouples only `aleatoric_from="dirichlet"`, where the head shares no parameter
        with the predictive, so the clip was the *only* connection. It does NOT decouple the scalar
        `"readout"` head: that one consumes the readout trunk by design, so the auxiliary gradient
        is already inside `readout`'s `.grad` before either clip runs, and partitioning the
        parameters cannot separate a mixture inside one tensor. Measured with the split clip and
        `aleatoric_from="readout"`: readout `|delta|` = 0.1431 at w = 0.3 versus 1.5375 at w = 3.0
        -- still coupled, and `"readout"` is both the constructor and the CLI default. So for that
        arm `aleatoric_weight` remains a learning-rate knob on the predictive, and **any
        scalar-arm comparison that varies it is confounded**; the `"dirichlet"` arm is the one that
        can be varied cleanly.
        """
        seen = set()
        for module in (self.readout, self.scorer):
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def aux_parameters(self):
        """The supervised aleatoric head. Clip this as its own group; see `main_parameters`."""
        for p in self.aleatoric.parameters():
            yield p
