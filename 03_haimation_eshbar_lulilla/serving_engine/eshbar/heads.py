import json
import math
import os
import re
import torch
HEAD_FILE, HEAD_META = ('head.safetensors', 'head.json')
OPT_HEAD = '\n\nOptions:\n'
OPT_TAIL = '\n\nAnswer with the letter of the single best option and nothing else.'
_YES, _NO = ('Yes(?:: .*)?', 'No(?:: .*)?')

def yes_no_index(prompt, k):
    if int(k) != 2:
        return None
    i, j = (prompt.rfind(OPT_HEAD), prompt.rfind(OPT_TAIL))
    if i < 0 or j < i:
        return None
    block = prompt[i + len(OPT_HEAD):j]
    if re.fullmatch(f'A\\) {_YES}\\nB\\) {_NO}', block, flags=re.S):
        return (0, 1)
    if re.fullmatch(f'A\\) {_NO}\\nB\\) {_YES}', block, flags=re.S):
        return (1, 0)
    return None

def yes_no_tensor(prompts, ks):
    return torch.tensor([yes_no_index(t, int(k)) or (-1, -1) for t, k in zip(prompts, ks)], dtype=torch.long).reshape(-1, 2)
BINARY_SCHEMA = 'binary-laplace-head-v1'
KIND_CODE = {'choice': 0, 'noul': 1, 'score': 2}
_GHX, _GHW = __import__('numpy').polynomial.hermite.hermgauss(16)
GH_Z = torch.from_numpy(_GHX * math.sqrt(2.0)).to(torch.float64)
GH_W = torch.from_numpy(_GHW / math.sqrt(math.pi)).to(torch.float64)
BINARY_FIT = {'chunk': 4096}

def h_perp(h, VE):
    hd = h.double()
    hp = hd - hd @ VE @ VE.T
    return hp * torch.rsqrt(hp.pow(2).mean(-1, keepdim=True) + 1e-12)

def binary_ovr(mu_scaled, sigma_01, mask):
    m = mask
    z = mu_scaled.masked_fill(~m, -1000000000.0)
    lse_all = torch.logsumexp(z, dim=-1, keepdim=True)
    p0 = torch.softmax(z, dim=-1).masked_fill(~m, 0.0)
    margin = z - lse_all - torch.log1p(-p0.clamp_max(1.0 - 1e-12))
    kb = m.double().sum(dim=-1, keepdim=True).clamp_min(2.0)
    sig_eff = sigma_01 * (2.0 / kb).sqrt() * m.double()
    gz, gw = (GH_Z.to(z.device), GH_W.to(z.device))
    arg = margin[None, :, :] + sig_eff[None, :, :] * gz[:, None, None]
    f_u = torch.sigmoid(arg).masked_fill(~m[None, :, :], 0.0)
    I_k = torch.einsum('q,qbk->bk', gw, f_u).masked_fill(~m, 0.0)
    p = I_k / I_k.sum(dim=-1, keepdim=True).clamp_min(1e-15)

    def bern(q):
        q_c = q.clamp(1e-15, 1.0 - 1e-15)
        return -(q_c * q_c.log() + (1.0 - q_c) * (1.0 - q_c).log())
    mi_k = (bern(I_k) - torch.einsum('q,qbk->bk', gw, bern(f_u))).clamp_min(0.0) * m.double()
    return (p, mi_k.sum(dim=-1) / kb.squeeze(-1))

def binary_score(p_init, k_vec, mask, sig_kern, lam_uni, sig_uni):
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
        p_out = torch.einsum('bj,bjk->bk', p_out, cdf_diff)
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

def binary_extract(h, ell, ks, kinds, st, chunk=BINARY_FIT['chunk']):
    VE, Vp, mu_hp, E = (st['V_E'], st['V_perp'], st['mu_hp'].reshape(1, -1), st['E'])
    res_l, un_l = ([], [])
    for s0 in range(0, h.shape[0], chunk):
        hp_raw = h_perp(h[s0:s0 + chunk], VE)
        hp = hp_raw - mu_hp
        u = hp @ Vp
        res_l.append((hp - u @ Vp.T).pow(2).sum(-1).sqrt())
        un_l.append(hp_raw @ Vp)
    res = torch.cat(res_l) if res_l else torch.zeros(0, dtype=torch.float64)
    u_noul = torch.cat(un_l) if un_l else torch.zeros(0, Vp.shape[1], dtype=torch.float64)
    K = ell.shape[1]
    mask = torch.arange(K, device=ell.device)[None] < ks[:, None]
    m = mask.double()
    e = ell.double()
    z = (h.double() @ E.T)[:, :K]
    if K > z.shape[1]:
        z = torch.cat([z, e[:, z.shape[1]:K]], 1)
    ell_c = (e - (e * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    z_c = (z - (z * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    n = int(ks.shape[0])
    idx_n = (kinds == KIND_CODE['noul']).nonzero().flatten()
    dm_noul = torch.zeros(n, dtype=torch.float64)
    var_noul = torch.zeros(n, dtype=torch.float64)
    if len(idx_n) > 0:
        un = u_noul[idx_n]
        dm_noul[idx_n] = un @ st['mu'] / 2.0
        var_noul[idx_n] = (un @ st['S'] * un).sum(-1).clamp_min(0.0) / 4.0
    row_rms = ((ell_c.pow(2) * m).sum(1) / m.sum(1).clamp_min(1.0) + 1.0).sqrt()
    return {'ell_c': ell_c, 'dz_c': z_c - ell_c, 'mask': mask, 'res': res, 'dm_noul': dm_noul, 'var_noul': var_noul, 'row_rms': row_rms, 'kinds': kinds, 'k': ks}

def binary_probs(d, st, T, c):
    m = d['mask']
    kinds = d['kinds']
    mu = (d['ell_c'] + c['alpha'] * d['dz_c']).clone()
    is_noul = kinds == KIND_CODE['noul']
    if c['beta'] > 0.0 and bool(is_noul.any()):
        dm = c['beta'] * d['dm_noul'][is_noul]
        mu[is_noul, 0] += dm
        mu[is_noul, 1] -= dm
    if c['tau'] > 0.0:
        r_scale = torch.ones(len(mu), dtype=torch.float64)
        for kd in ('noul', 'choice', 'score'):
            r_scale[kinds == KIND_CODE[kd]] = 1.0 + c['tau'] * (st['rel_s'][kd] - 1.0)
        T_row = T * r_scale[:, None]
    else:
        T_row = T
    score_on = c['kern'] > 0.0 or c['lam_uni'] > 0.0
    if c['sig'] <= 0.0 and (not score_on) and (c['tau'] <= 0.0):
        return (torch.softmax((mu / T).masked_fill(~m, -1000000000.0), -1).masked_fill(~m, 0.0), torch.zeros(len(mu), dtype=torch.float64))
    mu_scaled = mu / T_row
    rr = d['res'] / st['res_mean'] if c['dnorm'] == 'ratio' else torch.ones_like(d['res'])
    sd_row = c['sig'] * rr * d['row_rms']
    sigma = sd_row[:, None].expand_as(mu).clone()
    if bool(is_noul.any()):
        sigma[is_noul, :2] = (sigma[is_noul, :2].pow(2) + c['infl'] * d['var_noul'][is_noul, None]).sqrt()
    p, mi = binary_ovr(mu_scaled, sigma / T_row * m.double(), m)
    is_score = kinds == KIND_CODE['score']
    if score_on and bool(is_score.any()):
        idx = is_score.nonzero().flatten()
        p[idx] = binary_score(p[idx], d['k'][idx], m[idx], c['kern'], c['lam_uni'], c['sig_uni'])
    return (p, mi)

class BinaryHead(torch.nn.Module):
    needs_kinds = True

    def __init__(self, state, meta):
        super().__init__()
        self.heads = [dict(h) for h in meta['heads']]
        self.c = dict(meta['constants'])
        self.variant = meta.get('variant', 'full')
        self.rel_s = {k: float(v) for k, v in meta['rel_s'].items()}
        self.res_mean = float(meta['res_mean'])
        for k in ('E', 'V_E', 'V_perp', 'mu_hp', 'mu', 'S'):
            self.register_buffer(k, state[k].double())

    @classmethod
    def from_dir(cls, d, meta):
        from safetensors.torch import load_file
        return cls(load_file(os.path.join(d, HEAD_FILE)), meta)

    def state(self):
        return {'E': self.E, 'V_E': self.V_E, 'V_perp': self.V_perp, 'mu_hp': self.mu_hp, 'mu': self.mu, 'S': self.S, 'rel_s': self.rel_s, 'res_mean': self.res_mean}

    def predict(self, h, ell, ks, n_samples=None, seed=None, draw_grid_size=None, draw_slot=0, temperature=1.0, kinds=None):
        if kinds is None:
            raise ValueError("binary head: kinds (the request's question types) are required")
        if draw_grid_size not in (None, 1) or int(draw_slot or 0) != 0:
            raise ValueError('binary head: no draw grid')
        kinds = torch.as_tensor(kinds, dtype=torch.long).reshape(-1)
        if kinds.shape[0] != ell.shape[0] or bool(((kinds == KIND_CODE['noul']) & (ks.cpu() != 2)).any()):
            raise ValueError('binary head: one kind per prompt; noul rows K = 2')
        st = self.state()
        p, mi = binary_probs(binary_extract(h, ell, ks, kinds, st), st, float(temperature), self.c)
        hp_ = -(p * p.clamp_min(1e-12).log()).sum(-1)
        return (p[None], mi[None], (hp_ - mi).clamp_min(0.0)[None])

def load_head(d):
    import hashlib
    p, j = (os.path.join(d, HEAD_FILE), os.path.join(d, HEAD_META))
    if not (os.path.exists(p) and os.path.exists(j)):
        raise FileNotFoundError(f'{d}: no head ({HEAD_FILE} + {HEAD_META})')
    meta = json.load(open(j))
    if meta.get('schema') != BINARY_SCHEMA:
        raise ValueError(f"{j}: unknown head schema {meta.get('schema')!r}")
    if hashlib.sha256(open(p, 'rb').read()).hexdigest() != meta['tensor_sha256']:
        raise ValueError(f'{p}: sha256 differs from {HEAD_META}')
    return (BinaryHead.from_dir(d, meta), meta)

def letter_logits_fp32(h, E, cap):
    z = h.float() @ E.float().T
    return cap * torch.tanh(z / cap) if cap else z

def readout_spec(spec):
    spec = spec or 'stock'
    if spec == 'stock':
        return ('stock', 0.0)
    mode, _, a = spec.partition(':')
    try:
        av = float(a)
    except ValueError:
        av = -1.0
    if mode != 'blend' or not (0.0 < av < 1.0 and math.isfinite(av)):
        raise ValueError(f"letter_readout {spec!r}: 'stock' or 'blend:A' with 0 < A < 1")
    return ('blend', av)

def letter_readout_fp32(h, E, cap, spec=None):
    mode, a = readout_spec(spec)
    if mode == 'stock':
        return letter_logits_fp32(h, E, cap)
    if not cap:
        raise ValueError('blend readout needs a softcap')
    z = h.float() @ E.float().T
    return cap * torch.tanh(z / cap) * (1.0 - a) + z * a
