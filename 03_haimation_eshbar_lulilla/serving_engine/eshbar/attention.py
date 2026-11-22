from __future__ import annotations
import os
import types
import torch
ATTN_NAME = 'packed_doc'
REF_NAME = 'dense_ref'
POLICIES = ('C', 'B')
POLICY_ATTR = 'attention_policy'
REF_MAX_TOKENS = 4096
KERNEL_LOG = {'calls': {}, 'max_mask_elems': 0, 'eligible': {}, 'kv_materialized': 0}
KV_MATERIALIZE = os.environ.get('ESHBAR_KV_MATERIALIZE', '1') != '0'

def has_stride0(*ts):
    return any((t is not None and any((st == 0 and n > 1 for st, n in zip(t.stride(), t.shape))) for t in ts))

def materialize_kv(k, v):
    if KV_MATERIALIZE and has_stride0(k, v):
        KERNEL_LOG['kv_materialized'] = KERNEL_LOG.get('kv_materialized', 0) + 1
        return (k.contiguous(), v.contiguous())
    return (k, v)
DEFAULT_CHUNK = int(os.environ.get('ESHBAR_ATTN_CHUNK', '2048'))
ATTN_IMPL = os.environ.get('ESHBAR_ATTN_IMPL', 'kernels')
_DOC_CACHE = {}

def doc_ids(position_ids, B, L, device):
    if position_ids is None:
        return None
    key = (position_ids.data_ptr(), tuple(position_ids.shape), position_ids._version, B, L, str(device))
    hit = _DOC_CACHE.get('k')
    if hit is not None and hit[0] == key and (hit[1] is position_ids):
        return hit[2]
    out = _doc_ids(position_ids, B, L, device)
    _DOC_CACHE['k'] = (key, position_ids, out)
    return out

def _doc_ids(position_ids, B, L, device):
    pos = position_ids
    if pos.dim() == 1:
        pos = pos[None]
    if pos.shape[-1] != L:
        raise RuntimeError(f'{ATTN_NAME}: position_ids length {pos.shape[-1]} != query length {L}')
    pos = pos.to(device).expand(B, L)
    first = pos[:, :1] - 1
    d = (torch.diff(pos, prepend=first, dim=-1) != 1).cumsum(-1)
    if bool((d == d[:, :1]).all()):
        return None
    return d

def policy_of(config):
    pol = getattr(config, POLICY_ATTR, None) if config is not None else None
    if pol is None:
        return 'C'
    if pol not in POLICIES:
        raise RuntimeError(f'{ATTN_NAME}: unknown attention policy {pol!r} (expected one of {POLICIES})')
    return pol

def _layer_kind(module, sliding_window):
    is_global = sliding_window is None
    ms = getattr(module, 'is_sliding', None)
    if ms is not None and bool(ms) == is_global:
        raise RuntimeError(f'{ATTN_NAME}: layer type (is_sliding={ms}) disagrees with sliding_window={sliding_window}')
    return 'global' if is_global else 'local'

def _log(kind, policy, path, mask):
    k = (kind, policy, path)
    KERNEL_LOG['calls'][k] = KERNEL_LOG['calls'].get(k, 0) + 1
    if mask is not None:
        KERNEL_LOG['max_mask_elems'] = max(KERNEL_LOG['max_mask_elems'], int(mask.numel()))

def _doc_ends(doc, L):
    idx = torch.arange(L, device=doc.device).expand_as(doc)
    last = torch.diff(doc, append=doc[:, -1:] + 1, dim=-1) != 0
    val = torch.where(last, idx + 1, torch.full_like(idx, L))
    return val.flip(-1).cummin(-1).values.flip(-1)

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)

def _eligible(q, k, v, mask, causal):
    sig = (tuple(q.shape[1:]), tuple(k.shape[2:]), str(q.dtype), mask is not None, bool(causal))
    if sig in KERNEL_LOG['eligible']:
        return
    out = {}
    try:
        from torch.backends.cuda import SDPAParams, can_use_efficient_attention, can_use_flash_attention
        try:
            prm = SDPAParams(q, k, v, mask, 0.0, bool(causal), False)
        except TypeError:
            prm = SDPAParams(q, k, v, mask, 0.0, bool(causal))
        out = {'flash': bool(can_use_flash_attention(prm, False)), 'efficient': bool(can_use_efficient_attention(prm, False))}
    except Exception as e:
        out = {'probe_error': f'{type(e).__name__}: {str(e)[:120]}'}
    KERNEL_LOG['eligible'][sig] = out

def allowed_backends():
    allowed = [x for x in os.environ.get('ESHBAR_SDPA_BACKENDS', 'efficient,flash').split(',') if x]
    if 'math' in allowed and os.environ.get('ESHBAR_ALLOW_SDPA_MATH') != '1':
        raise RuntimeError(f'{ATTN_NAME}: the SDPA math kernel is not an allowed fallback (it materialises [H, L, L]); ESHBAR_ALLOW_SDPA_MATH=1 is the explicit, recorded override')
    return allowed

def _sdpa(q, k, v, mask, causal, scale):
    if mask is not None:
        k, v = materialize_kv(k, v)
    if ATTN_IMPL == 'chunked_math':
        from torch.utils.checkpoint import checkpoint
        f = torch.nn.functional.scaled_dot_product_attention
        if q.device.type == 'cuda':
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def f(q_, k_, v_, m_, _f=f):
                with sdpa_kernel([SDPBackend.MATH]):
                    return _f(q_, k_, v_, attn_mask=m_, is_causal=causal, scale=scale)
            return checkpoint(f, q, k, v, mask, use_reentrant=False) if torch.is_grad_enabled() else f(q, k, v, mask)
        return checkpoint(lambda q_, k_, v_, m_: f(q_, k_, v_, attn_mask=m_, is_causal=causal, scale=scale), q, k, v, mask, use_reentrant=False) if torch.is_grad_enabled() else f(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)
    if q.device.type == 'cuda':
        from torch.nn.attention import SDPBackend, sdpa_kernel
        allowed = allowed_backends()
        _eligible(q, k, v, mask, causal)
        bk = {'efficient': SDPBackend.EFFICIENT_ATTENTION, 'flash': SDPBackend.FLASH_ATTENTION, 'cudnn': SDPBackend.CUDNN_ATTENTION, 'math': SDPBackend.MATH}
        with sdpa_kernel([bk[a] for a in allowed if a]):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)

def packed_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, softcap=None, **kwargs):
    if attention_mask is not None:
        raise RuntimeError(f'{ATTN_NAME}: got a materialised attention mask; this implementation derives masks from position_ids (pass attention_mask=None / no padding)')
    if dropout:
        raise RuntimeError(f'{ATTN_NAME}: attention dropout is not supported')
    if softcap is not None:
        raise RuntimeError(f'{ATTN_NAME}: attention logit softcapping is not supported')
    B, H, L, D = query.shape
    if key.shape[2] != L:
        raise RuntimeError(f'{ATTN_NAME}: key length {key.shape[2]} != query length {L} (no KV cache support)')
    policy = policy_of(getattr(module, 'config', None))
    kind = _layer_kind(module, sliding_window)
    bidir = policy == 'B' and kind == 'global'
    scale = scaling if scaling is not None else D ** (-0.5)
    n_rep = H // key.shape[1]
    k, v = (repeat_kv(key, n_rep), repeat_kv(value, n_rep))
    W = int(sliding_window) if kind == 'local' else None
    if W is not None and W >= L:
        W = None
    doc = doc_ids(kwargs.get('position_ids'), B, L, query.device)
    if doc is None and W is None and (ATTN_IMPL != 'chunked_math'):
        _log(kind, policy, 'single_bidir' if bidir else 'single_causal', None)
        out = _sdpa(query, k, v, None, L > 1 and (not bidir), scale)
        return (out.transpose(1, 2).contiguous(), None)
    k, v = materialize_kv(k, v)
    C = max(16, int(kwargs.get('attn_chunk') or DEFAULT_CHUNK))
    if doc is not None:
        idx = torch.arange(L, device=query.device).expand(B, L)
        start = torch.where(torch.diff(doc, prepend=doc[:, :1] - 1, dim=-1) != 0, idx, torch.zeros_like(idx)).cummax(-1).values
        start = start.min(0).values.tolist()
        if bidir:
            end = _doc_ends(doc, L).max(0).values.tolist()
    outs = []
    for s in range(0, L, C):
        e = min(L, s + C)
        if bidir:
            k0, k1 = (int(start[s]), int(end[e - 1])) if doc is not None else (0, L)
            m = (doc[:, s:e, None] == doc[:, None, k0:k1])[:, None] if doc is not None else None
        else:
            k0, k1 = (0 if W is None else max(0, s - W + 1), e)
            if doc is not None:
                k0 = max(k0, int(start[s]))
            qi = torch.arange(s, e, device=query.device)[:, None]
            ki = torch.arange(k0, k1, device=query.device)[None, :]
            m = ki <= qi
            if W is not None:
                m = m & (ki > qi - W)
            if doc is not None:
                m = m[None] & (doc[:, s:e, None] == doc[:, None, k0:k1])
                m = m[:, None]
            else:
                m = m[None, None]
        _log(kind, policy, 'chunked_bidir' if bidir else 'chunked_causal', m)
        outs.append(_sdpa(query[:, :, s:e], k[:, :, k0:k1], v[:, :, k0:k1], m, False, scale))
    out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
    return (out.transpose(1, 2).contiguous(), None)

def dense_mask(doc, B, L, kind, policy, W, device):
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    same = doc[:, :, None] == doc[:, None, :] if doc is not None else torch.ones(B, L, L, dtype=torch.bool, device=device)
    if policy == 'B' and kind == 'global':
        return same
    m = same & (j <= i)[None]
    if kind == 'local':
        m = m & (i - j < W)[None]
    return m

def dense_reference_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, softcap=None, **kwargs):
    if attention_mask is not None or dropout or softcap is not None:
        raise RuntimeError(f'{REF_NAME}: masks / dropout / softcap are not part of the reference contract')
    B, H, L, D = query.shape
    if L > REF_MAX_TOKENS:
        raise RuntimeError(f'{REF_NAME}: reference only (L={L} > {REF_MAX_TOKENS})')
    policy = policy_of(getattr(module, 'config', None))
    kind = _layer_kind(module, sliding_window)
    pos = kwargs.get('position_ids')
    doc = _doc_ids(pos, B, L, query.device) if pos is not None else None
    m = dense_mask(doc, B, L, kind, policy, int(sliding_window) if kind == 'local' else None, query.device)
    n_rep = H // key.shape[1]
    k, v = (repeat_kv(key, n_rep), repeat_kv(value, n_rep))
    sc = query @ k.transpose(-1, -2) * (scaling if scaling is not None else D ** (-0.5))
    sc = sc.masked_fill(~m[:, None], float('-inf'))
    out = torch.softmax(sc, dim=-1) @ v
    return (out.transpose(1, 2).contiguous(), None)

def _no_mask(*args, attention_mask=None, **kwargs):
    if attention_mask is not None and (not bool(torch.as_tensor(attention_mask).bool().all())):
        raise RuntimeError(f'{ATTN_NAME}: padded batches are not supported (pack documents with position_ids instead)')
    return None

def register():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register(ATTN_NAME, packed_attention)
    AttentionMaskInterface.register(ATTN_NAME, _no_mask)
    AttentionInterface.register(REF_NAME, dense_reference_attention)
    AttentionMaskInterface.register(REF_NAME, _no_mask)
    return ATTN_NAME

def require_policy(config, policy):
    impl = getattr(config, '_attn_implementation', None)
    if policy_of(config) != policy:
        raise RuntimeError(f'attention policy {policy_of(config)} != required {policy}')
    if policy != 'C' and impl not in (ATTN_NAME, REF_NAME):
        raise RuntimeError(f'attention policy {policy} requires attn_implementation {ATTN_NAME} (got {impl})')
    return policy
