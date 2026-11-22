"""Exact greedy speculative decoding for the E1 program writer: drop-in replacement for
eshbar.route.RouteEngine._generate (E1 bundle route.py md5 062b0a81ace0eb6493b9a5d6d2b479b0). Same signature, same return
value [(text | None, n_tokens), ...]; greedy-exact (see "Exactness").

Algorithm (per prompt, batch 1, greedy):
  1. Prefill exactly as the shipped call does it: the chat template, the same tokenisation, the engine's own StaticCache(4096)
     reset, HF's prepare_inputs_for_generation with generate's kwargs (2D mask, position ids, logits_to_keep=1), eager forward
     in 'sdpa'. Token 1 is the argmax of that forward, i.e. the shipped token 1. The prompt K/V are copied into the
     speculative cache.
  2. Loop: the drafter proposes k tokens d1..dk after the last committed token b (position n). The target runs ONE forward
     on [b, d1..dk] (positions n..n+k; one compiled CUDA graph that also builds HF's causal / sliding-window masks and does
     the bookkeeping on the GPU), takes the argmax g0..gk of every row, accepts the longest prefix with d_i == g_{i-1} and
     commits g0..gj (= d1..dj plus its own token). Rejected rows are not rolled back explicitly: every layer writes at a
     shared position tensor that simply does not advance past them, so they sit beyond the causal mask until overwritten.
     The host reads (j, g) once per round while the GPU already runs the next round's drafter.
  3. Stopping is HF's: stop after the first token in generation_config.eos_token_id or at max_new_tokens; the returned
     (text, n) comes from the shipped post-processing.

Cache layout. Gemma-4 31B has 50 sliding-window (1024) layers and 10 full layers. When L + 320 + k <= 1024 (the common
case) every position stays inside the window, so all layers get 1024 slots; otherwise every layer gets 4096 slots and the
sliding-window mask restricts attention (no ring buffer, so a rejected draft never evicts context). Both layouts are
compiled at warm-up (engine init). L + 320 + k > 4096 falls back to the shipped generate call.

Drafters:
  'mtp'   google/gemma-4-31B-it-assistant (Gemma-4 MTP drafter, 4 layers, 0.47B params, Apache-2.0): input = [target embedding
          of the last token, target final hidden state that produced it]; its layers attend to the target's last sliding /
          full layer K/V (no draft cache, no draft prefill). All k draft steps run in one compiled CUDA graph.
  'ngram' prompt lookup: continuation of the most recent earlier match of the last 3/2/1 tokens in prompt + output.
  'none'  plain one-token decoding through the same loop (k=0), for testing.
A proposal never changes the output, only the speed: only tokens equal to the target's own argmax are kept.

Exactness. In exact arithmetic the output is the greedy output. In bf16 the verify forward has k+1 query rows where the
shipped decode has 1, so kernels differ in the last bits; a token can then flip only where the shipped top-2 logits are tied
or 1 bf16 ulp apart. The shipped path itself flips at those ties from run to run (same GPU, same process), so the parity test
(parity_spec.py) compares every program token for token with two shipped runs and reports the first differing step, both
sides' logit margins and the bf16 ulp there.

Config (engine.spec_config dict or env): drafter (ESHBAR_SPEC_DRAFTER, default mtp), k (ESHBAR_SPEC_K), assistant
(ESHBAR_SPEC_ASSISTANT: local dir or HF id, default google/gemma-4-31B-it-assistant), loop ('fast' | 'hf'), short_full.

Install (no release file changes):
    import spec_decode; spec_decode.install(eshbar.route.RouteEngine, drafter='mtp', k=..., assistant=PATH_OR_HF_ID)
"""
import os
import time

import torch
import torch._dynamo

PROGRAM_PROMPT = 'Solve the problem by writing a Python function `solution()` that takes no arguments and returns the final answer: a number, or True/False if the question asks yes/no or true/false. It runs with exact arithmetic (decimals and fractions are exact), so do not add small epsilons or round to guard against float error.\n\nUse only: integers and decimal numbers; + - * / // % and ** with a constant exponent; comparisons; and/or/not; if/elif/else; for loops over range(...); while loops; break/continue; lists of numbers (indexing, len, sum, min, max); helper functions (no recursion); round, abs, min, max, int, float, math.floor, math.ceil.\nDo not use strings, dictionaries, sets, imports other than math, input() or print().\n\nIf the answer is a fraction, return it as an exact division such as 3 / 7 (not a rounded decimal). Do not write comments or docstrings. Use short, clear variable names. Reply with only the code in one ```python block.\n\nProblem: {question}'
CACHE_LEN = 4096
DEFAULTS = {'drafter': os.environ.get('ESHBAR_SPEC_DRAFTER', 'mtp'), 'k': int(os.environ.get('ESHBAR_SPEC_K', '0') or 0),
            'assistant': os.environ.get('ESHBAR_SPEC_ASSISTANT', 'google/gemma-4-31B-it-assistant'),
            'ngram_max': 3, 'ngram_min': 1}
DEFAULT_K = {'mtp': 6, 'ngram': 5, 'none': 0}


def _text_cfg(m):
    return m.config.get_text_config() if hasattr(m.config, 'get_text_config') else m.config


try:
    from transformers.cache_utils import Cache as _Cache, StaticLayer as _StaticLayer
except ImportError:  # pragma: no cover
    _Cache = _StaticLayer = object


class SlidingStaticLayer(_StaticLayer):
    """A StaticLayer (index_copy at a tensor write position, no ring buffer) flagged as sliding, so HF's mask builder picks
    it for the sliding-window mask (kv_length = its slot count)."""
    is_sliding = True


def make_cache(cfg, sliding_len, full_len):
    n = cfg.num_hidden_layers - (getattr(cfg, 'num_kv_shared_layers', 0) or 0)
    layers = [SlidingStaticLayer(max_cache_len=sliding_len) if t == 'sliding_attention' else _StaticLayer(max_cache_len=full_len)
              for t in cfg.layer_types[:n]]
    return _Cache(layers=layers)


class SpecLayer(_StaticLayer):
    """Static K/V storage written at a shared position tensor `wpos` (the rows of the current forward). The loop sets wpos
    before every forward, so a rejected draft is rolled back by simply not advancing: its rows sit beyond the causal mask
    and are overwritten by the next write."""

    def __init__(self, max_cache_len, wpos, sliding):
        super().__init__(max_cache_len=max_cache_len)
        self.wpos = wpos
        self.is_sliding = sliding

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys.index_copy_(2, self.wpos, key_states)
        self.values.index_copy_(2, self.wpos, value_states)
        return self.keys, self.values


def make_fast_cache(cfg, sliding_len, full_len, wpos):
    n = cfg.num_hidden_layers - (getattr(cfg, 'num_kv_shared_layers', 0) or 0)
    return _Cache(layers=[SpecLayer(sliding_len if t == 'sliding_attention' else full_len, wpos, t == 'sliding_attention')
                          for t in cfg.layer_types[:n]])


def _static(t):
    torch._dynamo.mark_static_address(t)
    return t


def _set_length(cache, n):
    for layer in cache.layers:
        layer.cumulative_length.fill_(n)


class NgramDrafter:
    """Prompt lookup: continuation of the most recent earlier occurrence of the last g tokens (g = ngram_max..ngram_min)."""
    name = 'ngram'

    def __init__(self, k, ngram_max=3, ngram_min=1):
        self.k, self.gmax, self.gmin = k, ngram_max, ngram_min

    def start(self, seq):
        self.idx = {}                      # (g, tuple) -> latest start position of that g-gram
        self.n_indexed = 0
        self._index(seq)

    def _index(self, seq):
        # index every g-gram that ends before the last token (so a match always has >= 1 continuation token)
        for end in range(self.n_indexed, len(seq) - 1):
            for g in range(self.gmin, self.gmax + 1):
                s = end - g + 1
                if s >= 0:
                    self.idx[(g, tuple(seq[s:end + 1]))] = s
        self.n_indexed = max(self.n_indexed, len(seq) - 1)

    def propose(self, seq):
        self._index(seq)
        for g in range(self.gmax, self.gmin - 1, -1):
            if len(seq) < g:
                continue
            s = self.idx.get((g, tuple(seq[-g:])))
            if s is not None:
                return seq[s + g: s + g + self.k]
        return []


class MTPDrafter:
    """Gemma-4 MTP assistant driven directly (HF's SinglePositionMultiTokenCandidateGenerator semantics, with static
    target K/V): step i feeds [embed_target(tok), h] at the position of the last committed token, attends to the target's
    last sliding / full layer K/V of positions < n (sliding: the last `window` of them), and returns (logits, next h)."""
    name = 'mtp'

    def __init__(self, path, target, k, device):
        from transformers import AutoModelForCausalLM
        self.k = k
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device, attn_implementation='sdpa').eval()
        self.embed = target.get_input_embeddings()
        tcfg = _text_cfg(target)
        types = tcfg.layer_types
        self.kv_layers = {'sliding_attention': max(i for i, t in enumerate(types) if t == 'sliding_attention'),
                          'full_attention': max(i for i, t in enumerate(types) if t == 'full_attention')}
        self.window = int(_text_cfg(self.model).sliding_window)
        self._fn = {}
        # fixed-address, fixed-stride inputs for the compiled (CUDA-graph) draft loop
        self.n_buf = torch.zeros(1, dtype=torch.long, device=device)
        self.tok_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.h_buf = torch.zeros((1, 1, tcfg.hidden_size), dtype=torch.bfloat16, device=device)

    def _draft(self, tok, h, n, ks, vs, kf, vf):
        a = self.model
        inner = a.model
        out = []
        idx_s = torch.arange(ks.shape[2], device=ks.device)
        idx_f = torch.arange(kf.shape[2], device=kf.device)
        mf = (idx_f < n)[None, None, None, :]
        ms = ((idx_s < n) & (idx_s >= n - self.window))[None, None, None, :]
        pos = n.view(1, 1)
        shared = {'sliding_attention': (ks, vs), 'full_attention': (kf, vf)}
        for _ in range(self.k):
            x = torch.cat([self.embed(tok), h], dim=-1)
            x = a.pre_projection(x)
            o = inner(input_ids=None, inputs_embeds=x, attention_mask={'full_attention': mf, 'sliding_attention': ms},
                      position_ids=pos, shared_kv_states=dict(shared), use_cache=False)
            hl = o.last_hidden_state
            h = a.post_projection(hl)
            logits = a.masked_embedding(hl, a.lm_head.weight) if a.config.use_ordered_embeddings else a.lm_head(hl)
            tok = logits.argmax(-1)
            out.append(tok)
        return torch.cat(out, dim=-1)

    def _draft_into(self, blk, tok, h, n, ks, vs, kf, vf):
        d = self._draft(tok, h, n, ks, vs, kf, vf)
        blk.copy_(torch.cat([tok, d], dim=1))

    def draft_into(self, blk, cache, compiled=True):
        """Fast loop: inputs are the drafter's own static buffers (already set by the verify graph); writes [tok, d1..dk] into blk."""
        ls, lf = cache.layers[self.kv_layers['sliding_attention']], cache.layers[self.kv_layers['full_attention']]
        key = ('into', ls.keys.shape[2], lf.keys.shape[2])
        fn = self._fn.get(key)
        if fn is None:
            fn = self._fn[key] = (torch.compile(self._draft_into, mode='reduce-overhead', fullgraph=False, dynamic=False)
                                  if compiled else self._draft_into)
        fn(blk, self.tok_buf, self.h_buf, self.n_buf, ls.keys, ls.values, lf.keys, lf.values)

    def propose(self, tok, h, n, cache, compiled=True):
        """tok [1,1] long, h [1,1,D] target final hidden, n = number of valid target positions (= position of tok)."""
        ls, lf = cache.layers[self.kv_layers['sliding_attention']], cache.layers[self.kv_layers['full_attention']]
        self.n_buf.fill_(n)
        self.tok_buf.copy_(tok.view(1, 1))
        self.h_buf.copy_(h.view(1, 1, -1))
        key = (ls.keys.shape[2], lf.keys.shape[2])
        if not compiled:
            return self._draft(self.tok_buf, self.h_buf, self.n_buf, ls.keys, ls.values, lf.keys, lf.values)
        fn = self._fn.get(key)
        if fn is None:
            fn = self._fn[key] = torch.compile(self._draft, mode='reduce-overhead', fullgraph=False, dynamic=False)
        return fn(self.tok_buf, self.h_buf, self.n_buf, ls.keys, ls.values, lf.keys, lf.values)


class SpecState:
    def __init__(self, model, tok, drafter='mtp', k=None, assistant=None, ngram_max=3, ngram_min=1, compiled=True, loop='fast',
                 short_full=1024):
        self.m, self.tok = model, tok
        self.loop = loop
        self.short_full = int(short_full)
        self.kind = drafter
        self.k = int(k) if k else DEFAULT_K[drafter]
        self.compiled = compiled
        self.cfg = _text_cfg(model)
        self.window = int(self.cfg.sliding_window)
        self.dev = model.device
        self.caches = {}
        self.drafter = None
        if drafter == 'mtp':
            self.drafter = MTPDrafter(assistant or DEFAULTS['assistant'], model, self.k, self.dev)
        elif drafter == 'ngram':
            self.drafter = NgramDrafter(self.k, ngram_max, ngram_min)
        elif drafter != 'none':
            raise ValueError(drafter)
        self.ones = torch.ones((1, CACHE_LEN + self.k + 2), dtype=torch.long, device=self.dev)
        gc = model.generation_config
        eos = gc.eos_token_id
        self.eos = set([eos] if isinstance(eos, int) else list(eos or []))
        self.stats = {'calls': 0, 'fallback': 0, 'rounds': 0, 'tokens': 0, 'accepted': 0, 'proposed': 0}
        self.last = None
        from torch._dynamo import config as dcfg
        # the speculative shapes (k+1 rows, two cache layouts) are extra compiled variants of the forward HF generate
        # compiles for its 1-row decode; keep room so neither path ever falls back to eager for lack of cache entries
        for name in ('cache_size_limit', 'recompile_limit'):
            if hasattr(dcfg, name):
                setattr(dcfg, name, max(getattr(dcfg, name), 64))
        if hasattr(dcfg, 'accumulated_cache_size_limit'):
            dcfg.accumulated_cache_size_limit = max(dcfg.accumulated_cache_size_limit, 512)
        # fast loop state (shared across layouts); the MTP drafter's buffers double as the loop's state
        M = self.k + 1
        self.M = M
        if self.kind == 'mtp':
            self.tok_buf, self.h_buf, self.n_buf = self.drafter.tok_buf, self.drafter.h_buf, self.drafter.n_buf
        else:
            self.tok_buf = torch.zeros((1, 1), dtype=torch.long, device=self.dev)
            self.h_buf = torch.zeros((1, 1, self.cfg.hidden_size), dtype=torch.bfloat16, device=self.dev)
            self.n_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        for t in (self.tok_buf, self.h_buf, self.n_buf):
            _static(t)
        self.blk = _static(torch.zeros((1, M), dtype=torch.long, device=self.dev))
        self.ar_m = torch.arange(M, device=self.dev)
        self.ar = {n: torch.arange(n, device=self.dev) for n in {self.window, self.short_full, CACHE_LEN}}
        self.rec_host = torch.empty(M + 1, dtype=torch.long, pin_memory=True)
        self.fast = {}

    def cache(self, layout):
        c = self.caches.get(layout)
        if c is None:
            sl = self.window if layout == 'short' else CACHE_LEN
            c = self.caches[layout] = make_cache(self.cfg, sl, CACHE_LEN)
        return c

    def _fwd(self):
        # the callable HF generate uses for its decode steps (same torch.compile config, mode reduce-overhead)
        return self.m.get_compiled_call(None) if self.compiled else self.m.__call__

    @torch.no_grad()
    def prefill(self, ship_cache, ids, need_h):
        """generate()'s prefill: same kwargs, same eager forward, into the engine's own StaticCache."""
        m = self.m
        L = ids.shape[1]
        am = self.ones[:, :L]
        pos = torch.arange(L, device=self.dev)[None]
        inp = m.prepare_inputs_for_generation(ids, next_sequence_length=L, past_key_values=ship_cache, attention_mask=am,
                                              position_ids=pos, use_cache=True, logits_to_keep=1, is_first_iteration=True)
        out = m(**inp, return_dict=True, output_hidden_states=need_h)
        logits = out.logits[:, -1, :].to(copy=True, dtype=torch.float32)
        h = out.hidden_states[-1][:, -1:, :].clone() if need_h else None
        return logits, h

    def copy_prompt(self, ship_cache, cache, L):
        for src, dst in zip(ship_cache.layers, cache.layers):
            if not dst.is_initialized:
                dst.lazy_initialization(src.keys[:, :, :1], src.values[:, :, :1])
            S = src.keys.shape[2]
            if L <= S:
                dst.keys[:, :, :L].copy_(src.keys[:, :, :L])
                dst.values[:, :, :L].copy_(src.values[:, :, :L])
            else:                           # shipped sliding layer after a prefill longer than its window: last S positions
                dst.keys[:, :, L - S:L].copy_(src.keys)
                dst.values[:, :, L - S:L].copy_(src.values)
        _set_length(cache, L)

    @torch.no_grad()
    def verify(self, cache, block, n, need_h):
        m = self.m
        M = block.shape[1]
        inp = m.prepare_inputs_for_generation(block, next_sequence_length=M, past_key_values=cache, attention_mask=self.ones[:, :n + M],
                                              position_ids=torch.arange(n, n + M, device=self.dev)[None], use_cache=True,
                                              logits_to_keep=M)
        out = self._fwd()(**inp, return_dict=True, output_hidden_states=need_h)
        g = out.logits[0].argmax(-1)
        h = out.hidden_states[-1][0].clone() if need_h else None
        return g, h, out.logits

    def fast_layout(self, layout):
        f = self.fast.get(layout)
        if f is None:
            # short: every position stays below the sliding window, so the full-attention layers need no more slots
            # than the window either (fewer masked keys to read); long: 4096 slots everywhere, no ring buffer
            sl, fl = (self.window, max(self.short_full, self.window)) if layout == 'short' else (CACHE_LEN, CACHE_LEN)
            wpos = _static(torch.zeros(self.M, dtype=torch.long, device=self.dev))
            cache = make_fast_cache(self.cfg, sl, fl, wpos)
            fn = (torch.compile(self._vfn, mode='reduce-overhead', fullgraph=False, dynamic=False) if self.compiled else self._vfn)
            f = self.fast[layout] = {'cache': cache, 'wpos': wpos, 'fn': fn, 'ar_s': self.ar[sl], 'ar_f': self.ar[fl]}
        return f

    def _vfn(self, blk, n_buf, wpos, cache, ar_m, ar_s, ar_f, tok_buf, h_buf, need_h):
        """One verify forward of the target over blk = [b, d1..dk] at positions n..n+k, plus the bookkeeping on the GPU:
        accepted count j, next token g_j, its hidden state, n += j + 1. Masks are HF's (causal; sliding: q - kv < window)."""
        M = blk.shape[1]
        wpos.copy_(n_buf + ar_m)
        q = wpos.view(-1, 1)
        ms = (ar_s.view(1, -1) <= q) & (ar_s.view(1, -1) > q - self.window)
        mf = ar_f.view(1, -1) <= q
        out = self.m(input_ids=blk, position_ids=wpos.view(1, -1), past_key_values=cache, use_cache=True, logits_to_keep=M,
                     attention_mask={'full_attention': mf[None, None], 'sliding_attention': ms[None, None]},
                     return_dict=True, output_hidden_states=need_h)
        logits = out.logits[0]
        g = logits.argmax(-1)
        if M > 1:
            j = torch.cumprod((blk[0, 1:] == g[:-1]).to(torch.long), 0).sum().view(1)
        else:
            j = torch.zeros(1, dtype=torch.long, device=blk.device)
        tok_buf.copy_(g.gather(0, j).view(1, 1))
        if need_h:
            h_buf.copy_(out.hidden_states[-1][0].index_select(0, j).view(1, 1, -1))
        n_buf.add_(j + 1)
        return torch.cat([j, g]), logits

    @torch.no_grad()
    def run_fast(self, ids, ship_cache, max_new, capture=None, layout=None):
        L = ids.shape[1]
        k, M = self.k, self.M
        if L + max_new + k > CACHE_LEN:
            return None
        layout = layout or ('short' if L + max_new + k <= self.window else 'long')
        f = self.fast_layout(layout)
        cache = f['cache']
        mtp = self.kind == 'mtp'
        ship_cache.reset()
        logits0, h = self.prefill(ship_cache, ids, mtp)
        if capture is not None:
            capture.append(('prefill', logits0[0].clone()))
        b = int(logits0[0].argmax())
        self.copy_prompt(ship_cache, cache, L)
        self.tok_buf.fill_(b)
        self.n_buf.fill_(L)
        if mtp:
            self.h_buf.copy_(h.view(1, 1, -1))
            self.drafter.draft_into(self.blk, cache, self.compiled)
        gen = [b]
        st = {'rounds': 0, 'accepted': 0, 'proposed': 0, 'layout': layout, 'acc_hist': [], 'loop': 'fast'}
        seq = None
        if self.kind == 'ngram':
            seq = ids[0].tolist() + gen
            self.drafter.start(seq)
        n = L
        ev = torch.cuda.Event()
        done = b in self.eos or len(gen) >= max_new
        while not done:
            if self.kind == 'ngram':
                dl = self.drafter.propose(seq)
                st['proposed'] += len(dl)
                dl = dl + [b] * (k - len(dl))
                self.blk.copy_(torch.tensor([[b] + dl], dtype=torch.long), non_blocking=False)
            elif not mtp:
                self.blk.copy_(self.tok_buf)
            blk_now = self.blk.clone() if capture is not None else None
            rec, lg = f['fn'](self.blk, self.n_buf, f['wpos'], cache, self.ar_m, f['ar_s'], f['ar_f'], self.tok_buf, self.h_buf, mtp)
            self.rec_host.copy_(rec, non_blocking=True)
            ev.record()
            if capture is not None:
                capture.append(('verify', n, blk_now[0].tolist(), lg.float().clone()))
            if mtp:
                self.drafter.draft_into(self.blk, cache, self.compiled)   # next round's drafts, overlapping the host below
            ev.synchronize()
            j = int(self.rec_host[0])
            new = self.rec_host[1:j + 2].tolist()
            st['rounds'] += 1
            st['accepted'] += j
            st['acc_hist'].append(j)
            for t in new:
                gen.append(t)
                if t in self.eos or len(gen) >= max_new:
                    done = True
                    break
            n += j + 1
            b = new[-1]
            if seq is not None:
                seq.extend(new)
        st['tokens'] = len(gen)
        st['ids'] = gen
        self.last = st
        self.stats['calls'] += 1
        self.stats['rounds'] += st['rounds']
        self.stats['tokens'] += len(gen)
        self.stats['accepted'] += st['accepted']
        return gen

    def warm(self, ids, ship_cache, max_new=24):
        """Compile both cache layouts now (engine init), so a long prompt never compiles inside a timed request."""
        t = time.perf_counter()
        for layout in ('short', 'long'):
            self.run(ids, ship_cache, max_new, layout=layout)
        self.warm_s = round(time.perf_counter() - t, 1)
        self.stats = {k: 0 for k in self.stats}
        self.last = None

    @torch.no_grad()
    def run(self, ids, ship_cache, max_new, capture=None, layout=None):
        """ids [1, L] on device. Returns the generated token list (HF stopping rules), or None if it cannot run here."""
        if self.loop == 'fast':
            return self.run_fast(ids, ship_cache, max_new, capture, layout)
        L = ids.shape[1]
        k = self.k
        if L + max_new + k > CACHE_LEN:
            return None
        layout = layout or ('short' if L + max_new + k <= self.window else 'long')
        cache = self.cache(layout)
        need_h = self.kind == 'mtp'
        ship_cache.reset()
        logits0, h = self.prefill(ship_cache, ids, need_h)
        if capture is not None:
            capture.append(('prefill', logits0[0].clone()))
        b = int(logits0[0].argmax())
        self.copy_prompt(ship_cache, cache, L)
        gen = [b]
        st = {'rounds': 0, 'accepted': 0, 'proposed': 0, 'layout': layout, 'acc_hist': []}
        seq = ids[0].tolist() + gen if self.kind == 'ngram' else None
        if self.kind == 'ngram':
            self.drafter.start(seq)
        n = L                                   # position of b == number of valid cache rows
        b_t = torch.tensor([[b]], dtype=torch.long, device=self.dev)
        done = b in self.eos or len(gen) >= max_new
        while not done:
            if self.kind == 'mtp':
                d = self.drafter.propose(b_t, h, n, cache, self.compiled)            # [1, k] on device
                block = torch.cat([b_t, d.view(1, -1)], dim=1)
                dl = None
            elif self.kind == 'ngram':
                dl = self.drafter.propose(seq)
                st['proposed'] += len(dl)
                dl = dl + [b] * (k - len(dl))                                         # padding is harmless: verified
                block = torch.tensor([[b] + dl], dtype=torch.long, device=self.dev)
            else:
                block = b_t
            g, hs, lg = self.verify(cache, block, n, need_h)
            M = block.shape[1]
            if M > 1:
                match = (block[0, 1:] == g[:-1]).to(torch.int32)
                j_t = torch.cumprod(match, 0).sum()
                both = torch.cat([j_t.view(1), g]).tolist()                        # one host sync per round
                j, gl = both[0], both[1:]
                dtoks = block[0, 1:].tolist() if dl is None else dl
            else:
                j, gl, dtoks = 0, g.tolist(), []
            if capture is not None:
                capture.append(('verify', n, block[0].tolist(), lg[0].float().clone()))
            new = dtoks[:j] + [gl[j]]
            st['rounds'] += 1
            st['accepted'] += j
            st['acc_hist'].append(j)
            for t in new:
                gen.append(t)
                if t in self.eos or len(gen) >= max_new:
                    done = True
                    break
            if done:
                break
            n = n + j + 1
            _set_length(cache, n)
            b = gl[j]
            b_t = g[j:j + 1].view(1, 1)
            if need_h:
                h = hs[j:j + 1].view(1, 1, -1)
            if self.kind == 'ngram':
                seq.extend(new)
        st['tokens'] = len(gen)
        st['ids'] = gen
        self.last = st
        self.stats['calls'] += 1
        self.stats['rounds'] += st['rounds']
        self.stats['tokens'] += len(gen)
        self.stats['accepted'] += st['accepted']
        return gen


def _spec(self):
    s = getattr(self, '_spec_state', None)
    if s is None or s.m is not self.model or s.tok is not self.tok:
        cfg = dict(DEFAULTS)
        cfg.update(getattr(self, 'spec_config', None) or {})
        s = self._spec_state = SpecState(self.model, self.tok, drafter=cfg['drafter'], k=cfg['k'] or None,
                                         assistant=cfg['assistant'], ngram_max=cfg['ngram_max'], ngram_min=cfg['ngram_min'],
                                         compiled=cfg.get('compiled', True), loop=cfg.get('loop', 'fast'),
                                         short_full=cfg.get('short_full', 1024))
        if cfg.get('warm', True):
            enc = self.tok(self.tok.apply_chat_template([{'role': 'user', 'content': PROGRAM_PROMPT.format(question='What is 12 times 7, plus 5?')}],
                                                        tokenize=False, add_generation_prompt=True), return_tensors='pt', add_special_tokens=False)
            s.warm(enc['input_ids'].to(self.model.device), self._decoder())
    return s


def _generate(self, texts):
    import torch
    m, tok = (self.model, self.tok)
    prev_attn = m.config._attn_implementation
    m.set_attn_implementation('sdpa')
    res = []
    try:
        spec = _spec(self)
        for t in texts:
            chat = tok.apply_chat_template([{'role': 'user', 'content': PROGRAM_PROMPT.format(question=t)}], tokenize=False, add_generation_prompt=True)
            enc = tok(chat, return_tensors='pt', add_special_tokens=False).to(m.device)
            L = enc['input_ids'].shape[1]
            if L + self.route_max_new > CACHE_LEN:
                res.append((None, 0))
                continue
            cache = self._decoder()
            ids = spec.run(enc['input_ids'], cache, self.route_max_new)
            if ids is None:                                     # too long for the speculative cache: the shipped call
                spec.stats['fallback'] += 1
                cache.reset()
                with torch.no_grad():
                    out = m.generate(**enc, past_key_values=cache, max_new_tokens=self.route_max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                ids = out[0, L:].tolist()
            n = len(ids)
            while n and ids[n - 1] == (tok.pad_token_id or tok.eos_token_id):
                n -= 1
            res.append((tok.decode(ids[:n], skip_special_tokens=True), n))
        return res
    finally:
        m.set_attn_implementation(prev_attn)


def install(cls, **spec_config):
    """Replace cls._generate (RouteEngine) with the speculative one; spec_config: drafter, k, assistant, ngram_max/min."""
    import sys
    mod = sys.modules.get(cls.__module__)
    if mod is not None and getattr(mod, 'PROGRAM_PROMPT', PROGRAM_PROMPT) != PROGRAM_PROMPT:
        raise RuntimeError('PROGRAM_PROMPT differs from the route.py this was built for')
    cls._generate = _generate
    cls.spec_config = dict(spec_config)
    return cls
