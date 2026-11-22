#!/usr/bin/env python3
"""LM-HEAD FINE-TUNE: teach Qwen3.8-27B's OWN answer distribution to match human label distributions.

claude_lmscore.py showed the untouched model's letter distribution transfers better zero-shot than
any head trained on cached features (typed-decisions acc 0.686, KL 0.448, ECE 0.073). This keeps
that readout -- the LM head at the first answer token, restricted to the option letters -- and
trains LoRA adapters so that distribution matches the (soft) human target on jev_data:

    loss = - sum_o t_o * log softmax_{o in options}( lm_head(h_last)[letter_o] )

Nothing from jev_real is used, so evaluation there stays zero-shot. Training uses the exact prompt
claude_lmscore.py scores with, and a fresh random option order per example (choice questions), so
the adapter cannot learn a slot prior -- the positional bias R-25 diagnosed. LoRA is train_e2e.py's
(zero-initialised, fp32 masters), so step 0 is exactly the base model.

(53) --whole-case / --noul-defs train on claude_lmscore.py's whole-case / noul-definition prompts (default off).
(49) --rps-weight / --ordinal-readout: an RPS term and an ordinal readout for `score` questions (trainers/claude_ordinal.py;
default off: prompts, draws, losses and checkpoints unchanged). Score a --ordinal-readout run with the same readout.
EXACT RESUME (--resume auto, the default; see the "exact resume" section below): every checkpoint also writes a resume file
(optimizer, scheduler, data cursor, every RNG, loss trace) beside --out and under the resume S3 prefix, as do a spot interruption
notice, SIGTERM / SIGINT and an exception; a re-run with the same arguments continues from it bit for bit. --out is unchanged.

    python3 trainers/claude_lmtune.py --data /data/local/jev_data --steps 400 \
        --out /data/local/runs/lmtune.pt
    python3 trainers/claude_lmscore.py --lora-ckpt /data/local/runs/lmtune.pt ...

DATA PARALLEL (2026-10-03; see the "data parallel" section below, research/ddp.md): under torchrun (fleet/ddp_train.sh NGPU -- ...)
one run uses NGPU GPUs; --batch-size / --grad-accum are PER RANK, a step is NGPU x grad_accum x batch_size examples drawn exactly as
one GPU would draw them. WORLD_SIZE unset or 1: unchanged.

SPEED OPTIONS (speed, 2026-10-04; research/trainer_speed.md). Every default is the old behaviour (world 1: the same --out bytes and
logs; data parallel: the same numerics). Same values, only the host's waiting changes (resume-free): --dp-stop fold (DDP stop
requests ride the gradient all-reduce, read one step late: no host sync in a step), --h2d async (inputs copied to the GPU from pinned
memory without draining the queue), --tok-cache N (memoised tokenisation), --gc-every K / --gc-budget N (fewer layers recomputed).
Other kernels or shapes (in the identity when set): --compile-mode (inductor mode of --compile-layers), --compile-det pin (inductor's
deterministic mode in every layer compile: torch 2.13/2.14 drop $TORCHINDUCTOR_DETERMINISTIC after the first traced frame),
--pack-pad-to N (pack lengths rounded up to a multiple of N), --dp-share lpt (DDP: a tighter deal of a step's examples to ranks).

AUX VBLL (aux-vbll, 2026-10-04; see the "aux VBLL" section below): --aux-vbll also trains a variational Bayesian last layer (AuxVBLL)
on the DETACHED final hidden state at the decision position, jointly, on the same examples and soft targets, with its own optimizer:
answer + aleatoric + epistemic uncertainty from one forward. Nothing it does reaches the LoRA (same LoRA bits with or without it, under
data parallelism too). Saved in --out / the resume file as `aux_vbll`; scored by claude_lmscore.py --aux-vbll. Off = unchanged.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# (speedprobe3) --compile-layers: inductor's emulate_precision_casts must be a MODULE DEFAULT (torch 2.13 keeps config overrides per
# thread, and the backward compiles in the CUDA autograd thread), i.e. set in the environment before transformers imports
# torch._inductor.config (it does, at import). Only when the flag is on the command line; compile_decoder_layers verifies it.
import sys as _sys  # noqa: E402
if any(x == "--compile-layers" or x.startswith("--compile-layers=") for x in _sys.argv[1:]):
    os.environ.setdefault("TORCHINDUCTOR_EMULATE_PRECISION_CASTS", "1")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ls = _load("ls", os.path.join(HERE, "claude_lmscore.py"))
cf = ls.cf
ckpt = _load("ckpt", os.path.join(HERE, "claude_ckpt.py"))


def make_example(tok, unit, rng, max_len, permute, plain=False, whole_case=False, noul_defs=False, stats=None,
                 with_levels=False, chat=None):
    """(prompt text, target in the prompt's option order) or None if it cannot fit.
    `whole_case` (53; units grouped by ls.attach_case_questions): the case listing's question order is a nuisance like the
    option order, so with `permute` it is drawn fresh from `rng` (AFTER the option order, and only for a case of 2+
    questions, so the default path's draws are unchanged); without, it is the fixed scoring order (seed 0).
    `with_levels` (49): return a claude_ordinal.Example -- the same (text, target) pair, plus .levels = the option order
    (letter j shows level levels[j]) for a `score` question -- so the ordinal terms work in level order.
    `chat` (66): the model type's ls.chat_preset() entry (None for every backbone but Muse-Glimmer: prompts unchanged)."""
    k = len(unit["option_keys"])
    order = list(range(k))
    if permute and unit["qtype"] == "choice" and k >= 2:
        rng.shuffle(order)
    corder = None
    if whole_case and len(unit.get("case_questions") or ()) > 1:
        if permute:
            corder = list(range(len(unit["case_questions"])))
            rng.shuffle(corder)
        else:
            corder = ls.fixed_case_order(unit)
    text = ls.build_prompt(tok, unit, order, max_len, plain=plain, whole_case=whole_case, noul_defs=noul_defs,
                           case_order=corder, stats=stats, **({"chat": chat} if chat else {}))
    if text is None:
        return None
    v = np.asarray(unit["target"], dtype=np.float64)
    v = v / max(v.sum(), 1e-12)
    if with_levels:
        return ls.ordinal().Example(text, v[order], order if unit["qtype"] == "score" else None)
    return text, v[order]


def inner_model(model):
    """The module the model's own forward runs before its LM head (Gemma4ForConditionalGeneration: `model.model`, the
    multimodal wrapper that builds Gemma-4's masks -- NOT the bare text stack). (gemmafast) --exact-head fp32 reads it."""
    inner = getattr(model, getattr(model, "base_model_prefix", "model") or "model", None)
    if not isinstance(inner, torch.nn.Module) or inner is model:
        raise SystemExit("--exact-head fp32: no inner model under the LM head")
    return inner


def letter_head_fp32(model, h, label_ids):
    """(gemmafast) --exact-head fp32: the model's own output head -- lm_head, then output_multiplier and
    final_logit_softcapping when the config sets them, exactly the model's formula -- restricted to the option letters and
    computed in fp32 from the final hidden state h [B, H] (the full model computes it in bf16: every letter logit is rounded to a
    bf16 grid of ~0.12 near the cap, and tanh's backward (1 - y^2) is evaluated on bf16 y, so it moves in factor-2 steps or is
    0 once y rounds to 1). Equal to the model's head in exact arithmetic; only the rounding differs."""
    head = model.get_output_embeddings()
    z = h.float() @ head.weight[label_ids].float().T
    if getattr(head, "bias", None) is not None:
        z = z + head.bias[label_ids].float()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    mult = getattr(cfg, "output_multiplier", None)
    if mult:
        z = z * mult
    cap = getattr(cfg, "final_logit_softcapping", None)
    return cap * torch.tanh(z / cap) if cap else z


def to_dev(data, device, h2d="sync", **kw):
    """(speed) torch.tensor(data, device=device). h2d 'sync' (the default) is exactly that: on CUDA a pageable copy that first waits
    for the whole GPU queue to drain (a host sync per call). 'async': built on the host, pinned, copied with non_blocking=True --
    the same values, and the host goes on queueing work (the pinned block is held by the caching host allocator until the copy is
    done). On a CPU device the 'copy' is the tensor itself."""
    if h2d != "async":
        return torch.tensor(data, device=device, **kw)
    return host_to_dev(torch.tensor(data, **kw), device)


def host_to_dev(t, device):
    """(speed) --h2d async: a host tensor on `device` without a host sync (pinned + non_blocking on CUDA)."""
    if torch.device(device).type == "cuda":
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


class TokCache:
    """(speed E) --tok-cache N: `tok(text, add_special_tokens=...[, return_offsets_mapping=...])` for ONE string is memoised by
    (text, those two arguments) in an LRU of N entries; every other call and attribute goes to the tokenizer itself. A step
    tokenises each prompt several times (claude_lmscore.build_prompt's fit check, dp_share, exact_chunks, the forward); with the
    cache the prompt text, the token ids and the example stream are unchanged (tokenisation is deterministic; the cached
    encoding is the tokenizer's own object, only read by the callers). Results with return_tensors are never cached."""

    def __init__(self, tok, size):
        self.__dict__.update(_tok=tok, _size=max(int(size), 1), _memo=collections.OrderedDict(), hits=0, misses=0)

    def __call__(self, text, *args, **kw):
        if args or not isinstance(text, str) or set(kw) - {"add_special_tokens", "return_offsets_mapping"}:
            return self._tok(text, *args, **kw)
        key = (text, bool(kw.get("add_special_tokens", True)), bool(kw.get("return_offsets_mapping", False)))
        memo = self._memo
        enc = memo.get(key)
        if enc is not None:
            memo.move_to_end(key)
            self.__dict__["hits"] += 1
            return enc
        enc = self._tok(text, **kw)
        memo[key] = enc
        if len(memo) > self._size:
            memo.popitem(last=False)
        self.__dict__["misses"] += 1
        return enc

    def __getattr__(self, name):              # only for names not found on the proxy itself
        return getattr(self.__dict__["_tok"], name)

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)


def decoder_layers(model):
    """(speed D) The decoder layers in stack order: the longest ModuleList whose members all carry transformers' boolean
    `gradient_checkpointing` (GradientCheckpointingLayer); a multimodal model's vision tower is the shorter list."""
    best = []
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > len(best) and \
                all(isinstance(getattr(c, "gradient_checkpointing", None), bool) for c in m):
            best = list(m)
    if not best:
        raise SystemExit("--gc-every / --gc-budget: no checkpointable decoder layer list in this model")
    return best


def free_layer_ids(n_layers, gc_every, gc_budget, n_tokens):
    """(speed D) Indices of the decoder layers NOT checkpointed in one training forward of `n_tokens` tokens. --gc-every K: every
    layer i with i % K != 0 (K = 1: none, today). --gc-budget N (layer-tokens): floor(N / n_tokens) layers, spread evenly
    (activation memory grows with layers x tokens, so this bounds it for every forward size)."""
    if gc_every > 1:
        return {i for i in range(n_layers) if i % gc_every}
    if gc_budget > 0:
        k = min(n_layers, gc_budget // max(int(n_tokens), 1))
        return {(2 * j + 1) * n_layers // (2 * k) for j in range(k)} if k else set()
    return set()


@contextlib.contextmanager
def head_input(model, keep, pick):
    """(aux-vbll) With `keep` a list: while inside, every call of the model's output head (get_output_embeddings) appends
    pick(its input, detached) to it -- the final hidden state at exactly the positions whose logits the model computes
    (logits_to_keep), i.e. the decision state the letter logits are read from. A read-only forward pre-hook, removed on exit:
    the forward computes what it computes without it. `keep` None (every default path): nothing at all."""
    if keep is None:
        yield
        return
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda mod, args: keep.append(pick(args[0].detach())))
    try:
        yield
    finally:
        hook.remove()


def letter_logp(stack, lm_head, tok, texts, label_ids, device, model=None, head="model", h2d="sync", keep_h=None):
    """log p over each prompt's option letters, read at the last position: [B, Kmax] (-inf pad).
    With `model` (the --exact-path): the full model forward, last position only, one prompt at a time (no padding).
    `head` (gemmafast, exact path only): 'model' = the model's own head (bf16, the default, unchanged); 'fp32' = the same
    forward up to the final hidden state, then letter_head_fp32. `h2d` (speed): as to_dev. `keep_h` (aux-vbll): a list that
    receives each row's decision hidden state (detached; [1, H] per prompt on the exact path, [B, H] on the fast path); None (the
    default) keeps nothing and computes exactly as before."""
    nb = h2d == "async"
    if model is not None:
        zs = []
        for t in texts:
            e = tok(t, return_tensors="pt", add_special_tokens=False)
            e = {k: host_to_dev(x, device) for k, x in e.items()} if nb else e.to(device)
            if head == "fp32":
                hl = inner_model(model)(**e, use_cache=False).last_hidden_state[:, -1]
                if keep_h is not None:
                    keep_h.append(hl.detach())
                zs.append(letter_head_fp32(model, hl, label_ids))
                continue
            with head_input(model, keep_h, lambda x: x[:, -1]):
                zs.append(model(**e, use_cache=False, logits_to_keep=1).logits[:, -1].float())
        if head == "fp32":
            return torch.cat(zs, 0)
        return torch.cat(zs, 0)[:, label_ids]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: host_to_dev(x, device) for k, x in enc.items()} if nb else enc.to(device)
    h = stack(**enc, use_cache=False).last_hidden_state
    last = enc["attention_mask"].sum(1) - 1
    hl = h[torch.arange(len(texts), device=device), last]
    if keep_h is not None:
        keep_h.append(hl.detach())
    z = lm_head(hl).float()
    mult = getattr(lm_head, "logit_mult", None)             # (66) Muse-Glimmer's output_multiplier (before the cap), attached at load
    if mult:
        z = z * mult
    cap = getattr(lm_head, "softcap", None)                 # Gemma's final_logit_softcapping, attached at load
    if cap:
        z = cap * torch.tanh(z / cap)
    return z[:, label_ids]                                  # caller masks and normalises per row


CUDAGRAPH_MODES = ("max-autotune", "reduce-overhead")   # (speed B) the torch.compile modes that turn on inductor's CUDA graphs


def compile_decoder_layers(model, backend="inductor", recompile_limit=64, mode="default", det="env"):
    """(speedprobe3) --compile-layers: regional torch.compile (Module.compile, dynamic shapes) of every Gemma4TextDecoderLayer, our
    LoRA inside the compiled region. NOT bit-identical: inductor fuses (e.g. RMSNorm's reduction) and rounds where eager rounds only
    with emulate_precision_casts; measured at the size of the unbatched path's own equal-length-batch floor (research/
    gemma_speed_probe.md). Three things make it run as compiled, every call (each a failure seen on the GPU):
      - emulate_precision_casts must be a module default ($TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 before transformers is imported;
        set at the top of this file when the flag is given): torch 2.13 keeps config overrides PER THREAD, and the backward (the
        checkpoint recompute, AOTAutograd's lazily compiled backward) compiles in the CUDA autograd thread;
      - the recompile limit is per region (torch.compile(recompile_limit=)): the global one is per thread too, its default 8 is
        hit in the autograd thread, and a call past it silently runs EAGER;
      - each layer call gets a fresh shared_kv_states dict. The model threads one dict through all layers; the last sliding /
        full layer write their K/V into it and only KV-SHARED layers read it (Gemma-4-31B has none: asserted), so it is
        write-only. Its contents differ between a layer's forward and its checkpoint recompute, which would select another
        compiled graph on recompute (CheckpointError); a fresh dict changes no value.
    `mode` (speed B, --compile-mode): torch.compile's mode; 'default' passes none (the call is the old one). 'max-autotune*' tune
    GEMM templates (on-device benchmarking; TORCHINDUCTOR_DETERMINISTIC=1 skips it and keeps the ATen GEMMs) and run coordinate
    descent on the Triton configs; 'max-autotune' and 'reduce-overhead' add CUDA graphs (research/trainer_speed.md).
    `det` (speed B, --compile-det): 'env' (the default) leaves inductor's `deterministic` to $TORCHINDUCTOR_DETERMINISTIC -- which
    torch 2.13 / 2.14 drop in the compiling thread after the first traced frame (Dynamo restores torch.use_deterministic_algorithms'
    prior state on exit, and that setter also writes inductor's `deterministic`; overrides are per thread), so only the first
    forward graph and the backward graphs compiled in the CUDA autograd thread stay deterministic. 'pin': deterministic=True in
    every layer compile's own options (forward and backward, any thread): no on-device benchmarking that can change numerics.
    Returns the compiled layers (for layers_eager)."""
    import inspect
    import threading
    import torch._inductor.config as icfg
    if backend == "inductor":
        seen = {}
        t = threading.Thread(target=lambda: seen.update(v=bool(icfg.emulate_precision_casts)))
        t.start()
        t.join()
        if not (icfg.emulate_precision_casts and seen.get("v")):
            raise SystemExit("--compile-layers inductor: emulate_precision_casts is not on in every thread; export "
                             "TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 before python starts (or pass --compile-layers on the command line)")
    layers = [m for m in model.modules() if type(m).__name__ == "Gemma4TextDecoderLayer"]
    if not layers:
        raise SystemExit("--compile-layers: no Gemma4TextDecoderLayer in this model (Gemma-4 only)")
    shared = [i for i, L in enumerate(layers) if getattr(getattr(L, "self_attn", None), "is_kv_shared_layer", False)]
    if shared:
        raise SystemExit(f"--compile-layers: layers {shared[:5]} read shared K/V across layers; the per-call dict would change them")
    kw = {"backend": backend, "dynamic": True}
    if det == "pin":                          # torch.compile takes mode OR options: the mode's options, plus deterministic
        from torch._inductor import list_mode_options
        kw["options"] = {**(list_mode_options(mode, True) if mode and mode != "default" else {}), "deterministic": True}
    elif mode and mode != "default":
        kw["mode"] = mode
    if "recompile_limit" in inspect.signature(torch.compile).parameters:
        kw["recompile_limit"] = recompile_limit
    for L in layers:
        def _fwd(*args, _orig=L.forward, **k):
            if "shared_kv_states" in k:
                k["shared_kv_states"] = {}
            return _orig(*args, **k)
        L.forward = _fwd
        L.compile(**kw)
    return layers


class layers_eager:
    """with layers_eager(layers): the compiled layers run their eager code (bit-identical to the uncompiled model; the fresh
    write-only shared_kv_states dict changes no value). claude_lmtune evaluates this way, so a compiled run's val numbers are
    measured by the same forward as every other run's."""

    def __init__(self, layers):
        self.layers = list(layers or [])

    def __enter__(self):
        self.saved = [L._compiled_call_impl for L in self.layers]
        for L in self.layers:
            L._compiled_call_impl = None
        return self

    def __exit__(self, *exc):
        for L, c in zip(self.layers, self.saved):
            L._compiled_call_impl = c
        return False


def exact_letter_logits(model, tok, texts, label_ids, device, mode, head="model", pad_to=0, h2d="sync", keep_h=None):
    """--exact-batch: letter logits [B, Kmax] for several prompts through ONE full-model forward (the model's own head and
    final softcap, as the unbatched --exact-path). 'pad': right-padded rows with an attention mask, each row read at its OWN
    last real token. 'pack': the prompts concatenated in one row with per-prompt position_ids restarting at 0 and no
    attention_mask, which transformers turns into a block-diagonal causal mask (no padding at all). Both equal the unbatched
    path in exact arithmetic (gemmafast, 2026-10-02: bit-identical in float64 on a Gemma-4-31B-config model, tests in
    test_gemmafast.py); in bf16 they differ at batch-shape level (GEMM tiling), which on Gemma-4 is as large as the
    unbatched path's own equal-length-batch floor -- research/gemma_fast.md. `head` as letter_logp.
    `pad_to` (speed B, --pack-pad-to; pack only): the row is filled up to a multiple of pad_to tokens with ONE extra segment of
    pad tokens (positions from 0: its own block of the mask, after every real token), so a forward has one of a few lengths;
    no real token attends to it and it is never read, so the logits are the same in exact arithmetic (GEMM tiling aside).
    `h2d` (speed): as to_dev. `keep_h` (aux-vbll): as letter_logp ([B, H], the rows in `texts` order)."""
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    lens = [len(x) for x in ids]
    if mode == "pack":
        if pad_to or h2d == "async":
            flat, pos = [i for x in ids for i in x], [p for n in lens for p in range(n)]
            extra = -len(flat) % pad_to if pad_to else 0
            if extra:
                flat += [tok.pad_token_id if tok.pad_token_id is not None else flat[-1]] * extra
                pos += list(range(extra))
            inp, pos = to_dev([flat], device, h2d), to_dev([pos], device, h2d)
            last = to_dev(np.cumsum(lens) - 1, device, h2d)
        else:                                   # the old copies, exactly
            inp = torch.tensor([[i for x in ids for i in x]], device=device)
            pos = torch.tensor([[p for n in lens for p in range(n)]], device=device)
            last = torch.tensor(np.cumsum(lens) - 1, device=device)
        if head == "fp32":
            h = inner_model(model)(input_ids=inp, position_ids=pos, use_cache=False).last_hidden_state[0, last]
            if keep_h is not None:
                keep_h.append(h.detach())
            return letter_head_fp32(model, h, label_ids)
        with head_input(model, keep_h, lambda x: x[0]):                # (aux-vbll) the head's input [1, B, H] -> [B, H]
            z = model(input_ids=inp, position_ids=pos, use_cache=False, logits_to_keep=last).logits[0]      # [B, V]
        return z.float()[:, label_ids]
    if mode != "pad":
        raise ValueError(f"exact batch mode {mode!r}")
    if tok.pad_token_id is None:
        raise SystemExit("--exact-batch pad needs a pad token")
    inp = torch.full((len(ids), max(lens)), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), max(lens)), dtype=torch.long)
    for b, x in enumerate(ids):                             # RIGHT padding: real tokens keep positions 0..n-1
        inp[b, : len(x)] = torch.tensor(x)
        att[b, : len(x)] = 1
    last = to_dev([n - 1 for n in lens], device, h2d)
    rows = torch.arange(len(ids), device=device)
    if h2d == "async":                                      # (speed) the same tensors, copied without a host sync
        inp, att = host_to_dev(inp, device), host_to_dev(att, device)
    if head == "fp32":
        h = inner_model(model)(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False).last_hidden_state
        hl = h[rows, last]
        if keep_h is not None:
            keep_h.append(hl.detach())
        return letter_head_fp32(model, hl, label_ids)
    with head_input(model, keep_h, lambda x: x[rows, rows]):           # (aux-vbll) [B, B, H] at every row's last -> [B, H]
        z = model(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False, logits_to_keep=last).logits
    return z[rows, rows].float()[:, label_ids]             # z[b, j] = row b at position last[j]; keep j == b


def exact_chunks(tok, examples, budget, mode):
    """Split `examples` into forwards of at most `budget` tokens (PADDED tokens for 'pad', which first sorts by length so
    rows of similar length share a forward). Every example lands in exactly one chunk; a prompt longer than the budget
    gets a chunk of its own. The loss is a sum over examples, so the grouping does not change what a step computes."""
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t, _ in examples]
    idx = sorted(range(len(examples)), key=lambda i: lens[i]) if mode == "pad" else list(range(len(examples)))
    out, cur, cmax, csum = [], [], 0, 0
    for i in idx:
        m, s = max(cmax, lens[i]), csum + lens[i]
        if cur and (m * (len(cur) + 1) if mode == "pad" else s) > budget:
            out.append(cur)
            cur, m, s = [], lens[i], lens[i]
        cur.append(i)
        cmax, csum = m, s
    if cur:
        out.append(cur)
    return [[examples[i] for i in c] for c in out]


def fast_chunks(tok, examples, mode, batch_size, budget):
    """--microbatch (fast path): split a step's examples into forwards with less padding. Both modes first sort by prompt length
    (a stable sort, so ties keep draw order). 'regroup': batch_size rows per forward, as 'fixed' but length-grouped. 'budget': rows
    are added while rows x longest stays within `budget` padded tokens (a prompt longer than the budget gets a forward of its own).
    Every example lands in exactly one forward; the caller weights each forward's mean loss by len(chunk) / len(examples), so a
    step still computes the mean over its examples, as 'fixed' does."""
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t, _ in examples]
    idx = sorted(range(len(examples)), key=lambda i: lens[i])
    if mode == "regroup":
        groups = [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    elif mode == "budget":
        groups, cur, cmax = [], [], 0
        for i in idx:
            m = max(cmax, lens[i])
            if cur and m * (len(cur) + 1) > budget:
                groups.append(cur)
                cur, m = [], lens[i]
            cur.append(i)
            cmax = m
        if cur:
            groups.append(cur)
    else:
        raise ValueError(f"microbatch mode {mode!r}")
    return [[examples[i] for i in g] for g in groups]


def soft_ce(z, targets, ks):
    """Soft-target CE over each row's first k letters; also returns log p for metrics."""
    kmax = z.shape[1]
    mask = torch.arange(kmax, device=z.device)[None, :] < ks[:, None]
    logp = torch.log_softmax(z.masked_fill(~mask, -1e9), -1)
    return -(targets * logp.masked_fill(~mask, 0.0)).sum(-1).mean(), logp


def brier_rows(logp, targets, ks):
    """(50) multi-class Brier per row over its first k letters: sum_j (p_j - q_j)^2."""
    mask = torch.arange(logp.shape[1], device=logp.device)[None, :] < ks[:, None]
    return ((logp.exp() - targets) ** 2).masked_fill(~mask, 0.0).sum(-1)


def example_loss(z, tgt, ks, examples, readout="none", rps_weight=0.0, brier_weight=0.0):
    """(49) soft_ce read through the ordinal readout on `score` rows, plus rps_weight * RPS (level order) on them; (50) plus
    brier_weight * Brier on every row. All per example, so the batched, one-prompt and --exact-batch paths still compute the
    same step. Defaults: soft_ce itself."""
    if readout == "none" and not rps_weight:
        loss, logp = soft_ce(z, tgt, ks)
    else:
        od = ls.ordinal()
        lev, sc = od.batch_levels(examples, z.shape[1], z.device)
        loss, logp = soft_ce(od.readout_logits(z, ks, lev, sc, readout), tgt, ks)
        if rps_weight:
            loss = loss + rps_weight * od.rps_rows(logp, tgt, lev, sc).mean()
    if brier_weight:
        loss = loss + brier_weight * brier_rows(logp, tgt, ks).mean()
    return loss, logp


def batch_tensors(examples, device, h2d="sync"):
    """(texts, targets [B, Kmax] fp32, letter counts [B], Kmax). `h2d` (speed): 'sync' copies each row to the device as it goes
    (on CUDA 1 + B host syncs); 'async' fills the same fp32 rows on the host and copies them once without a sync (as to_dev)."""
    kmax = max(len(t) for _, t in examples)
    if h2d == "async":
        tgt = torch.zeros(len(examples), kmax)
        for b, (_, t) in enumerate(examples):
            tgt[b, : len(t)] = torch.tensor(t, dtype=torch.float32)
        return [x for x, _ in examples], host_to_dev(tgt, device), to_dev([len(t) for _, t in examples], device, h2d), kmax
    tgt = torch.zeros(len(examples), kmax, device=device)
    ks = torch.tensor([len(t) for _, t in examples], device=device)
    for b, (_, t) in enumerate(examples):
        tgt[b, : len(t)] = torch.tensor(t, dtype=torch.float32, device=device)
    return [x for x, _ in examples], tgt, ks, kmax


# --------------------------------------------------------------------------- aux VBLL (aux-vbll, 2026-10-04)
# --aux-vbll: a variational Bayesian last layer (Harrison et al. 2024, VBLL) trained JOINTLY with the LoRA, on the same examples and
# soft targets, but on DETACHED inputs: the final hidden state h at the decision position (the input of the model's own head at the
# position the letter logits are read from) and those letter logits. So its loss never reaches the LoRA: the LoRA's forwards,
# backwards, gradient all-reduce buckets, clipping and AdamW steps are those of a run without it (tested bit for bit). September's
# lesson (AGENT_NOTES.md): the VBLL head on cached features, prior and KL temper chosen by leave-one-source-out, was our best
# epistemic signal, and joint SVI THROUGH the LM collapsed -- hence the detach.
# Per step: after each training forward's own backward, the head's closed-form bound on that forward's examples is backpropagated
# into the head alone, weighted as the main loss is (so the step trains it on the GLOBAL step's mean); under data parallelism its
# gradients are summed in their own buckets AFTER the LoRA's; then (every rank, the same values) the KL term, a per-head clip at 1.0
# and the head's own AdamW (--aux-vbll-lr, the trainer's warmup + cosine shape) after the LoRA's step.
AUX_VBLL_COVS = ("diag", "full")
AUX_VBLL_DEFAULTS = {"aux_vbll_kl": "1", "aux_vbll_prior": "1", "aux_vbll_cov": "diag", "aux_vbll_lr": 1e-3, "aux_vbll_dim": 128,
                     "aux_vbll_samples": 64}


def aux_vbll_grid(a):
    """Canonicalise the --aux-vbll-* lists on `a` (so '1' and '1.0' give one identity) -> the heads: one dict per (prior, kl, cov)
    of the product, in that nesting order. ValueError on a bad value."""
    def floats(s, name, zero_ok):
        try:
            v = [float(x) for x in str(s).split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"--aux-vbll-{name}: a comma list of numbers, got {s!r}") from None
        if not v or any(not math.isfinite(x) or x < 0 or (x == 0 and not zero_ok) for x in v):
            raise ValueError(f"--aux-vbll-{name}: every value must be finite and {'>=' if zero_ok else '>'} 0, got {s!r}")
        return v
    kls = floats(a.aux_vbll_kl, "kl", True)                   # 0 = the bound alone: no pull towards the prior
    priors = floats(a.aux_vbll_prior, "prior", False)
    covs = [x.strip() for x in str(a.aux_vbll_cov).split(",") if x.strip()]
    if not covs or any(c not in AUX_VBLL_COVS for c in covs):
        raise ValueError(f"--aux-vbll-cov: a comma list of {'/'.join(AUX_VBLL_COVS)}, got {a.aux_vbll_cov!r}")
    a.aux_vbll_kl, a.aux_vbll_prior = ",".join(f"{x:g}" for x in kls), ",".join(f"{x:g}" for x in priors)
    a.aux_vbll_cov = ",".join(covs)
    if a.aux_vbll_dim < 1 or a.aux_vbll_samples < 1 or not a.aux_vbll_lr > 0:
        raise ValueError("--aux-vbll-dim / --aux-vbll-samples >= 1 and --aux-vbll-lr > 0")
    return [{"prior": s, "kl": k, "cov": c} for s in priors for k in kls for c in covs]


class AuxVBLL(torch.nn.Module):
    """(aux-vbll) G independent variational Bayesian last layers on the decision hidden state h [B, H]. For the option shown under
    letter j (unembedding row e_j) of a K-option prompt:

        z_j = tau * ell_j + phi_j^T beta,      q(beta) = N(mu, S),   p(beta) = N(0, s I)
        phi_j = u * (c_j - mean_{k<K} c_k) / sqrt(F),   u = rms(P rms(h)),   c_j = rms(Q rms(e_j))

    ell_j = the main readout's letter logit (detached): the prior mean beta = 0 is "no correction", so where the training corpus
    says little the predictive falls back on the LM's own readout, softened by the posterior variance (claude_vbll.py's design,
    whose leave-one-source-out version was the best epistemic signal). P [F, H] and tau are point estimates (the VBLL feature map;
    P learned), beta is the Bayesian weight; S = diag(sigma^2) ('diag', mean-field) or L L^T ('full'). The letter codes c are FIXED
    (Q a seeded random projection, saved as a buffer) and both u and c have unit RMS: with learned codes / unnormalised features a
    tempered KL simply shrinks phi to 0 (toy: the centred codes collapse, MI ~1e-7), leaving no epistemic signal at all.
    Permutation-equivariant: phi_j depends on (h, letter j) only, the same function for every slot, any K, any order -- as the LM
    head. Centring over the K shown letters removes a part common to all options (it cancels in the softmax but would inflate v).
    A linear layer on [h ; e_j] would not do: its h part is common to all options and cancels, leaving only a letter prior.
    Training: the closed-form Jensen bound (Harrison et al.'s discriminative classification bound; no sampling, no RNG)
        E_q[sum_j t_j log softmax(z)_j] >= sum_j t_j m_j - log sum_j exp(m_j + v_j / 2),   m = E_q z,  v = Var_q z,
    per step loss = mean_examples(-bound) + kl * KL(q || p) / n_train (kl = 1: the ELBO; larger: tempered towards the prior).
    Predictive: Monte Carlo over q with a FIXED set of draws (a seeded generator: deterministic, chunk-independent, never the global
    RNG); -> E_q softmax, mutual information (epistemic), expected entropy (aleatoric), sum_j Var_q p_j and the mean closed-form
    logit variance. Every parameter has a leading head axis G (the --aux-vbll-* grid); heads never interact (separate losses,
    per-head clipping). Its init draws come from its own generator, so the LoRA's init draws are those of a run without it."""

    def __init__(self, letters, heads, dim=128, seed=0):
        super().__init__()
        G, F, H = len(heads), int(dim), int(letters.shape[1])
        self.heads, self.F, self.H, self.seed = [dict(h) for h in heads], F, H, int(seed)
        g = torch.Generator().manual_seed(int(seed))
        self.P = torch.nn.Parameter(torch.randn(G, F, H, generator=g) / math.sqrt(H))
        Q = torch.randn(G, F, H, generator=g) / math.sqrt(H)
        s = torch.tensor([float(h["prior"]) for h in heads], dtype=torch.float32)
        self.mu = torch.nn.Parameter(torch.zeros(G, F))
        self.L_diag = torch.nn.Parameter((0.1 * s.sqrt()).expm1().log()[:, None].repeat(1, F))   # sigma = 0.1 sqrt(s) at init
        full = torch.tensor([h["cov"] == "full" for h in heads])
        self.L_off = torch.nn.Parameter(torch.zeros(G, F, F)) if bool(full.any()) else None
        self.log_tau = torch.nn.Parameter(torch.zeros(G))
        e = letters.detach().float().cpu()
        c = torch.einsum("gfh,kh->gkf", Q, e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6))
        self.register_buffer("codes", c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + 1e-6))     # [G, letters, F]
        self.register_buffer("prior", s, persistent=False)
        self.register_buffer("kl_w", torch.tensor([float(h["kl"]) for h in heads], dtype=torch.float32), persistent=False)
        self.register_buffer("full", full.float()[:, None, None], persistent=False)

    @classmethod
    def from_checkpoint(cls, ck, letters):
        """The head of a checkpoint's `aux_vbll` entry (aux_meta + state), on the CPU; `letters` [52, H] only sizes it (the saved
        codes are loaded)."""
        m = cls(letters, ck["heads"], ck["dim"], ck.get("seed", 0))
        m.load_state_dict(ck["state"])
        return m

    def meta(self):
        return {"heads": [dict(h) for h in self.heads], "dim": self.F, "hidden": self.H, "seed": self.seed}

    def sigma(self):
        return torch.nn.functional.softplus(self.L_diag) + 1e-6

    def chol(self):
        return torch.tril(self.L_off, -1) * self.full + torch.diag_embed(self.sigma())

    def moments(self, h, ell, ks, want_phi=False):
        """-> mean [G, B, K], var [G, B, K], mask [B, K] (K = ell.shape[1]; option j < ks[b] is shown) (, phi [G, B, K, F])."""
        kmax = ell.shape[1]
        mask = torch.arange(kmax, device=ell.device)[None, :] < ks[:, None]
        m = mask.float()
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        u = torch.einsum("gfh,bh->gbf", self.P, h)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
        w = self.codes[:, :kmax]
        wbar = torch.einsum("bk,gkf->gbf", m, w) / ks.to(w.dtype).clamp_min(1)[None, :, None]
        phi = u[:, :, None, :] * (w[:, None, :, :] - wbar[:, :, None, :]) * (m[None, :, :, None] / math.sqrt(self.F))
        mean = self.log_tau.exp()[:, None, None] * ell.float()[None] + torch.einsum("gbkf,gf->gbk", phi, self.mu)
        if self.L_off is None:
            var = (phi.pow(2) * self.sigma().pow(2)[:, None, None, :]).sum(-1)
        else:
            var = torch.einsum("gbkf,gfe->gbke", phi, self.chol()).pow(2).sum(-1)
        return (mean, var, mask, phi) if want_phi else (mean, var, mask)

    def bound(self, h, ell, tgt, ks):
        """The closed-form lower bound on E_q[sum_j t_j log softmax(z)_j], per head and example: [G, B]."""
        mean, var, mask = self.moments(h, ell, ks)
        lse = torch.logsumexp((mean + var / 2).masked_fill(~mask[None], -1e9), -1)
        return (tgt.float()[None] * mean.masked_fill(~mask[None], 0.0)).sum(-1) - lse

    def kl(self):
        """KL(q || p) per head: [G]."""
        s, sig = self.prior, self.sigma()
        tr = sig.pow(2).sum(-1)
        if self.L_off is not None:
            tr = tr + (torch.tril(self.L_off, -1) * self.full).pow(2).sum((-1, -2))
        return 0.5 * (tr / s + self.mu.pow(2).sum(-1) / s - self.F + self.F * s.log() - 2 * sig.log().sum(-1))

    def kl_loss(self, n_train):
        return (self.kl_w * self.kl()).sum() / max(int(n_train), 1)

    @torch.no_grad()
    def predict(self, h, ell, ks, n_samples=64, seed=0):
        """The predictive under q from `n_samples` fixed draws (seeded): -> p [G, B, K] (E_q softmax; 0 beyond each row's K),
        epistemic = mutual information [G, B], aleatoric = expected entropy [G, B], sum_j Var_q p_j [G, B], mean closed-form logit
        variance over the shown options [G, B]."""
        mean, var, mask, phi = self.moments(h, ell, ks, want_phi=True)
        eps = torch.randn(self.mu.shape[0], int(n_samples), self.F, generator=torch.Generator().manual_seed(int(seed)))
        eps = eps.to(phi.device)
        if self.L_off is None:
            beta = self.mu[:, None] + eps * self.sigma()[:, None]
        else:
            beta = self.mu[:, None] + torch.einsum("gef,gsf->gse", self.chol(), eps)
        z = self.log_tau.exp()[:, None, None, None] * ell.float()[None, None] + torch.einsum("gbkf,gsf->gsbk", phi, beta)
        ps = torch.softmax(z.masked_fill(~mask[None, None], -1e9), -1)          # [G, S, B, K]
        p = ps.mean(1)

        def ent(q):
            return -(q * q.clamp_min(1e-12).log()).sum(-1)
        alea = ent(ps).mean(1)
        epi = (ent(p) - alea).clamp_min(0.0)
        pvar = ps.var(1, unbiased=False).sum(-1)
        lvar = (var * mask[None]).sum(-1) / ks.to(var.dtype).clamp_min(1)[None]
        return p, epi, alea, pvar, lvar

    def clip_(self, max_norm=1.0):
        """clip_grad_norm_ per head (each head's own gradient norm, as torch's formula); -> the norms [G]."""
        ps = [p for p in self.parameters() if p.grad is not None]
        sq = sum(p.grad.reshape(p.shape[0], -1).pow(2).sum(1) for p in ps)
        coef = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
        for p in ps:
            p.grad.mul_(coef.view(-1, *([1] * (p.dim() - 1))))
        return sq.sqrt()


def aux_head_name(h):
    return f"s={h['prior']:g} kl={h['kl']:g} {h['cov']}"


def combine_readouts(p_main, p_aux, mode, w):
    """(aux-vbll) post hoc: 'mean' = (1 - w) p_main + w p_aux (a linear pool); 'geo' = p_main^(1-w) p_aux^w renormalised (a
    log-linear pool: a product of experts whose exponents sum to 1, so the LM's logits that the aux head also reads are not counted
    twice). Rows of [n, K] (zeros beyond a row's K stay 0)."""
    p_main, p_aux = np.asarray(p_main, dtype=np.float64), np.asarray(p_aux, dtype=np.float64)
    if mode == "mean":
        return (1 - w) * p_main + w * p_aux
    if mode != "geo":
        raise ValueError(f"combine mode {mode!r}")
    live = (p_main > 0) | (p_aux > 0)
    lg = (1 - w) * np.log(np.clip(p_main, 1e-300, 1)) + w * np.log(np.clip(p_aux, 1e-300, 1))
    lg = np.where(live, lg, -np.inf)
    lg = lg - lg.max(1, keepdims=True)
    q = np.where(live, np.exp(lg), 0.0)
    return q / q.sum(1, keepdims=True)


# --------------------------------------------------------------------------- exact resume (spot preemptions)
# The RESUME FILE `<out stem>.<identity hash>.resume` (beside --out, and under the resume S3 prefix) holds the scoring checkpoint's
# `lora` and `config` keys plus `train_state`: optimizer, scheduler, the data cursor, every RNG a step draws from (the example rng,
# Python's / numpy's / torch's global ones, CUDA's), the step, the val numbers and the per-step loss trace. The train order itself
# is re-derived from the seed and checked against a fingerprint. Written: with every checkpoint (--save-every / --eval-every,
# asynchronously); on a spot rebalance hint (then training continues); on a spot interruption notice, SIGTERM or SIGINT (then the
# run stops); on an exception (then it is re-raised) -- each of those synchronously, upload confirmed. Never mid optimizer step,
# so the state is always a step boundary; tmp file, load-back check, then rename, so a good file is never replaced by a bad one.
# --out and its S3 copies are exactly as before. Not a .pt name: fleet/s3_sync.sh mirrors runs/*.pt every 5 min.
# The name carries a hash of everything that shapes the run (RESUME_FREE_ARGS aside), so jobs that share an --out basename
# (task_46.sh a vs a clean) never share a resume file and a re-run with other arguments starts fresh beside the old state.
RESUME_VERSION = 1
# arguments that cannot change what a step computes: cadence, I/O and the resume machinery. --data and --init-lora are compared by
# CONTENT instead (the train / val fingerprints, the file's sha256): a re-run on another gq slot has another $JEV_D, so the same
# corpus and init sit under another path. Everything else in `vars(a)` must match.
RESUME_FREE_ARGS = ("eval_every", "save_every", "ckpt_s3", "out", "resume", "resume_s3", "spot_watch", "preempt_wait", "data",
                    "init_lora", "nockpt_max_tokens", "nockpt_max_padded",   # nockpt_*: bit-identical speed options (2026-10-02)
                    # (speed, 2026-10-04) the same values, only when the host waits / what is recomputed: a run may switch them
                    # on resume. dp_stop moves a stop's step boundary, not what any step computes; gc_*: as nockpt_*
                    "dp_stop", "h2d", "tok_cache", "gc_every", "gc_budget")
IMDS_URL = "http://169.254.169.254"       # EC2 instance metadata; $JEV_IMDS_URL overrides it (tests)


class Preempted(Exception):
    """Raised by the training thread itself at a check point once a stop was requested (signal or spot notice)."""


class Interrupts:
    """Requests from the signal handlers and the spot watcher thread. They only set fields (a handler that printed could hit a
    reentrant-write error on stdout); the training thread acts at its check points -- between micro-batches, eval chunks and
    steps, never inside the optimizer step -- so what it saves is always a consistent step boundary."""

    def __init__(self):
        self.reason = self.detail = None   # first stop request: 'SIGTERM' / 'SIGINT' / 'spot instance-action'
        self.exit_code = None
        self.last_signal = None
        self.rebalance = None              # a spot rebalance hint not yet acted on

    def stop(self, reason, code, detail=""):
        if self.reason is None:
            self.reason, self.exit_code, self.detail = reason, code, detail


def install_signal_handlers(intr):
    """SIGTERM / SIGINT -> a stop request (exit 128 + signum after the resume state is saved); a second SIGINT raises at once.
    Returns the previous handlers for restore_signal_handlers (none when not on the main thread)."""
    def handler(signum, frame):
        intr.last_signal = signum
        if intr.reason == "SIGINT" and signum == signal.SIGINT:
            raise KeyboardInterrupt
        intr.stop(signal.Signals(signum).name, 128 + signum)
    old = {}
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            old[s] = signal.signal(s, handler)
        except ValueError:                    # not the main thread (embedded use): the periodic snapshots still run
            pass
    return old


def restore_signal_handlers(old):
    for s, h in old.items():
        signal.signal(s, h if h is not None else signal.SIG_DFL)


class SpotWatcher(threading.Thread):
    """Polls the EC2 instance metadata service (IMDSv2: a session token from PUT /latest/api/token, refreshed before it expires or on
    a 401) every `interval` s for spot/instance-action (the interruption notice, ~2 min ahead; 404 until then) and
    events/recommendations/rebalance (an earlier, softer hint). Each kind is reported once through on_notice(kind, body). Off EC2
    (no answer to three token requests in a row before IMDS ever answered) it ends quietly with one log line."""

    def __init__(self, on_notice, base=None, interval=None):
        super().__init__(name="spot-watch", daemon=True)
        self.base = (base or os.environ.get("JEV_IMDS_URL") or IMDS_URL).rstrip("/")
        self.interval = float(interval if interval is not None else os.environ.get("JEV_SPOT_POLL", "5"))
        self.on_notice, self.done = on_notice, threading.Event()
        self.seen, self.token, self.token_t, self.reached, self.fails = set(), None, 0.0, False, 0
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open   # IMDS is never behind a proxy

    def _req(self, path, method="GET", headers=None):
        try:
            with self._open(urllib.request.Request(self.base + path, method=method, headers=headers or {}), timeout=2) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, ""

    def poll(self):
        if self.token is None or time.time() - self.token_t > 3000:
            code, body = self._req("/latest/api/token", "PUT", {"X-aws-ec2-metadata-token-ttl-seconds": "3600"})
            if code != 200:
                raise OSError(f"IMDS token HTTP {code}")
            self.token, self.token_t = body.strip(), time.time()
            if not self.reached:
                self.reached = True
                print(f"spot watcher: on ({self.base}, every {self.interval:g} s)", flush=True)
        for kind, path in (("instance-action", "/latest/meta-data/spot/instance-action"),
                           ("rebalance", "/latest/meta-data/events/recommendations/rebalance")):
            if kind in self.seen:
                continue
            code, body = self._req(path, headers={"X-aws-ec2-metadata-token": self.token})
            if code == 401:                   # token expired: a new one next poll
                self.token = None
                return
            if code == 200:
                self.seen.add(kind)
                self.on_notice(kind, body.strip())

    def run(self):
        while not self.done.is_set() and "instance-action" not in self.seen:
            try:
                self.poll()
                self.fails = 0
            except (OSError, ValueError) as e:  # URLError, timeouts and refused connections are OSErrors
                self.fails += 1
                if not self.reached and self.fails >= 3:
                    print(f"spot watcher: off (no instance metadata service: {str(e)[:100]})", flush=True)
                    return
            self.done.wait(self.interval)

    def stop(self):
        self.done.set()


def units_fingerprint(units):
    """sha256 over the units in order, by content (no paths): equal iff the same examples come in the same order. The whole-case
    listing (`case_questions`) is derived from the units themselves, so it is left out."""
    h = hashlib.sha256()
    for u in units:
        h.update(json.dumps({k: x for k, x in u.items() if k != "case_questions"}, sort_keys=True, ensure_ascii=False,
                            default=str).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:16]


def _plain(x):
    """JSON round trip, so an identity read back from a checkpoint compares equal to one built from argparse."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


# options added AFTER exact resume shipped: left out of the identity while at their default, so every earlier run's identity hash
# (and so its resume file) is unchanged by the new flag (2026-10-02, (50) --brier-weight)
IDENT_OMIT_AT_DEFAULT = {"brier_weight": 0.0, "exact_head": "model",   # exact_head: (gemmafast) 2026-10-02
                         "microbatch": "fixed", "microbatch_tokens": 0,  # --microbatch (2026-10-02): default runs keep their identity
                         "compile_layers": "off",                         # --compile-layers (speedprobe3, 2026-10-03): likewise
                         "world_size": 1,                                 # (ddp, 2026-10-03) torchrun's WORLD_SIZE (train())
                         # (speed, 2026-10-04) other kernels (autotuned GEMMs, CUDA graphs) / other forward lengths: other
                         # bf16 rounding, so part of a run's identity when set
                         "compile_mode": "default", "compile_det": "env", "pack_pad_to": 0, "dp_share": "snake",
                         # (aux-vbll, 2026-10-04) the head trains alongside the LoRA, so its options shape the run; the sub-options
                         # are refused without --aux-vbll, so off = every one at its default = every identity as before
                         "aux_vbll": False, **AUX_VBLL_DEFAULTS}


def run_identity(a, extra):
    ident = {k: x for k, x in sorted(vars(a).items())
             if k not in RESUME_FREE_ARGS and not (k in IDENT_OMIT_AT_DEFAULT and x == IDENT_OMIT_AT_DEFAULT[k])}
    ident.update(extra)
    return _plain(ident)


def identity_hash(ident):
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def resume_s3_prefix(a):
    """--resume-s3, else $JEV_RESUME_S3, else derived from the checkpoint prefix (--ckpt-s3 / $JEV_CKPT_S3): the fleet's per-node
    `<root>/<node>/live` maps to the SHARED `<root>/resume`, so a job re-run on another node finds the state; any other prefix P
    to `P/resume`. Empty = local resume files only."""
    if a.resume_s3 is not None:
        return a.resume_s3.rstrip("/")
    if os.environ.get("JEV_RESUME_S3") is not None:
        return os.environ["JEV_RESUME_S3"].rstrip("/")
    ck = (os.environ.get("JEV_CKPT_S3", "") if a.ckpt_s3 is None else a.ckpt_s3).rstrip("/")
    if not ck:
        return ""
    m = re.match(r"^(s3://.+)/[^/]+/live$", ck)
    return f"{m.group(1)}/resume" if m else f"{ck}/resume"


class S3Unavailable(Exception):
    pass


def _aws(*args):
    try:
        return subprocess.run(["aws", *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise S3Unavailable("no aws CLI") from None


def s3_head_step(url, tries=3):
    """The `step` metadata of an S3 object: None if there is no object, -1 if it has no such metadata; S3Unavailable on any
    other error (after `tries` attempts) -- the caller must not then assume the object is absent."""
    bucket, _, key = url[len("s3://"):].partition("/")
    err = ""
    for i in range(tries):
        r = _aws("s3api", "head-object", "--bucket", bucket, "--key", key)
        if r.returncode == 0:
            md = (json.loads(r.stdout or "{}").get("Metadata") or {})
            return int(md.get("step", -1))
        if re.search(r"\(404\)|Not Found|NoSuchKey", r.stderr or ""):
            return None
        err = (r.stderr or "").strip()[-300:]
        time.sleep(2 * (i + 1))
    raise S3Unavailable(err or "head-object failed")


def load_resume(path, mmap=False):
    ck = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if not isinstance(ck, dict) or not {"lora", "config", "train_state"} <= set(ck):
        raise ValueError(f"{path} is not a claude_lmtune resume file")
    return ck


def resume_step(path):
    return int(load_resume(path, mmap=True)["train_state"]["step"])


class ResumeWriter:
    """The resume file: torch.save to a tmp file, load it back, os.replace, then (with an S3 URL) `aws s3 cp` with the step as
    object metadata (the lookup compares steps without a download). save(): on one background thread, one write in flight (as
    claude_ckpt.AsyncCheckpointer). save_sync(): on the calling thread after the in-flight write; True only once the file is in
    place AND uploaded. A failed write is logged and leaves the previous file; training goes on."""

    def __init__(self, path, s3_url):
        self.path, self.s3_url = path, s3_url
        self.saved_step = None                 # last step whose file is in place (and uploaded, with an S3 URL)
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resume")
        self._fut = None

    def _snapshot(self, payload):
        snap = ckpt._to_host(payload)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return snap

    def save(self, payload, step):
        snap = self._snapshot(payload)
        self.drain()
        self._fut = self._ex.submit(self._write, snap, step)

    def save_sync(self, payload, step):
        snap = self._snapshot(payload)
        self.drain()
        return self._write(snap, step)

    def drain(self):
        fut, self._fut = self._fut, None
        if fut is not None:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 -- _write logs its own failures; this is a last resort
                print(f"  [resume] background write failed: {e!r}"[:400], flush=True)

    def close(self):
        self.drain()
        self._ex.shutdown(wait=True)

    def _write(self, snap, step):
        t0 = time.time()
        d = os.path.dirname(os.path.abspath(self.path))
        tmp = os.path.join(d, f".{os.path.basename(self.path)}.tmp{step}")
        try:
            os.makedirs(d, exist_ok=True)
            torch.save(snap, tmp)
            got = resume_step(tmp)              # the load-back check, before the good file is replaced
            if got != step:
                raise ValueError(f"read back step {got}")
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            try:
                os.remove(tmp)
            except OSError:
                pass
            print(f"  [resume] step {step}: NOT written ({e!r}); the previous resume file stands"[:400], flush=True)
            return False
        msg = f"  [resume] step {step}: {os.path.getsize(self.path) / 1e6:.0f} MB written in {time.time() - t0:.1f}s"
        ok = True
        if self.s3_url:
            try:
                r = _aws("s3", "cp", "--only-show-errors", "--metadata", f"step={step}", self.path, self.s3_url)
                ok, err = r.returncode == 0, (r.stderr or "")[-200:]
            except S3Unavailable as e:
                ok, err = False, str(e)
            msg += f"; S3 {'ok' if ok else 'FAILED: ' + err} ({time.time() - t0:.1f}s total)"
        print(msg, flush=True)
        if ok:
            self.saved_step = step
        return ok


def locate_resume(local, s3_url):
    """--resume auto: the newer (by step) of the local resume file and the S3 one; the S3 copy is downloaded only when it is newer.
    -> (path or None, s3_ok, where). s3_ok False: the S3 state is unknown (lookup error), so this run must not upload over it."""
    lstep = None
    if os.path.exists(local):
        try:
            lstep = resume_step(local)
        except Exception as e:  # noqa: BLE001 -- a torn or foreign file: ignore it, a fresh state replaces it
            print(f"resume: local {local} unreadable ({e!r}); ignored", flush=True)
    s3_ok, sstep, where = True, None, "local"
    if s3_url:
        try:
            sstep = s3_head_step(s3_url)
        except S3Unavailable as e:
            s3_ok = False
            print(f"resume: S3 lookup of {s3_url} FAILED ({e}); using local state only, and this run will NOT upload resume "
                  f"state (it might be older than what is there)", flush=True)
    if sstep is not None and (lstep is None or sstep < 0 or sstep > lstep):
        tmp = local + ".download"
        try:
            r = _aws("s3", "cp", "--only-show-errors", s3_url, tmp)
            if r.returncode != 0:
                raise S3Unavailable((r.stderr or "")[-200:])
            got = resume_step(tmp)
            if lstep is None or got > lstep:
                os.replace(tmp, local)
                lstep, where = got, "S3"
        except Exception as e:  # noqa: BLE001
            s3_ok = False
            print(f"resume: download of {s3_url} FAILED ({e!r}); using local state only, no S3 uploads"[:400], flush=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return (local if lstep is not None else None), s3_ok, where


def fetch_resume_arg(spec, local):
    """--resume PATH: a local resume file, or an s3:// one fetched beside `local` (removed once loaded)."""
    if spec.startswith("s3://"):
        tmp = local + ".fetched"
        try:
            r = _aws("s3", "cp", "--only-show-errors", spec, tmp)
        except S3Unavailable as e:
            raise SystemExit(f"--resume {spec}: {e}") from None
        if r.returncode != 0:
            raise SystemExit(f"--resume {spec}: download failed: {(r.stderr or '')[-200:]}")
        return tmp
    if not os.path.isfile(spec):
        raise SystemExit(f"--resume {spec}: no such file ('auto', 'off' or a resume file)")
    return spec


def note_other_resume_states(local, s3_url, stem, ident):
    """A fresh start beside resume states of OTHER configurations of the same --out: say so, with the identity fields that differ
    for the local ones (a re-run that should have resumed but did not -- data rebuilt differently, an argument changed -- shows
    here). The other states are left alone."""
    d, name = os.path.split(local)
    pat = re.compile(re.escape(stem) + r"\.[0-9a-f]{12}\.resume$")
    for f in sorted(f for f in (os.listdir(d) if os.path.isdir(d) else []) if pat.match(f) and f != name)[:3]:
        try:
            saved = _plain(load_resume(os.path.join(d, f), mmap=True)["train_state"].get("identity") or {})
            diff = ", ".join(k for k in sorted(set(saved) | set(ident)) if saved.get(k) != ident.get(k))
        except Exception as e:  # noqa: BLE001
            diff = f"unreadable: {e!r}"[:120]
        print(f"NOTE: resume state of ANOTHER configuration of this output, not used: {f} (differs in: {diff})", flush=True)
    if s3_url:
        try:
            r = _aws("s3", "ls", s3_url.rsplit("/", 1)[0] + "/" + stem + ".")
            for ln in [ln.split()[-1] for ln in (r.stdout or "").splitlines() if ln.split()][:6]:
                if pat.match(ln) and ln != name:
                    print(f"NOTE: resume state of ANOTHER configuration of this output on S3, not used: {ln}", flush=True)
        except S3Unavailable:
            pass


def rng_state(rng):
    """Every RNG a training step can draw from, as plain values (weights_only-loadable)."""
    ns = np.random.get_state()
    return {"rng": rng.getstate(), "random": random.getstate(),
            "numpy": [ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None}


def set_rng_state(rng, s):
    def tup(x):
        return tuple(tup(y) for y in x) if isinstance(x, (list, tuple)) else x
    rng.setstate(tup(s["rng"]))
    random.setstate(tup(s["random"]))
    n = s["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None:
        if torch.cuda.is_available() and torch.cuda.device_count() == len(s["cuda"]):
            torch.cuda.set_rng_state_all(s["cuda"])
        else:
            print(f"WARNING: the resume state has {len(s['cuda'])} CUDA RNG state(s) and this process sees "
                  f"{torch.cuda.device_count() if torch.cuda.is_available() else 0} device(s): CUDA RNG not restored", flush=True)


def refuse_mismatch(saved, ident, where):
    """Resuming a different configuration would train a model nobody asked for: stop, naming every field that differs."""
    saved = _plain(saved or {})
    diff = [k for k in sorted(set(saved) | set(ident)) if saved.get(k) != ident.get(k)]
    if diff:
        lines = "\n".join(f"  {k}: resume state {saved.get(k)!r} / this run {ident.get(k)!r}"[:300] for k in diff)
        raise SystemExit(f"REFUSING TO RESUME from {where}: it was written by a different configuration\n{lines}\n"
                         f"Fix the arguments, or pass --resume off to train from scratch.")


# --------------------------------------------------------------------------- data parallel (torchrun; research/ddp.md)
# `torchrun --nproc_per_node N trainers/claude_lmtune.py ...` (fleet/ddp_train.sh): N ranks, each a FULL model copy on its own GPU
# (LOCAL_RANK); only the LoRA gradients cross ranks. --batch-size and --grad-accum are PER RANK. A step is the GLOBAL batch of
# N x grad_accum x batch_size examples, and EVERY rank draws all of it with the single-GPU sampler (same cursor, same RNG draws), so
# the example stream depends on the seed and the global batch only: N ranks at b x a draw exactly what one GPU at b' x a' draws
# whenever b' x a' = N x b x a. Rank r trains on its share: on the default fixed path and the one-prompt exact path, micro-batches
# r*a .. r*a+a-1 of the step's N*a in draw order (each forward then IS a forward of the one-GPU run at b x (a*N)); on the paths
# that re-split a step (--microbatch regroup / budget, --exact-batch), a length-balanced deal (dp_share) that the existing
# chunking then splits as before. Every forward's loss is weighted by its share of the GLOBAL step; after all local accumulation
# the gradients are SUMMED over ranks (DataParallel.allreduce_grads; an explicit all-reduce, not torch DDP: the fast path runs the
# bare text stack and the head outside model.forward, chunk counts differ by rank, checkpointing and compile need no hooks). So
# every rank holds the one-GPU step's gradient, clips and steps AdamW identically, and keeps bit-identical parameters (checksummed
# every $JEV_DDP_CHECK_EVERY steps, default 50, and at the last; 0 = off). Rank 0 alone logs (the others' stdout goes to
# /dev/null), writes the checkpoint and resume state, uploads, and watches for spot notices; evaluation is sharded by val chunk and
# reduced in chunk order (the one-GPU numbers, bit for bit). A stop request on any rank (signal, rank 0's spot notice) is agreed at
# points every rank reaches together (the end of a step's forwards / backwards, after its optimizer step); rank 0 saves, every
# rank waits for that save, then all exit. WORLD_SIZE unset or 1: none of this runs.
def dp_share(tok, examples, world, rank, mode="snake"):
    """(ddp) Rank `rank`'s share of a step's examples on the paths that re-split a step: sorted by prompt length (longest first,
    stable), dealt in snake order (ranks 0..W-1, then W-1..0, ...), so every rank gets len/W examples of about the same total
    length (a step waits for its slowest rank); returned in draw order, so --exact-batch pack still packs in draw order.
    `mode` (speed, --dp-share): 'snake' (the default, above); 'lpt' = longest first, each example to the rank with the fewest
    tokens so far among those with fewer than ceil(len/W) examples (ties: the lowest rank). Same counts, tighter totals: on
    heavy-tailed prompt lengths (64 prompts of ~520 tokens on 8 ranks) the busiest rank carries ~2% more than the mean instead
    of ~10% with snake (research/trainer_speed.md). Every rank computes the same deal from the same lengths."""
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t, _ in examples]
    order = sorted(range(len(examples)), key=lambda i: -lens[i])
    if mode == "lpt":
        cap, tot, cnt, mine = -(-len(examples) // world), [0] * world, [0] * world, []
        for i in order:
            k = min((r for r in range(world) if cnt[r] < cap), key=lambda r: (tot[r], r))
            tot[k] += lens[i]; cnt[k] += 1
            if k == rank:
                mine.append(i)
        return [examples[i] for i in sorted(mine)]
    if mode != "snake":
        raise ValueError(f"dp share mode {mode!r}")
    mine = [i for j, i in enumerate(order) if (j % world if (j // world) % 2 == 0 else world - 1 - j % world) == rank]
    return [examples[i] for i in sorted(mine)]


class Later:
    """(speed A) A device tensor's values for the host LATER, without waiting now: on CUDA copied into pinned memory behind an
    event (queued after whatever produced the tensor); get() waits for that event -- by the next step long passed, since the host
    cannot run a whole step ahead of the GPU -- and returns .tolist(). On a CPU device: the tensor itself."""

    def __init__(self, t):
        self.ev = None
        if t.is_cuda:
            self.buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            self.buf.copy_(t, non_blocking=True)
            self.ev = torch.cuda.Event()
            self.ev.record()
        else:
            self.buf = t

    def get(self):
        if self.ev is not None:
            self.ev.synchronize()
        return self.buf.tolist()


class DataParallel:
    """(ddp) The torchrun process group, from $WORLD_SIZE / $RANK / $LOCAL_RANK. `on` is False at world 1 (or unset), and then
    nothing below runs: every call in train() sits behind dp.on, or is a no-op for rank 0 of 1."""

    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE") or 1)
        self.on = self.world > 1
        self.rank = int(os.environ.get("RANK") or 0) if self.on else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK") or 0) if self.on else 0
        self.check_every = int(os.environ.get("JEV_DDP_CHECK_EVERY", "50") or 0)
        self.dev, self.dist, self.checks = None, None, 0
        self.pending_stop, self.pending_checks = None, []   # (speed A) --dp-stop fold: last step's stop codes / checksums

    def init(self, device):
        """nccl on CUDA (GPU $LOCAL_RANK, made the current device, so 'cuda' means it everywhere -- transformers' device_map
        'cuda' too), gloo on CPU. The timeout ($JEV_DDP_TIMEOUT_MIN, default 30) must cover rank 0's longest solo stretch: its
        first model load (the others wait, then read the HF cache it filled) and a synchronous resume save + upload."""
        import datetime
        import torch.distributed as dist
        if device not in ("cuda", "cpu"):
            raise SystemExit(f"--device {device}: under torchrun use 'cuda' (each rank takes GPU $LOCAL_RANK) or 'cpu'")
        kw = {"timeout": datetime.timedelta(minutes=float(os.environ.get("JEV_DDP_TIMEOUT_MIN", "30")))}
        if device == "cuda":
            torch.cuda.set_device(self.local_rank)
            self.dev = torch.device("cuda", self.local_rank)
            kw.update(backend="nccl", device_id=self.dev)
        else:
            self.dev = torch.device("cpu")
            kw.update(backend="gloo")
        dist.init_process_group(**kw)
        if (dist.get_rank(), dist.get_world_size()) != (self.rank, self.world):
            raise SystemExit(f"process group rank {dist.get_rank()} of {dist.get_world_size()} != $RANK {self.rank} of "
                             f"$WORLD_SIZE {self.world}")
        self.dist = dist

    def close(self):
        self.barrier()
        self.dist.destroy_process_group()

    def barrier(self):
        """(collective) a 1-element all-reduce on this rank's device."""
        self.dist.all_reduce(torch.zeros(1, device=self.dev))

    def bcast_obj(self, x):
        """(collective) rank 0's `x` on every rank."""
        box = [x]
        self.dist.broadcast_object_list(box, src=0, device=self.dev)
        return box[0]

    def broadcast_params(self, params):
        """(collective) rank 0's tensors on every rank, in place (equal anyway: the same seed draws the same LoRA init)."""
        for p in params:
            self.dist.broadcast(p.data, src=0)

    def share(self, tok, examples, mode="snake"):
        return dp_share(tok, examples, self.world, self.rank, mode) if self.on else examples

    def agree_stop(self, intr):
        """(collective) Every rank's stop request; the first in rank order is adopted by all (one reason, one exit code on every
        rank). True if any rank has one."""
        v = torch.zeros(self.world, dtype=torch.int64, device=self.dev)
        if intr.reason is not None:
            v[self.rank] = intr.exit_code or 1
        self.dist.all_reduce(v)
        return self._adopt(intr, v.tolist())

    def _adopt(self, intr, codes):
        """Every rank's stop code (index = rank, 0 = none): the first in rank order is adopted. True if any."""
        hit = [(r, c) for r, c in enumerate(codes) if c]
        if not hit:
            return False
        r, c = hit[0]
        if r != self.rank:
            name = "spot instance-action" if c == 75 else signal.Signals(c - 128).name if c > 128 else f"exit {c}"
            intr.reason, intr.exit_code, intr.detail = f"{name} (rank {r})", c, intr.detail or ""
        return True

    def allreduce_grads_fold(self, params, intr, bucket_bytes=64 << 20):
        """(collective; speed A, --dp-stop fold) allreduce_grads with no host sync: the same buckets in the same order (so the
        same sums, bit for bit), then this rank's stop code in an int64 slot per rank, all-reduced right behind them and copied
        to the host without waiting (take_stop reads it at the next step's agree point, every rank the same values). The
        gradient-presence all-reduce becomes a local check: every rank trains every adapter (each runs every layer on a
        non-empty share), so a missing gradient is refused here instead of agreed on."""
        missing = sum(p.grad is None for p in params)
        if missing:
            raise RuntimeError(f"(ddp) --dp-stop fold: {missing} LoRA tensors have no gradient on rank {self.rank}; fold needs "
                               f"every rank to train every adapter (use --dp-stop agree)")
        self._reduce_buckets([p.grad for p in params], bucket_bytes)
        v = torch.zeros(self.world, dtype=torch.int64, device=self.dev)
        if intr.reason is not None:
            v[self.rank] = intr.exit_code or 1
        self.dist.all_reduce(v)
        self.pending_stop = Later(v)

    def take_stop(self, intr):
        """(speed A) The stop codes folded into the LAST step's gradient all-reduce (as agree_stop decides them). Every rank calls
        it at the same point, so all see the same codes and all stop, or none."""
        if self.pending_stop is None:
            return False
        codes, self.pending_stop = self.pending_stop.get(), None
        return self._adopt(intr, codes)

    def allreduce_grads(self, params, bucket_bytes=64 << 20):
        """(collective) SUM every trainable parameter's .grad over the ranks, in place, in flat buckets of <= 64 MiB (one dtype
        each). A parameter with no gradient on any rank stays None (AdamW skips it, as on one GPU); one with a gradient on some
        ranks only cannot happen (every rank runs every layer) and is refused rather than guessed."""
        have = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=self.dev)
        self.dist.all_reduce(have)
        h = have.tolist()
        if any(0 < x < self.world for x in h):
            raise RuntimeError("(ddp) the ranks disagree on which LoRA parameters have a gradient")
        self._reduce_buckets([p.grad for p, x in zip(params, h) if x], bucket_bytes)

    def reduce_aux(self, params, bucket_bytes=64 << 20):
        """(collective; aux-vbll) SUM the aux head's gradients over the ranks, in buckets of their own AFTER the LoRA's: the LoRA
        buckets -- and with them the order in which every LoRA gradient is summed -- are exactly those of a run without the head.
        Every rank runs the head on a non-empty share, so every head tensor has a gradient; a missing one is refused."""
        missing = sum(p.grad is None for p in params)
        if missing:
            raise RuntimeError(f"(ddp) --aux-vbll: {missing} head tensors have no gradient on rank {self.rank}")
        self._reduce_buckets([p.grad for p in params], bucket_bytes)

    def sum_(self, t):
        """(collective) all-reduce SUM of one tensor, in place."""
        self.dist.all_reduce(t)
        return t

    def gather_table(self, rows, n, width):
        """(collective; aux-vbll) per-chunk rows (chunk index, `width` floats), each chunk computed on one rank -> every chunk's
        row on every rank, in chunk order; exact, as gather_rows."""
        t = torch.zeros(n, width, dtype=torch.float64)
        for r in rows:
            t[r[0]] = torch.tensor(r[1:], dtype=torch.float64)
        return [(i, *x) for i, x in enumerate(self.sum_(t.to(self.dev)).tolist())]

    def _reduce_buckets(self, grads, bucket_bytes):
        """SUM `grads` over the ranks in place: consecutive tensors of one dtype in flat buckets of <= bucket_bytes."""
        i = 0
        while i < len(grads):
            j, size = i, 0
            while j < len(grads) and grads[j].dtype == grads[i].dtype and \
                    (j == i or size + grads[j].numel() * grads[j].element_size() <= bucket_bytes):
                size += grads[j].numel() * grads[j].element_size()
                j += 1
            flat = torch.cat([g.reshape(-1) for g in grads[i:j]])
            self.dist.all_reduce(flat)
            off = 0
            for g in grads[i:j]:
                g.copy_(flat[off : off + g.numel()].view_as(g))
                off += g.numel()
            i = j

    def verify_same(self, params, step, last=False, lazy=False):
        """(collective; every check_every steps, and at the last step) Assert that every rank holds BIT-identical trainable
        parameters: integer checksums of the raw bits (sum, and sum weighted by position), compared across ranks. Integer sums do
        not depend on the reduction order, so equal bits give equal checksums on any device. `lazy` (speed A, --dp-stop fold):
        the sums are read without waiting, and compared at the next step's agree point (resolve_checks)."""
        if not self.check_every or (step % self.check_every and not last):
            return
        s = torch.zeros(self.world, 2, dtype=torch.int64, device=self.dev)
        off = 0
        for p in params:
            x = p.detach().reshape(-1)
            b = x.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()]).to(torch.int64)
            s[self.rank, 0] += b.sum()
            s[self.rank, 1] += (b * torch.arange(off + 1, off + 1 + b.numel(), device=b.device, dtype=torch.int64)).sum()
            off += b.numel()
        self.dist.all_reduce(s)
        if lazy:
            self.pending_checks.append((step, Later(s)))
            return
        self._compare(step, s.tolist())

    def resolve_checks(self):
        """(speed A) Compare the checksums verify_same(lazy=True) queued."""
        pend, self.pending_checks = self.pending_checks, []
        for step, h in pend:
            self._compare(step, h.get())

    def _compare(self, step, rows):
        if any(r != rows[0] for r in rows):
            raise RuntimeError(f"(ddp) after step {step} the ranks' LoRA parameters DIFFER: checksums {rows}")
        self.checks += 1

    def step_stats(self, step_sum, step_n, peak):
        """(collective) the global step's loss sum and example count, and the largest peak memory of any rank (logging only)."""
        t = torch.tensor([float(step_sum), float(step_n)], dtype=torch.float64, device=self.dev)
        self.dist.all_reduce(t)
        m = torch.tensor([float(peak)], dtype=torch.float64, device=self.dev)
        self.dist.all_reduce(m, op=self.dist.ReduceOp.MAX)
        return float(t[0]), int(t[1]), float(m[0])

    def gather_rows(self, rows, n):
        """(collective) evaluate()'s per-chunk rows (chunk index, mean loss, rows, correct, RPS sum, score rows), each chunk
        computed on one rank -> every chunk's row on every rank, in chunk order. Each value arrives exactly (a float64 sum of
        itself and zeros), so the caller's sums are the one-GPU sums."""
        t = torch.zeros(n, 5, dtype=torch.float64)
        for r in rows:
            t[r[0]] = torch.tensor(r[1:], dtype=torch.float64)
        t = t.to(self.dev)
        self.dist.all_reduce(t)
        return [(i, x[0], int(x[1]), int(x[2]), x[3], int(x[4])) for i, x in enumerate(t.tolist())]


def main() -> int:
    """(ddp) Under torchrun (WORLD_SIZE > 1) ranks other than 0 write nothing to stdout (stderr, tracebacks included, stays) and
    the process group is closed after a normal finish; at world 1, train() exactly as before."""
    dp = DataParallel()
    if not dp.on:
        return train(dp)
    if dp.rank:
        with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
            rc = train(dp)
    else:
        rc = train(dp)
    dp.close()
    return rc


def train(dp) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ls.MODEL)
    ap.add_argument("--data", default="/data/local/jev_data")
    ap.add_argument("--allow-nc", action="store_true",
                    help="also train on non-commercial / research-only units (default: dropped, see cache_features.NC_LICENCE)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-mlp", action="store_true", help="also adapt gate/up/down projections")
    ap.add_argument("--lora-extra", default="", help="more targets, comma list; dotted ones match the module path "
                    "suffix, e.g. linear_attn.in_proj_qkv,linear_attn.in_proj_z,linear_attn.out_proj (Gated DeltaNet)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--no-permute", action="store_true")
    ap.add_argument("--exact-path", action="store_true", help="train and evaluate through the FULL model forward, one prompt "
                    "per forward (no padding, no bare text stack); for Gemma-4-12B-it, whose fast path is wrong")
    ap.add_argument("--exact-batch", choices=("off", "pad", "pack"), default="off",
                    help="with --exact-path: send the step's examples through the FULL model forward together instead of one "
                         "per forward (same examples, same order draws, same loss). 'pad' = right-padded rows + attention mask, "
                         "'pack' = one row, per-prompt position_ids (block-diagonal causal mask, no padding). Numerics differ "
                         "from 'off' at bf16 batch-shape level: run trainers/claude_exact_probe.py first (research/gemma_speed.md)")
    ap.add_argument("--exact-tokens", type=int, default=8192,
                    help="--exact-batch token budget per forward (padded tokens for 'pad'); bounds activation memory")
    ap.add_argument("--exact-head", choices=("model", "fp32"), default="model",
                    help="(gemmafast) with --exact-path: 'model' = the letter logits from the model's own head (bf16 lm_head and "
                         "softcap; the default, unchanged); 'fp32' = the same forward up to the final hidden state, then the model's "
                         "head formula on the option letters in fp32 (equal in exact arithmetic; removes the bf16 rounding of the "
                         "capped logits and of tanh's backward, which dominate Gemma-4's batch-shape noise -- research/gemma_fast.md). "
                         "Score such a checkpoint with claude_lmscore.py --exact-head fp32")
    ap.add_argument("--compile-layers", choices=("off", "inductor"), default="off",
                    help="(speedprobe3) speed, NOT bit-identical: regional torch.compile of every Gemma4 text decoder layer "
                         "(inductor, dynamic shapes, emulate_precision_casts; compile_decoder_layers). Training forwards / backwards "
                         "run compiled; evaluation runs the same layers eagerly (val numbers comparable with every other run). "
                         "'off' = unchanged (the default; identity and outputs as before)")
    ap.add_argument("--compile-mode", choices=("default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"),
                    default="default",
                    help="(speed B) with --compile-layers: torch.compile's mode for the compiled layers. 'default' = as before. "
                         "max-autotune* autotune GEMM templates (skipped under TORCHINDUCTOR_DETERMINISTIC=1, which keeps cuBLAS) and "
                         "tune Triton configs; 'max-autotune' and 'reduce-overhead' add CUDA graphs (pair them with --pack-pad-to, "
                         "or every new forward length records new graphs). Other kernels: part of the run identity when set. "
                         "research/trainer_speed.md")
    ap.add_argument("--compile-det", choices=("env", "pin"), default="env",
                    help="(speed B) with --compile-layers: 'env' (the default) = inductor's deterministic mode as "
                         "$TORCHINDUCTOR_DETERMINISTIC sets it, which torch 2.13 / 2.14 silently drop in the compiling thread after "
                         "the first traced frame (compile_decoder_layers); 'pin' = deterministic=True in every layer compile (no "
                         "numerics-changing autotuning anywhere, GEMM templates included). Part of the run identity when 'pin'")
    ap.add_argument("--pack-pad-to", type=int, default=0,
                    help="(speed B) --exact-batch pack, TRAINING forwards: fill each packed row up to a multiple of this many "
                         "tokens with one extra segment of pad tokens that no prompt attends to (logits equal in exact arithmetic; "
                         "a few forward lengths instead of one per step, for CUDA graphs / static shapes). 0 = off (the default)")
    ap.add_argument("--gc-every", type=int, default=1,
                    help="(speed D) gradient-checkpoint only every K-th decoder layer (i %% K == 0) in training forwards; the others "
                         "keep their activations (no recompute, more memory). Same values. 1 = every layer (the default)")
    ap.add_argument("--gc-budget", type=int, default=0,
                    help="(speed D) instead of --gc-every: per training forward of T tokens (padded tokens on padded paths), "
                         "leave floor(N / T) evenly spaced decoder layers un-checkpointed (N in layer-tokens; activation memory "
                         "~ layers x tokens, ~0.43 MiB per layer-token eager on Gemma-4-31B). Same values. 0 = off (the default)")
    ap.add_argument("--dp-stop", choices=("agree", "fold"), default="agree",
                    help="(speed A) data parallel only: 'agree' (the default) = a stop request is agreed by two all-reduces + host "
                         "reads per step (before the gradient all-reduce and after the optimizer step). 'fold' = no host sync in a "
                         "step: each rank's stop code is reduced right behind the gradient buckets and read one step later (a "
                         "stop lands one step later; the last step agrees as before), the grad-presence all-reduce becomes a local "
                         "check and the parameter checksums are compared a step late. Same gradients, bit for bit")
    ap.add_argument("--dp-share", choices=("snake", "lpt"), default="snake",
                    help="(speed) data parallel, paths that re-split a step (--exact-batch, --microbatch regroup / budget): how the "
                         "step's examples are dealt to ranks. 'snake' (the default) = by length in snake order; 'lpt' = longest first "
                         "to the least-loaded rank, equal counts (busiest rank ~2%% over the mean instead of ~10%%; dp_share). The "
                         "same examples and loss; other forwards per rank, so part of the run identity when 'lpt'")
    ap.add_argument("--h2d", choices=("sync", "async"), default="sync",
                    help="(speed A) host-to-GPU input copies (token ids, positions, targets): 'sync' (the default) = blocking "
                         "copies, each draining the GPU queue first; 'async' = pinned memory + non_blocking (the same values)")
    ap.add_argument("--tok-cache", type=int, default=0,
                    help="(speed E) memoise single-prompt tokenisation in an LRU of this many entries (TokCache: the same ids, "
                         "prompts and example stream); 0 = off (the default)")
    ap.add_argument("--aux-vbll", action="store_true",
                    help="(aux-vbll) also train an auxiliary VARIATIONAL BAYESIAN LAST LAYER (AuxVBLL) on the DETACHED final hidden "
                         "state at the decision position: per option z = tau * letter logit + phi(h, letter)^T beta, Gaussian "
                         "q(beta), KL to N(0, prior I) tempered by --aux-vbll-kl; same examples and soft targets, its own AdamW; "
                         "nothing reaches the LoRA (the same LoRA bits with or without it). In --out / resume as 'aux_vbll'; val logs "
                         "its soft-CE / Brier / epistemic; score with claude_lmscore.py --aux-vbll. Off (the default) = unchanged")
    ap.add_argument("--aux-vbll-kl", default=AUX_VBLL_DEFAULTS["aux_vbll_kl"],
                    help="(aux-vbll) KL temper(s) lambda: loss = mean(-bound) + lambda * KL / n_train, n_train = train units "
                         "(1 = the ELBO; larger = nearer the prior, i.e. the LM's own readout; claude_vbll_loso.py's lam). A comma "
                         "list trains one head per value (a grid with --aux-vbll-prior and --aux-vbll-cov)")
    ap.add_argument("--aux-vbll-prior", default=AUX_VBLL_DEFAULTS["aux_vbll_prior"],
                    help="(aux-vbll) prior variance(s) s of beta (features have ~unit norm, so s ~ the prior variance of a logit "
                         "correction); comma list = grid")
    ap.add_argument("--aux-vbll-cov", default=AUX_VBLL_DEFAULTS["aux_vbll_cov"],
                    help="(aux-vbll) posterior covariance(s): 'diag' (mean-field) or 'full' (Cholesky); comma list = grid")
    ap.add_argument("--aux-vbll-lr", type=float, default=AUX_VBLL_DEFAULTS["aux_vbll_lr"],
                    help="(aux-vbll) the head's AdamW lr (its own optimizer; the trainer's warmup + cosine shape)")
    ap.add_argument("--aux-vbll-dim", type=int, default=AUX_VBLL_DEFAULTS["aux_vbll_dim"],
                    help="(aux-vbll) feature dimension F (the Bayesian weight's size)")
    ap.add_argument("--aux-vbll-samples", type=int, default=AUX_VBLL_DEFAULTS["aux_vbll_samples"],
                    help="(aux-vbll) posterior draws for the head's predictive in val (a fixed seeded set: deterministic)")
    ap.add_argument("--plain-prompt", action="store_true", help="no chat template, ends 'Answer:', letters read with a "
                    "leading space (base models; the same prompt claude_lmscore.py --plain-prompt scores)")
    ap.add_argument("--whole-case", action="store_true",
                    help="(53) train on claude_lmscore.py --whole-case prompts: every question of the unit's case listed "
                         "(unlettered) before the asked one, the listing order shuffled per example; score with --whole-case")
    ap.add_argument("--noul-defs", action="store_true",
                    help="(53) show a noul's yes/no definitions when the data has them (claude_lmscore.py --noul-defs)")
    ap.add_argument("--nockpt-max-tokens", type=int, default=int(os.environ.get("JEV_NOCKPT_MAX_TOKENS", "0") or 0),
                    help="speed only, bit-identical: a TRAINING forward whose longest prompt has at most this many tokens runs without "
                         "gradient checkpointing (recompute is deterministic, so the step is unchanged; measured ~26 MiB per token of "
                         "activations on Gemma-4-31B). 0 = checkpoint every forward (the old behaviour)")
    ap.add_argument("--nockpt-max-padded", type=int, default=int(os.environ.get("JEV_NOCKPT_MAX_PADDED", "0") or 0),
                    help="speed only, bit-identical: also skip checkpointing for a TRAINING forward whose PADDED size (rows x longest "
                         "prompt, what its activation memory scales with) is at most this many tokens; either criterion skips. "
                         "0 = off (2026-10-02)")
    ap.add_argument("--microbatch", choices=("fixed", "regroup", "budget"), default="fixed",
                    help="fast (non-exact) path only: how a step's grad_accum x batch_size examples (the same draws, in the same order) "
                         "are split into forwards. 'fixed' = draw order, batch_size rows per forward (the old behaviour). 'regroup' = "
                         "sorted by length, then batch_size rows per forward. 'budget' = sorted by length, packed into forwards of at "
                         "most --microbatch-tokens PADDED tokens. Same examples and loss (the mean over the step's examples) in exact "
                         "arithmetic; less padding; bf16 numerics differ at batch-shape level (2026-10-02, user: 'do it')")
    ap.add_argument("--microbatch-tokens", type=int, default=0,
                    help="--microbatch budget: padded tokens per forward; 0 = batch_size x max_len (today's worst case)")
    ap.add_argument("--brier-weight", type=float, default=0.0,
                    help="(50) add this x multi-class Brier (sum over the row's letters of (p - q)^2) to the soft CE on every row")
    ap.add_argument("--rps-weight", type=float, default=0.0,
                    help="(49) add this x RPS (ranked probability score over the cumulative distribution in LEVEL order, not "
                         "the shuffled letter order) to the soft CE of every `score` example; 0 = off")
    ap.add_argument("--ordinal-readout", choices=("none", "cumlink", "smooth"), default="none",
                    help="(49) read `score` questions through an ordinal map of the letter logits, in training and val: "
                         "'cumlink' = location-scale cumulative logit derived from the logits (no new parameters), "
                         "'smooth' = projection onto unimodal distributions (trainers/claude_ordinal.py); score the "
                         "checkpoint with the same claude_lmscore.py --ordinal-readout")
    ap.add_argument("--loader", choices=("auto", "causal-lm", "image-text"), default="auto",
                    help="(66) model class, as claude_lmscore.py --loader: 'auto' = AutoModelForCausalLM whenever transformers maps "
                         "the model type to one (unchanged for every backbone so far), else AutoModelForImageTextToText (Muse-Glimmer)")
    ap.add_argument("--chat-preset", choices=("auto", "none"), default="auto",
                    help="(66) as claude_lmscore.py --chat-preset: the model type's CHAT_PRESETS entry (only Muse-Glimmer has one)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100, help="async checkpoint cadence (steps); evals also save")
    ap.add_argument("--ckpt-s3", default=None, help="S3 prefix for checkpoint upload; default $JEV_CKPT_S3")
    ap.add_argument("--eval-units", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-lora", default="", help="start from this claude_lmtune.py checkpoint's adapters (continued training, "
                    "e.g. (46) distillation); its LoRA rank/targets must match this run's")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="auto",
                    help="'auto' (default): continue from this configuration's resume state (beside --out, else on S3; the newer "
                         "wins), else start fresh; resume state is written with every checkpoint, on a spot notice, SIGTERM / "
                         "SIGINT and an exception. A PATH (local or s3://): continue from that file, refusing it if another "
                         "configuration wrote it. 'off': no resume state at all (the pre-resume behaviour)")
    ap.add_argument("--resume-s3", default=None,
                    help="S3 prefix for resume state; default $JEV_RESUME_S3, else from --ckpt-s3 / $JEV_CKPT_S3: the fleet's "
                         "per-node <root>/<node>/live -> the shared <root>/resume (a re-run on another node finds it), any "
                         "other P -> P/resume; '' = local only")
    ap.add_argument("--spot-watch", choices=("auto", "off"), default="auto",
                    help="poll EC2 instance metadata for spot interruption / rebalance notices (quietly off when not on EC2)")
    ap.add_argument("--preempt-wait", type=float, default=600,
                    help="after a spot interruption notice and the resume save, wait this long for the instance to go (exit "
                         "75 if it does not): exiting at once lets the job script carry on and the queue runner record a "
                         "FAILURE (two skip the job) and claim new work on a dying node")
    a = ap.parse_args()
    if a.exact_batch != "off" and not a.exact_path:
        raise SystemExit("--exact-batch applies to the --exact-path only")
    if a.exact_head != "model" and not a.exact_path:
        raise SystemExit("--exact-head applies to the --exact-path only")
    batched = a.exact_path and a.exact_batch != "off"
    if a.microbatch != "fixed" and a.exact_path:
        raise SystemExit("--microbatch applies to the fast path only (the exact path has --exact-batch)")
    regroup = a.microbatch != "fixed"           # (2026-10-02) the fast path's step examples are collected, then re-split
    mb_budget = a.microbatch_tokens or a.batch_size * a.max_len
    if a.rps_weight < 0 or a.brier_weight < 0:
        raise SystemExit("--rps-weight / --brier-weight must be >= 0")
    if (a.compile_mode != "default" or a.compile_det != "env") and a.compile_layers == "off":
        raise SystemExit("--compile-mode / --compile-det apply to --compile-layers")
    if a.pack_pad_to < 0 or (a.pack_pad_to and a.exact_batch != "pack"):
        raise SystemExit("--pack-pad-to: a positive token multiple, with --exact-batch pack")
    if a.gc_every < 1 or a.gc_budget < 0 or (a.gc_every > 1 and a.gc_budget):
        raise SystemExit("--gc-every K >= 1 or --gc-budget N >= 0, not both")
    if a.tok_cache < 0:
        raise SystemExit("--tok-cache: an entry count >= 0")
    aux_heads = None                            # (aux-vbll) off: none of it runs
    try:
        _grid = aux_vbll_grid(a)                # canonical lists on `a`, so the identity sees one spelling per value
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if a.aux_vbll:
        aux_heads = _grid
    elif any(getattr(a, k) != v for k, v in AUX_VBLL_DEFAULTS.items()):
        raise SystemExit("--aux-vbll-kl / -prior / -cov / -lr / -dim / -samples apply with --aux-vbll")
    fold = dp.on and a.dp_stop == "fold"        # (speed A) only under data parallel
    cudagraphs = a.compile_layers != "off" and a.compile_mode in CUDAGRAPH_MODES
    ordinal = bool(a.rps_weight) or a.ordinal_readout != "none"       # (49); off = the pre-(49) path exactly
    a.world_size = dp.world                   # (ddp) in the resume identity only when > 1 (IDENT_OMIT_AT_DEFAULT)
    if dp.on:
        dp.init(a.device)
        print(f"DDP world {dp.world}, per-rank batch {a.batch_size} x accum {a.grad_accum}, global "
              f"{dp.world * a.batch_size * a.grad_accum} examples/step (rank 0 logs, saves and watches; eval sharded; parameter "
              f"checksum every {dp.check_every or 'never'} steps)"
              + ("; --dp-stop fold: stop requests ride the gradient all-reduce, acted on one step later" if fold else "")
              + ("; --dp-share lpt" if a.dp_share != "snake" else ""),
              flush=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    te = _load("te", os.path.join(HERE, "train_e2e.py"))
    if dp.rank:                               # (ddp) rank 0 loads first (filling the HF cache once); the others then read it
        dp.barrier()
    mcfg = AutoConfig.from_pretrained(a.model)
    if hasattr(ls, "model_loader"):          # (66)
        loader, model_cls = ls.model_loader(mcfg, a.loader)
    else:                                     # a pre-(66) claude_lmscore.py beside this file (a job that fetched only this one)
        if a.loader == "image-text":
            raise SystemExit("--loader image-text needs the (66) claude_lmscore.py next to claude_lmtune.py")
        loader, model_cls = "causal-lm", AutoModelForCausalLM
    chat = ls.chat_preset(mcfg.model_type, a.plain_prompt) if a.chat_preset == "auto" and hasattr(ls, "chat_preset") else None
    if chat:
        print(f"chat preset for {mcfg.model_type!r}: {json.dumps(chat)}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if getattr(tok, "padding_side", "right") != "right":
        # the last real token is read from attention_mask.sum() - 1, which assumes RIGHT padding; set it rather than refuse
        # (Gemma's tokenizer defaults to left padding)
        print(f"tokenizer padding_side was {tok.padding_side!r}; setting 'right'", flush=True)
        tok.padding_side = "right"
    label_ids = [tok((" " + c) if a.plain_prompt else c, add_special_tokens=False)["input_ids"] for c in ls.LABELS]
    if any(len(x) != 1 for x in label_ids):
        raise SystemExit("an option letter is not a single token")
    label_ids = torch.tensor([x[0] for x in label_ids], device=a.device)

    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    model = model_cls.from_pretrained(a.model, dtype=dtype, device_map=a.device)
    if dp.on and not dp.rank:
        dp.barrier()
    if loader != "causal-lm":                         # (66)
        print(f"loader {loader}: {type(model).__name__} ({model_cls.__name__}) for model_type {mcfg.model_type!r}, text-only "
              f"prompts", flush=True)
    stack = None if a.exact_path else cf.find_text_stack(model, tok, a.device)
    full = model if a.exact_path else None
    lm_head = model.get_output_embeddings()
    _tc = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    lm_head.softcap = getattr(_tc, "final_logit_softcapping", None)
    lm_head.logit_mult = getattr(_tc, "output_multiplier", None)   # (66) None for every backbone but Muse-Glimmer
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    # --nockpt-max-tokens: the layers transformers marked for checkpointing; toggled per training forward (attribute only, no re-wrap)
    _gc_layers = [m for m in model.modules() if isinstance(getattr(m, "gradient_checkpointing", None), bool)]

    _gc_count = [0, 0]                        # [forwards without checkpointing, all training forwards]

    def gc_for(texts):
        """Checkpoint this training forward unless every prompt is short enough (--nockpt-max-tokens) or the padded forward is
        small enough (--nockpt-max-padded: rows x longest prompt)."""
        on = True
        if a.nockpt_max_tokens > 0 or a.nockpt_max_padded > 0:
            longest = max(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
            if a.nockpt_max_tokens > 0 and longest <= a.nockpt_max_tokens:
                on = False
            if a.nockpt_max_padded > 0 and longest * len(texts) <= a.nockpt_max_padded:
                on = False
        for m in _gc_layers:
            m.gradient_checkpointing = on
        _gc_count[0] += int(not on); _gc_count[1] += 1
        if _gc_count[1] in (1, 100, 1000) or _gc_count[1] % 10000 == 0:
            print(f"  [nockpt] training forwards without checkpointing: {_gc_count[0]} of {_gc_count[1]} "
                  f"(--nockpt-max-tokens {a.nockpt_max_tokens}, --nockpt-max-padded {a.nockpt_max_padded}, "
                  f"{len(_gc_layers)} checkpointed layers)", flush=True)
        return on

    _dec = decoder_layers(model) if a.gc_every > 1 or a.gc_budget else []   # (speed D) off: nothing below runs
    _gc_sel = [0, 0, 0]                       # [training forwards, un-checkpointed layers summed, largest layers x tokens]

    def gc_select(texts, packed, on=True):
        """(speed D) --gc-every / --gc-budget: after gc_for's on/off for this training forward, which decoder layers checkpoint.
        `packed`: the forward's tokens are the prompts' sum (+ --pack-pad-to fill), else rows x longest (padded)."""
        if not _dec:
            return
        n_tok = 0
        if a.gc_budget:
            lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            n_tok = (sum(lens) + (-sum(lens) % a.pack_pad_to if a.pack_pad_to else 0)) if packed else max(lens) * len(lens)
        free = free_layer_ids(len(_dec), a.gc_every, a.gc_budget, n_tok) if on else set()
        for i, m in enumerate(_dec):
            m.gradient_checkpointing = on and i not in free
        _gc_sel[0] += 1; _gc_sel[1] += len(free); _gc_sel[2] = max(_gc_sel[2], len(free) * n_tok)
        if _gc_sel[0] in (1, 100, 1000) or _gc_sel[0] % 10000 == 0:
            print(f"  [gc] training forward {_gc_sel[0]}: {len(free)} of {len(_dec)} decoder layers not checkpointed"
                  + (f" ({n_tok} tokens, --gc-budget {a.gc_budget}; largest layers x tokens so far {_gc_sel[2]})" if a.gc_budget
                     else f" (--gc-every {a.gc_every})")
                  + f"; mean {_gc_sel[1] / _gc_sel[0]:.1f} per forward", flush=True)

    def fwd_begin():
        """(speed B) CUDA-graph modes: a new training forward (inductor's cudagraph trees may overwrite last forward's outputs)."""
        if cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
    model.requires_grad_(False)               # freeze EVERYTHING first; see train_e2e.py
    targets = te.LORA_TARGETS if a.lora_mlp else ("q_proj", "k_proj", "v_proj", "o_proj")
    if a.lora_extra:
        targets = tuple(targets) + tuple(t for t in a.lora_extra.split(",") if t)
    # (66) a multimodal model whose vision tower the target rule would also hit (Muse-Glimmer) keeps LoRA on its language model;
    # None for Qwen / LFM2 / Gemma-4 (no hit outside it), so apply_lora runs exactly as before for them
    scope = ls.lora_scope(model, targets) if hasattr(ls, "lora_scope") else None
    te.apply_lora(model, a.lora_r, targets=targets, **({"scope": scope} if scope else {}))
    if a.init_lora:                           # refuse a partial load: a half-initialised adapter trains a different model
        ck0 = torch.load(a.init_lora, map_location="cpu")
        if int(ck0["config"]["lora_r"]) != a.lora_r or list(ck0["config"]["lora_targets"]) != list(targets) \
                or bool(ck0["config"].get("plain_prompt", False)) != bool(a.plain_prompt) \
                or ck0["config"].get("lora_scope") != scope:
            raise SystemExit(f"--init-lora config {ck0['config']} does not match r={a.lora_r} targets={targets}")
        want = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(ck0["lora"]) != want:
            raise SystemExit(f"--init-lora key mismatch: {len(set(ck0['lora']) ^ want)} keys differ")
        model.load_state_dict(ck0["lora"], strict=False)
        print(f"LoRA initialised from {a.init_lora} (step {ck0['config'].get('steps')})", flush=True)
    compiled_layers = None                    # (speedprobe3) --compile-layers; evaluate() runs them eagerly
    if a.compile_layers != "off":
        compiled_layers = compile_decoder_layers(model, a.compile_layers, mode=a.compile_mode, det=a.compile_det)
        print(f"compile-layers: {len(compiled_layers)} decoder layers compiled ({a.compile_layers}, "
              + (f"mode {a.compile_mode}, " if a.compile_mode != "default" else "")
              + ("deterministic pinned, " if a.compile_det == "pin" else "")
              + "dynamic, emulate_precision_casts); evaluation runs them eagerly", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if dp.on:                                 # (ddp) rank 0's initial adapters everywhere, as torch DDP does
        dp.broadcast_params(params)
    print(f"trainable {sum(p.numel() for p in params):,} params in {len(params)} tensors", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    train = list(cf.iter_units(a.data, ["train"], set(), None))
    val = list(cf.iter_units(a.data, ["val"], set(), None))
    if not a.allow_nc:                        # user 2026-10-01: never TRAIN on non-commercial data (eval may use it)
        # an older cache_features.py on a node has no is_noncommercial / no unit licence: fall back to the source family
        _nc = getattr(cf, "is_noncommercial", None) or (lambda u: str(u.get("source") or "").split("/")[0].split("__")[0] in {"ms_marco"})
        for nm, us in (("train", train), ("val", val)):
            drop = [u for u in us if _nc(u)]
            if drop:
                fams = {}
                for u in drop:
                    f = str(u["source"]).split("/")[0]; fams[f] = fams.get(f, 0) + 1
                print(f"NC FILTER ({nm}): dropped {len(drop)} non-commercial units by source {fams}", flush=True)
        train = [u for u in train if not _nc(u)]
        val = [u for u in val if not _nc(u)]
    if a.whole_case:                          # group BEFORE the K filter, so the listing shows the whole case
        for nm, us in (("train", train), ("val", val)):
            print(f"whole-case grouping ({nm}): {ls.attach_case_questions(us)}", flush=True)
    if a.noul_defs:
        nn = [u for u in train if u["qtype"] == "noul"]
        print(f"noul definitions: {sum(1 for u in nn if ls.noul_definitions(u.get('noul_criteria')))} of {len(nn)} "
              f"train noul units have them", flush=True)
    train = [u for u in train if len(u["option_keys"]) <= len(ls.LABELS)]
    val = [u for u in val if len(u["option_keys"]) <= len(ls.LABELS)]
    rng = random.Random(a.seed)
    rng.shuffle(train)
    val = random.Random(a.seed + 1).sample(val, min(a.eval_units, len(val)))
    wc_val, wc_train = {}, {}                 # (53) which rendering each prompt got (the --max-len overflow paths)
    if a.tok_cache:                           # (speed E) every later tok(...) of one prompt goes through the memo
        tok = TokCache(tok, a.tok_cache)
    val_ex = [e for e in (make_example(tok, u, random.Random(0), a.max_len, False, a.plain_prompt, a.whole_case,
                                       a.noul_defs, wc_val, with_levels=ordinal, chat=chat) for u in val) if e]
    print(f"train units {len(train)}, val units {len(val_ex)}", flush=True)
    if a.whole_case:
        print(f"whole-case val prompts: {dict(sorted(wc_val.items()))}", flush=True)

    aux = aux_opt = aux_sched = caps = aux_tr = None   # (aux-vbll) off: nothing below touches any of these
    aux_params, aux_v, aux_n, aux_labels = [], None, len(train), []
    if aux_heads:
        aux = AuxVBLL(lm_head.weight[label_ids].detach(), aux_heads, a.aux_vbll_dim, seed=a.seed).to(label_ids.device)
        aux_params = list(aux.parameters())
        if dp.on:                             # (ddp) equal anyway (its own seeded generator), as the LoRA
            dp.broadcast_params(aux_params)
        aux_opt = torch.optim.AdamW(aux_params, lr=a.aux_vbll_lr, betas=(0.9, 0.95), weight_decay=0.0)
        aux_sched = torch.optim.lr_scheduler.LambdaLR(
            aux_opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        aux_labels, caps = label_ids.tolist(), []
        names = "; ".join(f"{g}: {aux_head_name(h)}" for g, h in enumerate(aux_heads))
        per_ex = ", ".join("%.3g" % (h["kl"] / max(aux_n, 1)) for h in aux_heads)
        print(f"aux-vbll: {len(aux_heads)} head(s) [{names}], F {aux.F}, hidden {aux.H}, {sum(p.numel() for p in aux_params):,} "
              f"params; n_train {aux_n} (KL weight per example: {per_ex}); lr {a.aux_vbll_lr:g}, its own AdamW; inputs detached "
              f"(the LoRA never sees it)", flush=True)

    def aux_train(z, tgt, ks, w):
        """(aux-vbll) after a training forward's own backward: the head's bound on that forward's examples (their decision states,
        kept by the forward, and letter logits, both detached), weighted as the main loss was, backpropagated into the head alone."""
        nonlocal aux_tr
        h = torch.cat(caps, 0)
        caps.clear()
        if h.shape[0] != z.shape[0]:
            raise RuntimeError(f"--aux-vbll: {h.shape[0]} decision states kept for {z.shape[0]} prompts")
        bnd = aux.bound(h, z.detach(), tgt, ks)                         # [G, B]
        (-bnd.mean(1).sum() * w).backward()
        row = torch.cat([(-bnd.detach()).double().sum(1), bnd.new_full((1,), bnd.shape[1], dtype=torch.float64)])
        aux_tr = row if aux_tr is None else aux_tr + row

    def aux_eval_row(cap, z, tgt, ks, logp):
        """(aux-vbll) one val chunk: per head [soft-CE sum, Brier sum, correct, epistemic sum, aleatoric sum, logit-var sum], then the
        main readout's Brier sum and the row count."""
        h = torch.cat(cap, 0)
        p, epi, alea, _, lvar = aux.predict(h, z, ks, a.aux_vbll_samples)
        mask = (torch.arange(p.shape[-1], device=p.device)[None, :] < ks[:, None])[None]
        ce = -(tgt[None] * p.clamp_min(1e-12).log()).sum(-1).sum(1)
        br = ((p - tgt[None]) ** 2).masked_fill(~mask, 0.0).sum(-1).sum(1)
        acc = (p.argmax(-1) == tgt.argmax(-1)[None]).sum(1).to(ce.dtype)
        per = torch.stack([ce, br, acc, epi.sum(1), alea.sum(1), lvar.sum(1)], 1).flatten().tolist()
        return per + [float(brier_rows(logp, tgt, ks).sum()), float(len(ks))]

    intr = Interrupts()

    def check(sync=False):                    # a check point: act on a stop request now (never inside the optimizer step)
        if dp.on and not sync:                # (ddp) only at points every rank reaches together, where all agree (agree_stop)
            return
        if (dp.agree_stop(intr) if dp.on else intr.reason is not None):
            raise Preempted(intr.reason)

    def evaluate():
        nonlocal aux_v
        model.eval()
        tot, n, correct = 0.0, 0, 0
        rps_sum, n_sc = 0.0, 0                # (49) val RPS on score units; "val soft-CE" stays the readout's CE alone
        rows = []                             # per chunk: (index, mean loss, rows, correct, RPS sum, score rows)
        arows = []                            # (aux-vbll) per chunk: (index, aux_eval_row)
        with torch.no_grad(), layers_eager(compiled_layers):   # (speedprobe3) no-op unless --compile-layers
            chunks = exact_chunks(tok, val_ex, a.exact_tokens, a.exact_batch) if batched else \
                [val_ex[i : i + a.batch_size] for i in range(0, len(val_ex), a.batch_size)]
            for ci, chunk in enumerate(chunks):
                if dp.on and ci % dp.world != dp.rank:   # (ddp) chunk ci is rank (ci mod world)'s; the same chunks as one GPU
                    continue
                check()
                texts, tgt, ks, kmax = batch_tensors(chunk, a.device, a.h2d)
                cap = [] if aux is not None else None   # (aux-vbll) the chunk's decision states
                if batched:             # (speed) never --pack-pad-to here: val numbers stay those of the unpadded forward
                    z = exact_letter_logits(model, tok, texts, label_ids[:kmax], a.device, a.exact_batch, a.exact_head,
                                            h2d=a.h2d, keep_h=cap)
                else:
                    z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, full, a.exact_head, a.h2d,
                                    keep_h=cap)
                loss, logp = example_loss(z, tgt, ks, chunk, a.ordinal_readout)
                rps_c, nsc_c = 0.0, 0
                if ordinal:
                    lev, sc = ls.ordinal().batch_levels(chunk, kmax, a.device)
                    rps_c, nsc_c = float(ls.ordinal().rps_rows(logp, tgt, lev, sc).sum()), int(sc.sum())
                rows.append((ci, float(loss), len(texts), int((logp.argmax(-1) == tgt.argmax(-1)).sum()), rps_c, nsc_c))
                if aux is not None:
                    arows.append((ci, *aux_eval_row(cap, z, tgt, ks, logp)))
        if dp.on:                             # (ddp) every chunk's row on every rank, summed below in chunk order as one GPU sums
            rows = dp.gather_rows(rows, len(chunks))
            if aux is not None:
                arows = dp.gather_table(arows, len(chunks), 6 * len(aux.heads) + 2)
        for _, loss_c, n_c, correct_c, rps_c, nsc_c in rows:
            tot += loss_c * n_c; n += n_c
            correct += correct_c
            rps_sum += rps_c; n_sc += nsc_c
        model.train()
        if ordinal:
            print(f"  val RPS(score) {rps_sum / max(n_sc, 1):.4f} over {n_sc} score units", flush=True)
        if aux is not None:                   # (aux-vbll) the head's val numbers next to the main readout's, summed in chunk order
            sums = [0.0] * (6 * len(aux.heads) + 2)
            for r in arows:
                for i, x in enumerate(r[1:]):
                    sums[i] += x
            nv, taus = max(sums[-1], 1.0), aux.log_tau.detach().exp().tolist()
            keys = ("soft_ce", "brier", "acc", "epistemic", "aleatoric", "logit_var")
            aux_v = {"heads": [{k: sums[6 * g + i] / nv for i, k in enumerate(keys)} for g in range(len(aux.heads))],
                     "main_soft_ce": tot / max(n, 1), "main_brier": sums[-2] / nv, "n": int(sums[-1])}
            for g, (h, m) in enumerate(zip(aux.heads, aux_v["heads"])):
                print(f"  aux-vbll val head {g} ({aux_head_name(h)}): soft-CE {m['soft_ce']:.4f}  Brier {m['brier']:.4f}  acc "
                      f"{m['acc']:.4f}  epistemic {m['epistemic']:.3e}  aleatoric {m['aleatoric']:.4f}  logit-var "
                      f"{m['logit_var']:.3e}  tau {taus[g]:.3f}", flush=True)
            print(f"  aux-vbll val main readout: soft-CE {aux_v['main_soft_ce']:.4f}  Brier {aux_v['main_brier']:.4f}  "
                  f"({aux_v['n']} val prompts)", flush=True)
        return tot / max(n, 1), correct / max(n, 1)

    # exact resume: this run's identity, its resume state if one exists (see the section above main)
    cursor, start, post_pending = 0, 0, False
    v, best, v0 = (float("nan"), float("nan")), (float("nan"), float("nan"), 0), None
    losses, loss_hist = [], []                # the logged loss of every step: floats, the newest still as tensors (no sync)
    writer, ident, ihash, code_sha = None, None, "", ""
    if a.resume != "off":
        t_id = time.time()
        code_sha = file_sha256(os.path.abspath(__file__))
        ident = run_identity(a, {"train_fp": units_fingerprint(train), "val_fp": units_fingerprint(val),
                                 "init_lora_sha256": file_sha256(a.init_lora) if a.init_lora else "",
                                 "lora_targets": list(targets), "lora_scope": scope, "loader": loader, "chat_preset": chat,
                                 "model_revision": getattr(mcfg, "_commit_hash", None), "resume_version": RESUME_VERSION})
        ihash = identity_hash(ident)
        stem = os.path.splitext(os.path.basename(a.out))[0]
        rlocal = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"{stem}.{ihash}.resume")
        rprefix = resume_s3_prefix(a)
        rs3 = f"{rprefix}/{stem}.{ihash}.resume" if rprefix else ""
        if dp.rank:                           # (ddp) rank 0 looks (and downloads); every rank then reads the file it chose
            rpath, s3_ok, where = None, True, ""
        elif a.resume == "auto":
            rpath, s3_ok, where = locate_resume(rlocal, rs3)
        else:
            rpath, s3_ok, where = fetch_resume_arg(a.resume, rlocal), True, a.resume
        if dp.on:
            rpath, s3_ok, where = dp.bcast_obj((rpath, s3_ok, where))
        print(f"resume: identity {ihash} ({time.time() - t_id:.1f}s); state {rlocal}" + (f" + {rs3}" if rs3 and s3_ok else "")
              + (f"; continuing from the {where} copy" if rpath else "; none yet: fresh start"), flush=True)
        if rpath is None and not dp.rank:
            note_other_resume_states(rlocal, rs3 if s3_ok else "", stem, ident)
        if not dp.rank:                       # (ddp) rank 0 alone writes resume state
            writer = ResumeWriter(rlocal, rs3 if s3_ok else "")
        if rpath:
            ck = load_resume(rpath)
            if rpath.endswith(".fetched"):        # an explicit s3:// --resume, now in memory
                if dp.on:                         # (ddp) on every rank
                    dp.barrier()
                if not dp.rank:
                    os.remove(rpath)
            ts = ck["train_state"]
            refuse_mismatch(ts.get("identity"), ident, where)
            want = {n for n, p in model.named_parameters() if p.requires_grad}
            if set(ck["lora"]) != want:
                raise SystemExit(f"resume state {rpath}: LoRA key mismatch ({len(set(ck['lora']) ^ want)} keys differ)")
            model.load_state_dict(ck["lora"], strict=False)
            opt.load_state_dict(ts["optimizer"])
            sched.load_state_dict(ts["scheduler"])
            if aux is not None:                   # (aux-vbll) the identity says the state has a head; refuse a file without one
                if "aux_vbll" not in ck or "aux_vbll" not in ts:
                    raise SystemExit(f"resume state {rpath}: no aux_vbll head state (written without --aux-vbll?)")
                aux.load_state_dict(ck["aux_vbll"]["state"])
                aux_opt.load_state_dict(ts["aux_vbll"]["optimizer"])
                aux_sched.load_state_dict(ts["aux_vbll"]["scheduler"])
                aux_v = ts["aux_vbll"]["val"]
            set_rng_state(rng, ts["rng"])
            cursor, start, post_pending = int(ts["cursor"]), int(ts["step"]), bool(ts["post_pending"])
            v0, v, best = tuple(ts["val"]["v0"]), tuple(ts["val"]["last"]), tuple(ts["val"]["best"])
            losses = [float(x) for x in ts["losses"]]
            wc_train.update(ts["wc_train"])
            if writer is not None and os.path.abspath(rpath) == rlocal:
                writer.saved_step = start
            for k, mine in (("code_sha256", code_sha), ("torch", str(torch.__version__))):
                if ts.get(k) != mine:
                    print(f"NOTE: the resume state was written with {k} {ts.get(k)}, this run has {mine}", flush=True)
            print(f"RESUMED at step {start}/{a.steps}: cursor {cursor}, written {ts.get('time')} ({ts.get('reason')}); last val "
                  f"soft-CE {v[0]:.4f} acc {v[1]:.4f}, best {best[0]:.4f} at step {best[2]}"
                  + ("; redoing that step's eval / checkpoint" if post_pending else ""), flush=True)
            del ck, ts

    if v0 is None:
        v0 = evaluate()
        print(f"step 0  val soft-CE {v0[0]:.4f}  acc {v0[1]:.4f}  (the untouched model)", flush=True)
    model.train()
    t0 = time.time()
    saver = ckpt.AsyncCheckpointer(a.out, s3_prefix=a.ckpt_s3)

    def config(step):
        return {"model": a.model, "lora_r": a.lora_r, "lora_targets": list(targets),
                "steps": step, "lr": a.lr, "batch_size": a.batch_size,
                "grad_accum": a.grad_accum, "max_len": a.max_len,
                "permute": not a.no_permute, "data": a.data, "init_lora": a.init_lora, "plain_prompt": a.plain_prompt, "exact_path": a.exact_path,
                "exact_batch": a.exact_batch, "exact_tokens": a.exact_tokens,
                "val_soft_ce": v[0], "val_acc": v[1],
                # (53) only when on, so a default run's checkpoint config is unchanged
                **({"whole_case": True} if a.whole_case else {}),
                **({"noul_defs": True} if a.noul_defs else {}),
                # (49) likewise; claude_lmscore.py refuses to score a readout-trained adapter otherwise
                **({"rps_weight": a.rps_weight, "ordinal_readout": a.ordinal_readout} if ordinal
                   else {}),
                **({"brier_weight": a.brier_weight} if a.brier_weight else {}),   # (50) only when on
                **({"microbatch": a.microbatch, "microbatch_tokens": mb_budget} if regroup else {}),   # (2026-10-02) only when on
                **({"exact_head": a.exact_head} if a.exact_head != "model" else {}),   # (gemmafast) likewise
                **({"compile_layers": a.compile_layers} if a.compile_layers != "off" else {}),   # (speedprobe3) likewise
                **({"compile_mode": a.compile_mode} if a.compile_mode != "default" else {}),     # (speed) likewise
                **({"compile_det": a.compile_det} if a.compile_det != "env" else {}),            # (speed) likewise
                **({"pack_pad_to": a.pack_pad_to} if a.pack_pad_to else {}),                     # (speed) likewise
                **({"dp_share": a.dp_share} if a.dp_share != "snake" else {}),                   # (speed) likewise
                # (ddp) likewise; batch_size / grad_accum above are per rank
                **({"world_size": dp.world, "global_batch": dp.world * a.batch_size * a.grad_accum} if dp.on else {}),
                # (66) likewise: claude_lmscore.py rebuilds the adapters under the same scope
                **({"lora_scope": scope} if scope else {}),
                **({"chat_preset": chat} if chat else {}),
                **({"loader": loader} if loader != "causal-lm" else {}),
                # (aux-vbll) only when on: the head's spec, training settings and last val numbers (its tensors: the `aux_vbll` key)
                **({"aux_vbll": {**aux.meta(), "lr": a.aux_vbll_lr, "samples": a.aux_vbll_samples, "n_train": aux_n,
                                 "label_ids": aux_labels, "val": aux_v}} if aux is not None else {})}

    def aux_entry():
        """(aux-vbll) the checkpoint's `aux_vbll` key: everything claude_lmscore.py --aux-vbll needs to rebuild the head."""
        return {"state": aux.state_dict(), **aux.meta(), "samples": a.aux_vbll_samples, "label_ids": aux_labels}

    def resume_payload(step, reason, rs, cur, wc, pending):
        """The resume file: --out's two keys + train_state. `pending`: stopped after the step's optimizer update but before its
        eval / checkpoint, which the resumed run then redoes. (aux-vbll) With the head: --out's `aux_vbll` key too, and its optimizer,
        scheduler and last val numbers in train_state."""
        if loss_hist:
            losses.extend(torch.stack(loss_hist).float().cpu().tolist())
            loss_hist.clear()
        ts = {"version": RESUME_VERSION, "step": step, "identity": ident, "identity_hash": ihash,
              "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "cursor": cur, "rng": rs,
              "post_pending": pending, "val": {"v0": list(v0), "last": list(v), "best": list(best)},
              "losses": losses[:step], "wc_train": wc, "reason": reason,
              "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_sha256": code_sha,
              "torch": str(torch.__version__), "device": a.device}
        if aux is None:
            return {"lora": {n: p for n, p in model.named_parameters() if p.requires_grad}, "config": config(step), "train_state": ts}
        ts["aux_vbll"] = {"optimizer": aux_opt.state_dict(), "scheduler": aux_sched.state_dict(), "val": aux_v}
        return {"lora": {n: p for n, p in model.named_parameters() if p.requires_grad}, "config": config(step),
                "aux_vbll": aux_entry(), "train_state": ts}

    def save(step, with_resume=True):
        if dp.rank:                           # (ddp) rank 0 alone writes and uploads
            return
        ms = saver.save({"lora": {n: p for n, p in model.named_parameters() if p.requires_grad}, "config": config(step),
                         **({"aux_vbll": aux_entry()} if aux is not None else {})}, step)
        print(f"  [ckpt] step {step} snapshot, training blocked {ms:.0f} ms", flush=True)
        if writer is not None and with_resume:
            writer.save(resume_payload(step, "checkpoint", rng_state(rng), cursor, dict(wc_train), False), step)

    def post(step):                           # after a step's optimizer update: eval / checkpoint on their cadence
        nonlocal v, best
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate()
            if math.isnan(best[0]) or v[0] < best[0]:
                best = (v[0], v[1], step)
            print(f"step {step}  val soft-CE {v[0]:.4f}  acc {v[1]:.4f}", flush=True)
            if a.whole_case:
                print(f"  whole-case train prompts so far: {dict(sorted(wc_train.items()))}", flush=True)
            save(step)
        elif step % a.save_every == 0:
            save(step)

    phase, boundary, cur_step, rebalanced = "idle", None, start, False

    def emergency(reason):
        """Write the newest CONSISTENT state synchronously (load-back checked, then uploaded): while examples / forwards /
        backwards are in flight, the last step boundary (parameters and optimizer are untouched until opt.step); from the step's
        eval / checkpoint on, the step itself. Inside the optimizer step the parameters may be half-updated: nothing is written
        and the last snapshot stands."""
        if writer is None:
            return False
        writer.drain()
        if phase == "opt":
            print(f"[resume] {reason} INSIDE the optimizer update of step {cur_step}: the state may be half-updated, so nothing "
                  f"is written; the last resume state (step {writer.saved_step}) stands", flush=True)
            return False
        if phase == "fb":
            (st, rs, cur, wc), pending = boundary, False
        else:
            st, rs, cur, wc, pending = cur_step, rng_state(rng), cursor, dict(wc_train), phase == "post"
        if st <= 0 or writer.saved_step == st:
            print(f"[resume] {reason}: the state after step {st} is " + ("already on file" if st else "a fresh start")
                  + "; nothing to write", flush=True)
            return st > 0
        try:
            ok = writer.save_sync(resume_payload(st, reason, rs, cur, wc, pending), st)
        except Exception as e:  # noqa: BLE001 -- e.g. a dead CUDA context: report, keep the original error
            print(f"[resume] {reason}: resume save FAILED ({e!r}); the last resume state (step {writer.saved_step}) "
                  f"stands"[:400], flush=True)
            return False
        print(f"[resume] {reason}: state after step {st} " + ("saved" + (" and uploaded" if writer.s3_url else "") if ok
                                                              else "NOT fully saved (see above)"), flush=True)
        return ok

    old_handlers, watcher = {}, None
    if writer is not None or (dp.on and a.resume != "off"):   # (ddp) every rank listens; rank 0 alone writes
        old_handlers = install_signal_handlers(intr)
        if a.spot_watch == "auto" and not dp.rank:        # (ddp) one watcher per node: rank 0's
            def on_notice(kind, body):        # the watcher thread
                if kind == "rebalance":
                    intr.rebalance = body or kind
                else:
                    intr.stop("spot " + kind, 75, body)
                print(f"SPOT NOTICE {kind}: {body[:200]} (during step {cur_step + 1})", flush=True)
            watcher = SpotWatcher(on_notice)
            watcher.start()
    try:
        if post_pending:
            phase = "post"
            post(start)
        phase = "idle"
        if start >= a.steps and not post_pending:
            save(a.steps, with_resume=False)  # a finished run, re-run: re-emit --out from the final state
        for step in range(start + 1, a.steps + 1):
            cur_step = step
            if writer is not None:
                boundary = (step - 1, rng_state(rng), cursor, dict(wc_train))
            phase = "fb"
            step_ex = []
            step_sum, step_n = 0.0, 0         # [step-mean]: the loss over ALL the step's examples (logging only)
            aux_tr = None                     # (aux-vbll) this step's -bound sums per head + example count (logging only)
            for mb in range(a.grad_accum * dp.world):   # (ddp) every rank draws the GLOBAL step (world x grad_accum micro-
                check()                                 # batches, the one-GPU stream) and runs its own grad_accum of them
                ex = []
                while len(ex) < a.batch_size:
                    e = make_example(tok, train[cursor % len(train)], rng, a.max_len, not a.no_permute, a.plain_prompt,
                                     a.whole_case, a.noul_defs, wc_train, with_levels=ordinal, chat=chat)
                    cursor += 1
                    if e:
                        ex.append(e)
                if batched or regroup:        # collect the step's grad_accum * batch_size examples (the same draws, in the
                    step_ex.extend(ex)        # same order, as below) and run them together after this loop
                    continue
                if mb // a.grad_accum != dp.rank:      # (ddp) another rank's micro-batch
                    continue
                if a.exact_path:              # one prompt per forward/backward (memory), same loss as the batched mean
                    for e1 in ex:
                        check()
                        texts, tgt, ks, kmax = batch_tensors([e1], a.device, a.h2d)
                        gc_select(texts, False, gc_for(texts))
                        fwd_begin()
                        z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, full, a.exact_head, a.h2d,
                                        keep_h=caps)
                        loss, _ = example_loss(z, tgt, ks, [e1], a.ordinal_readout, a.rps_weight, a.brier_weight)
                        (loss / (a.grad_accum * dp.world * len(ex))).backward()   # (ddp) 1 / the GLOBAL step's count
                        step_sum = step_sum + loss.detach(); step_n += 1
                        if aux is not None:   # (aux-vbll) after the LoRA's backward, on detached inputs, the same weight
                            aux_train(z, tgt, ks, 1.0 / (a.grad_accum * dp.world * len(ex)))
                    continue
                texts, tgt, ks, kmax = batch_tensors(ex, a.device, a.h2d)
                gc_select(texts, False, gc_for(texts))
                z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, h2d=a.h2d, keep_h=caps)
                loss, _ = example_loss(z, tgt, ks, ex, a.ordinal_readout, a.rps_weight, a.brier_weight)
                (loss / (a.grad_accum * dp.world)).backward()                  # (ddp) len(ex) / the GLOBAL step's count
                step_sum = step_sum + loss.detach() * len(ex); step_n += len(ex)
                if aux is not None:           # (aux-vbll)
                    aux_train(z, tgt, ks, 1.0 / (a.grad_accum * dp.world))
            if batched:                       # sum_i loss_i / n, as the one-prompt-per-forward loop above
                for chunk in exact_chunks(tok, dp.share(tok, step_ex, a.dp_share), a.exact_tokens, a.exact_batch):   # (ddp)
                    check()
                    texts, tgt, ks, kmax = batch_tensors(chunk, a.device, a.h2d)
                    gc_select(texts, a.exact_batch == "pack")
                    fwd_begin()
                    z = exact_letter_logits(model, tok, texts, label_ids[:kmax], a.device, a.exact_batch, a.exact_head,
                                            pad_to=a.pack_pad_to, h2d=a.h2d, keep_h=caps)
                    loss, _ = example_loss(z, tgt, ks, chunk, a.ordinal_readout, a.rps_weight, a.brier_weight)
                    (loss * (len(chunk) / len(step_ex))).backward()
                    step_sum = step_sum + loss.detach() * len(chunk); step_n += len(chunk)
                    if aux is not None:       # (aux-vbll)
                        aux_train(z, tgt, ks, len(chunk) / len(step_ex))
            if regroup:                       # --microbatch regroup / budget: the same examples, length-grouped forwards
                for chunk in fast_chunks(tok, dp.share(tok, step_ex, a.dp_share), a.microbatch, a.batch_size, mb_budget):
                    check()
                    texts, tgt, ks, kmax = batch_tensors(chunk, a.device, a.h2d)
                    gc_select(texts, False, gc_for(texts))
                    z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, h2d=a.h2d, keep_h=caps)
                    loss, _ = example_loss(z, tgt, ks, chunk, a.ordinal_readout, a.rps_weight, a.brier_weight)
                    (loss * (len(chunk) / len(step_ex))).backward()
                    step_sum = step_sum + loss.detach() * len(chunk); step_n += len(chunk)
                    if aux is not None:       # (aux-vbll)
                        aux_train(z, tgt, ks, len(chunk) / len(step_ex))
            lazy = fold and step < a.steps    # (speed A) --dp-stop fold; the last step agrees as before
            if dp.on:                         # (ddp) one agreed stop check, then the GLOBAL step's gradient on every rank
                if fold:                      # (speed A) the stop codes / checksums of the LAST step, read now (no wait)
                    dp.resolve_checks()
                    if dp.take_stop(intr):
                        raise Preempted(intr.reason)
                if lazy:
                    dp.allreduce_grads_fold(params, intr)
                else:
                    check(sync=True)
                    dp.allreduce_grads(params)
                if aux is not None:           # (aux-vbll) its own buckets, after the LoRA's (theirs exactly as without it)
                    dp.reduce_aux(aux_params)
            phase = "opt"
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if aux is not None:               # (aux-vbll) after the LoRA's update; KL once per step (every rank: the same values)
                aux.kl_loss(aux_n).backward()
                aux.clip_(1.0)
                aux_opt.step(); aux_sched.step(); aux_opt.zero_grad(set_to_none=True)
            loss_hist.append(loss.detach())
            phase = "post"
            if not lazy:                      # (speed A) fold: this step's stop codes are read at the next step's agree point
                check(sync=True)
            if dp.on:                         # (aux-vbll) the head's tensors checksummed with the LoRA's (none when off)
                dp.verify_same(params + aux_params, step, last=step == a.steps, lazy=lazy)
            if step % 10 == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else 0.0
                if dp.on:                     # (ddp) the GLOBAL step's mean; the largest peak of any rank
                    step_sum, step_n, mem = dp.step_stats(step_sum, step_n, mem)
                print(f"  [step-mean] step {step} loss {float(step_sum) / max(step_n, 1):.5f} over {step_n} examples", flush=True)
                el = time.time() - t0
                print(f"step {step:4d}  loss {float(loss.detach()):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{el / (step - start):.2f}s/step  peak {mem:.1f} GiB"
                      + (f"  global batch {dp.world * a.batch_size * a.grad_accum} ({dp.world} ranks)" if dp.on else ""), flush=True)
                if aux is not None:           # (aux-vbll) the GLOBAL step's mean -bound per head, KL / n_train, tau
                    tr_ = dp.sum_(aux_tr) if dp.on else aux_tr
                    tr_, kl_, tau_ = tr_.tolist(), aux.kl().detach().tolist(), aux.log_tau.detach().exp().tolist()
                    print(f"  [aux-vbll] step {step}  -bound/ex " + " ".join(f"{x / max(tr_[-1], 1):.4f}" for x in tr_[:-1])
                          + "  KL/n_train " + " ".join(f"{x / max(aux_n, 1):.3e}" for x in kl_)
                          + "  tau " + " ".join(f"{x:.3f}" for x in tau_), flush=True)
            post(step)
            phase = "idle"
            if intr.rebalance is not None and not rebalanced and writer is not None:
                rebalanced = True             # a hint, not a deadline: save now (upload confirmed), keep training
                writer.drain()
                ok = writer.saved_step == step or \
                    writer.save_sync(resume_payload(step, "spot rebalance", rng_state(rng), cursor, dict(wc_train), False), step)
                print(f"SPOT REBALANCE hint: resume state after step {step} {'saved' if ok else 'NOT saved'}; training continues",
                      flush=True)
    except Preempted:
        ok = emergency(intr.reason)
        code = intr.exit_code
        if dp.on:                             # (ddp) no rank leaves before rank 0's state (and --out write) is on file:
            saver.close()                     # torchrun stops every rank once one exits
            dp.barrier()
        if intr.reason.startswith("spot"):
            print(f"PREEMPT: spot interruption notice {intr.detail[:200]}; resume state {'saved' if ok else 'NOT saved'}; waiting "
                  f"up to {a.preempt_wait:g}s for the instance to stop", flush=True)
            end = time.time() + a.preempt_wait
            while time.time() < end and intr.last_signal is None:
                time.sleep(0.2)
            if intr.last_signal is not None:
                code = 128 + intr.last_signal
        print(f"PREEMPT: {intr.reason} during step {cur_step}: stopping, exit {code} (a re-run resumes)", flush=True)
        saver.close()
        if writer is not None:                # (ddp) None on ranks > 0
            writer.close()
        return code
    except BaseException as e:
        if writer is not None:
            print(f"[resume] {type(e).__name__} during step {cur_step} ({phase}): saving the last consistent state, then "
                  f"re-raising", flush=True)
            emergency(f"exception {type(e).__name__}")
            if not isinstance(e, KeyboardInterrupt):
                try:
                    saver.close()             # the --out write in flight, before the process (or a test) moves on
                except Exception:  # noqa: BLE001 -- never mask the original error
                    pass
        raise
    finally:
        if watcher is not None:
            watcher.stop()
        restore_signal_handlers(old_handlers)
    saver.close()
    if writer is not None:
        writer.close()
    if dp.on:
        print(f"[ddp] the {dp.world} ranks' LoRA parameters were bit-identical at all {dp.checks} checks (every "
              f"{dp.check_every or 'never'} steps and the last)", flush=True)
    if a.tok_cache:
        print(f"tok-cache: {tok.hits} hits, {tok.misses} misses (LRU of {a.tok_cache})", flush=True)
    print(f"saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
