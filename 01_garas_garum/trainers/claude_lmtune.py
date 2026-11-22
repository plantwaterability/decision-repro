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
(speed, 2026-10-06; research/claude_bayes_handoff/SPEED_RESEARCH_IMPL.md) --stop-after-step N: steps 1..N of the --steps run (same
warmup / cosine horizon), --out + resume state after step N, exit 0; a re-run without it resumes the same run (resume-free).
--phase-log W [--phase-sync]: log-only PHASES lines (wall seconds per phase, real / padded training tokens; resume-free).
--letter-head selected: the letter logits from the 52 letter rows of the frozen head (ls.FrozenLabelHead; in the identity when set).
--pinned-staging: training forwards' inputs through two reusable pinned slots, one H2D copy each on a copy stream, staged one forward
ahead within a step, ordered by CUDA events (PinnedStager); the same draws in the same order, the same values (resume-free).

AUX VBLL (aux-vbll, 2026-10-04; see the "aux VBLL" section below): --aux-vbll also trains a variational Bayesian last layer (AuxVBLL)
on the DETACHED final hidden state at the decision position, jointly, on the same examples and soft targets, with its own optimizer:
answer + aleatoric + epistemic uncertainty from one forward. Nothing it does reaches the LoRA (same LoRA bits with or without it, under
data parallelism too). Saved in --out / the resume file as `aux_vbll`; scored by claude_lmscore.py --aux-vbll. Off = unchanged.

NUMERIC HEAD (p95, 2026-10-04; see the "numeric mixture head" section below, research/p95_parametric_head.md): --num-head mix:K trains a
small head on the decision-position final hidden state JOINTLY with the LoRA (its gradient reaches the LoRA) that outputs a K-component
logistic / Gaussian mixture on a numeric question's declared axis (build_numeric_jev.py records), trained by the cross-entropy of the
exact posterior's bin masses under the mixture's CDF differences over the same bins (+ optional closed-form binned CRPS). Only numeric
units train it; --num-head-letters off (default) swaps their letter loss for the head's. In --out / resume as `num_head`; scored by
claude_lmscore.py --num-head. Off = unchanged (outputs, logs, identity).

PRIOR OFFSETS (p101, 2026-10-04; see the "prior offsets" section below, research/p101_likelihood_ratio_head.md): --prior-offset meta[:KEY]
adds, for every unit whose record carries meta.log_prior_bins (or meta.KEY: one log prior mass per option, the question's stored
option order), log pi to the option-letter logits BEFORE the softmax / loss -- mapped through the example's option order -- so the
letters are trained to be the EVIDENCE l_i(D) and q = softmax(log pi + l). Everything else (losses, targets, draws) unchanged; units
without the key untouched. Off = unchanged (outputs, logs, identity).

LEARNED DATA MIXTURE (p103, 2026-10-04; see the "learned data mixture" section below, research/p103_learned_mixture.md): --mixture learned
splits train/ into groups by file name (--mixture-groups numeric=num_*,replay=*), reads each group as its own seeded stream without
replacement, draws every example slot's group from weights w, and every --mixture-every steps moves w by an exponentiated-gradient step on
the held-out excess loss of --mixture-val sets (DoReMi-style, online, one run): w_g <- w_g exp(eta s_g) / Z, smoothed, bounded. Streams,
draws, weights and history are resume state; every rank draws the same. --mixture static (the default) = unchanged (outputs, logs,
identity); with --mixture-val it only monitors the val sets.

HIERARCHICAL SAMPLING (datavar, 2026-10-04; see the "hierarchical sampling" section below, research/data_variants.md): --sampling hier
draws every example slot down [capability ->] source -> case -> question instead of reading the pooled shuffle: a source (one per
train file, or --sampling-sources groups of files) by --sampling-source-weights (uniform / cases / units / NAME=W), then that source's
next CASE (base_id or record_id; a fresh seeded permutation of its cases per pass), then one question of that case (without
replacement within the case). --sampling-capability-map FILE adds a capability level above the source. The draw RNG and one cursor
per source are the whole resume state; every rank draws the same. --sampling pooled (the default) = unchanged (outputs, logs,
identity).

(ss) LORA SCALE RULE (2026-10-08, review AM P2): --lora-scale-rule rs [--lora-scale-anchor 8] sets every adapter's multiplier to
anchor / sqrt(r) (rank-stabilised LoRA, arXiv 2312.03732; anchor 8 = the default's r16 value) instead of 32 / r. Saved in --out's config
(lora_scale_rule / lora_scale_anchor / lora_scale / lora_alpha), applied by every loader (train_e2e.lora_scale_from_config), part of
the resume identity when set; --init-lora refuses another convention. Off (the default) = unchanged (outputs, logs, identity).
(ss) DYNAMICS LOG (2026-10-07; see the "dynamics log" section below): --dyn-log PATH writes one JSON line per TRAINING example
forwarded -- step, epoch, draw index, the unit's rowid uid and source, its option count, the gold option (stored order), the gold
option's and the target's probability under the loss's own readout, the example's own loss term, correct -- the per-example training
dynamics of a data map (Swayamdipta et al. 2020: confidence / variability / correctness per unit; analysis/data_map.py). LOG ONLY:
the readout the loss already computed, detached, one device->host copy per step; flushed every step; a resumed run first drops the
records past its resumed step; under data parallelism one file per rank (PATH.rank<r>). Resume-free. Off (the default) = unchanged
(outputs, logs, identity).
(grok1, 2026-10-09; AGENT_NOTES 04:35Z registration) OPTIMISER OPTIONS + EVAL LOG, all opt-in. --weight-decay W: decoupled AdamW
weight decay W on the LoRA parameters (the only parameters this optimizer holds; an --aux-vbll / --num-head head keeps its own
optimizer at 0). --lr-schedule constant: the linear --warmup, then the peak --lr flat to --steps (no cosine). Both are saved in --out's
config when set and are part of the resume identity when set. --eval-log PATH (log only, resume-free, one process): at every val eval
one JSON line -- the val soft-CE / acc / Brier (soft target) / KL = CE - H(target), the per-unit CE / Brier / correct / p (unit order
in a header line, by rowid uid), the LoRA norms (sum ||A||^2, sum ||B||^2, sum ||scale B A||_F^2) and the LR; a resumed run first drops
the lines past its resumed step. Defaults (0.0 / cosine / off) = unchanged (outputs, logs, identity).
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
                 with_levels=False, chat=None, order_out=None):
    """(prompt text, target in the prompt's option order) or None if it cannot fit.
    `order_out` (p101): a list that receives the option order (letter j shows option order[j]); no draws, nothing else changes.
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
    if order_out is not None:
        order_out[:] = order
    corder = None
    if whole_case and len(unit.get("case_questions") or ()) > 1:
        if permute:
            corder = list(range(len(unit["case_questions"])))
            rng.shuffle(corder)
        else:
            corder = ls.fixed_case_order(unit)
    build_prompt = getattr(tok, "pretok_build_prompt", None) or ls.build_prompt   # (pretok) --token-cache: same prompt, cached ids
    text = build_prompt(tok, unit, order, max_len, plain=plain, whole_case=whole_case, noul_defs=noul_defs,
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


# --------------------------------------------------------------------------- (speed, 2026-10-06) --pinned-staging
# research/claude_bayes_handoff/KERNEL_DATA_PATH_RESEARCH.md "Pinning and ping-pong buffers": --h2d async pins inline and copies on the
# compute stream, so a forward's input copy queues behind the previous forward's compute. --pinned-staging packs each TRAINING forward's
# host inputs (token ids, mask or positions, targets, letter counts, ordinal levels) into one of TWO reusable host slots (page-locked on
# CUDA) and copies them in ONE transfer into that slot's device buffer on a dedicated copy stream, one forward ahead of the compute.
# Ordering is by explicit CUDA events only (no global synchronisation):
#   stage(n)   host: wait for the host slot's previous H2D (its event) before overwriting it, fill it; copy stream: wait for the device
#              slot's previous compute-use event, copy, record `ready`;
#   acquire(n) compute stream waits for `ready` (every consumer of the views runs after the copy);
#   release(n) after the forward's LAST consumer (its backward, activation-checkpoint recomputation included, and the aux head):
#              record the device slot's compute-use event on the compute stream.
# A device buffer is never freed while a copy or a consumer may still use it: a slot only grows (capacity buckets: powers of two), at a
# point where every use of the old buffer is already queued on the compute stream (the stream it was allocated on), and a fresh buffer
# is first written only after the copy stream waited for the compute stream (the caching allocator may hand out memory whose last
# compute-stream use is still queued). Every staged tensor starts on a STAGE_ALIGN-byte boundary, as a fresh allocation does.
STAGE_ALIGN = 512


def stage_layout(tensors):
    """(speed) [(name, byte offset, bytes, dtype, shape)] of a dict of CPU tensors packed into ONE byte buffer, each at a STAGE_ALIGN-
    aligned offset; and the bytes used."""
    out, off = [], 0
    for name, t in tensors.items():
        nb = t.numel() * t.element_size()
        out.append((name, off, nb, t.dtype, tuple(t.shape)))
        off += -(-nb // STAGE_ALIGN) * STAGE_ALIGN
    return out, off


class CudaStaging:
    """(speed) --pinned-staging backend on CUDA: page-locked host buffers, device buffers allocated on the compute (current) stream, ONE
    dedicated copy stream, CUDA events."""
    sync = False

    def __init__(self, device):
        d = torch.device(device)
        self.device = d if d.index is not None else torch.device("cuda", torch.cuda.current_device())
        self.copy = torch.cuda.Stream(device=self.device)

    @property
    def compute(self):                         # the stream the forwards / backwards run on
        return torch.cuda.current_stream(self.device)

    def host_buffer(self, n, dtype=torch.uint8):
        return torch.empty(n, dtype=dtype, pin_memory=True)

    def device_buffer(self, n, dtype=torch.uint8):
        return torch.empty(n, dtype=dtype, device=self.device)

    def event(self):
        return torch.cuda.Event()

    def record(self, ev, stream):
        ev.record(stream)

    def wait(self, stream, ev):
        stream.wait_event(ev)

    def query(self, ev):
        return ev.query()

    def host_wait(self, ev):
        ev.synchronize()

    def fill(self, dst, src):                   # a host-slot write (the host's own memcpy)
        dst.copy_(src)

    def copy_(self, stream, dst, src):
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)


class SyncStaging:
    """(speed) --pinned-staging without CUDA (e.g. --device cpu, the tests): the same slot code with ordinary buffers and a synchronous
    copy; no streams, so every 'event' is complete when recorded. The values and the example stream are those of the default loop."""
    sync = True
    compute = copy = None

    def __init__(self, device):
        self.device = torch.device(device)

    def host_buffer(self, n, dtype=torch.uint8):
        return torch.empty(n, dtype=dtype)

    def device_buffer(self, n, dtype=torch.uint8):
        return torch.empty(n, dtype=dtype, device=self.device)

    def event(self):
        return None

    def record(self, ev, stream):
        pass

    def wait(self, stream, ev):
        pass

    def query(self, ev):
        return True

    def host_wait(self, ev):
        pass

    def fill(self, dst, src):
        dst.copy_(src)

    def copy_(self, stream, dst, src):
        dst.copy_(src)


def staging_backend(device):
    """(speed) --pinned-staging's backend for `device` (the tests replace this with a recording fake)."""
    return CudaStaging(device) if torch.device(device).type == "cuda" else SyncStaging(device)


class Staged:
    """(speed) One staged forward: its slot, its fill number, the device views {name: tensor} and the caller's `info`."""
    __slots__ = ("slot", "fill", "views", "info")

    def __init__(self, slot, fill, views, info):
        self.slot, self.fill, self.views, self.info = slot, fill, views, info


class PinnedStager:
    """(speed, 2026-10-06) --pinned-staging: two (`nslots`) capacity-bucketed reusable host slots + device slots, ONE copy per forward on
    the backend's copy stream, ordered by events (see the section comment above). Slot state: free -> staged (stage) -> acquired
    (acquire) -> free (release); staging into a slot whose previous fill is acquired but not released is refused (it would overwrite
    inputs a consumer may still read)."""

    def __init__(self, backend, nslots=2, min_bytes=1 << 16):
        self.be, self.min_bytes = backend, int(min_bytes)
        self.slots = [{"host": None, "dev": None, "cap": 0, "copied": None, "used": None, "state": "free", "fill": -1}
                      for _ in range(int(nslots))]
        self.fills, self.grows, self.host_blocked, self.peak = 0, 0, 0, 0

    def stage(self, tensors, info=None):
        """Host inputs {name: contiguous CPU tensor} -> Staged (device views, valid after acquire until release)."""
        be = self.be
        k = self.fills % len(self.slots)
        s = self.slots[k]
        if s["state"] == "acquired":
            raise RuntimeError(f"pinned staging: slot {k} (fill {s['fill']}) is acquired and not released; refusing to overwrite it")
        layout, need = stage_layout(tensors)
        if s["copied"] is not None:               # never overwrite a host slot before its previous H2D completed
            if not be.query(s["copied"]):
                self.host_blocked += 1
            be.host_wait(s["copied"])
        if need > s["cap"]:                        # grow to the next power of two: the old buffers' uses are all queued already
            cap = max(self.min_bytes, 1 << (max(need, 1) - 1).bit_length())
            s["host"], s["dev"], s["cap"] = be.host_buffer(cap), be.device_buffer(cap), cap
            ev = be.event()                        # a fresh device block: the copy stream first waits for the compute stream
            be.record(ev, be.compute)
            s["used"] = ev
            self.grows += 1
        h, d = s["host"], s["dev"]
        views = {}
        for name, off, nb, dtype, shape in layout:
            if nb:
                be.fill(h[off: off + nb].view(dtype).view(shape), tensors[name])
            views[name] = d[off: off + nb].view(dtype).view(shape)
        if s["used"] is not None:                  # the device slot's previous compute use is over
            be.wait(be.copy, s["used"])
        be.copy_(be.copy, d[:need], h[:need])
        ev = be.event()
        be.record(ev, be.copy)                     # 'ready' (and the host slot's copy-done)
        s.update(copied=ev, state="staged", fill=self.fills)
        self.peak = max(self.peak, need)
        st = Staged(k, self.fills, views, info)
        self.fills += 1
        return st

    def acquire(self, st):
        """Compute stream waits for the copy; -> the device views."""
        s = self.slots[st.slot]
        if s["fill"] != st.fill or s["state"] != "staged":
            raise RuntimeError(f"pinned staging: acquire of fill {st.fill} but slot {st.slot} holds fill {s['fill']} ({s['state']})")
        self.be.wait(self.be.compute, s["copied"])
        s["state"] = "acquired"
        return st.views

    def release(self, st):
        """After the last consumer of the views (backward incl. recomputation): the device slot's compute-use event."""
        s = self.slots[st.slot]
        if s["fill"] != st.fill or s["state"] != "acquired":
            raise RuntimeError(f"pinned staging: release of fill {st.fill} but slot {st.slot} holds fill {s['fill']} ({s['state']})")
        ev = self.be.event()
        self.be.record(ev, self.be.compute)
        s.update(used=ev, state="free")

    def summary(self):
        return (f"pinned-staging: {self.fills} training forwards staged through {len(self.slots)} slots (largest {self.peak} bytes, "
                f"slot buffers grown {self.grows} times); the host waited for a slot's previous copy {self.host_blocked} times")


PHASE_ORDER = ("load", "prep", "resume", "initial_eval", "draw", "stage", "fwd", "bwd", "comm", "opt", "log", "mix_eval", "eval", "save",
               "misc", "gpu_wait", "final_drain")


class PhaseClock:
    """(speed, 2026-10-06) --phase-log W: LOG-ONLY wall clocks per phase and real / padded TRAINING tokens; a PHASES line every W steps,
    a startup line and totals at the end. lap(name) books the host time since the previous lap to `name` (time.perf_counter: no
    device read, no sync, nothing computed differently). On a GPU the host runs ahead, so a host clock books GPU work to whichever
    phase the host first BLOCKS in (a log line's loss read, an eval's .cpu(), a checkpoint snapshot). --phase-sync (CUDA only) adds
    torch.cuda.synchronize() at COARSE boundaries only -- window ends and around the startup / eval / checkpoint phases, never inside a
    step's forwards, backwards or optimizer update -- booking the wait as 'gpu_wait', and records a CUDA event at every lap; at those
    boundaries the events become GPU-timeline seconds per phase (the 'gpu' part; idle GPU time included). every = 0 (the default):
    each method returns at once."""

    def __init__(self, every, sync=False, event=None):
        self.every = int(every)
        self.on = self.every > 0
        self.sync = self.on and bool(sync)
        self.event = event or (lambda: torch.cuda.Event(enable_timing=True))
        self.t = time.perf_counter()
        self.win, self.tot, self.gwin, self.gtot = (collections.Counter() for _ in range(4))
        self.toks, self.tok_tot = [], [0, 0, 0, 0]          # this window's (real, padded, rows) per forward; totals + forwards
        self.ev, self.ev0, self.w0, self.t_w0 = [], None, 0, self.t

    def lap(self, name, sync=False):
        if not self.on:
            return
        now = time.perf_counter()
        self.win[name] += now - self.t
        self.tot[name] += now - self.t
        self.t = now
        if self.sync:
            e = self.event()
            e.record()
            self.ev.append((name, e))
            if sync:
                self.drain()

    def drain(self):
        """(--phase-sync) wait for the GPU ('gpu_wait'), then turn the events recorded since the last drain into GPU seconds."""
        torch.cuda.synchronize()
        self.lap_wait()
        prev = self.ev0
        for name, e in self.ev:
            if prev is not None:
                g = prev.elapsed_time(e) / 1000.0
                self.gwin[name] += g
                self.gtot[name] += g
            prev = e
        self.ev0, self.ev = prev, []

    def lap_wait(self):
        now = time.perf_counter()
        self.win["gpu_wait"] += now - self.t
        self.tot["gpu_wait"] += now - self.t
        self.t = now

    @staticmethod
    def _fmt(c):
        return " ".join(f"{k} {c[k]:.3f}" for k in PHASE_ORDER if k in c) + \
            "".join(f" {k} {v:.3f}" for k, v in sorted(c.items()) if k not in PHASE_ORDER)

    def _tok_line(self, real, padded, rows, fwd, wall):
        return (f"train tokens real {real} padded {padded} ({real / max(padded, 1):.1%} real) in {fwd} forwards, {rows} examples; "
                f"{real / max(wall, 1e-9):.0f} real tok/s")

    def startup(self, step):
        """After the initial eval: the startup phases (load, prep, resume, initial_eval) on one line; the first window starts here."""
        if not self.on:
            return
        if self.sync:
            self.drain()
        print(f"PHASES startup (s): {self._fmt(self.win)}" + (f" | gpu {self._fmt(self.gwin)}" if self.gwin else ""), flush=True)
        self.win.clear(); self.gwin.clear()
        self.w0, self.t_w0 = step, time.perf_counter()

    def window(self, step, last=False):
        """After a step's post(): the time since the last lap is 'misc'; every `every` steps (and at `last`) one PHASES line."""
        if not self.on:
            return
        self.lap("misc")
        if step % self.every and not last:
            return
        if self.sync:
            self.drain()
        wall, n = time.perf_counter() - self.t_w0, step - self.w0
        real, padded, rows = (sum(x[i] for x in self.toks) for i in range(3))
        for i, x in enumerate((real, padded, rows, len(self.toks))):
            self.tok_tot[i] += x
        print(f"PHASES steps {self.w0 + 1}-{step} ({n} steps, {wall:.2f}s, {wall / max(n, 1):.3f} s/step; s): {self._fmt(self.win)}"
              + (f" | gpu {self._fmt(self.gwin)}" if self.gwin else "") + " | " + self._tok_line(real, padded, rows, len(self.toks),
                                                                                                 wall), flush=True)
        self.win.clear(); self.gwin.clear(); self.toks.clear()
        self.w0, self.t_w0 = step, time.perf_counter()

    def final(self):
        """At the very end (after the checkpoint / resume writers drained): every phase's total, and the token totals."""
        if not self.on:
            return
        self.lap("final_drain")
        train = sum(v for k, v in self.tot.items() if k not in ("load", "prep", "resume", "initial_eval", "final_drain"))
        print(f"PHASES total (s): {self._fmt(self.tot)}" + (f" | gpu {self._fmt(self.gtot)}" if self.gtot else "")
              + " | " + self._tok_line(*self.tok_tot[:3], self.tok_tot[3], train), flush=True)


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


@contextlib.contextmanager
def head_input_grad(model, keep, pick):
    """(p95) head_input WITHOUT the detach: the kept decision states stay in the graph, so a loss on them (the numeric head's)
    reaches the LoRA. `keep` None (every path without --num-head): nothing at all."""
    if keep is None:
        yield
        return
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda mod, args: keep.append(pick(args[0])))
    try:
        yield
    finally:
        hook.remove()


def letter_logp(stack, lm_head, tok, texts, label_ids, device, model=None, head="model", h2d="sync", keep_h=None, keep_g=None,
                sel=None, tok_stats=None, staged=None):
    """log p over each prompt's option letters, read at the last position: [B, Kmax] (-inf pad).
    With `model` (the --exact-path): the full model forward, last position only, one prompt at a time (no padding).
    `head` (gemmafast, exact path only): 'model' = the model's own head (bf16, the default, unchanged); 'fp32' = the same
    forward up to the final hidden state, then letter_head_fp32. `h2d` (speed): as to_dev. `keep_h` (aux-vbll): a list that
    receives each row's decision hidden state (detached; [1, H] per prompt on the exact path, [B, H] on the fast path); None (the
    default) keeps nothing and computes exactly as before. `keep_g` (p95): the same states NOT detached (the numeric head).
    `sel` (speed, --letter-head selected; fast path): an ls.FrozenLabelHead over the FULL letter list, whose first len(label_ids)
    columns are the letters asked for (every caller passes label_ids[:kmax]). `tok_stats` (speed, --phase-log): a list that receives
    (real tokens, padded tokens, rows) of each forward, counted from the host-side encoding (no device read). `staged` (speed,
    --pinned-staging): the tokeniser's encoding ALREADY ON THE DEVICE (PinnedStager views of the same tensors; the caller counts
    tok_stats): the fast path's padded batch {input_ids, attention_mask, ...}, or on the exact path a list of one such dict per prompt.
    None (every default) tokenises and copies here, exactly as before."""
    nb = h2d == "async"
    if model is not None:
        zs = []
        for i, t in enumerate(texts):
            if staged is not None:
                e = staged[i]
            else:
                e = tok(t, return_tensors="pt", add_special_tokens=False)
                if tok_stats is not None:
                    tok_stats.append((int(e["input_ids"].numel()),) * 2 + (1,))
                e = {k: host_to_dev(x, device) for k, x in e.items()} if nb else e.to(device)
            if head == "fp32":
                hl = inner_model(model)(**e, use_cache=False).last_hidden_state[:, -1]
                if keep_h is not None:
                    keep_h.append(hl.detach())
                if keep_g is not None:
                    keep_g.append(hl)
                zs.append(letter_head_fp32(model, hl, label_ids))
                continue
            with head_input(model, keep_h, lambda x: x[:, -1]):
                with head_input_grad(model, keep_g, lambda x: x[:, -1]):
                    zs.append(model(**e, use_cache=False, logits_to_keep=1).logits[:, -1].float())
        if head == "fp32":
            return torch.cat(zs, 0)
        return torch.cat(zs, 0)[:, label_ids]
    if staged is not None:
        enc = staged
    else:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        if tok_stats is not None:
            tok_stats.append((int(enc["attention_mask"].sum()), int(enc["attention_mask"].numel()), len(texts)))
        enc = {k: host_to_dev(x, device) for k, x in enc.items()} if nb else enc.to(device)
    h = stack(**enc, use_cache=False).last_hidden_state
    last = enc["attention_mask"].sum(1) - 1
    hl = h[torch.arange(len(texts), device=device), last]
    if keep_h is not None:
        keep_h.append(hl.detach())
    if keep_g is not None:
        keep_g.append(hl)
    if sel is not None:                                     # (speed) the letter rows only; multiplier / softcap inside
        return sel(hl)[:, : len(label_ids)]
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


def exact_host_inputs(tok, texts, mode, pad_to=0):
    """(speed, --pinned-staging) exact_letter_logits' inputs built on the HOST, with the values, dtypes and shapes it builds itself:
    'pack' -> {inp [1, N], pos [1, N], last [B]}, 'pad' -> {inp [B, T], att [B, T], last [B]}; and the forward's tok_stats tuple."""
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    lens = [len(x) for x in ids]
    stats = (sum(lens), (sum(lens) + (-sum(lens) % pad_to if pad_to else 0)) if mode == "pack" else max(lens) * len(lens), len(lens))
    if mode == "pack":
        flat, pos = [i for x in ids for i in x], [p for n in lens for p in range(n)]
        extra = -len(flat) % pad_to if pad_to else 0
        if extra:
            flat += [tok.pad_token_id if tok.pad_token_id is not None else flat[-1]] * extra
            pos += list(range(extra))
        return {"inp": torch.tensor([flat]), "pos": torch.tensor([pos]), "last": torch.tensor(np.cumsum(lens) - 1)}, stats
    if mode != "pad":
        raise ValueError(f"exact batch mode {mode!r}")
    if tok.pad_token_id is None:
        raise SystemExit("--exact-batch pad needs a pad token")
    inp = torch.full((len(ids), max(lens)), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), max(lens)), dtype=torch.long)
    for b, x in enumerate(ids):                             # RIGHT padding, as exact_letter_logits
        inp[b, : len(x)] = torch.tensor(x)
        att[b, : len(x)] = 1
    return {"inp": inp, "att": att, "last": torch.tensor([n - 1 for n in lens])}, stats


def exact_letter_logits(model, tok, texts, label_ids, device, mode, head="model", pad_to=0, h2d="sync", keep_h=None, keep_g=None,
                        tok_stats=None, staged=None):
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
    `h2d` (speed): as to_dev. `keep_h` (aux-vbll): as letter_logp ([B, H], the rows in `texts` order); `keep_g` (p95): the same,
    not detached. `tok_stats` (speed, --phase-log): as letter_logp (pack: the row's tokens, --pack-pad-to fill included). `staged`
    (speed, --pinned-staging): exact_host_inputs' tensors ALREADY ON THE DEVICE (the caller counts tok_stats); None (every default)
    builds and copies them here, exactly as before."""
    if staged is None:
        ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        lens = [len(x) for x in ids]
        if tok_stats is not None:
            tok_stats.append((sum(lens), (sum(lens) + (-sum(lens) % pad_to if pad_to else 0)) if mode == "pack" else max(lens) * len(lens),
                              len(lens)))
    if mode == "pack":
        if staged is not None:
            inp, pos, last = staged["inp"], staged["pos"], staged["last"]
        elif pad_to or h2d == "async":
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
            if keep_g is not None:
                keep_g.append(h)
            return letter_head_fp32(model, h, label_ids)
        with head_input(model, keep_h, lambda x: x[0]):                # (aux-vbll) the head's input [1, B, H] -> [B, H]
            with head_input_grad(model, keep_g, lambda x: x[0]):        # (p95) the same, in the graph
                z = model(input_ids=inp, position_ids=pos, use_cache=False, logits_to_keep=last).logits[0]      # [B, V]
        return z.float()[:, label_ids]
    if mode != "pad":
        raise ValueError(f"exact batch mode {mode!r}")
    if staged is not None:
        inp, att, last = staged["inp"], staged["att"], staged["last"]
    else:
        if tok.pad_token_id is None:
            raise SystemExit("--exact-batch pad needs a pad token")
        inp = torch.full((len(ids), max(lens)), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(ids), max(lens)), dtype=torch.long)
        for b, x in enumerate(ids):                         # RIGHT padding: real tokens keep positions 0..n-1
            inp[b, : len(x)] = torch.tensor(x)
            att[b, : len(x)] = 1
        last = to_dev([n - 1 for n in lens], device, h2d)
    rows = torch.arange(len(texts), device=device)
    if h2d == "async" and staged is None:                   # (speed) the same tensors, copied without a host sync
        inp, att = host_to_dev(inp, device), host_to_dev(att, device)
    if head == "fp32":
        h = inner_model(model)(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False).last_hidden_state
        hl = h[rows, last]
        if keep_h is not None:
            keep_h.append(hl.detach())
        if keep_g is not None:
            keep_g.append(hl)
        return letter_head_fp32(model, hl, label_ids)
    with head_input(model, keep_h, lambda x: x[rows, rows]):           # (aux-vbll) [B, B, H] at every row's last -> [B, H]
        with head_input_grad(model, keep_g, lambda x: x[rows, rows]):  # (p95) the same, in the graph
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


def example_loss(z, tgt, ks, examples, readout="none", rps_weight=0.0, brier_weight=0.0, levels=None):
    """(49) soft_ce read through the ordinal readout on `score` rows, plus rps_weight * RPS (level order) on them; (50) plus
    brier_weight * Brier on every row. All per example, so the batched, one-prompt and --exact-batch paths still compute the
    same step. Defaults: soft_ce itself. (p101) Rows whose example carries prior offsets (PriorExample) are read as
    z + log pi first (with_prior_offsets): every path and evaluate() go through here. `levels` (speed, --pinned-staging): the
    ordinal batch_levels(examples) already on the device (staged); None (every default) builds and copies them here."""
    z = with_prior_offsets(z, examples)
    if readout == "none" and not rps_weight:
        loss, logp = soft_ce(z, tgt, ks)
    else:
        od = ls.ordinal()
        lev, sc = levels if levels is not None else od.batch_levels(examples, z.shape[1], z.device)
        loss, logp = soft_ce(od.readout_logits(z, ks, lev, sc, readout), tgt, ks)
        if rps_weight:
            loss = loss + rps_weight * od.rps_rows(logp, tgt, lev, sc).mean()
    if brier_weight:
        loss = loss + brier_weight * brier_rows(logp, tgt, ks).mean()
    return loss, logp


def example_loss_rows(z, tgt, ks, examples, rps_weight=0.0, brier_weight=0.0):
    """(p95) example_loss (no ordinal readout) per row: [B], whose mean is example_loss's value in exact arithmetic. The numeric
    head with --num-head-letters off drops its rows' letter terms from a forward's loss."""
    mask = torch.arange(z.shape[1], device=z.device)[None, :] < ks[:, None]
    logp = torch.log_softmax(z.masked_fill(~mask, -1e9), -1)
    rows = -(tgt * logp.masked_fill(~mask, 0.0)).sum(-1)
    if rps_weight:
        od = ls.ordinal()
        lev, sc = od.batch_levels(examples, z.shape[1], z.device)
        rows = rows + rps_weight * od.rps_rows(logp, tgt, lev, sc)
    if brier_weight:
        rows = rows + brier_weight * brier_rows(logp, tgt, ks)
    return rows


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


# --------------------------------------------------------------------------- numeric mixture head (p95, 2026-10-04)
# --num-head mix:K: a small head on the decision-position final hidden state h (the input of the model's own head at the position the
# letter logits are read from) that outputs a K-component mixture (logistic or Gaussian components) for a NUMERIC question: a
# `score` question whose levels are equal-width bins on a declared axis (build_numeric_jev.py: meta.bin_edges, meta.axis). Unlike
# --aux-vbll it trains JOINTLY: its loss is part of the training forward's loss, so its gradient reaches the LoRA through h. Only
# numeric units train it. A forward without numeric rows computes exactly what it computed before; with --num-head-letters off (the
# default) a numeric row's letter loss is replaced by the head's (same per-example weight), with 'on' the head's is added to it.
# COORDINATES. The mixture lives in the axis's own coordinate u (display units x, axis [lo, hi], w = hi - lo), chosen by which ends of
# the axis are folded (the question states it): both folded -> 'std' u = 4 (x - lo) / w - 2 (the axis is [-2, 2]); neither (a bounded
# quantity on its whole range: coin, A/B difference) -> 'logit' u = logit((x - lo) / w); low end at the support bound and high end
# folded (a rate on [0, hi]) -> 'log' u = log(2 (x - lo) / w); the mirror -> 'neglog' u = -log(2 (hi - x) / w). In every case the first
# and last bins reach u = -inf / +inf (by folding or by the transform), so the K-bin masses are differences of the mixture CDF at the
# K - 1 interior edges plus the two tails, and sum to 1 by construction. A learned per-coordinate output bias (4 rows) tells the head
# which coordinate it is writing in; the question's own text says the same thing.
# LOSS per numeric example: the cross-entropy of the exact posterior's bin masses (the label) under the head's bin masses -- a
# discretised NLL on the same bins the letters are scored on, so directly comparable with (94) -- plus, with --num-head-crps W,
# W x sum_k (F(e_k) - G_k)^2 over the interior edges: the closed-form mixture CDF against the exact posterior's binned CDF, = the
# binned CRPS minus the exact posterior's own (bin units; analysis/eval_numeric.py's `crps` minus sum G (1 - G)), which is (94)'s
# RPS term in level order. The mixture math runs in float64 (stable log-masses in both tails).
# OPTIMISATION: the head has its own AdamW (--num-head-lr, the trainer's warmup + cosine shape) and its own clip at 1.0, stepped after
# the LoRA's; the LoRA's clip / AdamW see only the LoRA's gradients (which include the head loss's share). Its output layer starts at
# zero (every input: the same broad mixture, ~ uniform bins), so step 1's LoRA gradient is the letters' alone. Its init draws come
# from its own generator.
# Under data parallelism its gradients are summed in buckets of their own after the LoRA's.
NUM_HEAD_FAMILIES = ("logistic", "gaussian")
NUM_TRANSFORMS = ("std", "logit", "log", "neglog")
NUM_HEAD_DEFAULTS = {"num_head_family": "logistic", "num_head_letters": "off", "num_head_crps": 0.0, "num_head_lr": 1e-3,
                     "num_head_dim": 256}
NUM_S_MIN = 1e-3                  # component scale floor, u units (a bin is ~0.08 u wide at the middle of every axis)
NUM_S_RAW_MAX = 30.0              # softplus input cap: scale <= ~30 u, far wider than any axis


def num_head_k(spec):
    """--num-head: 'off' -> 0, 'mix' -> 4, 'mix:K' -> K (1..16). ValueError otherwise."""
    s = str(spec).strip().lower()
    if s == "off":
        return 0
    if s == "mix":
        return 4
    m = re.fullmatch(r"mix:(\d+)", s)
    if not m or not 1 <= int(m.group(1)) <= 16:
        raise ValueError(f"--num-head: 'off', 'mix' or 'mix:K' with 1 <= K <= 16, got {spec!r}")
    return int(m.group(1))


def axis_transform(fold_low, fold_high):
    """The coordinate of an axis from which of its ends are folded (see the section comment)."""
    return {(True, True): "std", (False, False): "logit", (False, True): "log", (True, False): "neglog"}[
        (bool(fold_low), bool(fold_high))]


def to_u(x, lo, hi, tf):
    """Display value(s) x on the axis [lo, hi] -> the coordinate u of transform `tf` (numpy float64)."""
    x, lo, hi = np.asarray(x, dtype=np.float64), float(lo), float(hi)
    w = hi - lo
    if tf == "std":
        return 4.0 * (x - lo) / w - 2.0
    if tf == "logit":
        p = (x - lo) / w
        return np.log(p) - np.log1p(-p)
    if tf == "log":
        return np.log(2.0 * (x - lo) / w)
    if tf == "neglog":
        return -np.log(2.0 * (hi - x) / w)
    raise ValueError(f"transform {tf!r}")


def from_u(u, lo, hi, tf):
    """The inverse of to_u (numpy float64, broadcasting lo / hi)."""
    u, lo, hi = np.asarray(u, dtype=np.float64), np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    w = hi - lo
    if tf == "std":
        return lo + w * (u + 2.0) / 4.0
    if tf == "logit":
        return lo + w * np.where(u >= 0, 1.0 / (1.0 + np.exp(-np.abs(u))), np.exp(-np.abs(u)) / (1.0 + np.exp(-np.abs(u))))
    if tf == "log":
        return lo + 0.5 * w * np.exp(u)
    if tf == "neglog":
        return hi - 0.5 * w * np.exp(-u)
    raise ValueError(f"transform {tf!r}")


def num_spec(edges, axis, family=None):
    """A numeric question's axis -> {K, lo, hi, tf, tfi, u (the K - 1 interior edges in u, float64), family}. ValueError if the axis
    is not usable (fewer than 2 bins, edges not strictly increasing, an interior edge outside the transform's domain)."""
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or len(e) < 3 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
        raise ValueError(f"bin edges must be >= 3 strictly increasing finite numbers, got {list(edges)[:6]}...")
    tf = axis_transform(axis.get("fold_low"), axis.get("fold_high"))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = to_u(e[1:-1], e[0], e[-1], tf)
    if not np.all(np.isfinite(u)) or not np.all(np.diff(u) > 0):
        raise ValueError(f"interior edges not finite / increasing in the {tf} coordinate")
    return {"K": len(e) - 1, "lo": float(e[0]), "hi": float(e[-1]), "tf": tf, "tfi": NUM_TRANSFORMS.index(tf), "u": u,
            "family": family}


def numeric_index(data_dir, splits):
    """(p95) {(record id, qid): num_spec} for every record of `splits` (files as cache_features.iter_units reads them) whose meta
    declares a numeric axis (meta.bin_edges, meta.axis with fold_low / fold_high) and whose `score` question has len(bin_edges) - 1
    levels. Lines without "bin_edges" are skipped unparsed."""
    from pathlib import Path
    out = {}
    for split in splits:
        for p in sorted(Path(data_dir, split).glob("*.jsonl")):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if '"bin_edges"' not in line:
                        continue
                    rec = json.loads(line)
                    m = rec.get("meta") or {}
                    ax, ed = m.get("axis"), m.get("bin_edges")
                    if not isinstance(ax, dict) or not isinstance(ed, list):
                        continue
                    for qid, q in rec["questions"].items():
                        if q.get("type") == "score" and len(q.get("levels") or ()) == len(ed) - 1:
                            out[(rec["id"], qid)] = num_spec(ed, ax, m.get("family"))
    return out


def num_index_fingerprint(index, keys):
    """sha256 over the numeric specs of `keys` (in order): the axes a run trained on, for its identity."""
    h = hashlib.sha256()
    for k in keys:
        s = index.get(k)
        if s is not None:
            h.update(json.dumps([list(k), s["K"], s["tf"], s["lo"], s["hi"]]).encode())
            h.update(np.ascontiguousarray(s["u"]).tobytes())
    return h.hexdigest()[:16]


class NumExample(tuple):
    """(p95) make_example's (text, target) -- unpacked and indexed as before, the ordinal .levels kept -- plus .num: the unit's
    num_spec with its level-order target `t`, or None for a non-numeric unit."""

    def __new__(cls, e, num):
        x = super().__new__(cls, tuple(e))
        x.levels = getattr(e, "levels", None)
        x.num = num
        return x


def num_batch(nums, device):
    """Specs of B numeric rows -> (tf [B] long, u [B, Kmax-1] float64, kb [B] long, t [B, Kmax] float64 or None); u / t padded with
    zeros (num_log_masses replaces padded edges; t is 0 there)."""
    E = max(n["K"] for n in nums) - 1
    u = np.zeros((len(nums), E))
    t = np.zeros((len(nums), E + 1)) if all("t" in n for n in nums) else None
    for b, n in enumerate(nums):
        u[b, : n["K"] - 1] = n["u"]
        if t is not None:
            t[b, : n["K"]] = n["t"]
    return (torch.tensor([n["tfi"] for n in nums], dtype=torch.long, device=device), torch.tensor(u, device=device),
            torch.tensor([n["K"] for n in nums], dtype=torch.long, device=device),
            None if t is None else torch.tensor(t, device=device))


def _log1mexp(d):
    """log(1 - exp(-d)) for d > 0, stable at both ends (Maechler 2012); both branches finite, so where() never leaks a NaN."""
    small = d < math.log(2.0)
    d1 = d.clamp(min=1e-200, max=math.log(2.0))
    d2 = d.clamp(min=math.log(2.0))
    return torch.where(small, torch.log(-torch.expm1(-d1)), torch.log1p(-torch.exp(-d2)))


def _std_cdf(z, family):
    return torch.sigmoid(z) if family == "logistic" else torch.special.ndtr(z)


def num_log_masses(logpi, mu, s, u, kb, family):
    """log bin masses [B, Kmax] of the mixtures (logpi / mu / s [B, C], float64) on rows with interior edges u [B, Kmax-1] (the first
    kb - 1 entries real, strictly increasing) and kb [B] bins: bin 0 = (-inf, u_0], bin k = (u_{k-1}, u_k], bin kb-1 = (u_{kb-2}, +inf).
    Per component, in log space: logistic log(sig(b) - sig(a)) = logsig(b) + logsig(-a) + log1mexp(b - a), exact in both tails;
    Gaussian log(Phi(b) - Phi(a)) = log Phi(b) + log1mexp(log Phi(b) - log Phi(a)), mirrored (log S) when a > 0. Entries >= kb:
    -inf, with a zero gradient (they are computed from finite padding first). -> (logm [B, Kmax], valid [B, Kmax] bool)."""
    B, C = mu.shape
    E = u.shape[1]
    j = torch.arange(E, device=u.device)
    real = j[None, :] < (kb[:, None] - 1)
    last = u.gather(1, (kb - 2).clamp(min=0)[:, None])
    u = torch.where(real, u, last + (j[None, :] - kb[:, None] + 2).to(u.dtype))     # padding: finite, still increasing
    z = (u[:, None, :] - mu[:, :, None]) / s[:, :, None]                             # [B, C, E]
    if family == "logistic":
        lF, lS = torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z)
        mid = lF[..., 1:] + lS[..., :-1] + _log1mexp(z[..., 1:] - z[..., :-1])
    elif family == "gaussian":
        lF, lS = torch.special.log_ndtr(z), torch.special.log_ndtr(-z)
        up = z[..., :-1] > 0                                                         # both edges in the upper tail: use S
        big = torch.where(up, lS[..., :-1], lF[..., 1:])
        small = torch.where(up, lS[..., 1:], lF[..., :-1])
        mid = big + _log1mexp(big - small)
    else:
        raise ValueError(f"family {family!r}")
    lm = torch.cat([lF[..., :1], mid, lS[..., -1:]], -1)                             # [B, C, E + 1]
    k = torch.arange(E + 1, device=u.device)[None, None, :]
    kl = (kb - 1)[:, None, None]
    tail = lS.gather(2, (kb - 2).clamp(min=0)[:, None, None].expand(B, C, 1))
    lm = torch.where(k == kl, tail, lm)                                              # every row's last bin kb-1: its upper tail
    logm = torch.logsumexp(logpi[:, :, None] + lm, 1)
    valid = k[0] <= kl[:, 0]
    return logm.masked_fill(~valid, float("-inf")), valid


def num_cdf(logpi, mu, s, u, family):
    """The mixture CDF at u [B, E] (float64): sum_c pi_c F((u - mu_c) / s_c), closed form. [B, E]."""
    z = (u[:, None, :] - mu[:, :, None]) / s[:, :, None]
    return (logpi.exp()[:, :, None] * _std_cdf(z, family)).sum(1)


def num_crps_rows(logpi, mu, s, u, kb, t, family):
    """sum over each row's real interior edges of (F_mix(u_k) - G_k)^2, G = the target's cumulative masses: the binned CRPS minus the
    exact posterior's own, in bin units. [B]."""
    E = u.shape[1]
    real = torch.arange(E, device=u.device)[None, :] < (kb[:, None] - 1)
    d = num_cdf(logpi, mu, s, u, family) - torch.cumsum(t, 1)[:, :E]
    return (d * d).masked_fill(~real, 0.0).sum(1)


def num_head_rows(head, h, tf, u, kb, t, crps_w=0.0):
    """Per numeric row: (loss = CE + crps_w x CRPS-excess, CE, CRPS-excess), float64 [B] each."""
    logpi, mu, s = head(h, tf)
    logm, valid = num_log_masses(logpi, mu, s, u, kb, head.family)
    ce = -(t * logm.masked_fill(~valid, 0.0)).sum(1)
    cr = num_crps_rows(logpi, mu, s, u, kb, t, head.family)
    return ce + float(crps_w) * cr, ce, cr


class NumHead(torch.nn.Module):
    """(p95) h [B, H] (+ the row's coordinate index tf [B]) -> a C-component mixture in u: log weights, locations, scales [B, C]
    (float64). x = h / rms(h); y = W2 silu(W1 x + b1) + b2 + tf_bias[tf]; weights softmax(y_0), locations y_1, scales
    NUM_S_MIN + softplus(min(y_2, NUM_S_RAW_MAX)). W2 starts at zero (every input: weights equal, locations spread over [-1, 1],
    scales 0.5 -- about as good as uniform bins on the (94) val axes), so at init no gradient reaches h. Init draws come from its own
    generator."""

    def __init__(self, hidden, n_comp=4, family="logistic", dim=256, seed=0):
        super().__init__()
        if family not in NUM_HEAD_FAMILIES:
            raise ValueError(f"num head family {family!r}")
        C, H, D = int(n_comp), int(hidden), int(dim)
        self.C, self.H, self.D, self.family, self.seed = C, H, D, family, int(seed)
        g = torch.Generator().manual_seed(int(seed))
        self.W1 = torch.nn.Parameter(torch.randn(D, H, generator=g) / math.sqrt(H))
        self.b1 = torch.nn.Parameter(torch.zeros(D))
        self.W2 = torch.nn.Parameter(torch.zeros(3 * C, D))
        b2 = torch.zeros(3 * C)
        if C > 1:
            b2[C : 2 * C] = torch.linspace(-1.0, 1.0, C)
        b2[2 * C :] = math.log(math.expm1(0.5 - NUM_S_MIN))         # scales 0.5: on the (94) val axes ~ uniform (skill -0.02)
        self.b2 = torch.nn.Parameter(b2)
        self.tf_bias = torch.nn.Parameter(torch.zeros(len(NUM_TRANSFORMS), 3 * C))

    @classmethod
    def from_checkpoint(cls, ent):
        """The head of a checkpoint's `num_head` entry, on the CPU."""
        m = cls(ent["hidden"], ent["n_comp"], ent["family"], ent["dim"], ent.get("seed", 0))
        m.load_state_dict(ent["state"])
        return m

    def meta(self):
        return {"n_comp": self.C, "family": self.family, "dim": self.D, "hidden": self.H, "seed": self.seed,
                "transforms": list(NUM_TRANSFORMS), "s_min": NUM_S_MIN, "s_raw_max": NUM_S_RAW_MAX,
                "coords": "x display units on [lo, hi], w = hi - lo: std u = 4 (x - lo) / w - 2; logit u = logit((x - lo) / w); "
                          "log u = log(2 (x - lo) / w); neglog u = -log(2 (hi - x) / w)"}

    def forward(self, h, tf):
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        F = torch.nn.functional
        y = F.linear(F.silu(F.linear(x, self.W1, self.b1)), self.W2, self.b2) + self.tf_bias[tf]
        y = y.double()
        C = self.C
        return (torch.log_softmax(y[:, :C], -1), y[:, C : 2 * C],
                NUM_S_MIN + F.softplus(y[:, 2 * C :].clamp(max=NUM_S_RAW_MAX)))


def clip_params_(params, max_norm=1.0):
    """torch.nn.utils.clip_grad_norm_'s formula on `params` (their grads, in place); -> the total norm. A separate function so the
    LoRA's clip stays the only clip_grad_norm_ call of a step (test_ddp.py's child records exactly those)."""
    gs = [p.grad for p in params if p.grad is not None]
    if not gs:
        return 0.0
    total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach(), 2.0) for g in gs]), 2.0)
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for g in gs:
        g.detach().mul_(coef.to(g.device, g.dtype))
    return float(total)


def num_summaries(logpi, mu, s, family, tfs, lo, hi, qs=(0.05, 0.5, 0.95), span=60.0, grid=1201, iters=200):
    """Continuous summaries of mixtures in DISPLAY units (numpy float64): rows logpi / mu / s [n, C] in u, tfs [n] transform names,
    lo / hi [n]. -> {mean, sd [n]; q [n, len(qs)]}. Moments: trapezoid over each component's own grid mu_c + s_c t, t in [-span, span]
    (step 0.1: the mixture truncated where < 1e-26 of its mass lies outside; exact to ~1e-12 on std / logit axes; on a 'log' axis a
    logistic component's moments are heavy-tailed -- E x^k is infinite once k s >= 1 -- and the truncated values are reported, within
    ~1e-3 of the true sd at s = 0.4). Quantiles: bisection on the closed-form mixture CDF in u, then x = from_u(u)."""
    logpi, mu, s = (np.asarray(x, dtype=np.float64) for x in (logpi, mu, s))
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    tfs = np.asarray(tfs)
    n = len(mu)
    t = np.linspace(-span, span, grid)
    w = np.full(grid, t[1] - t[0])
    w[[0, -1]] /= 2
    if family == "logistic":
        f = np.exp(-np.abs(t)) / (1.0 + np.exp(-np.abs(t))) ** 2
        cdf = lambda z: np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    elif family == "gaussian":
        f = np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
        cdf = lambda z: torch.special.ndtr(torch.from_numpy(np.asarray(z, dtype=np.float64))).numpy()   # no scipy needed
    else:
        raise ValueError(family)
    pi = np.exp(logpi)
    mean, sd, q = np.full(n, np.nan), np.full(n, np.nan), np.full((n, len(qs)), np.nan)
    for tf in sorted(set(tfs.tolist())):
        r = np.nonzero(tfs == tf)[0]
        if not len(r):
            continue
        x = from_u(mu[r, :, None] + s[r, :, None] * t[None, None, :], lo[r, None, None], hi[r, None, None], tf)   # [m, C, grid]
        wf = (w * f)[None, None, :]
        m1 = (pi[r] * (wf * x).sum(-1)).sum(-1)
        var = (pi[r] * (wf * (x - m1[:, None, None]) ** 2).sum(-1)).sum(-1)
        mean[r], sd[r] = m1, np.sqrt(np.maximum(var, 0.0))
        a = (mu[r] - span * s[r]).min(1)[:, None].repeat(len(qs), 1)
        b = (mu[r] + span * s[r]).max(1)[:, None].repeat(len(qs), 1)
        qq = np.asarray(qs, dtype=np.float64)[None, :]
        for _ in range(iters):
            mid = (a + b) / 2
            F_ = (pi[r][:, None, :] * cdf((mid[:, :, None] - mu[r][:, None, :]) / s[r][:, None, :])).sum(-1)
            below = F_ < qq
            a, b = np.where(below, mid, a), np.where(below, b, mid)
        q[r] = from_u((a + b) / 2, lo[r, None], hi[r, None], tf)
    return {"mean": mean, "sd": sd, "q": q}


# --------------------------------------------------------------------------- prior offsets (p101, 2026-10-04)
# --prior-offset meta[:KEY]: a likelihood-ratio readout (neural ratio estimation style; Hermans et al. 2020) over a question's options:
#     q_i = softmax_i( log pi_i + l_i ),   l = the option-letter logits,   pi = the caller-supplied prior mass per option,
# so the network's letters are trained (same soft CE against the exact posterior, + RPS if set) to be the EVIDENCE l_i(D), and a prior
# enters by arithmetic: swapping it is exact given l, no data should give a flat l, independent evidence adds, l can be cached. pi comes
# from the record: meta[KEY] (default log_prior_bins; build_numeric_jev.py / build_numeric_evidence.py), a list of log masses in the
# question's STORED option order (levels for `score`, criteria keys for `choice`) or a {option key: log mass} dict. An example keeps
# its unit's offsets in LETTER order (off[order], order = the option permutation make_example drew), centred at its max (softmax-
# invariant; keeps float32 exact near the bins that matter), floored at PO_FLOOR; example_loss adds them to its row before anything
# else, on every training path and in evaluate(). Rows without offsets are taken as they are (torch.where: bit-identical values and
# gradients), and a forward without any offset row is untouched. The offsets' content enters the run identity (prior_offset_fp).
PO_FLOOR = -1e4                   # log pi floor (a zero-mass option: exp(-1e4) = 0 in float32 and float64)


def prior_offset_key(spec):
    """--prior-offset: 'off' -> None, 'meta' -> 'log_prior_bins', 'meta:KEY' -> KEY. ValueError otherwise."""
    s = str(spec).strip()
    if s == "off":
        return None
    if s == "meta":
        return "log_prior_bins"
    m = re.fullmatch(r"meta:([A-Za-z_][A-Za-z0-9_]*)", s)
    if not m:
        raise ValueError(f"--prior-offset: 'off', 'meta' or 'meta:KEY', got {spec!r}")
    return m.group(1)


def offsets_in_option_order(val, q):
    """meta[KEY] of a question -> float64 [k] log masses in its stored option order (cache_features.question_options' keys), or
    ValueError: a list of k numbers, or a dict keyed by option key; NaN / +inf refused; -inf (and anything lower) -> PO_FLOOR."""
    if q["type"] == "choice":
        keys = list(q["criteria"])
    elif q["type"] == "noul":
        keys = ["yes", "no"]
    elif q["type"] == "score":
        keys = list(q["levels"])
    else:
        raise ValueError(f"question type {q['type']!r}")
    return offsets_for_keys(val, keys)


def offsets_for_keys(val, keys):
    """offsets_in_option_order for a unit's option keys (cache_features' `option_keys`)."""
    if isinstance(val, dict):
        if set(map(str, val)) != set(map(str, keys)):
            raise ValueError(f"offset keys {sorted(val)[:4]} != option keys {keys[:4]}")
        v = [val[k] if k in val else val[str(k)] for k in keys]
    else:
        v = list(val)
    x = np.asarray(v, dtype=np.float64)
    if x.shape != (len(keys),) or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError(f"{len(v)} offsets for {len(keys)} options (or NaN / +inf)")
    return np.maximum(x, PO_FLOOR)


def prior_offset_index(data_dir, splits, key):
    """{(record id, qid): offsets in stored option order} for every question of `splits` whose record's meta has `key` (files as
    cache_features.iter_units reads them; lines without the key's name are skipped unparsed). A malformed entry is refused."""
    from pathlib import Path
    out, needle = {}, f'"{key}"'
    for split in splits:
        for p in sorted(Path(data_dir, split).glob("*.jsonl")):
            with p.open(encoding="utf-8") as f:
                for ln, line in enumerate(f):
                    if needle not in line:
                        continue
                    rec = json.loads(line)
                    val = (rec.get("meta") or {}).get(key)
                    if val is None:
                        continue
                    for qid, q in rec["questions"].items():
                        try:
                            out[(rec["id"], qid)] = offsets_in_option_order(val, q)
                        except ValueError as e:
                            raise SystemExit(f"--prior-offset: {p}:{ln + 1} question {qid}: meta.{key}: {e}") from None
    return out


def po_index_fingerprint(index, keys):
    """sha256 over the offsets of `keys` (in order; units without offsets add their key only): the priors a run trained with."""
    h = hashlib.sha256()
    for k in keys:
        h.update(json.dumps(list(k)).encode())
        s = index.get(k)
        if s is not None:
            h.update(np.ascontiguousarray(s).tobytes())
    return h.hexdigest()[:16]


class PriorExample(tuple):
    """(p101) make_example's (text, target) -- unpacked and indexed as before, the ordinal .levels kept -- plus .prior_off: the
    unit's log prior in LETTER order, centred at its max (float64 [k])."""

    def __new__(cls, e, off):
        x = super().__new__(cls, tuple(e))
        x.levels = getattr(e, "levels", None)
        x.prior_off = off
        return x


def po_letters(off, order):
    """Offsets in stored option order -> letter order (letter j shows option order[j]), centred at the max."""
    x = np.asarray(off, dtype=np.float64)[np.asarray(order, dtype=np.int64)]
    return x - x.max()


def with_prior_offsets(z, examples):
    """(p101) z [B, Kmax] letter logits -> z + log pi on the rows whose example carries .prior_off (letter order, its first k
    letters); every other row is z's own values (torch.where, so values and gradients are bit-identical). No offset row: z itself."""
    offs = [getattr(e, "prior_off", None) for e in examples]
    if all(o is None for o in offs):
        return z
    O = torch.zeros(z.shape, dtype=torch.float64)
    has = torch.zeros(len(offs), dtype=torch.bool)
    for b, o in enumerate(offs):
        if o is not None:
            O[b, : len(o)] = torch.from_numpy(np.asarray(o, dtype=np.float64))
            has[b] = True
    if z.is_cuda:                             # pinned + non_blocking (as --h2d async): no host sync for the offsets; same values
        O, has = host_to_dev(O, z.device), host_to_dev(has, z.device)
    return torch.where(has.to(z.device)[:, None], z + O.to(z.device, z.dtype), z)


# --------------------------------------------------------------------------- learned data mixture (p103, 2026-10-04)
# --mixture learned: the train units are split into GROUPS by file name (--mixture-groups numeric=num_*,replay=*: the first matching
# pattern, in spec order, takes a train/*.jsonl file; every file must match one, every group must get one). Each group is a stream read
# WITHOUT replacement: pass p of group g is a permutation of its units drawn by random.Random("<seed>|mixture|<g>|<p>") (a fresh shuffle
# per pass, cycling), read by the group's own cursor. Every example slot of a step draws its group from the current weights w with the
# mixture's own RNG (random.Random("<seed>|mixture-draw")), then takes that group's next unit; the option-order draws stay the example
# rng's, as before. All of it is deterministic in the seed and the weights, so under data parallelism every rank draws the same global
# step (as the static sampler), and the cursors, the draw RNG and the weights are the resume state.
# WEIGHTS (Xie et al. 2023, DoReMi; online, single run). Held-out VAL SETS (--mixture-val NAME[@GROUP]=PATH: a directory of *.jsonl, a
# file or a glob; at most --mixture-val-units units each, a seed-independent fixed sample) are scored every --mixture-every steps (and at
# step 0, and at the last step) with the main readout's per-unit soft CE (the val path's loss: prior offsets / ordinal readout applied;
# with --num-head a numeric unit's loss is the head's CE), sharded over ranks by chunk and summed in chunk order (the one-GPU numbers).
# At an update (step t > 0, t a multiple of --mixture-every, or the last step), per val set v attached to group g(v):
#     L_v  = mean_i CE_i,        H_v = mean_i H(t_i) (the labels' own entropy, so KL_v = L_v - H_v = mean KL(t || q) >= 0)
#     R_v  = H_v ('initial': a perfect model; the step-0 model enters through D_v)  |  the reference run's L_v ('file:PATH', DoReMi)
#     D_v  = max(L_v(0) - H_v, kl_floor) ('rel': units of v's step-0 excess)  |  1 ('abs': nats)
#     e_v  = (L_v - R_v) / D_v                      [initial + rel: e_v = KL_v(t) / KL_v(0), the share of v's step-0 excess NOT yet removed
#                                                    (= 1 - its relative improvement): the sets that improved LEAST have the largest e]
#     s_g  = mean_{v -> g} u_v * clip(e_v, lo, hi)  (u = --mixture-utility, default 1: u_v > 1 = v's remaining excess counts u_v times,
#                                                    more weight to its group while it has excess left; a group without val sets gets
#                                                    the others' weight-averaged signal: neutral to first order)
#     w_g <- w_g exp(eta s_g) / sum_h w_h exp(eta s_h)          (exponentiated gradient, i.e. a multiplicative-weights step)
#     w   <- (1 - eps) w + eps p                     (p = uniform or the static unit proportions: --mixture-smooth)
#     w   <- the bounds' projection (--mixture-min / --mixture-max: w_g -> clip(lambda w_g, min_g, max_g), lambda so that sum = 1)
# Every quantity is float64 host arithmetic on values every rank holds, so every rank computes the same w. NOTE (research/
# p103_learned_mixture.md): this is DoReMi's minimax objective (max over groups of the excess loss) -- a utility choice, not a neutral
# one; with 'initial' a group that cannot improve (data the parent already fit) keeps the largest e, and with eps > 0 the weights go to the
# fixed point w*_g ~ eps p_g / (1 - (1 - eps) exp(-eta d_g)) set by the persistent gap d_g, eta and eps. Default clip [0, 2]: e < 0 (only
# with a file reference: better than the reference) counts as 0, DoReMi's max(excess, 0); e > 2 (excess more than doubled) as 2.
# --mixture static (the default) changes nothing; with --mixture-val it only MONITORS (the same per-set losses logged and saved, the
# static sampler untouched: the same LoRA bits as without it).
MIXTURE_DEFAULTS = {"mixture": "static", "mixture_groups": "", "mixture_val_units": 300, "mixture_every": 250, "mixture_norm": "rel",
                    "mixture_kl_floor": 0.01, "mixture_eta": 1.0, "mixture_eps": 0.1, "mixture_smooth": "uniform",
                    "mixture_clip": "0:2", "mixture_init": "static", "mixture_min": "", "mixture_max": "", "mixture_utility": ""}
MIXTURE_LEARNED_ONLY = ("mixture_groups", "mixture_norm", "mixture_kl_floor", "mixture_eta", "mixture_eps", "mixture_smooth",
                        "mixture_clip", "mixture_init", "mixture_min", "mixture_max", "mixture_utility")
MIXTURE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
MIXTURE_STATE_VERSION = 1


def mixture_groups_spec(spec):
    """--mixture-groups 'g1=pat,g2=pat2,g1=pat3' -> [(group, [patterns])] in order of first appearance. ValueError on a bad entry."""
    out = {}
    for item in [x.strip() for x in str(spec).split(",") if x.strip()]:
        g, eq, pat = item.partition("=")
        g, pat = g.strip(), pat.strip()
        if not eq or not MIXTURE_NAME.fullmatch(g) or not pat or "/" in pat:
            raise ValueError(f"--mixture-groups: entries are NAME=PATTERN (a file-name pattern under train/), got {item!r}")
        out.setdefault(g, []).append(pat)
    if len(out) < 2:
        raise ValueError(f"--mixture-groups: at least two groups, got {spec!r}")
    return list(out.items())


def mixture_assign_files(stems, groups):
    """train file stems -> {stem: group index}: the first group (in spec order) one of whose patterns matches the stem or the file
    name. ValueError when a file matches no group or a group matches no file."""
    import fnmatch
    out = {}
    for s in stems:
        for gi, (_, pats) in enumerate(groups):
            if any(fnmatch.fnmatchcase(s, p) or fnmatch.fnmatchcase(s + ".jsonl", p) for p in pats):
                out[s] = gi
                break
        else:
            raise ValueError(f"--mixture-groups: train file {s}.jsonl matches no group's pattern")
    empty = [g for gi, (g, _) in enumerate(groups) if gi not in out.values()]
    if empty:
        raise ValueError(f"--mixture-groups: group(s) {empty} match no train file")
    return out


def mixture_val_spec(spec):
    """--mixture-val 'name@group=PATH,...' (@group omitted: the group of the same name) -> [{name, group, path}]. ValueError."""
    out, seen = [], set()
    for item in [x.strip() for x in str(spec).split(",") if x.strip()]:
        key, eq, path = item.partition("=")
        name, at, group = key.strip().partition("@")
        name, group, path = name.strip(), (group.strip() if at else name.strip()), path.strip()
        if not eq or not path or not MIXTURE_NAME.fullmatch(name) or not MIXTURE_NAME.fullmatch(group):
            raise ValueError(f"--mixture-val: entries are NAME[@GROUP]=PATH, got {item!r}")
        if name in seen:
            raise ValueError(f"--mixture-val: val set {name!r} named twice")
        seen.add(name)
        out.append({"name": name, "group": group, "path": path})
    if not out:
        raise ValueError("--mixture-val: no val set")
    return out


def mixture_val_files(path):
    """A val set's PATH -> its jsonl files (sorted): a directory's *.jsonl, a file, or a glob pattern. ValueError if none."""
    import glob as _glob
    from pathlib import Path
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    elif any(c in path for c in "*?["):
        files = [Path(x) for x in sorted(_glob.glob(path)) if x.endswith(".jsonl")]
    else:
        files = [p] if p.is_file() else []
    if not files:
        raise ValueError(f"--mixture-val: no *.jsonl at {path!r}")
    return files


def mixture_named(spec, names, what, lo=0.0, hi=float("inf")):
    """'a=0.2,b=0.5' -> {name: float} over known `names`, each value in [lo, hi]. ValueError."""
    out = {}
    for item in [x.strip() for x in str(spec).split(",") if x.strip()]:
        k, eq, v = item.partition("=")
        k = k.strip()
        try:
            x = float(v)
        except ValueError:
            x = float("nan")
        if not eq or k not in names or not (math.isfinite(x) and lo <= x <= hi):
            raise ValueError(f"{what}: entries are NAME=VALUE, NAME one of {list(names)}, VALUE in [{lo:g}, {hi:g}]; got {item!r}")
        out[k] = x
    return out


def mixture_clip_range(spec):
    """--mixture-clip 'C' -> (-C, C); 'LO:HI' -> (LO, HI); LO < HI, both finite. ValueError."""
    s = str(spec).strip()
    try:
        lo, hi = ((float(x) for x in s.split(":", 1)) if ":" in s else (-float(s), float(s)))
    except ValueError:
        raise ValueError(f"--mixture-clip: 'C' (symmetric) or 'LO:HI', got {spec!r}") from None
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"--mixture-clip: need LO < HI, finite, got {spec!r}")
    return lo, hi


def mixture_init_weights(spec, names, static_p):
    """--mixture-init 'static' | 'uniform' | 'w1,w2,...' (group order) | 'g=w,...' (every group) -> weights summing to 1, all > 0."""
    s = str(spec).strip()
    if s == "static":
        w = list(static_p)
    elif s == "uniform":
        w = [1.0] * len(names)
    elif "=" in s:
        d = mixture_named(s, names, "--mixture-init")
        if set(d) != set(names):
            raise ValueError(f"--mixture-init: a weight for every group {list(names)}, got {spec!r}")
        w = [d[g] for g in names]
    else:
        try:
            w = [float(x) for x in s.split(",")]
        except ValueError:
            raise ValueError(f"--mixture-init: 'static', 'uniform' or weights, got {spec!r}") from None
        if len(w) != len(names):
            raise ValueError(f"--mixture-init: {len(w)} weights for {len(names)} groups {list(names)}")
    if any(not (math.isfinite(x) and x > 0) for x in w):
        raise ValueError(f"--mixture-init: every group's weight must be > 0, got {spec!r}")
    t = sum(w)
    return [x / t for x in w]


def mixture_excess(L, R, L0, H, norm="rel", kl_floor=0.01):
    """e_v = (L - R) / D, D = max(L0 - H, kl_floor) ('rel') or 1 ('abs'); R = H for the 'initial' reference."""
    D = max(L0 - H, kl_floor) if norm == "rel" else 1.0
    return (L - R) / D


def mixture_project(w, lo, hi, iters=200):
    """w (sums to 1) -> the nearest distribution of the form clip(lambda w_g, lo_g, hi_g) (lambda by bisection; ratios kept among
    unclipped groups). w itself when every bound already holds (no rounding then)."""
    if all(a <= x <= b for x, a, b in zip(w, lo, hi)):
        return list(w)
    f = lambda lam: sum(min(max(lam * x, a), b) for x, a, b in zip(w, lo, hi))
    a_, b_ = 0.0, max(b / x for x, b in zip(w, hi) if x > 0) + 1.0
    for _ in range(iters):
        m = (a_ + b_) / 2
        a_, b_ = (m, b_) if f(m) < 1.0 else (a_, m)
    lam = (a_ + b_) / 2
    v = [min(max(lam * x, a), b) for x, a, b in zip(w, lo, hi)]
    t = sum(v)
    return [x / t for x in v]


def mixture_update(w, s, eta, eps, p, lo=None, hi=None):
    """One exponentiated-gradient step on the weights (float64 host arithmetic; see the section comment):
    w_g exp(eta s_g) / sum_h w_h exp(eta s_h), then (1 - eps) w + eps p, then the bounds' projection."""
    z = [math.log(x) + eta * y for x, y in zip(w, s)]
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    t = sum(ex)
    v = [(1.0 - eps) * (x / t) + eps * q for x, q in zip(ex, p)]
    G = len(v)
    return mixture_project(v, lo or [0.0] * G, hi or [1.0] * G)


class Mixture:
    """(p103) The learned (or monitored) data mixture: per-group streams, the group draws, the weights and their update history.
    `units`: per group, its units in load order (None in a static monitor). `val`: the val sets [{name, group (index or None), utility,
    H, n}]. Host-side and deterministic: state() / load_state() round-trip it exactly (plain values, weights_only-loadable)."""

    def __init__(self, groups, units, seed, w0, val, *, eta=1.0, eps=0.1, p=None, clip=(0.0, 2.0), lo=None, hi=None, norm="rel",
                 kl_floor=0.01, ref=None, learned=True):
        self.learned, self.groups, self.units, self.seed = bool(learned), list(groups), units, int(seed)
        G = len(self.groups)
        self.n = [len(x) for x in units] if units is not None else []
        self.w = list(w0) if w0 is not None else []
        self.w0 = list(self.w)
        self.val = [dict(v) for v in val]
        self.eta, self.eps, self.clip, self.norm, self.kl_floor = float(eta), float(eps), tuple(clip), norm, float(kl_floor)
        self.p = list(p) if p is not None else [1.0 / max(G, 1)] * G
        self.lo, self.hi = list(lo or [0.0] * G), list(hi or [1.0] * G)
        self.ref = ref                           # None ('initial': R_v = H_v) or {val name: reference loss} ('file:')
        self.cursors = [0] * G
        self.draw_rng = random.Random(f"{self.seed}|mixture-draw")
        self.L0, self.history = None, []
        self._perm = {}

    # ---- streams and draws
    def perm(self, g, p):
        """Group g's pass-p order: a permutation of range(n_g) from random.Random('<seed>|mixture|<name>|<p>')."""
        c = self._perm.get(g)
        if c is None or c[0] != p:
            order = list(range(self.n[g]))
            random.Random(f"{self.seed}|mixture|{self.groups[g]}|{p}").shuffle(order)
            self._perm[g] = c = (p, order)
        return c[1]

    def take(self, g):
        """The next unit of group g's stream (without replacement within a pass; a new shuffle per pass)."""
        c = self.cursors[g]
        self.cursors[g] = c + 1
        return self.units[g][self.perm(g, c // self.n[g])[c % self.n[g]]]

    def draw_group(self):
        r, acc = self.draw_rng.random(), 0.0
        for g, x in enumerate(self.w):
            acc += x
            if r < acc:
                return g
        return max(g for g, x in enumerate(self.w) if x > 0)    # r within rounding of 1

    def next_unit(self):
        return self.take(self.draw_group())

    # ---- the update
    def observe(self, step, losses):
        """Record the val sets' mean losses {name: L} at `step`; at step > 0 (learned) update the weights. -> the history entry."""
        names = [v["name"] for v in self.val]
        if set(losses) != set(names):
            raise RuntimeError(f"mixture: losses for {sorted(losses)} != val sets {names}")
        if self.L0 is None:
            if step != 0:
                raise RuntimeError(f"mixture: the first val losses must be step 0's, got step {step}")
            self.L0 = {k: float(losses[k]) for k in names}
        rec = {"step": int(step), "loss": {k: float(losses[k]) for k in names},
               "kl": {v["name"]: float(losses[v["name"]]) - v["H"] for v in self.val}}
        if self.learned:
            rec["w_before"] = list(self.w)
            if step > 0:
                R = self.ref if self.ref is not None else {v["name"]: v["H"] for v in self.val}
                e = {v["name"]: mixture_excess(rec["loss"][v["name"]], R[v["name"]], self.L0[v["name"]], v["H"], self.norm,
                                               self.kl_floor) for v in self.val}
                s = self.signal(e)
                self.w = mixture_update(self.w, s, self.eta, self.eps, self.p, self.lo, self.hi)
                rec["excess"], rec["signal"] = e, dict(zip(self.groups, s))
            rec["w_after"], rec["drawn"] = list(self.w), list(self.cursors)
        self.history.append(rec)
        return rec

    def signal(self, e):
        """s_g = mean over the val sets attached to g of utility x clip(e_v); a group with none gets the weight-averaged signal of
        the groups that have one (it neither gains nor loses to first order)."""
        lo, hi = self.clip
        out = [None] * len(self.groups)
        for g in range(len(self.groups)):
            vs = [v for v in self.val if v["group"] == g]
            if vs:
                out[g] = sum(v["utility"] * min(max(e[v["name"]], lo), hi) for v in vs) / len(vs)
        have = [g for g, x in enumerate(out) if x is not None]
        wsum = sum(self.w[g] for g in have)
        fill = sum(self.w[g] * out[g] for g in have) / wsum if have and wsum > 0 else 0.0
        return [fill if x is None else x for x in out]

    def mean_weights(self, step):
        """The time average of the sampling weights over steps 1..step (DoReMi's output weights): each update's w_after holds from
        its step to the next update."""
        if not self.learned or step <= 0:
            return list(self.w)
        acc, prev, w = [0.0] * len(self.w), 0, self.w0
        for h in self.history:
            if h["step"] >= step:
                break
            if h["step"] > prev:
                acc = [a + x * (h["step"] - prev) for a, x in zip(acc, w)]
                prev = h["step"]
            w = h["w_after"]
        acc = [a + x * (step - prev) for a, x in zip(acc, w)]
        return [a / step for a in acc]

    # ---- resume state
    def state(self):
        st = self.draw_rng.getstate()
        return {"version": MIXTURE_STATE_VERSION, "learned": self.learned, "groups": list(self.groups),
                "val": [v["name"] for v in self.val], "cursors": list(self.cursors), "w": list(self.w),
                "draw_rng": [st[0], list(st[1]), st[2]], "L0": dict(self.L0) if self.L0 is not None else None,
                "history": json.loads(json.dumps(self.history))}

    def load_state(self, s):
        if s.get("version") != MIXTURE_STATE_VERSION or list(s["groups"]) != self.groups or list(s["val"]) != \
                [v["name"] for v in self.val] or bool(s["learned"]) != self.learned:
            raise ValueError(f"mixture state {s.get('groups')} / {s.get('val')} does not match this run's {self.groups} / "
                             f"{[v['name'] for v in self.val]}")
        self.cursors, self.w = [int(x) for x in s["cursors"]], [float(x) for x in s["w"]]
        r = s["draw_rng"]
        self.draw_rng.setstate((int(r[0]), tuple(int(x) for x in r[1]), r[2]))
        self.L0 = {k: float(x) for k, x in s["L0"].items()} if s["L0"] is not None else None
        self.history = json.loads(json.dumps(s["history"]))
        self._perm = {}

    def summary(self, step):
        """The checkpoint config's `mixture` block."""
        out = {"mode": "learned" if self.learned else "static",
               "val_sets": [{**{k: v[k] for k in ("name", "n", "H", "utility")},
                             "group": self.groups[v["group"]] if v["group"] is not None else v.get("group_name"),
                             **({"fp": v["fp"]} if "fp" in v else {})} for v in self.val],
               "L0": self.L0, "history": json.loads(json.dumps(self.history))}
        if self.learned:
            out.update(groups=list(self.groups), units=list(self.n), weights=list(self.w), weights_mean=self.mean_weights(step),
                       drawn=list(self.cursors), w0=list(self.w0), eta=self.eta, eps=self.eps, smooth_p=list(self.p),
                       clip=list(self.clip), min=list(self.lo), max=list(self.hi), norm=self.norm, kl_floor=self.kl_floor,
                       ref="initial" if self.ref is None else {"file": dict(self.ref)})
        return out


def label_entropy(t):
    """H(t) in nats of a (normalised) target, float64; 0 log 0 = 0."""
    x = np.asarray(t, dtype=np.float64)
    x = x[x > 0]
    return float(-(x * np.log(x)).sum())


def mixture_ref_losses(path):
    """'file:PATH' -> ({val name: reference loss}, sha256 of the file, {val name: units fingerprint} where the file records one): a JSON
    {"loss": {...}} / {"losses": {...}} / {name: loss} (+ optional "val_sets": [{name, fp}], as fleet/task_103.sh's ref_losses_s<S>.json),
    or a claude_lmtune checkpoint / resume file whose config.mixture holds a history (its last entry's losses: a reference RUN's final
    model, as DoReMi's reference model)."""
    sha = file_sha256(path)
    vs = None
    if path.endswith(".json"):
        with open(path) as f:
            j = json.load(f)
        d = (j.get("loss") or j.get("losses") or j) if isinstance(j, dict) else None
        vs = j.get("val_sets") if isinstance(j, dict) and isinstance(j.get("loss") or j.get("losses"), dict) else None
    else:
        ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        m = ((ck.get("config") or {}).get("mixture") or {}) if isinstance(ck, dict) else {}
        hist, vs = m.get("history"), m.get("val_sets")
        d = hist[-1]["loss"] if hist else None
    if not isinstance(d, dict) or not d or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in d.values()):
        raise ValueError(f"--mixture-ref file:{path}: no per-val-set losses found (a JSON {{name: loss}} or a checkpoint with a mixture "
                         f"history)")
    fps = {str(v["name"]): v["fp"] for v in (vs or []) if isinstance(v, dict) and v.get("fp")}
    return {str(k): float(x) for k, x in d.items()}, sha, fps


# --------------------------------------------------------------------------- hierarchical sampling (datavar, 2026-10-04)
# --sampling hier: every example slot draws down a hierarchy instead of reading the pooled shuffle (research/data_variants.md):
#     [capability ->] source -> case -> question.
# SOURCES: one per train file (its stem), or --sampling-sources NAME=PATTERN,... (--mixture-groups' rule: fnmatch on the stem or the
# file name, the first NAME in spec order whose pattern matches takes the file; a NAME may repeat; every file must match one, every
# NAME must get one; ONE source is allowed: then only the case level acts). CASES: inside a source, the units sharing a case key --
# the unit's base_id (--sampling-case base, the default: a record and its option-reshuffled variants) or its record_id ('record') --
# in order of first appearance in load order; a case's questions in load order. Built after the NC / K filters (and the whole-case
# listing), from the default `train` list (loaded file by file, so each unit's file is known; the same list, the same order). A source
# whose units the filters ALL remove (ms_marco under the NC filter) is DROPPED with a logged note: no weight, no draws, a NAME=W entry
# for it ignored; only a pattern that matches no train FILE is refused.
# A SLOT: with --sampling-capability-map, a capability tag by the capability weights, then a source by the source weights
# renormalised within that tag; without, a source by the source weights. Each level is ONE .random() of the draw RNG
# random.Random("<seed>|hier-draw") against the cumulative weights (Mixture.draw_group's rule). Then that source's NEXT CASE: its cursor
# C_s, pass p = C_s // n_s, the case at position C_s % n_s of pass p's permutation of its n_s cases (random.Random("<seed>|hier|
# <source>|<p>"), a fresh shuffle per pass: uniform within the source, without replacement per pass). Then ONE QUESTION of that case:
# a case is visited exactly once per pass of its source, so its visit index is p, and with m questions it shows question p % m of the
# permutation of its questions for round p // m (random.Random("<seed>|hier-q|<source>|<case index>|<p // m>"); no draw when m = 1):
# without replacement within the case. So the sampler's whole state is the draw RNG and one cursor per source (the resume state), and
# under data parallelism every rank holds the same units and draws the same global step (as the static sampler). The example rng (the
# option orders) is untouched and the pooled shuffle still runs, so a slot's option order is drawn as before.
# SOURCE WEIGHTS (--sampling-source-weights): 'uniform' (the default: every source alike); 'cases' (proportional to its number of
# cases: every independent case counts once, whatever its number of questions); 'units' (proportional to its questions: the pooled
# sampler's source shares, with the case balancing inside each source); 'cases:T' / 'units:T' (proportional to n^T, T > 0); or
# 'NAME=W,...' (every source named, or '*=W' for the rest). CAPABILITIES (--sampling-capability-map FILE: a JSON {source: tag}, '*' =
# the tag of every unnamed source; entries naming no source of this run are ignored and counted in the log): the capability weights
# (--sampling-capability-weights) 'uniform' (the default), 'sources' (a tag's weight = the sum of its sources' weights: the same source
# marginals as without the map) or 'TAG=W,...' (every tag, or '*=W'). A source's marginal share of the slots is w_cap(tag) x w_s /
# (the sum of w over the tag's sources): hier_marginals(), logged and in the checkpoint config.
# --sampling pooled (the default) changes nothing (outputs, logs, identity).
SAMPLING_DEFAULTS = {"sampling": "pooled", "sampling_sources": "", "sampling_case": "base", "sampling_source_weights": "uniform",
                     "sampling_capability_weights": "uniform"}
SAMPLING_STATE_VERSION = 1
_HS_KIND = re.compile(r"(uniform|cases|units)(?::(.*))?")


def sampling_sources_spec(spec):
    """--sampling-sources '' -> None (one source per train file); 'NAME=PATTERN,...' -> [(name, [patterns])] in order of first
    appearance (a NAME may repeat to add patterns; one source is allowed). ValueError on a bad entry."""
    out = {}
    s = str(spec).strip()
    for item in [x.strip() for x in s.split(",") if x.strip()]:
        g, eq, pat = item.partition("=")
        g, pat = g.strip(), pat.strip()
        if not eq or not MIXTURE_NAME.fullmatch(g) or not pat or "/" in pat:
            raise ValueError(f"--sampling-sources: entries are NAME=PATTERN (a file-name pattern under train/), got {item!r}")
        out.setdefault(g, []).append(pat)
    if s and not out:
        raise ValueError(f"--sampling-sources: no NAME=PATTERN entry in {spec!r}")
    return list(out.items()) or None


def sampling_assign_files(stems, sources):
    """train file stems (load order) -> (source names, per source its patterns, {stem: source index}). `sources` None: one source per
    file, named by its stem. ValueError when a file matches no source or a source matches no file."""
    import fnmatch
    if sources is None:
        return list(stems), [[s] for s in stems], {s: i for i, s in enumerate(stems)}
    out = {}
    for s in stems:
        for gi, (_, pats) in enumerate(sources):
            if any(fnmatch.fnmatchcase(s, p) or fnmatch.fnmatchcase(s + ".jsonl", p) for p in pats):
                out[s] = gi
                break
        else:
            raise ValueError(f"--sampling-sources: train file {s}.jsonl matches no source's pattern")
    empty = [g for gi, (g, _) in enumerate(sources) if gi not in out.values()]
    if empty:
        raise ValueError(f"--sampling-sources: source(s) {empty} match no train file")
    return [g for g, _ in sources], [list(p) for _, p in sources], out


def hier_named(spec, names, what):
    """'a=0.2,b=1,*=0.5' -> {name: weight}: every key one of `names` or '*', each named once, every weight finite and > 0. ValueError."""
    out = {}
    for item in [x.strip() for x in str(spec).split(",") if x.strip()]:
        k, eq, v = item.partition("=")
        k = k.strip()
        try:
            x = float(v)
        except ValueError:
            x = float("nan")
        if not eq or (k != "*" and k not in names) or k in out or not (math.isfinite(x) and x > 0):
            raise ValueError(f"{what}: entries are NAME=W with NAME one of {list(names)} or '*' (each once) and W > 0; got {item!r}")
        out[k] = x
    if not out:
        raise ValueError(f"{what}: no NAME=W entry in {spec!r}")
    return out


def hier_spec_canon(spec, kinds, what):
    """A weight spec's syntax check before the sources are known, and one spelling per value for the identity ('cases:1' -> 'cases',
    'units:0.50' -> 'units:0.5'). `kinds`: the keywords allowed ('cases' / 'units' take an optional ':T')."""
    s = str(spec).strip()
    m = _HS_KIND.fullmatch(s)
    if s in kinds and s not in ("cases", "units"):
        return s
    if m and m.group(1) in kinds and m.group(1) in ("cases", "units"):
        if m.group(2) is None:
            return m.group(1)
        try:
            t = float(m.group(2))
        except ValueError:
            t = float("nan")
        if not (math.isfinite(t) and t > 0):
            raise ValueError(f"{what}: '{m.group(1)}:T' needs a finite T > 0, got {spec!r}")
        return m.group(1) if t == 1.0 else f"{m.group(1)}:{t:g}"
    if "=" not in s:
        raise ValueError(f"{what}: one of {list(kinds)} or NAME=W,..., got {spec!r}")
    hier_named(s, [k.partition("=")[0].strip() for k in s.split(",") if k.strip()], what)   # weights checked; names later
    return s


def hier_source_weights(spec, names, units, cases, dropped=()):
    """--sampling-source-weights -> per-source weights (in `names` order, sum 1, all > 0) given each source's unit and case counts.
    `dropped`: sources the filters emptied (not in `names`): a NAME=W entry for one is accepted and ignored."""
    s = hier_spec_canon(spec, ("uniform", "cases", "units"), "--sampling-source-weights")
    m = _HS_KIND.fullmatch(s)
    if m:
        t = float(m.group(2)) if m.group(2) else 1.0
        n = cases if m.group(1) == "cases" else units if m.group(1) == "units" else [1] * len(names)
        w = [float(x) ** t for x in n]
    else:
        d = hier_named(s, list(names) + [g for g in dropped if g not in names], "--sampling-source-weights")
        miss = [g for g in names if g not in d and "*" not in d]
        if miss:
            raise ValueError(f"--sampling-source-weights: no weight for source(s) {miss} (name every source, or give '*=W')")
        w = [d.get(g, d.get("*")) for g in names]
    if not w or any(not (math.isfinite(x) and x > 0) for x in w):
        raise ValueError(f"--sampling-source-weights {spec!r}: every source's weight must be finite and > 0, got {w}")
    t = sum(w)
    return [x / t for x in w]


def hier_capability_map_load(path):
    """--sampling-capability-map FILE -> {source (or '*'): tag}: a non-empty JSON object of non-empty strings. ValueError / OSError."""
    with open(path, encoding="utf-8") as f:
        try:
            j = json.load(f)
        except ValueError as e:
            raise ValueError(f"--sampling-capability-map {path}: not JSON ({e})") from None
    if not isinstance(j, dict) or not j or any(not isinstance(v, str) or not v.strip() for v in j.values()):
        raise ValueError(f"--sampling-capability-map {path}: a non-empty JSON object {{source: tag}} with string tags ('*' = "
                         f"every other source)")
    return {str(k): v.strip() for k, v in j.items()}


def hier_capability_tags(cmap, names):
    """-> (per source its tag, the map keys naming no source of this run). ValueError when a source has no tag."""
    miss = [g for g in names if g not in cmap and "*" not in cmap]
    if miss:
        raise ValueError(f"--sampling-capability-map: no tag for source(s) {miss} (map every source, or give '*')")
    return [cmap.get(g, cmap.get("*")) for g in names], sorted(k for k in cmap if k != "*" and k not in names)


def hier_capability_weights(spec, caps, tags, w, dropped=()):
    """--sampling-capability-weights over the tags `caps` (first-appearance order), given each source's tag and weight. `dropped`:
    tags whose every source the filters emptied (not in `caps`): a TAG=W entry for one is accepted and ignored."""
    s = hier_spec_canon(spec, ("uniform", "sources"), "--sampling-capability-weights")
    if s == "uniform":
        c = [1.0] * len(caps)
    elif s == "sources":
        c = [sum(x for x, t in zip(w, tags) if t == k) for k in caps]
    else:
        d = hier_named(s, list(caps) + [k for k in dropped if k not in caps], "--sampling-capability-weights")
        miss = [k for k in caps if k not in d and "*" not in d]
        if miss:
            raise ValueError(f"--sampling-capability-weights: no weight for tag(s) {miss} (name every tag, or give '*=W')")
        c = [d.get(k, d.get("*")) for k in caps]
    if not c or any(not (math.isfinite(x) and x > 0) for x in c):
        raise ValueError(f"--sampling-capability-weights {spec!r}: every tag's weight must be finite and > 0, got {c}")
    t = sum(c)
    return [x / t for x in c]


def hier_resolve(w, tags, caps, cw):
    """-> (per tag its source indices, per tag the within-tag source weights (sum 1), per source its marginal share)."""
    members = [[i for i, t in enumerate(tags) if t == k] for k in caps]
    within = [[w[i] / sum(w[j] for j in mem) for i in mem] for mem in members]
    marg = [0.0] * len(w)
    for k, mem in enumerate(members):
        for i, x in zip(mem, within[k]):
            marg[i] = cw[k] * x
    return members, within, marg


def hier_dropped_tags(cmap, dropped, caps):
    """The tags of the filter-emptied sources that no remaining source carries (their TAG=W entries are ignored, not refused)."""
    return [t for t in dict.fromkeys(cmap.get(g, cmap.get("*")) for g in dropped) if t is not None and t not in caps]


def hier_marginals(counts, spec="uniform", cap_map=None, cap_spec="uniform"):
    """The effective source shares of --sampling hier, from counts alone (no model, no data): `counts` {source: (units, cases)} or
    {source: {"units": u, "cases": c}} AFTER the NC / K filters, in source order; `spec` = --sampling-source-weights; `cap_map` = a
    capability map {source: tag} ('*' default) or None; `cap_spec` = --sampling-capability-weights. A source with 0 units is dropped as
    the trainer drops it (no weight; a NAME=W entry for it is ignored) and gets share 0.0. -> {source: marginal share of the example
    slots} (sum 1 over the sources with units)."""
    pick = lambda v, k, i: v[k] if isinstance(v, dict) else v[i]
    allnames = list(counts)
    names = [g for g in allnames if int(pick(counts[g], "units", 0)) > 0]
    dropped = [g for g in allnames if g not in names]
    if not names:
        raise ValueError("hier_marginals: no source has units")
    units = [int(pick(counts[g], "units", 0)) for g in names]
    cases = [int(pick(counts[g], "cases", 1)) for g in names]
    w = hier_source_weights(spec, names, units, cases, dropped)
    if cap_map is None:
        share = dict(zip(names, w))
    else:
        tags, _ = hier_capability_tags(cap_map, names)
        caps = list(dict.fromkeys(tags))
        cw = hier_capability_weights(cap_spec, caps, tags, w, hier_dropped_tags(cap_map, dropped, caps))
        share = dict(zip(names, hier_resolve(w, tags, caps, cw)[2]))
    return {g: share.get(g, 0.0) for g in allnames}


def _hs_pick(r, w):
    """Mixture.draw_group's rule: the first index whose cumulative weight exceeds r (the last positive one if r is within rounding
    of 1)."""
    acc = 0.0
    for g, x in enumerate(w):
        acc += x
        if r < acc:
            return g
    return max(g for g, x in enumerate(w) if x > 0)


class HierSampler:
    """(datavar) --sampling hier: [capability ->] source -> case -> question draws (see the section comment). `cases`: per source its
    cases in first-appearance order, each the list of its units in load order; `w` the source weights; `tags` / `caps` / `cap_w`: each
    source's capability tag, the tags (first-appearance order) and their weights (None: no capability level). Host-side and
    deterministic: state() / load_state() round-trip it exactly (plain values, weights_only-loadable)."""

    def __init__(self, names, cases, seed, w, tags=None, caps=None, cap_w=None):
        self.names, self.cases, self.seed = list(names), cases, int(seed)
        self.n = [len(c) for c in cases]
        self.units = [sum(len(x) for x in c) for c in cases]
        self.w = list(w)
        if caps:
            self.tags, self.caps, self.cap_w = list(tags), list(caps), list(cap_w)
            self.members, self.within, self.marg = hier_resolve(self.w, self.tags, self.caps, self.cap_w)
        else:
            self.tags = self.caps = self.cap_w = self.members = self.within = None
            self.marg = list(self.w)
        self.cursors = [0] * len(self.names)
        self.draw_rng = random.Random(f"{self.seed}|hier-draw")
        self._perm = {}

    def perm(self, s, p):
        """Source s's pass-p order of its cases: a permutation of range(n_s) from random.Random('<seed>|hier|<source>|<p>')."""
        c = self._perm.get(s)
        if c is None or c[0] != p:
            order = list(range(self.n[s]))
            random.Random(f"{self.seed}|hier|{self.names[s]}|{p}").shuffle(order)
            self._perm[s] = c = (p, order)
        return c[1]

    def question(self, s, c, p):
        """Case c of source s on its source's pass p (= its visit index): question p % m of round p // m's permutation of its m."""
        qs = self.cases[s][c]
        m = len(qs)
        if m == 1:
            return qs[0]
        order = list(range(m))
        random.Random(f"{self.seed}|hier-q|{self.names[s]}|{c}|{p // m}").shuffle(order)
        return qs[order[p % m]]

    def draw_source(self):
        if self.caps is None:
            return _hs_pick(self.draw_rng.random(), self.w)
        k = _hs_pick(self.draw_rng.random(), self.cap_w)
        return self.members[k][_hs_pick(self.draw_rng.random(), self.within[k])]

    def take(self, s):
        """Source s's next case (without replacement within a pass, a fresh shuffle per pass) and that case's next question."""
        C = self.cursors[s]
        self.cursors[s] = C + 1
        p = C // self.n[s]
        return self.question(s, self.perm(s, p)[C % self.n[s]], p)

    def next_unit(self):
        return self.take(self.draw_source())

    def state(self):
        st = self.draw_rng.getstate()
        return {"version": SAMPLING_STATE_VERSION, "sources": list(self.names), "cases": list(self.n), "cursors": list(self.cursors),
                "draw_rng": [st[0], list(st[1]), st[2]]}

    def load_state(self, s):
        if s.get("version") != SAMPLING_STATE_VERSION or list(s["sources"]) != self.names or list(s["cases"]) != self.n:
            raise ValueError(f"sampling state {s.get('sources')} / cases {s.get('cases')} does not match this run's {self.names} / "
                             f"{self.n}")
        self.cursors = [int(x) for x in s["cursors"]]
        r = s["draw_rng"]
        self.draw_rng.setstate((int(r[0]), tuple(int(x) for x in r[1]), r[2]))
        self._perm = {}

    def summary(self, files, case_mode, spec, cap_spec=None):
        """The checkpoint config's `sampling` block."""
        out = {"mode": "hier", "case": case_mode, "source_weights": spec,
               "sources": [{"name": g, "files": files[i], "units": self.units[i], "cases": self.n[i],
                            "q_per_case": self.units[i] / self.n[i], "weight": self.w[i], "share": self.marg[i],
                            **({"capability": self.tags[i]} if self.caps else {}), "drawn": self.cursors[i],
                            "passes": self.cursors[i] / self.n[i]} for i, g in enumerate(self.names)]}
        if self.caps:
            out["capabilities"] = [{"tag": k, "weight": self.cap_w[j], "sources": [self.names[i] for i in self.members[j]]}
                                   for j, k in enumerate(self.caps)]
            out["capability_weights"] = cap_spec
        return out


# --------------------------------------------------------------------------- dynamics log (ss, 2026-10-07)
# --dyn-log PATH: per-example TRAINING DYNAMICS for data maps (Swayamdipta et al. 2020, dataset cartography: per unit, confidence = the
# mean gold probability over epochs, variability = its std, correctness = the share of epochs it is the argmax). One JSON line per
# training example forwarded, {"step", "epoch", "i", "uid", "src", "k", "gold", "p_gold", "p_tgt", "loss", "correct"}:
#   step     the optimizer step the example trained in; i its draw index within the step (0.., in the draw order of the GLOBAL step,
#            which every DDP rank draws whole); epoch the pass over train/ the draw came from (the pooled sampler: cursor // len(train);
#            null under --mixture learned / --sampling hier, which have no single pass);
#   uid      claude_lmscore.py's rowid-1 unit id (unit_uids): json [record_id, qid], plus a repeat index m >= 1 for the m-th repeat of
#            that pair, over train/ in load order before any filter (cf.iter_units' order; group by group under --mixture learned), so
#            a data map joins a scorer's test_uid rows; src the unit's `source` (the record's, else its file stem); k its option count;
#   gold     the argmax of the unit's soft target in its STORED option order (the unpermuted order; the first on a tie): the same
#            option every epoch, whatever letter it was shown under; p_gold that option's probability under the loss's own readout
#            (example_loss's log p: the softmax over the k option letters of z, after --prior-offset's log pi and --ordinal-readout's
#            map); p_tgt = sum_o t_o p_o; correct = the readout's argmax letter (the first on a tie) shows the gold option;
#   loss     the example's own term of the step's loss: soft CE + rps_weight x RPS + brier_weight x Brier of its row (a forward's loss
#            is their mean, so in exact arithmetic its rows average to it); under --num-head a numeric row's term is the head's
#            (--num-head-letters off) or letters + head (on). p_gold / p_tgt / correct always read the letters.
# LOGGING ONLY: the log p example_loss already returns, kept detached (no draw, no op in any graph, nothing a step computes changes).
# Per step its forwards' log p go to the host in ONE copy, queued after the step's backwards and read after its optimizer step (one
# host wait); the lines (sorted by i, floats to 6 significant digits) are written and flushed. Eval / val forwards are never logged.
# Data parallel: each rank writes PATH.rank<r>, the examples it forwarded (every (step, i) in exactly one file). RESUME: at start the
# file keeps only the records of steps <= the resumed step (a stopped run logs steps past its last saved state, which this run redoes;
# a kill may leave a torn last line): rewritten through a tmp file + os.replace when anything is dropped; no file (another node): a
# new one, from the resumed step on. Off (the default): none of it runs.
def dyn_line_step(line):
    """(ss) The step of one dynamics-log line (bytes), None for a torn / foreign line."""
    if not line.endswith(b"\n"):
        return None
    if line.startswith(b'{"step": '):           # DynLog writes the step first: no JSON parse for the common case
        head = line[9: line.find(b",")]
        if head.isdigit():
            return int(head)
    try:
        s = json.loads(line).get("step")
    except (ValueError, AttributeError):
        return None
    return s if isinstance(s, int) else None


def dyn_loss_rows(logp, tgt, ks, examples, rps_weight=0.0, brier_weight=0.0):
    """(ss) example_loss's per-row terms from its own log p [B, Kmax] (host tensors): soft CE + rps_weight x RPS + brier_weight x
    Brier, whose mean over the rows is example_loss's value in exact arithmetic."""
    mask = torch.arange(logp.shape[1])[None, :] < ks[:, None]
    rows = -(tgt * logp.masked_fill(~mask, 0.0)).sum(-1)
    if rps_weight:
        lev, sc = ls.ordinal().batch_levels(examples, logp.shape[1], "cpu")
        rows = rows + rps_weight * ls.ordinal().rps_rows(logp, tgt, lev, sc)
    if brier_weight:
        rows = rows + brier_weight * brier_rows(logp, tgt, ks)
    return rows


def _sig6(x):
    return float(f"{x:.6g}")


class DynLog:
    """(ss) --dyn-log (see the section comment): draw() at every draw, fwd() after every training forward's example_loss, head() with
    the numeric head's rows of that forward, ready() after the step's backwards (the one device->host copy, queued), write() after its
    optimizer step (read, lines out, flush). open() trims and opens the file before the first step."""

    def __init__(self, path, uids, n_pass=None, rps_weight=0.0, brier_weight=0.0, head_letters=None):
        self.path, self.uids, self.n_pass = path, uids, n_pass
        self.rps, self.brier, self.head_letters = rps_weight, brier_weight, head_letters
        self.f, self.meta, self.fwds, self.pend, self.i, self.failed = None, {}, [], None, 0, False

    def open(self, step):
        kept = drop = torn = 0
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                for ln in f:
                    s = dyn_line_step(ln)
                    kept += s is not None and s <= step
                    drop += s is not None and s > step
                    torn += s is None
            if drop or torn:
                tmp = f"{self.path}.trim.tmp"           # a fixed name: one a kill left is overwritten next time
                with open(self.path, "rb") as f, open(tmp, "wb") as g:
                    for ln in f:
                        s = dyn_line_step(ln)
                        if s is not None and s <= step:
                            g.write(ln)
                    g.flush()
                    os.fsync(g.fileno())
                os.replace(tmp, self.path)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")
        print(f"dyn-log: {self.path}: {kept} records of steps <= {step} kept" + (f", {drop} past it dropped" if drop else "")
              + (f", {torn} torn line(s) dropped" if torn else "") + "; one line per training example from here on", flush=True)

    def draw(self, e, unit, order, cursor):
        gold = int(np.argmax(np.asarray(unit["target"], dtype=np.float64)))
        self.meta[id(e)] = (e, self.i, {"epoch": cursor // self.n_pass if self.n_pass else None, "uid": self.uids[id(unit)],
                                        "src": unit.get("source"), "k": len(order), "gold": gold}, order.index(gold))
        self.i += 1

    def fwd(self, chunk, logp):
        self.fwds.append([chunk, logp.detach(), None])

    def head(self, rows, hrow):
        self.fwds[-1][2] = (list(rows), hrow.detach())

    def ready(self):
        parts = [t.reshape(-1).float() for _, lp, h in self.fwds for t in ((lp,) if h is None else (lp, h[1]))]
        self.pend = Later(torch.cat(parts)) if parts else None
        self.fwds = [(chunk, tuple(lp.shape), h and h[0]) for chunk, lp, h in self.fwds]   # the device tensors are not needed now

    def write(self, step):
        vals, off, recs = torch.tensor(self.pend.get() if self.pend is not None else [], dtype=torch.float32), 0, []
        for chunk, (B, K), hr in self.fwds:
            logp = vals[off: off + B * K].view(B, K)
            off += B * K
            tgt = torch.zeros(B, K)
            for b, (_, t) in enumerate(chunk):
                tgt[b, : len(t)] = torch.tensor(t, dtype=torch.float32)
            rows = dyn_loss_rows(logp, tgt, torch.tensor([len(t) for _, t in chunk]), chunk, self.rps, self.brier)
            if hr:                                # (p95) the numeric head's rows of this forward
                h = vals[off: off + len(hr)]
                off += len(hr)
                rows[hr] = h if self.head_letters == "off" else rows[hr] + h
            p, rows = logp.exp(), rows.tolist()
            for b, e in enumerate(chunk):
                _, i, rec, gl = self.meta[id(e)]
                pb, tb, lb = p[b, : len(e[1])], tgt[b, : len(e[1])], logp[b, : len(e[1])]
                # (10-08, review LATE P2) distribution-aware fit to the SOFT target, from the normalised log-probs (no underflow):
                # h_tgt = H(target), ce_tgt = soft cross-entropy, kl_tgt = KL(target || prediction) = ce - H, l2_tgt = ||target - p||^2.
                # A correctly learned high-entropy posterior has kl_tgt ~ 0 however low p_gold is. Independent of the training loss.
                tn = tb / tb.sum() if float(tb.sum()) > 0 else tb
                nz = tn > 0
                h_t = float(-(tn[nz] * tn[nz].log()).sum())
                ce_t = float(-(tn[nz] * lb[nz]).sum())
                recs.append((i, {"step": step, "epoch": rec["epoch"], "i": i, "uid": rec["uid"], "src": rec["src"], "k": rec["k"],
                                 "gold": rec["gold"], "p_gold": _sig6(float(pb[gl])), "p_tgt": _sig6(float((tb * pb).sum())),
                                 "h_tgt": _sig6(h_t), "ce_tgt": _sig6(ce_t), "kl_tgt": _sig6(ce_t - h_t),
                                 "l2_tgt": _sig6(float(((tn - pb) ** 2).sum())),
                                 "loss": _sig6(rows[b]), "correct": int(pb.argmax()) == gl}))
        self.meta, self.fwds, self.pend, self.i = {}, [], None, 0
        try:
            self.f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for _, r in sorted(recs, key=lambda x: x[0])))
            self.f.flush()
        except OSError as e:                      # logging only: a full disk must not stop the run
            if not self.failed:
                print(f"dyn-log: WRITE FAILED at step {step} ({e!r}); training goes on, this step's lines are lost (further failures "
                      f"silent)"[:400], flush=True)
            self.failed = True

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


# --------------------------------------------------------------------------- (grok1) eval log (2026-10-09)
# --eval-log PATH (opt-in, LOG ONLY, resume-free; one process): everything a grokking read needs from the val evals, nothing that
# changes them. Every process start appends a HEADER line {"step": start, "kind": "header", "n", "uids", "src", "h", "val_fp"} (the val
# examples in eval order: rowid uids, sources, target entropies), then one EVAL line per val eval {"step", "kind": "eval", "lr", "val":
# {n, ce, acc, brier, h, kl}, "lora": {modules, a2, b2, dw2, dw}, "unit": {ce, brier, correct, p}}. "ce" / "acc" are the trainer's own
# printed val numbers; the per-unit CE is the same soft CE row by row (soft_ce's formula), Brier = brier_rows', correct = the val acc
# rule, p = the predicted distribution over the unit's options in stored order (val prompts are never permuted). LoRA norms: sum
# over adapters of ||A||_F^2, ||B||_F^2 and ||scale B A||_F^2 (the effective update; via the r x r Grams, fp64).
# "lr" = the LR the next update uses. A resumed run first drops the lines past its resumed step (DynLog's trim); a redone eval at the
# resumed step repeats that step's line (a reader keeps the last).
def eval_unit_rows(chunk, logp, tgt, ks):
    """(grok1) {id(example): (soft CE, Brier, correct, [p over the example's options])} for one val chunk, from the eval's own readout
    (soft_ce / brier_rows formulas; p in the example's option order = the stored order, val prompts are never permuted)."""
    lp = logp.detach().float()
    t = tgt.detach().float()
    mask = torch.arange(lp.shape[1], device=lp.device)[None, :] < ks[:, None]
    ce = -(t * lp.masked_fill(~mask, 0.0)).sum(-1)
    pr = lp.exp()
    br = ((pr - t) ** 2).masked_fill(~mask, 0.0).sum(-1)
    cor = (logp.argmax(-1) == tgt.argmax(-1))
    rows = torch.cat([torch.stack([ce.double(), br.double(), cor.double()], 1), pr.double()], 1).cpu().tolist()
    return {id(e): (r[0], r[1], int(r[2]), r[3: 3 + len(e[1])]) for e, r in zip(chunk, rows)}


def lora_norms(model, lora_cls):
    """(grok1) {modules, a2, b2, dw2, dw}: sums over every `lora_cls` adapter of ||A||_F^2, ||B||_F^2 and ||scale B A||_F^2
    (= scale^2 sum((B^T B) * (A A^T)), r x r Grams, fp64); dw = sqrt(dw2)."""
    n, a2, b2, dw2 = 0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, lora_cls):
                A, B = m.A.detach().double(), m.B.detach().double()
                a2 += float((A * A).sum())
                b2 += float((B * B).sum())
                dw2 += float(m.scaling) ** 2 * float(((B.T @ B) * (A @ A.T)).sum())
                n += 1
    return {"modules": n, "a2": a2, "b2": b2, "dw2": dw2, "dw": math.sqrt(max(dw2, 0.0))}


class EvalLog:
    """(grok1) --eval-log (see the section comment). open(step, header) trims past lines and appends this start's header;
    write(step, lr, v, rows, norms) appends one eval line (rows: per val example, in header order)."""

    def __init__(self, path):
        self.path, self.f, self.failed = path, None, False

    def open(self, step, header):
        kept = drop = torn = 0
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                for ln in f:
                    s = dyn_line_step(ln)
                    kept += s is not None and s <= step
                    drop += s is not None and s > step
                    torn += s is None
            if drop or torn:
                tmp = f"{self.path}.trim.tmp"
                with open(self.path, "rb") as f, open(tmp, "wb") as g:
                    for ln in f:
                        s = dyn_line_step(ln)
                        if s is not None and s <= step:
                            g.write(ln)
                    g.flush()
                    os.fsync(g.fileno())
                os.replace(tmp, self.path)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")
        self._put({"step": step, "kind": "header", **header})
        print(f"eval-log: {self.path}: {kept} lines of steps <= {step} kept" + (f", {drop} past it dropped" if drop else "")
              + (f", {torn} torn line(s) dropped" if torn else "") + f"; {header['n']} val units per eval", flush=True)

    def _put(self, rec):
        try:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()
        except OSError as e:                      # logging only: a full disk must not stop the run
            if not self.failed:
                print(f"eval-log: WRITE FAILED at step {rec.get('step')} ({e!r}); training goes on (further failures silent)"[:400],
                      flush=True)
            self.failed = True

    def write(self, step, lr, v, rows, h, norms):
        n = len(rows)
        brier = sum(r[1] for r in rows) / max(n, 1)
        hm = sum(h) / max(len(h), 1)
        self._put({"step": step, "kind": "eval", "lr": lr,
                   "val": {"n": n, "ce": v[0], "acc": v[1], "brier": brier, "h": hm, "kl": v[0] - hm},
                   "lora": norms,
                   "unit": {"ce": [_sig6(r[0]) for r in rows], "brier": [_sig6(r[1]) for r in rows], "correct": [r[2] for r in rows],
                            "p": [[_sig6(x) for x in r[3]] for r in rows]}})
        print(f"  [eval-log] step {step}  val Brier {brier:.4f}  KL {v[0] - hm:.4f} (H {hm:.4f})  LoRA |A|^2 {norms['a2']:.4g}  "
              f"|B|^2 {norms['b2']:.4g}  |dW| {norms['dw']:.4g}  lr {lr:.4g}", flush=True)

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


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
                    "dp_stop", "h2d", "tok_cache", "gc_every", "gc_budget",
                    # (p103) paths: the val sets' and the reference's CONTENT is in the identity instead (mixture_fp)
                    "mixture_val", "mixture_ref")
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
                         "aux_vbll": False, **AUX_VBLL_DEFAULTS,
                         # (p95, 2026-10-04) the numeric head: likewise (sub-options refused without --num-head)
                         "num_head": "off", **NUM_HEAD_DEFAULTS,
                         # (p101, 2026-10-04) prior offsets on the letters
                         "prior_offset": "off",
                         # (p103, 2026-10-04) the learned data mixture (learned-only options refused under static)
                         **MIXTURE_DEFAULTS}
# (datavar, 2026-10-04) the hierarchical sampler: likewise (its sub-options are refused without --sampling hier); the capability map's
# PATH is resume-free like --mixture-val's (its content enters the identity as sampling_fp)
IDENT_OMIT_AT_DEFAULT.update(SAMPLING_DEFAULTS)
RESUME_FREE_ARGS += ("sampling_capability_map",)
# (speed, 2026-10-06) --stop-after-step only moves where a run pauses (the LR schedule is --steps'), --phase-log / --phase-sync only
# log / wait: resume-free, so a stopped run continues without them. --letter-head selected rounds the letter logits differently: in
# the identity when set (every earlier identity unchanged).
RESUME_FREE_ARGS += ("stop_after_step", "phase_log", "phase_sync")
IDENT_OMIT_AT_DEFAULT["letter_head"] = "full"
# (ss, 2026-10-08, review AM P2) the LoRA scale rule: in the identity when set (a run under another adapter multiplier is another run);
# every identity at the default (alpha / r) is unchanged
IDENT_OMIT_AT_DEFAULT.update({"lora_scale_rule": "alpha_over_r", "lora_scale_anchor": 8.0})
# (speed, 2026-10-06) --pinned-staging moves only WHEN / HOW the same input values reach the device (and when the host draws a step's
# examples relative to its forwards, never past the step): resume-free, a run may switch it on resume.
RESUME_FREE_ARGS += ("pinned_staging",)

# (pretok, 2026-10-06) --token-cache PATH: the same prompts, token ids and example stream from cached component ids: resume-free.
RESUME_FREE_ARGS += ("token_cache",)
# (ss, 2026-10-07) --dyn-log PATH only logs what the loss already computed: resume-free (every identity unchanged; a resumed run may
# switch it on or off).
RESUME_FREE_ARGS += ("dyn_log",)
# (grok1, 2026-10-09) the LoRA optimizer's weight decay and the LR schedule shape: in the identity when set (another run); every
# identity at the defaults (0.0 / cosine) is unchanged. --eval-log PATH only logs what the val eval computed: resume-free.
IDENT_OMIT_AT_DEFAULT.update({"weight_decay": 0.0, "lr_schedule": "cosine"})
RESUME_FREE_ARGS += ("eval_log",)


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

    def reduce_num(self, params, bucket_bytes=64 << 20):
        """(collective; p95) SUM the numeric head's gradients over the ranks, in buckets of their own AFTER the LoRA's (and the aux
        head's). The trainer gives every head tensor a gradient on every rank first (zeros where a rank's share had no numeric row)."""
        missing = sum(p.grad is None for p in params)
        if missing:
            raise RuntimeError(f"(ddp) --num-head: {missing} head tensors have no gradient on rank {self.rank}")
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
    ap.add_argument("--lora-scale-rule", choices=("alpha_over_r", "rs"), default="alpha_over_r",
                    help="(ss, 2026-10-08) the adapter multiplier: 'alpha_over_r' = 32 / r (the default; every earlier run); 'rs' = "
                         "--lora-scale-anchor / sqrt(r) (rank-stabilised LoRA, arXiv 2312.03732; anchor 8 = the default's r16 value: "
                         "r16 2, r64 1, r256 0.5). Saved in --out's config (lora_scale_rule / _anchor / lora_scale / lora_alpha); "
                         "claude_lmscore.py and the other loaders apply it; part of the resume identity when set")
    ap.add_argument("--lora-scale-anchor", type=float, default=8.0, help="(ss) with --lora-scale-rule rs: scale = anchor / sqrt(r)")
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
    ap.add_argument("--letter-head", choices=("full", "selected"), default="full",
                    help="(speed, 2026-10-06) fast path: 'full' (the default, unchanged) = the full-vocabulary LM head, then the "
                         "option letters; 'selected' = the same map from the 52 letter rows only (ls.FrozenLabelHead; gradients reach "
                         "the LoRA through h as before). Equal in exact arithmetic, bf16 rounding may differ: in the run identity when set")
    ap.add_argument("--stop-after-step", type=int, default=0,
                    help="(speed, 2026-10-06) a PREFIX of the --steps run: train steps 1..N with --steps' warmup / cosine schedule "
                         "unchanged, write --out and the resume state after step N, exit 0. A re-run without the flag (or with a "
                         "larger N) resumes the SAME run (resume-free: not in the identity). Needs --resume auto / PATH. 0 = off")
    ap.add_argument("--phase-log", type=int, default=0,
                    help="(speed, 2026-10-06) LOG ONLY: every N steps a PHASES line (host wall seconds per phase: draw / fwd / bwd / "
                         "comm / opt / log / eval / save / misc; real vs padded training tokens), a startup line (load / prep / resume / "
                         "initial eval) and totals at the end (PhaseClock). No device read, no sync. 0 = off (the default)")
    ap.add_argument("--phase-sync", action="store_true",
                    help="(speed, 2026-10-06) with --phase-log on CUDA: synchronise at COARSE boundaries only (window ends, around "
                         "startup / eval / checkpoint phases; 'gpu_wait') and add CUDA-event GPU seconds per phase. Same values")
    ap.add_argument("--pinned-staging", action="store_true",
                    help="(speed, 2026-10-06) training forwards' inputs (ids, mask / positions, targets, letter counts, ordinal "
                         "levels) through TWO "
                         "reusable pinned host slots + device slots, ONE H2D copy each on a dedicated copy stream, staged one forward "
                         "ahead WITHIN a step, ordered by CUDA events (PinnedStager; no global sync). The same draws in the same order "
                         "(never past the step), the same values: resume-free. Evaluation unchanged (--h2d). Without CUDA the same "
                         "loop runs with synchronous copies. Refused with CUDA-graph --compile-mode")
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
    ap.add_argument("--num-head", default="off",
                    help="(p95) 'mix' or 'mix:K' (K = 1..16 components; 'mix' = mix:4): also train a NUMERIC MIXTURE HEAD (NumHead) on the "
                         "decision-position final hidden state, JOINTLY with the LoRA (its gradient reaches the LoRA), on numeric units "
                         "only (records with meta.bin_edges / meta.axis: build_numeric_jev.py). It outputs a K-component mixture in the "
                         "axis's own coordinate (std / logit / log by the axis's folded ends); loss = CE of the exact bin masses under the "
                         "mixture's CDF differences over the same bins (+ --num-head-crps). In --out / resume as 'num_head'; score with "
                         "claude_lmscore.py --num-head. 'off' (the default) = unchanged")
    ap.add_argument("--num-head-family", choices=NUM_HEAD_FAMILIES, default=NUM_HEAD_DEFAULTS["num_head_family"],
                    help="(p95) mixture components: 'logistic' (the default: closed-form CDF, exponential tails -- the shape of a Beta "
                         "or Gamma posterior's tails in logit / log coordinates; Beta(1, 1) in logit is exactly one component) or "
                         "'gaussian'")
    ap.add_argument("--num-head-letters", choices=("on", "off"), default=NUM_HEAD_DEFAULTS["num_head_letters"],
                    help="(p95) 'off' (the default): a numeric unit trains the head INSTEAD of its letter readout (same per-example "
                         "weight: the readout is swapped, nothing else); 'on': the head's loss is ADDED to the letter loss (both "
                         "readouts trained; the numeric units then weigh double)")
    ap.add_argument("--num-head-crps", type=float, default=NUM_HEAD_DEFAULTS["num_head_crps"],
                    help="(p95) add this x the binned CRPS excess sum_k (F_mix(e_k) - G_k)^2 (closed-form mixture CDF at the interior "
                         "edges vs the exact posterior's; = --rps-weight's RPS in level order, for the head) to every numeric row; 0 = off")
    ap.add_argument("--num-head-lr", type=float, default=NUM_HEAD_DEFAULTS["num_head_lr"],
                    help="(p95) the head's own AdamW lr (the trainer's warmup + cosine shape; its own clip at 1.0)")
    ap.add_argument("--num-head-dim", type=int, default=NUM_HEAD_DEFAULTS["num_head_dim"],
                    help="(p95) the head's hidden width (h -> dim -> 3K)")
    ap.add_argument("--prior-offset", default="off",
                    help="(p101) 'meta' or 'meta:KEY': for every unit whose record has meta.log_prior_bins (or meta.KEY: one log prior mass "
                         "per option, stored option order), train the option letters as EVIDENCE: q = softmax(log pi + letter logits), "
                         "the offsets mapped through the drawn option order and added before the softmax / loss (soft CE + RPS as set). "
                         "Other units untouched. Score with claude_lmscore.py --prior-offset (test_evidence_logits: apply any prior "
                         "offline, analysis/apply_prior.py). 'off' (the default) = unchanged")
    ap.add_argument("--mixture", choices=("static", "learned"), default="static",
                    help="(p103) 'static' (the default) = today's sampler: every train unit in one seeded shuffle read by one cursor. "
                         "'learned' = per-group streams (--mixture-groups) drawn by weights that an exponentiated-gradient step updates "
                         "every --mixture-every steps from the held-out per-group excess loss (--mixture-val; DoReMi-style, online; "
                         "research/p103_learned_mixture.md has the formula). With --mixture-val, 'static' only MONITORS the val sets")
    ap.add_argument("--mixture-groups", default=MIXTURE_DEFAULTS["mixture_groups"],
                    help="(p103, learned) NAME=PATTERN,...: file-name patterns under train/ (fnmatch on the stem or the file name); a file "
                         "goes to the FIRST group (spec order) one of whose patterns matches; a NAME may repeat to add patterns. E.g. "
                         "'numeric=num_*,replay=*'")
    ap.add_argument("--mixture-val", default="",
                    help="(p103) NAME[@GROUP]=PATH,...: held-out val sets (PATH a directory of *.jsonl, a jsonl file or a glob), each "
                         "attached to a train group (@GROUP; default the group named NAME). Scored at step 0, every --mixture-every "
                         "steps and at the last step with the main readout's per-unit soft CE")
    ap.add_argument("--mixture-val-units", type=int, default=MIXTURE_DEFAULTS["mixture_val_units"],
                    help="(p103) at most this many units per val set (a fixed sample, the same for every seed)")
    ap.add_argument("--mixture-every", type=int, default=MIXTURE_DEFAULTS["mixture_every"],
                    help="(p103) val-set scoring / weight-update cadence in steps")
    ap.add_argument("--mixture-ref", default="initial",
                    help="(p103, learned) the reference R_v of each val set: 'initial' (R_v = the labels' entropy; with --mixture-norm rel "
                         "e_v = KL_v(t) / KL_v(0), the share of v's step-0 excess not yet removed) or 'file:PATH' (a reference run's "
                         "losses: a JSON {name: loss} or a claude_lmtune checkpoint with a mixture history -- its last losses; DoReMi's "
                         "reference model)")
    ap.add_argument("--mixture-norm", choices=("rel", "abs"), default=MIXTURE_DEFAULTS["mixture_norm"],
                    help="(p103, learned) excess units: 'rel' (the default) divides L - R by v's step-0 excess max(L(0) - H, "
                         "--mixture-kl-floor), H = the labels' entropy; 'abs' = nats")
    ap.add_argument("--mixture-kl-floor", type=float, default=MIXTURE_DEFAULTS["mixture_kl_floor"],
                    help="(p103, learned) floor of the 'rel' denominator (nats)")
    ap.add_argument("--mixture-eta", type=float, default=MIXTURE_DEFAULTS["mixture_eta"],
                    help="(p103, learned) exponentiated-gradient step size: w_g <- w_g exp(eta s_g), normalised")
    ap.add_argument("--mixture-eps", type=float, default=MIXTURE_DEFAULTS["mixture_eps"],
                    help="(p103, learned) smoothing after each update: w <- (1 - eps) w + eps p (0 <= eps < 1)")
    ap.add_argument("--mixture-smooth", choices=("uniform", "static"), default=MIXTURE_DEFAULTS["mixture_smooth"],
                    help="(p103, learned) the smoothing target p: uniform over groups, or the static unit proportions")
    ap.add_argument("--mixture-clip", default=MIXTURE_DEFAULTS["mixture_clip"],
                    help="(p103, learned) clip of each val set's excess before the utility: 'LO:HI' (default 0:2: DoReMi's max(e, 0), "
                         "and a cap at a doubled excess) or 'C' -> [-C, C]")
    ap.add_argument("--mixture-init", default=MIXTURE_DEFAULTS["mixture_init"],
                    help="(p103, learned) initial weights: 'static' (unit proportions), 'uniform', 'w1,w2,...' (group order) or "
                         "'g=w,...'; normalised")
    ap.add_argument("--mixture-min", default=MIXTURE_DEFAULTS["mixture_min"],
                    help="(p103, learned) per-group minimum share, 'g=x,...' (default 0)")
    ap.add_argument("--mixture-max", default=MIXTURE_DEFAULTS["mixture_max"],
                    help="(p103, learned) per-group maximum share, 'g=x,...' (default 1)")
    ap.add_argument("--mixture-utility", default=MIXTURE_DEFAULTS["mixture_utility"],
                    help="(p103, learned) per-VAL-SET utility multipliers 'v=1.0,...' (default 1): s_g = mean over g's val sets of "
                         "u_v x clip(e_v); u_v > 1 = v's remaining excess counts more (more weight to its group while it has some)")
    ap.add_argument("--sampling", choices=("pooled", "hier"), default=SAMPLING_DEFAULTS["sampling"],
                    help="(datavar) 'pooled' (the default) = today's sampler (one seeded shuffle of every train unit, or --mixture "
                         "learned's group streams). 'hier' = every example slot draws [capability ->] source -> case -> question: a "
                         "source by --sampling-source-weights, that source's next case (a fresh seeded permutation of its cases per "
                         "pass), one question of that case (without replacement within the case)")
    ap.add_argument("--sampling-sources", default=SAMPLING_DEFAULTS["sampling_sources"],
                    help="(datavar, hier) '' (the default) = one source per train file; else NAME=PATTERN,... (as --mixture-groups: "
                         "fnmatch on the file stem or name, the first matching NAME takes a file, a NAME may repeat; one source allowed)")
    ap.add_argument("--sampling-case", choices=("base", "record"), default=SAMPLING_DEFAULTS["sampling_case"],
                    help="(datavar, hier) the case key inside a source: the unit's base_id (the default: a record and its option-"
                         "reshuffled variants) or its record_id")
    ap.add_argument("--sampling-source-weights", default=SAMPLING_DEFAULTS["sampling_source_weights"],
                    help="(datavar, hier) 'uniform' (the default), 'cases' (proportional to a source's number of cases), 'units' (to its "
                         "questions: the pooled source shares), 'cases:T' / 'units:T' (to n^T), or NAME=W,... (every source, or '*=W')")
    ap.add_argument("--sampling-capability-map", default="",
                    help="(datavar, hier) a JSON file {source: capability tag} ('*' = every other source): a slot draws a tag first "
                         "(--sampling-capability-weights), then a source by the source weights renormalised within that tag")
    ap.add_argument("--sampling-capability-weights", default=SAMPLING_DEFAULTS["sampling_capability_weights"],
                    help="(datavar, hier + map) 'uniform' (the default), 'sources' (a tag's weight = the sum of its sources' weights: "
                         "the source marginals of no map) or TAG=W,... (every tag, or '*=W')")
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
    ap.add_argument("--token-cache", default="",
                    help="(pretok, 2026-10-06) a cache from trainers/token_cache.py build (same --model / --max-len / --plain-prompt / "
                         "--noul-defs, else refused): prompt token ids assembled from cached components, verified per unit at build "
                         "(unverified units fall back). The same prompts, ids, example stream and loss; resume-free. '' = off")
    ap.add_argument("--dyn-log", default=None,
                    help="(ss, 2026-10-07) LOG ONLY: one JSON line per TRAINING example forwarded, {step, epoch, i, uid, src, k, gold, "
                         "p_gold, p_tgt, loss, correct} (the 'dynamics log' section; data maps: analysis/data_map.py): uid = the scorer's "
                         "rowid uid, gold = the target's argmax in STORED option order, p_* under the loss's own readout, loss = the "
                         "example's own term. One device->host copy per step, flushed every step; a resumed run first drops the records "
                         "past its resumed step; data parallel: PATH.rank<r> per rank. Resume-free. Default: off (unchanged)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="(grok1, 2026-10-09) decoupled AdamW weight decay on the LoRA parameters (p <- p (1 - lr wd) per step; the "
                         "aux-vbll / num-head heads keep their own optimizers at 0). Saved in --out's config and part of the resume "
                         "identity when set. 0.0 (the default) = unchanged")
    ap.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine",
                    help="(grok1, 2026-10-09) the LoRA optimizer's LR shape: 'cosine' (the default, unchanged) = linear --warmup x "
                         "cosine over --steps; 'constant' = linear --warmup, then the peak --lr flat to --steps. Saved in --out's config "
                         "and part of the resume identity when set; not with --aux-vbll / --num-head (their optimizers follow the cosine)")
    ap.add_argument("--eval-log", default=None,
                    help="(grok1, 2026-10-09) LOG ONLY: at every val eval one JSON line (val soft-CE / acc / Brier / KL, per-unit CE / "
                         "Brier / correct, LoRA norms sum ||A||^2 / ||B||^2 / ||scale B A||^2, LR) after a header line per process start "
                         "(the 'eval log' section). A resumed run first drops the lines past its resumed step. One process only (not "
                         "under data parallelism). Resume-free. Default: off (unchanged)")
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
    if a.lora_scale_rule == "alpha_over_r" and a.lora_scale_anchor != 8.0:
        raise SystemExit("--lora-scale-anchor applies to --lora-scale-rule rs only")
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
    if a.letter_head != "full" and a.exact_path:
        raise SystemExit("--letter-head selected applies to the fast path only (the exact path reads the model's own head)")
    if a.stop_after_step < 0 or (a.stop_after_step and a.resume == "off"):
        raise SystemExit("--stop-after-step N: N >= 1, with resume state on (--resume auto / PATH), so the run can continue")
    if a.phase_log < 0 or (a.phase_sync and not a.phase_log):
        raise SystemExit("--phase-log N >= 0; --phase-sync applies with --phase-log")
    if a.dyn_log and (a.dyn_log.endswith(".gz") or not hasattr(ls, "unit_uids")):   # (ss) a plain appended file; rowid-1 uids
        raise SystemExit("--dyn-log: a plain-text path (lines are appended and flushed every step: not .gz), and the (rowid) "
                         "claude_lmscore.py with unit_uids beside this file")
    if not (math.isfinite(a.weight_decay) and a.weight_decay >= 0):       # (grok1)
        raise SystemExit("--weight-decay: a finite value >= 0")
    if a.lr_schedule != "cosine" and (a.aux_vbll or a.num_head != "off"):
        raise SystemExit("--lr-schedule constant: not with --aux-vbll / --num-head (their optimizers follow the warmup x cosine shape)")
    if a.eval_log and (a.eval_log.endswith(".gz") or dp.on or not hasattr(ls, "unit_uids")):
        raise SystemExit("--eval-log: a plain-text path (not .gz), one process (not under data parallelism), and the (rowid) "
                         "claude_lmscore.py with unit_uids beside this file")
    stop_at = a.stop_after_step if a.stop_after_step < a.steps else 0   # (speed) N >= --steps: the whole run, as without it
    clock = PhaseClock(a.phase_log, a.phase_sync and torch.device(a.device).type == "cuda")   # (speed) off: every call returns
    aux_heads = None                            # (aux-vbll) off: none of it runs
    try:
        _grid = aux_vbll_grid(a)                # canonical lists on `a`, so the identity sees one spelling per value
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if a.aux_vbll:
        aux_heads = _grid
    elif any(getattr(a, k) != v for k, v in AUX_VBLL_DEFAULTS.items()):
        raise SystemExit("--aux-vbll-kl / -prior / -cov / -lr / -dim / -samples apply with --aux-vbll")
    try:                                        # (p95) canonical spelling ('mix' -> 'mix:4'), so the identity sees one per value
        nh_K = num_head_k(a.num_head)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    a.num_head = f"mix:{nh_K}" if nh_K else "off"
    if not nh_K and any(getattr(a, k) != v for k, v in NUM_HEAD_DEFAULTS.items()):
        raise SystemExit("--num-head-family / -letters / -crps / -lr / -dim apply with --num-head")
    if nh_K:
        if a.aux_vbll:
            raise SystemExit("--num-head with --aux-vbll: not supported (the aux head reads DETACHED decision states, the numeric head "
                             "trains through them); run them separately")
        if a.ordinal_readout != "none":
            raise SystemExit("--num-head with --ordinal-readout: not supported (the head replaces the letter readout on numeric units; "
                             "an ordinal map of the letters there is a different experiment)")
        if not (a.num_head_crps >= 0 and math.isfinite(a.num_head_crps)) or not a.num_head_lr > 0 or a.num_head_dim < 1:
            raise SystemExit("--num-head-crps >= 0, --num-head-lr > 0, --num-head-dim >= 1")
    try:                                        # (p101) canonical spelling ('meta:log_prior_bins' -> 'meta')
        po_key = prior_offset_key(a.prior_offset)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    a.prior_offset = "off" if po_key is None else ("meta" if po_key == "log_prior_bins" else f"meta:{po_key}")
    if po_key is not None:
        for bad, why in ((nh_K, "--num-head (the head replaces the letters on numeric units)"),
                         (a.aux_vbll, "--aux-vbll (the aux head reads the raw letter logits)"),
                         (a.ordinal_readout != "none", "--ordinal-readout (an ordinal map of evidence logits is another model)")):
            if bad:
                raise SystemExit(f"--prior-offset with {why}: not supported")
    mix_learned = a.mixture == "learned"        # (p103) static without --mixture-val: none of the mixture code runs
    mix_on = mix_learned or bool(a.mixture_val)
    mix_groups = mix_vspec = mix_clip = None
    if not mix_learned and (any(getattr(a, k) != MIXTURE_DEFAULTS[k] for k in MIXTURE_LEARNED_ONLY) or a.mixture_ref != "initial"):
        raise SystemExit("--mixture-groups / -ref / -norm / -kl-floor / -eta / -eps / -smooth / -clip / -init / -min / -max / -utility "
                         "apply with --mixture learned")
    if not mix_on and (a.mixture_val_units != MIXTURE_DEFAULTS["mixture_val_units"]
                       or a.mixture_every != MIXTURE_DEFAULTS["mixture_every"]):
        raise SystemExit("--mixture-val-units / --mixture-every apply with --mixture-val")
    if mix_on:
        if mix_learned and not (a.mixture_groups and a.mixture_val):
            raise SystemExit("--mixture learned needs --mixture-groups and --mixture-val")
        try:
            mix_vspec = mixture_val_spec(a.mixture_val)
            if mix_learned:
                mix_groups = mixture_groups_spec(a.mixture_groups)
                gnames = [g for g, _ in mix_groups]
                for v in mix_vspec:
                    if v["group"] not in gnames:
                        raise ValueError(f"--mixture-val: val set {v['name']!r} is attached to {v['group']!r}, not a group of "
                                         f"--mixture-groups {gnames} (use NAME@GROUP=PATH)")
                lo_, hi_ = mixture_clip_range(a.mixture_clip)
                a.mixture_clip = f"{hi_:g}" if lo_ == -hi_ else f"{lo_:g}:{hi_:g}"     # one spelling per value in the identity
                mix_clip = (lo_, hi_)
                mixture_named(a.mixture_utility, [v["name"] for v in mix_vspec], "--mixture-utility")
                mn_b = mixture_named(a.mixture_min, gnames, "--mixture-min", 0.0, 1.0)
                mx_b = mixture_named(a.mixture_max, gnames, "--mixture-max", 0.0, 1.0)
                if any(mn_b.get(g, 0.0) > mx_b.get(g, 1.0) for g in gnames) or sum(mn_b.values()) > 1.0 \
                        or sum(mx_b.get(g, 1.0) for g in gnames) < 1.0:
                    raise ValueError("--mixture-min / --mixture-max: need min <= max per group, sum of minima <= 1 <= sum of maxima")
                if a.mixture_init not in ("static", "uniform"):
                    mixture_init_weights(a.mixture_init, gnames, [1.0] * len(gnames))
        except ValueError as e:
            raise SystemExit(str(e)) from None
        if mix_learned:
            if not (math.isfinite(a.mixture_eta) and a.mixture_eta > 0) or not 0.0 <= a.mixture_eps < 1.0 \
                    or not (math.isfinite(a.mixture_kl_floor) and a.mixture_kl_floor > 0):
                raise SystemExit("--mixture-eta > 0, 0 <= --mixture-eps < 1, --mixture-kl-floor > 0")
            if a.mixture_ref != "initial" and not (a.mixture_ref.startswith("file:") and os.path.isfile(a.mixture_ref[5:])):
                raise SystemExit(f"--mixture-ref: 'initial' or 'file:PATH' with an existing file, got {a.mixture_ref!r}")
        if a.mixture_every < 1 or a.mixture_val_units < 1:
            raise SystemExit("--mixture-every >= 1 and --mixture-val-units >= 1")
    hier = a.sampling == "hier"                 # (datavar) pooled: none of the hierarchical sampler's code runs
    hs_sources = hs_cmap = None
    if not hier and (any(getattr(a, k) != v for k, v in SAMPLING_DEFAULTS.items() if k != "sampling")
                     or a.sampling_capability_map):
        raise SystemExit("--sampling-sources / -case / -source-weights / -capability-map / -capability-weights apply with "
                         "--sampling hier")
    if hier:
        if mix_learned:
            raise SystemExit("--sampling hier with --mixture learned: two samplers for one draw (--mixture-val alone, the static "
                             "monitor, coexists)")
        if a.sampling_capability_weights != SAMPLING_DEFAULTS["sampling_capability_weights"] and not a.sampling_capability_map:
            raise SystemExit("--sampling-capability-weights applies with --sampling-capability-map")
        try:                                    # syntax now; source / tag names once the train files are known
            hs_sources = sampling_sources_spec(a.sampling_sources)
            a.sampling_source_weights = hier_spec_canon(a.sampling_source_weights, ("uniform", "cases", "units"),
                                                        "--sampling-source-weights")
            a.sampling_capability_weights = hier_spec_canon(a.sampling_capability_weights, ("uniform", "sources"),
                                                            "--sampling-capability-weights")
            if a.sampling_capability_map:
                hs_cmap = hier_capability_map_load(a.sampling_capability_map)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        except OSError as e:
            raise SystemExit(f"--sampling-capability-map {a.sampling_capability_map}: {e.strerror or e}") from None
    fold = dp.on and a.dp_stop == "fold"        # (speed A) only under data parallel
    cudagraphs = a.compile_layers != "off" and a.compile_mode in CUDAGRAPH_MODES
    if a.pinned_staging and cudagraphs:
        raise SystemExit(f"--pinned-staging with --compile-mode {a.compile_mode}: refused. Inductor's CUDA graphs (cudagraph trees) "
                         f"manage their own input / output memory and replay outside the eager stream order the staging events are "
                         f"built for; that combination is untested, so it is not offered (use --compile-mode default or drop "
                         f"--pinned-staging)")
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
    if a.token_cache:                         # (pretok) the cache's key checked here (refused before the model loads)
        tok = _load("token_cache", os.path.join(HERE, "token_cache.py")).attach(
            a.token_cache, tok, max_len=a.max_len, plain=a.plain_prompt, noul_defs=a.noul_defs, chat=chat, ls=ls,
            whole_case=a.whole_case)
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
    # (ss 10-08) an explicit adapter multiplier only under a non-default scale rule; the default call is the old one
    if a.lora_scale_rule != "alpha_over_r" and not hasattr(te, "lora_scaling"):
        raise SystemExit(f"--lora-scale-rule {a.lora_scale_rule}: {te.__file__} predates scale rules (stage train_e2e.py with the trainer)")
    lora_scale = None if a.lora_scale_rule == "alpha_over_r" else te.lora_scaling(a.lora_r, a.lora_scale_rule, a.lora_scale_anchor)
    te.apply_lora(model, a.lora_r, targets=targets, **({"scope": scope} if scope else {}),
                  **({"scale": lora_scale} if lora_scale is not None else {}))
    if a.init_lora:                           # refuse a partial load: a half-initialised adapter trains a different model
        ck0 = torch.load(a.init_lora, map_location="cpu")
        if int(ck0["config"]["lora_r"]) != a.lora_r or list(ck0["config"]["lora_targets"]) != list(targets) \
                or bool(ck0["config"].get("plain_prompt", False)) != bool(a.plain_prompt) \
                or ck0["config"].get("lora_scope") != scope \
                or te.lora_scale_from_config(ck0["config"]) != lora_scale:   # (ss 10-08) the same adapter multiplier
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
    # (grok1) --weight-decay: AdamW's decoupled decay on these (LoRA) parameters; 0.0 = the old call. --lr-schedule constant: warmup, flat
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
    if a.lr_schedule == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    if a.weight_decay or a.lr_schedule != "cosine":
        print(f"optimizer (grok1): AdamW weight_decay {a.weight_decay:g} on the {len(params)} LoRA tensors (decoupled); LR schedule "
              f"{a.lr_schedule} (warmup {a.warmup})", flush=True)
    lsel = None                              # (speed) --letter-head selected: the 52 letter rows of the frozen head, cached once
    if a.letter_head == "selected":
        try:
            lsel = ls.FrozenLabelHead(lm_head, label_ids, lm_head.logit_mult, lm_head.softcap)
        except (AttributeError, TypeError, ValueError) as e:     # AttributeError: a claude_lmscore.py without FrozenLabelHead
            raise SystemExit(f"--letter-head selected: {e!r}") from None
        print(f"letter head: selected ({lsel.weight.shape[0]} rows of {lm_head.weight.shape[0]}, {lsel.weight.dtype}; multiplier "
              f"{lsel.multiplier}, softcap {lsel.softcap})", flush=True)
    stager = None                             # (speed) --pinned-staging: off = the training loop below exactly as before
    if a.pinned_staging:
        stager = PinnedStager(staging_backend(a.device))
        if stager.be.sync:
            print(f"pinned-staging: --device {a.device} has no CUDA streams: the staged loop runs with SYNCHRONOUS slot copies (no "
                  f"pinned memory, no copy stream, no events); the same draws and values", flush=True)
        else:
            print(f"pinned-staging: {len(stager.slots)} reusable pinned host slots + device slots (capacity buckets from "
                  f"{stager.min_bytes >> 10} KiB), one copy stream, CUDA events; each training forward's inputs in ONE H2D copy, "
                  f"staged one forward ahead within a step; evaluation unchanged (--h2d {a.h2d})", flush=True)
    clock.lap("load", sync=True)

    mix_gid = mix_fmap = None                 # (p103) learned: the group of every train unit (by object), the file -> group map
    if mix_learned:                           # the same units, loaded group by group (files assigned by --mixture-groups)
        from pathlib import Path as _Path
        try:
            mix_fmap = mixture_assign_files([p.stem for p in sorted(_Path(a.data, "train").glob("*.jsonl"))], mix_groups)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        train, mix_gid = [], {}
        for gi in range(len(mix_groups)):
            us = list(cf.iter_units(a.data, ["train"], {s for s, g in mix_fmap.items() if g == gi}, None))
            mix_gid.update((id(u), gi) for u in us)
            train.extend(us)
    elif hier:                                # (datavar) the default `train` list, read file by file so each unit's source is known
        from pathlib import Path as _Path
        hs_stems = [p.stem for p in sorted(_Path(a.data, "train").glob("*.jsonl"))]   # iter_units' own file order
        try:
            hs_names, hs_pats, hs_fmap = sampling_assign_files(hs_stems, hs_sources)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        train, hs_sid, hs_loaded = [], {}, [0] * len(hs_names)
        for s_ in hs_stems:
            us = list(cf.iter_units(a.data, ["train"], {s_}, None))
            hs_sid.update((id(u), hs_fmap[s_]) for u in us)
            hs_loaded[hs_fmap[s_]] += len(us)
            train.extend(us)
    else:
        train = list(cf.iter_units(a.data, ["train"], set(), None))
    val = list(cf.iter_units(a.data, ["val"], set(), None))
    # (ss) --dyn-log: every train unit's rowid uid over the load order, before any filter (beside the units: a key in them would change
    # units_fingerprint); the log itself opens once the resumed step is known
    dyn, dyn_uids = None, ({id(u): x for u, x in zip(train, ls.unit_uids(train))} if a.dyn_log else None)
    # (grok1) --eval-log: likewise every val unit's rowid uid over its load order (the log opens once the resumed step is known)
    elog, ev_uids, ev_last = None, ({id(u): x for u, x in zip(val, ls.unit_uids(val))} if a.eval_log else None), None
    if not a.allow_nc:                     # user 2026-10-01: never TRAIN on non-commercial data (eval may use it)
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
    mix_units = None                          # (p103) learned: each group's units after the filters, in load order
    if mix_learned:
        mix_units = [[] for _ in mix_groups]
        for u in train:
            mix_units[mix_gid[id(u)]].append(u)
        empty = [g for (g, _), us in zip(mix_groups, mix_units) if not us]
        if empty:
            raise SystemExit(f"--mixture-groups: group(s) {empty} have no train units after the filters")
    hs = None                                 # (datavar) --sampling hier: the sampler (None: the pooled sampler draws)
    if hier:                                  # the hierarchy from the filtered units in load order (before the pooled shuffle)
        t_hs = time.time()
        ckey = "base_id" if a.sampling_case == "base" else "record_id"
        by = [{} for _ in hs_names]           # per source: case key -> its units (dicts keep first-appearance order)
        for u in train:
            k = u[ckey]
            by[hs_sid[id(u)]].setdefault(k if isinstance(k, (str, int)) else json.dumps(k, sort_keys=True, default=str), []).append(u)
        hs_files = [[s_ for s_ in hs_stems if hs_fmap[s_] == i] for i in range(len(hs_names))]
        keep = [i for i, d in enumerate(by) if d]
        if not keep:
            raise SystemExit("--sampling hier: no train units after the filters")
        # a source the NC / K filters emptied (e.g. ms_marco) is DROPPED: no weight, no draws; noted in the log and the config
        hs_dropped = [{"name": hs_names[i], "files": hs_files[i], "units_loaded": hs_loaded[i]} for i in range(len(by)) if not by[i]]
        hs_names, hs_pats, hs_files = [hs_names[i] for i in keep], [hs_pats[i] for i in keep], [hs_files[i] for i in keep]
        hs_cases = [list(by[i].values()) for i in keep]
        n_u, n_c = [sum(len(c) for c in cs) for cs in hs_cases], [len(cs) for cs in hs_cases]
        dnames = [x["name"] for x in hs_dropped]
        try:
            hs_w = hier_source_weights(a.sampling_source_weights, hs_names, n_u, n_c, dnames)
            hs_tags = hs_caps = hs_cw = None
            hs_unused = []
            if hs_cmap is not None:
                hs_tags, hs_unused = hier_capability_tags(hs_cmap, hs_names)
                hs_caps = list(dict.fromkeys(hs_tags))
                hs_cw = hier_capability_weights(a.sampling_capability_weights, hs_caps, hs_tags, hs_w,
                                                hier_dropped_tags(hs_cmap, dnames, hs_caps))
        except ValueError as e:
            raise SystemExit(str(e)) from None
        hs = HierSampler(hs_names, hs_cases, a.seed, hs_w, hs_tags, hs_caps, hs_cw)
        print(f"sampling hier: {len(hs_names)} source(s), {sum(n_u)} units in {sum(n_c)} cases (case = {ckey}); source weights "
              f"{a.sampling_source_weights}"
              + (f"; capability level: {len(hs_caps)} tag(s) from {a.sampling_capability_map}, weights "
                 f"{a.sampling_capability_weights}" + (f" ({len(hs_unused)} map entries name no source here)" if hs_unused else "")
                 if hs_caps else "; no capability level")
              + (f"; {len(hs_dropped)} source(s) dropped (no units after the filters)" if hs_dropped else ""), flush=True)
        for x in hs_dropped:
            print(f"  [sampling] NOTE: source {x['name']} ({'|'.join(x['files'])}) DROPPED: the NC / K filters removed all "
                  f"{x['units_loaded']} of its units; it gets no weight and no draws", flush=True)
        named = {k.partition("=")[0].strip() for k in str(a.sampling_source_weights).split(",") if "=" in k}
        if named & set(dnames):
            print(f"  [sampling] NOTE: --sampling-source-weights entries for dropped source(s) {sorted(named & set(dnames))} "
                  f"ignored", flush=True)
        if hs_caps:
            named = {k.partition("=")[0].strip() for k in str(a.sampling_capability_weights).split(",") if "=" in k}
            gone = sorted(named & set(hier_dropped_tags(hs_cmap, dnames, hs_caps)))
            if gone:
                print(f"  [sampling] NOTE: --sampling-capability-weights entries for tag(s) {gone} (only dropped sources) ignored",
                      flush=True)
        if hs_caps:
            for j, k in enumerate(hs_caps):
                print(f"  [sampling] capability {k}: weight {hs_cw[j]:.4f}, {len(hs.members[j])} source(s)", flush=True)
        for i, g in enumerate(hs_names):
            print(f"  [sampling] source {g} ({'|'.join(hs_pats[i])}: {len(hs_files[i])} file(s)): {n_u[i]} units, {n_c[i]} cases "
                  f"({n_u[i] / n_c[i]:.2f} q/case), weight {hs_w[i]:.4f}, share {hs.marg[i]:.4f}"
                  + (f" [{hs_tags[i]}]" if hs_caps else ""), flush=True)
        print(f"  [sampling] set up in {time.time() - t_hs:.1f}s", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(train)
    val = random.Random(a.seed + 1).sample(val, min(a.eval_units, len(val)))
    wc_val, wc_train = {}, {}                 # (53) which rendering each prompt got (the --max-len overflow paths)
    if a.tok_cache:                           # (speed E) every later tok(...) of one prompt goes through the memo
        tok = TokCache(tok, a.tok_cache)
    val_mk = [(u, make_example(tok, u, random.Random(0), a.max_len, False, a.plain_prompt, a.whole_case, a.noul_defs, wc_val,
                               with_levels=ordinal, chat=chat)) for u in val]
    val_ex = [e for _, e in val_mk if e]
    print(f"train units {len(train)}, val units {len(val_ex)}", flush=True)
    if a.whole_case:
        print(f"whole-case val prompts: {dict(sorted(wc_val.items()))}", flush=True)

    nh_index, nh_fp = None, ""                  # (p95) off: none of the numeric head's code runs
    if nh_K:
        t_nh = time.time()
        nh_index = numeric_index(a.data, ["train", "val"])

        def nh_spec(u, idx=None):
            s = (nh_index if idx is None else idx).get((u["record_id"], u["qid"]))     # (p103) idx: a mixture val set's index
            return s if s is not None and u["qtype"] == "score" and len(u["option_keys"]) == s["K"] else None

        def nh_wrap(e, u, idx=None):
            """(p95) the example, tagged with its unit's axis and level-order target (None: not a numeric unit)."""
            s = nh_spec(u, idx)
            if s is None:
                return NumExample(e, None)
            v = np.asarray(u["target"], dtype=np.float64)
            return NumExample(e, {**s, "t": v / max(v.sum(), 1e-12)})

        nh_tr_units = [u for u in train if nh_spec(u) is not None]
        if not nh_tr_units:
            raise SystemExit(f"--num-head: no numeric units (records with meta.bin_edges / meta.axis and a matching `score` question) "
                             f"in {a.data}/train")
        nh_fp = num_index_fingerprint(nh_index, [(u["record_id"], u["qid"]) for u in train + val])
        val_ex = [nh_wrap(e, u) for u, e in val_mk if e]
        tfc = collections.Counter(nh_spec(u)["tf"] for u in nh_tr_units)
        print(f"num-head: numeric train units {len(nh_tr_units)} of {len(train)} (coordinates {dict(sorted(tfc.items()))}), numeric val "
              f"prompts {sum(e.num is not None for e in val_ex)} of {len(val_ex)}; index of {len(nh_index)} numeric questions read in "
              f"{time.time() - t_nh:.1f}s", flush=True)

    po_index, po_fp = None, ""                  # (p101) off: none of the prior-offset code runs
    if po_key is not None:
        t_po = time.time()
        po_index = prior_offset_index(a.data, ["train", "val"], po_key)

        def po_wrap(e, u, order, idx=None):
            """(p101) the example, tagged with its unit's offsets in letter order (`order` None: the unit's own order)."""
            off = (po_index if idx is None else idx).get((u["record_id"], u["qid"]))     # (p103) idx: a mixture val set's index
            if off is None:
                return e
            if len(off) != len(u["option_keys"]):
                raise SystemExit(f"--prior-offset: {u['qid']}: {len(off)} offsets for {len(u['option_keys'])} options")
            return PriorExample(e, po_letters(off, list(range(len(off))) if order is None else order))

        po_tr = sum((u["record_id"], u["qid"]) in po_index for u in train)
        if not po_tr:
            raise SystemExit(f"--prior-offset {a.prior_offset}: no train unit's record has meta.{po_key} in {a.data}/train")
        val_ex = [po_wrap(e, u, None) for u, e in val_mk if e]
        po_fp = po_index_fingerprint(po_index, [(u["record_id"], u["qid"]) for u in train + val])
        print(f"prior-offset: meta.{po_key} on {po_tr} of {len(train)} train units, {sum(isinstance(e, PriorExample) for e in val_ex)} "
              f"of {len(val_ex)} val prompts; q = softmax(log pi + letters) on those; index of {len(po_index)} questions read in "
              f"{time.time() - t_po:.1f}s", flush=True)

    mix, mix_ex, mix_fp_parts = None, [], None  # (p103) off: none of the mixture code runs
    if mix_on:
        t_mx = time.time()
        _ncf = None if a.allow_nc else (getattr(cf, "is_noncommercial", None) or
                                        (lambda u: str(u.get("source") or "").split("/")[0].split("__")[0] in {"ms_marco"}))
        gnames = [g for g, _ in mix_groups] if mix_learned else []
        utils = mixture_named(a.mixture_utility, [v["name"] for v in mix_vspec], "--mixture-utility") if mix_learned else {}
        mval = []
        for vsp in mix_vspec:
            try:
                files = mixture_val_files(vsp["path"])
            except ValueError as e:
                raise SystemExit(str(e)) from None
            dirs = collections.OrderedDict()
            for f in files:
                dirs.setdefault(f.parent, set()).add(f.stem)
            us, nh_i, po_i = [], {}, {}
            for d, stems in dirs.items():
                us.extend(cf.iter_units(str(d.parent), [d.name], stems, None))
                if nh_K:                      # (p95) the val set's numeric axes
                    nh_i.update(numeric_index(str(d.parent), [d.name]))
                if po_key is not None:        # (p101) its log priors
                    po_i.update(prior_offset_index(str(d.parent), [d.name], po_key))
            n_raw = len(us)
            n_nc = 0
            if _ncf is not None:
                n_nc = sum(1 for u in us if _ncf(u))
                us = [u for u in us if not _ncf(u)]
            if a.whole_case:
                ls.attach_case_questions(us)
            us = [u for u in us if len(u["option_keys"]) <= len(ls.LABELS)]
            if len(us) > a.mixture_val_units:     # a fixed sample: the same units for every seed and arm
                us = random.Random(f"mixval|{vsp['name']}").sample(us, a.mixture_val_units)
            ex, keep = [], []
            for u in us:
                e = make_example(tok, u, random.Random(0), a.max_len, False, a.plain_prompt, a.whole_case, a.noul_defs, {},
                                 with_levels=ordinal, chat=chat)
                if not e:
                    continue
                if nh_K:
                    e = nh_wrap(e, u, nh_i)
                if po_key is not None:
                    e = po_wrap(e, u, None, po_i)
                ex.append(e)
                keep.append(u)
            if not ex:
                raise SystemExit(f"--mixture-val {vsp['name']}: no usable units at {vsp['path']} ({n_raw} read, {n_nc} NC dropped)")
            H = float(np.mean([label_entropy(e[1]) for e in ex]))
            mval.append({"name": vsp["name"], "group": gnames.index(vsp["group"]) if mix_learned else None,
                         "group_name": vsp["group"], "utility": float(utils.get(vsp["name"], 1.0)), "H": H, "n": len(ex),
                         "n_read": n_raw, "n_nc": n_nc, "files": len(files), "fp": units_fingerprint(keep)})
            mix_ex.append(ex)
        ref_d, ref_sha = None, ""
        if mix_learned and a.mixture_ref != "initial":
            try:
                ref_d, ref_sha, ref_fps = mixture_ref_losses(a.mixture_ref[5:])
            except ValueError as e:
                raise SystemExit(str(e)) from None
            missing = [v["name"] for v in mval if v["name"] not in ref_d]
            if missing:
                raise SystemExit(f"--mixture-ref {a.mixture_ref}: no reference loss for val set(s) {missing} (has {sorted(ref_d)})")
            other = [v["name"] for v in mval if ref_fps.get(v["name"]) not in (None, v["fp"])]
            if other:                         # the excess L - R is only meaningful on the SAME units
                raise SystemExit(f"--mixture-ref {a.mixture_ref}: val set(s) {other} were scored on OTHER units in the reference run "
                                 f"(units fingerprint {[ref_fps[k] for k in other]} vs this run's "
                                 f"{[v['fp'] for v in mval if v['name'] in other]}): same names, data, --max-len and --mixture-val-units")
            ref_d = {v["name"]: ref_d[v["name"]] for v in mval}
        if mix_learned:
            n_g = [len(x) for x in mix_units]
            p_static = [x / sum(n_g) for x in n_g]
            w0 = mixture_init_weights(a.mixture_init, gnames, p_static)
            mn = mixture_named(a.mixture_min, gnames, "--mixture-min", 0.0, 1.0)
            mxx = mixture_named(a.mixture_max, gnames, "--mixture-max", 0.0, 1.0)
            lo_b, hi_b = [mn.get(g, 0.0) for g in gnames], [mxx.get(g, 1.0) for g in gnames]
            w0 = mixture_project(w0, lo_b, hi_b)
            mix = Mixture(gnames, mix_units, a.seed, w0, mval, eta=a.mixture_eta, eps=a.mixture_eps,
                          p=p_static if a.mixture_smooth == "static" else [1.0 / len(gnames)] * len(gnames), clip=mix_clip,
                          lo=lo_b, hi=hi_b, norm=a.mixture_norm, kl_floor=a.mixture_kl_floor, ref=ref_d, learned=True)
            files_by = collections.Counter(mix_fmap.values())
            print("mixture learned: groups " + ", ".join(
                f"{g} ({'|'.join(pats)}: {files_by[gi]} files, {n_g[gi]} units, static share {p_static[gi]:.4f})"
                for gi, (g, pats) in enumerate(mix_groups))
                + f"; init {' / '.join(f'{x:.4f}' for x in w0)} ({a.mixture_init}); EG eta {a.mixture_eta:g}, smoothing eps "
                f"{a.mixture_eps:g} toward {a.mixture_smooth}, excess clip [{mix_clip[0]:g}, {mix_clip[1]:g}], norm {a.mixture_norm} "
                f"(kl floor {a.mixture_kl_floor:g}), reference "
                + ("initial (e = KL(t) / KL(0))" if ref_d is None and a.mixture_norm == "rel" else
                   "initial (e = KL(t), nats)" if ref_d is None else f"file {a.mixture_ref[5:]}")
                + (f", bounds min {lo_b} max {hi_b}" if a.mixture_min or a.mixture_max else "")
                + f"; update every {a.mixture_every} steps", flush=True)
        else:
            mix = Mixture([], None, a.seed, None, mval, learned=False)
            print(f"mixture static (MONITOR ONLY: the static sampler is unchanged); val sets scored every {a.mixture_every} steps",
                  flush=True)
        for v in mval:
            print(f"  [mixture] val set {v['name']}" + (f" -> group {v['group_name']}" if mix_learned else "")
                  + f": {v['n']} units of {v['n_read']} read from {v['files']} file(s) ({v['n_nc']} NC dropped), label entropy "
                  f"{v['H']:.4f}" + (f", utility {v['utility']:g}" if mix_learned else "")
                  + (f", reference loss {ref_d[v['name']]:.4f}" if ref_d else ""), flush=True)
        if mix_learned:
            orphans = [g for gi, g in enumerate(gnames) if not any(v["group"] == gi for v in mval)]
            if orphans:
                print(f"  [mixture] NOTE: group(s) {orphans} have no val set: they get the others' weight-averaged signal (neutral to "
                      f"first order; they move by second-order effects and smoothing)", flush=True)
        mix_fp_parts = ref_sha
        print(f"  [mixture] set up in {time.time() - t_mx:.1f}s", flush=True)

    def mix_fp():
        """(p103) the mixture's content for the run identity: groups (patterns, units), val sets (attachment, units), reference."""
        ref_sha = mix_fp_parts
        h = {"mode": a.mixture, "val": [[v["name"], v["group_name"], v["fp"]] for v in mix.val],
             "ref": "initial" if not ref_sha else ["file", ref_sha]}
        if mix.learned:
            h["groups"] = [[g, pats, len(u), units_fingerprint(u)] for (g, pats), u in zip(mix_groups, mix_units)]
        return hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()[:16]

    def hs_fp():
        """(datavar) the hierarchy's content for the run identity: per source its patterns, files, unit / case counts, case sizes and
        units (in case order, by content); the case mode; the resolved weights; the capability map's content, tags and weights."""
        h = {"case": a.sampling_case, "weights": hs.w,
             "sources": [[g, hs_pats[i], hs_files[i], hs.units[i], hs.n[i],
                          hashlib.sha256(json.dumps([len(c) for c in hs.cases[i]]).encode()).hexdigest()[:16],
                          units_fingerprint([u for c in hs.cases[i] for u in c])] for i, g in enumerate(hs.names)],
             "cap": None if hs.caps is None else {"map": hs_cmap, "tags": hs.tags, "caps": hs.caps, "weights": hs.cap_w},
             "dropped": hs_dropped}
        return hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()[:16]

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

    nh = nh_opt = nh_sched = gcaps = nh_tr = nh_v = None   # (p95) off: nothing below touches any of these
    nh_params = []
    if nh_K:
        nh = NumHead(lm_head.weight.shape[1], nh_K, a.num_head_family, a.num_head_dim, seed=a.seed).to(label_ids.device)
        nh_params = list(nh.parameters())
        if dp.on:                             # (ddp) equal anyway (its own seeded generator), as the LoRA
            dp.broadcast_params(nh_params)
        nh_opt = torch.optim.AdamW(nh_params, lr=a.num_head_lr, betas=(0.9, 0.95), weight_decay=0.0)
        nh_sched = torch.optim.lr_scheduler.LambdaLR(
            nh_opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        gcaps = []
        print(f"num-head: {a.num_head} {nh.family}, dim {nh.D}, hidden {nh.H}, {sum(p.numel() for p in nh_params):,} params, lr "
              f"{a.num_head_lr:g} (its own AdamW + clip 1.0); letters on numeric units {a.num_head_letters}; CRPS weight "
              f"{a.num_head_crps:g}; trained JOINTLY (its gradient reaches the LoRA)", flush=True)

    def nh_loss(z, tgt, ks, chunk, loss):
        """(p95) a training forward's loss with the numeric head, from its decision states (kept in the graph) and letter logits:
        numeric rows' head loss replaces (letters off) or joins (letters on) their letter loss, each row still weighted 1 / len(chunk)
        as in `loss` (the forward's mean). A forward without numeric rows returns `loss` itself."""
        nonlocal nh_tr
        h = torch.cat(gcaps, 0)
        gcaps.clear()
        if h.shape[0] != z.shape[0]:
            raise RuntimeError(f"--num-head: {h.shape[0]} decision states kept for {z.shape[0]} prompts")
        rows = [b for b, e in enumerate(chunk) if e.num is not None]
        if not rows:
            return loss
        idx = torch.tensor(rows, device=z.device)
        tf, u, kb, t = num_batch([chunk[b].num for b in rows], z.device)
        hrow, ce, cr = num_head_rows(nh, h.index_select(0, idx), tf, u, kb, t, a.num_head_crps)
        if a.num_head_letters == "on":
            out = loss + hrow.to(loss.dtype).sum() / len(chunk)
        else:
            lrow = example_loss_rows(z, tgt, ks, chunk, a.rps_weight, a.brier_weight)
            keep = torch.ones(len(chunk), dtype=torch.bool, device=z.device)
            keep[idx] = False
            out = (lrow[keep].sum() + hrow.to(lrow.dtype).sum()) / len(chunk)
        nh_tr += torch.stack([ce.detach().sum(), cr.detach().sum(), ce.new_tensor(float(len(rows)))])
        if dyn is not None:                   # (ss) --dyn-log: the numeric rows' own terms (detached)
            dyn.head(rows, hrow)
        return out

    def nh_eval_row(gc, z, tgt, ks, chunk, logp):
        """(p95) one val chunk's numeric rows: [head CE sum, letters' CE sum, exact entropy sum, log K sum, head CRPS-excess sum, rows]."""
        h = torch.cat(gc, 0)
        rows = [b for b, e in enumerate(chunk) if e.num is not None]
        if not rows:
            return [0.0] * 6
        idx = torch.tensor(rows, device=z.device)
        tf, u, kb, t = num_batch([chunk[b].num for b in rows], z.device)
        _, ce, cr = num_head_rows(nh, h.index_select(0, idx), tf, u, kb, t)
        mask = torch.arange(logp.shape[1], device=z.device)[None, :] < ks[:, None]
        lce = -(tgt * logp.masked_fill(~mask, 0.0)).sum(-1).index_select(0, idx)
        ent = -(t * t.clamp_min(1e-300).log()).sum(1)
        return [float(ce.sum()), float(lce.sum()), float(ent.sum()), float(kb.double().log().sum()), float(cr.sum()), float(len(rows))]

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
        nonlocal aux_v, nh_v, ev_last
        model.eval()
        tot, n, correct = 0.0, 0, 0
        ev_u = {} if elog is not None else None   # (grok1) --eval-log: per-unit rows of this eval (id(example) -> CE, Brier, correct)
        rps_sum, n_sc = 0.0, 0                # (49) val RPS on score units; "val soft-CE" stays the readout's CE alone
        rows = []                             # per chunk: (index, mean loss, rows, correct, RPS sum, score rows)
        arows = []                            # (aux-vbll) per chunk: (index, aux_eval_row)
        nrows = []                            # (p95) per chunk: (index, nh_eval_row)
        with torch.no_grad(), layers_eager(compiled_layers):   # (speedprobe3) no-op unless --compile-layers
            chunks = exact_chunks(tok, val_ex, a.exact_tokens, a.exact_batch) if batched else \
                [val_ex[i : i + a.batch_size] for i in range(0, len(val_ex), a.batch_size)]
            for ci, chunk in enumerate(chunks):
                if dp.on and ci % dp.world != dp.rank:   # (ddp) chunk ci is rank (ci mod world)'s; the same chunks as one GPU
                    continue
                check()
                texts, tgt, ks, kmax = batch_tensors(chunk, a.device, a.h2d)
                cap = [] if aux is not None else None   # (aux-vbll) the chunk's decision states
                gcap = [] if nh is not None else None   # (p95) likewise, for the numeric head
                if batched:             # (speed) never --pack-pad-to here: val numbers stay those of the unpadded forward
                    z = exact_letter_logits(model, tok, texts, label_ids[:kmax], a.device, a.exact_batch, a.exact_head,
                                            h2d=a.h2d, keep_h=cap, keep_g=gcap)
                else:
                    z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, full, a.exact_head, a.h2d,
                                    keep_h=cap, keep_g=gcap, sel=lsel)
                loss, logp = example_loss(z, tgt, ks, chunk, a.ordinal_readout)
                rps_c, nsc_c = 0.0, 0
                if ordinal:
                    lev, sc = ls.ordinal().batch_levels(chunk, kmax, a.device)
                    rps_c, nsc_c = float(ls.ordinal().rps_rows(logp, tgt, lev, sc).sum()), int(sc.sum())
                rows.append((ci, float(loss), len(texts), int((logp.argmax(-1) == tgt.argmax(-1)).sum()), rps_c, nsc_c))
                if ev_u is not None:
                    ev_u.update(eval_unit_rows(chunk, logp, tgt, ks))
                if aux is not None:
                    arows.append((ci, *aux_eval_row(cap, z, tgt, ks, logp)))
                if nh is not None:
                    nrows.append((ci, *nh_eval_row(gcap, z, tgt, ks, chunk, logp)))
        if dp.on:                             # (ddp) every chunk's row on every rank, summed below in chunk order as one GPU sums
            rows = dp.gather_rows(rows, len(chunks))
            if aux is not None:
                arows = dp.gather_table(arows, len(chunks), 6 * len(aux.heads) + 2)
            if nh is not None:
                nrows = dp.gather_table(nrows, len(chunks), 6)
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
        if nh is not None:                    # (p95) the head on the numeric val rows, next to the letters on the same rows
            sm = [0.0] * 6
            for r in nrows:
                for i, x in enumerate(r[1:]):
                    sm[i] += x
            nn = max(sm[5], 1.0)
            gain = sm[3] - sm[2]              # sum log K - sum H(exact): the exact posterior's gain over uniform
            nh_v = {"n": int(sm[5]), "head_ce": sm[0] / nn, "letters_ce": sm[1] / nn, "exact_entropy": sm[2] / nn,
                    "head_skill": (sm[3] - sm[0]) / gain if gain > 0 else float("nan"),
                    "letters_skill": (sm[3] - sm[1]) / gain if gain > 0 else float("nan"), "head_crps_excess": sm[4] / nn}
            if nh_v["n"]:
                print(f"  [num-head] val: {nh_v['n']} numeric prompts  head CE {nh_v['head_ce']:.4f} (skill {nh_v['head_skill']:.4f})  "
                      f"letters CE {nh_v['letters_ce']:.4f} (skill {nh_v['letters_skill']:.4f})  exact H {nh_v['exact_entropy']:.4f}  "
                      f"head CRPS-excess {nh_v['head_crps_excess']:.4f}", flush=True)
            else:
                print("  [num-head] val: no numeric val prompts", flush=True)
        if ev_u is not None:                  # (grok1) a complete eval only (a stop mid-eval raises before this)
            ev_last = [ev_u[id(e)] for e in val_ex]
        return tot / max(n, 1), correct / max(n, 1)

    def mixture_eval():
        """(p103) every mixture val set's mean per-unit loss {name: L} under the current model: the main readout's soft CE per unit
        (the val path's readout: prior offsets / ordinal readout applied; --num-head: a numeric unit's head CE), chunked and sharded
        over ranks exactly as evaluate(), each chunk's float64 sum gathered and summed in chunk order (the one-GPU numbers)."""
        model.eval()
        out = {}
        with torch.no_grad(), layers_eager(compiled_layers):
            for vs, ex in zip(mix.val, mix_ex):
                chunks = exact_chunks(tok, ex, a.exact_tokens, a.exact_batch) if batched else \
                    [ex[i : i + a.batch_size] for i in range(0, len(ex), a.batch_size)]
                rows = []
                for ci, chunk in enumerate(chunks):
                    if dp.on and ci % dp.world != dp.rank:
                        continue
                    check()
                    texts, tgt, ks, kmax = batch_tensors(chunk, a.device, a.h2d)
                    gcap = [] if nh is not None else None
                    if batched:
                        z = exact_letter_logits(model, tok, texts, label_ids[:kmax], a.device, a.exact_batch, a.exact_head,
                                                h2d=a.h2d, keep_g=gcap)
                    else:
                        z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, full, a.exact_head, a.h2d,
                                        keep_g=gcap, sel=lsel)
                    _, logp = example_loss(z, tgt, ks, chunk, a.ordinal_readout)
                    mask = torch.arange(logp.shape[1], device=logp.device)[None, :] < ks[:, None]
                    ce = -(tgt * logp.masked_fill(~mask, 0.0)).sum(-1).double()
                    if nh is not None:            # (p95) numeric units: the head's CE (the readout they are answered with)
                        nr = [b for b, e in enumerate(chunk) if e.num is not None]
                        if nr:
                            idx = torch.tensor(nr, device=z.device)
                            tf, u_, kb, t_ = num_batch([chunk[b].num for b in nr], z.device)
                            _, hce, _ = num_head_rows(nh, torch.cat(gcap, 0).index_select(0, idx), tf, u_, kb, t_)
                            ce = ce.index_copy(0, idx, hce.to(ce.dtype))
                    rows.append((ci, float(ce.sum()), float(len(chunk))))
                if dp.on:                     # (ddp) every chunk's row on every rank, in chunk order
                    rows = dp.gather_table(rows, len(chunks), 2)
                tot = cnt = 0.0
                for _, s_, n_ in sorted(rows):
                    tot += s_; cnt += n_
                out[vs["name"]] = tot / max(cnt, 1.0)
        model.train()
        return out

    def mixture_log(rec):
        """(p103) one mixture eval / update, two lines."""
        st = rec["step"]
        print(f"  [mixture] step {st} val: " + "; ".join(
            f"{v['name']} CE {rec['loss'][v['name']]:.4f} KL {rec['kl'][v['name']]:.4f}"
            + (f" e {rec['excess'][v['name']]:+.4f}" if "excess" in rec else "") for v in mix.val), flush=True)
        if mix.learned:
            tot = max(sum(rec["drawn"]), 1)
            print(f"  [mixture] step {st} " + (("signal " + " ".join(f"{g} {rec['signal'][g]:+.4f}" for g in mix.groups) + "; ")
                                               if "signal" in rec else "")
                  + "weights " + ", ".join(f"{g} {b:.4f} -> {c:.4f}" for g, b, c in zip(mix.groups, rec["w_before"], rec["w_after"]))
                  + "; drawn " + ", ".join(f"{g} {d} ({d / tot:.1%})" for g, d in zip(mix.groups, rec["drawn"]))
                  + "; mean weights " + ", ".join(f"{g} {x:.4f}" for g, x in zip(mix.groups, mix.mean_weights(st))), flush=True)

    def mixture_observe(step):
        mixture_log(mix.observe(step, mixture_eval()))

    clock.lap("prep", sync=True)
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
                                 "model_revision": getattr(mcfg, "_commit_hash", None), "resume_version": RESUME_VERSION,
                                 **({"num_fp": nh_fp} if nh is not None else {}),     # (p95) the numeric axes trained on
                                 **({"prior_offset_fp": po_fp} if po_index is not None else {}),    # (p101) the priors used
                                 **({"mixture_fp": mix_fp()} if mix is not None else {}),         # (p103) groups / val sets / ref
                                 **({"sampling_fp": hs_fp()} if hs is not None else {})})         # (datavar) the hierarchy
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
            if nh is not None:                    # (p95) likewise for the numeric head
                if "num_head" not in ck or "num_head" not in ts:
                    raise SystemExit(f"resume state {rpath}: no num_head state (written without --num-head?)")
                nh.load_state_dict(ck["num_head"]["state"])
                nh_opt.load_state_dict(ts["num_head"]["optimizer"])
                nh_sched.load_state_dict(ts["num_head"]["scheduler"])
                nh_v = ts["num_head"]["val"]
            if mix is not None:                   # (p103) streams, draws, weights, history
                if "mixture" not in ts:
                    raise SystemExit(f"resume state {rpath}: no mixture state (written without --mixture / --mixture-val?)")
                mix.load_state(ts["mixture"])
            if hs is not None:                    # (datavar) the per-source cursors and the draw RNG
                if "sampling" not in ts:
                    raise SystemExit(f"resume state {rpath}: no sampling state (written without --sampling hier?)")
                try:
                    hs.load_state(ts["sampling"])
                except ValueError as e:
                    raise SystemExit(f"resume state {rpath}: {e}") from None
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

    if stop_at and start > stop_at:           # (speed) the resume state is past the requested prefix: it cannot be undone
        raise SystemExit(f"--stop-after-step {stop_at}: the resume state is already at step {start}; drop the flag (or raise it) to "
                         f"continue, or pass --resume off to start over")
    if dyn_uids is not None:                  # (ss) --dyn-log: this rank's file, kept up to the resumed step, opened for appending
        dyn = DynLog(a.dyn_log + (f".rank{dp.rank}" if dp.on else ""), dyn_uids, len(train) if mix_units is None and hs is None
                     else None, a.rps_weight, a.brier_weight, a.num_head_letters if nh is not None else None)
        dyn.open(start)
    if ev_uids is not None:                   # (grok1) --eval-log: kept up to the resumed step; this start's header (eval order)
        elog = EvalLog(a.eval_log)
        ev_keep = [u for u, e in val_mk if e]
        elog.open(start, {"n": len(val_ex), "uids": [ev_uids[id(u)] for u in ev_keep], "src": [u.get("source") for u in ev_keep],
                          "h": [_sig6(label_entropy(e[1])) for e in val_ex], "val_fp": units_fingerprint(val)})
    clock.lap("resume", sync=True)
    if v0 is None:
        v0 = evaluate()
        print(f"step 0  val soft-CE {v0[0]:.4f}  acc {v0[1]:.4f}  (the untouched model)", flush=True)
        if elog is not None:                  # (grok1)
            elog.write(0, opt.param_groups[0]["lr"], v0, ev_last, [label_entropy(e[1]) for e in val_ex], lora_norms(model, te.LoRALinear))
    if mix is not None and mix.L0 is None:    # (p103) the val sets' step-0 losses (the 'initial' reference); restored on resume
        mixture_observe(0)
    clock.lap("initial_eval", sync=True)
    clock.startup(start)
    model.train()
    t0 = time.time()
    saver = ckpt.AsyncCheckpointer(a.out, s3_prefix=a.ckpt_s3)

    def config(step):
        return {"model": a.model, "lora_r": a.lora_r, "lora_targets": list(targets),
                # (ss 10-08) the scale rule, only when set: claude_lmscore.py & co. apply train_e2e.lora_scale_from_config(config)
                **({"lora_scale_rule": a.lora_scale_rule, "lora_scale_anchor": a.lora_scale_anchor, "lora_scale": lora_scale,
                    "lora_alpha": te.LORA_ALPHA} if lora_scale is not None else {}),
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
                **({"weight_decay": a.weight_decay} if a.weight_decay else {}),   # (grok1) only when set
                **({"lr_schedule": a.lr_schedule} if a.lr_schedule != "cosine" else {}),   # (grok1) likewise
                **({"microbatch": a.microbatch, "microbatch_tokens": mb_budget} if regroup else {}),   # (2026-10-02) only when on
                **({"exact_head": a.exact_head} if a.exact_head != "model" else {}),   # (gemmafast) likewise
                **({"letter_head": a.letter_head} if a.letter_head != "full" else {}),  # (speed, 2026-10-06) likewise
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
                                 "label_ids": aux_labels, "val": aux_v}} if aux is not None else {}),
                # (p95) only when on: the numeric head's spec, training settings and last val numbers (tensors: the `num_head` key)
                **({"num_head": {**nh.meta(), "spec": a.num_head, "letters": a.num_head_letters, "crps": a.num_head_crps,
                                 "lr": a.num_head_lr, "val": nh_v}} if nh is not None else {}),
                # (p101) only when on: claude_lmscore.py notes an evidence-trained adapter scored without its offsets
                **({"prior_offset": a.prior_offset} if po_index is not None else {}),
                # (p103) only when on: groups, weights, draws, val sets, the update history
                **({"mixture": mix.summary(step)} if mix is not None else {}),
                # (datavar) only when on: sources, files, units, cases, weights / shares, capabilities, draws, passes
                **({"sampling": {**hs.summary(hs_files, a.sampling_case, a.sampling_source_weights,
                                              a.sampling_capability_weights), "dropped": hs_dropped}} if hs is not None else {})}

    def nh_entry():
        """(p95) the checkpoint's `num_head` key: everything claude_lmscore.py --num-head needs to rebuild the head."""
        return {"state": nh.state_dict(), **nh.meta(), "letters": a.num_head_letters, "crps": a.num_head_crps}

    def aux_entry():
        """(aux-vbll) the checkpoint's `aux_vbll` key: everything claude_lmscore.py --aux-vbll needs to rebuild the head."""
        return {"state": aux.state_dict(), **aux.meta(), "samples": a.aux_vbll_samples, "label_ids": aux_labels}

    def resume_payload(step, reason, rs, cur, wc, pending, mx=None, hx=None):
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
        if mix is not None:                   # (p103) `mx`: the mixture state of that boundary (None: the current one)
            ts["mixture"] = mx if mx is not None else mix.state()
        if hs is not None:                    # (datavar) `hx`: the sampler state of that boundary (None: the current one)
            ts["sampling"] = hx if hx is not None else hs.state()
        if aux is None and nh is None:
            return {"lora": {n: p for n, p in model.named_parameters() if p.requires_grad}, "config": config(step), "train_state": ts}
        out = {"lora": {n: p for n, p in model.named_parameters() if p.requires_grad}, "config": config(step)}
        if aux is not None:
            ts["aux_vbll"] = {"optimizer": aux_opt.state_dict(), "scheduler": aux_sched.state_dict(), "val": aux_v}
            out["aux_vbll"] = aux_entry()
        if nh is not None:                    # (p95) the head in --out's key, its optimizer / scheduler / val in train_state
            ts["num_head"] = {"optimizer": nh_opt.state_dict(), "scheduler": nh_sched.state_dict(), "val": nh_v}
            out["num_head"] = nh_entry()
        out["train_state"] = ts
        return out

    def save(step, with_resume=True):
        if dp.rank:                           # (ddp) rank 0 alone writes and uploads
            return
        ms = saver.save({"lora": {n: p for n, p in model.named_parameters() if p.requires_grad}, "config": config(step),
                         **({"aux_vbll": aux_entry()} if aux is not None else {}),
                         **({"num_head": nh_entry()} if nh is not None else {})}, step)
        print(f"  [ckpt] step {step} snapshot, training blocked {ms:.0f} ms", flush=True)
        if writer is not None and with_resume:
            writer.save(resume_payload(step, "checkpoint", rng_state(rng), cursor, dict(wc_train), False), step)

    def post(step):                           # after a step's optimizer update: eval / checkpoint on their cadence
        nonlocal v, best
        if mix is not None and (step % a.mixture_every == 0 or step == a.steps):   # (p103) val sets; learned: the update
            clock.lap("misc", sync=True)
            mixture_observe(step)
            clock.lap("mix_eval", sync=True)
        if step % a.eval_every == 0 or step == a.steps:
            clock.lap("misc", sync=True)      # (speed) --phase-log: no-ops unless on
            v = evaluate()
            clock.lap("eval", sync=True)
            if math.isnan(best[0]) or v[0] < best[0]:
                best = (v[0], v[1], step)
            print(f"step {step}  val soft-CE {v[0]:.4f}  acc {v[1]:.4f}", flush=True)
            if elog is not None:              # (grok1) --eval-log: this eval's line (log only)
                elog.write(step, opt.param_groups[0]["lr"], v, ev_last, [label_entropy(e[1]) for e in val_ex],
                           lora_norms(model, te.LoRALinear))
            if a.whole_case:
                print(f"  whole-case train prompts so far: {dict(sorted(wc_train.items()))}", flush=True)
            save(step)
            clock.lap("save", sync=True)
        elif step % a.save_every == 0:
            clock.lap("misc", sync=True)
            save(step)
            clock.lap("save", sync=True)

    phase, boundary, cur_step, rebalanced = "idle", None, start, False
    mix_bound = mix_post = None               # (p103) the mixture state at the step boundary / before the step's post()
    hs_bound = None                           # (datavar) the sampler state at the step boundary (post() never draws)

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
        mx = None                             # (p103) the mixture state that goes with the saved boundary (None: the current one)
        hx = None                             # (datavar) likewise the sampler's
        if phase == "fb":
            (st, rs, cur, wc), pending = boundary, False
            mx = mix_bound
            hx = hs_bound
        else:
            st, rs, cur, wc, pending = cur_step, rng_state(rng), cursor, dict(wc_train), phase == "post"
            if phase == "post":               # before this step's mixture update: the resumed run redoes it
                mx = mix_post
        if st <= 0 or writer.saved_step == st:
            print(f"[resume] {reason}: the state after step {st} is " + ("already on file" if st else "a fresh start")
                  + "; nothing to write", flush=True)
            return st > 0
        try:
            ok = writer.save_sync(resume_payload(st, reason, rs, cur, wc, pending, mx, hx), st)
        except Exception as e:  # noqa: BLE001 -- e.g. a dead CUDA context: report, keep the original error
            print(f"[resume] {reason}: resume save FAILED ({e!r}); the last resume state (step {writer.saved_step}) "
                  f"stands"[:400], flush=True)
            return False
        print(f"[resume] {reason}: state after step {st} " + ("saved" + (" and uploaded" if writer.s3_url else "") if ok
                                                              else "NOT fully saved (see above)"), flush=True)
        return ok

    def draw_mb():
        """One micro-batch's examples: batch_size draws from the sampler in use (pooled cursor, the learned mixture or the
        hierarchical sampler), each a make_example on the example rng. A check point first. (Shared by the default loop and
        --pinned-staging, so both draw identically.)"""
        nonlocal cursor
        check()
        ex = []
        while len(ex) < a.batch_size:
            oo = [] if po_index is not None or dyn is not None else None    # (p101, ss) the option order this draw shows
            unit = train[cursor % len(train)] if mix_units is None else mix.next_unit()   # (p103) learned: a group draw
            if hs is not None:        # (datavar) --sampling hier: [capability ->] source -> case -> question
                unit = hs.next_unit()
            e = make_example(tok, unit, rng, a.max_len, not a.no_permute, a.plain_prompt,
                             a.whole_case, a.noul_defs, wc_train, with_levels=ordinal, chat=chat,
                             **({"order_out": oo} if oo is not None else {}))
            if e and nh is not None:  # (p95) tagged with its numeric axis (None: not numeric); no draws
                e = nh_wrap(e, unit)
            if e and po_index is not None:   # (p101) tagged with its log prior in letter order; no draws
                e = po_wrap(e, unit, oo)
            if e and dyn is not None:        # (ss) the draw's unit, letter order, index in the step and pass; no draws
                dyn.draw(e, unit, oo, cursor)
            cursor += 1
            if e:
                ex.append(e)
        clock.lap("draw")
        return ex

    def staged_step():
        """(speed) --pinned-staging: the fb phase of one step -- the default loop's draws, forwards, backwards, loss weights and check
        points, in the same order and with the same arithmetic -- with each forward's inputs staged one forward AHEAD (stager): while
        forward n is queued on the GPU the host draws / tokenises / packs forward n+1 and its copy runs on the copy stream. Draws are
        made lazily, as the next forward needs them, and NEVER past this step: step n+1's examples are drawn after step n's optimizer
        step and post() (eval / checkpoint / mixture update), as without the flag. So the example stream, every RNG, the cursor, the
        mixture / hierarchical sampler state at every boundary a checkpoint or an emergency save reads, the DDP ranks' draws and the
        resume identity are the default loop's. Kinds (one per default block): 'one' (exact path, one prompt), 'fixed' (fast path
        micro-batch), 'regroup' (--microbatch chunk), 'xbatch' (--exact-batch chunk). -> (step loss sum, examples, last loss or None)."""
        res = {"sum": 0.0, "n": 0, "loss": None}

        def items():                          # this rank's forwards, drawing as they are asked for
            if batched or regroup:
                step_ex = []
                for _mb in range(a.grad_accum * dp.world):
                    step_ex.extend(draw_mb())
                share = dp.share(tok, step_ex, a.dp_share)
                for chunk in (exact_chunks(tok, share, a.exact_tokens, a.exact_batch) if batched else
                              fast_chunks(tok, share, a.microbatch, a.batch_size, mb_budget)):
                    yield ("xbatch" if batched else "regroup", chunk, len(step_ex))
                return
            for mb in range(a.grad_accum * dp.world):
                ex = draw_mb()
                if mb // a.grad_accum != dp.rank:      # (ddp) another rank's micro-batch
                    continue
                if a.exact_path:
                    for e1 in ex:
                        yield ("one", [e1], len(ex))
                else:
                    yield ("fixed", ex, len(ex))

        def stage(item):                      # host: batch_tensors' targets / counts + the forward's encoding -> the stager
            kind, chunk, _ = item
            texts = [x for x, _ in chunk]
            kmax = max(len(t) for _, t in chunk)
            tgt = torch.zeros(len(chunk), kmax)
            for b, (_, t) in enumerate(chunk):
                tgt[b, : len(t)] = torch.tensor(t, dtype=torch.float32)
            ks = torch.tensor([len(t) for _, t in chunk])
            lv = {}                           # (49) the ordinal levels example_loss would copy (two blocking copies per forward)
            if ordinal:
                lv["lev"], lv["sc"] = ls.ordinal().batch_levels(chunk, kmax, "cpu")
            if kind == "xbatch":
                enc, stats = exact_host_inputs(tok, texts, a.exact_batch, a.pack_pad_to)
            elif kind == "one":
                enc = dict(tok(texts[0], return_tensors="pt", add_special_tokens=False).items())
                stats = (int(enc["input_ids"].numel()),) * 2 + (1,)
            else:
                enc = dict(tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).items())
                stats = (int(enc["attention_mask"].sum()), int(enc["attention_mask"].numel()), len(texts))
            st = stager.stage({**{"in." + k: v for k, v in enc.items()}, "tgt": tgt, "ks": ks, **lv},
                              info=(texts, kmax, list(enc), stats))
            clock.lap("stage")
            return st

        def run(item, st):                    # compute: the default block of this kind, on the staged inputs
            kind, chunk, n_all = item
            texts, kmax, keys, stats = st.info
            if kind != "fixed":               # the default's per-prompt / per-chunk check point ('fixed': in draw_mb only)
                check()
            v = stager.acquire(st)
            tgt, ks, enc = v["tgt"], v["ks"], {k: v["in." + k] for k in keys}
            if kind == "xbatch":
                gc_select(texts, a.exact_batch == "pack")
                fwd_begin()
                z = exact_letter_logits(model, tok, texts, label_ids[:kmax], a.device, a.exact_batch, a.exact_head,
                                        pad_to=a.pack_pad_to, h2d=a.h2d, keep_h=caps, keep_g=gcaps, staged=enc)
            elif kind == "one":
                gc_select(texts, False, gc_for(texts))
                fwd_begin()
                z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, full, a.exact_head, a.h2d,
                                keep_h=caps, keep_g=gcaps, staged=[enc])
            else:
                gc_select(texts, False, gc_for(texts))
                z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, h2d=a.h2d, keep_h=caps, keep_g=gcaps,
                                sel=lsel, staged=enc)
            if tstats is not None:
                tstats.append(stats)
            loss, lp = example_loss(z, tgt, ks, chunk, a.ordinal_readout, a.rps_weight, a.brier_weight,
                                    levels=(v["lev"], v["sc"]) if ordinal else None)
            if dyn is not None:               # (ss) before the slot's release: lp is a fresh tensor, nothing here reads the slot
                dyn.fwd(chunk, lp)
            if nh is not None:                # (p95)
                loss = nh_loss(z, tgt, ks, chunk, loss)
            clock.lap("fwd")
            if kind == "one":                 # the default blocks' weights, with their arithmetic (division / product)
                (loss / (a.grad_accum * dp.world * n_all)).backward()
                res["sum"] = res["sum"] + loss.detach(); res["n"] += 1
                if aux is not None:
                    aux_train(z, tgt, ks, 1.0 / (a.grad_accum * dp.world * n_all))
            elif kind == "fixed":
                (loss / (a.grad_accum * dp.world)).backward()
                res["sum"] = res["sum"] + loss.detach() * len(chunk); res["n"] += len(chunk)
                if aux is not None:
                    aux_train(z, tgt, ks, 1.0 / (a.grad_accum * dp.world))
            else:
                (loss * (len(chunk) / n_all)).backward()
                res["sum"] = res["sum"] + loss.detach() * len(chunk); res["n"] += len(chunk)
                if aux is not None:
                    aux_train(z, tgt, ks, len(chunk) / n_all)
            stager.release(st)                # after the LAST consumer: the backward (checkpoint recomputation) and the aux head
            res["loss"] = loss
            clock.lap("bwd")

        it = items()
        nxt = next(it, None)
        st_nxt = stage(nxt) if nxt is not None else None
        while nxt is not None:
            cur, st_cur = nxt, st_nxt
            nxt = next(it, None)              # forward n+1's draws (this step's only) ...
            st_nxt = stage(nxt) if nxt is not None else None   # ... and its staging, before forward n runs
            run(cur, st_cur)
        return res["sum"], res["n"], res["loss"]

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
            if mix is not None:
                mix_post = mix.state()
            post(start)
        phase = "idle"
        if start >= a.steps and not post_pending:
            save(a.steps, with_resume=False)  # a finished run, re-run: re-emit --out from the final state
        last_step = stop_at or a.steps        # (speed) --stop-after-step: the last step of THIS process (the schedule is --steps')
        tstats = clock.toks if clock.on else None   # (speed) --phase-log: each training forward's real / padded tokens
        for step in range(start + 1, last_step + 1):
            cur_step = step
            if writer is not None:
                boundary = (step - 1, rng_state(rng), cursor, dict(wc_train))
                if mix is not None:
                    mix_bound = mix.state()
                if hs is not None:            # (datavar)
                    hs_bound = hs.state()
            phase = "fb"
            step_ex = []
            step_sum, step_n = 0.0, 0         # [step-mean]: the loss over ALL the step's examples (logging only)
            aux_tr = None                     # (aux-vbll) this step's -bound sums per head + example count (logging only)
            if nh is not None:                # (p95) this step's head CE / CRPS-excess sums and numeric row count (logging only)
                nh_tr = torch.zeros(3, dtype=torch.float64, device=a.device)
            if stager is not None:            # (speed) --pinned-staging: the blocks below, inputs staged one forward ahead
                step_sum, step_n, last_loss = staged_step()
                if last_loss is not None:
                    loss = last_loss
            # (ddp) every rank draws the GLOBAL step (world x grad_accum micro-batches, the one-GPU stream) and runs its own grad_accum
            for mb in range(a.grad_accum * dp.world if stager is None else 0):
                ex = draw_mb()
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
                                        keep_h=caps, keep_g=gcaps, tok_stats=tstats)
                        loss, lp = example_loss(z, tgt, ks, [e1], a.ordinal_readout, a.rps_weight, a.brier_weight)
                        if dyn is not None:   # (ss) the readout's log p, detached
                            dyn.fwd([e1], lp)
                        if nh is not None:    # (p95) a numeric row: the head's loss (in the graph) instead of / with the letters'
                            loss = nh_loss(z, tgt, ks, [e1], loss)
                        clock.lap("fwd")
                        (loss / (a.grad_accum * dp.world * len(ex))).backward()   # (ddp) 1 / the GLOBAL step's count
                        step_sum = step_sum + loss.detach(); step_n += 1
                        if aux is not None:   # (aux-vbll) after the LoRA's backward, on detached inputs, the same weight
                            aux_train(z, tgt, ks, 1.0 / (a.grad_accum * dp.world * len(ex)))
                        clock.lap("bwd")
                    continue
                texts, tgt, ks, kmax = batch_tensors(ex, a.device, a.h2d)
                gc_select(texts, False, gc_for(texts))
                z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, h2d=a.h2d, keep_h=caps, keep_g=gcaps,
                                sel=lsel, tok_stats=tstats)
                loss, lp = example_loss(z, tgt, ks, ex, a.ordinal_readout, a.rps_weight, a.brier_weight)
                if dyn is not None:           # (ss)
                    dyn.fwd(ex, lp)
                if nh is not None:            # (p95)
                    loss = nh_loss(z, tgt, ks, ex, loss)
                clock.lap("fwd")
                (loss / (a.grad_accum * dp.world)).backward()                  # (ddp) len(ex) / the GLOBAL step's count
                step_sum = step_sum + loss.detach() * len(ex); step_n += len(ex)
                if aux is not None:           # (aux-vbll)
                    aux_train(z, tgt, ks, 1.0 / (a.grad_accum * dp.world))
                clock.lap("bwd")
            if batched and stager is None:    # sum_i loss_i / n, as the one-prompt-per-forward loop above
                for chunk in exact_chunks(tok, dp.share(tok, step_ex, a.dp_share), a.exact_tokens, a.exact_batch):   # (ddp)
                    check()
                    texts, tgt, ks, kmax = batch_tensors(chunk, a.device, a.h2d)
                    gc_select(texts, a.exact_batch == "pack")
                    fwd_begin()
                    z = exact_letter_logits(model, tok, texts, label_ids[:kmax], a.device, a.exact_batch, a.exact_head,
                                            pad_to=a.pack_pad_to, h2d=a.h2d, keep_h=caps, keep_g=gcaps, tok_stats=tstats)
                    loss, lp = example_loss(z, tgt, ks, chunk, a.ordinal_readout, a.rps_weight, a.brier_weight)
                    if dyn is not None:       # (ss) the chunk's examples are the drawn objects (dyn.draw keyed them)
                        dyn.fwd(chunk, lp)
                    if nh is not None:        # (p95)
                        loss = nh_loss(z, tgt, ks, chunk, loss)
                    clock.lap("fwd")
                    (loss * (len(chunk) / len(step_ex))).backward()
                    step_sum = step_sum + loss.detach() * len(chunk); step_n += len(chunk)
                    if aux is not None:       # (aux-vbll)
                        aux_train(z, tgt, ks, len(chunk) / len(step_ex))
                    clock.lap("bwd")
            if regroup and stager is None:    # --microbatch regroup / budget: the same examples, length-grouped forwards
                for chunk in fast_chunks(tok, dp.share(tok, step_ex, a.dp_share), a.microbatch, a.batch_size, mb_budget):
                    check()
                    texts, tgt, ks, kmax = batch_tensors(chunk, a.device, a.h2d)
                    gc_select(texts, False, gc_for(texts))
                    z = letter_logp(stack, lm_head, tok, texts, label_ids[:kmax], a.device, h2d=a.h2d, keep_h=caps, keep_g=gcaps,
                                    sel=lsel, tok_stats=tstats)
                    loss, lp = example_loss(z, tgt, ks, chunk, a.ordinal_readout, a.rps_weight, a.brier_weight)
                    if dyn is not None:       # (ss) length-sorted chunk: dyn.draw keyed the drawn objects, so each row finds its draw
                        dyn.fwd(chunk, lp)
                    if nh is not None:        # (p95)
                        loss = nh_loss(z, tgt, ks, chunk, loss)
                    clock.lap("fwd")
                    (loss * (len(chunk) / len(step_ex))).backward()
                    step_sum = step_sum + loss.detach() * len(chunk); step_n += len(chunk)
                    if aux is not None:       # (aux-vbll)
                        aux_train(z, tgt, ks, len(chunk) / len(step_ex))
                    clock.lap("bwd")
            if nh is not None:                # (p95) a step (or rank share) without numeric rows still steps the head, with zeros,
                for p_ in nh_params:          # on every rank alike (so one process and N ranks take the same AdamW steps)
                    if p_.grad is None:
                        p_.grad = torch.zeros_like(p_)
            if dyn is not None:               # (ss) the step's log p to the host: one copy, queued behind its backwards (no wait)
                dyn.ready()
            lazy = fold and step < last_step  # (speed A) --dp-stop fold; the last step (of this process) agrees as before
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
                if nh is not None:            # (p95) likewise: the numeric head's own buckets, after the LoRA's
                    dp.reduce_num(nh_params)
            clock.lap("comm")
            phase = "opt"
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if aux is not None:               # (aux-vbll) after the LoRA's update; KL once per step (every rank: the same values)
                aux.kl_loss(aux_n).backward()
                aux.clip_(1.0)
                aux_opt.step(); aux_sched.step(); aux_opt.zero_grad(set_to_none=True)
            if nh is not None:                # (p95) after the LoRA's update: its own clip at 1.0 and AdamW
                clip_params_(nh_params, 1.0)
                nh_opt.step(); nh_sched.step(); nh_opt.zero_grad(set_to_none=True)
            clock.lap("opt")
            loss_hist.append(loss.detach())
            phase = "post"
            if mix is not None:               # (p103) a stop inside post() saves this; the resumed run redoes the update
                mix_post = mix.state()
            if dyn is not None:               # (ss) after the optimizer step was queued: one host wait, this step's lines out + flush
                dyn.write(step)
            if not lazy:                      # (speed A) fold: this step's stop codes are read at the next step's agree point
                check(sync=True)
            if dp.on:                         # (aux-vbll) the head's tensors checksummed with the LoRA's (none when off)
                dp.verify_same(params + aux_params + nh_params, step, last=step == last_step, lazy=lazy)
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
                if nh is not None:            # (p95) the GLOBAL step's numeric rows: head CE and CRPS-excess per row
                    c_, r_, n_ = (dp.sum_(nh_tr) if dp.on else nh_tr).tolist()
                    print(f"  [num-head] step {step}  numeric rows {int(n_)}  head CE/row {c_ / max(n_, 1):.4f}  CRPS-excess/row "
                          f"{r_ / max(n_, 1):.4f}", flush=True)
            clock.lap("log")
            post(step)
            clock.window(step, last=step == last_step)
            phase = "idle"
            if intr.rebalance is not None and not rebalanced and writer is not None:
                rebalanced = True             # a hint, not a deadline: save now (upload confirmed), keep training
                writer.drain()
                ok = writer.saved_step == step or \
                    writer.save_sync(resume_payload(step, "spot rebalance", rng_state(rng), cursor, dict(wc_train), False), step)
                print(f"SPOT REBALANCE hint: resume state after step {step} {'saved' if ok else 'NOT saved'}; training continues",
                      flush=True)
        if stop_at:                           # (speed) --stop-after-step: the state after that step on file before a clean exit
            if start == stop_at and not post_pending:   # a re-run at the stop: re-emit --out (+ resume state if not already on file)
                save(stop_at, with_resume=writer is not None and writer.saved_step != stop_at)
            elif stop_at % a.eval_every and stop_at % a.save_every:   # post() did not checkpoint this step: --out + resume now
                save(stop_at)
            clock.lap("save", sync=True)
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
        if dyn is not None:                   # (ss) every finished step's lines are already flushed
            dyn.close()
        if elog is not None:                  # (grok1)
            elog.close()
    saver.close()
    if writer is not None:
        writer.close()
    clock.final()                             # (speed) --phase-log totals; no-op when off
    if stop_at and not dp.rank:               # (speed) --stop-after-step: where the run paused; a pause that could not be saved fails
        ok = writer is not None and writer.saved_step == stop_at
        print(f"STOPPED after step {stop_at} of {a.steps} (--stop-after-step; the warmup / cosine schedule is the {a.steps}-step one): "
              f"--out and resume state {'saved' if ok else 'NOT saved'}; re-run without --stop-after-step (or with a larger one) to "
              f"continue this run", flush=True)
        if not ok:
            return 1
    if dp.on:
        print(f"[ddp] the {dp.world} ranks' LoRA parameters were bit-identical at all {dp.checks} checks (every "
              f"{dp.check_every or 'never'} steps and the last)", flush=True)
    if a.tok_cache:
        print(f"tok-cache: {tok.hits} hits, {tok.misses} misses (LRU of {a.tok_cache})", flush=True)
    if a.token_cache:                         # (pretok) prompts assembled / built directly, checks
        print(tok.pretok.summary(), flush=True)
    if stager is not None:                    # (speed) --pinned-staging
        print(stager.summary(), flush=True)
    if mix is not None and mix.learned:       # (p103)
        tot = max(sum(mix.cursors), 1)
        print("mixture: final weights " + ", ".join(f"{g} {x:.4f}" for g, x in zip(mix.groups, mix.w)) + "; mean over the run "
              + ", ".join(f"{g} {x:.4f}" for g, x in zip(mix.groups, mix.mean_weights(a.steps))) + "; drawn "
              + ", ".join(f"{g} {d} ({d / tot:.1%})" for g, d in zip(mix.groups, mix.cursors))
              + "; passes " + ", ".join(f"{g} {d / n:.3f}" for g, d, n in zip(mix.groups, mix.cursors, mix.n)), flush=True)
    if hs is not None:                        # (datavar)
        tot = max(sum(hs.cursors), 1)
        print("sampling: drawn " + ", ".join(f"{g} {d} ({d / tot:.1%})" for g, d in zip(hs.names, hs.cursors))
              + "; passes over cases " + ", ".join(f"{g} {d / n:.3f}" for g, d, n in zip(hs.names, hs.cursors, hs.n)), flush=True)
    print(f"saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
