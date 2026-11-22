#!/usr/bin/env python3
"""(FULLFT) FULL fine-tuning of Gemma-4-31B-it's text stack + the residual VBLL decision head on one p8 node (8 x H100 80 GB), FSDP2.

  torchrun --nproc-per-node 8 trainers/fullft_train.py --manifest train.jsonl --val val.jsonl --out DIR [memory flags] ...
  torchrun --nproc-per-node 8 trainers/fullft_train.py --smoke --length 32768 --out rec.json [memory flags]      # memory / speed smoke
  python3 trainers/fullft_train.py --tiny --smoke --length 64 --out /tmp/x.json                                 # CPU

Model: google/gemma-4-31B-it at a PINNED revision; only the text stack (Gemma4TextModel, 30.7B) is built and trained; the vision tower is
untouched and copied unchanged at export. FSDP2 full-shard per decoder layer + root, BF16 compute (MixedPrecisionPolicy), FP32 master
weights, FP32 gradient reduction (default), AdamW. The decision readout is LETTER-ONLY: the final hidden state h at each decision's
last token, native letter logits ell = softcap(h E[letters]^T) in fp32 (the tied embedding's 52 rows; full-vocabulary logits are
never formed), the residual VBLL head on detached (h, ell).

(v5, stock_decision_v1 Stage 1) --attention-policy C | B: the model config carries the policy (fullft_mem.py: global layers causal
(C) or bidirectional within their document (B); local layers causal-windowed in both); training, --extract-features, the export's
config.json and the serving engine read the same value. --stock-only: the only accepted initialisation is the pinned stock snapshot
(no --init-from, no exported fine-tune, no head / adapter files, no contender path). --preset sdv1_pilot fixes the pilot's settings
(ROADMAP Stage 2) and refuses conflicting flags. --max-row-tokens / --k-range refuse rows outside the manifest contract.
--extract-features FILE: forward-only over --manifest (same build, policy and packing), writes h / ell / K / t / w per row.

Memory options (fullft_mem.py; each measured by the smoke): --attn jev_mem (no [L, L] tensors; sdpa / eager = the old path),
--ac (per-layer activation checkpointing), --offload-ac (checkpointed layer inputs to pinned host memory for rows >= --offload-min
tokens), --mlp-chunk N (sequence-chunked MLP with nested checkpoints for rows >= --mlp-chunk-min), --optim adamw | adamw-bf16 (bf16
moments, stochastic rounding; fidelity caveat) | adamw-cpu (FSDP2 CPUOffloadPolicy: params / grads / optimizer on the host, step on
the CPU) | adamw-fused (torch's fused fp32 AdamW kernel; no full-size step temporaries), --evict-states-min N (v2: before a step whose
longest row has >= N tokens, the AdamW moments move to pinned host memory for the forward / backward and come back for the step:
exact fp32 AdamW, the same kernel on the same values, 28.6 GiB per GPU freed during the backward at 31B / 8), --freeze-embed (the embedding stays at its pretrained value: a recipe change, disclosed), --reduce-dtype bf16 (gradient
reduce-scatter in bf16: a numerics change, disclosed), --pack-tokens N (documents packed into rows of <= N tokens with per-document
positions: block-diagonal causal attention).

Data (fullft_render.py writes it; preservation.py joins parent probabilities): one JSON row per EXPOSURE, in exposure order, with the
rendered prompt and its identity (decision_uid, renderer_sha256, prompt_sha256, option_order = option keys in letter order), gold
(dict by option key), weight, and for preservation-eligible rows `parent_preservation.probabilities` (dict by option key). The
header row (`"manifest_header": true`) pins total_decision_weight (sum of weights over UNIQUE decisions) and renderer_sha256. The
trainer re-checks: prompt sha256, token count, renderer hash, the weight total, gold sums, parent keys.

Loss (fullft_head.py): backbone sum_i w_i [(1 - a) CE(t_i, p_i) + a KL(q_i || p_i)] / W_step (a = --alpha on eligible rows); head
-sum_i w_i bound_i / W_step + sum_g kl_g KL_g / W_total (W_total pinned), head on DETACHED features.

Why detached: (1) the backbone's update is exactly the declared backbone objective with or without the head (tested bit for bit), so
recipe comparisons (A-D, alpha 0 vs 0.15) are not confounded by a second objective acting on the features; (2) the earlier lesson that
variational objectives backpropagated through the LM collapsed (claude_lmtune aux-vbll notes; EMO's negative all-weight read), while
the last-layer head trained on detached states was the best epistemic signal; (3) the head stays a posterior CONDITIONAL on the
backbone's features (the contract's stated meaning). Cost: the features are not shaped for the head, and the head must track feature
drift during training (its own AdamW at 1e-3; the backbone's cosine LR makes the features settle late). Revisit only with an
ablation (--head-grad, not built).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import random
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # jev_labels.py + its sha-pinned label list sit beside this trainer
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fullft_head as fh  # noqa: E402
import fullft_mem as fm  # noqa: E402

VERSION = "fullft_train v6.0 (2026-11-06)"          # v5 = stock_decision_v1 (the registered pilot keeps v4 / v4.2 under fullft11 / 12)
# v6.0 (vision round 1, research/vision_trainer_plan_2026-11-06.md): --vision (policy V + frozen tower + projector in a plain
# dict OUTSIDE the parameters / FSDP / .to_empty; byte-identity asserted vs the checkpoint and recorded in init_record),
# v2 render manifests (images + EXPANDED n_tokens + image_layout pinned in the header; Tokens expands the marker runs and
# re-checks the recorded spans), micro_inputs emits the span channel + pixel tensors, Decider.forward splices frozen tower
# outputs into inputs_embeds at the spans. Single-image rows only. Gate 0 smoke = a real-manifest --max-steps slice, not --smoke.
# v5.2: --bundle ROLE=MANIFEST[@ROWS_SHA256] (repeatable): when the run completes its schedule, the trained model (in memory, no reload)
# writes OUT/bundles/ROLE.pt in the --extract-features format (h, ell, k, t, w, loss_by_option, has_loss, ids, E_letters, meta) for
# each given manifest (head_fit / cal_select / devP), + OUT/bundles/bundles.json (rows sha256, file sha256, step), so head ablations
# run on the CPU from the bundles. A training manifest is refused as a bundle (rows sha256 equal). With --extract-features the bundles
# are written after it from the same loaded model (one export load for several pools).
# v5.1: --extract-features also writes loss_by_option [N, 52] (letter order, 0.0 where absent) + has_loss [N] from the manifest rows'
# loss_by_option (the expected-regret term of a decision head); a malformed loss_by_option fails the manifest read.
CODE_REVISION = "v5 (v4.2 + attention policy C / B, stock-only init, sdv1_pilot preset, row limits, feature extraction)"
# ROADMAP Stage 2 fixed settings (plan.json "training" / "bayesian_head"); --preset sdv1_pilot applies them and refuses any conflicting flag
SDV1_PILOT = {"optim": "adamw-fused", "lr": 2.5e-6, "warmup": 50, "lr_floor": 0.1, "decisions_per_step": 64, "epochs": 1.0,
              "weight_decay": 0.0, "clip": 1.0, "alpha": 0.0, "head": "on", "head_prior": "1", "head_kl": "1", "head_cov": "full",
              "head_dim": 128, "head_lr": 1e-3, "head_samples": 64, "eval_every": 200, "eval_at_start": True, "save_every": 200,
              "param_dtype": "bf16", "reduce_dtype": "fp32", "max_row_tokens": 8192, "k_range": "2,52", "stock_only": True,
              "freeze_embed": False, "val_max": 0, "attn": "jev_mem"}
SDV1_PILOT_STEPS = 600                               # 38,400 exposures / 64; evals + checkpoints at 0 / 200 / 400 / 600
CONTENDER_MARKERS = ("garas", "pplx", "perplexity", "vllm-sr", "plantwaterer", "decider")
# (v6.1 restore) the mainline's extendable labels: A-Z, a-z, then single-token AA.., AAA.. (jev_labels_gemma4.json, sha-pinned
# 30435b1faa49af2a; 2,591 in all). The first 52 are the letters every earlier jev head was fitted on, so K<=52 renders are
# byte-identical; C7 replay rows with K>52 (POP909 129-way etc.) load and train like the esbar_code_v2e lineage. The
# v6.0 vision branch had regressed this to a hard 52 and died "bad option order" on replay row 34791.
import jev_labels
LABELS = jev_labels.labels()
N_LETTERS = len(LABELS)
BASE_REPO = "google/gemma-4-31B-it"
BASE_REV = "842da3794eaa0b77d5f08bae87a17459d91ff475"
# (v5) the pinned STOCK backbones (--backbone; default 31b = the behaviour of every earlier version). The text stack class follows the
# checkpoint config's model_type (TEXT_FAMILIES); both families share the attention module, the layer_scalar / embed_scale / rotary
# buffers and the "model.language_model." checkpoint prefix.
BACKBONES = {"31b": {"repo": BASE_REPO, "revision": BASE_REV},
             "12b": {"repo": "google/gemma-4-12B-it", "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                     # the 12B repo at this revision ships one model.safetensors (no index) and NO tokenizer files; Gemma-4 shares one
                     # tokenizer (vocab 262,144; the same special token ids in both configs): the pinned 31B snapshot's files are used
                     "tokenizer_from": "31b"}}
TEXT_FAMILIES = {"gemma4_text": ("transformers.models.gemma4.modeling_gemma4", "Gemma4TextModel", "Gemma4TextRotaryEmbedding"),
                 "gemma4_unified_text": ("transformers.models.gemma4_unified.modeling_gemma4_unified", "Gemma4UnifiedTextModel",
                                         "Gemma4UnifiedTextRotaryEmbedding")}


def text_classes(cfg):
    """(v5) (text model class, rotary class) for a text config's model_type; unknown families are refused"""
    import importlib
    mt = getattr(cfg, "model_type", None)
    if mt not in TEXT_FAMILIES:
        raise SystemExit(f"fullft: text model_type {mt!r} is not supported (supported: {sorted(TEXT_FAMILIES)})")
    mod, tm, rot = TEXT_FAMILIES[mt]
    m = importlib.import_module(mod)
    return getattr(m, tm), getattr(m, rot)
TEXT_PREFIX = "model.language_model."


# --------------------------------------------------------------------------------------------------------------------- configs
def tiny_text_config(layers=6, window=48, family="gemma4_text", kv_shared=0):
    if family == "gemma4_unified_text":
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig as TC
    else:
        from transformers import Gemma4TextConfig as TC
    pattern = (["sliding_attention", "sliding_attention", "full_attention"] * ((layers + 2) // 3))[:layers]
    return TC(vocab_size=300, vocab_size_per_layer_input=300, hidden_size=64, intermediate_size=96, num_hidden_layers=layers,
                            layer_types=pattern, num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                            num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=window, final_logit_softcapping=30.0,
                            hidden_size_per_layer_input=0, num_kv_shared_layers=int(kv_shared), enable_moe_block=False,
                            rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                                                "rope_type": "proportional"},
                                             "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}},
                            max_position_embeddings=65536, tie_word_embeddings=True, pad_token_id=0)


class Decider(torch.nn.Module):
    """the text stack + the letter readout. forward(ids [1, T], pos [1, T], last [n]) -> (h [n, H] fp32, ell [n, 52] fp32).

    (vision v6) with `vision` attached: forward(ids, pos, last, spans=None, images=None), where images is a list of
    (i0, K, pixel_tensor) in EXPANDED-id coordinates: the span [i0, i0+K) of ids are image-soft tokens whose embeddings
    come from the FROZEN tower + projector instead of embed_tokens; `spans` [1, T] is the policy-V group channel. The
    frozen tower / projector live in `self.vision` (a plain dict: NOT submodules, so they are outside named_parameters,
    the optimizer, FSDP and .to_empty() by construction -- a silent non-frozen tower is the worst failure here)."""

    def __init__(self, text, letter_ids, attn):
        super().__init__()
        self.text = text
        self.register_buffer("letters", torch.as_tensor(letter_ids, dtype=torch.long), persistent=False)
        self.cap = float(getattr(text.config, "final_logit_softcapping", 0) or 0)
        self.masks = {"full_attention": None, "sliding_attention": None} if attn == fm.ATTN_NAME else None
        self.vision = None                    # {"tower": nn.Module, "projector": nn.Module, "dtype": torch.dtype}

    def forward(self, ids, pos, last, spans=None, images=None):
        if images:
            if self.vision is None:
                raise RuntimeError("Decider.forward got image spans but --vision built no tower")
            v = self.vision
            emb = self.text.embed_tokens(ids)
            with torch.no_grad():                            # the tower is FROZEN; grads never flow into it (tests assert)
                for i0, K, px in images:
                    pxv = px["pixel_values"] if isinstance(px, dict) else px
                    ppos = px["image_position_ids"] if isinstance(px, dict) else None
                    pxv = pxv.to(emb.device).to(v["dtype"])
                    t = v["tower"](pxv, ppos.to(emb.device) if ppos is not None else None)
                    feats = v["projector"](t if torch.is_tensor(t) else t.last_hidden_state)
                    if feats.shape[0] != K or feats.shape[-1] != emb.shape[-1]:
                        raise RuntimeError(f"vision splice: tower gave {tuple(feats.shape)} for K={K}, emb {tuple(emb.shape)}")
                    emb[0, i0:i0 + K] = feats.to(emb.dtype)
            fm.set_image_spans(self.text, spans)
            try:
                out = self.text(inputs_embeds=emb, position_ids=pos, attention_mask=self.masks, use_cache=False)
            finally:
                fm.set_image_spans(self.text, None)
        else:
            out = self.text(input_ids=ids, position_ids=pos, attention_mask=self.masks, use_cache=False)
        h = out.last_hidden_state[0].index_select(0, last)
        h = h if h.dtype == torch.float64 else h.float()
        E = self.text.embed_tokens.weight.index_select(0, self.letters)
        return h, fh.letter_logits(h, E, self.cap)


def rebuild_buffers(text, cfg, device, layer_scalar=None):
    """to_empty leaves buffers uninitialised: recompute the known ones (rotary, embed_scale); layer_scalar from `layer_scalar`
    ({fqn: tensor}, the checkpoint's) or 1.0 (random init); refuse any other buffer."""
    rot = dict(text_classes(cfg)[1](cfg, device="cpu").named_buffers())
    n = 0
    with torch.no_grad():
        for name, b in text.named_buffers():
            if name.startswith("rotary_emb."):
                b.copy_(rot[name[len("rotary_emb."):]].to(b.device, b.dtype))
            elif name.endswith("embed_scale"):
                b.fill_(cfg.hidden_size ** 0.5)
            elif name.endswith("layer_scalar"):
                if layer_scalar is None:
                    b.fill_(1.0)
                else:
                    b.copy_(layer_scalar[name].to(b.device, b.dtype))
            else:
                raise SystemExit(f"fullft: unknown buffer {name}: cannot rebuild it")
            n += 1
    return n


def local(t):
    return t.to_local() if hasattr(t, "to_local") else t


# ------------------------------------------------------------------------------------------------------------------ weights I/O
def weight_map(mdir):
    """(v5) {tensor name: file} from model.safetensors.index.json, or (single-file / index-less checkpoints, e.g. the 12B) from the
    *.safetensors files' own key lists"""
    ix = os.path.join(mdir, "model.safetensors.index.json")
    if os.path.exists(ix):
        return json.load(open(ix))["weight_map"]
    from safetensors import safe_open
    out = {}
    for f in sorted(x for x in os.listdir(mdir) if x.endswith(".safetensors")):
        with safe_open(os.path.join(mdir, f), framework="pt") as sf:
            for k in sf.keys():
                out[k] = f
    if not out:
        raise SystemExit(f"{mdir}: no safetensors weights")
    return out


def tokenizer_dir(a, mdir):
    """(v5) the dir holding the tokenizer: the model dir, or for a pinned backbone that ships none (12B) its declared source's
    snapshot (tokenizer files only)"""
    if mdir and os.path.exists(os.path.join(mdir, "tokenizer.json")):
        return mdir
    src = next((b.get("tokenizer_from") for b in BACKBONES.values() if (b["repo"], b["revision"]) == (a.model, a.revision)), None)
    if not src:
        raise SystemExit(f"{mdir}: no tokenizer.json and no declared tokenizer source for {a.model}@{a.revision}")
    from huggingface_hub import snapshot_download
    b = BACKBONES[src]
    return snapshot_download(b["repo"], revision=b["revision"], allow_patterns=["tokenizer*", "*.jinja", "special_tokens_map.json"])


def snapshot_dir(repo, revision):
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*"])


def load_pretrained(model, mdir, cfg):
    """rank-local load of the text weights from the HF safetensors (each rank slices its own shard: distribute_tensor with
    src_data_rank=None, no communication). Strict: every text parameter and layer_scalar must be present. -> count."""
    from safetensors import safe_open
    idx = weight_map(mdir)
    names = dict(model.text.named_parameters())
    want = set(names) | {n for n, _ in model.text.named_buffers() if n.endswith("layer_scalar")}
    have = {k[len(TEXT_PREFIX):]: f for k, f in idx.items() if k.startswith(TEXT_PREFIX)}
    missing = sorted(want - set(have))
    if missing:
        raise SystemExit(f"fullft: {len(missing)} text tensors missing from the checkpoint, e.g. {missing[:3]}")
    by_file = {}
    for n in sorted(want):
        by_file.setdefault(have[n], []).append(n)
    scal, n_loaded = {}, 0
    with torch.no_grad():
        for f, ns in sorted(by_file.items()):
            with safe_open(os.path.join(mdir, f), framework="pt") as sf:
                for n in ns:
                    full = sf.get_tensor(TEXT_PREFIX + n)
                    if n not in names:
                        scal[n] = full
                        continue
                    p = names[n]
                    if hasattr(p, "to_local"):
                        from torch.distributed.tensor import distribute_tensor
                        dt = distribute_tensor(full.to(torch.float32), p.device_mesh, p.placements, src_data_rank=None)
                        p.to_local().copy_(dt.to_local())
                    else:
                        if tuple(p.shape) != tuple(full.shape):
                            raise SystemExit(f"fullft: {n} shape {tuple(full.shape)} != {tuple(p.shape)}")
                        p.copy_(full.to(p.dtype))
                    n_loaded += 1
    return n_loaded, scal


def gather_full_state(model, dev=None):
    """collective: every text parameter as a full CPU tensor on rank 0 (None elsewhere), in name order."""
    out = {}
    for n, p in model.text.named_parameters():
        full = full_of(p, dev or local(p).device)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            out[n] = full.detach().to("cpu")
        del full
    for n, b in model.text.named_buffers():
        if n.endswith("layer_scalar") and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            out[n] = b.detach().to("cpu")
    return out


def export_hf(out_dir, base_dir, text_state, dtype=torch.bfloat16, shard_bytes=5 << 30):
    """rank 0: a COMPLETE HF checkpoint dir (Gemma4ForConditionalGeneration): the fine-tuned text tensors (bf16) under their original
    names, every other tensor (vision tower, ...) copied unchanged from the base snapshot, base config / tokenizer files copied."""
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    idx = weight_map(base_dir)
    tensors = {}
    for k, f in sorted(idx.items()):
        if k.startswith(TEXT_PREFIX):
            n = k[len(TEXT_PREFIX):]
            if n not in text_state:
                raise SystemExit(f"export: no fine-tuned tensor for {k}")
            t = text_state[n]
            tensors[k] = t.to(dtype) if t.is_floating_point() and not n.endswith("layer_scalar") else t
        else:
            with safe_open(os.path.join(base_dir, f), framework="pt") as sf:
                tensors[k] = sf.get_tensor(k)
    extra = set(TEXT_PREFIX + n for n in text_state) - set(idx)
    if extra:
        raise SystemExit(f"export: fine-tuned tensors absent from the base index: {sorted(extra)[:3]}")
    shards, cur, size = [], {}, 0
    for k in sorted(tensors):
        b = tensors[k].numel() * tensors[k].element_size()
        if cur and size + b > shard_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = tensors[k]
        size += b
    if cur:
        shards.append(cur)
    wmap = {}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: v.contiguous() for k, v in sh.items()}, os.path.join(out_dir, name), metadata={"format": "pt"})
        for k in sh:
            wmap[k] = name
    json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())}, "weight_map": wmap},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    for f in os.listdir(base_dir):
        if f.endswith((".json", ".jinja")) and f != "model.safetensors.index.json" or f.startswith("tokenizer"):
            shutil.copy2(os.path.join(base_dir, f), os.path.join(out_dir, f))
    return len(shards)


# ---------------------------------------------------------------------------------------------------------------------- data
def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def loss_of(r):
    """(v5.1) a manifest row's per-option losses in its option_order -> (K floats | None, status): status 'ok' (every shown option
    has a finite loss), 'absent' (no loss_by_option), 'partial' (some shown option lacks one: NOT used, recorded). The field is
    r['loss_by_option'] (or r['meta']['loss_by_option']): a dict option key -> loss (a JSON string of one is accepted) or a list in
    option_order. Anything else, or a non-finite loss, raises (fails closed). sdv1_head.loss_of is the same rule (tested)."""
    lo = r.get("loss_by_option")
    if lo is None:
        lo = (r.get("meta") or {}).get("loss_by_option")
    if lo is None:
        return None, "absent"
    if isinstance(lo, str):
        lo = json.loads(lo)
    keys = list(r["option_order"])
    if isinstance(lo, dict):
        if any(k not in lo for k in keys):
            return None, "partial"
        vals = [lo[k] for k in keys]
    elif isinstance(lo, (list, tuple)) and len(lo) == len(keys):
        vals = list(lo)
    else:
        raise ValueError(f"{r.get('decision_uid')}: loss_by_option is neither a dict over the options nor a list in option order")
    vals = [float(v) for v in vals]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{r.get('decision_uid')}: non-finite loss_by_option")
    return vals, "ok"


def read_manifest(path, expect_renderer=None):
    """-> (header, rows). The header is an inline row {"manifest_header": true, ...} or the sidecar PATH.header.json (the rows file
    then stays a plain unit list that preservation.py can join). Checks the header and every row's identity / probability fields
    (fails closed)."""
    header, rows = None, []
    if os.path.exists(path + ".header.json"):
        header = json.load(open(path + ".header.json"))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("manifest_header"):
                if header is not None:
                    raise ValueError(f"{path}: two header rows")
                header = r
                continue
            rows.append(r)
    if header is None:
        raise ValueError(f"{path}: no manifest header (total_decision_weight / renderer_sha256 must be pinned)")
    ren = header["renderer_sha256"]
    if expect_renderer and ren != expect_renderer:
        raise ValueError(f"{path}: renderer {ren} != expected {expect_renderer}")
    # (v6 vision) a v2 manifest with a C7 text-replay slice: the replay rows are ALREADY-rendered v1 rows and keep their
    # own renderer hash; the header pins it and ONLY image-less rows may carry it (a vision row must match the header's)
    replay_ren = header.get("replay_renderer_sha256")
    allowed = {ren, replay_ren} if replay_ren else {ren}
    uniq = {}
    for i, r in enumerate(rows):
        if r["renderer_sha256"] not in allowed:
            raise ValueError(f"{path}: row {i} renderer {r['renderer_sha256']} differs from the header")
        if r["renderer_sha256"] != ren and r.get("images"):
            raise ValueError(f"{path}: row {i}: a vision row carrying the REPLAY renderer hash (only text rows can)")
        if sha256_text(r["prompt"]) != r["prompt_sha256"]:
            raise ValueError(f"{path}: row {i} ({r['decision_uid']}) prompt sha256 mismatch")
        keys = list(r["option_order"])
        if len(keys) < 2 or len(set(keys)) != len(keys) or len(keys) > N_LETTERS:
            raise ValueError(f"{path}: row {i}: bad option order")
        loss_of(r)                                   # (v5.1) a malformed loss_by_option fails here
        g = r["gold"]
        if set(g) != set(keys) or any(not math.isfinite(float(v)) or float(v) < 0 for v in g.values()) \
                or abs(sum(float(v) for v in g.values()) - 1) > 1e-6:
            raise ValueError(f"{path}: row {i} ({r['decision_uid']}): gold is not a distribution over the shown options")
        pp = r.get("parent_preservation")
        if pp is not None:
            p = pp["probabilities"]
            if set(p) != set(keys) or abs(sum(p.values()) - 1) > 1e-5 or any(v < 0 for v in p.values()):
                raise ValueError(f"{path}: row {i}: parent probabilities invalid")
            if pp.get("gold_overwritten") is not False:
                raise ValueError(f"{path}: row {i}: parent entry does not certify gold_overwritten=False")
            if r.get("numeric_family"):
                raise ValueError(f"{path}: row {i}: numeric rows carry no parent")
        w = float(r.get("weight", 1.0))
        if not (w >= 0 and math.isfinite(w)):
            raise ValueError(f"{path}: row {i}: bad weight")
        imgs = r.get("images")                                     # (v6 vision) v2 rows: per-image identity + expanded span
        if imgs is not None:
            ly = header.get("image_layout")
            if not ly or not {"boi_id", "soft_id", "eoi_id", "marker_id"} <= set(ly):
                raise ValueError(f"{path}: row {i}: images but no image_layout in the header (v2 manifest?)")
            if len(imgs) > 1:
                raise ValueError(f"{path}: row {i}: vision round 1 is single-image, got {len(imgs)}")
            for im in imgs:
                if not {"sha256", "n_expanded", "span"} <= set(im):
                    raise ValueError(f"{path}: row {i}: image entry needs sha256 / n_expanded / span")
                if int(im["n_expanded"]) != int(im["span"][1]):
                    raise ValueError(f"{path}: row {i}: image span len {im['span'][1]} != n_expanded {im['n_expanded']}")
        prev = uniq.setdefault(r["decision_uid"], w)
        if prev != w:
            raise ValueError(f"{path}: decision {r['decision_uid']} carries two weights")
    tot = float(sum(uniq.values()))
    if abs(tot - float(header["total_decision_weight"])) > 1e-6 * max(1.0, tot):
        raise ValueError(f"{path}: total_decision_weight {header['total_decision_weight']} != sum over unique decisions {tot}")
    return header, rows


def row_vectors(r):
    """letter-order gold t [k], parent q [k] or None, k."""
    keys = r["option_order"]
    t = np.array([float(r["gold"][k]) for k in keys])
    pp = r.get("parent_preservation")
    q = np.array([float(pp["probabilities"][k]) for k in keys]) if pp else None
    return t, q, len(keys)


def lpt(lengths, world):
    """longest-processing-time assignment of items to `world` bins -> list of index lists."""
    bins, load = [[] for _ in range(world)], [0] * world
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        b = min(range(world), key=lambda j: (load[j], j))
        bins[b].append(i)
        load[b] += lengths[i]
    return bins


def plan_step(items, lengths, world, pack_tokens):
    """one global step's exposures -> per rank a list of micro-batches (lists of exposure indices); every rank gets the SAME number
    of micro-batches (None = a dummy forward, zero weight), as FSDP's collectives require."""
    share = lpt([lengths[i] for i in items], world)
    per = []
    for b in share:
        its = [items[j] for j in b]
        rows = fm.pack_rows([lengths[i] for i in its], pack_tokens)
        per.append([[its[j] for j in row] for row in rows])
    m = max(len(p) for p in per)
    return [p + [None] * (m - len(p)) for p in per]


# ---------------------------------------------------------------------------------------------------------------------- utils
class _CheckpointExit(Exception):
    """(v6.1) the watcher's clean between-steps exit: SIGTERM / SIGINT (the wrapper's pre-wipe handshake) or the EC2
    spot notice. The step in flight completes first; the save is at MOST one step behind (never torn)."""

    def __init__(self, step, why):
        super().__init__(f"checkpoint-exit at step {step}: {why}")
        self.step, self.why = int(step), str(why)


class Dist:
    def __init__(self, on, rank, world, dev):
        self.on, self.rank, self.world, self.dev = on, rank, world, dev

    def sum_(self, tensors):
        if not self.on:
            return tensors
        flat = torch.cat([t.reshape(-1).float() for t in tensors]).to(self.dev)
        torch.distributed.all_reduce(flat)
        o = 0
        for t in tensors:
            n = t.numel()
            t.copy_(flat[o:o + n].view_as(t).to(t.dtype))
            o += n
        return tensors

    def gather_obj(self, x):
        if not self.on:
            return [x]
        out = [None] * self.world
        torch.distributed.all_gather_object(out, x)
        return out


def clip_grads(params, max_norm, dist):
    """global L2 clip over (sharded) gradients; works for GPU and CPU-offloaded DTensor grads (local squares, one all-reduce)."""
    gs = [local(p.grad) for p in params if p.grad is not None]
    per_dev = {}
    for g in gs:
        s = g.double().pow(2).sum()
        per_dev[g.device] = per_dev[g.device] + s if g.device in per_dev else s
    sq = sum((v.cpu() for v in per_dev.values()), torch.zeros((), dtype=torch.float64)).reshape(1)
    if dist.on:
        t = sq.to(dist.dev)
        torch.distributed.all_reduce(t)
        sq = t.cpu()
    norm = float(sq.sqrt())
    coef = max_norm / (norm + 1e-6)
    if coef < 1:
        for g in gs:
            g.mul_(coef)
    return norm


def lr_at(step, base, warmup, steps, floor=0.0):
    w = min(1.0, (step + 1) / max(1, warmup))
    c = 0.5 * (1 + math.cos(math.pi * min(step, steps) / max(1, steps)))
    return base * w * (floor + (1 - floor) * c)


def gib(x):
    return round(x / 2 ** 30, 3)


class MemTrace:
    """smoke: allocated GiB and the peak since the previous mark, at each layer's forward end and backward entry / exit (rank 0, the
    measured step). The overall peak is kept across the resets."""

    def __init__(self, dev):
        self.dev, self.on, self.in_fwd, self.rows, self.max = dev, False, False, [], 0

    def mark(self, name):
        if not self.on:
            return
        if self.dev.type == "cuda":
            p = torch.cuda.max_memory_allocated(self.dev)
            self.max = max(self.max, p)
            self.rows.append([name, gib(torch.cuda.memory_allocated(self.dev)), gib(p)])
            torch.cuda.reset_peak_memory_stats(self.dev)
        else:
            self.rows.append([name, None, None])

    def peak(self):
        return max(self.max, torch.cuda.max_memory_allocated(self.dev)) if self.dev.type == "cuda" else 0


TRACE = None


def attach_trace(model, text, tr):
    """forward hooks: model start / end, each decoder layer's end; on the ORIGINAL forward only (not checkpoint recomputes), a gradient
    hook on each layer's output ("bwd> Li": backward enters layer i; its row's peak covers layer i+1's backward) and on layer 0's
    input ("bwd< L0": layer 0 done). Layer i's input is layer i-1's output, so one hook per boundary."""
    def pre(mod, args):
        tr.in_fwd = torch.is_grad_enabled()
        tr.mark("fwd start")

    def post(mod, args, out):
        tr.mark("fwd end")
        tr.in_fwd = False
    model.register_forward_pre_hook(pre)
    model.register_forward_hook(post)
    for i, layer in enumerate(text.layers):
        def hook(mod, args, out, i=i):
            if not (tr.on and tr.in_fwd):
                return
            tr.mark(f"fwd L{i}")
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(o, torch.Tensor) and o.requires_grad:
                o.register_hook(lambda g, i=i: tr.mark(f"bwd> L{i}"))
            x = args[0] if args and i == 0 else None
            if isinstance(x, torch.Tensor) and x.requires_grad:
                x.register_hook(lambda g, i=i: tr.mark(f"bwd< L{i}"))
        layer.register_forward_hook(hook)


def full_of(p, dev):
    """the full tensor of a (possibly sharded) parameter. A CPU-resident shard (--optim adamw-cpu) is gathered on the GPU (NCCL), not
    through gloo."""
    if not hasattr(p, "full_tensor"):
        return p.detach()
    loc = p.to_local()
    if loc.device.type == "cpu" and dev.type == "cuda":
        from torch.distributed.tensor import DTensor
        p = DTensor.from_local(loc.to(dev), p.device_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride())
    return p.full_tensor()


# ---------------------------------------------------------------------------------------------------------------------- build
def vision_identity(v, base_dir):
    """(--vision) the live tower's fqn -> sha256(shape|bytes) map, verified against the same map built straight off the
    checkpoint (see build_vision's record); a silent non-frozen tower is the round's worst failure, so the export's
    "tower byte-identical" claim rests on this. v = the build_vision dict (or the checkpoint's own when v is None)."""
    m = v["tower"]
    proj = v["projector"]
    pw = proj.embedding_projection.weight
    got = {"model.embed_vision.embedding_projection.weight": (tuple(pw.shape), _bytes_sha(pw.detach()))}
    for n, t in m.state_dict().items():
        got["model.vision_tower." + n] = (tuple(t.shape), _bytes_sha(t.detach()))
    return got


def _bytes_sha(t):
    import numpy as np
    t = t.detach().cpu().contiguous()
    nf = t.to(torch.float32).numpy() if t.is_floating_point() else t.numpy()
    h = hashlib.sha256(str(tuple(t.shape)).encode())
    h.update(nf.tobytes())
    return h.hexdigest()


def checkpoint_vision_identity(base_dir):
    """the same map computed straight from the checkpoint's safetensors (the reference the load is asserted against)."""
    from safetensors import safe_open
    idx = weight_map(base_dir)
    got = {}
    by_file = {}
    for k, f in idx.items():
        if not k.startswith(TEXT_PREFIX) and (k.startswith("model.vision_tower.") or k.startswith("model.embed_vision.")):
            by_file.setdefault(f, []).append(k)
    for f, keys in sorted(by_file.items()):
        with safe_open(os.path.join(base_dir, f), framework="pt") as sf:
            for k in keys:
                t = sf.get_tensor(k)
                got[k] = (tuple(t.shape), _bytes_sha(t.float()))
    return got


def build_vision(base_dir, text_cfg, dist):
    """(--vision) the FROZEN Gemma-4 vision tower + multimodal projector, loaded from the same checkpoint as the text stack
    (the index's non-text tensors: model.vision_tower.* + model.embed_vision.*). Returned as a plain dict of modules -- NOT
    children of the Decider -- so named_parameters() / .to_empty() / FSDP / the optimizer never touch them. Loads in fp32,
    requires_grad_(False), eval(); calls verify the byte identity against the checkpoint and record it (init record)."""
    import torch.nn as nn
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder, Gemma4TextConfig
    hcfg = AutoConfig.from_pretrained(base_dir)
    vcfg = getattr(hcfg, "vision_config", None)
    if vcfg is None:
        raise SystemExit(f"--vision: {base_dir} carries no vision_config (the tower must come from the same checkpoint)")
    with torch.device("meta"):
        tower = Gemma4VisionModel(vcfg)
        proj = Gemma4MultimodalEmbedder(vcfg, text_cfg)   # pre-projection RMSNorm + embedding_projection (the checkpoint's own)
    tower.to_empty(device="cpu")
    proj.to_empty(device="cpu")
    tower.float()
    proj.float()
    idx = weight_map(base_dir)
    ckpt = {}
    by_file = {}
    for k, f in idx.items():
        if k.startswith("model.vision_tower.") or k.startswith("model.embed_vision."):
            ckpt[k] = None
            by_file.setdefault(f, []).append(k)
    if not ckpt:
        raise SystemExit(f"--vision: no vision_tower / embed_vision tensors in {base_dir}")
    for f, keys in sorted(by_file.items()):
        from safetensors import safe_open
        with safe_open(os.path.join(base_dir, f), framework="pt") as sf:
            for k in keys:
                ckpt[k] = sf.get_tensor(k).float().cpu()
    with torch.no_grad():
        for n, buf in tower.state_dict().items():
            full = "model.vision_tower." + n
            if full not in ckpt:
                raise SystemExit(f"--vision: the tower wants {n}, the checkpoint has no tensor {full}")
            buf.copy_(ckpt[full].view_as(buf))
        pk = [k for k in ckpt if k.endswith("embedding_projection.weight")]
        if len(pk) != 1:
            raise SystemExit(f"--vision: expected one embedding_projection tensor, found {pk}")
        proj.embedding_projection.weight.copy_(ckpt[pk[0]].view_as(proj.embedding_projection.weight))
        # every checkpoint vision tensor accounted for (load is strict, both directions)
        miss = sorted(set(ckpt) - {f"model.vision_tower.{n}" for n in tower.state_dict()} - set(pk))
        if miss:
            raise SystemExit(f"--vision: checkpoint vision tensors unused by the tower: {miss[:4]}")
    tower.eval()
    proj.eval()
    for m in (tower, proj):
        for p in m.parameters():
            p.requires_grad_(False)
    # (v6.1) place the frozen tower + projector on the training device: they are loaded/verified on CPU (byte identity
    # is checked against checkpoint bytes at load time, BEFORE this move) and the dict-held modules are outside FSDP,
    # so nothing else ever moves them; the GPU run_step forward needs them beside `emb` (attempt-3 Gate 0 died on
    # "mat2 is on cpu" exactly here).
    if dist.dev is not None:
        tower.to(dist.dev)
        proj.to(dist.dev)
    if dist.rank == 0:
        print(f"vision tower frozen: {len(tower.state_dict())} tower tensors + projector "
              f"{tuple(proj.embedding_projection.weight.shape)} from {base_dir}", flush=True)
    return {"tower": tower, "projector": proj, "dtype": torch.float32, "base_dir": base_dir}


def build(a, dist):
    from transformers import AutoConfig
    if a.attn == fm.ATTN_NAME:
        fm.register()
        fm.ATTN_IMPL, fm.DEFAULT_CHUNK = a.attn_impl, int(a.attn_chunk)
    if a.tiny_dir or a.init_from:
        mdir = a.tiny_dir or a.init_from
        cfg = AutoConfig.from_pretrained(mdir).get_text_config()
    elif a.tiny:
        mdir, cfg = None, tiny_text_config(a.tiny_layers, a.tiny_window, a.tiny_family, a.tiny_kv_shared)
    else:
        mdir = None if a.smoke else snapshot_dir(a.model, a.revision)
        cfg = AutoConfig.from_pretrained(mdir or a.model, revision=None if mdir else a.revision).get_text_config()
    if a.vision and (a.smoke or mdir is None):
        # Gate 0 (the plan's 40-min 2-GPU smoke) runs on the REAL manifest slice with --max-steps, not --smoke: SyntheticTokens
        # has no images, and the vision splice's shape contract is checkpoint-real only
        raise SystemExit("--vision needs a checkpoint dir (the pinned snapshot / --init-from / --tiny-dir) for the tower, and not --smoke")
    if a.vision and a.attention_policy != "V":
        raise SystemExit("--vision needs --attention-policy V (the image-span mask is the stock behaviour; causal would confound the round)")
    cfg._attn_implementation = a.attn
    prior = getattr(cfg, fm.POLICY_ATTR, None)
    if a.attention_policy:
        if prior is not None and prior != a.attention_policy and not (a.vision and prior == "B" and a.attention_policy == "V"):
            raise SystemExit(f"the init config carries attention policy {prior}; --attention-policy {a.attention_policy} refused")
        if prior == "B" and a.attention_policy == "V" and a.vision:
            print("fullft: vision round 1 continues a policy-B checkpoint at policy V (superset; no image rows = identical masks)",
                  flush=True)
        fm.set_policy(cfg, a.attention_policy)
    elif prior not in (None, "C"):
        raise SystemExit(f"the init config carries attention policy {prior}: pass --attention-policy {prior} explicitly")
    fm.require_policy(cfg, fm.policy_of(cfg))
    a.init_record = stock_only_check(a, mdir) if a.stock_only else {"stock_only": False, "dir": mdir}
    if mdir is None:
        letter_ids = list(range(10, 10 + N_LETTERS))
        tok = None
    else:
        from transformers import AutoTokenizer
        a.tokenizer_dir = tokenizer_dir(a, mdir)
        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
        ids = [tok(c, add_special_tokens=False)["input_ids"] for c in LABELS]
        if any(len(x) != 1 for x in ids):
            raise SystemExit("a letter is not a single token")
        letter_ids = [x[0] for x in ids]
    with torch.device("meta"):
        text = text_classes(cfg)[0](cfg)
        model = Decider(text, letter_ids, a.attn)
    if a.vision:
        model.vision = build_vision(mdir, cfg, dist)
    if a.freeze_embed:
        text.embed_tokens.weight.requires_grad_(False)
    if a.ac:
        fm.enable_checkpointing(text, offload=a.offload_ac, min_tokens=a.offload_min, force_cpu=a.force_cpu_offload)
    if a.mlp_chunk:
        fm.chunk_mlps(text, a.mlp_chunk, a.mlp_chunk_min)
    cuda = dist.dev.type == "cuda"
    if dist.on:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
        mesh = init_device_mesh("cuda" if cuda else "cpu", (dist.world,))
        pdt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.param_dtype if cuda else "fp32"]
        rdt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.reduce_dtype]
        if a.test_float64:
            pdt = rdt = torch.float64
        mp = MixedPrecisionPolicy(param_dtype=pdt, reduce_dtype=rdt)
        kw = {"mesh": mesh, "mp_policy": mp}
        if a.optim == "adamw-cpu":
            kw["offload_policy"] = CPUOffloadPolicy(pin_memory=cuda)
        for layer in text.layers:
            fully_shard(layer, **kw)
        fully_shard(model, **kw)
        # FSDP averages gradients over ranks; the loss is already divided by the GLOBAL step weight, so scale it by the world size
        # (a power of two on p8: exact in binary floating point). Not set_gradient_divide_factor(1.0): its PREMUL_SUM reduce is a
        # backend-specific op (gloo refuses it) and must be set on every FSDP module, not only the root.
        dist.loss_scale = float(dist.world)
    else:
        dist.loss_scale = 1.0
    pdev = torch.device("cpu") if a.optim == "adamw-cpu" else dist.dev
    model.to_empty(device=pdev)
    if pdev != dist.dev:                                 # host-resident params: buffers still live on the GPU
        for mod in model.modules():
            for bn, b in list(mod._buffers.items()):
                if b is not None:
                    mod._buffers[bn] = b.to(dist.dev)
    model.double() if a.test_float64 else model.float()
    scal = None
    if mdir is None or a.smoke:
        with torch.no_grad():
            g = torch.Generator(device=pdev).manual_seed(1234 + dist.rank)
            for p in model.parameters():
                local(p).normal_(0.0, 0.02, generator=g)
    else:
        n, scal = load_pretrained(model, mdir, cfg)
        if dist.rank == 0:
            print(f"loaded {n} text tensors from {mdir}", flush=True)
    nb = rebuild_buffers(text, cfg, dist.dev, scal)
    with torch.no_grad():                                # to_empty left the readout's letter-id buffer uninitialised too
        model.letters.copy_(torch.as_tensor(letter_ids, dtype=torch.long, device=model.letters.device))
    if model.letters.tolist() != list(letter_ids):
        raise SystemExit("fullft: letter ids not set")
    if a.vision:
        expect = checkpoint_vision_identity(mdir)
        got = vision_identity(model.vision, mdir)
        if got != expect:
            badk = [k for k in expect if expect.get(k) != got.get(k)][:4]
            raise SystemExit(f"fullft --vision: tower not byte-identical to the checkpoint, e.g. {badk}")
        a.init_record["vision"] = {"tower_dir": mdir, "identity": {k: got[k][1] for k in sorted(got)},
                                   "frozen": True,
                                   "note": "tower + projector outside named_parameters (a plain dict), byte-identical to the checkpoint"}
    return model, text, cfg, tok, letter_ids, mdir, nb + 1


def make_optimizer(a, params):
    if a.optim in ("adamw", "adamw-cpu"):
        kw = {"foreach": True} if a.optim == "adamw" else {}
        return torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay, **kw)
    if a.optim == "adamw-fused":
        return torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay, fused=True)
    if a.optim == "adamw-bf16":
        return fm.LowMemAdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay, seed=a.seed)
    raise SystemExit(f"--optim {a.optim}")


# --------------------------------------------------------------------------------------------------------------------- steps
def micro_inputs(rows, toks, dev, pad_id):
    """packed row of the given exposures -> ids, pos, last, t [n, 52], q [n, 52], has_parent [n], ks [n], w [n],
    and (vision rows only) spans [1, T] group channel + images [(i0, K, pixel_tensor), ...] in EXPANDED coordinates."""
    ids, pos, last = fm.packed_batch([toks[i] for i in rows], pad_id=pad_id)
    n = len(rows)
    t = torch.zeros(n, N_LETTERS)
    q = torch.zeros(n, N_LETTERS)
    hp = torch.zeros(n, dtype=torch.bool)
    ks = torch.zeros(n, dtype=torch.long)
    w = torch.zeros(n)
    spans = torch.zeros(1, ids.shape[1], dtype=torch.long) if getattr(toks, "n_vision", 0) else None
    images = [] if getattr(toks, "n_vision", 0) else None
    off = 0
    for j, i in enumerate(rows):
        tt, qq, k, ww = toks.vectors[i]
        t[j, :k] = torch.as_tensor(tt)
        if qq is not None:
            q[j, :k] = torch.as_tensor(qq)
            hp[j] = True
        ks[j], w[j] = k, ww
        if images is not None:
            for (g, (s0, K, sha)) in enumerate(toks.image_at(i)):
                spans[0, off + s0:off + s0 + K] = j + 1            # group = the doc index in this packed row
                images.append((off + s0, K, toks.load_pixels(sha)))
            off += toks.lengths[i]
    if spans is None:
        return [x.to(dev) for x in (ids, pos, last, t, q, hp, ks, w)]
    pv = [(i0, K, px.to(dev) if hasattr(px, "to") else px) for i0, K, px in images]
    return [x.to(dev) for x in (ids, pos, last, t, q, hp, ks, w)] + [spans.to(dev), pv]


class Tokens:
    """token ids of the manifest rows, tokenised lazily (checked against n_tokens); the letter-order vectors.

    (vision v6) v2 rows (images present) are EXPANDED here: each image marker token (boi, e.g. id 258880 for Gemma-4 -- the
    id pinned in the manifest header's image_layout) is replaced by the recorded run [boi, soft*K, eoi]; `n_tokens` in v2
    is the EXPANDED count, so the length check stays failing closed. `image_at(i)` -> list of (start, K, pixel_sha) in
    expanded-id coordinates; `load_pixels(shas)` reads the staged processor output tensors the splice needs."""

    IMAGE_LAYOUT_KEYS = ("boi_id", "soft_id", "eoi_id", "marker_id")

    def __init__(self, rows, tok, images_dir=None, layout=None, limit_image_cache=8):
        self.rows, self.tok, self.cache = rows, tok, {}
        self.vectors = []
        self.images_dir, self._pix_cache, self._pix_limit = images_dir, {}, limit_image_cache
        self.layout = layout
        self.n_vision = 0
        for r in rows:
            imgs = r.get("images")
            if imgs and layout is None:
                raise ValueError(f"{r.get('decision_uid')}: vision row but no image layout in the header (v2 manifest?)")
            if imgs:
                self.n_vision += 1
            t, q, k = row_vectors(r)
            self.vectors.append((t, q, k, float(r.get("weight", 1.0))))
        self.lengths = [int(r["n_tokens"]) for r in rows]

    def _expand(self, ids, imgs):
        """the marker runs in TEXT ids -> [boi, soft*K, eoi] (left to right); the span coordinates are recomputed and checked
        against the manifest's recorded spans (the length check stays failing closed)."""
        ly = self.layout
        mark = ly["marker_id"]
        out, spans, ti = [], [], 0
        for im in imgs:
            K = int(im["n_expanded"])
            j = ids.index(mark, ti)
            out.extend(ids[ti:j])
            s0 = len(out) + 1                       # the soft run; boi / eoi stay ordinary tokens
            out += [ly["boi_id"]] + [ly["soft_id"]] * K + [ly["eoi_id"]]
            spans.append((s0, K))
            ti = j + 1
        out.extend(ids[ti:])
        return out, spans

    def image_at(self, i):
        """(start, K, pixel_sha) of every image of row i, EXPANDED coordinates."""
        imgs = self.rows[i].get("images")
        if not imgs:
            return []
        ids = self.tok(self.rows[i]["prompt"], add_special_tokens=False)["input_ids"]
        _, spans = self._expand(ids, imgs)
        rec = [(int(im["span"][0]), int(im["span"][1]), im["sha256"]) for im in imgs]
        if spans != [(s, k) for s, k, _ in rec]:
            raise ValueError(f"row {i}: span {spans} != the manifest's recorded {rec}")
        return rec

    def __getitem__(self, i):
        x = self.cache.get(i)
        if x is None:
            r = self.rows[i]
            x = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
            if r.get("images"):
                x, _ = self._expand(x, r["images"])
            if len(x) != self.lengths[i]:
                raise ValueError(f"row {i} ({r['decision_uid']}): {len(x)} tokens, the manifest says {self.lengths[i]}")
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[i] = x
        return x

    def load_pixels(self, sha):
        p = self._pix_cache.get(sha)
        if p is None:
            if not self.images_dir:
                raise ValueError(f"image {sha[:8]}: no --vision-images dir staged")
            import torch as _t
            p = _t.load(os.path.join(self.images_dir, sha + ".pt"), weights_only=False)
            self._pix_cache[sha] = p
            if len(self._pix_cache) > self._pix_limit:
                self._pix_cache.pop(next(iter(self._pix_cache)))
        return p


class SyntheticTokens:
    """smoke: `n` documents of `length` tokens each, random ids, random soft targets over k options, half with a parent."""

    def __init__(self, n, length, vocab, seed=0, k=4):
        g = np.random.default_rng(seed)
        self.ids = [g.integers(0, vocab, length).tolist() for _ in range(n)]
        self.lengths = [length] * n
        self.vectors = []
        for i in range(n):
            t = g.dirichlet(np.ones(k))
            q = g.dirichlet(np.ones(k)) if i % 2 == 0 else None
            self.vectors.append((t, q, k, 1.0))

    def __getitem__(self, i):
        return self.ids[i]


def run_step(a, model, head, opt, plan_rows, toks, dist, W_step, pad_id, train=True):
    """one optimizer step's forwards / backwards on this rank -> stats dict (sums over this rank's decisions)."""
    st = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "n": 0, "tokens": 0, "head_negbound": None}
    for rows in plan_rows:
        if rows is None:                       # dummy forward: FSDP's collectives need the same count on every rank
            ids = torch.full((1, 16), pad_id, dtype=torch.long, device=dist.dev)
            pos = torch.arange(16, device=dist.dev)[None]
            h, ell = model(ids, pos, torch.tensor([15], device=dist.dev))
            (ell.sum() * 0.0).backward()
            continue
        ids, pos, last, t, q, hp, ks, w, spans, imgs = micro_inputs(rows, toks, dist.dev, pad_id)
        h, ell = model(ids, pos, last, spans=spans, images=imgs)
        loss_i, ce_i, kl_i = fh.decision_losses(ell, ks, t, q, hp, a.alpha, a.ineligible_gold_weight)
        L = (w * loss_i).sum() / W_step * dist.loss_scale
        if TRACE is not None:
            TRACE.mark("loss")
        L.backward()
        if TRACE is not None:
            TRACE.mark("bwd end")
        if head is not None:
            nb = head.data_backward(h, ell, t, ks, w, W_step)
            st["head_negbound"] = nb if st["head_negbound"] is None else st["head_negbound"] + nb
        st["loss"] += float((w * loss_i).sum().detach())
        st["ce"] += float((w * ce_i).sum())
        st["kl"] += float((w * kl_i).sum())
        st["n"] += len(rows)
        st["tokens"] += int(ids.shape[1])
    return st


def evaluate(a, model, head, toks, idx, dist, pad_id, samples):
    """forward-only over `idx` (sharded over ranks, padded with dummies) -> rank 0: per-decision readouts in index order."""
    model.eval()
    plan = plan_step(idx, toks.lengths, dist.world, a.pack_tokens)[dist.rank]
    res = []
    with torch.no_grad():
        for rows in plan:
            if rows is None:
                ids = torch.full((1, 16), pad_id, dtype=torch.long, device=dist.dev)
                model(ids, torch.arange(16, device=dist.dev)[None], torch.tensor([15], device=dist.dev))
                continue
            ids, pos, last, t, q, hp, ks, w, spans, imgs = micro_inputs(rows, toks, dist.dev, pad_id)
            h, ell = model(ids, pos, last, spans=spans, images=imgs)
            ro = fh.readouts(head.head, h, ell, ks, samples) if head is not None else None
            for j, i in enumerate(rows):
                k = int(ks[j])
                r = {"i": int(i), "k": k, "t": t[j, :k].tolist(), "native": None}
                lp, _ = fh.masked_logp(ell[j:j + 1], ks[j:j + 1])
                r["native"] = lp.exp()[0, :k].tolist()
                if ro is not None:
                    r["mean"] = ro["mean"][:, j, :k].tolist()
                    r["predictive"] = ro["predictive"][:, j, :k].tolist()
                    r["epi"] = ro["epi"][:, j].tolist()
                res.append(r)
    model.train()
    allr = dist.gather_obj(res)
    if dist.rank != 0:
        return None
    out = sorted([r for rr in allr for r in rr], key=lambda r: r["i"])
    if [r["i"] for r in out] != sorted(idx):
        raise RuntimeError("eval: rows lost or duplicated across ranks")
    return out


def eval_metrics(res):
    def nll(p, t):
        return float(-(np.asarray(t) * np.log(np.clip(np.asarray(p), 1e-12, 1))).sum())
    m = {"n": len(res), "native_nll": float(np.mean([nll(r["native"], r["t"]) for r in res]))}
    m["native_acc"] = float(np.mean([np.asarray(r["t"])[int(np.argmax(r["native"]))] >= max(r["t"]) - 1e-9 for r in res]))
    if res and "predictive" in res[0]:
        G = len(res[0]["predictive"])
        m["predictive_nll"] = [float(np.mean([nll(r["predictive"][g], r["t"]) for r in res])) for g in range(G)]
        m["mean_nll"] = [float(np.mean([nll(r["mean"][g], r["t"]) for r in res])) for g in range(G)]
    return m


# ----------------------------------------------------------------------------------------------------------------------- main
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=BASE_REPO)
    ap.add_argument("--revision", default=BASE_REV)
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="",
                    help="(v5) a pinned stock backbone: sets --model / --revision (conflicting values refused); default = the 31B pins")
    ap.add_argument("--tiny-family", choices=sorted(TEXT_FAMILIES), default="gemma4_text", help=argparse.SUPPRESS)
    ap.add_argument("--tiny-kv-shared", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--init-from", default="", help="a local HF dir to load the text weights from (default: the pinned snapshot)")
    ap.add_argument("--pinned-code", default=HERE, help="dir holding the pinned production claude_lmtune.py (AuxVBLL)")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--expect-rows-sha256", default="", help="(v4.2) the FROZEN digest the training file's bytes must have (refused otherwise)")
    ap.add_argument("--allow-code-change-on-resume", action="store_true",
                    help="(v4.2) resume a run whose resume shard was written by other trainer code (recorded in train_record)")
    ap.add_argument("--val", default="")
    ap.add_argument("--val-max", type=int, default=2000)
    ap.add_argument("--expect-renderer", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decisions-per-step", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the exposure manifest (1 = every exposure once)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-floor", type=float, default=0.1, help="cosine floor as a fraction of --lr")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.15, help="parent-preservation weight on eligible rows (spec 0.15; recipe C: 0)")
    ap.add_argument("--ineligible-gold-weight", choices=("one", "1-alpha"), default="one")
    ap.add_argument("--head", choices=("on", "off"), default="on", help="off ONLY for a deterministic-head-free control")
    ap.add_argument("--head-prior", default=fh.GRID["prior"])
    ap.add_argument("--head-kl", default=fh.GRID["kl"])
    ap.add_argument("--head-cov", default=fh.GRID["cov"])
    ap.add_argument("--head-dim", type=int, default=fh.GRID["dim"])
    ap.add_argument("--head-lr", type=float, default=fh.GRID["lr"])
    ap.add_argument("--head-samples", type=int, default=fh.GRID["samples"])
    ap.add_argument("--attn", default=fm.ATTN_NAME, choices=(fm.ATTN_NAME, "sdpa", "eager"))
    ap.add_argument("--attn-impl", choices=("kernels", "chunked_math"), default="kernels",
                    help="jev_mem: SDPA's memory-efficient / flash kernels, or the chunked-math fallback (each chunk checkpointed)")
    ap.add_argument("--attn-chunk", type=int, default=2048, help="jev_mem query chunk (sliding / packed / chunked_math)")
    ap.add_argument("--ac", type=int, default=1)
    ap.add_argument("--offload-ac", type=int, default=0)
    ap.add_argument("--offload-min", type=int, default=8192)
    ap.add_argument("--force-cpu-offload", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mlp-chunk", type=int, default=0)
    ap.add_argument("--mlp-chunk-min", type=int, default=16384)
    ap.add_argument("--optim", choices=("adamw", "adamw-fused", "adamw-bf16", "adamw-cpu"), default="adamw")
    ap.add_argument("--evict-states-min", type=int, default=0,
                    help="move the optimizer moments to pinned host memory for steps whose longest row has >= N tokens (0 = never)")
    ap.add_argument("--mem-snapshot", default="", help="smoke: rank 0 CUDA memory history of the measured step -> this pickle")
    ap.add_argument("--mem-warn-gib", type=float, default=74.0, help="(v4) print MEM_WARN when a step's live peak (max over ranks) exceeds this")
    ap.add_argument("--freeze-embed", action="store_true")
    ap.add_argument("--param-dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--reduce-dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--pack-tokens", type=int, default=16384)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-at-start", action="store_true", help="also evaluate --val before the first step (step 0 = the init)")
    ap.add_argument("--save-every", type=int, default=250, help="resume shards every N steps and at the end (0 = never)")
    ap.add_argument("--no-export", action="store_true", help="skip the final HF export (tests)")
    ap.add_argument("--keep-resume", type=int, default=2, help="newest resume dirs kept locally (0 = all)")
    ap.add_argument("--resume", default="", help="resume dir root (default --out)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = the schedule's)")
    # smoke
    ap.add_argument("--smoke", action="store_true", help="memory / speed smoke: random weights, synthetic rows, --steps steps")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--docs-per-row", type=int, default=1)
    ap.add_argument("--rows-per-rank", type=int, default=1, help="smoke: packed rows (micro-batches) per rank per step")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--tiny-layers", type=int, default=6)
    ap.add_argument("--tiny-window", type=int, default=48)
    ap.add_argument("--tiny-dir", default="", help="tests: a tiny HF Gemma-4 checkpoint dir (weights + tokenizer)")
    ap.add_argument("--device", default="")
    ap.add_argument("--test-float64", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dump-grads", default="", help=argparse.SUPPRESS)
    ap.add_argument("--attention-policy", choices=fm.POLICIES, default=None,
                    help="(v5) C = causal; B = global layers bidirectional within their document, local layers causal-windowed; "
                         "(v6) V = B + image-span bidirectional inside each span (stock vision behaviour; --vision)")
    ap.add_argument("--vision", action="store_true",
                    help="(v6) vision round 1: rows may carry images; the tower + projector load FROZEN from the init checkpoint "
                         "(byte-identity asserted + recorded); requires --attention-policy V")
    ap.add_argument("--vision-images", default="", help="(v6) the staged processor-output dir the splice reads: <sha256>.pt per image")
    ap.add_argument("--stock-only", action="store_true", help="(v5) refuse every initialisation but the pinned stock snapshot")
    ap.add_argument("--preset", choices=("", "sdv1_pilot"), default="", help="(v5) fixed settings; conflicting flags are refused")
    ap.add_argument("--max-row-tokens", type=int, default=0, help="(v5) refuse manifest rows with more tokens (0 = off)")
    ap.add_argument("--k-range", default="", help="(v5) 'LO,HI': refuse manifest rows whose K is outside")
    ap.add_argument("--extract-features", default="", help="(v5) forward only: h / ell / K / t / w of every --manifest row -> this file")
    ap.add_argument("--bundle", action="append", default=[], help="(v5.2) ROLE=MANIFEST[@ROWS_SHA256][#RENDERER]: at the end of a completed run "
                    "write OUT/bundles/ROLE.pt (the --extract-features format) from the trained model in memory")
    ap.add_argument("--bundle-partial", action="store_true", help="(v5.2) also write the bundles when --max-steps stops before the schedule's end")
    a = ap.parse_args(argv)
    if a.backbone:
        b = BACKBONES[a.backbone]
        for k in ("model", "revision"):
            if getattr(a, k) not in (ap.get_default(k), b["repo" if k == "model" else "revision"]):
                raise SystemExit(f"--backbone {a.backbone} fixes --{k} {b['repo' if k == 'model' else 'revision']}")
        a.model, a.revision = b["repo"], b["revision"]
    if a.preset:
        a = apply_preset(a, ap)
    return a


def apply_preset(a, ap):
    """--preset sdv1_pilot: every SDV1_PILOT value applied; a flag the user set to another value is refused (never silently
    overridden); an explicit --attention-policy and (real data) --expect-rows-sha256 are required"""
    bad = [k for k, v in SDV1_PILOT.items() if getattr(a, k) != ap.get_default(k) and getattr(a, k) != v]
    if bad:
        raise SystemExit(f"--preset {a.preset} fixes {', '.join(f'--{k.replace(chr(95), chr(45))}={SDV1_PILOT[k]}' for k in bad)}; "
                         "conflicting values refused")
    for k, v in SDV1_PILOT.items():
        setattr(a, k, v)
    if a.attention_policy is None:
        raise SystemExit(f"--preset {a.preset} needs an explicit --attention-policy ({' | '.join(fm.POLICIES)})")
    if not (a.smoke or a.tiny or a.tiny_dir) and not a.expect_rows_sha256:
        raise SystemExit(f"--preset {a.preset} needs --expect-rows-sha256 (the frozen manifest digest)")
    return a


def stock_only_check(a, mdir):
    """(v5) the student's initialisation must be the pinned STOCK snapshot: no --init-from, the pinned repo / revision, the snapshot
    dir named by that revision, no head / adapter files, no exported fine-tune marker, no contender name in the path. Tests: a
    --tiny-dir equal to $JEV_STOCK_ONLY_FIXTURE stands in for the snapshot (recorded as a fixture). -> the record's init entry."""
    if a.init_from:
        raise SystemExit(f"--stock-only: --init-from {a.init_from} refused (only the pinned stock snapshot initialises the student)")
    pins = [(b["repo"], b["revision"]) for b in BACKBONES.values()]
    if (a.model, a.revision) not in pins:
        raise SystemExit(f"--stock-only: model {a.model}@{a.revision} is not a pinned stock backbone {pins}")
    if a.smoke:
        return {"stock_only": True, "kind": "smoke (random weights, the stock config)"}
    if a.tiny and not a.tiny_dir:
        raise SystemExit("--stock-only: --tiny builds a random model, not the stock snapshot")
    if a.tiny_dir:
        fx = os.environ.get("JEV_STOCK_ONLY_FIXTURE", "")
        if not fx or os.path.realpath(fx) != os.path.realpath(a.tiny_dir):
            raise SystemExit(f"--stock-only: {a.tiny_dir} is not the stock snapshot")
        kind = "test fixture ($JEV_STOCK_ONLY_FIXTURE)"
    else:
        rp = os.path.realpath(mdir)
        if os.path.basename(rp) != a.revision:
            raise SystemExit(f"--stock-only: {mdir} is not the snapshot dir of revision {a.revision}")
        kind = "pinned snapshot"
    for f in ("vbll_head.json", "vbll_head.safetensors", "vbll_head_grid.pt", "adapter_config.json", "adapter_model.safetensors"):
        if os.path.exists(os.path.join(mdir, f)):
            raise SystemExit(f"--stock-only: {mdir} holds {f} (a head / adapter: not the stock model)")
    cj = json.load(open(os.path.join(mdir, "config.json")))
    tc = cj.get("text_config") or {}
    if "jev_requires_head" in cj or "jev_requires_attention" in cj or fm.POLICY_ATTR in cj or fm.POLICY_ATTR in tc:
        raise SystemExit(f"--stock-only: {mdir} is an exported fine-tune (jev markers in config.json)")
    parts = [x for x in re.split(r"[/\\]", os.path.realpath(mdir).lower()) if x]
    hit = [x for x in parts if x == "d3" or any(m in x for m in CONTENDER_MARKERS)]
    if hit:
        raise SystemExit(f"--stock-only: contender marker in the init path: {hit}")
    ix = os.path.join(mdir, "model.safetensors.index.json")
    return {"stock_only": True, "kind": kind, "repo": a.model, "revision": a.revision, "dir": mdir,
            "index_sha256": sha256_file(ix) if os.path.exists(ix) else None,
            "config_sha256": sha256_file(os.path.join(mdir, "config.json"))}


def pg_backend(cuda):
    """NCCL for GPU tensors AND gloo for CPU tensors. v1 created an NCCL-only group: the first collective on a CPU tensor (the
    adamw-cpu shards' full_tensor) failed with "No backend type associated with device type cpu"."""
    return "cpu:gloo,cuda:nccl" if cuda else "cpu:gloo"


def preflight_collectives(dev, world):
    """one all-reduce per device type the run uses (CPU always: host-resident state, gathers): fail in seconds, before the model
    is built, if the process group cannot reduce one of them."""
    for d in [torch.device("cpu")] + ([dev] if dev.type != "cpu" else []):
        t = torch.ones(1, device=d)
        try:
            torch.distributed.all_reduce(t)
        except RuntimeError as e:
            raise SystemExit(f"fullft: the process group cannot reduce {d.type} tensors: {e}")
        if int(t.item()) != world:
            raise SystemExit(f"fullft: {d.type} all-reduce gave {t.item()}, expected {world}")


def setup_dist(a):
    on = int(os.environ.get("WORLD_SIZE", "1")) > 1
    cuda = torch.cuda.is_available() and a.device != "cpu" and not a.tiny
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(dev)
        torch.backends.cuda.enable_cudnn_sdp(False)
    if on:
        torch.distributed.init_process_group(os.environ.get("FULLFT_PG_BACKEND") or pg_backend(cuda))
    rank = torch.distributed.get_rank() if on else 0
    world = torch.distributed.get_world_size() if on else 1
    if on:
        preflight_collectives(dev, world)
    return Dist(on, rank, world, dev)


def main(argv=None):
    a = parse(argv)
    dist = setup_dist(a)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rec = {"version": VERSION, "code_revision": CODE_REVISION, "args": vars(a), "world": dist.world, "torch": torch.__version__, "oom": False,
           "gpu": torch.cuda.get_device_name(dist.dev) if dist.dev.type == "cuda" else "cpu"}
    t0 = time.time()
    model, text, cfg, tok, letter_ids, mdir, nb = build(a, dist)
    rec["params"] = sum(p.numel() for p in model.parameters())
    rec["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rec["buffers_rebuilt"] = nb
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(a, params)
    pad_id = int(cfg.pad_token_id or 0)
    a.data_rows_sha256 = None
    rec["code"] = code_identity(a.pinned_code)
    rec["init"] = a.init_record
    rec["attention_policy"] = fm.policy_of(cfg)
    if a.smoke:
        n_docs = a.rows_per_rank * a.docs_per_row * dist.world
        doc_len = a.length // a.docs_per_row
        toks = SyntheticTokens(n_docs, doc_len, cfg.vocab_size, seed=a.seed)
        header = {"total_decision_weight": float(n_docs) * 1000.0}
        steps, D = a.steps, n_docs
        order = list(range(n_docs))
        pack = a.length
    else:
        header, rows = read_manifest(a.manifest, a.expect_renderer or None)
        got = sha256_file(a.manifest)                    # (v4.2) bind training to the ACTUAL bytes read, not only to the header's claim
        if header.get("rows_sha256") and os.path.exists(a.manifest + ".header.json") and header["rows_sha256"] != got:  # (sidecar header)
            raise SystemExit(f"fullft: {a.manifest} bytes sha256 {got} != its header's rows_sha256 {header['rows_sha256']} (refused)")
        if a.expect_rows_sha256 and got != a.expect_rows_sha256:
            raise SystemExit(f"fullft: {a.manifest} bytes sha256 {got} != the expected (registered) {a.expect_rows_sha256} (refused)")
        a.data_rows_sha256 = got
        rec["data"] = {"rows_sha256": got, "header_rows_sha256": header.get("rows_sha256"), "expected_rows_sha256": a.expect_rows_sha256 or None,
                       "rows": len(rows)}
        check_row_limits(a, rows)
        toks = Tokens(rows, tok, images_dir=getattr(a, "vision_images", None) or None,
                      layout=(header.get("image_layout") or None))
        if a.extract_features:
            extract_features(a, model, toks, rows, dist, pad_id, rec)
            if a.bundle:                             # (v5.2) more pools through the SAME loaded model (one load for every role)
                write_bundles(a, model, tok, dist, pad_id, rec, 0)
            if dist.on:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            return 0
        D = a.decisions_per_step
        rng = np.random.default_rng(a.seed)
        n_ex = int(round(len(rows) * a.epochs))
        order = [int(x) for e in range(int(math.ceil(a.epochs))) for x in rng.permutation(len(rows))][:n_ex]
        steps = int(math.ceil(len(order) / D))
        pack = a.pack_tokens
    sched_steps = steps                                  # the LR / head schedules' length never depends on --max-steps
    if a.preset == "sdv1_pilot" and not (a.smoke or a.tiny_dir) and sched_steps != SDV1_PILOT_STEPS:
        raise SystemExit(f"--preset sdv1_pilot: the manifest gives {sched_steps} steps, the plan fixes {SDV1_PILOT_STEPS} (38,400 exposures)")
    rec["schedule_steps"] = sched_steps
    if a.max_steps:
        steps = min(steps, a.max_steps)
    head = None
    if a.head == "on":
        lt = fh.pinned_trainer(a.pinned_code)
        with torch.no_grad():
            E = full_of(model.text.embed_tokens.weight, dist.dev)
            E = E.index_select(0, torch.as_tensor(letter_ids, device=E.device))
        heads = fh.heads_of(a.head_prior, a.head_kl, a.head_cov)
        head = fh.HeadState(lt, E.detach().float().cpu(), heads, a.head_dim, a.seed, a.head_lr, float(header["total_decision_weight"]),
                            a.warmup, sched_steps, dist.dev)
        rec["head"] = {"heads": heads, "dim": a.head_dim, "W_total": float(header["total_decision_weight"])}
    rec["build_s"] = round(time.time() - t0, 1)
    if dist.rank == 0:
        print(f"{VERSION}: world {dist.world}, params {rec['params']:,} (trainable {rec['trainable']:,}), steps {steps}, "
              f"decisions/step {D}, pack {pack}, attn {a.attn}, ac {a.ac} offload {a.offload_ac}, mlp_chunk {a.mlp_chunk}, "
              f"optim {a.optim}, alpha {a.alpha}, head {a.head}", flush=True)
    cuda = dist.dev.type == "cuda"
    global TRACE
    if a.smoke and dist.rank == 0:
        TRACE = MemTrace(dist.dev)
        attach_trace(model, text, TRACE)
    ev = fm.StateEvictor(opt, pinned=cuda) if a.evict_states_min else None
    times, log = [], []
    mem = {"run_peak": 0, "retries": 0, "warns": 0, "eval_peak": 0}

    # (v6.1 checkpoint insurance) the watcher flags the main loop to checkpoint-and-exit between steps: SIGTERM/SIGINT
    # (the wrapper's pre-wipe / reaper handshake), the EC2 spot notice (IMDSv2), or any unexpected thread error. The
    # save is the trainer's OWN save_resume (resume-identical restart); a signal already inside a step finishes that
    # step first (a clean collectives boundary), so a save is at most one step late, never torn.
    save_flags = {"why": None}

    def _flag(why):
        if save_flags["why"] is None:
            save_flags["why"] = why

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, (lambda s: lambda *_: _flag(f"signal-{s.name}"))(sig))
        except (ValueError, AttributeError):
            pass

    def _spot_watcher():
        import urllib.request
        while True:
            try:
                with urllib.request.urlopen("http://169.254.169.254/latest/api/token", data=b"", timeout=2,
                                             headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}) as tr:
                    tokn = tr.read().decode()
                req = urllib.request.Request("http://169.254.169.254/latest/meta-data/spot/instance-action",
                                             headers={"X-aws-ec2-metadata-token": tokn})
                with urllib.request.urlopen(req, timeout=3) as r:
                    if r.status == 200 and r.read():
                        _flag("spot-notice")
                        return
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(10)

    import threading
    if not a.smoke and os.environ.get("FULLFT_NO_SPOT_WATCH", "0") != "1":
        threading.Thread(target=_spot_watcher, daemon=True).start()

    def step_mem():
        """(v4, non-smoke CUDA runs) this step's live peak and the allocator's reserved bytes (max over ranks), the cudaMalloc retries
        since the previous step (max over ranks: the "expandable_segments: memory mapping failed" reclaim path); resets the peak."""
        p = torch.cuda.max_memory_allocated(dist.dev)
        r = torch.cuda.memory_reserved(dist.dev)
        n = int(torch.cuda.memory_stats(dist.dev).get("num_alloc_retries", 0))
        t = torch.tensor([p, r, n - mem["retries"]], dtype=torch.float64, device=dist.dev)
        mem["retries"] = n
        if dist.on:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        torch.cuda.reset_peak_memory_stats(dist.dev)
        mem["run_peak"] = max(mem["run_peak"], float(t[0]))
        return {"peak_gib": gib(float(t[0])), "reserved_gib": gib(float(t[1])), "retries": int(t[2])}
    try:
        start = 0 if a.smoke else load_resume(a, model, opt, head, dist)
        if not a.smoke and a.val and a.eval_at_start and start == 0:
            ep = do_eval(a, model, head, tok, dist, pad_id, 0)
            mem["eval_peak"] = max(mem["eval_peak"], ep or 0)
        for step in range(start, steps):
            ft = torch.tensor([1.0 if save_flags["why"] else 0.0], dtype=torch.float64)
            if dist.on:
                torch.distributed.all_reduce(ft, op=torch.distributed.ReduceOp.MAX)
            if float(ft[0]) > 0:                                # ANY rank's flag is EVERY rank's exit (collective symmetry)
                why = dist.gather_obj(save_flags["why"])
                if dist.rank == 0:
                    print(f"CHECKPOINT_EXIT step {step}: {why}", flush=True)
                raise _CheckpointExit(step, why or "rank-0")
            if step == start + 1 and cuda:
                torch.cuda.synchronize()
                rec["after_states_alloc_gib"] = gib(torch.cuda.memory_allocated(dist.dev))
                torch.cuda.reset_peak_memory_stats(dist.dev)
            ts = time.time()
            items = order if a.smoke else order[step * D:(step + 1) * D]
            W_step = float(sum(toks.vectors[i][3] for i in items))
            if a.smoke:
                plan = [[[r * a.docs_per_row * dist.world + rk * a.docs_per_row + j for j in range(a.docs_per_row)]
                         for r in range(a.rows_per_rank)] for rk in range(dist.world)]
            else:
                plan = plan_step(items, toks.lengths, dist.world, pack)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.lr, a.warmup, sched_steps, a.lr_floor)
            measured = step == start + 1
            if TRACE is not None:
                TRACE.on = measured
                if measured and cuda and a.mem_snapshot:
                    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=400000)
            longest = max((sum(toks.lengths[i] for i in row) for p in plan for row in p if row), default=0)
            if ev is not None and longest >= a.evict_states_min:
                ev.evict()
            if TRACE is not None:
                TRACE.mark("step start")
            st = run_step(a, model, head, opt, plan[dist.rank], toks, dist, W_step, pad_id)
            if a.dump_grads and step == start:
                torch.save({"rank": dist.rank, "world": dist.world, "loss": st["loss"],
                            "grads": {n: local(p.grad).detach().clone() for n, p in model.named_parameters() if p.grad is not None},
                            "head": {n: p.grad.detach().clone() for n, p in head.head.named_parameters()} if head is not None else {}},
                           f"{a.dump_grads}.rank{dist.rank}.pt")
            gnorm = clip_grads(params, a.clip, dist)
            if ev is not None:
                ev.restore()
            if TRACE is not None:
                TRACE.mark("before opt.step")
            opt.step()
            opt.zero_grad(set_to_none=True)
            if TRACE is not None:
                TRACE.mark("after opt.step")
                if measured and cuda and a.mem_snapshot:
                    torch.cuda.memory._dump_snapshot(a.mem_snapshot)
                    torch.cuda.memory._record_memory_history(enabled=None)
                TRACE.on = False
            hn = head.step(dist.sum_ if dist.on else None) if head is not None else None
            if cuda:
                torch.cuda.synchronize()
            tot = torch.tensor([st["loss"], st["ce"], st["kl"], st["n"], st["tokens"]], dtype=torch.float64)
            if dist.on:
                tot = tot.to(dist.dev)
                torch.distributed.all_reduce(tot)
                tot = tot.cpu()
            dt = time.time() - ts
            if step >= start + 1:
                times.append(dt)
            row = {"step": step, "loss": float(tot[0]) / W_step, "ce": float(tot[1]) / W_step, "kl": float(tot[2]) / W_step,
                   "decisions": int(tot[3]), "tokens": int(tot[4]), "s": round(dt, 3), "gnorm": gnorm,
                   "lr": opt.param_groups[0]["lr"]}
            if head is not None:
                row["head_kl_term"] = float(head.kl_term().detach())
                row["head_gnorm"] = [round(float(x), 4) for x in hn]
            if cuda and not a.smoke:
                row["mem"] = step_mem()
                row["longest_row"] = int(longest)
                if row["mem"]["peak_gib"] > a.mem_warn_gib:
                    mem["warns"] += 1
                    if dist.rank == 0:
                        print(f"MEM_WARN step {step}: live peak {row['mem']['peak_gib']} GiB > {a.mem_warn_gib} (longest row {longest} "
                              f"tokens)", flush=True)
            if not math.isfinite(row["loss"]):
                raise SystemExit(f"fullft: non-finite loss at step {step}")
            log.append(row)
            if step == start:
                rec["loss0"] = row["loss"]
            if dist.rank == 0:
                print("STEP " + json.dumps(row), flush=True)
            if not a.smoke and a.val and ((step + 1) % a.eval_every == 0 or step + 1 == steps):
                ep = do_eval(a, model, head, tok, dist, pad_id, step + 1)
                mem["eval_peak"] = max(mem["eval_peak"], ep or 0)
            if not a.smoke and a.save_every and ((step + 1) % a.save_every == 0 or step + 1 == steps):
                save_resume(a, model, opt, head, step + 1, dist)
    except _CheckpointExit as e:                                # (v6.1) signal / spot-notice / wipe: save and leave cleanly
        if not a.smoke and a.save_every:
            save_resume(a, model, opt, head, int(e.step), dist)
        rec["checkpoint_exit"] = {"step": int(e.step), "why": str(e.why)}
        if dist.on:
            torch.distributed.barrier()
    except Exception as e:                                      # (v6.1) exception saves: never lose a run to a crash
        rec["exception_exit"] = f"{type(e).__name__}: {str(e)[:400]}"
        if dist.rank == 0:
            print(f"EXCEPTION_EXIT {rec['exception_exit']}", flush=True)
        if isinstance(e, torch.cuda.OutOfMemoryError):
            rec["oom"] = True
            rec["oom_message"] = str(e)[:400]
            if TRACE is not None:
                TRACE.mark("OOM")
                if a.mem_snapshot:
                    torch.cuda.memory._dump_snapshot(a.mem_snapshot)
                    torch.cuda.memory._record_memory_history(enabled=None)
        try:
            _st = int(locals().get("step", locals().get("start", 0)) or 0)
            if not a.smoke and a.save_every and _st > 0:
                save_resume(a, model, opt, head, _st, dist)
        except Exception as e2:                                 # noqa: BLE001
            rec["exception_save_failed"] = str(e2)[:200]
        if not rec.get("oom"):
            raise
    if TRACE is not None:
        rec["mem_trace"] = TRACE.rows
    if ev is not None:
        rec["state_eviction"] = dict(ev.stats)
    if cuda:
        rec["peak_alloc_gib"] = gib(TRACE.peak() if TRACE is not None else max(mem["run_peak"], mem["eval_peak"],
                                                                                torch.cuda.max_memory_allocated(dist.dev)))
        if not a.smoke:
            rec["mem_telemetry"] = {"train_peak_gib": gib(mem["run_peak"]), "eval_peak_gib": gib(mem["eval_peak"]),
                                    "mem_warn_steps": mem["warns"], "warn_gib": a.mem_warn_gib}
        rec["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved(dist.dev))
        rec["total_gib"] = gib(torch.cuda.get_device_properties(dist.dev).total_memory)
        rec["alloc_retries"] = int(torch.cuda.memory_stats(dist.dev).get("num_alloc_retries", 0))      # (v5) the whole run, this rank
        if dist.on:
            t = torch.tensor([rec["peak_alloc_gib"], rec["peak_reserved_gib"], rec.get("after_states_alloc_gib", 0.0), rec["alloc_retries"]],
                             device=dist.dev)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            rec["max_over_ranks"] = {"peak_alloc_gib": round(float(t[0]), 3), "peak_reserved_gib": round(float(t[1]), 3),
                                     "after_states_alloc_gib": round(float(t[2]), 3), "alloc_retries": int(t[3])}
    if times:
        rec["step_s"] = round(float(np.median(times)), 3)
        tok_step = log[-1]["tokens"] if log else 0
        rec["tokens_per_step_node"] = tok_step
        rec["tokens_per_s_node"] = round(tok_step / rec["step_s"], 1)
        rec["tokens_per_s_per_gpu"] = round(tok_step / rec["step_s"] / dist.world, 1)
    rec["offload_stats"] = dict(fm.STATS)
    rec["attention"] = attention_record(cfg)
    rec["log"] = log[-8:]
    if dist.rank == 0:
        if a.smoke:
            tmp = a.out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f, indent=1)
            os.replace(tmp, a.out)
            print("FULLFT_SMOKE " + json.dumps({k: v for k, v in rec.items() if k not in ("log", "args")}), flush=True)
        else:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "train_record.json"), "w") as f:
                json.dump(rec, f, indent=1)
    if a.bundle and not a.smoke and not rec["oom"] and (steps == sched_steps or a.bundle_partial):
        write_bundles(a, model, tok, dist, pad_id, rec, steps)
    if not a.smoke and not rec["oom"] and not a.no_export:
        final_export(a, model, head, tok, mdir, letter_ids, header, dist)
    if dist.on:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0 if not rec["oom"] or a.smoke else 1


def write_bundles(a, model, tok, dist, pad_id, rec, step):
    """(v5.2) the --bundle manifests through the trained model in memory -> OUT/bundles/ROLE.pt + bundles.json (rank 0)"""
    bdir = os.path.join(a.out, "bundles")
    os.makedirs(bdir, exist_ok=True)
    keep, done = a.extract_features, {}
    for spec in a.bundle:
        role, _, rest = spec.partition("=")
        path, _, rest = rest.partition("@")
        want, _, ren = rest.partition("#")              # ROLE=MANIFEST[@ROWS_SHA256][#RENDERER_SHA256]: a per-pool expected renderer
        if not role or not path:
            raise SystemExit(f"--bundle {spec!r}: ROLE=MANIFEST[@ROWS_SHA256][#RENDERER_SHA256]")
        got = sha256_file(path)
        if want and got != want:
            raise SystemExit(f"--bundle {role}: {path} sha256 {got} != expected {want}")
        if got == getattr(a, "data_rows_sha256", None):
            raise SystemExit(f"--bundle {role}: that is the TRAINING manifest (a head pool must not be the backbone's training rows)")
        hdr, rows = read_manifest(path, ren or a.expect_renderer or None)
        check_row_limits(a, rows)
        a.extract_features = os.path.join(bdir, f"{role}.pt")
        extract_features(a, model, Tokens(rows, tok, images_dir=getattr(a, "vision_images", None) or None,
                                          layout=(hdr.get("image_layout") or None)), rows, dist, pad_id, rec)
        if dist.rank == 0:
            done[role] = {"manifest": os.path.basename(path), "rows_sha256": got, "rows": len(rows), "file": f"{role}.pt",
                          "renderer_sha256": hdr["renderer_sha256"],
                          "file_sha256": sha256_file(a.extract_features)}
    a.extract_features = keep
    if dist.rank == 0:
        json.dump({"trainer": VERSION, "step": int(step), "schedule_steps": rec.get("schedule_steps"), "code": rec.get("code"),
                   "param_dtype": a.param_dtype if dist.dev.type == "cuda" else "fp32",
                   "note": "CUDA: the forward uses the bf16 cast of the fp32 master weights = the bf16 export's weights",
                   "train_rows_sha256": getattr(a, "data_rows_sha256", None), "attention_policy": rec.get("attention_policy"),
                   "bundles": done}, open(os.path.join(bdir, "bundles.json"), "w"), indent=1)
        print("BUNDLES " + json.dumps({k: v["rows"] for k, v in done.items()}), flush=True)


def attention_record(cfg):
    """(v5) the policy, kernel settings and the dispatch log (calls per layer kind / path, the largest mask, CUDA kernel eligibility)"""
    return {"policy": fm.policy_of(cfg), "impl": fm.ATTN_IMPL, "chunk": fm.DEFAULT_CHUNK,
            "sdpa_backends": os.environ.get("JEV_SDPA_BACKENDS", "efficient,flash"),
            "math_override": os.environ.get("JEV_ALLOW_SDPA_MATH") == "1",
            "calls": {"|".join(k): v for k, v in sorted(fm.KERNEL_LOG["calls"].items())},
            "max_mask_elems": fm.KERNEL_LOG["max_mask_elems"],
            "eligible": {json.dumps(list(k)): v for k, v in fm.KERNEL_LOG["eligible"].items()}}


def check_row_limits(a, rows):
    """(v5) the manifest contract: no row above --max-row-tokens, every K inside --k-range (the manifest excludes and reports them)"""
    if a.max_row_tokens:
        over = [r["decision_uid"] for r in rows if int(r["n_tokens"]) > a.max_row_tokens]
        if over:
            raise SystemExit(f"{len(over)} manifest rows exceed --max-row-tokens {a.max_row_tokens}, e.g. {over[:2]}")
    if a.k_range:
        lo, hi = (int(x) for x in a.k_range.split(","))
        badk = [r["decision_uid"] for r in rows if not lo <= len(r["option_order"]) <= hi]
        if badk:
            raise SystemExit(f"{len(badk)} manifest rows have K outside [{lo}, {hi}], e.g. {badk[:2]}")


def extract_features(a, model, toks, rows, dist, pad_id, rec):
    """(v5) forward only over every manifest row (sharded over ranks, the training packing, the model's attention policy): per row
    h [H] fp32, ell [52] fp32, K, t [52] (letter order), w -> one torch file (rank 0 merges the ranks' shards, index order)."""
    model.eval()
    idx = list(range(len(rows)))
    plan = plan_step(idx, toks.lengths, dist.world, a.pack_tokens)[dist.rank]
    H, ELL, KS, T, Wt, I = [], [], [], [], [], []
    n_fwd = 0
    with torch.no_grad():
        for rws in plan:
            if rws is None:
                ids = torch.full((1, 16), pad_id, dtype=torch.long, device=dist.dev)
                model(ids, torch.arange(16, device=dist.dev)[None], torch.tensor([15], device=dist.dev))
                continue
            ids, pos, last, t, q, hp, ks, w, spans, imgs = micro_inputs(rws, toks, dist.dev, pad_id)
            h, ell = model(ids, pos, last, spans=spans, images=imgs)
            n_fwd += 1
            H.append(h.float().cpu())
            ELL.append(ell.float().cpu())
            KS.append(ks.cpu())
            T.append(t.float().cpu())
            Wt.append(w.float().cpu())
            I.extend(int(i) for i in rws)
    shard = {"i": torch.tensor(I, dtype=torch.long), "h": torch.cat(H) if H else torch.zeros(0), "ell": torch.cat(ELL) if ELL else torch.zeros(0),
             "k": torch.cat(KS) if KS else torch.zeros(0, dtype=torch.long), "t": torch.cat(T) if T else torch.zeros(0),
             "w": torch.cat(Wt) if Wt else torch.zeros(0), "forwards": n_fwd}
    torch.save(shard, f"{a.extract_features}.rank{dist.rank}")
    with torch.no_grad():                                # collective: the letter rows of the (tied) embedding the head's codes use
        E = full_of(model.text.embed_tokens.weight, dist.dev).index_select(0, model.letters).float().cpu()
    if dist.on:
        torch.distributed.barrier()
    if dist.rank == 0:
        parts = [torch.load(f"{a.extract_features}.rank{r}", weights_only=False) for r in range(dist.world)]
        parts = [x for x in parts if x["i"].numel()]
        i = torch.cat([x["i"] for x in parts])
        order = torch.argsort(i)
        if i[order].tolist() != idx:
            raise RuntimeError("extract-features: rows lost or duplicated across ranks")
        cat = lambda k: torch.cat([x[k] for x in parts])[order]  # noqa: E731
        T_ = cat("t")
        LBO = torch.zeros(len(idx), T_.shape[1], dtype=torch.float64)  # (v5.1) letter order, 0.0 where absent; fp64: the
        # identity binding rounds the losses to 6 decimals and losses can be large (an fp32 copy could round differently)
        HAS = torch.zeros(len(idx), dtype=torch.bool)
        lcount = {"ok": 0, "absent": 0, "partial": 0}
        for n_, j in enumerate(idx):
            vals, st_ = loss_of(rows[j])
            lcount[st_] += 1
            if st_ == "ok":
                LBO[n_, :len(vals)] = torch.tensor(vals, dtype=torch.float64)
                HAS[n_] = True
        out = {"h": cat("h"), "ell": cat("ell"), "k": cat("k"), "t": T_, "w": cat("w"), "loss_by_option": LBO, "has_loss": HAS,
               "decision_uid": [rows[j]["decision_uid"] for j in idx], "prompt_sha256": [rows[j]["prompt_sha256"] for j in idx],
               "source": [rows[j].get("source_family") or rows[j].get("source") for j in idx], "E_letters": E,
               "meta": {"trainer": VERSION, "code": rec["code"], "data": rec.get("data"), "attention_policy": rec["attention_policy"],
                        "init": rec.get("init"), "rows": len(idx), "backbone_forwards": sum(x["forwards"] for x in parts),
                        "label_ids": [int(x) for x in model.letters.tolist()],
                        "pack_tokens": a.pack_tokens, "loss_by_option_rows": lcount}}
        tmp = a.extract_features + ".partial"
        torch.save(out, tmp)
        os.replace(tmp, a.extract_features)
        for r in range(dist.world):
            os.remove(f"{a.extract_features}.rank{r}")
        print("FEATURES " + json.dumps({"file": a.extract_features, **out["meta"]}, default=str), flush=True)
    model.train()


def do_eval(a, model, head, tok, dist, pad_id, step):
    vh, vrows = read_manifest(a.val)
    vt = Tokens(vrows, tok, images_dir=getattr(a, "vision_images", None) or None,
                layout=(vh.get("image_layout") or None))
    idx = list(range(min(len(vrows), a.val_max))) if a.val_max else list(range(len(vrows)))
    cuda = dist.dev.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()                      # (v4.1, 10-13) free the training step's cached blocks first: the lrhi run's step-625 eval
        torch.cuda.reset_peak_memory_stats(dist.dev)  # died with CUBLAS_STATUS_EXECUTION_FAILED right after a failed 20 MB mapping
    res = evaluate(a, model, head, vt, idx, dist, pad_id, a.head_samples)
    peak = None
    if cuda:                                          # (v4) the eval's live peak; then drop the eval-shaped cache blocks
        t = torch.tensor([float(torch.cuda.max_memory_allocated(dist.dev))], dtype=torch.float64, device=dist.dev)
        if dist.on:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        peak = float(t[0])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dist.dev)
    if dist.rank == 0:
        m = eval_metrics(res)
        m["step"] = step
        if peak is not None:
            m["mem_peak_gib"] = gib(peak)
        print("EVAL " + json.dumps(m), flush=True)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"eval_step{step}.jsonl"), "w") as f:
            for r in res:
                r["decision_uid"] = vrows[r["i"]]["decision_uid"]
                r["prompt_sha256"] = vrows[r["i"]]["prompt_sha256"]
                f.write(json.dumps(r) + "\n")
    return peak


def save_resume(a, model, opt, head, step, dist):
    """per-rank resume shard (same world size required): local parameter shards, optimizer state (local tensors), head (rank 0)."""
    d = os.path.join(a.out, f"resume_step{step}")
    os.makedirs(d, exist_ok=True)
    ost = opt.state_dict()

    def loc(x):
        if isinstance(x, dict):
            return {k: loc(v) for k, v in x.items()}
        if isinstance(x, list):
            return [loc(v) for v in x]
        return local(x).detach().cpu() if isinstance(x, torch.Tensor) else x
    blob = {"step": step, "world": dist.world, "rank": dist.rank, "params": {n: local(p).detach().cpu() for n, p in model.named_parameters()},
            "optim": loc(ost), "version": VERSION, "ident": run_ident(a), "data_rows_sha256": getattr(a, "data_rows_sha256", None),
            "code": code_identity(a.pinned_code)}
    if head is not None:
        blob["head"] = head.state()                    # replicated (every rank holds the same head); saved on every rank
    tmp = os.path.join(d, f"rank{dist.rank}.pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, os.path.join(d, f"rank{dist.rank}.pt"))
    if dist.on:
        torch.distributed.barrier()
    if dist.rank == 0:
        with open(os.path.join(d, "COMPLETE"), "w") as f:
            f.write(json.dumps({"step": step, "world": dist.world, "time": time.time(), "ident": run_ident(a),
                                "data_rows_sha256": getattr(a, "data_rows_sha256", None), "code": code_identity(a.pinned_code)}))
        if a.keep_resume:                              # prune older COMPLETE resume dirs (per-file deletes), newest kept
            done = sorted((int(x[len("resume_step"):]), x) for x in os.listdir(a.out) if x.startswith("resume_step")
                          and os.path.exists(os.path.join(a.out, x, "COMPLETE")))
            for _, x in done[:-a.keep_resume]:
                rdir = os.path.join(a.out, x)
                os.remove(os.path.join(rdir, "COMPLETE"))
                for f in os.listdir(rdir):
                    os.remove(os.path.join(rdir, f))
                os.rmdir(rdir)
    if dist.on:
        torch.distributed.barrier()


RESUME_FREE = ("bundle", "bundle_partial",
               "mem_warn_gib", "out", "eval_every", "save_every", "keep_resume", "val", "val_max", "max_steps", "dump_grads", "no_export", "resume",
               "pinned_code", "init_from", "tiny_dir", "manifest", "evict_states_min", "mem_snapshot", "eval_at_start", "expect_rows_sha256",
               "allow_code_change_on_resume", "tokenizer_dir", "init_record", "extract_features")   # (v5) node-local paths / record-only


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


_CODE_ID = {}


def code_identity(pinned_code):
    """(v4.2) md5 of the trainer's own files and of the pinned production trainer (AuxVBLL) it loads"""
    key = str(pinned_code)
    if key not in _CODE_ID:
        out = {}
        for f in ("fullft_train.py", "fullft_mem.py", "fullft_head.py"):
            out[f] = hashlib.md5(open(os.path.join(HERE, f), "rb").read()).hexdigest()
        pt = os.path.join(pinned_code, "claude_lmtune.py")
        out["claude_lmtune.py"] = hashlib.md5(open(pt, "rb").read()).hexdigest() if os.path.exists(pt) else None
        _CODE_ID[key] = out
    return _CODE_ID[key]


def run_ident(a):
    """the arguments that shape what a step computes (resume refuses another identity)."""
    return hashlib.sha256(json.dumps({k: v for k, v in sorted(vars(a).items()) if k not in RESUME_FREE}, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def load_resume(a, model, opt, head, dist):
    """the newest COMPLETE resume_step* dir under --out (or --resume DIR): parameters, optimizer (local tensors re-wrapped as DTensors
    on their parameter's mesh), head; refuses another world size or run identity. -> the step to continue from (0 = fresh)."""
    root = a.resume or a.out
    if not os.path.isdir(root):
        return 0
    cands = sorted((int(x[len("resume_step"):]), x) for x in os.listdir(root) if x.startswith("resume_step")
                   and os.path.exists(os.path.join(root, x, "COMPLETE")))
    if not cands:
        return 0
    step, d = cands[-1]
    blob = torch.load(os.path.join(root, d, f"rank{dist.rank}.pt"), map_location="cpu", weights_only=False)
    if blob["world"] != dist.world or blob["rank"] != dist.rank:
        raise SystemExit(f"resume: {d} was written by world {blob['world']} (this run: {dist.world})")
    if blob.get("data_rows_sha256") != getattr(a, "data_rows_sha256", None):
        raise SystemExit(f"resume: {d} trained on data {blob.get('data_rows_sha256')}, this run reads {getattr(a, 'data_rows_sha256', None)} "
                         "(another training file: refused)")
    if blob.get("code") != code_identity(a.pinned_code):
        if not a.allow_code_change_on_resume:
            raise SystemExit(f"resume: {d} was written by trainer code {blob.get('code')}, this is {code_identity(a.pinned_code)} (refused; "
                             "--allow-code-change-on-resume records an explicit override)")
        print(f"RESUME_CODE_CHANGE {json.dumps({'from': blob.get('code'), 'to': code_identity(a.pinned_code)})}", flush=True)
    if blob.get("ident") != run_ident(a):
        raise SystemExit(f"resume: {d} has run identity {blob.get('ident')}, this run {run_ident(a)} (other arguments)")
    with torch.no_grad():
        for n, p in model.named_parameters():
            local(p).copy_(blob["params"][n])
    sd = blob["optim"]
    plist = [p for g in opt.param_groups for p in g["params"]]
    for idx, st in sd["state"].items():
        p = plist[int(idx)]
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor) and k != "step" and hasattr(p, "device_mesh") and not isinstance(opt, fm.LowMemAdamW):
                from torch.distributed.tensor import DTensor
                st[k] = DTensor.from_local(v.to(local(p).device), p.device_mesh, p.placements, run_check=False, shape=p.shape,
                                           stride=p.stride())
            elif isinstance(v, torch.Tensor) and k != "step":
                st[k] = v.to(local(p).device)
    opt.load_state_dict(sd)
    if head is not None:
        head.load(blob["head"])
    if dist.rank == 0:
        print(f"RESUMED at step {step} from {os.path.join(root, d)}", flush=True)
    return step


def final_export(a, model, head, tok, mdir, letter_ids, header, dist):
    """collective: the HF checkpoint (bf16 text weights + unchanged vision tower) and the selected head, rank 0 writes."""
    state = gather_full_state(model, dist.dev)
    if dist.rank == 0 and mdir:
        d = os.path.join(a.out, "export")
        n = export_hf(d, mdir, state)
        import shutil
        td = getattr(a, "tokenizer_dir", None) or mdir     # (v5) a backbone without tokenizer files (12B): the export carries the one used
        for f in os.listdir(td):
            if (f.startswith("tokenizer") or f.endswith(".jinja")) and not os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(td, f), os.path.join(d, f))
        cj = os.path.join(d, "config.json")
        cfgj = json.load(open(cj))
        cfgj["jev_requires_head"] = {"file": fh.HEAD_FILE, "meta": fh.HEAD_META, "engine": "garas_vbll_engine.GarasVBLLEngine",
                                     "readout": "posterior_predictive", "trainer": VERSION}
        pol = fm.policy_of(model.text.config)          # (v5) the attention policy travels with the weights; serving must apply it
        (cfgj["text_config"] if isinstance(cfgj.get("text_config"), dict) else cfgj)[fm.POLICY_ATTR] = pol
        cfgj["jev_requires_attention"] = {"impl": fm.ATTN_NAME, "policy": pol, "mem": fm.VERSION}
        json.dump(cfgj, open(cj, "w"), indent=2)
        if head is not None:
            sel = int(os.environ.get("JEV_HEAD_SELECTED", "-1"))
            if sel < 0:                     # the HEAD_RULE runs on DEV outside the trainer; until then export every head's grid file
                torch.save({"state": head.head.state_dict(), **head.head.meta()}, os.path.join(d, "vbll_head_grid.pt"))
            else:
                fh.export_head(d, head.head, letter_ids, sel, a.head_samples, header["total_decision_weight"])
        print(f"EXPORTED {d}: {n} weight shards", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
