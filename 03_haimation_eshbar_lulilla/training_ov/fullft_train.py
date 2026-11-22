#!/usr/bin/env python3
"""(FULLFT) FULL fine-tuning of Gemma-4-31B-it's text stack + the residual VBLL decision head on one p8 node (8 x H100 80 GB), FSDP2.

  torchrun --nproc-per-node 8 trainers/fullft_train.py --manifest train.jsonl --val val.jsonl --out DIR [memory flags] ...
  torchrun --nproc-per-node 8 trainers/fullft_train.py --smoke --length 32768 --out rec.json [memory flags]      # memory / speed smoke
  python3 trainers/fullft_train.py --tiny --smoke --length 64 --out /tmp/x.json                                 # CPU

Model: google/gemma-4-31B-it at a PINNED revision; only the text stack (Gemma4TextModel, 30.7B) is built and trained; the vision tower is
untouched and copied unchanged at export. FSDP2 full-shard per decoder layer + root, BF16 compute (MixedPrecisionPolicy), FP32 master
weights, FP32 gradient reduction (default), AdamW. The decision readout is LETTER-ONLY: the final hidden state h at each decision's
last token, native letter logits ell = softcap(h E[letters]^T) in fp32 (the tied embedding's rows for jev_labels' labels, any K; full-vocabulary logits are
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
import random
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fullft_head as fh  # noqa: E402
import fullft_mem as fm  # noqa: E402

VERSION = "fullft_train v6.1 (2026-10-15)"            # v5 = stock_decision_v1 (the registered pilot keeps v4 / v4.2 under fullft11 / 12)
# v6.1 (2026-10-15, MACKAY-HEAD's step-1 audit): the hybrid takes no score-row smoothing and starts exactly at the letter readout
# (fullft_span v3); evals carry rule_gap_pt (decider vs native accuracy, pt) and, for --readout span at steps 400 / 600, kill_rule.
# v6 (2026-10-15, the Latent01 step-2 design; coordinator spec + approvals): --readout span | hybrid trains the EVIDENCE-SPAN decision
# readout (fullft_span.py) END TO END in place of the letter readout: the backbone loss is (1 - a) CE + a KL on the decider's integrated p
# (letter: unchanged, byte-for-byte the v5.1 path; --span-mode ref, the default, = the reference head with a corrected sigma init; vi = a
# shared-weight variational posterior trained on the ELBO, det = its deterministic twin: fullft_span v2). The span head lives OUTSIDE FSDP (fp32, replicated on every rank, its own AdamW;
# gradients summed over ranks once per step; the posterior KL / W_unique added once per step). Two stages: Stage A --stageA-steps head-only
# steps (the backbone forward-only, the head at --stageA-lr after a --stageA-warmup ramp; the first steps of a separately seeded exposure
# order), then the usual schedule with the backbone at --lr and the head at --span-lr (same warmup / cosine / floor). Stage A is not
# resumable (an interruption reruns it); its finished head is saved (stageA_head.pt) so a later restart skips it; Stage B resume shards
# carry the head and its optimizer. Requires --head off (the AuxVBLL head reads letter logits). Option spans: fullft_span.option_token_spans
# (sdv1_optfeat's verified mapping) for every training / dev row at load; a row without spans is refused. Evals record both readouts:
# native = the letter softmax (as before), decider = the span readout's p, with its two sigma parts (sig_learned, sig_post).
# v5.1: --extract-features also writes loss_by_option [N, N_LETTERS] (label order, 0.0 where absent) + has_loss [N] from the manifest rows'
# loss_by_option (the expected-regret term of a decision head); a malformed loss_by_option fails the manifest read.
CODE_REVISION = "v6 (v5 + attention policy BF, the evidence-span readout --readout span | hybrid, two-stage)"
# ROADMAP Stage 2 fixed settings (plan.json "training" / "bayesian_head"); --preset sdv1_pilot applies them and refuses any conflicting flag
SDV1_PILOT = {"optim": "adamw-fused", "lr": 2.5e-6, "warmup": 50, "lr_floor": 0.1, "decisions_per_step": 64, "epochs": 1.0,
              "weight_decay": 0.0, "clip": 1.0, "alpha": 0.0, "head": "on", "head_prior": "1", "head_kl": "1", "head_cov": "full",
              "head_dim": 128, "head_lr": 1e-3, "head_samples": 64, "eval_every": 200, "eval_at_start": True, "save_every": 200,
              "param_dtype": "bf16", "reduce_dtype": "fp32", "max_row_tokens": 8192, "k_range": "", "stock_only": True,
              "freeze_embed": False, "val_max": 0, "attn": "jev_mem"}
SDV1_PILOT_STEPS = 600                               # 38,400 exposures / 64; evals + checkpoints at 0 / 200 / 400 / 600
CONTENDER_MARKERS = ("garas", "pplx", "perplexity", "vllm-sr", "plantwaterer", "decider")
import jev_labels  # noqa: E402
LABELS = jev_labels.labels()                         # every K: A-Z, a-z, then single-token AA.., AAA.. (no K cap)
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
    """the text stack + the letter readout. forward(ids [1, T], pos [1, T], last [n]) -> (h [n, H] fp32, ell [n, N_LETTERS] fp32)."""

    def __init__(self, text, letter_ids, attn):
        super().__init__()
        self.text = text
        self.register_buffer("letters", torch.as_tensor(letter_ids, dtype=torch.long), persistent=False)
        self.cap = float(getattr(text.config, "final_logit_softcapping", 0) or 0)
        self.masks = {"full_attention": None, "sliding_attention": None} if attn == fm.ATTN_NAME else None

    def forward(self, ids, pos, last, want_hs=False):
        out = self.text(input_ids=ids, position_ids=pos, attention_mask=self.masks, use_cache=False)
        h = out.last_hidden_state[0].index_select(0, last)
        h = h if h.dtype == torch.float64 else h.float()
        E = self.text.embed_tokens.weight.index_select(0, self.letters)
        if want_hs:                                  # (v6) the span readout pools option spans from the whole packed row
            return h, fh.letter_logits(h, E, self.cap), out.last_hidden_state[0]
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
    uniq = {}
    for i, r in enumerate(rows):
        if r["renderer_sha256"] != ren:
            raise ValueError(f"{path}: row {i} renderer differs from the header")
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
    cfg._attn_implementation = a.attn
    prior = getattr(cfg, fm.POLICY_ATTR, None)
    if a.attention_policy:
        if prior is not None and prior != a.attention_policy:
            raise SystemExit(f"the init config carries attention policy {prior}; --attention-policy {a.attention_policy} refused")
        fm.set_policy(cfg, a.attention_policy)
    elif prior not in (None, "C"):
        raise SystemExit(f"the init config carries attention policy {prior}: pass --attention-policy {prior} explicitly")
    fm.require_policy(cfg, fm.policy_of(cfg))
    a.init_record = stock_only_check(a, mdir) if a.stock_only else {"stock_only": False, "dir": mdir}
    if mdir is None:                                 # random tiny configs (tests): ids wrap into the tiny vocabulary
        letter_ids = [10 + j % (cfg.vocab_size - 10) for j in range(N_LETTERS)]
        tok = None
    else:
        from transformers import AutoTokenizer
        a.tokenizer_dir = tokenizer_dir(a, mdir)
        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
        letter_ids = jev_labels.token_ids(tok)
    with torch.device("meta"):
        text = text_classes(cfg)[0](cfg)
        model = Decider(text, letter_ids, a.attn)
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
def span_inputs(rows, toks, last, dev):
    """(v6) the packed row's option spans: opt_row [M], opt_k [M], opt_tok [M, 2] (packed-row token indices), is_score [n]"""
    orow, ok, otok, sc = [], [], [], []
    for j, i in enumerate(rows):
        n_i = toks.lengths[i]
        off = int(last[j]) - n_i + 1
        sp = toks.spans(i)
        for k, (f0, f1) in enumerate(sp):
            orow.append(j)
            ok.append(k)
            otok.append((off + f0, off + f1))
        sc.append(toks.is_score(i))
    return (torch.tensor(orow, dtype=torch.long, device=dev), torch.tensor(ok, dtype=torch.long, device=dev),
            torch.tensor(otok, dtype=torch.long, device=dev).reshape(-1, 2), torch.tensor(sc, dtype=torch.bool, device=dev))


def micro_inputs(rows, toks, dev, pad_id):
    """packed row of the given exposures -> ids, pos, last, t [n, N_LETTERS], q [n, N_LETTERS], has_parent [n], ks [n], w [n]"""
    ids, pos, last = fm.packed_batch([toks[i] for i in rows], pad_id=pad_id)
    n = len(rows)
    t = torch.zeros(n, N_LETTERS)
    q = torch.zeros(n, N_LETTERS)
    hp = torch.zeros(n, dtype=torch.bool)
    ks = torch.zeros(n, dtype=torch.long)
    w = torch.zeros(n)
    for j, i in enumerate(rows):
        tt, qq, k, ww = toks.vectors[i]
        t[j, :k] = torch.as_tensor(tt)
        if qq is not None:
            q[j, :k] = torch.as_tensor(qq)
            hp[j] = True
        ks[j], w[j] = k, ww
    return [x.to(dev) for x in (ids, pos, last, t, q, hp, ks, w)]


class Tokens:
    """token ids of the manifest rows, tokenised lazily (checked against n_tokens); the letter-order vectors."""

    def __init__(self, rows, tok, spans=False):
        self.rows, self.tok, self.cache = rows, tok, {}
        self.vectors = []
        for r in rows:
            t, q, k = row_vectors(r)
            self.vectors.append((t, q, k, float(r.get("weight", 1.0))))
        self.lengths = [int(r["n_tokens"]) for r in rows]
        self._spans = None
        if spans:                                    # (v6) every row's option token spans, checked once at load
            import fullft_span as fsp
            self._spans, bad = [], []
            for i, r in enumerate(rows):
                sp = fsp.option_token_spans(tok, r["prompt"], len(r["option_order"]), LABELS, self[i])
                if sp is None or len(sp) != len(r["option_order"]) or any(not (0 <= f0 <= f1 < self.lengths[i]) for f0, f1 in sp):
                    bad.append(r["decision_uid"])
                    sp = None
                self._spans.append(sp)
            if bad:
                raise SystemExit(f"fullft --readout span: {len(bad)} of {len(rows)} rows have no option spans (first: {bad[:3]})")
            self.span_stats = {"rows": len(rows), "options": sum(len(x) for x in self._spans)}

    def spans(self, i):
        return self._spans[i]

    def is_score(self, i):
        return (self.rows[i].get("kind") or self.rows[i].get("qtype")) == "score"

    def __getitem__(self, i):
        x = self.cache.get(i)
        if x is None:
            x = self.tok(self.rows[i]["prompt"], add_special_tokens=False)["input_ids"]
            if len(x) != self.lengths[i]:
                raise ValueError(f"row {i} ({self.rows[i]['decision_uid']}): {len(x)} tokens, the manifest says {self.lengths[i]}")
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[i] = x
        return x


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


class SpanState:
    """(v6) the evidence-span head (fullft_span.EvidenceSpanHead): fp32, REPLICATED on every rank (identical seeded init), outside FSDP;
    its own AdamW; per step: gradients summed over ranks (/ the FSDP loss scale), + KL / W_unique (identical on every rank), clipped."""

    def __init__(self, a, hidden, dev, W_unique):
        import fullft_span as fsp
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(a.seed) + 7919)
            self.head = fsp.EvidenceSpanHead(hidden, a.span_rank, hybrid=a.readout == "hybrid", mode=a.span_mode, prior_lam=a.span_prior_lam,
                                             score=a.span_score == "on" and a.readout != "hybrid")   # the hybrid starts AT the letter readout
        self.head = self.head.to(dev).double() if a.test_float64 else self.head.to(dev).float()
        self.params = list(self.head.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=a.span_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        self.W_unique = float(W_unique)
        self.stage = "B"
        self.seed, self.tick = int(a.seed), 0           # (vi) the training draws: a fresh seeded antithetic set per step, identical on every rank

    def train_eps(self):
        return self.head.draws(self.seed * 1000003 + 17 + self.tick) if self.head.mode == "vi" else None

    def eval_eps(self):
        return self.head.draws(0) if self.head.mode == "vi" else None

    def step(self, dist, lr, clip):
        for p_ in self.params:
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
        if dist.on:
            dist.sum_([p_.grad for p_ in self.params])
        for p_ in self.params:
            p_.grad.div_(dist.loss_scale)
        kl = self.head.kl() / self.W_unique
        if kl.requires_grad:
            kl.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.params, clip))
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.tick += 1
        return gn, float(kl.detach())

    def state(self):
        return {"head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()}, "opt": self.opt.state_dict(), "meta": self.head.meta(),
                "tick": self.tick}

    def load(self, st):
        self.head.load_state_dict(st["head"])
        self.opt.load_state_dict(st["opt"])
        self.tick = int(st.get("tick", 0))


def span_forward(a, model, span, ids, pos, last, ks, rows, toks, dev, stage_a, eps=None, mean_only=False):
    """(v6) the span readout of one packed row -> (h, ell, res)"""
    import contextlib
    import fullft_span as fsp
    orow, ok, otok, isc = span_inputs(rows, toks, last, dev)
    with torch.no_grad() if stage_a else contextlib.nullcontext():
        h, ell, hs = model(ids, pos, last, want_hs=True)
    hl, hm = fsp.pool_spans(hs, otok)
    if stage_a:
        h, ell, hl, hm = h.detach(), ell.detach(), hl.detach(), hm.detach()
    hd = span.head.U_dec.weight.dtype
    res = span.head.forward_from_states(h.to(hd), hl.to(hd), hm.to(hd), orow, ok, ks, is_score=isc, max_k=N_LETTERS, ell=ell.to(hd),
                                        eps=eps, mean_only=mean_only)
    return h, ell, res


def cc_offsets(rows, toks, ell):
    """(ell + per-row case-control log-odds offsets, number of rows carrying one). The manifest field cc_logit_offset holds one
    value per DISPLAYED option position; it enters the training objective only (backbone loss and the head's data term),
    never evaluation or serving. Same arithmetic as Wave 7's fullft_resume _run_step_with_cc_offset hook."""
    if not hasattr(toks, "rows"):
        return ell, 0
    off, n = None, 0
    for j, i in enumerate(rows):
        v = toks.rows[i].get("cc_logit_offset")
        if v:
            if off is None:
                off = torch.zeros(ell.shape, dtype=torch.float32)
            v = [float(x) for x in v[:ell.shape[1]]]
            off[j, :len(v)] = torch.tensor(v)
            n += 1
    return (ell if off is None else ell + off.to(device=ell.device, dtype=ell.dtype)), n


def run_step(a, model, head, opt, plan_rows, toks, dist, W_step, pad_id, train=True, span=None):
    """one optimizer step's forwards / backwards on this rank -> stats dict (sums over this rank's decisions)."""
    st = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "n": 0, "tokens": 0, "head_negbound": None, "cc_rows": 0}
    if span is not None:                             # (v6) the span readout (Stage A: head only; Stage B: end to end)
        import fullft_span as fsp
        stage_a = span.stage == "A"
        if train and hasattr(toks, "rows") and any(toks.rows[i].get("cc_logit_offset") for rs in plan_rows if rs for i in rs):
            raise SystemExit("cc_logit_offset rows reached the span readout, which does not apply them")
        for rows in plan_rows:
            if rows is None:
                ids = torch.full((1, 16), pad_id, dtype=torch.long, device=dist.dev)
                pos = torch.arange(16, device=dist.dev)[None]
                if stage_a:
                    with torch.no_grad():
                        model(ids, pos, torch.tensor([15], device=dist.dev))
                else:
                    h, ell = model(ids, pos, torch.tensor([15], device=dist.dev))
                    (ell.sum() * 0.0).backward()
                continue
            ids, pos, last, t, q, hp, ks, w = micro_inputs(rows, toks, dist.dev, pad_id)
            h, ell, res = span_forward(a, model, span, ids, pos, last, ks, rows, toks, dist.dev, stage_a, eps=span.train_eps())
            if "logp_samples" in res:                # (vi) the ELBO's expected log-likelihood: the per-draw loss averaged over the draws
                parts = [fsp.decision_losses_logp(lp_, res["mask"], t.to(lp_.dtype), q.to(lp_.dtype), hp, a.alpha, a.ineligible_gold_weight)
                         for lp_ in res["logp_samples"]]
                loss_i = torch.stack([x[0] for x in parts]).mean(0)
                ce_i = torch.stack([x[1] for x in parts]).mean(0)
                kl_i = torch.stack([x[2] for x in parts]).mean(0)
            else:
                loss_i, ce_i, kl_i = fsp.decision_losses_logp(res["logp"], res["mask"], t.to(res["logp"].dtype), q.to(res["logp"].dtype), hp,
                                                              a.alpha, a.ineligible_gold_weight)
            L = (w.to(loss_i.dtype) * loss_i).sum() / W_step * dist.loss_scale
            L.backward()
            st["loss"] += float((w.to(loss_i.dtype) * loss_i).sum().detach())
            st["ce"] += float((w.to(ce_i.dtype) * ce_i).sum())
            st["kl"] += float((w.to(kl_i.dtype) * kl_i).sum())
            st["n"] += len(rows)
            st["tokens"] += int(ids.shape[1])
        return st
    for rows in plan_rows:
        if rows is None:                       # dummy forward: FSDP's collectives need the same count on every rank
            ids = torch.full((1, 16), pad_id, dtype=torch.long, device=dist.dev)
            pos = torch.arange(16, device=dist.dev)[None]
            h, ell = model(ids, pos, torch.tensor([15], device=dist.dev))
            (ell.sum() * 0.0).backward()
            continue
        ids, pos, last, t, q, hp, ks, w = micro_inputs(rows, toks, dist.dev, pad_id)
        h, ell = model(ids, pos, last)
        ell_loss, ncc = cc_offsets(rows, toks, ell) if train else (ell, 0)
        loss_i, ce_i, kl_i = fh.decision_losses(ell_loss, ks, t, q, hp, a.alpha, a.ineligible_gold_weight)
        L = (w * loss_i).sum() / W_step * dist.loss_scale
        if TRACE is not None:
            TRACE.mark("loss")
        L.backward()
        if TRACE is not None:
            TRACE.mark("bwd end")
        if head is not None:
            nb = head.data_backward(h, ell_loss, t, ks, w, W_step)
            st["head_negbound"] = nb if st["head_negbound"] is None else st["head_negbound"] + nb
        st["cc_rows"] += ncc
        st["loss"] += float((w * loss_i).sum().detach())
        st["ce"] += float((w * ce_i).sum())
        st["kl"] += float((w * kl_i).sum())
        st["n"] += len(rows)
        st["tokens"] += int(ids.shape[1])
    return st


def evaluate(a, model, head, toks, idx, dist, pad_id, samples, span=None):
    """forward-only over `idx` (sharded over ranks, padded with dummies) -> rank 0: per-decision readouts in index order."""
    model.eval()
    if span is not None:
        span.head.eval()
    plan = plan_step(idx, toks.lengths, dist.world, a.pack_tokens)[dist.rank]
    res = []
    with torch.no_grad():
        for rows in plan:
            if rows is None:
                ids = torch.full((1, 16), pad_id, dtype=torch.long, device=dist.dev)
                model(ids, torch.arange(16, device=dist.dev)[None], torch.tensor([15], device=dist.dev))
                continue
            ids, pos, last, t, q, hp, ks, w = micro_inputs(rows, toks, dist.dev, pad_id)
            if span is not None:                     # (v6) both readouts: the letter softmax (native) and the span decider
                h, ell, sres = span_forward(a, model, span, ids, pos, last, ks, rows, toks, dist.dev, True, eps=span.eval_eps())
            else:
                h, ell = model(ids, pos, last)
                sres = None
            ro = fh.readouts(head.head, h, ell, ks, samples) if head is not None else None
            for j, i in enumerate(rows):
                k = int(ks[j])
                r = {"i": int(i), "k": k, "t": t[j, :k].tolist(), "native": None}
                lp, _ = fh.masked_logp(ell[j:j + 1], ks[j:j + 1])
                r["native"] = lp.exp()[0, :k].tolist()
                if sres is not None:
                    r["decider"] = sres["p"][j, :k].double().tolist()
                    r["mu"] = sres["z"][j, :k].double().tolist()
                    r["sig_learned"] = sres["sig_learned"][j, :k].double().tolist()
                    r["sig_post"] = sres["sig_post"][j, :k].double().tolist()
                    r["mi"] = float(sres["mi"][j])
                    if "p_mean" in sres:                 # (vi / det) the posterior-mean readout (S = 0)
                        r["decider_mean"] = sres["p_mean"][j, :k].double().tolist()
                if ro is not None:
                    r["mean"] = ro["mean"][:, j, :k].tolist()
                    r["predictive"] = ro["predictive"][:, j, :k].tolist()
                    r["epi"] = ro["epi"][:, j].tolist()
                res.append(r)
    model.train()
    if span is not None:
        span.head.train()
    allr = dist.gather_obj(res)
    if dist.rank != 0:
        return None
    out = sorted([r for rr in allr for r in rr], key=lambda r: r["i"])
    if [r["i"] for r in out] != sorted(idx):
        raise RuntimeError("eval: rows lost or duplicated across ranks")
    return out


def kill_rule(readout, step, m):
    """(v6.1, coordinator 10-15) the span-only arms' kill rule, read post hoc: at step 400 decider_acc more than 2 pt below the arm's own
    native letter readout, or at step 600 more than 1 pt below -> "fail" (failed by rule); otherwise "pass"; None where it does not apply"""
    if readout != "span" or step not in (400, 600) or "rule_gap_pt" not in m:
        return None
    return "fail" if m["rule_gap_pt"] < (-2.0 if step == 400 else -1.0) else "pass"


def eval_metrics(res):
    def nll(p, t):
        return float(-(np.asarray(t) * np.log(np.clip(np.asarray(p), 1e-12, 1))).sum())
    m = {"n": len(res), "native_nll": float(np.mean([nll(r["native"], r["t"]) for r in res]))}
    m["native_acc"] = float(np.mean([np.asarray(r["t"])[int(np.argmax(r["native"]))] >= max(r["t"]) - 1e-9 for r in res]))
    if res and "decider" in res[0]:                  # (v6) the span readout, and its two sigma parts (mean over options)
        m["decider_nll"] = float(np.mean([nll(r["decider"], r["t"]) for r in res]))
        m["decider_acc"] = float(np.mean([np.asarray(r["t"])[int(np.argmax(r["decider"]))] >= max(r["t"]) - 1e-9 for r in res]))
        m["sig_learned_mean"] = float(np.mean([np.mean(r["sig_learned"]) for r in res]))
        m["sig_post_mean"] = float(np.mean([np.mean(r["sig_post"]) for r in res]))
        m["sig_post_max"] = float(max(max(r["sig_post"]) for r in res))
        # (v6.1, coordinator 10-15) the span-only arms' KILL RULE, read post hoc at steps 400 / 600: decider_acc more than 2 pt (400) / 1 pt
        # (600) below the same arm's native letter readout -> failed by rule (do_eval writes kill_rule for --readout span; the run is NOT
        # stopped). rule_gap_pt is reported for the hybrid too (descriptive).
        m["rule_gap_pt"] = round(100.0 * (m["decider_acc"] - m["native_acc"]), 3)
        if "decider_mean" in res[0]:
            m["decider_mean_nll"] = float(np.mean([nll(r["decider_mean"], r["t"]) for r in res]))
            m["decider_mean_acc"] = float(np.mean([np.asarray(r["t"])[int(np.argmax(r["decider_mean"]))] >= max(r["t"]) - 1e-9 for r in res]))
            m["mi_mean"] = float(np.mean([r["mi"] for r in res]))
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
    ap.add_argument("--smoke-k", type=int, default=4, help="smoke: options per synthetic decision (any K: the label list has no cap)")
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
                    help="(v5) C = causal; B = global layers bidirectional within their document, local layers causal-windowed")
    ap.add_argument("--stock-only", action="store_true", help="(v5) refuse every initialisation but the pinned stock snapshot")
    ap.add_argument("--preset", choices=("", "sdv1_pilot"), default="", help="(v5) fixed settings; conflicting flags are refused")
    ap.add_argument("--max-row-tokens", type=int, default=0, help="(v5) refuse manifest rows with more tokens (0 = off)")
    ap.add_argument("--k-range", default="", help="(v5) 'LO,HI': refuse manifest rows whose K is outside")
    ap.add_argument("--extract-features", default="", help="(v5) forward only: h / ell / K / t / w of every --manifest row -> this file")
    ap.add_argument("--readout", choices=("letter", "span", "hybrid"), default="letter", help="(v6) the trained decision readout")
    ap.add_argument("--span-rank", type=int, default=256, help="(v6) the span head's projection rank r")
    ap.add_argument("--span-mode", choices=("ref", "vi", "det"), default="ref", help="(v6) ref = the reference head (corrected sigma init); "
                    "vi = the shared-weight variational posterior (ELBO, 64 antithetic draws, KL / W_unique); det = vi's deterministic twin")
    ap.add_argument("--span-prior-lam", type=float, default=0.0, help="(v6) the prior precision of N(0, I / lambda) on w_mu (0 = F = 3 r)")
    ap.add_argument("--span-score", choices=("on", "off"), default="on", help="(v6) the score-interval integral on manifest score rows")
    ap.add_argument("--span-lr", type=float, default=1e-4, help="(v6) the span head's Stage-B LR (the backbone's warmup / cosine / floor)")
    ap.add_argument("--stageA-steps", type=int, default=0, help="(v6) head-only steps before the schedule (backbone forward-only)")
    ap.add_argument("--stageA-lr", type=float, default=1e-3, help="(v6) the Stage-A head LR (after --stageA-warmup steps of linear ramp)")
    ap.add_argument("--stageA-warmup", type=int, default=10)
    a = ap.parse_args(argv)
    if a.backbone:
        b = BACKBONES[a.backbone]
        for k in ("model", "revision"):
            if getattr(a, k) not in (ap.get_default(k), b["repo" if k == "model" else "revision"]):
                raise SystemExit(f"--backbone {a.backbone} fixes --{k} {b['repo' if k == 'model' else 'revision']}")
        a.model, a.revision = b["repo"], b["revision"]
    if a.preset:
        a = apply_preset(a, ap)
    if a.readout != "letter":
        if a.smoke:
            raise SystemExit("--readout span / hybrid: not with --smoke (no option spans)")
        if a.head != "off":
            raise SystemExit("--readout span / hybrid: the AuxVBLL head reads letter logits; pass --head off")
        if a.preset:
            raise SystemExit("--readout span / hybrid: --preset fixes the letter readout's head; pass the settings explicitly")
        if a.readout == "hybrid" and a.span_mode != "ref":
            raise SystemExit("--readout hybrid: --span-mode ref only")
    elif a.stageA_steps or a.span_mode != "ref":
        raise SystemExit("--stageA-steps / --span-mode: the span / hybrid readouts only")
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
        toks = SyntheticTokens(n_docs, doc_len, cfg.vocab_size, seed=a.seed, k=a.smoke_k)
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
        if a.extract_features:
            toks = Tokens(rows, tok)
        else:
            toks = Tokens(rows, tok, spans=a.readout != "letter")
            if a.readout != "letter":
                rec["spans"] = toks.span_stats
        if a.extract_features:
            extract_features(a, model, toks, rows, dist, pad_id, rec)
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
    span = None
    if a.readout != "letter":                        # (v6) the evidence-span readout's head (replicated, outside FSDP)
        span = SpanState(a, int(cfg.hidden_size), dist.dev, float(header["total_decision_weight"]))
        rec["span_head"] = {**span.head.meta(), "params": sum(p_.numel() for p_ in span.params), "W_unique": span.W_unique,
                            "stageA_steps": a.stageA_steps, "stageA_lr": a.stageA_lr, "span_lr": a.span_lr}
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
        start = 0 if a.smoke else load_resume(a, model, opt, head, dist, span)
        if not a.smoke and a.val and a.eval_at_start and start == 0:
            ep = do_eval(a, model, head, tok, dist, pad_id, 0, span)
            mem["eval_peak"] = max(mem["eval_peak"], ep or 0)
        if span is not None and a.stageA_steps and start == 0 and not a.smoke:     # (v6) Stage A: the head alone
            run_stage_a(a, model, span, toks, rows, dist, pad_id, D, pack, rec)
            if a.val:
                ep = do_eval(a, model, head, tok, dist, pad_id, "stageA", span)
                mem["eval_peak"] = max(mem["eval_peak"], ep or 0)
        for step in range(start, steps):
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
            st = run_step(a, model, head, opt, plan[dist.rank], toks, dist, W_step, pad_id, span=span)
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
            sgn = span.step(dist, lr_at(step, a.span_lr, a.warmup, sched_steps, a.lr_floor), a.clip) if span is not None else None
            if cuda:
                torch.cuda.synchronize()
            tot = torch.tensor([st["loss"], st["ce"], st["kl"], st["n"], st["tokens"], st["cc_rows"]], dtype=torch.float64)
            if dist.on:
                tot = tot.to(dist.dev)
                torch.distributed.all_reduce(tot)
                tot = tot.cpu()
            dt = time.time() - ts
            if step >= start + 1:
                times.append(dt)
            row = {"step": step, "loss": float(tot[0]) / W_step, "ce": float(tot[1]) / W_step, "kl": float(tot[2]) / W_step,
                   "decisions": int(tot[3]), "tokens": int(tot[4]), "cc_rows": int(tot[5]), "s": round(dt, 3), "gnorm": gnorm,
                   "lr": opt.param_groups[0]["lr"]}
            if head is not None:
                row["head_kl_term"] = float(head.kl_term().detach())
                row["head_gnorm"] = [round(float(x), 4) for x in hn]
            if sgn is not None:
                row["span_gnorm"], row["span_kl_term"] = round(sgn[0], 5), sgn[1]
                row["span_lr"] = span.opt.param_groups[0]["lr"]
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
                ep = do_eval(a, model, head, tok, dist, pad_id, step + 1, span)
                mem["eval_peak"] = max(mem["eval_peak"], ep or 0)
            if not a.smoke and a.save_every and ((step + 1) % a.save_every == 0 or step + 1 == steps):
                save_resume(a, model, opt, head, step + 1, dist, span)
    except torch.cuda.OutOfMemoryError as e:
        rec["oom"] = True
        rec["oom_message"] = str(e)[:400]
        if TRACE is not None:
            TRACE.mark("OOM")
            if a.mem_snapshot:
                torch.cuda.memory._dump_snapshot(a.mem_snapshot)
                torch.cuda.memory._record_memory_history(enabled=None)
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
    if not a.smoke and not rec["oom"] and not a.no_export:
        final_export(a, model, head, tok, mdir, letter_ids, header, dist, span)
    if dist.on:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0 if not rec["oom"] or a.smoke else 1


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
    h [H] fp32, ell [N_LETTERS] fp32, K, t [N_LETTERS] (label order), w -> one torch file (rank 0 merges the ranks' shards, index order)."""
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
            ids, pos, last, t, q, hp, ks, w = micro_inputs(rws, toks, dist.dev, pad_id)
            h, ell = model(ids, pos, last)
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
        E = full_of(model.text.embed_tokens.weight, dist.dev).index_select(0, model.letters[:jev_labels.BASIS_LABELS]).float().cpu()  # the head's reference basis
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
        KW = int(cat("k").max())                         # label columns stored: the largest K present (the readout covers every label)
        T_ = cat("t")[:, :KW]
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
        out = {"h": cat("h"), "ell": cat("ell")[:, :KW], "k": cat("k"), "t": T_, "w": cat("w"), "loss_by_option": LBO, "has_loss": HAS,
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


def run_stage_a(a, model, span, toks, rows, dist, pad_id, D, pack, rec):
    """(v6) Stage A: --stageA-steps head-only steps (the backbone forward-only, no backbone update) on the first exposures of a
    separately seeded order; the head at --stageA-lr after a linear --stageA-warmup ramp. Skipped when a matching stageA_head.pt exists
    (a restart after Stage A); otherwise (re)run from the start."""
    path = os.path.join(a.resume or a.out, "stageA_head.pt")
    ident = {"run_ident": run_ident(a), "data": a.data_rows_sha256, "steps": a.stageA_steps, "code": code_identity(a.pinned_code)}
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("ident") == ident:
            span.load(blob)
            rec["stageA"] = {"loaded": path, "log": blob.get("log")}
            if dist.rank == 0:
                print(f"STAGEA_LOADED {path}", flush=True)
            return
    span.stage = "A"
    rng = np.random.default_rng(int(a.seed) + 10007)
    orderA = [int(x) for x in rng.permutation(len(rows))]
    while len(orderA) < a.stageA_steps * D:
        orderA += [int(x) for x in rng.permutation(len(rows))]
    log = []
    for sa in range(a.stageA_steps):
        items = orderA[sa * D:(sa + 1) * D]
        W_step = float(sum(toks.vectors[i][3] for i in items))
        plan = plan_step(items, toks.lengths, dist.world, pack)
        st = run_step(a, model, None, None, plan[dist.rank], toks, dist, W_step, pad_id, span=span)
        lr = a.stageA_lr * min(1.0, (sa + 1) / max(1, a.stageA_warmup))
        gn, klv = span.step(dist, lr, a.clip)
        tot = torch.tensor([st["loss"], st["ce"], st["kl"], st["n"]], dtype=torch.float64)
        if dist.on:
            tot = tot.to(dist.dev)
            torch.distributed.all_reduce(tot)
            tot = tot.cpu()
        row = {"stageA_step": sa, "loss": float(tot[0]) / W_step, "ce": float(tot[1]) / W_step, "kl": float(tot[2]) / W_step,
               "decisions": int(tot[3]), "span_gnorm": round(gn, 5), "span_kl_term": klv, "lr": lr}
        log.append(row)
        if not math.isfinite(row["loss"]):
            raise SystemExit(f"fullft: non-finite Stage-A loss at step {sa}")
        if dist.rank == 0:
            print("STAGEA " + json.dumps(row), flush=True)
    span.stage = "B"
    for g in span.opt.param_groups:                  # Stage B sets the head LR every step (--span-lr schedule)
        g["lr"] = a.span_lr
    rec["stageA"] = {"steps": a.stageA_steps, "log": log[-5:]}
    if dist.rank == 0:
        os.makedirs(a.out, exist_ok=True)
        tmp = os.path.join(a.out, "stageA_head.pt.tmp")
        torch.save({**span.state(), "ident": ident, "log": log}, tmp)
        os.replace(tmp, os.path.join(a.out, "stageA_head.pt"))
    if dist.on:
        torch.distributed.barrier()


_VAL = {}


def do_eval(a, model, head, tok, dist, pad_id, step, span=None):
    if a.val not in _VAL:                            # (v6) the dev rows (and their option spans) once per process
        vh, vrows = read_manifest(a.val)
        _VAL[a.val] = (vrows, Tokens(vrows, tok, spans=span is not None))
    vrows, vt = _VAL[a.val]
    idx = list(range(min(len(vrows), a.val_max))) if a.val_max else list(range(len(vrows)))
    cuda = dist.dev.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()                      # (v4.1, 10-13) free the training step's cached blocks first: the lrhi run's step-625 eval
        torch.cuda.reset_peak_memory_stats(dist.dev)  # died with CUBLAS_STATUS_EXECUTION_FAILED right after a failed 20 MB mapping
    res = evaluate(a, model, head, vt, idx, dist, pad_id, a.head_samples, span)
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
        kr = kill_rule(a.readout, step, m)            # (v6.1) the span-only arms' kill rule (post-read)
        if kr is not None:
            m["kill_rule"] = kr
        if peak is not None:
            m["mem_peak_gib"] = gib(peak)
        print("EVAL " + json.dumps(m), flush=True)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"eval_step{step}.jsonl" if isinstance(step, int) else f"eval_{step}.jsonl"), "w") as f:
            for r in res:
                r["decision_uid"] = vrows[r["i"]]["decision_uid"]
                r["prompt_sha256"] = vrows[r["i"]]["prompt_sha256"]
                f.write(json.dumps(r) + "\n")
    return peak


def save_resume(a, model, opt, head, step, dist, span=None):
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
    if span is not None:
        blob["span"] = span.state()                    # (v6) replicated; every rank
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


RESUME_FREE = ("mem_warn_gib", "out", "eval_every", "save_every", "keep_resume", "val", "val_max", "max_steps", "dump_grads", "no_export", "resume",
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
        for f in ("fullft_span.py", "sdv1_optfeat.py"):  # (v6) the span readout's code (recorded when present)
            if os.path.exists(os.path.join(HERE, f)):
                out[f] = hashlib.md5(open(os.path.join(HERE, f), "rb").read()).hexdigest()
        pt = os.path.join(pinned_code, "claude_lmtune.py")
        out["claude_lmtune.py"] = hashlib.md5(open(pt, "rb").read()).hexdigest() if os.path.exists(pt) else None
        _CODE_ID[key] = out
    return _CODE_ID[key]


def run_ident(a):
    """the arguments that shape what a step computes (resume refuses another identity)."""
    return hashlib.sha256(json.dumps({k: v for k, v in sorted(vars(a).items()) if k not in RESUME_FREE}, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def load_resume(a, model, opt, head, dist, span=None):
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
    if span is not None:
        if "span" not in blob:
            raise SystemExit(f"resume: {d} carries no span head")
        span.load(blob["span"])
    if dist.rank == 0:
        print(f"RESUMED at step {step} from {os.path.join(root, d)}", flush=True)
    return step


def final_export(a, model, head, tok, mdir, letter_ids, header, dist, span=None):
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
        if span is not None:                           # (v6) the trained decision readout travels with the weights
            from safetensors.torch import save_file
            save_file({k: v.detach().cpu().float().contiguous() for k, v in span.head.state_dict().items()}, os.path.join(d, "evidence_head.safetensors"))
            hm = {**span.head.meta(), "readout": a.readout, "letter_ids": [int(x) for x in letter_ids], "labels": LABELS, "trainer": VERSION,
                  "tensor_sha256": sha256_file(os.path.join(d, "evidence_head.safetensors"))}
            json.dump(hm, open(os.path.join(d, "evidence_head.json"), "w"), indent=1, sort_keys=True)
            cfgj["jev_decision_readout"] = {"readout": a.readout, "file": "evidence_head.safetensors", "meta": "evidence_head.json",
                                            "spans": "the R2 Options block (sdv1_optfeat.option_spans) on the rendered prompt"}
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
