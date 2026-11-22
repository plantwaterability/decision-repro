#!/usr/bin/env python3
"""(SCREEN4) Attention mode for the four-arm screen: CAUSAL vs NONCAUSAL (bidirectional) FULL-ATTENTION layers, one code path.

STAGE 2 of the screen (design tmp/screen4/design.md section 10): the joint-bidirectional-encoding ablation, run only after the stage-1
recipe is established. Primary backbone: Gemma-4-31B-it / Garas (99) -- 60 decoder layers, 10 full-attention + 50 sliding-window
layers; only the 10 FULL-attention layers become bidirectional (the analogue of pplx-decider-v1.1's hook, which turns the 16
full-attention layers of its Qwen3.8 hybrid bidirectional and leaves the 48 Gated-DeltaNet layers causal). The module is generic over
config.layer_types and is also tested on a Qwen3.5 hybrid. It puts the mask in at the DECODER LAYER (transformers 5.12.1, the fleet's,
gives every full-attention layer the one mask its text model built; a dict handed to the text model is refused: stage 2's wrapper calls
the TEXT model directly with a 2D padding mask and position ids):

  * a pre-hook on the TEXT MODEL reads the forward's 2D padding mask / position ids, derives a SEGMENT id per token (-1 = padding) and
    builds the full-attention mask ONCE per forward;
  * a pre-hook on every decoder layer whose config.layer_types entry is "full_attention" replaces that layer's `attention_mask` kwarg
    with it. Every other layer type is left exactly as the model built it: linear_attention (DeltaNet: an inherently causal
    recurrence; its own 2D padding mask) and sliding_attention (Gemma: causal within the window) are never touched.

Masks (per batch row b, query q, key k; True = may attend):
    causal     seg[q] == seg[k] and seg[k] >= 0 and k <= q
    noncausal  seg[q] == seg[k] and seg[k] >= 0                    (bidirectional inside one example; never across examples / padding)
  a padding query attends to itself only (no all-False row; its output is never read).
BOTH modes use this explicit mask (the arms differ only in its contents, not in the attention kernel: OPTISO A1 measured a kernel switch
moving bf16 probabilities by TV 0.01). With no hooks at all the model's own causal path is the reference: causal mode equals it.

Padding: right OR left; positions must start at 0 on each example's first real token (`positions_from_mask`; the model's default arange
is wrong under left padding). Packing (several examples in one row, positions restarting at 0) is supported on ALL-ATTENTION models
(block-diagonal masks) when attention_mask=None (only then does transformers build packing-aware masks for the other layer types); on a model with linear-attention layers it is REFUSED: a recurrent state would carry across the examples unless
the kernel is given sequence boundaries, which the torch fallback ignores (fail-closed; the screen pads, it never packs on the hybrid).
No KV cache in noncausal mode (a future token would change cached keys). sdpa (bool mask) or eager (additive float mask) only; any
kernel that ignores an arbitrary mask (flash_attention_*) is refused. cuDNN SDPA is the caller's switch (the screen disables it in every
arm, as the release does: model.py `torch.backends.cuda.enable_cudnn_sdp(False)`).

    ctl = AttentionMode(text_model, "noncausal")     # installs the hooks; ctl.mode is stamped into checkpoints / records
    out = text_model(input_ids=ids, attention_mask=am, position_ids=positions_from_mask(am), use_cache=False)
    ctl.remove()
"""
from __future__ import annotations

import torch

VERSION = "screen4_attn v1 (2026-10-12)"
MODES = ("causal", "noncausal")
FULL = "full_attention"
MASKABLE = ("sdpa", "eager")


def text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def attn_impl(model):
    cfg = text_config(model)
    return getattr(cfg, "_attn_implementation", None) or getattr(model.config, "_attn_implementation", None)


def decoder_layers(text_model):
    """The decoder layer ModuleList of a text model (Qwen3_5TextModel.layers, Gemma4TextModel.layers, ...)."""
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise SystemExit("screen4_attn: the model has no `layers` ModuleList (pass the TEXT model, e.g. Qwen3_5Model.language_model)")
    return layers


def layer_types(text_model):
    types = list(getattr(text_config(text_model), "layer_types", None) or [])
    n = len(decoder_layers(text_model))
    if len(types) < n:
        raise SystemExit(f"screen4_attn: config.layer_types has {len(types)} entries for {n} layers")
    return types[:n]


def positions_from_mask(attention_mask):
    """[B, L] position ids that start at 0 on each row's first real token (left or right padding); 0 on padding."""
    am = attention_mask.long()
    pos = am.cumsum(-1) - 1
    return pos.clamp(min=0) * am


def segments(attention_mask, position_ids=None, packed=False):
    """[B, L] segment ids: -1 = padding; with packed=True a new segment starts wherever position_ids == 0 on a real token."""
    am = attention_mask.bool()
    if not packed:
        return torch.where(am, torch.zeros_like(am, dtype=torch.long), torch.full_like(am, -1, dtype=torch.long))
    if position_ids is None:
        raise ValueError("packed segments need position_ids (restarting at 0 for every example)")
    pos = position_ids if position_ids.dim() == 2 else position_ids[0]
    starts = (pos == 0) & am
    seg = starts.long().cumsum(-1) - 1
    if (am & (seg < 0)).any():
        raise ValueError("a packed row's first real token does not have position 0")
    return torch.where(am, seg, torch.full_like(seg, -1))


def build_mask(seg, mode, impl, dtype):
    """[B, 1, L, L] mask for the full-attention layers (module docstring)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}")
    L = seg.shape[-1]
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, None, :] >= 0)
    if mode == "causal":
        idx = torch.arange(L, device=seg.device)
        same = same & (idx[None, :] <= idx[:, None])[None]
    eye = torch.eye(L, dtype=torch.bool, device=seg.device)[None]
    allowed = same | ((seg[:, :, None] < 0) & eye)               # a padding query sees itself only
    allowed = allowed[:, None]
    if impl == "sdpa":
        return allowed
    return torch.zeros(allowed.shape, dtype=dtype, device=seg.device).masked_fill(~allowed, torch.finfo(dtype).min)


class AttentionMode:
    """Installs the two hooks (module docstring) on a TEXT model; `remove()` takes them off. One instance per model."""

    def __init__(self, text_model, mode, allow_packed=None):
        if mode not in MODES:
            raise SystemExit(f"screen4_attn: mode must be one of {MODES}, got {mode!r}")
        if getattr(text_model, "_screen4_attention_mode", None) is not None:
            raise SystemExit("screen4_attn: an AttentionMode is already installed on this model")
        impl = attn_impl(text_model)
        if impl not in MASKABLE:
            raise SystemExit(f"screen4_attn: attention implementation {impl!r} cannot take an explicit mask (sdpa / eager only)")
        self.model, self.mode, self.impl = text_model, mode, impl
        self.types = layer_types(text_model)
        if FULL not in self.types:
            raise SystemExit("screen4_attn: the model has no full-attention layer")
        self.recurrent = any(t == "linear_attention" for t in self.types)
        self.allow_packed = (not self.recurrent) if allow_packed is None else bool(allow_packed)
        if self.allow_packed and self.recurrent:
            raise SystemExit("screen4_attn: packing on a model with linear-attention layers is refused (state would carry across)")
        self.full_layers = [i for i, t in enumerate(self.types) if t == FULL]
        self._mask, self._handles, self.calls = None, [], 0
        self._handles.append(text_model.register_forward_pre_hook(self._model_hook, with_kwargs=True))
        layers = decoder_layers(text_model)
        for i in self.full_layers:
            self._handles.append(layers[i].register_forward_pre_hook(self._layer_hook, with_kwargs=True))
        text_model._screen4_attention_mode = self

    # ------------------------------------------------------------------ hooks
    def _model_hook(self, module, args, kwargs):
        if args:
            raise SystemExit("screen4_attn: call the text model with keyword arguments (input_ids=..., attention_mask=...)")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            if self.mode == "noncausal":
                raise SystemExit("screen4_attn: noncausal attention cannot use a KV cache (pass use_cache=False)")
            raise SystemExit("screen4_attn: the screen never decodes; pass use_cache=False")
        emb = kwargs.get("inputs_embeds")
        ref = kwargs.get("input_ids") if emb is None else emb
        if ref is None:
            raise SystemExit("screen4_attn: no input_ids / inputs_embeds")
        B, L = ref.shape[0], ref.shape[1]
        am = kwargs.get("attention_mask")
        if isinstance(am, dict):
            raise SystemExit("screen4_attn: a per-layer-type mask dict reached the text model (a multimodal wrapper or a packed fast path "
                             "built it): call the TEXT model directly with the 2D padding mask and position ids")
        if am is None:
            am = torch.ones(B, L, dtype=torch.long, device=ref.device)
        if not isinstance(am, torch.Tensor) or am.dim() != 2 or tuple(am.shape) != (B, L):
            raise SystemExit("screen4_attn: expected the 2D [batch, length] padding mask")
        if not am.bool().any(-1).all():
            raise SystemExit("screen4_attn: an all-padding row")
        pos = kwargs.get("position_ids")
        seg = segments(am, pos, packed=False)
        if pos is not None:
            p2 = pos if pos.dim() == 2 else pos[0]
            restarts = ((p2 == 0) & am.bool()).sum(-1)
            if (restarts > 1).any():
                if not self.allow_packed:
                    raise SystemExit("screen4_attn: a packed row (positions restart) on a model whose packing is refused")
                if kwargs.get("attention_mask") is not None:
                    raise SystemExit("screen4_attn: a packed row WITH an explicit 2D mask: transformers then builds the other layer "
                                     "types' masks without packing (they would attend across examples); pass attention_mask=None")
                seg = segments(am, pos, packed=True)
        dtype = (emb.dtype if emb is not None else next(module.parameters()).dtype)
        self._mask = build_mask(seg, self.mode, self.impl, dtype)
        self.calls += 1
        return args, kwargs

    def _layer_hook(self, module, args, kwargs):
        # Gradient checkpointing re-runs this hook during backward with the CURRENT mask: callers must not run another forward
        # between a forward and its backward (the screen's trainer never does). The shape check catches a stale mask of another shape.
        if self._mask is None:
            raise SystemExit("screen4_attn: a full-attention layer ran outside a text-model forward")
        h = args[0] if args else kwargs.get("hidden_states")
        if h is not None and (h.shape[0] != self._mask.shape[0] or h.shape[1] != self._mask.shape[-1]):
            raise SystemExit(f"screen4_attn: stale mask {tuple(self._mask.shape)} for hidden states {tuple(h.shape)}")
        kwargs["attention_mask"] = self._mask
        return args, kwargs

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles, self._mask = [], None
        self.model._screen4_attention_mode = None

    def describe(self):
        return {"version": VERSION, "mode": self.mode, "impl": self.impl, "full_layers": self.full_layers,
                "n_layers": len(self.types), "recurrent_layers": sum(t == "linear_attention" for t in self.types),
                "packing": self.allow_packed}
