"""Shared pieces of the Qwen3.8 vision decision trainer and its eval engine.

- prompt rendering identical to the base-scan engine (kit 25c6ec7 prompt shape, enable_thinking=False, readout rows
  continue "The answer is (");
- non-causal attention for the full-attention layers only (the Gated DeltaNet layers stay causal by construction);
- the 26-way decision head over the last prompt token, initialised from the LM-head rows of the letters A..Z.
"""
import io

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TAIL = "Answer with the option's letter from the given choices directly."
MAX_PIXELS = 1600 * 32 * 32
ATTN_BIDIR = "qv_fullattn_bidir"
BIDIR_CALLS = {"n": 0}


def prompt_text(question, options, options_in_image=False):
    if options_in_image:
        return f"{question} {TAIL}"
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\n{opts}\n{TAIL}"


def render(proc, question, options, n_images, options_in_image=False, readout=None):
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": prompt_text(question, options, options_in_image)})
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    return text + readout if readout else text


def fit(w, h, max_pixels=MAX_PIXELS):
    if w * h <= max_pixels:
        return w, h
    s = (max_pixels / (w * h)) ** 0.5
    return max(1, int(w * s)), max(1, int(h * s))


def load_image(path, max_pixels=MAX_PIXELS):
    from PIL import Image

    im = Image.open(path)
    im.load()
    w, h = fit(im.width, im.height, max_pixels)
    if (w, h) != (im.width, im.height):
        im = im.resize((w, h), Image.LANCZOS)
    return im.convert("RGB")


def set_pixel_cap(proc, max_pixels=MAX_PIXELS):
    ip = proc.image_processor
    sz = getattr(ip, "size", None)
    if isinstance(sz, dict) and "longest_edge" in sz:
        ip.size = {**sz, "longest_edge": int(max_pixels)}
    elif getattr(sz, "longest_edge", None) is not None:
        sz.longest_edge = int(max_pixels)
    if getattr(ip, "max_pixels", None) is not None:
        ip.max_pixels = int(max_pixels)


def register_bidir():
    """Full-attention layers: no causal mask over the prompt (batch 1, no padding, no packing)."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    def bidir(module, query, key, value, attention_mask, **kw):
        BIDIR_CALLS["n"] += 1
        kw["is_causal"] = False
        return sdpa_attention_forward(module, query, key, value, None, **kw)

    AttentionInterface.register(ATTN_BIDIR, bidir)
    AttentionMaskInterface.register(ATTN_BIDIR, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    return ATTN_BIDIR


def letter_ids(tok, n=len(LETTERS)):
    ids = [tok.convert_tokens_to_ids(l) for l in LETTERS[:n]]
    unk = tok.unk_token_id
    if any(t is None or t == unk for t in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f"letter tokens not single ids: {ids}")
    for l, t in zip(LETTERS, ids):
        if tok.decode([t]) != l:
            raise RuntimeError(f"letter {l} token {t} decodes to {tok.decode([t])!r}")
    return ids


def lm_head_of(model):
    for name in ("lm_head",):
        m = getattr(model, name, None)
        if m is not None:
            return m
    return model.get_output_embeddings()


def inner_model(model):
    """The Qwen3_5Model inside a (possibly PEFT-wrapped) Qwen3_5ForConditionalGeneration."""
    m = model
    if type(m).__name__.startswith("Peft"):
        m = m.base_model.model
    return m.model


def last_hidden(model, enc):
    out = inner_model(model)(**enc, use_cache=False)
    return out.last_hidden_state[0, -1]
