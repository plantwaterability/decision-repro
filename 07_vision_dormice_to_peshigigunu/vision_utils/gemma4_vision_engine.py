"""Minimal Gemma-4 image+text engine for the Decision Index vision kit (kit commit 25c6ec7).

Called as engine(state, questions, images=[...], prompt={...}) by decision_index.vision.run.

- Loads a Gemma-4 checkpoint with transformers' Gemma4ForConditionalGeneration and the stock processor
  (the checkpoint's own processor_config.json, max_soft_tokens raised from 280 to 560 = the board's Gemma-4 budget, ~1.3 MP).
- Prompt shape = the kit's vllm-vl: one user turn with the images, then the question, `A. option` lines and
  "Answer with the option's letter from the given choices directly."; for readout rows (R-Bench-M, MMMU-Pro vision)
  an assistant turn "The answer is (" is continued.
- One forward pass; the next-token logits (after the model's final softcap) restricted to the valid option letters; softmax at T=1.
- Images: the kit's 1.6 MP cap (aspect kept, LANCZOS) before the processor.
- attention: "stock"  = transformers sdpa (causal text, bidirectional within each image block, as Gemma 4 ships);
             "polB"   = our text models' serving policy: full-attention layers bidirectional over the whole prompt,
                        sliding-window layers as stock (causal, window 1024, bidirectional within image blocks).
"""
import math
import time

from decision_index.engines.base import Engine, Unsupported
from decision_index.vision import images as img
from decision_index.vision.engines import MAX_PIXELS, TAIL, answer, pixels, prompt_text, resize_policy, single_choice, softmax

ATTN_POLB = "polB_sdpa"
POLB_CALLS = {"global_bidir": 0, "sliding_masked": 0}


def register_polB():
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    def polB(module, query, key, value, attention_mask, **kw):
        if getattr(module, "is_sliding", None) is False:
            POLB_CALLS["global_bidir"] += 1
            kw["is_causal"] = False
            return sdpa_attention_forward(module, query, key, value, None, **kw)
        POLB_CALLS["sliding_masked"] += 1
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register(ATTN_POLB, polB)
    AttentionMaskInterface.register(ATTN_POLB, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    return ATTN_POLB


class Gemma4VisionEngine(Engine):
    name = "gemma4-vl-letters"
    latency = ("CUDA-synchronized in-process request wall time per row: image load and 1.6 MP cap, processor, one forward pass "
               "(batch 1) and the letter readout; excludes model loading.")

    def __init__(self, model, attention="stock", max_soft_tokens=560, max_pixels=MAX_PIXELS, device="cuda:0", dtype="bfloat16",
                 image_root=None, label=None, readout="row", **options):
        super().__init__(**options)
        import torch
        import transformers
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

        if readout not in ("row", "none"):
            raise ValueError("readout is row (use the row's readout prefix) or none (always the plain generation prompt)")
        self.readout_mode = readout
        if attention not in ("stock", "polB"):
            raise ValueError("attention is stock or polB")
        self.torch = torch
        self.model_path, self.attention, self.device, self.image_root, self.max_pixels = model, attention, device, image_root, max_pixels
        self.proc = AutoProcessor.from_pretrained(model)
        self.proc.image_processor.max_soft_tokens = int(max_soft_tokens)
        self.max_soft_tokens = int(max_soft_tokens)
        self.tok = self.proc.tokenizer
        m = Gemma4ForConditionalGeneration.from_pretrained(model, dtype=getattr(torch, dtype), attn_implementation="sdpa").to(device)
        if attention == "polB":
            register_polB()
            m.set_attn_implementation({"text_config": ATTN_POLB})
        self.model = m.eval()
        tc = m.config.get_text_config()
        self.text_attn = tc._attn_implementation
        self.vision_attn = m.config.vision_config._attn_implementation
        if attention == "polB" and self.text_attn != ATTN_POLB:
            raise RuntimeError(f"text attention is {self.text_attn}, wanted {ATTN_POLB}")
        self.image_token_id = m.config.image_token_id
        self.letters = "ABCDEFGHIJKL"
        self.letter_ids = [self.tok.convert_tokens_to_ids(l) for l in self.letters]
        unk = self.tok.unk_token_id
        if any(t is None or t == unk for t in self.letter_ids) or len(set(self.letter_ids)) != len(self.letter_ids):
            raise RuntimeError(f"letter tokens not single ids: {self.letter_ids}")
        for l, t in zip(self.letters, self.letter_ids):
            if self.tok.decode([t]) != l:
                raise RuntimeError(f"letter {l} token {t} decodes to {self.tok.decode([t])!r}")
        self.provenance = {"kind": "transformers-gemma4-vl", "model": model, "label": label, "transformers": transformers.__version__,
                           "torch": torch.__version__, "dtype": dtype, "attention": attention, "readout_mode": readout, "text_attn_impl": self.text_attn,
                           "vision_attn_impl": self.vision_attn, "max_soft_tokens": self.max_soft_tokens, "max_pixels": max_pixels,
                           "image_resize": resize_policy(max_pixels) + f"; then the Gemma 4 processor (max_soft_tokens {self.max_soft_tokens})",
                           "processor_image_budget": {"max_soft_tokens": self.max_soft_tokens, "patch_size": self.proc.image_processor.patch_size,
                                                      "pooling_kernel_size": self.proc.image_processor.pooling_kernel_size},
                           "letter_ids": dict(zip(self.letters, self.letter_ids)),
                           "policy": "one user turn (images, then question, lettered options and the kit's letter instruction); readout rows continue "
                                     "an assistant turn 'The answer is ('; one forward; next-token logits restricted to the option letters; softmax T=1"}

    def runtime(self):
        t = self.torch
        return {"gpu": t.cuda.get_device_name(self.device) if str(self.device).startswith("cuda") else None, "device": str(self.device), "torch": t.__version__}

    def warmup(self):
        pass

    def render(self, q, n_images, prompt):
        content = [{"type": "image"} for _ in range(n_images)]
        content.append({"type": "text", "text": prompt_text(q, prompt)})
        msgs = [{"role": "user", "content": content}]
        readout = (prompt or {}).get("readout") if self.readout_mode == "row" else None
        if readout:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": readout}]})
            return self.proc.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, state, questions, images=(), prompt=None):
        torch = self.torch
        t0 = time.perf_counter()
        key, q, keys = single_choice(questions)
        k = len(keys)
        if k > len(self.letters):
            raise Unsupported(f"{k} options")
        pils = [img.pil(p, self.image_root, self.max_pixels).convert("RGB") for p in images or []]
        text = self.render(q, len(pils), prompt)
        enc = self.proc(text=[text], images=[pils] if pils else None, return_tensors="pt")
        enc = {n: v.to(self.device) for n, v in enc.items()}
        ids = enc["input_ids"][0]
        with torch.inference_mode():
            out = self.model(**enc, use_cache=False, logits_to_keep=1)
        z = out.logits[0, -1].float()
        vals = [float(z[t]) for t in self.letter_ids[:k]]
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        choice = keys[max(range(k), key=lambda i: vals[i])]
        top_all = int(torch.argmax(z))
        raw = {"letter_logits": vals, "input_tokens": int(ids.numel()), "image_tokens": int((ids == self.image_token_id).sum()),
               "image_pixels": [pixels(p, self.image_root, self.max_pixels) for p in images or []],
               "top_token_is_letter": top_all in self.letter_ids[:k], "top_token": self.tok.decode([top_all]),
               "engine_ms": (time.perf_counter() - t0) * 1000}
        return answer(key, keys, choice, softmax(vals), self.model_path, {"input_tokens": int(ids.numel())}), raw
