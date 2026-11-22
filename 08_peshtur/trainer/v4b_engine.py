"""Standalone d3 + OCR eval engine for Qwen3.5-4B distilled student checkpoints (ops/v4b).

Matches vx_train_vl.py (--layout d3, --merger-lr-mult > 0, non-causal full attention via qv_common.register_bidir,
26-way head.pt in fp32) bit-for-bit on encoding and arithmetic:
  - Loads stock Qwen/Qwen3.5-4B (AutoModelForImageTextToText, bf16, sdpa)
  - Sets full-attention layers non-causal via qv_common.register_bidir() (GDN layers stay causal)
  - If ckpt is provided:
      1. Merges adapter.safetensors into language-model linear layers in fp32, single bf16 rounding
      2. Loads merger.safetensors (if present) into qv_layout.merger_fp32(qv_layout.merger_of(model))
      3. Loads head.pt (26 x H weight + bias in fp32)
  - If ckpt is None: uses stock lm_head letter rows A..Z (fp32, bias 0) for zero-shot baseline.
"""
import json
import os
import time
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported
from decision_index.vision import images as img
from decision_index.vision.engines import answer, pixels, single_choice

import qv_common as C
import qv_layout as LY
import qv_ocr as O
import wx_engine as WXE


class V4BEngine(Engine):
    name = "v4b-d3"
    latency = "CUDA-synchronized in-process request wall time per row (image loading, prompt, one forward pass); excludes model loading."

    def __init__(self, model_dir, ckpt=None, device="cuda:0", image_root=None, label=None, ocr=None, **options):
        super().__init__(**options)
        import torch
        import transformers
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.image_root = image_root
        self.max_pixels = WXE.D3_MAX_PIXELS
        torch.cuda.set_device(device)

        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.proc.tokenizer.padding_side = "left"
        self.proc.image_processor.size = {
            "shortest_edge": WXE.D3_MIN_PIXELS,
            "longest_edge": WXE.D3_MAX_PIXELS,
        }
        lids = C.letter_ids(self.proc.tokenizer)

        m = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
        )
        m.set_attn_implementation({"text_config": C.register_bidir()})
        self.calls = C.BIDIR_CALLS

        lm = C.lm_head_of(m)
        W = lm.weight.detach()[lids].float().clone().to(device)
        b = torch.zeros(len(lids), dtype=torch.float32, device=device)

        merged = 0
        has_merger = False
        if ckpt:
            ck = Path(ckpt)
            st = json.loads((ck / "state.json").read_text())
            if st.get("layout") != "d3":
                raise ValueError(f"{ck}: trained with layout {st.get('layout')!r}, not 'd3'")
            sc = float(st["lora_alpha"]) / float(st["lora_r"])
            sd = load_file(str(ck / "adapter.safetensors"), device=str(device))
            with torch.no_grad():
                for k in sd:
                    if not k.endswith(".lora_A.weight"):
                        continue
                    path = k[len("base_model.model.") : -len(".lora_A.weight")]
                    mod = m.get_submodule(path)
                    w = mod.weight.float()
                    w.add_(sd[k.replace(".lora_A.weight", ".lora_B.weight")].float() @ sd[k].float(), alpha=sc)
                    mod.weight.copy_(w.to(mod.weight.dtype))
                    del w
                    merged += 1
            del sd
            mf = ck / "merger.safetensors"
            if mf.exists():
                mg = LY.merger_fp32(LY.merger_of(m))
                mg.load_state_dict(load_file(str(mf), device=str(device)), strict=True)
                has_merger = True
            h = torch.load(ck / "head.pt", map_location=device)
            W = h["weight"].float().to(device)
            b = h["bias"].float().to(device)
            torch.cuda.empty_cache()

        self.model = m.eval()
        self.W = W
        self.b = b
        self.ocr_text = O.load_ocr(ocr)[0] if ocr else None
        self.provenance = {
            "kind": "v4b-d3",
            "label": label,
            "model_dir": str(model_dir),
            "ckpt": str(ckpt) if ckpt else None,
            "lora_modules_merged": merged,
            "merger_loaded": has_merger,
            "ocr": ocr,
            "min_pixels": WXE.D3_MIN_PIXELS,
            "max_pixels": WXE.D3_MAX_PIXELS,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        }

    def runtime(self):
        t = self.torch
        return {"gpu": t.cuda.get_device_name(self.device), "device": str(self.device), "torch": t.__version__}

    def warmup(self):
        pass

    def __call__(self, state, questions, images=(), prompt=None):
        torch = self.torch
        t0 = time.perf_counter()
        key, q, keys = single_choice(questions)
        k = len(keys)
        if k > len(C.LETTERS):
            raise Unsupported(f"{k} options")
        ims = list(images or [])
        pils = [img.pil(p, self.image_root, self.max_pixels).convert("RGB") for p in ims]
        block = O.ocr_prefix(self.ocr_text, ims) if self.ocr_text is not None else ""
        st = block if block else state
        content = [{"type": "image"} for _ in pils] + [
            {"type": "text", "text": WXE.d3_user(st, q, list(C.LETTERS[:k]))}
        ]
        text = self.proc.apply_chat_template(
            [{"role": "system", "content": WXE.D3_SYSTEM}, {"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if text.count("<|image_pad|>") != len(pils) or "<|video_pad|>" in text:
            raise ValueError("literal image / video placeholder in the row text")
        enc = self.proc(text=[text], images=pils if pils else None, padding=True, return_tensors="pt")
        enc = {n: v.to(self.device) for n, v in enc.items()}
        with torch.inference_mode():
            h = C.last_hidden(self.model, enc).float()
            z = h @ self.W[:k].T + self.b[:k]
            p = torch.softmax(z, -1)
        torch.cuda.synchronize()
        probs = [float(x) for x in p]
        choice = keys[max(range(k), key=lambda i: probs[i])]
        ntok = int(enc["input_ids"].shape[1])
        raw = {
            "probs": probs,
            "input_tokens": ntok,
            "engine_ms": (time.perf_counter() - t0) * 1000,
            "ocr_chars": len(block),
            "image_pixels": [pixels(x, self.image_root, self.max_pixels) for x in ims],
        }
        return answer(key, keys, choice, probs, "v4b-d3", {"input_tokens": ntok}), raw
