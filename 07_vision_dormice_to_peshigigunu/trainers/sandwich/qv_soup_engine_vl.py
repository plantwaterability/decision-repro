"""VL copy of qv_soup_engine.py (586658b4), v2 (QV_LAYOUT_ANY=1: test-time layout reads): layout= (qv_layout.py; images_first = the stock prompt) and a checkpoint's trained vision merger
(merger.safetensors, fp32 as in training; single checkpoint only). A layout that differs from the checkpoint's state.json is refused.

Decision Index vision engine for adapter soups of qv_train checkpoints (and optional option-order averaging).

W = W_base + mean_m[(alpha/r) * B_m @ A_m] for every LoRA target; head = mean of member heads. perm=2 averages the head's
probabilities over the original and the reversed option order (mapped back); rows with options inside the image keep one pass.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "code"))  # VL: the stock qv_common lives in code/
import qv_common as C  # noqa: E402
import qv_layout as LY  # noqa: E402
from decision_index.engines.base import Engine, Unsupported  # noqa: E402
from decision_index.vision import images as img  # noqa: E402
from decision_index.vision.engines import MAX_PIXELS, answer, pixels, single_choice  # noqa: E402


class QVSoupEngine(Engine):
    name = "qv-soup-head"
    latency = "CUDA-synchronized in-process request wall time per row (all forward passes for the row); excludes model loading."

    def __init__(self, model, ckpts, device="cuda:0", perm=1, image_root=None, label=None, max_pixels=MAX_PIXELS, layout="images_first", **options):
        super().__init__(**options)
        import torch
        import transformers
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.torch, self.device, self.image_root, self.max_pixels, self.perm = torch, device, image_root, max_pixels, int(perm)
        if layout not in LY.LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}")
        self.layout = layout
        specs = ckpts.split(",") if isinstance(ckpts, str) else list(ckpts)
        cks = [Path(c.split("@")[0]) for c in specs]  # "path@w" = delta weight w; "path@w:h" = delta weight w, head weight h; default 1/n each
        wts = [float(c.split("@")[1].split(":")[0]) if "@" in c else 1.0 / len(specs) for c in specs]
        hws = [float(c.split(":")[1]) if "@" in c and ":" in c.split("@")[1] else (float(c.split("@")[1]) if "@" in c else 1.0 / len(specs)) for c in specs]
        self.proc = AutoProcessor.from_pretrained(model)
        C.set_pixel_cap(self.proc, max_pixels)
        m = Qwen3_5ForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        m.set_attn_implementation({"text_config": C.register_bidir()})
        sds, scales = [], []
        for ck in cks:  # adapters are small (~0.5 GB each); merge one module at a time so no full-size fp32 copy ever exists
            st = json.loads((ck / "state.json").read_text())
            scales.append(st["lora_alpha"] / st["lora_r"])
            sds.append(load_file(str(ck / "adapter.safetensors"), device=str(device)))
        n_mod = 0
        with torch.no_grad():
            for k in sds[0]:
                if not k.endswith(".lora_A.weight"):
                    continue
                path = k[len("base_model.model."):-len(".lora_A.weight")]
                kb = k.replace(".lora_A.weight", ".lora_B.weight")
                mod = m.get_submodule(path)
                w = mod.weight.float()
                for sd, sc, wt in zip(sds, scales, wts):
                    if k in sd:
                        w.add_(sd[kb].float() @ sd[k].float(), alpha=sc * wt)
                mod.weight.copy_(w.to(mod.weight.dtype))
                del w
                n_mod += 1
        del sds
        torch.cuda.empty_cache()
        states = [json.loads((ck / "state.json").read_text()) for ck in cks]
        bad = [str(ck) for ck, st in zip(cks, states) if st.get("layout", "images_first") != layout]
        if bad and os.environ.get("QV_LAYOUT_ANY") != "1":  # read with the layout it was trained with, unless QV_LAYOUT_ANY=1 (v2: test-time layout reads)
            raise ValueError(f"layout {layout!r} != the layout in state.json of {bad}")
        self.layout_mismatch = bad
        mg_files = [ck / "merger.safetensors" for ck in cks if (ck / "merger.safetensors").exists()]
        if any(float(st.get("merger_lr_mult", 0.0)) > 0 for st in states) and len(mg_files) != len(cks):
            raise ValueError("a checkpoint trained with the merger has no merger.safetensors")
        self.merger_file = None
        if mg_files:  # VL (arm uoq2): the checkpoint's trained vision merger, fp32 as in training
            if len(cks) != 1:
                raise ValueError("a checkpoint with a trained merger cannot be part of a soup")
            LY.merger_fp32(LY.merger_of(m)).load_state_dict(load_file(str(mg_files[0]), device=str(device)), strict=True)
            self.merger_file = str(mg_files[0])
        self.model = m.eval()
        heads = [torch.load(ck / "head.pt", map_location=device) for ck in cks]
        hw = hws
        hn = sum(hw)  # heads: weighted mean, weights normalised
        self.W = sum(h["weight"].float() * x for h, x in zip(heads, hw)) / hn
        self.b = sum(h["bias"].float() * x for h, x in zip(heads, hw)) / hn
        self.image_token_id = m.config.image_token_id
        self.provenance = {"kind": "qv-soup-head", "layout": layout, "layout_mismatch_allowed": self.layout_mismatch, "merger": self.merger_file, "model": model, "ckpts": [str(c) for c in cks], "weights": wts, "head_weights": hws, "perm": self.perm, "label": label,
                           "merged_modules": n_mod, "transformers": transformers.__version__, "torch": torch.__version__,
                           "policy": "kit prompt shape, enable_thinking=False, readout rows append 'The answer is ('; merged mean LoRA deltas; mean head; "
                                     + ("probabilities averaged over original + reversed option order (options-in-image rows: one pass)" if self.perm == 2 else "one forward")}

    def runtime(self):
        t = self.torch
        return {"gpu": t.cuda.get_device_name(self.device), "device": str(self.device), "torch": t.__version__}

    def warmup(self):
        pass

    def _probs(self, q_text, opts, pils, pr):
        torch = self.torch
        text = LY.render(self.proc, q_text, opts, len(pils), pr.get("options_in_image", False), pr.get("readout"), pr.get("ocr_block", ""), self.layout)
        enc = self.proc(text=[text], images=pils if pils else None, return_tensors="pt")
        enc = {n: v.to(self.device) for n, v in enc.items()}
        with torch.inference_mode():
            h = C.last_hidden(self.model, enc).float()
            z = h @ self.W[: len(opts)].T + self.b[: len(opts)]
        return torch.softmax(z, -1), int(enc["input_ids"].shape[1])

    def __call__(self, state, questions, images=(), prompt=None):
        torch = self.torch
        t0 = time.perf_counter()
        key, q, keys = single_choice(questions)
        k = len(keys)
        if k > len(C.LETTERS):
            raise Unsupported(f"{k} options")
        pils = [img.pil(p, self.image_root, self.max_pixels).convert("RGB") for p in images or []]
        pr = prompt or {}
        opts = [q["criteria"][kk] for kk in keys]
        p, ntok = self._probs(q["instructions"], opts, pils, pr)
        passes = 1
        if self.perm == 2 and not pr.get("options_in_image") and k > 1:
            pr_rev, _ = self._probs(q["instructions"], opts[::-1], pils, pr)
            p = (p + pr_rev.flip(0)) / 2
            passes = 2
        torch.cuda.synchronize()
        probs = [float(x) for x in p]
        choice = keys[max(range(k), key=lambda i: probs[i])]
        raw = {"probs": probs, "passes": passes, "input_tokens": ntok, "engine_ms": (time.perf_counter() - t0) * 1000,
               "image_pixels": [pixels(x, self.image_root, self.max_pixels) for x in images or []]}
        return answer(key, keys, choice, probs, "qv-soup", {"input_tokens": ntok}), raw
