#!/usr/bin/env python3
"""t4b_diag.py: Fast 8-GPU diagnostic on stratified 1,100 rows of s4400 (s4400[::4]) to isolate:
  1. stock_causal:  Qwen3.5-4B zero-shot, causal t4b_sandwich_v1
  2. stock_bidir:   Qwen3.5-4B zero-shot, non-causal full-attention layers, t4b_sandwich_v1
  3. stock_plain:   Qwen3.5-4B zero-shot, simple single-question letter prompt (no 'Answer: (')
  4. t4b1_s35:      t4b1 ckpt_03516 merged at 0.35x LoRA + 0.35x readout blend with stock lm_head
  5. t4b1_ck250:    t4b1 ckpt_00250 (early checkpoint, 250 steps)
Run one arm per call (sharded across 8 GPUs via launcher) -> writes results.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4b_layout as LYT
from decision_index.engines.base import Engine, Unsupported


def register_bidir():
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    name = "t4b_fullattn_bidir"

    def bidir(module, query, key, value, attention_mask, **kw):
        kw["is_causal"] = False
        return sdpa_attention_forward(module, query, key, value, None, **kw)

    AttentionInterface.register(name, bidir)
    AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    return name


class DiagEngine(Engine):
    name = "t4b-diag"

    def __init__(self, model, mode="stock_causal", ckpt="", scale_mult="1.0", device="cuda:0", **options):
        super().__init__(**options)
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

        self.torch = torch
        self.dev = torch.device(device)
        self.mode = mode
        self.tok = AutoTokenizer.from_pretrained(model)
        self.codes, self.code_ids = LYT.build_codes(self.tok, LYT.N_CODES)

        # Plain single-letter codes for stock_plain (after enable_thinking=False generation prompt)
        self.plain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.plain_ids = [self.tok.encode(c, add_special_tokens=False)[0] for c in self.plain_letters]

        cfg = AutoConfig.from_pretrained(model).get_text_config()
        m = Qwen3_5ForCausalLM.from_pretrained(model, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa").to(self.dev)
        if mode == "stock_bidir":
            m.set_attn_implementation(register_bidir())

        W0 = m.get_output_embeddings().weight.detach()[self.code_ids].float().clone()
        self.head = torch.nn.Linear(W0.shape[1], len(self.code_ids), bias=True).to(self.dev, torch.float32)
        with torch.no_grad():
            self.head.weight.copy_(W0)
            self.head.bias.zero_()

        sm = float(scale_mult)
        if ckpt and ckpt != "none":
            ck = Path(ckpt)
            st = json.loads((ck / "state.json").read_text())
            sd = load_file(str(ck / "adapter.safetensors"), device=str(self.dev))
            sc = (float(st["lora_alpha"]) / float(st["lora_r"])) * sm
            with torch.no_grad():
                for k in sd:
                    if not k.endswith(".lora_A.weight"):
                        continue
                    mod = m.get_submodule(k[len("base_model.model.") : -len(".lora_A.weight")])
                    w = mod.weight.float()
                    w.add_(sd[k.replace(".lora_A.weight", ".lora_B.weight")].float() @ sd[k].float(), alpha=sc)
                    mod.weight.copy_(w.to(mod.weight.dtype))
            h = torch.load(ck / "readout.pt", map_location=self.dev)
            with torch.no_grad():
                self.head.weight.copy_((1.0 - sm) * W0 + sm * h["weight"].float())
                self.head.bias.copy_(sm * h["bias"].float())

        self.backbone = m.model.eval()
        self.model = m.eval()

    def synchronize(self):
        self.torch.cuda.synchronize(self.dev)

    def _score_plain(self, state_text, q):
        torch = self.torch
        keys, opt_texts = LYT.options_of(q)
        if len(keys) > len(self.plain_letters):
            # Fallback to t4b_sandwich codes for K>52
            return self._score_t4b(state_text, LYT.instructions_of(q), q)
        opts = "\n".join(f"{self.plain_letters[j]}) {opt_texts[j]}" for j in range(len(opt_texts)))
        msg = f"{state_text}\n\nQuestion: {LYT.instructions_of(q)}\n\nOptions:\n{opts}\n\nAnswer with the letter of the single best option and nothing else."
        prompt = self.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) > 32768:
            raise Unsupported("too long")
        t_ids = torch.tensor([ids], dtype=torch.long, device=self.dev)
        with torch.inference_mode():
            z = self.model(input_ids=t_ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()[self.plain_ids[: len(keys)]]
            probs = torch.softmax(z, -1).cpu().tolist()
        return keys, probs

    def _score_t4b(self, state_text, preview, q):
        torch = self.torch
        keys, opt_texts = LYT.options_of(q)
        if len(keys) > len(self.codes):
            raise Unsupported("too many options")
        prompt = LYT.render_prompt(self.tok, state_text, preview, LYT.instructions_of(q), opt_texts, self.codes)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) > 32768:
            raise Unsupported("too long")
        t_ids = torch.tensor([ids], dtype=torch.long, device=self.dev)
        with torch.inference_mode():
            h = self.backbone(input_ids=t_ids, use_cache=False).last_hidden_state[0, -1].float()
            z = self.head(h)[: len(keys)]
            probs = torch.softmax(z, -1).cpu().tolist()
        return keys, probs

    def __call__(self, state, questions):
        state_text = LYT.describe(state)
        preview = LYT.question_preview(questions)
        answers = {}
        for qkey, q in questions.items():
            kind = q.get("type")
            if kind not in ("choice", "noul", "score"):
                raise Unsupported(f"type {kind}")
            if self.mode == "stock_plain":
                keys, probs = self._score_plain(state_text, q)
            else:
                keys, probs = self._score_t4b(state_text, preview, q)
            dist = {k: float(p) for k, p in zip(keys, probs)}
            if kind == "noul":
                answers[qkey] = {"type": "noul", "noul": min(1.0, max(0.0, dist["true"]))}
            elif kind == "score":
                answers[qkey] = {"type": "score", "score": float(sum(i * p for i, p in enumerate(probs))), "probabilities": dist, "legend": {str(i): v for i, v in enumerate(q["criteria"])}}
            else:
                best = max(keys, key=lambda kk: (dist[kk], -keys.index(kk)))
                answers[qkey] = {"type": "choice", "choice": best, "probabilities": dist}
        return {"model": f"diag:{self.mode}", "answers": answers}, None
