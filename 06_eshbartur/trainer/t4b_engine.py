"""Standalone Decision Index kit engine for t4b (Qwen3.5-4B, 4.21B pure causal text).

Loads Qwen/Qwen3.5-4B via native Qwen3_5ForCausalLM (4.21B, drops unused vision tower),
merges adapter.safetensors in fp32 -> bf16, loads 255-way readout.pt, and serves all 3
Decision Index question types (choice, noul, score) with up to 255 options in 1 pass.
Uses causal Qwen3_5DynamicCache prefix sharing on multi-question requests.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4b_layout as LYT
from decision_index.engines.base import Engine, Unsupported


class T4BEngine(Engine):
    name = "t4b"
    latency = "in-process request wall time on one GPU"

    def __init__(
        self,
        model,
        ckpt=None,
        ckpt_spec=None,
        temperature=1.0,
        max_len=65536,
        device="cuda:0",
        use_prefix_cache=True,
        batch_multiq=True,
        max_batch_tokens=12288,
        layout=None,
        bidir=None,
        read_twice_tokens=None,
        **options,
    ):
        super().__init__(**options)
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

        t0 = time.time()
        self.torch = torch
        self.dev = torch.device(device)
        self.max_len = int(max_len)
        self.temperature = float(temperature)
        self.max_batch_tokens = int(max_batch_tokens)
        self.use_prefix_cache = bool(int(use_prefix_cache)) if isinstance(use_prefix_cache, str) else bool(use_prefix_cache)
        self.batch_multiq = bool(int(batch_multiq)) if isinstance(batch_multiq, str) else bool(batch_multiq)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.codes, self.code_ids = LYT.build_codes(self.tok, LYT.N_CODES)

        # Parse weighted checkpoint spec: "path1:w1;path2:w2" (or comma-separated; single ckpt -> [(ckpt, 1.0)])
        ck_items = []
        if ckpt_spec:
            for part in str(ckpt_spec).replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" in part:
                    p_str, w_str = part.rsplit(":", 1)
                    ck_items.append((Path(p_str.strip()), float(w_str)))
                else:
                    ck_items.append((Path(part), 1.0))
        elif ckpt:
            ck_items.append((Path(ckpt), 1.0))

        st = {}
        if ck_items:
            st = json.loads((ck_items[0][0] / "state.json").read_text())

        self.layout = layout or st.get("layout") or LYT.LAYOUT_V1
        if self.layout not in (LYT.LAYOUT_V1, LYT.LAYOUT_V2):
            raise SystemExit(f"Unexpected checkpoint layout: {self.layout!r}")
        if bidir is not None:
            self.bidir = bool(int(bidir)) if isinstance(bidir, str) else bool(bidir)
        else:
            self.bidir = bool(st.get("bidir", False))
        if read_twice_tokens is not None:
            self.read_twice_tokens = int(read_twice_tokens)
        else:
            self.read_twice_tokens = int(st.get("read_twice_tokens", 0))
        if self.bidir:
            self.use_prefix_cache = False
            if ck_items and not st.get("batch_safe", True):
                self.batch_multiq = False

        cfg = AutoConfig.from_pretrained(model).get_text_config()
        m, info = Qwen3_5ForCausalLM.from_pretrained(
            model,
            config=cfg,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        missing = [k for k in (info.get("missing_keys") or []) if k != "lm_head.weight"]
        if missing:
            raise SystemExit(f"Qwen3_5ForCausalLM missing keys: {missing[:5]}")
        m = m.to(self.dev)
        if self.bidir:
            m.set_attn_implementation(LYT.register_bidir())

        W0_codes = m.get_output_embeddings().weight.detach()[self.code_ids].float().clone()
        merged_n, scale, step = 0, 0.0, 0
        if ck_items:
            # Single-rounding fp32 weighted LoRA merge + weighted readout blend (exact for soups & WiSE-FT)
            deltas = {}
            sum_w = sum(w for _, w in ck_items)
            h_w = (1.0 - sum_w) * W0_codes
            h_b = torch.zeros(len(self.code_ids), dtype=torch.float32, device=self.dev)
            for ck, cw in ck_items:
                sti = json.loads((ck / "state.json").read_text())
                if sti.get("codes") != self.codes or sti.get("code_ids") != self.code_ids:
                    raise SystemExit(f"Checkpoint {ck} codes/code_ids mismatch tokenizer!")
                sd = load_file(str(ck / "adapter.safetensors"), device=str(self.dev))
                sc = (float(sti["lora_alpha"]) / float(sti["lora_r"])) * cw
                scale += sc
                step = int(sti["step"])
                with torch.no_grad():
                    for k in sd:
                        if not k.endswith(".lora_A.weight"):
                            continue
                        mod_name = k[len("base_model.model.") : -len(".lora_A.weight")]
                        d_i = (sd[k.replace(".lora_A.weight", ".lora_B.weight")].float() @ sd[k].float()) * sc
                        if mod_name in deltas:
                            deltas[mod_name].add_(d_i)
                        else:
                            deltas[mod_name] = d_i
                del sd
                hi = torch.load(ck / "readout.pt", map_location=self.dev)
                h_w.add_(hi["weight"].float(), alpha=cw)
                h_b.add_(hi["bias"].float(), alpha=cw)
            with torch.no_grad():
                for mod_name, d_tot in deltas.items():
                    mod = m.get_submodule(mod_name)
                    w = mod.weight.float()
                    w.add_(d_tot)
                    mod.weight.copy_(w.to(mod.weight.dtype))
                    merged_n += 1
            del deltas
            torch.cuda.empty_cache()
            self.head = torch.nn.Linear(h_w.shape[1], h_w.shape[0], bias=True).to(self.dev, torch.float32)
            with torch.no_grad():
                self.head.weight.copy_(h_w / self.temperature)
                self.head.bias.copy_(h_b / self.temperature)
        else:
            # Zero-shot mode: initialize readout from tied/untied lm_head rows
            self.head = torch.nn.Linear(W0_codes.shape[1], len(self.code_ids), bias=True).to(self.dev, torch.float32)
            with torch.no_grad():
                self.head.weight.copy_(W0_codes / self.temperature)
                self.head.bias.zero_()

        self.backbone = m.model.eval()
        self.model = m.eval()
        self.provenance = {
            "kind": "in-process",
            "model": str(model),
            "ckpt": str(ckpt) if ckpt else None,
            "ckpt_spec": str(ckpt_spec) if ckpt_spec else None,
            "temperature": self.temperature,
            "step": step,
            "merged_lora_modules": merged_n,
            "lora_scale": round(scale, 6),
            "layout": self.layout,
            "bidir": self.bidir,
            "read_twice_tokens": self.read_twice_tokens,
            "n_codes": len(self.codes),
            "max_len": self.max_len,
            "use_prefix_cache": self.use_prefix_cache,
            "batch_multiq": self.batch_multiq,
            "load_seconds": round(time.time() - t0, 2),
        }

    def synchronize(self):
        self.torch.cuda.synchronize(self.dev)

    def _compile(self, state, questions):
        state_text = LYT.describe(state)
        preview = LYT.question_preview(questions)
        compiled = []
        for qkey, q in questions.items():
            kind = q.get("type")
            if kind not in ("choice", "noul", "score"):
                raise Unsupported(f"Unsupported question type: {kind!r}")
            keys, opt_texts = LYT.options_of(q)
            if len(keys) < 2:
                raise Unsupported(f"Question {qkey!r} has < 2 options")
            if len(keys) > len(self.codes):
                raise Unsupported(f"Question {qkey!r} has {len(keys)} > {len(self.codes)} options")
            ids, _ = LYT.build_prompt_ids(
                self.tok,
                state_text,
                preview,
                LYT.instructions_of(q),
                opt_texts,
                self.codes,
                layout=self.layout,
                read_twice_tokens=self.read_twice_tokens,
            )
            if not ids or len(ids) > self.max_len:
                raise Unsupported(f"Question {qkey!r} length {len(ids)} > max_len {self.max_len}")
            compiled.append((qkey, q, keys, ids))
        return compiled

    def _score_compiled(self, compiled):
        torch = self.torch
        out_logits = {}
        if self.bidir:
            # Non-causal full attention: batch multi-question requests safely using right-padded bool mask
            with torch.inference_mode():
                if self.batch_multiq and len(compiled) > 1:
                    order = sorted(range(len(compiled)), key=lambda i: len(compiled[i][3]))
                    batches, cur = [], []
                    for i in order:
                        it = compiled[i]
                        L = len(it[3])
                        if cur and (len(cur) >= 8 or L * (len(cur) + 1) > self.max_batch_tokens):
                            batches.append(cur)
                            cur = [it]
                        else:
                            cur.append(it)
                    if cur:
                        batches.append(cur)
                else:
                    batches = [[c] for c in compiled]

                for batch in batches:
                    if len(batch) == 1:
                        qkey, q, keys, ids = batch[0]
                        t_ids = torch.tensor([ids], dtype=torch.long, device=self.dev)
                        out = self.backbone(input_ids=t_ids, use_cache=False)
                        h = out.last_hidden_state[0, -1].float()
                        del out
                        out_logits[qkey] = self.head(h)[: len(keys)].cpu().tolist()
                    else:
                        bsz = len(batch)
                        lens = [len(c[3]) for c in batch]
                        maxlen = max(lens)
                        t_ids = torch.full((bsz, maxlen), self.pad_id, dtype=torch.long, device=self.dev)
                        mask = torch.zeros((bsz, maxlen), dtype=torch.long, device=self.dev)
                        for b, c in enumerate(batch):
                            L = len(c[3])
                            t_ids[b, :L] = torch.tensor(c[3], dtype=torch.long, device=self.dev)
                            mask[b, :L] = 1
                        out = self.backbone(input_ids=t_ids, attention_mask=mask, use_cache=False)
                        last_idx = torch.tensor([L - 1 for L in lens], dtype=torch.long, device=self.dev)
                        hs = out.last_hidden_state[torch.arange(bsz, device=self.dev), last_idx].float()
                        zs = self.head(hs).cpu()
                        del out
                        for b, (qkey, q, keys, _) in enumerate(batch):
                            out_logits[qkey] = zs[b, : len(keys)].tolist()
            return out_logits

        seqs = [c[3] for c in compiled]
        plen = LYT.common_prefix_len(seqs) if (self.use_prefix_cache and len(compiled) > 1) else 0
        with torch.inference_mode():
            cache = None
            if plen > 0:
                p_ids = torch.tensor([seqs[0][:plen]], dtype=torch.long, device=self.dev)
                p_mask = torch.ones_like(p_ids)
                p_out = self.backbone(input_ids=p_ids, attention_mask=p_mask, use_cache=True)
                cache = p_out.past_key_values
                del p_out
            for qkey, q, keys, ids in compiled:
                if cache is not None:
                    suf = ids[plen:]
                    s_ids = torch.tensor([suf], dtype=torch.long, device=self.dev)
                    s_mask = torch.ones((1, len(ids)), dtype=torch.long, device=self.dev)
                    s_pos = torch.arange(plen, len(ids), dtype=torch.long, device=self.dev).unsqueeze(0)
                    bc = copy.deepcopy(cache)
                    out = self.backbone(
                        input_ids=s_ids,
                        attention_mask=s_mask,
                        position_ids=s_pos,
                        past_key_values=bc,
                        use_cache=True,
                    )
                    h = out.last_hidden_state[0, -1].float()
                    del out, bc
                else:
                    t_ids = torch.tensor([ids], dtype=torch.long, device=self.dev)
                    out = self.backbone(input_ids=t_ids, use_cache=False)
                    h = out.last_hidden_state[0, -1].float()
                    del out
                z = self.head(h)[: len(keys)]
                out_logits[qkey] = z.cpu().tolist()
        return out_logits

    def __call__(self, state, questions):
        torch = self.torch
        compiled = self._compile(state, questions)
        logits_map = self._score_compiled(compiled)
        answers = {}
        for qkey, q, keys, _ in compiled:
            z = torch.tensor(logits_map[qkey], dtype=torch.float32)
            probs = torch.softmax(z, dim=-1).tolist()
            dist = {k: float(p) for k, p in zip(keys, probs)}
            kind = q["type"]
            if kind == "noul":
                answers[qkey] = {"type": "noul", "noul": min(1.0, max(0.0, dist["true"]))}
            elif kind == "score":
                exp_score = sum(i * p for i, p in enumerate(probs))
                answers[qkey] = {
                    "type": "score",
                    "score": float(exp_score),
                    "probabilities": dist,
                    "legend": {str(i): v for i, v in enumerate(q["criteria"])},
                }
            else:
                best = max(keys, key=lambda kk: (dist[kk], -keys.index(kk)))
                answers[qkey] = {"type": "choice", "choice": best, "probabilities": dist}
        return {"model": self.name, "answers": answers}, None
