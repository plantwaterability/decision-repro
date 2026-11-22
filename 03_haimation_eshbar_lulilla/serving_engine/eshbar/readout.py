from __future__ import annotations
import json
import os
import numpy as np
from . import render as core
from . import heads as _heads
from .prompting import PromptEngine, check_state_shown
from decision_index.engines.base import Engine, Unsupported

class ReadoutEngine(PromptEngine):

    def __init__(self, model='', device='cuda', max_len=32768, tokens_per_batch=32768, attn='packed_doc', samples=0, _backend=None, dtype='bfloat16', ordinal=False, **options):
        Engine.__init__(self, **options)
        self.max_len, self.tokens_per_batch = (int(max_len), int(tokens_per_batch))
        self.ordinal = str(ordinal).lower() in ('1', 'true', 'yes')
        self._kind, self._ord_keys = ({}, set())
        if _backend is not None:
            self.tok, self.chat, self._forward, self._sync, self.head, self.head_meta = _backend
            self.samples = int(samples) or int(self.head_meta.get('samples', 64))
            return
        self._load_readout(model, device, attn, samples, dtype)

    def _load_readout(self, model, device, attn, samples, dtype='bfloat16'):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        mdir = model
        if not os.path.isdir(mdir):
            from huggingface_hub import snapshot_download
            mdir = snapshot_download(model)
        cfg_raw = json.load(open(os.path.join(mdir, 'config.json')))
        if 'requires_head' not in cfg_raw:
            raise SystemExit(f'{mdir}: config.json has no requires_head marker')
        tcr = cfg_raw['text_config'] if isinstance(cfg_raw.get('text_config'), dict) else cfg_raw
        policy = tcr.get('attention_policy', 'C')
        if policy != 'C' and attn != 'packed_doc':
            raise SystemExit(f"{mdir}: attention policy {policy} needs attn='packed_doc' (got {attn!r}: it would serve the model causally)")
        self.head, self.head_meta = _heads.load_head(mdir)
        if len(self.head.heads) != 1:
            raise SystemExit('the export must carry exactly the selected head')
        mcfg = AutoConfig.from_pretrained(mdir)
        _, model_cls = core.model_loader(mcfg, 'auto')
        self.chat = core.chat_preset(mcfg.model_type, False)
        tok = AutoTokenizer.from_pretrained(mdir)
        tok.padding_side = 'right'
        from . import labels as _labels
        label_ids = _labels.token_ids(tok)
        n_head = len(self.head_meta['label_ids'])
        if label_ids[:n_head] != list(self.head_meta['label_ids']):
            raise SystemExit('the head was trained with other label tokens')
        kw = {}
        if attn == 'packed_doc':
            from . import attention as fm
            fm.register()
            kw['attn_implementation'] = fm.ATTN_NAME
        if dtype not in ('bfloat16', 'float32'):
            raise SystemExit(f'dtype {dtype!r}: bfloat16 (serving) or float32 (parity tests)')
        m = model_cls.from_pretrained(mdir, dtype=getattr(torch, dtype), device_map=device, **kw).eval()
        if attn == 'packed_doc':
            fm.require_policy(m.config.get_text_config() if hasattr(m.config, 'get_text_config') else m.config, policy)
        self.tok, self.model, self.device = (tok, m, device)
        self.head = self.head.to(device)
        lab = torch.tensor(label_ids, device=device)
        tcfg = mcfg.get_text_config() if hasattr(mcfg, 'get_text_config') else mcfg
        cap = float(getattr(tcfg, 'final_logit_softcapping', 0) or 0)
        mode = 'pack' if attn == 'packed_doc' else 'pad'
        readout = self.head_meta.get('letter_readout')
        _heads.readout_spec(readout)

        @torch.no_grad()
        def forward(texts):
            keep = []
            z = core.exact_letter_logits(m, tok, texts, lab, device, mode, head='fp32', keep_h=keep)
            h = torch.cat(keep, 0)
            E = m.get_output_embeddings().weight[lab]
            ell = _heads.letter_logits_fp32(h, E, cap)
            if not torch.equal(ell, z):
                raise RuntimeError('letter readout mismatch')
            if readout and readout != 'stock':
                ell = _heads.letter_readout_fp32(h, E, cap, readout)
            return (h, ell)
        self._forward = forward
        self._sync = (lambda: torch.cuda.synchronize()) if str(device).startswith('cuda') else lambda: None
        self.samples = int(samples) or int(self.head_meta.get('samples', 64))
        import transformers

    def run_prompts(self, items):
        import torch
        out = {}
        if not hasattr(self, 'last_raw'):
            self.last_raw = {}
        lens = [len(self.tok(t, add_special_tokens=False)['input_ids']) for _, _, t in items]
        i = 0
        while i < len(items):
            j, mx = (i, 0)
            while j < len(items) and max(mx, lens[j]) * (j - i + 1) <= max(self.tokens_per_batch, lens[i]):
                mx = max(mx, lens[j])
                j += 1
            h, ell = self._forward([t for _, _, t in items[i:j]])
            ks = torch.tensor([len(o) for _, o, _ in items[i:j]], device=ell.device)
            kmax = int(ks.max())
            kw = {}
            if getattr(self.head, 'needs_kinds', False):
                kw['kinds'] = torch.tensor([_heads.KIND_CODE.get(self._kind.get(qid, 'choice'), 0) for qid, _, _ in items[i:j]], dtype=torch.long)
            if getattr(self.head, 'needs_yes_no', False):
                kw['yes_no'] = _heads.yes_no_tensor([t for _, _, t in items[i:j]], [len(o) for _, o, _ in items[i:j]]).to(ell.device)
            p, epi, alea = self.head.predict(h, ell[:, :kmax], ks, self.samples, int(self.head_meta.get('draw_seed', 0)), self.head_meta.get('draw_grid_size'), int(self.head_meta.get('draw_slot', 0)), temperature=float(self.head_meta.get('temperature', 1.0)), **kw)
            p = p[0].double().cpu().numpy()
            nat = torch.softmax(ell[:, :kmax].double().masked_fill(torch.arange(kmax, device=ell.device)[None] >= ks[:, None], float('-inf')), -1).cpu().numpy()
            for b, (qid, order, _) in enumerate(items[i:j]):
                k = len(order)
                pb = p[b, :k] / p[b, :k].sum()
                q = np.zeros(k)
                q[np.asarray(order)] = pb
                nq = np.zeros(k)
                nq[np.asarray(order)] = nat[b, :k]
                out[qid] = q.tolist()
                self.last_raw[qid] = {'native': nq.tolist(), 'epistemic': float(epi[0, b]), 'aleatoric': float(alea[0, b])}
                if 'yes_no' in kw:
                    self.last_raw[qid]['yes_no'] = bool(kw['yes_no'][b, 0] >= 0)
                if 'kinds' in kw:
                    self.last_raw[qid]['kind'] = self._kind.get(qid, 'choice')
            i = j
        return out

    def prompt(self, u, qid):
        if qid.split('|', 1)[-1] in self._ord_keys:
            order = list(range(len(u['option_keys'])))
            text = core.build_prompt(self.tok, u, order, self.max_len, chat=self.chat, noul_defs=True)
            check_state_shown(u['state'], text, qid.split('|', 1)[1], self.max_len)
            self._kind[qid] = 'score'
            return (order, text)
        order, text = super().prompt(u, qid)
        self._kind[qid] = u['qtype'] if u['qtype'] in ('choice', 'noul') else 'choice'
        return (order, text)

    def __call__(self, state, questions):
        self.last_raw = {}
        self._kind = {}
        self._ord_keys = {k for k, q in questions.items() if self.ordinal and q.get('type') == 'choice' and (q.get('ordinal') is True)}
        resp, _ = super().__call__(state, questions)
        return (resp, {'head': getattr(self, 'last_raw', {})})
