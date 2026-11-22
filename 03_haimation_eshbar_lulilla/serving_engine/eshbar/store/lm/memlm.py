from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import ExactScorer, GraphTensors, KnowledgeGraph
from .modules import MemConfig, MemoryBlock, copy_inputs, copy_logprobs_for_targets, copy_mixture_probs

class StopForward(Exception):
    pass

class MemContext:

    def __init__(self):
        self.graph: GraphTensors | None = None
        self.enabled = True
        self.capture: list | None = None
        self.aux_idx = None
        self.aux_mention = None
        self.state = None
        self.write_mask = None
        self.stop_after_block = False
        self.clear()

    def clear(self):
        self.cand_ids = self.cand_w = self.kg_aux = self.copy_x = self.link_cache = self.gates = None
        self.relm_cache = self.dec_cache = None
        self.pad_cache = self.pad_logit_dir = self.ro_gate = None
        self.cand_hist_ids = self.cand_hist_w = None
        self.lam_hist = None
        self.tok_ids = None

class MemLayer(nn.Module):

    def __init__(self, layer: nn.Module, block: MemoryBlock, ctx: MemContext):
        super().__init__()
        self.layer, self.block, self.ctx = (layer, block, ctx)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('layer'), name)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.layer(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        if self.ctx.capture is not None:
            self.ctx.capture.append(h)
            raise StopForward
        if self.ctx.enabled:
            h = self.block(h, self.ctx)
        if self.ctx.stop_after_block:
            raise StopForward
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    skip = ('vision', 'visual', 'audio', 'mtp', 'encoder')
    cands = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ModuleList) and n.endswith('layers') and (not any((s in n for s in skip)))]
    if not cands:
        raise ValueError('could not find decoder layers')
    return max(cands, key=lambda x: len(x[1]))[1]

def pick_layer(model: nn.Module, n_layers: int, frac: float=2 / 3) -> int:
    tc = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config
    target = round(frac * (n_layers - 1))
    types = getattr(tc, 'layer_types', None)
    if types:
        full = [i for i, t in enumerate(types) if 'full' in str(t)]
        if full:
            return min(full, key=lambda i: abs(i - target))
    return target

@dataclass
class Batch:
    ids: torch.Tensor
    labels: torch.Tensor
    attn: torch.Tensor
    spans: list
    aux_idx: torch.Tensor | None = None
    aux_ent: torch.Tensor | None = None
    aux_rel: torch.Tensor | None = None
    aux_hops: torch.Tensor | None = None
    aux_mention: torch.Tensor | None = None
    prefix: 'Batch | None' = None
    ptr_w: torch.Tensor | None = None
    ptr_r: torch.Tensor | None = None
    beta_tgt: torch.Tensor | None = None
    addr_pair: torch.Tensor | None = None
    addr_neg: torch.Tensor | None = None
    addr_same: torch.Tensor | None = None
    plain: bool = False
    aux_rel_mention: torch.Tensor | None = None
    aux_rel_span_start: torch.Tensor | None = None

def _cross_pairs(pair: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    last: dict[int, set] = {}
    for b, p1, p2 in neg.tolist():
        last.setdefault(b, set()).update((p1, p2))
    out = [(b, q, p) for b, q, v in pair.tolist() for p in sorted(last.get(b, ())) if p != v]
    return torch.tensor(out, dtype=torch.long, device=pair.device).view(-1, 3)

class MemoryLM:

    def __init__(self, model, tokenizer, cfg: MemConfig=MemConfig(), layer_idx: int | None=None, name_prefix: str='Entity:', add_bos: bool | None=None, device=None):
        self.lm, self.tok, self.cfg = (model, tokenizer, cfg)
        for p in model.parameters():
            p.requires_grad_(False)
        layers = find_decoder_layers(model)
        self.layer_idx = pick_layer(model, len(layers)) if layer_idx is None else layer_idx % len(layers)
        tc = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config
        dev = torch.device(device) if device is not None else next(layers[self.layer_idx].parameters()).device
        self.device = dev
        self.block = MemoryBlock(tc.hidden_size, cfg).to(dev)
        end = tokenizer('\n', add_special_tokens=False).input_ids
        assert not cfg.end_gate or len(end) == 1, f'end_gate needs a one-token line end, got {end}'
        self.block.end_id = end[-1]
        self.ctx = MemContext()
        layers[self.layer_idx] = MemLayer(layers[self.layer_idx], self.block, self.ctx)
        self.name_prefix = name_prefix
        bos = tokenizer.bos_token_id
        if add_bos is None:
            mtype = str(getattr(model.config, 'model_type', ''))
            add_bos = tokenizer('a').input_ids[:1] == [bos] or mtype.startswith('gemma')
        self.bos = [bos] if bos is not None and add_bos else []

    def tokenize_name(self, name: str) -> list[int]:
        return self.tok(' ' + name, add_special_tokens=False).input_ids

    @torch.no_grad()
    def encode_names(self, names: list[str], bs: int=256, pool: str | None=None) -> torch.Tensor:
        pool = pool or self.cfg.key_pool
        pre = self.bos + self.tok(self.name_prefix, add_special_tokens=False).input_ids
        out = []
        was = self.ctx.enabled
        for i in range(0, len(names), bs):
            seqs = [pre + self.tokenize_name(n) for n in names[i:i + bs]]
            T = max(map(len, seqs))
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=self.device)
            attn = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=self.device)
            name_mask = attn.clone()
            name_mask[:, :len(pre)] = 0
            self.ctx.capture, self.ctx.enabled = ([], False)
            try:
                self.lm(input_ids=ids, attention_mask=attn, use_cache=False)
            except StopForward:
                pass
            finally:
                cap = self.ctx.capture
                self.ctx.capture, self.ctx.enabled = (None, was)
            dtype, h = (cap[0].dtype, self.block.standardize(cap[0]))
            if pool == 'first':
                out.append(h[:, len(pre)].to(dtype))
            elif pool == 'last':
                last = attn.sum(1) - 1
                out.append(h[torch.arange(h.shape[0]), last].to(dtype))
            else:
                m = name_mask[..., None].float()
                out.append(((h * m).sum(1) / m.sum(1)).to(dtype))
        return torch.cat(out)

    @torch.no_grad()
    def hidden_at_layer(self, ids, attn=None) -> torch.Tensor:
        was = self.ctx.enabled
        self.ctx.capture, self.ctx.enabled = ([], False)
        try:
            self.lm(input_ids=ids, attention_mask=attn, use_cache=False)
        except StopForward:
            pass
        finally:
            cap = self.ctx.capture
            self.ctx.capture, self.ctx.enabled = (None, was)
        return cap[0]

    def stage1_loss(self, b: Batch, h, prefix_h=None, aux_weight: float=1.0):
        state = self._prefix_state(b, cached_h=prefix_h) if b.prefix is not None else None
        self.block_forward(h, b.attn, b.aux_idx, b.aux_mention, state)
        return self.aux_loss(b, aux_weight)

    def load_graph(self, kg: KnowledgeGraph, scorer=None, max_name_tokens: int=12):
        toks = [self.tokenize_name(n)[:max_name_tokens] for n in kg.entities]
        L = max(map(len, toks))
        name_tok = torch.full((len(toks), L), -1, dtype=torch.long)
        alias: dict = {}
        for i, t in enumerate(toks):
            name_tok[i, :len(t)] = torch.tensor(t)
            alias[tuple(t)] = i if tuple(t) not in alias else -1
        self.ctx.graph = GraphTensors(ent_keys=self.encode_names(kg.entities, pool=self.cfg.key_pool), rel_keys=self.encode_names(kg.relations, pool=self.cfg.rel_key_pool), name_tok=name_tok.to(self.device), name_len=torch.tensor(list(map(len, toks)), device=self.device), scorer=scorer or ExactScorer(kg, self.cfg.max_tails, self.device), kg=kg, rel_keys_first=self.encode_names(kg.relations, pool='first') if self.cfg.relm_raw_cos or self.cfg.relm_first else None, alias=alias, rel_alias=self._alias(kg.relations))

    def _alias(self, names):
        out: dict = {}
        for i, n in enumerate(names):
            t = tuple(self.tokenize_name(n))
            out[t] = i if t not in out else -1
        return out

    def unload_graph(self):
        self.ctx.graph = None

    def trainable_parameters(self):
        return [p for p in self.block.parameters() if p.requires_grad]

    def lm_step(self, keep=None, **kw):
        if keep is not None:
            kw['logits_to_keep'] = keep
        out = self.lm(**kw)
        d = self.ctx.pad_logit_dir
        if d is not None and self.ctx.enabled:
            if isinstance(keep, int) and keep > 0:
                d = d[:, -keep:]
            elif torch.is_tensor(keep):
                d = d[:, keep.to(d.device)]
            W = self.lm.get_output_embeddings().weight
            out.logits = out.logits + (d.to(W.device, W.dtype) @ W.T).to(out.logits.device, out.logits.dtype)
        return out

    def forward(self, ids, attn=None, aux_idx=None, state=None, aux_mention=None, keep=None):
        self.ctx.clear()
        self.ctx.aux_idx = aux_idx
        self.ctx.state = state
        self.ctx.aux_mention = aux_mention
        self.ctx.write_mask = attn
        self.ctx.tok_ids = ids
        return self.lm_step(keep=keep, input_ids=ids, attention_mask=attn, use_cache=False).logits

    def block_forward(self, h, attn=None, aux_idx=None, aux_mention=None, state=None):
        self.ctx.clear()
        self.ctx.aux_idx = aux_idx
        self.ctx.state = state
        self.ctx.aux_mention = aux_mention
        self.ctx.write_mask = attn
        return self.block(h.to(self.device), self.ctx)

    def write_session(self, ids, attn=None, state=None):
        self.ctx.clear()
        self.ctx.aux_idx = None
        self.ctx.state = state
        self.ctx.aux_mention = None
        self.ctx.write_mask = attn
        self.ctx.tok_ids = ids
        self.ctx.stop_after_block = True
        try:
            self.lm(input_ids=ids, attention_mask=attn, use_cache=False)
        except StopForward:
            pass
        finally:
            self.ctx.stop_after_block = False
        return self.ctx.state

    def _prefix_state(self, b: Batch, cached_h=None):
        if b.prefix is None:
            return None
        if cached_h is None:
            self.write_session(b.prefix.ids, b.prefix.attn)
        else:
            self.block_forward(cached_h, b.prefix.attn)
        self._prefix_aux = getattr(self.block.pad, 'last_aux', None) if self.block.pad is not None else None
        return self.ctx.state

    @staticmethod
    def answer_positions(b: Batch):
        return (b.labels[:, 1:] != -100).any(0).nonzero().squeeze(-1)

    def _logits_at(self, b: Batch, bi, ti, state):
        base = getattr(self.lm, getattr(self.lm, 'base_model_prefix', ''), None)
        if base is None or base is self.lm:
            return None
        self.ctx.clear()
        self.ctx.aux_idx = b.aux_idx
        self.ctx.state = state
        self.ctx.aux_mention = b.aux_mention
        self.ctx.write_mask = b.attn
        h = base(input_ids=b.ids, attention_mask=b.attn, use_cache=False).last_hidden_state[bi, ti]
        head = self.lm.get_output_embeddings()
        logits = head(h.to(head.weight.device))
        cap = getattr(self.lm.config.get_text_config(), 'final_logit_softcapping', None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        d = self.ctx.pad_logit_dir
        if d is not None and self.ctx.enabled:
            W = head.weight
            logits = logits + (d[bi.to(d.device), ti.to(d.device)].to(W.device, W.dtype) @ W.T).to(logits.dtype)
        return logits

    def answer_logprobs(self, b: Batch, sparse_logits: bool=True, stop_weight: float=1.0):
        bi, ti = (b.labels[:, 1:] != -100).nonzero(as_tuple=True)
        state = self._prefix_state(b)
        logits = self._logits_at(b, bi, ti, state) if sparse_logits else None
        if logits is None:
            keep = self.answer_positions(b).to(b.ids.device)
            full = self.forward(b.ids, b.attn, b.aux_idx, state=state, aux_mention=b.aux_mention, keep=keep if sparse_logits else None)
            col = full if sparse_logits else full[:, keep]
            logits = col[bi, torch.searchsorted(keep, ti)]
        tgt = b.ids[bi, ti + 1]
        lp = copy_logprobs_for_targets(self.block, self.ctx, b.ids, logits, tgt, keep=(bi, ti))
        w = torch.ones_like(lp)
        if stop_weight != 1.0:
            end = torch.zeros(b.ids.shape, dtype=torch.bool, device=b.ids.device)
            for r, _, e, _ in b.spans:
                end[r, e - 1] = e < b.ids.shape[1]
            w = torch.where(end[bi, ti].to(lp.device), stop_weight, 1.0)
        return (lp, w)

    def plain_loss(self, b: Batch, aux_weight: float=1.0):
        with torch.no_grad():
            was = self.ctx.enabled
            self.ctx.enabled = False
            try:
                base = self.forward(b.ids, b.attn).float().log_softmax(-1)
            finally:
                self.ctx.enabled = was
        lp = self.forward(b.ids, b.attn).float().log_softmax(-1)
        if self.ctx.graph is not None and self.ctx.cand_ids is not None:
            nxt, w, mass, lam = (t.to(lp.device) for t in copy_inputs(self.block, self.ctx, b.ids))
            ptr = torch.zeros_like(lp).scatter_add_(-1, nxt, w) / mass.clamp(min=1e-06)[..., None]
            lp = ((1 - lam[..., None]) * lp.exp() + lam[..., None] * ptr + 1e-12).log()
        m = b.attn.bool()
        kl = (base.exp() * (base - lp)).sum(-1)[m].mean()
        aux, logs = self.aux_loss(b, aux_weight) if aux_weight else (0.0, {})
        logs['kl_plain'] = kl.item()
        return (kl + aux, logs)

    def question_kl(self, b: Batch):
        m = (b.labels[:, 1:] == -100) & (b.attn[:, 1:] > 0)
        bi, ti = m.nonzero(as_tuple=True)
        with torch.no_grad():
            was = self.ctx.enabled
            self.ctx.enabled = False
            try:
                base = self.forward(b.ids, b.attn)[bi, ti].float().log_softmax(-1)
            finally:
                self.ctx.enabled = was
        logits = self.forward(b.ids, b.attn, state=None)
        p = logits[bi, ti].float().softmax(-1)
        if self.ctx.graph is not None and self.ctx.cand_ids is not None:
            nxt, w, mass, lam = (t[bi, ti].to(p.device) for t in copy_inputs(self.block, self.ctx, b.ids))
            ptr = torch.zeros_like(p).scatter_add_(-1, nxt, w) / mass.clamp(min=1e-06)[:, None]
            p = (1 - lam[:, None]) * p + lam[:, None] * ptr
        return (base.exp() * (base - (p + 1e-12).log())).sum(-1).mean()

    def loss(self, b: Batch, aux_weight: float=1.0, sparse_logits: bool=True, keep_base: float=0.0, stop_weight: float=1.0):
        if b.plain:
            return self.plain_loss(b, aux_weight)
        lp, mask = self.answer_logprobs(b, sparse_logits, stop_weight)
        nll = -(lp * mask).sum() / mask.sum().clamp(min=1)
        aux, logs = self.aux_loss(b, aux_weight) if aux_weight else (0.0, {})
        logs['nll'] = nll.item()
        total = nll + aux
        if keep_base and b.prefix is None and (b.aux_idx is not None):
            kl = self.question_kl(b)
            logs['kl_question'] = kl.item()
            total = total + keep_base * kl
        return (total, logs)

    def aux_loss(self, b: Batch, aux_weight: float=1.0):
        loss, logs = (torch.zeros((), device=self.device), {})
        if self.block.pad is not None and (b.ptr_r is not None or b.prefix is not None):
            pad_aux = self._pad_aux_loss(b)
            loss = loss + pad_aux
            logs['aux_pad'] = pad_aux.item()
        g = self.ctx.gates
        if g is not None and self.ctx.graph is not None and (self.block.kg is not None) and self.cfg.kg_gate_loss:
            tgt = torch.zeros(b.labels.shape, device=g.device)
            if b.aux_idx is not None:
                tgt[:, :-1] = (b.labels[:, 1:] != -100).float()
            m = b.attn.bool()
            w = torch.where(tgt[m] > 0, 4.0, 1.0)
            gv = torch.nan_to_num(g[..., 0][m], nan=0.5)
            gl = F.binary_cross_entropy(gv.clamp(1e-05, 1 - 1e-05), tgt[m], weight=w)
            loss = loss + self.cfg.kg_gate_loss * gl
            logs['aux_gate'] = gl.item()
        rg = self.ctx.ro_gate
        if rg is not None:
            tgt = torch.zeros(b.labels.shape, device=rg.device)
            if b.addr_pair is not None:
                tgt[:, :-1] = (b.labels[:, 1:] != -100).float()
            m = b.attn.bool()
            w = torch.where(tgt[m] > 0, 4.0, 1.0)
            rl = F.binary_cross_entropy(rg[..., 0][m].clamp(1e-05, 1 - 1e-05), tgt[m], weight=w)
            loss = loss + rl
            logs['aux_ro_gate'] = rl.item()
        a = self.ctx.kg_aux
        if a is not None and b.aux_idx is not None and b.aux_idx.numel():
            e = a['ent']
            ok = b.aux_mention >= 0
            hit = (e['q_w'] * (e['q_ids'] == b.aux_ent[:, None])).sum(-1)
            aux = -torch.log(hit + 1e-06).mean()
            if ok.any():
                aux = aux + F.cross_entropy(e['att'][ok], b.aux_mention[ok]) + F.cross_entropy(e['tok'][ok], b.aux_ent[ok])
            logs['link_acc'] = (hit > 0.5).float().mean().item()
            for hop, rl in enumerate(a['rel']):
                if (b.aux_rel[:, hop] != -100).any():
                    aux = aux + F.cross_entropy(rl, b.aux_rel[:, hop], ignore_index=-100)
                    sel = b.aux_rel[:, hop] != -100
                    logs[f'rel{hop}_acc'] = (rl[sel].argmax(-1) == b.aux_rel[sel, hop]).float().mean().item()
            aux = aux + F.cross_entropy(a['halt'], b.aux_hops)
            if self.cfg.halt_rel_aux and a.get('halt_rel') is not None:
                hr = F.cross_entropy(a['halt_rel'], b.aux_hops)
                aux = aux + self.cfg.halt_rel_aux * hr
                logs['aux_halt_rel'] = hr.item()
            rm = a.get('relm')
            if rm is not None:
                T = b.ids.shape[1]
                for hop in range(len(rm['p'])):
                    gold = b.aux_rel[:, hop]
                    sel = gold != -100
                    if not sel.any():
                        continue
                    aux = aux - torch.log(rm['p'][hop][sel].gather(1, gold[sel, None]).squeeze(1) + 1e-06).mean()
                    logs[f'relm{hop}_acc'] = (rm['p'][hop][sel].argmax(-1) == gold[sel]).float().mean().item()
                    if b.aux_rel_mention is not None:
                        pos = b.aux_rel_mention[:, hop]
                        ok = sel & (pos >= 0)
                        if ok.any():
                            first = b.aux_rel_span_start[:, hop] if b.aux_rel_span_start is not None else pos
                            first = torch.where(first >= 0, first, pos)
                            lp = rm['att'][hop][ok].log_softmax(-1)
                            cols = torch.arange(lp.shape[-1], device=lp.device)[None, :]
                            span = (cols >= first[ok, None]) & (cols <= pos[ok, None])
                            aux = aux - lp.masked_fill(~span, -1000000000.0).logsumexp(-1).mean()
                            rows = b.aux_idx[ok] // T * T
                            aux = aux + F.cross_entropy(rm['tok'][rows + pos[ok]], gold[ok])
                            if self.cfg.relm_first:
                                aux = aux + F.cross_entropy(rm['tok'][rows + first[ok]], gold[ok])
            loss = loss + aux
            logs['aux'] = aux.item()
        return (aux_weight * loss, logs)

    def _pad_aux_loss(self, b: Batch):

        def ptr_ce(logits, idx):
            return F.cross_entropy(logits[idx[:, 0], idx[:, 1]], idx[:, 2])

        def beta_bce(beta, tgt):
            m = tgt >= 0
            if not m.any():
                return beta.sum() * 0
            w = torch.where(tgt[m] > 0, 4.0, 1.0)
            return F.binary_cross_entropy(beta[m].clamp(1e-05, 1 - 1e-05), tgt[m].float(), weight=w)
        cur = self.block.pad.last_aux
        loss = torch.zeros((), device=self.device)
        if b.ptr_r is not None and b.ptr_r.numel():
            loss = loss + ptr_ce(cur['ptr_r'], b.ptr_r)
        if b.beta_tgt is not None:
            loss = loss + beta_bce(cur['beta'], b.beta_tgt)
        pre = getattr(self, '_prefix_aux', None) if b.prefix is not None else None
        if b.prefix is not None and pre is None:
            return loss
        if pre is not None and b.prefix.beta_tgt is not None:
            loss = loss + beta_bce(pre['beta'], b.prefix.beta_tgt)
        wa, wb = (pre, b.prefix) if pre is not None else (cur, b)
        if wb.ptr_w is not None and wb.ptr_w.numel():
            loss = loss + ptr_ce(wa['ptr_w'], wb.ptr_w)
        if self.cfg.cont_keys:
            return loss + self._cont_key_loss(cur, wa, wb, b)
        if wb.ptr_w is not None and wb.ptr_w.numel():
            K = wa['kw'][wb.ptr_w[:, 0], wb.ptr_w[:, 1]].clamp(min=1e-09)
            pm = K.mean(0)
            mi = -(pm * pm.log()).sum() + (K * K.log()).sum(-1).mean()
            loss = loss - mi / math.log(K.shape[-1])
        if wb.addr_same is not None and wb.addr_same.numel():
            s = wb.addr_same
            same_w = (wa['kw'][s[:, 0], s[:, 1]] * wa['kw'][s[:, 0], s[:, 2]]).sum(-1)
            loss = loss - torch.log(same_w + 1e-06).mean()
        if b.addr_pair is not None and b.addr_pair.numel():
            p = b.addr_pair
            same = (cur['kr'][p[:, 0], p[:, 1]] * wa['kw'][p[:, 0], p[:, 2]]).sum(-1)
            loss = loss - torch.log(same + 1e-06).mean()
        if wb.addr_neg is not None and wb.addr_neg.numel():
            q = wb.addr_neg
            diff = (wa['kw'][q[:, 0], q[:, 1]] * wa['kw'][q[:, 0], q[:, 2]]).sum(-1)
            loss = loss - torch.log(1 - diff + 1e-06).mean()
            if b.addr_pair is not None and b.addr_pair.numel():
                c = _cross_pairs(b.addr_pair, q)
                if c.numel():
                    hit = (cur['kr'][c[:, 0], c[:, 1]] * wa['kw'][c[:, 0], c[:, 2]]).sum(-1)
                    loss = loss - torch.log(1 - hit + 1e-06).mean()
        return loss

    def _cont_key_loss(self, cur, wa, wb, b: Batch):
        loss = torch.zeros((), device=self.device)
        if wb.addr_same is not None and wb.addr_same.numel():
            s = wb.addr_same
            loss = loss + (1 - (wa['kw'][s[:, 0], s[:, 1]] * wa['kw'][s[:, 0], s[:, 2]]).sum(-1)).mean()
        if b.addr_pair is not None and b.addr_pair.numel():
            p = b.addr_pair.tolist()
            neg = wb.addr_neg.tolist() if wb.addr_neg is not None else []
            last = sorted({(x, v) for x, _, v in p} | {(x, y) for x, y1, y2 in neg for y in (y1, y2)})
            at = {k: i for i, k in enumerate(last)}
            idx = torch.tensor(last, device=self.device)
            sim = cur['kr'][b.addr_pair[:, 0], b.addr_pair[:, 1]] @ wa['kw'][idx[:, 0], idx[:, 1]].T
            loss = loss + F.cross_entropy(sim / 0.1, torch.tensor([at[x, v] for x, _, v in p], device=self.device))
        if wb.addr_neg is not None and wb.addr_neg.numel():
            q = wb.addr_neg
            loss = loss + ((wa['kw'][q[:, 0], q[:, 1]] * wa['kw'][q[:, 0], q[:, 2]]).sum(-1) ** 2).mean()
        return loss

    @torch.no_grad()
    def exact_match(self, b: Batch, stops: bool=False):
        T = b.ids.shape[1]
        keep = sorted({t for _, s, e, _ in b.spans for t in range(s - 1, min(e - 1 + stops, T - 1))})
        keep_t = torch.tensor(keep, device=b.ids.device)
        logits = self.forward(b.ids, b.attn, None, state=self._prefix_state(b), keep=keep_t)
        col = {t: k for k, t in enumerate(keep)}
        use_ptr = self.ctx.graph is not None and self.ctx.cand_ids is not None
        if use_ptr and self.cfg.carry_commit:
            copy_inputs(self.block, self.ctx, b.ids)
        pred = {}
        for t in keep:
            k = col[t]
            pred[t] = (copy_mixture_probs(self.block, self.ctx_slice(t), b.ids[:, :t + 1], logits[:, k:k + 1]) if use_ptr else logits[:, k].float().softmax(-1)).argmax(-1)
        res = []
        for bi, s, e, kind in b.spans:
            ok = all((int(pred[t][bi]) == int(b.ids[bi, t + 1]) for t in range(s - 1, e - 1)))
            if stops:
                res.append((kind, ok, ok and e < T and ('\n' in self.tok.decode([int(pred[e - 1][bi])]))))
            else:
                res.append((kind, ok))
        return res

    def ctx_slice(self, t):
        c = MemContext()
        c.graph, c.cand_ids, c.cand_w, c.copy_x = (self.ctx.graph, self.ctx.cand_ids[:, t:t + 1], self.ctx.cand_w[:, t:t + 1], self.ctx.copy_x[:, t:t + 1])
        c.cand_hist_ids, c.cand_hist_w = (self.ctx.cand_hist_ids[:, :t + 1], self.ctx.cand_hist_w[:, :t + 1])
        c.lam_hist = self.ctx.lam_hist[:, :t] if self.ctx.lam_hist is not None else None
        return c

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int=24, stop: str='\n', persist: bool=False, use_memory: bool=True) -> str:
        was = self.ctx.enabled
        self.ctx.enabled = use_memory
        ids = torch.tensor([self.bos + self.tok(prompt, add_special_tokens=False).input_ids], device=self.device)
        self.ctx.clear()
        self.ctx.aux_idx = None
        self.ctx.write_mask = None
        self.ctx.aux_mention = None
        self.ctx.state = self.block.persistent_state.clone() if persist and self.block.pad is not None else None
        self.ctx.tok_ids = ids
        out = self.lm_step(keep=1, input_ids=ids, use_cache=True)
        gen = []
        for _ in range(max_new_tokens):
            p = copy_mixture_probs(self.block, self.ctx, ids, out.logits) if use_memory else out.logits[:, -1].softmax(-1)
            nxt = p.argmax(-1, keepdim=True)
            gen.append(nxt.item())
            ids = torch.cat([ids, nxt], 1)
            if stop and stop in self.tok.decode(gen):
                break
            self.ctx.tok_ids = ids
            out = self.lm_step(keep=1, input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
        if persist and self.block.pad is not None and (self.ctx.state is not None):
            self.block.persistent_state.copy_(self.ctx.state[:1])
        self.ctx.enabled = was
        return self.tok.decode(gen).split(stop)[0] if stop else self.tok.decode(gen)

    @torch.no_grad()
    def remember(self, text: str):
        ids = torch.tensor([self.bos + self.tok(text, add_special_tokens=False).input_ids], device=self.device)
        state = self.write_session(ids, None, state=self.block.persistent_state.clone())
        self.ctx.write_mask = None
        self.block.persistent_state.copy_(state[:1])

    def reset_memory(self):
        self.block.persistent_state.zero_()

    def save(self, path: str, meta: dict | None=None):
        torch.save({'cfg': self.cfg.__dict__, 'layer_idx': self.layer_idx, 'block': self.block.state_dict(), 'meta': meta}, path)

    def load(self, path: str, strict: bool=True):
        sd = torch.load(path, map_location=self.device)
        assert sd['layer_idx'] == self.layer_idx, 'adapter was trained at a different layer'
        own = self.block.state_dict()
        sd = {k: v for k, v in sd['block'].items() if strict or (k in own and own[k].shape == v.shape)}
        for k in ('in_shift', 'in_scale'):
            sd.setdefault(k, own[k])
        return self.block.load_state_dict(sd, strict=strict)

    @torch.no_grad()
    def read(self, h, ids, pos: int | None=None, k: int=3, window: int | None=1024) -> dict:
        T = ids.shape[1]
        pos = T - 1 if pos is None else pos % T
        start = max(0, pos + 1 - window) if window else 0
        h, ids, pos = (h[:, start:pos + 1], ids[:, start:pos + 1], pos - start)
        was = self.block.training
        self.block.eval()
        self.ctx.clear()
        self.ctx.state = None
        self.ctx.aux_mention = None
        self.ctx.aux_idx = torch.tensor([pos], device=self.device)
        self.ctx.write_mask = torch.ones_like(ids, device=self.device)
        self.ctx.tok_ids = ids.to(self.device)
        try:
            self.block(h.to(self.device), self.ctx)
        finally:
            self.block.train(was)
        aux, out = (self.ctx.kg_aux, {'hops': []})
        for r in aux['rel']:
            p, i = r[0].float().softmax(-1).topk(min(k, r.shape[-1]))
            out['hops'].append(dict(rel=i.tolist(), p=p.tolist()))
        out['halt'] = aux['halt'][0].float().softmax(-1).tolist()
        cid, cw = (self.ctx.cand_ids[0, pos], self.ctx.cand_w[0, pos].float())
        dense = torch.zeros(int(cid.max()) + 1, device=cw.device).scatter_add_(0, cid, cw)
        w, e = dense.topk(min(k, dense.numel()))
        out['candidates'] = [(int(a), float(b)) for a, b in zip(e, w) if b > 0]
        return out

    @torch.no_grad()
    def calibrate(self, texts: list[str], skip: int=1):
        hs = []
        for t in texts:
            ids = torch.tensor([self.bos + self.tok(t, add_special_tokens=False).input_ids], device=self.device)
            hs.append(self.hidden_at_layer(ids, torch.ones_like(ids))[0, skip:].float())
        h = torch.cat(hs)
        self.block.in_shift.copy_(h.mean(0))
        self.block.in_scale.copy_(h.std(0).clamp(min=0.001))
