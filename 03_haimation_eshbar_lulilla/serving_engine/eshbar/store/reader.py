import json
import time
import torch
from .table import Tab, pick, text, top
from .lm import MemConfig, MemoryLM
from .lm.memlm import MemLayer
_S = {}

def _block(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        meta = json.load(open(path[:-len('.safetensors')] + '.json'))
        return (dict(block=load_file(path), cfg=meta['cfg'], layer_idx=meta['layer_idx']), meta)
    return (torch.load(path, map_location='cpu', weights_only=True), {})

def load(blk_path, tab_path, soft=0.1, k=5, graphs=None):
    sd, meta = _block(blk_path)
    _S.update(sd=sd, tab=Tab(tab_path), soft=soft, k=k, m=None, model=None, g=None, graphs=bool(meta.get('graphs', False)) if graphs is None else graphs)

class _Graphs:

    def __init__(self, m):
        self.m, self.g, self.pool, self.lazy = (m, {}, torch.cuda.graph_pool_handle(), True)
        lm = m.lm.model
        self.t = getattr(lm, 'language_model', None) or lm

    def _run(self, ids, pos):
        from collections import UserDict
        t = self.t
        h = t.embed_tokens(ids)
        pe = {lt: t.rotary_emb(h, pos, lt) for lt in t.unique_layer_types}
        kv = UserDict()
        for i in range(self.m.layer_idx + 1):
            layer = t.layers[i]
            layer = layer.layer if isinstance(layer, MemLayer) else layer
            out = layer(h, None, shared_kv_states=kv, position_embeddings=pe[t.config.layer_types[i]], attention_mask=None, position_ids=pos, past_key_values=None)
            h = out[0] if isinstance(out, tuple) else out
        return h.clone()

    def capture(self, n, device):
        buf, pos = (torch.zeros((1, n), dtype=torch.long, device=device), torch.arange(n, device=device)[None])
        for _ in range(2):
            self._run(buf, pos)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._run(buf, pos)
        self.g[n] = (graph, buf, out, pos)

    def __call__(self, ids):
        n = ids.shape[1]
        if n not in self.g:
            if not self.lazy:
                return None
            self.capture(n, ids.device)
        graph, buf, out, _ = self.g[n]
        buf.copy_(ids)
        graph.replay()
        return out

def attach(model, tok):
    if _S.get('model') is model:
        return _S['m']
    sd = _S['sd']
    m = MemoryLM(model, tok, MemConfig(**{**sd['cfg'], 'rel_exists_soft': _S['soft']}), layer_idx=sd['layer_idx'])
    m.block.load_state_dict(sd['block'])
    m.block.eval()
    _S['tab'].put(m)
    m.ctx.enabled = False
    _S.update(m=m, model=model, g=_Graphs(m) if _S['graphs'] and torch.cuda.is_available() else None)
    return m

@torch.no_grad()
def warm(model, tok, lengths=range(8, 97)):
    m = attach(model, tok)
    g = _S['g']
    if g is None:
        return {'graphs': 0, 's': 0.0}
    torch.cuda.synchronize()
    t = time.time()
    for n in lengths:
        if n not in g.g:
            g.capture(n, m.device)
    torch.cuda.synchronize()
    g.lazy = False
    return {'graphs': len(g.g), 's': round(time.time() - t, 2)}

def _hidden(m, ids):
    if _S['g'] is not None:
        try:
            h = _S['g'](ids)
            if h is not None:
                return h
        except Exception:
            _S['g'] = None
    return m.hidden_at_layer(ids, torch.ones_like(ids))

@torch.no_grad()
def read(model, tok, question, options=None):
    m = attach(model, tok)
    ids = torch.tensor([m.bos + tok(text(question), add_special_tokens=False).input_ids], device=m.device)
    m.read(_hidden(m, ids), ids, k=_S['k'])
    vals, ab = top(m, _S['tab'], _S['k'])
    return {'choice': pick(vals, options) if options and vals else None, 'abstain': ab, 'values': vals}
