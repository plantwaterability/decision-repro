from __future__ import annotations
import json
import math
import os
import numpy as np
from . import gate as le
from . import orders as oe
from . import heads as _heads

def _logodds(p0, p1):
    return math.log(max(p0, 1e-300)) - math.log(max(p1, 1e-300))

class BinaryEngine(le.GatedEngine):

    def __init__(self, model='', **options):
        super().__init__(model=model, **options)
        self._act_path = os.environ.get('ESHBAR_BINARY_LOG') or None

    def _head_delta(self, h_row, ell_letters):
        import torch
        inner = getattr(self.head, 'inner', self.head)
        st = inner.state()
        T = float(self.head_meta.get('temperature', 1.0))
        ks = torch.tensor([2])
        kinds = torch.tensor([_heads.KIND_CODE['noul']])
        d = _heads.binary_extract(h_row.cpu(), ell_letters.cpu(), ks, kinds, st)
        p_on, mi = _heads.binary_probs(d, st, T, inner.c)
        p_off, _ = _heads.binary_probs(d, st, T, dict(inner.c, beta=0.0, infl=0.0))
        p_on, p_off = (p_on[0].double().tolist(), p_off[0].double().tolist())
        return (_logodds(p_on[0], p_on[1]) - _logodds(p_off[0], p_off[1]), float(mi[0]), p_on, p_off)

    def _log(self, rec):
        if self._act_path:
            with open(self._act_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

    def run_prompts(self, items):
        import torch
        if not self.order_avg:
            out = oe.OrderAvgEngine.run_prompts(self, items)
            if self._act_path:
                for qid, order, _ in items:
                    lr = self.last_raw.get(qid, {})
                    nat = lr.get('native')
                    self._log({'qid': qid, 'kind': self._kind.get(qid, 'choice'), 'route': 'unaveraged', 'abs_head_minus_native': float(np.max(np.abs(np.asarray(out[qid]) - np.asarray(nat)))) if nat is not None else None, 'mi': lr.get('epistemic')})
            return out
        expanded = []
        for qid, order, text in items:
            expanded.append((qid, order, text))
            for j, (o, t) in enumerate(self._alts.get(qid, [])):
                qj = f'{qid}#oa{j + 1}'
                self._kind[qj] = self._kind.get(qid, 'choice')
                expanded.append((qj, o, t))
        cap, pos = ({}, [0])
        fwd0 = self._forward

        def fwd(texts):
            h, ell = fwd0(texts)
            for b in range(len(texts)):
                cap[expanded[pos[0] + b][0]] = (h[b:b + 1].detach(), ell[b:b + 1].detach())
            pos[0] += len(texts)
            return (h, ell)
        self._forward = fwd
        try:
            raw = oe.ServingEngine.run_prompts(self, expanded)
        finally:
            self._forward = fwd0
        if pos[0] != len(expanded):
            raise RuntimeError(f'captured {pos[0]} prompts of {len(expanded)}')
        out = {}
        for qid, order, _ in items:
            alts = self._alts.get(qid, [])
            if not alts:
                out[qid] = raw[qid]
                self._log({'qid': qid, 'kind': self._kind.get(qid, 'choice'), 'route': 'no_alternatives'})
                continue
            names = [qid] + [f'{qid}#oa{j + 1}' for j in range(len(alts))]
            if self._kind.get(qid) == 'noul':
                nats = [self.last_raw[q]['native'] for q in names]
                p_len = oe.combine(nats, oe.T_NOUL)
                avg = np.asarray(nats, dtype=np.float64).mean(0)
                h_b, ell_b = cap[qid]
                ell_letters = ell_b[:, :2].clone()
                ell_letters[0, 0] = math.log(max(avg[order[0]], 1e-300))
                ell_letters[0, 1] = math.log(max(avg[order[1]], 1e-300))
                delta, mi, p_on, p_off = self._head_delta(h_b, ell_letters)
                lo = _logodds(p_len[order[0]], p_len[order[1]]) + delta
                pa = 1.0 / (1.0 + math.exp(-lo)) if lo > -700 else 0.0
                p = np.zeros(2)
                p[order[0]], p[order[1]] = (pa, 1.0 - pa)
                out[qid] = p.tolist()
                self.last_raw[qid]['orders'] = {'orders': len(names), 'readout': 'native+head'}
                self.last_raw[qid]['binary'] = {'delta': delta, 'mi': mi, 'p_len': p_len.tolist(), 'p_v4': p.tolist(), 'p_on': p_on, 'p_off': p_off, 'head_per_order': [raw[q] for q in names]}
                self._log({'qid': qid, 'kind': 'noul', 'route': 'averaged', 'abs_delta': abs(delta), 'abs_p_v4_minus_p_len': float(np.max(np.abs(p - p_len))), 'mi': mi})
            else:
                p = oe.combine([raw[q] for q in names], oe.T_HEAD)
                out[qid] = p.tolist()
                self.last_raw[qid]['orders'] = {'orders': len(names), 'readout': 'head'}
                self._log({'qid': qid, 'kind': self._kind.get(qid, 'choice'), 'route': 'averaged', 'abs_head_minus_native': float(np.max(np.abs(np.asarray(raw[qid]) - np.asarray(self.last_raw[qid]['native'])))), 'mi': self.last_raw[qid].get('epistemic')})
            for q in names[1:]:
                self.last_raw.pop(q, None)
                self._kind.pop(q, None)
        return out
