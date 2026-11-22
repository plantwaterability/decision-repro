from __future__ import annotations
import numpy as np
from . import render as core
from .serving import ServingEngine
SEED = 20261021
T_HEAD = 0.95438
T_NOUL = 0.93738
K_AVG = 4

def random_orders(K, k=K_AVG):
    return [list(range(K))] + [[int(x) for x in np.random.default_rng(SEED + 1000 * j + K).permutation(K)] for j in range(1, k)]

def relettered(base_order, r):
    return [base_order[j] for j in r]

def combine(prob_lists, T):
    P = np.asarray(prob_lists, dtype=np.float64).mean(0)
    Q = np.where(P > 0, np.power(np.maximum(P, 1e-300), 1.0 / T), 0.0)
    return Q / Q.sum()

class OrderAvgEngine(ServingEngine):

    def __init__(self, model='', order_avg='on', order_avg_k=K_AVG, **options):
        super().__init__(model=model, **options)
        self.order_avg = str(order_avg).lower() not in ('off', '0', 'false', 'no') and int(order_avg_k) > 1
        self.order_avg_k = int(order_avg_k)
        self._alts = {}

    def __call__(self, state, questions):
        self._alts = {}
        return super().__call__(state, questions)

    def prompt(self, u, qid):
        order, text = super().prompt(u, qid)
        if self.order_avg:
            K = len(order)
            if self._kind.get(qid) == 'noul':
                rs = [list(range(K)), list(range(K))[::-1]] if K == 2 else [list(range(K))]
            else:
                rs = random_orders(K, self.order_avg_k)
            alts = []
            for r in rs[1:]:
                o = relettered(order, r)
                t = core.build_prompt(self.tok, u, o, self.max_len, chat=self.chat, noul_defs=True)
                alts.append((o, t))
            self._alts[qid] = alts
        return (order, text)

    def run_prompts(self, items):
        if not self.order_avg:
            return super().run_prompts(items)
        expanded = []
        for qid, order, text in items:
            expanded.append((qid, order, text))
            for j, (o, t) in enumerate(self._alts.get(qid, [])):
                qj = f'{qid}#oa{j + 1}'
                self._kind[qj] = self._kind.get(qid, 'choice')
                expanded.append((qj, o, t))
        raw = super().run_prompts(expanded)
        out = {}
        for qid, order, _ in items:
            alts = self._alts.get(qid, [])
            if not alts:
                out[qid] = raw[qid]
                continue
            names = [qid] + [f'{qid}#oa{j + 1}' for j in range(len(alts))]
            if self._kind.get(qid) == 'noul':
                p = combine([self.last_raw[q]['native'] for q in names], T_NOUL)
            else:
                p = combine([raw[q] for q in names], T_HEAD)
            out[qid] = p.tolist()
            self.last_raw[qid]['orders'] = {'orders': len(names), 'readout': 'native' if self._kind.get(qid) == 'noul' else 'head'}
            for q in names[1:]:
                self.last_raw.pop(q, None)
                self._kind.pop(q, None)
        return out
