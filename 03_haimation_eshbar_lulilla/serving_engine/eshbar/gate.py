from __future__ import annotations
from . import orders as oe
L_STAR = 939

def request_length(state, questions):
    st = state or ''
    tot = 0
    for q in (questions or {}).values():
        tot += len(st) + len(q.get('instructions') or '') + sum((len(str(k)) + len(str(v)) for k, v in (q.get('criteria') or {}).items()))
    return tot

class GatedEngine(oe.OrderAvgEngine):

    def __init__(self, model='', l_star=L_STAR, **options):
        super().__init__(model=model, **options)
        self._avg_allowed = self.order_avg
        self.l_star = int(l_star)
        self.last_choice = None

    def __call__(self, state, questions):
        L = request_length(state, questions)
        self.order_avg = self._avg_allowed and L <= self.l_star
        self.last_choice = {'L': L, 'averaged': self.order_avg}
        try:
            return super().__call__(state, questions)
        finally:
            self.order_avg = self._avg_allowed
