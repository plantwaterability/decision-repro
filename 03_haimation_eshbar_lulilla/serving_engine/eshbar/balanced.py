from __future__ import annotations
from . import gate_long as glong
from . import orders as oe
_ORIG_RANDOM_ORDERS = oe.random_orders

def balanced_orders(K, k=oe.K_AVG):
    if K <= 4:
        return _ORIG_RANDOM_ORDERS(K, k)
    steps = [int(round(j * K / k)) % K for j in range(k)]
    used = []
    for s in steps:
        while s in used:
            s = (s + 1) % K
        used.append(s)
    return [[(i + s) % K for i in range(K)] for s in used]
oe.random_orders = balanced_orders

class BalancedEngine(glong.GatedLongEngine):

    def __init__(self, model='', l_star=6000, l_star_noul=6000, **options):
        super().__init__(model=model, l_star=l_star, l_star_noul=l_star_noul, **options)
