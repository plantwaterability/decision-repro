from __future__ import annotations
from . import render as gc
from . import binary as lbe
from . import gate as le

class GatedLongEngine(lbe.BinaryEngine):

    def __init__(self, model='', l_star=6000, l_star_noul=6000, **options):
        super().__init__(model=model, l_star=l_star, **options)
        self.l_star_noul = int(l_star_noul)

    def __call__(self, state, questions):
        L_raw = le.request_length(state, questions)
        st_txt = gc.render_state(state, 'text')
        L_ren = le.request_length(st_txt, questions)
        all_noul = bool(questions) and all((q.get('type') == 'noul' for q in (questions or {}).values()))
        cutoff = self.l_star_noul if all_noul else self.l_star
        self.order_avg = self._avg_allowed and (L_raw <= 939 or (len(questions or {}) == 1 and L_ren <= cutoff))
        self.last_choice = {'L_raw': L_raw, 'L_ren': L_ren, 'all_noul': all_noul, 'cutoff': cutoff, 'averaged': self.order_avg}
        try:
            return super(le.GatedEngine, self).__call__(state, questions)
        finally:
            self.order_avg = self._avg_allowed
