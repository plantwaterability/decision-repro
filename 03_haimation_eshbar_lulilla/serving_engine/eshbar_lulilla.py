from eshbar.engine import Engine as _PinnedEngine

TAU_NUM, TAU_TRUTH = 0.925, 0.75


class Engine(_PinnedEngine):
    route_tau_pinned = (TAU_NUM, TAU_TRUTH)
