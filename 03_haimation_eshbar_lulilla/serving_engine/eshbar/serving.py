from __future__ import annotations
import torch
from . import heads as _heads
from .readout import ReadoutEngine
BINARY_SCHEMA = 'binary-laplace-head-v1'

class CpuHead:

    def __init__(self, head):
        self.inner = head.cpu()
        self.heads = getattr(head, 'heads', [])
        self.needs_kinds = bool(getattr(head, 'needs_kinds', False))
        self.needs_yes_no = bool(getattr(head, 'needs_yes_no', False))

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def predict(self, h, ell, ks, *args, **kw):
        kw = {k: v.cpu() if torch.is_tensor(v) else v for k, v in kw.items()}
        return self.inner.predict(h.cpu(), ell.cpu(), ks.cpu(), *args, **kw)

class ServingEngine(ReadoutEngine):

    def __init__(self, model='', device='cuda', require_schema=BINARY_SCHEMA, head_device='auto', **options):
        super().__init__(model=model, device=device, **options)
        meta = self.head_meta
        if require_schema and meta.get('schema') != require_schema:
            raise SystemExit(f"{model}: the serving head is {meta.get('schema')!r} not {require_schema!r}")
        hd = head_device
        if hd == 'auto':
            hd = 'cpu' if isinstance(self.head, _heads.BinaryHead) else str(device)
        if hd == 'cpu' and (not isinstance(self.head, CpuHead)):
            self.head = CpuHead(self.head)
        self.head_device = hd
