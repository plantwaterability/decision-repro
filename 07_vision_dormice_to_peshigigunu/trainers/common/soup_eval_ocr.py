"""soup_eval.py with the OCR block in the state (qv_ocr format: before the question, only on rows where OCR found text).
usage: python soup_eval_ocr.py <label> <ckpts comma-separated> <perm 1|2> <rows> <image_root> <out_dir> <model_dir> <ocr_cache.jsonl>
raw_output carries ocr_ms (the cached per-image OCR wall time summed over the row's images) for latency accounting."""
import json
import sys
import time

label, ckpts, perm, rows_f, root, out_dir, model_dir, cache_f = sys.argv[1:9]
import decision_index.vision.run as R  # noqa: E402
from decision_index.engines.base import Engine  # noqa: E402
import qv_ocr as O  # noqa: E402
from qv_soup_engine import QVSoupEngine  # noqa: E402

TEXT, MS = O.load_ocr(cache_f)


class OCRWrap(Engine):
    """Wraps the soup engine: the OCR block goes before the question; OCR time is reported in raw_output."""

    def __init__(self, inner, **options):
        super().__init__(**options)
        self.inner = inner
        self.provenance = {**inner.provenance, "ocr": "PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec, block before the question on rows with text"}
        self.latency = inner.latency

    def runtime(self):
        return self.inner.runtime()

    def warmup(self):
        pass

    def __call__(self, state, questions, images=(), prompt=None):
        ims = list(images or [])
        key = next(iter(questions))
        q = dict(questions[key])
        q["instructions"] = O.ocr_prefix(TEXT, ims) + q["instructions"]
        resp, raw = self.inner(state, {key: q}, images=images, prompt=prompt)
        raw = dict(raw)
        raw["ocr_ms"] = sum(MS[p] for p in ims)
        return resp, raw


eng = OCRWrap(QVSoupEngine(model=model_dir, ckpts=ckpts, device="cuda:0", perm=int(perm), image_root=root, label=label))
R.load_engine = lambda name, **kw: eng
t = time.time()
final = R.run(f"soup_ocr:{label}", {"label": label, "ckpts": ckpts, "perm": perm, "ocr": cache_f}, root, out_dir, rows_path=rows_f, workers=1, chunk=50)
print(json.dumps({"final": final, "wall_s": round(time.time() - t, 1)}))
