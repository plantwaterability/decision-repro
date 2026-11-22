"""VL copy of soup_eval_ocr.py (37a674ff): env QV_LAYOUT=images_first (default; the stock path, the block prefixed to the question) or
sandwich (the bare question goes to the engine, the OCR block travels in the prompt dict and is placed after the images).

soup_eval.py with the OCR block in the state (qv_ocr format: before the question, only on rows where OCR found text).
usage: python soup_eval_ocr.py <label> <ckpts comma-separated> <perm 1|2> <rows> <image_root> <out_dir> <model_dir> <ocr_cache.jsonl>
raw_output carries ocr_ms (the cached per-image OCR wall time summed over the row's images) for latency accounting."""
import json
import sys
import time

import os
from pathlib import Path

label, ckpts, perm, rows_f, root, out_dir, model_dir, cache_f = sys.argv[1:9]
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "code"))  # VL: the stock qv_ocr / qv_common live in code/
LAYOUT = os.environ.get("QV_LAYOUT", "images_first")
import decision_index.vision.run as R  # noqa: E402
from decision_index.engines.base import Engine  # noqa: E402
import qv_ocr as O  # noqa: E402
from qv_soup_engine_vl import QVSoupEngine  # noqa: E402

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
        block = O.ocr_prefix(TEXT, ims)
        if LAYOUT == "images_first":
            q["instructions"] = block + q["instructions"]  # the stock path
        else:
            prompt = {**(prompt or {}), "ocr_block": block}  # sandwich: the question in front of the images stays bare
        resp, raw = self.inner(state, {key: q}, images=images, prompt=prompt)
        raw = dict(raw)
        raw["ocr_chars"] = len(block)
        raw["ocr_ms"] = sum(MS[p] for p in ims)
        return resp, raw


eng = OCRWrap(QVSoupEngine(model=model_dir, ckpts=ckpts, device="cuda:0", perm=int(perm), image_root=root, label=label, layout=LAYOUT))
R.load_engine = lambda name, **kw: eng
t = time.time()
final = R.run(f"soup_ocr:{label}", {"label": label, "ckpts": ckpts, "perm": perm, "ocr": cache_f, "layout": LAYOUT}, root, out_dir, rows_path=rows_f, workers=1, chunk=50)
print(json.dumps({"final": final, "wall_s": round(time.time() - t, 1), "layout": LAYOUT, "merger": eng.inner.merger_file, "layout_mismatch": eng.inner.layout_mismatch}))
