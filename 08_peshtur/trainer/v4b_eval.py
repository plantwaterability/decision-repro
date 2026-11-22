"""Shard runner for V4BEngine (matches wx_eval_d3s.py CLI).
usage: python v4b_eval.py <label> <ckpt_dir|none> <perm_ignored> <rows.jsonl> <image_root> <out_dir> <model_dir> <ocr_cache>
"""
import json
import sys
import time

label, ckpt, _perm, rows_f, root, out_dir, model_dir, ocr = sys.argv[1:9]
import decision_index.vision.run as R  # noqa: E402
from v4b_engine import V4BEngine  # noqa: E402

eng = V4BEngine(
    model_dir,
    ckpt=None if ckpt in ("none", "", "-") else ckpt,
    device="cuda:0",
    image_root=root,
    label=label,
    ocr=None if ocr in ("none", "", "-") else ocr,
)
R.load_engine = lambda name, **kw: eng
t = time.time()
final = R.run(
    f"v4b:{label}",
    {"label": label, "ckpt": ckpt, "model_dir": model_dir, "ocr": ocr, "layout": "d3"},
    root,
    out_dir,
    rows_path=rows_f,
    workers=1,
    chunk=50,
    compact=False,
)
print(
    json.dumps(
        {
            "final": final,
            "wall_s": round(time.time() - t, 1),
            "attn_calls": eng.calls["n"],
            "lora_merged": eng.provenance["lora_modules_merged"],
            "merger_loaded": eng.provenance["merger_loaded"],
        }
    )
)
