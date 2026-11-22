"""Run one zero-training variant over the gate set through the kit runner.
usage: python soup_eval.py <label> <ckpts comma-separated> <perm 1|2> <gate_rows> <image_root> <out_dir> <model_dir>"""
import json
import sys
import time

label, ckpts, perm, rows_f, root, out_dir, model_dir = sys.argv[1:8]
import decision_index.vision.run as R  # noqa: E402
from qv_soup_engine import QVSoupEngine  # noqa: E402

eng = QVSoupEngine(model=model_dir, ckpts=ckpts, device="cuda:0", perm=int(perm), image_root=root, label=label)
R.load_engine = lambda name, **kw: eng
t = time.time()
final = R.run(f"soup:{label}", {"label": label, "ckpts": ckpts, "perm": perm}, root, out_dir, rows_path=rows_f, workers=1, chunk=50)
print(json.dumps({"final": final, "wall_s": round(time.time() - t, 1)}))
