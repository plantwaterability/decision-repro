"""Offline (CPU) check: the release engine's prompt text == WX's wx_engine d3 prompt text (WX_D3_STATE=ocr) on every row, with the OCR
state from the cached OCR file the reference read used; and the release engine's OCR block == qv_ocr.ocr_prefix. No model is loaded.

  m2_prompt_eq.py REPO_DIR NAME WX_CODE_DIR ROWS_JSONL OCR_CACHE_JSONL PROCESSOR_DIR OUT_JSON"""
import hashlib
import importlib
import json
import sys

repo, name, wx_code, rows_f, cache_f, proc_dir, out_f = sys.argv[1:8]
sys.path.insert(0, repo)
mod = importlib.import_module(name)
sys.path.insert(1, wx_code)
assert hashlib.md5(open(f"{wx_code}/wx_engine.py", "rb").read()).hexdigest() == "165f1ad31cee86227233ecee99514562"
assert hashlib.md5(open(f"{wx_code}/qv_ocr.py", "rb").read()).hexdigest() == "246b6430d8a1cb8e4ab47614c305ab8f"
import qv_ocr  # noqa: E402
import wx_engine as W  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

proc = AutoProcessor.from_pretrained(proc_dir)
cfg = json.load(open(f"{repo}/{name}.json"))
codes = cfg["readout"]["codes"]
assert codes == json.load(open(f"{repo}/{name}.json"))["readout"]["codes"]
wx_text, _ = qv_ocr.load_ocr(cache_f)
mine = {}
for l in open(cache_f):
    r = json.loads(l)
    mine[r["image"]] = mod.ocr_text(r["lines"])
n = same_text = same_block = with_ocr = 0
bad = []
for l in open(rows_f):
    if not l.strip():
        continue
    r = json.loads(l)
    ims = r["images"]
    (key, q), = r["questions"].items()
    k = len(q["criteria"])
    wb = qv_ocr.ocr_prefix(wx_text, ims)
    mb = mod.ocr_block([mine[p] for p in ims])
    same_block += wb == mb
    with_ocr += bool(wb)
    st = wb if wb else r["state"]
    content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": W.d3_user(st, q, codes[:k])}]
    t_wx = proc.apply_chat_template([{"role": "system", "content": W.D3_SYSTEM}, {"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    t_me = mod.render(proc, (mb if mb else r["state"]), q, codes[:k], len(ims))
    same_text += t_wx == t_me
    if t_wx != t_me and len(bad) < 5:
        bad.append({"id": r["id"], "wx": t_wx[-300:], "release": t_me[-300:]})
    n += 1
res = {"rows": n, "same_prompt_text": same_text, "same_ocr_block": same_block, "rows_with_ocr_state": with_ocr, "examples_differing": bad,
       "system_equal": W.D3_SYSTEM == mod.SYSTEM, "PROMPT_EQ": n > 0 and same_text == n and same_block == n and W.D3_SYSTEM == mod.SYSTEM}
json.dump(res, open(out_f, "w"), indent=1)
print("PROMPT_EQ " + json.dumps({k: v for k, v in res.items() if k != "examples_differing"}))
sys.exit(0 if res["PROMPT_EQ"] else 1)
