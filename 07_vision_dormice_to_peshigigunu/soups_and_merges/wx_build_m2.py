"""WX (student wxs1 prep): M2 = WeMM-Decision + 0.5 x peshashaga's update in the stock Qwen3.8 packaging (VX's trainer loads it like stock):
wemm_sf (WeMM's 1,183 tensors bit for bit, lm_head = stock) -> wx_engine.build_weights(model.model, a = 1, b = 0.5) on the GPU in venv_w (the exact
arithmetic of the M2 reads: 496 matrices, fp32 sum of the 8 terms in file order, one bf16 rounding) -> save_pretrained + processor / tokenizer
files + WeMM's readout.safetensors / decision_config.json (the d3 head init) + WEMM_LICENSE / WEMM_NOTICE + MANIFEST.json (sha256 per file).
usage: PYTHONPATH=<kit>:<code> python wx_build_m2.py <wemm_sf dir> <peshashaga dir> <wemm dir> <out dir> <stock dir (index only)>"""
import hashlib, json, os, shutil, sys
import torch
import wx_engine as WXE
from transformers import AutoModelForImageTextToText

src, pesh, wemm, out, stock = sys.argv[1:6]
m = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map={"": "cuda:0"})
stats = WXE.build_weights(m.model, stock, pesh, 1.0, 0.5, "cuda:0")  # a = 1: only the stock index is opened, no stock tensor is read
assert stats["a_tensors"] == 0 and stats["b_matrices"] == 496, stats
os.makedirs(out, exist_ok=True)
m.save_pretrained(out, max_shard_size="5GB")
for f in os.listdir(src):
    if (f.endswith((".json", ".jinja", ".txt")) and "safetensors" not in f and f not in ("config.json", "MANIFEST.json")) or f.startswith("WEMM_"):
        if not os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(src, f), os.path.join(out, f))
for f in ("readout.safetensors", "decision_config.json"):
    shutil.copy(os.path.join(wemm, f), os.path.join(out, f))
man = {"what": "M2 = Rover00/WeMM-Decision @1543fbfc + 0.5 x plantwaterer/peshashaga @61320dc8 peshashaga_update.safetensors (sha256 3eed0e55...), "
               "built by wx_engine.build_weights (a 1, b 0.5) in venv_w on H200; stock Qwen3.8-27B @1d4bf0f2 packaging, lm_head = stock",
       "licence": "Apache-2.0 (WeMM-Decision LICENSE + NOTICE as WEMM_LICENSE / WEMM_NOTICE; modified: merged with 0.5 x peshashaga's update)",
       "build": stats, "torch": torch.__version__, "files": []}
for f in sorted(os.listdir(out)):
    if f == "MANIFEST.json":
        continue
    h = hashlib.sha256()
    with open(os.path.join(out, f), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    man["files"].append({"name": f, "sha256": h.hexdigest()})
json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in man.items() if k != "files"} | {"files": len(man["files"])}), flush=True)
