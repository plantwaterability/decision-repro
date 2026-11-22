"""WX step 4 (draft input, no training): WeMM-Decision @1543fbfc in the stock Qwen3.8-27B packaging, so VX's trainer / soup engine load it as they
load stock (AutoModelForImageTextToText -> Qwen3_5ForConditionalGeneration): stock model, every model.language_model.* / model.visual.* tensor
replaced by WeMM's (1,183 tensors, bit for bit), lm_head = stock (WeMM ships none; its readout rows equal stock's letter rows to ~1 bf16 ulp, and
uoq1's 26-way head starts from the stock letter rows), stock processor / tokenizer / template files (byte-identical to WeMM's), WeMM's LICENSE +
NOTICE kept. Check: every saved tensor read back equals WeMM's (or stock's lm_head) bit for bit. MANIFEST.json = sha256 per file. CPU only.
usage: CUDA_VISIBLE_DEVICES= python wx_build_init.py <stock_dir> <wemm_dir> <out_dir>"""
import hashlib, json, os, shutil, sys
import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

stock, wemm, out = sys.argv[1:4]
m = AutoModelForImageTextToText.from_pretrained(stock, dtype=torch.bfloat16, device_map={"": "cpu"})
params = dict(m.named_parameters())
widx = json.load(open(os.path.join(wemm, "model.safetensors.index.json")))["weight_map"]
H = {}


def get(d, idx, k):
    f = idx[k]
    if (d, f) not in H:
        H[(d, f)] = safe_open(os.path.join(d, f), "pt", device="cpu")
    return H[(d, f)].get_tensor(k)


n = 0
with torch.no_grad():
    for k in widx:
        t = get(wemm, widx, k)
        p = params["model." + k]
        assert p.shape == t.shape and p.dtype == t.dtype, (k, p.shape, t.shape, p.dtype, t.dtype)
        p.copy_(t); n += 1
assert n == len(widx) == 1183, n
os.makedirs(out, exist_ok=True)
m.save_pretrained(out, max_shard_size="5GB")
for f in os.listdir(stock):
    if f.endswith((".json", ".jinja", ".txt")) and "safetensors" not in f and f != "config.json" and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(stock, f), os.path.join(out, f))
for f in ("LICENSE", "NOTICE"):
    shutil.copy(os.path.join(wemm, f), os.path.join(out, f"WEMM_{f}"))
del m
# read back: every WeMM tensor bit-identical; lm_head = stock's
oidx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
sidx = json.load(open(os.path.join(stock, "model.safetensors.index.json")))["weight_map"]
bad = [k for k in widx if not torch.equal(get(out, oidx, "model." + k), get(wemm, widx, k))]
lm_ok = torch.equal(get(out, oidx, "lm_head.weight"), get(stock, sidx, "lm_head.weight"))
extra = sorted(set(oidx) - {"model." + k for k in widx} - {"lm_head.weight"})
man = {"what": "Rover00/WeMM-Decision @1543fbfcb171c7990607369e4c17329d3a7a6385 weights in the stock Qwen/Qwen3.8-27B @1d4bf0f2 packaging; lm_head = stock",
       "licence": "Apache-2.0 (WeMM LICENSE + NOTICE kept as WEMM_LICENSE / WEMM_NOTICE)", "tensors_from_wemm": n, "readback_bad": bad[:5],
       "lm_head_equals_stock": lm_ok, "tensors_not_from_wemm": extra, "files": []}
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
sys.exit(0 if (not bad and lm_ok and not extra) else 1)
