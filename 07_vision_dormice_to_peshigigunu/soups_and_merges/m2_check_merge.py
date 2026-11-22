"""m2_check_merge.py: merge parity. The release weights built by m2_build.py (from our S3 copies) against the weights the reference read
used, recomputed with the reference engine's own code from the public pinned files (for M2: WX's wx_engine.build_weights on
Rover00/WeMM-Decision @1543fbfc loaded with transformers' Qwen3_5Model, + b x peshashaga's update from plantwaterer/peshashaga @61320dc8).

  m2_check_merge.py SPEC_ENV RELEASE_DIR WX_CODE_DIR WEMM_HF_DIR STOCK_HF_DIR PESH_HF_DIR OUT_JSON [device]

Checks, all bit for bit (torch.equal):
  raw     every tensor of the reference-built model == the release checkpoint's tensor of the same name ("model." + name);
  loaded  the release directory loaded with transformers' Qwen3_5ForConditionalGeneration: every parameter of its .model == the reference's
          (covers any load-time conversion); lm_head == the stock checkpoint's lm_head; key sets equal (release = reference + lm_head);
  readout the release readout.safetensors has WeMM-Decision's sha256;
  config  the release config's text / vision settings vs WeMM-Decision's (differing keys listed; only bookkeeping keys allowed).
Exit 0 only if everything is identical."""
import hashlib
import json
import os
import sys
import time

import torch

spec_f, rel, wx_code, wemm, stock, pesh, out_f = sys.argv[1:8]
dev = sys.argv[8] if len(sys.argv) > 8 else "cuda:0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2lib import dump, read_env, sha256_file  # noqa: E402

t0 = time.time()
env = read_env(spec_f)
kind, a, b = env["REF_RECOMPUTE"].split()
assert kind == "wx", kind
a, b = float(a), float(b)
WX_MD5 = "165f1ad31cee86227233ecee99514562"
assert hashlib.md5(open(os.path.join(wx_code, "wx_engine.py"), "rb").read()).hexdigest() == WX_MD5, "wx_engine.py is not the handoff's file"
sys.path.insert(0, wx_code)
import wx_engine  # noqa: E402  (WX's engine module: imports the kit and transformers exactly as the M2 read did)
from safetensors import safe_open  # noqa: E402
from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Model  # noqa: E402

flags = {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
         "float32_matmul_precision": torch.get_float32_matmul_precision()}
ref = Qwen3_5Model.from_pretrained(wemm, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": dev})
logs = []
st = wx_engine.build_weights(ref, stock, pesh, a, b, dev, log=logs.append)
P = dict(ref.named_parameters())
ridx = json.load(open(os.path.join(rel, "model.safetensors.index.json")))["weight_map"]
H = {}


def rget(k):
    f = ridx[k]
    if f not in H:
        H[f] = safe_open(os.path.join(rel, f), "pt", device=dev)
    return H[f].get_tensor(k)


raw_same, raw_bad = 0, []
with torch.no_grad():
    for n, p in P.items():
        k = "model." + n
        if k not in ridx:
            raw_bad.append({"name": n, "why": "missing in release"}); continue
        r = rget(k)
        if r.dtype == p.dtype and r.shape == p.shape and torch.equal(r, p):
            raw_same += 1
        else:
            raw_bad.append({"name": n, "dtype": str(r.dtype), "max_abs": float((r.float() - p.float()).abs().max()) if r.shape == p.shape else None})
        del r
extra = sorted(set(ridx) - {"model." + n for n in P})
sidx = json.load(open(os.path.join(stock, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(stock, sidx["lm_head.weight"]), "pt", device=dev) as h:
    lm_stock = h.get_tensor("lm_head.weight")
lm_ok = torch.equal(rget("lm_head.weight"), lm_stock)
del lm_stock
H.clear(); torch.cuda.empty_cache()
# the release directory as the engine loads it
m = Qwen3_5ForConditionalGeneration.from_pretrained(rel, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": dev})
Q = dict(m.model.named_parameters())
loaded_same = sum(1 for n, p in P.items() if n in Q and Q[n].dtype == p.dtype and Q[n].shape == p.shape and torch.equal(Q[n], p))
loaded_missing = sorted(set(P) ^ set(Q))[:10]
with safe_open(os.path.join(stock, sidx["lm_head.weight"]), "pt", device=dev) as h:
    lm_loaded_ok = torch.equal(m.lm_head.weight, h.get_tensor("lm_head.weight"))


def cfgdiff(x, y):
    dx, dy = x.to_dict(), y.to_dict()
    return sorted(k for k in set(dx) | set(dy) if dx.get(k, "<absent>") != dy.get(k, "<absent>"))


BOOK = {"transformers_version", "architectures", "dtype", "torch_dtype", "_name_or_path", "model_type", "auto_map", "custom_pipelines", "rope_parameters", "output_attentions"}
cd_text = cfgdiff(ref.config.text_config, m.config.text_config)
cd_vis = cfgdiff(ref.config.vision_config, m.config.vision_config)
wemm_ro = sha256_file(os.path.join(wemm, "readout.safetensors"))
rel_ro = sha256_file(os.path.join(rel, "readout.safetensors"))
res = {"reference": f"wx_engine.build_weights (md5 {WX_MD5}) a={a} b={b} on Qwen3_5Model({wemm})", "build_stats": st, "build_logs": logs,
       "flags_at_build": flags, "reference_tensors": len(P), "raw_identical": raw_same, "raw_differing": raw_bad[:20], "release_extra_keys": extra,
       "lm_head_equals_stock": bool(lm_ok), "loaded_identical": loaded_same, "loaded_name_mismatch": loaded_missing, "loaded_lm_head_equals_stock": bool(lm_loaded_ok),
       "readout_sha256_release": rel_ro, "readout_sha256_wemm": wemm_ro, "config_text_diff": cd_text, "config_vision_diff": cd_vis,
       "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__, "transformers": __import__("transformers").__version__,
       "seconds": round(time.time() - t0, 1)}
res["ALL_IDENTICAL"] = bool(raw_same == len(P) and not raw_bad and extra == ["lm_head.weight"] and lm_ok and loaded_same == len(P) and not loaded_missing
                            and lm_loaded_ok and rel_ro == wemm_ro and set(cd_text) <= BOOK and set(cd_vis) <= BOOK and st.get("b_matrices") == 496 * (b != 0))
dump(res, out_f)
print("MERGE_CHECK " + json.dumps({k: v for k, v in res.items() if k not in ("build_logs", "raw_differing")}), flush=True)
sys.exit(0 if res["ALL_IDENTICAL"] else 1)
