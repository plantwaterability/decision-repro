"""Compare the vision tower + projector tensors (model.vision_tower.*, model.embed_vision.*) of each checkpoint with google/gemma-4-31B-it,
byte for byte, and spot-check that text weights differ.
usage: python verify_vision_tower.py <ref_dir> <dir> [<dir> ...]
"""
import hashlib
import json
import sys
from pathlib import Path

from safetensors import safe_open


def tensors(d, pred):
    wm = json.loads((Path(d) / "model.safetensors.index.json").read_text())["weight_map"]
    by = {}
    for k, f in wm.items():
        if pred(k):
            by.setdefault(f, []).append(k)
    out = {}
    for f, ks in by.items():
        with safe_open(str(Path(d) / f), "pt") as fh:
            for k in ks:
                t = fh.get_tensor(k)
                out[k] = (str(t.dtype), tuple(t.shape), hashlib.sha256(t.contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest())
    return out


vis = lambda k: k.startswith("model.vision_tower.") or k.startswith("model.embed_vision.")
probe = {"model.language_model.layers.30.self_attn.q_proj.weight", "model.language_model.layers.59.mlp.down_proj.weight",
         "model.language_model.embed_tokens.weight"}
ref = sys.argv[1]
rv = tensors(ref, vis)
rt = tensors(ref, lambda k: k in probe)
report = {"ref": ref, "vision_tensors": len(rv)}
for d in sys.argv[2:]:
    dv = tensors(d, vis)
    dt = tensors(d, lambda k: k in probe)
    same = sum(dv.get(k) == v for k, v in rv.items())
    report[d] = {"vision_tensors": len(dv), "identical_to_ref": same, "all_identical": same == len(rv) == len(dv),
                 "text_probe_identical": {k: dt.get(k) == v for k, v in rt.items()}}
print(json.dumps(report, indent=1))
