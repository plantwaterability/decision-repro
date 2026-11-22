"""m2_build.py: materialise one merge spec (merge.env) as release weights in the stock Qwen3.8-27B packaging, on one GPU. Publishes nothing.

  m2_build.py SPEC_ENV INPUT_ROOT CFG_BASE_JSON OUT_DIR BUILD_JSON [device]

INPUT_ROOT/base/     the base's small files (config.json, generation_config.json, tokenizer / processor / template files, LICENSE,
                     model.safetensors.index.json) and, only when the spec needs base tensors, the base's shards
INPUT_ROOT/m_<lab>/  each member as fetched: full = model-*.safetensors + model.safetensors.index.json (stock or WeMM key names);
                     lowrank = <x>.json with an "update" block + the update file; peft = adapter.safetensors + state.json
INPUT_ROOT/readout/  readout.safetensors + decision_config.json (the d3 readout and its codes), copied unchanged
CFG_BASE_JSON        the engine settings that are not part of the merge (OCR models and settings)

Arithmetic per output tensor (fp32 on the GPU, ONE bf16 rounding; for M2 this is exactly wx_engine.build_weights, which made the 83.53 read):
  one full member, coefficient 1    w = W                       (no base tensor read; tensors no low-rank member touches are copied as bits)
  one full member, coefficient c    w = B + c * (W - B)         (wx_engine: s + a * (w - s))
  several full members              w = B; w = w + c_i * (W_i - B) for each, in spec order
  no full member                    w = B
  then every low-rank / PEFT member in spec order, every term j in file order:
                                    w.add_(B_j.float() @ A_j.float(), alpha=scale_j * c)          (wx_engine / the soup engine)
Output keys = the base checkpoint's keys minus mtp.* (transformers' class has no MTP head); a key a full member lacks (lm_head for WeMM's
own packaging) comes from the base. Shards of at most 5e9 bytes in the base index's key order; safetensors metadata {"format": "pt"}."""
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2lib import dump, read_env, sha256_file, sha256_many, spec_members  # noqa: E402

spec_f, root, cfg_base_f, out, build_f = sys.argv[1:6]
dev = sys.argv[6] if len(sys.argv) > 6 else "cuda:0"
t0 = time.time()
env = read_env(spec_f)
members = spec_members(env)
WORK = env["WORK"]
SMALL = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
         "preprocessor_config.json", "video_preprocessor_config.json"]
os.makedirs(out, exist_ok=True)
assert not any(f.endswith(".safetensors") for f in os.listdir(out)), f"{out} already holds weights"


class Ckpt:
    """A sharded safetensors checkpoint read tensor by tensor."""

    def __init__(self, d):
        self.d = d
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            self.map = {}
            for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
                with safe_open(os.path.join(d, f), "pt") as h:
                    self.map.update({k: f for k in h.keys()})
        self.h = {}

    def key(self, out_key):
        if out_key in self.map:
            return out_key
        if out_key.startswith("model.") and out_key[6:] in self.map:
            return out_key[6:]
        return None

    def get(self, k):
        f = self.map[k]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(k)


base_dir = os.path.join(root, "base")
bidx = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
keys = [k for k in bidx if not k.startswith("mtp.")]
full = [(lab, c, Ckpt(os.path.join(root, f"m_{lab}"))) for lab, kind, c, _, _ in members if kind == "full"]
need_base_all = not (len(full) == 1 and full[0][1] == 1.0)
base = Ckpt(base_dir) if (need_base_all or any(f[2].key(k) is None for f in full for k in keys)) else None
if base is not None:
    missing = [k for k in keys if base.key(k) is None]
    assert not missing, f"base shards missing for {missing[:3]} (fetch the base's shards for this spec)"

# low-rank terms: out_key -> [(B_j, A_j, alpha)] in spec order, term order
terms, lr_info = {}, []
for lab, kind, c, src, hub in members:
    d = os.path.join(root, f"m_{lab}")
    if kind == "lowrank":
        js = [f for f in sorted(os.listdir(d)) if f.endswith(".json") and "update" in json.load(open(os.path.join(d, f)))]
        assert len(js) == 1, (lab, js)
        mc = json.load(open(os.path.join(d, js[0])))
        scales = [float(x) for x in mc["update"]["scales"]]
        upd = load_file(os.path.join(d, mc["update"]["file"]))
        n = 0
        for k in upd:
            if not k.endswith(".A"):
                continue
            path = k[:-2]
            ok = path + ".weight" if path + ".weight" in bidx else "model." + path + ".weight"
            assert ok in bidx, (lab, k)
            A, B = upd[k], upd[path + ".B"]
            assert A.shape[0] == B.shape[0] == len(scales) == int(mc["update"]["terms"]), (lab, k, A.shape, B.shape)
            terms.setdefault(ok, []).extend((B[j], A[j], scales[j] * c) for j in range(len(scales)))
            n += 1
        assert n == int(mc["update"]["matrices"]), (lab, n, mc["update"]["matrices"])
        lr_info.append({"label": lab, "kind": kind, "coef": c, "json": js[0], "update_file": mc["update"]["file"],
                        "update_sha256": sha256_file(os.path.join(d, mc["update"]["file"])), "declared_sha256": mc["update"].get("sha256"),
                        "matrices": n, "terms": len(scales), "scales": scales})
    elif kind == "peft":
        st = json.load(open(os.path.join(d, "state.json")))
        s = float(st["lora_alpha"]) / float(st["lora_r"])
        sd = load_file(os.path.join(d, "adapter.safetensors"))
        n = 0
        for k in sd:
            if not k.endswith(".lora_A.weight"):
                continue
            path = k[len("base_model.model."):-len(".lora_A.weight")]
            ok = path + ".weight"
            assert ok in bidx, (lab, k)
            terms.setdefault(ok, []).append((sd[k.replace(".lora_A.weight", ".lora_B.weight")], sd[k], s * c))
            n += 1
        lr_info.append({"label": lab, "kind": kind, "coef": c, "matrices": n, "alpha_over_r": s,
                        "adapter_sha256": sha256_file(os.path.join(d, "adapter.safetensors"))})

# shard plan in the base index's key order
def nbytes(k):
    src = full[0][2] if full and full[0][2].key(k) is not None else base
    kk = src.key(k)
    with safe_open(os.path.join(src.d, src.map[kk]), "pt") as h:
        sl = h.get_slice(kk)
        shape, dt = sl.get_shape(), sl.get_dtype()
    n = 1
    for x in shape:
        n *= x
    return n * {"BF16": 2, "F16": 2, "F32": 4}[dt]


sizes = {k: nbytes(k) for k in keys}
plan, cur, cb = [], [], 0
for k in keys:
    if cur and cb + sizes[k] > 5_000_000_000:
        plan.append(cur); cur, cb = [], 0
    cur.append(k); cb += sizes[k]
plan.append(cur)
N = len(plan)
fname = lambda i: f"model-{i + 1:05d}-of-{N:05d}.safetensors"  # noqa: E731
stats = {"tensors": 0, "from_full_bits": 0, "from_base_bits": 0, "full_arith": 0, "lowrank_matrices": 0, "lowrank_terms": 0}
wmap = {}
with torch.no_grad():
    for i, ks in enumerate(plan):
        tens = {}
        for k in ks:
            lr = terms.get(k)
            if len(full) == 1 and full[0][1] == 1.0 and full[0][2].key(k) is not None:
                t = full[0][2].get(full[0][2].key(k))
                if lr is None:
                    tens[k] = t.contiguous(); stats["from_full_bits"] += 1; stats["tensors"] += 1; wmap[k] = fname(i); continue
                w = t.to(dev).float()
            elif not full or all(f[2].key(k) is None for f in full):
                t = base.get(base.key(k))
                if lr is None:
                    tens[k] = t.contiguous(); stats["from_base_bits"] += 1; stats["tensors"] += 1; wmap[k] = fname(i); continue
                w = t.to(dev).float()
            else:
                s = base.get(base.key(k)).to(dev).float()
                if len(full) == 1:
                    lab, c, ck = full[0]
                    w = s + c * (ck.get(ck.key(k)).to(dev).float() - s)
                else:
                    w = s
                    for lab, c, ck in full:
                        if ck.key(k) is not None:
                            w = w + c * (ck.get(ck.key(k)).to(dev).float() - s)
                stats["full_arith"] += 1
                t = s
            if lr is not None:
                for Bj, Aj, alpha in lr:
                    w.add_(Bj.to(dev).float() @ Aj.to(dev).float(), alpha=alpha)
                    stats["lowrank_terms"] += 1
                stats["lowrank_matrices"] += 1
            tens[k] = w.to(torch.bfloat16).cpu().contiguous()
            stats["tensors"] += 1; wmap[k] = fname(i)
            del w
        save_file(tens, os.path.join(out, fname(i)), metadata={"format": "pt"})
        print(f"[build] {fname(i)} {len(ks)} tensors {sum(sizes[k] for k in ks) / 1e9:.2f} GB  {time.time() - t0:.0f}s", flush=True)
        del tens
assert stats["tensors"] == len(keys) and stats["lowrank_matrices"] == len(terms), (stats, len(keys), len(terms))
dump({"metadata": {"total_size": sum(sizes.values())}, "weight_map": wmap}, os.path.join(out, "model.safetensors.index.json"))
for f in SMALL:
    shutil.copyfile(os.path.join(base_dir, f), os.path.join(out, f))
if env["READOUT"].startswith("head:"):  # a trained 26-way answer head of a member checkpoint (head.pt: weight [26, H], bias [26]), unchanged values
    hl = env["READOUT"][5:]; hd = os.path.join(root, f"m_{hl}")
    sj = os.path.join(hd, "state.json")
    if os.path.exists(sj):
        assert float(json.load(open(sj)).get("head_softcap", 0.0)) == 0.0, "softcapped heads are not supported by the engine"
    h = torch.load(os.path.join(hd, "head.pt"), map_location="cpu", weights_only=True)
    save_file({"weight": h["weight"].float().contiguous(), "bias": h["bias"].float().contiguous()}, os.path.join(out, "readout.safetensors"), metadata={"format": "pt"})
    dc = {"codes": [chr(65 + i) for i in range(h["weight"].shape[0])], "token_ids": None, "temperature": 1.0}
else:
    rd = os.path.join(root, "readout")
    shutil.copyfile(os.path.join(rd, "readout.safetensors"), os.path.join(out, "readout.safetensors"))
    dc = json.load(open(os.path.join(rd, "decision_config.json")))
    assert dc["prompt"] == "d3" and dc["pooling"] == "last" and dc["attention_mode"] == "noncausal_full_attention" and float(dc["temperature"]) == 1.0, dc
R = load_file(os.path.join(out, "readout.safetensors"))["weight"]
assert tuple(R.shape) == (len(dc["codes"]), 5120) and R.dtype == torch.float32, (R.shape, R.dtype)
assert dc["codes"][:26] == [chr(65 + i) for i in range(26)]
env_pp = env["PROMPT"]; env_st = env["D3_STATE"]
assert env_pp == "d3" and env_st in ("ocr", "row"), (env_pp, env_st)
files = sorted(os.listdir(out))
sums = sha256_many([os.path.join(out, f) for f in files])
sums = {os.path.basename(p): h for p, h in sums.items()}
cfg = json.load(open(cfg_base_f))
cfg = {"name": WORK, "model_class": "Qwen3_5ForConditionalGeneration",
       "files": {f: sums[f] for f in files},
       "weights": {"index": "model.safetensors.index.json", "shards": N, "tensors": len(keys), "dtype": "bfloat16", "bytes": sum(sizes.values())},
       "readout": {"file": "readout.safetensors", "rows": len(dc["codes"]), "hidden_size": int(R.shape[1]), "dtype": "float32",
                   "bias": "bias" in load_file(os.path.join(out, "readout.safetensors")),
                   "codes": dc["codes"], "token_ids": dc.get("token_ids"), "temperature": float(dc["temperature"]), "source": env["READOUT"]},
       "prompt": {"format": env_pp, "state": env_st},
       "images": {"min_pixels": int(env["MIN_PIXELS"]), "max_pixels": int(env["MAX_PIXELS"])},
       **{k: v for k, v in cfg.items() if k not in ("name", "files", "weights", "readout", "prompt", "images")},
       "merge": {"base": env["BASE"], "members": [{"label": lab, "kind": kind, "coefficient": c, "origin": hub} for lab, kind, c, src, hub in members],
                 "formula": "base + sum_i coefficient_i * (member_i - base); low-rank members as sum_j scale_j B_j A_j; fp32, one bf16 rounding per tensor",
                 "lowrank_matrices": stats["lowrank_matrices"]}}
dump(cfg, os.path.join(out, f"{WORK}.json"))
cfg_sha = sha256_file(os.path.join(out, f"{WORK}.json"))
build = {"BUILD_OK": True, "spec_id": env["SPEC_ID"], "work": WORK, "spec": {k: v for k, v in env.items()}, "stats": stats, "shards": N,
         "out_sha256": {**{f: sums[f] for f in files}, f"{WORK}.json": cfg_sha}, "cfg_sha256": cfg_sha,
         "lowrank": lr_info, "full": [{"label": lab, "coef": c, "keys_found": sum(ck.key(k) is not None for k in keys)} for lab, c, ck in full],
         "base_tensors_read": base is not None, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev),
         "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
         "float32_matmul_precision": torch.get_float32_matmul_precision(), "seconds": round(time.time() - t0, 1)}
dump(build, build_f)
print("BUILD_OK " + json.dumps({k: build[k] for k in ("spec_id", "stats", "shards", "base_tensors_read", "torch", "gpu", "tf32_matmul", "seconds")}), flush=True)
