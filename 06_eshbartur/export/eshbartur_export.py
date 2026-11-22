"""Export standalone merged Apache-2.0 pure-text Hugging Face weights for eshbartur (Qwen3.5-4B + t4b3 ckpt_03516 + T=1.40).

Arithmetic per tensor (fp32 on GPU, single bf16 rounding, identical to T4BEngine):
  - Loads Qwen/Qwen3.5-4B as pure-text Qwen3_5ForCausalLM (4.21B parameters; vision encoder and mtp.* omitted).
  - Merges 336 LoRA matrices (r=32, alpha=64, scale=2.0) in fp32 with single bf16 rounding.
  - Saves standard <= 5 GB safetensors shards + model.safetensors.index.json + text config.json + tokenizer files + LICENSE.
  - Exports readout.safetensors with weight = head.weight / 1.40 and bias = head.bias / 1.40 (fp32, T=1.40 folded).
  - Safetensors metadata is strictly {"format": "pt"} (zero internal names or paths).
"""
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

BASE_REPO = "Qwen/Qwen3.5-4B"
BASE_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CALIB_T = 1.40
SMALL_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "LICENSE",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) < 5:
        raise SystemExit("usage: eshbartur_export.py BASE_DIR CKPT_DIR OUT_DIR BUILD_JSON [device]")
    base_dir, ckpt_dir, out_dir, build_json = sys.argv[1:5]
    dev = sys.argv[5] if len(sys.argv) > 5 else "cuda:0"
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    if any(f.endswith(".safetensors") for f in os.listdir(out_dir)):
        raise SystemExit(f"{out_dir} already holds safetensors weights")

    st = json.load(open(os.path.join(ckpt_dir, "state.json")))
    assert st.get("layout") == "t4b_bidir_re2_v1", st
    assert bool(st.get("bidir")) is True, st
    assert int(st.get("read_twice_tokens", 0)) == 2048, st
    scale = float(st["lora_alpha"]) / float(st["lora_r"])

    cfg = AutoConfig.from_pretrained(base_dir).get_text_config()
    m, info = Qwen3_5ForCausalLM.from_pretrained(
        base_dir,
        config=cfg,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    missing = [k for k in (info.get("missing_keys") or []) if k != "lm_head.weight"]
    assert not missing, f"missing keys: {missing[:5]}"
    m = m.to(dev)

    sd = load_file(os.path.join(ckpt_dir, "adapter.safetensors"), device=str(dev))
    merged = 0
    with torch.no_grad():
        for k in sd:
            if not k.endswith(".lora_A.weight"):
                continue
            mod_name = k[len("base_model.model.") : -len(".lora_A.weight")]
            mod = m.get_submodule(mod_name)
            w = mod.weight.float()
            w.add_(sd[k.replace(".lora_A.weight", ".lora_B.weight")].float() @ sd[k].float(), alpha=scale)
            mod.weight.copy_(w.to(mod.weight.dtype))
            del w
            merged += 1
    del sd
    torch.cuda.empty_cache()
    assert merged > 0, "no LoRA modules merged"

    # Save pure-text config.json cleanly
    cfg.architectures = ["Qwen3_5ForCausalLM"]
    cfg.save_pretrained(out_dir)

    # Gather state dict (exclude tied lm_head.weight if tie_word_embeddings is True)
    full_sd = m.state_dict()
    tie = bool(getattr(cfg, "tie_word_embeddings", True))
    save_keys = [k for k in full_sd if not (tie and k == "lm_head.weight")]
    sizes = {k: int(full_sd[k].numel() * full_sd[k].element_size()) for k in save_keys}

    plan, cur, cb = [], [], 0
    for k in save_keys:
        if cur and cb + sizes[k] > 5_000_000_000:
            plan.append(cur)
            cur, cb = [], 0
        cur.append(k)
        cb += sizes[k]
    plan.append(cur)
    n_shards = len(plan)
    fname = lambda i: f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"  # noqa: E731

    wmap = {}
    for i, ks in enumerate(plan):
        tens = {k: full_sd[k].detach().cpu().contiguous() for k in ks}
        for k in ks:
            wmap[k] = fname(i)
        save_file(tens, os.path.join(out_dir, fname(i)), metadata={"format": "pt"})
        print(f"[eshbartur_export] {fname(i)} {len(ks)} tensors {sum(sizes[k] for k in ks)/1e9:.2f} GB ({time.time()-t0:.1f}s)", flush=True)
        del tens
    del m, full_sd
    torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": sum(sizes.values())}, "weight_map": wmap}, fh, indent=2)
        fh.write("\n")

    for f in SMALL_FILES:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, f))
        elif f != "LICENSE":
            raise FileNotFoundError(f"missing base file {src}")

    # Export readout.safetensors with held-out calibration temperature T = 1.40 folded into fp32 weight & bias
    h = torch.load(os.path.join(ckpt_dir, "readout.pt"), map_location="cpu", weights_only=True)
    W_ro = (h["weight"].float() / CALIB_T).contiguous()
    b_ro = (h["bias"].float() / CALIB_T).contiguous()
    assert W_ro.ndim == 2 and W_ro.shape[0] == len(st["codes"]) == 255 and b_ro.shape == (255,), (W_ro.shape, b_ro.shape)
    save_file({"weight": W_ro, "bias": b_ro}, os.path.join(out_dir, "readout.safetensors"), metadata={"format": "pt"})

    # Verify tokenizer codes match
    tok = AutoTokenizer.from_pretrained(out_dir)
    assert len(st["codes"]) == len(st["code_ids"]) == 255

    # Hash ONLY the exact model/config/tokenizer/readout files the engine reads
    shard_files = [fname(i) for i in range(n_shards)]
    engine_files = sorted(
        shard_files
        + ["model.safetensors.index.json", "readout.safetensors", "config.json"]
        + [f for f in SMALL_FILES if f != "LICENSE" and os.path.isfile(os.path.join(out_dir, f))]
    )
    with ThreadPoolExecutor(16) as ex:
        hashes = dict(zip(engine_files, ex.map(lambda f: sha256_file(os.path.join(out_dir, f)), engine_files)))

    rel_cfg = {
        "name": "eshbartur",
        "model_class": "Qwen3_5ForCausalLM",
        "files": hashes,
        "weights": {
            "index": "model.safetensors.index.json",
            "shards": n_shards,
            "tensors": len(save_keys),
            "dtype": "bfloat16",
            "bytes": sum(sizes.values()),
            "base": f"hf:{BASE_REPO}@{BASE_REV}",
            "merged_lora_matrices": merged,
            "lora_r": int(st["lora_r"]),
            "lora_alpha": int(st["lora_alpha"]),
        },
        "readout": {
            "file": "readout.safetensors",
            "rows": int(W_ro.shape[0]),
            "hidden_size": int(W_ro.shape[1]),
            "dtype": "float32",
            "bias": True,
            "codes": st["codes"],
            "code_ids": st["code_ids"],
            "temperature": 1.0,
            "calibration_temperature_folded": CALIB_T,
        },
        "prompt": {
            "layout": "bidir_re2_v1",
            "bidir_full_attention": True,
            "read_twice_tokens": 2048,
            "question_types": ["choice", "noul", "score"],
            "max_options": 255,
            "max_len": 65536,
        },
    }
    with open(os.path.join(out_dir, "eshbartur.json"), "w") as fh:
        json.dump(rel_cfg, fh, indent=1)
        fh.write("\n")

    build = {
        "BUILD_OK": True,
        "name": "eshbartur",
        "base": f"{BASE_REPO}@{BASE_REV}",
        "step": int(st["step"]),
        "merged_lora_matrices": merged,
        "tensors": len(save_keys),
        "shards": n_shards,
        "bytes": sum(sizes.values()),
        "readout_shape": list(W_ro.shape),
        "calibration_temperature_folded": CALIB_T,
        "cfg_sha256": sha256_file(os.path.join(out_dir, "eshbartur.json")),
        "files_sha256": hashes,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(dev),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(build_json)), exist_ok=True)
    with open(build_json, "w") as fh:
        json.dump(build, fh, indent=2)
        fh.write("\n")
    print("BUILD_OK " + json.dumps({k: build[k] for k in ("name", "step", "merged_lora_matrices", "tensors", "shards", "bytes", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
