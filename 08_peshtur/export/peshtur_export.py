"""Export standalone merged Apache-2.0 Hugging Face weights for peshtur (Qwen3.5-4B + v4b2 ckpt_05051).

Arithmetic per tensor (fp32 on GPU, single bf16 rounding, identical to V4BEngine):
  - Untouched tensors (vision tower, vision merger, embeddings, norms, lm_head): copied as exact raw bits from Qwen/Qwen3.5-4B.
  - 336 language-model LoRA matrices (r=16, alpha=32, scale=2.0): w = base.float() + 2.0 * (B.float() @ A.float()) -> bf16.
  - Unused mtp.* tensors are omitted.
  - readout.safetensors stores weight [26, 2560] and bias [26] in fp32 from ckpt_05051/head.pt.
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
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE_REPO = "Qwen/Qwen3.5-4B"
BASE_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SMALL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
]
OCR_CFG = {
    "library": "PaddleOCR",
    "detection": {
        "name": "PP-OCRv5_mobile_det",
        "repo": "PaddlePaddle/PP-OCRv5_mobile_det",
        "revision": "0d63e78e2b680928f6b1747d76a08db6e645efb7",
    },
    "recognition": {
        "name": "en_PP-OCRv5_mobile_rec",
        "repo": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
        "revision": "267c36e24c331595590fe7bd72bde2436fd286f2",
    },
    "det_limit_side_len": 960,
    "rec_batch_size": 8,
    "min_score": 0.5,
    "max_lines": 60,
    "max_chars": 1200,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) < 5:
        raise SystemExit("usage: peshtur_export.py BASE_DIR CKPT_DIR OUT_DIR BUILD_JSON [device]")
    base_dir, ckpt_dir, out_dir, build_json = sys.argv[1:5]
    dev = sys.argv[5] if len(sys.argv) > 5 else "cuda:0"
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    if any(f.endswith(".safetensors") for f in os.listdir(out_dir)):
        raise SystemExit(f"{out_dir} already holds safetensors weights")

    st = json.load(open(os.path.join(ckpt_dir, "state.json")))
    assert st.get("layout") == "d3", st
    assert float(st.get("head_softcap", 0.0)) == 0.0, st
    assert float(st.get("merger_lr_mult", 0.0)) == 0.0, st
    assert not os.path.exists(os.path.join(ckpt_dir, "merger.safetensors")), "unexpected merger.safetensors"
    scale = float(st["lora_alpha"]) / float(st["lora_r"])

    bidx = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in bidx if not k.startswith("mtp.")]

    handles = {}

    def get_base(k):
        f = bidx[k]
        if f not in handles:
            handles[f] = safe_open(os.path.join(base_dir, f), "pt", device="cpu")
        return handles[f].get_tensor(k)

    def nbytes(k):
        f = bidx[k]
        if f not in handles:
            handles[f] = safe_open(os.path.join(base_dir, f), "pt", device="cpu")
        sl = handles[f].get_slice(k)
        n = 1
        for x in sl.get_shape():
            n *= x
        return n * {"BF16": 2, "F16": 2, "F32": 4}[sl.get_dtype()]

    sd = load_file(os.path.join(ckpt_dir, "adapter.safetensors"), device="cpu")
    lora = {}
    for k in sd:
        if not k.endswith(".lora_A.weight"):
            continue
        path = k[len("base_model.model.") : -len(".lora_A.weight")]
        ok = path + ".weight"
        assert ok in bidx, f"adapter key {ok} not in base index"
        lora[ok] = (sd[k.replace(".lora_A.weight", ".lora_B.weight")], sd[k])
    assert len(lora) > 0, "no LoRA weights found"

    sizes = {k: nbytes(k) for k in keys}
    plan, cur, cb = [], [], 0
    for k in keys:
        if cur and cb + sizes[k] > 5_000_000_000:
            plan.append(cur)
            cur, cb = [], 0
        cur.append(k)
        cb += sizes[k]
    plan.append(cur)
    n_shards = len(plan)
    fname = lambda i: f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"  # noqa: E731

    stats = {"tensors": 0, "copied_bits": 0, "merged_lora": 0}
    wmap = {}
    with torch.no_grad():
        for i, ks in enumerate(plan):
            tens = {}
            for k in ks:
                t = get_base(k)
                if k in lora:
                    B, A = lora[k]
                    w = t.to(dev).float()
                    w.add_(B.to(dev).float() @ A.to(dev).float(), alpha=scale)
                    tens[k] = w.to(torch.bfloat16).cpu().contiguous()
                    del w
                    stats["merged_lora"] += 1
                else:
                    tens[k] = t.contiguous()
                    stats["copied_bits"] += 1
                stats["tensors"] += 1
                wmap[k] = fname(i)
            save_file(tens, os.path.join(out_dir, fname(i)), metadata={"format": "pt"})
            print(f"[peshtur_export] {fname(i)} {len(ks)} tensors {sum(sizes[k] for k in ks)/1e9:.2f} GB ({time.time()-t0:.1f}s)", flush=True)
            del tens

    assert stats["tensors"] == len(keys) and stats["merged_lora"] == len(lora), (stats, len(keys), len(lora))
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": sum(sizes.values())}, "weight_map": wmap}, fh, indent=2)
        fh.write("\n")

    for f in SMALL_FILES:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, f))
        elif f != "LICENSE":
            raise FileNotFoundError(f"missing base file {src}")

    h = torch.load(os.path.join(ckpt_dir, "head.pt"), map_location="cpu", weights_only=True)
    W_ro = h["weight"].float().contiguous()
    b_ro = h["bias"].float().contiguous()
    assert W_ro.ndim == 2 and W_ro.shape[0] == 26 and b_ro.shape == (26,), (W_ro.shape, b_ro.shape)
    save_file({"weight": W_ro, "bias": b_ro}, os.path.join(out_dir, "readout.safetensors"), metadata={"format": "pt"})

    # Hash ONLY the exact model/config/tokenizer/processor/readout files the engine reads
    shard_files = [fname(i) for i in range(n_shards)]
    engine_files = sorted(
        shard_files
        + ["model.safetensors.index.json", "readout.safetensors"]
        + [f for f in SMALL_FILES if f != "LICENSE" and os.path.isfile(os.path.join(out_dir, f))]
    )
    with ThreadPoolExecutor(16) as ex:
        hashes = dict(zip(engine_files, ex.map(lambda f: sha256_file(os.path.join(out_dir, f)), engine_files)))

    codes = [chr(65 + i) for i in range(W_ro.shape[0])]
    cfg = {
        "name": "peshtur",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "files": hashes,
        "weights": {
            "index": "model.safetensors.index.json",
            "shards": n_shards,
            "tensors": len(keys),
            "dtype": "bfloat16",
            "bytes": sum(sizes.values()),
            "base": f"hf:{BASE_REPO}@{BASE_REV}",
            "merged_lora_matrices": stats["merged_lora"],
            "lora_r": int(st["lora_r"]),
            "lora_alpha": int(st["lora_alpha"]),
        },
        "readout": {
            "file": "readout.safetensors",
            "rows": int(W_ro.shape[0]),
            "hidden_size": int(W_ro.shape[1]),
            "dtype": "float32",
            "bias": True,
            "codes": codes,
            "temperature": 1.0,
        },
        "prompt": {"format": "d3", "state": "ocr"},
        "images": {"min_pixels": 589824, "max_pixels": 1638400},
        "ocr": OCR_CFG,
    }
    with open(os.path.join(out_dir, "peshtur.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
        fh.write("\n")

    build = {
        "BUILD_OK": True,
        "name": "peshtur",
        "base": f"{BASE_REPO}@{BASE_REV}",
        "step": int(st["step"]),
        "stats": stats,
        "shards": n_shards,
        "bytes": sum(sizes.values()),
        "readout_shape": list(W_ro.shape),
        "cfg_sha256": sha256_file(os.path.join(out_dir, "peshtur.json")),
        "files_sha256": hashes,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(dev),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(build_json)), exist_ok=True)
    with open(build_json, "w") as fh:
        json.dump(build, fh, indent=2)
        fh.write("\n")
    print("BUILD_OK " + json.dumps({k: build[k] for k in ("name", "step", "stats", "shards", "bytes", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
