"""Independent bit-exact merge & metadata verification for peshtur exported HF repo."""
import json
import os
import struct
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import Qwen3_5ForConditionalGeneration


def main():
    if len(sys.argv) < 5:
        raise SystemExit("usage: peshtur_check_merge.py BASE_DIR CKPT_DIR REPO_DIR OUT_JSON [device]")
    base_dir, ckpt_dir, repo_dir, out_json = sys.argv[1:5]
    dev = sys.argv[5] if len(sys.argv) > 5 else "cuda:0"
    t0 = time.time()

    # 1. Check safetensors metadata + index metadata for zero internal leaks
    leaks = []
    for f in sorted(os.listdir(repo_dir)):
        p = os.path.join(repo_dir, f)
        if f.endswith(".safetensors"):
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            meta = hdr.get("__metadata__")
            if meta != {"format": "pt"}:
                leaks.append((f, meta))
        elif f == "model.safetensors.index.json":
            j = json.load(open(p))
            blob = json.dumps(j.get("metadata")).lower()
            for w in ("v4b", "plantwaterer", "trainers", "jev", "s3:", "/mnt", "claude", "anthropic"):
                if w in blob:
                    leaks.append((f, w))

    # 2. Load base model and apply adapter.safetensors identically to V4BEngine
    st = json.load(open(os.path.join(ckpt_dir, "state.json")))
    scale = float(st["lora_alpha"]) / float(st["lora_r"])
    m_ref = Qwen3_5ForConditionalGeneration.from_pretrained(base_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": dev})
    sd = load_file(os.path.join(ckpt_dir, "adapter.safetensors"), device=str(dev))
    merged = 0
    with torch.no_grad():
        for k in sd:
            if not k.endswith(".lora_A.weight"):
                continue
            path = k[len("base_model.model.") : -len(".lora_A.weight")]
            mod = m_ref.get_submodule(path)
            w = mod.weight.float()
            w.add_(sd[k.replace(".lora_A.weight", ".lora_B.weight")].float() @ sd[k].float(), alpha=scale)
            mod.weight.copy_(w.to(mod.weight.dtype))
            del w
            merged += 1
    del sd
    ref_sd = {k: v.detach().cpu() for k, v in m_ref.state_dict().items()}
    del m_ref
    torch.cuda.empty_cache()

    # 3. Load exported standalone repo via transformers and verify bit-exact equality
    m_exp, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        repo_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu", output_loading_info=True
    )
    missing = info.get("missing_keys") or []
    unexpected = info.get("unexpected_keys") or []
    exp_sd = m_exp.state_dict()
    same_keys = set(ref_sd) == set(exp_sd)
    diff_tensors = []
    if same_keys:
        for k in sorted(ref_sd):
            if not torch.equal(ref_sd[k], exp_sd[k]):
                d = (ref_sd[k].float() - exp_sd[k].float()).abs().max().item()
                diff_tensors.append({"key": k, "max_abs_diff": d})

    # 4. Check readout.safetensors bit-identical to head.pt
    h_ref = torch.load(os.path.join(ckpt_dir, "head.pt"), map_location="cpu", weights_only=True)
    h_exp = load_file(os.path.join(repo_dir, "readout.safetensors"))
    ro_w_eq = bool(torch.equal(h_ref["weight"].float(), h_exp["weight"]))
    ro_b_eq = bool(torch.equal(h_ref["bias"].float(), h_exp["bias"]))

    res = {
        "ALL_IDENTICAL": bool(same_keys and not diff_tensors and not missing and not unexpected and ro_w_eq and ro_b_eq and not leaks),
        "merged_lora_matrices": merged,
        "reference_tensors": len(ref_sd),
        "exported_tensors": len(exp_sd),
        "same_keys": same_keys,
        "missing_keys": missing[:10],
        "unexpected_keys": unexpected[:10],
        "diff_tensors": diff_tensors[:10],
        "readout_weight_equal": ro_w_eq,
        "readout_bias_equal": ro_b_eq,
        "metadata_leaks": leaks,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2)
        fh.write("\n")
    print("MERGE_CHECK " + json.dumps(res), flush=True)
    if not res["ALL_IDENTICAL"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
