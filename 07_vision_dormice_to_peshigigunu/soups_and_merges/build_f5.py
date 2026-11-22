"""Build the release weight files on CPU from the seven member checkpoints of the weight average F5 (heads at the same weights).

  build_f5.py NAME SRC_RENDERED_DIR OUT_DIR MEMBERS_ROOT ckpt@weight ...

update: per LoRA target, A [n,16,in] and B [n,out,16] fp32, stored UNSCALED; the engine applies scale_j = (alpha / r) x weight_j with
w.add_(B_j @ A_j, alpha=scale_j), one term at a time in the members' order: the soup engine's own expression (qv_soup_engine.py:
w.add_(B.float() @ A.float(), alpha=(alpha / r) * w_j)). head: the soup engine's expression (weighted mean, weights normalised).
SRC_RENDERED_DIR holds the engine and environment files already rendered for NAME (NAME.py, selfcheck.py, install.sh, pyproject.toml,
uv.lock, .python-version); they are copied unchanged. Writes NAME.json and build.json (member sha256, checks) into OUT_DIR."""
import hashlib, json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

name, src, out, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
MEMBERS = [(m.split("@")[0], float(m.split("@")[1])) for m in sys.argv[5:]]
assert abs(sum(w for _, w in MEMBERS) - 1.0) < 1e-12, MEMBERS
out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(32)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


sds, scales, src_meta = [], [], []
for ck, wt in MEMBERS:
    st = json.loads((root / ck / "state.json").read_text())
    assert st["lora_r"] == 16 and st["lora_alpha"] == 32 and not st.get("merge_from"), (ck, {k: st.get(k) for k in ("lora_r", "lora_alpha", "merge_from")})
    scales.append(st["lora_alpha"] / st["lora_r"] * wt)  # the soup engine's sc * wt, the same double
    sds.append(load_file(str(root / ck / "adapter.safetensors")))
    src_meta.append({"member": ck, "weight": wt, "scale": scales[-1], "adapter_sha256": sha256(root / ck / "adapter.safetensors"),
                     "head_sha256": sha256(root / ck / "head.pt"), "state_sha256": sha256(root / ck / "state.json"),
                     "step": st.get("step"), "lora_r": st["lora_r"], "lora_alpha": st["lora_alpha"]})
keys = [k for k in sds[0] if k.endswith(".lora_A.weight")]
assert all(set(sd) == set(sds[0]) for sd in sds) and len(sds[0]) == 2 * len(keys), "member key sets differ"
upd, n_checked = {}, 0
g = torch.Generator().manual_seed(0)
for k in keys:
    kb = k.replace(".lora_A.weight", ".lora_B.weight")
    path = k[len("base_model.model."):-len(".lora_A.weight")]
    As = [sd[k] for sd in sds]; Bs = [sd[kb] for sd in sds]
    assert all(a.dtype == torch.float32 and a.shape == As[0].shape and a.shape[0] == 16 for a in As)
    assert all(b.dtype == torch.float32 and b.shape == Bs[0].shape and b.shape[1] == 16 for b in Bs)
    A = torch.stack(As).contiguous()
    B = torch.stack(Bs).contiguous()
    upd[path + ".A"], upd[path + ".B"] = A, B
    if n_checked < 12 or path.endswith(("up_proj", "in_proj_qkv")) and n_checked < 40:  # the engine's arithmetic vs the soup engine's (CPU)
        w0 = torch.randn(B.shape[1], A.shape[2], generator=g) * 0.02
        w_ref = w0.clone()
        for a, b, (ck, wt) in zip(As, Bs, MEMBERS):
            w_ref.add_(b.float() @ a.float(), alpha=2.0 * wt)
        w_new = w0.clone()
        for a, b, s in zip(A, B, scales):
            w_new.add_(b @ a, alpha=s)
        assert torch.equal(w_ref, w_new), f"merge arithmetic differs at {path}"
        n_checked += 1
save_file(upd, str(out / f"{name}_update.safetensors"))
heads = [torch.load(root / ck / "head.pt", map_location="cpu") for ck, _ in MEMBERS]
hw = [w for _, w in MEMBERS]; hn = sum(hw)
W = sum(h["weight"].float() * x for h, x in zip(heads, hw)) / hn
b = sum(h["bias"].float() * x for h, x in zip(heads, hw)) / hn
assert W.shape[0] == 26 and b.shape == (26,), (W.shape, b.shape)
save_file({"weight": W.contiguous(), "bias": b.contiguous()}, str(out / f"{name}_head.safetensors"))
cfg = {"name": name, "base_model": "Qwen/Qwen3.8-27B", "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
       "update": {"file": f"{name}_update.safetensors", "sha256": sha256(out / f"{name}_update.safetensors"), "matrices": len(keys), "terms": len(MEMBERS),
                  "rank_per_term": 16, "scales": scales, "dtype": "float32"},
       "head": {"file": f"{name}_head.safetensors", "sha256": sha256(out / f"{name}_head.safetensors"), "options": 26, "hidden_size": int(W.shape[1]), "dtype": "float32"},
       "ocr": {"library": "PaddleOCR",
               "detection": {"name": "PP-OCRv5_mobile_det", "repo": "PaddlePaddle/PP-OCRv5_mobile_det", "revision": "0d63e78e2b680928f6b1747d76a08db6e645efb7"},
               "recognition": {"name": "en_PP-OCRv5_mobile_rec", "repo": "PaddlePaddle/en_PP-OCRv5_mobile_rec", "revision": "267c36e24c331595590fe7bd72bde2436fd286f2"},
               "det_limit_side_len": 960, "rec_batch_size": 8, "min_score": 0.5, "max_lines": 60, "max_chars": 1200},
       "max_pixels": 1638400}
(out / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
for f in (f"{name}.py", "selfcheck.py", "install.sh", "pyproject.toml", "uv.lock", ".python-version"):
    shutil.copyfile(src / f, out / f)
(out / "install.sh").chmod(0o755)
info = {"BUILD_OK": True, "name": name, "matrices": len(keys), "terms": len(MEMBERS), "scales": scales, "params": sum(v.numel() for v in upd.values()),
        "checked_modules_cpu": n_checked, "members": src_meta, "sizes": {p.name: p.stat().st_size for p in out.iterdir()}, "cfg": cfg}
(out.parent / "build.json").write_text(json.dumps(info, indent=1) + "\n")
print(json.dumps(info))
