"""Build the dormice weight files on CPU from the three member checkpoints (SU50 = U@0.5 + A2@0.25 + A3@0.25).
update: per LoRA target, A [3,16,in] and B [3,out,16] fp32 with (alpha/r)*weight folded into B (1.0, 0.5, 0.5: exact).
head: 0.5*U + 0.25*A2 + 0.25*A3 (weight [26,H], bias [26]) fp32, the soup engine's expression.
usage: python build_bundle.py <out_dir>"""
import hashlib, json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

Q = Path("/data/local/qv")
MEMBERS = [("runs/u_union/ckpt_03000", 0.5), ("runs/a2_mix/ckpt_01000", 0.25), ("runs/a3_mix/ckpt_01000", 0.25)]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(16)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

sds, scales, src = [], [], {}
for ck, wt in MEMBERS:
    st = json.loads((Q / ck / "state.json").read_text())
    assert st["lora_r"] == 16 and st["lora_alpha"] == 32 and not st.get("merge_from"), st
    scales.append(st["lora_alpha"] / st["lora_r"] * wt)
    sds.append(load_file(str(Q / ck / "adapter.safetensors")))
    src[ck] = {"adapter_sha256": sha256(Q / ck / "adapter.safetensors"), "head_sha256": sha256(Q / ck / "head.pt"), "weight": wt}
assert scales == [1.0, 0.5, 0.5], scales
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
    B = torch.stack([b * s for b, s in zip(Bs, scales)]).contiguous()
    for b, b2, s in zip(Bs, B, scales):
        assert torch.equal(b2 / s, b), "scale fold not exact"
    upd[path + ".A"], upd[path + ".B"] = A, B
    if n_checked < 12 or path.endswith(("up_proj", "in_proj_qkv")) and n_checked < 40:  # the engine's arithmetic vs the soup engine's, same device
        w0 = torch.randn(B.shape[1], A.shape[2], generator=g) * 0.02
        w_ref = w0.clone()
        for a, b, s in zip(As, Bs, [2.0 * 0.5, 2.0 * 0.25, 2.0 * 0.25]):
            w_ref.add_(b.float() @ a.float(), alpha=s)
        w_new = w0.clone()
        for a, b in zip(A, B):
            w_new.add_(b @ a)
        assert torch.equal(w_ref, w_new), f"merge arithmetic differs at {path}"
        n_checked += 1
save_file(upd, str(out / "dormice_update.safetensors"))
heads = [torch.load(Q / ck / "head.pt", map_location="cpu") for ck, _ in MEMBERS]
hw = [w for _, w in MEMBERS]; hn = sum(hw)
W = sum(h["weight"].float() * x for h, x in zip(heads, hw)) / hn
b = sum(h["bias"].float() * x for h, x in zip(heads, hw)) / hn
assert W.shape[0] == 26 and b.shape == (26,), (W.shape, b.shape)
save_file({"weight": W.contiguous(), "bias": b.contiguous()}, str(out / "dormice_head.safetensors"))
cfg = {"name": "dormice", "base_model": "Qwen/Qwen3.8-27B", "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
       "update": {"file": "dormice_update.safetensors", "sha256": sha256(out / "dormice_update.safetensors"), "matrices": len(keys), "terms": 3, "rank_per_term": 16, "dtype": "float32"},
       "head": {"file": "dormice_head.safetensors", "sha256": sha256(out / "dormice_head.safetensors"), "options": 26, "hidden_size": int(W.shape[1]), "dtype": "float32"},
       "max_pixels": 1638400}
(out / "dormice.json").write_text(json.dumps(cfg, indent=2) + "\n")
n_par = sum(v.numel() for v in upd.values())
print(json.dumps({"BUILD_OK": True, "matrices": len(keys), "params": n_par, "checked_modules": n_checked, "cfg": cfg, "members": src,
                  "sizes": {p.name: p.stat().st_size for p in out.iterdir()}}))
