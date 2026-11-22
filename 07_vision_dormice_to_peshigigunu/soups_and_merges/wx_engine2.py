"""WX student engine (wxs1 reads): wx_engine.WXEngine (unchanged file, md5 165f1ad3 = the M2 inference truth) in family d3 with the OCR state,
weights = M2 (a 1, b 0.5) + an optional trainer LoRA checkpoint merged exactly as VX's soup engine merges one member
(W = bf16(W_fp32 + (alpha / r) B @ A) per module), head = an optional trainer head.pt (26 x H weight + bias) instead of WeMM's readout.
The bias is folded in as an extra hidden unit (z = [h, 1] @ [W | b]^T) so WXEngine.__call__ runs unchanged.
With no checkpoint and no head file this IS the M2 engine (a reproduction check)."""
import json
from pathlib import Path

from wx_engine import WXEngine


class WXStudent(WXEngine):
    name = "wx-student"

    def __init__(self, wemm_dir, stock_dir, upd_dir, ckpt=None, a=1.0, b=0.5, device="cuda:0", image_root=None, label=None, ocr=None, **options):
        super().__init__("d3", wemm_dir, stock_dir, upd_dir, a=a, b=b, head="readout", device=device, image_root=image_root, label=label, ocr=ocr, **options)
        import torch
        from safetensors.torch import load_file
        self.aug = False
        merged = 0
        if ckpt:
            ck = Path(ckpt)
            st = json.loads((ck / "state.json").read_text())
            if st.get("layout") != "d3":
                raise ValueError(f"{ck}: trained with layout {st.get('layout')!r}, not d3")
            sc = st["lora_alpha"] / st["lora_r"]
            sd = load_file(str(ck / "adapter.safetensors"), device=str(device))
            with torch.no_grad():
                for k in sd:
                    if not k.endswith(".lora_A.weight"):
                        continue
                    path = k[len("base_model.model."):-len(".lora_A.weight")]
                    assert path.startswith("model."), path
                    mod = self.model.get_submodule(path[len("model."):])
                    w = mod.weight.float()
                    w.add_(sd[k.replace(".lora_A.weight", ".lora_B.weight")].float() @ sd[k].float(), alpha=sc)
                    mod.weight.copy_(w.to(mod.weight.dtype))
                    del w
                    merged += 1
            del sd
            torch.cuda.empty_cache()
            h = torch.load(ck / "head.pt", map_location=device)
            self.R = torch.cat([h["weight"].float(), h["bias"].float()[:, None]], 1).to(device)
            self.aug = True
        self.provenance = {**self.provenance, "kind": "wx-student", "ckpt": str(ckpt) if ckpt else None, "lora_modules_merged": merged,
                           "head": "trainer head.pt (bias folded)" if self.aug else "WeMM readout"}

    def _hidden(self, enc):
        h = super()._hidden(enc)
        return self.torch.cat([h, h.new_ones(1)]) if self.aug else h
