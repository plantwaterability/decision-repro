"""VL round: the trainer / eval-engine / eval-wrapper COPIES with the layout switch (and the trainable-merger option), generated from the stock
files by exact, counted replacements (md5 of each source asserted). Nothing under the stock code prefix or the shipped peshgi tree is touched.
usage: vl_gen_code.py <outdir>      sources: ops/vmx/ref/{qv_train.py f555b7e9, qv_soup_engine.py 586658b4, soup_eval_ocr.py 37a674ff}"""
import hashlib
import sys
from pathlib import Path

OPS = Path(__file__).resolve().parents[2]
REF = OPS / "vmx" / "ref"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()  # noqa: E731


def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:90], s.count(old))
    return s.replace(old, new)


# ------------------------------------------------------------------ trainer
T = (REF / "qv_train.py").read_text()
assert md5(T) == "f555b7e919b8877a72e4bf5954b449be", md5(T)
T = rep(T, '"""Qwen3.8-27B vision decision trainer:',
        '"""VL copy of qv_train.py (f555b7e9) with two switches, both off by default (then every code path is the stock one):\n'
        '  --layout sandwich        question (+ text options) in front of the images, in training and in every in-trainer read (qv_layout.py)\n'
        '  --merger-lr-mult X > 0   the vision merger (6 tensors) is trained with full fp32 weights at --lr * X, own gradient clip, saved as\n'
        '                           merger.safetensors in every checkpoint; the vision tower stays frozen\n\n'
        'Qwen3.8-27B vision decision trainer:')
T = rep(T, 'sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport qv_common as C  # noqa: E402\nimport qv_ocr as O  # noqa: E402\n',
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        'sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "code"))  # VL: the stock (md5-checked) qv_common / qv_ocr live in code/\n'
        'import qv_common as C  # noqa: E402\nimport qv_ocr as O  # noqa: E402\nimport qv_layout as LY  # noqa: E402\n')
T = rep(T, 'STOP = {"why": None}\n', 'STOP = {"why": None}\nVL = {"layout": "images_first", "hlen": 0}  # set in main() from --layout\n')
T = rep(T, '''    question = O.ocr_prefix(ocr, row["images"]) + row["question"]  # OCR block before the question (--ocr-train), else unchanged
    text = C.render(proc, question, row["options"], len(pils), pr.get("options_in_image", False), pr.get("readout"))
    enc = proc(text=[text], images=pils if pils else None, return_tensors="pt")
    n = int(enc["input_ids"].shape[1])
    return (enc, n) if n <= max_tokens else (None, n)
''', '''    # VL: the OCR block is handed over on its own; images_first renders C.render(proc, block + question, ...) exactly as the stock trainer
    text = LY.render(proc, row["question"], row["options"], len(pils), pr.get("options_in_image", False), pr.get("readout"),
                     O.ocr_prefix(ocr, row["images"]), VL["layout"])
    enc = proc(text=[text], images=pils if pils else None, return_tensors="pt")
    n = int(enc["input_ids"].shape[1])
    # the cap applies to the images_first length, so the layout changes no row's kept / skipped status
    n_cap = n - LY.added_tokens(enc["input_ids"][0], proc, VL["hlen"], VL["layout"], len(pils))
    return (enc, n) if n_cap <= max_tokens else (None, n)
''')
T = rep(T, '''        question = O.ocr_prefix(ocr, r["images"]) + q["instructions"]
        text = C.render(proc, question, [q["criteria"][k] for k in keys], len(pils), pr.get("options_in_image", False), pr.get("readout"))
''', '''        text = LY.render(proc, q["instructions"], [q["criteria"][k] for k in keys], len(pils), pr.get("options_in_image", False), pr.get("readout"),
                         O.ocr_prefix(ocr, r["images"]), VL["layout"])
''')
T = rep(T, '''                                          "probs": probs, "ocr": ocr is not None}))''',
        '''                                          "probs": probs, "ocr": ocr is not None, "layout": VL["layout"]}))''')
T = rep(T, '''    a = ap.parse_args()
''', '''    ap.add_argument("--layout", default="images_first", choices=LY.LAYOUTS, help="VL: prompt layout in training and in every in-trainer read")
    ap.add_argument("--merger-lr-mult", type=float, default=0.0, help="VL: > 0 trains the vision merger (full fp32 weights) at --lr times this")
    a = ap.parse_args()
    VL["layout"] = a.layout
''')
T = rep(T, '''    C.set_pixel_cap(proc)
    tok = proc.tokenizer
''', '''    C.set_pixel_cap(proc)
    VL["hlen"] = LY.header_len(proc) if a.layout == "sandwich" else 0
    tok = proc.tokenizer
''')
T = rep(T, '''    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr, "base": a.lr},
                             {"params": head_params, "lr": a.head_lr, "base": a.head_lr}], betas=(0.9, 0.999), eps=1e-8, weight_decay=a.wd)
''', '''    merger, merger_params = None, []
    if a.merger_lr_mult > 0:  # VL (arm uoq2): the merger's six tensors, full weights in fp32; lora_params above was taken before, so it is unchanged
        merger = LY.merger_fp32(LY.merger_of(model))
        merger_params = list(merger.parameters())
        for p in merger_params:
            p.requires_grad_(True)
        if C.inner_model(model).visual.dtype != torch.bfloat16:
            raise SystemExit("the vision tower's dtype changed with the fp32 merger")
    groups = [{"params": lora_params, "lr": a.lr, "base": a.lr}, {"params": head_params, "lr": a.head_lr, "base": a.head_lr}]
    if merger_params:
        groups.append({"params": merger_params, "lr": a.lr * a.merger_lr_mult, "base": a.lr * a.merger_lr_mult})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.wd)
''')
T = rep(T, '''        set_peft_model_state_dict(model, load_file(str(rp / "adapter.safetensors"), device=str(dev)))
        head.load_state_dict(torch.load(rp / "head.pt", map_location=dev))
        opt.load_state_dict(torch.load(rp / "optim.pt", map_location=dev))
        step0 = json.loads((rp / "state.json").read_text())["step"]
''', '''        rst = json.loads((rp / "state.json").read_text())
        if rst.get("layout", "images_first") != a.layout or float(rst.get("merger_lr_mult", 0.0)) != a.merger_lr_mult:
            raise SystemExit(f"resume checkpoint was trained with layout / merger {rst.get('layout')} / {rst.get('merger_lr_mult')}: fail closed")
        set_peft_model_state_dict(model, load_file(str(rp / "adapter.safetensors"), device=str(dev)))
        head.load_state_dict(torch.load(rp / "head.pt", map_location=dev))
        if merger is not None:
            merger.load_state_dict(load_file(str(rp / "merger.safetensors"), device=str(dev)), strict=True)
        opt.load_state_dict(torch.load(rp / "optim.pt", map_location=dev))
        step0 = rst["step"]
''')
T = rep(T, '''               "step0": step0, "steps": a.steps, "bidir": not a.no_bidir, "letter_ids": lids[:4],
''', '''               "step0": step0, "steps": a.steps, "bidir": not a.no_bidir, "letter_ids": lids[:4],
               "layout": a.layout, "header_tokens": VL["hlen"], "merger_params": sum(p.numel() for p in merger_params),
               "merger_lr": a.lr * a.merger_lr_mult if merger_params else None, "param_groups": [len(g["params"]) for g in opt.param_groups],
               "merger_shapes": {k: list(v.shape) for k, v in merger.state_dict().items()} if merger is not None else None,
               "merger_dtype": str(merger_params[0].dtype) if merger_params else None,
               "code": {"qv_common": C.__file__, "qv_ocr": O.__file__, "qv_layout": LY.__file__},
''')
T = rep(T, '''            save_file(sd, str(d / "adapter.safetensors"))
''', '''            save_file(sd, str(d / "adapter.safetensors"))
            if merger is not None:  # written before state.json, so a checkpoint with a state.json always has its merger
                save_file({k: v.detach().to("cpu").contiguous() for k, v in merger.state_dict().items()}, str(d / "merger.safetensors"))
''')
T = rep(T, '''                                                      "manifest": a.manifest, "seed": a.seed, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))''',
        '''                                                      "manifest": a.manifest, "seed": a.seed, "layout": a.layout, "merger_lr_mult": a.merger_lr_mult,
                                                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))''')
T = rep(T, '''            params = lora_params + head_params
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad.div_(world)
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
''', '''            params = lora_params + head_params
            for p in params + merger_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad.div_(world)
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            gm = torch.nn.utils.clip_grad_norm_(merger_params, a.clip) if merger_params else None  # own clip: the LoRA / head clip is as without it
''')
T = rep(T, '''                       "s": round(time.time() - t0, 2)})
            now = dt.datetime.now(dt.timezone.utc)''', '''                       "s": round(time.time() - t0, 2), **({"gnorm_merger": float(gm)} if gm is not None else {})})
            now = dt.datetime.now(dt.timezone.utc)''')
(out / "qv_train_vl.py").write_text(T)

# ------------------------------------------------------------------ eval engine
E = (REF / "qv_soup_engine.py").read_text()
assert md5(E) == "586658b4c5d0bf7cfb2c4583ae033ee9", md5(E)
E = rep(E, '"""Decision Index vision engine for adapter soups of qv_train checkpoints (and optional option-order averaging).',
        '"""VL copy of qv_soup_engine.py (586658b4): layout= (qv_layout.py; images_first = the stock prompt) and a checkpoint\'s trained vision merger\n'
        '(merger.safetensors, fp32 as in training; single checkpoint only). A layout that differs from the checkpoint\'s state.json is refused.\n\n'
        'Decision Index vision engine for adapter soups of qv_train checkpoints (and optional option-order averaging).')
E = rep(E, 'sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport qv_common as C  # noqa: E402\n',
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        'sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "code"))  # VL: the stock qv_common lives in code/\n'
        'import qv_common as C  # noqa: E402\nimport qv_layout as LY  # noqa: E402\n')
E = rep(E, 'label=None, max_pixels=MAX_PIXELS, **options):', 'label=None, max_pixels=MAX_PIXELS, layout="images_first", **options):')
E = rep(E, '''        self.torch, self.device, self.image_root, self.max_pixels, self.perm = torch, device, image_root, max_pixels, int(perm)
''', '''        self.torch, self.device, self.image_root, self.max_pixels, self.perm = torch, device, image_root, max_pixels, int(perm)
        if layout not in LY.LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}")
        self.layout = layout
''')
E = rep(E, '''        del sds
        torch.cuda.empty_cache()
        self.model = m.eval()
''', '''        del sds
        torch.cuda.empty_cache()
        states = [json.loads((ck / "state.json").read_text()) for ck in cks]
        bad = [str(ck) for ck, st in zip(cks, states) if st.get("layout", "images_first") != layout]
        if bad:  # a checkpoint is read with the layout it was trained with, or not at all
            raise ValueError(f"layout {layout!r} != the layout in state.json of {bad}")
        mg_files = [ck / "merger.safetensors" for ck in cks if (ck / "merger.safetensors").exists()]
        if any(float(st.get("merger_lr_mult", 0.0)) > 0 for st in states) and len(mg_files) != len(cks):
            raise ValueError("a checkpoint trained with the merger has no merger.safetensors")
        self.merger_file = None
        if mg_files:  # VL (arm uoq2): the checkpoint's trained vision merger, fp32 as in training
            if len(cks) != 1:
                raise ValueError("a checkpoint with a trained merger cannot be part of a soup")
            LY.merger_fp32(LY.merger_of(m)).load_state_dict(load_file(str(mg_files[0]), device=str(device)), strict=True)
            self.merger_file = str(mg_files[0])
        self.model = m.eval()
''')
E = rep(E, 'self.provenance = {"kind": "qv-soup-head", "model": model,', 'self.provenance = {"kind": "qv-soup-head", "layout": layout, "merger": self.merger_file, "model": model,')
E = rep(E, '''        text = C.render(self.proc, q_text, opts, len(pils), pr.get("options_in_image", False), pr.get("readout"))
''', '''        text = LY.render(self.proc, q_text, opts, len(pils), pr.get("options_in_image", False), pr.get("readout"), pr.get("ocr_block", ""), self.layout)
''')
(out / "qv_soup_engine_vl.py").write_text(E)

# ------------------------------------------------------------------ OCR eval wrapper
W = (REF / "soup_eval_ocr.py").read_text()
assert md5(W) == "37a674ff18a0ff40827792b60951e13c", md5(W)
W = rep(W, '"""soup_eval.py with the OCR block in the state',
        '"""VL copy of soup_eval_ocr.py (37a674ff): env QV_LAYOUT=images_first (default; the stock path, the block prefixed to the question) or\n'
        'sandwich (the bare question goes to the engine, the OCR block travels in the prompt dict and is placed after the images).\n\n'
        'soup_eval.py with the OCR block in the state')
W = rep(W, '''label, ckpts, perm, rows_f, root, out_dir, model_dir, cache_f = sys.argv[1:9]
import decision_index.vision.run as R  # noqa: E402
from decision_index.engines.base import Engine  # noqa: E402
import qv_ocr as O  # noqa: E402
from qv_soup_engine import QVSoupEngine  # noqa: E402
''', '''import os
from pathlib import Path

label, ckpts, perm, rows_f, root, out_dir, model_dir, cache_f = sys.argv[1:9]
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "code"))  # VL: the stock qv_ocr / qv_common live in code/
LAYOUT = os.environ.get("QV_LAYOUT", "images_first")
import decision_index.vision.run as R  # noqa: E402
from decision_index.engines.base import Engine  # noqa: E402
import qv_ocr as O  # noqa: E402
from qv_soup_engine_vl import QVSoupEngine  # noqa: E402
''')
W = rep(W, '''        q["instructions"] = O.ocr_prefix(TEXT, ims) + q["instructions"]
        resp, raw = self.inner(state, {key: q}, images=images, prompt=prompt)
        raw = dict(raw)
''', '''        block = O.ocr_prefix(TEXT, ims)
        if LAYOUT == "images_first":
            q["instructions"] = block + q["instructions"]  # the stock path
        else:
            prompt = {**(prompt or {}), "ocr_block": block}  # sandwich: the question in front of the images stays bare
        resp, raw = self.inner(state, {key: q}, images=images, prompt=prompt)
        raw = dict(raw)
        raw["ocr_chars"] = len(block)
''')
W = rep(W, 'eng = OCRWrap(QVSoupEngine(model=model_dir, ckpts=ckpts, device="cuda:0", perm=int(perm), image_root=root, label=label))',
        'eng = OCRWrap(QVSoupEngine(model=model_dir, ckpts=ckpts, device="cuda:0", perm=int(perm), image_root=root, label=label, layout=LAYOUT))')
W = rep(W, '{"label": label, "ckpts": ckpts, "perm": perm, "ocr": cache_f}', '{"label": label, "ckpts": ckpts, "perm": perm, "ocr": cache_f, "layout": LAYOUT}')
W = rep(W, 'print(json.dumps({"final": final, "wall_s": round(time.time() - t, 1)}))', 'print(json.dumps({"final": final, "wall_s": round(time.time() - t, 1), "layout": LAYOUT, "merger": eng.inner.merger_file}))')
(out / "soup_eval_ocr_vl.py").write_text(W)
for f in ("qv_train_vl.py", "qv_soup_engine_vl.py", "soup_eval_ocr_vl.py"):
    compile((out / f).read_text(), f, "exec")
    print(f, md5((out / f).read_text()))
