"""VL round: layout check on real rows, CPU only (processor + tokenizer + images; no model weights). Fails closed.

For every sampled row, with the STOCK trainer's own encode() (code/qv_train.py) as the reference:
  off   qv_train_vl.encode with --layout images_first gives the same tensors (input_ids, attention_mask, pixel_values, grid, ...) and
        qv_layout.render gives the same prompt string as qv_common.render: the old layout is byte-identical with the switch off.
  on    with --layout sandwich: the chat header is unchanged; everything from the first image to the last prompt token is the
        images_first prompt token for token (images, OCR block, question, options, tail, generation prompt, readout: the answer slot, i.e.
        the last prompt token the head reads, is unchanged); the tokens added in front of the first image decode to
        pre_text(question, options); the pixel tensors are identical; n_sandwich - added == n_images_first (the --max-tokens rule).
Row sources: train:<manifest>:<img_root>:<ocr_cache|->   (trainer rows)   kit:<rows>:<img_root>:<ocr_cache|->   (dev / gate rows, the
in-trainer read's and the eval engine's rendering). Types: one image, several images, readout, options in the image, each with and without
OCR text; --n rows per (source, type, ocr) cell in file order (--n 0 = every row). A no-image request is checked on a synthetic row.
usage: vl_layout_check.py <model_dir> <out.json> <source> [<source> ...] [--n 6] [--kit <kit_vis dir>] [--stock <code dir>]
Last line = JSON summary; exit 0 only if every check passed on every row."""
import collections
import json
import os
import re
import sys
from pathlib import Path

args = sys.argv[1:]
opt = {"--n": "6", "--kit": None, "--stock": str(Path(__file__).resolve().parent.parent / "code")}
for k in list(opt):
    if k in args:
        i = args.index(k)
        opt[k] = args[i + 1]
        del args[i:i + 2]
model_dir, out_path, sources = args[0], args[1], args[2:]
N = int(opt["--n"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, opt["--stock"])
if opt["--kit"]:
    sys.path.insert(2, opt["--kit"])
import torch  # noqa: E402
import qv_common as C  # noqa: E402
import qv_ocr as O  # noqa: E402
import qv_layout as LY  # noqa: E402
import qv_train as ST  # noqa: E402  (the stock trainer, imported for its encode())
import qv_train_vl as VT  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

assert Path(ST.__file__).resolve().parent == Path(opt["--stock"]).resolve() and Path(C.__file__).resolve().parent == Path(opt["--stock"]).resolve()
proc = AutoProcessor.from_pretrained(model_dir)
C.set_pixel_cap(proc)
tok = proc.tokenizer
H = LY.header_len(proc)
VS, VE, PAD = (tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>"))
CAP = 10 ** 9
show = lambda t: re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", t)  # noqa: E731


def same(a, b):
    return set(a.keys()) == set(b.keys()) and all(torch.equal(a[k], b[k]) for k in a.keys())


def typ(n, pr, block):
    t = "noimg" if n == 0 else ("one" if n == 1 else "multi")
    if pr.get("options_in_image"):
        t += "+oii"
    if pr.get("readout"):
        t += "+readout"
    return t + ("+ocr" if block else "")


def check(kind, row, root, ocr):
    """-> (type, {check: bool}, example)"""
    if kind == "train":
        q, opts, images = row["question"], row["options"], row["images"]
    else:
        qq = row["questions"]["answer"]
        q, opts, images = qq["instructions"], [qq["criteria"][k] for k in qq["criteria"]], row["images"]
    pr = row.get("prompt") or {}
    oii, ro = pr.get("options_in_image", False), pr.get("readout")
    block = O.ocr_prefix(ocr, images)
    r = {}
    t_old = C.render(proc, block + q, opts, len(images), oii, ro)  # the stock expression (qv_train.encode / kit_eval / soup_eval_ocr + engine)
    t_off = LY.render(proc, q, opts, len(images), oii, ro, block, "images_first")
    t_new = LY.render(proc, q, opts, len(images), oii, ro, block, "sandwich")
    r["off_prompt_string_identical"] = t_old == t_off
    if kind == "train":
        e_old, n_old = ST.encode(proc, row, root, CAP, ocr)
        VT.VL.update(layout="images_first", hlen=0)
        e_off, n_off = VT.encode(proc, row, root, CAP, ocr)
        VT.VL.update(layout="sandwich", hlen=H)
        e_new, n_new = VT.encode(proc, row, root, CAP, ocr)
        VT.VL.update(layout="images_first", hlen=0)
    else:
        from decision_index.vision import images as kimg

        pils = [kimg.pil(p, root, C.MAX_PIXELS).convert("RGB") for p in images]
        e_old = proc(text=[t_old], images=pils if pils else None, return_tensors="pt")
        e_off = proc(text=[t_off], images=pils if pils else None, return_tensors="pt")
        e_new = proc(text=[t_new], images=pils if pils else None, return_tensors="pt")
        n_old, n_off, n_new = (int(e["input_ids"].shape[1]) for e in (e_old, e_off, e_new))
    r["off_tensors_identical"] = same(e_old, e_off) and n_old == n_off
    io, inw = e_old["input_ids"][0], e_new["input_ids"][0]
    if len(images) == 0:
        r["noimg_unchanged"] = same(e_old, e_new) and t_old == t_new
        added = 0
    else:
        fo, fn = LY.first_vision_start(io, proc), LY.first_vision_start(inw, proc)
        added = fn - H
        pre = LY.pre_text(q, opts, oii)
        r["header_unchanged"] = fo == H and torch.equal(io[:H], inw[:H])
        r["from_first_image_identical"] = torch.equal(io[fo:], inw[fn:])  # images + OCR block + question + options + tail + answer slot
        r["answer_slot_last_token_same"] = int(io[-1]) == int(inw[-1]) and (len(inw) - len(io)) == added
        r["added_tokens_decode_to_pre_text"] = tok.decode(inw[H:fn].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False) == pre
        r["added_has_no_image_or_special"] = not bool(((inw[H:fn] == VS) | (inw[H:fn] == PAD) | (inw[H:fn] == VE)).any())
        r["image_tokens_same"] = int((io == PAD).sum()) == int((inw == PAD).sum()) and int((io == VS).sum()) == len(images) == int((inw == VS).sum())
        r["pixels_identical"] = all(torch.equal(e_old[k], e_new[k]) for k in e_old.keys() if k not in ("input_ids", "attention_mask", "mm_token_type_ids"))
        r["cap_rule_images_first_length"] = n_new - LY.added_tokens(inw, proc, H, "sandwich", len(images)) == n_old
        r["new_prompt_string"] = t_new == t_old.replace("<|vision_start|>", pre + "<|vision_start|>", 1)
        r["ocr_block_after_images_only"] = (O.HEADER not in t_new.split("<|vision_start|>")[0]) and (not block or t_new.split("<|vision_end|>")[-1].startswith(block))
    ex = {"id": row.get("id"), "n_images": len(images), "tokens_images_first": n_old, "tokens_sandwich": n_new, "added": added,
          "images_first": show(t_old)[:900], "sandwich": show(t_new)[:1400], "last_tokens": [tok.decode([int(x)]) for x in inw[-6:]]}
    return typ(len(images), pr, block), r, ex


res = {"header_tokens": H, "cells": {}, "examples": {}, "failed": []}
add_stats = collections.defaultdict(list)
for src in sources:
    kind, rows_f, root, ocr_f = src.split(":")
    ocr = O.load_ocr(ocr_f)[0] if ocr_f != "-" else None
    seen = collections.Counter()
    for l in open(rows_f):
        if not l.strip():
            continue
        row = json.loads(l)
        pr = row.get("prompt") or {}
        cell = f"{kind}:{typ(len(row['images']), pr, O.ocr_prefix(ocr, row['images']))}"
        if N and seen[cell] >= N:
            continue
        seen[cell] += 1
        t, r, ex = check(kind, row, root, ocr)
        c = res["cells"].setdefault(cell, {"rows": 0, "ok": 0})
        c["rows"] += 1
        ok = all(r.values())
        c["ok"] += ok
        add_stats[cell].append(ex["added"])
        if not ok:
            res["failed"].append({"cell": cell, "id": row.get("id"), "failed": [k for k, v in r.items() if not v]})
        if cell not in res["examples"]:
            res["examples"][cell] = {**ex, "checks": r}
# a request without images (none exists in the training / dev / public rows): rendered as images_first
t, r, ex = check("train", {"id": "synthetic_noimg", "images": [], "question": "Which is larger?", "options": ["2", "3"], "prompt": {}}, ".", None)
res["cells"]["synthetic:noimg"] = {"rows": 1, "ok": int(all(r.values()))}
res["examples"]["synthetic:noimg"] = {**ex, "checks": r}
if not all(r.values()):
    res["failed"].append({"cell": "synthetic:noimg", "failed": [k for k, v in r.items() if not v]})
res["added_tokens"] = {k: {"median": sorted(v)[len(v) // 2], "max": max(v)} for k, v in add_stats.items()}
res["rows"] = sum(c["rows"] for c in res["cells"].values())
res["ok"] = not res["failed"] and res["rows"] > 1
res["failed"] = res["failed"][:40]
Path(out_path).write_text(json.dumps(res, indent=1))
print(json.dumps({"ok": res["ok"], "rows": res["rows"], "header_tokens": H, "cells": {k: f"{v['ok']}/{v['rows']}" for k, v in res["cells"].items()},
                  "added_tokens": res["added_tokens"], "failed": res["failed"][:3]}))
sys.exit(0 if res["ok"] else 1)
