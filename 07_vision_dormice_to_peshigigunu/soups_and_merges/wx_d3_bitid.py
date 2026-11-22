"""WX: trainer encode (patched vx_train_vl.py, layout d3) vs wx_engine.py's encode, tensor for tensor, on sample rows.
mode dump-trainer: python wx_d3_bitid.py trainer <out.pt> ; mode dump-engine: python wx_d3_bitid.py engine <out.pt> ; mode compare: ... compare a.pt b.pt
Rows: 6 cells x up to 25 training rows (manifest row for the trainer, its kit twin for the engine, train OCR cache, ug1 image root)
+ 50 public kit rows through the trainer's kit_eval path (public OCR cache, public image root). env WX_ROOT, WX_CODE, WX_KIT."""
import hashlib, json, os, sys
R = os.environ.get("WX_ROOT", "/data/local/wx"); CODE = os.environ.get("WX_CODE", f"{R}/code"); KIT = os.environ.get("WX_KIT", f"{R}/kit/kit_vis")
sys.path.insert(0, KIT); sys.path.insert(0, CODE)
mode = sys.argv[1]
import torch  # noqa: E402

if mode == "compare":
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])  # our own dumps carry torch.__version__
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    bad, n = [], 0
    for k in a["encs"]:
        ea, eb = a["encs"][k], b["encs"].get(k)
        n += 1
        if eb is None or set(ea) != set(eb) or any(not torch.equal(ea[t], eb[t]) for t in ea):
            bad.append((k, sorted(ea), sorted(eb or {}), [t for t in ea if eb and t in eb and not torch.equal(ea[t], eb[t])]))
    print(json.dumps({"rows": n, "identical": n - len(bad), "versions": [a["versions"], b["versions"]], "cells": a["cells"], "bad": bad[:5]}))
    sys.exit(0 if not bad else 1)

import transformers  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
import qv_ocr as O  # noqa: E402

STOCK = f"{R}/dl/qwen38_1d4bf0f2"
kit_rows = {}
for l in open(f"{R}/teacher/ug1_nc_kit.jsonl"):
    r = json.loads(l); kit_rows[r["id"]] = r
man = [json.loads(l) for l in open(f"{R}/data/ug1_nc.jsonl") if l.strip()]
ocr_train = O.load_ocr(f"{R}/ocr/ocr_train_ug1.jsonl")[0]
ocr_pub = O.load_ocr(f"{R}/ocr/ocr_public.jsonl")[0]
cells = {"one_image": [], "several_images": [], "options_in_image": [], "with_ocr_text": [], "no_ocr_text": [], "readout": []}
for r in sorted(man, key=lambda r: hashlib.sha1(("bitid" + r["id"]).encode()).hexdigest()):
    pr = r.get("prompt") or {}
    has = bool(O.ocr_prefix(ocr_train, r["images"]))
    for c, ok in (("one_image", len(r["images"]) == 1), ("several_images", len(r["images"]) > 1), ("options_in_image", pr.get("options_in_image")),
                  ("with_ocr_text", has), ("no_ocr_text", not has), ("readout", bool(pr.get("readout")))):
        if ok and len(cells[c]) < 25:
            cells[c].append(r["id"])
    if all(len(v) >= 25 for v in cells.values()):
        break
ids = sorted({i for v in cells.values() for i in v})
pub = [json.loads(l) for l in open(f"{R}/public/suite/vision-rows-noWG.jsonl") if l.strip()]
pub = sorted(pub, key=lambda r: hashlib.sha1(("bitid" + r["id"]).encode()).hexdigest())[:50]
encs = {}


def cpu(e):
    return {k: v.detach().cpu() for k, v in e.items()}


if mode == "trainer":
    import vx_train_vl as T
    T.VL["layout"] = "d3"
    import wx_engine as WXE
    proc = AutoProcessor.from_pretrained(STOCK)
    proc.tokenizer.padding_side = "left"
    proc.image_processor.size = {"shortest_edge": WXE.D3_MIN_PIXELS, "longest_edge": WXE.D3_MAX_PIXELS}
    mrow = {r["id"]: r for r in man}
    for i in ids:
        enc, n = T.encode(proc, mrow[i], f"{R}/data/img", 10 ** 9, ocr_train)
        encs["train:" + i] = cpu(enc)
    for r in pub:
        q = r["questions"]["answer"]; keys = list(q["criteria"])
        encs["pub:" + r["id"]] = cpu(T.d3_encode(proc, q["instructions"], [q["criteria"][k] for k in keys], r["images"], f"{R}/public/suite", ocr_pub, r.get("state", {})))
else:
    torch.cuda.synchronize = lambda *a, **k: None  # the engine syncs after its forward; this dump runs no forward and no GPU
    from wx_engine import WXEngine
    from decision_index.vision.engines import MAX_PIXELS
    E = object.__new__(WXEngine)
    E.torch, E.device, E.max_pixels, E.family, E.temperature, E.state_mode = torch, "cpu", MAX_PIXELS, "d3", 1.0, "ocr"
    E.proc = AutoProcessor.from_pretrained(STOCK)
    E.proc.tokenizer.padding_side = "left"
    E.proc.image_processor.size = {"shortest_edge": 589824, "longest_edge": 1_638_400}
    E.codes = json.load(open(f"{R}/dl/wemm_1543fbfc/decision_config.json"))["codes"]
    E.R = torch.zeros(255, 5120)
    cap = {}

    def fake_hidden(enc):
        cap["enc"] = enc
        return torch.zeros(5120)
    E._hidden = fake_hidden
    for i in ids:
        r = kit_rows[i]; E.image_root, E.ocr_text = f"{R}/data/img", ocr_train
        E(r["state"], r["questions"], r["images"], r.get("prompt"))
        encs["train:" + i] = cpu(cap["enc"])
    for r in pub:
        E.image_root, E.ocr_text = f"{R}/public/suite", ocr_pub
        E(r["state"], r["questions"], r["images"], r.get("prompt"))
        encs["pub:" + r["id"]] = cpu(cap["enc"])
torch.save({"encs": encs, "versions": {"transformers": str(transformers.__version__), "torch": str(torch.__version__)}, "cells": {k: len(v) for k, v in cells.items()}}, sys.argv[2])
print(mode, len(encs), transformers.__version__)
