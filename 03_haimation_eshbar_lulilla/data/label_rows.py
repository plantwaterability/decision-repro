#!/usr/bin/env python3
"""The (tied) embedding rows of EVERY jev label, in label order, from an exported checkpoint: the rows AuxVBLL's letter codes are a
fixed function of (c_j = rms(Q rms(e_j))). Feature files store only the first BASIS_LABELS rows (E_letters, the head's reference
basis); a head built from those has codes for 52 labels and cannot score a K > 52 row. Label ids come from the feature file's meta,
and the first 52 rows must equal its E_letters bit for bit.

  python3 label_rows.py --model EXPORT_DIR --features FEATS.pt --out LABEL_ROWS.pt
"""
import argparse
import json
import os

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = torch.load(a.features, map_location="cpu", weights_only=False)
    ids = (f.get("meta") or {}).get("label_ids")
    if not ids:
        raise SystemExit(f"{a.features}: meta has no label_ids")
    wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in wm if k.endswith("embed_tokens.weight") and "language_model" in k] or [k for k in wm if k == "model.embed_tokens.weight"]
    if len(keys) != 1 or any(k.startswith("lm_head") for k in wm):
        raise SystemExit(f"expected one text embed_tokens tensor and a tied head; got {keys}, lm_head present: {any(k.startswith('lm_head') for k in wm)}")
    with safe_open(os.path.join(a.model, wm[keys[0]]), framework="pt") as sf:
        E = sf.get_tensor(keys[0]).index_select(0, torch.tensor(ids)).float()
    E0 = f["E_letters"].float()
    if not torch.equal(E[:E0.shape[0]], E0):
        raise SystemExit(f"first {E0.shape[0]} rows != the features' E_letters (max |diff| {float((E[:E0.shape[0]] - E0).abs().max()):.3g}): "
                         "wrong checkpoint for these features")
    torch.save({"E": E, "label_ids": [int(x) for x in ids], "model": os.path.abspath(a.model), "tensor": keys[0]}, a.out)
    print(json.dumps({"rows": E.shape[0], "hidden": E.shape[1], "basis_rows_match": E0.shape[0], "tensor": keys[0]}), flush=True)


if __name__ == "__main__":
    main()
