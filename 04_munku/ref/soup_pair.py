#!/usr/bin/env python3
"""Pair soup (TB1 fork, 2026-11-12): OUT = a*A + (1-a)*B over every tensor, fp32 math, A's dtype / shard layout / index / non-weight files.
The same arithmetic and checks as soup75_build.sh's embedded step, for two arbitrary exports with identical tensor names and shapes.
  soup_pair.py A_DIR B_DIR OUT_DIR a"""
import json, os, shutil, sys, multiprocessing as mp
x1, x0, xs, A = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); os.makedirs(xs, exist_ok=True)
i1 = json.load(open(f"{x1}/model.safetensors.index.json"))["weight_map"]; i0 = json.load(open(f"{x0}/model.safetensors.index.json"))["weight_map"]
assert set(i1) == set(i0), f"STRUCT key sets differ: only_A {sorted(set(i1) - set(i0))[:5]} only_B {sorted(set(i0) - set(i1))[:5]}"
shards = sorted(set(i1.values()))

def work(sh):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    out, st = {}, {"n": 0, "equal": 0, "rel": []}
    keys = sorted(k for k, v in i1.items() if v == sh); opened = {}
    with safe_open(f"{x1}/{sh}", "pt") as a:
        meta = a.metadata()
        for k in keys:
            f0 = i0[k]
            if f0 not in opened: opened[f0] = safe_open(f"{x0}/{f0}", "pt").__enter__()
            t1, t0 = a.get_tensor(k), opened[f0].get_tensor(k)
            assert t1.shape == t0.shape, f"STRUCT {k} shape {t1.shape} vs {t0.shape}"
            st["n"] += 1
            if t1.dtype == t0.dtype and torch.equal(t1, t0):
                st["equal"] += 1; out[k] = t1.clone(); continue
            if not (t1.dtype.is_floating_point and t0.dtype.is_floating_point):
                raise AssertionError(f"STRUCT non-float tensor differs: {k} {t1.dtype}/{t0.dtype}")
            g1, g0 = t1.float(), t0.float()
            out[k] = (A * g1 + (1.0 - A) * g0).to(t1.dtype)
            n = g1.norm().item(); st["rel"].append((g1 - g0).norm().item() / n if n > 0 else 0.0)
    save_file(out, f"{xs}/{sh}", metadata=meta)
    rel = st.pop("rel"); st["rel_mean"] = sum(rel) / len(rel) if rel else 0.0; st["rel_max"] = max(rel) if rel else 0.0
    return sh, st

with mp.get_context("fork").Pool(len(shards)) as pool:
    res = dict(pool.map(work, shards))
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json", "model.safetensors.index.json"):
    shutil.copy(f"{x1}/{f}", f"{xs}/{f}")
tot = {"alpha_A": A, "tensors": sum(v["n"] for v in res.values()), "identical": sum(v["equal"] for v in res.values()),
       "rel_A_vs_B_mean": sum(v["rel_mean"] for v in res.values()) / len(res), "rel_A_vs_B_max": max(v["rel_max"] for v in res.values())}
print("PAIR_STATS", json.dumps(tot))
