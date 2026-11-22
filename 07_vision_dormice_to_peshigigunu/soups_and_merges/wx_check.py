"""WX step 2 checks (weights only; safetensors reads, no model class): (1) WeMM's tokenizer / template / processor files vs stock's;
(2) readout.safetensors rows vs stock Qwen3.8-27B lm_head rows of decision_config's token ids; (3) tau_W = WeMM - stock per tensor group
(relative norm, identical tensors); (4) tau_P (peshashaga's update) norms and its cosine with tau_W on the 496 matrices.
usage: python wx_check.py <wemm_dir> <stock_dir> <pesh_dir> <out.json>"""
import collections, hashlib, json, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file

wd, sd, pd, outf = sys.argv[1:5]
dev = "cuda:0"
out = {}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
files = ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json"]
out["files_equal_to_stock"] = {f: sha(f"{wd}/{f}") == sha(f"{sd}/{f}") for f in files}
wc, sc = json.load(open(f"{wd}/config.json")), json.load(open(f"{sd}/config.json"))
out["text_config_diff"] = {k: (wc["text_config"].get(k), sc["text_config"].get(k)) for k in set(wc["text_config"]) | set(sc["text_config"]) if wc["text_config"].get(k) != sc["text_config"].get(k)}
out["vision_config_diff"] = {k: (wc["vision_config"].get(k), sc["vision_config"].get(k)) for k in set(wc["vision_config"]) | set(sc["vision_config"]) if wc["vision_config"].get(k) != sc["vision_config"].get(k)}
dc = json.load(open(f"{wd}/decision_config.json"))
widx = json.load(open(f"{wd}/model.safetensors.index.json"))["weight_map"]
sidx = json.load(open(f"{sd}/model.safetensors.index.json"))["weight_map"]
out["stock_key_prefixes"] = dict(collections.Counter(k.split(".")[0] + "." + k.split(".")[1] for k in sidx))
H = {}


def get(d, idx, k):
    f = idx[k]
    if (d, f) not in H:
        H[(d, f)] = safe_open(f"{d}/{f}", "pt", device="cpu")
    return H[(d, f)].get_tensor(k)


# (2) readout vs stock lm_head rows
R = load_file(f"{wd}/readout.safetensors")
out["readout_keys"] = {k: list(v.shape) + [str(v.dtype)] for k, v in R.items()}
Wr = R["weight"].to(dev).float()
lk = [k for k in sidx if k.endswith("lm_head.weight")][0]
L = get(sd, sidx, lk).to(dev)
rows = L[torch.tensor(dc["token_ids"], device=dev)]
d = (Wr - rows.float()).abs()
out["readout_vs_stock_lmhead"] = {"lm_head_key": lk, "rows": int(Wr.shape[0]), "max_abs": float(d.max()), "max_abs_first26": float(d[:26].max()),
                                  "rel_norm_median": float(((Wr - rows.float()).norm(dim=1) / rows.float().norm(dim=1)).median()),
                                  "bf16_equal_rows": int((Wr.to(torch.bfloat16) == rows).all(dim=1).sum()),
                                  "exact_fp32_of_bf16_rows": int((Wr == rows.float()).all(dim=1).sum())}
emb_k = [k for k in sidx if k.endswith("embed_tokens.weight")][0]
E = get(sd, sidx, emb_k).to(dev)
out["readout_vs_stock_embed_rows_max_abs"] = float((Wr - E[torch.tensor(dc["token_ids"], device=dev)].float()).abs().max())
del L, E, rows
# (3) tau_W per group
grp = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])  # sum ||d||^2, sum ||s||^2, n, identical
missing, shape_bad = [], []
tw = {}
for k in widx:
    skey = "model." + k
    if skey not in sidx:
        missing.append(k); continue
    w = get(wd, widx, k).to(dev); s = get(sd, sidx, skey).to(dev)
    if w.shape != s.shape:
        shape_bad.append(k); continue
    dd_ = (w.float() - s.float())
    parts = k.split(".")
    g = ("visual." + parts[1]) if parts[0] == "visual" else (("lm.layers." + parts[3]) if parts[1] == "layers" else ("lm." + parts[1]))
    e = grp[g]; e[0] += float((dd_ ** 2).sum()); e[1] += float((s.float() ** 2).sum()); e[2] += 1; e[3] += int(torch.equal(w, s))
    if k.startswith("language_model.layers") and w.dim() == 2:
        tw[k] = dd_.to("cpu", torch.bfloat16)
    del w, s, dd_
out["tau_W_missing_in_stock"] = missing[:5]; out["tau_W_shape_mismatch"] = shape_bad[:5]
out["tau_W_groups"] = {g: {"rel_norm": round((v[0] / max(v[1], 1e-30)) ** 0.5, 5), "tensors": v[2], "identical": v[3]} for g, v in sorted(grp.items())}
tot = [sum(v[i] for v in grp.values()) for i in range(4)]
out["tau_W_total"] = {"rel_norm": round((tot[0] / tot[1]) ** 0.5, 5), "tensors": tot[2], "identical": tot[3], "stock_tensors_not_in_wemm": [k for k in sidx if k.startswith("model.") and k[6:] not in widx][:5],
                      "stock_extra_prefixes": dict(collections.Counter(k.split(".")[0] for k in sidx if not (k.startswith("model.") and k[6:] in widx)))}
# (4) tau_P and cos(tau_W, tau_P)
cfg = json.load(open(f"{pd}/peshashaga.json"))
sc_ = [float(x) for x in cfg["update"]["scales"]]
U = load_file(f"{pd}/{cfg['update']['file']}")
dot = nW = nP = nS = 0.0; n = 0; nomatch = []
for k in U:
    if not k.endswith(".A"):
        continue
    path = k[:-2]; wk = path[len("model."):] + ".weight"
    A, B = U[k].to(dev).float(), U[path + ".B"].to(dev).float()
    tp = sum(sc_[j] * (B[j] @ A[j]) for j in range(len(sc_)))
    s = get(sd, sidx, "model." + wk).to(dev).float()
    nP += float((tp ** 2).sum()); nS += float((s ** 2).sum())
    if wk in tw:
        t = tw[wk].to(dev).float(); dot += float((t * tp).sum()); nW += float((t ** 2).sum())
    else:
        nomatch.append(wk)
    n += 1
out["tau_P"] = {"matrices": n, "rel_norm_on_its_matrices": round((nP / nS) ** 0.5, 5), "cos_with_tau_W": round(dot / max((nW * nP) ** 0.5, 1e-30), 5),
                "tau_W_rel_norm_on_same_matrices": round((nW / nS) ** 0.5, 5), "no_tau_W_match": nomatch[:3]}
json.dump(out, open(outf, "w"), indent=1)
print(json.dumps(out)[:6000])
