#!/usr/bin/env python3
"""(FULLFT) PARENT-PREDICTION EXTRACTION for preservation targets: score the EXACT rendered prompts of a render manifest
(fullft_render.py) with stock Gemma-4-31B-it or Garas / (99), and write a cache that preservation.py joins.

  score   --rows UNITS.jsonl --route stock|garas --parents PARENTS.json --model-dir SNAP [--adapter 99.pt] --out CACHE.jsonl
          [--shard I/N] [--batch exact|pack] [--attn jev_mem|sdpa]
  merge   --caches A.jsonl B.jsonl ... --out CACHE.jsonl            (shards / routes -> one cache; duplicate identities refused)
  parity  --data DIR --rows UNITS.jsonl --route ... --scorer-npz X.npz --out PARITY.json    (vs the pinned scorer's own output)
          [--save-rows OURS.npz] [--compute bf16|fp32]   (v2: per-unit letter log-probs; fp32 = the bf16-stored model computed in fp32)
  refscore --rows ROWS.jsonl --route stock|garas --parents P.json --model-dir SNAP [--adapter 99.pt] --out REF.jsonl
          (v3) a REFERENCE model's native letter probabilities on EVERY row of a render manifest (DEV), keyed by decision_uid +
          prompt_sha256 (no parent identity, no preservation contract); the same load / forward / batching as score
  compare --runs NAME=FILE.npz ... --pairs A:B,... --out CMP.json   (v2: TV / max|dp| / |dlogp| / KL / flips between any two runs,
          pinned-scorer npz or ours, joined by test_uid; by prompt length)

EXACT EQUIVALENT of the pinned production scorer (claude_lmscore.py e0672bd0, --exact-path): the same code loads the model
(model_loader, bf16, from_pretrained at the pinned commit) and, for Garas, the same adapter path (train_e2e.apply_lora with the
checkpoint's r / targets / scope, strict key match: claude_lmscore's own block); the same letter tokens; the same readout: the full
model forward with logits_to_keep=1, the model's own bf16 head, the first K letter logits, softmax. Differences, all disclosed in the
cache's `scored_by`: (1) the prompts are the manifest's (each one re-hashed against prompt_sha256 and re-tokenised against n_tokens)
instead of being re-drawn by plan_prompts; (2) attention runs through fullft_mem's jev_mem (memory-efficient kernel, no [L, L] scores)
so 32k prompts fit; the pinned path uses SDPA's math kernel on the 10 full-attention layers, which cannot run 32k prompts on one
H100 (128 GiB of scores); (3) --batch pack (optional, faster on short prompts) packs prompts into one row with per-prompt positions
(gemmafast's --exact-batch pack, equal in exact arithmetic). `parity` measures (1)+(2) against the pinned scorer itself on a sample:
prompt identity by sha1 (its test_input_sha) and letter log-probability differences (row_logp of --save-rows).

GOLD STAYS INDEPENDENT: the cache holds the parent's probabilities and identity only; gold is never read or written here, and
preservation.py's join keeps the unit's gold untouched (gold_overwritten false).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fullft_mem as fm  # noqa: E402
import fullft_render as fr  # noqa: E402

VERSION = "fullft_parent v3 (2026-10-13)"
PRESERVATION_MD5 = None            # set by --preservation-md5 (the data package's file); checked when given
IDENT = ("decision_uid", "parent_id", "parent_revision", "renderer_sha256", "prompt_sha256", "option_order")


def load_preservation(path, want_md5=None):
    m5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if want_md5 and m5 != want_md5:
        raise SystemExit(f"{path}: md5 {m5} != {want_md5}")
    spec = importlib.util.spec_from_file_location("preservation_pinned", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, m5


def shard_of(key, n):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n


def load_parent(a, spec, ls):
    """(model, tok, label_ids, provenance) exactly as claude_lmscore.py main() loads them (plus the jev_mem attention)."""
    from transformers import AutoConfig, AutoTokenizer
    mdir = a.model_dir
    commit = spec["base_revision"]
    if not os.path.isdir(mdir):
        raise SystemExit(f"--model-dir {mdir}: not a directory")
    ref = os.path.join(os.path.dirname(os.path.dirname(mdir.rstrip("/"))), "refs", "main")
    snap = os.path.basename(mdir.rstrip("/"))
    if snap != commit and not a.allow_unverified_snapshot:
        raise SystemExit(f"--model-dir is snapshot {snap}, the parent spec pins {commit}")
    mcfg = AutoConfig.from_pretrained(mdir)
    _, model_cls = ls.model_loader(mcfg, "auto")
    tok = AutoTokenizer.from_pretrained(mdir)
    tok.padding_side = "right"
    ids = [tok(c, add_special_tokens=False)["input_ids"] for c in ls.LABELS]
    if any(len(x) != 1 for x in ids):
        raise SystemExit("a label is not a single token")
    label_ids = torch.tensor([x[0] for x in ids])
    kw = {}
    if a.attn == fm.ATTN_NAME:
        fm.register()
        kw["attn_implementation"] = fm.ATTN_NAME
    dev = a.device
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    try:                                              # claude_lmscore's own load (device_map needs accelerate)
        model = model_cls.from_pretrained(mdir, dtype=dt, device_map=dev, **kw).eval()
    except ModuleNotFoundError as e:                  # a CPU test image without accelerate: the same weights, moved after loading
        if "accelerate" not in str(e):
            raise
        model = model_cls.from_pretrained(mdir, dtype=dt, **kw).to(dev).eval()
    prov = {"model_dir_snapshot": snap, "refs_main": open(ref).read().strip() if os.path.exists(ref) else None, "attn": a.attn,
            "batch": a.batch, "dtype": "bfloat16" if dev == "cuda" else "float32", "torch": torch.__version__}
    if spec.get("adapter_md5"):
        if not a.adapter:
            raise SystemExit("this parent needs --adapter (its LoRA checkpoint)")
        m5 = hashlib.md5(open(a.adapter, "rb").read()).hexdigest()
        if m5 != spec["adapter_md5"]:
            raise SystemExit(f"--adapter md5 {m5} != the parent spec's {spec['adapter_md5']}")
        tspec = importlib.util.spec_from_file_location("te_pinned", os.path.join(os.path.dirname(ls.__file__), "train_e2e.py"))
        te = importlib.util.module_from_spec(tspec)
        tspec.loader.exec_module(te)
        ck = torch.load(a.adapter, map_location="cpu")
        if bool(ck["config"].get("plain_prompt", False)):
            raise SystemExit("adapter trained with plain prompts; the manifest renders chat prompts")
        scope = ck["config"].get("lora_scope")
        te.apply_lora(model, int(ck["config"]["lora_r"]), targets=tuple(ck["config"]["lora_targets"]), **({"scope": scope} if scope else {}))
        want = {n for n, _ in model.named_parameters() if n.endswith(".A") or n.endswith(".B")}
        if set(ck["lora"]) != want:
            raise SystemExit(f"LoRA key mismatch: {len(set(ck['lora']) ^ want)} keys differ")
        model.load_state_dict(ck["lora"], strict=False)
        model.eval()
        prov["adapter_md5"] = m5
        prov["adapter_tensors"] = len(want)
    return model, tok, label_ids, prov


def text_model_of(model):
    """the Gemma-4 text stack inside a loaded model (the module holding embed_tokens, layers and norm)."""
    for _, m in model.named_modules():
        if hasattr(m, "embed_tokens") and hasattr(m, "layers") and hasattr(m, "norm") and "Text" in type(m).__name__:
            return m
    raise SystemExit("no Gemma-4 text model inside the loaded model")


class Fp32Compute:
    """(v2) the bf16-STORED model computed in fp32 -- research/gemma_fast.md's ground truth: each top-level text module (the
    embedding, every decoder layer, the final norm, the tied head) has its bf16 parameters / buffers cast to fp32 just before its
    forward and back after it (bf16 -> fp32 -> bf16 is exact), so the hidden states, every Linear, norm, attention and the softcap run
    in fp32 while only one module at a time holds an fp32 copy. TF32 is off while active."""

    def __init__(self, model):
        lm = text_model_of(model)
        self.mods = [lm.embed_tokens, *lm.layers, lm.norm, model.get_output_embeddings()]
        self.handles = []

    @staticmethod
    def _up(m, args):
        m._jev_cast = []
        for t in list(m.parameters()) + list(m.buffers()):
            if t.dtype == torch.bfloat16:
                t.data = t.data.float()
                m._jev_cast.append(t)
        return tuple(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in args)

    @staticmethod
    def _down(m, args, out):
        for t in getattr(m, "_jev_cast", []):
            t.data = t.data.to(torch.bfloat16)
        m._jev_cast = []

    def __enter__(self):
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        for m in self.mods:
            self.handles += [m.register_forward_pre_hook(self._up), m.register_forward_hook(self._down)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        return False


def forward_letters(model, tok, texts, label_ids, dev, batch):
    """letter logits [B, 52] fp32: 'exact' = equal-length rows, model(..., logits_to_keep=1) (the pinned --exact-path); 'pack' = one
    row, per-prompt positions, the model's head at every prompt's last token."""
    with torch.no_grad():
        if batch == "exact":
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            if not bool(enc["attention_mask"].bool().all()):
                raise RuntimeError("exact batches must hold equal-length prompts")
            z = model(input_ids=enc["input_ids"], use_cache=False, logits_to_keep=1).logits[:, -1].float()
            return z[:, label_ids.to(dev)]
        ids, pos, last = fm.packed_batch([tok(t, add_special_tokens=False)["input_ids"] for t in texts])
        z = model(input_ids=ids.to(dev), position_ids=pos.to(dev), use_cache=False, logits_to_keep=last.to(dev)).logits[0].float()
        return z[:, label_ids.to(dev)]


def batches(lens, mode, tokens_per_batch):
    """the scorer's grouping: length-sorted; exact = consecutive EQUAL-length prompts while rows x len <= budget; pack = consecutive
    prompts while the token sum <= budget. -> list of index lists."""
    order = sorted(range(len(lens)), key=lambda i: (lens[i], i))
    out, i = [], 0
    while i < len(order):
        j, tot = i, 0
        while j < len(order):
            L = lens[order[j]]
            if mode == "exact":
                if j > i and (L != lens[order[i]] or (j - i + 1) * L > tokens_per_batch):
                    break
            elif j > i and tot + L > tokens_per_batch:
                break
            tot += L
            j += 1
        out.append(order[i:max(j, i + 1)])
        i = max(j, i + 1)
    return out


def score(a):
    ls, md5s = fr.load_scorer(a.code)
    pres, pres_md5 = load_preservation(a.preservation, a.preservation_md5)
    parents = fr.parent_specs(a.parents)
    if a.route not in parents:
        raise SystemExit(f"PARENTS.json has no {a.route}")
    pid, prev, spec = parents[a.route]
    sc = spec.get("scorer") or {}
    for k, v in (("attn", a.attn), ("batch", a.batch)):
        if sc.get(k) is not None and sc[k] != v:
            raise SystemExit(f"--{k} {v}: the parent spec (part of parent_revision) says {sc[k]}")
    si, sn = (int(x) for x in a.shard.split("/"))
    rows, keys = {}, {}
    with open(a.rows, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("numeric_family") or r.get("parent_route") != a.route:
                continue
            if r["parent_id"] != pid or r["parent_revision"] != prev:
                raise SystemExit(f"{r['exposure_id']}: parent {r['parent_id']} / {r['parent_revision'][:12]} != this run's {pid} / {prev[:12]}")
            k = pres.identity(r)
            if shard_of(k, sn) != si:
                continue
            if k in rows:
                keys[k].append(r["exposure_id"])
                continue
            if fr.sha256_text(r["prompt"]) != r["prompt_sha256"]:
                raise SystemExit(f"{r['exposure_id']}: prompt sha256 mismatch")
            rows[k], keys[k] = r, [r["exposure_id"]]
    if a.limit:
        rows = dict(list(rows.items())[: a.limit])
    model, tok, label_ids, prov = load_parent(a, spec, ls)
    ks = list(rows)
    texts = [rows[k]["prompt"] for k in ks]
    lens = []
    for k, t in zip(ks, texts):
        n = len(tok(t, add_special_tokens=False)["input_ids"])
        if n != int(rows[k]["n_tokens"]):
            raise SystemExit(f"{rows[k]['exposure_id']}: {n} tokens, the manifest says {rows[k]['n_tokens']} (tokenizer differs)")
        lens.append(n)
    scored_by = {"version": VERSION, "equivalent_of": f"claude_lmscore.py {fr.PINNED_SCORER_MD5} --exact-path", **md5s, **prov,
                 "preservation_md5": pres_md5, "route": a.route, "parent_spec": spec}
    tmp = a.out + ".partial"
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists (immutable)")
    t0, done, ntok = time.time(), 0, 0
    with open(tmp, "w", encoding="utf-8") as g:
        for bi, idx in enumerate(batches(lens, a.batch, a.tokens_per_batch)):
            z = forward_letters(model, tok, [texts[i] for i in idx], label_ids, a.device, a.batch).cpu()
            for b, i in enumerate(idx):
                r = rows[ks[i]]
                order = r["option_order"]
                zk = z[b, :len(order)].double()
                p = torch.softmax(zk, -1).numpy()
                ent = {f: r[f] for f in IDENT}
                ent["probabilities"] = {order[j]: float(p[j]) for j in range(len(order))}
                ent["letter_logits"] = [float(x) for x in z[b, :len(order)]]
                ent["n_tokens"] = lens[i]
                ent["exposure_ids"] = keys[ks[i]]
                pres.probabilities(ent)                 # the contract's own validation
                if pres.identity(ent) != ks[i]:
                    raise SystemExit("identity changed while scoring")
                g.write(json.dumps(ent, ensure_ascii=False) + "\n")
                done += 1
                ntok += lens[i]
            if bi % 50 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(ks)} prompts  {ntok / max(el, 1e-9):.0f} tok/s  eta {(len(ks) - done) * el / max(done, 1) / 60:.1f} min",
                      flush=True)
    os.replace(tmp, a.out)
    el = time.time() - t0
    summ = {"route": a.route, "shard": a.shard, "prompts": done, "exposures": sum(len(v) for v in keys.values()), "tokens": ntok,
            "seconds": round(el, 1), "tokens_per_s": round(ntok / max(el, 1e-9), 1), "md5": hashlib.md5(open(a.out, "rb").read()).hexdigest(),
            "scored_by": scored_by}
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summ, f, indent=1, sort_keys=True)
    return summ


def refscore(a):
    """(v3) reference native probabilities on every row (DEV): {decision_uid, prompt_sha256, option_order, probabilities, letter_logits,
    n_tokens}; rows re-hashed and re-tokenised against the manifest; output immutable."""
    ls, md5s = fr.load_scorer(a.code)
    parents = fr.parent_specs(a.parents)
    if a.route not in parents:
        raise SystemExit(f"PARENTS.json has no {a.route}")
    pid, prev, spec = parents[a.route]
    rows = []
    with open(a.rows, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("manifest_header"):
                continue
            if fr.sha256_text(r["prompt"]) != r["prompt_sha256"]:
                raise SystemExit(f"{r.get('exposure_id')}: prompt sha256 mismatch")
            rows.append(r)
    if a.limit:
        rows = rows[: a.limit]
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists (immutable)")
    model, tok, label_ids, prov = load_parent(a, spec, ls)
    texts = [r["prompt"] for r in rows]
    lens = []
    for r, t in zip(rows, texts):
        n = len(tok(t, add_special_tokens=False)["input_ids"])
        if n != int(r["n_tokens"]):
            raise SystemExit(f"{r.get('exposure_id')}: {n} tokens, the manifest says {r['n_tokens']}")
        lens.append(n)
    t0, done, ntok = time.time(), 0, 0
    with open(a.out + ".partial", "w", encoding="utf-8") as g:
        for idx in batches(lens, a.batch, a.tokens_per_batch):
            z = forward_letters(model, tok, [texts[i] for i in idx], label_ids, a.device, a.batch).cpu()
            for b, i in enumerate(idx):
                r = rows[i]
                order = r["option_order"]
                zk = z[b, :len(order)].double()
                p = torch.softmax(zk, -1).numpy()
                g.write(json.dumps({"decision_uid": r["decision_uid"], "prompt_sha256": r["prompt_sha256"], "option_order": order,
                                    "probabilities": {order[j]: float(p[j]) for j in range(len(order))},
                                    "letter_logits": [float(x) for x in z[b, :len(order)]], "n_tokens": lens[i]},
                                   ensure_ascii=False) + "\n")
                done += 1
                ntok += lens[i]
    os.replace(a.out + ".partial", a.out)
    el = time.time() - t0
    summ = {"route": a.route, "rows": done, "tokens": ntok, "seconds": round(el, 1), "tokens_per_s": round(ntok / max(el, 1e-9), 1),
            "md5": hashlib.md5(open(a.out, "rb").read()).hexdigest(), "scored_by": {"version": VERSION, **md5s, **prov,
            "parent_spec": spec, "rows_file_sha256": fr.sha256_file(a.rows)}}
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summ, f, indent=1, sort_keys=True)
    return summ


def merge(a):
    pres, _ = load_preservation(a.preservation, a.preservation_md5)
    seen = set()
    n = 0
    with open(a.out + ".partial", "x", encoding="utf-8") as g:
        for c in a.caches:
            with open(c, encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    k = pres.identity(e)
                    pres.probabilities(e)
                    if k in seen:
                        raise SystemExit(f"duplicate cache identity in {c}: {e['decision_uid']}")
                    seen.add(k)
                    g.write(line if line.endswith("\n") else line + "\n")
                    n += 1
    os.replace(a.out + ".partial", a.out)
    return {"entries": n, "md5": hashlib.md5(open(a.out, "rb").read()).hexdigest()}


def parity(a):
    """our path vs the pinned scorer's own npz (claude_lmscore.py --n-perm 1 --exact-path --save-rows on --data / test split): the
    identity-order prompt text (sha1 == test_input_sha) and |our letter log-probs - row_logp|."""
    ls, md5s = fr.load_scorer(a.code)
    parents = fr.parent_specs(a.parents)
    pid, prev, spec = parents[a.route]
    z = np.load(a.scorer_npz, allow_pickle=False)
    units = list(ls.cf.iter_units(a.data, ["test"], set(), None))
    uids = ls.unit_uids(units)
    pos = {str(u): i for i, u in enumerate(z["test_uid"])}
    model, tok, label_ids, prov = load_parent(a, spec, ls)
    prov["compute"] = a.compute
    rows_unit = z["row_unit"]
    out = {"units": 0, "prompt_sha1_equal": 0, "max_abs_dlogp": 0.0, "mean_abs_dlogp": 0.0, "prov": prov}
    diffs = []
    keep = {"uid": [], "k": [], "n_tokens": [], "logp": [], "sha_equal": []}
    ctx = Fp32Compute(model) if a.compute == "fp32" else None
    for u, uid in zip(units, uids):
        if uid not in pos or len(u["option_keys"]) > 52:
            continue
        i = pos[uid]
        order = list(range(len(u["option_keys"])))
        text = ls.build_prompt(tok, u, order, a.max_len)
        eq = hashlib.sha1(text.encode("utf-8")).hexdigest() == str(z["test_input_sha"][i])
        out["prompt_sha1_equal"] += int(eq)
        rr = np.where(rows_unit == i)[0]
        if len(rr) != 1:
            continue
        ref = z["row_logp"][rr[0], : len(order)]
        if ctx is not None:
            with ctx:
                zz = forward_letters(model, tok, [text], label_ids, a.device, "exact").cpu()[0, : len(order)]
        else:
            zz = forward_letters(model, tok, [text], label_ids, a.device, "exact").cpu()[0, : len(order)]
        ours = torch.log_softmax(zz.double(), -1).numpy()
        diffs.append(float(np.abs(ours - ref).max()))
        out["units"] += 1
        lp = np.full(52, -1e9)
        lp[: len(order)] = ours
        for f, v in (("uid", str(uid)), ("k", len(order)), ("n_tokens", len(tok(text, add_special_tokens=False)["input_ids"])),
                     ("logp", lp), ("sha_equal", eq)):
            keep[f].append(v)
    if a.save_rows:
        np.savez_compressed(a.save_rows, uid=np.array(keep["uid"], dtype=str), k=np.array(keep["k"]), n_tokens=np.array(keep["n_tokens"]),
                            logp=np.array(keep["logp"]), sha_equal=np.array(keep["sha_equal"]), prov=json.dumps(prov))
    out["max_abs_dlogp"] = max(diffs) if diffs else None
    out["mean_abs_dlogp"] = float(np.mean(diffs)) if diffs else None
    out["p99_abs_dlogp"] = float(np.quantile(diffs, 0.99)) if diffs else None
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return out


def load_run(path):
    """uid -> (log p [k] float64, n_tokens or None) from a pinned-scorer npz (--save-rows, one row per unit: --n-perm 1) or ours."""
    z = np.load(path, allow_pickle=False)
    out = {}
    if "row_logp" in z.files:
        ru, ro, rl, tu = z["row_unit"], z["row_order"], z["row_logp"], z["test_uid"]
        seen = {}
        for r in range(len(ru)):
            seen.setdefault(int(ru[r]), []).append(r)
        for i, rs in seen.items():
            if len(rs) != 1:
                raise SystemExit(f"{path}: unit {tu[i]} has {len(rs)} rows (score with --n-perm 1)")
            r = rs[0]
            k = int((ro[r] >= 0).sum())
            if list(ro[r, :k]) != list(range(k)):
                raise SystemExit(f"{path}: unit {tu[i]} was scored in a non-identity order")
            out[str(tu[i])] = (rl[r, :k].astype(np.float64), None)
        return out
    for uid, k, n, lp in zip(z["uid"], z["k"], z["n_tokens"], z["logp"]):
        out[str(uid)] = (lp[: int(k)].astype(np.float64), int(n))
    return out


def pair_metrics(A, B, lens, thr=0.01):
    """A, B: uid -> log p. Per unit: TV, max|dp|, max|dlogp| (all letters), max|dlogp| over letters with p_A >= thr, KL(A||B),
    top-1 flip (outside near-ties: |p_A top1 - p_A top2| >= 0.02). -> summary (mean / p99 / max) overall and by length."""
    uids = sorted(set(A) & set(B))
    rows = []
    for u in uids:
        la, lb = A[u], B[u]
        if len(la) != len(lb):
            raise SystemExit(f"unit {u}: {len(la)} vs {len(lb)} options")
        pa, pb = np.exp(la), np.exp(lb)
        sa = np.sort(pa)[::-1]
        big = pa >= thr
        rows.append({"tv": 0.5 * float(np.abs(pa - pb).sum()), "maxdp": float(np.abs(pa - pb).max()),
                     "maxdlogp": float(np.abs(la - lb).max()), "maxdlogp_p01": float(np.abs(la - lb)[big].max()) if big.any() else 0.0,
                     "kl": float((pa * (la - lb)).sum()), "flip": bool(np.argmax(pa) != np.argmax(pb) and (len(sa) < 2 or sa[0] - sa[1] >= 0.02)),
                     "len": lens.get(u)})

    def summ(rs):
        if not rs:
            return {"n": 0}
        o = {"n": len(rs), "flips": int(sum(r["flip"] for r in rs))}
        for f in ("tv", "maxdp", "maxdlogp", "maxdlogp_p01", "kl"):
            v = np.array([r[f] for r in rs])
            o[f] = {"mean": round(float(v.mean()), 6), "p99": round(float(np.quantile(v, 0.99)), 6), "max": round(float(v.max()), 6)}
        return o
    out = {"all": summ(rows)}
    for name, lo, hi in (("len<=1024", 0, 1024), ("len>1024", 1025, 1 << 30)):
        out[name] = summ([r for r in rows if r["len"] is not None and lo <= r["len"] <= hi])
    return out


def compare(a):
    runs = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        runs[name] = load_run(path)
    lens = {}
    for r in runs.values():
        for u, (_, n) in r.items():
            if n is not None:
                lens[u] = n
    out = {"runs": {k: len(v) for k, v in runs.items()}, "pairs": {}}
    for pr in a.pairs.split(","):
        x, y = pr.split(":")
        out["pairs"][pr] = pair_metrics({u: v[0] for u, v in runs[x].items()}, {u: v[0] for u, v in runs[y].items()}, lens)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return {"pairs": {k: {m: v["all"].get(m, {}).get("mean") if isinstance(v["all"].get(m), dict) else v["all"].get(m)
                          for m in ("n", "tv", "maxdp", "maxdlogp", "flips")} for k, v in out["pairs"].items()}}


def bench(a):
    """forward-only throughput of the parent path on one GPU at given prompt lengths (random token ids; the real weights): -> per length
    and batching mode, tokens / s and peak memory. Feeds the extraction cost estimate (tokens per length stratum / tokens per s)."""
    ls, md5s = fr.load_scorer(a.code)
    parents = fr.parent_specs(a.parents)
    pid, prev, spec = parents[a.route]
    model, tok, label_ids, prov = load_parent(a, spec, ls)
    g = torch.Generator().manual_seed(0)
    out = {"route": a.route, "prov": prov, "rows": []}
    cuda = a.device == "cuda"
    for L in [int(x) for x in a.lengths.split(",")]:
        for mode in ("exact", "pack"):
            if mode == "pack" and L >= a.tokens_per_batch:
                continue
            nb = max(1, a.tokens_per_batch // L)
            ids = torch.randint(1000, 200000, (nb, L), generator=g)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            try:
                with torch.no_grad():
                    for it in range(a.bench_iters + 1):
                        t0 = time.time()
                        if mode == "exact":
                            model(input_ids=ids.to(a.device), use_cache=False, logits_to_keep=1)
                        else:
                            flat = ids.reshape(1, -1)
                            pos = torch.arange(L).repeat(nb)[None]
                            last = torch.arange(1, nb + 1) * L - 1
                            model(input_ids=flat.to(a.device), position_ids=pos.to(a.device), use_cache=False,
                                  logits_to_keep=last.to(a.device))
                        if cuda:
                            torch.cuda.synchronize()
                        if it:
                            times.append(time.time() - t0)
                row = {"length": L, "mode": mode, "rows": nb, "tokens_per_s": round(nb * L / float(np.median(times)), 1),
                       "s_per_forward": round(float(np.median(times)), 4), "oom": False}
            except torch.cuda.OutOfMemoryError as e:
                row = {"length": L, "mode": mode, "rows": nb, "oom": True, "msg": str(e)[:200]}
            if cuda:
                row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                torch.cuda.empty_cache()
            print("BENCH " + json.dumps(row), flush=True)
            out["rows"].append(row)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return {"rows": out["rows"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "merge", "parity", "bench", "compare", "refscore"))
    ap.add_argument("--code", default=HERE)
    ap.add_argument("--preservation", default=os.path.join(HERE, "preservation.py"))
    ap.add_argument("--preservation-md5", default="")
    ap.add_argument("--rows")
    ap.add_argument("--route", choices=("stock", "garas"))
    ap.add_argument("--parents")
    ap.add_argument("--model-dir")
    ap.add_argument("--allow-unverified-snapshot", action="store_true")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--attn", choices=(fm.ATTN_NAME, "sdpa"), default=fm.ATTN_NAME)
    ap.add_argument("--batch", choices=("exact", "pack"), default="exact")
    ap.add_argument("--tokens-per-batch", type=int, default=24576)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--caches", nargs="*")
    ap.add_argument("--data")
    ap.add_argument("--scorer-npz")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--lengths", default="512,2048,8192,32768")
    ap.add_argument("--bench-iters", type=int, default=3)
    ap.add_argument("--save-rows", default="", help="parity: per-unit letter log-probs -> this npz")
    ap.add_argument("--compute", choices=("bf16", "fp32"), default="bf16", help="parity: fp32 = the bf16-stored model computed in fp32")
    ap.add_argument("--runs", nargs="*", help="compare: NAME=FILE.npz (a pinned-scorer --save-rows npz or a parity --save-rows npz)")
    ap.add_argument("--pairs", default="", help="compare: A:B,C:D,...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"score": score, "merge": merge, "parity": parity, "bench": bench, "compare": compare, "refscore": refscore}[a.cmd](a)
    print("FULLFT_PARENT " + json.dumps({k: v for k, v in out.items() if k not in ("scored_by", "prov")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
