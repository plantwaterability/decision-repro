#!/usr/bin/env python3
"""Letter scoring for choice questions with MORE than 52 options (jev-bench banking77 77, clinc150 150, ledgar 100, massive 60).

claude_lmscore.py labels options with 52 single-token letters. Here each option gets a TWO-token label: a chunk letter and a
digit ("A1".."A9", "B1", ...; 9 options per chunk, so up to 234 options), and the label probability factorises exactly:

    p(option = c d) = p(c | prompt) * p(d | prompt, c)

Pass 1 reads p(c) at the answer position (over the chunk letters in use). Pass 2 needs p(d | c) for EVERY chunk c: the prompt's
cache is copied n_chunks times along the batch axis and each copy is fed its chunk letter, so pass 2 costs n_chunks single-token
steps, not n_chunks prompts. Qwen3.8's hybrid cache has no working batch_repeat_interleave for its linear-attention layers, so
every per-layer state tensor (keys/values, conv/recurrent states) is repeated directly (checked against a full recompute on a
tiny random Qwen3.8: agreement to bf16 rounding). The tokenizer must split "C4" into "C" + "4"; checked for every label at load.

Option order is a nuisance as in claude_lmscore.py: --n-perm random non-identity orders are averaged. Output npz has the same
schema, so claude_jev_pub_npz.py / claude_paired_cfg.py read it unchanged.

    python3 trainers/claude_lmscore_bigk.py --data /data/local/jev_bench_all2 --configs ledgar,massive,banking77,clinc150 \
        --lora-ckpt /data/local/runs/lmtune_ext4000.pt --out analysis/claude_ext4k_bigk_zeroshot_units.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import string
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ls", os.path.join(HERE, "claude_lmscore.py"))
ls = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ls)
cf = ls.cf
CHUNKS = string.ascii_uppercase
DIGITS = "123456789"
PROMPT = ls.PROMPT.replace("Answer with the letter of the single best option and nothing else.",
                           "Answer with the label of the single best option (a letter followed by a digit, e.g. C4) and nothing else.")


def labels_for(k):
    return [f"{CHUNKS[i // len(DIGITS)]}{DIGITS[i % len(DIGITS)]}" for i in range(k)]


def build(tok, unit, order, max_len, margin=16):
    labs = labels_for(len(order))
    opts = "\n".join(f"{labs[j]}) {unit['option_texts'][o]}" for j, o in enumerate(order))

    def render(state):
        msg = PROMPT.format(state=state, instructions=unit["instructions"], options=opts)
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    budget = max_len - len(tok(render(""), add_special_tokens=False)["input_ids"]) - margin
    if budget <= 0:
        return None
    state = unit["state"]
    enc = tok(state, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) > budget:
        state = state[: int(enc["offset_mapping"][budget - 1][1])]
    return render(state)


def expand_cache(cache, n):
    for layer in cache.layers:
        for name, v in list(vars(layer).items()):
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
                setattr(layer, name, v.repeat_interleave(n, 0))
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--configs", required=True, help="comma list of config names (qid prefixes)")
    ap.add_argument("--n-perm", type=int, default=2, help="random non-identity option orders per unit")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=ls.MODEL)
    ap.add_argument("--lora-ckpt", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "zeroshot" not in os.path.basename(a.out):
        raise SystemExit("--out must contain 'zeroshot'")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    c_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in CHUNKS]
    d_ids = [tok(d, add_special_tokens=False)["input_ids"] for d in DIGITS]
    if any(len(x) != 1 for x in c_ids + d_ids):
        raise SystemExit("a chunk letter or digit is not a single token")
    c_ids = torch.tensor([x[0] for x in c_ids]); d_ids = torch.tensor([x[0] for x in d_ids])
    for lab in labels_for(len(CHUNKS) * len(DIGITS)):
        got = tok(lab, add_special_tokens=False)["input_ids"]
        if got != [int(c_ids[CHUNKS.index(lab[0])]), int(d_ids[DIGITS.index(lab[1])])]:
            raise SystemExit(f"label {lab!r} does not tokenize as letter + digit: {got}")
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev).eval()
    if a.lora_ckpt:
        spec = importlib.util.spec_from_file_location("te", os.path.join(HERE, "train_e2e.py"))
        te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)
        ck = torch.load(a.lora_ckpt, map_location="cpu")
        if ck["config"].get("plain_prompt"):
            raise SystemExit("this scorer uses the chat template; the adapter was trained on plain prompts")
        te.apply_lora(model, int(ck["config"]["lora_r"]), targets=tuple(ck["config"]["lora_targets"]))
        want = {n for n, _ in model.named_parameters() if n.endswith(".A") or n.endswith(".B")}
        if set(ck["lora"]) != want:
            raise SystemExit(f"LoRA key mismatch: {len(set(ck['lora']) ^ want)} keys differ")
        model.load_state_dict(ck["lora"], strict=False)
        print(f"LoRA applied from {a.lora_ckpt}: {len(want)} adapter tensors", flush=True)
    want_cfg = set(a.configs.split(","))
    units = [u for u in cf.iter_units(a.data, [s for s in a.splits.split(",") if s], set(), None)
             if str(u["qid"]).split("/")[0] in want_cfg and u["qtype"] == "choice"
             and len(u["option_keys"]) <= len(CHUNKS) * len(DIGITS)]
    if a.limit:
        units = units[: a.limit]
    print(f"units {len(units)} (K range {min(len(u['option_keys']) for u in units)}-{max(len(u['option_keys']) for u in units)})",
          flush=True)
    rng = np.random.default_rng(a.seed)
    P, T, qid, src, ks, soft, ent = [], [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for ui, u in enumerate(units):
            k = len(u["option_keys"]); nch = (k + len(DIGITS) - 1) // len(DIGITS)
            dists = []
            for _ in range(a.n_perm):
                order = list(range(k))
                while order == list(range(k)):
                    order = list(rng.permutation(k))
                text = build(tok, u, order, a.max_len)
                if text is None:
                    break
                enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
                out = model(**enc, use_cache=True, logits_to_keep=1)
                lc = torch.log_softmax(out.logits[0, -1].float()[c_ids[:nch].to(dev)], -1)            # [nch]
                cache = expand_cache(out.past_key_values, nch)
                am = torch.cat([enc["attention_mask"].repeat(nch, 1),
                                torch.ones(nch, 1, dtype=enc["attention_mask"].dtype, device=dev)], 1)
                out2 = model(input_ids=c_ids[:nch].to(dev).view(nch, 1), attention_mask=am, past_key_values=cache,
                             use_cache=False, logits_to_keep=1)
                ld = out2.logits[:, -1].float()[:, d_ids.to(dev)]                                    # [nch, 9]
                lp = []
                for j in range(k):
                    c, d = divmod(j, len(DIGITS))
                    m = min(len(DIGITS), k - c * len(DIGITS))                                        # members in chunk c
                    lp.append(lc[c] + torch.log_softmax(ld[c, :m], -1)[d])
                p_pres = torch.stack(lp).exp().cpu().numpy()                                          # presentation order
                p = np.zeros(k); p[np.asarray(order)] = p_pres
                dists.append(p / p.sum())
                del out, out2, cache
            if not dists:
                continue
            ps = np.stack(dists); pm = ps.mean(0)
            v = np.asarray(u["target"], dtype=np.float64)
            P.append(pm); T.append(v / max(v.sum(), 1e-12)); ks.append(k); soft.append(u["label_kind"] == "soft")
            qid.append(str(u["qid"])); src.append(str(u["source"]))
            ent.append(-(ps * np.log(np.clip(ps, 1e-12, 1))).sum(1).mean())
            if (ui + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {ui + 1}/{len(units)} units  {(ui + 1) / el:.2f}/s  eta {(len(units) - ui - 1) / ((ui + 1) / el) / 60:.1f} min",
                      flush=True)
    kmax = max(ks); n = len(P)
    Pm = np.zeros((n, kmax)); Tm = np.zeros((n, kmax))
    for i in range(n):
        Pm[i, : ks[i]], Tm[i, : ks[i]] = P[i], T[i]
    tot = -(Pm * np.log(np.clip(Pm, 1e-12, 1))).sum(1)
    np.savez_compressed(a.out, test_p=Pm, test_t=Tm, test_ks=np.array(ks), test_soft=np.array(soft), test_qid=np.array(qid),
                        test_source=np.array(src), test_tot=tot, test_pred_ent=np.array(ent), test_mx=Pm.max(1),
                        test_perm_mi=tot - np.array(ent))
    acc = float((Pm.argmax(1) == Tm.argmax(1)).mean())
    print(f"wrote {a.out}: {n} units, pooled acc {acc:.4f}, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
