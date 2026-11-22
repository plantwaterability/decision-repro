#!/usr/bin/env python3
"""(v3 = build_manifest_v2 + dict instructions rendered with as_text = json.dumps(ensure_ascii=False), as the engine
and convert_decision_index do; v2 passed a dict to the template, i.e. Python repr. String instructions are unchanged.)
es-bar-lu-lil-la Stage-1 training manifest (sdv1-stage2-v1 schema, the trainer's contract).

Rows:
  base   liquamen's latest branch mixture (coteach_wave7_ppfixed_v1_25k_l4096_gemma.jsonl), prompts unchanged.
  loomd  Loom-verified pools from strandedcpu-7c (DI-format rows; hard negatives), rendered by claude_lmscore.build_prompt.
  bigk   K > 52 choice rows from CLINC150 train + BANKING77 train (minus the K-free held-out sample), same renderer.
Renderer id R3 = r2 (0fab3c66) + jev_labels (30435b1f): for K <= 52 R3 output is byte-identical to r2 (the label list's
first 52 entries are A-Z a-z, the renderer code is otherwise unchanged), so base rows are re-stamped, not re-rendered.
Decontamination: a row is dropped if its state (normalized, >= 25 chars) equals, or shares a 13-gram with, any DI 0.3
suite row or any held-out validation row (tmp/heldout_loom sets).

  python3 build_manifest.py --code DIR --model DIR --suite DIR --heldout DIR --pools DIR --base FILE --out FILE
         [--loomd-max N] [--bigk-max N] [--seed S]
"""
import argparse
import csv
import glob
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import urllib.request

R2 = "0fab3c66ad3bb2044c27a849022912e3d83cf46fb51e9f5b1807fbaa821884e3"
R3 = hashlib.sha256(f"r2:{R2}+jev_labels:30435b1faa49af2a".encode()).hexdigest()
SUFFIX = "Answer with the letter of the single best option and nothing else.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
MAX_TOKENS = 4096


def norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z ]", " ", str(s).lower())).strip()


def grams13(s):
    t = norm_text(s).split()
    return {" ".join(t[i:i + 13]) for i in range(len(t) - 12)}


def text_of(row):
    st = row.get("state")
    parts = [st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)]
    for q in (row.get("questions") or {}).values():
        parts.append(str(q.get("instructions") or ""))
    return "\n".join(parts)


class Gate:
    """Exact duplicates (normalized state, or the instructions of a state-less row) and 13-gram overlap of an item's OWN
    content against everything in the reference sets. Shared instruction templates are not content: a CLINC150 train
    utterance is checked by its utterance, not by the 15-word classification template every CLINC row carries."""

    def __init__(self):
        self.exact, self.grams = set(), set()
        self.short = {}                                  # first 3 words -> short reference items (< 13 words: no 13-grams)

    def add(self, row):
        st = row.get("state")
        s = norm_text(st if isinstance(st, str) else json.dumps(st, ensure_ascii=False))
        if len(s) >= 25:
            self.exact.add(s)
        for q in (row.get("questions") or {}).values():
            self.exact.add(norm_text(q.get("instructions") or ""))
        self.grams |= grams13(text_of(row))
        if isinstance(st, str) and 12 <= len(s) and len(s.split()) < 13:   # e.g. a BBH boolean expression
            w = s.split()
            if len(w) >= 3:
                self.short.setdefault(" ".join(w[:3]), set()).add(s)

    def clean(self, content, exact_keys=()):
        if any(len(norm_text(k)) >= 25 and norm_text(k) in self.exact for k in exact_keys):
            return False
        c = norm_text(content)
        w = c.split()
        for i in range(len(w) - 2):                      # a short reference item contained in the candidate's text
            for s in self.short.get(" ".join(w[i:i + 3]), ()):
                if s in c:
                    return False
        return not (grams13(content) & self.grams)


def jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def make_render(tok, clm, rng, split="train", tag="esbar1"):
    """DI-format choice question -> one sdv1-stage2-v1 manifest row (renderer R3; options shown in an rng-shuffled order), or None
    over MAX_TOKENS."""
    def render(uid, fam, state, instructions, criteria, gold_key, src, soft=None):
        keys = [str(k) for k in criteria]
        unit = {"state": state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
                "instructions": "" if instructions is None else instructions if isinstance(instructions, str)
                                else json.dumps(instructions, ensure_ascii=False),   # (v3) = garas_core / convert_decision_index as_text
                "qtype": "choice", "option_keys": keys, "noul_criteria": None,
                "option_texts": [v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in (criteria[k] for k in keys)]}  # as garas_core.as_text
        shown = keys[:]
        rng.shuffle(shown)
        res = clm.build_prompt(tok, unit, [keys.index(k) for k in shown], max_len=32768, noul_defs=True)
        prompt = res[0] if isinstance(res, tuple) else res
        assert prompt.endswith(SUFFIX), prompt[-200:]
        nt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        if nt > MAX_TOKENS:
            return None
        return {"ancestry_component_id": int(hashlib.md5(uid.encode()).hexdigest()[:8], 16), "decision_uid": uid, "di_ancestry": False,
                "exposure_id": f"{tag}:{uid}", "exposure_k": 0,
                "gold": {k: float(soft.get(k, 0.0)) for k in keys} if soft else {k: 1.0 if k == gold_key else 0.0 for k in keys},
                "kind": "choice", "loss_by_option": None, "n_tokens": nt, "numeric_family": False, "option_order": shown,
                "parent_id": None, "parent_preservation": None, "parent_route": "none", "parent_uid": uid.rsplit("|", 1)[0],
                "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "renderer_sha256": R3,
                "source": src, "source_family": src, "split": split, "target_basis": "soft_reference" if soft else "hard_fact", "weight": 1.0}
    return render


def main():
    ap = argparse.ArgumentParser()
    for k in ("code", "model", "suite", "heldout", "pools", "base", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--loomd-max", type=int, default=20000)
    ap.add_argument("--per-pool", default="", help="'tracking=6000,shapes=3000,chess_train=8000,...' (rows per pool file)")
    ap.add_argument("--bigk-max", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--exclude-groups", default="", help="file of pool group ids (one per line, or JSON lines with 'group') kept out "
                                                          "of training, e.g. the program library's eval split")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    excl = set()
    if a.exclude_groups:
        for l in open(a.exclude_groups):
            l = l.strip()
            if l:
                excl.add(json.loads(l)["group"] if l.startswith("{") else l)
    import claude_lmscore as clm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rng = random.Random(a.seed)

    gate = Gate()
    for f in ("selected-rows", "added-rows", "gsm8k-rows"):
        for r in jsonl(f"{a.suite}/{f}.jsonl.gz"):
            gate.add(r)
    for p in sorted(glob.glob(f"{a.heldout}/heldout_*.jsonl*") + glob.glob(f"{a.heldout}/kfree_rows.jsonl")):
        for r in jsonl(p):
            gate.add(r)
    print(f"gate: {len(gate.exact):,} exact keys, {len(gate.grams):,} 13-grams", flush=True)
    render = make_render(tok, clm, rng)
    out, counts = [], {}

    def keep(row, tag):
        if row is not None:
            out.append(row)
            counts[tag] = counts.get(tag, 0) + 1

    # base: liquamen's mixture, re-stamped (K <= 52: R3 == r2 byte for byte)
    for r in jsonl(a.base):
        if r.get("manifest_header"):
            continue
        assert r["renderer_sha256"] == R2 and len(r["option_order"]) <= 52
        r = dict(r, renderer_sha256=R3)
        keep(r, "base")

    # loomd: Loom-verified pools; each pool file gets its own quota (--per-pool "name=N,..." else --loomd-max / pools);
    # groups (4- / 10-option variants of one item) never split
    pools = sorted(glob.glob(f"{a.pools}/*.jsonl"))
    quota = dict(x.split("=") for x in a.per_pool.split(",") if x) if a.per_pool else {}
    for p in pools:
        name = os.path.basename(p)[:-len(".jsonl")]
        cap = int(quota.get(name, a.loomd_max // max(1, len(pools))))
        groups = {}
        for r in jsonl(p):
            g = r.get("group") or r["id"]
            if g in excl:
                counts[f"loomd_{name}_excluded_group"] = counts.get(f"loomd_{name}_excluded_group", 0) + 1
                continue
            groups.setdefault(g, []).append(r)
        gkeys = sorted(groups)
        rng.shuffle(gkeys)
        n = 0
        for g in gkeys:
            if n >= cap:
                break
            for r in groups[g]:
                st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
                if isinstance(r["state"], dict) and "fen" in r["state"]:
                    # chess: positions were excluded from the official test set by FEN and by placement + side at labelling
                    # (sf_label_chess.py); board diagrams share 13-word piece runs with unrelated positions, so no text gate
                    counts[f"loomd_{name}_fen_excluded_upstream"] = counts.get(f"loomd_{name}_fen_excluded_upstream", 0) + 1
                elif not gate.clean(st, exact_keys=(st,)):
                    counts[f"loomd_{name}_dropped_gate"] = counts.get(f"loomd_{name}_dropped_gate", 0) + 1
                    continue
                (qk, q), = r["questions"].items()
                row = render(f"loomd|{r['id']}", r["family"], r["state"], q["instructions"], q["criteria"], r["expected"][qk], r["family"],
                             soft=(r.get("soft_gold") or {}).get(qk))
                keep(row, f"loomd_{name}")
                n += row is not None

    # bigk: CLINC150 train (151-way) + BANKING77 train (77-way), minus the K-free held-out sample
    held = {json.loads(l)["questions"]["q1"]["instructions"] for l in open(f"{a.heldout}/kfree_rows.jsonl")}
    obj = json.load(urllib.request.urlopen("https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"))
    labels = sorted({p[1] for p in obj["train"]} | {"oos"})
    desc = ["out of scope: none of the listed intents" if l == "oos" else l.replace("_", " ") for l in labels]
    big = []
    for i, (t, l) in enumerate(obj["train"] + obj["oos_train"]):
        instr = "Classify the intent of this user request, or choose out of scope if none applies:\n" + t
        big.append((f"bigk|clinc150_train|{i}", "train_clinc150", instr, t, {f"option_{j}": d for j, d in enumerate(desc)}, f"option_{labels.index(l)}"))
    cats = json.load(urllib.request.urlopen("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json"))
    tr = list(csv.DictReader(io.StringIO(urllib.request.urlopen("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv").read().decode())))
    for i, r in enumerate(tr):
        instr = "Classify the banking intent of this user request:\n" + r["text"]
        big.append((f"bigk|banking77_train|{i}", "train_banking77", instr, r["text"], {f"option_{j}": c for j, c in enumerate(cats)}, f"option_{cats.index(r['category'])}"))
    rng.shuffle(big)
    n = 0
    for uid, fam, instr, utt, crit, gold in big:
        if n >= a.bigk_max:
            break
        if instr in held:
            counts["bigk_dropped_heldout"] = counts.get("bigk_dropped_heldout", 0) + 1
            continue
        if not gate.clean(utt, exact_keys=(instr,)):
            counts["bigk_dropped_gate"] = counts.get("bigk_dropped_gate", 0) + 1
            continue
        row = render(uid, fam, {}, instr, crit, gold, fam)
        keep(row, "bigk")
        n += row is not None

    rng.shuffle(out)
    body = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in out)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    header = {"manifest_header": True, "schema_version": "sdv1-stage2-v1", "renderer_sha256": R3, "rows": len(out),
              "rows_sha256": sha, "total_decision_weight": float(sum(r["weight"] for r in out))}
    json.dump(header, open(a.out + ".header.json", "w"), indent=2)
    kmax = max(len(r["option_order"]) for r in out)
    print(json.dumps({"counts": counts, "rows": len(out), "rows_sha256": sha, "renderer": R3, "k_max": kmax}), flush=True)


if __name__ == "__main__":
    main()
