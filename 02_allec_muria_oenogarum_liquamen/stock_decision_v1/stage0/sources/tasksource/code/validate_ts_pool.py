#!/usr/bin/env python3
"""Validate the converted tasksource pool with the PRODUCTION reader (cache_features.py iter_units / question_options, md5 49613a40)
and spot-check the pinned scorer's build_prompt (claude_lmscore.py e0672bd0) on a stratified sample (choice / noul / score).
Checks per record: exactly one unit, qtype == upstream kind, K == len(options) (noul 2), target == upstream target exactly
(noul: [p, 1 - p]), label_kind soft iff >1 positive option. Prompt check: every option text appears once in the rendered prompt in
the drawn order, the instructions and state appear, no target / meta text leaks (raw_target, upstream id)."""
import collections
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

POOL, CODE, TOK = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(CODE / "trainers"))
assert hashlib.md5((CODE / "trainers/cache_features.py").read_bytes()).hexdigest() == "49613a40faed8cd612175e81c3f3925f"
assert hashlib.md5((CODE / "trainers/claude_lmscore.py").read_bytes()).hexdigest() == "e0672bd0f515ccb17d97035c84c3d339"
spec = importlib.util.spec_from_file_location("ls", CODE / "trainers/claude_lmscore.py")
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)
cf = ls.cf if hasattr(ls, "cf") else __import__("cache_features")
recs = {}
for p in sorted((POOL / "train").glob("*.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        recs[r["id"]] = r
c = collections.Counter()
sample = collections.defaultdict(list)
rng = random.Random(20261013)
seen = collections.Counter()
for u in cf.iter_units(str(POOL), ["train"], set(), None):
    r = recs[u["record_id"]]
    m = r["meta"]
    seen[u["record_id"]] += 1
    c["units"] += 1
    if u["qtype"] != m["kind"]:
        c["qtype_mismatch"] += 1
    k = len(m["raw_options"]) if m["kind"] != "noul" else 2
    if len(u["option_keys"]) != k:
        c["K_mismatch"] += 1
    exp = m["raw_target"] if m["kind"] != "noul" else [m["raw_target"][0], 1.0 - min(m["raw_target"][0], 1.0)]
    if [float(x) for x in u["target"]] != [float(x) for x in exp]:
        c["target_mismatch"] += 1
    if m["kind"] != "noul" and list(u["option_texts"]) != list(m["raw_options"]):
        c["option_text_mismatch"] += 1
    soft = sum(1 for v in u["target"] if v > 0) > 1
    if (u["label_kind"] == "soft") != soft:
        c["label_kind_mismatch"] += 1
    key = (m["kind"], "soft" if soft else "hard", "K>52" if k > 52 else "ok")
    if len(sample[key]) < 120:
        sample[key].append(u)
    elif rng.random() < 0.01:
        sample[key][rng.randrange(120)] = u
c["records"] = len(recs)
c["records_with_one_unit"] = sum(1 for v in seen.values() if v == 1)
c["records_without_unit"] = len(recs) - len(seen)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOK)
pc = collections.Counter()
for key, us in sorted(sample.items()):
    for u in us[:60]:
        k = len(u["option_keys"])
        if k > 52:
            continue
        order = list(range(k))
        if u["qtype"] == "choice":
            rng.shuffle(order)
        for nd in (False, True):
            text = ls.build_prompt(tok, u, order, 1 << 30, noul_defs=nd)
            pc["prompts"] += 1
            texts = ls.option_texts(u, nd)
            ok = all(f"{ls.LABELS[j]}) {texts[o]}" in text for j, o in enumerate(order)) and u["instructions"] in text and u["state"] in text
            pc["ok" if ok else "bad"] += 1
            r = recs[u["record_id"]]
            if r["id"] in text and r["id"] not in u["state"]:
                pc["upstream_id_leak"] += 1
print(json.dumps({"reader": dict(c), "prompt_spotcheck": dict(pc), "sample_strata": {"|".join(k): len(v) for k, v in sample.items()}}))
