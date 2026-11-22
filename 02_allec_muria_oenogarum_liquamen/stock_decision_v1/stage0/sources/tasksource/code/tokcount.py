#!/usr/bin/env python3
"""tokcount.py: measured Gemma prompt lengths for every decision unit, under the production renderer (R1) and the proposed
renderer R2 (garas-render-2). Data preparation only: tokenizer files, no model weights, no downloads, no truncation.

R1  "garas-render-1" = trainers/claude_lmscore.py build_prompt(tok, unit, order, max_len, noul_defs=False), the renderer that
    trained and deploys Garas / (99) (pinned scorer md5 e0672bd0f515ccb17d97035c84c3d339; units from trainers/cache_features.py
    question_options, md5 49613a40faed8cd612175e81c3f3925f).
R2  "garas-render-2" = the same build_prompt with noul_defs=True (the existing option_texts path: "Yes: <def>" / "No: <def>")
    plus one new rule: option texts that collide EXACTLY after rendering get " [<option key>]" appended (colliding options only;
    targets, keys and order untouched). R2 equals production `--noul-defs` byte-for-byte on every unit without a collision.

Order: the stored (identity) option order; --parity measures the effect of a random order on token counts for a sample.
Splits: default train,val. `confirmation` is refused, and confirmation members of an archive are never extracted or parsed.

Input (--input): an s3:// tgz (downloaded with the aws CLI, md5-checked with --md5), a local .tgz, or a directory laid out
<root>/<split>/*.jsonl (the trainer's one-level layout). Output (--out DIR, must not exist):
  units_tokens_<pool>.jsonl.gz   one row per unit: uid = "<pool>|<member>|<line>|<qid>" (member = jsonl path from the split dir,
                                 e.g. "train/jev_rd3__courts.jsonl"; line 1-based), tokens, flags
  summary_<pool>.json            totals / strata / fit counts by split and source
  parity_<pool>.json             byte parity of the fast renderer against the real production build_prompt on a sample
  inputs_<pool>.json             input archive / member hashes, code hashes, tokenizer pin

  python3 tokcount.py --tokenizer TOKDIR --code CODEDIR --input s3://.../emo_pool_v1.tgz --md5 832a6bd9... --pool emo_pool_v1 \
      --splits train,val --work /data/local/garas_v2/tok/work --out /data/local/garas_v2/tok/out_emo_pool_v1 --procs 4
CODEDIR holds trainers/claude_lmscore.py (e0672bd0) and trainers/cache_features.py (49613a40); md5s are refused if different.
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import string
import subprocess
import sys
import tarfile
import time
from pathlib import Path

VERSION = "tokcount.py v1 (2026-10-12; garas_gemma_vbll v2 gates 4/5) + stock_decision_v1 K>52 skip (2026-10-13)"
SCORER_MD5 = "e0672bd0f515ccb17d97035c84c3d339"
LOADER_MD5 = "49613a40faed8cd612175e81c3f3925f"
R1_ID, R2_ID = "garas-render-1", "garas-render-2"
LENGTH_STRATA = [(1, 512), (513, 2048), (2049, 8192), (8193, 32768), (32769, 10 ** 12)]
MENU_STRATA = [(2, 4), (5, 12), (13, 32), (33, 52), (53, 10 ** 9)]
FIT_LIMITS = (2048, 4096, 8192, 32768)
MARGIN = 16                      # build_prompt's default margin
TRIVIAL_DEF = {"", "yes", "no", "true", "false", "y", "n"}
SPLIT_OK = ("train", "val")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ production code, extracted unchanged (no torch import)
def extract(path, names, consts=()):
    """exec the named top-level functions / assignments of a source file, verbatim, into a fresh namespace"""
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in n.targets))]
    found = {n.name for n in keep if isinstance(n, ast.FunctionDef)} | {t.id for n in keep if isinstance(n, ast.Assign) for t in n.targets}
    missing = (set(names) | set(consts)) - found
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    env = {"re": re, "string": string, "json": json, "hashlib": hashlib}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), env)
    return env


class Prod:
    def __init__(self, code):
        sc, ld = Path(code) / "trainers/claude_lmscore.py", Path(code) / "trainers/cache_features.py"
        for p, want in ((sc, SCORER_MD5), (ld, LOADER_MD5)):
            if md5_file(p) != want:
                raise SystemExit(f"{p}: md5 {md5_file(p)} is not the pinned {want}")
        s = extract(sc, ("noul_definitions", "option_texts"), ("LABELS", "PROMPT"))
        c = extract(ld, ("question_options", "is_noncommercial"), ("NC_LICENCE", "NC_SOURCES"))
        self.LABELS, self.PROMPT = s["LABELS"], s["PROMPT"]
        self.noul_definitions, self.option_texts = s["noul_definitions"], s["option_texts"]
        self.question_options, self.is_noncommercial = c["question_options"], c["is_noncommercial"]
        self.hashes = {"claude_lmscore.py": {"md5": SCORER_MD5, "sha256": sha256_file(sc)},
                       "cache_features.py": {"md5": LOADER_MD5, "sha256": sha256_file(ld)}}


def unit_of(P, rec, qid, q, split, stem):
    """= cache_features.iter_units' unit dict (the fields build_prompt and the trainer filters read)"""
    keys, texts, target = P.question_options(q, rec["labels"][qid])
    return {"split": split, "source": rec.get("source") or stem,
            "license": str(rec.get("license") or (rec.get("meta") or {}).get("license") or ""),
            "record_id": rec["id"], "qid": qid, "qtype": q["type"], "state": rec["state"], "instructions": q["instructions"],
            "option_keys": keys, "option_texts": texts, "noul_criteria": q.get("criteria") if q["type"] == "noul" else None,
            "target": target}


def r2_texts(P, unit):
    """R2 option texts: production noul definitions, then exact-collision disambiguation by option key"""
    texts = list(P.option_texts(unit, True))
    cnt = collections.Counter(texts)
    if len(cnt) != len(texts):
        texts = [f"{t} [{k}]" if cnt[t] > 1 else t for t, k in zip(texts, unit["option_keys"])]
    return texts


def render(tok, P, unit, texts, order, state):
    """the no-truncation branch of claude_lmscore.build_prompt (chat=None, plain=False, whole_case=False), verbatim"""
    opts = "\n".join(f"{P.LABELS[j]}) {texts[o]}" for j, o in enumerate(order))
    msg = P.PROMPT.format(state=state, instructions=unit["instructions"], options=opts)
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def stratum(n, strata):
    for i, (lo, hi) in enumerate(strata):
        if lo <= n <= hi:
            return i
    return -1


# ------------------------------------------------------------------ worker
G = {}


def init_worker(tokdir, code):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    G["tok"] = AutoTokenizer.from_pretrained(tokdir)
    G["P"] = Prod(code)


def lens(tok, texts):
    if not texts:
        return []
    return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]


def new_agg():
    return collections.defaultdict(collections.Counter)


def work(job):
    """job = (chunk index, split, stem, member, path, first line, last line, pool, outdir) -> part file + aggregates"""
    ci, split, stem, member, path, lo, hi, pool, outdir = job
    tok, P = G["tok"], G["P"]
    units, rows = [], []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if ln < lo:
                continue
            if ln > hi:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            for qid, q in rec["questions"].items():
                u = unit_of(P, rec, qid, q, split, stem)
                units.append((ln, u))
    agg = new_agg()
    B = 128
    part = Path(outdir) / "parts" / f"{ci:07d}.jsonl.gz"
    n_all = len(units)
    big = [(ln, u) for ln, u in units if len(u["option_keys"]) > len(P.LABELS)]     # (stock_decision_v1) K > 52: the production
    units = [(ln, u) for ln, u in units if len(u["option_keys"]) <= len(P.LABELS)]  # renderer cannot letter them; not rendered
    with gzip.open(part, "wt", encoding="utf-8") as out:
        for ln, u in big:
            out.write(json.dumps({"uid": f"{pool}|{member}|{ln}|{u['qid']}", "split": split, "stem": stem, "source": u["source"],
                                  "record_id": str(u["record_id"]), "qtype": u["qtype"], "K": len(u["option_keys"]),
                                  "nc": bool(P.is_noncommercial(u)), "tok_r1": None, "tok_r2": None, "k_gt_52_unrendered": True},
                                 ensure_ascii=False) + "\n")
            for key in (("all", split, "*"), ("stem", split, stem), ("root", split, stem.split("__")[0])):
                agg[key]["units"] += 1
                agg[key]["k_gt_52"] += 1
                agg[key]["k_gt_52_unrendered"] += 1
        for b0 in range(0, len(units), B):
            batch = units[b0:b0 + B]
            t_full, t_over, t_state, meta = [], [], [], []
            t2_full, t2_over, idx2 = [], [], []
            for i, (ln, u) in enumerate(batch):
                st = u["state"]
                nonstr = not isinstance(st, str)
                if nonstr:
                    st = str(st)
                k = len(u["option_keys"])
                order = list(range(k))
                tx1 = list(P.option_texts(u, False))
                tx2 = r2_texts(P, u)
                t_full.append(render(tok, P, u, tx1, order, st))
                t_over.append(render(tok, P, u, tx1, order, ""))
                t_state.append(st)
                if tx2 != tx1:
                    idx2.append(i)
                    t2_full.append(render(tok, P, u, tx2, order, st))
                    t2_over.append(render(tok, P, u, tx2, order, ""))
                d = P.noul_definitions(u["noul_criteria"]) if u["qtype"] == "noul" else None
                subst = bool(d) and any(norm(v) not in TRIVIAL_DEF for v in d.values())
                meta.append((ln, u, k, nonstr, bool(u["noul_criteria"]), bool(d), subst,
                             sum(len(str(v)) for v in (d or {}).values()),
                             len(set(tx1)) != k, len(set(map(norm, tx1))) != k, len(set(tx2)) != k))
            n_full, n_over, n_state = lens(tok, t_full), lens(tok, t_over), lens(tok, t_state)
            n2_full, n2_over = lens(tok, t2_full), lens(tok, t2_over)
            r2 = {i: (a, o) for i, a, o in zip(idx2, n2_full, n2_over)}
            for i, (ln, u, k, nonstr, crit, hasdef, subst, deflen, dup_x, dup_n, dup2) in enumerate(meta):
                f1, o1, s1 = n_full[i], n_over[i], n_state[i]
                f2, o2 = r2.get(i, (f1, o1))
                nc = bool(P.is_noncommercial(u))
                row = {"uid": f"{pool}|{member}|{ln}|{u['qid']}", "split": split, "stem": stem, "source": u["source"],
                       "record_id": str(u["record_id"]), "qtype": u["qtype"], "K": k, "nc": nc, "tok_r1": f1, "tok_r2": f2,
                       "over_r1": o1, "over_r2": o2, "state_tok": s1, "state_chars": len(t_state[i]),
                       "noul_crit": crit, "noul_def": hasdef, "noul_def_subst": subst, "noul_def_chars": deflen,
                       "dup_exact": dup_x, "dup_norm": dup_n, "dup_r2": dup2, "r2_changed": i in r2}
                if nonstr:
                    row["state_nonstring"] = True
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                fits = {L: k <= 52 and s1 <= L - o1 - MARGIN for L in FIT_LIMITS}
                fits2 = {L: k <= 52 and s1 <= L - o2 - MARGIN for L in FIT_LIMITS}
                kept = (not nc) and k <= 52
                for key in (("all", split, "*"), ("stem", split, stem), ("root", split, stem.split("__")[0])):
                    a = agg[key]
                    a["units"] += 1
                    a["trainer_kept"] += kept
                    a["nc"] += nc
                    a["k_gt_52"] += k > 52
                    a["tok_r1"] += f1
                    a["tok_r2"] += f2
                    a["tok_r1_kept"] += f1 if kept else 0
                    a["tok_r2_kept"] += f2 if kept else 0
                    a["state_tok"] += s1
                    a[f"len_r1_{stratum(f1, LENGTH_STRATA)}"] += 1
                    a[f"len_r2_{stratum(f2, LENGTH_STRATA)}"] += 1
                    a[f"lensum_r1_{stratum(f1, LENGTH_STRATA)}"] += f1
                    a[f"lensum_r2_{stratum(f2, LENGTH_STRATA)}"] += f2
                    if kept:
                        a[f"klen_r1_{stratum(f1, LENGTH_STRATA)}"] += 1
                        a[f"klensum_r1_{stratum(f1, LENGTH_STRATA)}"] += f1
                        a[f"klen_r2_{stratum(f2, LENGTH_STRATA)}"] += 1
                        a[f"klensum_r2_{stratum(f2, LENGTH_STRATA)}"] += f2
                        a[f"kmenu_{stratum(k, MENU_STRATA)}"] += 1
                    a[f"menu_{stratum(k, MENU_STRATA)}"] += 1
                    for L in FIT_LIMITS:
                        a[f"nofit_r1_{L}"] += not fits[L]
                        a[f"nofit_r2_{L}"] += not fits2[L]
                        if kept:
                            a[f"knofit_r1_{L}"] += not fits[L]
                    a[f"qtype_{u['qtype']}"] += 1
                    a["noul_crit"] += crit
                    a["noul_def"] += hasdef
                    a["noul_def_subst"] += subst
                    a["noul_def_chars"] += deflen
                    a["dup_exact"] += dup_x
                    a["dup_norm"] += dup_n
                    a["dup_r2"] += dup2
                    a["r2_changed"] += i in r2
                    a["r2_delta_tokens"] += f2 - f1
                    a["state_nonstring"] += nonstr
                    a[f"log2_r1_{max(0, f1 - 1).bit_length()}"] += 1
    return ci, {"|".join(k): dict(v) for k, v in agg.items()}, n_all


# ------------------------------------------------------------------ inputs
def split_rel(name, splits):
    parts = [p for p in name.split("/") if p not in ("", ".")]
    for i, p in enumerate(parts[:-1]):
        if p == "confirmation":
            return None
        if p in splits and i == len(parts) - 2:
            return "/".join(parts[i:])
    return None


def stage_input(a, work):
    info = {"input": a.input}
    splits = a.splits
    if os.path.isdir(a.input):
        root = Path(a.input)
        files = []
        for s in splits:
            for p in sorted((root / s).glob("*.jsonl")):
                if not p.name.startswith("._"):
                    files.append((s, p.stem, f"{s}/{p.name}", p))
        info["kind"] = "dir"
        info["members"] = {m: {"sha256": sha256_file(p), "md5": md5_file(p)} for _, _, m, p in files}
        return files, info
    tgz = a.input
    if a.input.startswith("s3://"):
        tgz = str(work / ("in_" + os.path.basename(a.input)))
        if not os.path.exists(tgz):
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", a.input, tgz + ".part"], check=True)
            os.replace(tgz + ".part", tgz)
    got = md5_file(tgz)
    if a.md5 and got != a.md5:
        raise SystemExit(f"{a.input}: md5 {got} != pinned {a.md5}")
    info.update(kind="tgz", md5=got, sha256=sha256_file(tgz), bytes=os.path.getsize(tgz))
    ex = work / f"extract_{a.pool}"
    files, members = [], {}
    if (ex / ".complete").exists():
        members = json.loads((ex / ".complete").read_text())
    else:
        ex.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r|gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".jsonl") or os.path.basename(m.name).startswith("._"):
                    continue
                rel = split_rel(m.name, splits)
                if rel is None:
                    continue
                dst = ex / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                h5, h256 = hashlib.md5(), hashlib.sha256()
                src = tf.extractfile(m)
                with open(dst, "wb") as f:
                    for b in iter(lambda: src.read(1 << 22), b""):
                        f.write(b); h5.update(b); h256.update(b)
                members[rel] = {"archive_member": m.name, "md5": h5.hexdigest(), "sha256": h256.hexdigest(), "bytes": m.size}
        (ex / ".complete").write_text(json.dumps(members))
    for rel in sorted(members):
        s, fn = rel.split("/", 1)
        files.append((s, Path(fn).stem, rel, ex / rel))
    info["members"] = members
    return files, info


def chunks(files, pool, outdir, lines_per_chunk):
    jobs = []
    for split, stem, member, path in files:
        n = 0
        with open(path, "rb") as f:
            for _ in f:
                n += 1
        for lo in range(1, n + 1, lines_per_chunk):
            jobs.append((len(jobs), split, stem, member, str(path), lo, min(n, lo + lines_per_chunk - 1), pool, str(outdir)))
    return jobs


# ------------------------------------------------------------------ parity against the real production function
def parity(a, files, outdir, prod):
    """re-render a deterministic sample through the REAL trainers/claude_lmscore.build_prompt (module imported as shipped)"""
    import importlib.util
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(Path(a.code) / "trainers"))
    spec = importlib.util.spec_from_file_location("prod_lmscore", Path(a.code) / "trainers/claude_lmscore.py")
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    want = collections.defaultdict(set)
    flagged = collections.Counter()
    with gzip.open(outdir / f"units_tokens_{a.pool}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("k_gt_52_unrendered"):            # (stock_decision_v1) K > 52 rows carry no token counts
                continue
            h = int(hashlib.sha1(r["uid"].encode()).hexdigest()[:8], 16)
            pick = (h % a.parity_every == 0) or (r["dup_exact"] and flagged["dup"] < 400) or \
                   (r["noul_def"] and h % 50 == 0 and flagged["noul"] < 1500) or (r["tok_r1"] > 8192 and flagged["long"] < 100)
            if pick:
                flagged["dup"] += r["dup_exact"]; flagged["noul"] += r["noul_def"]; flagged["long"] += r["tok_r1"] > 8192
                _, member, ln, qid = r["uid"].split("|")
                want[member].add((int(ln), qid))
    path_of = {m: p for _, _, m, p in files}
    res = collections.Counter()
    bad, order_delta = [], []
    rng = random.Random(0)
    for member in sorted(want):
        split, fn = member.split("/", 1)
        stem = Path(fn).stem
        need = want[member]
        lines = {ln for ln, _ in need}
        with open(path_of[member], encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if ln not in lines:
                    continue
                rec = json.loads(line)
                for qid, q in rec["questions"].items():
                    if (ln, qid) not in need:
                        continue
                    u = unit_of(prod, rec, qid, q, split, stem)
                    if not isinstance(u["state"], str):
                        res["skipped_nonstring_state"] += 1
                        continue
                    # the real loader's unit equals ours
                    keys, texts, target = ls.cf.question_options(q, rec["labels"][qid])
                    res["loader_parity"] += (keys, texts, list(target)) == (u["option_keys"], u["option_texts"], list(u["target"]))
                    k = len(u["option_keys"])
                    if k > 52:
                        res["skipped_k_gt_52"] += 1
                        continue
                    order = list(range(k))
                    real1 = ls.build_prompt(tok, u, order, 10 ** 9, noul_defs=False)
                    fast1 = render(tok, prod, u, prod.option_texts(u, False), order, u["state"])
                    real2 = ls.build_prompt(tok, u, order, 10 ** 9, noul_defs=True)
                    tx2 = r2_texts(prod, u)
                    fast2 = render(tok, prod, u, tx2, order, u["state"])
                    collide = len(set(prod.option_texts(u, True))) != k
                    res["units"] += 1
                    res["r1_equal"] += real1 == fast1
                    res["r2_equal_prod_noul_defs"] += (real2 == fast2) if not collide else 0
                    res["r2_collision_units"] += collide
                    if collide:   # R2 = production --noul-defs with the colliding texts suffixed; verify by reconstruction
                        u2 = dict(u, option_texts=tx2, noul_criteria=None)
                        res["r2_collision_equal"] += ls.build_prompt(tok, u2, order, 10 ** 9, noul_defs=False) == fast2
                    if real1 != fast1 or (not collide and real2 != fast2):
                        bad.append(f"{member}:{ln}:{qid}")
                    if q["type"] == "choice" and k >= 2:
                        perm = order[:]
                        rng.shuffle(perm)
                        n0 = len(tok(fast1, add_special_tokens=False)["input_ids"])
                        n1 = len(tok(ls.build_prompt(tok, u, perm, 10 ** 9), add_special_tokens=False)["input_ids"])
                        order_delta.append(n1 - n0)
    res = dict(res)
    res["mismatch_examples"] = bad[:20]
    res["pass"] = (res.get("units", 0) > 0 and res.get("r1_equal", 0) == res["units"]
                   and res.get("r2_equal_prod_noul_defs", 0) == res["units"] - res.get("r2_collision_units", 0)
                   and res.get("r2_collision_equal", 0) == res.get("r2_collision_units", 0)
                   and res.get("loader_parity", 0) == res["units"] + res.get("skipped_k_gt_52", 0))
    if order_delta:
        res["random_order_token_delta"] = {"n": len(order_delta), "mean": sum(order_delta) / len(order_delta),
                                           "mean_abs": sum(map(abs, order_delta)) / len(order_delta),
                                           "min": min(order_delta), "max": max(order_delta)}
    res["real_function"] = "trainers/claude_lmscore.build_prompt imported from the pinned file (max_len 1e9, chat=None)"
    return res


# ------------------------------------------------------------------ summary
def summarize(agg, pool):
    out = {"by_split": {}, "by_stem": {}, "by_root": {}}
    for key, c in agg.items():
        kind, split, name = key.split("|", 2)
        d = {"all": out["by_split"], "stem": out["by_stem"], "root": out["by_root"]}[kind]
        d[f"{split}|{name}" if kind != "all" else split] = dict(sorted(c.items()))
    tot = collections.Counter()
    for k, c in agg.items():
        if k.startswith("all|"):
            tot.update(c)
    out["total"] = dict(sorted(tot.items()))
    t = tot
    out["derived"] = {
        "units": t["units"], "trainer_kept_units": t["trainer_kept"], "tokens_r1": t["tok_r1"], "tokens_r2": t["tok_r2"],
        "tokens_r1_trainer_kept": t["tok_r1_kept"], "tokens_r2_trainer_kept": t["tok_r2_kept"],
        "mean_r1": t["tok_r1"] / max(1, t["units"]), "mean_r2": t["tok_r2"] / max(1, t["units"]),
        "length_strata": [f"{lo}-{hi if hi < 10 ** 11 else 'inf'}" for lo, hi in LENGTH_STRATA],
        "menu_strata": [f"{lo}-{hi if hi < 10 ** 8 else 'inf'}" for lo, hi in MENU_STRATA],
        "kept_units_per_length_stratum_r1": [t[f"klen_r1_{i}"] for i in range(len(LENGTH_STRATA))],
        "kept_mean_tokens_per_length_stratum_r1": [t[f"klensum_r1_{i}"] / t[f"klen_r1_{i}"] if t[f"klen_r1_{i}"] else None
                                                   for i in range(len(LENGTH_STRATA))],
        "kept_units_per_length_stratum_r2": [t[f"klen_r2_{i}"] for i in range(len(LENGTH_STRATA))],
        "kept_mean_tokens_per_length_stratum_r2": [t[f"klensum_r2_{i}"] / t[f"klen_r2_{i}"] if t[f"klen_r2_{i}"] else None
                                                   for i in range(len(LENGTH_STRATA))],
        "kept_units_per_menu_stratum": [t[f"kmenu_{i}"] for i in range(len(MENU_STRATA))],
        "kept_not_fitting_without_state_cut_r1": {str(L): t[f"knofit_r1_{L}"] for L in FIT_LIMITS},
        "all_not_fitting_without_state_cut_r1": {str(L): t[f"nofit_r1_{L}"] for L in FIT_LIMITS},
        "all_not_fitting_without_state_cut_r2": {str(L): t[f"nofit_r2_{L}"] for L in FIT_LIMITS},
        "noul_crit_units": t["noul_crit"], "noul_def_units": t["noul_def"], "noul_def_substantive_units": t["noul_def_subst"],
        "dup_exact_units": t["dup_exact"], "dup_norm_units": t["dup_norm"], "dup_after_r2_units": t["dup_r2"],
        "r2_changed_units": t["r2_changed"], "r2_minus_r1_tokens": t["r2_delta_tokens"],
        "state_nonstring_units": t["state_nonstring"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="local dir with the pinned tokenizer files (TOKENIZER_PIN.json)")
    ap.add_argument("--code", required=True, help="dir with trainers/claude_lmscore.py + trainers/cache_features.py (pinned)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--md5", default="")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--lines-per-chunk", type=int, default=4000)
    ap.add_argument("--parity-every", type=int, default=300)
    ap.add_argument("--no-parity", action="store_true")
    a = ap.parse_args()
    a.splits = [s for s in a.splits.split(",") if s]
    if any(s not in SPLIT_OK for s in a.splits):
        raise SystemExit(f"splits {a.splits}: only {SPLIT_OK} are measured; confirmation is never read")
    if "|" in a.pool:
        raise SystemExit("pool tag must not contain '|'")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: outputs are immutable, choose a new path")
    pin = json.loads((Path(a.tokenizer) / "TOKENIZER_PIN.json").read_text())
    for fn, r in pin["files"].items():
        if sha256_file(Path(a.tokenizer) / fn) != r["sha256"]:
            raise SystemExit(f"tokenizer file {fn} differs from its pin")
    prod = Prod(a.code)
    workdir = Path(a.work)
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".partial")
    (tmp / "parts").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files, info = stage_input(a, workdir)
    if not files:
        raise SystemExit("no input jsonl")
    jobs = chunks(files, a.pool, tmp, a.lines_per_chunk)
    print(f"{len(files)} files, {len(jobs)} chunks, staged in {time.time() - t0:.0f}s", flush=True)
    agg, n_units = collections.defaultdict(collections.Counter), 0
    with mp.get_context("fork").Pool(a.procs, initializer=init_worker, initargs=(a.tokenizer, a.code)) as pool:
        for i, (ci, part_agg, n) in enumerate(pool.imap_unordered(work, jobs)):
            n_units += n
            for k, v in part_agg.items():
                agg[k].update(v)
            if i % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} chunks, {n_units} units, {time.time() - t0:.0f}s", flush=True)
    final = tmp / f"units_tokens_{a.pool}.jsonl.gz"
    with open(final, "wb") as w:
        for j in jobs:
            p = tmp / "parts" / f"{j[0]:07d}.jsonl.gz"
            with open(p, "rb") as r:
                for b in iter(lambda: r.read(1 << 22), b""):
                    w.write(b)
            p.unlink()
    (tmp / "parts").rmdir()
    summ = summarize(agg, a.pool)
    summ.update(pool=a.pool, version=VERSION, renderers={"R1": R1_ID, "R2": R2_ID}, margin=MARGIN, fit_limits=FIT_LIMITS,
                order="stored identity order", splits=a.splits, units_table_md5=md5_file(final), seconds=round(time.time() - t0))
    (tmp / f"summary_{a.pool}.json").write_text(json.dumps(summ, indent=1) + "\n")
    import transformers, tokenizers
    inputs = {"pool": a.pool, "version": VERSION, "tool_sha256": sha256_file(__file__), "code": prod.hashes,
              "tokenizer_pin": {"repo": pin["repo"], "revision": pin["revision"],
                                "files": {k: v["sha256"] for k, v in pin["files"].items()}},
              "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, **info}
    (tmp / f"inputs_{a.pool}.json").write_text(json.dumps(inputs, indent=1) + "\n")
    if not a.no_parity:
        par = parity(a, files, tmp, prod)
        (tmp / f"parity_{a.pool}.json").write_text(json.dumps(par, indent=1) + "\n")
        print("PARITY", json.dumps({k: v for k, v in par.items() if k != "mismatch_examples"}), flush=True)
        if not par["pass"]:
            print("PARITY_FAILED", par["mismatch_examples"][:5], flush=True)
    os.replace(tmp, out)
    d = summ["derived"]
    print("DONE", json.dumps({k: d[k] for k in ("units", "trainer_kept_units", "tokens_r1", "tokens_r2", "mean_r1",
                                                  "kept_not_fitting_without_state_cut_r1", "noul_def_units", "dup_exact_units")}),
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
