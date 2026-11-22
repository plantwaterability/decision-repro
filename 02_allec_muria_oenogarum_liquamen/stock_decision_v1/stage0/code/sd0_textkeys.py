#!/usr/bin/env python3
"""Stage 0 candidate-side identity keys (stock_decision_v1). One pass over a candidate pool (same file / line / qid enumeration as
unittable_v2, so uids join every other table), writing per RECORD:
  rk = "<pool>|<member>|<line>", lineage key (meta.parent_key / group_id / base_id / id), parent key,
  state_hash  (xxh64 of the normalised whole state, 0 if < 8 words),
  text units  (xxh64 of every normalised line / sentence with >= 12 words, from the state AND every question's instructions and
               option descriptions, i.e. all model-visible text; tagged by field),
  minhash     (48 x uint32 over normalised 8-word shingles of the state, rows of zeros if < 20 words),
and per DECISION (record x qid): dec_hash = h(state | instructions | type | option keys + texts | gold rounded to 1e-6) and
dec_nogold (the same without gold) for exact-duplicate and conflicting-target detection.
Normalisation: NFKC, casefold, every non-word run -> one space. Nothing here reads labels except to hash gold for dec_hash.

  python3 sd0_textkeys.py --input POOL.tgz|DIR --pool TAG --splits train --unittable unittable_v2.py --reader cache_features.py --out DIR
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import xxhash

NPERM, SHINGLE, MIN_TU_WORDS, MIN_STATE_WORDS, MIN_MH_WORDS = 48, 8, 12, 8, 20
P31 = (1 << 31) - 1
_rng = np.random.default_rng(20261013)
A_ = _rng.integers(1, P31, NPERM, dtype=np.int64)
B_ = _rng.integers(0, P31, NPERM, dtype=np.int64)
WRE = re.compile(r"[^\w]+", re.UNICODE)
SENT = re.compile(r"(?<=[.!?;:])\s+")
FIELDS = {"state": 1, "instructions": 2, "criteria": 3, "options": 4}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    return " ".join(WRE.sub(" ", unicodedata.normalize("NFKC", s).casefold()).split())


def as_text(x):
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)


def units_of(text):
    out = set()
    for line in str(text).splitlines():
        n = norm(line)
        if len(n.split()) >= MIN_TU_WORDS:
            out.add(n)
            if len(n.split()) > 60:
                for s in SENT.split(line):
                    m = norm(s)
                    if len(m.split()) >= MIN_TU_WORDS:
                        out.add(m)
    return out


def minhash(words):
    if len(words) < MIN_MH_WORDS:
        return np.zeros(NPERM, dtype=np.uint32)
    sh = np.fromiter((xxhash.xxh64_intdigest(" ".join(words[i:i + SHINGLE]).encode()) % P31 for i in range(len(words) - SHINGLE + 1)),
                     dtype=np.int64)
    sh = np.unique(sh)
    return (((A_[:, None] * sh[None, :]) + B_[:, None]) % P31).min(axis=1).astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--unittable", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    UT = load(a.unittable, "unittable_v2")
    R = UT.load_reader(a.reader)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"keys_{a.pool}.npz"
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    rks, lin, par, sh, mh, tu_h, tu_r, tu_f, tu_w = [], [], [], [], [], [], [], [], []
    ndec = 0
    with gzip.open(out / f"dec_{a.pool}.partial.jsonl.gz", "wt", encoding="utf-8") as dg:
        for mem, f in UT.iter_files(a.input, set(a.splits.split(","))):
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
                ri = len(rks)
                rks.append(f"{a.pool}|{mem}|{line_no}")
                lineage = str(meta.get("group_id") or rec.get("base_id") or rec.get("id"))
                parent = str(meta.get("parent_key") or rec.get("base_id") or rec.get("id"))
                lin.append(lineage)
                par.append(parent)
                st = as_text(rec.get("state"))
                ns = norm(st)
                words = ns.split()
                sh.append(xxhash.xxh64_intdigest(ns.encode()) if len(words) >= MIN_STATE_WORDS else 0)
                mh.append(minhash(words))
                seen = {}
                for u in units_of(st):
                    seen.setdefault(u, FIELDS["state"])
                for qid, q in (rec.get("questions") or {}).items():
                    instr = as_text(q.get("instructions") or q.get("question") or "")
                    for u in units_of(instr):
                        seen.setdefault(u, FIELDS["instructions"])
                    crit = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
                    for v in crit.values():
                        for u in units_of(as_text(v)):
                            seen.setdefault(u, FIELDS["criteria"])
                    try:
                        keys, texts, target = R["question_options"](q, rec["labels"][qid])
                    except Exception:                       # noqa: BLE001 -- loader errors are excluded via the unit table
                        keys, texts, target = [], [], []
                    for t in texts:
                        for u in units_of(as_text(t)):
                            seen.setdefault(u, FIELDS["options"])
                    tsum = sum(target) or 1.0
                    body = "\x1f".join([ns, norm(instr), str(q.get("type")), "\x1e".join(f"{k}\x1d{norm(as_text(t))}" for k, t in zip(keys, texts))])
                    gold = "\x1e".join(f"{v / tsum:.6f}" for v in target)
                    dg.write(json.dumps({"uid": f"{a.pool}|{mem}|{line_no}|{qid}", "ri": ri,
                                         "dec": hashlib.sha256((body + "\x1c" + gold).encode()).hexdigest()[:20],
                                         "dec_nogold": hashlib.sha256(body.encode()).hexdigest()[:20]}, separators=(",", ":")) + "\n")
                    ndec += 1
                for u, fld in seen.items():
                    tu_h.append(xxhash.xxh64_intdigest(u.encode()))
                    tu_r.append(ri)
                    tu_f.append(fld)
                    tu_w.append(min(len(u.split()), 32767))
    os.replace(out / f"dec_{a.pool}.partial.jsonl.gz", out / f"dec_{a.pool}.jsonl.gz")
    np.savez(str(dst) + ".partial.npz", state_hash=np.array(sh, dtype=np.uint64), minhash=np.stack(mh) if mh else np.zeros((0, NPERM), np.uint32),
             tu_hash=np.array(tu_h, dtype=np.uint64), tu_rec=np.array(tu_r, dtype=np.int32), tu_field=np.array(tu_f, dtype=np.int8), tu_words=np.array(tu_w, dtype=np.int16))
    os.replace(str(dst) + ".partial.npz", dst)
    with gzip.open(out / f"recs_{a.pool}.jsonl.gz", "wt", encoding="utf-8") as g:
        for rk, l_, p_ in zip(rks, lin, par):
            g.write(json.dumps([rk, l_, p_], ensure_ascii=False) + "\n")
    print(json.dumps({"pool": a.pool, "records": len(rks), "decisions": ndec, "text_units": len(tu_h),
                      "state_hashed": int(sum(1 for x in sh if x)), "minhashed": int(sum(1 for m in mh if m.any()))}))


if __name__ == "__main__":
    main()
