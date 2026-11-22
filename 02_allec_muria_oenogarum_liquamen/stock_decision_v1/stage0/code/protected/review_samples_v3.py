#!/usr/bin/env python3
"""Recover candidate / protected text snippets (<= 300 chars) for the sampled strong hits in publish/hit_samples_v3.json so a reviewer can
judge false positives. Sealed populations are never sampled. Protected files > 300 MB are skipped (listed).

  PYTHONHASHSEED=0 python3 review_samples_v3.py --work W --pool POOL --input POOL_TGZ_OR_DIR --splits train,val
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--splits", default="train,val")
    a = ap.parse_args()
    W = Path(a.work)
    sys.path.insert(0, str(W / "code"))
    import evalindex_v3 as E
    E.load_v1(W / "code/code_v1")
    samples = [s for s in json.loads((W / "publish/hit_samples_v3.json").read_text()) if s["pool"] == a.pool]
    need = collections.defaultdict(list)
    for s in samples:
        _, mem, ln, qid = s["uid"].split("|")
        need[(mem, int(ln))].append(s)
    mems = {m for m, _ in need}
    for mem, lines in E.iter_members(a.input, set(a.splits.split(",")), 5000):
        if mem not in mems:
            continue
        for ln, line in lines:
            for s in need.get((mem, ln), []):
                rec = json.loads(line)
                qid = s["uid"].split("|")[3]
                s["_cand"] = texts_of(E, rec, qid, s["candidate_field"], "jev")
    stage = json.loads((W / "stage/stage_manifest.json").read_text())
    path = {(p, f.get("member") or Path(f["path"]).name): (f["path"], f["kind"]) for p, r in stage["populations"].items() for f in r.get("files", [])}
    byfile = collections.defaultdict(list)
    for s in samples:
        pop, fil = s["protected"].split("|", 1)
        byfile[(pop, fil)].append(s)
    skipped = []
    for (pop, fil), lst in byfile.items():
        p, kind = path.get((pop, fil), (None, None))
        if p is None or os.path.getsize(p) > 300e6:
            skipped.append(f"{pop}|{fil}")
            continue
        cs = []
        for s in lst:
            ct = s.get("_cand") or []
            sh = set()
            for t in ct:
                sh |= set(E.shingles(E.cnorm(t)).tolist())
            cs.append((s, sh, {k for t in ct for k in E.text_units(t)}))
        best = [(0.0, None) for _ in cs]
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for i, (s, sh, tu) in enumerate(cs):
                    for t in texts_of(E, rec, None, s["protected_field"], "rendered" if kind == "rendered" else "jev"):
                        if tu & set(E.text_units(t)):
                            sc = 2.0
                        elif sh:
                            psh = set(E.shingles(E.cnorm(t)).tolist())
                            sc = len(sh & psh) / max(1, min(len(sh), len(psh))) if psh else 0.0
                        else:
                            sc = 0.0
                        if sc > best[i][0]:
                            best[i] = (sc, t)
        for i, (s, _, _) in enumerate(cs):
            s["protected_snippet"] = (best[i][1] or "")[:300]
            s["match_score"] = round(best[i][0], 3)
    for s in samples:
        s["candidate_snippet"] = " || ".join(t[:300] for t in (s.pop("_cand", None) or [])[:2])[:600]
    out = W / f"publish/hit_samples_review_{a.pool}.json"
    out.write_text(json.dumps({"pool": a.pool, "skipped_large_files": skipped, "samples": samples}, indent=1, ensure_ascii=False) + "\n")
    print(out, len(samples))


def texts_of(E, rec, qid, field, kind):
    fid = E.FID.get(field)
    out = []
    if field == "state" and rec.get("state") is not None:
        flat, _ = E.flatten(rec["state"])
        out.append(flat)
    rf, _ = E.record_fields(rec, kind)
    out += [t for f, t in rf if f == fid]
    qs = rec.get("questions") if isinstance(rec.get("questions"), dict) else {}
    for q_, q in qs.items():
        if qid is not None and q_ != qid:
            continue
        out += [t for f, t in E.question_fields(q_, q) if f == fid]
    return out


if __name__ == "__main__":
    main()
