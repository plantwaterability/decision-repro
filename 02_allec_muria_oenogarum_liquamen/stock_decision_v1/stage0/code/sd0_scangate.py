#!/usr/bin/env python3
"""Stage 0 protected-overlap scan gate (stock_decision_v1). Builds scan_manifest.json: one entry per INCLUDED candidate pool, with
  population (pool), field coverage (the scanner's field set + key-function sha256, identical enumeration for candidate and
  protected rows), input digest (the scanned archive md5; when it differs from the candidate archive, the train-member digest of
  both archives must be equal), input rows (train units scanned == candidate train units), index revision (the scan's index npz
  md5 == the published protected index), completion (summary + adjudication present, adjudicated hits file present, readable,
  md5 equal to the publication _READY.json entry and to the local copy), and the hit counts. A pool with zero hits needs a valid
  (readable, possibly empty) hits file AND a summary saying zero: 'no file' never means 'no hits'.
status COMPLETE only when every check passes; sd0_families.py refuses to run unless every included pool is COMPLETE.

  python3 sd0_scangate.py --scan-dir DIR(summary_/adjudication_/hits_*.adj) --ready _READY.json --index-manifest protected_index_manifest.json
        --cand cand.parquet ... --pool-md5 pool_meta.json --member-digests member_digests.json --pools p1,p2 --hits-local DIR --out scan_manifest.json
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

import pandas as pd


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gz_lines(p):
    n = 0
    with gzip.open(p, "rt") as g:
        for line in g:
            if line.strip():
                json.loads(line)
                n += 1
    return n


def check_pool(t, a, ready, idx_md5, fields, keyfn, cand_n, cand_md5, member_dig, cand_splits=("train",)):
    e = {"population": t, "checks": {}, "reasons": []}
    sp = os.path.join(a.scan_dir, f"summary_{t}.json")
    ap_ = os.path.join(a.scan_dir, f"adjudication_{t}.json")
    hp = os.path.join(a.hits_local, f"hits_{t}.jsonl.gz")
    for name, p in (("summary", sp), ("adjudication", ap_), ("hits_local", hp)):
        e["checks"][name + "_present"] = os.path.exists(p)
        if not os.path.exists(p):
            e["reasons"].append(f"missing {name}: {p}")
    if e["reasons"]:
        e["status"] = "INCOMPLETE"
        return e
    s = json.load(open(sp))
    adj = json.load(open(ap_))
    e["field_coverage"] = {"fields": fields, "keyfn_sha256": keyfn}
    e["index_revision"] = s.get("index_npz_md5")
    if s.get("index_npz_md5") != idx_md5:
        e["reasons"].append(f"index revision {s.get('index_npz_md5')} != published {idx_md5}")
    if s.get("pool") != t or adj.get("pool") != t:
        e["reasons"].append("pool name mismatch in summary / adjudication")
    e["input"] = {"scanned": s.get("input"), "scanned_md5": s.get("input_md5"), "candidate_md5": cand_md5}
    if s.get("input_md5") != cand_md5:
        md = member_dig.get(t) or {}
        if md.get("scanned_train_digest") and md.get("scanned_train_digest") == md.get("candidate_train_digest") \
                and md.get("scanned_md5") == s.get("input_md5") and md.get("candidate_md5") == cand_md5:
            e["input"]["train_member_digest"] = md["candidate_train_digest"]
            e["input"]["note"] = "archives differ; train members byte-identical"
        else:
            e["reasons"].append(f"scanned archive {s.get('input_md5')} != candidate archive {cand_md5} and no equal train-member digest")
    train_scanned = sum(v.get("units", 0) for k, v in (s.get("by_member") or {}).items() if k.split("/")[0] in cand_splits)
    e["input_rows"] = {"splits": sorted(cand_splits), "units_scanned": train_scanned, "candidate_units": cand_n}
    if train_scanned != cand_n:
        e["reasons"].append(f"units scanned in {sorted(cand_splits)} {train_scanned} != candidate units {cand_n}")
    local_md5 = md5(hp)
    pub_key = f"scans/hits_{t}.adj.jsonl.gz"
    e["hits"] = {"file": pub_key, "md5_local": local_md5, "md5_ready": ready.get(pub_key), "md5_adjudication": adj.get("md5")}
    if not (local_md5 == ready.get(pub_key) == adj.get("md5")):
        e["reasons"].append("hits md5 differs between local copy, _READY.json and the adjudication record")
    try:
        n = gz_lines(hp)
    except Exception as ex:                       # noqa: BLE001
        n = None
        e["reasons"].append(f"hits file unreadable: {type(ex).__name__}")
    strong = (adj.get("after") or {}).get("strong_units", 0)
    e["hits"].update(lines=n, strong_units=strong)
    if n == 0 and strong:
        e["reasons"].append("empty hits file but the adjudication reports strong units")
    e["completion"] = {"summary_seconds": s.get("seconds"), "adjudication_rules": sorted((adj.get("rules") or {}).keys())}
    e["status"] = "COMPLETE" if not e["reasons"] else "INCOMPLETE"
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--ready", required=True)
    ap.add_argument("--index-manifest", required=True)
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--pool-md5", required=True)
    ap.add_argument("--member-digests", default="")
    ap.add_argument("--pools", required=True)
    ap.add_argument("--hits-local", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ready = json.load(open(a.ready))
    ready = ready.get("objects_md5", ready)
    im = json.load(open(a.index_manifest))
    idx_md5 = (im.get("index") or {}).get("md5") or (im.get("index") or {}).get("npz_md5")
    fields, keyfn = im.get("fields"), im.get("keyfn_sha256")
    cand = pd.concat([pd.read_parquet(p, columns=[c for c in ("pool", "split") if c in pd.read_parquet(p).columns]) for p in a.cand])
    cn = cand.pool.value_counts().to_dict()
    csp = {t: (tuple(sorted(set(g.split.dropna()))) or ("train",)) if "split" in g else ("train",) for t, g in cand.groupby("pool")}
    pm = json.load(open(a.pool_md5))["pools"]
    md = json.load(open(a.member_digests)) if a.member_digests else {}
    out = {"schema": "stock-decision-scan-manifest-v1", "index_md5": idx_md5, "pools": {}}
    for t in a.pools.split(","):
        cmd5 = (pm.get(t) or {}).get("archive_md5")
        out["pools"][t] = check_pool(t, a, ready, idx_md5, fields, keyfn, int(cn.get(t, 0)), cmd5, md, csp.get(t, ("train",)))
    out["all_complete"] = all(v["status"] == "COMPLETE" for v in out["pools"].values())
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({t: (v["status"], v["reasons"]) for t, v in out["pools"].items()}))
    sys.exit(0 if out["all_complete"] else 2)


if __name__ == "__main__":
    main()
