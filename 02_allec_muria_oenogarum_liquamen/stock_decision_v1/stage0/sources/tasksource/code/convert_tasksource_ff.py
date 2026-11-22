#!/usr/bin/env python3
"""stock_decision_v1 Stage 0: convert Tasksource `filtered-full` (pinned parquet shards) to the jev record format read by the
production reader (trainers/cache_features.py 49613a40 question_options / iter_units) and scorer (claude_lmscore.py e0672bd0).

One record per upstream row: id = upstream id, base_id = group_id, source = "tasksource/<source>", one question keyed by the
upstream question_id. Kinds:
  choice -> {"type": "choice", "instructions": question, "criteria": {option_text: option_text}}  (semantic key = exact option text)
  score  -> {"type": "score",  "instructions": question, "levels": [option texts in stored ordinal order]}   (labels keyed by text)
  noul   -> {"type": "noul",   "instructions": question}   upstream target [p_yes] -> {"yes": p, "no": 1 - p}  (no definitions upstream)
Targets are copied exactly (choice / score: aligned to options; noul: p and 1 - p). label_kind "soft" when more than one option
has positive mass, else "hard". Rows that fail a structural check are written to rejects.jsonl with the reason; nothing is fixed.

  python3 convert_tasksource_ff.py --shards DIR --pin PIN.json --sources-yaml SOURCES.yaml --out OUTDIR/tasksource_ff_v1
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import re

import pyarrow.parquet as pq
import yaml

SCHEMA = "stock-decision-tasksource-conversion-v1"


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_") or "unnamed"


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def convert_row(r, pin, ds_of):
    kind, opts, tgt = r["kind"], list(r["options"] or []), [float(x) for x in (r["target"] or [])]
    qid = str(r["question_id"])
    if not all(math.isfinite(x) and x >= 0 for x in tgt):
        return None, "target_nonfinite_or_negative"
    if kind in ("choice", "score"):
        if len(opts) < 2:
            return None, f"{kind}_K_lt_2"
        if len(tgt) != len(opts):
            return None, "target_option_length_mismatch"
        if len(set(opts)) != len(opts):
            return None, "duplicate_option_text"
        if any(not str(o).strip() for o in opts):
            return None, "empty_option_text"
        s = sum(tgt)
        if abs(s - 1.0) > 1e-6:
            return None, "target_sum_not_1"
        lab = {o: p for o, p in zip(opts, tgt)}
        q = {"type": "choice", "instructions": r["question"], "criteria": {o: o for o in opts}} if kind == "choice" else \
            {"type": "score", "instructions": r["question"], "levels": opts}
        soft = sum(1 for p in tgt if p > 0) > 1
    elif kind == "noul":
        if len(opts) != 0 or len(tgt) != 1:
            return None, "noul_shape"
        p = tgt[0]
        if p > 1.0 + 1e-12:
            return None, "noul_p_gt_1"
        p = min(p, 1.0)
        lab = {"yes": p, "no": 1.0 - p}
        q = {"type": "noul", "instructions": r["question"]}
        soft = 0.0 < p < 1.0
    else:
        return None, f"unknown_kind_{kind}"
    src = r["source"]
    ds = ds_of.get(src) or {}
    rec = {"id": r["id"], "base_id": r["group_id"], "source": f"tasksource/{src}", "state": r["state"],
           "questions": {qid: q}, "labels": {qid: lab}, "label_kind": {qid: "soft" if soft else "hard"},
           "license": r["license"],
           "meta": {"schema_version": SCHEMA, "upstream_repo": pin["repo"], "upstream_revision": pin["data_revision"],
                    "upstream_config": "filtered-full", "historical_source_revision": pin["historical_source_revision"],
                    "upstream_id": r["id"], "group_id": r["group_id"], "question_id": qid,
                    "parent_key": sha256_text(f"{r['group_id']}|{r['state']}"), "tasksource_source": src,
                    "upstream_dataset": ds.get("dataset"), "upstream_dataset_revisions": ds.get("revisions"),
                    "kind": kind, "variant": r["variant"], "license_use": r["license_use"], "upstream_split": r["split"],
                    "raw_options": opts, "raw_target": tgt,
                    "target_basis": "source_label_soft" if soft else "source_label_hard"}}
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--sources-yaml", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pin = json.load(open(a.pin))
    sy = yaml.safe_load(open(a.sources_yaml))
    ds_of = {k: {"dataset": v.get("dataset"), "revisions": v.get("revisions")} for k, v in (sy.get("sources") or {}).items()}
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(os.path.join(a.out, "train"))
    handles, cnt, rej, per = {}, collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    rejf = open(os.path.join(a.out, "..", "rejects_tasksource_ff_v1.jsonl"), "w")
    shards = sorted(glob.glob(os.path.join(a.shards, "*.parquet")))
    for sp in shards:
        t = pq.read_table(sp)
        cols = {c: t.column(c).to_pylist() for c in t.schema.names}
        for i in range(t.num_rows):
            r = {c: cols[c][i] for c in cols}
            cnt["rows_in"] += 1
            rec, why = convert_row(r, pin, ds_of)
            if rec is None:
                rej[why] += 1
                per[r["source"]]["rejected"] += 1
                rejf.write(json.dumps({"id": r["id"], "source": r["source"], "reason": why, "shard": os.path.basename(sp), "row": i}) + "\n")
                continue
            s = slug(r["source"])
            if s not in handles:
                handles[s] = open(os.path.join(a.out, "train", s + ".jsonl"), "w", encoding="utf-8")
            handles[s].write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            cnt["records_out"] += 1
            cnt[f"kind_{r['kind']}"] += 1
            per[r["source"]]["records"] += 1
            if rec["label_kind"][r["question_id"]] == "soft":
                cnt["soft"] += 1
    for h in handles.values():
        h.close()
    rejf.close()
    slugs = collections.Counter(slug(s) for s in per)
    if any(v > 1 for v in slugs.values()):
        raise SystemExit(f"slug collision: {[k for k, v in slugs.items() if v > 1]}")
    man = {"schema": SCHEMA, "pin": pin, "shards": [os.path.basename(s) for s in shards], "counts": dict(cnt), "rejects": dict(rej),
           "files": len(handles), "converter_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, sort_keys=True)
    json.dump({k: dict(v) for k, v in sorted(per.items())}, open(os.path.join(a.out, "counts.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({"counts": dict(cnt), "rejects": dict(rej), "files": len(handles)}))


if __name__ == "__main__":
    main()
