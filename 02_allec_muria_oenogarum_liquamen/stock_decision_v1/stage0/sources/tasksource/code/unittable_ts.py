#!/usr/bin/env python3
"""(stock_decision_v1 copy of unittable_v2.py with tasksource meta extras) Per-decision-unit table of a candidate pool, enumerated with the PRODUCTION reader's own functions (trainers/cache_features.py
question_options / is_noncommercial, AST-extracted so no torch import; md5 pinned). One row per (record, question) of every
train/val *.jsonl; a `confirmation/` split is never opened. uid = "<pool>|<member>|<line>|<qid>" (member relative to the pool
root, line 1-based) -- the key shared by the overlap scan, the tokenizer table and the exposure manifests.

  python3 unittable_v2.py --input s3://.../emo_pool_v1.tgz --pool emo_pool_v1 --reader /repo/trainers/cache_features.py --out DIR
"""
import argparse
import ast
import collections
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

READER_MD5 = "49613a40faed8cd612175e81c3f3925f"
SPLITS = ("train", "val", "validation", "dev", "test", "confirmation", "eval")


def load_reader(path):
    src = Path(path).read_bytes()
    if hashlib.md5(src).hexdigest() != READER_MD5:
        raise SystemExit(f"reader md5 differs from production {READER_MD5}")
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("question_options", "is_noncommercial"))
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("NC_LICENCE", "NC_SOURCES") for t in n.targets))]
    env = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), env)
    return env


def clean(name):
    return "/".join(x for x in name.split("/") if x not in (".", ""))


def member_of(name):
    parts = clean(name).split("/")
    for i, x in enumerate(parts[:-1]):
        if x in SPLITS:
            return "/".join(parts[i:])
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def iter_files(spec, splits):
    def ok(mem):
        return mem.endswith(".jsonl") and mem.split("/")[0] in splits and mem.split("/")[0] != "confirmation" \
            and not os.path.basename(mem).startswith("._")
    if spec.startswith("s3://") or spec.endswith(".tgz"):
        cmd = ["aws", "s3", "cp", "--only-show-errors", spec, "-"] if spec.startswith("s3://") else ["cat", spec]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        with tarfile.open(fileobj=p.stdout, mode="r|gz") as tf:
            for m in tf:
                if m.isfile() and ok(member_of(m.name)):
                    yield member_of(m.name), io.StringIO(tf.extractfile(m).read().decode("utf-8"))   # stream members are not seekable
        p.wait()
    else:
        root = Path(spec)
        for p in sorted(root.rglob("*.jsonl")):
            mem = str(p.relative_to(root))
            if ok(mem):
                with open(p, encoding="utf-8") as f:
                    yield mem, f


def noul_defs(crit):
    if not isinstance(crit, dict) or not crit:
        return False
    low = {str(k).strip().lower(): v for k, v in crit.items()}
    return any(low.get(k) is not None and str(low[k]).strip() for k in ("yes", "true", "no", "false"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = load_reader(a.reader)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"units_{a.pool}.jsonl.gz"
    if dst.exists():
        raise SystemExit(f"{dst} exists (immutable)")
    tot = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    with gzip.open(out / f"units_{a.pool}.partial.jsonl.gz", "wt", encoding="utf-8") as o:
        for mem, f in iter_files(a.input, set(a.splits.split(","))):
            split, stem = mem.split("/")[0], Path(mem).stem
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
                st = rec.get("state")
                schars = len(st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)) if st is not None else 0
                for qid, q in (rec.get("questions") or {}).items():
                    row = {"uid": f"{a.pool}|{mem}|{line_no}|{qid}", "split": split, "stem": stem,
                           "source": rec.get("source") or stem, "record_id": str(rec.get("id")),
                           "base_id": str(rec.get("base_id") or meta.get("variant_of") or rec.get("id")),
                           "group": str(meta.get("group_id") or ((meta.get("lineage") or {}).get("root") if isinstance(meta.get("lineage"), dict) else None)
                                        or meta.get("group") or ""),
                           "license": str(rec.get("license") or meta.get("license") or ""), "qtype": q.get("type"), "state_chars": schars}
                    row["nc"] = bool(R["is_noncommercial"]({"license": row["license"], "source": row["source"]}))
                    try:
                        keys, texts, target = R["question_options"](q, rec["labels"][qid])
                        lk = rec["label_kind"][qid]
                        if lk == "hard" and sum(1 for v in target if v > 0) > 1:
                            lk = "soft"
                        tsum = sum(target)
                        row.update(K=len(keys), label_kind=lk, tmax=round(max(target) / tsum, 6) if tsum > 0 else None,
                                   dup_opt=len(set(" ".join(str(t).split()) for t in texts)) < len(texts),
                                   noul_def=q.get("type") == "noul" and noul_defs(q.get("criteria")))
                    except Exception as e:                   # noqa: BLE001 -- the trainer would die on this unit: record it
                        row["loader_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                    for k in ("n_raters", "family", "generator", "numeric_family", "target_basis", "evidence_position", "template_id",
                              "label_method", "label_source", "di_source_training", "external_benchmark",
                              "upstream_id", "question_id", "parent_key", "tasksource_source", "upstream_dataset", "license_use", "kind",
                              "variant"):            # (stock_decision_v1 tasksource extras)
                        v = meta.get(k)
                        if v is not None and not isinstance(v, (dict, list)):
                            row[k] = v
                    tb = (meta.get("curation_v1") or {}).get("target_basis") if isinstance(meta.get("curation_v1"), dict) else None
                    if tb is not None and "target_basis" not in row:
                        row["target_basis"] = tb if not isinstance(tb, dict) else tb.get(qid)
                    o.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    tot["units"] += 1
                    per[stem]["units"] += 1
                    if row.get("loader_error"):
                        tot["loader_error"] += 1
                    if row["nc"]:
                        tot["nc"] += 1
                        per[stem]["nc"] += 1
                    if row.get("K", 0) > 52:
                        tot["k_gt_52"] += 1
                tot["records"] += 1
    os.replace(out / f"units_{a.pool}.partial.jsonl.gz", dst)
    h = hashlib.md5(dst.read_bytes()).hexdigest()
    summ = {"pool": a.pool, "input": a.input, "reader_md5": READER_MD5, "totals": dict(tot),
            "per_stem": {k: dict(v) for k, v in sorted(per.items())}, "units_file": dst.name, "units_md5": h}
    (out / f"units_{a.pool}.summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pool": a.pool, "totals": dict(tot), "units_md5": h}))


if __name__ == "__main__":
    main()
