#!/usr/bin/env python3
"""Build the clean karpo-distilled training JSONL for t4b1 directly from S3 structured requests.

Sources (all Apache-2.0 karpo-decision-31B soft targets, ZERO DMCA / ZERO NC-SA / ZERO droplist):
  1. fs5 (359,747 rows) minus droplist_roundH (LOOMD-phish_v5, train_hover), with native_ns
     replacing choice-coerced noul/score rows in their true native form (8,438 noul + 257 score).
  2. fs5h0 (60,984 rows) minus LOOMD-chess_train (26,586), bv1_discovery (400), droplist_roundH.
  3. fs5h (51,697 rows) minus fs5h_hellaswag (12,000, DMCA) and fs5h_cfcolor (4,841, CC BY-NC-SA).
  4. fs5h_multi (12 clean benchmark-shaped families from k25_families): up to --multi-per-req
     deterministic questions per multi-question request with full sibling-question preview.

Output JSONL row schema:
  {"id", "src", "kind", "state", "preview", "instructions", "options", "p", "w"}
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4b_layout as LYT

EXCLUDED_FAMILIES = {
    "LOOMD-chess_train",
    "bv1_discovery",
    "fs5h_hellaswag",
    "fs5h_cfcolor",
    "LOOMD-phish_v5",
    "train_hover",
}

MULTI_FAMILIES = {
    "fs5h_sata_mrc",
    "fs5h_sata_goemo",
    "LOOMD-sata_v5",
    "train_sata_multirc",
    "LOOMD-bfcl_v7",
    "LOOMD-toolret_v73",
    "fs5_acos",
    "train_acos_laptop",
    "train_acos_rest16",
    "LOOMD-home_v5",
    "gap_supply/synth_appliance_sata_case",
    "train_contractnli",
}


def opn(p: str):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, "r", encoding="utf-8")


def norm_dist(vals: list[float]):
    s = sum(vals)
    if not vals or s <= 0 or any(v < -1e-9 or v != v for v in vals):
        return None
    return [round(max(0.0, float(v)) / s, 6) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--multi-per-req", type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    d = Path(a.src_dir)
    st = collections.Counter()
    by_src = collections.Counter()
    by_kind = collections.Counter()

    # 1. Droplist
    droplist = set()
    for l in open(d / "droplist_roundH.txt"):
        s = l.strip()
        if s and not s.startswith("#"):
            droplist.add(s)
    print(f"Loaded droplist: {len(droplist)} ids", flush=True)

    # 2. Metadata map (eid -> (source_family, weight)) from mxa.jsonl + fs5h_rows.jsonl
    meta = {}
    for fname in ("mxa.jsonl", "fs5h_rows.jsonl"):
        p = d / fname
        n_m = 0
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("manifest_header"):
                continue
            eid = r["exposure_id"]
            if eid not in meta:
                meta[eid] = (str(r.get("source_family") or "unknown"), float(r.get("weight", 1.0)))
                n_m += 1
        print(f"Loaded metadata from {fname}: +{n_m} (total {len(meta)}) in {time.time()-t0:.1f}s", flush=True)

    # 3. Native noul/score replacements (8,695 rows)
    native_teach = {}
    for l in opn(str(d / "teacher_karpo_native_ns.jsonl.gz")):
        r = json.loads(l)
        if "p" in r:
            native_teach[r["id"]] = r["p"]
    native_ids = set()
    native_rows = []
    for l in opn(str(d / "native_ns.jsonl.gz")):
        r = json.loads(l)
        eid = r["id"]
        if eid in droplist:
            st["drop_native_droplist"] += 1
            continue
        sf = str(r.get("source_family") or meta.get(eid, ("unknown", 1.0))[0])
        if sf in EXCLUDED_FAMILIES:
            st["drop_native_excluded_fam"] += 1
            continue
        req = r["request"]
        if len(req["questions"]) != 1:
            st["drop_native_multiq"] += 1
            continue
        qid, qobj = next(iter(req["questions"].items()))
        tp = (native_teach.get(eid) or {}).get(qid)
        if tp is None:
            st["drop_native_no_teacher"] += 1
            continue
        keys, opt_texts = LYT.options_of(qobj)
        if set(tp) != set(keys) or len(keys) < 2 or len(keys) > LYT.N_CODES:
            st["drop_native_bad_keys"] += 1
            continue
        p = norm_dist([float(tp[k]) for k in keys])
        if p is None:
            st["drop_native_bad_p"] += 1
            continue
        w = meta.get(eid, (sf, 1.0))[1]
        ins = LYT.instructions_of(qobj)
        native_ids.add(eid)
        native_rows.append({
            "id": eid,
            "src": sf,
            "kind": qobj["type"],
            "state": LYT.describe(req["state"]),
            "preview": LYT.question_preview(req["questions"]),
            "instructions": ins,
            "options": opt_texts,
            "p": p,
            "w": round(w, 4),
        })
    print(f"Prepared native noul/score rows: {len(native_rows)} in {time.time()-t0:.1f}s", flush=True)

    # 4. Single-question pools: fs5, fs5h0, fs5h
    written = set()
    out_f = open(a.out, "w", encoding="utf-8")

    def emit(row: dict):
        if row["src"] in EXCLUDED_FAMILIES:
            raise RuntimeError(f"Contaminated source attempted: {row['src']}")
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        by_src[row["src"]] += 1
        by_kind[row["kind"]] += 1
        st["written"] += 1

    # Write native noul/score first so they supersede any choice-coerced twin
    for row in native_rows:
        written.add(row["id"])
        emit(row)
        st["from_native_ns"] += 1

    single_pools = [
        ("fs5", "fs5.jsonl.gz", "teacher_karpo_fs5.jsonl.gz"),
        ("fs5h0", "fs5h0.jsonl.gz", "teacher_karpo_fs5h0.jsonl.gz"),
        ("fs5h", "fs5h.jsonl.gz", "teacher_karpo_fs5h.jsonl.gz"),
    ]
    for pname, rfile, tfile in single_pools:
        teach = {}
        for l in opn(str(d / tfile)):
            r = json.loads(l)
            if "p" in r:
                teach[r["id"]] = r["p"]
        n_ok = 0
        for l in opn(str(d / rfile)):
            r = json.loads(l)
            eid = r["id"]
            if eid in written:
                st[f"skip_{pname}_dup_or_native"] += 1
                continue
            if eid in droplist:
                st[f"drop_{pname}_droplist"] += 1
                continue
            sf, w = meta.get(eid, ("unknown", 1.0))
            if sf in EXCLUDED_FAMILIES:
                st[f"drop_{pname}_excluded_fam"] += 1
                continue
            # Belt-and-suspenders ID checks for excluded pools
            if "CHESSD:" in eid or "|hellaswag|" in eid or "|cfcolor|" in eid:
                st[f"drop_{pname}_excluded_id"] += 1
                continue
            tp = teach.get(eid)
            if tp is None:
                st[f"drop_{pname}_no_teacher"] += 1
                continue
            req = r["request"]
            if len(req["questions"]) != 1:
                st[f"drop_{pname}_multiq"] += 1
                continue
            qid, qobj = next(iter(req["questions"].items()))
            keys, opt_texts = LYT.options_of(qobj)
            if set(tp) != set(keys) or len(keys) < 2 or len(keys) > LYT.N_CODES:
                st[f"drop_{pname}_bad_keys"] += 1
                continue
            p = norm_dist([float(tp[k]) for k in keys])
            if p is None:
                st[f"drop_{pname}_bad_p"] += 1
                continue
            written.add(eid)
            emit({
                "id": eid,
                "src": sf,
                "kind": qobj["type"],
                "state": LYT.describe(req["state"]),
                "preview": LYT.question_preview(req["questions"]),
                "instructions": LYT.instructions_of(qobj),
                "options": opt_texts,
                "p": p,
                "w": round(w, 4),
            })
            n_ok += 1
            st[f"from_{pname}"] += 1
        print(f"Pool {pname}: +{n_ok} rows (total {st['written']}) in {time.time()-t0:.1f}s", flush=True)

    # 5. Multi-question grouped pool (fs5h_multi) with full sibling preview
    mteach = {}
    for l in opn(str(d / "teacher_karpo_fs5h_multi.jsonl.gz")):
        r = json.loads(l)
        if "p" in r:
            mteach[r["id"]] = r["p"]
    n_mreq, n_mq = 0, 0
    for l in opn(str(d / "fs5h_multi.jsonl.gz")):
        r = json.loads(l)
        rid = r["id"]
        sf = str(r.get("source_family") or "unknown")
        if sf not in MULTI_FAMILIES or sf in EXCLUDED_FAMILIES or rid in droplist:
            st["drop_multi_fam_or_list"] += 1
            continue
        tp_all = mteach.get(rid)
        if not tp_all:
            st["drop_multi_no_teacher"] += 1
            continue
        req = r["request"]
        qs = req["questions"]
        if len(qs) < 2:
            st["drop_multi_single"] += 1
            continue
        state_text = LYT.describe(req["state"])
        preview = LYT.question_preview(qs)
        qkeys = list(qs.keys())
        # Deterministically sample up to --multi-per-req questions from each multi-q request
        if len(qkeys) > a.multi_per_req:
            order = LYT.permute_order(len(qkeys), LYT.seed_of("t4b-multi-pick", rid))
            pick = [qkeys[j] for j in sorted(order[: a.multi_per_req])]
        else:
            pick = qkeys
        emitted_here = 0
        for qk in pick:
            qobj = qs[qk]
            tp = tp_all.get(qk)
            if tp is None:
                continue
            keys, opt_texts = LYT.options_of(qobj)
            if set(tp) != set(keys) or len(keys) < 2 or len(keys) > LYT.N_CODES:
                continue
            p = norm_dist([float(tp[k]) for k in keys])
            if p is None:
                continue
            row_id = f"{rid}::{qk}"
            if row_id in written:
                continue
            written.add(row_id)
            emit({
                "id": row_id,
                "src": f"multi:{sf}",
                "kind": qobj["type"],
                "state": state_text,
                "preview": preview,
                "instructions": LYT.instructions_of(qobj),
                "options": opt_texts,
                "p": p,
                "w": 1.0,
            })
            emitted_here += 1
            n_mq += 1
            st["from_multi"] += 1
        if emitted_here:
            n_mreq += 1
    out_f.close()

    # Hard sanity assertions
    for bad in EXCLUDED_FAMILIES:
        assert by_src.get(bad, 0) == 0, f"Excluded family leaked: {bad}"
    assert by_kind["noul"] >= 8000, f"Too few native noul rows: {by_kind['noul']}"
    assert by_kind["score"] >= 200, f"Too few native score rows: {by_kind['score']}"
    assert st["written"] >= 420000, f"Too few total rows: {st['written']}"

    # Content hash
    h = hashlib.sha256()
    with open(a.out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    summary = {
        "total_rows": st["written"],
        "sha256": h.hexdigest(),
        "by_kind": dict(by_kind),
        "stats": dict(st),
        "multi_requests_used": n_mreq,
        "multi_questions_emitted": n_mq,
        "top_sources": dict(by_src.most_common(40)),
        "n_sources": len(by_src),
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.stats).write_text(json.dumps(summary, indent=2))
    print("T4B_DATA_OK", json.dumps({k: summary[k] for k in ("total_rows", "by_kind", "multi_requests_used", "multi_questions_emitted", "n_sources", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
