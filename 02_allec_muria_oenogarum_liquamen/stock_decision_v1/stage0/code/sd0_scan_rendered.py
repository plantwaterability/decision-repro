#!/usr/bin/env python3
"""Protected-index scan of RENDERED blocks (stock_decision_v1): U arms, M2 parts, mined rows -- anything delivered as Stage 0 decision
rows (sd0_render.py fields) rather than as pool records.

  prep     every decision row -> one jev record holding exactly its model-visible fields: state, the question's instructions, the
           option texts (choice / score -> option_text, noul -> noul_defs); id = source_record_id (record-id identities are checked
           too). Written to DIR/<block>/train/<block>.jsonl with uidmap.jsonl.gz (line -> decision_uid). Labels are not copied.
  report   after `evalindex_v3.py scan` + `adjudicate` on DIR/<block> (pool tag <block>): per-block scan manifest -- COMPLETE iff every
           decision was scanned as exactly one unit (units == decisions, no record without a question); STRONG / weak-only counts
           after the T1 / T2 template rules, STRONG decision uids, protected (population, file) ids only (no protected text is read).
           Optional --stage0-adj: STRONG rows whose decision uid is a Stage 0 unit are also looked up in that unit's own Stage 0
           adjudication (source-pool DF) and reported side by side.

  python3 sd0_scan_rendered.py prep --block NAME=DECISIONS.jsonl.gz [...] --out DIR
  python3 sd0_scan_rendered.py report --block NAME --dir DIR [--stage0-adj POOL=hits_POOL.adj.jsonl.gz ...] --out manifest.json"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys


def rows(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def to_record(d):
    opts = d["ordered_options"]
    if isinstance(opts, str):
        opts = json.loads(opts)
    crit = {str(o["key"]): str(o.get("text") if o.get("text") not in (None, "") else o["key"]) for o in opts}
    kind = str(d.get("kind") or "choice")
    return {"id": str(d.get("source_record_id") or d["decision_uid"]), "state": d.get("state") or "",
            "questions": {"q": {"type": kind, "instructions": d.get("question") or "", "criteria": crit}}}


def prep(a):
    out = {}
    for spec in a.block:
        name, path = spec.split("=", 1)
        bd = os.path.join(a.out, name)
        os.makedirs(os.path.join(bd, "train"), exist_ok=True)
        n = 0
        seen = set()
        with open(os.path.join(bd, "train", f"{name}.jsonl"), "w", encoding="utf-8") as f, \
                gzip.open(os.path.join(bd, "uidmap.jsonl.gz"), "wt", encoding="utf-8") as m:
            for d in rows(path):
                if d["decision_uid"] in seen:
                    raise SystemExit(f"{name}: duplicate decision uid {d['decision_uid']}")
                seen.add(d["decision_uid"])
                n += 1
                f.write(json.dumps(to_record(d), ensure_ascii=False, sort_keys=True) + "\n")
                m.write(json.dumps({"line": n, "decision_uid": d["decision_uid"]}) + "\n")
        out[name] = {"decisions": n, "input": path, "input_md5": hashlib.md5(open(path, "rb").read()).hexdigest()}
        json.dump(out[name], open(os.path.join(bd, "prep.json"), "w"), indent=1)
    print(json.dumps(out))


def report(a):
    bd = os.path.join(a.dir, a.block)
    pr = json.load(open(os.path.join(bd, "prep.json")))
    sd = os.path.join(bd, "scan")
    summ = json.load(open(os.path.join(sd, f"summary_{a.block}.json")))
    adj = json.load(open(os.path.join(sd, f"adjudication_{a.block}.json")))
    umap = {r["line"]: r["decision_uid"] for r in rows(os.path.join(bd, "uidmap.jsonl.gz"))}
    strong, weak = [], 0
    files = collections.Counter()
    kinds = collections.Counter()
    for r in rows(os.path.join(sd, f"hits_{a.block}.adj.jsonl.gz")):
        line = int(r["uid"].split("|")[2])
        if r["s"]:
            strong.append(umap[line])
            kinds.update(r["s"])
            for m_ in r["m"]:
                if m_[0] in r["s"]:
                    files[f"{m_[2]}|{m_[3]}"] += 1
        else:
            weak += 1
    t = summ["totals"]
    complete = t.get("units", 0) == pr["decisions"] and not t.get("records_without_questions") and len(umap) == pr["decisions"]
    s0 = {}
    if strong and a.stage0_adj:
        st0 = {}
        for spec in a.stage0_adj:
            pool, p = spec.split("=", 1)
            for r in rows(p):
                st0[r["uid"]] = r["s"]
        for u in strong:
            s0[u] = ("not a Stage 0 unit" if u.split("|", 1)[0] not in {x.split("=", 1)[0] for x in a.stage0_adj}
                     else ("strong in source pool" if st0.get(u) else ("weak / template in source pool" if u in st0 else "no hit in source pool")))
    man = {"schema": "stock-decision-rendered-scan-manifest-v1", "block": a.block, "input": pr["input"], "input_md5": pr["input_md5"],
           "decisions": pr["decisions"], "units_scanned": t.get("units", 0), "status": "COMPLETE" if complete else "INCOMPLETE",
           "index_npz_md5": summ["index_npz_md5"], "hits_md5": summ["hits_md5"], "adjudicated_md5": adj["md5"],
           "raw": {"strong_units": t.get("strong_units", 0), "weak_only_units": t.get("weak_only_units", 0)},
           "adjudication_moved": adj["moved"],
           "after_adjudication": {"strong_units": len(strong), "weak_only_units": weak, "strong_kinds": dict(kinds)},
           "strong_decision_uids": strong[:500], "strong_protected_files": dict(files.most_common(50)),
           "strong_vs_stage0_source_adjudication": dict(collections.Counter(s0.values())) if s0 else {},
           "policy": "STRONG after adjudication = withhold (stops publication until resolved); weak-only stays eligible; protected items are "
                     "identified by (population, file) only -- no protected or sealed text is read"}
    json.dump(man, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: man[k] for k in ("block", "decisions", "units_scanned", "status", "raw", "after_adjudication",
                                          "strong_vs_stage0_source_adjudication")}))
    sys.exit(0 if complete else 2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--block", action="append", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--block", required=True)
    r.add_argument("--dir", required=True)
    r.add_argument("--stage0-adj", action="append", default=[])
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": prep, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
