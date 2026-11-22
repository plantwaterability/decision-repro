#!/usr/bin/env python3
"""Stage 0 unified candidate-decision table (stock_decision_v1): one row per TRAIN decision unit of every candidate pool, joined
from the production-reader unit table (unittable_v2) and the R1/R2 token table (tokcount), with the source map applied.

Columns: uid, pool, rk (record key), qid, stem, root, upstream_source, canonical_family, generator_mechanism, area, skill, domain,
label_basis, m1_category, m1_eligible, m0_root (recipe A root), di_ancestry, licence_line, kind, K, k_bucket, tok_r1, tok_r2,
len_bucket, hard (one-hot gold), tmax, base_id, record_id, parent_key, lineage_key, base_ok, base_reason.
base_ok = train split, clean licence line, reader loads it, 2 <= K <= 52, R2 prompt <= 8,176 tokens (8,192 minus the renderer's
16-token margin; no truncation), no duplicate option texts (R2 would have to disambiguate them). Overlap, de-duplication,
conflicting targets and split reservation are applied later (sd0_split.py) and recorded there.

  python3 sd0_candidates.py --pool T=UNITS.jsonl.gz,TOKENS.jsonl.gz ... --roots source_roots_v2.json --map source_map_core_v1.json
        [--ts-map tasksource_map.json] --out cand.parquet
"""
import argparse
import gzip
import json
import os

import pandas as pd

KB = [(2, 2, "2"), (3, 5, "3-5"), (6, 16, "6-16"), (17, 32, "17-32"), (33, 52, "33-52")]
LB = [(1, 512, "<=512"), (513, 2048, "513-2048"), (2049, 8176, "2049-8192")]
MAX_TOK = 8176
# finer upstream sources than v2's cap roots (v2 merged them only for its 5% cap)
STEM_ROOT_OVERRIDE = {"jev_rd3__swiss": "swiss"}
M0_ROOTS = {"afd", "aidev", "banking77", "bbq", "casino", "circa", "clinc_oos", "code_contests", "commonsense_qa", "cosmos_qa", "courts",
            "swiss", "ctgov", "dmhc", "e_care", "esci", "fda", "folktables", "gdelt", "genomics", "go_emotions", "gsm8k", "helpsteer2",
            "ibm_argq", "jebadiah", "kaivoss", "lichess", "measuring_hate_speech", "medmcqa", "mghafiri", "nvd_cwe", "open_jev",
            "openreview", "physics", "procedural_typed_decisions", "pubmedqa", "safety", "scruples", "social_i_qa", "stanford_politeness",
            "swe_rebench", "tabfact", "vagmi", "wiki_census", "wikidata", "winogrande", "bayes_pref_flights", "synth_beta_rate",
            "synth_binary_evidence"}


def bucket(x, table):
    for lo, hi, name in table:
        if lo <= x <= hi:
            return name
    return None


def rows(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="append", required=True)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--ts-map", default="")
    ap.add_argument("--splits", default="train", help="unit-table splits to keep (default train)")
    ap.add_argument("--qsem", default="", help="sd0_qsemantics.py output: per-question declared target semantics")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stems = json.load(open(a.roots))["stems"]
    smap = json.load(open(a.map))["roots"]
    smap.setdefault("swiss", {**smap["courts"], "upstream_source": "swiss_federal_court", "canonical_family": "swiss_federal_court"})
    smap["courts"] = {**smap["courts"], "upstream_source": "courtlistener", "canonical_family": "courtlistener"}
    tsmap = json.load(open(a.ts_map))["sources"] if a.ts_map else {}
    out = []
    for spec in a.pool:
        tag, paths = spec.split("=", 1)
        up, tp = paths.split(",")
        tok = {r["uid"]: r for r in rows(tp)}
        for r in rows(up):
            if r.get("split") not in set(a.splits.split(",")):
                continue
            uid = r["uid"]
            pool, mem, line, qid = uid.split("|", 3)
            stem = r["stem"]
            meta_root = None
            if tag == "bayes_families_v1":
                parts = str(r.get("base_id", "")).split("/")
                meta_root = "bf_" + (parts[3] if len(parts) > 3 else "?")
            elif tag.startswith("tasksource"):
                meta_root = (tsmap.get(r.get("tasksource_source")) or {}).get("root")
            elif tag.startswith("gap_supply") or tag.startswith("long_knowledge"):
                meta_root = stem
            root = meta_root or STEM_ROOT_OVERRIDE.get(stem) or (stems.get(stem) or {}).get("root")
            m = (tsmap.get(r.get("tasksource_source")) if tag.startswith("tasksource") else smap.get(root)) or {}
            t = tok.get(uid) or {}
            K = int(r.get("K") or 0)
            t2 = t.get("tok_r2")
            reason = []
            if r.get("loader_error"):
                reason.append("loader_error")
            if r.get("nc"):
                reason.append("nc")
            if K < 2:
                reason.append("k_lt_2")
            if K > 52:
                reason.append("k_gt_52")
            if t2 is None:
                reason.append("no_token_count")
            elif t2 > MAX_TOK:
                reason.append("over_8k")
            if r.get("dup_opt"):
                reason.append("dup_option_text")
            if not m:
                reason.append("unmapped_source")
            gen = m.get("generator_mechanism")
            if root == "procedural_typed_decisions":
                gen = "procedural/" + stem.split("__")[-1]
            tm = r.get("tmax")
            out.append({"uid": uid, "pool": tag, "rk": f"{pool}|{mem}|{line}", "qid": qid, "stem": stem, "root": root,
                        "upstream_source": m.get("upstream_source"), "canonical_family": m.get("canonical_family"),
                        "generator_mechanism": gen, "area": m.get("area"), "skill": m.get("skill"), "domain": m.get("domain"),
                        "label_basis": m.get("label_basis"), "m1_category": m.get("m1_category"), "m1_eligible": bool(m.get("m1_eligible")),
                        "m0_root": root in M0_ROOTS and tag in ("emo_pool_v1", "synth_supply_v1"),
                        "di_ancestry": bool((stems.get(stem) or {}).get("di_ancestry")) or bool(m.get("di_root")) or m.get("m1_reason", "").startswith("Decision Index"),
                        "variant": r.get("variant"), "label_evidence": m.get("label_evidence"),
                        "td_workflow_imitation": m.get("td_workflow_imitation"), "di_target_benchmarks": ",".join(m.get("di_target_benchmarks") or []) or None,
                        "split": r.get("split"),
                        "licence_line": "NC" if r.get("nc") else "clean", "kind": r.get("qtype"), "K": K, "k_bucket": bucket(K, KB),
                        "tok_r1": t.get("tok_r1"), "tok_r2": t2, "len_bucket": bucket(t2, LB) if t2 else None,
                        "hard": bool(tm is not None and float(tm) >= 0.999999), "tmax": tm, "base_id": str(r.get("base_id")),
                        "record_id": str(r.get("record_id")), "parent_key": str(r.get("parent_key") or r.get("base_id")),
                        "lineage_key": str(r.get("group") or r.get("group_id") or r.get("base_id")),
                        "base_ok": not reason, "base_reason": ";".join(reason)})
        print(tag, sum(1 for x in out if x["pool"] == tag), flush=True)
    df = pd.DataFrame(out)
    if a.qsem:                                   # per-question target semantics declared by the generator (carried, not inferred)
        q = pd.read_parquet(a.qsem)
        df = df.merge(q, on="uid", how="left")
    for c_ in ("target_basis_q", "label_verification_q", "target_semantics_declared", "loss_by_option"):
        if c_ not in df:
            df[c_] = None
    df.to_parquet(a.out + ".partial", index=False)
    os.replace(a.out + ".partial", a.out)
    print(json.dumps({"rows": len(df), "base_ok": int(df.base_ok.sum()),
                      "reasons": df.loc[~df.base_ok, "base_reason"].value_counts().head(20).to_dict()}))


if __name__ == "__main__":
    main()
