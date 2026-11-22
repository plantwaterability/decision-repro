#!/usr/bin/env python3
"""Assemble the v3 protected-index deliverables from a finished build + scans (x2 work dir), sample hits for false-positive review
(non-sealed populations only; snippets of public / unsealed text), list excluded uids, flag unresolved source aliases, and upload
everything to S3 with md5 read-back; _READY.json last.

  PYTHONHASHSEED=0 python3 publish_v3.py --work /data/local/sdv1/prot --pools emo_pool_v1,... [--upload]
"""
import argparse
import collections
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

S3P = "s3://YOUR_BUCKET/jev-race/stock_decision_v1/stage0/protected/"
SEALED = ("final_v1_hl_index", "final_v1_rd_index", "confirm_2610_index", "confirm_p1_index")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tokens(s):
    return set(re.findall(r"[a-z]{3,}", s.lower().replace("_", " ")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    W = Path(a.work)
    sys.path.insert(0, str(W / "code"))
    import evalindex_v3 as E
    E.load_v1(W / "code/code_v1")
    man = json.loads((W / "index_v31/protected_index_v3_manifest.json").read_text())
    stage_man = json.loads((W / "stage/stage_manifest.json").read_text())
    srcpath = {}
    for pop, r in stage_man["populations"].items():
        for f in r.get("files", []):
            srcpath[(pop, f.get("member") or Path(f["path"]).name)] = f["path"]
    out = W / "publish"
    out.mkdir(exist_ok=True)
    pools = [p for p in a.pools.split(",") if p]
    rep = {"schema": "stock-decision-overlap-report-external-v1", "index_npz_md5": man["npz_md5"], "index_npz_sha256": man["npz_sha256"],
           "thresholds": man["params"], "strong_kinds": man["strong_kinds"], "weak_kinds": man["weak_kinds"],
           "template_rules": {"T1": "text_long hit whose every matched text unit occurs in >= 25 candidate TRAIN records -> weak text_template "
                                    "(= v2 adjudicate_text_v2, prespecified)",
                              "T2": "near_dup / eval_in_cand / cand_in_eval hit whose every candidate document is an instructions / "
                                    "option_text / noul_defs / question_other text occurring in >= 25 candidate TRAIN records -> weak "
                                    "doc_template (stated with T1 in evalindex_v3.py before any v3 scan result was seen)"},
           "policy": "STRONG after T1 / T2 = withhold from every training / head-fit / discovery pool; weak = eligible, listed. Field tags: "
                     "candidate field -> protected field.", "pools": {}}
    rng = random.Random(20261013)
    samples, aliases = [], collections.defaultdict(lambda: collections.Counter())
    uploads = []
    for pool in pools:
        sp = W / f"scans/summary_{pool}.json"
        ap_ = W / f"scans/adjudication_{pool}.json"
        if not sp.exists() or not ap_.exists():
            rep["pools"][pool] = {"status": "not scanned"}
            continue
        summ, adj = json.loads(sp.read_text()), json.loads(ap_.read_text())
        hits = W / f"scans/hits_{pool}.adj.jsonl.gz"
        by_kf, by_pop, by_file, excl = collections.Counter(), collections.Counter(), collections.Counter(), []
        by_cfield = collections.Counter()
        cat_rows = collections.defaultdict(list)
        n = 0
        for r in (json.loads(l) for l in gzip.open(hits, "rt")):
            if not r["s"]:
                continue
            n += 1
            excl.append(r["uid"])
            seen = set()
            for kind_, cf, pop, fil, pf in r["m"]:
                if kind_ not in E.STRONG:
                    continue
                t = (kind_, cf, pf)
                if t not in seen:
                    seen.add(t)
                    by_kf[f"{kind_}|{cf}->{pf}"] += 1
                by_file[f"{pop}|{fil}"] += 0
            for pop in r["sp"]:
                by_pop[pop] += 1
            fs = {f"{m[2]}|{m[3]}" for m in r["m"] if m[0] in E.STRONG}
            for f_ in fs:
                by_file[f_] += 1
            for cf in {m[1] for m in r["m"] if m[0] in E.STRONG}:
                by_cfield[cf] += 1
            stem = r["uid"].split("|")[1].split("/")[-1].rsplit(".", 1)[0]
            for f_ in fs:
                aliases[(pool, stem)][f_] += 1
            ok = [m for m in r["m"] if m[0] in E.STRONG and m[2] not in SEALED and m[0] in ("text_long", "near_dup", "eval_in_cand", "cand_in_eval")]
            if ok:
                m0 = ok[0]
                cat_rows[(m0[0], m0[1], m0[4])].append((r["uid"], m0))
        exf = out / f"excluded_uids_{pool}.txt.gz"
        with gzip.open(exf, "wt") as g:
            g.write("\n".join(sorted(excl)) + ("\n" if excl else ""))
        uploads += [exf, sp, ap_, hits, W / f"scans/hits_{pool}.jsonl.gz"]
        for cat, lst in cat_rows.items():
            for uid, m0 in rng.sample(lst, min(3, len(lst))):
                samples.append({"pool": pool, "uid": uid, "category": "|".join(cat), "protected": f"{m0[2]}|{m0[3]}", "_m": m0})
        rep["pools"][pool] = {"input": summ["input"], "input_md5": summ.get("input_md5"), "splits": summ["splits"],
                              "units": summ["totals"].get("units", 0), "strong_units_raw": summ["totals"].get("strong_units", 0),
                              "strong_units_after_T1_T2": n, "weak_only_units": adj["after"].get("weak_only_units", 0),
                              "moved_by_rule": adj["moved"], "strong_by_kind": {k[7:]: v for k, v in adj["after"].items() if k.startswith("strong:")},
                              "strong_units_by_kind_and_field": dict(by_kf.most_common()), "strong_units_by_candidate_field": dict(by_cfield),
                              "strong_units_by_population": dict(by_pop.most_common()),
                              "strong_units_by_protected_file_top40": dict(collections.Counter({k: v for k, v in by_file.items() if v}).most_common(40)),
                              "hits_raw": {"key": S3P + f"scans/hits_{pool}.jsonl.gz", "md5": md5(W / f"scans/hits_{pool}.jsonl.gz")},
                              "hits_adjudicated": {"key": S3P + f"scans/hits_{pool}.adj.jsonl.gz", "md5": md5(hits)},
                              "excluded_uids": {"key": S3P + f"excluded/excluded_uids_{pool}.txt.gz", "md5": md5(exf), "count": n}}
    for s in samples:
        m0 = s.pop("_m")
        s.update(kind=m0[0], candidate_field=m0[1], protected_field=m0[4])
    rep["hit_samples_for_review"] = {"n": len(samples), "file": "hit_samples_v3.json", "note": "3 random strong hits per (kind, candidate "
                                     "field, protected field) category per pool, sealed populations excluded"}
    (out / "hit_samples_v3.json").write_text(json.dumps(samples, indent=1) + "\n")
    unresolved = []
    for (pool, stem), c in aliases.items():
        st = tokens(stem)
        for f_, v in c.most_common(5):
            if v < 20:
                continue
            ft = tokens(f_.split("|", 1)[1])
            if not (st & ft):
                unresolved.append({"pool": pool, "candidate_stem": stem, "protected_file": f_, "strong_units": v})
    rep["unresolved_aliases"] = {"rule": "a candidate stem with >= 20 strong units on a protected file whose name shares no >= 3-letter token "
                                         "with the stem (the same upstream under another name, e.g. GSM8K inside RouterBench); listed for "
                                         "the parent's canonical source-family map", "items": sorted(unresolved, key=lambda x: -x["strong_units"])}
    v30 = {}
    for sp in sorted((W / "v30_raw/scans").glob("summary_*.json")):
        pool = sp.stem[len("summary_"):]
        sm = json.loads(sp.read_text())
        ad = W / f"v30_raw/scans/adjudication_{pool}.json"
        adj = json.loads(ad.read_text()) if ad.exists() else {}
        v30[pool] = {"units": sm["totals"].get("units"), "strong_units_raw": sm["totals"].get("strong_units"),
                     "strong_units_after_T1_T2": (adj.get("after") or {}).get("strong_units"), "moved": adj.get("moved")}
    rep["amendment_A1"] = {"what": "v3.1 shingles CONTENT fields only (state, candidate_document = |doc_ / |tool_ / |passage_ question texts, "
                                   "record_other, attachment_text, rendered prompt); instructions / option texts / noul definitions are matched by "
                                   "exact text units (T1) and v1 joint / question identities only; choice options are never tagged candidate documents",
                           "when": "2026-10-14 00:10Z, after the RAW v3.0 counts below (emo_pool_v1, jdref_gsm8k) and before any adjudicated v3.0 "
                                   "count was read (the v3.0 adjudicated count, read afterwards, is shown too)",
                           "why": "v3.0 near-dup / containment on template-bearing question fields flagged items that share a builder template but "
                                  "differ in state (different evaluation items); T2's exact-text DF rule cannot see slot-varied templates",
                           "v3_0_evidence": v30, "fixture": "A1_slot_template_not_strong_c37 (a slot-varied template with a different state is "
                                                            "not STRONG) added to regress_fixtures"}
    reg = W / "regress_v3.json"
    rep["regression"] = json.loads(reg.read_text()) if reg.exists() else None
    rep["regression_note"] = ("E2 field coverage: the reverse check finds exactly the errata's 11 jev_data-train GSM8K questions in 8 of 220 "
                              "scored BRIGHT rows, all in the candidate_document field. Forward attribution names 10 of the 11: near-dup / "
                              "containment attribution reports only the best-matching protected document per kind, and the 11th question "
                              "(jdref lines 4354-4355) is fully contained (cc 1.0) in DI 06_routerbench states too, which wins the tie; both "
                              "units are STRONG (cand_in_eval + text_long) and excluded either way. Exact text-unit attribution is complete. "
                              "E1 passes: jev_data gsm8k strong hits attribute to 06_routerbench (1,217 units in emo_pool_v1), none to DI 30_gsm8k.")
    rep["regression_fixtures_initial"] = json.loads((W / "regress_fixtures.json").read_text()) if (W / "regress_fixtures.json").exists() else None
    (out / "overlap_report_external.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    pim = {"schema": "stock-decision-protected-index-manifest-v1", "index": {"npz": S3P + "index/protected_index_v3.npz", "md5": man["npz_md5"],
                                                                          "sha256": man["npz_sha256"]},
           "built_utc": man["built_utc"], "python": man["python"], "pythonhashseed": man["pythonhashseed"],
           "code_sha256": man["code_sha256"], "keyfn_sha256": man["keyfn_sha256"], "v1_decontam_sha256": man["v1_decontam_sha256"],
           "registry_v3_sha256": man["registry_sha256"], "stage_manifest_sha256": man["stage_manifest_sha256"], "fields": man["fields"],
           "params": man["params"], "strong_kinds": man["strong_kinds"], "weak_kinds": man["weak_kinds"], "unique_docs": man["unique_docs"],
           "doc_occurrences": man["doc_occurrences"], "family_sizes": man["family_sizes"],
           "populations": {p: {**{k: v for k, v in man["coverage"][p].items() if k != "substitute"},
                               "digest_sha256": man["protected_panel_digest_sha256"].get(p),
                               "coverage_by_field": man["coverage_by_field"].get(p)} for p in man["populations"]},
           "not_included": json.loads((W / "code/registry_v3.json").read_text()).get("not_included"),
           "coverage_statement": man["coverage_statement"], "n_source_files": len(man["sources"]),
           "source_files": S3P + "index/protected_index_v3_manifest.json (sources[])"}
    (out / "protected_index_manifest.json").write_text(json.dumps(pim, indent=1, sort_keys=True) + "\n")
    if not a.upload:
        print("publish dir ready (no upload)")
        return
    ups = [(W / "index_v31/protected_index_v3.npz", "index/protected_index_v3.npz"),
           (W / "index_v31/protected_index_v3_manifest.json", "index/protected_index_v3_manifest.json"),
           (W / "stage/stage_manifest.json", "index/stage_manifest_v3.json"),
           (W / "code/registry_v3.json", "code/registry_v3.json"), (W / "code/evalindex_v3.py", "code/evalindex_v3.py"),
           (W / "code/build_registry_v3.py", "code/build_registry_v3.py"), (W / "code/publish_v3.py", "code/publish_v3.py"),
           (W / "regress_fixtures.json", "regression/regress_fixtures.json"), (W / "regress_v3.json", "regression/regress_v3.json"),
           (out / "overlap_report_external.json", "overlap_report_external.json"), (out / "protected_index_manifest.json", "protected_index_manifest.json"),
           (out / "hit_samples_v3.json", "hit_samples_v3.json")]
    for p in sorted((W / "v30_raw/scans").glob("summary_*.json")) + sorted((W / "v30_raw/scans").glob("adjudication_*.json")):
        ups.append((p, "v3.0_raw/" + p.name))
    for p in uploads:
        p = Path(p)
        ups.append((p, ("excluded/" if p.name.startswith("excluded_") else "scans/") + p.name))
    ready = {}
    for p, key in ups:
        if not Path(p).exists():
            continue
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(p), S3P + key], check=True)
        back = subprocess.run(f"aws s3 cp --only-show-errors {S3P + key} - | md5sum", shell=True, capture_output=True, text=True).stdout.split()[0]
        m = md5(p)
        if back != m:
            raise SystemExit(f"READBACK MISMATCH {key}")
        ready[key] = m
    rd = out / "_READY.json"
    rd.write_text(json.dumps({"schema": "stock-decision-protected-ready-v1", "objects_md5": ready, "pools": pools}, indent=1, sort_keys=True) + "\n")
    subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(rd), S3P + "_READY.json"], check=True)
    print(json.dumps({"uploaded": len(ready), "ready_md5": md5(rd)}))


if __name__ == "__main__":
    main()
