#!/usr/bin/env python3
"""(DI3, 2026-10-09) Decision Index 0.3 from our 0.2.1 build: the suite, gated by the kit's own hash checks, and the 2,638 rebuilt
GSM8K requests in our record format. Pre-registered in AGENT_NOTES.md "(DI3) DECISION INDEX 0.3 PUBLIC INDEX ..." (2026-10-09 07:45Z).
EVALUATION ONLY (the terms of trainers/convert_decision_index.py apply to every row).

Edition 0.3 reads the 0.2.1 files UNCHANGED plus gsm8k-rows.jsonl.gz (kit README "What changed in 0.3"; docs/suite.md):

  suite03   build gsm8k-rows with the KIT'S OWN builder (decision_index.suite.build.gsm8k_v3.build / write) from the pinned
            openai/gsm8k test parquet (the kit checks its sha256) and the 0.2.1 selected rows; then the kit's `suite import` for
            edition 0.3 (download.from_local, verify=True: refuses any file whose hash differs from editions.py) and Suite.verify
            (strict). Writes OUT/suite03_verify.json. Exit 0 only if every hash matches.
  gsm8k     the rebuilt GSM8K requests (catalog 30, run ids "v0.3-gsm8k:...") read through the kit's Suite under edition 0.3
            with exclusions, converted with convert_decision_index.convert_row (unchanged) into OUT/30_gsm8k_v3/eval/30_gsm8k_v3.jsonl
            (+ manifest.json). Refuses unless there are exactly 2,638 such requests, each with K <= 52 and a v0.3 run id.

  python3 trainers/convert_di03.py suite03 --kit KIT --suite02 DIR/suite-0.2 --parquet test-00000-of-00001.parquet --out DIR/suite-0.3
  python3 trainers/convert_di03.py gsm8k --kit KIT --suite-dir DIR/suite-0.3 --out DIR/di_v03
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITION = "0.3"
GSM8K_ID = 30
GSM8K_REQUESTS = 2638
RUN_PREFIX = "v0.3-gsm8k:"
SLUG = "30_gsm8k_v3"


def load_cdi():
    import importlib.util
    spec = importlib.util.spec_from_file_location("convert_decision_index", ROOT / "trainers/convert_decision_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kit_path(kit):
    if str(kit) not in sys.path:
        sys.path.insert(0, str(kit))


def cmd_suite03(a) -> int:
    kit_path(a.kit)
    from decision_index import editions
    from decision_index.suite import download
    from decision_index.suite.build import gsm8k_v3
    from decision_index.suite.io import Suite
    e = editions.get(EDITION)
    s02, out, work = Path(a.suite02), Path(a.out), Path(a.work)
    rows = s02 / editions.ROWS_FILE
    rows_v3 = gsm8k_v3.build(Path(a.parquet), rows)                      # the kit's builder; checks the parquet's sha256
    rep = gsm8k_v3.write(rows_v3, work / "release-v3-rebuilt")
    rep.update(expected_sha256=e["gsm8k_sha256"], byte_identical=rep["gsm8k_sha256"] == e["gsm8k_sha256"])
    result = {"edition": e["id"], "kit_edition": e["label"], "gsm8k": rep}
    if not rep["byte_identical"] or rep["requests"] != GSM8K_REQUESTS:
        result["ok"] = False
    else:
        imp = download.from_local(out, rows, edition=EDITION, added=s02 / editions.ADDED_FILE, gsm8k=Path(rep["out"]), verify=True)
        ver = Suite(out, EDITION).verify(strict=True)
        result.update(imported=imp, verify=ver, ok=bool(ver.get("match")))
    (Path(a.report) if a.report else out.parent / "suite03_verify.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({"ok": result["ok"], "gsm8k_sha256": rep["gsm8k_sha256"], "expected": e["gsm8k_sha256"],
                      "requests": rep["requests"]}))
    return 0 if result["ok"] else 1


def convert_gsm8k(rows, cdi) -> tuple[list, dict]:
    recs, stats = [], {"requests": 0, "units": 0, "questions": 0, "k52_questions": 0, "k_hist": Counter()}
    for row in rows:
        e = row["_evaluation"]
        if int(e["catalog_id"]) != GSM8K_ID:
            continue
        rid = str(e["run_id"])
        if not rid.startswith(RUN_PREFIX):
            raise SystemExit(f"GSM8K row {rid} is not a rebuilt 0.3 row (edition mismatch)")
        rec = cdi.convert_row(row, None, stats, "text")
        rec["meta"]["builder"] = "decision-index kit 0.3 (suite-0.3 gsm8k-rows) via trainers/convert_di03.py + convert_decision_index.convert_row"
        rec["meta"]["edition"] = EDITION
        rec["source"] = f"decision_index/{SLUG}"
        recs.append(rec)
        stats["requests"] += 1
        stats["questions"] += len(row["questions"])
    return recs, stats


def cmd_gsm8k(a) -> int:
    kit_path(a.kit)
    from decision_index.suite.io import Suite
    cdi = load_cdi()
    suite = Suite(Path(a.suite_dir), EDITION)
    ver = suite.verify(strict=not a.no_verify)
    recs, stats = convert_gsm8k(suite.rows(apply_exclusions=True), cdi)
    bad = [r["id"] for r in recs if any(len(q.get("criteria", {})) > cdi.MAX_K for q in r["questions"].values())]
    if len(recs) != a.expect or bad or stats["k52_questions"]:
        raise SystemExit(f"{len(recs)} rebuilt GSM8K requests (want {a.expect}); K>52: {len(bad)}")
    out = Path(a.out) / SLUG
    (out / "eval").mkdir(parents=True, exist_ok=True)
    with open(out / "eval" / f"{SLUG}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"name": "Decision Index 0.3 rebuilt GSM8K, converted for trainers/claude_lmscore.py", "edition": EDITION, "eval_only": True,
           "requests": len(recs), "questions": stats["questions"], "units_round1": stats["units"],
           "k_hist": dict(sorted(stats["k_hist"].items())), "suite_verify": ver, "slug": SLUG}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    print(json.dumps({k: man[k] for k in ("requests", "questions", "units_round1", "k_hist")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("suite03")
    s.add_argument("--kit", required=True)
    s.add_argument("--suite02", required=True, help="our 0.2.1 suite dir (selected-rows.jsonl.gz, added-rows.jsonl.gz, ...)")
    s.add_argument("--parquet", required=True, help="openai/gsm8k @740312ad main/test-00000-of-00001.parquet")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True, help="the suite-0.3 dir to stage")
    s.add_argument("--report", default="")
    s.set_defaults(fn=cmd_suite03)
    g = sub.add_parser("gsm8k")
    g.add_argument("--kit", required=True)
    g.add_argument("--suite-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--expect", type=int, default=GSM8K_REQUESTS)
    g.add_argument("--no-verify", action="store_true", help="tests only")
    g.set_defaults(fn=cmd_gsm8k)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
