"""Score dev-set reads (kit scorer over the dev rows; families carry board names, so the index uses board weights)."""
import json
import statistics
import sys
from pathlib import Path

kit, rows_f, base = sys.argv[1:4]
sys.path.insert(0, kit)
from decision_index.vision.score import score  # noqa: E402

rows = [json.loads(l) for l in open(rows_f) if l.strip()]
rep = {}
for d in sorted(p for p in Path(base).iterdir() if p.is_dir()):
    f = d / "results.jsonl"
    if not f.exists():
        continue
    ok = [json.loads(l) for l in open(f)]
    ok = [r for r in ok if r["status"] == "ok"]
    preds = {r["run_id"]: r["response"]["answers"]["answer"]["choice"] for r in ok}
    res = score(rows, preds)
    lat = sorted(r["total_wall_ms"] for r in ok)
    rep[d.name] = {"dev_index": round(res["public_vision_index"], 2), "rows_ok": len(ok), "missing": len(rows) - len(preds),
                   "skill": {b: round(x["skill_clipped"] * 100, 2) for b, x in res["benchmarks"].items()},
                   "acc": {b: round(x["accuracy"] * 100, 2) for b, x in res["benchmarks"].items()},
                   "latency_ms": {"median": round(statistics.median(lat)), "p80": round(lat[int(.8 * len(lat))])}}
Path(base, "dev_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep))
