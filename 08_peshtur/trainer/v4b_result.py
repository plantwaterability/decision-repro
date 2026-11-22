"""Summarize v4b1 final evaluation (dev gates, dev parity, RBH, HELD, public no-WG + halves)."""
import collections
import glob
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

from decision_index.vision.score import score

Q = Path("/data/local/qv")
NAME = sys.argv[1]
RDIR = Q / "runs" / NAME


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_shard_preds(d):
    preds, probs, lats = {}, {}, []
    for f in sorted(glob.glob(f"{d}/shard_*/results.jsonl")):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("status") == "ok":
                ans = r["response"]["answers"]["answer"]
                preds[r["run_id"]] = ans["choice"]
                if "probs" in ans:
                    probs[r["run_id"]] = ans["probs"]
                if "total_wall_ms" in r:
                    lats.append(r["total_wall_ms"])
    return preds, probs, lats


def main():
    tlog = load_jsonl(RDIR / "train_log.jsonl")
    steps = [x for x in tlog if x.get("event") == "step"]
    gates = {str(x["step"]): round(x["index"], 3) for x in tlog if x.get("event") in ("gate", "gate_baseline")}
    early_pub = [x for x in tlog if x.get("event") == "public_read"]

    pub_rows = load_jsonl(Q / "public/suite/vision-rows-noWG.jsonl")
    preds, _, lats = load_shard_preds(RDIR / "public_read")
    sc_all = score(pub_rows, preds)
    h0_rows = [r for r in pub_rows if int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 2 == 0]
    h1_rows = [r for r in pub_rows if int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 2 == 1]
    sc_h0 = score(h0_rows, {r["id"]: preds[r["id"]] for r in h0_rows if r["id"] in preds})
    sc_h1 = score(h1_rows, {r["id"]: preds[r["id"]] for r in h1_rows if r["id"] in preds})

    held_json = json.loads((RDIR / "held.json").read_text()) if (RDIR / "held.json").exists() else None
    parity_json = json.loads((RDIR / "dev_parity/parity.json").read_text()) if (RDIR / "dev_parity/parity.json").exists() else None
    rbh_json = json.loads((RDIR / "rbh_read/rbh_gate.json").read_text()) if (RDIR / "rbh_read/rbh_gate.json").exists() else None

    lats.sort()
    lat_summary = (
        {
            "n": len(lats),
            "median_ms": round(statistics.median(lats), 1),
            "mean_ms": round(statistics.mean(lats), 1),
            "p80_ms": round(lats[int(0.8 * len(lats))], 1),
            "p95_ms": round(lats[int(0.95 * len(lats))], 1),
        }
        if lats
        else None
    )

    out = {
        "run": NAME,
        "base": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "layout": "d3",
        "manifest": "wxs1_nc (323,264 rows, 0.5 gold + 0.5 equal mean of WeMM/vxg2/s2b)",
        "train": {
            "last_step": steps[-1]["step"] if steps else 0,
            "s_per_step_median": round(statistics.median([x["s"] for x in steps]), 2) if steps else None,
            "rows_skipped": sum(x.get("skipped", 0) for x in steps),
        },
        "dev_gates": gates,
        "early_public_step2500": early_pub[-1] if early_pub else None,
        "dev_parity": parity_json,
        "rbh": rbh_json,
        "held": held_json["arm"] if held_json else None,
        "public_noWG": {
            "rows_read": len(preds),
            "rows_total": len(pub_rows),
            "index": round(sc_all["public_vision_index"], 3),
            "halves": [round(sc_h0["public_vision_index"], 3), round(sc_h1["public_vision_index"], 3)],
            "skill": {k: round(v["skill_clipped"] * 100, 2) for k, v in sc_all["benchmarks"].items()},
            "acc": {k: round(v["accuracy"] * 100, 2) for k, v in sc_all["benchmarks"].items()},
            "h200_latency_ms": lat_summary,
        },
    }
    (RDIR / f"{NAME}_result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
