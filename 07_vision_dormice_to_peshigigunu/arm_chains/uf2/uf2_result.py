"""uf2 close-out numbers from the node's own files (no model call): dev gates, rows used / skipped (and the ADE rows among them), RBH two
numbers, proxy HELD per family and per subtask against the UE2 reference read in the same processes (CV-Bench split into Count / Relation /
Distance), the ONE public no-WG read against UE2's stored one, verdict by the preregistered band. Writes runs/<name>/held_split.json and
runs/<name>/uf2_result.json. usage: uf2_result.py <run name> <manifest name>"""
import collections, glob, json, os, subprocess, sys

name, man = sys.argv[1], sys.argv[2]
Q = os.environ.get("UF2_Q", "/data/local/qv")
R = f"{Q}/runs/{name}"
A = "s3://YOUR_BUCKET/jev-race/engineering/vision/recipe_a"
out = {"run": name, "manifest": man}


def jl(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


# training log: gates, skipped rows
ev = [json.loads(l) for l in open(f"{R}/train_log.jsonl") if l.strip()] if os.path.exists(f"{R}/train_log.jsonl") else []
steps = {e["step"]: e for e in ev if e.get("event") == "step"}
out["train"] = {"steps_logged": len(steps), "last_step": max(steps) if steps else None, "rows_used": sum(e["rows"] for e in steps.values()),
                "rows_skipped": sum(e["skipped"] for e in steps.values()), "steps_with_skips": sum(1 for e in steps.values() if e["skipped"]),
                "starts": sum(1 for e in ev if e.get("event") == "start"),
                "seconds_per_step_median": sorted(e["s"] for e in steps.values())[len(steps) // 2] if steps else None}
out["dev_gates"] = {str(e["step"]): {"index": round(e["index"], 2), "verdict": e["verdict"], "skill": e.get("skill")} for e in ev if e.get("event") == "gate"}
try:
    rc = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "uf2_rows_check.py"), f"{Q}/data/{man}.jsonl", "20261114",
                         f"{R}/train_log.jsonl"], capture_output=True, text=True, timeout=600)
    out["rows_check"] = json.loads(rc.stdout.strip().splitlines()[-1])
except Exception as e:  # noqa: BLE001
    out["rows_check"] = {"error": str(e)[:200]}
out["rbh"] = jl(f"{R}/rbh_read/rbh_gate.json")
out["held"] = jl(f"{R}/held.json")
# HELD split: per family, per (family, subtask), per proxy source; arm vs the UE2 reference, paired
try:
    rows = [json.loads(l) for l in open(f"{Q}/priv/priv_held.jsonl") if l.strip()]
    rid = lambda r: (r.get("_evaluation") or {}).get("run_id") or r.get("id")  # noqa: E731
    cnt = collections.Counter(rid(r) for r in rows)
    rows = [r for r in rows if cnt[rid(r)] == 1]

    def preds(d):
        p = {}
        for f in sorted(glob.glob(d + "/shard_*/results.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if r.get("status") == "ok":
                    p[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
        return p

    a, b = preds(f"{R}/held_read"), preds(f"{R}/held_ref")

    def table(key):
        g = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0, 0])
        for r in rows:
            x = g[key(r)]
            oa, ob = a.get(rid(r)) == r["expected"]["answer"], b.get(rid(r)) == r["expected"]["answer"]
            x[0] += 1; x[1] += oa; x[2] += ob; x[3] += 1 / len(r["questions"]["answer"]["criteria"]); x[4] += oa and not ob; x[5] += ob and not oa
        sk = lambda c, n, ch: round(100 * (c / n - ch / n) / (1 - ch / n), 2)  # noqa: E731
        return {k: {"n": n, "arm_acc": round(100 * ca / n, 2), "ue2_acc": round(100 * cb / n, 2), "arm_skill": sk(ca, n, ch), "ue2_skill": sk(cb, n, ch),
                    "acc_delta": round(100 * (ca - cb) / n, 2), "gained": ga, "lost": lo} for k, (n, ca, cb, ch, ga, lo) in sorted(g.items())}

    split = {"rows": len(rows), "arm_read": sum(rid(r) in a for r in rows), "ue2_read": sum(rid(r) in b for r in rows),
             "per_family": table(lambda r: r.get("benchmark")), "per_family_subtask": table(lambda r: f"{r.get('benchmark')} | {r.get('subtask')}"),
             "per_proxy_source": table(lambda r: r.get("set")), "reference": "ue2_noade/ckpt_03700 read on this node in the same processes"}
    json.dump(split, open(f"{R}/held_split.json", "w"), indent=1)
    out["held_split"] = {"per_family": split["per_family"], "cv_bench": {k: v for k, v in split["per_family_subtask"].items() if k.startswith("CV-Bench")}}
except Exception as e:  # noqa: BLE001
    out["held_split"] = {"error": str(e)[:300]}
# the ONE public no-WG read vs UE2's stored read (stored file only; no new read of UE2)
pub = jl(f"{R}/public_read/full_index.json")
if pub:
    ue2 = None
    try:
        ue2 = json.loads(subprocess.run(["aws", "s3", "cp", "--quiet", f"{A}/runs/ue2_noade/public_read/full_index.json", "-"], capture_output=True, text=True, timeout=120).stdout)
    except Exception:  # noqa: BLE001
        pass
    x = pub["nowg_index"]
    out["public_nowg"] = {"index": x, "rows": pub.get("n_nowg"), "missing": pub.get("missing_nowg"), "skill": pub.get("skill"), "latency_ms": pub.get("latency_ms"),
                          "ue2_index_reference": 78.15, "ue2_stored_nowg_index": (ue2 or {}).get("nowg_index"),
                          "skill_minus_ue2": {k: round(v - ue2["skill"][k], 2) for k, v in pub.get("skill", {}).items() if ue2 and k in ue2.get("skill", {})} if ue2 else None,
                          "verdict_by_prereg": "GAIN (>= 78.45)" if x >= 78.45 else ("NO CHANGE (77.85 .. 78.45)" if x >= 77.85 else "WORSE (< 77.85)"),
                          "winoground": "not read (never on this node)"}
else:
    out["public_nowg"] = None
json.dump(out, open(f"{R}/uf2_result.json", "w"), indent=1)
print(json.dumps({"dev": {k: v["index"] for k, v in out["dev_gates"].items()}, "skipped": out["train"]["rows_skipped"],
                  "ade_rows_trained": (out.get("rows_check") or {}).get("ade_rows_in_logged_steps_without_skips"),
                  "rbh": out["rbh"], "held_index_like": {k: (out["held"] or {}).get(k, {}).get("index_like") for k in ("arm", "ref")} if out["held"] else None,
                  "public_nowg": (out["public_nowg"] or {}).get("index"), "verdict": (out["public_nowg"] or {}).get("verdict_by_prereg")}))
