"""Cached-OCR parity runner, comparison, latency summary, and official 12,210-row assembly for peshtur."""
import copy
import glob
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from decision_index.vision.score import score_run


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_read_cached(argv):
    """Run exported peshtur.Engine with precomputed OCR cache (for exact parity vs v4b2 reference read)."""
    repo_dir, rows_f, img_root, out_dir, ocr_jsonl = argv[:5]
    sys.path.insert(0, str(Path(repo_dir).resolve()))
    import peshtur

    # Load cached OCR identically to qv_ocr.load_ocr
    ocr_map = {}
    for l in open(ocr_jsonl):
        r = json.loads(l)
        ocr_map[r["image"]] = peshtur.ocr_text(r["lines"])

    class CachedEngine(peshtur.Engine):
        def read_text(self, ref):
            return ocr_map[ref]

    eng = CachedEngine(model=repo_dir, image_root=img_root, warmup_tokens=0, verify=1)
    rows = [json.loads(l) for l in open(rows_f) if l.strip()]
    os.makedirs(out_dir, exist_ok=True)
    out_p = Path(out_dir) / "results.jsonl"
    t0 = time.time()
    with open(out_p, "w", encoding="utf-8") as fh:
        for row in rows:
            ev = row["_evaluation"]
            ts = time.perf_counter()
            resp, raw = eng(row.get("state"), row["questions"], images=row.get("images", ()))
            ms = (time.perf_counter() - ts) * 1000.0
            rec = {
                "run_id": ev["run_id"],
                "benchmark": ev["benchmark"],
                "set": ev["set"],
                "payload_sha256": ev["payload_sha256"],
                "image_sha256": ev["image_sha256"],
                "engine": "peshtur:Engine",
                "status": "ok",
                "response": resp,
                "total_wall_ms": ms,
                "model_request_wall_ms": ms,
            }
            fh.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(f"CACHED_READ_OK rows={len(rows)} sec={time.time()-t0:.1f}", flush=True)


def _load_results(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files and os.path.exists(pattern):
        files = [pattern]
    out = {}
    for f in files:
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("status") == "ok":
                out[r["run_id"]] = r
    return out


def cmd_cmp(argv):
    rows_f, new_pat, ref_pat, out_json, label = argv[:5]
    new_m = _load_results(new_pat)
    ref_m = _load_results(ref_pat)
    rows = [json.loads(l) for l in open(rows_f) if l.strip()]
    want_ids = [r["_evaluation"]["run_id"] for r in rows]
    common = [rid for rid in want_ids if rid in new_m and rid in ref_m]
    diff_ans, dps, exact_p = 0, [], 0
    for rid in common:
        a = new_m[rid]["response"]["answers"]["answer"]
        b = ref_m[rid]["response"]["answers"]["answer"]
        if a["choice"] != b["choice"]:
            diff_ans += 1
        pa, pb = a["probabilities"], b["probabilities"]
        md = max(abs(float(pa[k]) - float(pb[k])) for k in pa)
        dps.append(md)
        if md == 0.0:
            exact_p += 1
    dps.sort()
    n = len(common)
    q = lambda f: dps[min(n - 1, int(f * n))] if n else None  # noqa: E731
    # Score both on shared rows
    tmp_dir = Path(out_json).parent / f"_tmp_{label}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    common_set = set(common)
    r_sub = tmp_dir / "rows.jsonl"
    r_sub.write_text("\n".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) for r in rows if r["_evaluation"]["run_id"] in common_set) + "\n")
    n_sub = tmp_dir / "new.jsonl"
    n_sub.write_text("\n".join(json.dumps(new_m[rid], separators=(",", ":"), ensure_ascii=False) for rid in common) + "\n")
    f_sub = tmp_dir / "ref.jsonl"
    f_sub.write_text("\n".join(json.dumps(ref_m[rid], separators=(",", ":"), ensure_ascii=False) for rid in common) + "\n")
    sn = score_run(r_sub, n_sub)["public_vision_index"]
    sr = score_run(r_sub, f_sub)["public_vision_index"]
    out = {
        "label": label,
        "common": n,
        "answers_differ": diff_ans,
        "same_answer_frac": round((n - diff_ans) / n, 6) if n else 0.0,
        "probabilities_identical": exact_p,
        "mean_abs_dp": round(sum(dps) / n, 6) if n else None,
        "med_abs_dp": round(q(0.5), 6) if n else None,
        "p95_abs_dp": round(q(0.95), 6) if n else None,
        "p99_abs_dp": round(q(0.99), 6) if n else None,
        "max_abs_dp": round(dps[-1], 6) if n else None,
        "index_new_shared": round(sn, 4),
        "index_ref_shared": round(sr, 4),
        "index_diff": round(sn - sr, 4),
    }
    Path(out_json).write_text(json.dumps(out, indent=2) + "\n")
    print("CMP " + json.dumps(out), flush=True)


def cmd_lat(argv):
    res_f, env_f, mem_f, out_json, label = argv[:5]
    rows = [json.loads(l) for l in open(res_f) if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    ms = sorted(r["total_wall_ms"] for r in ok)
    n = len(ms)
    q = lambda f: ms[min(n - 1, int(f * n))]  # noqa: E731
    env = json.load(open(env_f))
    mem = []
    if os.path.exists(mem_f):
        for l in open(mem_f):
            p = l.split()
            if len(p) >= 2 and p[1].isdigit():
                mem.append(int(p[1]))
    raw = [r.get("raw_output") or {} for r in ok]
    by = {}
    for r in ok:
        by.setdefault(r["benchmark"], []).append(r["total_wall_ms"])
    med = lambda v: statistics.median(v) if v else None  # noqa: E731
    out = {
        "label": label,
        "n": n,
        "errors": len(rows) - n,
        "median_ms": round(statistics.median(ms), 2),
        "mean_ms": round(sum(ms) / n, 2),
        "p80_ms": round(q(0.8), 2),
        "p95_ms": round(q(0.95), 2),
        "p99_ms": round(q(0.99), 2),
        "max_ms": round(ms[-1], 2),
        "over_1000_ms": sum(x > 1000 for x in ms),
        "board_rule_each_le_1000": statistics.median(ms) <= 1000 and (sum(ms) / n) <= 1000 and q(0.8) <= 1000,
        "ocr_ms_median": round(med([x["ocr_ms"] for x in raw if "ocr_ms" in x]), 2) if any("ocr_ms" in x for x in raw) else None,
        "engine_ms_median": round(med([x["engine_ms"] for x in raw if "engine_ms" in x]), 2) if any("engine_ms" in x for x in raw) else None,
        "input_tokens_median": med([x["input_tokens"] for x in raw if "input_tokens" in x]),
        "input_tokens_max": max([x["input_tokens"] for x in raw if "input_tokens" in x], default=None),
        "image_tokens_max": max([x["image_tokens"] for x in raw if "image_tokens" in x], default=None),
        "loaded_seconds": round(env.get("loaded_seconds", 0.0), 2),
        "gpu": env.get("gpu"),
        "gpu_mem_max_mib": max(mem) if mem else None,
        "by_benchmark_median_ms": {b: round(statistics.median(v), 1) for b, v in sorted(by.items())},
    }
    Path(out_json).write_text(json.dumps(out, indent=2) + "\n")
    print("LAT " + json.dumps(out), flush=True)


def cmd_assemble(argv):
    """Assemble all 12,210 official rows (h200 4-part no-WG + h200-winoground) into OUT_DIR/runs/peshtur-vision-0.3/."""
    name, tmp_dir, out_dir = argv[0], Path(argv[1]), Path(argv[2])
    specs = argv[3:]
    KEYS = {
        "run_id", "benchmark", "set", "payload_sha256", "image_sha256", "started_utc",
        "engine", "status", "response", "completed_utc", "total_wall_ms", "model_request_wall_ms",
    }
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def lat(ms):
        s = sorted(ms)
        n = len(s)
        q = lambda f: s[min(n - 1, int(f * n))]  # noqa: E731
        return {
            "n": n, "median_ms": statistics.median(s), "mean_ms": sum(s) / n,
            "p80_ms": q(0.8), "p95_ms": q(0.95), "p99_ms": q(0.99), "max_ms": s[-1],
            "over_1000_ms": sum(x > 1000 for x in s),
        }

    parts = []
    for sp in specs:
        lab, runs = sp.split("=", 1)
        P = {"label": lab, "lines": [], "recs": [], "runs": [], "rows_files": []}
        for one in runs.split("+"):
            d, rows_f = (Path(x) for x in one.split("@"))
            raw_lines = [l for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
            outl, recs = [], []
            for line in raw_lines:
                r = json.loads(line)
                assert set(r) <= KEYS, sorted(set(r) - KEYS)
                assert r["status"] == "ok" and r["engine"] == f"{name}:Engine" and r["response"]["model"] == name, (r["run_id"], r["engine"], r["response"].get("model"))
                assert set(r["response"]) <= {"model", "answers", "usage"} and set(r["response"]["answers"]["answer"]) == {"type", "choice", "probabilities"}
                outl.append(json.dumps(r, separators=(",", ":"), ensure_ascii=False))
                recs.append(r)
            want = [json.loads(l)["_evaluation"]["run_id"] for l in rows_f.read_text().splitlines() if l.strip()]
            ids = [r["run_id"] for r in recs]
            assert len(ids) == len(set(ids)) and set(ids) == set(want), (lab, d, len(ids), len(want))
            env = json.loads((d / "environment.json").read_text())
            assert env["rows_sha256"] == sha256_file(rows_f), (lab, d, "rows_sha256 mismatch")
            assert env["engine_options"] == {}, (lab, d, env["engine_options"])
            assert env["model_source"]["weights_source"] == "repository directory", (lab, d)
            assert env["model_source"]["ocr"]["source"] == "hub", (lab, d)
            assert env["model_source"]["files_verified"] is True, (lab, d)
            env["rows_path"] = rows_f.name
            st = json.loads((d / "status.json").read_text())
            P["lines"] += outl
            P["recs"] += recs
            P["runs"].append({"env": env, "status": st, "dir": str(d)})
            P["rows_files"].append(rows_f)
        P["latency"] = lat([r["total_wall_ms"] for r in P["recs"]])
        parts.append(P)

    allids = [r["run_id"] for P in parts for r in P["recs"]]
    assert len(allids) == len(set(allids)) == 12210, f"expected 12210 unique rows, got {len(allids)}"
    e0 = parts[0]["runs"][0]["env"]
    for P in parts:
        for R in P["runs"]:
            e = R["env"]
            assert e["model_source"] == e0["model_source"] and e["torch"] == e0["torch"] and e["latency"] == e0["latency"], "settings mismatch across parts"

    rows_all = tmp_dir / "vision-rows.jsonl"
    seen, buf = set(), []
    for P in parts:
        for rf in P["rows_files"]:
            for l in rf.read_text().splitlines():
                if l.strip():
                    rid = json.loads(l)["_evaluation"]["run_id"]
                    assert rid not in seen
                    seen.add(rid)
                    buf.append(l)
    rows_all.write_text("\n".join(buf) + "\n")

    main_dir = out_dir / f"runs/{name}-vision-0.3"
    (main_dir / "parts").mkdir(parents=True, exist_ok=True)
    (main_dir / "results.jsonl").write_text("\n".join(l for P in parts for l in P["lines"]) + "\n")
    full = score_run(rows_all, main_dir / "results.jsonl", main_dir)
    assert full["complete"] is True and full["missing_rows"] == 0, full

    flat = []
    for P in parts:
        for i, (R, rf) in enumerate(zip(P["runs"], P["rows_files"])):
            nm = P["label"] if len(P["runs"]) == 1 else f"{P['label']}-{i + 1}of{len(P['runs'])}"
            ids = {json.loads(l)["_evaluation"]["run_id"] for l in rf.read_text().splitlines() if l.strip()}
            flat.append((nm, R, [r for r in P["recs"] if r["run_id"] in ids]))
            (main_dir / "parts" / f"environment.{nm}.json").write_text(json.dumps(R["env"], indent=2) + "\n")
            (main_dir / "parts" / f"status.{nm}.json").write_text(json.dumps(R["status"]) + "\n")

    envs = [f[1]["env"] for f in flat]
    budget = {
        "max_image_pixels": max(e["effective_image_budget"]["max_image_pixels"] for e in envs),
        "max_image_tokens": max(e["effective_image_budget"]["max_image_tokens"] for e in envs),
        "rows_with_image_tokens": sum(e["effective_image_budget"]["rows_with_image_tokens"] for e in envs),
    }
    env = {
        "engine": e0["engine"],
        "engine_options": {},
        "model_source": e0["model_source"],
        "torch": e0["torch"],
        "latency": e0["latency"],
        "rows": len(allids),
        "rows_path": "vision-rows.jsonl",
        "effective_image_budget": budget,
        "parts": [
            {
                "part": nm,
                "gpu": R["env"]["gpu"],
                "rows": len(recs),
                "benchmarks": sorted({r["benchmark"] for r in recs}),
                "rows_sha256": R["env"]["rows_sha256"],
                "loaded_seconds": R["env"]["loaded_seconds"],
                "effective_image_budget": R["env"]["effective_image_budget"],
            }
            for nm, R, recs in flat
        ],
        "note": "Parts made with the same files, engine and settings, merged; see parts/.",
    }
    (main_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    method = e0["latency"] + " One request at a time per process (--workers 1)."
    (main_dir / "latency.json").write_text(
        json.dumps({"method": method, "parts": {P["label"]: {"gpu": P["runs"][0]["env"]["gpu"], "processes": len(P["runs"]), **P["latency"]} for P in parts}}, indent=2) + "\n"
    )

    # Also compute no-WG 10-benchmark score for comparison
    rows_nowg = tmp_dir / "vision-rows-noWG.jsonl"
    rows_nowg.write_text("\n".join(l for l in buf if json.loads(l)["benchmark"] != "Winoground") + "\n")
    res_nowg = tmp_dir / "results-noWG.jsonl"
    res_nowg.write_text("\n".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) for P in parts for r in P["recs"] if r["benchmark"] != "Winoground") + "\n")
    nowg_idx = score_run(rows_nowg, res_nowg)["public_vision_index"]

    summary = {
        "name": name,
        "full_index": round(full["public_vision_index"], 4),
        "nowg_index": round(nowg_idx, 4),
        "winoground_skill": round(full["benchmarks"]["Winoground"]["skill_clipped"] * 100.0, 2),
        "complete": full["complete"],
        "missing_rows": full["missing_rows"],
        "rows": len(allids),
        "benchmarks": {
            b: {
                "rows": d["rows"],
                "answered": d["answered"],
                "accuracy": round(d["accuracy"] * 100.0, 2),
                "skill": round(d["skill_clipped"] * 100.0, 2),
                "weight": d["weight"],
            }
            for b, d in full["benchmarks"].items()
        },
        "latency": {P["label"]: {"gpu": P["runs"][0]["env"]["gpu"], "processes": len(P["runs"]), **P["latency"]} for P in parts},
        "budget": budget,
        "results_sha256": sha256_file(main_dir / "results.jsonl"),
    }
    (tmp_dir / "assemble_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ASSEMBLE_OK " + json.dumps({k: summary[k] for k in ("name", "full_index", "nowg_index", "winoground_skill", "complete", "rows")}), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    fn = {"read_cached": cmd_read_cached, "cmp": cmd_cmp, "lat": cmd_lat, "assemble": cmd_assemble}[cmd]
    fn(sys.argv[2:])
