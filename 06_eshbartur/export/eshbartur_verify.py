"""Parity check, 8-way suite-0.3 sharding, merge + validation, and latency summary for eshbartur."""
import collections
import gzip
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from decision_index.runner import iter_rows
from decision_index.suite.io import Suite, read_jsonl


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_parity(argv):
    """Compare exported eshbartur:Engine vs reference T4BEngine(T=1.40) across choice, noul, score, multi-q, RE2 & long rows."""
    repo_dir, base_dir, ckpt_dir, t4b_code_dir, s4400_gz, out_json = argv[:6]
    sys.path.insert(0, str(Path(repo_dir).resolve()))
    sys.path.insert(0, str(Path(t4b_code_dir).resolve()))
    import eshbartur
    import t4b_engine

    # Select ~240 diverse rows covering choice, noul, score, multi-q, and long states
    all_rows = list(read_jsonl(Path(s4400_gz)))
    buckets = {"noul": [], "score": [], "multi": [], "long": [], "choice": []}
    for r in all_rows:
        qs = r["questions"]
        kinds = {q["type"] for q in qs.values()}
        s_len = len(json.dumps(r["state"], ensure_ascii=False))
        if "noul" in kinds and len(buckets["noul"]) < 50:
            buckets["noul"].append(r)
        elif "score" in kinds and len(buckets["score"]) < 50:
            buckets["score"].append(r)
        elif len(qs) > 1 and len(buckets["multi"]) < 50:
            buckets["multi"].append(r)
        elif s_len > 9000 and len(buckets["long"]) < 40:
            buckets["long"].append(r)
        elif len(buckets["choice"]) < 60:
            buckets["choice"].append(r)
    seen, sample = set(), []
    for k in ("noul", "score", "multi", "long", "choice"):
        for r in buckets[k]:
            rid = r["_evaluation"]["run_id"]
            if rid not in seen:
                seen.add(rid)
                sample.append(r)

    # 1. Run exported standalone eshbartur:Engine
    eshbartur.BIDIR_STATE["calls"] = 0
    eng_new = eshbartur.Engine(model=repo_dir, device="cuda:0", warmup_tokens=0, verify=1)
    out_new = {}
    for r in sample:
        rid = r["_evaluation"]["run_id"]
        resp, _ = eng_new(r["state"], r["questions"])
        out_new[rid] = resp["answers"]
    bidir_calls = eshbartur.BIDIR_STATE["calls"]
    del eng_new
    import torch

    torch.cuda.empty_cache()

    # 2. Run reference T4BEngine(temperature=1.40)
    eng_ref = t4b_engine.T4BEngine(model=base_dir, ckpt=ckpt_dir, temperature=1.40, device="cuda:0")
    out_ref = {}
    for r in sample:
        rid = r["_evaluation"]["run_id"]
        resp, _ = eng_ref(r["state"], r["questions"])
        out_ref[rid] = resp["answers"]
    del eng_ref
    torch.cuda.empty_cache()

    # 3. Compare every question answer
    n_q, n_choice, choice_eq = 0, 0, 0
    prob_diffs, noul_diffs, score_diffs = [], [], []
    for r in sample:
        rid = r["_evaluation"]["run_id"]
        an, ar = out_new[rid], out_ref[rid]
        assert set(an) == set(ar)
        for qk in an:
            n_q += 1
            a, b = an[qk], ar[qk]
            assert a["type"] == b["type"]
            if a["type"] == "choice":
                n_choice += 1
                if a["choice"] == b["choice"]:
                    choice_eq += 1
                md = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
                prob_diffs.append(md)
            elif a["type"] == "noul":
                noul_diffs.append(abs(a["noul"] - b["noul"]))
            elif a["type"] == "score":
                score_diffs.append(abs(a["score"] - b["score"]))
                md = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
                prob_diffs.append(md)

    max_p = max(prob_diffs) if prob_diffs else 0.0
    max_n = max(noul_diffs) if noul_diffs else 0.0
    max_s = max(score_diffs) if score_diffs else 0.0
    res = {
        "rows": len(sample),
        "questions": n_q,
        "buckets": {k: len(v) for k, v in buckets.items()},
        "bidir_calls": bidir_calls,
        "choice_total": n_choice,
        "choice_exact_match": choice_eq,
        "choice_match_frac": round(choice_eq / n_choice, 6) if n_choice else 1.0,
        "max_prob_diff": round(max_p, 8),
        "max_noul_diff": round(max_n, 8),
        "max_score_diff": round(max_s, 8),
        "PARITY_OK": bool(bidir_calls > 0 and choice_eq == n_choice and max_p < 1e-3 and max_n < 1e-3 and max_s < 1e-3),
    }
    Path(out_json).write_text(json.dumps(res, indent=2) + "\n")
    print("PARITY " + json.dumps(res), flush=True)
    if not res["PARITY_OK"]:
        sys.exit(1)


def cmd_shard(argv):
    """Verify suite-0.3 strictly and partition all 140,620 in-edition rows round-robin across N shards."""
    suite_dir, out_dir, n_shards = Path(argv[0]), Path(argv[1]), int(argv[2])
    suite = Suite(suite_dir, "0.3")
    ver = suite.verify(strict=True)
    assert ver.get("match") and ver.get("uncompressed_match") and ver.get("added_match") and ver.get("gsm8k_match"), ver
    out_dir.mkdir(parents=True, exist_ok=True)
    fhs = [open(out_dir / f"shard_{i}.jsonl", "w", encoding="utf-8") for i in range(n_shards)]
    counts = [0] * n_shards
    total = 0
    for row in iter_rows(suite.row_paths, suite.in_edition):
        idx = total % n_shards
        fhs[idx].write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        counts[idx] += 1
        total += 1
    for fh in fhs:
        fh.close()
    assert total == 140620, f"expected 140620 rows, got {total}"
    manifest = {
        "edition": "0.3",
        "total_rows": total,
        "n_shards": n_shards,
        "shard_counts": counts,
        "suite_verify": ver,
        "shard_sha256": {f"shard_{i}.jsonl": sha256_file(out_dir / f"shard_{i}.jsonl") for i in range(n_shards)},
    }
    (out_dir / "shard_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SHARD_OK " + json.dumps({"total_rows": total, "shard_counts": counts}), flush=True)


def cmd_merge(argv):
    """Merge 8 shard results.jsonl into one ordered 140,620-row results.jsonl + results.jsonl.gz + clean environment.json."""
    suite_dir, shards_out_dir, n_shards, final_dir = Path(argv[0]), Path(argv[1]), int(argv[2]), Path(argv[3])
    suite = Suite(suite_dir, "0.3")
    want_ids = [r["_evaluation"]["run_id"] for r in iter_rows(suite.row_paths, suite.in_edition)]
    assert len(want_ids) == 140620

    by_id = {}
    dups = 0
    status_counts = collections.Counter()
    envs = []
    for i in range(n_shards):
        sd = shards_out_dir / f"shard_{i}"
        ep = sd / "environment.json"
        if ep.exists():
            e = json.loads(ep.read_text())
            e["rows_path"] = f"shard_{i}.jsonl"
            envs.append(e)
        rp = sd / "results.jsonl"
        for l in rp.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            rid = r["run_id"]
            if rid in by_id:
                dups += 1
            by_id[rid] = (r, l)
            status_counts[r.get("status")] += 1

    missing = [rid for rid in want_ids if rid not in by_id]
    assert not missing and dups == 0, f"missing={len(missing)} dups={dups}"
    assert status_counts["ok"] == 140620 and len(status_counts) == 1, dict(status_counts)

    # Verify engine/model labels and identical provenance across shards
    e0 = envs[0]
    for e in envs[1:]:
        assert e["engine"] == e0["engine"] and e["model_source"] == e0["model_source"] and e["torch"] == e0["torch"], "env mismatch"

    final_dir.mkdir(parents=True, exist_ok=True)
    res_path = final_dir / "results.jsonl"
    gz_path = final_dir / "results.jsonl.gz"
    wall_ms = []
    with open(res_path, "w", encoding="utf-8") as fh, gzip.open(gz_path, "wt", encoding="utf-8", compresslevel=6) as gz:
        for rid in want_ids:
            r, raw_l = by_id[rid]
            assert r["engine"] == "eshbartur:Engine" and r["response"]["model"] == "eshbartur", rid
            wall_ms.append(float(r["total_wall_ms"]))
            fh.write(raw_l + "\n")
            gz.write(raw_l + "\n")

    wall_ms.sort()
    n = len(wall_ms)
    q = lambda f: wall_ms[min(n - 1, int(f * n))]  # noqa: E731
    merged_env = {
        "engine": e0["engine"],
        "engine_options": e0["engine_options"],
        "model_source": e0["model_source"],
        "gpu": e0["gpu"],
        "device": e0["device"],
        "torch": e0["torch"],
        "loaded_seconds": round(sum(e["loaded_seconds"] for e in envs) / len(envs), 2),
        "frozen_corpus_sha256": suite.verify(strict=True)["sha256"],
        "rows_path": "suite-0.3",
        "runner_sha256": e0.get("runner_sha256"),
        "latency": e0["latency"],
        "sharding": {"shards": n_shards, "gpu": e0["gpu"], "total_rows": len(want_ids)},
    }
    (final_dir / "environment.json").write_text(json.dumps(merged_env, indent=2) + "\n")
    summary = {
        "rows": len(want_ids),
        "ok": status_counts["ok"],
        "errors": 0,
        "unsupported": 0,
        "missing": 0,
        "dups": 0,
        "sharded_h200_ms": {
            "median_ms": round(statistics.median(wall_ms), 2),
            "mean_ms": round(sum(wall_ms) / n, 2),
            "p80_ms": round(q(0.8), 2),
            "p95_ms": round(q(0.95), 2),
            "p99_ms": round(q(0.99), 2),
            "max_ms": round(wall_ms[-1], 2),
        },
        "results_gz_sha256": sha256_file(gz_path),
    }
    (final_dir / "merge.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MERGE_OK " + json.dumps(summary), flush=True)


def cmd_lat(argv):
    res_f, env_f, mem_f, out_json, label = argv[:5]
    rows = [json.loads(l) for l in open(res_f) if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    ms = sorted(float(r["total_wall_ms"]) for r in ok)
    n = len(ms)
    q = lambda f: ms[min(n - 1, int(f * n))]  # noqa: E731
    env = json.load(open(env_f))
    mem = []
    if os.path.exists(mem_f):
        for l in open(mem_f):
            p = l.split()
            if len(p) >= 2 and p[1].isdigit():
                mem.append(int(p[1]))
    by_fam = {}
    for r in ok:
        fam = r.get("family") or r.get("benchmark") or "unknown"
        by_fam.setdefault(fam, []).append(float(r["total_wall_ms"]))
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
        "loaded_seconds": round(env.get("loaded_seconds", 0.0), 2),
        "gpu": env.get("gpu"),
        "gpu_mem_max_mib": max(mem) if mem else None,
        "by_family_median_ms": {b: round(statistics.median(v), 1) for b, v in sorted(by_fam.items())},
    }
    Path(out_json).write_text(json.dumps(out, indent=2) + "\n")
    print("LAT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    fn = {"parity": cmd_parity, "shard": cmd_shard, "merge": cmd_merge, "lat": cmd_lat}[cmd]
    fn(sys.argv[2:])
