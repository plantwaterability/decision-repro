"""VL node S: score one public no-WG read against F5's stored images_first read (and, when given, a same-node control read).
usage: vls_score.py <public rows jsonl> <read dir> <label> <F5 preds json> [<control read dir>]
Writes <read dir>/public_noWG.json ({"index", "benchmarks", "preds"}; mx_halves.py input) and prints ONE JSON line: rows read, index, delta vs
F5 on the same rows, halves (sha1(id) % 2) vs F5, per-benchmark skill delta, answers changed / gained / lost, H200 latency, prompt tokens."""
import glob
import hashlib
import json
import statistics
import sys

from decision_index.vision.score import score

rows_f, d, label, f5_f = sys.argv[1:5]
ctl_d = sys.argv[5] if len(sys.argv) > 5 else None
allrows = [json.loads(l) for l in open(rows_f) if l.strip()]
gold = {r["id"]: r["expected"]["answer"] for r in allrows}


def read(dd):
    p, lat, tok, mm = {}, [], [], None
    for f in sorted(glob.glob(f"{dd}/shard_*/results.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("status") == "ok":
                p[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
                if "total_wall_ms" in r:
                    lat.append(r["total_wall_ms"])
                tok.append((r.get("raw_output") or {}).get("input_tokens", 0))
    for f in sorted(glob.glob(f"{dd}/shard_*.log"))[:1]:
        for l in open(f):
            if l.startswith("{") and '"layout"' in l:
                mm = json.loads(l)
    return p, lat, tok, mm


preds, lat, tok, meta = read(d)
f5 = json.load(open(f5_f))["preds"]
half = {r["id"]: int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 2 for r in allrows}
idx = lambda rows, p: score(rows, {k: v for k, v in p.items() if k in {r["id"] for r in rows}})  # noqa: E731
a, b = idx(allrows, preds), idx(allrows, f5)
json.dump({"index": a["public_vision_index"], "benchmarks": a["benchmarks"], "preds": preds}, open(f"{d}/public_noWG.json", "w"))
H = {}
for h in (0, 1):
    rr = [r for r in allrows if half[r["id"]] == h]
    x, y = idx(rr, preds)["public_vision_index"], idx(rr, f5)["public_vision_index"]
    H[f"half{h}"] = {"rows": len(rr), "index": round(x, 2), "F5": round(y, 2), "delta": round(x - y, 2)}
ch = [k for k in preds if k in f5 and preds[k] != f5[k]]
out = {"label": label, "rows": len(allrows), "read": len(preds), "missing": len(allrows) - len(preds), "index": round(a["public_vision_index"], 2),
       "F5_stored": round(b["public_vision_index"], 2), "delta_vs_F5": round(a["public_vision_index"] - b["public_vision_index"], 2), "halves_vs_F5": H,
       "changed_vs_F5": len(ch), "gained": sum(preds[k] == gold.get(k) for k in ch), "lost": sum(f5[k] == gold.get(k) for k in ch),
       "skill_delta_vs_F5": {k: round(100 * (v["skill_clipped"] - b["benchmarks"][k]["skill_clipped"]), 2) for k, v in a["benchmarks"].items()},
       "latency_ms": {"median": round(statistics.median(lat)), "p80": round(sorted(lat)[int(.8 * len(lat))])} if lat else None,
       "input_tokens_median": statistics.median(tok) if tok else None, "engine": meta}
if ctl_d:
    c, clat, _, _ = read(ctl_d)
    ci = idx(allrows, c)["public_vision_index"]
    out["vs_control"] = {"control_index": round(ci, 2), "delta": round(a["public_vision_index"] - ci, 2),
                         "halves": {f"half{h}": round(idx([r for r in allrows if half[r["id"]] == h], preds)["public_vision_index"]
                                                      - idx([r for r in allrows if half[r["id"]] == h], c)["public_vision_index"], 2) for h in (0, 1)},
                         "latency_ratio": round(statistics.median(lat) / statistics.median(clat), 3) if lat and clat else None}
print(json.dumps(out))
