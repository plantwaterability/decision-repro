"""vmx: is a public no-WG gain over F5 real? Two FIXED halves of the suite by row id (sha1(id) % 2), the index of each model on each half
(kit scorer), per-benchmark deltas, changed rows, and a paired bootstrap of the index delta (rows resampled within benchmark, 2,000 draws).
usage: mx_halves.py <rows.jsonl> <F5 preds json> <label>=<preds json> [...]      preds json: {"preds": {id: letter}}"""
import hashlib, json, sys
import numpy as np
from decision_index.vision.score import score
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
f5 = json.load(open(sys.argv[2]))["preds"]
half = {r["id"]: int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 2 for r in rows}
H = [[r for r in rows if half[r["id"]] == h] for h in (0, 1)]
W = {"R-Bench-M": 0.5, "MMMU-Pro vision": 0.25}
bench = sorted({r["benchmark"] for r in rows})
ix = {b: np.array([i for i, r in enumerate(rows) if r["benchmark"] == b]) for b in bench}
gold = [r["expected"]["answer"] for r in rows]; ids = [r["id"] for r in rows]
chance = np.array([1.0 / len(r["questions"]["answer"]["criteria"]) for r in rows])
def corr(p): return np.array([float(p.get(i) == g) for i, g in zip(ids, gold)])
def index(c, sel=None):
    tot = ws = 0.0
    for b in bench:
        s = ix[b] if sel is None else sel[b]
        m = chance[s].mean(); sk = (c[s].mean() - m) / (1 - m)
        w = W.get(b, 1.0); tot += w * min(1.0, max(0.0, sk)); ws += w
    return 100 * tot / ws
idx = lambda rs, p: score(rs, dict(p))
sk = lambda s: {k: round(x["skill_clipped"] * 100, 2) for k, x in s["benchmarks"].items()}
c0 = corr(f5)
rng = np.random.default_rng(20261115)
draws = [{b: rng.choice(ix[b], len(ix[b])) for b in bench} for _ in range(2000)]
out = {"rows": len(rows), "half_rows": [len(H[0]), len(H[1])], "F5": {"all": round(idx(rows, f5)["public_vision_index"], 2), "half0": round(idx(H[0], f5)["public_vision_index"], 2), "half1": round(idx(H[1], f5)["public_vision_index"], 2)}}
for a in sys.argv[3:]:
    lab, f = a.split("=", 1); p = json.load(open(f))["preds"]; c = corr(p)
    d = np.array([index(c, s) - index(c0, s) for s in draws])
    rec = {"missing": sum(1 for i in ids if i not in p), "bootstrap": {"delta_mean": round(float(d.mean()), 2), "se": round(float(d.std()), 2), "p_gt0": round(float((d > 0).mean()), 3), "p_ge0.3": round(float((d >= 0.3).mean()), 3), "ci80": [round(float(np.percentile(d, 10)), 2), round(float(np.percentile(d, 90)), 2)]}}
    for name, rs in (("all", rows), ("half0", H[0]), ("half1", H[1])):
        s1, s0 = idx(rs, p), idx(rs, f5); e = {r["id"]: r["expected"]["answer"] for r in rs}
        rec[name] = {"index": round(s1["public_vision_index"], 2), "F5": round(s0["public_vision_index"], 2), "delta": round(s1["public_vision_index"] - s0["public_vision_index"], 2),
                     "gained": sum(1 for i, g in e.items() if p.get(i) == g != f5.get(i)), "lost": sum(1 for i, g in e.items() if p.get(i) != g == f5.get(i)),
                     "skill_delta": {k: round(sk(s1)[k] - sk(s0)[k], 2) for k in sk(s1)}}
    rec["both_halves_ge_0.5"] = rec["half0"]["delta"] >= 0.5 and rec["half1"]["delta"] >= 0.5
    out[lab] = rec
print(json.dumps(out))
