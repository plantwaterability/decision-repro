"""WX: score a read (dir with shard_*/results.jsonl, or one results.jsonl) on the kit rows with the kit scorer; halves sha1(id) % 2; deltas vs
reference reads (paired bootstrap within benchmark, 1,000 reps) and per-row probability agreement with each reference (same answer, total
variation distance) on the rows both have.
usage: python wx_score.py <rows.jsonl> <read> <label> [ref_label=ref_path ...] > score.json"""
import glob, hashlib, json, os, sys
import numpy as np
from decision_index.vision import spec
from decision_index.vision.score import score as kit_score

rows_f, read, label = sys.argv[1:4]
refs = dict(a.split("=", 1) for a in sys.argv[4:])
rows = [json.loads(l) for l in open(rows_f) if l.strip()]


def load(path):
    files = sorted(glob.glob(f"{path}/shard_*/results.jsonl")) if os.path.isdir(path) else [path]
    out = {}
    for f in files:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "ok":
                    a = r["response"]["answers"]["answer"]
                    out[r["run_id"]] = (a["choice"], a.get("probabilities") or {})
    return out


X = load(read)
sub = [r for r in rows if r["id"] in X]
ids = [r["id"] for r in sub]
B = [b["name"] for b in spec.BENCHMARKS]
WT = np.array([b["weight"] for b in spec.BENCHMARKS])
bi = np.array([B.index(r["benchmark"]) for r in sub])
chance = np.array([1.0 / len(r["questions"]["answer"]["criteria"]) for r in sub])
half = np.array([int(hashlib.sha1(i.encode()).hexdigest(), 16) % 2 for i in ids])
present = sorted(set(bi.tolist()))
rng = np.random.default_rng(20261118)
NB = 1000
bidx = {j: np.where(bi == j)[0] for j in present}
samp = {j: rng.integers(0, len(v), size=(NB, len(v))) for j, v in bidx.items()}


def correct(P):
    return np.array([P.get(i, (None,))[0] == r["expected"]["answer"] for i, r in zip(ids, sub)])


def index(c, mask):
    num = den = 0.0
    for j in present:
        m = mask & (bi == j)
        if m.any():
            ch = chance[m].mean()
            num += WT[j] * min(1, max(0, (c[m].mean() - ch) / (1 - ch))); den += WT[j]
    return 100 * num / den


def boot(c):
    tot = np.zeros(NB); den = 0.0
    for j, v in bidx.items():
        cc = c[v][samp[j]].mean(1); ch = chance[v][samp[j]].mean(1)
        tot += WT[j] * np.clip((cc - ch) / (1 - ch), 0, 1); den += WT[j]
    return 100 * tot / den


ALL = np.ones(len(ids), bool)
cX = correct(X)
ks = kit_score(sub, {i: X[i][0] for i in ids})
out = {"label": label, "rows_scored": len(sub), "rows_in_file": len(rows), "complete": len(sub) == len(rows),
       "index": round(index(cX, ALL), 3), "kit_index": round(ks["public_vision_index"], 3),
       "halves": [round(index(cX, half == 0), 3), round(index(cX, half == 1), 3)],
       "skill": {k: round(100 * v["skill_clipped"], 2) for k, v in ks["benchmarks"].items()}, "vs": {}}
bX = boot(cX)
for rl, rp in refs.items():
    P = load(rp)
    common = [n for n, i in enumerate(ids) if i in P]
    cP = correct(P)
    same = np.mean([X[ids[n]][0] == P[ids[n]][0] for n in common]) if common else None
    tv = [0.5 * sum(abs(X[ids[n]][1].get(k, 0.0) - P[ids[n]][1].get(k, 0.0)) for k in set(X[ids[n]][1]) | set(P[ids[n]][1])) for n in common]
    pc = [abs(X[ids[n]][1].get(P[ids[n]][0], 0.0) - P[ids[n]][1].get(P[ids[n]][0], 0.0)) for n in common]
    full_ref = len(common) == len(ids)
    d = {"rows_common": len(common), "same_answer": round(float(same), 4) if same is not None else None,
         "tv_mean": round(float(np.mean(tv)), 4) if tv else None, "tv_p90": round(float(np.percentile(tv, 90)), 4) if tv else None,
         "tv_max": round(float(np.max(tv)), 4) if tv else None, "ref_choice_prob_absdiff_median": round(float(np.median(pc)), 4) if pc else None}
    if full_ref:
        iP = index(cP, ALL)
        d.update({"ref_index": round(iP, 3), "delta": round(out["index"] - iP, 3),
                  "delta_halves": [round(index(cX, half == k) - index(cP, half == k), 3) for k in (0, 1)],
                  "se": round(float(np.std(bX - boot(cP))), 3), "gained": int((cX & ~cP).sum()), "lost": int((~cX & cP).sum()),
                  "skill_delta": {B[j]: round(100 * ((cX[bi == j].mean() - cP[bi == j].mean()) / (1 - chance[bi == j].mean())), 2) for j in present}})
    out["vs"][rl] = d
print(json.dumps(out))
