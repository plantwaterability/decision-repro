#!/usr/bin/env python3
"""tz_unseen_score.py (TZ; prereg "PREREG TZ" Q2b / Q3(b)): accuracy (argmax == argmax(gold), ties -> first in the shown order) on the converted unseen
sets for probability reads ({"id","p"}) and MX/TB1's stored our-engine reads ({"uid","pred","gold"} over the shown order), on the questions every listed
model answered; micro + per-set mean; halves (sha1 group % 2, from tz_convert); paired difference vs a reference with a bootstrap se; probability
ensembles (equal, and weights fitted on one half by coordinate ascent on accuracy and read on the other half, both ways).
  tz_unseen_score.py DIR OUT.json   (DIR holds SET.req.jsonl.gz, SET.<model>.p.jsonl.gz, reads_<model>_SET.rows.jsonl.gz)"""
import collections, glob, gzip, hashlib, json, os, random, sys
DIR, OUT = sys.argv[1], sys.argv[2]
GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
def load(p): return [json.loads(l) for l in gzip.open(p, "rt")]
res = {}
for s in ("gde1v3s", "dev"):
    req = {r["id"]: r for r in load(f"{DIR}/{s}.req.jsonl.gz")}
    gold = {}
    for i, r in req.items():
        ks = r["keys"]; g = r["gold"]; gold[i] = max(ks, key=lambda k: (g.get(k, 0.0), -ks.index(k)))
    P, pred, stat = {}, {}, {}
    for f in sorted(glob.glob(f"{DIR}/{s}.*.p.jsonl.gz")):
        m = os.path.basename(f)[len(s) + 1:-len(".p.jsonl.gz")]; d = {}; err = 0
        for r in load(f):
            if "p" in r and r["id"] in req:
                ks = req[r["id"]]["keys"]
                if set(r["p"]) == set(ks): d[r["id"]] = [float(r["p"][k]) for k in ks]
                else: err += 1
            else: err += 1
        P[m] = d; pred[m] = {i: req[i]["keys"][max(range(len(v)), key=lambda j: (v[j], -j))] for i, v in d.items()}; stat[m] = {"usable": len(d), "errors_or_mismatch": err}
    for f in sorted(glob.glob(f"{DIR}/reads_*_{s}.rows.jsonl.gz")):
        m = "stored_" + os.path.basename(f)[len("reads_"):-len(f"_{s}.rows.jsonl.gz")]; d = {}; gmis = 0
        for r in load(f):
            if r["uid"] in req:
                ks = req[r["uid"]]["keys"]
                if r["pred"] < len(ks): d[r["uid"]] = ks[r["pred"]]
                if r["gold"] < len(ks) and ks[r["gold"]] != gold[r["uid"]]: gmis += 1
        pred[m] = d; stat[m] = {"usable": len(d), "stored_gold_differs_from_ours": gmis}
    models = sorted(pred); common = sorted(set(req).intersection(*[set(pred[m]) for m in models]))
    fam = {i: req[i]["fam"] for i in common}; half = {i: req[i]["half"] for i in common}
    def acc(pm, ids):
        return 100.0 * sum(pm[i] == gold[i] for i in ids) / max(1, len(ids))
    def setmean(pm, ids):
        by = collections.defaultdict(list)
        for i in ids: by[fam[i]].append(i)
        return sum(acc(pm, v) for v in by.values()) / max(1, len(by))
    H = {h: [i for i in common if half[i] == h] for h in (0, 1)}
    table = {m: {"micro": round(acc(pred[m], common), 3), "set_mean": round(setmean(pred[m], common), 3), "half0": round(acc(pred[m], H[0]), 3), "half1": round(acc(pred[m], H[1]), 3)} for m in models}
    ref = "karpo" if "karpo" in pred else "stored_mx_karpo"
    rng = random.Random(20261119)
    def boot(a, b, ids, n=1000):
        x = [(pred[a][i] == gold[i]) - (pred[b][i] == gold[i]) for i in ids]; mu = 100.0 * sum(x) / len(x)
        bs = []
        for _ in range(n):
            smp = [x[rng.randrange(len(x))] for _ in x]; bs.append(100.0 * sum(smp) / len(smp))
        m = sum(bs) / n; sd = (sum((v - m) ** 2 for v in bs) / (n - 1)) ** 0.5
        return round(mu, 3), round(sd, 3)
    paired = {m: {"vs": ref, "micro_diff_se": boot(m, ref, common), "half0": round(table[m]["half0"] - table[ref]["half0"], 3), "half1": round(table[m]["half1"] - table[ref]["half1"], 3),
                  "set_mean_diff": round(table[m]["set_mean"] - table[ref]["set_mean"], 3)} for m in models if m != ref}
    # per-set lg vs ref
    byfam = collections.defaultdict(list)
    for i in common: byfam[fam[i]].append(i)
    fams = {f: {m: round(acc(pred[m], v), 1) for m in models} | {"n": len(v)} for f, v in sorted(byfam.items())}
    # ensembles over probability readers
    def ens_pred(ws, ids):
        out = {}
        for i in ids:
            tot = sum(w for w in ws.values() if w > 0); v = None
            for m, w in ws.items():
                if w <= 0: continue
                pv = P[m][i]; v = [w * x for x in pv] if v is None else [a + w * x for a, x in zip(v, pv)]
            out[i] = req[i]["keys"][max(range(len(v)), key=lambda j: (v[j], -j))]
        return out
    def fit(members, A):
        ws = {m: 1.0 for m in members}; best = acc(ens_pred(ws, A), A)
        for _ in range(3):
            moved = False
            for m in members:
                for x in GRID:
                    if x == ws[m]: continue
                    t = dict(ws, **{m: x})
                    if not any(t.values()): continue
                    v = acc(ens_pred(t, A), A)
                    if v > best + 1e-9: best, ws, moved = v, t, True
            if not moved: break
        return ws, best
    ens = {}
    if all(m in P for m in ("karpo", "lg")):
        sets = {"karpo+lg": {"karpo": 1, "lg": 1}, "karpo2+lg": {"karpo": 2, "lg": 1}, "karpo+lg2": {"karpo": 1, "lg": 2}}
        if "kudhab" in P: sets.update({"kudhab+lg": {"kudhab": 1, "lg": 1}, "karpo+kudhab": {"karpo": 1, "kudhab": 1}, "karpo+kudhab+lg": {"karpo": 1, "kudhab": 1, "lg": 1}})
        for lab, ws in sets.items():
            pe = ens_pred(ws, common)
            ens[lab] = {"micro": round(acc(pe, common), 3), "half0": round(acc(pe, H[0]), 3), "half1": round(acc(pe, H[1]), 3), "set_mean": round(setmean(pe, common), 3),
                        "vs_karpo": {"micro": round(acc(pe, common) - table["karpo"]["micro"], 3), "half0": round(acc(pe, H[0]) - table["karpo"]["half0"], 3), "half1": round(acc(pe, H[1]) - table["karpo"]["half1"], 3)}}
        for members in (["karpo", "lg"],) + ((["karpo", "kudhab", "lg"],) if "kudhab" in P else ()):
            for a, b in ((0, 1), (1, 0)):
                ws, pa = fit(members, H[a]); pb = ens_pred(ws, H[b]); tw = sum(ws.values())
                ens[f"fit_{'+'.join(members)}_on_half{a}"] = {"weights": {m: round(w / tw, 3) for m, w in ws.items()}, "fit_half_acc": round(pa, 3), f"read_half{b}": round(acc(pb, H[b]), 3),
                                                                f"read_half{b}_vs_karpo": round(acc(pb, H[b]) - table["karpo"][f"half{b}"], 3)}
    res[s] = {"questions_parsed": len(req), "common": len(common), "halves": [len(H[0]), len(H[1])], "models": stat, "table": table, "paired": paired, "ensembles": ens, "per_set": fams}
json.dump(res, open(OUT, "w"), indent=1)
for s, r in res.items():
    print(f"== {s}: parsed {r['questions_parsed']}, common {r['common']} (halves {r['halves']})")
    for m, t in sorted(r["table"].items(), key=lambda kv: -kv[1]["micro"]):
        pdx = r["paired"].get(m)
        print(f"   {m:22s} micro {t['micro']:6.2f} set-mean {t['set_mean']:6.2f} halves {t['half0']:6.2f} / {t['half1']:6.2f}" + (f"  vs {pdx['vs']}: {pdx['micro_diff_se'][0]:+.2f} (se {pdx['micro_diff_se'][1]:.2f}), halves {pdx['half0']:+.2f} / {pdx['half1']:+.2f}, set-mean {pdx['set_mean_diff']:+.2f}" if pdx else ""))
    for k, v in r["ensembles"].items(): print("   ENS", k, json.dumps(v))
