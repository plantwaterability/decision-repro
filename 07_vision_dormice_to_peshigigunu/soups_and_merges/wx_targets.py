"""WX wxs1 targets (prereg AGENT_NOTES 'WX PREREG wxs1'): gold-corrected teacher mix on uoq1's 323,264 training rows.
  p_mix(row) = sum_t w_{b,t} p_t(row), t in {WeMM d3+OCR, vxg2, s2b}, b = the row's benchmark family;
  w_b = simplex weights (grid 0.1) maximising the mean log p_mix(gold) on the teachers' DEV + HELD reads of benchmark b (never a public row);
        benchmarks with < 100 fit rows (and the training-only 'Winoground' family) take the global fit over all dev + HELD rows;
  target = 0.5 one-hot(gold) + 0.5 p_mix (gold is the argmax; an exact tie -> 0.51 / 0.49), written as row["soft"] = {A: p, ...};
  the trainer adds its 0.05 label smoothing on top.
usage: python3 wx_targets.py <manifest ug1_nc.jsonl> <out_dir> TEACHER=<teacher_x.jsonl.gz>:<dev read>:<held read> ...   (reads: dirs with
shard_*/results.jsonl or a results.jsonl; dev rows file + HELD rows file via env WX_DEV_ROWS / WX_HELD_ROWS)"""
import collections, glob, gzip, hashlib, itertools, json, math, os, sys

man_f, out_dir = sys.argv[1:3]
specs = [a.split("=", 1) for a in sys.argv[3:]]
names = [n for n, _ in specs]
K = len(names)
os.makedirs(out_dir, exist_ok=True)


def read_probs(path):
    files = sorted(glob.glob(f"{path}/shard_*/results.jsonl")) if os.path.isdir(path) else [path]
    out = {}
    for f in files:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "ok":
                    out[r["run_id"]] = r["response"]["answers"]["answer"]["probabilities"]
    return out


def kit_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


EQUAL = os.environ.get("WX_EQUAL_ALL") == "1"  # amendment 20:01Z: equal 1/3 weights everywhere, no input from any set
fit_rows = {} if EQUAL else {"dev": kit_rows(os.environ["WX_DEV_ROWS"]), "held": kit_rows(os.environ["WX_HELD_ROWS"])}
# fit set: rows read by every teacher in both splits; per row: benchmark, gold index, per-teacher prob vectors
F = []
cover = {}
for n, s in specs:
    if EQUAL:
        break
    tf, dev_r, held_r = s.split(":")
    cover[n] = {"dev": read_probs(dev_r), "held": read_probs(held_r)}
for split, rows in fit_rows.items():
    for r in rows:
        rid = (r.get("_evaluation") or {}).get("run_id") or r["id"]
        keys = list(r["questions"]["answer"]["criteria"])
        ps = [cover[n][split].get(rid) for n in names]
        if any(p is None for p in ps):
            continue
        F.append((r["benchmark"], keys.index(r["expected"]["answer"]), [[p.get(k, 0.0) for k in keys] for p in ps], split))
fit_counts = collections.Counter((b, s) for b, _, _, s in F)
GRID = [tuple(x / 10 for x in c) for c in itertools.product(range(11), repeat=K) if sum(c) == 10]


def ll(rows, w):
    tot = 0.0
    for _, g, ps, _ in rows:
        tot += math.log(max(1e-6, sum(w[t] * ps[t][g] for t in range(K))))
    return tot / max(1, len(rows))


def acc(rows, w):
    c = 0
    for _, g, ps, _ in rows:
        m = [sum(w[t] * ps[t][j] for t in range(K)) for j in range(len(ps[0]))]
        c += max(range(len(m)), key=m.__getitem__) == g
    return c / max(1, len(rows))


eq = tuple(1.0 / K for _ in range(K))
best = lambda rows: max(GRID, key=lambda w: (round(ll(rows, w), 9), -sum(abs(a - b) for a, b in zip(w, eq))))
w_glob = eq if EQUAL else best(F)
by = collections.defaultdict(list)
for x in F:
    by[x[0]].append(x)
W = {}
fitrep = {}
for b, rows in by.items():
    w = w_glob if EQUAL else (best(rows) if len(rows) >= 100 else w_glob)
    W[b] = w
    singles = {names[t]: round(acc(rows, tuple(1.0 if u == t else 0.0 for u in range(K))), 4) for t in range(K)}
    fitrep[b] = {"rows": len(rows), "w": dict(zip(names, w)), "ll_mix": round(ll(rows, w), 4), "ll_equal": round(ll(rows, eq), 4),
                 "acc_mix": round(acc(rows, w), 4), "acc_single": singles}
if os.environ.get("WX_FIT_ONLY") == "1":  # preview: the fitted weights and the fit table only (no teacher file needed)
    print(json.dumps({"fit_rows": {f"{b}|{s}": c for (b, s), c in sorted(fit_counts.items())}, "w_global": dict(zip(names, w_glob)), "fit": fitrep}, indent=1))
    sys.exit(0)
# teachers on the training rows
T = []
for n, s in specs:
    tf = s.split(":")[0]
    d = {}
    with gzip.open(tf, "rt") as fh:
        for l in fh:
            r = json.loads(l); d[r["id"]] = r["probs"]
    T.append(d)
md5s = {n: hashlib.md5(open(s.split(":")[0], "rb").read()).hexdigest() for n, s in specs}
man_rows = [json.loads(l) for l in open(man_f) if l.strip()]
man_ids = [r["id"] for r in man_rows]
nopt = {r["id"]: len(r["options"]) for r in man_rows}
checks = {"manifest_rows": len(man_rows), "manifest_ids_unique": len(set(man_ids)) == len(man_ids)}
for n, t, (nn, s) in zip(names, T, specs):
    with gzip.open(s.split(":")[0], "rt") as fh:
        ids = [json.loads(l)["id"] for l in fh]
    bad_nan = sum(1 for i, p in t.items() if any(x != x or x in (float("inf"), float("-inf")) for x in p))
    bad_sum = sum(1 for i, p in t.items() if abs(sum(p) - 1.0) > 1e-3)
    bad_len = sum(1 for i, p in t.items() if i in nopt and len(p) != nopt[i])
    checks[n] = {"lines": len(ids), "unique_ids": len(set(ids)), "join_1to1": len(ids) == len(man_ids) and set(ids) == set(man_ids),
                 "nan": bad_nan, "sum_off": bad_sum, "len_mismatch": bad_len, "md5": md5s[n]}
ok = checks["manifest_ids_unique"] and all(checks[n]["join_1to1"] and checks[n]["lines"] == len(man_ids) == checks[n]["unique_ids"]
                                           and checks[n]["nan"] == 0 and checks[n]["sum_off"] == 0 and checks[n]["len_mismatch"] == 0 for n in names)
json.dump(checks, open(os.path.join(out_dir, "teacher_checks.json"), "w"), indent=1)
print(json.dumps({"teacher_checks_ok": ok, "checks": checks}), flush=True)
if not ok:
    sys.exit(9)
outp = os.path.join(out_dir, "wxs1_nc.jsonl")
stats = collections.Counter()
with open(outp, "w") as fo:
    for r in man_rows:
        k = len(r["options"]); g = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(r["gold"])
        ps = [t[r["id"]] for t in T]
        assert all(len(p) == k for p in ps), r["id"]
        fam = r.get("family") or r.get("benchmark")  # manifest rows carry 'family' (= the DI benchmark name, or the training-only 'Winoground')
        w = W.get(fam, w_glob)
        stats["fam_" + str(fam) + ("" if fam in W else "_globalw")] += 1
        m = [sum(w[t] * ps[t][j] for t in range(K)) for j in range(k)]
        z = sum(m) or 1.0
        m = [x / z for x in m]
        tgt = [0.5 * x for x in m]; tgt[g] += 0.5
        if any(tgt[j] >= tgt[g] for j in range(k) if j != g):
            tgt = [0.49 * x for x in m]; tgt[g] += 0.51; stats["tie_fixed"] += 1
        stats["rows"] += 1; stats["mix_right"] += max(range(k), key=m.__getitem__) == g
        r["soft"] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"[j]: round(tgt[j], 6) for j in range(k)}
        fo.write(json.dumps(r) + "\n")
sha = hashlib.sha256(open(outp, "rb").read()).hexdigest()
hdr = {"rows": stats["rows"], "rows_sha256": sha, "source_manifest": man_f, "source_sha256": hashlib.sha256(open(man_f, "rb").read()).hexdigest(),
       "teachers": dict(specs), "teacher_md5": md5s, "fit_rows": {f"{b}|{s}": c for (b, s), c in sorted(fit_counts.items())},
       "w_global": dict(zip(names, w_glob)), "w_by_benchmark": {b: dict(zip(names, w)) for b, w in W.items()}, "fit": fitrep,
       "mix_right_on_train": round(stats["mix_right"] / stats["rows"], 4), "tie_fixed": stats["tie_fixed"],
       "rows_by_family": {k[4:]: v for k, v in stats.items() if k.startswith("fam_")},
       "rule": ("target = 0.5 one-hot(gold) + 0.5 x mean of the three teachers (EQUAL weights, amendment 20:01Z); trainer smoothing 0.05" if EQUAL else "target = 0.5 one-hot(gold) + 0.5 p_mix; weights per benchmark by max mean log p_mix(gold) on dev + HELD (grid 0.1); trainer smoothing 0.05"), "teacher_checks": checks}
json.dump(hdr, open(os.path.join(out_dir, "wxs1_nc.header.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in hdr.items() if k not in ("fit", "fit_rows")}))
