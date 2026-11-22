import gzip, json, sys, collections
rows = [json.loads(l) for l in gzip.open(sys.argv[1], "rt")]
qt = collections.Counter(); rt = collections.Counter(); bench = collections.defaultdict(collections.Counter)
for r in rows:
    ts = {q.get("type") for q in r["questions"].values()}
    for q in r["questions"].values(): qt[q.get("type")] += 1
    k = "+".join(sorted(ts)); rt[k] += 1; bench[r["_evaluation"].get("catalog_id")][k] += 1
nb = len(bench); w = collections.Counter()
for b, c in bench.items():
    n = sum(c.values())
    for k, v in c.items(): w[k] += v / n / nb
print("rows", len(rows), "by question types in the row:", dict(rt))
print("questions", sum(qt.values()), dict(qt))
print("benchmarks (catalog ids)", nb, "| equal-weight-per-benchmark share by row type:", {k: round(v, 4) for k, v in w.items()})
print("benchmarks that are all noul:", sorted(str(b) for b, c in bench.items() if set(c) == {"noul"}), "all score:", sorted(str(b) for b, c in bench.items() if set(c) == {"score"}), "mixed:", sorted(str(b) for b, c in bench.items() if len(c) > 1))
