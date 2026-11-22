#!/usr/bin/env python3
"""kl_deltas.py COMPARE_SINGLE.json COMPARE_JOINT.json PREFIX_ROWS.jsonl.gz (KL agent, 11-16): single-view minus joint-view per benchmark (KOWN arm of gate_index's
prefix_compare.json), named benchmarks first, then every multi-question benchmark, then the single-question ones' spread (same request in both views: noise)."""
import gzip, json, sys, collections
S = json.load(open(sys.argv[1]))["arms"]["KOWN"]; J = json.load(open(sys.argv[2]))["arms"]["KOWN"]
names = {}; nq = collections.defaultdict(list)
for l in gzip.open(sys.argv[3], "rt"):
    r = json.loads(l); e = r["_evaluation"]; b = str(e["catalog_id"]); names.setdefault(b, f"{e.get('dataset')} / {r.get('family')}"); nq[b].append(len(r["questions"]))
print(f"INDEX single {S['index']:.3f} joint {J['index']:.3f} delta {S['index'] - J['index']:+.3f} (prefix, 31,198 rows, same node)")
print("AREAS single-joint: " + ", ".join(f"{a} {S['areas'][a] - J['areas'][a]:+.4f}" for a in S["areas"]))
want = ("sata", "acos", "bfcl", "contract", "toolret", "phish", "appliance", "api-bank", "apibank", "api_bank", "api bank")
rows = []
for b in sorted(set(S["bench"]) | set(J["bench"]), key=lambda x: int(x) if x.isdigit() else 999):
    s, j = S["bench"].get(b), J["bench"].get(b); q = nq.get(b, [0])
    rows.append((b, names.get(b, "?"), s, j, None if s is None or j is None else s - j, len(q), sum(q) / max(len(q), 1)))
fmt = lambda x: "  n/a " if x is None else f"{100 * x:6.2f}"
def line(t): b, nm, s, j, d, n, mq = t; return f"  {b:>3} {nm[:46]:46s} single {fmt(s)} joint {fmt(j)} delta {'   n/a' if d is None else f'{100 * d:+6.2f}'} rows {n:5d} q/row {mq:4.1f}"
print("NAMED (catalog id, benchmark, single, joint, delta in points of the benchmark's own score, prefix rows, questions per row):")
for t in rows:
    if any(w in t[1].lower() for w in want): print(line(t))
print("OTHER MULTI-QUESTION BENCHMARKS (> 1.05 questions per row):")
for t in rows:
    if t[6] > 1.05 and not any(w in t[1].lower() for w in want): print(line(t))
one = [t[4] for t in rows if t[6] <= 1.05 and t[4] is not None]
print(f"SINGLE-QUESTION BENCHMARKS {len(one)}: max |delta| {100 * max(abs(x) for x in one):.2f} points, mean {100 * sum(one) / len(one):+.3f} (same request in both views)")
print("NO SCORE IN ONE VIEW: " + ", ".join(f"{t[0]} {t[1][:30]}" for t in rows if t[4] is None))
