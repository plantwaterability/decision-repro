#!/usr/bin/env python3
"""kl_cmp.py LABEL_A COMPARE_A LABEL_B COMPARE_B LABEL_C COMPARE_C PREFIX_ROWS.jsonl.gz (KL agent, 11-16): per-benchmark prefix scores of model A against two reads of a
reference (B = same node, C = another read), KOWN arm of gate_index's prefix_compare.json, points of each benchmark's own score; the coordinator's gap list first."""
import gzip, json, sys, collections
la, fa, lb, fb, lc, fc, rows = sys.argv[1:8]
A, B, C = (json.load(open(f))["arms"]["KOWN"] for f in (fa, fb, fc))
nm = {}; cnt = collections.Counter()
for l in gzip.open(rows, "rt"):
    r = json.loads(l); e = r["_evaluation"]; b = str(e["catalog_id"]); cnt[b] += 1; nm.setdefault(b, e.get("dataset"))
gap = [("33", "SATA-Bench"), ("23", "cfcolor"), ("25", "GPQA Diamond"), ("22", "POP909"), ("4", "BANKING77"), ("30", "GSM8K"), ("29", "HellaSwag"), ("44", "CLadder")]
also = [("5", "CLINC150 (H0's largest addition)"), ("61", "HoVer (train_hover on the drop list)"), ("56", "PhishNChips (LOOMD-phish_v5 on the drop list)"), ("57", "MMLU-Pro"), ("31", "ChessBench (H0 chess left out)")]
f = lambda x: "   n/a" if x is None else f"{100 * x:6.2f}"
d = lambda x, y: "   n/a" if x is None or y is None else f"{100 * (x - y):+6.2f}"
print(f"INDEX {la} {A['index']:.3f} | {lb} {B['index']:.3f} ({la} - {lb} {A['index'] - B['index']:+.3f}) | {lc} {C['index']:.3f} ({la} - {lc} {A['index'] - C['index']:+.3f})  [fixed prefix, 31,198 rows]")
print("AREAS " + ", ".join(f"{k} {A['areas'][k] - B['areas'][k]:+.4f}" for k in A["areas"]) + f"  ({la} - {lb})")
def row(b, label):
    a, x, y = A["bench"].get(b), B["bench"].get(b), C["bench"].get(b)
    return f"  {b:>3} {label[:44]:44s} {la} {f(a)} | {lb} {f(x)} d {d(a, x)} | {lc} {f(y)} d {d(a, y)} | rows {cnt[b]}"
print(f"GAP LIST (points of the benchmark's own score; d = {la} minus the soup read):")
for b, label in gap: print(row(b, label))
print("ALSO MOVED BY THE DATA CHANGE:")
for b, label in also: print(row(b, label))
ds = sorted(((A["bench"][b] - B["bench"][b], b) for b in A["bench"] if A["bench"].get(b) is not None and B["bench"].get(b) is not None))
print("LARGEST LOSSES vs " + lb + ": " + "; ".join(f"{b} {nm.get(b, '?')[:22]} {100 * x:+.2f}" for x, b in ds[:6]))
print("LARGEST GAINS vs " + lb + ": " + "; ".join(f"{b} {nm.get(b, '?')[:22]} {100 * x:+.2f}" for x, b in ds[::-1][:6]))
