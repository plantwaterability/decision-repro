#!/usr/bin/env python3
"""kl_lr_choice.py EVALS_2.5e-6 EVALS_1e-6 EVALS_5e-6 (KL agent): K1's lr from the three 200-step dev reads.
Rule v2 (AGENT_NOTES "DEVIATION KL lr rule", fixed 11:4xZ 11-15 after the 2.5e-6 arm and the 1e-6 arm's step 100 were seen, before 1e-6's step 200
and before any 5e-6 read): 2.5e-6 (the ordered default) stays unless another arm's dev NLL at step 200 is lower than 2.5e-6's by more than 0.01;
then that arm (the lower of the two if both qualify); 5e-6 must also not rise from step 100 to step 200. Prints "LR reason".
(Rule v1, preregistered: "1e-6 if 2.5e-6's dev NLL at 200 is above its step-0 value ..." presumed the rise grows with lr; at step 100 it did not.)"""
import json, sys
def ev(f): return {r["step"]: r for r in map(json.loads, open(f))}
m, lo, hi = (ev(f) for f in sys.argv[1:4])
n0, m2, l2, h1, h2 = m[0]["nll"], m[200]["nll"], lo[200]["nll"], hi[100]["nll"], hi[200]["nll"]
tab = (f"dev NLL step 0 {n0:.4f} | step 100: 1e-6 {lo[100]['nll']:.4f}, 2.5e-6 {m[100]['nll']:.4f}, 5e-6 {h1:.4f} | step 200: 1e-6 {l2:.4f} (acc {lo[200]['acc']:.4f}), "
       f"2.5e-6 {m2:.4f} (acc {m[200]['acc']:.4f}), 5e-6 {h2:.4f} (acc {hi[200]['acc']:.4f}) | rule v1 would have said {'1e-6' if (m2 > n0 or m2 > l2 + 0.01) else ('5e-6' if (h2 < m2 - 0.02 and h2 <= h1) else '2.5e-6')}")
cand = []
if l2 < m2 - 0.01: cand.append((l2, "1e-6"))
if h2 < m2 - 0.01 and h2 <= h1: cand.append((h2, "5e-6"))
if cand: print(min(cand)[1], "rule v2, clearly lower dev NLL at 200:", tab)
else: print("2.5e-6", "rule v2, default stays:", tab)
