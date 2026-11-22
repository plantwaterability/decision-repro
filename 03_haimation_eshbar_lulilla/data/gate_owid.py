#!/usr/bin/env python3
"""Decontamination for mem-00's generated global_facts items (OWID statistics store): build_manifest.Gate (suite-0.3 + held-out sets:
exact, 13-grams, short-item containment) on question + options, PLUS a near-duplicate check against the 184 public global_facts rows
(MMLU 100 + MMLU-Pro 84), whose questions are mostly too short for 13-grams: content-word Jaccard >= 0.5, normalized containment either
way, or the same option-number set with Jaccard >= 0.25. Writes the failing qids with reasons.
  python3 gate_owid.py --suite S03 --heldout DIR --out FAIL.jsonl Q.clean.jsonl..."""
import argparse
import glob
import gzip
import json
import re
import sys

from build_manifest import Gate, jsonl, norm_text

STOP = set("the a an of in to for and or by on at as is was were be been what which who whom whose how many much about approximately "
           "roughly around percent percentage share number population people country countries world year years according estimated "
           "this that with from than per total".split())


def words(t):
    return {w for w in re.findall(r"[a-z0-9]+", norm_text(t)) if len(w) >= 3 and w not in STOP}


def nums(xs):
    out = set()
    for x in xs:
        m = re.findall(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
        if len(m) == 1:
            out.add(round(float(m[0]), 3))
    return frozenset(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    gate, pub = Gate(), []
    for f in ("selected-rows", "added-rows", "gsm8k-rows"):
        for r in jsonl(f"{a.suite}/{f}.jsonl.gz"):
            gate.add(r)
            if "global_facts" in json.dumps({k: v for k, v in r.items() if k not in ("questions", "state")}):
                for q in r["questions"].values():
                    st = r.get("state")
                    t = (st.get("question") if isinstance(st, dict) and st.get("question") else "") or q.get("instructions") or ""
                    pub.append((r["id"], norm_text(t), words(t), nums(q["criteria"].values())))
    for p in sorted(glob.glob(f"{a.heldout}/heldout_*.jsonl*") + glob.glob(f"{a.heldout}/kfree_rows.jsonl")):
        for r in jsonl(p):
            gate.add(r)
    print(json.dumps({"gate_exact": len(gate.exact), "gate_13grams": len(gate.grams), "public_global_facts": len(pub)}), flush=True)
    inv = {}
    for i, (_, _, w, _) in enumerate(pub):
        for x in w:
            inv.setdefault(x, []).append(i)
    with open(a.out, "w") as out:
        for path in a.files:
            c = {"in": 0, "fail": 0, "gate": 0, "jaccard": 0, "contain": 0, "optset": 0}
            for r in jsonl(path):
                c["in"] += 1
                q = r["question"]
                ro = r.get("options") or []
                opts = list(ro.values()) if isinstance(ro, dict) else list(ro)
                why = []
                if not gate.clean(q + " " + " ".join(str(o) for o in opts), exact_keys=(q,)):
                    why.append("gate")
                nq, wq, ns = norm_text(q), words(q), nums(opts)
                cand = {i for x in wq for i in inv.get(x, ())}
                best = None
                for i in cand:
                    pid, pt, pw, pn = pub[i]
                    j = len(wq & pw) / max(len(wq | pw), 1)
                    if j >= 0.5:
                        why.append(f"jaccard:{pid}:{j:.2f}")
                    if len(pt) >= 20 and (pt in nq or (len(nq) >= 20 and nq in pt)):
                        why.append(f"contain:{pid}")
                    if pn and ns == pn and j >= 0.25:
                        why.append(f"optset:{pid}:{j:.2f}")
                if why:
                    c["fail"] += 1
                    for k in ("gate", "jaccard", "contain", "optset"):
                        c[k] += any(w.startswith(k) for w in why)
                    out.write(json.dumps({"file": path.rsplit("/", 1)[-1], "qid": r["qid"], "why": why[:6]}) + "\n")
            print(path.rsplit("/", 1)[-1], json.dumps(c), flush=True)


if __name__ == "__main__":
    main()
