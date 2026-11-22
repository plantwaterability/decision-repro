#!/usr/bin/env python3
"""kl_parity_cmp.py make ROWS.jsonl.gz REQS.jsonl | cmp ROWS.jsonl.gz REF A OUT_STATS.json [OUT_RESULTS.jsonl]
make: the kit rows' (state, questions) as requests (id = the row's run_id).
cmp: per question, A's option probabilities against REF's (max |dp|, argmax agreement). REF / A = kit results (.jsonl or .jsonl.gz: an engine run)
or kl_train's predict output (rows with "probs": answers are built with karpo's own answers.answer); with OUT_RESULTS, A as a kit-format results
file (REF's rows with the answers replaced) for the scorer."""
import gzip, json, sys
def rid(r): return r["_evaluation"]["run_id"]
def op(p): return gzip.open(p, "rt") if p.endswith(".gz") else open(p)
if sys.argv[1] == "make":
    with open(sys.argv[3], "w") as f:
        for l in op(sys.argv[2]):
            r = json.loads(l); f.write(json.dumps({"id": rid(r), "state": r["state"], "questions": r["questions"]}) + "\n")
    sys.exit(0)
from karpo_decision.answers import answer
from karpo_decision.protocol import options
rows = {rid(r): r for r in map(json.loads, op(sys.argv[2]))}
def load(p):
    out = {}
    for r in map(json.loads, op(p)):
        if "run_id" in r: out[r["run_id"]] = (r["response"]["answers"], r["response"]["usage"]["input_tokens"], r) if r.get("status") == "ok" else (None, None, r)
        elif "probs" in r: out[r["id"]] = ({k: answer(rows[r["id"]]["questions"][k], v) for k, v in r["probs"].items()}, r["input_tokens"], None)
        else: out[r["id"]] = (None, None, None)
    return out
ref, mine = load(sys.argv[3]), load(sys.argv[4])
def vec(q, ans):
    if q["type"] == "noul": return [1 - ans["noul"], ans["noul"]]
    return [ans["probabilities"][k] for k in options(q)[0]]
nq = same = 0; dps = []; st = {"rows": 0, "both_ok": 0, "both_unsupported": 0, "status_mismatch": 0, "token_mismatch": 0, "missing": 0}; worst = []
f = open(sys.argv[6], "w") if len(sys.argv) > 6 else None
for k, row in rows.items():
    st["rows"] += 1
    if k not in ref or k not in mine: st["missing"] += 1; continue
    (ra, rt, rr), (ma, mt, _) = ref[k], mine[k]
    if ra is None or ma is None:
        st["both_unsupported" if (ra is None and ma is None) else "status_mismatch"] += 1
        if f: f.write(json.dumps(rr) + "\n")
        continue
    st["both_ok"] += 1; st["token_mismatch"] += int(rt != mt)
    for qk, q in row["questions"].items():
        x, y = vec(q, ma[qk]), vec(q, ra[qk]); d = max(abs(u - v) for u, v in zip(x, y)); dps.append(d); nq += 1
        same += int(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__))
        if d > 0.01: worst.append((round(d, 4), k, qk, len(x), mt))
    if f:
        out = json.loads(json.dumps(rr)); out["response"]["answers"] = ma; out["engine"] = "kl_train.predict"; f.write(json.dumps(out) + "\n")
dps.sort(); n = len(dps); worst.sort(reverse=True)
big = [w for w in worst if w[0] > 0.05]
stats = {**st, "questions": nq, "argmax_same": same, "argmax_agreement": round(same / max(nq, 1), 6), "dp_max": dps[-1] if dps else None,
         "dp_p50": dps[n // 2] if dps else None, "dp_p90": dps[int(0.9 * n)] if dps else None, "dp_p99": dps[int(0.99 * n)] if dps else None, "dp_p999": dps[int(0.999 * n)] if dps else None,
         "dp_over_1e-2": sum(d > 1e-2 for d in dps), "dp_over_5e-2": len(big), "median_tokens_of_rows_over_5e-2": sorted(w[4] for w in big)[len(big) // 2] if big else None,
         "worst": [w[:4] for w in worst[:6]]}
json.dump(stats, open(sys.argv[5], "w"), indent=1); print("PARITY", json.dumps(stats))
