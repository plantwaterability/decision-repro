"""VL round: dev parity. The eval engine's read of the trainer's gate rows (soup_eval_ocr_vl.py shards under <dir>/shard_*/results.jsonl)
against the trainer's own in-process read of the same rows at the final step (gate_<step>.json: preds + option probabilities).
Both must render the same prompt (layout) and use the same weights (LoRA, head, and the merger when it was trained); what remains is the
arithmetic difference between the unmerged fp32 LoRA path (trainer) and the LoRA merged into the bf16 weights (engine).
ok = every row read, >= 93% same answer, |index difference| <= 2.0 (a gross-mismatch guard for the one public read, not a precision claim).
usage: vl_dev_parity.py <gate rows jsonl> <dir with shard_*/results.jsonl> <gate json>"""
import glob
import json
import sys

from decision_index.vision.score import score

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
g = json.load(open(sys.argv[3]))
tp, tprob = g["preds"], g.get("probs", {})
ep, eprob = {}, {}
for f in sorted(glob.glob(sys.argv[2] + "/shard_*/results.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("status") == "ok":
            ep[r["run_id"]] = r["response"]["answers"]["answer"]["choice"]
            pr = (r.get("raw_output") or {}).get("probs")
            if pr:
                eprob[r["run_id"]] = pr
rid = lambda r: (r.get("_evaluation") or {}).get("run_id") or r.get("id")  # noqa: E731
ids = [(r["id"], rid(r)) for r in rows]
both = [(a, b) for a, b in ids if a in tp and b in ep]
agree = sum(tp[a] == ep[b] for a, b in both)
dp = [max(abs(x - y) for x, y in zip(tprob[a], eprob[b])) for a, b in both if a in tprob and b in eprob and len(tprob[a]) == len(eprob[b])]
dp.sort()
e_idx = score(rows, {a: ep[b] for a, b in ids if b in ep})["public_vision_index"] if len(both) == len(rows) else None
out = {"rows": len(rows), "trainer_read": sum(a in tp for a, _ in ids), "engine_read": sum(b in ep for _, b in ids), "same_answer": agree,
       "same_answer_frac": round(agree / max(1, len(both)), 4), "trainer_index": round(g["index"], 2), "engine_index": round(e_idx, 2) if e_idx is not None else None,
       "max_abs_prob_diff": {"median": round(dp[len(dp) // 2], 4), "p95": round(dp[int(.95 * len(dp))], 4), "max": round(dp[-1], 4)} if dp else None,
       "layout": g.get("layout")}
out["ok"] = bool(len(both) == len(rows) and out["same_answer_frac"] >= 0.93 and e_idx is not None and abs(e_idx - g["index"]) <= 2.0)
print(json.dumps(out))
