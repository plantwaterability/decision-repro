#!/usr/bin/env python3
"""Tool-memory rows from mem-00's held-out ToolRet-Training-20w queries (apigen + toolACE; BM25 top-32 candidates over the full tool
corpus), gated (gate_queries.py). Queries are split by a hash of the qid:
  fit   half -> HEAD-FIT rows in our own phrasing, one yes / no question per prompt (all gold candidates + the top-ranked non-gold ones,
               up to --negatives), for refitting the head with vs without the memory feature;
  eval  half -> DI-SHAPED rows for measurement only, the ToolRet-retrieval shape (state {query, task}; one yes / no question doc_k per
               BM25 candidate, the candidate in the instructions); and (--out-bfcl) for the dialogue-shaped toolACE queries also the
               BFCL-tool-selection shape (state {conversation, functions}: the gold tools + the top-ranked non-gold candidates, up to
               --bfcl-k, in a seeded order; one yes / no question per function). Never trained on.
Every row keeps qid and the candidate tool id per question, so a per-(qid, tool_id) memory score can be attached.

  python3 tool_rows.py --heldout heldout.jsonl --gated heldout_q.gated.jsonl --out-fit FIT.jsonl --out-eval EVAL.jsonl
"""
import argparse
import hashlib
import json
import random

TASK = "Rank candidate documents/tools by relevance to this query."
DOC_TASK = "Assess whether this candidate is relevant/useful to the query. Use the full text below; return the probability of relevance."
DI_CRIT = {"no": "Not relevant/useful to the query.", "yes": "Relevant/useful to the query."}
BFCL_INSTR = ("Given the complete user conversation and the published function schemas in state, mark each candidate tool yes if it should "
              "be invoked to answer the request, or no otherwise. This evaluates tool-name selection only; do not produce arguments or call "
              "ordering.\n\n\n\nCandidate tool: {name}")
BFCL_CRIT = {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}
OWN_INSTR = "Would the tool described below be a useful tool to call for the user's request in state?\n\nTool:\n{doc}"
OWN_CRIT = {"yes": "Yes, this tool is useful for the request.", "no": "No, this tool is not useful for the request."}


def half(qid):
    return "eval" if int(hashlib.sha256(f"toolmem|{qid}".encode()).hexdigest()[:8], 16) % 2 else "fit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--gated", required=True)
    ap.add_argument("--out-fit", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--out-bfcl", default=None)
    ap.add_argument("--bfcl-k", type=int, default=4)
    ap.add_argument("--all-eval", action="store_true", help="every query to the eval half (the seen-tools dev proxy, cross-fit downstream)")
    a = ap.parse_args()
    ok = {json.loads(l)["qid"] for l in open(a.gated)}
    rng = random.Random(20261102)
    bf = []
    fit, ev, c = [], [], {"skipped_gate": 0, "fit_queries": 0, "eval_queries": 0, "eval_no_gold_in_32": 0}
    for l in open(a.heldout):
        r = json.loads(l)
        if r["qid"] not in ok:
            c["skipped_gate"] += 1
            continue
        gold = set(r["gold"])
        cands = r["candidates"]
        if a.all_eval or half(r["qid"]) == "eval":
            c["eval_queries"] += 1
            c["eval_no_gold_in_32"] += not any(x["tool_id"] in gold for x in cands)
            qs, exp, ids = {}, {}, {}
            for k, x in enumerate(cands):
                qs[f"doc_{k}"] = {"type": "choice", "instructions": {"candidate": x["doc"], "task": DOC_TASK}, "criteria": dict(DI_CRIT)}
                exp[f"doc_{k}"] = "yes" if x["tool_id"] in gold else "no"
                ids[f"doc_{k}"] = x["tool_id"]
            ev.append({"id": f"toolmem|{r['qid']}", "family": f"toolmem-{r['source']}", "split": "heldout",
                       "state": {"query": r["query"], "task": TASK}, "questions": qs, "expected": exp,
                       "metadata": {"track": f"toolmem_{r['source']}", "qid": r["qid"], "tool_ids": ids, "n_gold": len(gold)}})
            if a.out_bfcl and r["source"].lower() == "toolace":
                fs = [x for x in cands if x["tool_id"] in gold] + [x for x in cands if x["tool_id"] not in gold]
                fs = fs[:max(a.bfcl_k, sum(x["tool_id"] in gold for x in cands))]
                if not any(x["tool_id"] in gold for x in fs):
                    continue
                rng.shuffle(fs)
                schemas, qs, exp, ids = [], {}, {}, {}
                for x in fs:
                    try:
                        sch = json.loads(x["doc"])
                    except ValueError:
                        sch = {"name": x["tool_id"], "description": x["doc"]}
                    name = str(sch.get("name") or x["tool_id"])
                    schemas.append(sch)
                    key = f"tool_{name}"
                    while key in qs:
                        key += "_"
                    qs[key] = {"type": "choice", "instructions": BFCL_INSTR.format(name=name), "criteria": dict(BFCL_CRIT)}
                    exp[key] = "yes" if x["tool_id"] in gold else "no"
                    ids[key] = x["tool_id"]
                bf.append({"id": f"toolmem-bfcl|{r['qid']}", "family": "toolmem-bfcl-toolace", "split": "heldout",
                           "state": {"conversation": [[{"role": "user", "content": r["query"]}]], "functions": schemas},
                           "questions": qs, "expected": exp,
                           "metadata": {"track": "toolmem_bfcl_toolace", "qid": r["qid"], "tool_ids": ids, "n_gold": len(gold)}})
        else:
            c["fit_queries"] += 1
            pos = [x for x in cands if x["tool_id"] in gold]
            neg = [x for x in cands if x["tool_id"] not in gold][:a.negatives]
            for x in pos + neg:
                fit.append({"id": f"toolmemfit|{r['qid']}|{x['tool_id']}", "family": f"toolmemfit-{r['source']}", "split": "train",
                            "group": f"toolmemfit|{r['qid']}", "state": {"request": r["query"]},
                            "questions": {"answer": {"type": "choice", "instructions": OWN_INSTR.format(doc=x["doc"]), "criteria": dict(OWN_CRIT)}},
                            "expected": {"answer": "yes" if x["tool_id"] in gold else "no"},
                            "meta": {"qid": r["qid"], "tool_id": x["tool_id"], "source": r["source"]}})
    for path, rows in ((a.out_fit, fit), (a.out_eval, ev)) + (((a.out_bfcl, bf),) if a.out_bfcl else ()):
        body = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        open(path, "w").write(body)
        c[path] = {"rows": len(rows), "sha256": hashlib.sha256(body.encode()).hexdigest()}
    c["eval_questions"] = sum(len(x["questions"]) for x in ev)
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
