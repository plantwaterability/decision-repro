#!/usr/bin/env python3
"""Stage-1 v3 tool replay pool from mem-00's train8k.jsonl (8,000 gated ToolRet-Training TRAIN-pool queries; every tool eval set excluded:
toolmem held-out / LOSO pools / dev400). One question per row (the manifest quota counts rows):
  ToolRet shape (all sources): state {query, task}, one yes / no question per candidate (gold + the top-3 BM25 non-gold), as the DI rows;
  BFCL shape (toolACE queries): state {conversation, functions = gold + top-3 BM25 non-gold, seeded order}, one yes / no per function.
Queries without a gold tool in BM25's 32 are skipped. Queries were gated (loso gate); tool documents are the shared ToolRet corpus and
carry no labels, so they are not re-gated.
  python3 v3_tool_rows.py train8k.jsonl OUT.jsonl"""
import hashlib, json, random, sys
TASK = "Rank candidate documents/tools by relevance to this query."
DOC_TASK = "Assess whether this candidate is relevant/useful to the query. Use the full text below; return the probability of relevance."
DI_CRIT = {"no": "Not relevant/useful to the query.", "yes": "Relevant/useful to the query."}
BFCL_INSTR = ("Given the complete user conversation and the published function schemas in state, mark each candidate tool yes if it should "
              "be invoked to answer the request, or no otherwise. This evaluates tool-name selection only; do not produce arguments or call "
              "ordering.\n\n\n\nCandidate tool: {name}")
BFCL_CRIT = {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}
src, outp = sys.argv[1], sys.argv[2]
rng = random.Random(20261103)
out, c = [], {"queries": 0, "no_gold": 0, "toolret_rows": 0, "bfcl_rows": 0}
for l in open(src):
    r = json.loads(l)
    c["queries"] += 1
    gold = set(r["gold"])
    cands = r["candidates"]
    pos = [x for x in cands if x["tool_id"] in gold]
    if not pos:
        c["no_gold"] += 1
        continue
    neg = [x for x in cands if x["tool_id"] not in gold][:3]
    for x in pos[:2] + neg:
        y = "yes" if x["tool_id"] in gold else "no"
        out.append({"id": f"TOOLV3:{r['qid']}:{x['tool_id']}", "family": "LOOMD-tools_v3", "split": "train", "group": f"toolv3|{r['qid']}",
                    "state": {"query": r["query"], "task": TASK},
                    "questions": {"doc": {"type": "choice", "instructions": {"candidate": x["doc"], "task": DOC_TASK}, "criteria": dict(DI_CRIT)}},
                    "expected": {"doc": y}, "meta": {"source": r["source"], "qid": r["qid"], "tool_id": x["tool_id"], "shape": "toolret"}})
        c["toolret_rows"] += 1
    if r["source"].lower() == "toolace":
        fs = pos[:2] + neg
        rng.shuffle(fs)
        schemas = []
        for x in fs:
            try:
                sch = json.loads(x["doc"])
            except ValueError:
                sch = {"name": x["tool_id"], "description": x["doc"]}
            schemas.append(sch)
        for x, sch in zip(fs, schemas):
            name = str(sch.get("name") or x["tool_id"])
            out.append({"id": f"TOOLV3B:{r['qid']}:{x['tool_id']}", "family": "LOOMD-tools_v3", "split": "train", "group": f"toolv3|{r['qid']}",
                        "state": {"conversation": [[{"role": "user", "content": r["query"]}]], "functions": schemas},
                        "questions": {"tool": {"type": "choice", "instructions": BFCL_INSTR.format(name=name), "criteria": dict(BFCL_CRIT)}},
                        "expected": {"tool": "yes" if x["tool_id"] in gold else "no"}, "meta": {"source": r["source"], "qid": r["qid"], "tool_id": x["tool_id"], "shape": "bfcl"}})
            c["bfcl_rows"] += 1
body = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
open(outp, "w").write(body)
c["rows"] = len(out); c["sha256"] = hashlib.sha256(body.encode()).hexdigest()
print(json.dumps(c))
