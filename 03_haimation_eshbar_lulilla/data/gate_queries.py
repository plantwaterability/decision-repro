#!/usr/bin/env python3
"""Decontamination gate for free-text queries (tool-memory data): build_manifest.Gate (DI 0.3 suite + held-out validation sets:
exact normalized states / instructions, content 13-grams, short-item containment) plus an exact check against the DI tool rows'
own query texts (ToolRet state.query, BFCL conversation turns, API-Bank dialogue turns), which the state-level exact keys miss
because those states also carry a task field. A query passes if it is clean under both.

  python3 gate_queries.py --suite DIR --heldout DIR --code DIR IN.jsonl... (rows {qid, query}) -> IN.gated.jsonl + counts
"""
import argparse
import glob
import gzip
import json
import sys

from build_manifest import Gate, jsonl, norm_text


def turns(st):
    if isinstance(st, str):
        try:
            st = json.loads(st)
        except ValueError:
            return []
    out = []
    if isinstance(st, dict):
        if isinstance(st.get("query"), str):
            out.append(st["query"])
        for conv in st.get("conversation") or []:
            for t in (conv if isinstance(conv, list) else [conv]):
                if isinstance(t, dict) and isinstance(t.get("content"), str):
                    out.append(t["content"])
        for t in st.get("dialogue") or []:
            if isinstance(t, dict) and isinstance(t.get("text"), str):
                out.append(t["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    gate, qx = Gate(), set()
    for f in ("selected-rows", "added-rows", "gsm8k-rows"):
        for r in jsonl(f"{a.suite}/{f}.jsonl.gz"):
            gate.add(r)
            for t in turns(r.get("state")):
                if len(norm_text(t)) >= 8:
                    qx.add(norm_text(t))
    for p in sorted(glob.glob(f"{a.heldout}/heldout_*.jsonl*") + glob.glob(f"{a.heldout}/kfree_rows.jsonl")):
        for r in jsonl(p):
            gate.add(r)
    print(json.dumps({"gate_exact": len(gate.exact), "gate_13grams": len(gate.grams), "tool_turns_exact": len(qx)}), flush=True)
    for p in a.files:
        c = {"in": 0, "pass": 0, "drop_tool_turn_exact": 0, "drop_gate": 0}
        with open(p.replace(".jsonl", ".gated.jsonl"), "w") as o:
            for r in jsonl(p):
                c["in"] += 1
                q = r["query"]
                if norm_text(q) in qx:
                    c["drop_tool_turn_exact"] += 1
                    continue
                if not gate.clean(q, exact_keys=(q,)):
                    c["drop_gate"] += 1
                    continue
                c["pass"] += 1
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(p, json.dumps(c), flush=True)


if __name__ == "__main__":
    main()
