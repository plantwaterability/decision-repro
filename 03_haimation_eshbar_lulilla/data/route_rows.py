#!/usr/bin/env python3
"""Program-route prompts for DI-format held-out rows (call_eval.py's PROGRAM_PROMPT, the whole-def template), one per choice question
whose options are all plain numbers or all truth words (the only rows the route can answer). Problem text: state.question when the state
has one (as rowprompts), else the state's text parts + the question's own instructions (generic BBH instruction dropped).
  python3 route_rows.py --pools DIR --tag T --out P.jsonl [--sample N --seed S] FILE[:TRACK,...]..."""
import argparse
import os
import gzip
import json
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="per track")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.pools)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(here, "..", "stage2"), os.path.join(here, "..", "engine")]   # the one shared renderer
    from stage2_text import problem_text
    from call_eval import PROGRAM_PROMPT
    from loom_api import parse_option_number
    truth = {"yes", "no", "true", "false"}
    by = {}
    for spec in a.files:
        path, _, tr = spec.partition(":")
        want = set(tr.split(",")) if tr else None
        op = gzip.open if path.endswith(".gz") else open
        for l in op(path, "rt"):
            r = json.loads(l)
            t = (r.get("metadata") or {}).get("track") or r["family"]
            if want and t not in want:
                continue
            for qk, q in r["questions"].items():
                if q.get("type", "choice") != "choice":
                    continue
                vals = [str(v).strip().rstrip(".").lower() for v in q["criteria"].values()]
                if not (all(v in truth for v in vals) or all(parse_option_number(str(v)) is not None for v in q["criteria"].values())):
                    continue
                text = problem_text(r["state"], q)
                by.setdefault(t, []).append({"pid": f"{a.tag}|{r['id']}|{qk}", "group": r["id"], "family": r["family"], "track": t, "id": r["id"],
                                             "question_text": text, "options": q["criteria"], "expected": r["expected"][qk],
                                             "template": "program", "question": PROGRAM_PROMPT.format(question=text)})
    rng = random.Random(a.seed)
    out = []
    for t in sorted(by):
        xs = by[t]
        if a.sample and len(xs) > a.sample:
            xs = sorted(rng.sample(xs, a.sample), key=lambda x: x["pid"])
        out += xs
    with open(a.out, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print(json.dumps({t: min(len(v), a.sample or len(v)) for t, v in sorted(by.items())}))


if __name__ == "__main__":
    main()
