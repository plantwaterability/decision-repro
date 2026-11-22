#!/usr/bin/env python3
"""Held-out DI-format rows -> an EVAL-ONLY sdv1-stage2-v1 manifest (renderer R3, split "eval"), for feature extraction and head
comparisons on families never used for selection (BBH originals, CLadder variants, CRUX synth, SVAMP, K-free, pool held-outs).
No gate: these rows are never trained on. Source = metadata.track (else family). Chess rows use their engine-best move.

  python3 render_eval.py --code DIR --model DIR --out FILE [--seed S] HELDOUT_FILE...
"""
import argparse
import hashlib
import json
import random
import sys

from build_manifest import R3, jsonl, make_render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import claude_lmscore as clm
    from transformers import AutoTokenizer
    render = make_render(AutoTokenizer.from_pretrained(a.model), clm, random.Random(a.seed), split="eval", tag="esbar_eval")
    out, counts = [], {}
    for p in a.files:
        for r in jsonl(p):
            src = (r.get("metadata") or {}).get("track") or r.get("family") or "?"
            for qk, q in r["questions"].items():
                if q.get("type", "choice") != "choice":
                    counts["skipped_nonchoice"] = counts.get("skipped_nonchoice", 0) + 1
                    continue
                uid = f"eval|{r['id']}|{qk}"
                row = render(uid, r.get("family"), r["state"], q.get("instructions") or "", q["criteria"], r["expected"][qk], src)
                if row is None:
                    counts[f"{src}_over_max_tokens"] = counts.get(f"{src}_over_max_tokens", 0) + 1
                    continue
                out.append(row)
                counts[src] = counts.get(src, 0) + 1
    body = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in out)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    json.dump({"manifest_header": True, "schema_version": "sdv1-stage2-v1", "renderer_sha256": R3, "rows": len(out), "rows_sha256": sha,
               "total_decision_weight": float(len(out))}, open(a.out + ".header.json", "w"), indent=2)
    print(json.dumps({"counts": counts, "rows": len(out), "rows_sha256": sha}), flush=True)


if __name__ == "__main__":
    main()
