#!/usr/bin/env python3
"""Re-render a frozen Gemma exposure manifest for Qwen without changing targets.

Decision text comes from the original Stage 0 decisions. Every source row is
retained in order, including its option order and complete parent-preservation
entry. Only the prompt identity/token count/renderer change. The parent is a
fixed teacher target from the ORIGINAL Gemma prompt, not a newly scored Qwen
teacher; this distinction is recorded in the header and every row's audit.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
PROMPT = """{state}

Question: {instructions}

Options:
{options}

Answer with the letter of the single best option and nothing else."""
CHANGED = {"prompt", "prompt_sha256", "n_tokens", "renderer_sha256", "qwen_render_audit"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def render_decision(tok, decision, order, max_len=8192, margin=16):
    """Single question, same R2 body; truncate only a prefix of the state.

    The original scorer's overhead-plus-margin policy is retained. Retokenize
    the complete prompt and shorten further if a boundary changes its length.
    An unfit question/options section refuses the whole build, never a row.
    """
    options = decision["ordered_options"]
    keys = [x["key"] for x in options]
    if len(order) < 2 or len(order) > len(LABELS) or len(set(keys)) != len(keys) or sorted(order) != sorted(keys):
        raise ValueError("option order must be a permutation of 2..52 distinct decision keys")
    state = decision["state"]
    if not isinstance(state, str) or not isinstance(decision["question"], str):
        raise ValueError("Stage 0 state and question must be strings")
    texts = {x["key"]: x["text"] for x in options}
    opts = "\n".join(f"{LABELS[i]}) {texts[k]}" for i, k in enumerate(order))

    def render(value):
        body = PROMPT.format(state=value, instructions=decision["question"], options=opts)
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    overhead = count(render(""))
    budget = max_len - overhead - margin
    if budget <= 0:
        raise ValueError(f"question/options exceed cap: overhead={overhead}, cap={max_len}, margin={margin}")
    enc = tok(state, add_special_tokens=False, return_offsets_mapping=True)
    n_state = len(enc["input_ids"])
    kept = min(n_state, budget)
    # Offsets, rather than decode(), preserve the original Unicode/whitespace.
    def prefix(n):
        return state if n == n_state else state[:int(enc["offset_mapping"][n - 1][1])] if n else ""
    shortened = prefix(kept)
    prompt = render(shortened)
    nt = count(prompt)
    while nt > max_len and kept:
        kept -= max(1, nt - max_len)
        kept = max(0, kept)
        shortened = prefix(kept)
        prompt = render(shortened)
        nt = count(prompt)
    if nt > max_len:
        raise ValueError("question/options do not fit even with empty state")
    return prompt, {"truncated": shortened != state, "state_tokens": n_state, "kept_state_tokens": kept,
                    "state_characters": len(state), "kept_state_characters": len(shortened), "n_tokens": nt}


def transform_row(tok, row, decision, renderer_sha256, max_len=8192, margin=16):
    if row["decision_uid"] != decision["decision_uid"]:
        raise ValueError("decision identity mismatch")
    if row["gold"] != {k: float(v) for k, v in decision["gold_by_option_key"].items()}:
        raise ValueError(f"gold mismatch for {row['decision_uid']}")
    text, audit = render_decision(tok, decision, row["option_order"], max_len, margin)
    result = dict(row)
    result.update(prompt=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  n_tokens=audit["n_tokens"], renderer_sha256=renderer_sha256)
    result["qwen_render_audit"] = {**audit, "source_prompt_sha256": row["prompt_sha256"],
                                   "source_renderer_sha256": row["renderer_sha256"],
                                   "parent_targets": "unchanged original Gemma-prompt teacher targets"}
    if {k: v for k, v in result.items() if k not in CHANGED} != {k: v for k, v in row.items() if k not in CHANGED}:
        raise AssertionError("non-rendering row fields changed")
    return result


def build(args):
    source_header = json.load(open(args.source_header, encoding="utf-8"))
    if source_header["rows_sha256"] != args.expected_rows_sha256:
        raise ValueError("source header does not match frozen rows sha256")
    if digest_file(args.decisions, "md5") != args.decisions_md5:
        raise ValueError("Stage 0 decisions md5 mismatch")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=REVISION, trust_remote_code=False)
    files = {name: digest_file(Path(args.tokenizer, name)) for name in
             ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")
             if Path(args.tokenizer, name).exists()}
    if not files or "tokenizer.json" not in files:
        raise ValueError("tokenizer must be a pinned local directory with tokenizer.json")
    letter_ids = [tok(x, add_special_tokens=False)["input_ids"] for x in LABELS]
    if any(len(ids) != 1 for ids in letter_ids) or len({ids[0] for ids in letter_ids}) != 52:
        raise ValueError("52 decision labels must be distinct single tokens")
    spec = {"builder": "qwen_fullft_render v1", "builder_sha256": digest_file(__file__), "model": MODEL_ID,
            "revision": REVISION, "tokenizer_files": files, "max_len": args.max_len, "margin": 16,
            "enable_thinking": False, "add_generation_prompt": True, "one_question": True,
            "body": PROMPT, "state_truncation": "original-text prefix at Qwen token offset; full-prompt cap rechecked",
            "parent_targets": "unchanged original Gemma-prompt teacher targets", "label_ids": [x[0] for x in letter_ids]}
    renderer_sha256 = hashlib.sha256(canonical(spec).encode()).hexdigest()
    wanted = set()
    source_sha = hashlib.sha256()
    n_source = 0
    with open_text(args.source) as f:
        for line in f:
            source_sha.update(line.encode("utf-8"))
            r = json.loads(line)
            if r.get("manifest_header"):
                raise ValueError("expected source rows with a separate header")
            wanted.add(r["decision_uid"])
            n_source += 1
    if source_sha.hexdigest() != args.expected_rows_sha256 or n_source != args.expected_rows:
        raise ValueError("source rows byte hash or row count mismatch")
    decisions = {}
    with open_text(args.decisions) as f:
        for line in f:
            d = json.loads(line)
            uid = d["decision_uid"]
            if uid in wanted:
                if uid in decisions:
                    raise ValueError(f"duplicate Stage 0 decision {uid}")
                decisions[uid] = d
    if set(decisions) != wanted:
        raise ValueError(f"missing {len(wanted - set(decisions))} Stage 0 decisions")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites: source and any previous renderer remain recoverable.
    if out.exists() or Path(str(out) + ".header.json").exists():
        raise ValueError("output already exists")
    partial = Path(str(out) + ".partial")
    stats = Counter()
    row_hash = hashlib.sha256()
    targets_before, targets_after = hashlib.sha256(), hashlib.sha256()
    with open_text(args.source) as src, open(partial, "xb") as dst:
        for line in src:
            r = json.loads(line)
            if r["renderer_sha256"] != source_header["renderer_sha256"]:
                raise ValueError("source row renderer differs from source header")
            if hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]:
                raise ValueError("source prompt hash mismatch")
            q = transform_row(tok, r, decisions[r["decision_uid"]], renderer_sha256, args.max_len)
            before = canonical({k: v for k, v in r.items() if k not in CHANGED}).encode() + b"\n"
            after = canonical({k: v for k, v in q.items() if k not in CHANGED}).encode() + b"\n"
            targets_before.update(before)
            targets_after.update(after)
            data = canonical(q).encode() + b"\n"
            row_hash.update(data)
            dst.write(data)
            stats["rows"] += 1
            stats["tokens"] += q["n_tokens"]
            stats["truncated_rows"] += int(q["qwen_render_audit"]["truncated"])
            stats["max_tokens"] = max(stats["max_tokens"], q["n_tokens"])
            stats["with_parent"] += int(r.get("parent_preservation") is not None)
            if stats["rows"] % 2000 == 0:
                print("RENDER_PROGRESS", stats["rows"], flush=True)
        dst.flush()
        os.fsync(dst.fileno())
    if targets_before.digest() != targets_after.digest() or stats["rows"] != args.expected_rows:
        raise AssertionError("row preservation audit failed")
    header = dict(source_header)
    header.update(renderer_sha256=renderer_sha256, renderer_spec=spec, rows_sha256=row_hash.hexdigest(),
                  tokenizer_revision=f"{MODEL_ID}@{REVISION}", render="Qwen R2, thinking disabled", materializer="qwen_fullft_render v1",
                  tokens=stats["tokens"], source_manifest={"rows_sha256": args.expected_rows_sha256,
                  "renderer_sha256": source_header["renderer_sha256"], "decisions_md5": args.decisions_md5},
                  rerender_audit={**stats, "non_render_fields_sha256": targets_before.hexdigest(), "dropped_rows": 0,
                  "order_gold_parent_and_metadata_unchanged": True})
    # Gemma-only convenience metadata must not purport to describe this renderer.
    header.pop("renderer_sha256_equivalent", None)
    partial.replace(out)
    with open(str(out) + ".header.json", "x", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    print("QWEN_RENDER_DONE", canonical({"out": str(out), "renderer_sha256": renderer_sha256,
          "rows_sha256": row_hash.hexdigest(), **header["rerender_audit"]}), flush=True)
    return header


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-header", required=True)
    ap.add_argument("--expected-rows-sha256", required=True)
    ap.add_argument("--expected-rows", type=int, required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--decisions-md5", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--out", required=True, help="uncompressed JSONL path")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
