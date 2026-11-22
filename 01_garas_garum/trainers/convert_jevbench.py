#!/usr/bin/env python3
"""Convert the real JEV benchmarks into the record format the rest of the pipeline consumes.

This is the pivot that makes the work measurable. Until now the corpus was self-built, so any
number was self-referential. `Praveenrajus/jev-bench` supplies:
  * train / validation / test splits (8000 / 500 / 1000 per config, 22 configs)
  * REAL per-rater label distributions for 4 configs, plus `meta.n_raters` and
    `meta.rater_agreement` -- the rater count a Miller-Madow entropy correction needs
  * published TypeSafe Jev 1.13.0 baselines to compare against
`LocalLLaMA/typed-decisions` supplies the other published eval target (Jev 1.13: acc 0.727,
KL 1.442, Brier 0.148, ECE 0.144; teacher ceiling 0.735).

Observed schemas (verified by fetching, not assumed):

  jev-bench columns: id, source, primitive, split, state, question, label, soft_label, meta
    question   JSON string: {type, instructions, criteria}
    soft_label JSON string: {option_key: probability}   <- the distribution we train on
    meta       JSON string: {license, domain, task_family, hf_id, n_raters, rater_agreement,...}

  typed-decisions columns: id, workflow, split, state, questions, gold, factors,
                           label_agreement, n_questions
    questions  JSON string: {qid: {type, instructions, criteria}}
               NOTE: for type "score", `criteria` is a LIST of level descriptions;
                     for type "choice" it is a DICT keyed by option.
    gold       JSON string: {qid: {type, label, confidence, probabilities, score?, noul?}}

Emits one JSON object per line in the build_jev_data.py schema, so cache_features.py and
train_head.py need no changes.

  python3 convert_jevbench.py --out jev_real --dry-run     # show what the shapes look like
  python3 convert_jevbench.py --out jev_real

NOUL DEFINITIONS (backlog 53). By default a noul record keeps only {type, instructions}: the true/false definitions
typed-decisions sends in `criteria` (400 of its 600 test nouls) are dropped. `--noul-defs` keeps them as
criteria {"yes": .., "no": ..} (ids unchanged: they hash the option KEYS). `--graft-onto DIR` then writes DIR's records
byte-for-byte except that a noul gains its definitions, matched by record id, so a run with them is paired with the
existing jev_real item by item:

  python3 convert_jevbench.py --out jev_real_ndraw --splits test --noul-defs
  python3 convert_jevbench.py --out jev_real_nd --splits test --noul-defs --graft-onto jev_real --from-converted jev_real_ndraw
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

JEVBENCH = "Praveenrajus/jev-bench"
TYPED = "LocalLLaMA/typed-decisions"


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def normalise(qtype, criteria, soft):
    """(option_keys, criteria_dict, level_list) in a single canonical shape.

    Handles the one real inconsistency between the two sources: `criteria` is a dict for
    `choice` but a LIST for `score`. Guessing here would silently mis-align every label, so
    each branch asserts instead.
    """
    if qtype == "choice":
        if isinstance(criteria, dict):
            keys = list(criteria)
            crit = {k: (str(v) if v is not None else "") for k, v in criteria.items()}
        elif isinstance(criteria, list):
            keys = [str(i) for i in range(len(criteria))]
            crit = {str(i): str(c) for i, c in enumerate(criteria)}
        else:
            raise ValueError(f"choice criteria is {type(criteria).__name__}")
        return keys, crit, None

    if qtype == "score":
        levels = criteria if isinstance(criteria, list) else [str(v) for v in (criteria or {}).values()]
        if not levels:
            # Some configs ship soft_label as a LIST of vote shares with no `criteria` at all
            # (measuring_hate_speech does). Fall back to positional level names.
            n = len(soft) if isinstance(soft, (list, tuple, dict)) else 0
            levels = [str(i) for i in range(n)]
        keys = [str(i) for i in range(len(levels))]
        return keys, {k: str(l) for k, l in zip(keys, levels)}, [str(l) for l in levels]

    if qtype == "noul":
        # soft_label keys are "true"/"false" in jev-bench and typed-decisions; the schema here
        # is "yes"/"no". Map explicitly rather than trusting order.
        keys = ["yes", "no"]
        return keys, {"yes": "Yes", "no": "No"}, None

    raise ValueError(f"unknown primitive {qtype!r}")


def align_labels(qtype, keys, soft):
    """A distribution over `keys`, in `keys` order, from a source-specific dict OR list.

    Two shapes in the wild: a dict keyed by option, and a positional LIST of vote shares
    (measuring_hate_speech). Assuming one and asserting the other is how labels get silently
    rotated, so both are handled explicitly.
    """
    if isinstance(soft, (int, float)):
        # civil_comments ships a noul as a bare scalar: the share of annotators who said yes.
        if qtype == "noul":
            return [float(soft), 1.0 - float(soft)]
        return None
    if isinstance(soft, (list, tuple)):
        if len(soft) != len(keys):
            return None
        return [float(v) for v in soft]
    if qtype == "noul":
        t = soft.get("true", soft.get("yes"))
        f = soft.get("false", soft.get("no"))
        # A one-hot rebuilt from a hard label (onehot_from_label) names only ONE side; the other is 1 minus it.
        # Requiring both dropped every hard-label noul config (boolq, fever, paws, ...), the bias the docstring
        # of onehot_from_label warns about.
        if t is None and f is not None:
            t = 1.0 - float(f)
        if f is None and t is not None:
            f = 1.0 - float(t)
        if t is None or f is None:
            return None
        return [float(t), float(f)]
    out, missing = [], 0
    for k in keys:
        if k in soft:
            out.append(float(soft[k]))
            continue
        # score levels may be keyed by the level text rather than an index
        hit = [v for kk, v in soft.items() if str(kk) == str(k)]
        if not hit:
            out.append(0.0); missing += 1
            continue
        out.append(float(hit[0]))
    # Missing keys are zeros ONLY when every key that IS given matched an option and they already carry all
    # the mass (a one-hot from a hard label); otherwise the label does not fit these options and is refused.
    if missing and (len(soft) + missing != len(keys) or abs(sum(out) - 1.0) > 1e-6):
        return None
    return out


def noul_criteria(criteria):
    """A noul's written definitions as {"yes": .., "no": ..} from criteria keyed true/false or yes/no (any case), or None.
    A side that is not given is ""."""
    if not isinstance(criteria, dict):
        return None
    low = {str(k).strip().lower(): v for k, v in criteria.items()}
    d = {}
    for ours, names in (("yes", ("true", "yes")), ("no", ("false", "no"))):
        d[ours] = next((str(low[n]).strip() for n in names if low.get(n) is not None and str(low[n]).strip()), "")
    return d if (d["yes"] or d["no"]) else None


def graft_noul_defs(base_lines, converted_lines):
    """`base_lines` (an existing split file's lines) with every noul question that a `--noul-defs` conversion of the same
    record (same id) gives definitions gaining them as `criteria`; every other line returned unchanged, byte for byte.
    Returns (lines, n_questions_grafted, n_base_records_whose_id_the_conversion_lacks) -- the last should be 0; more
    means the source dataset changed since `base_lines` was converted."""
    defs, seen = {}, set()
    for line in converted_lines:
        if not line.strip():
            continue
        r = json.loads(line)
        seen.add(r["id"])
        for qid, q in r["questions"].items():
            if q.get("type") == "noul" and isinstance(q.get("criteria"), dict):
                defs[(r["id"], qid)] = q["criteria"]
    out, n_graft, n_miss = [], 0, 0
    for line in base_lines:
        if not line.strip():
            out.append(line)
            continue
        r = json.loads(line)
        hit = False
        for qid, q in r["questions"].items():
            d = defs.get((r["id"], qid))
            if q.get("type") == "noul" and d is not None and "criteria" not in q:
                q["criteria"] = d
                hit = True
                n_graft += 1
        n_miss += r["id"] not in seen
        out.append(json.dumps(r, ensure_ascii=False) if hit else line)
    return out, n_graft, n_miss


def emit(out_lines, source, state, qid, question, dist, label_kind, meta, split, noul_defs=False):
    keys, crit, levels = normalise(question["type"], question.get("criteria"), dist)
    aligned = align_labels(question["type"], keys, dist)
    if aligned is None:
        return False, "unmapped_label_keys"
    total = sum(aligned)
    if total <= 0:
        return False, "zero_mass"
    aligned = [x / total for x in aligned]
    q = {"type": question["type"], "instructions": str(question.get("instructions", "")).strip()}
    if question["type"] == "choice":
        q["criteria"] = crit
    elif question["type"] == "score":
        q["levels"] = levels
    elif question["type"] == "noul" and noul_defs:
        d = noul_criteria(question.get("criteria"))
        if d:
            q["criteria"] = d
    if not q["instructions"]:
        q["instructions"] = "Which option applies?"
    rid = f"{source}-{sha(state + qid + json.dumps(keys, sort_keys=True))}"
    rec = {
        "id": rid, "base_id": rid, "source": source, "license": meta.get("license", "verify"),
        "state": state, "questions": {qid: q}, "labels": {qid: dict(zip(keys, aligned))},
        "label_kind": {qid: label_kind}, "meta": meta, "split": split,
    }
    out_lines.append(json.dumps(rec, ensure_ascii=False))
    return True, None


def onehot_from_label(qtype, label):
    """Hard-label configs ship `soft_label = None`; rebuild a one-hot from `label`.

    Only 4 of jev-bench's 22 configs carry real per-rater distributions. The other 18 are
    hard labels, and silently dropping them would bias the benchmark toward the soft configs.
    """
    if label is None:
        return None
    v = str(label).strip()
    if qtype == "noul":
        low = v.lower()
        if low in ("true", "yes", "1"):
            return {"true": 1.0}
        if low in ("false", "no", "0"):
            return {"false": 1.0}
        return None
    return {v: 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jev_real")
    ap.add_argument("--dry-run", action="store_true", help="print shapes, write nothing")
    ap.add_argument("--sources", default="jevbench,typed")
    ap.add_argument("--splits", default="train,val,test", help="output splits to convert (default all three)")
    ap.add_argument("--noul-defs", action="store_true", help="keep a noul's true/false definitions as criteria {yes, no}")
    ap.add_argument("--graft-onto", default="", help="with --from-converted: write this jev_real dir's records, adding the "
                    "noul definitions of --from-converted's records with the same id (see the module docstring)")
    ap.add_argument("--from-converted", default="", help="a --noul-defs conversion to take the definitions from")
    a = ap.parse_args()
    want = {s.strip() for s in a.sources.split(",")}
    out_splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    if a.graft_onto:
        if not a.from_converted:
            raise SystemExit("--graft-onto needs --from-converted")
        for split in out_splits:
            base = Path(a.graft_onto, split, "real.jsonl").read_text(encoding="utf-8").split("\n")
            conv = Path(a.from_converted, split, "real.jsonl").read_text(encoding="utf-8").split("\n")
            lines, n_graft, n_miss = graft_noul_defs(base, conv)
            d = Path(a.out) / split
            d.mkdir(parents=True, exist_ok=True)
            (d / "real.jsonl").write_text("\n".join(lines), encoding="utf-8")
            print(f"wrote {d / 'real.jsonl'}: {sum(1 for x in lines if x.strip())} records, noul definitions grafted onto "
                  f"{n_graft} questions; {n_miss} records not in --from-converted (should be 0)")
        return

    stats, failures, shapes = Counter(), Counter(), Counter()
    lines = {s: [] for s in ("train", "val", "test") if s in out_splits}
    split_map = {k: v for k, v in {"train": "train", "validation": "val", "test": "test"}.items() if v in out_splits}

    if "jevbench" in want:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names(JEVBENCH)
        print(f"{JEVBENCH}: {len(configs)} configs")
        for cfg in configs:
            ds = load_dataset(JEVBENCH, cfg)
            for src_split, dst_split in split_map.items():
                if src_split not in ds:
                    continue
                for row in ds[src_split]:
                    q = json.loads(row["question"])
                    raw = row.get("soft_label")
                    soft = json.loads(raw) if raw else onehot_from_label(
                        row["primitive"], row.get("label"))
                    if not soft:
                        failures["no_label_at_all"] += 1
                        continue
                    meta = json.loads(row["meta"])
                    meta = {**meta, "config": cfg, "primitive": row["primitive"],
                            "hf_id": meta.get("hf_id"), "original_split": src_split}
                    shapes[(row["primitive"], type(q.get("criteria")).__name__)] += 1
                    kind = "soft" if int(meta.get("n_raters") or 1) > 1 else "hard"
                    ok, why = emit(lines[dst_split], f"jevbench/{cfg}", row["state"],
                                   row["id"], q, soft, kind, meta, dst_split, a.noul_defs)
                    stats[f"jevbench/{cfg}/{dst_split}"] += 1 if ok else 0
                    if not ok:
                        failures[why] += 1

    if "typed" in want:
        ds = load_dataset(TYPED, "all")
        for src_split, dst_split in split_map.items():
            if src_split not in ds:
                continue
            for row in ds[src_split]:
                qs = json.loads(row["questions"])
                gold = json.loads(row["gold"])
                for qid, q in qs.items():
                    g = gold.get(qid) or {}
                    probs = g.get("probabilities")
                    if not probs:
                        failures["no_probabilities"] += 1
                        continue
                    shapes[(q["type"], type(q.get("criteria")).__name__)] += 1
                    # teacher-mean gold from 3 samples: genuinely soft, but not human.
                    meta = {"teacher_samples": 3, "workflow": row["workflow"],
                            "n_questions": row["n_questions"], "license": "apache-2.0",
                            "gold_confidence": g.get("confidence")}
                    ok, why = emit(lines[dst_split], "typed-decisions", row["state"], qid, q,
                                   probs, "soft", meta, dst_split, a.noul_defs)
                    stats[f"typed-decisions/{dst_split}"] += 1 if ok else 0
                    if not ok:
                        failures[why] += 1

    print("\ncriteria shapes (primitive, criteria type):")
    for k, v in shapes.most_common():
        print(f"  {k}: {v}")
    print("\nwritten by split/source:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    if failures:
        print(f"\nFAILURES: {dict(failures)}")

    if a.dry_run:
        print("\n(dry run: nothing written)")
        for s, ls in lines.items():
            if ls:
                print(f"\n--- sample {s} ---")
                print(json.dumps(json.loads(ls[0]), indent=2)[:1200])
        return

    for split, ls in lines.items():
        d = Path(a.out) / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "real.jsonl").write_text("\n".join(ls) + "\n", encoding="utf-8")
        print(f"wrote {d / 'real.jsonl'}: {len(ls)} records")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "manifest.json").write_text(json.dumps(
        {"variants": 0, "sources": {k: v for k, v in stats.items()}}, indent=2))
    print(f"manifest written to {Path(a.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
