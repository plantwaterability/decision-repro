"""synth_workflows.contract -- the unlabelled-record contract between the generator and the labeller.

build_jev_typed_synth.py writes UNLABELLED records; trainers/claude_verbal_label.py fills them. The full
contract is in research/typed_synth_spec.md section 5; this module is its executable form. Stdlib only, so
the labeller can import it anywhere:

    from synth_workflows.contract import validate_unlabelled, attach_labels, validate_labelled

A labelled record loads through trainers/cache_features.iter_units unchanged.
"""
from __future__ import annotations

from .base import N_QUESTIONS, option_keys, questions_errors

UNLABELLED_TOP = {"id", "base_id", "source", "license", "state", "questions", "meta", "split", "unlabelled"}
META_REQUIRED = {"generator", "generator_version", "workflow", "schema", "factors", "surface", "seed",
                 "case_index", "split", "dev_workflow", "question_order"}
LABEL_KINDS = ("soft", "hard", "noisy_hard")


def label_keys(q: dict, score_keys: str = "text") -> list:
    """The keys a labels[qid] dict must use: choice -> criteria keys; noul -> 'yes','no'; score -> the level
    TEXTS (score_keys='text', the default: passes build_jev_data.validate too) or '0'..'K-1' ('index', what
    the typed-decisions converter writes). cache_features.question_options accepts either for score."""
    keys = option_keys(q)
    if q["type"] == "score" and score_keys == "index":
        return [str(i) for i in range(len(keys))]
    return keys


def validate_unlabelled(rec: dict) -> list:
    """Every way `rec` breaks the unlabelled contract (empty list = fine)."""
    e = []
    missing = UNLABELLED_TOP - set(rec)
    if missing:
        e.append(f"missing top-level fields {sorted(missing)}")
    if "labels" in rec or "label_kind" in rec:
        e.append("an unlabelled record must not carry labels / label_kind")
    if rec.get("unlabelled") is not True:
        e.append("unlabelled must be true")
    if rec.get("base_id") != rec.get("id"):
        e.append("base_id must equal id")
    if rec.get("split") not in ("train", "val"):
        e.append(f"split {rec.get('split')!r} not train/val")
    if not isinstance(rec.get("state"), str) or not rec["state"].strip():
        e.append("empty state")
    e.extend(questions_errors(rec.get("questions") or {}))
    meta = rec.get("meta") or {}
    miss = META_REQUIRED - set(meta)
    if miss:
        e.append(f"meta missing {sorted(miss)}")
    if meta.get("split") != rec.get("split"):
        e.append("meta.split != split")
    if list(meta.get("question_order") or []) != list(rec.get("questions") or {}):
        e.append("meta.question_order must list the qids in stored order")
    return e


def attach_labels(rec: dict, labels: dict, label_kind: str = "soft", labeller_meta: dict | None = None,
                  tol: float = 1e-6) -> dict:
    """A labelled COPY of an unlabelled record: labels[qid] = {option_key: p} for all five questions.

    Probabilities are renormalised (they must already sum to ~1 within 1e-3); `unlabelled` is removed;
    `label_kind` is set per question; `labeller_meta` lands in meta['labeller']. Score labels may be keyed by
    level text or by '0'..'K-1'; they are re-keyed to level text so build_jev_data.validate passes as well.
    """
    out = {k: v for k, v in rec.items() if k != "unlabelled"}
    out["meta"] = dict(rec["meta"])
    if labeller_meta is not None:
        out["meta"]["labeller"] = labeller_meta
    labs, kinds = {}, {}
    for qid, q in rec["questions"].items():
        if qid not in labels:
            raise KeyError(f"{rec['id']}: no label for {qid}")
        lab = {str(k): float(v) for k, v in labels[qid].items()}
        keys = option_keys(q)
        vals = []
        for i, k in enumerate(keys):
            if k in lab:
                vals.append(lab[k])
            elif str(i) in lab and q["type"] == "score":
                vals.append(lab[str(i)])
            else:
                raise KeyError(f"{rec['id']}/{qid}: no probability for option {k!r}")
        extra = set(lab) - set(keys) - ({str(i) for i in range(len(keys))} if q["type"] == "score" else set())
        if extra:
            raise KeyError(f"{rec['id']}/{qid}: unknown option keys {sorted(extra)}")
        if any(v < -tol for v in vals):
            raise ValueError(f"{rec['id']}/{qid}: negative probability")
        s = sum(vals)
        if abs(s - 1.0) > 1e-3:
            raise ValueError(f"{rec['id']}/{qid}: probabilities sum to {s:.4f}")
        labs[qid] = {k: max(0.0, v) / s for k, v in zip(keys, vals)}
        kinds[qid] = label_kind
    out["labels"], out["label_kind"] = labs, kinds
    return out


def validate_labelled(rec: dict) -> list:
    """The labelled form: everything cache_features.iter_units needs, nothing left of `unlabelled`."""
    e = []
    if rec.get("unlabelled"):
        e.append("labelled record still says unlabelled")
    qs = rec.get("questions") or {}
    if len(qs) != N_QUESTIONS:
        e.append(f"{len(qs)} questions")
    for qid, q in qs.items():
        lab = (rec.get("labels") or {}).get(qid)
        if lab is None:
            e.append(f"{qid}: no labels")
            continue
        if set(lab) != set(option_keys(q)):
            e.append(f"{qid}: label keys {sorted(lab)[:4]} != option keys")
        if abs(sum(lab.values()) - 1.0) > 1e-6:
            e.append(f"{qid}: labels sum to {sum(lab.values())}")
        if (rec.get("label_kind") or {}).get(qid) not in LABEL_KINDS:
            e.append(f"{qid}: label_kind must be one of {LABEL_KINDS}")
    return e
