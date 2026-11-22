#!/usr/bin/env python3
"""stock_decision_v1 Stage 3, TEACHER SIDE: score a decision manifest with each pinned teacher through that teacher's OWN
released inference path, and write one prediction row per (teacher, decision, transform).

Teachers are predictors only. Nothing here loads teacher weights into a student, writes training data, or reads gold: the
request builders read a fixed whitelist of visible fields (state, question, kind, options, yes / no definitions), so gold,
rationales and every other manifest field can never reach a teacher (test_sdv1_teacher.py plants a canary in every other field).

INPUT: a decision manifest (JSONL, optionally .gz), one decision per line (a line with "manifest_header" is a header):
    decision_uid, parent_uid, split ("discovery" | "train" | "training"; anything naming a held-out pool is REFUSED),
    source_family, skill, domain (carried through),
    state (str | JSON object | list | null), question (str: the instructions; alias "instructions"),
    kind: choice | noul | score,
    ordered_options: [{"key": semantic_key, "text": description or null}, ...] in the presented order (aliases: "options";
        a list of strings means key = text; a {key: text} object keeps its order).
        choice: K >= 2 unique keys. score: the ordered levels (K >= 2), level order kept. noul: [] or keys from
        {true, false} (yes = true, no = false) whose texts are the yes / no definitions (alias "noul_definitions").
    Keys are shown to the System One teachers (d3, Perplexity) as "key: text" (text omitted when null or equal to the key),
    exactly like Decision Index requests, so keys should be meaningful labels (or equal to the option text).

TRANSFORMS (the fixed perturbation sample only; base for every decision):
    perm1   choice with K >= 2: the options in one seeded NON-identity order (seed = sha256("sdv1-perm-v1|" + decision_uid));
            probabilities stay keyed by semantic key, so they compare with base directly. noul / score: not_applicable
            (yes / no has each teacher's fixed native order; score levels keep their ordinal meaning).
    irrel1  one demonstrably irrelevant note from IRRELEVANT_NOTES (chosen by sha256("sdv1-irrel-v1|" + parent_uid), the same
            note for every decision of a parent) appended after a text state ("\n\n" + note), or added to a JSON-object state
            as "unrelated_note". A list state: not_applicable.
    The sample = the PERTURB_N parents with the smallest sha256("sdv1-perturb-sample-v1|" + parent_uid) (fixed, order-free).

OUTPUT (sdv1-teacher-prediction-v1, one JSON line each):
    teacher_id, teacher_pin_sha256, decision_uid, parent_uid, source_family, kind, K, transform, transform_spec,
    presented_keys, request_sha256 (canonical JSON of exactly what the teacher received), status (STATUSES),
    status_detail, probabilities {semantic key: p} (ONLY when status == ok; never fabricated), native_answer (the teacher's
    raw answer object), n_input_tokens, latency_s, extra (reasoning trace etc.), prediction_sha256 (sha256 of the canonical
    JSON of every field above except latency_s / prediction_sha256).

    python3 sdv1_teacher.py selftest
    python3 sdv1_teacher.py plan  --manifest M [--perturb-n 2000] [--shards N]
    python3 sdv1_teacher.py dry   --teacher T --pins teacher_pins.json --snapshot DIR [--adapter DIR] --decisions dryrun_v1.jsonl --out DIR
    python3 sdv1_teacher.py score --teacher T --pins teacher_pins.json --snapshot DIR [--adapter DIR] --manifest M --shard I/N --out FILE
    python3 sdv1_teacher.py merge --manifest M --inputs A.jsonl.gz ... --out ALL.jsonl.gz [--perturb-n 2000]
    python3 sdv1_teacher.py verify-snapshot --teacher T --pins teacher_pins.json --dir SNAPSHOT_DIR [--files adapter]
Standard library + numpy at import; torch / transformers / each teacher's own code only inside sdv1_backends.load_backend.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "sdv1_teacher v5 (2026-10-14)"
SCHEMA_PRED = "sdv1-teacher-prediction-v1"
KINDS = ("choice", "noul", "score")
ALLOWED_SPLITS = ("discovery", "train", "training", "dryrun")
# a split / path naming any held-out pool is refused before a single row is read into a request
BLOCK_WORDS = ("confirm", "seal", "dev", "cal", "test", "eval", "heldout", "held_out", "held-out", "valid", "protected",
               "final", "panel", "board")
STATUSES = ("ok", "unsupported_kind", "unsupported_menu", "unsupported_too_long", "unsupported_input", "timeout", "failure",
            "failure_oom", "invalid_output", "not_applicable")
TRANSFORMS = ("base", "perm1", "irrel1")
PERTURB_N = 2000
PERTURB_TAG = "sdv1-perturb-sample-v1"
PERM_TAG = "sdv1-perm-v1"
IRREL_TAG = "sdv1-irrel-v1"
PROB_TOL = 1e-4
OPAQUE_KEY = re.compile(r"^(opt|option|o|k|c|choice)?[_-]?\d+$", re.I)
# Demonstrably irrelevant notes (version sdv1-irrel-v1): fictional, schedule-free trivia about no real entity, no numbers that a
# question could need, no instructions. Gold is unchanged by construction; a teacher's change under this transform is a mining
# signal, never a label change.
IRRELEVANT_NOTES = (
    "Unrelated note: the reading room on the second floor has new blue curtains.",
    "Unrelated note: a volunteer repainted the garden bench beside the old oak tree.",
    "Unrelated note: the staff kitchen now keeps its spare mugs in the left cupboard.",
    "Unrelated note: someone left a striped umbrella in the lobby; it is at the front desk.",
    "Unrelated note: the hallway plants were moved closer to the window for more light.",
    "Unrelated note: the choir practised a new folk song during its last rehearsal.",
    "Unrelated note: a family of sparrows has built a nest above the bicycle shed.",
    "Unrelated note: the community noticeboard was cleaned and its old flyers recycled.",
    "Unrelated note: the cafe on the corner changed the colour of its awning to green.",
    "Unrelated note: the library cat was seen sleeping on the poetry shelf again.",
    "Unrelated note: a jigsaw puzzle of a lighthouse is half finished in the common room.",
    "Unrelated note: the fountain in the square was switched back on after cleaning.",
    "Unrelated note: the art club is collecting empty jars for a painting project.",
    "Unrelated note: the meeting room clock was replaced with a quieter model.",
    "Unrelated note: a neighbour's dog enjoys chasing leaves along the footpath.",
    "Unrelated note: the old piano in the hall was tuned by a visiting technician.",
    "Unrelated note: the walking group prefers the river path when the weather is mild.",
    "Unrelated note: fresh herbs from the rooftop planters were shared with the neighbours.",
    "Unrelated note: the craft fair organisers chose a hand-drawn poster design.",
    "Unrelated note: a kite got caught in the tall hedge near the playground.",
    "Unrelated note: the photocopier in the annex now prints slightly lighter pages.",
    "Unrelated note: the chess club's favourite board has a chipped white rook.",
    "Unrelated note: a mural of sunflowers is being painted on the side of the bakery.",
    "Unrelated note: the cloakroom hooks were rearranged so that coats hang lower.",
    "Unrelated note: the pond in the park has a few more lily pads than last spring.",
    "Unrelated note: the bookshop window display features paper cranes this season.",
    "Unrelated note: a stray tabby cat visits the courtyard most mornings.",
    "Unrelated note: the town band polished its brass instruments before the parade.",
    "Unrelated note: the cushions in the waiting area were swapped for softer ones.",
    "Unrelated note: a set of wind chimes now hangs by the greenhouse door.",
    "Unrelated note: the knitting circle finished a long scarf in many colours.",
    "Unrelated note: the gravel path to the shed was raked smooth on the weekend.",
)


NOUL_DISPLAY = re.compile(r"^\s*(yes|no|true|false)\s*(?::\s*(.+?))?\s*$", re.I | re.S)


class ManifestError(ValueError):
    pass


class InvalidOutput(ValueError):
    pass


class Result:
    """one backend answer: status in STATUSES; probabilities by semantic key only when ok."""

    __slots__ = ("status", "probabilities", "detail", "native_answer", "request", "n_input_tokens", "extra")

    def __init__(self, status, probabilities=None, detail="", native_answer=None, request=None, n_input_tokens=None,
                 extra=None):
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        self.status, self.probabilities, self.detail = status, probabilities, str(detail)[:500]
        self.native_answer, self.request, self.n_input_tokens, self.extra = native_answer, request, n_input_tokens, extra


# ------------------------------------------------------------------------------------------------------------------- basics
def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_canon(x):
    return sha256_text(canon(x))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def open_text(path, mode="rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(path, mode[0], encoding="utf-8")


def iter_jsonl(path):
    with open_text(path) as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield n, json.loads(line)
                except json.JSONDecodeError as e:
                    raise ManifestError(f"{path}:{n}: not JSON ({e})") from e


def blocked_word(text):
    t = str(text).lower()
    return next((w for w in BLOCK_WORDS if w in t), None)


def refuse_path(path):
    """the manifest FILE NAME must not name a held-out pool (a guard on top of the per-row split check)."""
    w = blocked_word(os.path.basename(path))
    if w:
        raise ManifestError(f"manifest file {path!r} names a held-out pool ({w!r}): teachers never see held-out items")


def refuse_key(key):
    """an S3 key (the job's input): no component after the bucket / namespace prefix may name a held-out pool."""
    rest = key.split("stock_decision_v1/", 1)[-1] if "stock_decision_v1/" in key else key
    w = blocked_word(rest)
    if w or "stock_decision_v1/" not in key:
        raise ManifestError(f"input key {key!r} is not a stock_decision_v1 teacher-eligible pool ({w!r})")


def check_split(split, allowed=ALLOWED_SPLITS):
    s = str(split or "").strip().lower()
    w = blocked_word(s)
    if w or s not in allowed:
        raise ManifestError(f"split {split!r} is not teacher-eligible (allowed {allowed}; held-out words refused)")
    return s


# ----------------------------------------------------------------------------------------------------------- decisions
def _options_of(row, kind):
    raw = row.get("ordered_options", row.get("options"))
    if raw is None:
        raw = []
    out = []
    if isinstance(raw, dict):
        raw = [{"key": k, "text": v} for k, v in raw.items()]
    for o in raw:
        if isinstance(o, str):
            out.append((o, None))
        elif isinstance(o, (list, tuple)) and len(o) == 2:
            out.append((o[0], o[1]))
        elif isinstance(o, dict) and "key" in o:
            out.append((o["key"], o.get("text", o.get("description"))))
        else:
            raise ManifestError(f"option {o!r}: expected a string, [key, text] or {{key, text}}")
    for k, t in out:
        if not isinstance(k, str) or not k.strip():
            raise ManifestError(f"option key {k!r}: keys are nonempty strings")
        if t is not None and not isinstance(t, (str, int, float, dict, list)):
            raise ManifestError(f"option text of {k!r} has type {type(t).__name__}")
    keys = [k for k, _ in out]
    if len(set(keys)) != len(keys):
        raise ManifestError(f"duplicate option keys {keys}")
    return out


def normalize_decision(row, allowed_splits=ALLOWED_SPLITS):
    """a manifest row -> the internal decision (ONLY the visible fields + identity); raises ManifestError."""
    uid = row.get("decision_uid")
    if not isinstance(uid, str) or not uid:
        raise ManifestError("decision_uid missing")
    parent = row.get("parent_uid")
    if not isinstance(parent, str) or not parent:
        raise ManifestError(f"{uid}: parent_uid missing")
    split = check_split(row.get("split"), allowed_splits)
    kind = row.get("kind")
    if kind not in KINDS:
        raise ManifestError(f"{uid}: kind {kind!r} not in {KINDS}")
    q = row.get("question", row.get("instructions"))
    if q is not None and not isinstance(q, str):
        raise ManifestError(f"{uid}: question must be a string (the instructions)")
    state = row.get("state")
    if state is not None and not isinstance(state, (str, dict, list)):
        raise ManifestError(f"{uid}: state has type {type(state).__name__}")
    opts = _options_of(row, kind)
    defs = None
    if kind == "noul":
        m = {"true": "true", "yes": "true", "false": "false", "no": "false"}
        given = dict(row.get("noul_definitions") or {})
        for k, t in opts:
            kk = m.get(k.strip().lower())
            if kk is None:
                raise ManifestError(f"{uid}: noul option key {k!r} is not true / false / yes / no")
            if isinstance(t, str):
                # a RENDERED display text (Stage 0 writes R2's "Yes" / "No" or "Yes: <definition>") is not a definition:
                # keep only what follows "Yes:" / "No:"; a bare "Yes" / "No" means no definition
                mm = NOUL_DISPLAY.match(t)
                if mm:
                    t = mm.group(2)
            if t is not None:
                given[kk] = t
        defs = {}
        for k, v in given.items():
            kk = m.get(str(k).strip().lower())
            if kk is None:
                raise ManifestError(f"{uid}: noul definition key {k!r}")
            if v is not None and str(v).strip():
                defs[kk] = v if isinstance(v, str) else canon(v)
        opts = [("true", defs.get("true")), ("false", defs.get("false"))]
        defs = defs or None
    elif len(opts) < 2:
        raise ManifestError(f"{uid}: a {kind} decision needs at least 2 options, got {len(opts)}")
    return {"decision_uid": uid, "parent_uid": parent, "split": split, "kind": kind, "state": state,
            "instructions": q or "", "options": opts, "noul_defs": defs,
            "source_family": row.get("source_family"), "skill": row.get("skill"), "domain": row.get("domain"),
            "record_uid": row.get("record_uid")}


def load_manifest(path, allowed_splits=ALLOWED_SPLITS, exclude_sha256=None, check_path=True):
    """-> (decisions in file order, info). Every row is validated; one bad row refuses the manifest. exclude_sha256: a set of
    sha256(decision_uid) / sha256(parent_uid) digests (a protected-panel digest list): matching rows are dropped and only
    COUNTED, never sent."""
    if check_path:
        refuse_path(path)
    out, seen, header, excluded = [], set(), None, 0
    for n, row in iter_jsonl(path):
        if row.get("manifest_header"):
            header = row
            continue
        try:
            d = normalize_decision(row, allowed_splits)
        except ManifestError as e:
            raise ManifestError(f"{path}:{n}: {e}") from None
        if d["decision_uid"] in seen:
            raise ManifestError(f"{path}:{n}: duplicate decision_uid {d['decision_uid']}")
        seen.add(d["decision_uid"])
        if exclude_sha256 and (sha256_text(d["decision_uid"]) in exclude_sha256 or sha256_text(d["parent_uid"]) in exclude_sha256):
            excluded += 1
            continue
        out.append(d)
    info = {"path": os.path.basename(path), "sha256": sha256_file(path), "rows": len(out), "excluded_protected": excluded,
            "parents": len({d["parent_uid"] for d in out}), "header": header,
            "opaque_keys": sum(1 for d in out if d["kind"] != "noul" for k, _ in d["options"] if OPAQUE_KEY.match(k))}
    return out, info


def perturb_parents(decisions, n=PERTURB_N):
    parents = sorted({d["parent_uid"] for d in decisions}, key=lambda p: (sha256_text(f"{PERTURB_TAG}|{p}"), p))
    chosen = parents[:n]
    return set(chosen), sha256_text("\n".join(sorted(chosen)))


def shard_of(parent_uid, n):
    return int(sha256_text(f"sdv1-shard|{parent_uid}")[:12], 16) % n


# ---------------------------------------------------------------------------------------------------------- transforms
def _seed(tag, uid):
    return int(sha256_text(f"{tag}|{uid}")[:16], 16)


def apply_transform(d, name):
    """-> (decision', spec) or (None, spec) when not applicable. Never touches anything but the presented visible fields."""
    if name == "base":
        return d, {"name": "base"}
    if name == "perm1":
        k = len(d["options"])
        if d["kind"] != "choice" or k < 2:
            return None, {"name": "perm1", "not_applicable": f"{d['kind']} K={k}: no meaning-preserving reorder"}
        rng = random.Random(_seed(PERM_TAG, d["decision_uid"]))
        ident = list(range(k))
        while True:
            order = ident[:]
            rng.shuffle(order)
            if order != ident:
                break
        e = dict(d, options=[d["options"][i] for i in order])
        return e, {"name": "perm1", "order": order, "tag": PERM_TAG}
    if name == "irrel1":
        i = _seed(IRREL_TAG, d["parent_uid"]) % len(IRRELEVANT_NOTES)
        note = IRRELEVANT_NOTES[i]
        st = d["state"]
        spec = {"name": "irrel1", "note_index": i, "note_sha256": sha256_text(note), "tag": IRREL_TAG}
        if st is None or (isinstance(st, str) and not st.strip()):
            return dict(d, state=note), dict(spec, placement="state_was_empty")
        if isinstance(st, str):
            return dict(d, state=st + "\n\n" + note), dict(spec, placement="after_state")
        if isinstance(st, dict) and "unrelated_note" not in st:
            return dict(d, state={**st, "unrelated_note": note}), dict(spec, placement="json_key_unrelated_note")
        return None, dict(spec, not_applicable="state is a list (or already has unrelated_note)")
    raise ValueError(f"unknown transform {name!r}")


# ------------------------------------------------------------------------------------------------------ request builders
def state_text(state):
    if state is None:
        return ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def option_display(key, text):
    """(key, description-or-None) as a System One criteria entry: the key alone when there is no distinct description."""
    if text is None:
        return None
    t = text if isinstance(text, str) else canon(text)
    return None if (not t.strip() or t == key) else t


def systemone_question(d):
    """the System One / Decision Index question for d3, Perplexity and the Garas engine (one question per request)."""
    q = {"type": d["kind"], "instructions": d["instructions"]}
    if d["kind"] == "choice":
        q["criteria"] = {k: option_display(k, t) for k, t in d["options"]}
    elif d["kind"] == "noul":
        defs = d.get("noul_defs") or {}
        if defs:
            q["criteria"] = {"false": defs.get("false"), "true": defs.get("true")}
    else:
        q["criteria"] = [(t if isinstance(t, str) and t.strip() else (canon(t) if t is not None else k)) for k, t in d["options"]]
    return q


def option_lines_texts(d):
    """the option texts a letter-readout prompt shows (choice: the text, or the key when there is none)."""
    if d["kind"] == "noul":
        defs = d.get("noul_defs") or {}
        return [("true", "Yes" + (f": {defs['true']}" if defs.get("true") else "")),
                ("false", "No" + (f": {defs['false']}" if defs.get("false") else ""))]
    return [(k, option_display(k, t) or k) for k, t in d["options"]]


def keys_of(d):
    return [k for k, _ in d["options"]]


def map_systemone_answer(d, ans):
    """a System One answer object (d3 / Garas engine) -> {semantic key: p}."""
    if not isinstance(ans, dict):
        raise InvalidOutput(f"answer is {type(ans).__name__}")
    keys = keys_of(d)
    if d["kind"] == "noul":
        p = ans.get("noul")
        if not isinstance(p, (int, float)) or not math.isfinite(p):
            raise InvalidOutput(f"noul answer {ans!r}")
        return {"true": float(p), "false": 1.0 - float(p)}
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        raise InvalidOutput("no probabilities in the answer")
    if d["kind"] == "score" and set(probs) == {str(i) for i in range(len(keys))} and set(probs) != set(keys):
        return {keys[int(i)]: float(v) for i, v in probs.items()}
    return {str(k): float(v) for k, v in probs.items()}


def map_index_probs(d, values, native_keys):
    """probabilities in a teacher's native option order (native_keys = the semantic key of each slot) -> {key: p}."""
    if len(values) != len(native_keys):
        raise InvalidOutput(f"{len(values)} probabilities for {len(native_keys)} options")
    return {k: float(v) for k, v in zip(native_keys, values)}


def validate_probabilities(d, probs, tol=PROB_TOL):
    if not isinstance(probs, dict):
        raise InvalidOutput("probabilities is not an object")
    want = set(keys_of(d))
    if set(probs) != want:
        raise InvalidOutput(f"keys {sorted(probs)[:6]} != options {sorted(want)[:6]}")
    vals = list(probs.values())
    if any((not isinstance(v, float)) or not math.isfinite(v) or v < -1e-12 for v in vals):
        raise InvalidOutput("non-finite or negative probability")
    s = sum(vals)
    if abs(s - 1.0) > tol:
        raise InvalidOutput(f"probabilities sum to {s}")
    return probs


# ------------------------------------------------------------------------------------------------------------ predictions
HASHED = ("schema", "teacher_id", "teacher_pin_sha256", "decision_uid", "parent_uid", "kind", "K", "transform",
          "transform_spec", "presented_keys", "request_sha256", "status", "status_detail", "probabilities", "native_answer",
          "n_input_tokens", "extra")


def prediction_sha256(row):
    return sha256_canon({k: row.get(k) for k in HASHED})


def make_row(teacher_id, pin_sha, d0, d, transform, spec, res, latency):
    row = {"schema": SCHEMA_PRED, "teacher_id": teacher_id, "teacher_pin_sha256": pin_sha, "decision_uid": d0["decision_uid"],
           "parent_uid": d0["parent_uid"], "source_family": d0.get("source_family"), "kind": d0["kind"],
           "K": len(d0["options"]), "transform": transform, "transform_spec": spec,
           "presented_keys": keys_of(d) if d is not None else None,
           "request_sha256": sha256_canon(res.request) if res.request is not None else None, "status": res.status,
           "status_detail": res.detail, "probabilities": res.probabilities if res.status == "ok" else None,
           "native_answer": res.native_answer, "n_input_tokens": res.n_input_tokens, "extra": res.extra,
           "latency_s": None if latency is None else round(latency, 4)}
    row["prediction_sha256"] = prediction_sha256(row)
    return row


def check_row(row, d0):
    """the contract every written row satisfies (the merge re-checks it)."""
    if row["status"] not in STATUSES:
        raise InvalidOutput(f"status {row['status']!r}")
    if row["status"] == "ok":
        validate_probabilities(d0, row["probabilities"])
    elif row["probabilities"] is not None:
        raise InvalidOutput("a non-ok row carries probabilities")
    if row["prediction_sha256"] != prediction_sha256(row):
        raise InvalidOutput("prediction_sha256 mismatch")


def work_items(decisions, perturb, transforms=TRANSFORMS):
    """(decision, transform) pairs: base for all; perm1 / irrel1 for the perturbation sample's parents."""
    items = [(d, "base") for d in decisions]
    for t in transforms:
        if t != "base":
            items += [(d, t) for d in decisions if d["parent_uid"] in perturb]
    return items


def run(backend, items, out_path, teacher_id, pin_sha, done=None, log_every=200, on_flush=None, flush_every_s=900):
    """score items with backend.predict_batch; append rows to out_path (resumable: `done` = (uid, transform) already written)."""
    done = set(done or ())
    todo = []
    for d0, t in items:
        if (d0["decision_uid"], t) in done:
            continue
        d, spec = apply_transform(d0, t)
        todo.append((d0, d, t, spec))
    stats = {"rows": 0, "status": {}, "seconds": 0.0, "tokens": 0}
    if getattr(backend, "streaming", False):
        return _run_streaming(backend, todo, out_path, teacher_id, pin_sha, stats, log_every, on_flush, flush_every_s)
    t0 = last_flush = time.time()
    with open_text(out_path, "at") as g:
        chunk = getattr(backend, "batch_size", 1)
        i = 0
        while i < len(todo):
            part = todo[i:i + chunk]
            i += len(part)
            runnable = [(d0, d, t, s) for d0, d, t, s in part if d is not None]
            na = [(d0, d, t, s) for d0, d, t, s in part if d is None]
            res, lat = [], None
            if runnable:
                ts = time.time()
                res = backend.predict_batch([d for _, d, _, _ in runnable])
                lat = (time.time() - ts) / len(runnable)
            rows = [make_row(teacher_id, pin_sha, d0, None, t, s, Result("not_applicable", detail=s.get("not_applicable", "")), None)
                    for d0, d, t, s in na]
            for (d0, d, t, s), r in zip(runnable, res):
                if r.status == "ok":
                    try:
                        validate_probabilities(d0, r.probabilities)
                    except InvalidOutput as e:
                        r = Result("invalid_output", detail=str(e), native_answer=r.native_answer, request=r.request,
                                   n_input_tokens=r.n_input_tokens, extra=r.extra)
                rows.append(make_row(teacher_id, pin_sha, d0, d, t, s, r, lat))
            for row in rows:
                g.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["rows"] += 1
                stats["status"][row["status"]] = stats["status"].get(row["status"], 0) + 1
                stats["tokens"] += row["n_input_tokens"] or 0
            if stats["rows"] // log_every != (stats["rows"] - len(rows)) // log_every:
                el = time.time() - t0
                print(f"  {stats['rows']}/{len(todo)} rows  {stats['status']}  {stats['rows'] / max(el, 1e-9):.2f} rows/s  "
                      f"eta {(len(todo) - stats['rows']) * el / max(stats['rows'], 1) / 60:.1f} min", flush=True)
            if on_flush and time.time() - last_flush > flush_every_s:
                g.flush()
                on_flush()
                last_flush = time.time()
    stats["seconds"] = round(time.time() - t0, 2)
    return stats


def _finish_row(teacher_id, pin_sha, d0, d, t, s, r, lat):
    if r.status == "ok":
        try:
            validate_probabilities(d0, r.probabilities)
        except InvalidOutput as e:
            r = Result("invalid_output", detail=str(e), native_answer=r.native_answer, request=r.request,
                       n_input_tokens=r.n_input_tokens, extra=r.extra)
    return make_row(teacher_id, pin_sha, d0, d, t, s, r, lat)


def _run_streaming(backend, todo, out_path, teacher_id, pin_sha, stats, log_every, on_flush, flush_every_s):
    """(v5) a backend with streaming = True and predict_one(d): keep backend.concurrency requests in flight across the whole
    shard (no per-chunk tail), write each row as it completes (order-free; resume reads (uid, transform) pairs)."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    t0 = last_flush = time.time()
    width = int(getattr(backend, "concurrency", 8))
    with open_text(out_path, "at") as g, ThreadPoolExecutor(width) as ex:
        def emit(row):
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats["status"][row["status"]] = stats["status"].get(row["status"], 0) + 1
            stats["tokens"] += row["n_input_tokens"] or 0
            if stats["rows"] % log_every == 0:
                el = time.time() - t0
                print(f"  {stats['rows']}/{len(todo)} rows  {stats['status']}  {stats['rows'] / max(el, 1e-9):.2f} rows/s  "
                      f"eta {(len(todo) - stats['rows']) * el / max(stats['rows'], 1) / 60:.1f} min", flush=True)
        pending, it = {}, iter(todo)
        while True:
            while len(pending) < 2 * width:
                nxt = next(it, None)
                if nxt is None:
                    break
                d0, d, t, s = nxt
                if d is None:
                    emit(make_row(teacher_id, pin_sha, d0, None, t, s, Result("not_applicable", detail=s.get("not_applicable", "")), None))
                    continue
                pending[ex.submit(backend.predict_one, d)] = (d0, d, t, s, time.time())
            if not pending:
                break
            fin, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in fin:
                d0, d, t, s, ts = pending.pop(fut)
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001 - an explicit status, never a distribution
                    r = Result("failure", detail=f"{type(e).__name__}: {e}")
                emit(_finish_row(teacher_id, pin_sha, d0, d, t, s, r, time.time() - ts))
            if on_flush and time.time() - last_flush > flush_every_s:
                g.flush()
                on_flush()
                last_flush = time.time()
    stats["seconds"] = round(time.time() - t0, 2)
    return stats


def read_done(path):
    done = set()
    if os.path.exists(path):
        with open_text(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break            # a torn last line from a killed run: everything before it stands
                done.add((r["decision_uid"], r["transform"]))
    return done


def repair_tail(path):
    """drop a torn last line (a killed writer) so appends stay valid JSONL."""
    if not os.path.exists(path):
        return 0
    good, bad = [], 0
    with open_text(path) as f:
        for line in f:
            try:
                json.loads(line)
                good.append(line if line.endswith("\n") else line + "\n")
            except json.JSONDecodeError:
                bad += 1
    if bad:
        with open_text(path, "wt") as g:
            g.writelines(good)
    return bad


# ----------------------------------------------------------------------------------------------------------- pins / CLI
def load_pin(pins_path, teacher):
    pins = json.load(open(pins_path, encoding="utf-8"))
    if teacher not in pins.get("teachers", {}):
        raise SystemExit(f"teacher {teacher!r} not in {pins_path}")
    pin = pins["teachers"][teacher]
    return pin, sha256_canon(pin)


def git_blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_snapshot(pin, d, which="files"):
    """every pinned file under d: LFS files by sha256, small files by git blob sha1; extra weight files refused."""
    files = pin[which]
    bad = []
    for rel, f in sorted(files.items()):
        p = os.path.join(d, rel)
        if not os.path.exists(p):
            bad.append(f"missing {rel}")
            continue
        if os.path.getsize(p) != f["size"]:
            bad.append(f"size {rel}")
            continue
        got = sha256_file(p) if f.get("sha256") else git_blob_sha1(p)
        if got != (f.get("sha256") or f.get("git_oid")):
            bad.append(f"hash {rel}")
    extra = [x for x in os.listdir(d) if x.endswith(".safetensors") and x not in files]
    bad += [f"unpinned weight file {x}" for x in extra]
    return bad


def cmd_plan(a):
    ds, info = load_manifest(a.manifest)
    pert, ph = perturb_parents(ds, a.perturb_n)
    items = work_items(ds, pert)
    kinds = {}
    for d in ds:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    kb = {"2": 0, "3-5": 0, "6-16": 0, "17-32": 0, "33-52": 0, "53+": 0}
    for d in ds:
        k = len(d["options"])
        kb["2" if k <= 2 else "3-5" if k <= 5 else "6-16" if k <= 16 else "17-32" if k <= 32 else "33-52" if k <= 52 else "53+"] += 1
    chars = [len(state_text(d["state"])) + len(d["instructions"]) + sum(len(str(t or k)) for k, t in d["options"]) for d in ds]
    chars.sort()
    applicable = sum(1 for d, t in items if apply_transform(d, t)[0] is not None)
    out = {"manifest": info, "decisions": len(ds), "parents": info["parents"], "kinds": kinds, "K_buckets": kb,
           "perturb_parents": len(pert), "perturb_sample_sha256": ph, "work_items": len(items), "applicable_items": applicable,
           "visible_chars": {"mean": round(sum(chars) / max(len(chars), 1)), "p50": chars[len(chars) // 2] if chars else 0,
                             "p95": chars[int(len(chars) * 0.95)] if chars else 0, "max": chars[-1] if chars else 0},
           "approx_tokens_mean": round(sum(chars) / max(len(chars), 1) / 3.6)}
    if a.shards:
        out["shard_items"] = [sum(1 for d, t in items if shard_of(d["parent_uid"], a.shards) == s) for s in range(a.shards)]
    print(json.dumps(out, indent=1))
    return 0


def cmd_estimate(a):
    """GPU-hours for a manifest from measured dry runs: items x seconds per item (the dry bench, scaled linearly by the mean visible
    length of the manifest's decisions over the dry set's) + model download / verify / load. A planning number, not a measurement."""
    ds, info = load_manifest(a.manifest)
    pert, _ = perturb_parents(ds, a.perturb_n)
    items = [(d, t) for d, t in work_items(ds, pert) if apply_transform(d, t)[0] is not None]
    vis = lambda d: len(state_text(d["state"])) + len(d["instructions"]) + sum(len(str(t or k)) for k, t in d["options"])  # noqa: E731
    mean_chars = sum(vis(d) for d, _ in items) / max(len(items), 1)
    out = {"manifest_sha256": info["sha256"], "decisions": len(ds), "items_applicable": len(items), "mean_visible_chars": round(mean_chars),
           "teachers": {}}
    tot = 0.0
    for p in a.dry:
        r = json.load(open(p))
        b = r.get("bench") or {}
        rps = b.get("rows_per_s") or (r["run"]["rows"] / max(r["run"]["seconds"], 1e-9))
        scale = (mean_chars / r["visible_chars_mean"]) ** a.length_exponent if r.get("visible_chars_mean") else 1.0
        sec = len(items) / rps * scale + (r.get("load_s") or 0) + a.download_s
        tot += sec / 3600
        out["teachers"][r["teacher"]] = {"bench_rows_per_s": rps, "length_scale": round(scale, 3), "gpu_hours": round(sec / 3600, 2),
                                         "peak_gpu_mem_mib": r.get("peak_gpu_mem_used_mib_nvidia_smi"), "gpu": r.get("gpu"),
                                         "think_tokens": r.get("think_tokens")}
    out["total_gpu_hours"] = round(tot, 2)
    out["ceiling_gpu_hours"] = 40
    print(json.dumps(out, indent=1))
    return 0


def _backend(a, pin, pin_sha):
    sys.path.insert(0, HERE)
    import sdv1_backends as sb
    return sb.load_backend(a.teacher, pin, snapshot=a.snapshot, adapter=getattr(a, "adapter", None), device=a.device,
                           options=json.loads(a.backend_options or "{}"))


def cmd_score(a):
    pin, pin_sha = load_pin(a.pins, a.teacher)
    excl = set(open(a.exclude_sha256).read().split()) if a.exclude_sha256 else None
    ds, info = load_manifest(a.manifest, exclude_sha256=excl)
    pert, ph = perturb_parents(ds, a.perturb_n)
    si, sn = (int(x) for x in a.shard.split("/"))
    mine = [d for d in ds if shard_of(d["parent_uid"], sn) == si]
    if a.sample_only:                    # (v4) only the fixed perturbation-sample parents: base + perm1 + irrel1 for each
        mine = [d for d in mine if d["parent_uid"] in pert]
    items = work_items(mine, pert)
    if a.limit:
        items = items[:a.limit]
    print(f"score {a.teacher} pin {pin_sha[:12]} manifest {info['sha256'][:12]} rows {info['rows']} (excluded {info['excluded_protected']}) "
          f"shard {si}/{sn}{' SAMPLE-ONLY' if a.sample_only else ''}: {len(mine)} decisions, {len(items)} items; perturbation sample "
          f"{len(pert)} parents {ph[:12]}", flush=True)
    bad = repair_tail(a.out)
    done = read_done(a.out)
    if done:
        print(f"RESUME {len(done)} rows already in {a.out} (torn lines dropped: {bad})", flush=True)
    be = _backend(a, pin, pin_sha)
    flush = None
    if a.flush_cmd:
        flush = lambda: os.system(a.flush_cmd)  # noqa: E731 (the job's own upload command; no user input)
    try:
        st = run(be, items, a.out, a.teacher, pin_sha, done=done, on_flush=flush)
    finally:
        if hasattr(be, "close"):
            be.close()
    summ = {"version": VERSION, "teacher": a.teacher, "teacher_pin_sha256": pin_sha, "manifest": info, "shard": a.shard,
            "perturb_n": a.perturb_n, "perturb_sample_sha256": ph, "sample_only": bool(a.sample_only), "items": len(items),
            "resumed_rows": len(done), "stats": st,
            "provenance": be.provenance(), "self_md5": md5_file(__file__)}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1, sort_keys=True)
    print(f"SCORE_DONE {a.teacher} shard {a.shard}: {st}", flush=True)
    return 0


def cmd_merge(a):
    ds, info = load_manifest(a.manifest)
    byuid = {d["decision_uid"]: d for d in ds}
    pert, ph = perturb_parents(ds, a.perturb_n)
    scope = [d for d in ds if d["parent_uid"] in pert] if a.sample_only else ds
    want = {(d["decision_uid"], t) for d, t in work_items(scope, pert)}
    got, teacher, pins = {}, set(), set()
    for p in a.inputs:
        with open_text(p) as f:
            for line in f:
                r = json.loads(line)
                k = (r["decision_uid"], r["transform"])
                if k not in want:
                    raise SystemExit(f"{p}: row {k} is not in this manifest's work list")
                if k in got:
                    if got[k]["prediction_sha256"] != r["prediction_sha256"]:
                        raise SystemExit(f"{p}: conflicting duplicate {k}")
                    continue
                check_row(r, byuid[r["decision_uid"]])
                got[k] = r
                teacher.add(r["teacher_id"])
                pins.add(r["teacher_pin_sha256"])
    if len(teacher) != 1 or len(pins) != 1:
        raise SystemExit(f"one teacher / pin per merge: {teacher} {pins}")
    missing = want - set(got)
    if missing:
        raise SystemExit(f"MERGE_INCOMPLETE: {len(missing)} of {len(want)} rows missing (e.g. {sorted(missing)[:3]})")
    order = [(d["decision_uid"], t) for d, t in work_items(scope, pert)]
    with open_text(a.out, "wt") as g:
        for k in order:
            g.write(json.dumps(got[k], ensure_ascii=False) + "\n")
    st = {}
    for r in got.values():
        st.setdefault(r["transform"], {}).setdefault(r["status"], 0)
        st[r["transform"]][r["status"]] += 1
    summ = {"teacher": teacher.pop(), "teacher_pin_sha256": pins.pop(), "manifest_sha256": info["sha256"], "rows": len(got),
            "sample_only": bool(a.sample_only), "decisions_covered": len({u for u, _ in got}),
            "perturb_sample_sha256": ph, "status_by_transform": st, "md5": md5_file(a.out)}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1, sort_keys=True)
    print("MERGE_OK", json.dumps(summ))
    return 0


def cmd_verify(a):
    pin, _ = load_pin(a.pins, a.teacher)
    bad = verify_snapshot(pin, a.dir, a.files)
    if bad:
        print(f"VERIFY_FAIL {a.teacher} {a.files}: {bad[:5]} ({len(bad)} problems)")
        return 3
    print(f"VERIFY_OK {a.teacher} {a.files}: {len(pin[a.files])} files")
    return 0


def cmd_dry(a):
    sys.path.insert(0, HERE)
    import sdv1_dry
    return sdv1_dry.dry(a, load_pin, _backend)


def cmd_selftest(a):
    sys.path.insert(0, HERE)
    import pytest  # noqa: F401  (the node image has pytest; the job runs the shipped tests)
    return os.system(f"{sys.executable} -m pytest -q -p no:cacheprovider {os.path.join(HERE, 'test_sdv1_teacher.py')}") >> 8


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", required=True)
    p.add_argument("--perturb-n", type=int, default=PERTURB_N)
    p.add_argument("--shards", type=int, default=0)
    p.set_defaults(fn=cmd_plan)
    for name, fn in (("score", cmd_score), ("dry", cmd_dry)):
        p = sub.add_parser(name)
        p.add_argument("--teacher", required=True)
        p.add_argument("--pins", required=True)
        p.add_argument("--snapshot", required=True)
        p.add_argument("--adapter", default=None)
        p.add_argument("--device", default="cuda")
        p.add_argument("--backend-options", default="")
        p.add_argument("--out", required=True)
        if name == "score":
            p.add_argument("--manifest", required=True)
            p.add_argument("--shard", default="0/1")
            p.add_argument("--perturb-n", type=int, default=PERTURB_N)
            p.add_argument("--exclude-sha256", default="")
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--sample-only", action="store_true", help="(v4) only the perturbation-sample parents")
            p.add_argument("--flush-cmd", default="")
        else:
            p.add_argument("--decisions", required=True)
            p.add_argument("--bench-repeats", type=int, default=0)
            p.add_argument("--min-easy-acc", type=float, default=0.75)
            p.add_argument("--parity-rows", type=int, default=8)
            p.add_argument("--require-parity", action="store_true")
        p.set_defaults(fn=fn)
    p = sub.add_parser("merge")
    p.add_argument("--manifest", required=True)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--perturb-n", type=int, default=PERTURB_N)
    p.add_argument("--sample-only", action="store_true", help="(v4) the shards cover only the perturbation-sample parents")
    p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("estimate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dry", nargs="+", required=True, help="dry_<teacher>.json reports")
    p.add_argument("--perturb-n", type=int, default=PERTURB_N)
    p.add_argument("--length-exponent", type=float, default=1.0)
    p.add_argument("--download-s", type=float, default=900.0)
    p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("check-key")
    p.add_argument("key")
    p.set_defaults(fn=lambda a: (refuse_key(a.key), print("KEY_OK", a.key), 0)[2])
    p = sub.add_parser("verify-snapshot")
    p.add_argument("--teacher", required=True)
    p.add_argument("--pins", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--files", default="files", choices=("files", "adapter_files"))
    p.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ManifestError as e:
        print(f"MANIFEST_REFUSED: {e}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
