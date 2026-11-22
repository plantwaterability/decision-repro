#!/usr/bin/env python3
"""
build_jev_ext3.py -- DRAFT (2026-09-30). A third extension corpus: the top zero-shot-eligible candidates of the
dataset scout (research/datasets_scout.md), converted to the jev_data record schema and deduplicated exactly as
build_jev_ext2.py does. It imports build_jev_ext2 and reuses its fingerprint index, eval-state fetchers (jev-bench
test, typed-decisions ALL splits, TDB v0.3, MMLU/ARC, LegalBench, PhishNChips/community, DICES, ChaosNLI), process()
(validate -> length -> cross-source exact dedup -> exact / leaf / 8-word-shingle dedup -> cap -> write) and the
cache_features read-back, so a record that passes here passes the same gate as ext2.

NOT YET RUN END TO END: the converters are written against rows read through the HF datasets-server on 2026-09-30
(see the scout report) and pass --self-test offline; run the --small smoke test on a node before a real build.

Sources (output source -> origin; licence; labels; question types). Soft = a distribution, not one-hot.
  tsgraded/<task>  tasksource/tasksource-jev-typed-decisions, the "SoftLabeling" tasks only, minus every task whose
                   source data is (or contains) one of our eval sets or is already in our corpora:
                       blocked: glue/stsb (= jev-bench stsb), chaos-mnli-ambiguity (= ChaosNLI), civil_comments/*
                       (= jev-bench civil_comments), scruples/verdict_votes (already ext2 scruples/anecdotes),
                       procedural-typed-decisions/* (already jev_ext_strict)
                   kept (see TS_GRADED): UNLI, DynaSent r1/r2 votes, LeWiDi md_agreement / hs_brexit / mp, oasst2
                   ratings (quality, helpfulness, creativity, humor, toxicity, violence), ProtoQA, WouldYouRather,
                   acceptability-prediction votes, SICK relatedness, STS-companion, google_wellformed_query.
                   Licence: per row (`license`, `license_use`); default keeps license_use == "commercial" only
                   (--ts-license-use commercial,unspecified widens it). By that rule SICK and acceptability rating
                   votes (non-commercial) and LeWiDi md_agreement / hs_brexit (unspecified) drop out by default;
                   oasst2 is kept through TS_LICENCE_OVERRIDE (upstream Apache-2.0, checked by hand). Labels: human annotator vote shares / mean
                   ratings (>= 5 annotators per item per the card;
                   oasst2 ratings are means of 3 per its sources.yaml). Types: noul, choice, score. Rows sharing a
                   `group_id` (several ratings of one item) become ONE record with several questions.
  typed/<name>     typed-decisions-schema synthetic corpora (state + questions{type, instructions, criteria} + a soft
                   gold per question) labelled by OPEN teachers, from the scout's ranked list (TYPED_SOURCES below).
                   The four typed-decisions workflow domains are dropped with --strict (as build_jev_ext.py --strict).

Rules carried over from the scout (do not relax without re-reading research/datasets_scout.md):
  * Nothing derived from LocalLLaMA/typed-decisions (train or test, incl. translations) is ever a source here; the
    typed-decisions states of BOTH splits are eval fingerprints (ext2's ev_typed), so a stray copy is dropped anyway.
  * No source labelled by Jev / TypeSafe, and no closed-model teacher, is in the default source list.
  * Every record's `license` is the row's own licence string (tsgraded) or the source licence (typed/*).

Splits: train / val ~95/5 by a hash of the per-source group key (meta.group), as ext2.
Output: <out>/{train,val}/<source>.jsonl ('/' -> '__') and <out>/manifest.json.

Usage (on a node; pandas + pyarrow + huggingface_hub; HF access is anonymous):
  export HF_HOME=/data/local/hf_jev_ext3
  python3 build_jev_ext3.py --out /data/local/jev_ext3 \\
      --eval jev_real=/data/local/jev_real/test/real.jsonl --eval extevals=/data/local/extevals \\
      --fingerprint jev_data=/data/local/jev_data --fingerprint jev_ext_strict=/data/local/jev_ext_strict \\
      --fingerprint jev_ext2=/data/local/jev_ext2
  python3 build_jev_ext3.py --self-test                                   # no network
  python3 build_jev_ext3.py --small --limit-rows 400 --out /tmp/x ...     # one shard / small files only
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

from claude_overlap import norm  # noqa: E402

import build_jev_ext2 as x2  # noqa: E402  fingerprint Index, eval fetchers, process(), report / summary helpers
from build_jev_data import (  # noqa: E402
    Writer,
    choice_q,
    noul_q,
    record,
    score_q,
    shuffled,
    validate,
)
from build_jev_ext import (  # noqa: E402
    Labeler,
    as_obj,
    final_check,
    humanize,
    instructions_text,
    kind_of,
    level_text,
    noul_instructions,
    plain,
    read_jsonl,
    state_text,
    text_of,
    unique_levels,
)

MAX_K = x2.MAX_K
VAL_PCT = x2.VAL_PCT

# --------------------------------------------------------------------------- tasksource graded (soft human labels)
TS_REPO = "tasksource/tasksource-jev-typed-decisions"
TS_SHARDS = 13
# kept SoftLabeling tasks -> output source (the family is what --sources selects). Card + build-report.jsonl, 2026-09-30.
TS_GRADED = {
    "UNLI": "tsgraded/unli",
    "dynasent/r1_votes": "tsgraded/dynasent", "dynasent/r2_votes": "tsgraded/dynasent",
    "lewidi/md_agreement": "tsgraded/lewidi", "lewidi/hs_brexit": "tsgraded/lewidi", "lewidi/mp": "tsgraded/lewidi",
    "oasst2/quality": "tsgraded/oasst2", "oasst2/helpfulness": "tsgraded/oasst2", "oasst2/creativity": "tsgraded/oasst2",
    "oasst2/humor": "tsgraded/oasst2", "oasst2/toxicity": "tsgraded/oasst2", "oasst2/violence": "tsgraded/oasst2",
    "proto_qa/proto_qa": "tsgraded/protoqa",
    "wouldyourather": "tsgraded/wouldyourather",
    "acceptability-prediction/rating_votes": "tsgraded/acceptability",
    "acceptability-prediction/binary_votes": "tsgraded/acceptability",
    "sick/relatedness": "tsgraded/sts", "sts-companion": "tsgraded/sts",
    "google_wellformed_query": "tsgraded/wellformed",
}
# never read, whatever --sources says: eval sets or already-trained-on sources (prefix match on the row's `source`)
TS_BLOCKED = ("glue/stsb", "chaos-mnli", "civil_comments", "scruples", "procedural-typed-decisions", "go_emotions",
              "measuring", "helpsteer", "HelpSteer", "lewidi")
TS_MAX_Q = 6
# tasksource marks a few copies "unspecified" whose upstream licence was checked by hand (scout, 2026-09-30): the row's
# license_use is treated as this value and the record's licence string says where it came from.
TS_LICENCE_OVERRIDE = {
    "oasst2/": ("commercial", "apache-2.0 (upstream OpenAssistant/oasst2; tasksource copy oasst2_dense_flat says unspecified)"),
}

# --------------------------------------------------------------------------- typed-decisions-schema synthetic sets
# name -> dict(repo, files, gold (the soft-target field), licence, teacher, closed (closed-model teacher: opt-in only),
#              keep (row filter), note). From the scout's ranking (research/datasets_scout.md, 2026-09-30).
TYPED_SOURCES = {
    # DeepSeek-V4.1-Flash (MIT, open weights) wrote the cases and labelled each 3x at T=0.7; `teacher` = mean of the
    # three verbalized distributions (the typed-decisions gold recipe, larger teacher); `gold` is the hard answer and is
    # NOT read. The card leaves out typed-decisions' four workflows. Correct option listed first ~half the time: choice
    # criteria are reshuffled here.
    "tds": dict(repo="n4ze3m/typed-decisions-synth", files=["data/train.jsonl", "data/validation.jsonl"], gold="teacher",
                licence="MIT (card YAML; teacher weights MIT)", teacher="deepseek-ai/DeepSeek-V4.1-Flash x3 mean, T=0.7",
                closed=False, keep=None),
    # The 675 scenarios v2 added (subset == "v2"); v1 (subset == "v1", 2,000 rows) is already in jev_ext_strict.
    # Claude author + blind Claude reviewer (rubric probabilities): CLOSED-TEACHER, same footing as v1 -> opt-in only.
    "mghafiri_v2": dict(repo="mghafiri/decision-model-scenarios-v2",
                        files=["data/train.jsonl", "data/validation.jsonl", "data/test.jsonl"], gold="targets",
                        licence="MIT (card YAML + LICENSE)", teacher="Claude author + Claude reviewer (closed)",
                        closed=True, keep=lambda r: r.get("subset") == "v2"),
}

LICENSES = {
    "tsgraded": "per row: the source dataset's licence(s) from tasksource (`license`); license_use filtered (see header)",
}
ORIGINS = {s: TS_REPO for s in set(TS_GRADED.values())}
CAPS = {"tsgraded/unli": 2500, "tsgraded/dynasent": 2500, "tsgraded/lewidi": 2000, "tsgraded/oasst2": 2500,
        "tsgraded/protoqa": 1500, "tsgraded/wouldyourather": 1000, "tsgraded/acceptability": 1500, "tsgraded/sts": 1500,
        "tsgraded/wellformed": 1000}
FAMILIES = ["tsgraded", "typed"]
STRICT_DOMAINS = re.compile(r"support|help ?desk|customer service|ticket|siem|security (alert|incident)|soc\b|invoice|"
                            r"procure|accounts payable|agent trace|tool (call|routing|selection)|observability", re.IGNORECASE)


def family_of(source):
    return source.split("/")[0]


def license_of(source):
    return LICENSES.get(source) or LICENSES[family_of(source)]


def cap_of(a, source):
    return max(1, min(a.max_per_source, int(round(CAPS[source] * a.scale))))


def make_record(source, state, qs, labs, meta, group, probes=(), license_=None):
    kinds = {q: kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, None)
    rec["base_id"] = rec["id"]
    rec["license"] = license_ or license_of(source)
    rec["meta"] = {"dataset": ORIGINS.get(source, "?"), "group": str(group), **meta}
    rec["_probes"] = [p for p in probes if p]
    return rec


def new_stats(source, lab):
    """ext2's per-source stats dict, with ext3's origin / licence tables."""
    return {"dataset": ORIGINS.get(source, "?"), "license": license_of(source), "cap": None, "candidates": 0,
            "written": Counter(), "units": Counter(), "label_kinds": Counter(), "qtypes": Counter(), "drops": Counter(),
            "dedup_exact": Counter(), "dedup_leaf": Counter(), "dedup_shingle": Counter(), "question_drops": lab.drops,
            "overlap_examples": [], "errors": [], "hf_sha": None}


# --------------------------------------------------------------------------- converters (pure: rows -> records)
def ts_question(row, rng, lab):
    """One tasksource row -> (question, label) in our schema, or None. kinds: noul [p]; choice / score: options +
    target over them. Choice criteria are the option texts under letter keys, shuffled; score levels keep the
    source order (ordered scale)."""
    kind, q = row.get("kind"), text_of(row.get("question"))
    opts, tgt = list(row.get("options") or []), list(row.get("target") or [])
    if not q:
        lab.drops["empty_instructions"] += 1
        return None
    if kind == "noul":
        d = lab.noul(tgt[0] if len(tgt) == 1 else None)
        return (noul_q(q), d) if d else None
    if kind not in ("choice", "score"):
        lab.drops["unsupported_question"] += 1
        return None
    opts = [text_of(o) for o in opts]
    if len(opts) < 2 or not all(opts) or len({norm(o) for o in opts}) != len(opts):
        lab.drops["bad_options"] += 1
        return None
    if kind == "score":
        levels = unique_levels(opts)
        d = lab.dist(levels, tgt)
        return (score_q(q, levels), d) if d else None
    keys = [chr(97 + j) if j < 26 else chr(65 + j - 26) for j in range(len(opts))]
    d = lab.dist(keys, tgt)
    if not d:
        return None
    crit = shuffled(dict(zip(keys, opts, strict=True)), rng)
    return choice_q(q, crit), {k: d[k] for k in crit}


def conv_tsgraded(rows, rng, lab, allowed_use=("commercial",)):
    """Rows (any order) -> one record per (output source, group_id, state): up to TS_MAX_Q questions about one item.
    Blocked / non-allow-listed tasks and rows whose license_use is not allowed are dropped and counted."""
    groups = defaultdict(list)
    for r in rows:
        src = str(r.get("source") or "")
        if src.startswith(TS_BLOCKED):
            lab.drops["blocked_task"] += 1
            continue
        out = TS_GRADED.get(src)
        if out is None:
            lab.drops["not_graded_task"] += 1
            continue
        use = r.get("license_use")
        for prefix, (use_, lic_) in TS_LICENCE_OVERRIDE.items():
            if src.startswith(prefix):
                r = dict(r, license_use=use_, license=lic_)
                use = use_
        if use not in allowed_use:
            lab.drops[f"license_use_{use}"] += 1
            continue
        if not text_of(r.get("state")):
            lab.drops["empty_state"] += 1
            continue
        groups[(out, r.get("group_id") or r.get("id"), norm(r["state"])[:500])].append(r)
    for (out, gid, _), grp in groups.items():
        qs, labs, seen, tasks, lics = {}, {}, set(), [], set()
        for r in grp:
            got = ts_question(r, rng, lab)
            if not got:
                continue
            qd, d = got
            sig = norm(qd["instructions"])
            if sig in seen:
                lab.drops["duplicate_question"] += 1
                continue
            seen.add(sig)
            qid = re.sub(r"\W+", "_", r["source"].split("/")[-1]).strip("_").lower() or "q"
            qid = qid if qid not in qs else f"{qid}_{len(qs)}"
            qs[qid], labs[qid] = qd, d
            tasks.append(r["source"])
            lics.add(str(r.get("license") or "unspecified"))
            if len(qs) >= TS_MAX_Q:
                break
        if qs:
            yield make_record(out, grp[0]["state"], qs, labs,
                              {"tasks": tasks, "variant": grp[0].get("variant"), "orig_split": grp[0].get("split"),
                               "license_use": grp[0].get("license_use")},
                              group=gid, probes=[grp[0]["state"]], license_=" | ".join(sorted(lics)))


def crit_text(k, v):
    """A choice criterion as display text: strings verbatim; {description|definition|...} dicts (mghafiri v2) by
    their description; else the humanized key."""
    if isinstance(v, dict):
        for f in ("description", "definition", "meaning", "text", "label"):
            if text_of(v.get(f)):
                return text_of(v[f])
        return level_text(v) or humanize(k)
    return text_of(v) or humanize(k)


def typed_question(q, t, rng, lab):
    """A typed-decisions-schema question + its gold -> (question, label) or None. Accepts the gold shapes seen in the
    candidates: {"probabilities": {...}} (choice/score/noul), {"noul": p}, a bare {option: p} dict, or a list."""
    if not isinstance(q, dict) or t is None:
        lab.drops["target_missing"] += 1
        return None
    instr = instructions_text(q.get("instructions"))
    if not instr:
        lab.drops["empty_instructions"] += 1
        return None
    typ, crit = q.get("type"), q.get("criteria")
    probs = t.get("probabilities") if isinstance(t, dict) and isinstance(t.get("probabilities"), (dict, list)) else t
    if typ == "noul":
        p = None
        if isinstance(t, dict) and "noul" in t:
            p = t["noul"]
        elif isinstance(probs, dict):
            for k in ("true", "yes", "True", "Yes", "1"):
                if k in probs:
                    p = probs[k]
                    break
        d = lab.noul(p)
        return (noul_q(noul_instructions(instr, crit)), d) if d else None
    if typ == "choice" and isinstance(crit, dict) and isinstance(probs, dict):
        keys = [str(k) for k in crit]
        probs = {str(k): v for k, v in probs.items()}
        if set(probs) != set(keys):
            lab.drops["target_count_mismatch"] += 1
            return None
        d = lab.dist(keys, [probs[k] for k in keys])
        if not d:
            return None
        criteria = shuffled({str(k): crit_text(k, v) for k, v in crit.items()}, rng)
        return choice_q(instr, criteria), {k: d[k] for k in criteria}
    if typ == "score" and isinstance(crit, list):
        levels = unique_levels([level_text(c) for c in crit])
        if isinstance(probs, list):
            vals = probs
        elif isinstance(probs, dict) and set(map(str, probs)) == {str(i) for i in range(len(levels))}:
            vals = [{str(k): v for k, v in probs.items()}[str(i)] for i in range(len(levels))]
        else:
            lab.drops["target_count_mismatch"] += 1
            return None
        d = lab.dist(levels, vals)
        return (score_q(instr, levels), d) if d else None
    lab.drops["unsupported_question"] += 1
    return None


def conv_typed(rows, source, rng, lab, strict=False, gold_key="gold"):
    """Whole-case rows {state, questions, <gold_key>} (JSON objects or JSON strings) -> one record per case."""
    for r in rows:
        state = state_text(as_obj(r.get("state")))
        Q, G = as_obj(r.get("questions")), as_obj(r.get(gold_key))
        if not state or not isinstance(Q, dict) or not isinstance(G, dict):
            lab.drops["bad_row"] += 1
            continue
        dom = " ".join(text_of(r.get(k)) for k in ("domain", "workflow", "family", "task") if r.get(k))
        if strict and STRICT_DOMAINS.search(dom):
            lab.drops["strict_domain"] += 1
            continue
        qs, labs = {}, {}
        for qid, q in Q.items():
            got = typed_question(q, G.get(qid), rng, lab)
            if got:
                qs[str(qid)], labs[str(qid)] = got
        if qs:
            yield make_record(source, state, qs, labs, {"orig_id": r.get("id") or r.get("state_id"), "domain": dom or None},
                              group=r.get("group") or r.get("id") or r.get("state_id") or norm(state)[:200], probes=[state])


# --------------------------------------------------------------------------- loaders (network) -> candidate records
def ts_rows(a, wanted_tasks):
    """Rows of the kept graded tasks, read shard by shard with a pyarrow filter on `source` (the 13 train shards are
    ~116 MB each; --small reads one). Validation/test splits are not read: they are tasksource's own eval rows."""
    import pyarrow.parquet as pq
    shards = range(1 if a.small else TS_SHARDS)
    cols = ["state", "kind", "id", "options", "target", "question", "source", "variant", "split", "group_id",
            "license", "license_use"]
    for k in shards:
        path = x2.hf_file(a, TS_REPO, f"data/train-{k:05d}-of-{TS_SHARDS:05d}.parquet")
        t = pq.read_table(path, columns=cols, filters=[("source", "in", sorted(wanted_tasks))])
        yield from t.to_pylist()


def load_tsgraded(a, rng, lab, source):
    tasks = [t for t, s in TS_GRADED.items() if s == source]
    rows = list(ts_rows(a, tasks))
    if a.limit_rows:
        rows = rows[: a.limit_rows]
    return conv_tsgraded(rows, rng, lab, allowed_use=tuple(a.ts_license_use))


def load_typed(a, rng, lab, source):
    spec = TYPED_SOURCES[source.split("/", 1)[1]]
    rows = []
    for fn in spec["files"][:1] if a.small else spec["files"]:
        p = x2.hf_file(a, spec["repo"], fn)
        for r in (x2.read_parquet(p) if fn.endswith(".parquet") else read_jsonl(p)):
            if spec["keep"] is None or spec["keep"](r):
                rows.append(r)
            else:
                lab.drops["filtered_rows"] += 1
    rows = x2.sample(rows, a.limit_rows or None, rng)
    return conv_typed(rows, source, rng, lab, strict=a.strict, gold_key=spec["gold"])


def register_typed():
    for name, sp in TYPED_SOURCES.items():
        CAPS.setdefault(f"typed/{name}", 8000 if name == "tds" else 3000)
        LICENSES[f"typed/{name}"] = sp["licence"]
        ORIGINS[f"typed/{name}"] = sp["repo"]


def loader_for(source):
    if source.startswith("tsgraded/"):
        return lambda a, rng, lab: load_tsgraded(a, rng, lab, source)
    if source.startswith("typed/"):
        return lambda a, rng, lab: load_typed(a, rng, lab, source)
    raise KeyError(source)


# --------------------------------------------------------------------------- pipeline (ext2's, with ext3's sources)
def build(a):
    out = Path(a.out)
    stale = [p for sp in ("train", "val", "test") for p in (out / sp).glob("*.jsonl")]
    if stale or (out / "manifest.json").exists():
        sys.exit(f"{out} already holds a corpus ({len(stale)} jsonl files); write to a new --out")
    t0 = time.time()
    idx, coverage = x2.load_fingerprints(a)
    writer, seen = Writer(out), set()
    manifest = {
        "builder": "build_jev_ext3.py (draft)", "seed": a.seed, "small": a.small, "limit_rows": a.limit_rows,
        "strict": a.strict, "ts_license_use": a.ts_license_use,
        "splits": {"train": 100 - VAL_PCT, "val": VAL_PCT, "by": "sha1 of the normalized per-source group key (meta.group)"},
        "caps": {s: cap_of(a, s) for s in a.sources}, "max_state_chars": a.max_state_chars,
        "dedup": {"as": "build_jev_ext2.py", "fingerprint_states": dict(idx.sizes), "eval_coverage": coverage},
        "licenses": {s: license_of(s) for s in a.sources}, "sources": {},
    }
    for source in a.sources:
        rng, lab = random.Random(f"{a.seed}:{source}"), Labeler(a.renorm_tol)
        st = new_stats(source, lab)
        print(f"[{source}] loading...", flush=True)
        try:
            cands = list(loader_for(source)(a, rng, lab))
            x2.process(source, cands, cap_of(a, source), idx, writer, rng, st, seen, a.max_state_chars)
        except Exception as e:  # noqa: BLE001
            st["errors"].append(f"FAILED {type(e).__name__}: {e}")
            print(f"[{source}] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        x2.report_source(source, st)
        manifest["sources"][source] = plain(st)
    writer.close()
    out.mkdir(parents=True, exist_ok=True)
    manifest["readback"] = final_check(out)
    manifest["seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    x2.print_summary(manifest)
    failed = [s for s, v in manifest["sources"].items() if any(e.startswith("FAILED") for e in v["errors"])]
    print(f"manifest written to {out / 'manifest.json'}" + (f"; FAILED sources: {failed}" if failed else ""))
    return 1 if failed else 0


# --------------------------------------------------------------------------- self-test (offline)
def self_test():
    register_typed()
    rng, lab = random.Random(0), Labeler(0.01)
    oasst = "User: Tell me a joke about cats.\nAssistant: Why did the cat sit on the computer? To keep an eye on the mouse."
    ts = [
        {"source": "oasst2/humor", "kind": "score", "question": "How humorous is the reply?", "state": oasst,
         "options": ["0: not at all", "1", "2", "3", "4: very"], "target": [0.0, 0.2, 0.4, 0.4, 0.0],
         "group_id": "g1", "license": "apache-2.0", "license_use": "commercial", "variant": "direct", "split": "train"},
        {"source": "oasst2/quality", "kind": "score", "question": "How good is the reply overall?", "state": oasst,
         "options": ["0: poor", "1", "2", "3", "4: excellent"], "target": [0.0, 0.0, 0.6, 0.4, 0.0],
         "group_id": "g1", "license": "apache-2.0", "license_use": "commercial", "variant": "direct", "split": "train"},
        {"source": "UNLI", "kind": "noul", "question": "How likely is the hypothesis to be true, given the premise?",
         "state": "Passage A: A man rides a bike. Passage B: A person is outdoors.", "options": [], "target": [0.752],
         "group_id": "g2", "license": "mit", "license_use": "commercial", "variant": "direct", "split": "train"},
        {"source": "proto_qa/proto_qa", "kind": "choice", "question": "Which answer would a respondent give?",
         "state": "Name something you need to play basketball.", "options": ["ball", "hoop", "shoes"],
         "target": [0.6, 0.3, 0.1], "group_id": "g3", "license": "cc-by-4.0", "license_use": "commercial"},
        {"source": "civil_comments/toxicity_share", "kind": "noul", "question": "toxic?", "state": "x y z",
         "options": [], "target": [0.2], "group_id": "g4", "license": "cc0-1.0", "license_use": "commercial"},
        {"source": "lewidi/mp", "kind": "noul", "question": "Is it ironic?", "state": "Great, another Monday.",
         "options": [], "target": [0.6], "group_id": "g5", "license": "unspecified", "license_use": "unspecified"},
    ]
    recs = list(conv_tsgraded(ts, rng, lab))
    for r in recs:
        validate(r)
    by = {r["source"]: r for r in recs}
    assert set(by) == {"tsgraded/oasst2", "tsgraded/unli", "tsgraded/protoqa"}, sorted(by)
    assert set(by["tsgraded/oasst2"]["questions"]) == {"humor", "quality"}, "one record per group_id"
    assert by["tsgraded/unli"]["labels"]["unli"]["yes"] == 0.752 and by["tsgraded/unli"]["label_kind"]["unli"] == "soft"
    assert lab.drops["blocked_task"] == 2 and not lab.drops["license_use_unspecified"], dict(lab.drops)   # lewidi now blocked (GDE-1 A)
    pq = by["tsgraded/protoqa"]
    assert sorted(pq["labels"]["proto_qa"].values()) == [0.1, 0.3, 0.6] and set(pq["questions"]["proto_qa"]["criteria"]) == {"a", "b", "c"}
    print(f"ok  tsgraded: {len(recs)} records, blocked + licence filters, grouping")

    rows = [{"id": "c1", "domain": "veterinary triage calls",
             "state": json.dumps({"pet": {"species": "dog", "age_years": 6}, "wound": "deep cut on the paw, bleeding"}),
             "questions": json.dumps({
                 "severity": {"type": "choice", "instructions": "Categorize the severity of the wound.",
                              "criteria": {"minor": "Superficial", "moderate": "Needs sutures", "severe": "Deep, bleeding"}},
                 "urgent": {"type": "noul", "instructions": "The pet must be seen today.",
                            "criteria": {"true": "Same-day visit", "false": "Can wait"}},
                 "pain": {"type": "score", "instructions": "Rate the pain.", "criteria": ["none", "mild", "strong"]}}),
             "gold": json.dumps({"severity": {"probabilities": {"minor": 0.01, "moderate": 0.26, "severe": 0.73}},
                                 "urgent": {"noul": 0.9}, "pain": {"probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}})},
            {"id": "c2", "domain": "customer support tickets", "state": "Ticket: refund please",
             "questions": {"q": {"type": "noul", "instructions": "Refund due."}}, "gold": {"q": {"noul": 0.5}}}]
    lab2 = Labeler(0.01)
    tr = list(conv_typed(rows, "typed/tds", rng, lab2, strict=True, gold_key="gold"))
    assert len(tr) == 1 and lab2.drops["strict_domain"] == 1
    t = tr[0]
    validate(t)
    assert t["labels"]["urgent"]["yes"] == 0.9 and "Yes means: Same-day visit" in t["questions"]["urgent"]["instructions"]
    assert t["labels"]["pain"]["strong"] == 0.6 and t["questions"]["pain"]["levels"] == ["none", "mild", "strong"]
    print("ok  typed: whole-case schema, noul descriptions in the instructions, strict domain filter")

    idx = x2.Index()
    ev = idx.tag("eval:jevbench/chaosnli", "eval")
    idx.add(ev, "Passage A: A man rides a bike. Passage B: A person is outdoors.")
    idx.freeze()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ext3"
        writer, seen, sts = Writer(out), set(), {}
        for r in recs + tr:
            r = dict(r, _probes=list(r.get("_probes", [])))
            s = r["source"]
            sts.setdefault(s, new_stats(s, Labeler(0.01)))
            x2.process(s, [r], 100, idx, writer, random.Random(1), sts[s], seen, 6500)
        writer.close()
        assert sts["tsgraded/unli"]["dedup_exact"] == {"eval:jevbench/chaosnli": 1}, sts["tsgraded/unli"]
        rb = final_check(out)
        assert "tsgraded/unli" not in rb["per_source"] and "tsgraded/oasst2" in rb["per_source"]
        print("ok  pipeline: ext2 dedup + write + iter_units read-back")
    print("\nself-test passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="all", help="comma-separated output sources or families (default all)")
    ap.add_argument("--out")
    ap.add_argument("--eval", action="append", default=[], help="TAG=PATH eval jsonl file or directory (repeatable)")
    ap.add_argument("--fingerprint", action="append", default=[], help="TAG=PATH training corpus (repeatable)")
    ap.add_argument("--no-fetch-evals", dest="fetch_evals", action="store_false")
    ap.add_argument("--skip-evals", default="")
    ap.add_argument("--allow-missing-evals", action="store_true")
    ap.add_argument("--allow-missing-corpora", action="store_true")
    ap.add_argument("--ts-license-use", default="commercial",
                    help="tasksource license_use values to keep: commercial[,unspecified][,non-commercial]")
    ap.add_argument("--strict", action="store_true", help="drop typed-decisions' four workflow domains (typed/*)")
    ap.add_argument("--allow-closed-teacher", action="store_true",
                    help="include sources labelled by closed models (e.g. typed/mghafiri_v2, Claude); the user's call")
    ap.add_argument("--max-per-source", type=int, default=3000)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--oversample", type=float, default=1.5)
    ap.add_argument("--max-state-chars", type=int, default=6500)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--cache", default=str(Path("~/.cache/huggingface/jev_ext3_urls").expanduser()))
    ap.add_argument("--renorm-tol", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.out:
        ap.error("--out is required")
    a.ts_license_use = [s.strip() for s in a.ts_license_use.split(",") if s.strip()]
    register_typed()
    closed = {f"typed/{n}" for n, sp in TYPED_SOURCES.items() if sp["closed"]}
    want = [s.strip() for s in a.sources.split(",") if s.strip()]
    a.sources = list(CAPS) if want == ["all"] else [s for s in CAPS if s in want or family_of(s) in want]
    if not a.allow_closed_teacher:
        skipped = [s for s in a.sources if s in closed]
        a.sources = [s for s in a.sources if s not in closed]
        if skipped:
            print(f"closed-model teacher sources skipped (pass --allow-closed-teacher to include): {skipped}")
    if not a.sources:
        ap.error(f"no sources selected; choose from {list(CAPS)} or families {FAMILIES}")
    a.skip_evals = {s.strip() for s in a.skip_evals.split(",") if s.strip()}
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
