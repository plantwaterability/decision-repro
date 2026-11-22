#!/usr/bin/env python3
"""
build_jev_ext5.py -- (EXT5, 2026-10-09) a CLEAN-LINE corpus from tasksource/tasksource-jev-typed-decisions, config
`filtered-full`, at a pinned repo revision. Pre-registered in AGENT_NOTES 2026-10-09 22:36Z "(EXT5)" before any build.

What it is: an allow-list of tasksource tasks in four groups (1 soft / probabilistic labels, 2 evidence updating, 3 rule
application, 4 agent-action safety) plus an OPTIONAL family listed separately. Each task's rows are converted to the jev_data
record schema exactly as build_jev_ext3.py converts tasksource rows (its ts_question), grouped into ONE record per
(output source, group_id, state) with up to 6 questions (tasksource's choice / noul / score variants of one item stay together).

Filters and gates (in this order):
  * allow-list only; a block list (eval sets / already-trained sources / closed-model output) is checked first;
  * per row: license_use == commercial (the snapshot's own policy, re-checked) and no NC / research-only licence string;
  * held sources (card says synthetic with an unstated / open generator) are skipped unless --include-held (the user's call);
    agent_action_safety (card mentions Gemini) is excluded outright;
  * G1: the sealed final_v1 exclusion indexes (rd_ / hl_exclusion_index: ids, state sha1s, entity keys, clusters) through
    fleet/ss_lib.load_exclusions + hits, on candidates BEFORE dedup (index files only; no sealed item is read);
  * G2: build_jev_ext2's process(): validate, length, cross-source exact dedup, fingerprint dedup (exact / leaf / 8-word shingle)
    against the corpora (--fingerprint) and every eval set (--eval files + ext2's fetched eval families), cap, write;
  * zero-shot assertion: no record derives from procedural-typed-decisions / typed-decisions (blocked; asserted on output).
Splits: train / val by sha1 of the per-source group key (ext2's rule). Output: <out>/{train,val}/<source>.jsonl, manifest.json,
licences.json (per source: tasks, licence strings, license_use, flags, written counts).

Usage (on a node):
  python3 build_jev_ext5.py --out /data/local/jev_ext5 --eval jev_real=... --eval extevals=... --eval di021=... \\
      --eval gde1v3=... --fingerprint jev_data=... --fingerprint jev_ext_strict=... --fingerprint jev_ext2=... \\
      --exclusion-index rd=.../rd_exclusion_index.json.gz --exclusion-index hl=.../hl_exclusion_index.json.gz
  python3 build_jev_ext5.py --self-test          # offline
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "analysis", HERE / "fleet"):
    sys.path.insert(0, str(_p))

from claude_overlap import norm  # noqa: E402

import build_jev_ext3 as x3  # noqa: E402  tasksource row converter (ts_question), ext2 machinery as x3.x2
import ss_lib  # noqa: E402  sealed exclusion indexes (load_exclusions / hits) and the NC licence regex
from build_jev_data import Writer, record  # noqa: E402
from build_jev_ext import Labeler, final_check, kind_of, plain, text_of  # noqa: E402

x2 = x3.x2

TS_REPO = "tasksource/tasksource-jev-typed-decisions"
TS_REVISION = "477db7523772a988510c336c68cb623bfd989755"
TS_CONFIG = "filtered-full"
TS_SHARDS = 13
TS_MAX_Q = 6
COLS = ["state", "kind", "id", "options", "target", "question", "source", "variant", "split", "group_id", "license", "license_use"]

# output source -> tasks (exact tasksource `source` values; a value ending in '/*' is a prefix), cap in records, flags
SOURCES = {
    # group 1: soft / probabilistic labels
    "ts5soft/unli": dict(tasks=["UNLI"], cap=6000, flags=["SA-derived text (SNLI pairs, CC BY-SA 4.0 upstream)"]),
    "ts5soft/dynasent": dict(tasks=["dynasent/r1_votes", "dynasent/r2_votes"], cap=6000, flags=[]),
    "ts5soft/scruples": dict(tasks=["scruples/verdict_votes"], cap=4000,
                             flags=["licence string 'Custom (DPI)'", "Reddit text", "expect dedup vs ext2 scruples"]),
    "ts5soft/acceptability": dict(tasks=["acceptability-prediction/binary_votes", "acceptability-prediction/rating_votes"], cap=2500,
                                  flags=["ext3 header once listed acceptability votes as NC; tasksource now says commercial"]),
    "ts5soft/probwords": dict(tasks=["probability_words_nli/reasoning_1hop", "probability_words_nli/reasoning_2hop"], cap=3000,
                              flags=["ShareAlike (CC BY-SA 4.0 DPI)"]),
    # group 2: evidence updating
    "ts5evid/defeasible": dict(tasks=["defeasible-nli/atomic", "defeasible-nli/snli", "defeasible-nli/social"], cap=3000,
                               flags=["snli part SA-derived text"]),
    "ts5evid/corr2cause": dict(tasks=["corr2cause"], cap=3000, flags=["programmatic"]),
    "ts5evid/vitaminc": dict(tasks=["vitaminc"], cap=3000, flags=["ShareAlike (cc-by-sa-3.0)"]),
    "ts5evid/wice": dict(tasks=["wice"], cap=2500, flags=["ShareAlike (cc-by-sa-4.0)"]),
    "ts5evid/health_fact": dict(tasks=["health_fact"], cap=3000, flags=[]),
    "ts5evid/temporal": dict(tasks=["temporal-nli"], cap=3000, flags=[]),
    # group 3: rule application
    "ts5rule/ruletaker": dict(tasks=["ruletaker"], cap=2500, flags=["programmatic"]),
    "ts5rule/pararule": dict(tasks=["PARARULE-Plus"], cap=2500, flags=["programmatic"]),
    "ts5rule/conceptrules": dict(tasks=["conceptrules_v2"], cap=2500, flags=["licence string 'Custom (DPI)'"]),
    "ts5rule/logical": dict(tasks=["logical-entailment"], cap=2500, flags=["programmatic"]),
    "ts5rule/babi": dict(tasks=["babi_nli/*"], cap=2500, flags=["programmatic"]),
    "ts5rule/case_hold": dict(tasks=["lex_glue/case_hold"], cap=3000, flags=[]),
    "ts5rule/unfair_tos": dict(tasks=["lex_glue/unfair_tos"], cap=1500, flags=[]),
    # group 4: agent-action safety (cards checked 2026-10-09 for model-generated text)
    "ts5safe/prompt_injections": dict(tasks=["prompt-injections"], cap=600,
                                      flags=["upstream deepset/prompt-injections card: CC BY 4.0 (tasksource: apache-2.0)"]),
    "ts5safe/promptshield": dict(tasks=["PromptShield"], cap=2500,
                                 flags=["origin unverified (card shows no model-generation mention)"]),
    # optional family: built and listed separately; NOT in test (a)
    "ts5opt/sdoh": dict(tasks=["SDOH-NLI"], cap=1500, flags=["optional", "origin to verify"]),
    "ts5opt/ethics": dict(tasks=["ethics/*"], cap=2500, flags=["optional"]),
    "ts5opt/moral_stories": dict(tasks=["moral_stories/full"], cap=2500, flags=["optional"]),
    # held (closed/synthetic-model rule): never built unless --include-held
    "ts5held/shell_safety": dict(tasks=["shell-safety-v2"], cap=2500, flags=["HELD: card says synthetic, generator unstated"]),
    "ts5held/pi_neuralchemy": dict(tasks=["Prompt-injection-dataset/full"], cap=2500,
                                   flags=["HELD: card says 'synthetically' generated"]),
    "ts5held/pi_slabs": dict(tasks=["prompt-injection-dataset"], cap=2500, flags=["HELD: card mentions Llama"]),
}
FAMILIES = ["ts5soft", "ts5evid", "ts5rule", "ts5safe", "ts5opt", "ts5held"]
DEFAULT_FAMILIES = ["ts5soft", "ts5evid", "ts5rule", "ts5safe", "ts5opt"]
TEST_A_FAMILIES = ["ts5soft", "ts5evid", "ts5rule", "ts5safe"]
EXCLUDED_TASKS = {"agent_action_safety": "card mentions Gemini: closed-model output"}
# ext3's block list WITHOUT "scruples" (ext5 wants scruples/verdict_votes; overlap with ext2's scruples is removed by the dedup gate),
# plus this registration's eval sets / closed-model sources.
BLOCKED = tuple(b for b in x3.TS_BLOCKED if b != "scruples") + (
    "fever", "lex_glue/ledgar", "sms_spam", "MedQA", "folio", "FOL-nli", "gen_debiased_nli/mnli", "HelpSteer3", "typed-decisions",
    "agent_action_safety")
CARD_LICENCE = {"prompt-injections": "cc-by-4.0 (upstream deepset/prompt-injections card; tasksource row says apache-2.0)"}


def family_of(source):
    return source.split("/")[0]


def task_map(sources):
    """[(exact or prefix task, output source)] for the selected output sources."""
    out = []
    for s in sources:
        for t in SOURCES[s]["tasks"]:
            out.append((t, s))
    return out


def route(task, tmap):
    for t, s in tmap:
        if (t.endswith("/*") and task.startswith(t[:-1])) or task == t:
            return s
    return None


def make_record(source, state, qs, labs, meta, group, probes, license_):
    kinds = {q: kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, None)
    rec["base_id"] = rec["id"]
    rec["license"] = license_
    rec["meta"] = {"dataset": TS_REPO, "revision": TS_REVISION, "config": TS_CONFIG, "group": str(group), **meta}
    rec["_probes"] = [p for p in probes if p]
    return rec


def convert(rows, tmap, rng, labs_by_source, allowed_use=("commercial",)):
    """Rows (any order) -> records per (output source, group_id, state). Drops are counted per output source (or 'unrouted')."""
    groups = defaultdict(list)
    for r in rows:
        src = str(r.get("source") or "")
        out = route(src, tmap)
        lab = labs_by_source[out or "unrouted"]
        if src.startswith(BLOCKED) or src in EXCLUDED_TASKS:
            lab.drops["blocked_task"] += 1
            continue
        if out is None:
            lab.drops["not_selected_task"] += 1
            continue
        use = r.get("license_use")
        if use not in allowed_use:
            lab.drops[f"license_use_{use}"] += 1
            continue
        lic = CARD_LICENCE.get(src) or str(r.get("license") or "unspecified")
        if ss_lib.NC_LICENCE.search(lic):
            lab.drops["licence_string_nc"] += 1
            continue
        if not text_of(r.get("state")):
            lab.drops["empty_state"] += 1
            continue
        groups[(out, r.get("group_id") or r.get("id"), norm(r["state"])[:500])].append(dict(r, _licence=lic))
    for (out, gid, _), grp in groups.items():
        lab = labs_by_source[out]
        qs, labs, seen, tasks, lics, variants = {}, {}, set(), [], set(), []
        for r in grp:
            got = x3.ts_question(r, rng, lab)
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
            variants.append(r.get("variant"))
            lics.add(r["_licence"])
            if len(qs) >= TS_MAX_Q:
                break
        if qs:
            yield make_record(out, grp[0]["state"], qs, labs,
                              {"tasks": tasks, "variants": variants, "orig_split": grp[0].get("split"),
                               "license_use": grp[0].get("license_use"), "flags": SOURCES[out]["flags"]},
                              group=gid, probes=[grp[0]["state"]], license_=" | ".join(sorted(lics)))


def read_rows(a, tasks_exact, prefixes):
    """All filtered-full train rows of the selected tasks (pyarrow filter on `source`; prefixes filtered in Python)."""
    import pyarrow.parquet as pq
    rows, shards = [], range(1 if a.small else TS_SHARDS)
    for k in shards:
        p = x2.hf_file(a, TS_REPO, f"{TS_CONFIG}/train-{k:05d}-of-{TS_SHARDS:05d}.parquet", revision=TS_REVISION)
        t = pq.read_table(p, columns=COLS)
        src = t.column("source").to_pylist()
        keep = [i for i, s in enumerate(src) if s in tasks_exact or any(s.startswith(pf) for pf in prefixes)]
        sub = t.take(keep).to_pylist() if keep else []
        rows.extend(sub)
        print(f"shard {k}: {len(sub)} selected rows of {t.num_rows}", flush=True)
    if a.limit_rows:
        rows = rows[: a.limit_rows]
    return rows


def snapshot_provenance(a):
    """The filtered-full filtering report (counts / policies / hashes) at the pinned revision, for the manifest."""
    try:
        p = x2.hf_file(a, TS_REPO, f"{TS_CONFIG}/filtering-report.json", revision=TS_REVISION)
        raw = Path(p).read_bytes()
        rep = json.loads(raw)
        return {"sha256": hashlib.sha256(raw).hexdigest(),
                **{k: rep.get(k) for k in ("source_revision", "base_repository_revision", "retained_rows", "license_use_policy",
                                            "prefilter_exclusions", "llm_rejected_rows", "review_model", "retained_content_sha256")}}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def new_stats(source):
    return {"dataset": TS_REPO, "license": "per row (tasksource `license`; card override where stricter)", "cap": None,
            "candidates": 0, "written": Counter(), "units": Counter(), "label_kinds": Counter(), "qtypes": Counter(),
            "drops": Counter(), "dedup_exact": Counter(), "dedup_leaf": Counter(), "dedup_shingle": Counter(),
            "question_drops": Counter(), "overlap_examples": [], "errors": [], "hf_sha": TS_REVISION}


def load_exclusion_indexes(specs, required=True):
    pairs = []
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--exclusion-index needs NAME=PATH, got {spec!r}")
        pairs.append((name, f"index:{path}"))
    if not pairs:
        if required:
            sys.exit("no --exclusion-index given: the sealed final_v1 indexes are required (or --allow-no-exclusion-index)")
        return None
    return ss_lib.load_exclusions(pairs)


def gate_exclusion(cands, X, st):
    if X is None:
        return cands
    kept = []
    for r in cands:
        why, _ = ss_lib.hits(r, X)
        if why:
            st["drops"]["exclusion_index"] += 1
            for w in why:
                st["drops"][f"exclusion:{w.split(':')[0]}"] += 1
        else:
            kept.append(r)
    return kept


def build(a):
    out = Path(a.out)
    stale = [p for sp in ("train", "val", "test") for p in (out / sp).glob("*.jsonl")]
    if stale or (out / "manifest.json").exists():
        sys.exit(f"{out} already holds a corpus ({len(stale)} jsonl files); write to a new --out")
    t0 = time.time()
    X = load_exclusion_indexes(a.exclusion_index, required=not a.allow_no_exclusion_index)
    idx, coverage = x2.load_fingerprints(a)
    tmap = task_map(a.sources)
    exact = {t for t, _ in tmap if not t.endswith("/*")}
    prefixes = tuple(t[:-1] for t, _ in tmap if t.endswith("/*"))
    rows = read_rows(a, exact, prefixes)
    labs = defaultdict(lambda: Labeler(a.renorm_tol))
    rng = random.Random(f"{a.seed}:convert")
    by_source = defaultdict(list)
    for rec in convert(rows, tmap, rng, labs, allowed_use=tuple(a.ts_license_use)):
        by_source[rec["source"]].append(rec)
    writer, seen = Writer(out), set()
    manifest = {
        "builder": "build_jev_ext5.py", "registration": "AGENT_NOTES 2026-10-09 22:36Z (EXT5)", "seed": a.seed, "small": a.small,
        "limit_rows": a.limit_rows, "ts_license_use": a.ts_license_use, "include_held": a.include_held,
        "snapshot": {"repo": TS_REPO, "revision": TS_REVISION, "config": TS_CONFIG, "filtering_report": snapshot_provenance(a)},
        "splits": {"train": 100 - x2.VAL_PCT, "val": x2.VAL_PCT, "by": "sha1 of the normalized per-source group key (meta.group)"},
        "caps": {s: max(1, int(round(SOURCES[s]["cap"] * a.scale))) for s in a.sources}, "max_state_chars": a.max_state_chars,
        "exclusion_indexes": (X or {}).get("info") if X else "NOT APPLIED (--allow-no-exclusion-index)",
        "dedup": {"as": "build_jev_ext2.py process()", "fingerprint_states": dict(idx.sizes), "eval_coverage": coverage},
        "blocked_prefixes": list(BLOCKED), "excluded_tasks": EXCLUDED_TASKS, "test_a_families": TEST_A_FAMILIES,
        "unrouted_drops": dict(labs["unrouted"].drops), "sources": {},
    }
    for source in a.sources:
        st = new_stats(source)
        st["question_drops"] = labs[source].drops
        cands = gate_exclusion(by_source.get(source, []), X, st)
        x2.process(source, cands, manifest["caps"][source], idx, writer, random.Random(f"{a.seed}:{source}"), st, seen,
                   a.max_state_chars)
        x2.report_source(source, st)
        manifest["sources"][source] = plain(st)
    writer.close()
    out.mkdir(parents=True, exist_ok=True)
    manifest["readback"] = final_check(out)
    bad = zero_shot_violations(out)
    manifest["zero_shot_check"] = {"violations": bad}
    licences = licence_table(out, a.sources)
    (out / "licences.json").write_text(json.dumps(licences, indent=2, ensure_ascii=False))
    manifest["seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    x2.print_summary(manifest)
    if bad:
        print(f"ZERO_SHOT_VIOLATION: {bad[:5]}")
        return 1
    print(f"manifest written to {out / 'manifest.json'}")
    return 0


def zero_shot_violations(out):
    bad = []
    for p in sorted(Path(out).glob("*/*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            tasks = (r.get("meta") or {}).get("tasks") or []
            if not str(r.get("source", "")).startswith("ts5") or any(
                    str(t).startswith(("procedural-typed-decisions", "typed-decisions")) for t in tasks):
                bad.append(r.get("id"))
    return bad


def licence_table(out, sources):
    tab = {s: {"tasks": SOURCES[s]["tasks"], "flags": SOURCES[s]["flags"], "licence_strings": Counter(),
               "license_use": Counter(), "records": Counter(), "units": Counter()} for s in sources}
    for p in sorted(Path(out).glob("*/*.jsonl")):
        split = p.parent.name
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            t = tab.get(r["source"])
            if t is None:
                continue
            t["licence_strings"][r.get("license")] += 1
            t["license_use"][str((r.get("meta") or {}).get("license_use"))] += 1
            t["records"][split] += 1
            t["units"][split] += len(r["questions"])
    return {s: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in d.items()} for s, d in tab.items()}


# --------------------------------------------------------------------------- self-test (offline)
def self_test():
    rng = random.Random(0)
    sel = [s for s in SOURCES if family_of(s) in DEFAULT_FAMILIES]
    tmap = task_map(sel)
    assert route("babi_nli/counting", tmap) == "ts5rule/babi" and route("shell-safety-v2", tmap) is None
    assert route("UNLI", tmap) == "ts5soft/unli" and route("ethics/virtue", tmap) == "ts5opt/ethics"
    assert not "scruples/verdict_votes".startswith(BLOCKED) and route("scruples/verdict_votes", tmap) == "ts5soft/scruples"
    assert "civil_comments/x".startswith(BLOCKED) and "procedural-typed-decisions/x".startswith(BLOCKED)
    assert not any(family_of(s) == "ts5held" for s in sel)
    st = "Premise: A man rides a bike. Hypothesis: A person is outdoors."
    rows = [
        {"source": "UNLI", "kind": "noul", "question": "How likely is the hypothesis, given the premise?", "state": st,
         "options": [], "target": [0.752], "group_id": "g1", "license": "mit", "license_use": "commercial", "variant": "direct"},
        {"source": "dynasent/r1_votes", "kind": "choice", "question": "What is the sentiment?", "state": "The food was cold.",
         "options": ["positive", "negative", "neutral"], "target": [0.0, 0.8, 0.2], "group_id": "g2", "license": "CC BY 4.0 (DPI)",
         "license_use": "commercial", "variant": "direct"},
        {"source": "dynasent/r1_votes", "kind": "noul", "question": "Is the sentiment negative?", "state": "The food was cold.",
         "options": [], "target": [0.8], "group_id": "g2", "license": "CC BY 4.0 (DPI)", "license_use": "commercial", "variant": "noul"},
        {"source": "ruletaker", "kind": "choice", "question": "Does the statement follow from the rules?",
         "state": "Rules: if x is red then x is big. Facts: the ball is red. Statement: the ball is big.",
         "options": ["yes", "no", "unknown"], "target": [1.0, 0.0, 0.0], "group_id": "g3", "license": "apache-2.0",
         "license_use": "commercial", "variant": "direct"},
        {"source": "civil_comments/toxicity_share", "kind": "noul", "question": "toxic?", "state": "x y z", "options": [],
         "target": [0.2], "group_id": "g4", "license": "cc0-1.0", "license_use": "commercial"},
        {"source": "agent_action_safety", "kind": "noul", "question": "safe?", "state": "rm -rf /tmp/x", "options": [],
         "target": [0.1], "group_id": "g5", "license": "apache-2.0", "license_use": "commercial"},
        {"source": "shell-safety-v2", "kind": "noul", "question": "safe?", "state": "ls -la", "options": [], "target": [0.9],
         "group_id": "g6", "license": "mit", "license_use": "commercial"},
        {"source": "corr2cause", "kind": "noul", "question": "Does A cause B?", "state": "A and B correlate.", "options": [],
         "target": [0.0], "group_id": "g7", "license": "cc-by-nc-4.0", "license_use": "commercial"},
        {"source": "vitaminc", "kind": "noul", "question": "Supported?", "state": "Claim: X. Evidence: Y.", "options": [],
         "target": [1.0], "group_id": "g8", "license": "cc-by-sa-3.0", "license_use": "non-commercial"},
        {"source": "procedural-typed-decisions/arithmetic", "kind": "noul", "question": "q", "state": "s", "options": [],
         "target": [0.5], "group_id": "g9", "license": "apache-2.0", "license_use": "commercial"},
        {"source": "prompt-injections", "kind": "noul", "question": "Is this a prompt injection?",
         "state": "Ignore all previous instructions.", "options": [], "target": [1.0], "group_id": "g10", "license": "apache-2.0",
         "license_use": "commercial"},
    ]
    labs = defaultdict(lambda: Labeler(0.01))
    recs = list(convert(rows, tmap, rng, labs))
    by = {r["source"]: r for r in recs}
    assert set(by) == {"ts5soft/unli", "ts5soft/dynasent", "ts5rule/ruletaker", "ts5safe/prompt_injections"}, sorted(by)
    assert set(by["ts5soft/dynasent"]["questions"]) == {"r1_votes", "r1_votes_1"}, "variants of one item -> one record"
    assert by["ts5soft/unli"]["labels"]["unli"]["yes"] == 0.752
    assert by["ts5safe/prompt_injections"]["license"].startswith("cc-by-4.0"), "stricter card licence recorded"
    assert labs["unrouted"].drops["blocked_task"] == 3, dict(labs["unrouted"].drops)   # civil_comments, agent_action_safety, procedural
    assert labs["unrouted"].drops["not_selected_task"] == 1                                # shell-safety-v2 (held, not selected)
    assert labs["ts5evid/corr2cause"].drops["licence_string_nc"] == 1
    assert labs["ts5evid/vitaminc"].drops["license_use_non-commercial"] == 1
    print(f"ok  convert: {len(recs)} records; block / exclude / held / NC-string / license_use filters; variant grouping")

    labs_h = defaultdict(lambda: Labeler(0.01))
    held = list(convert(rows, task_map(list(SOURCES)), rng, labs_h))
    assert "ts5held/shell_safety" in {r["source"] for r in held}, "--include-held routes held sources"
    assert labs_h["unrouted"].drops["blocked_task"] == 3, "agent_action_safety stays excluded even with held sources"
    print("ok  held sources only with --include-held; agent_action_safety always excluded")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ix = {"suite": "selftest", "ids": [], "state_sha1": [ss_lib.shash(by["ts5rule/ruletaker"]["state"])]}
        with gzip.open(tmp / "ix.json.gz", "wt", encoding="utf-8") as f:
            json.dump(ix, f)
        X = load_exclusion_indexes([f"t={tmp / 'ix.json.gz'}"])
        st = new_stats("x")
        kept = gate_exclusion([dict(r, _probes=list(r.get("_probes", []))) for r in recs], X, st)
        assert len(kept) == len(recs) - 1 and st["drops"]["exclusion_index"] == 1, st["drops"]
        print("ok  G1 sealed-index gate drops a state-sha1 hit")

        idx = x2.Index()
        ev = idx.tag("eval:selftest/chaosnli", "eval")
        st_unli = by["ts5soft/unli"]["state"]
        idx.add(ev, st_unli)
        idx.freeze()
        out = tmp / "ext5"
        writer, seen, sts = Writer(out), set(), {}
        for r in kept:
            s = r["source"]
            sts.setdefault(s, new_stats(s))
            x2.process(s, [r], 100, idx, writer, random.Random(1), sts[s], seen, 6500)
        writer.close()
        assert sts["ts5soft/unli"]["dedup_exact"] == {"eval:selftest/chaosnli": 1}, sts["ts5soft/unli"]
        rb = final_check(out)
        assert "ts5soft/unli" not in rb["per_source"] and "ts5soft/dynasent" in rb["per_source"], rb["per_source"]
        assert not zero_shot_violations(out)
        bad_dir = tmp / "bad" / "train"
        bad_dir.mkdir(parents=True)
        (bad_dir / "x.jsonl").write_text(json.dumps({"id": "b1", "source": "ts5rule/logical", "meta": {
            "tasks": ["procedural-typed-decisions/arithmetic"]}}) + "\n" + json.dumps({"id": "b2", "source": "rd3/x", "meta": {}}) + "\n")
        assert zero_shot_violations(tmp / "bad") == ["b1", "b2"], "zero-shot check must flag TD-derived and non-ext5 records"
        lt = licence_table(out, ["ts5soft/dynasent", "ts5safe/prompt_injections"])
        assert sum(lt["ts5soft/dynasent"]["records"].values()) == 1 and lt["ts5soft/dynasent"]["units"]
        print(f"ok  G2 ext2 dedup + write + read-back; zero-shot check; licence table ({st_unli[:20]}...)")
    print("\nself-test passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="default", help="comma-separated output sources or families; 'default' = groups 1-4 + opt")
    ap.add_argument("--include-held", action="store_true", help="also build the HELD sources (closed/synthetic-model rule; user's call)")
    ap.add_argument("--out")
    ap.add_argument("--eval", action="append", default=[], help="TAG=PATH eval jsonl file or directory (repeatable)")
    ap.add_argument("--fingerprint", action="append", default=[], help="TAG=PATH training corpus (repeatable)")
    ap.add_argument("--exclusion-index", action="append", default=[], help="NAME=PATH sealed exclusion index (.json.gz)")
    ap.add_argument("--allow-no-exclusion-index", action="store_true")
    ap.add_argument("--no-fetch-evals", dest="fetch_evals", action="store_false")
    ap.add_argument("--skip-evals", default="")
    ap.add_argument("--allow-missing-evals", action="store_true")
    ap.add_argument("--allow-missing-corpora", action="store_true")
    ap.add_argument("--ts-license-use", default="commercial")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-state-chars", type=int, default=6500)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--cache", default=str(Path("~/.cache/huggingface/jev_ext5_urls").expanduser()))
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
    if a.ts_license_use != ["commercial"]:
        ap.error("the clean line keeps license_use == commercial only")
    want = [s.strip() for s in a.sources.split(",") if s.strip()]
    fams = DEFAULT_FAMILIES + (["ts5held"] if a.include_held else [])
    if want == ["default"]:
        a.sources = [s for s in SOURCES if family_of(s) in fams]
    else:
        a.sources = [s for s in SOURCES if (s in want or family_of(s) in want) and (family_of(s) != "ts5held" or a.include_held)]
    if not a.sources:
        ap.error(f"no sources selected; choose from {list(SOURCES)} or families {FAMILIES}")
    a.skip_evals = {s.strip() for s in a.skip_evals.split(",") if s.strip()}
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
