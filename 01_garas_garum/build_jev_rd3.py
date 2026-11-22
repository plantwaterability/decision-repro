#!/usr/bin/env python3
"""
build_jev_rd3.py -- jev_rd3 (nearly ALL of the gated real-decision pool, one per-source cap) and jev_rd3_stackpool (a held-out
pool for the (92) Bayesian stack and the Bayesian last layer, drawn only from units no rd1 / rd2-trained model has seen; (99)
never trains on it). Sources, cleaning, gate, within-corpus dedupe, sampling and validation are build_jev_rd1.py's,
imported and unchanged (research/rd1_pipeline.md); this script runs them in stages so the memory-heavy gate runs once per
eval group and its hits are cached on disk:

  gate --group A   EXACTLY rd2's gate: discover_evals(root) + ext2's fetched eval families (what rd1 and rd2 indexed).
                   Checked by `build --repro-rd2`, which must reproduce the published jev_rd2 byte for byte.
  gate --group B   the eval sets added for rd3, as a SEPARATE gate (its own document-frequency statistics, so adding them
                   cannot turn an rd2-gate shingle into "boilerplate"): --eval TAG=DIR, e.g. the Decision Index 0.2.1 suite
                   (records keep their request ids) and GDE-1 v3, gated and pre-gate bundles, every split.
  pool             the gated candidate pool (A + B applied) as <work>/pool/{train,val}/<source>.jsonl (+ .keys sidecars),
                   for gate C = analysis/claude_di_overlap.py (the Decision Index board's item-level overlap rule) run on it.
  build            A + B + C (exact / near DI matches) applied, rd1's within-corpus dedupe, then:
                     1. STACKPOOL train: per source, units of the TRAIN pool whose record id AND normalised state are in
                        neither rd1 nor rd2 (train or val); quota water-filled over sources with equal weight, capped per
                        source (--stack-cap); inside a source rd1's sample_source (families sqrt-share, label strata
                        proportional, newest first). Within-pool duplicate states dropped. Removed from the train pool
                        together with every train record sharing a stackpool state.
                     2. rd3 train: sample_source per source at --train-cap (the per-source cap).
                     3. rd3 val: sample_source per source at --val-cap, BEFORE the stack val draw, so rd3 val reproduces
                        rd2 val (same seed, same pool) except where the new gates dropped a record.
                     4. STACKPOOL val: as 1 over the val records left after 3 (outside rd1 / rd2 / rd3 val), --stack-val-total
                        units, --stack-val-cap per source.
                   then the checks (asserted): stackpool vs rd1 / rd2 / rd3 (ids, states), rd3 train vs rd1 / rd2 val,
                   rd3 val vs rd3 train, and an exact id / normalised-state check of every output record against every
                   eval file of gates A and B (hits are dropped, counted as gate `exact_check`).
  validate         rd1's validate (cache_features.iter_units, the trainer's NC / K filters, make_example on every unit).

OUTPUT <out>/jev_rd3/{train,val}/<source>.jsonl + manifest.json + gate_report.json and <out>/jev_rd3_stackpool/{train,val}/
<source>.jsonl + manifest.json. Records keep rd1's layout (meta.rd1 = {source, split, family, date}) plus meta.rd3 =
{part, in_rd1, in_rd2} (part: train / val / stack_train / stack_val; in_rdX: the rdX split holding the record id, or null).

Node run (x11 head, 2026-10-04; layout = build_jev_rd1.py's under --root; research/rd3.md):
  PYTHONHASHSEED=0 python3 build_jev_rd3.py gate --group A --root R --work WK --workers 4
  PYTHONHASHSEED=0 python3 build_jev_rd3.py gate --group B --root R --work WK --workers 4 --no-fetch-evals \\
      --eval di=EXTRA/di/di_v021 --eval gde1v3=EXTRA/gde1v3_gated/gde1 --eval gde1v3pre=EXTRA/gde1v3_pregate/gde1
  PYTHONHASHSEED=0 python3 build_jev_rd3.py build --repro-rd2 --root R --work WK --hits A --out OUT/rd2_repro \\
      --train-cap 25000 --val-cap 500        # must equal jev_rd2 byte for byte
  PYTHONHASHSEED=0 python3 build_jev_rd3.py pool --root R --work WK --hits A,B
  PYTHONHASHSEED=0 python3 analysis/claude_di_overlap.py --di EXTRA/di/di_v021 --jev-real-test R/evals/jev_real/test/real.jsonl \\
      --corpus pool=WK/pool --out WK/di_audit --workers 4
  PYTHONHASHSEED=0 python3 build_jev_rd3.py build --root R --work WK --hits A,B --di-audit WK/di_audit --out OUT \\
      --rd1 REF/jev_rd1 --rd2 REF/jev_rd2 --eval-check ... --tokenizer Qwen/Qwen3.8-27B
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

import build_jev_rd1 as rd1  # noqa: E402  sources, cleaning, gate, sampling, validation: unchanged
from claude_overlap import norm  # noqa: E402

VERSION = "rd3 1.0 (2026-10-04)"
TRAIN_CAP, VAL_CAP = 150_000, 500
STACK_TOTAL, STACK_CAP = 17_000, 2_500
STACK_VAL_TOTAL, STACK_VAL_CAP = 2_000, 150
STACK_SEED_OFFSET = 7919          # stackpool draws: seed + 7919 (train) / + 7920 (val); rd3 keeps rd1's seed / seed + 1
DI_DROP_LEVELS = ("exact", "near")  # the board's counted levels; coincidental / partial are reported only
log = rd1.log


def rss():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return f"maxrss self {s:.1f} GB, children {c:.1f} GB"


def shash(state):
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return hashlib.sha1(norm(s).encode("utf-8")).digest()


def nq(r):
    return len(r["questions"])


# --------------------------------------------------------------------------- candidates (build_jev_rd1.main, verbatim)
def load_candidates(a, names):
    cf = rd1._load("cache_features", HERE / "trainers" / "cache_features.py")
    roots = {"laptop": os.path.join(a.root, "laptop"), "s3": os.path.join(a.root, "s3src")}
    cands, rep, t0 = [], {}, time.time()
    for name in names:
        spec = rd1.SPECS[name]
        r_ = rep[name] = {"files": [], "records_read": Counter(), "question_drops": Counter(), "licenses": Counter()}
        for dkey, sub, pat in spec["dirs"]:
            base = os.path.join(roots[dkey], sub)
            for split in ("train", "val"):
                files = sorted(glob.glob(os.path.join(base, split, pat)))
                if not files:
                    raise SystemExit(f"{name}: no {split} files at {base}/{split}/{pat}")
                for p in files:
                    if name == "openjev" and any(b in Path(p).name for b in rd1.OPENJEV_BLOCKED):
                        raise SystemExit(f"blocked Open-Jev sub present: {p}")
                    r_["files"].append(os.path.relpath(p, a.root))
                    for r in rd1.read_jsonl(p):
                        r_["records_read"][split] += 1
                        rec, drops = rd1.clean_record(r, name, spec, cf.is_noncommercial)
                        r_["question_drops"].update(drops)
                        if rec is None:
                            continue
                        r_["licenses"][rec["license"]] += 1
                        cands.append((f"{name}|{split}|{rec['id']}", rec))
        log(f"{name}: read {dict(r_['records_read'])}; question drops {dict(r_['question_drops'])}")
    log(f"candidates: {len(cands):,} records ({time.time() - t0:.0f}s; {rss()})")
    ids = Counter(k for k, _ in cands)
    dup_ids = [k for k, c in ids.items() if c > 1]
    if dup_ids:
        log(f"WARNING {len(dup_ids)} duplicate candidate keys (kept first), e.g. {dup_ids[:3]}")
        seen, cc = set(), []
        for k, r in cands:
            if k not in seen:
                seen.add(k)
                cc.append((k, r))
        cands = cc
    return cands, rep


def load_candidates_lean(a, names):
    """What the gate reads of each candidate, nothing more: (key, state, probes, entity keys), in load_candidates' order with
    the same duplicate-key rule. build_jev_rd1.run_gate keeps whole records; on a 30 GB head the forked check workers'
    copy-on-write of those records was OOM-killed (2026-10-04), so the gate stage holds only this."""
    cf = rd1._load("cache_features", HERE / "trainers" / "cache_features.py")
    roots = {"laptop": os.path.join(a.root, "laptop"), "s3": os.path.join(a.root, "s3src")}
    out, seen, ndup, t0 = [], set(), 0, time.time()
    for name in names:
        spec = rd1.SPECS[name]
        for dkey, sub, pat in spec["dirs"]:
            base = os.path.join(roots[dkey], sub)
            for split in ("train", "val"):
                for p in sorted(glob.glob(os.path.join(base, split, pat))):
                    for r in rd1.read_jsonl(p):
                        rec, _ = rd1.clean_record(r, name, spec, cf.is_noncommercial)
                        if rec is None:
                            continue
                        k = f"{name}|{split}|{rec['id']}"
                        if k in seen:
                            ndup += 1
                            continue
                        seen.add(k)
                        out.append((k, rec["state"], rd1.candidate_probes(rec), frozenset(rd1.entity_keys(rec))))
    log(f"lean candidates: {len(out):,} ({ndup} duplicate keys dropped; {time.time() - t0:.0f}s; {rss()})")
    return out


_L = {}


def _lean_chunk(bounds):
    """build_jev_rd1._check_chunk with the entity keys precomputed (entity_keys(r) is a pure function of the record)."""
    lo, hi = bounds
    gate, ng, x2 = _L["gate"], _L["ng"], _L["x2"]
    doc_tags, ent_index, cands = _L["doc_tags"], _L["ent_index"], _L["cands"]
    out = {}
    for j in range(lo, hi):
        k, st, probes, ekeys = cands[j]
        h = []
        for i, kind, nsh, smp in gate.check(st, probes):
            if kind in rd1.DROP_KINDS:
                h.append((kind, sorted(doc_tags[i]), {"n_shared": nsh, "sample": smp}))
        flat, _ = x2.flatten(st)
        c13 = ng.check(norm(flat))
        if c13:
            h.append(("ngram13", sorted(doc_tags[c13[0]]), {"frac": round(c13[1], 3), "n_shared": c13[2]}))
        for key in sorted(ekeys):
            if key in ent_index:
                h.append(("entity", sorted(ent_index[key]), {"key": key}))
        if h:
            out[k] = h
    return out


def _child_init():
    import gc
    gc.disable()


def lean_run_gate(cands, evals, a):
    """build_jev_rd1.run_gate step for step (same index, same candidate-side counts, same check, same chunking) over lean
    candidates; gc.freeze() before the fork and no GC in the workers, so they do not copy the parent's heap."""
    import gc
    import multiprocessing as mp
    ga = rd1._load("build_gde1_a", HERE / "build_gde1_a.py")
    import build_jev_ext2 as x2
    gate = rd1.fast_gate_class(ga, x2)(max_df=a.max_df, min_contain=a.min_contain, max_train_df=a.max_train_df,
                                       min_shared=a.min_shared, min_field_words=a.min_field_words)
    ng = rd1.Ngram13(max_df=a.max_df, max_train_df=a.max_train_df, frac=a.ngram_frac)
    doc_of, doc_tags, ent_index = {}, [], defaultdict(set)
    tag_items, failed = Counter(), []
    t0 = time.time()

    def add_doc(tag, state, fields, ekeys):
        if state is None:
            return
        flat, _ = x2.flatten(state)
        nf = norm(flat)
        key = hashlib.sha1((nf + "\x00" + "\x01".join(norm(f) for f in fields)).encode()).hexdigest()
        tag_items[tag] += 1
        i = doc_of.get(key)
        if i is None:
            i = len(doc_tags)
            doc_of[key] = i
            doc_tags.append({tag})
            gate.add(i, state, fields)
            ng.add(i, nf)
        else:
            doc_tags[i].add(tag)
        for k in ekeys:
            ent_index[k].add(tag)

    for tag, path in evals:
        for r in rd1.read_jsonl(path):
            add_doc(tag, r.get("state"), rd1.eval_fields(r), rd1.entity_keys(r))
        log(f"eval {tag}: {tag_items[tag]} items ({len(doc_tags)} distinct docs so far, {time.time() - t0:.0f}s)")
    if a.fetch_evals:
        for tag, st in rd1.fetched_evals(a.fetch_cache):
            if tag.startswith("FAILED/"):
                failed.append(tag)
                continue
            add_doc(tag, st, [], set())
        for tag in sorted(t for t in tag_items if t.startswith("fetched/")):
            log(f"eval {tag}: {tag_items[tag]} items")
    if failed and not a.allow_missing_evals:
        raise SystemExit(f"eval fetch failed for {failed} (rerun, or pass --allow-missing-evals)")
    log(f"eval index: {len(doc_tags):,} distinct docs from {sum(tag_items.values()):,} items in {len(tag_items)} tags; "
        f"{len(ent_index):,} entity keys; building ...")
    doc_of = None
    gate.freeze()
    ng.freeze()
    log(f"frozen ({time.time() - t0:.0f}s): {len(gate.H):,} shingle entries, {len(ng.H):,} 13-gram entries; {rss()}")
    t1 = time.time()
    n = gate.count_training((k, st, pr) for k, st, pr, _ in cands)
    for _k, st, _p, _e in cands:
        flat, _ = x2.flatten(st)
        ng.count(norm(flat))
    ng.finish_count()
    log(f"candidate-side frequencies over {n:,} candidate states ({time.time() - t1:.0f}s; {rss()})")
    hits = {}
    t1 = time.time()
    _L.update(gate=gate, ng=ng, x2=x2, doc_tags=doc_tags, ent_index=ent_index, cands=cands)
    step = 2000
    chunks = [(i, min(i + step, len(cands))) for i in range(0, len(cands), step)]
    if a.workers > 1:
        gc.collect()
        gc.freeze()
        with mp.get_context("fork").Pool(a.workers, initializer=_child_init) as pool:
            for n_done, part in enumerate(pool.imap(_lean_chunk, chunks), 1):
                hits.update(part)
                if n_done % 25 == 0:
                    log(f"  gate: {chunks[n_done - 1][1]:,} / {len(cands):,} checked ({time.time() - t1:.0f}s); hit "
                        f"{len(hits):,}")
        gc.unfreeze()
    else:
        for n_done, c in enumerate(chunks, 1):
            hits.update(_lean_chunk(c))
            if n_done % 25 == 0:
                log(f"  gate: {c[1]:,} / {len(cands):,} checked ({time.time() - t1:.0f}s); hit {len(hits):,}")
    log(f"gate done: {len(hits):,} of {len(cands):,} candidates hit something ({time.time() - t1:.0f}s; {rss()})")
    return hits, dict(tag_items), failed


def expand_evals(specs):
    """TAG=PATH (a jsonl file or a directory read recursively) -> [(tag/<subdirs>, file)]: one tag per set / split dir."""
    out = []
    for spec in specs:
        tag, _, p = spec.partition("=")
        if os.path.isfile(p):
            out.append((tag, p))
            continue
        for f in sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)):
            if Path(f).name.startswith("._") or "prediction" in Path(f).name:
                continue
            rel = os.path.relpath(os.path.dirname(f), p)
            out.append((tag if rel == "." else f"{tag}/{rel}", f))
    if not out:
        raise SystemExit(f"no eval files under {specs}")
    return out


def group_a_evals(a):
    evals = rd1.discover_evals(a.root)
    tags = sorted({t for t, _ in evals})
    must = ["jev_real/test", "extevals_b/test", "extevals_community/test", "extevals_tdb/test"]
    missing = [t for t in must if t not in tags]
    if missing:
        raise SystemExit(f"required eval sets missing: {missing}")
    return evals


# --------------------------------------------------------------------------- stage: gate
def cmd_gate(a):
    names = [s for s in a.sources.split(",") if s]
    os.environ.setdefault("HF_HOME", os.path.join(a.root, "evals", "hf"))
    a.fetch_cache = a.fetch_cache or os.path.join(a.root, "evals", "ext2_cache")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (the shingle / 13-gram indexes use str hash)")
    log(f"{VERSION} gate group {a.group}: sources {names}; workers {a.workers}")
    cands = load_candidates_lean(a, names)
    if a.group == "A":
        evals = group_a_evals(a)
        if not a.fetch_evals:
            raise SystemExit("group A is rd2's gate: it includes ext2's fetched eval families (drop --no-fetch-evals)")
    else:
        evals = expand_evals(a.eval)
        if a.fetch_evals:
            raise SystemExit("group B indexes only --eval sets: pass --no-fetch-evals")
    tags = sorted({t for t, _ in evals})
    log(f"eval files: {len(evals)} under {len(tags)} tags")
    for t in tags:
        log(f"  eval tag {t}: {sum(1 for x, _ in evals if x == t)} files")
    hits, tag_items, failed = lean_run_gate(cands, evals, a)
    if failed:
        raise SystemExit(f"eval fetch failed: {failed}")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    with gzip.open(work / f"hits_{a.group}.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"group": a.group, "version": VERSION, "n_candidates": len(cands), "eval_items_per_tag": tag_items,
                   "eval_files": evals, "hits": hits}, f, default=str)
    kinds = Counter(x[0] for h in hits.values() for x in h)
    log(f"GATE {a.group}: {len(hits):,} of {len(cands):,} candidates hit; kinds {dict(kinds)}; {rss()}")
    log("DONE")


def load_hits(work, groups):
    out = {}
    for g in groups:
        with gzip.open(Path(work) / f"hits_{g}.json.gz", "rt", encoding="utf-8") as f:
            d = json.load(f)
        out[g] = d
        log(f"hits {g}: {len(d['hits']):,} candidates hit, {sum(d['eval_items_per_tag'].values()):,} eval items in "
            f"{len(d['eval_items_per_tag'])} tags")
    return out


def apply_gates(cands, names, hits_by_group, extra_drop=None):
    """kept {(name, split): [records]} in candidate order; drop report per source (records / units by group, kind, eval set)."""
    extra_drop = extra_drop or {}
    drop_by = defaultdict(lambda: {"records": Counter(), "units": Counter(), "groups": Counter(), "kinds": Counter(),
                                   "eval_sets": Counter(), "entity_key_types": Counter(), "examples": []})
    per_tag = defaultdict(Counter)
    kept = {(n, s): [] for n in names for s in ("train", "val")}
    for k, r in cands:
        name, split, _ = k.split("|", 2)
        h = [(g, x) for g, d in hits_by_group.items() for x in d["hits"].get(k, ())]
        h += [("C", x) for x in extra_drop.get(k, ())]
        if not h:
            kept[(name, split)].append(r)
            continue
        d = drop_by[name]
        d["records"][split] += 1
        d["units"][split] += nq(r)
        for g in sorted({g for g, _ in h}):
            d["groups"][g] += 1
        for kind in sorted({x[0] for _, x in h}):
            d["kinds"][kind] += 1
        for t in sorted({t for _, x in h for t in x[1]}):
            d["eval_sets"][t] += 1
            per_tag[t][f"{name}/{split}"] += 1
        for _, x in h:
            if x[0] == "entity":
                d["entity_key_types"][x[2]["key"].split(":")[0]] += 1
        if len(d["examples"]) < 6:
            d["examples"].append({"id": r["id"], "split": split, "hits": [(g, x[0], x[1][:3], x[2]) for g, x in h[:3]]})
    return kept, drop_by, per_tag


def within_dedupe(kept, names):
    """build_jev_rd1.main's within-corpus dedupe, verbatim: duplicate (state, question set) records in train dropped; a val
    state that occurs in train (or a duplicate val record) dropped from val. Returns the train state set."""
    seen_train, within = set(), Counter()
    for name in names:
        uniq = []
        for r in kept[(name, "train")]:
            key = (norm(r["state"]), tuple(sorted(r["questions"])))
            if key in seen_train:
                within[f"{name}/train_dup"] += 1
                continue
            seen_train.add(key)
            uniq.append(r)
        kept[(name, "train")] = uniq
    train_states = {k[0] for k in seen_train}
    seen_val = set()
    for name in names:
        uniq = []
        for r in kept[(name, "val")]:
            ns = norm(r["state"])
            if ns in train_states or (ns, tuple(sorted(r["questions"]))) in seen_val:
                within[f"{name}/val_dup_or_in_train"] += 1
                continue
            seen_val.add((ns, tuple(sorted(r["questions"]))))
            uniq.append(r)
        kept[(name, "val")] = uniq
    log(f"within-corpus dedup drops: {dict(within)}")
    return within


# --------------------------------------------------------------------------- stage: pool (input of gate C)
def cmd_pool(a):
    names = [s for s in a.sources.split(",") if s]
    cands, _ = load_candidates(a, names)
    hb = load_hits(a.work, a.hits.split(","))
    kept, drop_by, _ = apply_gates(cands, names, hb)
    out = Path(a.work) / "pool"
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)
    keyof = {id(r): k for k, r in cands}
    n = Counter()
    for (name, split), rs in kept.items():
        with open(out / split / f"{name}.jsonl", "w", encoding="utf-8") as f, \
                open(out / split / f"{name}.keys", "w", encoding="utf-8") as g:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                g.write(keyof[id(r)] + "\n")
        n[split] += len(rs)
    log(f"POOL written: {dict(n)} records ({rss()})")
    log("DONE")


def load_di_audit(audit_dir, pool_dir):
    """Gate C: {candidate key: [("di_" + level, ["di/<benchmark>"], {run_id, kinds, contain})]} for every pool record the
    Decision Index overlap audit matched at a DROP level; plus a report of every level."""
    keys = {}
    for p in glob.glob(os.path.join(pool_dir, "*", "*.keys")):
        rel = os.path.relpath(p, pool_dir)[: -len(".keys")] + ".jsonl"
        keys[rel] = open(p, encoding="utf-8").read().splitlines()
    drop, rep = defaultdict(list), {"rows_by_level": Counter(), "records_by_level": defaultdict(set),
                                    "by_benchmark_level": Counter(), "drop_records_by_source": Counter()}
    with gzip.open(os.path.join(audit_dir, "matches.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            k = keys[m["file"]][m["line"]]
            rep["rows_by_level"][m["level"]] += 1
            rep["records_by_level"][m["level"]].add(k)
            rep["by_benchmark_level"][f"{m['benchmark']}|{m['level']}"] += 1
            if m["level"] in DI_DROP_LEVELS:
                drop[k].append(("di_" + m["level"], ["di/" + m["benchmark"]],
                                {"run_id": m["run_id"], "kinds": m["kinds"], "contain": m.get("contain")}))
    for k in drop:
        rep["drop_records_by_source"][k.split("|", 1)[0]] += 1
    rep["records_by_level"] = {lv: len(s) for lv, s in rep["records_by_level"].items()}
    log(f"DI audit (gate C): rows {dict(rep['rows_by_level'])}; records {rep['records_by_level']}; dropping "
        f"{len(drop)} records ({dict(rep['drop_records_by_source'])})")
    return dict(drop), rep


# --------------------------------------------------------------------------- prior corpora (rd1, rd2)
def load_prior(dirs):
    """{(tag, split): ({id: True}, {state hash})} for every <dir>/{train,val}/*.jsonl."""
    out = {}
    for tag, d in dirs.items():
        for split in ("train", "val"):
            ids, sts, n = set(), set(), 0
            for p in sorted(glob.glob(os.path.join(d, split, "*.jsonl"))):
                for r in rd1.read_jsonl(p):
                    n += 1
                    ids.add(str(r["id"]))
                    sts.add(shash(r["state"]))
            out[(tag, split)] = (ids, sts)
            log(f"prior {tag}/{split}: {n:,} records, {len(ids):,} ids, {len(sts):,} states")
    return out


def eval_index(files):
    """ids and normalised-state hashes of every eval record (the exact check), per tag. States of < 3 words (an empty or
    constant probe state) are not indexed: the gate's own exact rule covers them through the question text."""
    ids, sts, per = defaultdict(set), defaultdict(set), Counter()
    for tag, p in files:
        for r in rd1.read_jsonl(p):
            per[tag] += 1
            if "id" in r:
                ids[str(r["id"])].add(tag)
            st = r.get("state")
            if st is not None:
                s = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
                if len(norm(s).split()) >= 3:
                    sts[shash(s)].add(tag)
    return ids, sts, per


# --------------------------------------------------------------------------- stage: build
def draw_pool(pool, quota, spec, seed):
    """rd1.sample_source over `pool` (records with unique states) at `quota` units."""
    if quota <= 0 or not pool:
        return [], {}
    return rd1.sample_source(pool, spec, quota, seed)


def unique_states(rs):
    seen, out, dup = set(), [], 0
    for r in rs:
        h = shash(r["state"])
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        out.append(r)
    return out, dup


def cmd_build(a):
    names = [s for s in a.sources.split(",") if s]
    t0 = time.time()
    log(f"{VERSION} build: caps train {a.train_cap} / val {a.val_cap}; stackpool "
        f"{'OFF' if a.repro_rd2 else f'{a.stack_total} (cap {a.stack_cap}) + val {a.stack_val_total} (cap {a.stack_val_cap})'}")
    cands, rep = load_candidates(a, names)
    groups = a.hits.split(",")
    hb = load_hits(a.work, groups)
    for g, d in hb.items():
        if d["n_candidates"] != len(cands):
            raise SystemExit(f"hits {g} were computed over {d['n_candidates']} candidates, now {len(cands)}")
    di_drop, di_rep = ({}, None)
    if a.di_audit:
        di_drop, di_rep = load_di_audit(a.di_audit, os.path.join(a.work, "pool"))
    if a.repro_rd2 and (groups != ["A"] or di_drop):
        raise SystemExit("--repro-rd2 takes --hits A only and no --di-audit")
    kept, drop_by, per_tag = apply_gates(cands, names, hb, di_drop)
    avail_after_gates = {f"{n}|{s}": sum(nq(r) for r in kept[(n, s)]) for n in names for s in ("train", "val")}
    within = within_dedupe(kept, names)

    # ---- the exact id / state check against every eval FILE of the gates (fetched families are states-only: gate A)
    exact_drop = Counter()
    eval_files = [tuple(x) for g in groups for x in hb[g]["eval_files"]]
    if not a.repro_rd2:
        e_ids, e_sts, _ = eval_index(eval_files)
        for key in list(kept):
            k2 = []
            for r in kept[key]:
                hit = shash(r["state"]) in e_sts
                if hit:
                    exact_drop[f"{key[0]}/{key[1]}"] += 1
                    continue
                k2.append(r)
            kept[key] = k2
        id_hits = Counter(f"{n}/{s}" for (n, s), rs in kept.items() for r in rs if str(r["id"]) in e_ids)
        log(f"exact check vs {len(eval_files)} eval files: state hits dropped {dict(exact_drop)}; id-only coincidences "
            f"(kept, reported) {dict(id_hits)}")
    else:
        id_hits = Counter()

    # ---- prior corpora
    prior = load_prior({"rd1": a.rd1, "rd2": a.rd2}) if not a.repro_rd2 else {}
    p_ids = set().union(*(v[0] for v in prior.values())) if prior else set()
    p_sts = set().union(*(v[1] for v in prior.values())) if prior else set()

    def in_prior(r, tag):
        i = str(r["id"])
        for sp in ("train", "val"):
            if i in prior[(tag, sp)][0]:
                return sp
        return None

    out = Path(a.out)
    rd3_dir = out if a.repro_rd2 else out / "jev_rd3"
    sp_dir = out / "jev_rd3_stackpool"
    for d in ([rd3_dir] if a.repro_rd2 else [rd3_dir, sp_dir]):
        for split in ("train", "val"):
            (d / split).mkdir(parents=True, exist_ok=True)
            for old in (d / split).glob("*.jsonl"):
                old.unlink()

    # ---- 1. STACKPOOL train (before rd3 train is sampled, so capped sources still fill their cap)
    sp = {"train": {}, "val": {}}
    sp_rep = {"train": {}, "val": {}}
    sp_states = set()
    if not a.repro_rd2:
        elig = {}
        for name in names:
            e = [r for r in kept[(name, "train")] if str(r["id"]) not in p_ids and shash(r["state"]) not in p_sts]
            e, dup = unique_states(e)
            elig[name] = e
            sp_rep["train"][name] = {"pool_units": sum(nq(r) for r in kept[(name, "train")]),
                                     "eligible_units": sum(nq(r) for r in e), "eligible_records": len(e),
                                     "eligible_dup_states_dropped": dup}
        avail = {n: min(sp_rep["train"][n]["eligible_units"], a.stack_cap) for n in names}
        quota = rd1.water_fill(a.stack_total, avail, {n: 1.0 for n in names})
        for name in names:
            ch, alloc = draw_pool(elig[name], quota[name], rd1.SPECS[name], a.seed + STACK_SEED_OFFSET)
            sp["train"][name] = ch
            sp_states |= {shash(r["state"]) for r in ch}
            sp_rep["train"][name].update({"quota_units": quota[name], "units": sum(nq(r) for r in ch), "records": len(ch),
                                          "families": alloc})
        chosen_ids = {id(r) for rs in sp["train"].values() for r in rs}
        rm = Counter()
        for name in names:
            k2 = []
            for r in kept[(name, "train")]:
                if id(r) in chosen_ids:
                    continue
                if shash(r["state"]) in sp_states:
                    rm[name] += 1
                    continue
                k2.append(r)
            kept[(name, "train")] = k2
            sp_rep["train"][name]["same_state_records_removed_from_train"] = rm[name]
        log(f"STACKPOOL train: {sum(v['units'] for v in sp_rep['train'].values()):,} units / "
            f"{sum(v['records'] for v in sp_rep['train'].values()):,} records; eligible "
            f"{ {n: v['eligible_units'] for n, v in sp_rep['train'].items() if v['eligible_units']} }; extra same-state "
            f"train records removed {dict(rm)}")

    # ---- 2-3. rd3 train / val (rd1.main's sampling + annotation, verbatim)
    manifest = {"version": VERSION, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": a.seed,
                "caps": {"train_units": a.train_cap, "val_units": a.val_cap},
                "gate": {"groups": {g: {"eval_items_per_tag": hb[g]["eval_items_per_tag"],
                                        "n_eval_files": len(hb[g]["eval_files"])} for g in groups},
                         "kinds": rd1.DROP_KINDS, "max_df": a.max_df, "max_train_df": a.max_train_df,
                         "min_shared": a.min_shared, "min_contain": a.min_contain, "min_field_words": a.min_field_words,
                         "ngram13_frac": a.ngram_frac, "di_audit": di_rep, "di_drop_levels": DI_DROP_LEVELS,
                         "exact_check_dropped": dict(exact_drop), "exact_check_id_coincidences_kept": dict(id_hits)},
                "within_dedup": dict(within), "sources": {}, "totals": {}}
    tot = Counter()
    chosen_all = {"train": {}, "val": {}}
    for name in names:
        spec = rd1.SPECS[name]
        srep = {"files": rep[name]["files"], "records_read": dict(rep[name]["records_read"]),
                "question_drops": dict(rep[name]["question_drops"]),
                "available_units_after_gates": {s: avail_after_gates[f"{name}|{s}"] for s in ("train", "val")}}
        for split, cap in (("train", a.train_cap), ("val", a.val_cap)):
            pool = kept[(name, split)]
            chosen, alloc = rd1.sample_source(pool, spec, cap, a.seed + (0 if split == "train" else 1))
            for r in chosen:
                r["meta"] = dict(r["meta"], rd1={"source": name, "split": split, "family": spec["family"](r),
                                                 "date": (spec["date"] or (lambda _r: None))(r)})
                if not a.repro_rd2:
                    r["meta"]["rd3"] = {"part": split, "in_rd1": in_prior(r, "rd1"), "in_rd2": in_prior(r, "rd2")}
            chosen_all[split][name] = chosen
            units = sum(nq(r) for r in chosen)
            dates = sorted(d for d in ((spec["date"] or (lambda _r: None))(r) for r in chosen) if d)
            srep[split] = {"records": len(chosen), "units": units, "cap_units": cap,
                           "available_units_after_gate": sum(nq(r) for r in pool),
                           "available_records_after_gate": len(pool), "cap_applied": units < sum(nq(r) for r in pool),
                           "families": alloc,
                           "label_source_units": dict(Counter(c for r in chosen for c in r["label_source"].values())),
                           "label_kind_units": dict(Counter(k for r in chosen for k in r["label_kind"].values())),
                           "qtype_units": dict(Counter(q["type"] for r in chosen for q in r["questions"].values())),
                           "time_range": [dates[0], dates[-1]] if dates else None,
                           "median_date": dates[len(dates) // 2] if dates else None,
                           "licenses": dict(Counter(r["license"] for r in chosen).most_common())}
            if not a.repro_rd2:
                srep[split]["units_in_rd2_" + split] = sum(nq(r) for r in chosen if in_prior(r, "rd2") == split)
                srep[split]["units_new_vs_rd2"] = sum(nq(r) for r in chosen if in_prior(r, "rd2") is None)
            tot[f"{split}_units"] += units
            tot[f"{split}_records"] += len(chosen)
        d = drop_by.get(name)
        srep["gate_drops"] = {"records": dict(d["records"]), "units": dict(d["units"]), "by_group_records": dict(d["groups"]),
                              "by_kind_records": dict(d["kinds"]),
                              "by_eval_set_records": dict(d["eval_sets"].most_common()),
                              "entity_key_types": dict(d["entity_key_types"]), "examples": d["examples"]} if d else {}
        manifest["sources"][name] = srep

    # ---- 4. STACKPOOL val (from the val records rd3 val did not take)
    if not a.repro_rd2:
        taken_val = {id(r) for rs in chosen_all["val"].values() for r in rs}
        v_sts = {shash(r["state"]) for rs in chosen_all["val"].values() for r in rs}
        elig = {}
        for name in names:
            e = [r for r in kept[(name, "val")] if id(r) not in taken_val and str(r["id"]) not in p_ids
                 and shash(r["state"]) not in p_sts and shash(r["state"]) not in sp_states
                 and shash(r["state"]) not in v_sts]
            e, dup = unique_states(e)
            elig[name] = e
            sp_rep["val"][name] = {"eligible_units": sum(nq(r) for r in e), "eligible_records": len(e),
                                   "eligible_dup_states_dropped": dup}
        avail = {n: min(sp_rep["val"][n]["eligible_units"], a.stack_val_cap) for n in names}
        quota = rd1.water_fill(a.stack_val_total, avail, {n: 1.0 for n in names})
        for name in names:
            ch, alloc = draw_pool(elig[name], quota[name], rd1.SPECS[name], a.seed + STACK_SEED_OFFSET + 1)
            sp["val"][name] = ch
            sp_rep["val"][name].update({"quota_units": quota[name], "units": sum(nq(r) for r in ch), "records": len(ch),
                                        "families": alloc})
        for split in ("train", "val"):
            for name in names:
                spec = rd1.SPECS[name]
                for r in sp[split][name]:
                    r["meta"] = dict(r["meta"], rd1={"source": name, "split": split, "family": spec["family"](r),
                                                     "date": (spec["date"] or (lambda _r: None))(r)},
                                     rd3={"part": "stack_" + split, "in_rd1": None, "in_rd2": None})
                ds = sorted(d for d in ((spec["date"] or (lambda _r: None))(r) for r in sp[split][name]) if d)
                sp_rep[split][name]["time_range"] = [ds[0], ds[-1]] if ds else None
                sp_rep[split][name]["median_date"] = ds[len(ds) // 2] if ds else None
                sp_rep[split][name]["label_source_units"] = dict(Counter(c for r in sp[split][name]
                                                                         for c in r["label_source"].values()))
        log(f"STACKPOOL val: {sum(v['units'] for v in sp_rep['val'].values()):,} units / "
            f"{sum(v['records'] for v in sp_rep['val'].values()):,} records")

    # ---- checks (asserted)
    checks = {}
    if not a.repro_rd2:
        def idset(d):
            return {str(r["id"]) for rs in d.values() for r in rs}

        def stset(d):
            return {shash(r["state"]) for rs in d.values() for r in rs}
        T, V, ST, SV = chosen_all["train"], chosen_all["val"], sp["train"], sp["val"]
        sp_ids, sp_sts = idset(ST) | idset(SV), stset(ST) | stset(SV)
        checks = {
            "rd3_train_x_stackpool_ids": len(idset(T) & sp_ids), "rd3_train_x_stackpool_states": len(stset(T) & sp_sts),
            "rd3_val_x_stackpool_ids": len(idset(V) & sp_ids), "rd3_val_x_stackpool_states": len(stset(V) & sp_sts),
            "stackpool_x_prior_ids": len(sp_ids & p_ids), "stackpool_x_prior_states": len(sp_sts & p_sts),
            "rd3_train_x_rd1rd2_val_ids": len(idset(T) & (prior[("rd1", "val")][0] | prior[("rd2", "val")][0])),
            "rd3_train_x_rd1rd2_val_states": len(stset(T) & (prior[("rd1", "val")][1] | prior[("rd2", "val")][1])),
            "rd3_val_x_rd3_train_states": len(stset(V) & stset(T)),
            "stack_train_x_stack_val_states": len(stset(ST) & stset(SV)),
            "stackpool_dup_states": sum(len(rs) for rs in list(ST.values()) + list(SV.values())) - len(sp_sts),
        }
        bad = {k: v for k, v in checks.items() if v}
        rd2v = prior[("rd2", "val")][0]
        checks["rd2_val_records"] = len(rd2v)
        checks["rd2_val_in_rd3_val"] = len(rd2v & idset(V))
        checks["rd2_val_in_rd3_train"] = len(rd2v & idset(T))
        checks["rd2_train_records"] = len(prior[("rd2", "train")][0])
        checks["rd2_train_in_rd3_train"] = len(prior[("rd2", "train")][0] & idset(T))
        log(f"CHECKS {checks}")
        if bad or checks["rd2_val_in_rd3_train"]:
            raise SystemExit(f"CHECKS FAILED: {bad}")

    # ---- write
    def write(d, split, name, rs):
        with open(d / split / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ("train", "val"):
        for name in names:
            write(rd3_dir, split, name, chosen_all[split][name])
            if not a.repro_rd2 and sp[split][name]:
                write(sp_dir, split, name, sp[split][name])
    manifest["totals"] = dict(tot)
    manifest["checks"] = checks
    (rd3_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, ensure_ascii=False))
    (rd3_dir / "gate_report.json").write_text(json.dumps({"per_eval_set": {t: dict(c) for t, c in sorted(per_tag.items())},
                                                          "eval_items_per_tag": {g: hb[g]["eval_items_per_tag"] for g in groups}},
                                                         indent=1, default=str))
    log(f"TOTAL rd3 train {tot['train_units']:,} units / {tot['train_records']:,} records; val {tot['val_units']:,} / "
        f"{tot['val_records']:,} ({time.time() - t0:.0f}s; {rss()})")
    for name in names:
        s = manifest["sources"][name]
        log(f"  {name:14s} train {s['train']['units']:7d} u (avail {s['train']['available_units_after_gate']:7d}) val "
            f"{s['val']['units']:5d}; gate drops {s['gate_drops'].get('records', {})} by group "
            f"{s['gate_drops'].get('by_group_records', {})}"
            + ("" if a.repro_rd2 else f"; stack {sp_rep['train'][name]['units']} / {sp_rep['val'][name]['units']}"))
    if not a.repro_rd2:
        spm = {"version": VERSION, "built_utc": manifest["built_utc"], "seed_train": a.seed + STACK_SEED_OFFSET,
               "seed_val": a.seed + STACK_SEED_OFFSET + 1,
               "rule": {"train": f"TRAIN-split units of the gated pool whose record id and normalised state are in neither "
                                 f"rd1 nor rd2 (train or val); {a.stack_total} units water-filled over sources with equal "
                                 f"weight, at most {a.stack_cap} per source; rd1 sample_source inside a source (families "
                                 f"sqrt-share, label strata proportional, newest first); removed from rd3 train with every "
                                 f"train record sharing a state",
                        "val": f"VAL-split units left after rd3 val was sampled, outside rd1 / rd2 / rd3 val (ids, states) "
                               f"and the stackpool train states; {a.stack_val_total} units, at most {a.stack_val_cap} per "
                               f"source; same draw"},
               "per_source": sp_rep,
               "totals": {s: {"units": sum(v["units"] for v in sp_rep[s].values()),
                              "records": sum(v["records"] for v in sp_rep[s].values())} for s in ("train", "val")},
               "checks": checks}
        (sp_dir / "manifest.json").write_text(json.dumps(spm, indent=1, default=str, ensure_ascii=False))
    if a.validate_after:
        manifest["validation"] = rd1.validate(str(rd3_dir), a.tokenizer, a.max_len)
        (rd3_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, ensure_ascii=False))
        if not a.repro_rd2:
            spm["validation"] = rd1.validate(str(sp_dir), a.tokenizer, a.max_len)
            (sp_dir / "manifest.json").write_text(json.dumps(spm, indent=1, default=str, ensure_ascii=False))
    log(f"DONE ({rss()})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("gate", "pool", "build", "validate"))
    ap.add_argument("--root", default="/data/local/rd3/root")
    ap.add_argument("--work", default="/data/local/rd3/work")
    ap.add_argument("--out", default="/data/local/rd3/out")
    ap.add_argument("--sources", default=",".join(rd1.SPECS))
    ap.add_argument("--group", choices=("A", "B"), default="A")
    ap.add_argument("--hits", default="A,B", help="comma list of gate groups to apply")
    ap.add_argument("--di-audit", default=None, help="claude_di_overlap.py --out dir run over <work>/pool (gate C)")
    ap.add_argument("--repro-rd2", action="store_true", help="hits A only, no stackpool / meta.rd3: must equal jev_rd2")
    ap.add_argument("--rd1", default="/data/local/rd3/ref/jev_rd1")
    ap.add_argument("--rd2", default="/data/local/rd3/ref/jev_rd2")
    ap.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    ap.add_argument("--val-cap", type=int, default=VAL_CAP)
    ap.add_argument("--stack-total", type=int, default=STACK_TOTAL)
    ap.add_argument("--stack-cap", type=int, default=STACK_CAP)
    ap.add_argument("--stack-val-total", type=int, default=STACK_VAL_TOTAL)
    ap.add_argument("--stack-val-cap", type=int, default=STACK_VAL_CAP)
    ap.add_argument("--seed", type=int, default=rd1.SEED)
    ap.add_argument("--eval", action="append", default=[], help="group B: TAG=PATH (jsonl file or dir)")
    ap.add_argument("--no-fetch-evals", dest="fetch_evals", action="store_false")
    ap.add_argument("--fetch-cache", default=None)
    ap.add_argument("--allow-missing-evals", action="store_true")
    ap.add_argument("--max-df", type=int, default=20)
    ap.add_argument("--max-train-df", type=int, default=10)
    ap.add_argument("--min-shared", type=int, default=3)
    ap.add_argument("--min-contain", type=float, default=0.10)
    ap.add_argument("--min-field-words", type=int, default=8)
    ap.add_argument("--ngram-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--validate-after", action="store_true")
    ap.add_argument("--dir", default=None, help="validate: corpus dir")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    if a.stage == "gate":
        cmd_gate(a)
    elif a.stage == "pool":
        cmd_pool(a)
    elif a.stage == "build":
        cmd_build(a)
    else:
        rd1.validate(a.dir, a.tokenizer, a.max_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
