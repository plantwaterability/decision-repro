#!/usr/bin/env python3
"""
build_jev_rd1.py -- jev_rd1, the first mixed REAL-DATA training corpus: the training splits of the real-data builders
(research/*_pipeline.md), cleaned to the label sources the project trains on, gated against EVERY evaluation set, capped
per source and written as one flat jsonl per source (the trainer reads ONE directory level).

SOURCES (training splits only; every builder's own eval / test / probe splits are EVAL items here, never candidates):
  openreview   ICLR 2024 + 2025 reviewer ratings (votes)              research/openreview_pipeline.md
  courts       federal appeals (FJC IDB + CourtListener) + state supreme courts (SSCDP)   courts_pipeline.md
  genomics     ClinVar (submitter votes) + CIViC (expert curation)     genomics_pipeline.md
  physics      LVK GW alerts, USGS PAGER, SPC mesoscale discussions    physics_pipeline.md
  fda          openFDA recall enforcement (food / drug / device)       fda_dmhc_pipeline.md
  dmhc         California DMHC independent medical review               fda_dmhc_pipeline.md (USER approved 10-01)
  safety       OSHA SIR (coded / severity), OSHA enforcement, NTSB aviation, NHTSA investigations   safety_pipeline.md
  afd, aidev, swe_rebench, swiss                                        outcomes2_pipeline.md
  lichess, folktables, esci                                             misc2_pipeline.md
  ctgov, nvd_cwe                                                        rawsrc2_pipeline.md
  wd_classify  (Wikidata shape 4), wikidata_s1 (office_end, athlete_team, org_dissolved, iucn_status),
  wikidata_s2  (rfd, property_proposal)                                 wikidata_pipeline.md; shapes 1-2 from wikidata_A_v1
  openjev, wiki_census  (jev_ext4: Open-Jev rule-computed controls, Wikipedia census paragraphs; Spark-234K stays OFF)
  NOT used: wd_compare (optional, low priority: skipped), Spark-234K (model-written answers), every teacher-filled question.

CLEANING (per record, before the gate):
  * label_source: every kept question gets a top-level `label_source[qid]` in {outcome, votes, expert, exact, rule}
    (the builder's raw tag stays in meta.label_source). Dropped questions: any teacher / LLM tag (teacher_verbal:*) and
    the teacher qids (hazard_kind, evidence_strength_for_request, harm_if_delayed, hazard_controllability,
    recurrence_likelihood); USGS PAGER `published_probability:usgs_pager_floor` (a fixed default vector, not a probability).
    published / expert probabilities (LVK p_astro / em_bright, USGS PAGER, SPC forecaster) map to `expert`.
  * OpenReview: labels <- meta.labels_smoothed (Dirichlet-multinomial posterior; the pipeline doc says never fit the raw
    3-6-reviewer shares, 41-48% exact zeros); raw shares kept in meta.labels_raw.
  * a record whose licence string matches cache_features.is_noncommercial is dropped (never rewritten).
  * a record with no question left is dropped. Demographic / SDOH fields are kept exactly as the builders kept them.

GATE (a training or val record is dropped if ANY of these hits ANY eval item):
  exact / leaf / shingle   build_gde1_a.Gate turned the training way round (eval items indexed, every candidate streamed
                           through it): claude_overlap.norm whole-state equality; an eval field (JSON leaf, conversation
                           turn, question instruction / option text) of >= 8 words equal to a candidate state / line /
                           instruction; >= --min-shared (3, ext2's training-side rule) shared 8-word shingles with ONE eval
                           field covering >= 10% of the shorter, ignoring boilerplate shingles (held by > 20 eval items or
                           by > 10 candidate states -- the builders' templates).
  ngram13                  >= 50% of the candidate's word 13-grams in ONE eval state (synth_workflows/exclusion.py, the od1
                           audit), counting only non-boilerplate 13-grams (same two document-frequency limits).
  entity                   a named entity key shared with any eval item (CVE, NCT, QID, enwiki page id, SWE instance,
                           AIDev PR, Swiss decision, DMHC reference, FDA recall / event, safety source id, court case /
                           docket, OpenReview forum / group, ClinVar variation, CIViC evidence / citation (PMID), physics
                           event, Lichess puzzle / game / FEN, ESCI query / product, ACS household, GDE-1 item_key ...).
  Eval items: jev_real (test; train / val reported separately), extevals b / community / tdb, rw_eval_v1 (67), the (67b)
  FDA / DMHC (+ eval_cf) / safety evals, every GDE-1 set (laptop v3 superset, gde1_v2 (+gated), gde1_outcomes2, Wikidata
  A_v1 and B copies, eval + probe + eval_name_only + eval_state_removed + extended + dev + overlap), the held-out val
  splits the race scores (jev_data val / test, jev_ext_strict / ext2 / ext3 val), and ext2's fetched eval families
  (jev-bench, typed-decisions train+test, TDB + its withheld source tables, MMLU / ARC, LegalBench, PhishNChips / community,
  DICES, ChaosNLI). Question text (instructions, options) of eval items is indexed as fields, so probe splits whose state
  is constant are still covered. Within rd1: duplicate (state, questions) records dropped; a val state that also occurs
  in train is dropped from val.

SAMPLING (deterministic, --seed): per source, families (sub-sources / question signatures) get a sqrt-proportional share
of the cap (water-filled); inside a family the primary question's argmax label strata get a PROPORTIONAL share (base
rates preserved); inside a stratum the NEWEST records come first (the builder's decision / event / publication date;
seeded hash order where a source has no date). Caps: --train-cap 4000 / --val-cap 250 units per source.

OUTPUT: <out>/{train,val}/<source>.jsonl, <out>/manifest.json (per source: units, records, licences, label_source mix,
time range, cap, gate drops by kind / eval set / entity key), <out>/gate_report.json (per eval set, examples).
VALIDATION (--validate DIR, also run after a build): trainers/cache_features.iter_units over train and val, the
trainer's own filters (claude_lmtune.py: is_noncommercial, K <= len(LABELS)), per-source unit counts as the loader sees
them, an assert that zero units are non-commercial, and (with --tokenizer) claude_lmtune.make_example on EVERY unit.

Node layout used for the 2026-10-02 build (cpu2, /data/shared/rd1 = --root):
  s3src/x_<name>/   = s3://YOUR_BUCKET/jev-race/data/<name>.tgz extracted (laptop_data_2026-10-01,
                      wikidata_A_v1, wikidata_jev_B_final_2026-10-01, outcomes2_jev, realdata_fda_dmhc_v1,
                      realdata_safety_v1, jev_ext4, rw_eval_v1, jev_data_ref, gde1_v2, gde1_v2_gated, gde1_outcomes2,
                      jev_ext_strict, jev_ext2, jev_ext3)
  laptop/           = laptop /tmp/{openreview,courts,genomics,physics}_jev train + val (md5-checked: every other laptop
                      file used here is byte-identical to its S3 copy)
  evals/jev_real    = trainers/convert_jevbench.py --out evals/jev_real
  evals/extevals/X  = trainers/convert_extevals.py --cache evals/cache X --out evals/extevals/X   (X = b, community, tdb)

  PYTHONHASHSEED=0 python3 build_jev_rd1.py --root /data/shared/rd1 --out /data/shared/rd1/out/jev_rd1 \\
      --tokenizer Qwen/Qwen3.8-27B
  python3 build_jev_rd1.py --validate /data/shared/rd1/out/jev_rd1 --tokenizer Qwen/Qwen3.8-27B
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

from claude_overlap import norm  # noqa: E402  the eval-overlap normalization, reused verbatim

VERSION = "rd1 1.0 (2026-10-02)"
SEED = 20261002
TRAIN_CAP, VAL_CAP = 4000, 250
LABEL_SOURCES = ("outcome", "votes", "expert", "exact", "rule")
TEACHER_QIDS = {"hazard_kind", "evidence_strength_for_request", "harm_if_delayed", "hazard_controllability",
                "recurrence_likelihood"}
DROP_KINDS = ("exact", "leaf", "shingle", "ngram13", "entity")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rhash(seed, s):
    return hashlib.sha1(f"{seed}|{s}".encode()).hexdigest()


# --------------------------------------------------------------------------- sources
def m_(r):
    m = r.get("meta")
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except ValueError:
            m = {}
    return m if isinstance(m, dict) else {}


def d_meta(field):
    return lambda r: (str(m_(r).get(field))[:19] if m_(r).get(field) not in (None, "", "None") else None)


def d_openreview(r):
    return str(r.get("source", ""))[-4:]          # openreview/iclr2025 -> '2025'


def d_dmhc(r):
    m = m_(r)
    y = m.get("report_year")
    rank = (m.get("decision_time_estimate") or {}).get("within_year_sequence_rank")
    if y is None:
        return None
    return f"{int(y):04d}.{int(float(rank or 0) * 1e6):07d}"


def fam_source(r):
    return str(r.get("source") or "?")


def fam_courts(r):
    return "federal" if str(r.get("source", "")).startswith("courts/ca") else "state"


def fam_one(r):
    return "all"


# name -> spec. dirs: [(dir key, subdir, file glob)]; default_ls: label_source for questions whose builder wrote none.
SPECS = {
    "openreview": dict(dirs=[("laptop", "openreview_jev", "*.jsonl")], family=fam_one, date=d_openreview),
    "courts": dict(dirs=[("laptop", "courts_jev", "*.jsonl")], family=fam_courts, date=d_meta("decision_date")),
    "genomics": dict(dirs=[("laptop", "genomics_jev", "*.jsonl")], family=fam_source, date=None),
    "physics": dict(dirs=[("laptop", "physics_jev", "*.jsonl")], family=fam_source, date=d_meta("event_day")),
    "fda": dict(dirs=[("s3", "x_realdata_fda_dmhc_v1/fda_jev", "*.jsonl")], family=fam_source,
                date=d_meta("decision_date")),
    "dmhc": dict(dirs=[("s3", "x_realdata_fda_dmhc_v1/dmhc_jev", "*.jsonl")], family=fam_one, date=d_dmhc),
    "safety": dict(dirs=[("s3", "x_realdata_safety_v1/safety_jev", "*.jsonl")], family=fam_source,
                   date=d_meta("decision_date")),
    "afd": dict(dirs=[("s3", "x_laptop_data_2026-10-01/outcomes2_jev/afd", "*.jsonl")], family=fam_source,
                date=d_meta("decision_date")),
    "aidev": dict(dirs=[("s3", "x_laptop_data_2026-10-01/outcomes2_jev/aidev", "*.jsonl")], family=fam_one,
                  date=d_meta("decision_date")),
    "swe_rebench": dict(dirs=[("s3", "x_laptop_data_2026-10-01/outcomes2_jev/swe_rebench", "*.jsonl")], family=fam_one,
                        date=d_meta("info_date")),          # decision_date is the constant dataset release date
    "swiss": dict(dirs=[("s3", "x_laptop_data_2026-10-01/outcomes2_jev/swiss", "*.jsonl")], family=fam_one,
                  date=d_meta("decision_date")),
    "lichess": dict(dirs=[("s3", "x_laptop_data_2026-10-01/misc2_jev/lichess", "*.jsonl")], family=fam_one, date=None,
                    default_ls="rule"),
    "folktables": dict(dirs=[("s3", "x_laptop_data_2026-10-01/misc2_jev/folktables", "*.jsonl")], family=fam_source,
                       date=None, default_ls="outcome"),
    "esci": dict(dirs=[("s3", "x_laptop_data_2026-10-01/misc2_jev/esci", "*.jsonl")], family=fam_one, date=None,
                 default_ls="votes"),
    "ctgov": dict(dirs=[("s3", "x_laptop_data_2026-10-01/rawsrc2_jev/ctgov", "*.jsonl")], family=fam_source,
                  date=d_meta("decision_date")),
    "nvd_cwe": dict(dirs=[("s3", "x_laptop_data_2026-10-01/rawsrc2_jev/nvd_cwe", "*.jsonl")], family=fam_one,
                    date=d_meta("published")),
    "wd_classify": dict(dirs=[("s3", "x_wikidata_jev_B_final_2026-10-01/wikidata_jev/wd_classify", "*.jsonl")],
                        family=fam_one, date=None, default_ls="rule"),
    "wikidata_s1": dict(dirs=[("s3", f"x_wikidata_A_v1/{s}", "*.jsonl") for s in
                              ("s1_office_end", "s1_athlete_team", "s1_org_dissolved", "s1_iucn_status")],
                        family=fam_source, date=d_meta("decision_date"), default_ls="outcome"),
    "wikidata_s2": dict(dirs=[("s3", f"x_wikidata_A_v1/{s}", "*.jsonl") for s in ("s2_rfd", "s2_property_proposal")],
                        family=fam_source, date=d_meta("decision_date"), default_ls="outcome"),
    "openjev": dict(dirs=[("s3", "x_jev_ext4/jev_ext4", "openjev__*.jsonl")], family=fam_source, date=None,
                    default_ls="rule"),
    "wiki_census": dict(dirs=[("s3", "x_jev_ext4/jev_ext4", "wiki__census.jsonl")], family=fam_one,
                        date=d_meta("census_year"), default_ls="exact"),
}
# Open-Jev subs that must never be read (TD-workflow imitations / TD-adjacent; build_jev_ext4 OJ_BLOCKED + --strict)
OPENJEV_BLOCKED = ("mailroom", "retention", "silentfail", "customer_control", "workflow_controls", "spark")


def map_label_source(raw, default):
    """Builder tag -> one of LABEL_SOURCES, or None (drop the question)."""
    if raw in (None, "", "None"):
        return default
    t = str(raw).lower()
    if t.startswith("teacher") or "llm" in t or "verbal" in t:
        return None
    if t == "published_probability:usgs_pager_floor":
        return None
    if t.startswith("votes"):
        return "votes"
    if t.startswith("expert") or t.startswith("published_probability"):
        return "expert"
    if t.startswith("outcome"):
        return "outcome"
    if t.startswith("rule"):
        return "rule"
    if t.startswith("exact"):
        return "exact"
    return None


def clean_record(r, name, spec, is_nc):
    """(record, {drop reason: n questions}) -- record None when nothing is left (reason in the dict)."""
    drops = Counter()
    m = dict(m_(r))
    raw_ls = m.get("label_source")
    keep = []
    for q in r["questions"]:
        if q in TEACHER_QIDS:
            drops["teacher_qid"] += 1
            continue
        raw = raw_ls.get(q) if isinstance(raw_ls, dict) else raw_ls
        cls = map_label_source(raw, spec.get("default_ls"))
        if cls is None:
            drops["label_source:" + str(raw)[:48]] += 1
            continue
        lab = r["labels"].get(q)
        if not isinstance(lab, dict) or not lab:
            drops["no_label"] += 1
            continue
        vals = [float(v) for v in lab.values()]
        if min(vals) < 0 or sum(vals) <= 0 or not all(math.isfinite(v) for v in vals):
            drops["bad_label"] += 1
            continue
        keep.append((q, cls))
    if not keep:
        drops["record_no_questions"] += 1
        return None, drops
    qs = [q for q, _ in keep]
    rec = dict(r)
    if len(qs) < len(r["questions"]):
        allq = set(r["questions"])
        m = {k: ({q: v[q] for q in qs if q in v} if isinstance(v, dict) and set(v) == allq else v) for k, v in m.items()}
        rec["questions"] = {q: r["questions"][q] for q in qs}
        rec["labels"] = {q: r["labels"][q] for q in qs}
        rec["label_kind"] = {q: r["label_kind"][q] for q in qs}
    if name == "openreview":
        sm = m.get("labels_smoothed") or {}
        new = {}
        for q in qs:
            levels = rec["questions"][q].get("levels") or list(rec["labels"][q])
            v = sm.get(q)
            if isinstance(v, list) and len(v) == len(levels) and list(rec["labels"][q]) == list(levels):
                new[q] = {lv: float(x) for lv, x in zip(levels, v)}
            else:
                drops["openreview_no_smoothed"] += 1
                new[q] = rec["labels"][q]
        m["labels_raw"] = {q: rec["labels"][q] for q in qs}
        rec["labels"] = new
        rec["label_kind"] = {q: "soft" for q in qs}
    lic = rec.get("license") or m.get("license") or ""
    if not lic:
        drops["record_no_license"] += 1
        return None, drops
    rec["license"] = lic
    rec["source"] = rec.get("source") or name
    rec["label_source"] = {q: c for q, c in keep}
    rec["meta"] = m
    if is_nc({"source": rec["source"], "license": lic}):
        drops["record_noncommercial"] += 1
        return None, drops
    return rec, drops


# --------------------------------------------------------------------------- entity keys
LIST_KEYS = {"qids": "qid", "items": "qid", "evidence_ids": "civic_ev", "citation_ids": "pmid"}
SCALAR_KEYS = {"cve_id": "cve", "nct_id": "nct", "wikidata_item": "qid", "qid": "qid", "gold_qid": "qid",
               "pageid": "enwiki", "page_id": "enwiki", "instance_id": "swe_instance", "aidev_pr_id": "aidev_pr",
               "html_url": "url", "decision_id": "swiss_decision", "reference_id": "dmhc_ref",
               "recall_number": "fda_recall", "case_key": "court_case", "court_docket": "court_docket",
               "forum_url": "openreview_forum", "variation_id": "clinvar_variation", "usgs_event_id": "usgs_event",
               "puzzle_id": "lichess_puzzle", "game_id": "lichess_game", "fen_after_first_move": "fen",
               "serialno": "acs_household", "query_id": "esci_query", "product_id": "esci_product",
               "item_key": "item_key", "afd_conversation_id": "afd_conversation", "gw_id": "gw"}
# family-namespaced: the same field name means different things in different builders
FAMILY_KEYS = {"event_id": "event", "source_id": "source_id", "repo": "repo", "group": "group"}
GROUP_FAMILIES = {"courts", "physics", "fda_recalls", "dmhc_imr", "osha_sir", "osha_enforcement", "ntsb", "nhtsa",
                  "swiss", "ctgov", "nvd_cwe", "openreview", "wikidata", "wd_classify", "civic", "clinvar"}


def _as_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str) and v[:1] == "[":
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return [v]
    return [v]


def entity_keys(r):
    m = m_(r)
    fam = str(r.get("source") or "").split("/")[0].split("__")[0]
    keys = set()
    for f, ns in SCALAR_KEYS.items():
        v = m.get(f, r.get(f) if f == "item_key" else None)
        if v not in (None, "", "None"):
            keys.add(f"{ns}:{str(v).strip()}")
    for f, ns in LIST_KEYS.items():
        if f in m and m[f] not in (None, "", "None"):
            if f == "citation_ids" and fam != "civic":
                continue
            for x in _as_list(m[f]):
                if x not in (None, "", "None"):
                    keys.add(f"{ns}:{str(x).strip()}")
    cl = m.get("courtlistener")
    if isinstance(cl, dict) and cl.get("docket_id"):
        keys.add(f"cl_docket:{cl['docket_id']}")
    for f, ns in FAMILY_KEYS.items():
        v = m.get(f)
        if v in (None, "", "None"):
            continue
        if f == "group" and fam not in GROUP_FAMILIES:
            continue
        if f == "group" and fam == "wikidata" and "property proposal" not in str(v).lower() and not str(v).startswith("Q"):
            continue
        keys.add(f"{ns}:{fam}:{str(v).strip()}")
    return keys


# --------------------------------------------------------------------------- eval items
EVAL_SPLITS = {"test", "eval", "eval_cf", "test_temporal", "probe", "eval_name_only", "eval_state_removed", "extended",
               "dev", "overlap", "unlabelled"}
HELDOUT_SPLITS = {"val", "test"}
SKIP_PARTS = {"teacher_fill", "cache", "_cache", "wikidata_a_work", "wikidata_b_work", "__pycache__", "results", "hf",
              "ext2_cache", "out"}
SPLIT_NAMES = EVAL_SPLITS | {"train", "val"}


def canon_tag(path, root):
    """A readable eval-set tag for a jsonl path: tarball prefix and duplicate-copy prefixes removed."""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if parts[0] == "s3src":
        parts = parts[2:]                                      # s3src/x_<tarball>/...
    elif parts[0] == "evals":
        parts = parts[1:]
    parts = [p for p in parts[:-1]]
    if parts and parts[0] in ("gde1_outcomes2",):
        parts[0] = "gde1"
    if parts and parts[0] == "wikidata_jev":
        parts = parts[1:]
    if parts and parts[0] == "extevals":
        parts = ["extevals_" + parts[1]] + parts[2:]
    return "/".join(parts)


def discover_evals(root):
    """[(tag, path)] for every eval jsonl under the node layout."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "s3src", "x_*", "**", "*.jsonl"), recursive=True)) + \
            sorted(glob.glob(os.path.join(root, "evals", "**", "*.jsonl"), recursive=True)) + \
            sorted(glob.glob(os.path.join(root, "laptop", "**", "*.jsonl"), recursive=True)):
        parts = Path(p).parts
        if any(x in SKIP_PARTS for x in parts) or "prediction" in Path(p).name or Path(p).name.startswith("samples"):
            continue
        split = next((x for x in reversed(parts[:-1]) if x in SPLIT_NAMES), None)   # nearest split-like dir
        heldout = any(x in parts for x in ("x_jev_data_ref", "x_jev_ext_strict", "x_jev_ext2", "x_jev_ext3"))
        if heldout:
            if split in HELDOUT_SPLITS:
                out.append(("heldout/" + canon_tag(p, root), p))
            continue
        if "jev_real" in parts and split in ("train", "val"):
            out.append((canon_tag(p, root), p))                # typed-decisions + jev-bench train/val: reported apart
            continue
        if split in EVAL_SPLITS:
            out.append((canon_tag(p, root), p))
    return out


def eval_fields(r):
    """Question text of an eval item (instructions, option texts, levels): indexed as fields, so a probe split whose
    state is constant (the entity is in the question) and HealthBench rubric criteria are still covered."""
    out = []
    qs = r.get("questions") or {}
    if isinstance(qs, str):
        try:
            qs = json.loads(qs)
        except ValueError:
            qs = {}
    for q in (qs.values() if isinstance(qs, dict) else []):
        if not isinstance(q, dict):
            continue
        if q.get("instructions"):
            out.append(str(q["instructions"]))
        c = q.get("criteria")
        if isinstance(c, dict):
            out += [str(v) for v in c.values() if v]
        elif isinstance(c, list):
            out += [str(v) for v in c if v]
        out += [str(v) for v in (q.get("levels") or []) if v]
    return out


def fetched_evals(cache):
    """ext2's eval-family fetchers (state / text fields only): yields (tag, state)."""
    import build_jev_ext2 as x2
    a = SimpleNamespace(hf_token=None, cache=cache)
    for fam, fn in x2.EVAL_FETCHERS.items():
        t0, n = time.time(), 0
        try:
            for tag, st in fn(a):
                n += 1
                yield f"fetched/{tag}", st
            log(f"fetched eval family {fam}: {n} states ({time.time() - t0:.0f}s)")
        except Exception as e:  # noqa: BLE001
            log(f"FETCH FAILED {fam}: {type(e).__name__}: {str(e)[:300]}")
            yield f"FAILED/{fam}", None


# --------------------------------------------------------------------------- 13-gram index
class Ngram13:
    """od1's 13-gram rule as an index: eval states' word 13-grams (claude_overlap.norm of the flattened state); a candidate
    hits an eval item when >= frac of ITS 13-grams occur in that item, counting only 13-grams held by <= max_df eval items
    and <= max_train_df candidate states (the builders' templates are in thousands)."""

    def __init__(self, n=13, max_df=20, max_train_df=10, frac=0.5):
        self.n, self.max_df, self.max_train_df, self.frac = n, max_df, max_train_df, frac
        self._h, self._d = [], []

    def hashes(self, nf):
        import build_jev_ext2 as x2
        w = x2.WORD.findall(nf)
        return {hash(" ".join(w[i:i + self.n])) for i in range(len(w) - self.n + 1)}

    def add(self, doc, nf):
        import numpy as np
        hs = self.hashes(nf)
        if hs:
            self._h.append(np.fromiter(hs, dtype=np.int64, count=len(hs)))
            self._d.append(np.full(len(hs), doc, dtype=np.int32))

    def freeze(self):
        import numpy as np
        H = np.concatenate(self._h) if self._h else np.zeros(0, np.int64)
        D = np.concatenate(self._d) if self._d else np.zeros(0, np.int32)
        o = np.lexsort((D, H))
        self.H, self.D = H[o], D[o]
        keep = np.ones(len(self.H), dtype=bool)
        keep[1:] = (self.H[1:] != self.H[:-1]) | (self.D[1:] != self.D[:-1])
        self.H, self.D = self.H[keep], self.D[keep]
        self.U, c = np.unique(self.H, return_counts=True)
        self.DF = c[np.searchsorted(self.U, self.H)]
        self.TDF_U = np.zeros(len(self.U), dtype=np.int64)
        self._h, self._d = [], []

    def count(self, nf):
        import numpy as np
        hs = self.hashes(nf)
        if not hs or not len(self.U):
            return
        k = np.fromiter(hs, dtype=np.int64, count=len(hs))
        pos = np.minimum(np.searchsorted(self.U, k), len(self.U) - 1)
        mm = self.U[pos] == k
        if mm.any():
            self.TDF_U[pos[mm]] += 1

    def finish_count(self):
        import numpy as np
        self.TDF = self.TDF_U[np.searchsorted(self.U, self.H)] if len(self.U) else np.zeros(0, np.int64)

    def check(self, nf):
        """(doc, frac, n_shared) of the best eval item, or None."""
        import numpy as np
        hs = self.hashes(nf)
        if not hs or not len(self.H):
            return None
        k = np.fromiter(hs, dtype=np.int64, count=len(hs))
        lo, hi = np.searchsorted(self.H, k, "left"), np.searchsorted(self.H, k, "right")
        nz = np.nonzero(hi > lo)[0]
        if not len(nz):
            return None
        # DF / TDF are per hash (every entry of a hash carries the same value): filter BEFORE expanding, so a template
        # 13-gram held by thousands of eval items costs nothing
        nz = nz[(self.DF[lo[nz]] <= self.max_df) & (self.TDF[lo[nz]] <= self.max_train_df)]
        if not len(nz):
            return None
        ent = np.concatenate([np.arange(lo[j], hi[j]) for j in nz])
        u, c = np.unique(self.D[ent], return_counts=True)
        j = int(np.argmax(c))
        f = c[j] / len(hs)
        return (int(u[j]), float(f), int(c[j])) if f >= self.frac else None


# --------------------------------------------------------------------------- the gate
def fast_gate_class(ga, x2):
    """build_gde1_a.Gate whose check() applies the calibrated rule's boilerplate filter (eval DF <= max_df, candidate
    DF <= max_train_df) per HASH before expanding the hash's entries. The dropping kinds it returns (exact, leaf,
    calibrated shingle) are exactly Gate.check's: a calibrated hit needs >= min_shared non-boilerplate shingles in one
    field, which implies ext2's plain rule. Only the informational ext2_rule_only kind is not computed. The plain
    Gate.check expands every entry of a template shingle (held by ~10^4 eval items of the same builder) per candidate."""
    import numpy as np

    class FastGate(ga.Gate):
        def check(self, state, probes=()):
            nf, parts = self._texts(state, probes)
            hits = {}
            for p in parts:
                k = hash(p)
                for i in self.whole.get(k, ()):
                    hits[i] = ("exact", None, p[:120])
                for i in self.field.get(k, ()):
                    if i in hits:
                        continue
                    stock = self.field_tdf is not None and self.max_train_df and self.field_tdf[k] > self.max_train_df
                    ok = self.field_words.get(k, 0) >= self.min_field_words and not stock
                    hits[i] = ("leaf" if ok else "short_or_stock_field", None, p[:120])
            hs = x2.shingle_hashes(nf)
            if len(hs) < x2.SHINGLE_MIN_SHARED or not len(self.H):
                return [(i, *v) for i, v in hits.items()]
            keys = np.fromiter(hs, dtype=np.int64, count=len(hs))
            lo, hi = np.searchsorted(self.H, keys, "left"), np.searchsorted(self.H, keys, "right")
            nz = np.nonzero(hi > lo)[0]
            keep = np.ones(len(nz), dtype=bool)
            if self.max_df:
                keep &= self.DF[lo[nz]] <= self.max_df
            if self.max_train_df and self.TDF is not None:
                keep &= self.TDF[lo[nz]] <= self.max_train_df
            nz = nz[keep]
            if len(nz) < self.min_shared:
                return [(i, *v) for i, v in hits.items()]
            lens = hi[nz] - lo[nz]
            ent = np.concatenate([np.arange(lo[j], hi[j]) for j in nz])
            tsh = np.repeat(np.arange(len(nz), dtype=np.int64), lens)
            fids = self.F[ent].astype(np.int64)
            m = len(nz)
            fu, fc = np.unique(np.unique(fids * m + tsh) // m, return_counts=True)
            need = np.maximum(self.min_shared, self.min_contain * np.minimum(self.f_n[fu], len(hs)))
            cal = {}
            for f, n in zip(fu[fc >= need].tolist(), fc[fc >= need].tolist()):
                i = int(self.f_item[f])
                cal[i] = max(cal.get(i, 0), int(n))
            for i, n in cal.items():
                if i in hits and hits[i][0] in ga.DROP_KINDS:
                    continue
                sh = [hs[int(keys[nz[t]])] for t in np.unique(tsh[self.f_item[fids] == i])[:2]]
                hits[i] = ("shingle", n, sh)
            return [(i, *v) for i, v in hits.items()]

    return FastGate


_W = {}   # forked gate workers read the frozen indexes from here (copy-on-write)


def _check_chunk(bounds):
    lo, hi = bounds
    gate, ng, x2 = _W["gate"], _W["ng"], _W["x2"]
    doc_tags, ent_index, cands, stream = _W["doc_tags"], _W["ent_index"], _W["cands"], _W["stream"]
    out = {}
    for j in range(lo, hi):
        k, r = cands[j]
        _k, st, probes = stream[j]
        h = []
        for i, kind, nsh, smp in gate.check(st, probes):
            if kind in DROP_KINDS:
                h.append((kind, sorted(doc_tags[i]), {"n_shared": nsh, "sample": smp}))
        flat, _ = x2.flatten(st)
        c13 = ng.check(norm(flat))
        if c13:
            h.append(("ngram13", sorted(doc_tags[c13[0]]), {"frac": round(c13[1], 3), "n_shared": c13[2]}))
        for key in sorted(entity_keys(r)):
            if key in ent_index:
                h.append(("entity", sorted(ent_index[key]), {"key": key}))
        if h:
            out[k] = h
    return out


def candidate_probes(r):
    """Lines of >= 5 words and every question's instructions (build_gde1_b.probes_of's view)."""
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    lines = [ln.strip() for ln in st.split("\n") if len(ln.split()) >= 5]
    return lines + [q.get("instructions", "") for q in r["questions"].values() if q.get("instructions")]


def run_gate(cands, evals, a):
    """cands: list of (key, record). evals: [(tag, path)] + fetched. Returns {key: [(kind, [eval tags], detail)]},
    per-tag doc counts, eval-key index stats."""
    import numpy as np
    ga = _load("build_gde1_a", HERE / "build_gde1_a.py")
    import build_jev_ext2 as x2
    gate = fast_gate_class(ga, x2)(max_df=a.max_df, min_contain=a.min_contain, max_train_df=a.max_train_df,
                                   min_shared=a.min_shared, min_field_words=a.min_field_words)
    ng = Ngram13(max_df=a.max_df, max_train_df=a.max_train_df, frac=a.ngram_frac)
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
        for r in read_jsonl(path):
            add_doc(tag, r.get("state"), eval_fields(r), entity_keys(r))
        log(f"eval {tag}: {tag_items[tag]} items ({len(doc_tags)} distinct docs so far, {time.time() - t0:.0f}s)")
    if a.fetch_evals:
        for tag, st in fetched_evals(a.fetch_cache):
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
    gate.freeze()
    ng.freeze()
    log(f"frozen ({time.time() - t0:.0f}s): {len(gate.H):,} shingle entries, {len(ng.H):,} 13-gram entries")

    stream = [(k, r["state"], candidate_probes(r)) for k, r in cands]
    t1 = time.time()
    n = gate.count_training(iter(stream))
    for _k, st, _p in stream:
        flat, _ = x2.flatten(st)
        ng.count(norm(flat))
    ng.finish_count()
    log(f"candidate-side frequencies over {n:,} candidate states ({time.time() - t1:.0f}s)")

    hits = {}
    t1 = time.time()
    _W.update(gate=gate, ng=ng, x2=x2, doc_tags=doc_tags, ent_index=ent_index, cands=cands, stream=stream)
    step = 2000
    chunks = [(i, min(i + step, len(cands))) for i in range(0, len(cands), step)]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            for n_done, part in enumerate(pool.imap(_check_chunk, chunks), 1):
                hits.update(part)
                if n_done % 25 == 0:
                    log(f"  gate: {chunks[n_done - 1][1]:,} / {len(cands):,} checked ({time.time() - t1:.0f}s); hit {len(hits):,}")
    else:
        for n_done, c in enumerate(chunks, 1):
            hits.update(_check_chunk(c))
            if n_done % 25 == 0:
                log(f"  gate: {c[1]:,} / {len(cands):,} checked ({time.time() - t1:.0f}s); hit {len(hits):,}")
    log(f"gate done: {len(hits):,} of {len(cands):,} candidates hit something ({time.time() - t1:.0f}s)")
    return hits, dict(tag_items), failed


# --------------------------------------------------------------------------- sampling
def water_fill(budget, avail, weight):
    """Integer shares of `budget` over keys, proportional to weight[k], none above avail[k] (excess redistributed)."""
    out, left, active = {}, budget, {k for k in avail if avail[k] > 0}
    while active and left > 0:
        tot = sum(weight[k] for k in active)
        share = {k: left * weight[k] / tot for k in active}
        full = {k for k in active if share[k] >= avail[k]}
        if full:
            for k in full:
                out[k] = avail[k]
                left -= avail[k]
            active -= full
            continue
        base = {k: int(share[k]) for k in active}
        rem = left - sum(base.values())
        for k in sorted(active, key=lambda k: (-(share[k] - base[k]), str(k)))[:rem]:
            base[k] += 1
        out.update(base)
        left = 0
    for k in avail:
        out.setdefault(k, 0)
    return out


def primary_label(r, by_type=False):
    """Stratum label: the first question's argmax option (qid:option); by_type -> qtype:option-position, for families
    whose question ids are per record (Open-Jev sponsor / painting)."""
    q = next(iter(r["questions"]))
    lab = r["labels"][q]
    keys = list(lab)
    j = max(range(len(keys)), key=lambda i: float(lab[keys[i]]))
    return f"{r['questions'][q]['type']}:{j}" if by_type else f"{q}:{keys[j]}"


def _take(rs, budget, nq, got=0):
    """Records from rs (in order) while adding one brings the unit total closer to budget (multi-question records
    must not overshoot a small stratum several-fold). Returns (taken, units)."""
    out = []
    for r in rs:
        if got >= budget:
            break
        n = nq(r)
        if got + n <= budget or (budget - got) >= (got + n - budget):
            out.append(r)
            got += n
    return out, got


def sample_source(recs, spec, cap, seed):
    """recs: cleaned, gated records of one source and split. Returns (chosen records, allocation report)."""
    nq = lambda r: len(r["questions"])  # noqa: E731
    fam_of = spec["family"]
    date_of = spec["date"] or (lambda r: None)
    order = lambda rs: sorted(sorted(rs, key=lambda r: rhash(seed, r["id"])),  # noqa: E731
                              key=lambda r: date_of(r) or "", reverse=True)   # newest first; seeded hash inside a date
    by_fam = defaultdict(list)
    for r in recs:
        by_fam[fam_of(r)].append(r)
    avail = {f: sum(nq(r) for r in rs) for f, rs in by_fam.items()}
    fam_budget = water_fill(cap, avail, {f: math.sqrt(v) for f, v in avail.items()})
    chosen, rep = [], {}
    for f in sorted(by_fam):
        by_type = len({next(iter(r["questions"])) for r in by_fam[f]}) > 20
        cells = defaultdict(list)
        for r in by_fam[f]:
            cells[primary_label(r, by_type)].append(r)
        cav = {c: sum(nq(r) for r in rs) for c, rs in cells.items()}
        cb = water_fill(fam_budget[f], cav, cav)            # proportional: base rates preserved
        taken, got, ids = [], 0, set()
        for c in sorted(cells):
            t, u = _take(order(cells[c]), cb[c], nq)
            taken += t
            got += u
        ids = {r["id"] for r in taken}
        if got < fam_budget[f]:                             # top up from what the strata left (rounding, overshoot rule)
            t, got = _take(order([r for r in by_fam[f] if r["id"] not in ids]), fam_budget[f], nq, got)
            taken += t
        chosen += taken
        rep[f] = {"available_units": avail[f], "budget": fam_budget[f], "taken_units": got,
                  "strata": len(cells), "strata_by_type": by_type}
    return chosen, rep


# --------------------------------------------------------------------------- validation
def validate(out, tokenizer=None, max_len=2048):
    """The trainer's own loading path over <out>: cache_features.iter_units (train, val), claude_lmtune.py's NC and K
    filters, per-source counts, label sanity, and (tokenizer given) make_example on every unit."""
    t0 = time.time()
    lm = _load("claude_lmtune", HERE / "trainers" / "claude_lmtune.py")
    cf, ls = lm.cf, lm.ls
    res = {}
    for split in ("train", "val"):
        units = list(cf.iter_units(out, [split], set(), None))
        nc = [u for u in units if cf.is_noncommercial(u)]
        k_bad = [u for u in units if len(u["option_keys"]) > len(ls.LABELS)]
        bad_t = [u for u in units if not (sum(u["target"]) > 0 and min(u["target"]) >= 0)]
        per = Counter(str(u["source"]).split("/")[0] for u in units)
        per_file = Counter()
        for p in sorted(Path(out, split).glob("*.jsonl")):
            per_file[p.stem] = sum(len(json.loads(l)["questions"]) for l in open(p, encoding="utf-8"))
        log(f"VALIDATE {split}: {len(units):,} units via cache_features.iter_units; non-commercial {len(nc)}; "
            f"K > {len(ls.LABELS)}: {len(k_bad)}; bad targets {len(bad_t)}")
        log(f"VALIDATE {split} units per file: {dict(sorted(per_file.items()))}")
        log(f"VALIDATE {split} units per record-source family: {dict(sorted(per.items()))}")
        lk = Counter((u["qtype"], u["label_kind"]) for u in units)
        log(f"VALIDATE {split} (qtype, label_kind): {dict(sorted(lk.items()))}")
        assert not nc, f"{len(nc)} non-commercial units in {split}"
        assert not k_bad, f"{len(k_bad)} units with K > {len(ls.LABELS)} in {split}"
        assert not bad_t, f"{len(bad_t)} units with bad targets in {split}"
        info = {"units": len(units), "noncommercial": len(nc), "k_too_large": len(k_bad), "per_file": dict(per_file)}
        if tokenizer:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer)
            rng = random.Random(0)
            stats, none, cut = {}, 0, Counter()
            lens = []
            for u in units:
                ex = lm.make_example(tok, u, rng, max_len, split == "train", stats=stats)
                if ex is None:
                    none += 1
                    continue
                text, tgt = ex
                if abs(float(sum(tgt)) - 1) > 1e-6:
                    raise AssertionError(f"target does not sum to 1: {u['record_id']}")
                if norm(u["state"])[:4000] and norm(u["state"][-200:]) not in norm(text):
                    cut[str(u["source"]).split("/")[0]] += 1
                lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
            lens.sort()
            log(f"VALIDATE {split} make_example (max_len {max_len}): {len(units) - none:,} prompts built, {none} None; "
                f"state cut to fit: {sum(cut.values())} {dict(cut)}; prompt tokens p50 {lens[len(lens) // 2]} "
                f"p99 {lens[int(len(lens) * .99)]} max {lens[-1]}")
            assert none == 0, f"{none} units make_example could not build"
            info.update({"make_example_none": none, "state_cut": dict(cut), "tokens_p50": lens[len(lens) // 2],
                         "tokens_max": lens[-1]})
        res[split] = info
    log(f"VALIDATE OK ({time.time() - t0:.0f}s)")
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="/data/shared/rd1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sources", default=",".join(SPECS), help="comma list (default: all)")
    ap.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    ap.add_argument("--val-cap", type=int, default=VAL_CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval", action="append", default=[], help="extra TAG=PATH (jsonl file or dir) eval items")
    ap.add_argument("--no-discover-evals", dest="discover", action="store_false")
    ap.add_argument("--no-fetch-evals", dest="fetch_evals", action="store_false")
    ap.add_argument("--fetch-cache", default=None)
    ap.add_argument("--allow-missing-evals", action="store_true")
    ap.add_argument("--max-df", type=int, default=20)
    ap.add_argument("--max-train-df", type=int, default=10)
    ap.add_argument("--min-shared", type=int, default=3)
    ap.add_argument("--min-contain", type=float, default=0.10)
    ap.add_argument("--min-field-words", type=int, default=8)
    ap.add_argument("--ngram-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="forked gate workers")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: records per source file")
    ap.add_argument("--validate", default=None, help="validate an existing corpus dir and exit")
    ap.add_argument("--tokenizer", default=None, help="run claude_lmtune.make_example on every unit with this tokenizer")
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    if a.validate:
        validate(a.validate, a.tokenizer, a.max_len)
        return 0
    if os.environ.get("PYTHONHASHSEED") != "0":
        log("note: PYTHONHASHSEED is not 0 -- shingle hashes (str hash) differ run to run; set it for a bit-exact rebuild")
    a.out = a.out or os.path.join(a.root, "out", "jev_rd1")
    a.fetch_cache = a.fetch_cache or os.path.join(a.root, "evals", "ext2_cache")
    os.environ.setdefault("HF_HOME", os.path.join(a.root, "evals", "hf"))
    cf = _load("cache_features", HERE / "trainers" / "cache_features.py")
    roots = {"laptop": os.path.join(a.root, "laptop"), "s3": os.path.join(a.root, "s3src")}
    names = [s for s in a.sources.split(",") if s]
    t0 = time.time()
    log(f"{VERSION}: sources {names}; caps train {a.train_cap} / val {a.val_cap}; seed {a.seed}")

    # ---- candidates
    cands, rep = [], {}
    for name in names:
        spec = SPECS[name]
        r_ = rep[name] = {"files": [], "records_read": Counter(), "question_drops": Counter(), "licenses": Counter()}
        for dkey, sub, pat in spec["dirs"]:
            base = os.path.join(roots[dkey], sub)
            for split in ("train", "val"):
                files = sorted(glob.glob(os.path.join(base, split, pat)))
                if not files:
                    raise SystemExit(f"{name}: no {split} files at {base}/{split}/{pat}")
                for p in files:
                    if name == "openjev" and any(b in Path(p).name for b in OPENJEV_BLOCKED):
                        raise SystemExit(f"blocked Open-Jev sub present: {p}")
                    r_["files"].append(os.path.relpath(p, a.root))
                    for i, r in enumerate(read_jsonl(p)):
                        if a.limit and i >= a.limit:
                            break
                        r_["records_read"][split] += 1
                        rec, drops = clean_record(r, name, spec, cf.is_noncommercial)
                        r_["question_drops"].update(drops)
                        if rec is None:
                            continue
                        r_["licenses"][rec["license"]] += 1
                        cands.append((f"{name}|{split}|{rec['id']}", rec))
        log(f"{name}: read {dict(r_['records_read'])}; question drops {dict(r_['question_drops'])}")
    log(f"candidates: {len(cands):,} records ({time.time() - t0:.0f}s)")
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

    # ---- evals + gate
    evals = discover_evals(a.root) if a.discover else []
    for spec in a.eval:
        tag, _, p = spec.partition("=")
        files = [p] if os.path.isfile(p) else sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        evals += [(tag, f) for f in files]
    tags = sorted({t for t, _ in evals})
    log(f"eval files: {len(evals)} under {len(tags)} tags")
    for t in tags:
        log(f"  eval tag {t}: {sum(1 for x, _ in evals if x == t)} files")
    must = ["jev_real/test", "extevals_b/test", "extevals_community/test", "extevals_tdb/test"]
    missing = [t for t in must if t not in tags]
    if missing and not a.allow_missing_evals:
        raise SystemExit(f"required eval sets missing: {missing}")
    hits, tag_items, failed = run_gate(cands, evals, a)

    # ---- apply the gate; within-rd1 dedup
    drop_by = defaultdict(lambda: {"records": Counter(), "units": Counter(), "kinds": Counter(), "eval_sets": Counter(),
                                   "entity_key_types": Counter(), "examples": []})
    per_tag = defaultdict(lambda: Counter())
    kept = {(n, s): [] for n in names for s in ("train", "val")}
    for k, r in cands:
        name, split, _ = k.split("|", 2)
        h = hits.get(k)
        if h:
            d = drop_by[name]
            d["records"][split] += 1
            d["units"][split] += len(r["questions"])
            for kind in sorted({x[0] for x in h}):
                d["kinds"][kind] += 1
            for t in sorted({t for x in h for t in x[1]}):
                d["eval_sets"][t] += 1
                per_tag[t][f"{name}/{split}"] += 1
            for x in h:
                if x[0] == "entity":
                    d["entity_key_types"][x[2]["key"].split(":")[0]] += 1
            if len(d["examples"]) < 6:
                d["examples"].append({"id": r["id"], "split": split, "hits": [(x[0], x[1][:3], x[2]) for x in h[:3]]})
            continue
        kept[(name, split)].append(r)
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
    log(f"within-rd1 dedup drops: {dict(within)}")

    # ---- sample + write
    out = Path(a.out)
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)
        for old in (out / split).glob("*.jsonl"):
            old.unlink()
    manifest = {"version": VERSION, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": a.seed,
                "caps": {"train_units": a.train_cap, "val_units": a.val_cap},
                "gate": {"kinds": DROP_KINDS, "max_df": a.max_df, "max_train_df": a.max_train_df,
                         "min_shared": a.min_shared, "min_contain": a.min_contain, "min_field_words": a.min_field_words,
                         "ngram13_frac": a.ngram_frac, "eval_items_per_tag": tag_items, "fetch_failed": failed},
                "within_rd1_dedup": dict(within), "sources": {}, "totals": {}}
    tot = Counter()
    for name in names:
        spec = SPECS[name]
        srep = {"files": rep[name]["files"], "records_read": dict(rep[name]["records_read"]),
                "question_drops": dict(rep[name]["question_drops"])}
        for split, cap in (("train", a.train_cap), ("val", a.val_cap)):
            pool = kept[(name, split)]
            chosen, alloc = sample_source(pool, spec, cap, a.seed + (0 if split == "train" else 1))
            for r in chosen:
                r["meta"] = dict(r["meta"], rd1={"source": name, "split": split, "family": spec["family"](r),
                                                 "date": (spec["date"] or (lambda _r: None))(r)})
            with open(out / split / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in chosen:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            units = sum(len(r["questions"]) for r in chosen)
            ls_mix = Counter(c for r in chosen for c in r["label_source"].values())
            dates = sorted(d for d in ((spec["date"] or (lambda _r: None))(r) for r in chosen) if d)
            srep[split] = {"records": len(chosen), "units": units, "cap_units": cap,
                           "available_units_after_gate": sum(len(r["questions"]) for r in pool),
                           "available_records_after_gate": len(pool), "cap_applied": units < sum(len(r["questions"]) for r in pool),
                           "families": alloc, "label_source_units": dict(ls_mix),
                           "label_kind_units": dict(Counter(k for r in chosen for k in r["label_kind"].values())),
                           "qtype_units": dict(Counter(q["type"] for r in chosen for q in r["questions"].values())),
                           "time_range": [dates[0], dates[-1]] if dates else None,
                           "licenses": dict(Counter(r["license"] for r in chosen).most_common())}
            tot[f"{split}_units"] += units
            tot[f"{split}_records"] += len(chosen)
        d = drop_by.get(name)
        srep["gate_drops"] = {"records": dict(d["records"]), "units": dict(d["units"]), "by_kind_records": dict(d["kinds"]),
                              "by_eval_set_records": dict(d["eval_sets"].most_common()),
                              "entity_key_types": dict(d["entity_key_types"]), "examples": d["examples"]} if d else {}
        manifest["sources"][name] = srep
        log(f"{name}: train {srep['train']['units']} units / {srep['train']['records']} rec (avail "
            f"{srep['train']['available_units_after_gate']}), val {srep['val']['units']} / {srep['val']['records']}; "
            f"gate dropped {dict(d['records']) if d else {}} {dict(d['kinds']) if d else ''}")
    manifest["totals"] = dict(tot)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, ensure_ascii=False))
    (out / "gate_report.json").write_text(json.dumps({"per_eval_set": {t: dict(c) for t, c in sorted(per_tag.items())},
                                                      "eval_items_per_tag": tag_items}, indent=1, default=str))
    log(f"TOTAL train {tot['train_units']} units / {tot['train_records']} records; val {tot['val_units']} / "
        f"{tot['val_records']} ({time.time() - t0:.0f}s)")
    log("GATE DROPS per source (records train/val; kinds):")
    for name in names:
        d = drop_by.get(name)
        log(f"  {name:14s} {dict(d['records']) if d else {}}  {dict(d['kinds']) if d else {}}  "
            f"entity {dict(d['entity_key_types']) if d else {}}")
    log("GATE DROPS per eval set (candidate records hitting it, all sources):")
    for t, c in sorted(per_tag.items(), key=lambda x: -sum(x[1].values())):
        log(f"  {t:60s} {sum(c.values()):6d}  {dict(c.most_common(6))}")
    manifest["validation"] = validate(str(out), a.tokenizer, a.max_len)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, ensure_ascii=False))
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
