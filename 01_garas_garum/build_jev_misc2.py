#!/usr/bin/env python3
"""
build_jev_misc2.py -- misc2 (2026-10-01): six datasets suggested by outside researchers, vetted at the primary source
and converted for jev (single-pass typed decision model): training corpora for the zero-shot line and GDE-1-format
eval sets. Write-up: research/misc2_pipeline.md. The researchers' claims were treated as unverified; where they were
wrong is recorded in that file and in each manifest.

Verdicts (licences quoted with URLs in LICENCE_EVIDENCE below and in each manifest):
  civil_comments  EVAL ONLY. GDE-1 V set `civil_comments_votes` (CC0). jev-bench's civil_comments config (10,500 items:
                  test 2,000 = google/civil_comments test rows; train 8,000 + val 500 = google validation rows) spans
                  the whole archive in time (2015-09-29 .. 2017-11-11), so no training split can be date-disjoint from
                  it, and ext2's gate fingerprints only jev-bench TEST. Label = Jigsaw toxicity fraction x
                  toxicity_annotator_count -> vote counts, >= 10 raters only; jev-bench's state + question kept
                  verbatim so existing per-item predictions rescore.
  medqa           EVAL ONLY, K side panel `medqa_usmle`: no data licence (MIT covers the code; questions scraped from
                  commercial question banks); publish the converter, not the text.
  folio           EVAL ONLY, K side panel `folio`: FOLIO v0.0 validation (GitHub, CC BY-SA 4.0), every label re-proved
                  here with z3 from the authors' FOL; only items whose label the prover confirms are kept. (v2 on HF,
                  yale-nlp/FOLIO, MIT, is gated and the user's account has not accepted it; --folio-v2 reads it then.)
  lichess         TRAIN (small slice, `lichess`) + T side panel `lichess_puzzles` (CC0). State = the position AFTER the
                  opponent's move (Moves[0] applied to FEN); gold = Moves[1]; options = the solution + up to 3 legal
                  alternatives chosen deterministically (captures / checks first), never a mating move.
  folktables      TRAIN (`folktables/<task>`, ACS 2018, 6 states) + T side panel `folktables_acs` (ACS 2024: 4 held-out
                  states + the 6 training states). Demographics are kept in the state and in meta for bias tables.
  esci            TRAIN (`esci`, ESCI train-split queries) + T side panel `esci` (ESCI test-split queries); TDB's 100
                  esci-product-rerank queries are blocked from both.
  chaosnli        nothing to build: already a jev-bench eval (CC BY-NC 4.0), blocked in ext3, fingerprinted by ext2.
Domain T (new SIDE domain in analysis/gde1_score.py): "trained-source panel" -- the misc2 training line trains on
other items of the same source, so the set is not zero-shot for our models; scored per set, never in the headline.

Layout
  training  /tmp/misc2_jev/<dataset>/{train,val}/<source>.jsonl + manifest.json   (build_jev_data schema; 95/5 by group)
  eval      /tmp/gde1/<set>/eval/<source>.jsonl + manifest.json + samples.json   (GDE-1 schema: meta.domain, gde1_set,
            label_provenance, item_key, base_id = bootstrap cluster; V sets: meta.n_raters + meta.counts)
  scratch   /tmp/misc2_work (raw downloads, candidates, HF cache)

Dedup
  * Training: the ext2 gate verbatim (build_jev_ext2.load_fingerprints + process: exact / leaf / 8-word shingle drop
    against jev-bench test (22 configs), typed-decisions, TDB v0.3, MMLU/ARC, LegalBench, community, DICES, ChaosNLI,
    plus every GDE-1 set under /tmp/gde1 incl. the ones built here). The training corpora themselves (jev_data, ...)
    live on S3 / nodes: pass --fingerprint TAG=PATH there; on the laptop the cross-corpus check is recorded as not run.
  * Eval: build_gde1_a's Gate (eval items indexed, training states streamed) against (a) this builder's own training
    candidates and (b) targeted upstream proxies of the training sources that could plausibly share text (MedQA:
    medmcqa + pubmedqa; Civil Comments: MHS + GoEmotions). `gate --corpus TAG=PATH` re-gates on a node.

Usage (from the repo):
  W="uv run --no-project --python 3.13 --with numpy --with pyarrow --with huggingface_hub --with chess --with zstandard \\
     --with z3-solver --with fsspec --with aiohttp python"
  $W build_jev_misc2.py civil_comments | medqa | folio | lichess | folktables | esci | chaosnli
  $W build_jev_misc2.py train            # ext2 gate + write /tmp/misc2_jev (after the dataset subcommands)
  $W build_jev_misc2.py check            # read everything back through cache_features.iter_units
  $W build_jev_misc2.py gate --corpus jev_data=/data/local/jev_data --corpus ...   # on a node
  $W build_jev_misc2.py --self-test      # offline fixtures
HF token: only for gated reads (--folio-v2), read in-process from ~/.hf_read_token into a request header; never
printed, logged, written or put on a command line.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))
WORK = Path(os.environ.get("MISC2_WORK", "/tmp/misc2_work"))
os.environ.setdefault("HF_HOME", str(WORK / "hf"))     # before anything imports huggingface_hub

from claude_overlap import norm  # noqa: E402  the eval-overlap normalization, reused verbatim

import build_gde1_a as ga  # noqa: E402  Gate / gate_items / vote_stats / corpus_stream
import build_gde1_b as gb  # noqa: E402  lab_dict / kind_of / split_stats
import build_jev_ext2 as x2  # noqa: E402  the ext2 gate: Index / load_fingerprints / process
from build_jev_data import Writer, choice_q, noul_q, option_keys, record, validate  # noqa: E402
from build_jev_ext import final_check, load_cache_features, plain  # noqa: E402

VERSION = "misc2 0.1 (2026-10-01)"
OUT_TRAIN = Path(os.environ.get("MISC2_OUT_TRAIN", "/tmp/misc2_jev"))
OUT_EVAL = Path(os.environ.get("MISC2_OUT_EVAL", "/tmp/gde1"))
SEED = 20261001
CAP_TRAIN = 20000
CAP_EVAL = 8000
CUTOFF = "2025-07-01"
MAX_STATE_CHARS = 6500
UA = "jev-misc2 (research data build)"
TOKEN_FILE = "~/.hf_read_token"
EVAL_DOMAINS = ("V", "P", "O", "X", "K", "T")

PINS = {
    "google_civil_comments": ("google/civil_comments", "f2970eb3a55777454c94069077cc8d9b5866312d"),
    "kaggle_mirror": ("TheMrguiller/jigsaw-unintended-bias-in-toxicity-classification",
                      "c0fe541912004c8d18cc4b6f3dc3849c42700743"),
    "wilds_civilcomments": "https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/",
    "jevbench": ("Praveenrajus/jev-bench", "2797e3c13020b33188177a19fd9bbb707f172a21"),
    "medqa": ("GBaker/MedQA-USMLE-4-options", "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"),
    "folio_gh": ("Yale-LILY/FOLIO", "5d7bb84c7edab3fb358e057d2807f19cf5cf5e2d"),
    "folio_hf": ("yale-nlp/FOLIO", "4c1f14bf39ca6a6de242bd8f75ef313b4b436818"),
    "lichess": "https://database.lichess.org/lichess_db_puzzle.csv.zst",
    "acs": "https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{st}.zip",
    "acs_dict": "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_{year}.csv",
    "folktables": ("socialfoundations/folktables", "ebace19d8065efc43589f0067bf183398eb156b9"),
    "esci": ("amazon-science/esci-data", "7916cdf6ab75a462e77f20ab40428a10923998d5"),
    "tdb": ("SamuelChien821/typed-decision-bench", "79ec2b9bf193d69ceaaaf86ad9d0b79623adb101"),
}

# Record-level licence strings. Training strings must not match cache_features.NC_LICENCE; eval-only NC / research-only
# strings must (so the trainer refuses them even if a file is misplaced).
LICENCE = {
    "civil_comments_votes": "CC0-1.0 (Civil Comments / Jigsaw Unintended Bias: google/civil_comments card 'This dataset is "
                            "released under CC0, as is the underlying comment text'; WILDS CivilComments 'distributed "
                            "under CC0'); jev: EVAL ONLY (same source as the jev-bench civil_comments eval)",
    "medqa_usmle": "No data licence (MedQA, Jin et al. 2020: the jind11/MedQA MIT licence covers the code only; the USMLE "
                   "questions were scraped from commercial question banks, paper App. A; the GBaker re-upload's "
                   "cc-by-4.0 tag carries no grant from the authors). EVAL ONLY, research use only: never train; "
                   "publish the converter, not the text",
    "folio": "CC-BY-SA-4.0 (FOLIO v0.0, github.com/Yale-LILY/FOLIO LICENSE); jev: eval side panel",
    "folio_v2": "MIT (FOLIO v2, huggingface.co/datasets/yale-nlp/FOLIO card; gated); jev: eval side panel",
    "lichess": "CC0-1.0 (Lichess database exports: 'released under the Creative Commons CC0 license. Use them for "
               "research, commercial purpose, publication, anything you like')",
    "folktables": "CC0-1.0 (ACS PUMS, US Census Bureau: api.census.gov PUMS metadata license "
                  "creativecommons.org/publicdomain/zero/1.0; Census terms: never use the data to identify any person "
                  "or household); task definitions from folktables (MIT)",
    "esci": "Apache-2.0 (amazon-science/esci-data LICENSE + README; product text from amazon.com, NOTICE: Copyright "
            "Amazon.com, Inc. or its affiliates)",
}

LICENCE_EVIDENCE = {
    "civil_comments": [
        ("https://huggingface.co/datasets/google/civil_comments (README @ f2970eb, card YAML license: cc0-1.0)",
         "This data set is an exact replica of the data released for the Jigsaw Unintended Bias in Toxicity "
         "Classification Kaggle challenge. This dataset is released under CC0, as is the underlying comment text."),
        ("https://api.figshare.com/v2/articles/7376747 (the original Civil Comments archive)", "license: CC0"),
        ("https://raw.githubusercontent.com/p-lambda/wilds/main/wilds/datasets/civilcomments_dataset.py",
         "This dataset is in the public domain and is distributed under CC0."),
        ("toxicity_annotator_count", "exists only in the Kaggle files (login) and mirrors; the mirror used here "
         "(TheMrguiller/..., no licence of its own; the data is CC0) agrees with the official WILDS release on all "
         "42,870 overlapping ids (toxicity and count) and with google/civil_comments on text + toxicity"),
    ],
    "medqa": [
        ("https://github.com/jind11/MedQA LICENSE @ 27b02f6", "MIT License, Copyright (c) 2022 Di Jin ... this software"),
        ("arXiv 2009.13081 v1 §3.2", "We include problems from both real exams and mock tests; all are freely "
         "accessible online for public usage. ... To comply with fair use of law, we shuffle the order of answer options "
         "and randomly delete one of the wrong options for each question for USMLE and MCMLE datasets"),
        ("arXiv 2009.13081 v1 App. A / Table A.11", "For USMLE and MCMLE datasets, we scraped several websites that "
         "provide the question banks (step1.medbullets.com, amboss.com/us/usmle, lecturio.com/usmle-step-1)"),
        ("https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options @ 0fb93dd", "card YAML license: cc-by-4.0 "
         "(the uploader's tag; the card holds only the citation, no grant from the authors)"),
    ],
    "folio": [
        ("https://github.com/Yale-LILY/FOLIO LICENSE @ 5d7bb84", "Attribution-ShareAlike 4.0 International (data/v0.0)"),
        ("https://github.com/Yale-LILY/FOLIO README @ 5d7bb84", "[UPDATE]: We are excited to release a new version of "
         "FOLIO [here](https://huggingface.co/datasets/yale-nlp/FOLIO) with significant quality improvement and error "
         "fixing!"),
        ("https://huggingface.co/api/datasets/yale-nlp/FOLIO @ 4c1f14b", "license: mit; gated: auto (the user's token "
         "gets HTTP 403 on the files: the gate has not been accepted)"),
        ("arXiv 2209.00840 abstract", "The logical correctness of the premises and conclusions is ensured by their FOL "
         "annotations, which are automatically verified by an FOL inference engine."),
        ("https://huggingface.co/api/datasets/Yale-LILY/FOLIO", "HTTP 404 'Repository not found' (the researchers' id)"),
    ],
    "lichess": [
        ("https://database.lichess.org/", "Database exports are released under the Creative Commons CC0 license. Use "
         "them for research, commercial purpose, publication, anything you like. You can download, modify and "
         "redistribute them, without asking for permission."),
        ("https://database.lichess.org/#puzzles", "FEN is the position before the opponent makes their move. The "
         "position to present to the player is after applying the first move to that FEN. The second move is the "
         "beginning of the solution."),
        ("https://database.lichess.org/#puzzles", "All player moves of the solution are 'only moves'. ... An exception "
         "is made for mates in one: there can be several. Any move that checkmates should win the puzzle."),
    ],
    "folktables": [
        ("https://github.com/socialfoundations/folktables LICENSE.txt @ ebace19", "MIT License / Copyright (c) 2021 ZYKLS"),
        ("folktables README", "The data itself is governed by the terms of use provided by the Census Bureau. For more "
         "information, see https://www.census.gov/data/developers/about/terms-of-service.html"),
        ("https://api.census.gov/data/2018/acs/acs1/pums.json and .../2024/acs/acs1/pums.json",
         "\"license\": \"https://creativecommons.org/publicdomain/zero/1.0/\", \"accessLevel\": \"public\""),
        ("https://www.census.gov/data/developers/about/terms-of-service.html", "users will not use these data, alone or "
         "in combination with any other Census or non-Census data, to identify any individual person, household, "
         "business or other entity"),
    ],
    "esci": [
        ("https://github.com/amazon-science/esci-data LICENSE + README @ 7916cdf", "Apache License 2.0; 'This project is "
         "licensed under the Apache-2.0 License.'"),
        ("arXiv 2206.06588 §2", "A minimum of three annotations were collected for each pair, and an automatic "
         "aggregation mechanism selected the majority vote as the gold label."),
    ],
    "chaosnli": [
        ("https://github.com/easonnie/ChaosNLI README @ f358e23", "Chaos NLI is licensed under Creative Commons-Non "
         "Commercial 4.0."),
    ],
}


# =========================================================================== small helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(s):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def uhash(*parts):
    """A uniform number in [0, 1) from the parts: seeded, order-independent selection."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:13], 16) / float(16 ** 13)


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_token():
    """The user's HF read token, IN-PROCESS only: never printed, logged, written or passed on a command line."""
    p = Path(TOKEN_FILE).expanduser()
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def hf_url(repo, rev, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def fetch(url, dest, token=False, tries=6):
    """Cached streaming download to dest (WORK/raw/...). Retries 429 / 5xx / network errors; any other HTTP error is
    raised at once (a gated 403 must not be retried)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA}
    if token:
        tok = read_token()
        if not tok:
            raise SystemExit(f"{url}: needs the HF token ({TOKEN_FILE}) and it is missing")
        headers["Authorization"] = "Bearer " + tok
        del tok
    try:
        for t in range(tries):
            try:
                req = urllib.request.Request(url, headers=headers)
                tmp = dest.with_name(dest.name + ".part")
                with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                    for b in iter(lambda: r.read(1 << 20), b""):
                        f.write(b)
                tmp.replace(dest)
                return dest
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or t == tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                if t == tries - 1:
                    raise
            time.sleep(4 * 2 ** t)
    finally:
        headers.clear()
    return dest


def hf_api_sha(repo):
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}",
                                                           headers={"User-Agent": UA}), timeout=60) as r:
            return json.load(r).get("sha")
    except Exception as e:  # noqa: BLE001
        return f"unresolved ({type(e).__name__})"


def raw(*parts):
    return WORK.joinpath("raw", *parts)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def entropy(vals):
    return -sum(x * math.log(x) for x in vals if x > 0)


def hist(xs, edges):
    out = Counter()
    for x in xs:
        lab = next((f"<{e}" for e in edges if x < e), f">={edges[-1]}")
        out[lab] += 1
    return dict(out)


# =========================================================================== eval records, writing, read-back
def eval_rec(gset, domain, source, state, qs, labs, meta, base_id, licence, rec_id):
    assert domain in EVAL_DOMAINS, domain
    kinds = {q: gb.kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, None)
    rec["id"] = rec_id
    rec["base_id"] = base_id
    rec["license"] = licence
    rec["meta"] = {"gde1_set": gset, "domain": domain, "gde1_part": "misc2", "gde1_version": VERSION,
                   "eval_only": True, **meta}
    validate(rec)
    return rec


def train_rec(source, state, qs, labs, meta, group, licence, probes=()):
    kinds = {q: gb.kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, None)
    rec["base_id"] = rec["id"]
    rec["license"] = licence
    rec["meta"] = {"dataset": source.split("/")[0], "group": str(group), "builder": VERSION, **meta}
    rec["_probes"] = [p for p in probes if p]
    validate(rec)
    return rec


def readback_eval(root, integral=False):
    """Every record re-validated and round-tripped through cache_features.question_options; every unit read back
    through cache_features.iter_units (integral=True: labels * meta.n_raters must be integers, V sets)."""
    cf = load_cache_features()
    n_rec, by_src = 0, Counter()
    for p in sorted(Path(root, "eval").glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            rec = json.loads(line)
            validate(rec)
            m = rec["meta"]
            assert m["domain"] in EVAL_DOMAINS and m["gde1_set"] and m.get("item_key") and rec.get("base_id"), rec["id"]
            for qid, q in rec["questions"].items():
                keys, texts, target = cf.question_options(q, rec["labels"][qid])
                assert keys == option_keys(q) and len(texts) == len(target) <= 52, (rec["id"], qid)
                assert all(abs(t - rec["labels"][qid][k]) < 1e-12 for k, t in zip(keys, target)), (rec["id"], qid)
            n_rec += 1
            by_src[rec["source"]] += 1
    units, qtypes, kinds, lens = 0, Counter(), Counter(), []
    with quiet():
        us = list(cf.iter_units(str(root), ["eval"], set(), None))
    for u in us:
        assert abs(sum(u["target"]) - 1.0) < 1e-9, u["record_id"]
        if integral:
            n = u["n_raters"]
            assert n and all(abs(t * n - round(t * n)) < 1e-6 for t in u["target"]), u["record_id"]
        units += 1
        qtypes[u["qtype"]] += 1
        kinds[u["label_kind"]] += 1
        lens.append(len(u["state"]))
    lens.sort()
    files = {p.name: file_sha256(p) for p in sorted(Path(root, "eval").glob("*.jsonl"))}
    return {"records": n_rec, "records_by_source": dict(by_src), "units": units, "qtypes": dict(qtypes),
            "label_kind": dict(kinds), "state_chars": {"median": lens[len(lens) // 2], "max": lens[-1]} if lens else None,
            "files_sha256": files}


def write_eval(gset, recs, manifest, notes=None, integral=False, sample_text=True):
    """<OUT_EVAL>/<gset>/eval/<source>.jsonl (this set's own files are replaced; nothing else is touched) +
    manifest.json + samples.json; then the read-back, recorded in the manifest."""
    root = OUT_EVAL / gset
    d = root / "eval"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.jsonl"):
        old.unlink()
    files = {}
    for r in recs:
        fn = d / (r["source"].replace("/", "__") + ".jsonl")
        if fn not in files:
            files[fn] = fn.open("w", encoding="utf-8")
        files[fn].write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    for f in files.values():
        f.close()
    for name, text in (notes or {}).items():
        (root / name).write_text(text, encoding="utf-8")
    manifest = {"set": gset, "builder": VERSION, "built": time.strftime("%Y-%m-%d %H:%M:%S"), **manifest}
    manifest["readback"] = readback_eval(root, integral=integral)
    manifest["split_stats"] = gb.split_stats(recs)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    picks = [recs[i] for i in sorted({0, len(recs) // 3, (2 * len(recs)) // 3}) if i < len(recs)]
    samples = [{**r, "state": (r["state"][:1500] + (" ..." if len(r["state"]) > 1500 else "")) if sample_text
                else "<not reproduced: converter-only set>"} for r in picks]
    (root / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False), encoding="utf-8")
    rb = manifest["readback"]
    log(f"{gset}: wrote {rb['records']} records / {rb['units']} units (read back through cache_features.iter_units)")
    return manifest


# =========================================================================== training candidates (pre-gate)
def cands_path(ds):
    return WORK / "cands" / f"{ds}.jsonl.gz"


def write_cands(ds, recs, info):
    p = cands_path(ds)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)
    p.with_suffix(".info.json").write_text(json.dumps(info, indent=1, default=str, ensure_ascii=False))
    log(f"{ds}: {len(recs)} training candidates -> {p} (gated and capped by `train`)")


def read_cands(ds):
    p = cands_path(ds)
    if not p.exists():
        return []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# =========================================================================== eval-side gate (build_gde1_a.Gate)
def _parquet_rows(path, cols):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pylist()


def proxy_source(name):
    """A callable returning a fresh iterable of (tag, state, probes) for one training-source proxy: the upstream rows
    the training corpus samples from, read in full (a superset of what the corpus holds)."""
    if name.startswith("misc2/"):
        ds = name.split("/", 1)[1]

        def gen():
            for r in read_cands(ds):
                yield name, r["state"], r.get("_probes") or []
        return gen
    if name == "jev_data/mhs":
        def gen():
            p = fetch(hf_url("ucberkeley-dlab/measuring-hate-speech", "main", "measuring-hate-speech.parquet"),
                      raw("proxy", "mhs.parquet"))
            for t in {r["text"] for r in _parquet_rows(p, ["text"]) if r["text"]}:
                yield name, t, []
        return gen
    if name == "jev_data/go_emotions":
        def gen():
            p = fetch(hf_url("google-research-datasets/go_emotions", "main", "raw/train-00000-of-00001.parquet"),
                      raw("proxy", "go_emotions_raw.parquet"))
            for t in {r["text"] for r in _parquet_rows(p, ["text"]) if r["text"]}:
                yield name, t, []
        return gen
    if name == "jev_ext2/medical/medmcqa":
        def gen():
            for sp in ("train", "validation", "test"):
                p = fetch(hf_url("openlifescienceai/medmcqa", "main", f"data/{sp}-00000-of-00001.parquet"),
                          raw("proxy", f"medmcqa_{sp}.parquet"))
                for r in _parquet_rows(p, ["question", "opa", "opb", "opc", "opd"]):
                    q = str(r.get("question") or "")
                    yield name, {"question": q, "options": [r.get(k) for k in ("opa", "opb", "opc", "opd")]}, [q]
        return gen
    if name == "jev_ext2/medical/pubmedqa":
        def gen():
            for cfg in ("pqa_labeled", "pqa_artificial"):
                p = fetch(hf_url("qiaojin/PubMedQA", "main", f"{cfg}/train-00000-of-00001.parquet"),
                          raw("proxy", f"pubmedqa_{cfg}.parquet"))
                for r in _parquet_rows(p, ["question", "context"]):
                    ctx = r.get("context") or {}
                    body = "\n".join(ctx.get("contexts") or []) if isinstance(ctx, dict) else str(ctx)
                    q = str(r.get("question") or "")
                    yield name, {"question": q, "context": body}, [q]
        return gen
    raise KeyError(name)


def gate_eval(recs, proxies, fields=lambda r: [], content=lambda r: r["state"]):
    """Drop eval records whose item overlaps a training-proxy state (build_gde1_a.Gate defaults: exact; leaf >= 8
    words in <= 10 training states; >= 5 shared 8-word shingles covering >= 10% of the shorter, ignoring shingles held
    by > 20 eval items or found in > 10 training states). Returns (kept records, report)."""
    if not proxies or not recs:
        return recs, {"status": "no proxies"}
    items, seen = [], set()
    for r in recs:
        k = r["meta"]["item_key"]
        if k not in seen:
            seen.add(k)
            items.append({"key": k, "state": content(r), "fields": fields(r)})
    srcs = [proxy_source(n) for n in proxies]

    def stream():
        for s in srcs:
            yield from s()
    hits, read = ga.gate_items(items, stream)
    drop = {k for k, h in hits.items() if ga.dropping(h)}
    kinds, reported, ex = Counter(), Counter(), []
    for k, h in hits.items():
        if k in drop:
            kinds[f"{h[0][1]}:{h[0][0]}"] += 1
            if len(ex) < 5:
                ex.append({"item": k, "match": [(t, kd, n, (s if isinstance(s, str) else (s or [])[:2]))
                                                for t, kd, n, s in h[:2]]})
        else:
            reported[h[0][1]] += 1
    kept = [r for r in recs if r["meta"]["item_key"] not in drop]
    return kept, {"proxies": proxies, "training_states_read": dict(read), "items_checked": len(items),
                  "items_dropped": len(drop), "dropped_by_kind_and_source": dict(kinds),
                  "reported_not_dropped": dict(reported), "examples": ex,
                  "rule": "build_gde1_a.Gate defaults (see research/gde1_part_a.md §2.2)", "when": "before sampling/cap"}


# =========================================================================== 1. Civil Comments (V, eval only)
CC_SET = "civil_comments_votes"
CC_SOURCE = "gde1/civil_comments_votes"
CC_QID = "toxic"
CC_MIN_RATERS = 10
CC_ARCHIVE = ("2015-09-29", "2017-11-11")       # first / last created_date in the archive (WILDS release)
CC_IDENT = ["male", "female", "transgender", "other_gender", "heterosexual", "homosexual_gay_or_lesbian", "bisexual",
            "other_sexual_orientation", "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
            "black", "white", "asian", "latino", "other_race_or_ethnicity", "physical_disability",
            "intellectual_or_learning_disability", "psychiatric_or_mental_illness", "other_disability"]
CC_SUB = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
CC_LABEL_NOTE = ("vote share: raters who rated the comment Toxic or Very Toxic, out of toxicity_annotator_count (Jigsaw "
                 "Unintended Bias: `toxicity` is that fraction; toxicity x count is an integer for every item)")


def cc_fetch():
    g, gr = PINS["google_civil_comments"]
    m, mr = PINS["kaggle_mirror"]
    j, jr = PINS["jevbench"]
    paths = {
        "google_test": fetch(hf_url(g, gr, "data/test-00000-of-00001.parquet"), raw("civil", "test-00000-of-00001.parquet")),
        "google_validation": fetch(hf_url(g, gr, "data/validation-00000-of-00001.parquet"),
                                   raw("civil", "validation-00000-of-00001.parquet")),
        "kaggle_mirror_test": fetch(hf_url(m, mr, "data/test-00000-of-00001.parquet"), raw("civil", "kaggle_mirror_test.parquet")),
        "wilds": fetch(PINS["wilds_civilcomments"], raw("civil", "civilcomments_v1.0.tar.gz")),
    }
    for s in ("train", "validation", "test"):
        paths[f"jevbench_{s}"] = fetch(hf_url(j, jr, f"data/civil_comments/{s}.jsonl"), raw("jevbench", f"cc_{s}.jsonl"))
    return paths


def cc_wilds(path, want_ids):
    """id -> metadata from the official WILDS CivilComments release (all_data_with_identities.csv, 448,000 comments of
    the Kaggle data with identity labels), restricted to want_ids (None = dates only, every id)."""
    csv.field_size_limit(1 << 30)
    out, dates = {}, Counter()
    with tarfile.open(path) as tf:
        f = tf.extractfile("./all_data_with_identities.csv")
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
            i = int(float(r["id"]))
            day = r["created_date"][:10]
            dates[day] += 1
            if want_ids is not None and i not in want_ids:
                continue
            fl = lambda k: float(r[k]) if r.get(k) not in (None, "") else None  # noqa: E731
            out[i] = {"created_date": day, "article_id": int(float(r["article_id"])),
                      "publication_id": int(float(r["publication_id"])), "wilds_split": r["split"],
                      "toxicity": fl("toxicity"), "toxicity_annotator_count": int(float(r["toxicity_annotator_count"])),
                      "identity_annotator_count": int(float(r["identity_annotator_count"] or 0)),
                      "identity": {k: fl(k) for k in CC_IDENT if fl(k)}}
    return out, dates


def cc_jevbench_rows(paths):
    return {s: [json.loads(line) for line in open(paths[f"jevbench_{s}"], encoding="utf-8")]
            for s in ("train", "validation", "test")}


def cc_jev_real_id(state, qid):
    """trainers/convert_jevbench.emit's record id for a jev-bench noul row (keys yes/no)."""
    return f"jevbench/civil_comments-{sha(state + qid + json.dumps(['yes', 'no'], sort_keys=True))}"


def cc_map(google, mirror, jb):
    """jev-bench civil_comments items -> google/civil_comments rows -> Kaggle comment ids (the mirror's id/text/toxicity/
    count), with the counts. Returns ({jb id: info}, report)."""
    gtext = {s: defaultdict(list) for s in google}
    for s, t in google.items():
        for i, x in enumerate(t["text"]):
            gtext[s][x].append(i)
    mt = defaultdict(list)
    for i, x, y, c in zip(mirror["id"], mirror["text"], mirror["toxicity"], mirror["toxicity_annotator_count"]):
        mt[x].append((int(i), float(y), int(c)))
    out, rep = {}, {}
    for split, rows in jb.items():
        c = Counter()
        for r in rows:
            text = json.loads(r["state"])
            tox = float(r["soft_label"])
            if split == "test":
                gi = int(r["id"].rsplit("/", 1)[1])
                ok = gi < len(google["test"]["text"]) and google["test"]["text"][gi] == text and \
                    abs(google["test"]["toxicity"][gi] - tox) < 1e-5
                grows = [("test", gi)] if ok else []
            else:
                grows = [("validation", i) for i in gtext["validation"].get(text, [])
                         if abs(google["validation"]["toxicity"][i] - tox) < 1e-5] or \
                        [("test", i) for i in gtext["test"].get(text, []) if abs(google["test"]["toxicity"][i] - tox) < 1e-5]
            ids = sorted({(i, n) for i, y, n in mt.get(text, []) if abs(y - tox) < 1e-5})
            c["google_row_found" if grows else "google_row_missing"] += 1
            c["google_split:" + (grows[0][0] if grows else "none")] += 1
            c["kaggle_id_unique" if len(ids) == 1 else "kaggle_id_ambiguous(duplicate text)" if ids else "kaggle_id_missing"] += 1
            counts = {n for _, n in ids}
            if len(counts) == 1 and next(iter(counts)) >= CC_MIN_RATERS:
                c[f"raters>={CC_MIN_RATERS}"] += 1
            out[r["id"]] = {"split": split, "google_rows": grows, "kaggle_ids": [i for i, _ in ids],
                            "counts": sorted(counts), "text": text, "toxicity": tox, "state": r["state"],
                            "label": r["label"], "question": r["question"]}
        rep[split] = {"items": len(rows), **dict(c)}
    return out, rep


def cc_build(a):
    import pyarrow.parquet as pq
    paths = cc_fetch()
    google = {s: pq.read_table(paths[f"google_{s}"]).to_pydict() for s in ("test", "validation")}
    mirror = pq.read_table(paths["kaggle_mirror_test"], columns=["id", "text", "toxicity",
                                                                 "toxicity_annotator_count"]).to_pydict()
    jb = cc_jevbench_rows(paths)
    log(f"civil: google test {len(google['test']['text'])}, validation {len(google['validation']['text'])}; Kaggle "
        f"mirror test {len(mirror['id'])}; jev-bench {', '.join(f'{k} {len(v)}' for k, v in jb.items())}")
    jmap, map_rep = cc_map(google, mirror, jb)
    # jev-bench's state encoding (json.dumps of the comment, ASCII-escaped or not): reproduce it exactly for pairing
    enc = None
    for asc in (True, False):
        if all(json.dumps(v["text"], ensure_ascii=asc) == v["state"] for v in jmap.values() if v["split"] == "test"):
            enc = asc
            break
    assert enc is not None, "jev-bench civil_comments states are not json.dumps(text)"
    jb_q = json.loads(jb["test"][0]["question"])
    assert all(r["question"] == jb["test"][0]["question"] for rs in jb.values() for r in rs)
    # Kaggle id per google TEST row (unique text+toxicity match only)
    mt = defaultdict(list)
    for i, x, y, c in zip(mirror["id"], mirror["text"], mirror["toxicity"], mirror["toxicity_annotator_count"]):
        mt[x].append((int(i), float(y), int(c)))
    jb_trval_norm = {norm(v["text"]) for v in jmap.values() if v["split"] != "test"}
    jb_test_by_row = {v["google_rows"][0][1]: k for k, v in jmap.items() if v["split"] == "test" and v["google_rows"]}
    pool, why = [], Counter()
    seen_text = set()
    for gi, (text, tox) in enumerate(zip(google["test"]["text"], google["test"]["toxicity"])):
        ids = sorted({(i, n) for i, y, n in mt.get(text, []) if abs(y - tox) < 1e-5})
        if len(ids) != 1:
            why["kaggle_id_not_unique_or_missing"] += 1
            continue
        kid, n = ids[0]
        if n < CC_MIN_RATERS:
            why[f"raters<{CC_MIN_RATERS}"] += 1
            continue
        nt = norm(text)
        if nt in jb_trval_norm:
            why["text_in_jevbench_train_or_val"] += 1
            continue
        if nt in seen_text or not nt:
            why["duplicate_text_within_pool"] += 1
            continue
        k = round(tox * n)
        if abs(tox * n - k) > 1e-4:
            why["non_integral_votes"] += 1
            continue
        seen_text.add(nt)
        pool.append({"gi": gi, "kid": kid, "n": n, "k": k, "text": text, "jb": jb_test_by_row.get(gi)})
    want = {p["kid"] for p in pool} | {i for v in jmap.values() for i in v["kaggle_ids"]}
    wilds, all_dates = cc_wilds(paths["wilds"], want)
    # ---- jev-bench dates: is a date-disjoint training split possible?
    jdates = defaultdict(list)
    for v in jmap.values():
        for i in v["kaggle_ids"][:1]:
            if i in wilds:
                jdates[v["split"]].append(wilds[i]["created_date"])
    alld = sorted(all_dates)
    jall = sorted(d for x in jdates.values() for d in x)
    days_with_jb = len(set(jall))
    date_rep = {"archive_dates": [alld[0], alld[-1]], "archive_days_with_comments": len(alld),
                "jevbench_items_with_a_date(WILDS subset)": {s: len(x) for s, x in jdates.items()},
                "jevbench_date_range": {s: [min(x), max(x)] for s, x in jdates.items() if x},
                "archive_days_holding_a_dated_jevbench_item(lower bound)": days_with_jb,
                "verdict": ("NO date-disjoint training split exists: the dated jev-bench items alone span "
                            f"{jall[0]} .. {jall[-1]}, i.e. the whole archive ({alld[0]} .. {alld[-1]}); the Civil "
                            "Comments splits are random, not by time. Training on Civil Comments is therefore refused "
                            "(user rule: disjoint by comment id AND by date), and the source stays EVAL ONLY.")}
    # ---- sample: every jev-bench test item with >= 10 raters (pairing), then a seeded random fill
    rng = random.Random(f"{SEED}:{CC_SET}")
    forced = [p for p in pool if p["jb"]]
    rest = [p for p in pool if not p["jb"]]
    rng.shuffle(rest)
    sel = forced + rest[: max(0, a.cap - len(forced))]
    gsub = {k: google["test"][k] for k in CC_SUB if k in google["test"]}
    recs = []
    for p in sel:
        state = json.dumps(p["text"], ensure_ascii=enc)
        q = {"type": "noul", "instructions": jb_q["instructions"],
             "criteria": {"yes": jb_q["criteria"]["true"], "no": jb_q["criteria"]["false"]}}
        labs = {CC_QID: {"yes": p["k"] / p["n"], "no": (p["n"] - p["k"]) / p["n"]}}   # exact shares (labels x n integral)
        w = wilds.get(p["kid"], {})
        jbi = jmap.get(p["jb"]) if p["jb"] else None
        meta = {"label_provenance": "crowd votes (Jigsaw Unintended Bias raters)", "label_method": CC_LABEL_NOTE,
                "n_raters": p["n"], "counts": {CC_QID: [p["k"], p["n"] - p["k"]]},
                "kaggle_id": p["kid"], "google_civil_comments_row": f"test/{p['gi']}",
                "stratum": "jevbench_test" if jbi else "random",
                "jevbench_id": p["jb"], "jev_real_id": cc_jev_real_id(state, p["jb"]) if jbi else None,
                "jevbench_hard_label": jbi["label"] if jbi else None,
                "subtype_fractions": {k: round(float(gsub[k][p["gi"]]), 6) for k in gsub},
                "created_date": w.get("created_date"), "article_id": w.get("article_id"),
                "publication_id": w.get("publication_id"),
                "identity_annotated": bool(w), "identity_annotator_count": w.get("identity_annotator_count"),
                "identity_fractions": w.get("identity"),
                "identity_mentions": sorted(k for k, v in (w.get("identity") or {}).items() if v >= 0.5) if w else None,
                "item_key": f"kaggle:{p['kid']}", "item_date": w.get("created_date"),
                "item_date_range": list(CC_ARCHIVE), "post_2025_07": False,
                "panel": "V (crowd vote shares, >= 10 raters)"}
        base = f"{CC_SOURCE}/article/{w['article_id']}" if w.get("article_id") else f"{CC_SOURCE}/comment/{p['kid']}"
        recs.append(eval_rec(CC_SET, "V", CC_SOURCE, state, {CC_QID: q}, labs, meta, base, LICENCE[CC_SET],
                             f"{CC_SOURCE}-{p['kid']}"))
    recs, gate = gate_eval(recs, [] if a.no_proxy else ["jev_data/mhs", "jev_data/go_emotions"],
                           fields=lambda r: [json.loads(r["state"])])
    stats = {"all": ga.vote_stats([r["meta"]["counts"][CC_QID] for r in recs]),
             "by_stratum": {s: ga.vote_stats([r["meta"]["counts"][CC_QID] for r in recs if r["meta"]["stratum"] == s])
                            for s in ("jevbench_test", "random")}}
    ident = defaultdict(list)
    for r in recs:
        for g in r["meta"]["identity_mentions"] or []:
            ident[g].append(r["meta"]["counts"][CC_QID])
        ident["(identity-annotated)" if r["meta"]["identity_annotated"] else "(not identity-annotated)"].append(
            r["meta"]["counts"][CC_QID])
    stats["by_identity_mention(>=0.5 of identity raters; bias slices)"] = {
        g: {"units": len(v), "mean_yes_share": round(statistics.fmean(c[0] / sum(c) for c in v), 4),
            "brier_noise_floor": ga.vote_stats(v).get("brier_noise_floor")} for g, v in sorted(ident.items()) if len(v) >= 30}
    jb_pair = sum(1 for r in recs if r["meta"]["stratum"] == "jevbench_test")
    man = {"domain": "V", "license": LICENCE[CC_SET], "license_evidence": LICENCE_EVIDENCE["civil_comments"],
           "eval_only": True, "verdict": "EVAL ONLY (no training corpus is written for Civil Comments)",
           "sources": {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in paths.items()},
           "pins": {k: PINS[k] for k in ("google_civil_comments", "kaggle_mirror", "wilds_civilcomments", "jevbench")},
           "jevbench_civil_comments": {
               "which_items": "train 8,000 + validation 500 are google/civil_comments VALIDATION rows (matched on text "
                              "+ toxicity); test 2,000 are google/civil_comments TEST rows (row index = the id suffix, "
                              "verified text + toxicity)",
               "mapping": map_rep, "dates": date_rep,
               "ext2_gate_coverage": "build_jev_ext2.ev_jevbench fingerprints jev-bench TEST only (data/<cfg>/test.jsonl): "
                                     "the 8,500 train + validation items are NOT in ext2's index",
               "state_encoding": f"json.dumps(comment, ensure_ascii={enc}) (reproduces all 2,000 test states)"},
           "pool": {"population": f"google/civil_comments TEST rows with a unique Kaggle id and >= {CC_MIN_RATERS} raters, "
                                  "minus any comment whose text is a jev-bench train/validation item",
                    "eligible": len(pool), "excluded": dict(why)},
           "sampling": {"cap_units": a.cap, "forced_jevbench_test_items": len(forced), "random_fill": len(sel) - len(forced),
                        "seed": f"{SEED}:{CC_SET}", "weights": "none needed: jev-bench's own test items are a random draw "
                        "of the same population"},
           "question": "jev-bench's civil_comments question verbatim (instructions + true/false definitions as noul "
                       "criteria yes/no), so jev-bench per-item predictions rescore against the vote shares "
                       f"(meta.jev_real_id; {jb_pair} paired items)",
           "dedup_vs_training_proxy": gate, "stats": stats,
           "dedup_vs_training_real": "PENDING: `build_jev_misc2.py gate --corpus ...` on a node"}
    write_eval(CC_SET, recs, man, notes={"DO_NOT_TRAIN.txt": "GDE-1 eval set: never place under a training directory.\n"},
               integral=True)


# =========================================================================== 2. MedQA (K side panel, eval only)
MQ_SET = "medqa_usmle"
MQ_SOURCE = "gde1/medqa_usmle"
MQ_QID = "answer"
MQ_INSTR = ("This is a multiple-choice question from MedQA (USMLE-style medical licensing exam questions collected from "
            "exam question banks). Exactly one answer option is correct. Which one is it?")
MQ_STEP = {"step1": "USMLE Step 1", "step2&3": "USMLE Step 2/3"}


def mmlu_stream():
    """MMLU test (an EVAL set, not training): only to report MedQA items that also appear there."""
    p = fetch(hf_url("cais/mmlu", "main", "all/test-00000-of-00001.parquet"), raw("proxy", "mmlu_all_test.parquet"))
    for r in _parquet_rows(p, ["question", "choices", "subject"]):
        yield f"eval/mmlu/{r['subject']}", {"question": r["question"], "choices": list(r["choices"] or [])}, [r["question"]]


def mq_records(rows):
    recs, rep = [], Counter()
    for r in rows:
        q, opts, ans = str(r.get("question") or "").strip(), r.get("options") or {}, str(r.get("answer_idx") or "")
        keys = sorted(opts)
        if not q or keys != ["A", "B", "C", "D"] or ans not in opts or not all(str(opts[k]).strip() for k in keys):
            rep["dropped/bad_row"] += 1
            continue
        if len({norm(opts[k]) for k in keys}) < 4:
            rep["dropped/duplicate_options"] += 1
            continue
        step = MQ_STEP.get(r.get("meta_info"), str(r.get("meta_info") or "USMLE"))
        state = f"{step}-style question:\n{q}"
        qs = {MQ_QID: choice_q(MQ_INSTR, {k: str(opts[k]).strip() for k in keys})}
        labs = {MQ_QID: gb.lab_dict(keys, [1.0 if k == ans else 0.0 for k in keys])}
        key = sha(norm(q))
        meta = {"label_provenance": "answer key (exam question bank; MedQA, Jin et al. 2020)",
                "panel": "K (knowledge QA side panel; never in the headline)", "step": r.get("meta_info"),
                "item_key": f"medqa:{key}", "item_date": "2020-09", "post_2025_07": False, "do_not_train": True,
                "public_release": "converter + manifest only (scraped question-bank text)"}
        recs.append(eval_rec(MQ_SET, "K", MQ_SOURCE, state, qs, labs, meta, f"{MQ_SOURCE}/{key}", LICENCE[MQ_SET],
                             f"{MQ_SOURCE}-{key}"))
        rep["records"] += 1
    return recs, dict(rep)


def mq_build(a):
    repo, rev = PINS["medqa"]
    pt = fetch(hf_url(repo, rev, "phrases_no_exclude_test.jsonl"), raw("medqa", "phrases_no_exclude_test.jsonl"))
    ptr = fetch(hf_url(repo, rev, "phrases_no_exclude_train.jsonl"), raw("medqa", "phrases_no_exclude_train.jsonl"))
    test = [json.loads(line) for line in open(pt, encoding="utf-8")]
    train = [json.loads(line) for line in open(ptr, encoding="utf-8")]
    recs, rep = mq_records(test)
    recs, ndup = gb.dedup_within(recs)
    rep["dropped/duplicate_question_within_test"] = ndup
    train_q = {norm(r["question"]) for r in train}
    n0 = len(recs)
    recs = [r for r in recs if norm(r["state"].split("\n", 1)[1]) not in train_q]
    rep["dropped/question_also_in_medqa_train"] = n0 - len(recs)
    gate = {"status": "skipped (--no-proxy)"}
    mmlu = {"status": "skipped (--no-proxy)"}
    if not a.no_proxy:
        recs, gate = gate_eval(recs, ["jev_ext2/medical/medmcqa", "jev_ext2/medical/pubmedqa"],
                               fields=lambda r: [r["state"].split("\n", 1)[1]])
        items = [{"key": r["meta"]["item_key"], "state": r["state"], "fields": [r["state"].split("\n", 1)[1]]} for r in recs]
        hits, read = ga.gate_items(items, mmlu_stream)
        mmlu = {"note": "MMLU is an EVAL set: overlaps are reported, nothing is dropped",
                "mmlu_states_read": sum(read.values()), "items_with_a_dropping_kind_match": sum(ga.dropping(h) for h in hits.values()),
                "by_subject": dict(Counter(h[0][0] for h in hits.values() if ga.dropping(h)))}
    ans = Counter(max(r["labels"][MQ_QID], key=r["labels"][MQ_QID].get) for r in recs)
    man = {"domain": "K", "license": LICENCE[MQ_SET], "license_evidence": LICENCE_EVIDENCE["medqa"], "eval_only": True,
           "verdict": "EVAL ONLY (K side panel): no data licence; never train", "pins": {"medqa": PINS["medqa"]},
           "sources": {"test": {"path": str(pt), "sha256": file_sha256(pt), "rows": len(test)},
                       "train(read only for the within-MedQA duplicate check)": {"path": str(ptr), "sha256": file_sha256(ptr),
                                                                               "rows": len(train)}},
           "selection": "the US 4-option TEST split (1,273 questions), the standard MedQA benchmark split; train is "
                        "never converted (other systems commonly fine-tune on it)",
           "build_report": rep, "answer_letter_hist": dict(sorted(ans.items())),
           "step_hist": dict(Counter(r["meta"]["step"] for r in recs)),
           "dedup_vs_training_proxy": gate, "overlap_with_mmlu_test(report only)": mmlu,
           "dedup_vs_training_real": "PENDING: `build_jev_misc2.py gate --corpus ...` on a node"}
    write_eval(MQ_SET, recs, man, sample_text=False, notes={
        "DO_NOT_TRAIN.txt": "MedQA (USMLE) eval set: never train on it (no data licence).\n",
        "PUBLISH_CONVERTER_ONLY.txt": "The questions were scraped from commercial USMLE question banks (MedQA paper, "
                                      "App. A). Publish build_jev_misc2.py + manifest.json sha256 values, not these files.\n"})


# =========================================================================== 3. FOLIO (K side panel, eval only)
FO_SET = "folio"
FO_SOURCE = "gde1/folio"
FO_QID = "conclusion"
FO_LABEL = {"True": "true", "False": "false", "Unknown": "unknown", "Uncertain": "unknown"}
FO_CRIT = {"true": "True: the conclusion follows from the premises (it holds whenever every premise holds).",
           "false": "False: the opposite of the conclusion follows from the premises.",
           "unknown": "Unknown: the premises do not settle it (the conclusion holds in some situations consistent with "
                      "the premises and fails in others)."}
FO_INSTR = ("Treat every premise as true and assume nothing else (read them as statements of first-order logic). Is "
            "the conclusion true, false, or unknown given the premises?")
FOL_TOKEN = re.compile(r"\s*(?:(∀|∃|¬|∧|∨|→|↔|⊕|≠|=|\(|\)|,)|(\w[\w’'\-]*(?:\.\w+)*)|(\.))")   # \w: Unicode names
FOL_NAME = re.compile(r"\w[\w’'\-]*(?:\.\w+)*")


class FolError(Exception):
    pass


def fol_tokens(s):
    s = str(s).replace("⟷", "↔").replace("​", "").strip()
    pos, out = 0, []
    while pos < len(s):
        m = FOL_TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise FolError(f"bad character at {s[pos:pos + 12]!r}")
        out.append(m.group(1) or m.group(2) or m.group(3))
        pos = m.end()
        while pos < len(s) and s[pos].isspace():
            pos += 1
    return out


class FolParser:
    """Recursive descent over FOLIO's FOL strings. Precedence (loosest first): ↔, → (right-assoc), ⊕, ∨, ∧, then ¬ /
    quantifiers / parentheses / atoms. A quantifier's scope extends as far right as possible. Terms are constants,
    bound variables or function applications; atoms are P(t, ...), a bare proposition, or t = t / t ≠ t."""

    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, want=None):
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise FolError(f"expected {want!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self):
        f = self.iff()
        while self.peek() == ".":
            self.eat()
        if self.peek() is not None:
            raise FolError(f"trailing tokens {self.t[self.i:self.i + 5]}")
        return f

    def iff(self):
        left = self.imp()
        while self.peek() == "↔":
            self.eat()
            left = ("iff", left, self.imp())
        return left

    def imp(self):
        left = self.xor()
        if self.peek() == "→":
            self.eat()
            return ("imp", left, self.imp())
        return left

    def _chain(self, op, tag, sub):
        left = sub()
        while self.peek() == op:
            self.eat()
            left = (tag, left, sub())
        return left

    def xor(self):
        return self._chain("⊕", "xor", self.or_)

    def or_(self):
        return self._chain("∨", "or", self.and_)

    def and_(self):
        return self._chain("∧", "and", self.unary)

    def unary(self):
        t = self.peek()
        if t == "¬":
            self.eat()
            return ("not", self.unary())
        if t in ("∀", "∃"):
            self.eat()
            vs = [self.name()]
            while self.peek() == ",":
                self.eat()
                vs.append(self.name())
            while self.peek() == ".":
                self.eat()
            body = self.iff()
            for v in reversed(vs):
                body = ("all" if t == "∀" else "ex", v, body)
            return body
        if t == "(":
            self.eat()
            f = self.iff()
            self.eat(")")
            return f
        return self.atom()

    def name(self):
        v = self.eat()
        if not FOL_NAME.fullmatch(v):
            raise FolError(f"expected a name, got {v!r}")
        return v

    def term(self):
        n = self.name()
        if self.peek() == "(":
            self.eat()
            args = [self.term()]
            while self.peek() == ",":
                self.eat()
                args.append(self.term())
            self.eat(")")
            return ("fn", n, args)
        return ("sym", n)

    def atom(self):
        left = self.term()
        if self.peek() in ("=", "≠"):
            op = self.eat()
            e = ("eq", left, self.term())
            return ("not", e) if op == "≠" else e
        return ("pred", left[1], left[2]) if left[0] == "fn" else ("prop", left[1])


def fol_parse(s):
    return FolParser(fol_tokens(s)).parse()


class FolSig:
    def __init__(self):
        import z3
        self.z3, self.U, self.syms = z3, z3.DeclareSort("U"), {}

    def get(self, kind, name, arity=0):
        z3 = self.z3
        k = (kind, name, arity)
        if k not in self.syms:
            nm = f"{kind}_{name}_{arity}"
            if kind == "c":
                self.syms[k] = z3.Const(nm, self.U)
            elif kind == "p0":
                self.syms[k] = z3.Bool(nm)
            elif kind == "P":
                self.syms[k] = z3.Function(nm, *([self.U] * arity), z3.BoolSort())
            else:
                self.syms[k] = z3.Function(nm, *([self.U] * arity), self.U)
        return self.syms[k]


def fol_z3(f, sig, env=None):
    z3, env = sig.z3, env or {}

    def term(t):
        if t[0] == "sym":
            return env[t[1]] if t[1] in env else sig.get("c", t[1])
        return sig.get("f", t[1], len(t[2]))(*[term(x) for x in t[2]])
    k = f[0]
    if k == "not":
        return z3.Not(fol_z3(f[1], sig, env))
    if k in ("and", "or", "imp", "iff", "xor"):
        a, b = fol_z3(f[1], sig, env), fol_z3(f[2], sig, env)
        return {"and": z3.And, "or": z3.Or, "imp": z3.Implies, "xor": z3.Xor}[k](a, b) if k != "iff" else a == b
    if k in ("all", "ex"):
        v = z3.Const(f"v_{f[1]}_{len(env)}", sig.U)
        body = fol_z3(f[2], sig, {**env, f[1]: v})
        return z3.ForAll([v], body) if k == "all" else z3.Exists([v], body)
    if k == "pred":
        return sig.get("P", f[1], len(f[2]))(*[term(x) for x in f[2]])
    if k == "prop":
        if f[1] in env:
            raise FolError(f"variable {f[1]!r} used as a formula")
        return sig.get("p0", f[1])
    if k == "eq":
        return term(f[1]) == term(f[2])
    raise FolError(f"unknown node {k}")


def fol_verdict(premises, conclusion, timeout_ms=20000):
    """('true' | 'false' | 'unknown' | 'inconsistent' | 'solver_unknown' | 'parse_error', detail): whether the premises
    entail the conclusion, its negation, or neither, by z3 over an uninterpreted domain."""
    try:
        import z3
    except ImportError:
        return "no_z3", "z3-solver not installed"
    try:
        sig = FolSig()
        zp = [fol_z3(fol_parse(p), sig) for p in premises]
        zc = fol_z3(fol_parse(conclusion), sig)
    except (FolError, z3.Z3Exception) as e:
        return "parse_error", str(e)[:160]

    def check(extra):
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(*zp)
        s.add(extra)
        return s.check()
    r_neg, r_pos = check(z3.Not(zc)), check(zc)
    if r_neg == z3.unsat and r_pos == z3.unsat:
        return "inconsistent", "premises unsatisfiable"
    if r_neg == z3.unsat:
        return "true", ""
    if r_pos == z3.unsat:
        return "false", ""
    if r_neg == z3.sat and r_pos == z3.sat:
        return "unknown", ""
    return "solver_unknown", f"{r_neg}/{r_pos}"


def as_list(x):
    if isinstance(x, list):
        return [str(s).strip() for s in x if str(s).strip()]
    return [s.strip() for s in str(x or "").split("\n") if s.strip()]


def fo_records(rows, version):
    recs, rep, mism = [], Counter(), []
    for i, r in enumerate(rows):
        prem, pf = as_list(r.get("premises")), as_list(r.get("premises-FOL"))
        concl, cf_ = str(r.get("conclusion") or "").strip(), str(r.get("conclusion-FOL") or "").strip()
        lab = FO_LABEL.get(str(r.get("label")))
        rep["rows"] += 1
        if not prem or not concl or lab is None:
            rep["dropped/bad_row"] += 1
            continue
        if not cf_ or not pf:
            rep["dropped/no_fol_annotation"] += 1
            continue
        verdict, detail = fol_verdict(pf, cf_)
        rep[f"z3/{verdict}"] += 1
        if verdict != lab:
            rep["dropped/z3_disagrees_or_cannot_decide"] += 1
            if len(mism) < 12:
                mism.append({"row": i, "label": lab, "z3": verdict, "detail": detail, "conclusion": concl[:160],
                             "conclusion_fol": cf_[:160]})
            continue
        story = sha("\n".join(norm(p) for p in prem))
        state = "Premises:\n" + "\n".join(f"{j + 1}. {p}" for j, p in enumerate(prem)) + f"\n\nConclusion: {concl}"
        keys = list(FO_CRIT)
        rng = random.Random(f"{SEED}:folio:{sha(state)}")
        order = keys[:]
        rng.shuffle(order)
        qs = {FO_QID: choice_q(FO_INSTR, {k: FO_CRIT[k] for k in order})}
        labs = {FO_QID: gb.lab_dict(order, [1.0 if k == lab else 0.0 for k in order])}
        meta = {"label_provenance": "exact (first-order logic: the authors' label, re-proved here with z3 from the "
                                    "authors' FOL annotation; items the prover does not confirm are dropped)",
                "panel": "K (logic side panel; never in the headline)", "folio_version": version,
                "story_id": r.get("story_id"), "example_id": r.get("example_id"), "z3": verdict,
                "item_key": f"folio:{sha(state)}", "item_date": "2022-09", "post_2025_07": False}
        recs.append(eval_rec(FO_SET, "K", FO_SOURCE, state, qs, labs, meta, f"{FO_SOURCE}/story/{story}",
                             LICENCE["folio_v2" if version.startswith("v2") else FO_SET], f"{FO_SOURCE}-{sha(state)}"))
    return recs, dict(rep), mism


def fo_build(a):
    if a.folio_v2:
        repo, rev = PINS["folio_hf"]
        files = {s: fetch(hf_url(repo, rev, f"folio_v2_{s}.jsonl"), raw("folio", f"folio_v2_{s}.jsonl"), token=True)
                 for s in ("train", "validation")}
        version = "v2 (yale-nlp/FOLIO, HF; UNTESTED converter path)"
    else:
        repo, rev = PINS["folio_gh"]
        files = {s: fetch(f"https://raw.githubusercontent.com/{repo}/{rev}/data/v0.0/folio-{s}.jsonl",
                          raw("folio", f"folio-{s}.jsonl")) for s in ("train", "validation")}
        version = "v0.0 (github.com/Yale-LILY/FOLIO)"
    rows = {s: [json.loads(line) for line in open(p, encoding="utf-8")] for s, p in files.items()}
    recs, rep, mism = [], {}, {}
    for s, rs in rows.items():
        rr, rep[s], mism[s] = fo_records(rs, version)
        for r in rr:
            r["meta"]["folio_split"] = s
        recs += rr
    recs, ndup = gb.dedup_within(recs)
    man = {"domain": "K", "license": LICENCE["folio_v2" if a.folio_v2 else FO_SET], "license_evidence": LICENCE_EVIDENCE["folio"],
           "eval_only": True, "verdict": "EVAL (K side panel). The licence would allow training; kept as a small exact-label "
           "side panel as suggested", "version": version, "pins": {"folio": (repo, rev)},
           "sources": {s: {"path": str(p), "sha256": file_sha256(p), "rows": len(rows[s])} for s, p in files.items()},
           "label_check": "z3 (uninterpreted domain) over the authors' premises-FOL and conclusion-FOL: True = premises "
                          "entail the conclusion, False = they entail its negation, Unknown = both consistent; a record "
                          "is kept only when z3's verdict equals the released label",
           "build_report": rep, "z3_disagreements(examples)": mism, "within_set_duplicates_dropped": ndup,
           "label_hist": dict(Counter(max(r["labels"][FO_QID], key=r["labels"][FO_QID].get) for r in recs)),
           "dedup_vs_training": "no training source of ours carries FOLIO or FOL story text (checked: build_jev_data, "
                                "ext, ext2, ext3, misc2 source lists); node gate: `gate --corpus ...`"}
    write_eval(FO_SET, recs, man, notes={"DO_NOT_TRAIN.txt": "GDE-1 eval side panel: never place under a training "
                                                            "directory.\n"})


# =========================================================================== 4. Lichess puzzles (training slice + T panel)
LI_SET = "lichess_puzzles"
LI_EVAL_SOURCE = "gde1/lichess_puzzles"
LI_TRAIN_SOURCE = "lichess"
LI_QID = "best_move"
LI_BINS = [0, 1000, 1400, 1800, 2200, 2600, 10 ** 6]          # puzzle-rating strata
LI_EVAL_PER_BIN = 1000                                        # 6 bins -> 6,000 eval units
LI_TRAIN_PER_BIN = 834                                        # -> ~5,000 training units (cap 5,000)
LI_TRAIN_CAP = 5000
LI_MIN_PLAYS, LI_MAX_RD, LI_MIN_POP = 100, 90, 50              # well-measured, well-liked puzzles only
LI_N_DISTRACTORS = 3
LI_PIECE = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}


def li_bin(rating):
    for lo, hi in zip(LI_BINS, LI_BINS[1:]):
        if lo <= rating < hi:
            return f"{lo}-{hi - 1}" if hi < 10 ** 6 else f">={lo}"
    return None


def li_game(url):
    m = re.search(r"lichess\.org/([A-Za-z0-9]{8})", url or "")
    return m.group(1) if m else (url or "")


def li_side(gid):
    """Train / eval assignment by GAME (not puzzle): one game never feeds both."""
    return "eval" if uhash("lichess-split", gid) < 0.5 else "train"


def li_move_text(board, m):
    """SAN without the check / mate suffix (a '#' would give the answer away), plus from-to squares."""
    import chess
    san = board.san(m).rstrip("+#")
    return f"{san} ({chess.square_name(m.from_square)}-{chess.square_name(m.to_square)})"


def li_pieces(board, color):
    import chess
    parts = []
    for pt in (6, 5, 4, 3, 2, 1):
        sq = sorted(chess.square_name(s) for s in board.pieces(pt, color))
        if sq:
            parts.append(f"{LI_PIECE[pt]}{'s' if len(sq) > 1 else ''} {' '.join(sq)}")
    return ", ".join(parts)


def li_convert(row):
    """One puzzle row -> (state, question, label dict, meta) or (None, reason)."""
    import chess
    try:
        board = chess.Board(row["FEN"])
    except ValueError:
        return None, "bad_fen"
    mv = row["Moves"].split()
    if len(mv) < 2:
        return None, "short_solution"
    m0 = chess.Move.from_uci(mv[0])
    if m0 not in board.legal_moves:
        return None, "first_move_illegal"
    who_moved = "White" if board.turn == chess.WHITE else "Black"
    last = board.san(m0)
    board.push(m0)
    sol = chess.Move.from_uci(mv[1])
    if sol not in board.legal_moves:
        return None, "solution_illegal"

    def mates(m):
        board.push(m)
        r = board.is_checkmate()
        board.pop()
        return r
    others = [m for m in board.legal_moves if m != sol and not mates(m)]
    if not others:
        return None, "no_non_mating_alternative"
    rng = random.Random(f"{SEED}:lichess:{row['PuzzleId']}")
    forcing = sorted((m for m in others if board.is_capture(m) or board.gives_check(m)), key=lambda m: m.uci())
    quiet_ = sorted((m for m in others if m not in forcing), key=lambda m: m.uci())
    rng.shuffle(forcing)
    picks = forcing[:2]
    rest = forcing[2:] + quiet_
    rng.shuffle(rest)
    picks += rest[: LI_N_DISTRACTORS - len(picks)]
    opts = [sol] + picks
    rng.shuffle(opts)
    keys = [chr(65 + i) for i in range(len(opts))]
    to_move = "White" if board.turn == chess.WHITE else "Black"
    crit = {k: li_move_text(board, m) for k, m in zip(keys, opts)}
    if len(set(crit.values())) != len(crit):
        return None, "duplicate_option_text"
    diagram = "\n".join(f"{8 - i} {line}" for i, line in enumerate(str(board).split("\n"))) + "\n  a b c d e f g h"
    state = (f"A chess position from a real game played on lichess.org (a Lichess puzzle). {to_move} to move.\n"
             f"{who_moved} has just played {last.rstrip('+#')} ({chess.square_name(m0.from_square)}-"
             f"{chess.square_name(m0.to_square)}).\n"
             f"FEN: {board.fen()}\n"
             f"Board (White in capitals, Black in lower case, rank 8 at the top):\n{diagram}\n"
             f"White: {li_pieces(board, chess.WHITE)}\nBlack: {li_pieces(board, chess.BLACK)}")
    instr = (f"Which move should {to_move} play? This position is a Lichess puzzle: exactly one of the moves below is the "
             f"puzzle's solution (the only clearly best move, found by the Lichess puzzle generator with Stockfish); the "
             f"others are legal moves that are clearly worse. Which move is the solution?")
    lab = gb.lab_dict(keys, [1.0 if m == sol else 0.0 for m in opts])
    rating = int(row["Rating"])
    meta = {"puzzle_id": row["PuzzleId"], "rating": rating, "rating_bin": li_bin(rating),
            "rating_deviation": int(row["RatingDeviation"]), "popularity": int(row["Popularity"]),
            "nb_plays": int(row["NbPlays"]), "themes": row["Themes"].split(), "opening_tags": row["OpeningTags"].split(),
            "game_id": li_game(row["GameUrl"]), "n_legal_moves": board.legal_moves.count(), "n_options": len(keys),
            "solution_is_capture": board.is_capture(sol), "solution_gives_check": board.gives_check(sol),
            "solution_mates": mates(sol), "solution_uci": sol.uci(), "fen_after_first_move": board.fen()}
    return (state, choice_q(instr, crit), lab, meta), "ok"


def li_select(path, per_side, seed=SEED, limit=0):
    """Stream the puzzle CSV once; keep, per (side, rating bin), the `per_side[side]` eligible puzzles with the
    smallest seeded hash (a uniform sample, deterministic). Returns ({(side, bin): [rows]}, counters)."""
    import zstandard
    heaps, rep = defaultdict(list), Counter()
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", newline="")
        for n, row in enumerate(csv.DictReader(text)):
            if limit and n >= limit:
                break
            rep["rows"] += 1
            try:
                rating, rd, pop, plays = (int(row[k]) for k in ("Rating", "RatingDeviation", "Popularity", "NbPlays"))
            except (TypeError, ValueError):
                rep["dropped/bad_numbers"] += 1
                continue
            if plays < LI_MIN_PLAYS or rd > LI_MAX_RD or pop < LI_MIN_POP:
                rep["dropped/quality_filter"] += 1
                continue
            side, b = li_side(li_game(row["GameUrl"])), li_bin(rating)
            rep[f"eligible/{side}"] += 1
            k = per_side[side]
            pri = -uhash(seed, "lichess", row["PuzzleId"])
            h = heaps[(side, b)]
            if len(h) < k:
                heapq.heappush(h, (pri, row["PuzzleId"], row))
            elif pri > h[0][0]:
                heapq.heapreplace(h, (pri, row["PuzzleId"], row))
    return {k: [x[2] for x in sorted(v, key=lambda x: -x[0])] for k, v in heaps.items()}, rep


def li_build(a):
    path = fetch(PINS["lichess"], raw("lichess", "lichess_db_puzzle.csv.zst"))
    per = {"eval": int(LI_EVAL_PER_BIN * 1.15), "train": int(LI_TRAIN_PER_BIN * 1.25)}
    log("lichess: streaming the puzzle database (one pass) ...")
    sel, rep = li_select(path, per, limit=a.limit_rows)
    ev, tr, conv = [], [], Counter()
    seen_fen = set()
    for (side, b), rows in sorted(sel.items()):
        want = LI_EVAL_PER_BIN if side == "eval" else LI_TRAIN_PER_BIN
        got = 0
        for row in rows:
            if got >= want:
                break
            out, why = li_convert(row)
            conv[f"{side}/{why}"] += 1
            if out is None:
                continue
            state, q, lab, meta = out
            fk = " ".join(meta["fen_after_first_move"].split()[:4])
            if fk in seen_fen:
                conv[f"{side}/duplicate_position"] += 1
                continue
            seen_fen.add(fk)
            if side == "eval":
                meta.update({"label_provenance": "computed (the puzzle's stored first solution move; Lichess puzzle "
                                                 "generator, Stockfish NNUE); distractors: legal non-mating moves",
                             "panel": "T (trained-source side panel: misc2 trains on puzzles from OTHER games)",
                             "item_key": f"lichess:{row['PuzzleId']}", "item_date": None, "post_2025_07": False,
                             "source_snapshot": "lichess_db_puzzle.csv.zst (Last-Modified 2026-09-09)"})
                ev.append(eval_rec(LI_SET, "T", LI_EVAL_SOURCE, state, {LI_QID: q}, {LI_QID: lab}, meta,
                                   f"{LI_EVAL_SOURCE}/game/{meta['game_id']}", LICENCE["lichess"],
                                   f"{LI_EVAL_SOURCE}-{row['PuzzleId']}"))
            else:
                meta["label_provenance"] = "computed (Lichess puzzle solution, Stockfish)"
                tr.append(train_rec(LI_TRAIN_SOURCE, state, {LI_QID: q}, {LI_QID: lab}, meta, meta["game_id"],
                                    LICENCE["lichess"]))
            got += 1
        conv[f"{side}/kept_bin_{b}"] = got
    assert not ({r["meta"]["game_id"] for r in ev} & {r["meta"]["game_id"] for r in tr}), "game in both sides"
    write_cands("lichess", tr, {"build_report": dict(rep), "conversion": dict(conv)})
    gate = {"structural": "eval and training puzzles come from disjoint GAMES (sha1 of the game id) and every position "
                          "(FEN board + side + castling + e.p.) is kept once across both sides; a text gate is not used "
                          "between them (the state template would match every pair)",
            "games_in_both": 0, "eval_games": len({r["meta"]["game_id"] for r in ev}),
            "train_games": len({r["meta"]["game_id"] for r in tr})}
    stats = {}
    for name, rs in (("eval", ev), ("train_candidates", tr)):
        stats[name] = {
            "units": len(rs), "by_rating_bin": dict(Counter(r["meta"]["rating_bin"] for r in rs)),
            "n_options_hist": dict(Counter(r["meta"]["n_options"] for r in rs)),
            "solution_letter_hist": dict(Counter(max(r["labels"][LI_QID], key=r["labels"][LI_QID].get) for r in rs)),
            "solution_is_capture": round(statistics.fmean(r["meta"]["solution_is_capture"] for r in rs), 4) if rs else None,
            "solution_gives_check": round(statistics.fmean(r["meta"]["solution_gives_check"] for r in rs), 4) if rs else None,
            "solution_mates": round(statistics.fmean(r["meta"]["solution_mates"] for r in rs), 4) if rs else None,
            "top_themes": Counter(t for r in rs for t in r["meta"]["themes"]).most_common(12)}
    man = {"domain": "T", "license": LICENCE["lichess"], "license_evidence": LICENCE_EVIDENCE["lichess"],
           "eval_only": True, "verdict": "T side panel (eval) + a small training slice from other games",
           "source": {"url": PINS["lichess"], "path": str(path), "sha256": file_sha256(path)},
           "selection": {"quality_filter": f"NbPlays >= {LI_MIN_PLAYS}, RatingDeviation <= {LI_MAX_RD}, Popularity >= "
                                           f"{LI_MIN_POP}", "strata": LI_BINS[:-1], "eval_per_bin": LI_EVAL_PER_BIN,
                         "split": "by GAME (sha1 of the lichess game id): eval games and training games are disjoint; "
                                  "identical positions (FEN board + side + castling + e.p.) are kept once",
                         "sample": "per stratum, the eligible puzzles with the smallest seeded hash", "seed": SEED},
           "question": "state = the position AFTER the opponent's move (the CSV's FEN + Moves[0]); gold = Moves[1]; "
                       f"options = the solution + {LI_N_DISTRACTORS} legal alternatives (up to 2 captures / checks, the "
                       "rest random; every move that checkmates is excluded, since Lichess accepts any mate in a "
                       "mate-in-one), seeded by the puzzle id, shuffled; SAN shown without +/# suffixes",
           "build_report": dict(rep), "conversion": dict(conv), "stats": stats, "dedup_vs_own_training": gate,
           "no_stockfish": "no engine was run; labels are the stored solutions"}
    write_eval(LI_SET, ev, man, notes={"DO_NOT_TRAIN.txt": "GDE-1 eval side panel: never place under a training "
                                                          "directory.\n"})


# =========================================================================== 5. Folktables / ACS PUMS (training + T panel)
FT_SET = "folktables_acs"
FT_EVAL_SOURCE = "gde1/folktables_acs"
FT_TRAIN_YEAR, FT_EVAL_YEAR = 2018, 2024
FT_TRAIN_STATES = ["AL", "CO", "CT", "IA", "NM", "OR"]
FT_HELDOUT_STATES = ["GA", "MN", "NV", "ME"]
FT_FIPS = {"AL": "01", "CO": "08", "CT": "09", "IA": "19", "NM": "35", "OR": "41", "GA": "13", "MN": "27", "NV": "32",
           "ME": "23"}
FT_TRAIN_PER_TASK = 4000          # 5 tasks -> 20,000 training units
FT_EVAL_PER_STRATUM = 800         # per task: 800 held-out-state + 800 training-state units (2024) -> 8,000 in all


def _num(r, k):
    v = r.get(k, "")
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def _gt(r, k, x):
    v = _num(r, k)
    return v is not None and v > x


FT_TASKS = {
    "acs_income": dict(
        features=["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "REL", "WKHP", "SEX", "RAC1P"],
        eligible=lambda r: _gt(r, "AGEP", 16) and _gt(r, "PINCP", 100) and _gt(r, "WKHP", 0) and _gt(r, "PWGTP", 0.5),
        target=lambda r: _num(r, "PINCP") > 50000,
        question="Was this person's total personal income in the 12 months before the survey more than $50,000 (as "
                 "reported, in that year's dollars, not inflation-adjusted)?",
        yes="Yes: total personal income above $50,000.", no="No: $50,000 or less.", folktables="ACSIncome"),
    "acs_employment": dict(
        features=["AGEP", "SCHL", "MAR", "REL", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "DEAR", "DEYE",
                  "DREM", "SEX", "RAC1P"],
        eligible=lambda r: _num(r, "AGEP") is not None and _num(r, "AGEP") >= 16 and _num(r, "ESR") is not None,
        target=lambda r: _num(r, "ESR") == 1,
        question="At the time of the survey, was this person a civilian who was employed and at work (not unemployed, "
                 "not outside the labour force, not in the armed forces, and not temporarily away from a job)?",
        yes="Yes: a civilian, employed and at work.", no="No.", folktables="ACSEmployment"),
    "acs_public_coverage": dict(
        features=["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "DEAR", "DEYE",
                  "DREM", "PINCP", "ESR", "ST", "FER", "RAC1P"],
        eligible=lambda r: _num(r, "AGEP") is not None and _num(r, "AGEP") < 65 and _num(r, "PINCP") is not None
        and _num(r, "PINCP") <= 30000,
        target=lambda r: _num(r, "PUBCOV") == 1,
        question="Does this person have public health insurance coverage (for example Medicaid, Medicare, or VA health "
                 "care)?",
        yes="Yes: covered by public health insurance.", no="No public health coverage.", folktables="ACSPublicCoverage"),
    "acs_mobility": dict(
        features=["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "CIT", "MIL", "ANC", "NATIVITY", "REL", "DEAR", "DEYE",
                  "DREM", "RAC1P", "GCL", "COW", "ESR", "WKHP", "JWMNP", "PINCP"],
        eligible=lambda r: _gt(r, "AGEP", 18) and not _gt(r, "AGEP", 34) and _num(r, "MIG") is not None,
        target=lambda r: _num(r, "MIG") == 1,
        question="Did this person live in the same house one year before the survey?",
        yes="Yes: same house one year earlier.", no="No: moved within the past year.", folktables="ACSMobility"),
    "acs_travel_time": dict(
        features=["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "MIG", "REL", "RAC1P", "PUMA", "ST", "CIT", "OCCP", "JWTR",
                  "POWPUMA", "POVPIP"],
        eligible=lambda r: _gt(r, "AGEP", 16) and _gt(r, "PWGTP", 0.5) and _num(r, "ESR") == 1 and _num(r, "JWMNP") is not None,
        target=lambda r: _num(r, "JWMNP") > 20,
        question="Is this person's usual one-way travel time to work more than 20 minutes?",
        yes="Yes: more than 20 minutes.", no="No: 20 minutes or less.", folktables="ACSTravelTime"),
}
FT_FIELD = {"AGEP": "Age", "COW": "Class of worker", "SCHL": "Educational attainment", "MAR": "Marital status",
            "OCCP": "Occupation", "POBP": "Place of birth", "REL": "Relationship to the household's reference person",
            "WKHP": "Usual hours worked per week (past 12 months)", "SEX": "Sex", "RAC1P": "Race",
            "DIS": "Disability", "ESP": "Employment status of parents", "CIT": "Citizenship status",
            "MIG": "Mobility status (lived here 1 year ago)", "MIL": "Military service", "ANC": "Ancestry",
            "NATIVITY": "Nativity", "DEAR": "Hearing difficulty", "DEYE": "Vision difficulty",
            "DREM": "Cognitive difficulty", "PINCP": "Total personal income (past 12 months, survey-year dollars)",
            "ESR": "Employment status", "ST": "State of residence", "FER": "Gave birth to a child within the past 12 months",
            "GCL": "Grandparent living with own grandchildren", "JWMNP": "Travel time to work",
            "JWTR": "Means of transportation to work", "PUMA": "Public use microdata area (PUMA) of residence",
            "POWPUMA": "Place-of-work PUMA", "POVPIP": "Income-to-poverty ratio"}
FT_ALIASES = {"REL": ("RELP", "RELSHIPP"), "JWTR": ("JWTR", "JWTRNS"), "ST": ("ST", "STATE")}
FT_EXTRA = ["SERIALNO", "SPORDER", "PWGTP", "HISP", "PUBCOV", "ESR", "PINCP", "MIG", "JWMNP", "AGEP", "SEX", "RAC1P"]
FT_RACE = {"1": "White alone", "2": "Black alone", "3": "American Indian or Alaska Native", "4": "American Indian or "
           "Alaska Native", "5": "American Indian or Alaska Native", "6": "Asian alone", "7": "Native Hawaiian or Pacific "
           "Islander alone", "8": "Some other race alone", "9": "Two or more races"}


def acs_dictionary(year):
    p = fetch(PINS["acs_dict"].format(year=year), raw("acs", f"PUMS_Data_Dictionary_{year}.csv"))
    data = p.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    names, vals = {}, defaultdict(list)
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 5 and row[0] == "NAME":
            names.setdefault(row[1], row[4])
        elif len(row) >= 7 and row[0] == "VAL" and (row[4], row[5]) not in {(a, b) for a, b, _ in vals[row[1]]}:
            vals[row[1]].append((row[4], row[5], row[6]))
    return names, vals, p


def acs_label(vals, var, v):
    entries = vals.get(var, [])
    if v in ("", None):
        return next((lab for lo, _, lab in entries if lo.strip().lower().startswith("b")), "N/A")
    try:
        iv = int(float(v))
    except ValueError:
        iv = None
    for lo, hi, lab in entries:
        if lo.strip().lower().startswith("b"):
            continue
        try:
            if iv is not None and int(lo) <= iv <= int(hi):
                return lab
        except ValueError:
            if lo == v:
                return lab
    return None


def ft_value(var, v, vals, col):
    if var in ("AGEP", "WKHP", "PINCP", "JWMNP", "POVPIP") and v not in ("", None):
        n = int(float(v))
        if var == "AGEP":
            return "under 1 year" if n == 0 else f"{n} years"
        if var == "WKHP":
            return "99 or more hours" if n >= 99 else f"{n} hours"
        if var == "PINCP":
            return "none" if n == 0 else (f"${n:,}" if n > 0 else f"a loss of ${-n:,}")
        if var == "JWMNP":
            return f"{n} minutes"
        return "501% or more" if n >= 501 else f"{n}%"
    if var in ("PUMA", "POWPUMA"):
        return acs_label(vals, col, v) if v in ("", None) else f"code {str(v).zfill(5)}"
    lab = acs_label(vals, col, v)
    return lab if lab is not None else f"code {v}"


def ft_state(task, r, vals, cols, year):
    lines = [f"Person record from the US Census Bureau's American Community Survey (ACS), {year} 1-year Public Use "
             f"Microdata Sample."]
    for var in FT_TASKS[task]["features"]:          # r is keyed by canonical name; cols[var] = the dictionary's name
        lines.append(f"- {FT_FIELD[var]}: {ft_value(var, r.get(var, ''), vals, cols[var])}")
    return "\n".join(lines)


def ft_columns(header):
    cols = {}
    for var in {f for t in FT_TASKS.values() for f in t["features"]} | set(FT_EXTRA):
        alts = FT_ALIASES.get(var, (var,))
        hit = next((c for c in alts if c in header), None)
        if hit is None:
            raise SystemExit(f"ACS file lacks {var} (tried {alts})")
        cols[var] = hit
    return cols


def ft_rows(year, st):
    """(header column map, row iterator) of one state's person file, streamed from the zip (canonical names: REL, JWTR,
    ST map to RELP / RELSHIPP, JWTR / JWTRNS, ST / STATE by year)."""
    p = fetch(PINS["acs"].format(year=year, st=st.lower()), raw("acs", f"{year}_csv_p{st.lower()}.zip"))
    zf = zipfile.ZipFile(p)
    member = next(n for n in zf.namelist() if n.lower().startswith("psam_p") and n.lower().endswith(".csv"))
    fh = io.TextIOWrapper(zf.open(member), encoding="utf-8", newline="")
    rd = csv.DictReader(fh)
    cols = ft_columns(rd.fieldnames)
    return p, cols, rd


def ft_canon(r, cols):
    return {var: r.get(c, "") for var, c in cols.items()}


def ft_build(a):
    plan = [(FT_TRAIN_YEAR, st, "train") for st in FT_TRAIN_STATES] + \
           [(FT_EVAL_YEAR, st, "eval_heldout_state") for st in FT_HELDOUT_STATES] + \
           [(FT_EVAL_YEAR, st, "eval_training_state") for st in FT_TRAIN_STATES]
    want = {"train": int(FT_TRAIN_PER_TASK * 1.2), "eval_heldout_state": FT_EVAL_PER_STRATUM,
            "eval_training_state": FT_EVAL_PER_STRATUM}
    heaps, files, rep, dicts = defaultdict(list), {}, Counter(), {}
    for year in (FT_TRAIN_YEAR, FT_EVAL_YEAR):
        dicts[year] = acs_dictionary(year)
    for year, st, stratum in plan:
        p, cols, rd = ft_rows(year, st)
        files[f"{year}/{st}"] = {"url": PINS["acs"].format(year=year, st=st.lower()), "sha256": file_sha256(p),
                                 "columns": {k: v for k, v in cols.items() if k != v}}
        n = 0
        for r0 in rd:
            n += 1
            r = ft_canon(r0, cols)
            for task, T in FT_TASKS.items():
                if not T["eligible"](r):
                    continue
                rep[f"{year}/{stratum}/{task}/eligible"] += 1
                pri = -uhash(SEED, task, year, st, r["SERIALNO"], r["SPORDER"])
                h = heaps[(task, stratum)]
                item = (pri, f"{year}:{st}:{r['SERIALNO']}:{r['SPORDER']}", (year, st, r))
                if len(h) < want[stratum]:
                    heapq.heappush(h, item)
                elif pri > h[0][0]:
                    heapq.heapreplace(h, item)
            if a.limit_rows and n >= a.limit_rows:
                break
        rep[f"{year}/{st}/persons_read"] = n
        log(f"folktables: {year} {st} ({stratum}): {n} persons")
    ev, tr = [], []
    for (task, stratum), h in sorted(heaps.items()):
        T = FT_TASKS[task]
        for _, key, (year, st, r) in sorted(h, key=lambda x: -x[0]):
            names, vals, _ = dicts[year]
            cols_y = {var: (next(c for c in FT_ALIASES[var] if c in vals) if var in FT_ALIASES else var)
                      for var in FT_FIELD}
            state = ft_state(task, r, vals, cols_y, year)
            y = bool(T["target"](r))
            q = noul_q(T["question"])
            q["criteria"] = {"yes": T["yes"], "no": T["no"]}
            lab = gb.noul_lab(1.0 if y else 0.0)
            hisp = acs_label(vals, "HISP", r["HISP"]) or ""
            demo = {"sex": acs_label(vals, "SEX", r["SEX"]), "race": FT_RACE.get(str(int(float(r["RAC1P"])))),
                    "hispanic": None if not r["HISP"] else int(float(r["HISP"])) != 1, "hispanic_origin": hisp,
                    "age": int(float(r["AGEP"]))}
            meta = {"task": task, "folktables_task": T["folktables"], "survey_year": year, "state": st,
                    "serialno": r["SERIALNO"], "sporder": r["SPORDER"], "person_weight": int(float(r["PWGTP"] or 0)),
                    "stratum": stratum, "demographics": demo,
                    "label_provenance": "recorded survey response (ACS PUMS); folktables task definition"}
            hh = f"{year}:{st}:{r['SERIALNO']}"
            if stratum == "train":
                tr.append(train_rec(f"folktables/{task}", state, {task: q}, {task: lab}, meta, hh, LICENCE["folktables"]))
            else:
                meta.update({"panel": "T (trained-source side panel: misc2 trains on ACS 2018 of other years/states)",
                             "item_key": f"acs:{key}:{task}", "item_date": str(year), "post_2025_07": False})
                ev.append(eval_rec(FT_SET, "T", FT_EVAL_SOURCE, state, {task: q}, {task: lab}, meta,
                                   f"{FT_EVAL_SOURCE}/household/{hh}", LICENCE["folktables"],
                                   f"{FT_EVAL_SOURCE}-{sha(key + task)}"))
    write_cands("folktables", tr, {"build_report": dict(rep)})
    gate = {"structural": f"training = ACS {FT_TRAIN_YEAR} only, eval = ACS {FT_EVAL_YEAR} only (independent annual "
                          "samples; households cannot coincide by id), and the held-out states are never read for "
                          "training; a text gate is not used between them (the record template would match every pair)"}

    def rate(rs, task, key):
        g = defaultdict(list)
        for r in rs:
            if r["meta"]["task"] == task:
                g[str(r["meta"]["demographics"].get(key))].append(r["labels"][task]["yes"])
        return {k: {"n": len(v), "yes_rate": round(statistics.fmean(v), 4)} for k, v in sorted(g.items()) if len(v) >= 20}
    bias = {}
    for task in FT_TASKS:
        bias[task] = {}
        for name, rs in (("train_candidates", tr), ("eval_heldout_state", [r for r in ev if r["meta"]["stratum"] ==
                                                                          "eval_heldout_state"]),
                         ("eval_training_state", [r for r in ev if r["meta"]["stratum"] == "eval_training_state"])):
            sub = [r for r in rs if r["meta"]["task"] == task]
            bias[task][name] = {"n": len(sub), "yes_rate": round(statistics.fmean(r["labels"][task]["yes"] for r in sub), 4)
                                if sub else None, "by_sex": rate(rs, task, "sex"), "by_race": rate(rs, task, "race"),
                                "by_hispanic": rate(rs, task, "hispanic")}
    man = {"domain": "T", "license": LICENCE["folktables"], "license_evidence": LICENCE_EVIDENCE["folktables"],
           "eval_only": True, "verdict": "TRAIN (ACS 2018, 6 states) + T side panel (ACS 2024: 4 held-out states and the "
                                         "6 training states)",
           "design": {"train": f"{FT_TRAIN_YEAR}, states {FT_TRAIN_STATES}, {FT_TRAIN_PER_TASK} units per task (95/5 "
                               "train/val by household)",
                      "eval": f"{FT_EVAL_YEAR}: held-out states {FT_HELDOUT_STATES} (time + place shift) and training "
                              f"states (time shift only), {FT_EVAL_PER_STRATUM} units per task and stratum",
                      "unit": "one record per (person, task) with ONLY that task's folktables features in the state "
                              "(a merged state would leak: PINCP and ESR are features of other tasks)",
                      "departures_from_folktables": [
                          "rows whose TARGET is N/A are dropped instead of being labelled 0: ACSEmployment drops people "
                          "under 16 (ESR N/A), ACSTravelTime drops people who work from home (JWMNP N/A)",
                          "2019+ files rename RELP -> RELSHIPP (recoded categories), JWTR -> JWTRNS and ST -> STATE: the "
                          "renamed column is read and rendered with that year's data dictionary labels",
                          "features are rendered as text with the year's PUMS data dictionary; PINCP is nominal dollars "
                          "(folktables does not apply ADJINC either)"],
                      "demographics": "kept in the state (they are folktables features) and in meta.demographics (sex, "
                                      "race, Hispanic origin, age) for the bias tables below",
                      "sampling": "per (task, stratum) the eligible persons with the smallest seeded hash (uniform, "
                                  "unweighted; meta.person_weight = PWGTP for weighted analyses)"},
           "sources": files, "data_dictionaries": {y: {"path": str(d[2]), "sha256": file_sha256(d[2])} for y, d in dicts.items()},
           "build_report": dict(rep), "bias_tables(yes rate by group)": bias, "dedup_vs_own_training": gate,
           "census_terms": "never use these records to identify any person or household (Census API terms)"}
    write_eval(FT_SET, ev, man, notes={"DO_NOT_TRAIN.txt": "GDE-1 eval side panel: never place under a training "
                                                          "directory.\n"})


# =========================================================================== 6. Amazon ESCI (training + T panel)
ES_SET = "esci"
ES_EVAL_SOURCE = "gde1/esci"
ES_TRAIN_SOURCE = "esci"
ES_QID = "relation"
ES_KEYS = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}
ES_CRIT = {   # the ESCI paper's definitions (arXiv 2206.06588 §2)
    "exact": "Exact: the product is relevant for the query and satisfies all the query specifications.",
    "substitute": "Substitute: the product is somewhat relevant; it fails to fulfil some aspects of the query but can "
                  "be used as a functional substitute.",
    "complement": "Complement: the product does not fulfil the query, but could be used in combination with an exact "
                  "item.",
    "irrelevant": "Irrelevant: the product is irrelevant, or it fails to fulfil a central aspect of the query."}
ES_INSTR = ("Shoppers' search queries on amazon.com were paired with products, and trained annotators judged how each "
            "product relates to the query (the released label is the majority of at least three annotators). Which "
            "relation did they judge this product to have to the query?")
ES_TRAIN_UNITS, ES_EVAL_UNITS, ES_PER_QUERY = 20000, 4000, 3


def es_media(path):
    repo, rev = PINS["esci"]
    return f"https://media.githubusercontent.com/media/{repo}/{rev}/{path}"


def es_products(ids, url=None, local=None):
    """{product_id: (title, brand, color)} for the US-locale ids wanted, read from the 1.1 GB products parquet with
    HTTP range requests (only the id / title / brand / color / locale column chunks; bullets and descriptions are never
    downloaded)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    cols = ["product_id", "product_title", "product_brand", "product_color", "product_locale"]
    if local:
        t = pq.read_table(local, columns=cols)
    else:
        import fsspec
        with fsspec.filesystem("http").open(url or es_media("shopping_queries_dataset/shopping_queries_dataset_products.parquet"),
                                            block_size=8 << 20) as f:
            t = pq.read_table(f, columns=cols)
    t = t.filter(pc.and_(pc.equal(t["product_locale"], "us"), pc.is_in(t["product_id"], value_set=__import__("pyarrow").array(sorted(ids)))))
    d = t.to_pydict()
    return {i: (ti, b, c) for i, ti, b, c in zip(d["product_id"], d["product_title"], d["product_brand"], d["product_color"])}


def es_state(query, prod):
    title, brand, color = (str(x).strip() if x else "" for x in prod)
    lines = [f"Shopping search query (amazon.com, US store): {query}", "", "Candidate product:", f"Title: {title}"]
    if brand:
        lines.append(f"Brand: {brand}")
    if color:
        lines.append(f"Color: {color}")
    return "\n".join(lines)


def es_tdb():
    repo, rev = PINS["tdb"]
    p = fetch(hf_url(repo, rev, "bench/suites/retrieval-and-knowledge/items.jsonl"), raw("tdb", "retrieval-and-knowledge.jsonl"))
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    rows = [r for r in rows if (r.get("meta") or {}).get("task") == "esci-product-rerank"]
    return rows, p


def es_build(a):
    import pyarrow.parquet as pq
    ex_path = fetch(es_media("shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
                    raw("esci", "shopping_queries_dataset_examples.parquet"))
    t = pq.read_table(ex_path, columns=["example_id", "query", "query_id", "product_id", "product_locale", "esci_label",
                                        "small_version", "split"]).to_pydict()
    tdb, tdb_path = es_tdb()
    tdb_ids = {int(r["meta"]["group"]) for r in tdb}
    tdb_q = {norm(r["state"]["query"]) for r in tdb}
    tdb_titles = {norm((r["state"].get("product") or {}).get("title") or "") for r in tdb}
    by_q, qsplit, qtext, rep = defaultdict(list), defaultdict(set), {}, Counter()
    for i in range(len(t["query_id"])):
        if t["product_locale"][i] != "us":
            continue
        qid = int(t["query_id"][i])
        qsplit[qid].add(t["split"][i])
        qtext[qid] = t["query"][i]
        by_q[qid].append(i)
    q_by_side = {"train": [], "eval": []}
    for qid, rows in by_q.items():
        if len(qsplit[qid]) > 1:
            rep["dropped_query/in_both_splits"] += 1
            continue
        if qid in tdb_ids or norm(qtext[qid]) in tdb_q:
            rep["dropped_query/tdb_esci_query"] += 1
            continue
        q_by_side["eval" if "test" in qsplit[qid] else "train"].append(qid)
    test_qtext = {norm(qtext[q]) for q in q_by_side["eval"]}
    n0 = len(q_by_side["train"])
    q_by_side["train"] = [q for q in q_by_side["train"] if norm(qtext[q]) not in test_qtext]
    rep["dropped_query/train_query_text_equals_a_test_query"] = n0 - len(q_by_side["train"])
    target = {"train": int(ES_TRAIN_UNITS * 1.15), "eval": ES_EVAL_UNITS}
    picks = {}
    for side, qs in q_by_side.items():
        qs = sorted(qs, key=lambda q: uhash(SEED, "esci-q", q))
        out = []
        for q in qs:
            rows = sorted(by_q[q], key=lambda i: uhash(SEED, "esci-p", t["example_id"][i]))[:ES_PER_QUERY]
            out += rows
            if len(out) >= target[side]:
                break
        picks[side] = out[: target[side]]
    eval_pids = {t["product_id"][i] for i in picks["eval"]}
    n0 = len(picks["train"])
    picks["train"] = [i for i in picks["train"] if t["product_id"][i] not in eval_pids]
    rep["dropped_train_pair/product_also_in_eval"] = n0 - len(picks["train"])
    need = {t["product_id"][i] for s in picks.values() for i in s}
    log(f"esci: reading {len(need)} products from the products parquet (range requests) ...")
    prods = es_products(need, local=a.esci_products)
    ev, tr = [], []
    for side, idxs in picks.items():
        for i in idxs:
            pid, qid = t["product_id"][i], int(t["query_id"][i])
            prod = prods.get(pid)
            if not prod or not str(prod[0] or "").strip():
                rep[f"{side}/dropped/no_title"] += 1
                continue
            if norm(prod[0]) in tdb_titles:
                rep[f"{side}/dropped/product_title_in_tdb"] += 1
                continue
            lab_key = ES_KEYS[t["esci_label"][i]]
            state = es_state(t["query"][i], prod)
            rng = random.Random(f"{SEED}:esci:{t['example_id'][i]}")
            order = list(ES_CRIT)
            rng.shuffle(order)
            qs = {ES_QID: choice_q(ES_INSTR, {k: ES_CRIT[k] for k in order})}
            labs = {ES_QID: gb.lab_dict(order, [1.0 if k == lab_key else 0.0 for k in order])}
            meta = {"query_id": qid, "product_id": pid, "example_id": int(t["example_id"][i]),
                    "esci_label": t["esci_label"][i], "small_version": int(t["small_version"][i]),
                    "esci_split": t["split"][i],
                    "label_provenance": "human: majority vote of >= 3 annotators (only the majority label is released; "
                                        "no vote counts)"}
            if side == "eval":
                meta.update({"panel": "T (trained-source side panel: misc2 trains on ESCI train-split queries)",
                             "item_key": f"esci:{t['example_id'][i]}", "item_date": "2022-06", "post_2025_07": False})
                ev.append(eval_rec(ES_SET, "T", ES_EVAL_SOURCE, state, qs, labs, meta, f"{ES_EVAL_SOURCE}/query/{qid}",
                                   LICENCE["esci"], f"{ES_EVAL_SOURCE}-{t['example_id'][i]}"))
            else:
                tr.append(train_rec(ES_TRAIN_SOURCE, state, qs, labs, meta, qid, LICENCE["esci"], probes=[prod[0]]))
    write_cands("esci", tr, {"build_report": dict(rep)})

    def lab_hist(rs):
        return dict(Counter(r["meta"]["esci_label"] for r in rs))
    man = {"domain": "T", "license": LICENCE["esci"], "license_evidence": LICENCE_EVIDENCE["esci"], "eval_only": True,
           "verdict": "TRAIN (train-split queries) + T side panel (test-split queries)", "pins": {"esci": PINS["esci"],
                                                                                                    "tdb": PINS["tdb"]},
           "sources": {"examples": {"path": str(ex_path), "sha256": file_sha256(ex_path)},
                       "products": es_media("shopping_queries_dataset/shopping_queries_dataset_products.parquet") +
                                   " (range-read: id, title, brand, color, locale columns)",
                       "tdb_items": {"path": str(tdb_path), "sha256": file_sha256(tdb_path), "esci_items": len(tdb),
                                     "esci_queries": len(tdb_ids)}},
           "selection": {"locale": "us", "split": "by ESCI query: the dataset's own query-stratified train / test split; "
                                                  "TDB v0.3's esci-product-rerank queries (all from ESCI test) are "
                                                  "removed from both sides by id and by text; a training query whose "
                                                  "text equals a test query is dropped; training pairs whose product "
                                                  "is also in an eval pair are dropped",
                         "per_query": ES_PER_QUERY, "eval_units": ES_EVAL_UNITS, "seed": SEED,
                         "state": "query + product title / brand / color (bullets and descriptions are not used)"},
           "build_report": dict(rep), "label_hist": {"eval": lab_hist(ev), "train_candidates": lab_hist(tr)},
           "label_note": "single released label per pair = the majority of >= 3 annotators (paper §2); audited "
                         "agreement 91% overall, > 96% exact vs not-exact. Hard one-hot targets.",
           "queries": {"eval": len({r["meta"]["query_id"] for r in ev}), "train_candidates":
                       len({r["meta"]["query_id"] for r in tr})},
           "tdb_note": "training on ESCI makes TDB's esci-product-rerank task IN-DISTRIBUTION for the misc2 line: report "
                       "it as a fitted row from then on"}
    write_eval(ES_SET, ev, man, notes={"DO_NOT_TRAIN.txt": "GDE-1 eval side panel: never place under a training "
                                                          "directory.\n"})


# =========================================================================== 7. ChaosNLI (check only)
def chaosnli_check(a):
    ext3 = (HERE / "build_jev_ext3.py").read_text(encoding="utf-8")
    ext2 = (HERE / "build_jev_ext2.py").read_text(encoding="utf-8")
    facts = {"licence": LICENCE_EVIDENCE["chaosnli"],
             "in_jevbench_eval": "chaosnli" in x2.JEVBENCH_CONFIGS,
             "blocked_in_ext3_TS_BLOCKED": '"chaos-mnli"' in ext3,
             "fingerprinted_by_ext2_gate": "def ev_chaosnli" in ext2 and '"chaosnli": ev_chaosnli' in ext2,
             "verdict": "EVAL ONLY (CC BY-NC 4.0). Nothing to build: it is jev-bench's chaosnli config (MNLI 1,599) and "
                        "the convert_chaos_extra sets; every training build is fingerprinted against it."}
    assert facts["in_jevbench_eval"] and facts["blocked_in_ext3_TS_BLOCKED"] and facts["fingerprinted_by_ext2_gate"], facts
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    p = OUT_TRAIN / "chaosnli_check.json"
    p.write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


# =========================================================================== training write: the ext2 gate
TRAIN_SOURCES = {"lichess": {LI_TRAIN_SOURCE: LI_TRAIN_CAP},
                 "folktables": {f"folktables/{t}": FT_TRAIN_PER_TASK for t in FT_TASKS},
                 "esci": {ES_TRAIN_SOURCE: ES_TRAIN_UNITS}}
TRAIN_LICENCE = {"lichess": LICENCE["lichess"], "folktables": LICENCE["folktables"], "esci": LICENCE["esci"]}
MISC2_SETS = [CC_SET, MQ_SET, FO_SET, LI_SET, FT_SET, ES_SET]
OWN_SETS = {LI_SET, FT_SET, ES_SET}     # same-source eval sets: disjoint by construction (game / year+state / query)


def new_stats(ds, src):
    return {"dataset": ds, "license": TRAIN_LICENCE[ds], "cap": None, "candidates": 0, "written": Counter(),
            "units": Counter(), "label_kinds": Counter(), "qtypes": Counter(), "drops": Counter(),
            "dedup_exact": Counter(), "dedup_leaf": Counter(), "dedup_shingle": Counter(), "question_drops": Counter(),
            "overlap_examples": [], "errors": [], "hf_sha": None}


def cmd_train(a):
    evals = [f"gde1_{p.name}={p}" for p in sorted(OUT_EVAL.iterdir()) if p.is_dir() and p.name not in OWN_SETS
             and any(p.rglob("*.jsonl"))] + list(a.eval)
    ns = SimpleNamespace(eval=evals, fingerprint=list(a.fingerprint), allow_missing_corpora=True,
                         fetch_evals=not a.no_fetch_evals, skip_evals=set(a.skip_evals), allow_missing_evals=False,
                         hf_token=None, cache=str(WORK / "urlcache"))
    t0 = time.time()
    idx, coverage = x2.load_fingerprints(ns)
    cf = load_cache_features()
    seen, top = set(), {"builder": VERSION, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}
    gate_desc = {"as": "build_jev_ext2.load_fingerprints + process (exact / leaf / 8-word-shingle; >= 3 shared shingles "
                       "with ONE fingerprint state drops the TRAINING record)",
                 "fingerprint_states": dict(idx.sizes), "eval_coverage": coverage, "given_evals": evals,
                 "training_corpora": list(a.fingerprint) or "NOT CHECKED on the laptop (jev_data / jev_ext_strict / "
                 "jev_ext2 / jev_ext3 live on S3 / nodes): cross-corpus duplicates only; no eval is affected",
                 "own_eval_sets": {s: "not in the index: disjoint by construction (see the set's manifest), and the "
                                      "shared record template would match every pair" for s in sorted(OWN_SETS)}}
    for ds, srcs in TRAIN_SOURCES.items():
        cands = read_cands(ds)
        if not cands:
            log(f"[{ds}] no candidates under {cands_path(ds)}: run `build_jev_misc2.py {ds}` first; skipped")
            continue
        out = OUT_TRAIN / ds
        for sp in ("train", "val"):
            for old in (out / sp).glob("*.jsonl") if (out / sp).exists() else []:
                old.unlink()
        writer = Writer(out)
        man = {"dataset": ds, "builder": VERSION, "license": TRAIN_LICENCE[ds],
               "license_evidence": LICENCE_EVIDENCE[ds], "splits": {"train": 95, "val": 5, "by": "sha1 of meta.group "
                                                                  "(lichess: game; folktables: household; esci: query)"},
               "dedup": gate_desc, "sources": {}}
        info_p = cands_path(ds).with_suffix(".info.json")
        if info_p.exists():
            man["candidates_build_report"] = json.loads(info_p.read_text())
        for src, cap in srcs.items():
            cs = [c for c in cands if c["source"] == src]
            st = new_stats(ds, src)
            x2.process(src, cs, cap, idx, writer, random.Random(f"{SEED}:{src}"), st, seen, MAX_STATE_CHARS)
            x2.report_source(src, st)
            man["sources"][src] = plain(st)
        writer.close()
        man["readback"] = final_check(out)
        labs, nc = defaultdict(Counter), 0
        for p in sorted(out.rglob("*.jsonl")):
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                nc += bool(cf.NC_LICENCE.search(r.get("license") or ""))
                for q, lab in r["labels"].items():
                    labs[f"{p.parent.name}/{r['source']}/{q}"][max(lab, key=lab.get)] += 1
        assert nc == 0, f"{ds}: {nc} training records carry a non-commercial licence string"
        man["label_hist"] = {k: dict(v) for k, v in sorted(labs.items())}
        man["nc_licence_check"] = "0 records match cache_features.NC_LICENCE"
        man["files_sha256"] = {str(p.relative_to(out)): file_sha256(p) for p in sorted(out.rglob("*.jsonl"))}
        (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
        top["datasets"][ds] = {"readback_total": man["readback"]["total"],
                               "dedup_drops": {s: {k: dict(v[f"dedup_{k}"]) for k in ("exact", "leaf", "shingle")
                                                   if v[f"dedup_{k}"]} for s, v in man["sources"].items()}}
    top["seconds"] = round(time.time() - t0, 1)
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    (OUT_TRAIN / "manifest.json").write_text(json.dumps(top, indent=1, default=str))
    x2.print_summary({"sources": {s: v for ds in top["datasets"] for s, v in
                                  json.loads((OUT_TRAIN / ds / "manifest.json").read_text())["sources"].items()}})


# =========================================================================== node gate, check
def cmd_gate(a):
    """Re-gate the WRITTEN misc2 eval sets against real training corpora (TAG=PATH jsonl dirs) with build_gde1_a's
    Gate; drop overlapping records and rewrite (unless --dry-run); append the result to each manifest."""
    if not a.corpus:
        raise SystemExit("gate needs --corpus TAG=PATH (repeatable)")
    recs_by, items, seen = {}, [], set()
    for gset in MISC2_SETS:
        root = OUT_EVAL / gset
        if not (root / "manifest.json").exists():
            continue
        recs_by[gset] = [json.loads(line) for p in sorted((root / "eval").glob("*.jsonl")) for line in p.open(encoding="utf-8")]
        for r in recs_by[gset]:
            k = r["meta"]["item_key"]
            if k not in seen:
                seen.add(k)
                st = json.loads(r["state"]) if gset == CC_SET else r["state"]
                items.append({"key": k, "state": st, "fields": []})
    hits, read = ga.gate_items(items, lambda: ga.corpus_stream(a.corpus))
    drop = {k for k, h in hits.items() if ga.dropping(h)}
    for gset, recs in recs_by.items():
        keep = [r for r in recs if r["meta"]["item_key"] not in drop]
        root = OUT_EVAL / gset
        man = json.loads((root / "manifest.json").read_text())
        ex = [{"item": r["meta"]["item_key"], "match": hits[r["meta"]["item_key"]][:2]} for r in recs
              if r["meta"]["item_key"] in drop][:5]
        man.setdefault("dedup_vs_training_real_runs", []).append({
            "when": time.strftime("%Y-%m-%d %H:%M:%S"), "corpora": a.corpus, "training_states_read": dict(Counter(
                t.split("/")[0] for t in read.elements())) if hasattr(read, "elements") else dict(read),
            "records_dropped": len(recs) - len(keep), "examples": ex if gset != MQ_SET else "suppressed",
            "dry_run": bool(a.dry_run)})
        if not a.dry_run and len(keep) < len(recs):
            for p in (root / "eval").glob("*.jsonl"):
                p.unlink()
            for r in keep:
                with (root / "eval" / (r["source"].replace("/", "__") + ".jsonl")).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            man["readback_after_gate"] = readback_eval(root, integral=gset == CC_SET)
        if not a.dry_run:
            man["dedup_vs_training_real"] = "run (see dedup_vs_training_real_runs)"
        (root / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False, default=str))
        log(f"[gate] {gset}: {len(recs) - len(keep)} of {len(recs)} records overlap" + (" (dry run)" if a.dry_run else ""))


def cmd_check(a):
    rep = {}
    for gset in MISC2_SETS:
        root = OUT_EVAL / gset
        if (root / "eval").exists():
            rb = readback_eval(root, integral=gset == CC_SET)
            rep[gset] = {k: rb[k] for k in ("records", "units", "qtypes", "label_kind", "state_chars")}
    for ds in TRAIN_SOURCES:
        if (OUT_TRAIN / ds / "train").exists():
            with quiet():
                rep[f"train:{ds}"] = final_check(OUT_TRAIN / ds)["total"]
    print(json.dumps(rep, indent=1, default=str))
    return rep


# =========================================================================== self-test (offline)
def self_test():
    global OUT_EVAL, OUT_TRAIN
    cf = load_cache_features()
    # ---- licences: training strings pass the trainer's NC filter, the research-only one is caught
    for k in ("lichess", "folktables", "esci", "folio", "civil_comments_votes"):
        assert not cf.NC_LICENCE.search(LICENCE[k]), k
    assert cf.NC_LICENCE.search(LICENCE[MQ_SET])
    print("ok  licence strings vs cache_features.NC_LICENCE")
    # ---- FOL parser + z3
    t = fol_parse("∀x (Man(x) → Mortal(x)) ∧ ¬∃y (Rock(y) ∧ Mortal(y))")    # scope runs right (NLTK's convention)
    assert t[0] == "all" and t[2][0] == "and" and t[2][2][0] == "not" and t[2][2][1][0] == "ex", t
    t = fol_parse("(∀x (Man(x) → Mortal(x))) ∧ ¬∃y (Rock(y) ∧ Mortal(y))")
    assert t[0] == "and" and t[1][0] == "all" and t[2][0] == "not" and t[2][1][0] == "ex", t
    assert fol_parse("LostTo(igaŚwiątek, y42.3billion)")[0] == "pred"
    assert fol_parse("a = b ⊕ P(a) → Q(b)")[0] == "imp"
    try:
        fol_parse("P(a))")
        raise AssertionError("trailing paren accepted")
    except FolError:
        pass
    try:
        import z3  # noqa: F401
        prem = ["∀x (Man(x) → Mortal(x))", "Man(socrates)"]
        assert fol_verdict(prem, "Mortal(socrates)")[0] == "true"
        assert fol_verdict(prem, "¬Mortal(socrates)")[0] == "false"
        assert fol_verdict(prem, "Greek(socrates)")[0] == "unknown"
        assert fol_verdict(["P(a)", "¬P(a)"], "Q(a)")[0] == "inconsistent"
        assert fol_verdict(prem, "Mortal(socrates")[0] == "parse_error"
        rows = [{"premises": ["All men are mortal.", "Socrates is a man."], "premises-FOL": prem,
                 "conclusion": "Socrates is mortal.", "conclusion-FOL": "Mortal(socrates)", "label": "True"},
                {"premises": ["All men are mortal.", "Socrates is a man."], "premises-FOL": prem,
                 "conclusion": "Socrates is Greek.", "conclusion-FOL": "Greek(socrates)", "label": "False"},
                {"premises": ["x"], "premises-FOL": ["P(a)"], "conclusion": "y", "label": "Unknown"}]
        fr, frep, _ = fo_records(rows, "v0.0")
        assert len(fr) == 1 and frep["dropped/z3_disagrees_or_cannot_decide"] == 1 and frep["dropped/no_fol_annotation"] == 1
        assert fr[0]["labels"][FO_QID]["true"] == 1.0 and fr[0]["base_id"].startswith(f"{FO_SOURCE}/story/")
        print("ok  FOL parser, z3 verdicts, FOLIO records (label kept only when z3 agrees)")
    except ImportError:
        fr = []
        print("--  z3 not installed: FOL verdict tests skipped")
    # ---- MedQA
    mr, mrep = mq_records([{"question": "A 30-year-old man has chest pain. Which drug is first line?", "answer_idx": "C",
                            "options": {"A": "Aspirin", "B": "Heparin", "C": "Nitroglycerin", "D": "Morphine"},
                            "meta_info": "step2&3"}, {"question": "", "options": {}, "answer_idx": "A"}])
    assert len(mr) == 1 and mr[0]["labels"][MQ_QID]["C"] == 1.0 and mrep["dropped/bad_row"] == 1
    assert mr[0]["state"].startswith("USMLE Step 2/3-style question:")
    print("ok  MedQA records")
    # ---- Civil Comments mapping + jev-bench id formula (trainers/convert_jevbench.emit)
    google = {"test": {"text": ["you are an idiot", "nice article"], "toxicity": [0.8, 0.0]},
              "validation": {"text": ["thanks", "nice article"], "toxicity": [0.1, 0.0]}}
    mirror = {"id": [11, 12, 13, 14], "text": ["you are an idiot", "nice article", "thanks", "nice article"],
              "toxicity": [0.8, 0.0, 0.1, 0.0], "toxicity_annotator_count": [10, 4, 10, 6]}
    jb = {"test": [{"id": "civil_comments/test/0", "state": json.dumps("you are an idiot"), "soft_label": "0.8",
                    "label": "1", "question": "{}"}],
          "train": [{"id": "civil_comments/train/5", "state": json.dumps("thanks"), "soft_label": "0.1", "label": "0",
                     "question": "{}"}],
          "validation": [{"id": "civil_comments/validation/9", "state": json.dumps("nice article"), "soft_label": "0.0",
                          "label": "0", "question": "{}"}]}
    jm, jrep = cc_map(google, mirror, jb)
    assert jm["civil_comments/test/0"]["kaggle_ids"] == [11] and jm["civil_comments/test/0"]["counts"] == [10]
    assert jm["civil_comments/train/5"]["google_rows"] == [("validation", 0)]
    assert jrep["validation"]["kaggle_id_ambiguous(duplicate text)"] == 1
    st = json.dumps("you are an idiot")
    assert cc_jev_real_id(st, "civil_comments/test/0") == \
        "jevbench/civil_comments-" + hashlib.sha1((st + "civil_comments/test/0" + '["yes", "no"]').encode()).hexdigest()[:16]
    print("ok  Civil Comments: jev-bench -> google row -> Kaggle id mapping; jev_real id")
    # ---- Lichess
    try:
        import chess  # noqa: F401
        row = {"PuzzleId": "00008", "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
               "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1", "Rating": "1797", "RatingDeviation": "76", "Popularity": "95",
               "NbPlays": "10183", "Themes": "crushing hangingPiece long middlegame",
               "GameUrl": "https://lichess.org/787zsVup/black#48", "OpeningTags": ""}
        (state, q, lab, meta), why = li_convert(row)
        assert why == "ok" and "White to move" in state and "Black has just played Bxg3 (f2-g3)" in state, state[:200]
        assert meta["fen_after_first_move"].startswith("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w")
        sol = [k for k, v in lab.items() if v == 1.0]
        assert len(sol) == 1 and q["criteria"][sol[0]] == "Rxe7 (e6-e7)" and len(q["criteria"]) == 4
        # a mate in one with TWO mates (Ra8# and Rb8#): the second mate must never be offered as a "wrong" option
        row2 = dict(row, PuzzleId="m1", FEN="6k1/5ppp/8/8/8/8/1R3PPP/R5K1 b - - 0 1", Moves="g8h8 a1a8")
        (s2, q2, l2, m2), w2 = li_convert(row2)
        assert w2 == "ok" and m2["solution_mates"] and q2["criteria"][max(l2, key=l2.get)] == "Ra8 (a1-a8)"
        assert not any("b2-b8" in v for v in q2["criteria"].values()), q2["criteria"]
        assert all("#" not in v and "+" not in v for v in q2["criteria"].values())
        assert li_side("787zsVup") in ("train", "eval") and li_bin(1797) == "1400-1799" and li_bin(2700) == ">=2600"
        print("ok  Lichess: state after the opponent's move, gold = Moves[1], 4 options, no mating distractor")
    except ImportError:
        print("--  python-chess not installed: Lichess tests skipped")
    # ---- Folktables rendering, eligibility, targets
    vals = {"SEX": [("1", "1", "Male"), ("2", "2", "Female")], "RAC1P": [("1", "1", "White alone")],
            "COW": [("b", "b", "N/A (less than 16 years old)"), ("1", "1", "Employee of a private for-profit company")],
            "SCHL": [("21", "21", "Bachelor's degree")], "MAR": [("1", "1", "Married")],
            "OCCP": [("0010", "0010", "MGR-Chief Executives And Legislators")], "POBP": [("008", "008", "Colorado/CO")],
            "RELP": [("00", "00", "Reference person")],
            "JWMNP": [("bbb", "bbb", "N/A (not a worker or worker who worked from home)")]}
    r = {"AGEP": "45", "COW": "1", "SCHL": "21", "MAR": "1", "OCCP": "10", "POBP": "8", "REL": "0", "WKHP": "45",
         "SEX": "2", "RAC1P": "1", "PINCP": "64000", "PWGTP": "12", "ESR": "1", "JWMNP": ""}
    cols = {v: ("RELP" if v == "REL" else v) for v in FT_FIELD}
    s = ft_state("acs_income", r, vals, cols, 2018)
    assert "- Occupation: MGR-Chief Executives And Legislators" in s and "- Sex: Female" in s and "$" not in s, s
    assert "- Relationship to the household's reference person: Reference person" in s
    assert FT_TASKS["acs_income"]["eligible"](r) and FT_TASKS["acs_income"]["target"](r)
    assert not FT_TASKS["acs_travel_time"]["eligible"](r), "work-from-home (JWMNP N/A) must be dropped"
    assert not FT_TASKS["acs_employment"]["eligible"]({"AGEP": "12", "ESR": ""})
    assert ft_value("PINCP", "-500", vals, "PINCP") == "a loss of $500" and ft_value("POVPIP", "501", vals, "POVPIP") == "501% or more"
    print("ok  Folktables: per-task feature states, folktables filters, N/A targets dropped")
    # ---- ESCI state
    es = es_state("usb c cable 6ft", ("Anker USB-C Cable, 6 ft", "Anker", None))
    assert "Brand: Anker" in es and "Color" not in es and "Title: Anker USB-C Cable, 6 ft" in es
    print("ok  ESCI state")
    # ---- eval write + read-back, ext2 training gate, cands round trip (temp dirs)
    with tempfile.TemporaryDirectory() as tmp:
        OUT_EVAL, OUT_TRAIN = Path(tmp) / "gde1", Path(tmp) / "misc2_jev"
        recs = mr + fr
        for rr in recs:
            rr["meta"]["gde1_set"] = "toy"
        man = write_eval("toy", recs, {"domain": "K"})
        assert man["readback"]["units"] == len(recs) and (OUT_EVAL / "toy" / "samples.json").exists()
        v = eval_rec("toyv", "V", "gde1/toyv", json.dumps("a comment"), {"toxic": {"type": "noul", "instructions": "Toxic?",
                     "criteria": {"yes": "y", "no": "n"}}}, {"toxic": gb.lab_dict(["yes", "no"], [3, 7])},
                     {"n_raters": 10, "counts": {"toxic": [3, 7]}, "item_key": "k1"}, "gde1/toyv/c1", "CC0-1.0", "toyv-1")
        assert write_eval("toyv", [v], {"domain": "V"}, integral=True)["readback"]["units"] == 1
        idx = x2.Index()
        tag = idx.tag("eval:toy", "eval")
        idx.add(tag, "Shopping search query (amazon.com, US store): usb c cable 6ft\n\nCandidate product:\nTitle: Anker "
                     "USB-C Cable, 6 ft, braided nylon fast charging cord for phones tablets laptops")
        idx.freeze()
        mk = lambda query, title: train_rec("esci", es_state(query, (title, "Anker", None)), {ES_QID: choice_q(  # noqa: E731
            ES_INSTR, ES_CRIT)}, {ES_QID: gb.lab_dict(list(ES_CRIT), [1, 0, 0, 0])}, {}, 7, LICENCE["esci"], probes=[title])
        c1 = mk("usb c cable 6ft", "Anker USB-C Cable, 6 ft, braided nylon fast charging cord for phones tablets laptops")
        c2 = mk("right angle hdmi cable", "Cable Matters right angle HDMI cable 10 ft")
        st = new_stats("esci", "esci")
        w = Writer(OUT_TRAIN / "esci")
        x2.process("esci", [c1, c2], 10, idx, w, random.Random(0), st, set(), MAX_STATE_CHARS)
        w.close()
        assert sum(st["written"].values()) == 1 and sum(st["dedup_exact"].values()) + sum(st["dedup_shingle"].values()) == 1, st
        with quiet():
            fc = final_check(OUT_TRAIN / "esci")
        assert fc["total"]["units"]["train"] + fc["total"]["units"]["val"] == 1
        print("ok  eval write + iter_units read-back (K and V), ext2 gate drops a training pair that shares an eval state")
    print("\nself-test passed")
    return 0


# =========================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--cap", type=int, default=CAP_EVAL, help="eval units per set (default 8,000)")
        p.add_argument("--no-proxy", action="store_true", help="skip the eval-side gate against training proxies")
        p.add_argument("--limit-rows", type=int, default=0, help="smoke test: read only this many source rows")
        p.add_argument("--folio-v2", action="store_true", help="FOLIO v2 from yale-nlp/FOLIO (gated; HF token)")
        p.add_argument("--esci-products", default=None, help="a local copy of the ESCI products parquet")
        return p
    for name in ("civil_comments", "medqa", "folio", "lichess", "folktables", "esci", "chaosnli"):
        common(sub.add_parser(name))
    p = sub.add_parser("train", help="ext2 gate + write the training corpora from the candidates")
    p.add_argument("--eval", action="append", default=[], help="extra TAG=PATH eval fingerprints")
    p.add_argument("--fingerprint", action="append", default=[], help="TAG=PATH training corpora (on a node)")
    p.add_argument("--no-fetch-evals", action="store_true")
    p.add_argument("--skip-evals", action="append", default=[])
    p = sub.add_parser("gate", help="re-gate the misc2 eval sets against real training corpora (on a node)")
    p.add_argument("--corpus", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    fn = {"civil_comments": cc_build, "medqa": mq_build, "folio": fo_build, "lichess": li_build, "folktables": ft_build,
          "esci": es_build, "chaosnli": chaosnli_check, "train": cmd_train, "gate": cmd_gate,
          "check": lambda a: cmd_check(a)}.get(a.cmd)
    if fn is None:
        ap.error("choose a subcommand or --self-test")
    fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
