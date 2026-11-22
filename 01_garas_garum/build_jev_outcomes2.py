#!/usr/bin/env python3
"""
build_jev_outcomes2.py -- packaged REAL-OUTCOME decision datasets suggested by two outside researchers, vetted against
primary sources (research/outcomes2_pipeline.md) and converted to the jev_data record schema. Two outputs:
  * TRAINING corpora (zero-shot line), behind build_jev_ext2's dedup gate (fingerprint Index against every eval set,
    exact / leaf / 8-word-shingle; x2.process -> validate, length, cross-source exact dedup, gate, cap, write):
        <train-out>/<dataset>/{train,val}/<source>.jsonl + manifest.json        (default /tmp/outcomes2_jev)
  * GDE-1 eval sets (domain O, real outcomes), in build_gde1_b.py's layout and conventions:
        <eval-out>/<set>/eval/<source>.jsonl + manifest.json (+ check_report.json, samples.json)   (default /tmp/gde1)
The eval sets are written FIRST and the eval-out directory is always added to the gate's eval fingerprints, so no
training record can share a state with a GDE-1 item (including this builder's own eval sets).

Datasets (subcommand -> verdict; the researchers' claims are checked in research/outcomes2_pipeline.md)
  aidev        hao-li/AIDev @ c63c8a57, AIDev-pop tables (repos > 100 stars; pull_request + repository parquet, 58 MB).
               BOTH. "This pull request will be merged" (noul pr_merged); gold = merged_at set vs closed unmerged
               (open PRs dropped). State = creation-time fields only: repository name / language / licence, agent,
               creation date, title, description. Never: reviews, comments, commits (their dates are not in the data,
               so post-creation commits cannot be separated: no commit or diff field is used), stars / forks (a
               collection-time snapshot). GitHub Copilot PRs are DROPPED: the Copilot agent opens a "[WIP]" draft at
               once and rewrites title and body as it works, so the stored final text encodes post-creation progress
               (WIP-titled PRs are closed unmerged 87% of the time vs 26% overall). Other WIP titles and bodies that
               announce their own closing are dropped. Split: 25% of repositories held out by hash; EVAL = held-out
               repos, PRs created 2025-08-01 .. 2025-09-30 (outcomes after 2025-07; right-censoring limited by ending
               3.5 weeks before the data cutoff); TRAIN/VAL = other repos, created < 2025-07-01 and closed < 2025-08-01,
               and only repositories under a permissive licence (the card: "Each source repository retains its
               original copyright and license ... Do not assume a universal re-license").
  afd          ConvoKit wiki-articles-for-deletion-corpus (= emayfield/AFD_Decision_Corpus, Mayfield & Black 2019).
               TRAIN only (every outcome 2005-2018 is in web-scale pretraining: the archived debate page starts with
               "The result was ..."). Two disjoint variants: afd/nomination (nomination statement only) and
               afd/nomination_votes (+ the votes signed BEFORE the close, earliest first). Questions: afd_close
               (choice: delete / keep / no consensus / merge / redirect) and afd_deleted (noul). Gold = the closing
               admin's normalized outcome label; ambiguous multi-label closes, withdrawals, userfy / transwiki / move
               are dropped. Never in the state: outcome_rationale, any utterance at or after the close, votes without
               a timestamp, usernames / speaker metadata. Closes 2005-01-01 .. 2018-12-31 only.
  swiss        rcds/swiss_judgment_prediction_xl @ 648b5910 (streamed prefix of the xz files, not the 1.3 GB).
               BOTH. "appeal_approved" (noul: approval incl. partial vs dismissal). State = the FACTS section only
               (head + tail if long), law area, division, language, year; never the considerations. Dispositive
               words (gutgeheissen, abgewiesen, admis, rejete, accolto, respinto) occur in ~54% of fact sections with
               no association with the label (they describe the lower courts and the parties' requests); records are
               dropped only where the facts state THIS court's ruling (a same-day Federal Supreme Court ruling) or carry
               the considerations / operative-part heading. Diachronic: TRAIN = the XL train split (2002-2015),
               EVAL = the XL test split (2018-2022; pre-cutoff, flagged), 2016-2017 unused (buffer). Multilingual.
               The original rcds/swiss_judgment_prediction files (Zenodo 7109926) are CC BY-NC-SA 4.0 and are NOT read.
  swe_rebench  nebius/SWE-rebench-openhands-trajectories @ 35455389 (column projection: no trajectory is read) joined
               with nebius/SWE-rebench @ 89cdfbab (problem_statement). BOTH. "patch_resolves_issue" (noul); gold =
               the stored `resolved` test result. State = repository + issue + the agent's final patch ONLY: no
               trajectory, tool output, test output, logs, exit status. Splits are repository-disjoint (a global
               hash bucket shared with swe_agent: 20% of repositories are eval-only), at most 3 (train) / 2 (eval)
               patches per issue; empty patches dropped. Training: permissive repository licences only.
  swe_agent    nebius/SWE-agent-trajectories @ 68195a14 (+ nebius/SWE-bench-extra, princeton-nlp/SWE-bench dev for the
               issue text). EVAL ONLY: every patch comes from a Llama-based swe-agent-llama-* model and the card
               requires Llama 3.1 License compliance for these outputs ("include 'Llama' at the beginning of any such
               AI model name" when training a distributed model). Same question, state and repository bucket.
  ecthr        coastalcph/lex_glue @ c23fdff1, ecthr_a + ecthr_b (row-aligned). EVAL ONLY: the text is HUDOC content,
               whose terms require prior written permission for commercial use (the LexGLUE card YAML says cc-by-4.0,
               its licensing section is empty). One noul per article the Court considered (ecthr_b labels, plus any
               violated article): "violation_art_<n>"; gold = article in ecthr_a (violated). State = the facts
               paragraphs only. Splits = LexGLUE's chronological dev (2016-17) + test (2017-19); pre-cutoff, flagged.
  autocast, peerread   SKIP (verdict printed; see the write-up).
  forecastbench market questions: build_gde1_b.py forecastbench --market manifold,metaculus (separate set).

Labels. EVAL: the realised outcome, one-hot. TRAIN (default --blend 0.5, as build_jev_courts / safety / fda):
t = lam * onehot + (1 - lam) * a leave-one-out base rate of the source's cell (meta.label_method has the cell),
smoothed toward the leave-one-out global rate (prior strength 20); meta.outcome keeps the realised option and
meta.label_source[qid] = "outcome". --blend 1 gives one-hot.

Every record: `license` (source licence string; eval-only ones read as non-commercial to the trainer's NC filter),
meta.dataset, meta.source_revision, meta.label_provenance, meta.decision_date / info_date (+ resolution_date per
question on eval records), base_id (bootstrap cluster on eval sets: repository, debate, case).

Usage (laptop: scratch <= 2 GB under --work; HF access is anonymous):
  uv run --no-project --with pyarrow --with numpy --with huggingface_hub python build_jev_outcomes2.py all \\
      --work /tmp/outcomes2_work --train-out /tmp/outcomes2_jev --eval-out /tmp/gde1 --allow-missing-corpora
  ... aidev | afd | swiss | swe_rebench | swe_agent | ecthr       (one dataset; same flags)
  ... check                                                        (iter_units read-back, scans, stats)
  ... --self-test                                                  (offline fixtures)
On a node (authoritative gate with the training-corpus fingerprints): fleet/task_62d.sh.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import math
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")   # public data only; never pick up a cached token

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

from claude_overlap import norm  # noqa: E402

import build_gde1_b as gb  # noqa: E402  GDE-1 conventions: dedup_within, cap_units, split_stats, write_set, check_set
import build_jev_ext2 as x2  # noqa: E402  the training gate: Index, load_fingerprints, process, report_source
from build_jev_data import Writer, choice_q, record, shuffled, validate  # noqa: E402
from build_jev_ext import Labeler, final_check, plain  # noqa: E402

BUILDER = "build_jev_outcomes2.py v1 (2026-10-01)"
WORK = "/tmp/outcomes2_work"
TRAIN_OUT = "/tmp/outcomes2_jev"
EVAL_OUT = "/tmp/gde1"
SEED = 2026
EVAL_CAP = 8000
TRAIN_CAP = 20000
UA = "jev-outcomes2/0.1 (research dataset build; public data, polite rate)"

# --------------------------------------------------------------------------- pins
AIDEV = ("hao-li/AIDev", "c63c8a57a2de34fc03fa83722412824af4d8753b")
AFD_URL = ("https://zissou.infosci.cornell.edu/convokit/datasets/wiki-articles-for-deletion-corpus/"
           "wiki-articles-for-deletion-corpus.zip")
AFD_BYTES = 401810632            # Content-Length on 2026-10-01 (Last-Modified 2021-02-19); sha256 goes to the manifest
SWISS = ("rcds/swiss_judgment_prediction_xl", "648b5910046f2ebdfd9a3821e396bba242560359")
OPENHANDS = ("nebius/SWE-rebench-openhands-trajectories", "35455389ab51bf5e2306bfd436ef72d0f98bf882")
REBENCH = ("nebius/SWE-rebench", "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881")
SWE_AGENT = ("nebius/SWE-agent-trajectories", "68195a1450865274106246d0d0296a1d6807b88e")
SWE_EXTRA = ("nebius/SWE-bench-extra", "11dcbfb30e19552df2a2f8030bd764adc95c92a5")
SWE_BENCH = ("princeton-nlp/SWE-bench", "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6")
LEXGLUE = ("coastalcph/lex_glue", "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c")
OPENHANDS_RELEASED = "2025-12-27"   # HF lastModified of the pinned revision: an upper bound for the agent runs
SWE_AGENT_RELEASED = "2024-12-23"

# --------------------------------------------------------------------------- licences (record strings) + evidence
# Training licence strings must NOT match cache_features.NC_LICENCE; the eval-only ones deliberately do.
LICENSES = {
    "aidev": ("CC-BY-4.0 (hao-li/AIDev card YAML 'license: cc-by-4.0'); the card adds: 'Each source repository retains "
              "its original copyright and license ... Do not assume a universal re-license' -- the repository's "
              "licence is in meta.repo_license (training records: permissive licences only)"),
    "afd": ("GPL-3.0 (AFD_Decision_Corpus, Mayfield & Black 2019, github.com/emayfield/AFD_Decision_Corpus; ConvoKit: "
            "'The original Wikipedia Articles for Deletion dataset ... is licensed under the GNU General Public "
            "License v3.0') over English Wikipedia contributions (CC BY-SA 3.0/4.0 + GFDL, authors' copyright): "
            "redistributing these records requires GPL-3.0 terms plus CC BY-SA attribution and share-alike"),
    "swiss": ("CC-BY-SA-4.0 (rcds/swiss_judgment_prediction_xl card YAML 'license: cc-by-sa-4.0'; its text says "
              "'We release the data under CC-BY-4.0'; the stricter share-alike reading is applied). (c) Swiss Federal "
              "Supreme Court decisions, anonymised by the court"),
    "swe_rebench": ("CC-BY-4.0 (nebius/SWE-rebench-openhands-trajectories LICENSE + card; nebius/SWE-rebench card: "
                    "'please respect the license of each specific repository'); patches written by "
                    "Qwen3-Coder-480B-A35B-Instruct (Apache-2.0); repository licence in meta.repo_license"),
    "swe_agent": ("EVAL ONLY, do not train: CC-BY-4.0 (nebius/SWE-agent-trajectories card) but the patches are outputs "
                  "of Llama-based swe-agent-llama-* models and the card requires Llama 3.1 License compliance for "
                  "them (a model trained on Llama outputs must carry 'Llama' at the start of its name); the "
                  "public GDE-1 eval is a non-commercial use"),
    "ecthr": ("EVAL ONLY, non-commercial: LexGLUE (coastalcph/lex_glue) card YAML 'cc-by-4.0', licensing section "
              "empty; the text is ECHR-CEDH HUDOC content: reproduction 'for private use or for the purposes of "
              "information and education' only, 'commercial use, must request prior written permission'"),
}
EVIDENCE = {
    "aidev": {"urls": [f"https://huggingface.co/datasets/{AIDEV[0]}/blob/{AIDEV[1]}/README.md"],
              "quotes": ["license: cc-by-4.0", "This dataset aggregates content from GitHub repositories. Each source "
                         "repository retains its original copyright and license (e.g., MIT, Apache-2.0, GPL family, "
                         "Creative Commons variants, etc.). Files, patches/diffs, and any other artifacts originating "
                         "from those repositories remain governed by their original licenses.",
                         "Users must verify and comply with the specific license of any source repository or file they "
                         "extract or use from this collection. Do not assume a universal re-license.",
                         "Intended Uses: Fine-tuning or post-training: fine-tuning or post-training your LLMs/agents "
                         "based on the patches"]},
    "afd": {"urls": ["https://github.com/emayfield/AFD_Decision_Corpus (LICENSE: GNU GPL v3; GitHub API licence "
                     "gpl-3.0; the README states no separate data licence; the corpus files are downloaded, not in "
                     "the repository)",
                     "https://github.com/CornellNLP/ConvoKit/blob/master/docs/source/wiki-articles-for-deletion-corpus.rst",
                     "https://en.wikipedia.org/wiki/Wikipedia:Copyrights",
                     "https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/"],
            "quotes": ["ConvoKit: 'The original Wikipedia Articles for Deletion dataset (see source here) is licensed "
                       "under the GNU General Public License v3.0.'",
                       "Wikipedia: 'Most of Wikipedia's text ... [is] co-licensed under the Creative Commons "
                       "Attribution-ShareAlike 4.0 International License (CC BY-SA) and the GNU Free Documentation "
                       "License (GFDL)'",
                       "CC: 'GPLv3: The GNU General Public License version 3 was declared a BY-SA-Compatible License for "
                       "version 4.0 on 8 October 2015. Note that compatibility with the GPLv3 is one-way only'"]},
    "swiss": {"urls": [f"https://huggingface.co/datasets/{SWISS[0]}/blob/{SWISS[1]}/README.md",
                       "https://zenodo.org/records/7109926 (the ORIGINAL rcds/swiss_judgment_prediction files: "
                       "license cc-by-nc-sa-4.0 -- not read by this builder)"],
              "quotes": ["license: cc-by-sa-4.0", "We release the data under CC-BY-4.0 which complies with the court "
                         "licensing (https://www.bger.ch/files/live/sites/bger/files/pdf/de/urteilsveroeffentlichung_d.pdf)",
                         "(the linked BGer PDF is a publication policy with no licence terms; read 2026-10-01)"]},
    "swe_rebench": {"urls": [f"https://huggingface.co/datasets/{OPENHANDS[0]}/blob/{OPENHANDS[1]}/LICENSE",
                             f"https://huggingface.co/datasets/{REBENCH[0]}/blob/{REBENCH[1]}/README.md",
                             "https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct (license: apache-2.0)"],
                    "quotes": ["license: cc-by-4.0 (card YAML); LICENSE file: 'Attribution 4.0 International'",
                               "SWE-rebench: 'The dataset is licensed under the Creative Commons Attribution 4.0 "
                               "license. However, please respect the license of each specific repository on which a "
                               "particular instance is based.'"]},
    "swe_agent": {"urls": [f"https://huggingface.co/datasets/{SWE_AGENT[0]}/blob/{SWE_AGENT[1]}/README.md",
                           "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"],
                  "quotes": ["The dataset is licensed under the Creative Commons Attribution 4.0 license. ... "
                             "Additionally, a notice for users: if you intend to use the outputs of these models, you "
                             "must comply with the Llama 3.1 License.",
                             "Llama 3.1: 'If you use the Llama Materials or any outputs or results of the Llama "
                             "Materials to create, train, fine tune, or otherwise improve an AI model, which is "
                             "distributed or made available, you shall also include \"Llama\" at the beginning of any "
                             "such AI model name.'", "model_name values: swe-agent-llama-70b / -8b / -405b (all rows)"]},
    "ecthr": {"urls": [f"https://huggingface.co/datasets/{LEXGLUE[0]}/blob/{LEXGLUE[1]}/README.md",
                       "https://arxiv.org/abs/2110.00976", "https://www.echr.coe.int/copyright-and-disclaimer"],
              "quotes": ["card YAML: license: cc-by-4.0; '### Licensing Information [More Information Needed]'",
                         "LexGLUE paper: 'We use all datasets in accordance with copyright terms and under the licenses "
                         "set forth by their creators.'",
                         "ECHR: 'The information and texts available on the Court's website may be reproduced provided "
                         "the source is acknowledged (c ECHR-CEDH) and the reproduction is made for private use or for "
                         "the purposes of information and education in connection with the Court's activities. ... "
                         "Persons wishing to use information and texts on the Court's website for any use other than "
                         "those listed above, and in particular for commercial use, must request prior written "
                         "permission'"]},
}
GDE1_SETS = {"aidev": "aidev_prs", "swiss": "swiss_judgment", "swe_rebench": "swe_rebench_oh",
             "swe_agent": "swe_agent", "ecthr": "ecthr"}
TRAINABLE = ("aidev", "afd", "swiss", "swe_rebench")
DATASETS = ("aidev", "afd", "swiss", "swe_rebench", "swe_agent", "ecthr")
SKIPPED = {
    "autocast": ("SKIP (train and eval). 'EleutherAI/autocast' does not exist on the Hub (HTTP 404, also with a token). "
                 "The real Autocast (github.com/andyzoujm/autocast, Zou et al. 2022): 'We obtained permission from "
                 "Metaculus to host the dataset on GitHub for research purposes only.' Research-only -> never train; "
                 "the questions (Metaculus, GJ Open, CSET-Foretell) resolved by mid-2022, so their outcomes are in "
                 "pretraining and the set cannot be a GDE-1 O eval; the news corpus is CC-NEWS (Common Crawl terms; "
                 "publisher copyright). ForecastBench market questions (permission documented) cover the same need."),
    "peerread": ("SKIP (train and eval). github.com/allenai/PeerRead has no LICENSE (API licence null) and the HF card "
                 "says 'unknown'; only data/acl_2017 and data/conll_2016 carry a LICENSE.txt (CC BY 4.0) and their "
                 "review files have no `accepted` field (labels need a title match against acl_accepted.txt, a list of "
                 "19,743 ACL titles); iclr_2017 is OpenReview content with no licence file; NIPS is not distributed "
                 "('Due to licensing constraints'); arXiv labels are heuristic venue matches. Decisions are 2013-2017 "
                 "(in pretraining). ICLR 2017 review files carry the 'ICLR committee final decision' note among the "
                 "reviews, and the parsed PDFs are the latest OpenReview revisions (post-review). No overlap with our "
                 "OpenReview pipeline (ICLR 2024-2026, scores only)."),
}
QIDS = ("pr_merged", "afd_close", "afd_deleted", "appeal_approved", "patch_resolves_issue")   # + violation_art_<n>


# =========================================================================== small helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def uh(*parts):
    return gb.unit_hash(*parts)


def day(ts):
    """ISO date (UTC) from an ISO string or a unix timestamp."""
    if ts is None or ts == "":
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    return str(ts)[:10]


def clip(text, n, tail=0, mark="[...]"):
    """Keep the first n - tail characters and the last `tail` (the end of a fact section or an issue often holds the
    request), with a marker between; unchanged if short enough."""
    text = str(text or "").strip()
    if len(text) <= n:
        return text
    if tail <= 0:
        return text[:n].rstrip() + " " + mark
    return text[:n - tail].rstrip() + f"\n{mark}\n" + text[-tail:].lstrip()


def tidy(text):
    text = re.sub(r"\r\n?", "\n", str(text or ""))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def hf_url(pin, path):
    return f"https://huggingface.co/datasets/{pin[0]}/resolve/{pin[1]}/{path}"


def http_open(url, headers=None, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})}),
                                  timeout=timeout)


def download(url, dest, tries=6):
    """Stream a URL to disk (cached; .part then rename)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for t in range(tries):
        try:
            with http_open(url, timeout=600) as r, tmp.open("wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if t == tries - 1:
                raise
            log(f"  download retry {t + 1}: {e}")
            time.sleep(5 * (t + 1))
    return dest


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP Range requests, so pyarrow can read selected parquet COLUMNS of a remote
    file without downloading it (the openhands trajectories file is 2.1 GB; its model_patch column is 123 MB). The HF
    resolve URL redirects to a signed CDN URL, which is re-resolved on 403 / expiry."""

    def __init__(self, url, block=8 << 20):
        super().__init__()
        self.src, self.block, self.pos, self.cache = url, block, 0, {}
        self._resolve()

    def _resolve(self):
        with http_open(self.src, {"Range": "bytes=0-0"}, timeout=120) as r:
            self.url = r.geturl()
            cr = r.headers.get("Content-Range") or ""
            self.size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, a, b):
        for t in range(6):
            try:
                with http_open(self.url, {"Range": f"bytes={a}-{b - 1}"}, timeout=300) as r:
                    data = r.read()
                if len(data) == b - a:
                    return data
                raise OSError(f"short read {len(data)} of {b - a}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if t == 5:
                    raise
                if isinstance(e, urllib.error.HTTPError) and e.code in (403, 410):
                    self._resolve()
                time.sleep(3 * (t + 1))
        return b""

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n >= self.block:                  # a whole column chunk: one request
            data = self._get(self.pos, self.pos + n)
        else:
            out, p, end = [], self.pos, self.pos + n
            while p < end:
                k = p // self.block
                if k not in self.cache:
                    if len(self.cache) > 32:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[k] = self._get(k * self.block, min(self.size, (k + 1) * self.block))
                blk = self.cache[k]
                s = p - k * self.block
                take = blk[s:s + (end - p)]
                out.append(take)
                p += len(take)
            data = b"".join(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def parquet_columns(url, dest, columns):
    """Selected columns of a remote parquet file, cached locally (zstd parquet)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = HttpRangeFile(url)
    pf = pq.ParquetFile(f)
    have = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in have]
    t = pa.concat_tables([pf.read_row_group(i, columns=cols) for i in range(pf.metadata.num_row_groups)])
    tmp = dest.with_name(dest.name + ".part")
    pq.write_table(t, tmp, compression="zstd")
    tmp.replace(dest)
    log(f"  {dest.name}: {t.num_rows} rows, columns {cols}, {time.time() - t0:.0f}s")
    return dest


def read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def noul_crit(yes, no):
    return {"yes": yes, "no": no}


def onehot(keys, k):
    return {x: (1.0 if x == k else 0.0) for x in keys}


BOILER_MIN_WORDS = 5


def boiler_lines(texts_groups, min_count, min_groups):
    """Lines (>= 5 words, normalized) that recur across many items AND many groups (repositories, ...): PR / issue
    templates, agent footers. They carry no item information, and left in they make ext2's 3-shingle rule match
    unrelated items (a training PR and an eval PR sharing only a footer). Computed from text only, never labels."""
    cnt, grp = Counter(), defaultdict(set)
    for text, g in texts_groups:
        for ln in {norm(x) for x in str(text or "").split("\n")}:
            if len(re.findall(r"\w+", ln)) >= BOILER_MIN_WORDS:
                cnt[ln] += 1
                if len(grp[ln]) < min_groups:
                    grp[ln].add(g)
    return frozenset(ln for ln, c in cnt.items() if c >= min_count and len(grp[ln]) >= min_groups)


def strip_boiler(text, boiler):
    if not boiler:
        return str(text or "")
    return "\n".join(x for x in str(text or "").split("\n") if norm(x) not in boiler)


# =========================================================================== record makers
def train_rec(source, dataset, state, qs, outcome, meta, group, probes, pin):
    """A training CANDIDATE: one-hot labels (blended later), meta.outcome = the realised option per question."""
    labs = {q: onehot(gb_keys(qs[q]), outcome[q]) for q in qs}
    rec = record(source, state, qs, labs, {q: "hard" for q in qs}, None)
    rec["base_id"] = rec["id"]
    rec["license"] = LICENSES[dataset]
    rec["meta"] = {"dataset": pin, "group": str(group), "label_source": {q: "outcome" for q in qs},
                   "outcome": dict(outcome), **meta}
    rec["_probes"] = [p for p in probes if p]
    return rec


def gb_keys(q):
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["yes", "no"]
    return list(q["levels"])


def eval_rec(dataset, source, state, qs, outcome, meta, base_id, rec_id):
    gset = GDE1_SETS[dataset]
    labs = {q: onehot(gb_keys(qs[q]), outcome[q]) for q in qs}
    rec = record(source, state, qs, labs, {q: "hard" for q in qs}, None)
    rec["id"] = rec_id
    rec["base_id"] = base_id
    rec["license"] = LICENSES[dataset]
    rec["meta"] = {"gde1_set": gset, "domain": "O", "gde1_part": "outcomes2", "eval_only": True,
                   "label_source": {q: "outcome" for q in qs}, "outcome": dict(outcome), **meta}
    validate(rec)
    return rec


# =========================================================================== leave-one-out base-rate blend (train)
def blend(cands, cell_of, lam, a=20.0):
    """t = lam * onehot + (1 - lam) * LOO cell rate, the cell rate shrunk toward the LOO global rate with strength a.
    Item i's own outcome never enters its own rate. lam >= 1 leaves the one-hot labels."""
    if lam >= 1:
        return cands
    tot, cell = defaultdict(Counter), defaultdict(Counter)
    for r in cands:
        for q, y in r["meta"]["outcome"].items():
            tot[q][y] += 1
            cell[(q, cell_of(r))][y] += 1
    for r in cands:
        c = cell_of(r)
        for q, y in r["meta"]["outcome"].items():
            keys = list(r["labels"][q])
            n_g = sum(tot[q].values()) - 1
            g = {k: (tot[q][k] - (k == y) + 1.0) / (n_g + len(keys)) for k in keys}
            cc = cell[(q, c)]
            n_c = sum(cc.values()) - 1
            rate = {k: (cc[k] - (k == y) + a * g[k]) / (n_c + a) for k in keys}
            r["labels"][q] = gb.lab_dict(keys, [lam * (k == y) + (1 - lam) * rate[k] for k in keys])
            r["label_kind"][q] = gb.kind_of(r["labels"][q])
            r["meta"].setdefault("label_method", {})[q] = (f"{lam}*onehot(outcome) + {round(1 - lam, 6)}*LOO rate of "
                                                         f"cell [{c}] shrunk to the LOO global rate (a={a})")
    return cands


# =========================================================================== the gate's shingle rule, on templated text
class DfIndex(x2.Index):
    """build_jev_ext2.Index (exact and leaf rules unchanged) with its 3-shingle rule made robust to STOCK PHRASING, the
    failure mode documented in research/gde1_part_a.md 2.2 / gde1_part_b.md 6: legal formulae, PR / issue templates and
    common code lines are shared by thousands of unrelated documents, and the plain rule dropped 95% of the Swiss and
    40% of the AIDev / SWE training candidates against their own held-out eval sets on such shingles alone.
      rare     >= 3 shared shingles with ONE fingerprint state, counting only shingles found in <= MAX_DF fingerprint
               states, and covering >= MIN_COVER (10%) of the shorter of the two texts' shingles (part A's calibrated
               gate, build_gde1_a.Gate: df bound 20, coverage 10%; ext2's count of 3 is kept);
      contain  >= 3 shared shingles with one state AND covering >= MIN_CONTAIN (80%) of the candidate's shingles, whatever
               their df (shingles in more than MAX_DF_CONTAIN states are skipped for cost): a near-copy of an eval item
               that sits in a cluster of > 20 near-identical items is still caught.
    Everything else (eval-tag priority, reporting) is the parent's."""
    MAX_DF = 20
    MIN_COVER = 0.10
    MIN_CONTAIN = 0.8
    MAX_DF_CONTAIN = 2000

    def _shingle(self, nf):
        import numpy as np
        hs = x2.shingle_hashes(nf)
        if len(hs) < x2.SHINGLE_MIN_SHARED or self.H is None or not len(self.H):
            return None
        keys = np.fromiter(hs, dtype=np.int64, count=len(hs))
        lo, hi = np.searchsorted(self.H, keys, "left"), np.searchsorted(self.H, keys, "right")
        df = hi - lo
        if getattr(self, "NS", None) is None:
            self.NS = np.bincount(self.S, minlength=len(self.ST)) if len(self.S) else np.zeros(0, np.int64)
        rules = [("rare", (df > 0) & (df <= self.MAX_DF), 0)]
        if (df > 0).sum() >= max(x2.SHINGLE_MIN_SHARED, self.MIN_CONTAIN * len(hs)):
            rules.append(("contain", (df > 0) & (df <= self.MAX_DF_CONTAIN), self.MIN_CONTAIN * len(hs)))
        for _, mask, need in rules:
            nz = np.nonzero(mask)[0]
            if len(nz) < max(x2.SHINGLE_MIN_SHARED, need):
                continue
            ids = np.concatenate([self.S[lo[i]:hi[i]] for i in nz])
            u, c = np.unique(ids, return_counts=True)
            sel = (c >= x2.SHINGLE_MIN_SHARED) & (c >= need)
            if not need:     # rare rule: part A's coverage condition, >= MIN_COVER of the shorter text's shingles
                sel &= c >= self.MIN_COVER * np.minimum(len(hs), self.NS[u])
            if not sel.any():
                continue
            sid, n = min(zip(u[sel].tolist(), c[sel].tolist()),
                         key=lambda x: (self.klass[self.ST[x[0]]] != "eval", -x[1]))
            shared = [hs[int(keys[i])] for i in nz if (self.S[lo[i]:hi[i]] == sid).any()][:2]
            return ("shingle", int(self.ST[sid]), n, shared)
        return None


def load_gate(a):
    """x2.load_fingerprints with DfIndex in place of Index (the module attribute is swapped for the call only);
    --gate plain keeps ext2's Index exactly."""
    if getattr(a, "gate", "calibrated") == "plain":
        return x2.load_fingerprints(a)
    orig = x2.Index
    x2.Index = DfIndex
    try:
        return x2.load_fingerprints(a)
    finally:
        x2.Index = orig


GROUP_CAP = {"aidev": 100, "swe_rebench": 30}     # records per repository (train and eval): no repo dominates


def cap_groups(recs, k, seed, tag, key):
    if not k:
        return recs, 0
    by = defaultdict(list)
    for r in recs:
        by[key(r)].append(r)
    out, drop = [], 0
    for g in sorted(by):
        rs = sorted(by[g], key=lambda r: uh(seed, tag, r["id"]))
        out += rs[:k]
        drop += max(0, len(rs) - k)
    return out, drop


# =========================================================================== AIDev
AIDEV_FILES = ("pull_request.parquet", "repository.parquet")
AGENTS = {"OpenAI_Codex": "OpenAI Codex", "Devin": "Devin", "Cursor": "Cursor", "Claude_Code": "Claude Code",
          "Google_Jules": "Google Jules", "Copilot": "GitHub Copilot"}
AIDEV_DROP_AGENTS = {"Copilot": "the Copilot coding agent opens a '[WIP]' draft PR at once and rewrites its title and "
                                "description as it works: the stored final text encodes post-creation progress"}
PERMISSIVE = {"MIT", "MIT-0", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "ISC", "Unlicense", "CC0-1.0",
              "Zlib", "BSL-1.0", "UPL-1.0", "CC-BY-4.0", "PostgreSQL", "Python-2.0", "NCSA", "X11"}
WIP = re.compile(r"^\s*(\[\s*(wip|draft)\s*\]|\(\s*wip\s*\)|wip\b|draft\s*:)", re.I)
SELF_CLOSE = re.compile(r"(closing (this|in favou?r)|superseded by|closed in favou?r of|this (pr|pull request) "
                        r"(was|has been|is now) (merged|closed|abandoned)|\[(closed|abandoned|superseded)\]|"
                        r"no longer (needed|required)\b)", re.I)
# agent session links and agent footers: whole lines removed (no item information; a requester's name in Devin's)
AGENT_LINE = re.compile(r"chatgpt\.com/codex/tasks/|app\.devin\.ai/sessions/|cursor\.(com|sh)/\S*agent|"
                        r"jules\.google\.com/|claude\.ai/code|^\W*Requested by\b|Co-Authored-By:|"
                        r"Generated with \W*Claude Code|Link to Devin run|^\s*-{4,}\s*$", re.I)
AIDEV_Q = {"type": "noul",
           "instructions": "Will this pull request be merged into the repository, rather than closed without merging?",
           "criteria": noul_crit("A maintainer merges the pull request.",
                                 "The pull request is closed without being merged.")}
AIDEV_EVAL = ("2025-08-01", "2025-09-30")     # created in this window (inclusive), held-out repositories
AIDEV_TRAIN_BEFORE = "2025-07-01"             # created before; closed before AIDEV_TRAIN_CLOSED_BEFORE
AIDEV_TRAIN_CLOSED_BEFORE = "2025-08-01"
AIDEV_HELDOUT = 0.25
BODY_MAX = 2500


# Blocks that review / CI bots write INTO the description after the PR is opened (several say they "update as commits
# are pushed"): post-creation information, removed whole before anything else.
BOT_BLOCKS = {
    "ellipsis": r"<!-- ELLIPSIS_HIDDEN -->.*?(?:<!-- ELLIPSIS_HIDDEN -->|\Z)",
    "auto_generated(coderabbit/cubic/mrge/cypress)": r"<!-- This is an auto-generated (?:comment|description)[^>]*-->.*?"
                                                     r"(?:<!-- [Ee]nd of auto-generated[^>]*-->|\Z)",
    "pr_codex": r"<!-- start pr-codex -->.*?(?:<!-- end pr-codex -->|\Z)",
    "cursor_bugbot": r"<!-- CURSOR_SUMMARY -->.*?(?:<!-- /CURSOR_SUMMARY -->|\Z)",
    "deploy_preview": r"<!-- PREVIEW-TABLE-START -->.*?(?:<!-- PREVIEW-TABLE-END -->|\Z)",
    "reviewable": r"<!-- Reviewable:start -->.*?(?:<!-- Reviewable:end -->|\Z)",
    "coverage_or_test_links": r"<!-- (CLI_COVERAGE|DBUX-TEST-URL)[_-]START -->.*?(?:<!-- \1[_-]END -->|\Z)",
    "propel": r"<!-- Summary by @[\w-]+ -->.*\Z",
    "summary_heading_to_end": r"(?m)^#+\s*Summary by (?:Sourcery|CodeRabbit|cubic|mrge)\b.*\Z",
}
BOT_RE = {k: re.compile(v, re.S) for k, v in BOT_BLOCKS.items()}


def strip_bots(body, counts=None):
    b = str(body or "")
    for k, pat in BOT_RE.items():
        b, n = pat.subn("", b)
        if n and counts is not None:
            counts[f"bot_blocks_removed/{k}"] += n
    return b


def aidev_clean(body):
    b = re.sub(r"<!--.*?-->", "", strip_bots(body), flags=re.S)
    return "\n".join(x for x in b.replace("\r", "").split("\n") if not AGENT_LINE.search(x))


def aidev_state(pr, repo, boiler=frozenset()):
    """Header lines keep fixed word runs short (< 8 words, broken by the date / repository), so the shingle gate
    compares descriptions, not this template."""
    lic = repo.get("license") or "not stated"
    lang = repo.get("language") or "unknown"
    body = clip(tidy(strip_boiler(aidev_clean(pr.get("body")), boiler)), BODY_MAX, mark="[... description truncated]")
    return "\n".join([f"Opened {day(pr['created_at'])} by {AGENTS.get(pr['agent'], pr['agent'])}",
                      f"Repository: {repo.get('full_name') or pr.get('repo_url')} ({lang}; licence {lic})",
                      f"Title: {tidy(pr.get('title')) or '(no title)'}", "Description:", body or "(empty)"])


def aidev_convert(prs, repos, seed=SEED, boiler=None):
    """-> ({'train': candidates, 'eval': records}, report). Pure. boiler: template lines to strip (default: computed
    over every PR description, >= 25 PRs in >= 10 repositories)."""
    rep, out = Counter(), {"train": [], "eval": []}
    rp = {r["id"]: r for r in repos}
    if boiler is None:
        boiler = boiler_lines(((aidev_clean(p.get("body")), p.get("repo_id")) for p in prs), 25, 10)
    rep["boilerplate_lines_stripped"] = len(boiler)
    assoc = Counter()
    for pr in prs:
        rep["rows"] += 1
        if pr.get("state") != "closed":
            rep["dropped/open_unresolved"] += 1
            continue
        merged = pr.get("merged_at") is not None
        if pr.get("agent") in AIDEV_DROP_AGENTS:
            rep[f"dropped/agent_{pr['agent']}"] += 1
            continue
        title = str(pr.get("title") or "")
        if WIP.match(title):
            rep["dropped/wip_title"] += 1
            assoc[("wip_title", merged)] += 1
            continue
        strip_bots(pr.get("body"), rep)
        if SELF_CLOSE.search(str(pr.get("body") or "")):
            rep["dropped/body_announces_close_or_merge"] += 1
            assoc[("self_close_text", merged)] += 1
            continue
        repo = rp.get(pr["repo_id"])
        if repo is None:
            rep["dropped/no_repository_row"] += 1
            continue
        created, closed = day(pr["created_at"]), day(pr.get("closed_at"))
        if not closed or closed < created:
            rep["dropped/bad_dates"] += 1
            continue
        held = uh(seed, "aidev-heldout-repo", pr["repo_id"]) < AIDEV_HELDOUT
        y = "yes" if merged else "no"
        state = aidev_state(pr, repo, boiler)
        meta = {"label_provenance": "outcome (GitHub merge state recorded by AIDev: merged_at set vs closed unmerged)",
                "decision_date": created, "info_date": created, "aidev_pr_id": pr["id"], "pr_number": pr.get("number"),
                "html_url": pr.get("html_url"), "repo": repo.get("full_name"), "repo_id": pr["repo_id"],
                "repo_license": repo.get("license"), "agent": pr["agent"], "created_at": pr["created_at"],
                "closed_at": pr.get("closed_at"), "source_revision": AIDEV[1], "td_workflow": None}
        if held:
            if not (AIDEV_EVAL[0] <= created <= AIDEV_EVAL[1]):
                rep["heldout_repo/outside_eval_window"] += 1
                continue
            m = dict(meta, resolution_date={"pr_merged": closed}, time_slice=f"created {AIDEV_EVAL[0]}..{AIDEV_EVAL[1]}")
            out["eval"].append(eval_rec("aidev", "aidev/pr_merge", state, {"pr_merged": dict(AIDEV_Q)},
                                        {"pr_merged": y}, m, base_id=f"aidev/repo{pr['repo_id']}",
                                        rec_id=f"gde1-aidev-{pr['id']}"))
            rep["eval/candidates"] += 1
            continue
        if created >= AIDEV_TRAIN_BEFORE or closed >= AIDEV_TRAIN_CLOSED_BEFORE:
            rep["train_repo/outside_train_window"] += 1
            continue
        if repo.get("license") not in PERMISSIVE:
            rep[f"train_repo/licence_not_permissive({repo.get('license')})"] += 1
            continue
        out["train"].append(train_rec("aidev/pr_merge", "aidev", state, {"pr_merged": dict(AIDEV_Q)}, {"pr_merged": y},
                                      dict(meta, resolution_date={"pr_merged": closed}), group=f"repo{pr['repo_id']}",
                                      probes=[title, pr.get("body")], pin=f"{AIDEV[0]}@{AIDEV[1][:8]}"))
        rep["train/candidates"] += 1
    rep.update({f"leak_assoc/{k}/{'merged' if m else 'closed'}": v for (k, m), v in assoc.items()})
    return out, dict(rep)


def aidev_fetch(a):
    root = Path(a.work) / "aidev" / AIDEV[1][:12]
    for f in AIDEV_FILES:
        download(hf_url(AIDEV, f), root / f)
    return root


def load_aidev(a):
    root = aidev_fetch(a)
    prs, repos = read_parquet(root / "pull_request.parquet"), read_parquet(root / "repository.parquet")
    out, rep = aidev_convert(prs, repos, a.seed)
    censor = Counter((p["created_at"][:7], p["state"]) for p in prs if p.get("agent") not in AIDEV_DROP_AGENTS)
    rep["open_share_by_created_month"] = {m: round(censor[(m, "open")] / max(1, censor[(m, "open")] +
                                                                          censor[(m, "closed")]), 3)
                                          for m in sorted({k[0] for k in censor})}
    return out, rep, {"source": {"repo": AIDEV[0], "revision": AIDEV[1], "files": list(AIDEV_FILES),
                                 "subset": "AIDev-pop (repositories with > 100 stars)"},
                      "time_split": {"eval": f"held-out repositories ({AIDEV_HELDOUT:.0%} by hash), created "
                                             f"{AIDEV_EVAL[0]}..{AIDEV_EVAL[1]}",
                                     "train": f"other repositories, created < {AIDEV_TRAIN_BEFORE}, closed < "
                                              f"{AIDEV_TRAIN_CLOSED_BEFORE}, permissive repository licence"},
                      "dropped_agents": AIDEV_DROP_AGENTS}


# =========================================================================== Wikipedia AfD
AFD_DIR = "wiki-articles-for-deletion-corpus/"
AFD_MAP = {"delete": "delete", "delete speedy": "delete", "copyright delete": "delete",
           "copyright delete speedy": "delete", "keep": "keep", "keep speedy": "keep", "no-consensus": "no_consensus",
           "merge": "merge", "merge redirect": "merge", "redirect": "redirect", "redirect speedy": "redirect"}
AFD_OPTIONS = {"delete": "Delete: the article is deleted (including a speedy deletion).",
               "keep": "Keep: the debate closes with consensus to keep the article (including a speedy keep).",
               "no_consensus": "No consensus: the debate closes without consensus, so the article is kept by default.",
               "merge": "Merge: the article's content is merged into another article.",
               "redirect": "Redirect: the article is replaced by a redirect to another article, without a merge."}
AFD_YEARS = ("2005-01-01", "2018-12-31")
AFD_LEAK = re.compile(r"the result (was|of the debate)|this page is an archive|please do not modify it|"
                      r"subsequent comments should be made|non-admin closure|\(NAC\)|closed as|closing as", re.I)
AFD_MAX_VOTES = 30
AFD_VOTE_CHARS = 350
AFD_NOM_CHARS = 2500


def wiki_plain(text):
    t = str(text or "")
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\d{1,2}:\d{2}, \d{1,2} \w+ \d{4} \(UTC\)", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"^\s*\*?\s*VOTE\b\s*[-:,.]*\s*", "", t)
    return tidy(t)


def afd_group(title):
    return re.sub(r"\s*\((\d+(st|nd|rd|th)|second|third|fourth|fifth) nomination\)\s*$", "", str(title or ""),
                  flags=re.I).strip().lower()


def afd_records(convs, utts, variant_of, seed=SEED):
    """convs: {cid: meta}; utts: {cid: [utterance dicts]} -> (candidates by source, report). Pure."""
    rep, out = Counter(), defaultdict(list)
    rng = random.Random(f"{seed}:afd")
    for cid, m in convs.items():
        lab = AFD_MAP.get(str(m.get("outcome_label") or ""))
        close_ts = m.get("outcome_timestamp")
        if lab is None:
            rep[f"dropped/label_{m.get('outcome_label')}"] += 1
            continue
        if not close_ts:
            rep["dropped/no_close_timestamp"] += 1
            continue
        close = day(close_ts)
        if not (AFD_YEARS[0] <= close <= AFD_YEARS[1]):
            rep["dropped/outside_2005_2018"] += 1
            continue
        us = sorted((u for u in utts.get(cid, []) if u.get("timestamp")), key=lambda u: u["timestamp"])
        noms = [u for u in us if (u.get("meta") or {}).get("type") == "nomination" and u["timestamp"] < close_ts]
        if not noms or len(wiki_plain(noms[0].get("text"))) < 20:
            rep["dropped/no_pre_close_nomination"] += 1
            continue
        nom = noms[0]
        nom_text = clip(wiki_plain(nom.get("text")), AFD_NOM_CHARS)
        if AFD_LEAK.search(nom_text):
            rep["dropped/nomination_outcome_words"] += 1
            continue
        title = str(m.get("article_title") or "").strip()
        variant = variant_of(cid)
        lines = [f'Articles for Deletion (English Wikipedia), {close[:4]}: "{title}"',
                 f"Nominated on {day(nom['timestamp'])}. Nomination statement:", nom_text]
        n_votes_all = sum(1 for u in utts.get(cid, []) if (u.get("meta") or {}).get("type") == "vote")
        votes = [u for u in us if (u.get("meta") or {}).get("type") == "vote" and u["timestamp"] < close_ts
                 and (u.get("meta") or {}).get("label")]
        rep["votes/after_close_or_untimed_excluded"] += n_votes_all - len(votes)
        info = day(nom["timestamp"])
        shown = 0
        if variant == "nomination_votes":
            vl = []
            for u in votes[:AFD_MAX_VOTES]:
                txt = clip(wiki_plain(u.get("text")), AFD_VOTE_CHARS)
                if AFD_LEAK.search(txt):
                    rep["votes/outcome_words_skipped"] += 1
                    continue
                vl.append(f"- {day(u['timestamp'])} {u['meta']['label']}: {txt or '(no rationale)'}")
            head = f"Votes signed before the close ({len(vl)} shown, earliest first):"
            budget = 6000 - len("\n".join(lines))
            keep = []
            for v in vl:
                if len("\n".join(keep + [v])) + len(head) > budget:
                    break
                keep.append(v)
            lines += [head.replace(f"({len(vl)} shown", f"({len(keep)} shown")] + (keep or ["- (none)"])
            shown = len(keep)
            if keep:
                info = max(info, max(day(u["timestamp"]) for u in votes[:shown]))
        state = "\n".join(lines)
        crit = shuffled(dict(AFD_OPTIONS), rng)
        qs = {"afd_close": choice_q("How will the closing administrator close this deletion debate?", crit),
              "afd_deleted": {"type": "noul", "instructions": "Will the debate close with the article deleted?",
                              "criteria": noul_crit("The closing decision is to delete the article.",
                                                    "The article is kept, merged, redirected or closed without "
                                                    "consensus.")}}
        outcome = {"afd_close": lab, "afd_deleted": "yes" if lab == "delete" else "no"}
        meta = {"label_provenance": "outcome (closing administrator's normalized label, AfD corpus)",
                "decision_date": close, "info_date": info, "afd_conversation_id": cid, "article_title": title,
                "close_label_raw_normalized": m.get("outcome_label"), "variant": variant,
                "n_votes_total": n_votes_all, "n_votes_before_close": len(votes), "n_votes_shown": shown,
                "wikipedia_url": "https://en.wikipedia.org/wiki/Wikipedia:Articles_for_deletion/" +
                                 title.replace(" ", "_"), "close_year": int(close[:4]), "td_workflow": None}
        src = f"afd/{variant}"
        out[src].append(train_rec(src, "afd", state, qs, outcome, meta, group=afd_group(title),
                                  probes=[nom_text], pin="ConvoKit wiki-articles-for-deletion-corpus (AFD_Decision_Corpus 1.0)"))
        rep[f"candidates/{src}"] += 1
    return dict(out), dict(rep)


def load_afd(a):
    root = Path(a.work) / "afd"
    z_path = download(AFD_URL, root / "wiki-afd.zip")
    if z_path.stat().st_size != AFD_BYTES:
        log(f"afd: WARNING size {z_path.stat().st_size} != pinned {AFD_BYTES}")
    z = zipfile.ZipFile(z_path)
    convs = {cid: v["meta"] for cid, v in json.load(z.open(AFD_DIR + "conversations.json")).items()}
    want = max(1, int(a.afd_cap * a.oversample))
    elig = [cid for cid, m in convs.items() if AFD_MAP.get(str(m.get("outcome_label") or "")) and
            m.get("outcome_timestamp") and AFD_YEARS[0] <= day(m["outcome_timestamp"]) <= AFD_YEARS[1]]

    def variant_of(cid):
        return "nomination" if uh(a.seed, "afd-variant", cid) < 0.5 else "nomination_votes"
    pick = set()
    for v in ("nomination", "nomination_votes"):
        ids = sorted((c for c in elig if variant_of(c) == v), key=lambda c: uh(a.seed, "afd-sample", c))
        pick.update(ids[:want])
    log(f"afd: {len(convs)} debates, {len(elig)} eligible, {len(pick)} sampled; streaming utterances")
    utts = defaultdict(list)
    cid_re = re.compile(rb'"conversation_id": "(\d+)"')
    with z.open(AFD_DIR + "utterances.jsonl") as fh:
        for line in fh:
            m = cid_re.search(line)
            if m and m.group(1).decode() in pick:
                u = json.loads(line)
                utts[u["conversation_id"]].append({"timestamp": u.get("timestamp"), "text": u.get("text"),
                                                   "meta": {"type": (u.get("meta") or {}).get("type"),
                                                            "label": (u.get("meta") or {}).get("label")}})
    out, rep = afd_records({c: convs[c] for c in pick}, utts, variant_of, a.seed)
    rep["debates_in_corpus"], rep["eligible_debates"], rep["sampled_debates"] = len(convs), len(elig), len(pick)
    rep["label_dist_eligible"] = dict(Counter(AFD_MAP[str(convs[c]["outcome_label"])] for c in elig))
    rep["zip_sha256"] = sha256_of(z_path)
    return out, rep


# =========================================================================== Swiss judgment prediction (XL)
SWISS_FIELDS = ("decision_id", "year", "language", "facts", "label", "law_area", "law_sub_area", "chamber", "court",
                "canton", "region")
SWISS_LANG = {"de": "German", "fr": "French", "it": "Italian"}
SWISS_DISP = re.compile(r"gutgeheissen|gutzuheissen|abgewiesen|abzuweisen|\badmis\b|\badmise?s?\b|rejet[ée]e?s?|"
                        r"accolt[oaie]|respint[oaie]", re.I)
SWISS_FED = re.compile(r"Bundesgericht|Tribunal f[ée]d[ée]ral|Tribunale federale", re.I)
SWISS_TODAY = re.compile(r"heut|aujourd'hui|aujourd’hui|ce jour|in data odierna|\boggi\b", re.I)
SWISS_SECTION = re.compile(r"(^|\n)\s*(Erwägungen?\s*:|Das Bundesgericht zieht in Erwägung|Demnach erkennt das "
                           r"Bundesgericht|Considérant en droit|Le Tribunal fédéral considère|Par ces motifs|"
                           r"Considerando in diritto|Il Tribunale federale considera|Per questi motivi)", re.I)
SWISS_Q = {"type": "noul",
           "instructions": ("Will the Swiss Federal Supreme Court approve this appeal (in full or in part), rather than "
                            "dismiss it?"),
           "criteria": noul_crit("The appeal is approved, fully or partly (gutgeheissen / admis / accolto).",
                                 "The appeal is dismissed (abgewiesen / rejeté / respinto).")}
SWISS_FACTS_MAX, SWISS_FACTS_TAIL = 5200, 2000


def swiss_leak(facts):
    """'section' / 'same_day_ruling' if the facts carry this court's ruling, else None."""
    if SWISS_SECTION.search(facts):
        return "considerations_or_operative_heading_in_facts"
    for s in re.split(r"(?<=[.;:])\s+", facts):
        if SWISS_FED.search(s) and SWISS_DISP.search(s) and SWISS_TODAY.search(s):
            return "same_day_federal_ruling"
    return None


def swiss_state(r):
    area = (r.get("law_area") or "law area not recorded").replace("_", " ")
    sub = r.get("law_sub_area")
    if sub and str(sub).lower() not in ("nan", "none"):
        area += f" / {str(sub).replace('_', ' ')}"
    # every fixed word run is broken by the year within 8 words (ext2's shingle size), so the header itself never
    # makes two decisions look alike
    lines = [f"Division {str(r.get('chamber') or '?').replace('CH_BGer_', '')} ({area}); "
             f"{SWISS_LANG.get(r.get('language'), r.get('language'))}; decided {r['year']}"]
    if r.get("canton") and r["canton"] not in ("CH", None):
        lines.append(f"Canton of origin: {r['canton']} ({r.get('region')})")
    lines += [clip(tidy(r["facts"]), SWISS_FACTS_MAX, SWISS_FACTS_TAIL)]
    return "\n".join(lines)


def swiss_convert(rows, split, rep, assoc):
    out = []
    for r in rows:
        rep[f"{split}/rows"] += 1
        facts = str(r.get("facts") or "")
        if len(facts) < 200 or r.get("label") not in ("approval", "dismissal"):
            rep[f"{split}/dropped/no_facts_or_label"] += 1
            continue
        y = "yes" if r["label"] == "approval" else "no"
        assoc[(split, "dispositive_word_anywhere", bool(SWISS_DISP.search(facts)), y)] += 1
        assoc[(split, "dispositive_word_last_600", bool(SWISS_DISP.search(facts[-600:])), y)] += 1
        why = swiss_leak(facts)
        if why:
            rep[f"{split}/dropped/{why}"] += 1
            continue
        state = swiss_state(r)
        year = str(r["year"])
        meta = {"label_provenance": "outcome (FSCS judgment: approval / dismissal, binarised by the dataset)",
                "decision_date": f"{year}-01-01", "info_date": f"{year}-01-01", "date_precision": "year",
                "decision_id": r["decision_id"], "language": r.get("language"), "law_area": r.get("law_area"),
                "law_sub_area": r.get("law_sub_area"), "chamber": r.get("chamber"), "canton": r.get("canton"),
                "region": r.get("region"), "year": int(year), "source_revision": SWISS[1], "td_workflow": None}
        if split == "test":
            m = dict(meta, resolution_date={"appeal_approved": f"{year}-12-31"},
                     time_slice=f"decided {year} (XL test split 2018-2022; PRE-cutoff: knowledge probe required)")
            out.append(eval_rec("swiss", "swiss/fscs", state, {"appeal_approved": dict(SWISS_Q)},
                                {"appeal_approved": y}, m, base_id=f"swiss/{r['decision_id']}",
                                rec_id=f"gde1-swiss-{r['decision_id']}"))
        else:
            out.append(train_rec("swiss/fscs", "swiss", state, {"appeal_approved": dict(SWISS_Q)},
                                 {"appeal_approved": y}, meta, group=r["decision_id"], probes=[facts[:2000]],
                                 pin=f"{SWISS[0]}@{SWISS[1][:8]}"))
    return out


def swiss_stream(a, split, max_rows):
    """The first `max_rows` rows of the XL split, read from the start of the xz stream (the rows are not ordered by
    year or language), cached as jsonl.gz with only SWISS_FIELDS."""
    cache = Path(a.work) / "swiss" / SWISS[1][:12] / f"{split}_first{max_rows}.jsonl.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh], json.loads(cache.with_suffix(".info").read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    dec, buf, rows, nbytes = lzma.LZMADecompressor(), b"", [], 0
    with http_open(hf_url(SWISS, f"data/huggingface/{split}.jsonl.xz"), timeout=600) as r:
        while len(rows) < max_rows:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            nbytes += len(chunk)
            buf += dec.decompress(chunk)
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line.strip() and len(rows) < max_rows:
                    o = json.loads(line)
                    rows.append({k: o.get(k) for k in SWISS_FIELDS})
    tmp = cache.with_name(cache.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for o in rows:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    tmp.replace(cache)
    info = {"split": split, "rows_read": len(rows), "compressed_bytes_read": nbytes}
    cache.with_suffix(".info").write_text(json.dumps(info))
    log(f"swiss {split}: {len(rows)} rows from the first {nbytes / 1e6:.0f} MB of the xz stream")
    return rows, info


def load_swiss(a):
    rep, assoc = Counter(), Counter()
    tr, tr_info = swiss_stream(a, "train", a.swiss_train_rows)
    te, te_info = swiss_stream(a, "test", a.swiss_test_rows)
    out = {"train": swiss_convert(tr, "train", rep, assoc), "eval": swiss_convert(te, "test", rep, assoc)}
    rep = dict(rep)
    rep["leak_association"] = {f"{s}/{k}/{'hit' if h else 'no_hit'}": {"n": assoc[(s, k, h, 'yes')] +
                                                                           assoc[(s, k, h, 'no')],
                                                                      "approval_rate": round(assoc[(s, k, h, 'yes')] /
                                                                                             max(1, assoc[(s, k, h, 'yes')] +
                                                                                                 assoc[(s, k, h, 'no')]), 4)}
                               for (s, k, h, _) in sorted(assoc, key=str)}
    rep["stream"] = {"train": tr_info, "test": te_info}
    rep["years"] = {"train": dict(Counter(r["year"] for r in tr)), "test": dict(Counter(r["year"] for r in te))}
    rep["languages"] = {"train": dict(Counter(r["language"] for r in tr)), "test": dict(Counter(r["language"] for r in te))}
    return out, rep, {"source": {"repo": SWISS[0], "revision": SWISS[1], "read": "prefix of data/huggingface/"
                                 "{train,test}.jsonl.xz", **rep["stream"]},
                      "time_split": {"eval": "XL test split (decided 2018-2022)", "train": "XL train split (2002-2015)",
                                     "unused": "XL validation (2016-2017), buffer"},
                      "pre_cutoff_warning": "decided 2018-2022 and published online: run the state-free knowledge "
                                            "probe before using this set in the O headline"}


# =========================================================================== SWE (openhands + SWE-agent)
SWE_HELDOUT = 0.20
SWE_STATE_MAX = 9000
ISSUE_MAX, ISSUE_TAIL = 2000, 500
SWE_Q = {"type": "noul",
         "instructions": ("Does this patch resolve the issue? That is: once it is applied, do the tests that the "
                          "repository's own fix for this issue makes pass now pass, without breaking tests that passed "
                          "before?"),
         "criteria": noul_crit("The patch passes the issue's fail-to-pass tests and keeps the pass-to-pass tests "
                               "passing.", "At least one of those tests fails with the patch applied.")}


SCRATCH_DIRS = (".openhands/",)
SCRATCH_PARTS = (".egg-info/", "__pycache__/")


def clean_patch(patch):
    """The agent's final diff without its scratch files: .openhands/TASKS.md (the agent's own progress log, with
    'verified' claims: a trajectory artefact), files the agent CREATED at the repository root (reproduce_issue.py,
    issue.md, comprehensive_test.py, final_verification.py, ...: 21,680 / 14,243 / 11,728 / 8,573 of the 67,074
    openhands patches) and build artefacts (*.egg-info, __pycache__). The harness runs the repository's own tests, so
    none of these files decides the label. -> (cleaned diff, [removed paths])."""
    blocks = re.split(r"(?m)^(?=diff --git a/)", str(patch or ""))
    keep, removed = [], []
    for b in blocks:
        m = re.match(r"diff --git a/(\S+) b/(\S+)", b)
        if not m:
            if b.strip():
                keep.append(b)
            continue
        path = m.group(2)
        new = bool(re.search(r"(?m)^new file mode", b[:400]))
        if path.startswith(SCRATCH_DIRS) or any(x in path for x in SCRATCH_PARTS) or (new and "/" not in path):
            removed.append(path)
            continue
        keep.append(b)
    return "".join(keep).strip(), removed


def swe_heldout(repo, seed=SEED):
    """ONE repository bucket for both SWE sources, so the openhands training repos never meet either eval set."""
    return uh(seed, "swe-heldout-repo", str(repo).lower()) < SWE_HELDOUT


def swe_state(repo, issue, patch, boiler=frozenset()):
    """Short fixed headers (see aidev_state); issue-template lines are stripped."""
    return "\n".join([f"Repository: {repo}", "Issue:", clip(tidy(strip_boiler(issue, boiler)), ISSUE_MAX, ISSUE_TAIL),
                      "Agent patch:", str(patch).strip()])


def swe_convert(rows, issues, dataset, seed=SEED, per_issue_train=3, per_issue_eval=2):
    """rows: {instance_id, repo, patch, resolved, model, traj_id, exit_status}; issues: {instance_id: {problem_statement,
    created_at, license}} -> ({'train', 'eval'}, report). Pure."""
    rep, out = Counter(), {"train": [], "eval": []}
    by = defaultdict(list)
    for r in rows:
        rep["rows"] += 1
        iss = issues.get(r["instance_id"])
        if iss is None or not str(iss.get("problem_statement") or "").strip():
            rep["dropped/no_issue_text"] += 1
            continue
        if not str(r.get("patch") or "").strip():
            rep["dropped/empty_patch"] += 1
            continue
        clean, removed = clean_patch(r["patch"])
        rep["patch_files_removed(scratch)"] += len(removed)
        if not clean:
            rep["dropped/patch_only_scratch_files"] += 1
            continue
        by[r["instance_id"]].append(dict(r, patch=clean, removed=removed))
    src = "swe/rebench_openhands" if dataset == "swe_rebench" else "swe/agent_trajectories"
    repo_of = {iid: grp[0]["repo"] for iid, grp in by.items()}
    boiler = boiler_lines(((issues[i]["problem_statement"], repo_of[i]) for i in by), 20, 10)
    rep["boilerplate_lines_stripped"] = len(boiler)
    released = OPENHANDS_RELEASED if dataset == "swe_rebench" else SWE_AGENT_RELEASED
    for iid in sorted(by):
        grp = sorted(by[iid], key=lambda r: uh(seed, "swe-pick", r["traj_id"]))
        repo = grp[0]["repo"]
        iss = issues[iid]
        held = swe_heldout(repo, seed)
        if dataset == "swe_agent" and not held:
            rep["dropped/not_heldout_repo(eval_only_source)"] += len(grp)
            continue
        lic = iss.get("license")
        if not held and not lic_permissive(lic):
            rep[f"train/dropped/repo_licence_not_permissive"] += len(grp)
            continue
        seen, kept = set(), 0
        for r in grp:
            if kept >= (per_issue_eval if held else per_issue_train):
                rep["dropped/per_issue_cap"] += 1
                continue
            state = swe_state(repo, iss["problem_statement"], r["patch"], boiler)
            if len(state) > SWE_STATE_MAX:
                rep["dropped/state_too_long"] += 1
                continue
            k = norm(r["patch"])
            if k in seen:
                rep["dropped/same_patch_same_issue"] += 1
                continue
            seen.add(k)
            kept += 1
            y = "yes" if int(bool(r["resolved"])) else "no"
            created = day(iss.get("created_at"))
            meta = {"label_provenance": "outcome (stored test result of the patch: SWE-rebench/SWE-bench harness)",
                    "decision_date": released, "info_date": min(created or released, released),
                    "instance_id": iid, "repo": repo, "repo_license": lic, "trajectory_id": r["traj_id"],
                    "model": r["model"], "exit_status_meta_only": str(r.get("exit_status") or "")[:60],
                    "scratch_files_removed_from_patch": r.get("removed") or [],
                    "fix_pr_created_at": iss.get("created_at"), "td_workflow": None,
                    "source_revision": (OPENHANDS if dataset == "swe_rebench" else SWE_AGENT)[1]}
            if held:
                m = dict(meta, resolution_date={"patch_resolves_issue": released},
                         time_slice=f"agent runs released {released}; repository-disjoint")
                out["eval"].append(eval_rec(dataset, src, state, {"patch_resolves_issue": dict(SWE_Q)},
                                            {"patch_resolves_issue": y}, m, base_id=f"{src}/{iid}",
                                            rec_id=f"gde1-{dataset}-{gb.sha16(r['traj_id'])}"))
            else:
                out["train"].append(train_rec(src, "swe_rebench", state, {"patch_resolves_issue": dict(SWE_Q)},
                                              {"patch_resolves_issue": y}, meta, group=f"repo:{repo}",
                                              probes=[iss["problem_statement"][:2000]],
                                              pin=f"{OPENHANDS[0]}@{OPENHANDS[1][:8]} + {REBENCH[0]}@{REBENCH[1][:8]}"))
    rep["eval/candidates"], rep["train/candidates"] = len(out["eval"]), len(out["train"])
    return out, dict(rep)


PERMISSIVE_WORDS = re.compile(r"^(mit|apache|bsd|isc|unlicense|the unlicense|cc0|zlib|0bsd|python software|psf|"
                              r"mit-cmu|historical permission|zope public license|boost|postgresql|ncsa|x11|upl)", re.I)


def lic_permissive(lic):
    s = str(lic or "").strip()
    if not s or s.lower() in ("none", "null", "other", "noassertion"):
        return False
    return bool(PERMISSIVE_WORDS.match(s) or PERMISSIVE_WORDS.match(s.replace("License", "").strip())
                or s in PERMISSIVE)


def load_swe_rebench(a):
    root = Path(a.work) / "swe"
    oh = read_parquet(parquet_columns(hf_url(OPENHANDS, "trajectories.parquet"), root / "oh_traj_cols.parquet",
                                      ["trajectory_id", "instance_id", "repo", "model_patch", "exit_status", "resolved"]))
    issues = {}
    for f in ("test-00000-of-00002", "test-00001-of-00002", "filtered-00000-of-00001"):
        for r in read_parquet(parquet_columns(hf_url(REBENCH, f"data/{f}.parquet"), root / f"rebench_{f}.parquet",
                                              ["instance_id", "repo", "created_at", "problem_statement",
                                               "license_name"])):
            issues.setdefault(r["instance_id"], {"problem_statement": r["problem_statement"],
                                                 "created_at": r["created_at"], "license": r["license_name"]})
    rows = [{"instance_id": r["instance_id"], "repo": r["repo"], "patch": r["model_patch"], "resolved": r["resolved"],
             "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct (OpenHands 0.54.0)", "traj_id": r["trajectory_id"],
             "exit_status": r["exit_status"]} for r in oh]
    out, rep = swe_convert(rows, issues, "swe_rebench", a.seed)
    return out, rep, {"source": {"trajectories": f"{OPENHANDS[0]}@{OPENHANDS[1]} (columns trajectory_id, instance_id, "
                                                 "repo, model_patch, exit_status, resolved; the trajectory column is "
                                                 "never read)", "issues": f"{REBENCH[0]}@{REBENCH[1]}"},
                      "split": f"repository-disjoint: {SWE_HELDOUT:.0%} of repositories (hash) are eval-only for both "
                               "SWE sources; <= 3 (train) / <= 2 (eval) patches per issue"}


def load_swe_agent(a):
    root = Path(a.work) / "swe"
    rows = []
    for i in range(12):
        f = parquet_columns(hf_url(SWE_AGENT, f"data/train-{i:05d}-of-00012.parquet"), root / f"sa_{i:02d}.parquet",
                            ["instance_id", "model_name", "target", "exit_status", "generated_patch"])
        for j, r in enumerate(read_parquet(f)):
            rows.append({"instance_id": r["instance_id"], "patch": r["generated_patch"], "resolved": r["target"],
                         "model": r["model_name"], "traj_id": f"sa{i:02d}-{j}", "exit_status": r["exit_status"]})
    issues = {}
    for r in read_parquet(parquet_columns(hf_url(SWE_EXTRA, "data/train-00000-of-00001.parquet"), root / "extra.parquet",
                                          ["instance_id", "repo", "created_at", "problem_statement", "license"])):
        issues[r["instance_id"]] = {"problem_statement": r["problem_statement"], "created_at": r["created_at"],
                                    "license": r["license"], "repo": r["repo"]}
    for r in read_parquet(parquet_columns(hf_url(SWE_BENCH, "data/dev-00000-of-00001.parquet"),
                                          root / "swebench_dev.parquet",
                                          ["instance_id", "repo", "created_at", "problem_statement"])):
        issues.setdefault(r["instance_id"], {"problem_statement": r["problem_statement"], "created_at": r["created_at"],
                                             "license": None, "repo": r["repo"]})
    for r in rows:
        r["repo"] = (issues.get(r["instance_id"]) or {}).get("repo") or r["instance_id"].rsplit("-", 1)[0]
    out, rep = swe_convert(rows, issues, "swe_agent", a.seed)
    rep["models"] = dict(Counter(r["model"] for r in rows))
    return out, rep, {"source": {"trajectories": f"{SWE_AGENT[0]}@{SWE_AGENT[1]} (columns instance_id, model_name, "
                                                 "target, exit_status, generated_patch; no trajectory / eval_logs)",
                                 "issues": [f"{SWE_EXTRA[0]}@{SWE_EXTRA[1]}", f"{SWE_BENCH[0]}@{SWE_BENCH[1]} (dev)"]},
                      "split": "eval only; the held-out repository bucket shared with swe_rebench"}


# =========================================================================== ECtHR (LexGLUE)
ECHR_ARTICLES = ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"]
ECHR_TITLES = {"2": "right to life", "3": "prohibition of torture and inhuman or degrading treatment",
               "5": "right to liberty and security", "6": "right to a fair trial",
               "8": "right to respect for private and family life",
               "9": "freedom of thought, conscience and religion", "10": "freedom of expression",
               "11": "freedom of assembly and association", "14": "prohibition of discrimination",
               "P1-1": "protection of property"}
ECHR_LEAK = re.compile(r"there (has|have) been (a |no )?violations? of|finds? (that )?there (has|have) been|"
                       r"just satisfaction|holds? unanimously|the court (holds|finds|concludes)\b", re.I)
ECHR_SPLITS = {"validation": ("2016-01-01", "2017-12-31", "LexGLUE dev (decided 2016-2017)"),
               "test": ("2017-01-01", "2019-12-31", "LexGLUE test (decided 2017-2019)")}
ECHR_FACTS_MAX = 9000


def echr_art_name(art):
    return "Article 1 of Protocol No. 1" if art == "P1-1" else f"Article {art}"


def ecthr_convert(a_rows, b_rows, split):
    rep, out = Counter(), []
    lo, hi, desc = ECHR_SPLITS[split]
    for i, (ra, rb) in enumerate(zip(a_rows, b_rows)):
        rep["rows"] += 1
        if ra["text"] != rb["text"]:
            rep["dropped/a_b_misaligned"] += 1
            continue
        considered = sorted(set(rb["labels"]) | set(ra["labels"]))
        if not considered:
            rep["dropped/no_article_considered"] += 1
            continue
        rep["questions/violated_not_in_considered_added"] += len(set(ra["labels"]) - set(rb["labels"]))
        facts = "\n".join(str(p).strip() for p in ra["text"] if str(p).strip())
        if ECHR_LEAK.search(facts):
            rep["dropped/facts_state_a_finding"] += 1
            continue
        state = "\n".join(["European Court of Human Rights: the facts of an application, as set out in the judgment",
                           clip(facts, ECHR_FACTS_MAX, mark="[... remaining fact paragraphs omitted]")])
        qs, outcome = {}, {}
        for k in considered:
            art = ECHR_ARTICLES[k]
            q = f"violation_art_{art.replace('-', '_').lower()}"
            qs[q] = {"type": "noul",
                     "instructions": (f"Will the Court find a violation of {echr_art_name(art)} ({ECHR_TITLES[art]}) "
                                      "of the European Convention on Human Rights in this case?"),
                     "criteria": noul_crit(f"The judgment finds a violation of {echr_art_name(art)}.",
                                           f"The Court examines {echr_art_name(art)} but finds no violation of it.")}
            outcome[q] = "yes" if k in ra["labels"] else "no"
        cid = gb.sha16(facts)
        meta = {"label_provenance": "outcome (ECtHR judgment: violated articles, LexGLUE ecthr_a; considered: ecthr_b)",
                "decision_date": lo, "info_date": lo, "date_precision": f"split period only ({desc})",
                "resolution_date": {q: hi for q in qs}, "lexglue_split": split, "lexglue_row": i,
                "time_slice": f"{desc}; PRE-cutoff: knowledge probe required", "td_workflow": None,
                "source_revision": LEXGLUE[1]}
        out.append(eval_rec("ecthr", "ecthr/lexglue", state, qs, outcome, meta, base_id=f"ecthr/{cid}",
                            rec_id=f"gde1-ecthr-{split}-{i}"))
    return out, dict(rep)


def load_ecthr(a):
    root = Path(a.work) / "ecthr"
    recs, rep = [], {}
    for split in ("validation", "test"):
        rows = {c: read_parquet(download(hf_url(LEXGLUE, f"{c}/{split}-00000-of-00001.parquet"),
                                         root / f"{c}_{split}.parquet")) for c in ("ecthr_a", "ecthr_b")}
        r, rp = ecthr_convert(rows["ecthr_a"], rows["ecthr_b"], split)
        recs += r
        rep[split] = rp
    return {"train": [], "eval": recs}, rep, {"source": {"repo": LEXGLUE[0], "revision": LEXGLUE[1],
                                                         "configs": ["ecthr_a", "ecthr_b"], "splits": list(ECHR_SPLITS)},
                                              "pre_cutoff_warning": "judgments 2016-2019: knowledge probe required",
                                              "facts_caveat": "the facts are the Court's own 'THE FACTS' section, "
                                                              "drafted with the judgment (not a pre-decision filing)"}


# =========================================================================== eval writer (GDE-1)
SCANS = {"aidev_prs": re.compile(r"\b(merged|closing this|superseded|abandoned)\b", re.I),
         "swiss_judgment": re.compile(r"Erwägung|Considérant en droit|Considerando in diritto|Dispositiv|"
                                      r"Demnach erkennt", re.I),
         "swe_rebench_oh": re.compile(r"\b(PASSED|FAILED|resolved|test(s)? pass(ed)?|exit_status)\b"),
         "swe_agent": re.compile(r"\b(PASSED|FAILED|resolved|test(s)? pass(ed)?|exit_status)\b"),
         "ecthr": re.compile(r"violation of article|just satisfaction|the court (holds|finds)", re.I)}
for _k, _v in SCANS.items():
    gb.SCAN.setdefault(_k, _v)


def finish_eval(a, dataset, recs, rep, extra):
    gset = GDE1_SETS[dataset]
    manifest = {"set": gset, "domain": "O", "builder": BUILDER, "built": datetime.now().isoformat(timespec="seconds"),
                "license": LICENSES[dataset], "license_evidence": EVIDENCE[dataset], "eval_only": True,
                "build_report": rep, **extra, "splits": {}}
    if a.eval_fingerprint:
        idx, sizes = gb.build_index(a.eval_fingerprint)
        manifest["dedup_vs_training"] = {"fingerprints": sizes, "paths": a.eval_fingerprint, "when": "before the cap"}
    else:
        idx = None
        manifest["dedup_vs_training"] = {"status": "NOT RUN here: run `build_gde1_b.py dedup --kind built --sets "
                                                   f"{gset} --fingerprint ...` against jev_data, jev_ext_strict, "
                                                   "jev_ext2, jev_ext3 and outcomes2_jev (fleet/task_62d.sh)"}
    recs, ndup = gb.dedup_within(recs)
    info = {"within_set_exact_state_duplicates_dropped": ndup}
    recs, info["dropped_by_per_repository_cap"] = cap_groups(recs, GROUP_CAP.get(dataset), a.seed, f"{gset}/eval-group",
                                                            lambda r: r["meta"].get("repo") or r["base_id"])
    if idx is not None:
        recs, info["vs_training"] = gb.dedup_against(recs, idx)
    recs, info["cap"] = gb.cap_units(recs, a.eval_cap, a.seed, f"{gset}/eval")
    info["stats"] = gb.split_stats(recs)
    manifest["splits"]["eval"] = info
    gb.write_set(a.eval_out, gset, {"eval": recs}, manifest)
    return recs


# =========================================================================== training writer (ext2 gate)
CELLS = {"aidev": lambda r: r["meta"]["agent"],
         "afd": lambda r: f"{r['source']}|{r['meta']['close_year']}",
         "swiss": lambda r: f"{r['meta']['law_area']}|{r['meta']['language']}",
         "swe_rebench": lambda r: r["meta"]["repo"]}
CAPS = {"aidev/pr_merge": TRAIN_CAP, "afd/nomination": TRAIN_CAP // 2, "afd/nomination_votes": TRAIN_CAP // 2,
        "swiss/fscs": TRAIN_CAP, "swe/rebench_openhands": TRAIN_CAP}
MAX_CHARS = {"swe/rebench_openhands": SWE_STATE_MAX}


def new_stats(source, dataset, lab):
    return {"dataset": dataset, "license": LICENSES[dataset], "cap": None, "candidates": 0, "written": Counter(),
            "units": Counter(), "label_kinds": Counter(), "qtypes": Counter(), "drops": Counter(),
            "dedup_exact": Counter(), "dedup_leaf": Counter(), "dedup_shingle": Counter(), "question_drops": lab.drops,
            "overlap_examples": [], "errors": [], "hf_sha": None}


def corpus_stats(out):
    """Base rates and counts of a written corpus (realised outcomes from meta.outcome)."""
    st = {}
    for sp in ("train", "val"):
        for p in sorted((Path(out) / sp).glob("*.jsonl")):
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                s = st.setdefault(r["source"], {"records": Counter(), "outcome": defaultdict(Counter),
                                                "mean_label": defaultdict(lambda: defaultdict(float)), "dates": []})
                s["records"][sp] += 1
                s["dates"].append(r["meta"].get("decision_date"))
                for q, y in (r["meta"].get("outcome") or {}).items():
                    s["outcome"][q][y] += 1
                    for k, v in r["labels"][q].items():
                        s["mean_label"][q][k] += v
    rep = {}
    for src, s in st.items():
        n = sum(s["records"].values())
        ds = sorted(d for d in s["dates"] if d)
        rep[src] = {"records": dict(s["records"]),
                    "outcome_rate": {q: {k: round(v / sum(c.values()), 4) for k, v in sorted(c.items())}
                                     for q, c in s["outcome"].items()},
                    "mean_label": {q: {k: round(v / n, 4) for k, v in sorted(m.items())} for q, m in s["mean_label"].items()},
                    "decision_date_range": [ds[0], ds[-1]] if ds else None}
    return rep


def write_train(a, dataset, cands_by_source, idx, coverage, extra):
    out = Path(a.train_out) / dataset
    stale = [p for sp in ("train", "val", "test") for p in (out / sp).glob("*.jsonl")]
    if stale or (out / "manifest.json").exists():
        if not a.overwrite:
            sys.exit(f"{out} already holds a corpus; pass --overwrite (removes its jsonl files) or a new --train-out")
        for p in stale + [out / "manifest.json"]:
            if p.exists():
                p.unlink()
    writer, seen = Writer(out), set()
    manifest = {"builder": BUILDER, "dataset": dataset, "seed": a.seed, "blend": a.blend,
                "gate": {"mode": getattr(a, "gate", "calibrated"), "rule": DfIndex.__doc__ if getattr(a, "gate", "calibrated")
                         == "calibrated" else "build_jev_ext2.Index (exact; leaf >= 5 words; >= 3 shared 8-word shingles)"},
                "splits": {"train": 100 - x2.VAL_PCT, "val": x2.VAL_PCT,
                           "by": "sha1 of meta.group (repository / article / decision); the eval split is separate "
                                 "(time + repository/ID held out, written to the GDE-1 set)"},
                "dedup": {"as": "build_jev_ext2.py Index + process", "fingerprint_states": dict(idx.sizes),
                          "fingerprint_paths": dict(idx.paths), "eval_coverage": coverage,
                          "corpus_fingerprints_present": sorted(t for t in idx.sizes if t.startswith("corpus:")),
                          "status": ("AUTHORITATIVE (training-corpus fingerprints present)"
                                     if any(t.startswith("corpus:") for t in idx.sizes) else
                                     "PROVISIONAL: eval-gated only; the cross-corpus dedup against jev_data / "
                                     "jev_ext_strict / jev_ext2 / jev_ext3 runs on a node (fleet/task_62d.sh)")},
                "license": LICENSES[dataset], "license_evidence": EVIDENCE.get(dataset), **extra, "sources": {}}
    for source, cands in cands_by_source.items():
        cap = min(CAPS[source], a.train_cap)
        rng, lab = random.Random(f"{a.seed}:{source}"), Labeler(0.01)
        st = new_stats(source, dataset, lab)
        blend(cands, CELLS[dataset], a.blend)
        cands, ndrop = cap_groups(cands, GROUP_CAP.get(dataset), a.seed, f"{source}/train-group",
                                  lambda r: r["meta"]["group"])
        st["drops"]["group_cap"] = ndrop
        x2.process(source, cands, cap, idx, writer, rng, st, seen, MAX_CHARS.get(source, a.max_state_chars))
        x2.report_source(source, st)
        manifest["sources"][source] = plain(st)
    writer.close()
    out.mkdir(parents=True, exist_ok=True)
    manifest["readback"] = final_check(out)
    manifest["stats"] = corpus_stats(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    log(f"{dataset}: training corpus written to {out}")
    return manifest


# =========================================================================== orchestration
LOADERS = {"aidev": load_aidev, "afd": lambda a: (*load_afd(a), {}), "swiss": load_swiss,
           "swe_rebench": load_swe_rebench, "swe_agent": load_swe_agent, "ecthr": load_ecthr}


def run(a, datasets):
    staged, summary = {}, {}
    for ds in datasets:
        log(f"== {ds}: loading")
        got = LOADERS[ds](a)
        if ds == "afd":
            cands, rep, extra = got
            staged[ds] = (cands, rep, extra)
            summary[ds] = {"train_candidates": {k: len(v) for k, v in cands.items()}}
            continue
        out, rep, extra = got
        if out["eval"] and not a.no_eval:
            recs = finish_eval(a, ds, out["eval"], rep, extra)
            summary[ds] = {"eval_records": len(recs), "eval_units": gb.n_units(recs)}
        if ds in TRAINABLE and out["train"]:
            src = out["train"][0]["source"]
            staged[ds] = ({src: out["train"]}, rep, extra)
            summary.setdefault(ds, {})["train_candidates"] = len(out["train"])
    if a.no_train or not staged:
        print(json.dumps(summary, indent=1, default=str))
        return summary
    # the gate: every eval family + the eval-out directory (all GDE-1 sets, this builder's included)
    a.eval = list(a.eval) + [f"gde1_all={a.eval_out}"]
    idx, coverage = load_gate(a)
    for ds, (cands, rep, extra) in staged.items():
        m = write_train(a, ds, cands, idx, coverage, {"build_report": rep, **extra})
        summary.setdefault(ds, {})["train"] = m["stats"]
    print(json.dumps(summary, indent=1, default=str))
    return summary


def cmd_check(a):
    cf = gb.load_cf()
    rep = {}
    for ds, gset in GDE1_SETS.items():
        if (Path(a.eval_out) / gset / "eval").is_dir():
            rep[gset] = gb.check_set(a.eval_out, gset, cf)
    for ds in TRAINABLE:
        out = Path(a.train_out) / ds
        if (out / "train").is_dir():
            scan = Counter()
            pat = {"aidev": SCANS["aidev_prs"], "afd": AFD_LEAK, "swiss": SCANS["swiss_judgment"],
                   "swe_rebench": SCANS["swe_rebench_oh"]}[ds]
            for p in sorted(out.rglob("*.jsonl")):
                for line in p.open(encoding="utf-8"):
                    for h in {m.group(0).lower() for m in pat.finditer(json.loads(line)["state"])}:
                        scan[h] += 1
            rep[f"train:{ds}"] = {"readback": final_check(out)["total"], "stats": corpus_stats(out),
                                  "outcome_word_scan": dict(scan.most_common(15)) or "0 hits"}
    print(json.dumps(rep, indent=1, default=str))
    (Path(a.work) / "check_report.json").write_text(json.dumps(rep, indent=1, default=str))


# =========================================================================== self-test (offline)
def self_test():
    from synth_workflows.exclusion import EXCLUDED_QIDS
    assert not set(QIDS) & set(EXCLUDED_QIDS) and not any(q.startswith("violation_art") for q in EXCLUDED_QIDS)
    for ds in TRAINABLE:
        assert not re.search(r"\bNC\b|non[- ]?commercial|research[- ]only", LICENSES[ds], re.I), ds
    for ds in ("swe_agent", "ecthr"):
        assert re.search(r"\bNC\b|non[- ]?commercial|research[- ]only", LICENSES[ds], re.I) or "EVAL ONLY" in LICENSES[ds]
    # ---- AIDev
    free = [i for i in range(1, 400) if uh(SEED, "aidev-heldout-repo", i) >= AIDEV_HELDOUT]
    held = next(i for i in range(1, 400) if uh(SEED, "aidev-heldout-repo", i) < AIDEV_HELDOUT)
    r1, r2 = free[:2]
    repos = [{"id": r1, "full_name": "o/a", "license": "MIT", "language": "Python"},
             {"id": r2, "full_name": "o/b", "license": "GPL-3.0", "language": "Go"},
             {"id": held, "full_name": "o/held", "license": None, "language": "Rust"}]

    def pr(i, repo, agent="OpenAI_Codex", state="closed", merged=True, created="2025-06-01T10:00:00Z",
           closed="2025-06-02T10:00:00Z", title="Fix parser bug", body="## Summary\n- fix\n<!-- template -->\n"
           "https://chatgpt.com/codex/tasks/task_e_123"):
        return {"id": i, "number": i, "title": title, "body": body, "agent": agent, "state": state, "repo_id": repo,
                "created_at": created, "closed_at": closed if state == "closed" else None,
                "merged_at": closed if merged else None, "html_url": f"https://github.com/x/pull/{i}"}
    prs = [pr(1, r1), pr(2, r1, merged=False, title="Add retry to the HTTP client", body="Adds exponential backoff."),
           pr(3, r1, state="open", merged=False), pr(4, r1, agent="Copilot"),
           pr(5, r1, title="[WIP] Add feature"), pr(6, r1, body="Closing this in favour of #9"), pr(7, r2),
           pr(8, held, created="2025-08-05T00:00:00Z", closed="2025-08-09T00:00:00Z", merged=False,
              title="Port the scheduler to asyncio", body="Rewrites the job loop on asyncio tasks."),
           pr(9, held, created="2025-06-05T00:00:00Z"), pr(10, r1, created="2025-07-05T00:00:00Z")]
    out, rep = aidev_convert(prs, repos)
    assert [r["meta"]["aidev_pr_id"] for r in out["train"]] == [1, 2], rep
    assert [r["meta"]["aidev_pr_id"] for r in out["eval"]] == [8], rep
    assert rep["dropped/agent_Copilot"] == 1 and rep["dropped/wip_title"] == 1 and rep["dropped/open_unresolved"] == 1
    assert rep["dropped/body_announces_close_or_merge"] == 1 and rep["train_repo/licence_not_permissive(GPL-3.0)"] == 1
    s = out["train"][0]["state"]
    assert "chatgpt.com" not in s and "template" not in s and "o/a (Python; licence MIT)" in s
    assert s.startswith("Opened 2025-06-01 by OpenAI Codex\nRepository: o/a")
    assert out["train"][1]["meta"]["outcome"] == {"pr_merged": "no"} and out["eval"][0]["labels"]["pr_merged"]["no"] == 1
    ev = out["eval"][0]
    assert ev["meta"]["resolution_date"] == {"pr_merged": "2025-08-09"} and ev["base_id"] == f"aidev/repo{held}"
    print("ok  aidev: closed-only, Copilot / WIP / self-closing dropped, held-out repos -> eval, licence filter")
    # ---- AfD
    ts = lambda d: datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp()  # noqa: E731
    convs = {"100000001": {"article_title": "Foo Band (2nd nomination)", "outcome_label": "delete",
                           "outcome_timestamp": ts("2010-05-08")},
             "100000002": {"article_title": "Bar", "outcome_label": "keep no-consensus", "outcome_timestamp": ts("2010-05-08")},
             "100000003": {"article_title": "Baz", "outcome_label": "keep", "outcome_timestamp": ts("2019-02-01")},
             "100000004": {"article_title": "Qux", "outcome_label": "merge redirect", "outcome_timestamp": ts("2012-01-09")}}
    U = lambda t, d, txt, label=None: {"timestamp": ts(d), "text": txt, "meta": {"type": t, "label": label}}  # noqa: E731
    utts = {"100000001": [U("nomination", "2010-05-01", "Non-notable band, no reliable sources [[WP:BAND]]. --Nom 12:00, 1 May 2010 (UTC)"),
                          U("vote", "2010-05-02", "*VOTE per nom, fails WP:BAND", "delete"),
                          U("vote", "2010-05-03", "*VOTE has two albums", "keep"),
                          U("vote", "2010-05-09", "*VOTE late vote after close", "keep"),
                          {"timestamp": None, "text": "untimed", "meta": {"type": "vote", "label": "keep"}}],
            "100000004": [U("nomination", "2012-01-01", "Duplicates the main article on the topic, should be merged.")]}
    cands, rep = afd_records(convs, utts, lambda c: "nomination_votes" if c == "100000001" else "nomination")
    nv, nm = cands["afd/nomination_votes"][0], cands["afd/nomination"][0]
    assert rep["dropped/label_keep no-consensus"] == 1 and rep["dropped/outside_2005_2018"] == 1
    assert "late vote" not in nv["state"] and "untimed" not in nv["state"] and "fails WP:BAND" in nv["state"]
    assert "2 shown" in nv["state"] and "VOTE" not in nv["state"] and "(UTC)" not in nv["state"]
    assert nv["meta"]["outcome"] == {"afd_close": "delete", "afd_deleted": "yes"} and nv["meta"]["group"] == "foo band"
    assert nm["meta"]["outcome"]["afd_close"] == "merge" and "Votes" not in nm["state"]
    assert set(nv["questions"]["afd_close"]["criteria"]) == set(AFD_OPTIONS)
    print("ok  afd: label map, 2005-2018, votes strictly before the close, no untimed votes, variants")
    # ---- Swiss
    base = {"decision_id": "d1", "year": 2010, "language": "de", "law_area": "civil_law", "law_sub_area": None,
            "chamber": "CH_BGer_004", "court": "CH_BGer", "canton": "CH", "region": "Federation"}
    long_facts = "Sachverhalt:\nA. Das Obergericht hat die Klage abgewiesen. " + "Text. " * 80
    rows = [dict(base, facts=long_facts, label="approval"),
            dict(base, decision_id="d2", facts=long_facts + "\nErwägungen:\n1. Die Beschwerde ist begründet.",
                 label="approval"),
            dict(base, decision_id="d3", facts=long_facts + " Mit Entscheid vom heutigen Tag hat das Bundesgericht "
                 "die Beschwerde abgewiesen.", label="dismissal"),
            dict(base, decision_id="d4", facts="kurz", label="dismissal")]
    rep, assoc = Counter(), Counter()
    tr = swiss_convert(rows, "train", rep, assoc)
    assert [r["meta"]["decision_id"] for r in tr] == ["d1"], rep
    assert rep["train/dropped/considerations_or_operative_heading_in_facts"] == 1
    assert rep["train/dropped/same_day_federal_ruling"] == 1 and tr[0]["state"].startswith("Division 004 (civil law)")
    te = swiss_convert([dict(base, decision_id="d9", year=2021, facts=long_facts, label="dismissal")], "test", rep, assoc)
    assert te[0]["meta"]["domain"] == "O" and te[0]["meta"]["resolution_date"] == {"appeal_approved": "2021-12-31"}
    print("ok  swiss: facts only, own-ruling leaks dropped, lower-court dispositive words kept")
    # ---- SWE
    rh = next(f"own/r{i}" for i in range(500) if swe_heldout(f"own/r{i}"))
    rt = next(f"own/r{i}" for i in range(500) if not swe_heldout(f"own/r{i}"))
    issues = {"i1": {"problem_statement": "Crash when x is None", "created_at": "2021-01-01", "license": "MIT License"},
              "i2": {"problem_statement": "Bug in y", "created_at": "2022-01-01", "license": "MIT License"},
              "i3": {"problem_statement": "GPL bug", "created_at": "2022-01-01", "license": "GNU General Public License v3.0"}}
    P = lambda iid, repo, t, res, patch="diff --git a/x b/x\n+fix": {"instance_id": iid, "repo": repo, "patch": patch,  # noqa: E731
                                                                     "resolved": res, "model": "m", "traj_id": t,
                                                                     "exit_status": "submit"}
    rows = [P("i1", rt, "t1", 1), P("i1", rt, "t2", 0, "diff 2"), P("i1", rt, "t3", 0, "diff 3"),
            P("i1", rt, "t4", 1, "diff 4"), P("i1", rt, "t5", 1, ""), P("i2", rh, "t6", 0), P("i2", rh, "t7", 1),
            P("i3", rt, "t8", 1)]
    out, rep = swe_convert(rows, issues, "swe_rebench")
    assert len(out["train"]) == 3 and rep["dropped/per_issue_cap"] == 1 and rep["dropped/empty_patch"] == 1, rep
    assert len(out["eval"]) == 1 and rep["dropped/same_patch_same_issue"] == 1, rep   # t6, t7 share one patch
    assert rep["train/dropped/repo_licence_not_permissive"] == 1
    assert "Agent patch:" in out["train"][0]["state"] and "submit" not in out["train"][0]["state"]
    out2, rep2 = swe_convert(rows, issues, "swe_agent")
    assert not out2["train"] and len(out2["eval"]) == 1 and out2["eval"][0]["license"] == LICENSES["swe_agent"]
    print("ok  swe: issue + patch only, per-issue caps, shared repository bucket, licence filter, swe_agent eval-only")
    # ---- ECtHR
    fa = [{"text": ["5. The applicant was arrested.", "6. He complained."], "labels": [3]},
          {"text": ["7. The Court finds that there has been a violation."], "labels": [3]},
          {"text": ["8. Nothing."], "labels": []}]
    fb = [{"text": fa[0]["text"], "labels": [1, 3]}, {"text": fa[1]["text"], "labels": [3]}, {"text": fa[2]["text"], "labels": []}]
    er, rp = ecthr_convert(fa, fb, "test")
    assert len(er) == 1 and rp["dropped/facts_state_a_finding"] == 1 and rp["dropped/no_article_considered"] == 1
    assert er[0]["meta"]["outcome"] == {"violation_art_3": "no", "violation_art_6": "yes"}, er[0]["meta"]["outcome"]
    print("ok  ecthr: one noul per considered article, finding-in-facts dropped")
    # ---- blend
    c = [train_rec("aidev/pr_merge", "aidev", f"s{i}", {"pr_merged": dict(AIDEV_Q)}, {"pr_merged": "yes" if i % 4 else "no"},
                   {"agent": "A" if i < 20 else "B"}, group=i, probes=[], pin="x") for i in range(40)]
    blend(c, CELLS["aidev"], 0.5)
    assert all(abs(sum(r["labels"]["pr_merged"].values()) - 1) < 1e-9 for r in c)
    y, n = c[1]["labels"]["pr_merged"], c[0]["labels"]["pr_merged"]
    assert 0.5 < y["yes"] < 1 and 0 < n["yes"] < 0.5 and c[0]["label_kind"]["pr_merged"] == "soft"
    print("ok  blend: lam * onehot + (1 - lam) * LOO cell rate, own outcome excluded")
    # ---- boilerplate: a train PR and an eval PR that share only a template + footer are NOT matched once the
    #      template lines are stripped (and are, by ext2's 3-shingle rule, if they are left in)
    tmpl = ("## Type of change\n- [x] Bug fix (non-breaking change which fixes an issue)\n- [ ] New feature "
            "(non-breaking change which adds functionality)\n- [ ] Breaking change (fix or feature that would cause "
            "existing functionality to change)\n")
    foot = "\nGenerated with [Claude Code](https://claude.ai/code)\nCo-Authored-By: Claude <noreply@anthropic.com>"
    tp = pr(20, r1, agent="Claude_Code", title="Handle empty config files", body=tmpl + "Empty files crash the loader." + foot)
    ep = pr(21, held, agent="Claude_Code", created="2025-08-05T00:00:00Z", closed="2025-08-06T00:00:00Z",
            title="Speed up CSV export", body=tmpl + "Streams rows instead of building a list." + foot)
    bo = boiler_lines([(aidev_clean(tmpl), g) for g in range(12) for _ in range(3)], 25, 10)
    assert len(bo) == 3, bo
    for boiler, want_drop in ((frozenset(), True), (bo, False)):
        ix = x2.Index()
        ix.add(ix.tag("eval:x", "eval"), aidev_state(ep, repos[2], boiler))
        ix.freeze()
        hit = ix.check(aidev_state(tp, repos[0], boiler))
        assert bool(hit) == want_drop, (boiler, hit)
    assert "Co-Authored-By" not in aidev_state(tp, repos[0]) and "claude.ai" not in aidev_state(tp, repos[0])
    print("ok  boilerplate: template-only overlap is a shingle hit unless stripped; agent footers removed")
    # ---- DfIndex: stock shingles (in > 20 eval states) do not count; rare shared shingles do; a near-copy of an item
    #      in a cluster of > 20 identical items is caught by the containment rule
    stock = "the court of appeal dismissed the appeal and confirmed the judgment of the district court in full"
    di = DfIndex()
    t = di.tag("eval:x", "eval")
    for i in range(30):
        di.add(t, f"case {i} alpha{i} beta{i} gamma{i}. {stock}. delta{i} epsilon{i} zeta{i} eta{i} theta{i} iota{i}")
    twin_text = " ".join(f"w{j}" for j in range(40))
    for i in range(25):
        di.add(t, f"{twin_text} copy {i}")
    di.add(t, "a rare sentence about the zebra crossing on main street near the old mill and the river bank today")
    di.freeze()
    assert di.check(f"new case omega psi chi. {stock}. sigma tau upsilon phi") is None, "stock phrasing must not drop"
    assert di.check("we saw a rare sentence about the zebra crossing on main street near the old mill and more"), "rare"
    assert di.check(twin_text + " copy 99"), "a near-copy inside a large identical cluster is still caught"
    long_a = " ".join(f"a{j}" for j in range(200)) + " the unusual phrase one two three four five six seven eight nine"
    di2 = DfIndex()
    di2.add(di2.tag("eval:y", "eval"), long_a)
    di2.freeze()
    long_b = " ".join(f"b{j}" for j in range(200)) + " the unusual phrase one two three four five six seven eight nine"
    assert di2.check(long_b) is None, "3 shared shingles in 200-word texts: under the 10% coverage"
    assert di2.check(long_a[:600]), "a long shared passage is caught"
    plain_ix = x2.Index()
    plain_ix.add_rows("eval:x", "eval", [f"case {i} alpha{i} beta{i} gamma{i}. {stock}." for i in range(30)])
    plain_ix.freeze()
    assert plain_ix.check(f"new case omega psi chi. {stock}. sigma tau upsilon phi"), "the plain ext2 rule drops it"
    print("ok  DfIndex: stock shingles ignored, rare overlaps and near-copies caught (the plain rule drops stock text)")
    # ---- pipeline: eval set -> gate -> training corpus -> read-back
    with tempfile.TemporaryDirectory() as td:
        class A:
            work, train_out, eval_out, seed, eval_cap, train_cap, blend, overwrite = td, f"{td}/tr", f"{td}/ev", SEED, 8000, 20000, 0.5, False
            eval_fingerprint, max_state_chars = [], 6500
        out, rep = aidev_convert(prs, repos)
        planted = dict(out["eval"][0])
        finish_eval(A, "aidev", out["eval"], rep, {})
        cf = gb.load_cf()
        r = gb.check_set(A.eval_out, "aidev_prs", cf, quiet=True)
        assert r["schema_problems"] == "none" and r["total_units"] == 1, r
        idx = DfIndex()
        for p in sorted(Path(A.eval_out).rglob("*.jsonl")):
            idx.add_rows("given:gde1", "eval", x2.jsonl_states(p), p)
        idx.freeze()
        twin = train_rec("aidev/pr_merge", "aidev", planted["state"], {"pr_merged": dict(AIDEV_Q)}, {"pr_merged": "no"},
                         {"agent": "OpenAI_Codex", "decision_date": "2025-06-01"}, group="g", probes=[], pin="x")
        m = write_train(A, "aidev", {"aidev/pr_merge": out["train"] + [twin]}, idx, {}, {})
        st = m["sources"]["aidev/pr_merge"]
        assert st["dedup_exact"] == {"given:gde1": 1}, st
        assert m["readback"]["total"]["records"]["train"] + m["readback"]["total"]["records"]["val"] == 2
        units = list(cf.iter_units(str(Path(A.train_out) / "aidev"), ["train", "val"], set(), None))
        assert len(units) == 2 and not any(cf.is_noncommercial(u) for u in units)
        ev_units = list(cf.iter_units(str(Path(A.eval_out) / "aidev_prs"), ["eval"], set(), None))
        assert ev_units and ev_units[0]["noul_criteria"]
    print("ok  pipeline: GDE-1 write + check_set, ext2 gate drops a planted eval twin, iter_units read-back")
    print("\nself-test passed")
    return 0


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--work", default=WORK, help="scratch / download cache (keep <= 2 GB on the laptop)")
        p.add_argument("--train-out", default=TRAIN_OUT)
        p.add_argument("--eval-out", default=EVAL_OUT)
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--eval-cap", type=int, default=EVAL_CAP, help="max units per GDE-1 set")
        p.add_argument("--train-cap", type=int, default=TRAIN_CAP, help="max records per training source")
        p.add_argument("--blend", type=float, default=0.5, help="training label = lam*onehot + (1-lam)*LOO rate (1 = one-hot)")
        p.add_argument("--oversample", type=float, default=1.6)
        p.add_argument("--max-state-chars", type=int, default=6500)
        p.add_argument("--overwrite", action="store_true", help="replace an existing corpus's jsonl files")
        p.add_argument("--no-train", action="store_true")
        p.add_argument("--no-eval", action="store_true")
        p.add_argument("--afd-cap", type=int, default=TRAIN_CAP // 2)
        p.add_argument("--swiss-train-rows", type=int, default=32000)
        p.add_argument("--swiss-test-rows", type=int, default=12000)
        # the ext2 gate (same flags as build_jev_ext2 / ext3)
        p.add_argument("--eval", action="append", default=[], help="TAG=PATH eval jsonl file or directory")
        p.add_argument("--fingerprint", action="append", default=[], help="TAG=PATH training corpus")
        p.add_argument("--no-fetch-evals", dest="fetch_evals", action="store_false")
        p.add_argument("--skip-evals", default="")
        p.add_argument("--allow-missing-evals", action="store_true")
        p.add_argument("--allow-missing-corpora", action="store_true")
        p.add_argument("--gate", choices=["calibrated", "plain"], default="calibrated",
                       help="training gate: ext2's Index with part A's stock-phrase calibration (default) or plain ext2")
        p.add_argument("--eval-fingerprint", action="append", default=[], metavar="TAG=PATH",
                       help="training corpora to dedup the GDE-1 sets against before the cap (build_gde1_b rule)")
        p.add_argument("--cache", default=None, help="ext2 non-Hub cache (default <work>/ext2_urls)")
        p.add_argument("--hf-token", default=None)
    for name in DATASETS + ("all",):
        common(sub.add_parser(name))
    common(sub.add_parser("check"))
    for name in SKIPPED:
        sub.add_parser(name)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd in SKIPPED:
        print(f"{a.cmd}: {SKIPPED[a.cmd]}")
        return 0
    if not a.cmd:
        ap.print_help()
        return 2
    a.cache = a.cache or str(Path(a.work) / "ext2_urls")
    a.skip_evals = {s.strip() for s in a.skip_evals.split(",") if s.strip()}
    if a.cmd == "check":
        return cmd_check(a)
    run(a, list(DATASETS) if a.cmd == "all" else [a.cmd])
    return 0


if __name__ == "__main__":
    sys.exit(main())
