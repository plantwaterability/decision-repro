#!/usr/bin/env python3
"""
build_jev_openreview.py -- pipeline P3 (open peer review), OpenReview part: ICLR reviewer scores as multi-rater
("V") score questions in the jev_data record schema (build_jev_data.py `record`, build_jev_ext2.py `make_record`).

Design: research/realworld_data.md, "### P3" and the OpenReview catalogue row. Write-up, terms, statistics:
research/openreview_pipeline.md.

Sources (all through the official OpenReview API v2, https://api2.openreview.net, with the openreview-py client):
  openreview/iclr2024   ICLR.cc/2024/Conference   ratings = FINAL (post-rebuttal); the public API keeps no rating history
  openreview/iclr2025   ICLR.cc/2025/Conference   ratings = FINAL (post-rebuttal); same
  openreview/iclr2026   ICLR.cc/2026/Conference   ratings = INITIAL: ICLR reverted every review's text and scores to
                                                 the start of the discussion period after the 2025-11-27 security
                                                 incident (blog.iclr.cc 2025-12-03, 2026-03-31)
  TMLR is NOT used: its `Review` form has no numeric rating (free-text fields only); the one ordinal field is the
  reviewer's `Official_Recommendation` (Accept / Leaning Accept / Leaning Reject / Reject), entered "based on your
  initial assessment and the discussion with the authors" -- post-discussion and a decision vote, so it fails the
  "numeric ratings" condition.
Licence: reviews are OpenReview "Comments", released CC BY 4.0 under the OpenReview Terms of Use (2024-09-24);
titles, abstracts, keywords and primary areas are "Metadata", CC0 1.0 dedication; ICLR 2025 / 2026 submissions
also carry license "CC BY 4.0". Every record carries LICENSE and meta.forum_url (CC BY attribution).

One record per submission with >= MIN_REVIEWS official reviews that carry a rating.
  state      JSON: venue (name, year, fixed description) + the submission's title, primary_area, keywords and
             abstract (clipped to ABSTRACT_CHARS). Never: authors, review text, meta-review, decision, venue
             status, TL;DR, PDF. The public note holds the LATEST abstract (the API exposes no submission
             history to non-authors), so sentences carrying a URL are cut (post-acceptance code / project links
             are the main camera-ready addition), and records whose submission text matches an outcome
             pattern (STRONG_OUTCOME: acceptance / venue-tier / camera-ready / withdrawal notices, thanks to
             reviewers), contains an author's full name, or whose abstract is a placeholder (< MIN_ABSTRACT_CHARS,
             typically blanked after withdrawal) are dropped.
  questions  reviewer_rating     score 4: reject / marginally below / marginally above / accept (mapping per year,
                                 RATING_MAP; ICLR 2024-25 scale {1,3,5,6,8,10}, ICLR 2026 {0,2,4,6,8,10})
             soundness_rating, presentation_rating, contribution_rating   score 4: ICLR's 1 poor .. 4 excellent
             No accept / reject question: ICLR <= 2025 decisions are surely in pretraining.
  labels     the empirical distribution over the paper's reviewers (vote shares). meta.n_raters, meta.raw (every
             reviewer's raw value and edited-after-release flag, in review order), meta.rating_stage,
             meta.label_provenance = "human raters (ICLR official reviewers)", meta.label_source[qid] = "votes:<n>"
             and meta.label_method[qid] (as build_jev_courts.py), meta.labels_smoothed (the Dirichlet-multinomial
             posterior mean of realworld_data.md 3.1, a fitted per venue-year and question; lists in level order).
  ids        "<source>-<sha1(forum id)[:16]>"; base_id = id.
Splits (by forum, near-duplicates grouped across years: same normalized title, or >= SHINGLE_MIN_SHARED shared
8-word shingles of the URL-stripped abstracts covering >= 10% of the shorter one):
  train / val   ICLR 2024 + 2025; val = VAL_PCT% of groups by hash. A 2024 / 2025 paper whose group also holds a
                2026 submission (a resubmission) is dropped from train / val so the eval is never trained on.
  eval/test     ICLR 2026 only (held out). meta.prior_iclr_version marks resubmissions of 2024 / 2025 papers.
Output: <out>/{train,val}/openreview__iclr{2024,2025}.jsonl, <out>/eval/test/openreview__iclr2026.jsonl,
        <out>/manifest.json (counts, scale maps, label statistics, agreement ceiling, leak scans),
        <out>/_cache/<venue>.jsonl.gz (compact API extract: no author names, no review text; holds the venue status
        only for the leak diagnostics -- never copied into a record).

Credentials: ~/.openreview (line 1 email, line 2 password) is read ONLY inside the fetch process and sent only to
OpenReview's /login endpoint. Nothing about it is printed, logged or written; the session token stays in memory.
If login asks for MFA or a challenge, the fetch stops.

Usage:
  uv run --no-project --with openreview-py python build_jev_openreview.py fetch --out /tmp/openreview_jev
  mkdir -p /tmp/jevstub && echo 'AutoTokenizer = None' > /tmp/jevstub/transformers.py \\
      && echo 'class Tensor: pass' > /tmp/jevstub/torch.py
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_openreview.py build --out /tmp/openreview_jev
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_openreview.py self-test
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_jev_data import Writer, option_keys, record, score_q, validate  # noqa: E402

API = "https://api2.openreview.net"
CRED_FILE = "~/.openreview"
UA = "jev-openreview-p3 (openreview-py; research dataset build; polite, sequential)"
MIN_REVIEWS = 3
ABSTRACT_CHARS = 1200
MIN_ABSTRACT_CHARS = 250     # shorter = a placeholder ("n/a", ".", the title, "We have withdrawn our paper.") left by
                             # authors after withdrawal: an outcome leak and an empty state (shortest real one: 194)
VAL_PCT = 10
SHINGLE_WORDS = 8
SHINGLE_MIN_SHARED = 3
SHINGLE_MIN_CONTAIN = 0.10   # ... and at least this share of the shorter abstract's shingles (URL sentences cut first:
                             # "our code is available at https anonymous 4open science ..." chained unrelated papers)
SHINGLE_MAX_DF = 20          # shingles in more than this many abstracts are boilerplate, ignored for dedup
MAX_KEYWORDS = 10
LICENSE = "OpenReview, CC BY 4.0 (review scores: CC BY 4.0 per OpenReview Terms; title/abstract: metadata, CC0 1.0)"

VENUES = {
    "iclr2024": {"venue_id": "ICLR.cc/2024/Conference", "name": "ICLR", "year": 2024, "role": "train",
                 "stage": "final"},
    "iclr2025": {"venue_id": "ICLR.cc/2025/Conference", "name": "ICLR", "year": 2025, "role": "train",
                 "stage": "final"},
    "iclr2026": {"venue_id": "ICLR.cc/2026/Conference", "name": "ICLR", "year": 2026, "role": "eval",
                 "stage": "initial"},
}
STAGE_TEXT = {
    "final": "Scores are final: reviewers may revise them after the author response.",
    "initial": "Scores are initial: given before the author response.",
}
STAGE_META = {"final": "final (post-rebuttal)", "initial": "initial (pre-rebuttal)"}
STAGE_NOTE = {  # manifest only
    "final": "the public API exposes no rating history (edits visible to non-authors carry no content), so the "
             "initial scores are unavailable; meta.reviews_modified_after_release counts reviews edited after release",
    "initial": "ICLR 2026 reverted every review's text and scores to the start of the discussion period "
               "(blog.iclr.cc 2025-12-03); meta.reviews_modified_after_release counts those reverts (2025-11-28)",
}

# ---------------------------------------------------------------- questions
RATING_LEVELS = [
    "reject: the reviewer recommends rejection ('strong reject' or 'reject, not good enough')",
    "marginally below the acceptance threshold: leaning reject",
    "marginally above the acceptance threshold: leaning accept",
    "accept: the reviewer recommends acceptance ('accept, good paper' or 'strong accept')",
]
# raw venue value -> level index. Each year's option texts were read off the review invitations (see manifest).
RATING_MAP = {
    "iclr2024": {1: 0, 3: 0, 5: 1, 6: 2, 8: 3, 10: 3},
    "iclr2025": {1: 0, 3: 0, 5: 1, 6: 2, 8: 3, 10: 3},
    "iclr2026": {0: 0, 2: 0, 4: 1, 6: 2, 8: 3, 10: 3},
}
_OPT_2024_25 = ["1: strong reject", "3: reject, not good enough", "5: marginally below the acceptance threshold",
                 "6: marginally above the acceptance threshold", "8: accept, good paper",
                 "10: strong accept, should be highlighted at the conference"]
VENUE_RATING_OPTIONS = {  # verbatim from the expired per-paper Official_Review invitations (get_invitations(expired=True))
    "iclr2024": _OPT_2024_25,
    "iclr2025": _OPT_2024_25,
    "iclr2026": ["0: strong reject.", "2: reject, not good enough",
                 "4: marginally below the acceptance threshold. But would not mind if paper is accepted",
                 "6: marginally above the acceptance threshold. But would not mind if paper is rejected",
                 "8: accept, good paper (poster)",
                 "10: strong accept, should be highlighted at the conference as spotlight or oral."],
}
VENUE_DIM_OPTIONS = "1 poor, 2 fair, 3 good, 4 excellent (soundness, presentation, contribution; all three years)"
DIM_LEVELS = {
    "soundness_rating": [
        "1 poor: major flaws; the central claims are unsupported",
        "2 fair: some claims weakly supported or notable gaps",
        "3 good: claims mostly supported; minor issues",
        "4 excellent: rigorous; claims fully supported",
    ],
    "presentation_rating": [
        "1 poor: hard to follow; key details missing",
        "2 fair: understandable with effort; several unclear parts",
        "3 good: clear and well organized; minor issues",
        "4 excellent: exceptionally clear and well placed in prior work",
    ],
    "contribution_rating": [
        "1 poor: little new or useful; incremental",
        "2 fair: modest; limited significance or originality",
        "3 good: solid and original; others in the area will use it",
        "4 excellent: significant, highly original, broad impact",
    ],
}
DIM_FIELD = {"soundness_rating": "soundness", "presentation_rating": "presentation",
             "contribution_rating": "contribution"}
DIM_WHAT = {
    "soundness_rating": "soundness: are the methods sound and the central claims supported by evidence?",
    "presentation_rating": "presentation: clarity, writing, and placement relative to prior work?",
    "contribution_rating": "contribution: importance, originality and value of the results to the field?",
}
QIDS = ["reviewer_rating", *DIM_LEVELS]


def _rating_bins_text(key):
    groups = defaultdict(list)
    for raw, b in RATING_MAP[key].items():
        groups[b].append(str(raw))
    return " | ".join(",".join(groups[b]) for b in sorted(groups))


LABEL_METHOD = {   # meta.label_method[qid]: the exact construction (raw vote shares; smoothed copy in meta)
    key: {"reviewer_rating": f"shares; rating bins {_rating_bins_text(key)}",
          **{q: f"shares; {DIM_FIELD[q]} 1-4" for q in DIM_LEVELS}}
    for key in RATING_MAP}


def instructions(qid, v):
    head = (f"A paper submitted to {v['name']} {v['year']}, assessed by several anonymous expert reviewers. "
            f"Pick one official reviewer at random.")
    if qid == "reviewer_rating":
        body = "Which overall recommendation does that reviewer give?"
    else:
        body = f"How does that reviewer rate the paper's {DIM_WHAT[qid]}"
    return f"{head} {body} {STAGE_TEXT[v['stage']]}"


def levels_of(qid):
    return RATING_LEVELS if qid == "reviewer_rating" else DIM_LEVELS[qid]


def bin_of(qid, key, raw):
    if raw is None:
        return None
    if qid == "reviewer_rating":
        return RATING_MAP[key].get(raw)
    return raw - 1 if 1 <= raw <= 4 else None


# ---------------------------------------------------------------- small helpers
def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_int(v):
    """ICLR 2024 stores '6: marginally above ...' / '3 good'; 2025-26 store ints."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.match(r"\s*(\d+)", str(v))
    return int(m.group(1)) if m else None


def val(content, k):
    x = (content or {}).get(k)
    return x.get("value") if isinstance(x, dict) else x


# ---------------------------------------------------------------- credentials + fetch
def login():
    """Log in with ~/.openreview. The file is read here and nowhere else; nothing from it is printed or stored."""
    import openreview
    import requests

    with open(os.path.expanduser(CRED_FILE), encoding="utf-8") as f:
        lines = f.read().splitlines()
    user = lines[0].strip() if lines else ""
    pw = lines[1] if len(lines) > 1 else ""
    del lines
    if not user or not pw:
        raise SystemExit("credential file is incomplete (contents not shown)")
    secrets = (user, pw)

    def scrub(x):
        s = str(x)
        for t in secrets or ():
            if t:
                s = s.replace(t, "<redacted>")
        return s[:300]

    try:
        r = requests.post(f"{API}/login", json={"id": user, "password": pw}, timeout=60,
                          headers={"User-Agent": UA, "Accept": "application/json"})
        body = r.json() if "application/json" in r.headers.get("Content-Type", "") else {}
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"login request failed: {type(e).__name__}: {scrub(e)}") from None
    finally:
        user = pw = None
    if r.status_code != 200:
        name = body.get("name") if isinstance(body, dict) else None
        msg = body.get("message") if isinstance(body, dict) else r.text[:200]
        raise SystemExit(f"login refused: HTTP {r.status_code} {name}: {scrub(msg)}")
    if body.get("mfaPending"):
        raise SystemExit("login requires MFA; stopping (not attempting it)")
    token = body.get("token")
    if not token:
        raise SystemExit("login returned no token")
    secrets = None
    del body
    client = openreview.api.OpenReviewClient(baseurl=API, token=token)
    del token
    return client


def name_in_text(name, text_lc):
    name = re.sub(r"\s+", " ", str(name or "")).strip().lower()
    if len(name.split()) < 2 or len(name) < 6:
        return False
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text_lc) is not None


def status_of(venueid, venue_id):
    venueid = str(venueid or "")
    if venueid == venue_id:
        return "accepted"
    tail = venueid.rsplit("/", 1)[-1].lower()
    for k in ("withdrawn", "desk_rejected", "rejected"):
        if k in tail:
            return k
    return "other"


def compact(note, key):
    """One submission note (with directReplies) -> the cache row. Authors are used only for the name check."""
    v = VENUES[key]
    ct = note.content or {}
    title, abstract = val(ct, "title") or "", val(ct, "abstract") or ""
    kws = val(ct, "keywords") or []
    if isinstance(kws, str):
        kws = [kws]
    area = val(ct, "primary_area")
    text_lc = " ".join([title, abstract, " ".join(map(str, kws))]).lower()
    authors = val(ct, "authors") or []
    author_hit = any(name_in_text(a, text_lc) for a in authors if isinstance(a, str))
    del authors
    reviews = []
    for r in (note.details or {}).get("directReplies") or []:
        if r.get("ddate") or not any(i.endswith("/-/Official_Review") for i in r.get("invitations", [])):
            continue
        rc = r.get("content") or {}
        reviews.append({"cdate": r.get("cdate"), "mdate": r.get("mdate") or r.get("tmdate"),
                        **{f: parse_int(val(rc, f)) for f in ("rating", "soundness", "presentation", "contribution",
                                                             "confidence")}})
    reviews.sort(key=lambda x: (x["cdate"] or 0))
    return {"forum": note.id, "number": note.number, "venue": key, "cdate": note.cdate,
            "title": title, "abstract": abstract, "keywords": [str(k) for k in kws], "primary_area": area,
            "license": getattr(note, "license", None), "author_name_in_text": author_hit,
            "status": status_of(val(ct, "venueid"), v["venue_id"]), "reviews": reviews}


def fetch(a):
    cache = Path(a.out, "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    client = login()
    log("logged in (token held in memory only)")
    for key in a.venues:
        path = cache / f"{key}.jsonl.gz"
        if path.exists() and not a.refetch:
            log(f"{key}: cached at {path}, skipping (use --refetch)")
            continue
        inv = f"{VENUES[key]['venue_id']}/-/Submission"
        rows, after, page_no, t0 = [], None, 0, time.time()
        while True:
            params = {"invitation": inv, "limit": a.page, "details": "directReplies", "sort": "id"}
            if after:
                params["after"] = after
            for attempt in range(6):
                try:
                    notes = client.get_notes(**params)
                    break
                except Exception as e:  # noqa: BLE001
                    wait = min(300, 15 * 2 ** attempt)
                    log(f"{key}: page {page_no} failed ({type(e).__name__}: {str(e)[:160]}); retry in {wait}s")
                    time.sleep(wait)
            else:
                raise SystemExit(f"{key}: page {page_no} failed 6 times; stopping")
            if not notes:
                break
            rows.extend(compact(n, key) for n in notes)
            after, page_no = notes[-1].id, page_no + 1
            log(f"{key}: page {page_no}: {len(rows)} submissions so far ({time.time() - t0:.0f}s)")
            if len(notes) < a.page:
                break
            time.sleep(a.sleep)
        tmp = path.with_suffix(".part")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(path)
        log(f"{key}: wrote {len(rows)} submissions -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
        time.sleep(a.sleep)


def read_cache(out, key):
    with gzip.open(Path(out, "_cache", f"{key}.jsonl.gz"), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- state + leak controls
URL = re.compile(r"(https?://|www\.|github\.com|gitlab\.com|huggingface\.co/|anonymous\.4open|\.github\.io|"
                 r"bit\.ly/|tinyurl|zenodo\.org|colab\.research)", re.IGNORECASE)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\\(\[])")
STRONG_OUTCOME = re.compile(
    r"camera[- ]?ready|desk[- ]?reject"
    r"|\b(accepted|to appear|published|presented)\b[^.]{0,60}\b(ICLR|NeurIPS|ICML|conference|proceedings|TMLR)\b"
    r"|\bICLR\b[^.]{0,30}\b(oral|spotlight|poster|notable)\b|\b(oral|spotlight)\b[^.]{0,30}\bICLR\b"
    r"|\b(this|our) (paper|submission|work|manuscript) (was|has been|is|got) (accepted|rejected|withdrawn)\b"
    r"|\bwithdr[ae]w(n|ing)?\b[^.]{0,40}\b(paper|submission|manuscript|work)\b"
    r"|\b(paper|submission|manuscript)\b[^.]{0,30}\bwithdrawn\b|\bdecided? to withdraw\b"
    r"|\bthank (the|all|our)?\s*(anonymous )?(reviewers|area chairs?|ACs?)\b",
    re.IGNORECASE)
CODE_PHRASE = re.compile(r"\b(code|implementation|project page|models?|datasets?|benchmark)\b[^.]{0,50}\b"
                         r"(available|released|open[- ]sourced?|publicly)\b", re.IGNORECASE)
OUTCOME_WORDS = ["accepted", "acceptance", "accept", "oral", "spotlight", "poster", "withdrawn", "withdraw",
                 "rejected", "rejection", "reject", "camera-ready", "camera ready", "notable", "top 5%", "top 25%",
                 "published", "to appear", "rebuttal", "meta-review", "reviewer"]
OUTCOME_RE = {w: re.compile(r"(?<!\w)" + re.escape(w) + r"(?!\w)", re.IGNORECASE) for w in OUTCOME_WORDS}


def clean_ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def strip_url_sentences(text):
    sents = SENT.split(clean_ws(text))
    kept = [s for s in sents if not URL.search(s)]
    return " ".join(kept).strip(), len(sents) - len(kept)


def clip(text, n=ABSTRACT_CHARS):
    if len(text) <= n:
        return text
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end >= int(n * 0.6):
        return cut[:end + 1]
    return cut[:cut.rfind(" ")].rstrip(" ,;:") + " ..."


def split_keywords(kws):
    out = []
    for k in kws or []:
        parts = re.split(r"\s*;\s*", str(k)) if ";" in str(k) else [str(k)]
        out.extend(clean_ws(p).strip(" ,.") for p in parts)
    seen, res = set(), []
    for k in out:
        if k and k.lower() not in seen and len(k) <= 80:
            seen.add(k.lower())
            res.append(k)
    return res[:MAX_KEYWORDS]


def venue_block(v):
    return {"name": f"{v['name']} {v['year']}", "kind": "machine-learning conference, main track",
            "review_model": "double-blind; several anonymous expert reviewers per submission"}


def make_state(row):
    v = VENUES[row["venue"]]
    abstract, n_cut = strip_url_sentences(row["abstract"])
    sub = {"title": clean_ws(row["title"]), "primary_area": clean_ws(row.get("primary_area")) or None,
           "keywords": split_keywords(row.get("keywords")), "abstract": clip(abstract)}
    sub = {k: x for k, x in sub.items() if x}
    state = json.dumps({"submission": sub, "venue": venue_block(v)}, ensure_ascii=False, sort_keys=True)
    return state, sub, n_cut


def submission_text(sub):
    return " ".join([sub.get("title", ""), sub.get("primary_area") or "", " ".join(sub.get("keywords", [])),
                     sub.get("abstract", "")])


# ---------------------------------------------------------------- dedup across years
def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t or "").lower()).strip()


def shingles(text, n=SHINGLE_WORDS):
    w = re.findall(r"\w+", str(text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def dup_groups(rows):
    """Union-find over forums: identical normalized title, or >= SHINGLE_MIN_SHARED shared non-boilerplate
    8-word shingles of the URL-stripped abstracts that also cover >= SHINGLE_MIN_CONTAIN of the shorter one.
    Returns forum -> group key (the smallest forum id in the group)."""
    parent = {r["forum"]: r["forum"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    by_title = defaultdict(list)
    for r in rows:
        t = norm_title(r["title"])
        if len(t) >= 12:
            by_title[t].append(r["forum"])
    for fs in by_title.values():
        for f in fs[1:]:
            union(fs[0], f)
    sh = {r["forum"]: shingles(strip_url_sentences(r["abstract"])[0]) for r in rows}
    index = defaultdict(list)
    for f, ss in sh.items():
        for s in ss:
            index[s].append(f)
    shared = Counter()
    for fs in index.values():
        if 1 < len(fs) <= SHINGLE_MAX_DF:
            fs = sorted(set(fs))
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    shared[(fs[i], fs[j])] += 1
    for (x, y), c in shared.items():
        if c >= max(SHINGLE_MIN_SHARED, SHINGLE_MIN_CONTAIN * min(len(sh[x]), len(sh[y]))):
            union(x, y)
    return {f: find(f) for f in parent}


# ---------------------------------------------------------------- labels + statistics
def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def dm_loglik(counts_list, alpha):
    a = sum(alpha)
    tot = 0.0
    for c in counts_list:
        n = sum(c)
        tot += math.lgamma(a) - math.lgamma(n + a)
        tot += sum(math.lgamma(ck + ak) - math.lgamma(ak) for ck, ak in zip(c, alpha))
    return tot


def fit_dm(counts_list):
    """Empirical-Bayes Dirichlet-multinomial: m = pooled base rate, a = argmax marginal likelihood (log grid +
    golden refinement). Returns (a, m)."""
    K = len(counts_list[0])
    tot = [sum(c[k] for c in counts_list) for k in range(K)]
    m = [(t + 0.5) / (sum(tot) + 0.5 * K) for t in tot]
    grid = [10 ** (x / 20) for x in range(-30, 81)]          # 0.03 .. 1e4
    best = max(grid, key=lambda a: dm_loglik(counts_list, [a * mk for mk in m]))
    lo, hi = best / 10 ** 0.05, best * 10 ** 0.05
    g = (math.sqrt(5) - 1) / 2
    f = lambda la: dm_loglik(counts_list, [math.exp(la) * mk for mk in m])  # noqa: E731
    x0, x1 = math.log(lo), math.log(hi)
    for _ in range(40):
        c, d = x1 - g * (x1 - x0), x0 + g * (x1 - x0)
        if f(c) > f(d):
            x1 = d
        else:
            x0 = c
    return math.exp((x0 + x1) / 2), m


def label_stats(dists):
    if not dists:
        return {}
    return {"n": len(dists), "mean_max_p": round(statistics.fmean(max(d) for d in dists), 4),
            "mean_entropy_nats": round(statistics.fmean(entropy(d) for d in dists), 4),
            "share_one_hot": round(sum(1 for d in dists if max(d) > 1 - 1e-9) / len(dists), 4),
            "share_exact_zeros": round(sum(sum(1 for x in d if x == 0) for d in dists) / (len(dists) * len(dists[0])), 4)}


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else float("nan")


def agreement_ceiling(papers, a, m, reps=50, seed=0):
    """papers: list of (raw ratings, bins). Returns inter-rater statistics -- the ceiling any predictor of a
    reviewer's score faces, since reviewers of one paper disagree among themselves."""
    rng = random.Random(seed)
    K = len(m)
    r1, rhalf = [], []
    for _ in range(reps):
        xs, ys = [], []
        hx, hy = [], []
        for raw, _ in papers:
            i, j = rng.sample(range(len(raw)), 2)
            xs.append(raw[i])
            ys.append(raw[j])
            if len(raw) >= 4:
                idx = list(range(len(raw)))
                rng.shuffle(idx)
                h = len(raw) // 2
                hx.append(statistics.fmean(raw[k] for k in idx[:h]))
                hy.append(statistics.fmean(raw[k] for k in idx[h:]))
        r1.append(pearson(xs, ys))
        rhalf.append(pearson(hx, hy))
    rho1, rh = statistics.fmean(r1), statistics.fmean(rhalf)
    n_bar = statistics.fmean(len(raw) for raw, _ in papers)
    # pairwise exact agreement on the 4 bins vs chance; leave-one-out prediction from the other reviewers
    po_num = po_den = 0
    nll_peer = nll_base = brier_peer = brier_base = acc_peer = acc_base = 0.0
    n_loo = 0
    for _, bins in papers:
        c = Counter(bins)
        n = len(bins)
        po_num += sum(v * (v - 1) for v in c.values())
        po_den += n * (n - 1)
        for b in bins:
            rest = [c[k] - (1 if k == b else 0) for k in range(K)]
            q = [(rest[k] + a * m[k]) / (n - 1 + a) for k in range(K)]
            nll_peer -= math.log(q[b])
            nll_base -= math.log(m[b])
            brier_peer += sum((q[k] - (k == b)) ** 2 for k in range(K))
            brier_base += sum((m[k] - (k == b)) ** 2 for k in range(K))
            acc_peer += 1.0 if max(range(K), key=lambda k: (q[k], -abs(k - 1.5))) == b else 0.0
            acc_base += 1.0 if max(range(K), key=lambda k: m[k]) == b else 0.0
            n_loo += 1
    po = po_num / po_den
    pe = sum(x * x for x in m)
    return {
        "papers": len(papers), "mean_reviewers": round(n_bar, 2),
        "single_rater_r": round(rho1, 4),
        "split_half_r": round(rh, 4), "split_half_papers": len([1 for raw, _ in papers if len(raw) >= 4]),
        "split_half_spearman_brown": round(2 * rh / (1 + rh), 4),
        "panel_mean_reliability_at_mean_n": round(n_bar * rho1 / (1 + (n_bar - 1) * rho1), 4),
        "bin_pairwise_agreement": round(po, 4), "bin_chance_agreement": round(pe, 4),
        "bin_kappa": round((po - pe) / (1 - pe), 4),
        "loo_from_peers": {"nll": round(nll_peer / n_loo, 4), "brier": round(brier_peer / n_loo, 4),
                           "top1": round(acc_peer / n_loo, 4)},
        "loo_base_rate": {"nll": round(nll_base / n_loo, 4), "brier": round(brier_base / n_loo, 4),
                          "top1": round(acc_base / n_loo, 4)},
    }


# ---------------------------------------------------------------- build
def scan_outcome_words(texts):
    hits = Counter()
    for t in texts:
        for w, rx in OUTCOME_RE.items():
            if rx.search(t):
                hits[w] += 1
    return dict(sorted(hits.items(), key=lambda kv: -kv[1]))


def build_records(rows, groups, eval_groups, drops, diag):
    """rows of one venue -> (records, per-question count vectors). Pure: no I/O."""
    out, counts = [], defaultdict(list)
    for row in rows:
        key = row["venue"]
        v = VENUES[key]
        revs = [r for r in row["reviews"] if r.get("rating") is not None]
        if len(revs) < MIN_REVIEWS:
            drops["fewer_than_3_rated_reviews"] += 1
            continue
        if row.get("author_name_in_text"):
            drops["author_name_in_text"] += 1
            continue
        state, sub, n_cut = make_state(row)
        if not sub.get("title") or len(sub.get("abstract", "")) < MIN_ABSTRACT_CHARS:
            drops["placeholder_or_short_abstract"] += 1
            continue
        if STRONG_OUTCOME.search(submission_text(sub)):
            drops["strong_outcome_pattern"] += 1
            diag["strong_outcome_examples"].append(STRONG_OUTCOME.search(submission_text(sub)).group(0))
            continue
        g = groups[row["forum"]]
        if v["role"] == "train" and g in eval_groups:
            drops["resubmitted_to_iclr2026(eval)"] += 1
            continue
        qs, labs, raw, cvec = {}, {}, {}, {}
        for qid in QIDS:
            field = "rating" if qid == "reviewer_rating" else DIM_FIELD[qid]
            vals = [r.get(field) for r in revs]
            bins = [bin_of(qid, key, x) for x in vals]
            if any(x is not None and b is None for x, b in zip(vals, bins)):
                drops[f"unmapped_{field}_value"] += 1
            good = [b for b in bins if b is not None]
            if len(good) != len(revs):     # keep labels * n_raters integral (jev_bayes.recover_counts)
                drops[f"question_dropped_missing_{field}"] += 1
                continue
            lv = levels_of(qid)
            c = [sum(1 for b in good if b == k) for k in range(len(lv))]
            qs[qid] = score_q(instructions(qid, v), lv)
            labs[qid] = {lv[k]: c[k] / len(good) for k in range(len(lv))}
            raw[field] = vals
            cvec[qid] = c
        if "reviewer_rating" not in qs:
            drops["no_rating_question"] += 1
            continue
        source = f"openreview/{key}"
        kinds = {q: ("hard" if sum(1 for x in labs[q].values() if x > 0) == 1 else "soft") for q in qs}
        rec = record(source, state, qs, labs, kinds, None)
        rec["id"] = rec["base_id"] = f"{source}-{sha(row['forum'])}"
        rec["license"] = LICENSE
        release = diag["release_ms"][key]
        modified = [bool((r.get("mdate") or 0) > max(r.get("cdate") or 0, release) + 3600_000) for r in revs]
        rec["meta"] = {
            "dataset": f"OpenReview API v2, {v['venue_id']}",
            "forum_url": f"https://openreview.net/forum?id={row['forum']}",
            "group": sha(g), "venue": f"{v['name']} {v['year']}", "n_raters": len(revs),
            "rating_stage": STAGE_META[v["stage"]],
            "raw": {**raw, "confidence": [r.get("confidence") for r in revs],
                    "modified_after_release": [int(x) for x in modified]},
            "reviews_modified_after_release": sum(modified),
            "label_provenance": "human raters (ICLR official reviewers)",
            "label_source": {q: f"votes:{len(revs)}" for q in qs},
            "label_method": {q: LABEL_METHOD[key][q] for q in qs},
            "abstract_url_sentences_removed": n_cut,
            "submission_license": row.get("license"),
        }
        if v["role"] == "eval":
            rec["meta"]["prior_iclr_version"] = g in diag["train_groups"]
        rec["_counts"], rec["_group"] = cvec, g
        validate(rec)
        out.append(rec)
        for qid, c in cvec.items():
            counts[qid].append(c)
    return out, counts


def release_ms(rows):
    """End of the review-release bulk edit: the busy hours (>= 2% of the peak hour, within 24 h of it) of review
    mdates. ICLR 2024 / 2025 released in one hour; ICLR 2026's release ran 02:00-10:00 UTC on 2025-11-12."""
    hours = Counter(int((r.get("mdate") or 0) // 3600_000) for row in rows for r in row["reviews"] if r.get("mdate"))
    if not hours:
        return 0
    h0, c0 = hours.most_common(1)[0]
    busy = [h for h, c in hours.items() if c >= 0.02 * c0 and abs(h - h0) <= 24]
    return (max(busy) + 1) * 3600_000


class CompactWriter(Writer):
    """build_jev_data.Writer with compact JSON separators (same files, same layout, ~5% smaller)."""

    def write(self, split, source, rec):
        key = (split, source)
        if key not in self.files:
            path = self.out / split / f"{source.replace('/', '__')}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files[key] = path.open("w", encoding="utf-8")
        self.files[key].write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def split_of_group(g):
    return "val" if int(hashlib.sha1(("openreview-val:" + g).encode()).hexdigest(), 16) % 100 < VAL_PCT else "train"


def build(a):
    out = Path(a.out)
    rows_by = {k: read_cache(out, k) for k in VENUES if Path(out, "_cache", f"{k}.jsonl.gz").exists()}
    missing = [k for k in VENUES if k not in rows_by]
    if missing:
        raise SystemExit(f"no cache for {missing}: run `fetch` first")
    all_rows = [r for rows in rows_by.values() for r in rows]
    log(f"loaded {len(all_rows)} submissions: " + ", ".join(f"{k} {len(v)}" for k, v in rows_by.items()))
    groups = dup_groups(all_rows)
    eval_groups = {groups[r["forum"]] for r in rows_by["iclr2026"]}
    train_groups = {groups[r["forum"]] for k in ("iclr2024", "iclr2025") for r in rows_by[k]}
    n_groups = len(set(groups.values()))
    log(f"dedup: {len(all_rows)} forums -> {n_groups} groups")

    diag = {"strong_outcome_examples": [], "train_groups": train_groups,
            "release_ms": {k: release_ms(rows) for k, rows in rows_by.items()}}
    drops = {k: Counter() for k in rows_by}
    recs, counts = {}, {}
    for k, rows in rows_by.items():
        recs[k], counts[k] = build_records(rows, groups, eval_groups, drops[k], diag)
        log(f"{k}: {len(recs[k])} records; drops {dict(drops[k])}")

    # Dirichlet-multinomial smoothing per venue-year x question (stored in meta, labels stay raw shares)
    dm = {}
    for k in recs:
        dm[k] = {}
        for qid, cl in counts[k].items():
            alpha, m = fit_dm(cl)
            dm[k][qid] = {"a": round(alpha, 4), "m": [round(x, 4) for x in m]}
        for r in recs[k]:
            sm = {}
            for qid, c in r["_counts"].items():
                alpha, m = dm[k][qid]["a"], dm[k][qid]["m"]
                n = sum(c)
                sm[qid] = [round((c[i] + alpha * m[i]) / (n + alpha), 5) for i in range(len(c))]
            r["meta"]["labels_smoothed"] = sm   # lists in the question's level order; a, m in the manifest

    # write
    for sub in ("train", "val", "eval"):
        p = out / sub
        if p.exists():
            for f in p.rglob("*.jsonl"):
                f.unlink()
    w_train, w_eval = CompactWriter(out), CompactWriter(out / "eval")
    split_counts = Counter()
    for k, rs in recs.items():
        for r in rs:
            g = r.pop("_group")
            r.pop("_counts")
            if VENUES[k]["role"] == "eval":
                w_eval.write("test", r["source"], r)
                split_counts[(k, "test")] += 1
            else:
                s = split_of_group(g)
                w_train.write(s, r["source"], r)
                split_counts[(k, s)] += 1
    w_train.close()
    w_eval.close()

    # statistics
    stats = {}
    for k, rs in recs.items():
        ids = {r["id"] for r in rs}
        st = {"records": len(rs), "splits": {s: n for (kk, s), n in split_counts.items() if kk == k},
              "submissions_fetched": len(rows_by[k]),
              "submission_status_counts(cache only, never in records)": dict(Counter(row["status"] for row in rows_by[k])),
              "status_of_records(cache only, never in records)": dict(Counter(
                  row["status"] for row in rows_by[k] if f"openreview/{k}-{sha(row['forum'])}" in ids)),
              "reviews_per_record": dict(sorted(Counter(r["meta"]["n_raters"] for r in rs).items())),
              "share_records_with_review_modified_after_release":
                  round(sum(1 for r in rs if r["meta"]["reviews_modified_after_release"]) / max(1, len(rs)), 4),
              "share_reviews_modified_after_release":
                  round(sum(r["meta"]["reviews_modified_after_release"] for r in rs)
                        / max(1, sum(r["meta"]["n_raters"] for r in rs)), 4),
              "raw_rating_values": dict(sorted(Counter(x for r in rs for x in r["meta"]["raw"]["rating"]).items())),
              "questions": {}, "dirichlet": dm[k], "drops": dict(drops[k])}
        for qid in QIDS:
            lv = levels_of(qid)
            raw_d = [[r["labels"][qid][x] for x in lv] for r in rs if qid in r["labels"]]
            sm_d = [r["meta"]["labels_smoothed"][qid] for r in rs if qid in r["labels"]]
            st["questions"][qid] = {"raw_shares": label_stats(raw_d), "smoothed": label_stats(sm_d),
                                    "bin_marginal": [round(statistics.fmean(d[i] for d in raw_d), 4)
                                                     for i in range(len(lv))] if raw_d else []}
        st["agreement_ceiling"] = {}
        for qid in QIDS:
            field = "rating" if qid == "reviewer_rating" else DIM_FIELD[qid]
            papers = []
            for r in rs:
                raw = [x for x in r["meta"]["raw"].get(field, []) if x is not None]
                bins = [bin_of(qid, k, x) for x in raw]
                if len(raw) >= 2 and None not in bins:
                    papers.append((raw, bins))
            if papers and qid in dm[k]:
                st["agreement_ceiling"][qid] = agreement_ceiling(papers, dm[k][qid]["a"], dm[k][qid]["m"])
        # the same for reviewer_rating on reviews never edited after release (initial == final for those);
        # papers with >= 3 such reviews. In 2026 an "edit" is the revert, so this is close to the full set there.
        papers = []
        for r in rs:
            raw = [x for x, mo in zip(r["meta"]["raw"]["rating"], r["meta"]["raw"]["modified_after_release"])
                   if x is not None and not mo]
            if len(raw) >= MIN_REVIEWS:
                papers.append((raw, [bin_of("reviewer_rating", k, x) for x in raw]))
        if papers:
            st["agreement_ceiling"]["reviewer_rating(unedited reviews only)"] = agreement_ceiling(
                papers, dm[k]["reviewer_rating"]["a"], dm[k]["reviewer_rating"]["m"])
        allp = [r for r in rs]
        st["pooled_all_questions"] = {
            "raw_shares": label_stats([[r["labels"][q][x] for x in levels_of(q)] for r in allp for q in r["labels"]]),
            "smoothed": label_stats([r["meta"]["labels_smoothed"][q] for r in allp for q in r["labels"]])}
        stats[k] = st

    # leak scans over the WRITTEN states (submission block only; the venue block is fixed text)
    leak = {}
    for k, rs in recs.items():
        subs = [json.loads(r["state"])["submission"] for r in rs]
        texts = [submission_text(s) for s in subs]
        venue_blocks = {json.dumps(json.loads(r["state"])["venue"], sort_keys=True) for r in rs}
        url_left = sum(1 for t in texts if URL.search(t))
        st_rows = {row["forum"]: row for row in rows_by[k]}
        by_status_cut, by_status_code, by_status_n = Counter(), Counter(), Counter()
        for row in rows_by[k]:
            _, sub, n_cut = make_state(row)
            by_status_cut[(row["status"], n_cut > 0)] += 1
            by_status_n[row["status"]] += 1
            if CODE_PHRASE.search(sub.get("abstract", "")):
                by_status_code[row["status"]] += 1
        leak[k] = {"outcome_word_hits_in_states": scan_outcome_words(texts),
                   "strong_outcome_hits_in_states": sum(1 for t in texts if STRONG_OUTCOME.search(t)),
                   "url_left_in_states": url_left, "distinct_venue_blocks": len(venue_blocks),
                   "url_sentence_cut_by_status": {f"{s}|cut={c}": n for (s, c), n in sorted(by_status_cut.items())},
                   "code_release_phrase_rate_by_status(after URL cut)": {
                       s: round(by_status_code[s] / n, 4) for s, n in sorted(by_status_n.items())},
                   "examples": {w: [] for w in ("accepted", "rejected", "oral", "spotlight", "withdrawn")}}
        for t in texts:
            for w in leak[k]["examples"]:
                mm = OUTCOME_RE[w].search(t)
                if mm and len(leak[k]["examples"][w]) < 3:
                    leak[k]["examples"][w].append(t[max(0, mm.start() - 70): mm.end() + 50])
        del st_rows
    leak["strong_outcome_examples_dropped"] = diag["strong_outcome_examples"][:20]

    manifest = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "out": str(out), "license": LICENSE,
        "venues": {k: {**v, "rating_stage": STAGE_META[v["stage"]], "rating_stage_note": STAGE_NOTE[v["stage"]],
                       "release_ms": diag["release_ms"][k]} for k, v in VENUES.items()},
        "labels_smoothed": "meta.labels_smoothed[qid] = Dirichlet-multinomial posterior mean (c_k + a m_k) / (n + a), "
                           "a list in the question's level order; a, m per venue-year and question in stats.*.dirichlet",
        "venue_rating_options": VENUE_RATING_OPTIONS, "venue_dim_options": VENUE_DIM_OPTIONS,
        "rating_levels": RATING_LEVELS, "rating_map": {k: {str(x): b for x, b in m.items()} for k, m in RATING_MAP.items()},
        "dim_levels": DIM_LEVELS, "min_reviews": MIN_REVIEWS, "val_pct": VAL_PCT,
        "dedup": {"forums": len(all_rows), "groups": n_groups,
                  "iclr2026_records_with_prior_version": sum(1 for r in recs["iclr2026"] if r["meta"]["prior_iclr_version"])},
        "stats": stats, "leak": leak,
        "typed_decisions_gold_reference": {"mean_max_p": 0.659, "mean_entropy_nats": 0.767, "score_entropy_nats": 0.930},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
    log(f"manifest -> {out / 'manifest.json'}")
    final_check(out)


# ---------------------------------------------------------------- final check through cache_features
def load_cache_features():
    spec = importlib.util.spec_from_file_location("cache_features", HERE / "trainers" / "cache_features.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)          # needs torch / transformers importable (real or stub modules)
    return cf


def final_check(out):
    cf = load_cache_features()
    total = Counter()
    for root, splits in ((Path(out), ["train", "val"]), (Path(out) / "eval", ["test"])):
        n_rec = 0
        for split in splits:
            for p in sorted(Path(root, split).glob("*.jsonl")):
                with p.open(encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        validate(rec)
                        for qid, q in rec["questions"].items():
                            keys, texts, target = cf.question_options(q, rec["labels"][qid])
                            assert keys == option_keys(q), (rec["id"], qid)
                            assert all(abs(t - rec["labels"][qid][k]) < 1e-12 for k, t in zip(keys, target))
                        n_rec += 1
        n_units = 0
        for u in cf.iter_units(str(root), splits, set(), None):
            assert abs(sum(u["target"]) - 1.0) < 1e-6 and len(u["target"]) == 4, u["record_id"]
            assert u["n_raters"] and u["n_raters"] >= MIN_REVIEWS, u["record_id"]
            assert all(abs(t * u["n_raters"] - round(t * u["n_raters"])) < 1e-6 for t in u["target"]), u["record_id"]
            total[(str(root), u["split"], u["qid"])] += 1
            n_units += 1
        log(f"final check {root} {splits}: {n_rec} records valid, {n_units} units through cf.iter_units")
    for (root, split, qid), n in sorted(total.items()):
        print(f"   {Path(root).name:16s} {split:5s} {qid:22s} {n:7d}")


# ---------------------------------------------------------------- self-test (no network)
def self_test():
    assert parse_int("6: marginally above the acceptance threshold") == 6 and parse_int("3 good") == 3
    assert parse_int(4) == 4 and parse_int(None) is None and parse_int("n/a") is None
    assert bin_of("reviewer_rating", "iclr2025", 5) == 1 and bin_of("reviewer_rating", "iclr2026", 4) == 1
    assert bin_of("reviewer_rating", "iclr2026", 0) == 0 and bin_of("reviewer_rating", "iclr2024", 10) == 3
    assert bin_of("reviewer_rating", "iclr2026", 5) is None and bin_of("soundness_rating", "iclr2024", 4) == 3
    t, n = strip_url_sentences("We propose X. Code is at https://github.com/a/b. It works well.")
    assert t == "We propose X. It works well." and n == 1, t
    assert STRONG_OUTCOME.search("This paper was accepted at ICLR 2024 as a spotlight.")
    assert STRONG_OUTCOME.search("Camera-ready version.") and not STRONG_OUTCOME.search("We use rejection sampling.")
    assert STRONG_OUTCOME.search("We have decided to withdraw the submission.") and STRONG_OUTCOME.search(
        "We thank the reviewers for their comments.") and STRONG_OUTCOME.search("This paper has been withdrawn")
    assert not STRONG_OUTCOME.search("Auditing data controller compliance with data withdrawal requests.")
    assert name_in_text("Jane Q. Doe", "work by jane q. doe et al") and not name_in_text("Bayes", "bayes rule")
    assert split_keywords(["a; b", "A", "c"]) == ["a", "b", "c"]
    assert len(clip("word " * 400)) <= ABSTRACT_CHARS + 4
    rows = [{"forum": "f1", "title": "A study of graphs and things", "abstract": " ".join(f"w{i}" for i in range(40))},
            {"forum": "f2", "title": "A STUDY of graphs, and things!", "abstract": "x y z"},
            {"forum": "f3", "title": "Unrelated paper about cats", "abstract": " ".join(f"w{i}" for i in range(5, 30))},
            {"forum": "f4", "title": "Totally different", "abstract": "q r s t"}]
    g = dup_groups(rows)
    assert g["f1"] == g["f2"] == g["f3"] == "f1" and g["f4"] == "f4", g
    a, m = fit_dm([[3, 0, 0, 0], [0, 3, 0, 0], [1, 1, 1, 0], [0, 0, 2, 1], [0, 0, 0, 3]] * 20)
    assert 0.03 < a < 1e4 and abs(sum(m) - 1) < 1e-9
    VENUES["selftest"] = {"venue_id": "X/2025/Conference", "name": "ICLR", "year": 2025, "role": "train", "stage": "final"}
    RATING_MAP["selftest"] = RATING_MAP["iclr2025"]
    LABEL_METHOD["selftest"] = LABEL_METHOD["iclr2025"]
    row = {"forum": "abc", "number": 1, "venue": "selftest", "cdate": 0, "title": "A title here",
           "abstract": "We study things. " * 20 + "Code: https://github.com/x/y.", "keywords": ["k1; k2"], "primary_area": "area",
           "license": "CC BY 4.0", "author_name_in_text": False, "status": "rejected",
           "reviews": [{"cdate": 1, "mdate": 2, "rating": 3, "soundness": 2, "presentation": 3, "contribution": 2,
                        "confidence": 4},
                       {"cdate": 1, "mdate": 2, "rating": 6, "soundness": 3, "presentation": 3, "contribution": 3,
                        "confidence": 3},
                       {"cdate": 1, "mdate": 9_000_000, "rating": 6, "soundness": 3, "presentation": 2,
                        "contribution": 3, "confidence": 3}]}
    drops, diag = Counter(), {"strong_outcome_examples": [], "train_groups": set(), "release_ms": {"selftest": 0}}
    recs, counts = build_records([row], {"abc": "abc"}, set(), drops, diag)
    assert len(recs) == 1, drops
    r = recs[0]
    assert r["id"] == f"openreview/selftest-{sha('abc')}" and "github" not in r["state"]
    lab = r["labels"]["reviewer_rating"]
    assert abs(lab[RATING_LEVELS[0]] - 1 / 3) < 1e-9 and abs(lab[RATING_LEVELS[2]] - 2 / 3) < 1e-9
    assert r["meta"]["n_raters"] == 3 and r["meta"]["raw"]["rating"] == [3, 6, 6]
    assert r["meta"]["reviews_modified_after_release"] == 1 and r["meta"]["raw"]["modified_after_release"] == [0, 0, 1]
    assert set(r["questions"]) == set(QIDS) and counts["reviewer_rating"] == [[1, 0, 2, 0]]
    st = json.loads(r["state"])
    assert set(st) == {"submission", "venue"} and st["submission"]["keywords"] == ["k1", "k2"]
    row2 = dict(row, reviews=row["reviews"][:2])
    assert not build_records([row2], {"abc": "abc"}, set(), drops, diag)[0]
    assert drops["fewer_than_3_rated_reviews"] == 1
    assert r["meta"]["label_source"]["reviewer_rating"] == "votes:3"
    assert LABEL_METHOD["iclr2026"]["reviewer_rating"].endswith("bins 0,2 | 4 | 6 | 8,10")
    del VENUES["selftest"], RATING_MAP["selftest"], LABEL_METHOD["selftest"]
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch", help="download the venues through the API (needs openreview-py and ~/.openreview)")
    f.add_argument("--out", default="/tmp/openreview_jev")
    f.add_argument("--venues", default=",".join(VENUES))
    f.add_argument("--page", type=int, default=1000, help="submissions per API page (with their direct replies)")
    f.add_argument("--sleep", type=float, default=5.0, help="seconds between API calls")
    f.add_argument("--refetch", action="store_true")
    b = sp.add_parser("build", help="cache -> corpus, statistics, final check (no network)")
    b.add_argument("--out", default="/tmp/openreview_jev")
    sp.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "fetch":
        a.venues = [v.strip() for v in a.venues.split(",") if v.strip()]
        unknown = [v for v in a.venues if v not in VENUES]
        if unknown:
            raise SystemExit(f"unknown venues {unknown}; known: {list(VENUES)}")
        fetch(a)
    elif a.cmd == "build":
        build(a)
    else:
        self_test()


if __name__ == "__main__":
    main()
