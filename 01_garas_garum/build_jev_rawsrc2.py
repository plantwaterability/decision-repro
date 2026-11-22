#!/usr/bin/env python3
"""
build_jev_rawsrc2.py -- decision datasets built from RAW public registries (researchers' list "Y", CLAUDE_STATE (74)):
ClinicalTrials.gov trial outcomes, NVD CVE -> CWE (expert key) and Supreme Court Database dispositions, plus a
verification probe of chufangao/CTO. Write-up, licences, counts and every caveat: research/rawsrc2_pipeline.md.

Output (jev_data record schema: build_jev_data.record / validate; every file reads back through
trainers/cache_features.iter_units)
  training corpora   /tmp/rawsrc2_jev/<source>/{train,val}/<source>__<question>.jsonl + manifest.json
  GDE-1 eval sets    /tmp/gde1/<set>/eval/<set>__<question>.jsonl (+ dev/ for SCDB) + manifest.json, check_report.json,
                     samples.json. EVAL ONLY: never place these under a training directory.

Sources (subcommand = source)
  ctgov    ClinicalTrials.gov API v2 (keyless; NLM terms, quoted in LICENSES). Interventional studies with a terminal or
           halted status. The registration history API (/api/int/studies/<nct>/history) refuses non-browser clients
           (HTTP 403 to this builder's descriptive User-Agent, 2026-10-01; the public API v2 serves only the CURRENT
           record), so the state is the current record restricted to REGISTRATION-DESIGN fields that the status
           machinery does not rewrite: titles, sponsor and its class, collaborators, study type, phase, design
           (allocation, model, purpose, masking), arms and interventions, conditions and keywords, eligibility (sex,
           ages, healthy volunteers, criteria text), brief summary, oversight flags, and the first-posted month.
           Primary outcome measures are left out: results submission rewrites them (results-era wording leaked). NEVER in the state: overall status, why stopped, status-verified date, start /
           primary-completion / completion dates and their ACTUAL/ESTIMATED types, enrollment (count and type: ACTUAL
           leaks, withdrawn = 0), locations and sites, results / references / documents / limitations sections,
           last-update dates, NCT ids (masked). Study-status sentences in free text are removed (STATUS_RX).
           Questions
             final_status        choice 4  completed / terminated / withdrawn / suspended (source ctgov/final_status)
             results_posted_12m  noul      a completed or terminated trial posts results on ClinicalTrials.gov within
                                           12 months of its actual primary completion (source ctgov/results_posted;
                                           a separate record, so its presence never reveals final_status; primary
                                           completion from 2008-10 on, when the results database existed)
             (planned-completion + 12 months: NOT BUILT -- needs the registration-time planned date, which only the
             refused history API holds)
           Resolution dates: completed / terminated = the ACTUAL completion date (else ACTUAL primary completion);
           withdrawn / suspended = the status-verified month (ESTIMATED; flagged in meta), and train additionally needs
           the last update before the train cut. results_posted_12m resolves at primary completion + 12 months.
           Splits by resolution: train < 2025-01-01 <= val < 2025-07-01 <= eval <= data version. final_status EVAL is
           the strict post-cutoff cohort (first posted on / after 2025-07-01): without the history API, a status
           change cannot be dated (re-verification and revised completion dates move old halts into the window).
           results_posted_12m EVAL = actual primary completion on / after 2025-07-01. A trial lives in ONE split
           (questions eligible for an earlier split are dropped). Retrospectively registered trials (first posted on /
           after the resolution) are dropped.
  nvd_cwe  CVE -> CWE class (expert key, decision_data_guide.md Category 3). State = the CVE content AS RECEIVED BY NVD
           at publication (build_gde1_b.py's KEV logic, reused: the NVD Change History "New CVE Received" event,
           CVE id masked), WITHOUT the CNA's CWE line, with explicit CWE-<n> mentions masked and cwe.mitre.org links
           removed. Gold = the CWE NVD's analysts assigned (source nvd@nist.gov) when present, else the CNA's (record
           label_source expert:nvd / expert:cna; eval keeps them as separate sources). Options = the 25 CWEs NVD
           assigns most often in the train window + other; any CWE maps to its nearest listed ancestor along CWE-1000
           ChildOf links (NVD's own CVMAP rule), NVD-CWE-Other -> other, NVD-CWE-noinfo -> not asked. Several gold CWEs
           split the mass equally. Splits by NVD publication: train 2023-11 (the first month NVD logged "New CVE
           Received") .. 2024-12, val 2025-01 .. 06, eval 2025-07 .. 2026-06. GDE-1 side panel E (expert key).
           ZERO-SHOT LINE: HOLD. realworld_data.md excluded NVD / KEV / EPSS from training as security triage; the
           train split is written with a HOLD notice and meta.td_workflow = "security" until the user rules.
  scdb     Supreme Court Database 2026 Release 01. Licence: the last one the SCDB published is CC BY-NC 3.0 US (old
           site); the new PSU site states none -> NON-COMMERCIAL: EVAL ONLY (GDE-1 O), nothing under /tmp/rawsrc2_jev.
           State = pre-decision fields only (term, chief, case name, docket, jurisdiction route, parties and their
           types, source / origin courts, the court below's disposition, agency, issue area, oral argument date or
           none). Question case_disposition: affirmed / reversed / vacated_remanded / other. eval = decided on / after
           2025-07-01 (OT2025); dev = OT2015 .. OT2024 decided before 2025-07-01 (PRE-CUTOFF, knowledge-contaminated:
           probe / calibration only, never a headline).
  cto      chufangao/CTO probe: licence, the human_labels file's composition, overlap with ctgov. Writes nothing to
           the corpora (see the doc for why).
  gate     ext2 dedup (build_jev_ext2.Index: exact / >= 5-word leaf / shingle) of the TRAIN and VAL corpora against
           every eval fingerprint: the GDE-1 sets on disk (content view, template lines stripped), ext2's fetched eval
           families (jev-bench, typed-decisions states, TDB incl. TrialGPT, MMLU/ARC, LegalBench, community, DICES,
           ChaosNLI) and --eval files. Records whose NCT / CVE id appears in any eval text are dropped too. Shingle
           rule (--shingle-rule): `calibrated` (default; >= 5 shared 8-word shingles covering >= 10% of the shorter
           text, shingles in > 20 fingerprint states ignored -- build_jev_openreview / build_gde1_a constants) or
           `ext2` (>= 3 shared shingles); under `calibrated` the plain-ext2 matches are reported, not dropped (the
           plain rule dropped 36% of CT.gov train on eligibility boilerplate). Then the new GDE-1 sets are checked
           against the --fingerprint training corpora. Rewrites files in place; logged in the manifests.
  check    read-back (iter_units), schema, outcome-word scans, leak probes, label statistics, samples.

Labels (FDA / DMHC conventions): train / val = LAMBDA * onehot + (1 - LAMBDA) * p_hat, p_hat a hierarchical
Dirichlet-smoothed base rate over fields the state shows, LEAVE-ONE-OUT on train (build_jev_fda.group_out_rates /
fit_kappas; val gets train-only rates); eval = the realised outcome / key (one-hot; several gold CWEs split equally),
with meta.base_rate and meta.smoothed_target kept.

Usage (laptop; the gate needs huggingface_hub + pyarrow, the rest numpy + scipy; pandas for scdb's value labels):
  UVR="uv run --no-project --with numpy --with scipy --with pandas --with pyarrow --with huggingface_hub python"
  $UVR build_jev_rawsrc2.py ctgov   [--fetch-only]
  $UVR build_jev_rawsrc2.py nvd_cwe [--fetch-only]
  $UVR build_jev_rawsrc2.py scdb
  $UVR build_jev_rawsrc2.py cto
  HF_HOME=/tmp/rawsrc2_work/hf $UVR build_jev_rawsrc2.py gate --fingerprint TAG=PATH ...
  $UVR build_jev_rawsrc2.py check
  $UVR build_jev_rawsrc2.py --self-test                    # offline fixtures
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

import build_gde1_b as gb  # noqa: E402  NVD-at-publication state logic (KEV), GDE-1 helpers
from build_jev_data import choice_q, noul_q, option_keys, record, validate  # noqa: E402
from build_jev_fda import blend, calib_report, encode_levels, fit_kappas, group_out_rates  # noqa: E402
from claude_overlap import norm  # noqa: E402

BUILDER = "build_jev_rawsrc2.py v1 (2026-10-01)"
UA = "jev-rawsrc2-research/0.1 (decision-dataset build from public registries; polite paging)"
WORK = "/tmp/rawsrc2_work"
OUT = "/tmp/rawsrc2_jev"
GDE1 = "/tmp/gde1"
TRAIN_END = "2025-01-01"      # train: resolved before this day
VAL_END = "2025-07-01"        # val: resolved in [TRAIN_END, VAL_END); eval: resolved on / after VAL_END
EVAL_FROM = VAL_END
LAMBDA = 0.5
SEED = 2026
gb.UA = UA                     # build_gde1_b.http_get (the NVD fetchers) sends this project's User-Agent


# =========================================================================== small helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha16(s):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def rank(*parts):
    """A uniform number in [0, 1) from the parts: seeded, input-order-independent selection."""
    return gb.unit_hash(*parts)


def dlo(s):
    """Earliest calendar day of a YYYY-MM or YYYY-MM-DD date (None if missing)."""
    s = str(s or "")[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return date.fromisoformat(s)
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return date.fromisoformat(s + "-01")
    return None


def dhi(s):
    """Latest calendar day of a YYYY-MM or YYYY-MM-DD date (None if missing)."""
    s = str(s or "")[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return date.fromisoformat(s)
    if re.fullmatch(r"\d{4}-\d{2}", s):
        y, m = int(s[:4]), int(s[5:7])
        return date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return None


def clip(text, n):
    t = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) <= n:
        return t
    cut = t[:n].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + " …"


def write_gz(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Refused(SystemExit):
    """A source refused this builder (HTTP 401 / 403 that is not a rate limit). Never retried under another identity."""


_LAST = {}


def polite_get(url, gap, timeout=300, tries=6, accept="application/json", refuse_codes=(401, 403)):
    """GET with this project's descriptive User-Agent, a minimum gap between requests to one host, and backoff on
    429 / 5xx. A 401 / 403 is a refusal: stop and report (no other User-Agent is ever tried)."""
    host = urllib.parse.urlparse(url).netloc
    for t in range(tries):
        wait = _LAST.get(host, 0.0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST[host] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read(300).decode("utf-8", "replace")
            if e.code in refuse_codes:
                raise Refused(f"REFUSED: HTTP {e.code} from {host} for {url[:160]} ({body[:120]!r}). Stopping; this "
                              "builder does not retry under another identity.") from None
            if e.code not in (429, 500, 502, 503, 504) or t == tries - 1:
                raise
            log(f"  HTTP {e.code}; retry {t + 1} in {gap * 4 * 2 ** t:.0f}s")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if t == tries - 1:
                raise
            log(f"  {type(e).__name__}: {e}; retry {t + 1} in {gap * 4 * 2 ** t:.0f}s")
        time.sleep(gap * 4 * 2 ** t)
    raise RuntimeError("unreachable")


def entropy(vals):
    return -sum(x * math.log(x) for x in vals if x > 0)


def lab_dict(keys, vals):
    return gb.lab_dict(keys, vals)


def onehot(keys, k):
    return {x: (1.0 if x == k else 0.0) for x in keys}


def kind_of(lab):
    return "hard" if sum(1 for v in lab.values() if v > 0) == 1 else "soft"


# =========================================================================== licences (quoted from primary sources)
CTG_DATA_VERSION = "2026-10-01T09:00:05"     # /api/v2/version dataTimestamp at fetch time; overwritten by the fetch
LICENSES = {
    "ctgov": ("ClinicalTrials.gov data (U.S. National Library of Medicine), retrieved through the ClinicalTrials.gov API "
              "v2; data processed by ClinicalTrials.gov on {version}. NLM Terms and Conditions: 'In any publication or "
              "distribution of these data, you should: Attribute the source of the data as ClinicalTrials.gov ... "
              "Clearly display the date the data were processed by ClinicalTrials.gov ... State any modifications made "
              "to the content of the data, along with a complete description of the modifications.' 'The "
              "ClinicalTrials.gov data carry an international copyright outside the United States and its Territories "
              "or Possessions. Some ClinicalTrials.gov data may be subject to the copyright of third parties.' "
              "Modifications: selected registration-design fields of the current record only, free text clipped, "
              "trial ids masked, study-status sentences removed (research/rawsrc2_pipeline.md)."),
    "nvd_cwe": ("CVE record text: CVE Terms of Use (MITRE grants a perpetual, worldwide, non-exclusive, no-charge, "
                "royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly "
                "display, publicly perform, sublicense, and distribute CVE; reproduce MITRE's copyright designation and "
                "this license). Publication-time copy (NVD CVE Change History API) and the CWE assignments (NVD CVE API) "
                "from NIST's NVD: 'All NIST publications are available in the public domain according to Title 17 of "
                "the United States Code'. This product uses the NVD API but is not endorsed or certified by the NVD; "
                "the content is modified (fields selected, ids masked), so it is not attributed as the NVD's own. CWE "
                "names and hierarchy: CWE Terms of Use ('CWE is free to use by any organization or individual for any "
                "research, development, and/or commercial purposes'); Copyright (c) 2006-2026, The MITRE Corporation; "
                "CWE is a trademark of The MITRE Corporation."),
    "scdb": ("The Supreme Court Database (Harold J. Spaeth et al.), 2026 Release 01 (SCDB_2026_01, released 2026-09-07), "
             "scdb.la.psu.edu. Licence: Creative Commons BY-NC 3.0 US, the licence the SCDB site published "
             "(rel=license link on scdb.wustl.edu/data.php, archived 2025-10-09); the new PSU site states no licence. "
             "NON-COMMERCIAL: evaluation only, never training."),
}
LICENSE_EVIDENCE = {
    "ctgov": {"url": "https://clinicaltrials.gov/about-site/terms-conditions",
              "read_from": "the page's own content bundle https://clinicaltrials.gov/chunk-W433262W.js (sha256 "
                           "ee577d202734ac589a5fd724ffc12b5ebc4655773872fd4ab25fc3659261811b, fetched 2026-10-01; the "
                           "page is rendered by JavaScript)",
              "also": "https://clinicaltrials.gov/api/oas/v2: 'the most recent version of the clinical trials' (no "
                      "history in the public API); robots.txt: 'Disallow: /api/' with 'Allow: /api/int/' for the SPA",
              "history_api": "https://clinicaltrials.gov/api/int/studies/<nct>/history: HTTP 403 to this builder's "
                             "User-Agent (every /api/int/ path; the public /api/v2 answers 200)"},
    "nvd_cwe": {"cve": "github.com/CVEProject/cve-website src/views/Legal/TermsOfUse.vue @ e03b94dd (as research/gde1_part_b.md)",
                "nvd": "https://nvd.nist.gov/developers/terms-of-use and /general/faq, read from the site bundle "
                       "https://nvd.nist.gov/main-QN4PLKKZ.js (sha256 dbd9f659695821edbf7f971560639e0dc06945f5052204a3"
                       "6139c6ce41f73daa, 2026-10-01): 'All NIST publications are available in the public domain "
                       "according to Title 17 of the United States Code, however services which utilize or access the "
                       "NVD are asked to display the following notice prominently within the application: \"This "
                       "product uses data from the NVD API but is not endorsed or certified by the NVD.\"'; 'If you "
                       "modify the content accessed through the API, you may not attribute the source as the NVD.'",
                "cwe": "https://cwe.mitre.org/about/termsofuse.html (2026-10-01): 'CWE is free to use by any "
                       "organization or individual for any research, development, and/or commercial purposes ... Any "
                       "copy you make for such purposes is authorized on the condition that you reproduce MITRE's "
                       "copyright designation and this license in any such copy.'"},
    "scdb": {"current_site": "https://scdb.la.psu.edu/ (home, data/2026-release-01, about, how-to-cite-us, "
                             "documentation, data): no licence or terms text; footer 'Copyright (c)2026 The Pennsylvania "
                             "State University'",
             "old_site": "http://scdb.wustl.edu/data.php: HTTP 403 to this builder (2026-10-01); Wayback snapshot "
                         "20251009222733 (SCDB_2025_01 page) carries <a rel=\"license\" "
                         "href=\"http://creativecommons.org/licenses/by-nc/3.0/us/\"> and the commented notice 'The "
                         "Supreme Court Database is licensed under a Creative Commons License.'",
             "verdict": "NC (last stated licence) / unstated on the new site -> eval only"},
}


# =========================================================================== ClinicalTrials.gov: fetch
CTG_API = "https://clinicaltrials.gov/api/v2/studies"
CTG_VERSION = "https://clinicaltrials.gov/api/v2/version"
CTG_GAP = 1.3                  # seconds between requests (robots.txt asks for Crawl-delay: 1)
CTG_STATUSES = ("COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED")
CTG_FULL_FIELDS = ",".join([
    "protocolSection.identificationModule", "protocolSection.statusModule", "protocolSection.sponsorCollaboratorsModule",
    "protocolSection.oversightModule", "protocolSection.descriptionModule.briefSummary", "protocolSection.conditionsModule",
    "protocolSection.designModule", "protocolSection.armsInterventionsModule", "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.eligibilityModule", "hasResults"])


def ctg_slim_status(s):
    ps = s.get("protocolSection") or {}
    sm = ps.get("statusModule") or {}

    def ds(k):
        x = sm.get(k) or {}
        return [x.get("date"), x.get("type")]
    return {"nct": (ps.get("identificationModule") or {}).get("nctId"), "status": sm.get("overallStatus"),
            "verified": sm.get("statusVerifiedDate"), "start": ds("startDateStruct"),
            "pc": ds("primaryCompletionDateStruct"), "comp": ds("completionDateStruct"),
            "first_submit": sm.get("studyFirstSubmitDate"),
            "first_post": (sm.get("studyFirstPostDateStruct") or {}).get("date"),
            "results_first_submit": sm.get("resultsFirstSubmitDate"),
            "results_first_post": (sm.get("resultsFirstPostDateStruct") or {}).get("date"),
            "last_update_post": (sm.get("lastUpdatePostDateStruct") or {}).get("date"),
            "has_results": bool(s.get("hasResults"))}


def ctg_fetch_index(cache):
    """Every interventional study with status completed / terminated / withdrawn / suspended: id + status module only
    (~300k studies, 1,000 per page)."""
    cache = Path(cache)
    dest = cache / "index.jsonl.gz"
    if dest.exists():
        return json.loads((cache / "index_meta.json").read_text())
    ver = json.loads(polite_get(CTG_VERSION, CTG_GAP))
    params = {"filter.overallStatus": "|".join(CTG_STATUSES), "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
              "fields": "protocolSection.identificationModule.nctId,protocolSection.statusModule,hasResults",
              "pageSize": 1000, "countTotal": "true", "format": "json"}
    rows, token, total, page = [], None, None, 0
    while True:
        p = dict(params)
        if token:
            p["pageToken"] = token
            p.pop("countTotal", None)
        d = json.loads(polite_get(CTG_API + "?" + urllib.parse.urlencode(p), CTG_GAP))
        if total is None:
            total = d.get("totalCount")
        rows += [ctg_slim_status(s) for s in d.get("studies", [])]
        page += 1
        if page % 25 == 0:
            log(f"  ctgov index: {len(rows):,} / {total:,}")
        token = d.get("nextPageToken")
        if not token:
            break
    if total is not None and len(rows) != total:
        raise SystemExit(f"ctgov index: got {len(rows)} of {total} studies; rerun")
    write_gz(dest, rows)
    meta = {"fetched": datetime.now().isoformat(timespec="seconds"), "api_version": ver.get("apiVersion"),
            "data_timestamp": ver.get("dataTimestamp"), "total": total, "query": params}
    (cache / "index_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"ctgov index: {len(rows):,} studies (data {ver.get('dataTimestamp')})")
    return meta


def ctg_fetch_full(cache, ncts, batch=100):
    """The design / eligibility modules of the selected studies (resumable: ids already in full.jsonl.gz are skipped).
    Stored as returned (protocolSection subset), so `build` can be re-run offline."""
    cache = Path(cache)
    dest = cache / "full.jsonl.gz"
    have = {}
    if dest.exists():
        have = {r["protocolSection"]["identificationModule"]["nctId"]: r for r in read_gz(dest)}
    todo = sorted(set(ncts) - set(have))
    log(f"ctgov full records: {len(have):,} cached, {len(todo):,} to fetch")
    new = []
    for i in range(0, len(todo), batch):
        ids = todo[i:i + batch]
        p = {"filter.ids": ",".join(ids), "fields": CTG_FULL_FIELDS, "pageSize": batch, "format": "json"}
        d = json.loads(polite_get(CTG_API + "?" + urllib.parse.urlencode(p), CTG_GAP))
        new += d.get("studies", [])
        if (i // batch) % 20 == 0:
            log(f"  ctgov full: {i + len(ids):,} / {len(todo):,}")
        if len(new) >= 5000:                      # checkpoint
            have.update({r["protocolSection"]["identificationModule"]["nctId"]: r for r in new})
            write_gz(dest, have.values())
            new = []
    have.update({r["protocolSection"]["identificationModule"]["nctId"]: r for r in new})
    write_gz(dest, have.values())
    return have


# =========================================================================== NVD: fetch (history = build_gde1_b.nvd_fetch)
NVD_CVES = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_FROM = "2023-11-01"        # NVD's change history has "New CVE Received" events from mid-November 2023 on
NVD_TO = "2026-07-01"          # eval publications end 2026-06-30 (NVD analysis lags; coverage is reported per month)
CWE_XML = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"


def nvd_slim(cve):
    return {"id": cve.get("id"), "published": cve.get("published"), "lastModified": cve.get("lastModified"),
            "vulnStatus": cve.get("vulnStatus"), "sourceIdentifier": cve.get("sourceIdentifier"),
            "weaknesses": [{"source": w.get("source"), "type": w.get("type"),
                            "values": [x.get("value") for x in w.get("description") or [] if x.get("lang") == "en"]}
                           for w in cve.get("weaknesses") or []],
            "cveTags": cve.get("cveTags") or []}


def nvd_fetch_cves(cache, start, end, chunk_days=119, per_page=2000, spacing=7.0):
    """The CURRENT NVD record (weaknesses = CWE assignments, vulnStatus) of every CVE published in [start, end), one
    jsonl.gz per <= 119-day window; slimmed to the fields the label needs. Public rate: 5 requests / rolling 30 s."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    t0, t_end = date.fromisoformat(start), date.fromisoformat(end)
    while t0 < t_end:
        t1 = min(t0 + timedelta(days=chunk_days), t_end)
        dest = cache / f"cves_{t0}_{t1}.jsonl.gz"
        if dest.exists():
            t0 = t1
            continue
        rows, idx, total = [], 0, None
        while total is None or idx < total:
            qs = urllib.parse.urlencode({"pubStartDate": f"{t0}T00:00:00.000", "pubEndDate": f"{t1}T00:00:00.000",
                                         "resultsPerPage": per_page, "startIndex": idx})
            d = json.loads(gb.http_get(NVD_CVES + "?" + qs, timeout=600, tries=9, backoff=8.0))
            total = int(d["totalResults"])
            got = d.get("vulnerabilities", [])
            rows += [nvd_slim(v["cve"]) for v in got]
            idx += len(got)
            log(f"  NVD cves {t0}..{t1}: {idx}/{total}")
            if not got:
                break
            time.sleep(spacing)
        if len(rows) != total:
            raise SystemExit(f"NVD cves {t0}..{t1}: got {len(rows)} of {total}; rerun")
        write_gz(dest, rows)
        time.sleep(spacing)
        t0 = t1


def nvd_fetch_all(cache):
    cache = Path(cache)
    gb.nvd_fetch(cache / "history", NVD_FROM, NVD_TO)
    nvd_fetch_cves(cache / "cves", NVD_FROM, NVD_TO)
    gb.nvd_sources_fetch(cache / "sources.json")
    z = cache / "cwec_latest.xml.zip"
    if not z.exists():
        z.write_bytes(polite_get(CWE_XML, 2.0, accept="application/zip"))
    return cache


# =========================================================================== ClinicalTrials.gov: selection
FS_KEYS = ["completed", "terminated", "withdrawn", "suspended"]
FS_CRITERIA = {
    "completed": "Completed: the trial ends normally -- participants are no longer examined or treated, and the last "
                 "participant's last visit has taken place.",
    "terminated": "Terminated: the trial stops early, after it has begun enrolling participants, and will not restart.",
    "withdrawn": "Withdrawn: the trial stops before it enrols its first participant.",
    "suspended": "Suspended: the trial is halted early but may restart (it was still recorded as suspended when these "
                 "data were taken).",
}
FS_INSTR = ("Which final status will this trial have on ClinicalTrials.gov? The trial is registered as described; "
            "decide from the registration alone.")
RP_INSTR = ("This trial reached its primary completion (it was completed or terminated). Are its summary results "
            "posted on ClinicalTrials.gov within 12 months after the actual primary completion date?")
RP_CRITERIA = {"yes": "The trial's results section (results tables) is first posted on ClinicalTrials.gov no later than "
                      "12 months after its actual primary completion date.",
               "no": "No results are posted within those 12 months (they are posted later, or never)."}
CTG_CAPS = {("final_status", "train"): 14000, ("final_status", "val"): 1500, ("final_status", "eval"): 5000,
            ("results_posted", "train"): 6000, ("results_posted", "val"): 800, ("results_posted", "eval"): 3000}
OVERSAMPLE = 1.12              # selected before the state filters; capped again after them
RESULTS_DB_FROM = date(2008, 10, 1)   # the results database (FDAAA 801) opened in September 2008: no earlier postings


def split_of_day(d, end):
    """train / val / eval for a resolution day (eval also needs d <= the data date `end`)."""
    if d is None:
        return None
    if d < date.fromisoformat(TRAIN_END):
        return "train"
    if d < date.fromisoformat(VAL_END):
        return "val"
    return "eval" if d <= end else None


def ctg_final_status_resolution(r, end):
    """(split, resolution day, estimated?, reason) for final_status, or (None, None, None, reason).

    EVAL is the strict post-cutoff cohort: first posted on / after EVAL_FROM. Without the history API the day a status
    became public is unknown: re-verification moves the status-verified month of long-withdrawn trials into the eval
    window, and terminated trials can carry an ACTUAL completion date long after the termination was posted (both
    found against chufangao/CTO's 2024 snapshot), so only a trial that did not exist before the cutoff is
    guaranteed to resolve after it."""
    st = r["status"]
    fp = dlo(r["first_post"])
    if fp is None:
        return None, None, None, "no_first_post_date"
    res = _ctg_fs_resolution(r, end, st, fp)
    if res[0] == "eval" and fp < date.fromisoformat(EVAL_FROM):
        return None, None, None, "eval_window_but_registered_before_cutoff (status timing unverifiable)"
    return res


def _ctg_fs_resolution(r, end, st, fp):
    if st in ("COMPLETED", "TERMINATED"):
        comp, pc = r["comp"], r["pc"]
        src = comp if comp[1] == "ACTUAL" and comp[0] else (pc if pc[1] == "ACTUAL" and pc[0] else None)
        if src is None:
            return None, None, None, "no_actual_end_date"
        lo, hi = dlo(src[0]), dhi(src[0])
        sp_lo, sp_hi = split_of_day(lo, end), split_of_day(hi, end)
        if sp_lo != sp_hi or sp_lo is None:                    # a month-precision date straddling a cut, or too late
            return None, None, None, "end_date_straddles_cut_or_after_data"
        if fp >= lo:
            return None, None, None, "registered_on_or_after_its_end (retrospective)"
        return sp_lo, lo, False, "actual end date"
    if st in ("WITHDRAWN", "SUSPENDED"):
        v_lo, v_hi = dlo(r["verified"]), dhi(r["verified"])
        lu = dlo(r["last_update_post"])
        if v_lo is None or lu is None:
            return None, None, None, "no_verified_or_update_date"
        sp_lo, sp_hi = split_of_day(v_lo, end), split_of_day(v_hi, end)
        if sp_lo != sp_hi or sp_lo is None:
            return None, None, None, "verified_month_straddles_cut_or_after_data"
        if fp > v_hi:
            return None, None, None, "verified_before_first_post"
        if sp_lo == "train" and lu >= date.fromisoformat(TRAIN_END):
            return None, None, None, "halted_train_candidate_updated_after_train_cut"
        if sp_lo == "val" and lu >= date.fromisoformat(VAL_END):
            return None, None, None, "halted_val_candidate_updated_after_val_cut"
        if sp_lo == "eval" and lu < date.fromisoformat(EVAL_FROM):
            return None, None, None, "halted_eval_candidate_not_updated_in_eval_window"
        return sp_lo, max(v_lo, fp), True, "status-verified month (estimate of when the halt was recorded)"
    return None, None, None, f"status_{st}"


def ctg_results_resolution(r, end):
    """(split, deadline, label yes?, reason) for results_posted_12m."""
    if r["status"] not in ("COMPLETED", "TERMINATED"):
        return None, None, None, "not_completed_or_terminated"
    pc = r["pc"]
    if pc[1] != "ACTUAL" or not pc[0]:
        return None, None, None, "no_actual_primary_completion"
    pc_lo, pc_hi = dlo(pc[0]), dhi(pc[0])          # month-precision dates: [first, last] day of the month
    if pc_lo is None or pc_lo < RESULTS_DB_FROM:
        return None, None, None, "primary_completion_before_the_results_database (launched 2008-09)"
    fp = dlo(r["first_post"])
    if fp is None or pc_lo is None or fp >= pc_lo:
        return None, None, None, "registered_on_or_after_primary_completion"
    dl_lo, dl_hi = pc_lo + timedelta(days=365), pc_hi + timedelta(days=365)
    if dl_hi > end:
        return None, None, None, "deadline_after_data_date"
    if pc_lo >= date.fromisoformat(EVAL_FROM):
        sp = "eval"                                  # every posting (after primary completion) falls in the eval window
    elif dl_hi < date.fromisoformat(TRAIN_END):
        sp = "train"
    elif date.fromisoformat(TRAIN_END) <= dl_lo and dl_hi < date.fromisoformat(VAL_END):
        sp = "val"
    else:
        return None, None, None, "deadline_in_val_eval_gap_or_straddles_a_cut"
    rp = dlo(r["results_first_post"])
    if rp and rp < pc_lo:
        return None, None, None, "results_posted_before_primary_completion (inconsistent dates)"
    if rp and dl_lo < rp <= dl_hi:
        return None, None, None, "posting_inside_the_month_precision_ambiguity"
    return sp, dl_hi, bool(rp and rp <= dl_lo), "actual primary completion + 365 days" + (
        " (month-precision date: last day of the month)" if pc_lo != pc_hi else "")


def ctg_select(index, end, caps=CTG_CAPS, seed=SEED):
    """Eligibility, the one-split-per-trial rule, and a seeded uniform sample per (question, split)."""
    rep = Counter()
    cand = {}
    order = {"train": 0, "val": 1, "eval": 2}
    for r in index:
        fs = ctg_final_status_resolution(r, end)
        rp = ctg_results_resolution(r, end)
        rep[f"final_status/{fs[0] or 'drop:' + fs[3]}"] += 1
        rep[f"results_posted/{rp[0] or 'drop:' + rp[3]}"] += 1
        qs = {}
        if fs[0]:
            qs["final_status"] = fs
        if rp[0]:
            qs["results_posted"] = rp
        if not qs:
            continue
        sp = max((v[0] for v in qs.values()), key=order.get)
        for q in list(qs):
            if qs[q][0] != sp:
                rep[f"{q}/dropped_trial_in_later_split_{sp}"] += 1
                del qs[q]
        cand[r["nct"]] = (sp, r, qs)
    picked = defaultdict(list)
    for q in ("final_status", "results_posted"):
        for sp in ("train", "val", "eval"):
            pool = [n for n, (s, _, qs) in cand.items() if s == sp and q in qs]
            pool.sort(key=lambda n: rank(seed, "ctgov", q, n))
            k = int(math.ceil(caps[(q, sp)] * OVERSAMPLE))
            picked[(q, sp)] = pool[:k]
            rep[f"pool/{q}/{sp}"] = len(pool)
    return cand, picked, dict(rep)


# =========================================================================== ClinicalTrials.gov: state
STATUS_RX = re.compile(
    r"\b(?:(?:this |the )?(?:study|trial|protocol|enrol+ment|recruitment|accrual)\s+(?:was|has been|had been|is|were|"
    r"have been|will be)\s+(?:prematurely |permanently |temporarily )?(?:terminated|withdrawn|suspended|halted|stopped|"
    r"closed|completed|cancel+ed|discontinued|put on hold|ended|abandoned)|(?:terminated|stopped|halted|closed|ended) "
    r"(?:early|prematurely)|early termination|prematurely (?:terminated|stopped|closed|ended)|study (?:termination|"
    r"withdrawal)|why stopped|no (?:participants|patients|subjects) (?:were |have been )?(?:enrol+ed|recruited)|"
    r"(?:never|not) (?:started|initiated|opened)|futility|(?:low|slow|poor|insufficient) (?:enrol+ment|accrual|"
    r"recruitment)|sponsor decision|business decision|lack of funding|funding (?:was )?withdrawn|was withdrawn|"
    r"(?:have|has|were|was) (?:been )?enrol+ed|results (?:showed|show|demonstrated|indicated|suggest)|"
    r"(?:the )?results (?:of this|of the) (?:study|trial) (?:showed|were))\b", re.I)
NCT_RX = re.compile(r"\bNCT\s?\d{8}\b", re.I)
SPONSOR_CLASS = {"INDUSTRY": "industry", "NIH": "US National Institutes of Health",
                 "FED": "US federal agency other than NIH", "OTHER_GOV": "other government body", "INDIV": "individual",
                 "NETWORK": "research network", "OTHER": "other (university, hospital, foundation, ...)",
                 "AMBIG": "ambiguous", "UNKNOWN": "unknown"}
PHASE = {"EARLY_PHASE1": "early phase 1", "PHASE1": "phase 1", "PHASE2": "phase 2", "PHASE3": "phase 3",
         "PHASE4": "phase 4", "NA": "not applicable"}
CTG_TEMPLATE = re.compile(r"^(CLINICAL TRIAL REGISTRATION .*|First posted on ClinicalTrials\.gov: .*|Design: .*|"
                          r"Phase: .*|Lead sponsor class: .*|Oversight: .*|Eligibility: .*|Brief summary:|"
                          r"Eligibility criteria \(abridged\):|Primary outcome measures:|Interventions:|Arms: .*|"
                          r"Conditions: .*|Keywords: .*|Collaborators: .*)$")


def lc(x):
    return str(x or "").replace("_", " ").lower()


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+", str(text or "")) if s.strip()]


def scrub_free_text(text, rep, tag):
    """Mask trial ids; remove sentences / lines with study-status wording (counted)."""
    t = NCT_RX.sub("NCT[masked]", str(text or ""))
    keep = []
    for s in sentences(t):
        if STATUS_RX.search(s):
            rep[f"status_sentence_removed/{tag}"] += 1
            continue
        keep.append(s.strip())
    return " ".join(keep) if tag == "summary" else "\n".join(keep)


def ctg_state(st, rep, max_summary=700, max_criteria=1500, include_outcomes=False):
    """The registration-design state of one study (protocolSection subset) -> (text or None, info).

    Primary outcome measures are OFF by default: the current record carries the results-era wording of outcome
    measures (titles and time frames are rewritten when results are submitted, e.g. 'Number of Participants With ...',
    'At the primary completion date, the median ...'); in the first build their presence lifted the results-posted
    rate 3.3x and lowered the withdrawn rate 2x (research/rawsrc2_pipeline.md)."""
    ps = st["protocolSection"]
    idm = ps.get("identificationModule") or {}
    sm = ps.get("statusModule") or {}
    sp = ps.get("sponsorCollaboratorsModule") or {}
    ov = ps.get("oversightModule") or {}
    de = ps.get("descriptionModule") or {}
    co = ps.get("conditionsModule") or {}
    dm = ps.get("designModule") or {}
    ai = ps.get("armsInterventionsModule") or {}
    om = ps.get("outcomesModule") or {}
    el = ps.get("eligibilityModule") or {}
    titles = " ".join([str(idm.get("briefTitle") or ""), str(idm.get("officialTitle") or "")])
    if STATUS_RX.search(titles):
        rep["drop/status_wording_in_title"] += 1
        return None, None
    fp = str((sm.get("studyFirstPostDateStruct") or {}).get("date") or "")[:7]
    di = dm.get("designInfo") or {}
    mk = di.get("maskingInfo") or {}
    phases = [PHASE.get(p, lc(p)) for p in dm.get("phases") or []]
    lead = sp.get("leadSponsor") or {}
    colls = [c.get("name") for c in sp.get("collaborators") or [] if c.get("name")]
    lines = ["CLINICAL TRIAL REGISTRATION (design as registered; no status, dates or enrolment)",
             f"First posted on ClinicalTrials.gov: {fp}",
             f"Title: {NCT_RX.sub('NCT[masked]', clip(idm.get('briefTitle'), 300))}"]
    if idm.get("officialTitle") and idm.get("officialTitle") != idm.get("briefTitle"):
        lines.append(f"Official title: {NCT_RX.sub('NCT[masked]', clip(idm.get('officialTitle'), 400))}")
    lines.append(f"Lead sponsor: {clip(lead.get('name'), 160)}")
    lines.append(f"Lead sponsor class: {SPONSOR_CLASS.get(lead.get('class'), lc(lead.get('class')))}")
    if colls:
        lines.append(f"Collaborators: {len(colls)} ({clip('; '.join(colls[:5]), 300)}"
                     f"{'; ...' if len(colls) > 5 else ''})")
    lines.append(f"Phase: {', '.join(phases) if phases else 'not stated'}")
    design = [f"allocation {lc(di.get('allocation')) or 'not stated'}",
              f"model {lc(di.get('interventionModel')) or 'not stated'}",
              f"primary purpose {lc(di.get('primaryPurpose')) or 'not stated'}",
              f"masking {lc(mk.get('masking')) or 'not stated'}"
              + (f" ({', '.join(lc(w) for w in mk.get('whoMasked') or [])})" if mk.get("whoMasked") else "")]
    lines.append("Design: " + "; ".join(design))
    arms = ai.get("armGroups") or []
    if arms:
        types = Counter(lc(a.get("type")) or "unspecified" for a in arms)
        lines.append(f"Arms: {len(arms)} ({', '.join(f'{n} {t}' for t, n in types.most_common())})")
    ivs = ai.get("interventions") or []
    if ivs:
        lines.append("Interventions:")
        for iv in ivs[:8]:
            lines.append(f"- {lc(iv.get('type'))}: {NCT_RX.sub('NCT[masked]', clip(iv.get('name'), 160))}")
        if len(ivs) > 8:
            lines.append(f"- ... and {len(ivs) - 8} more")
    conds = co.get("conditions") or []
    if conds:
        lines.append(f"Conditions: {clip('; '.join(conds[:8]), 400)}{'; ...' if len(conds) > 8 else ''}")
    kws = co.get("keywords") or []
    if kws:
        lines.append(f"Keywords: {clip('; '.join(kws[:8]), 300)}")
    pos = [o for o in om.get("primaryOutcomes") or [] if o.get("measure")]
    kept_pos = []
    for o in pos:
        txt = f"{clip(o.get('measure'), 200)}" + (f" [{clip(o.get('timeFrame'), 100)}]" if o.get("timeFrame") else "")
        if STATUS_RX.search(txt):
            rep["status_sentence_removed/primary_outcome"] += 1
            continue
        kept_pos.append(NCT_RX.sub("NCT[masked]", txt))
    if kept_pos and include_outcomes:
        lines.append("Primary outcome measures:")
        lines += [f"- {t}" for t in kept_pos[:3]]
        if len(kept_pos) > 3:
            lines.append(f"- ... and {len(kept_pos) - 3} more")
    ages = " to ".join(x for x in (el.get("minimumAge"), el.get("maximumAge")) if x) or "no age limits stated"
    hv = el.get("healthyVolunteers")
    lines.append(f"Eligibility: sex {lc(el.get('sex')) or 'not stated'}; ages {ages}"
                 + (f" ({', '.join(lc(a) for a in el.get('stdAges') or [])})" if el.get("stdAges") else "")
                 + f"; healthy volunteers {'accepted' if hv else 'not accepted' if hv is False else 'not stated'}")
    crit = scrub_free_text(el.get("eligibilityCriteria"), rep, "criteria")
    if crit.strip():
        lines += ["Eligibility criteria (abridged):", clip(crit, max_criteria)]
    summ = scrub_free_text(de.get("briefSummary"), rep, "summary")
    if summ.strip():
        lines += ["Brief summary:", clip(summ, max_summary)]
    flags = []
    if ov.get("oversightHasDmc") is not None:
        flags.append(f"data monitoring committee {'yes' if ov.get('oversightHasDmc') else 'no'}")
    if ov.get("isFdaRegulatedDrug") is not None:
        flags.append(f"FDA-regulated drug {'yes' if ov.get('isFdaRegulatedDrug') else 'no'}")
    if ov.get("isFdaRegulatedDevice") is not None:
        flags.append(f"FDA-regulated device {'yes' if ov.get('isFdaRegulatedDevice') else 'no'}")
    if flags:
        lines.append("Oversight: " + "; ".join(flags))
    text = "\n".join(lines)
    if STATUS_RX.search(text):
        rep["drop/status_wording_after_scrub"] += 1
        return None, None
    info = {"phase": "|".join(dm.get("phases") or []) or "none", "sponsor_class": lead.get("class") or "UNKNOWN",
            "purpose": di.get("primaryPurpose") or "none", "allocation": di.get("allocation") or "none",
            "fda_drug": str(ov.get("isFdaRegulatedDrug")), "first_post": fp,
            "era": "pre2008" if fp < "2008" else ("2008-2016" if fp < "2017" else "2017+")}
    return text, info


# =========================================================================== shared: labels, records, writing
def loo_rates(items, levels_fn, y, K, fit_mask):
    """Hierarchical leave-one-out (each item its own group) Dirichlet-smoothed rates, prior strengths fitted by LOO log
    score on the fit set (build_jev_fda.fit_kappas / group_out_rates). Returns (P [N, K], kappas, report)."""
    import numpy as np
    N = len(items)
    lv = encode_levels([[levels_fn(it, L) for it in items] for L in range(levels_fn.n_levels)])
    groups = np.arange(N)
    y = np.asarray(y)
    fit = np.asarray(fit_mask, dtype=bool)
    kap, ll = fit_kappas(lv, y, K, fit, groups)
    P = group_out_rates(lv, y, K, fit, groups, kap)
    prior = np.bincount(y[fit & (y >= 0)], minlength=K) / max(1, (fit & (y >= 0)).sum())
    rep = {"levels": levels_fn.names, "kappas": kap, "loo_log_score_train": round(ll, 5),
           "class_rates_train": [round(float(x), 4) for x in prior],
           "loo_on_train": calib_report(P[fit], y[fit]) if fit.any() else None}
    for name, m in (("val", "val"), ("eval", "eval")):
        mask = np.array([it.get("split") == m for it in items])
        if mask.any():
            rep[f"train_only_rates_on_{name}"] = calib_report(P[mask], y[mask])
            rep[f"global_rate_on_{name}"] = calib_report(np.tile(prior, (int(mask.sum()), 1)), y[mask])
    return P, kap, rep


class Levels:
    """levels_fn(item, L) -> the key of item at level L (0 = global)."""

    def __init__(self, names, fn):
        self.names, self.fn, self.n_levels = names, fn, len(names)

    def __call__(self, it, L):
        return self.fn(it, L)


def train_meta(dataset, group, as_of, decision, resolution, split_reason, lsrc, lmeth, raw, base, lam, extra=None):
    return {"dataset": dataset, "group": group, "as_of": as_of, "decision_date": decision,
            "resolution_date": resolution, "split_reason": split_reason, "label_source": lsrc, "label_method": lmeth,
            "raw_outcome": raw, "lambda": lam, "base_rate": base, "sample_weight": 1.0, **(extra or {})}


def build_rec(source, state, qs, labs, meta, base_id, licence, rec_id=None):
    kinds = {q: kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, None)
    if rec_id:
        rec["id"] = rec_id
    rec["base_id"] = base_id
    rec["license"] = licence
    rec["meta"] = meta
    validate(rec)
    return rec


def gde1_meta(gset, domain, meta):
    return {"gde1_set": gset, "domain": domain, "gde1_part": "rawsrc2", "eval_only": True, **meta}


def write_corpus(root, recs_by_split, prefix):
    """<root>/<split>/<source with / -> __>.jsonl. Only this builder's own files are replaced (never a directory removal)."""
    root = Path(root)
    counts = {}
    for sp, recs in recs_by_split.items():
        d = root / sp
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob(f"{prefix}*.jsonl"):
            old.unlink()
        fhs = {}
        for r in recs:
            fn = r["source"].replace("/", "__") + ".jsonl"
            if fn not in fhs:
                fhs[fn] = (d / fn).open("w", encoding="utf-8")
            fhs[fn].write(json.dumps(r, ensure_ascii=False) + "\n")
        for fh in fhs.values():
            fh.close()
        counts[sp] = {"records": len(recs), "units": sum(len(r["questions"]) for r in recs)}
    return counts


def read_dir(root, splits=("train", "val", "eval", "dev")):
    out = defaultdict(list)
    for sp in splits:
        for p in sorted((Path(root) / sp).glob("*.jsonl")):
            out[sp] += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return out


def label_stats(recs):
    out = {}
    for sp, v in recs.items():
        per = defaultdict(lambda: {"n": 0, "max_p": 0.0, "ent": 0.0, "raw": Counter()})
        for r in v:
            for q, lab in r["labels"].items():
                d = per[q]
                d["n"] += 1
                d["max_p"] += max(lab.values())
                d["ent"] += entropy(lab.values())
                raw = (r["meta"].get("raw_outcome") or {}).get(q)
                d["raw"][str(raw) if not isinstance(raw, (list, dict)) else json.dumps(raw)] += 1
        out[sp] = {q: {"units": d["n"], "mean_max_p": round(d["max_p"] / d["n"], 4),
                       "mean_entropy_nats": round(d["ent"] / d["n"], 4),
                       "realised": dict(d["raw"].most_common(30))} for q, d in per.items()}
    return out


# =========================================================================== ClinicalTrials.gov: build
def cmd_ctgov(a):
    cache = Path(a.work) / "cache" / "ctgov"
    meta = ctg_fetch_index(cache)
    data_day = date.fromisoformat(str(meta["data_timestamp"])[:10])
    index = list(read_gz(cache / "index.jsonl.gz"))
    cand, picked, rep = ctg_select(index, data_day, seed=a.seed)
    want = sorted({n for v in picked.values() for n in v})
    log(f"ctgov: {len(cand):,} eligible trials; {len(want):,} selected for full records")
    if a.fetch_only:
        ctg_fetch_full(cache, want)
        return
    full = ctg_fetch_full(cache, want)
    licence = LICENSES["ctgov"].format(version=meta["data_timestamp"])
    rep = Counter(rep)
    states = {}
    for n in want:
        st = full.get(n)
        if st is None:
            rep["drop/full_record_missing"] += 1
            continue
        text, info = ctg_state(st, rep)
        if text:
            states[n] = (text, info)
    # items per question (whole trials; one split per trial is already enforced)
    items = {"final_status": [], "results_posted": []}
    for (q, sp), ncts in picked.items():
        k = 0
        for n in ncts:
            if n not in states or k >= CTG_CAPS[(q, sp)]:
                rep[f"cap_or_state_drop/{q}/{sp}"] += n not in states
                continue
            _, r, qs = cand[n]
            res = qs[q]
            it = {"nct": n, "split": sp, "row": r, "state": states[n][0], "info": states[n][1], "res": res}
            items[q].append(it)
            k += 1
    # labels
    out = {"train": [], "val": [], "eval": []}
    reports = {}
    lv_fs = Levels(["global", "phase", "phase x sponsor class", "... x primary purpose", "... x allocation"],
                   lambda it, L: "|".join(["all", it["info"]["phase"], it["info"]["sponsor_class"], it["info"]["purpose"],
                                           it["info"]["allocation"]][:L + 1]))
    lv_rp = Levels(["global", "sponsor class", "sponsor class x phase", "... x registration era", "... x FDA-regulated drug"],
                   lambda it, L: "|".join(["all", it["info"]["sponsor_class"], it["info"]["phase"], it["info"]["era"],
                                           it["info"]["fda_drug"]][:L + 1]))
    for q, its in items.items():
        if not its:
            continue
        if q == "final_status":
            keys = FS_KEYS
            y = [FS_KEYS.index(it["row"]["status"].lower()) for it in its]
            lv = lv_fs
        else:
            keys = ["yes", "no"]
            y = [0 if it["res"][2] else 1 for it in its]
            lv = lv_rp
        P, _, lrep = loo_rates(its, lv, y, len(keys), [it["split"] == "train" for it in its])
        reports[q] = lrep
        for i, it in enumerate(its):
            sp, n, r = it["split"], it["nct"], it["row"]
            p_hat = lab_dict(keys, P[i])
            t = blend(y[i], P[i], len(keys), a.lam)
            yk = keys[y[i]]
            if q == "final_status":
                _, rday, est, why = it["res"]
                qs = {"final_status": choice_q(FS_INSTR, dict(FS_CRITERIA))}
                qid, src = "final_status", "ctgov/final_status"
                lmeth_eval = "onehot(overallStatus at data version)"
            else:
                _, rday, _, why = it["res"]
                est = False
                qs = {"results_posted_12m": {**noul_q(RP_INSTR), "criteria": dict(RP_CRITERIA)}}
                qid, src = "results_posted_12m", "ctgov/results_posted"
                lmeth_eval = "onehot(resultsFirstPostDate <= actual primary completion + 365 days)"
            fp = r["first_post"]
            common = {"nct_id": n, "data_version": meta["data_timestamp"], "first_posted": fp,
                      "record_last_update": r["last_update_post"], "resolution_estimated": est,
                      "resolution_rule": why, "state_fields": "registration-design fields of the CURRENT record "
                      "(history API refused); assumed unchanged since first posting", "td_workflow": None,
                      "info_date_note": "the state's fields are current-record values assumed unchanged since first "
                                        "posting; meta.record_last_update is the record's last change"}
            if sp == "eval":
                m = gde1_meta("ctgov", "O", {
                    "label_provenance": "outcome (ClinicalTrials.gov registry record)", "decision_date": fp,
                    "info_date": fp, "resolution_date": {qid: rday.isoformat()}, "raw_outcome": {qid: yk},
                    "base_rate": {qid: p_hat}, "smoothed_target": {qid: lab_dict(keys, t)},
                    "label_method": {qid: lmeth_eval}, **common})
                labs = {qid: onehot(keys, yk)}
                rec = build_rec(src, it["state"], qs, labs, m, f"ctgov/{n}", licence,
                                rec_id=f"gde1-ctgov-{qid}-{n}")
            else:
                m = train_meta(f"ClinicalTrials.gov API v2 (data {meta['data_timestamp']})", f"nct:{n}", fp, fp,
                               {qid: rday.isoformat()}, f"{sp}: resolved {rday} ({why})", {qid: "outcome"},
                               {qid: f"{a.lam}*{lmeth_eval.replace('onehot', 'onehot', 1)} + {1 - a.lam}*leave-one-out "
                                     f"hierarchical base rate ({' > '.join(lv.names)}; train-only fit)"},
                               {qid: yk}, {qid: p_hat}, a.lam, common)
                labs = {qid: lab_dict(keys, t)}
                rec = build_rec(src, it["state"], qs, labs, m, f"ctgov/{n}", licence,
                                rec_id=f"{src}-{sha16(n + qid)}")
            out[sp].append(rec)
    # exact-state duplicates (sister registrations with identical text): one per source and split, and none of a
    # train / val state that is identical to an eval state of the same question
    eval_states = {(r["source"], norm(r["state"])) for r in out["eval"]}
    for sp in ("eval", "val", "train"):
        seen, keep = set(), []
        for r in sorted(out[sp], key=lambda r: (r["meta"].get("first_posted") or "", r["id"])):
            k = (r["source"], norm(r["state"]))
            if k in seen or (sp != "eval" and k in eval_states):
                rep[f"dedup/{sp}/{r['source']}/identical_state_dropped"] += 1
                continue
            seen.add(k)
            keep.append(r)
        out[sp] = keep
    train_counts = write_corpus(Path(a.out) / "ctgov", {k: out[k] for k in ("train", "val")}, "ctgov__")
    eval_counts = write_corpus(Path(a.gde1) / "ctgov", {"eval": out["eval"]}, "ctgov__")
    common_manifest = {
        "builder": BUILDER, "built": datetime.now().isoformat(timespec="seconds"), "seed": a.seed, "lambda": a.lam,
        "license": licence, "license_evidence": LICENSE_EVIDENCE["ctgov"], "fetch": meta,
        "splits": {"train": f"resolved < {TRAIN_END}", "val": f"{TRAIN_END} <= resolved < {VAL_END}",
                   "eval": f"resolved {EVAL_FROM} .. {data_day} (GDE-1 domain O)"},
        "caps": {f"{q}/{s}": v for (q, s), v in CTG_CAPS.items()}, "build_report": dict(sorted(rep.items())),
        "label_models": reports,
        "not_built": {"completion_within_planned_plus_12m": "needs the registration-time planned primary completion "
                      "date; the current record overwrites it with the ACTUAL date and the history API refused us"}}
    (Path(a.out) / "ctgov" / "manifest.json").write_text(json.dumps(
        {**common_manifest, "role": "TRAINING corpus (train / val)", "counts": train_counts}, indent=1, default=str))
    (Path(a.gde1) / "ctgov" / "manifest.json").write_text(json.dumps(
        {**common_manifest, "set": "ctgov", "domain": "O", "eval_only": True, "counts": eval_counts,
         "splits_stats": {"eval": gb.split_stats(out["eval"])},
         "dedup_vs_training": {"status": "NOT RUN: run `build_jev_rawsrc2.py gate`"}}, indent=1, default=str))
    log(f"ctgov: train {train_counts.get('train')}, val {train_counts.get('val')}, eval {eval_counts.get('eval')}")


# =========================================================================== NVD CVE -> CWE
CWE_NS = "{http://cwe.mitre.org/cwe-7}"
N_TOP_CWE = 25
CWE_INSTR = ("Which weakness class (CWE) does this vulnerability belong to? The options are the CWE classes NVD's "
             "analysts assign most often; a more specific CWE counts as the listed class it falls under, and 'other' "
             "covers every other weakness.")
CWE_CAPS = {"train": 20000, "val": 2000, "eval": 8000}
CWE_REF_RX = re.compile(r"cwe\.mitre\.org", re.I)
CWE_URL_RX = re.compile(r"(?:https?://)?cwe\.mitre\.org\S*", re.I)
CWE_ID_RX = re.compile(r"\bCWE[-_ ]?\d{1,4}\b", re.I)
NVD_SOURCE = "nvd@nist.gov"


def cwe_taxonomy(zpath):
    """{id: name} for weaknesses and categories, {id: [ChildOf parents in the CWE-1000 research view]}, version."""
    import xml.etree.ElementTree as ET
    z = zipfile.ZipFile(zpath)
    root = ET.fromstring(z.read(z.namelist()[0]))
    names, parents = {}, defaultdict(list)
    for w in root.iter(CWE_NS + "Weakness"):
        i = w.get("ID")
        names[i] = w.get("Name")
        for rw in w.iter(CWE_NS + "Related_Weakness"):
            if rw.get("Nature") == "ChildOf" and rw.get("View_ID") == "1000" and rw.get("CWE_ID") not in parents[i]:
                parents[i].append(rw.get("CWE_ID"))
    for c in root.iter(CWE_NS + "Category"):
        names[c.get("ID")] = c.get("Name")
    return names, dict(parents), {"version": root.get("Version"), "date": root.get("Date"), "file": z.namelist()[0]}


def cwe_num(v):
    m = re.fullmatch(r"CWE-(\d+)", str(v).strip())
    return m.group(1) if m else None


def cwe_map(v, options, parents):
    """A CWE value -> the option keys it falls under (nearest listed ancestors along CWE-1000 ChildOf), ['other'], or
    None (no information)."""
    v = str(v).strip()
    if v == "NVD-CWE-noinfo":
        return None
    if v == "NVD-CWE-Other":
        return ["other"]
    n = cwe_num(v)
    if n is None:
        return None
    frontier, seen = [n], {n}
    while frontier:
        hit = sorted({f"cwe_{x}" for x in frontier if f"cwe_{x}" in options})
        if hit:
            return hit
        nxt = []
        for x in frontier:
            for p in parents.get(x, []):
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        frontier = nxt
    return ["other"]


def cwe_gold(weaknesses):
    """(values, provenance) from the current NVD record: NVD's own assignment if any, else the CNA's."""
    nvd, cna = [], []
    for w in weaknesses or []:
        vals = [x for x in w.get("values") or [] if x]
        (nvd if w.get("source") == NVD_SOURCE else cna).extend(v for v in vals if v not in (nvd + cna))
    if nvd:
        if all(v == "NVD-CWE-noinfo" for v in nvd):
            return None, "nvd_noinfo"
        return [v for v in nvd if v != "NVD-CWE-noinfo"], "nvd"
    cna = [v for v in cna if v != "NVD-CWE-noinfo"]
    return (cna, "cna") if cna else (None, "no_cwe")


def cwe_dist(values, options, parents):
    """Equal mass per gold CWE, each CWE's mass split over the option keys it maps to; None if nothing maps."""
    mass = Counter()
    maps = [m for m in (cwe_map(v, options, parents) for v in values) if m]
    if not maps:
        return None
    for m in maps:
        for k in m:
            mass[k] += 1.0 / len(maps) / len(m)
    return mass


def nvd_state(ev, cve, pub):
    """build_gde1_b.kev_state (the publication-time record), WITHOUT the CNA's CWE, with explicit CWE ids masked in the
    description and cwe.mitre.org references dropped."""
    e = dict(ev)
    e["cwe"] = []
    e["description"] = CWE_ID_RX.sub("CWE-[masked]", CWE_URL_RX.sub("[CWE link masked]", ev["description"]))
    e["refs"] = [CWE_ID_RX.sub("CWE-[masked]", u) for u in ev["refs"] if not CWE_REF_RX.search(u)]
    return gb.kev_state(e, cve, pub)


def nvd_split(pub):
    if NVD_FROM <= pub < TRAIN_END:
        return "train"
    if TRAIN_END <= pub < VAL_END:
        return "val"
    if VAL_END <= pub < NVD_TO:
        return "eval"
    return None


def nvd_candidates(events, cves, names):
    """Publication-time states + current gold for every CVE first received in the NVD window."""
    rep = Counter()
    first = {}
    for ch in events:
        cid = str(ch.get("cveId", "")).upper()
        if not cid:
            continue
        if cid in first:
            rep["events/duplicate_new_cve_received"] += 1
            if str(ch.get("created")) >= str(first[cid].get("created")):
                continue
        first[cid] = ch
    cands, cover = [], defaultdict(Counter)
    for cid, ch in first.items():
        pub = str(ch.get("created"))[:10]
        sp = nvd_split(pub)
        if sp is None:
            rep["cve/outside_window"] += 1
            continue
        ev = gb.parse_event(ch)
        ev["source"] = ch.get("sourceIdentifier") or "unknown"
        ev["cna_name"] = (names or {}).get(ev["source"])
        month = pub[:7]
        cur = cves.get(cid)
        if not ev["description"] or gb.REJECT.match(ev["description"]):
            rep["cve/rejected_or_empty_at_publication"] += 1
            continue
        if cur is None:
            rep["cve/not_in_current_nvd_window_fetch"] += 1
            cover[month]["missing_current"] += 1
            continue
        if cur.get("vulnStatus") == "Rejected":
            rep["cve/rejected_later"] += 1
            cover[month]["rejected_later"] += 1
            continue
        vals, prov = cwe_gold(cur.get("weaknesses"))
        cover[month]["published"] += 1
        cover[month][f"gold_{prov}"] += 1
        cover[month][f"status_{cur.get('vulnStatus')}"] += 1
        if vals is None:
            rep[f"gold/{prov}"] += 1
            continue
        cands.append({"cve": cid, "pub": pub, "split": sp, "ev": ev, "vals": vals, "prov": prov,
                      "vuln_status": cur.get("vulnStatus"), "cna": ev["source"], "change_id": ch.get("cveChangeId"),
                      "received": str(ch.get("created"))})
    return cands, dict(rep), {m: dict(c) for m, c in sorted(cover.items())}


def cmd_nvd_cwe(a):
    import numpy as np
    cache = Path(a.work) / "cache" / "nvd"
    nvd_fetch_all(cache)
    if a.fetch_only:
        return
    events = gb.nvd_load(cache / "history")
    cves = {}
    for p in sorted((cache / "cves").glob("cves_*.jsonl.gz")):
        for r in read_gz(p):
            cves[r["id"].upper()] = r
    names = gb.cna_names(cache / "sources.json")
    cwe_names, parents, cwe_ver = cwe_taxonomy(cache / "cwec_latest.xml.zip")
    cands, rep, coverage = nvd_candidates(events, cves, names)
    rep = Counter(rep)
    log(f"nvd_cwe: {len(events):,} New CVE Received events, {len(cves):,} current records, {len(cands):,} with gold")
    # options: the N most frequent NVD-assigned CWEs in TRAIN (NVD's own assignments only)
    freq = Counter(v for c in cands if c["split"] == "train" and c["prov"] == "nvd" for v in c["vals"] if cwe_num(v))
    top = [cwe_num(v) for v, _ in freq.most_common(N_TOP_CWE)]
    options = {f"cwe_{n}" for n in top}
    keys = [f"cwe_{n}" for n in top] + ["other"]
    criteria = {f"cwe_{n}": f"CWE-{n}: {cwe_names.get(n, 'name not in the CWE list')} (or a more specific weakness "
                            f"beneath it)" for n in top}
    criteria["other"] = ("Another weakness: any CWE not covered by the options above, including NVD's 'NVD-CWE-Other' "
                         "placeholder.")
    # states and label distributions
    items = []
    for c in cands:
        dist = cwe_dist(c["vals"], options, parents)
        if dist is None:
            rep["gold/unmappable"] += 1
            continue
        state = nvd_state(c["ev"], c["cve"], c["pub"])
        if CWE_ID_RX.search(state) or CWE_REF_RX.search(state):
            rep["leak/cwe_id_left_in_state"] += 1
            continue
        if CWE_ID_RX.search(c["ev"]["description"]):
            rep["note/description_named_a_cwe_id (masked)"] += 1
        c.update(state=state, dist=dist)
        items.append(c)
    # exact-state duplicates (templated vendor batches): within a split keep the earliest; eval drops a group whose
    # gold differs; a train / val state identical to an eval state is dropped from train / val
    by_split = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_split[it["split"]][norm(it["state"])].append(it)
    kept = []
    eval_states = set(by_split["eval"])
    for sp, groups in by_split.items():
        for k, grp in groups.items():
            grp.sort(key=lambda c: (c["pub"], c["cve"]))
            if sp != "eval" and k in eval_states:
                rep[f"dedup/{sp}_state_identical_to_eval"] += len(grp)
                continue
            if sp == "eval" and len({json.dumps(sorted(g["dist"].items())) for g in grp}) > 1:
                rep["dedup/eval_identical_state_conflicting_gold_dropped"] += len(grp)
                continue
            rep[f"dedup/{sp}_identical_state_duplicates_dropped"] += len(grp) - 1
            kept.append(grp[0])
    # caps. train and eval PREFER NVD's own assignment: every NVD-labelled CVE is kept, CNA-labelled ones fill up to
    # the cap (seeded uniform within each provenance stratum); eval records carry sample_weight = stratum size / taken,
    # so a weighted set-level score estimates the full population of eval-window CVEs with a gold CWE. val is uniform.
    chosen = []
    for sp in ("train", "val", "eval"):
        pool = sorted((c for c in kept if c["split"] == sp), key=lambda c: rank(a.seed, "nvd_cwe", c["cve"]))
        cap = CWE_CAPS[sp]
        nvd = [c for c in pool if c["prov"] == "nvd"]
        cna = [c for c in pool if c["prov"] != "nvd"]
        rep[f"pool/{sp}"] = len(pool)
        rep[f"pool/{sp}/nvd"] = len(nvd)
        if sp == "val":
            take = pool[:cap]
            for c in take:
                c["w"] = 1.0
        else:
            k_nvd = min(len(nvd), cap)
            k_cna = min(len(cna), cap - k_nvd)
            take = nvd[:k_nvd] + cna[:k_cna]
            for c in nvd[:k_nvd]:
                c["w"] = len(nvd) / k_nvd
            for c in cna[:k_cna]:
                c["w"] = len(cna) / k_cna
            rep[f"sampling/{sp}/nvd_taken"] = k_nvd
            rep[f"sampling/{sp}/cna_taken"] = k_cna
            rep[f"sampling/{sp}/cna_inclusion_prob"] = round(k_cna / len(cna), 5) if cna else None
        chosen += take
    # leave-one-out base rates: global > CNA (single-gold items count; multi-gold items get rates but are not counted)
    y = [keys.index(max(c["dist"], key=c["dist"].get)) if len(c["dist"]) == 1 else -1 for c in chosen]
    lv = Levels(["global", "assigning CNA"], lambda it, L: "|".join(["all", it["cna"]][:L + 1]))
    P, _, lrep = loo_rates(chosen, lv, y, len(keys), [c["split"] == "train" for c in chosen])
    out = {"train": [], "val": [], "eval": []}
    for i, c in enumerate(chosen):
        sp = c["split"]
        gold = np.array([c["dist"].get(k, 0.0) for k in keys])
        gold = gold / gold.sum()
        p_hat = lab_dict(keys, P[i])
        t = a.lam * gold + (1 - a.lam) * P[i]
        qs = {"cwe_class": choice_q(CWE_INSTR, dict(criteria))}
        lsrc = f"expert:{c['prov']}"
        raw = {"cwe_class": c["vals"], "mapped": dict(c["dist"])}
        common = {"cve_id": c["cve"], "published": c["pub"], "nvd_received": c["received"], "cna": c["cna"],
                  "cna_name": c["ev"].get("cna_name"), "vuln_status": c["vuln_status"], "gold_provenance": c["prov"],
                  "n_gold_cwes": len(c["vals"]), "nvd_change_id": c["change_id"], "cwe_list": cwe_ver,
                  "td_workflow": "security"}
        if sp == "eval":
            src = "nvd_cwe/nvd_analyst" if c["prov"] == "nvd" else "nvd_cwe/cna_assigned"
            m = gde1_meta("nvd_cwe", "E", {
                "label_provenance": f"expert key ({'NVD analyst' if c['prov'] == 'nvd' else 'assigning CNA'} CWE, "
                                    f"current NVD record)", "panel": "E (expert-key side panel; never in the headline)",
                "decision_date": c["pub"], "info_date": c["pub"], "label_source": {"cwe_class": lsrc},
                "raw_outcome": raw, "base_rate": {"cwe_class": p_hat},
                "smoothed_target": {"cwe_class": lab_dict(keys, t)}, "sample_weight": c["w"],
                "sampling": {"stratum": c["prov"], "inclusion_prob": round(1.0 / c["w"], 6),
                             "rule": "every NVD-labelled CVE, CNA-labelled ones sampled to the cap"}, **common})
            rec = build_rec(src, c["state"], qs, {"cwe_class": lab_dict(keys, gold)}, m, f"nvd_cwe/{c['cve']}",
                            LICENSES["nvd_cwe"], rec_id=f"gde1-nvd_cwe-{c['cve']}")
        else:
            src = "nvd_cwe/cwe_class"
            m = train_meta("NVD CVE Change History (New CVE Received) + NVD CVE API weaknesses", f"cve:{c['cve']}",
                           c["pub"], c["pub"], None, f"{sp}: NVD publication {c['pub']}", {"cwe_class": lsrc},
                           {"cwe_class": f"{a.lam}*gold (NVD's CWE, else the CNA's; mapped to options; equal split) + "
                                         f"{1 - a.lam}*leave-one-out base rate (global > assigning CNA; train-only)"},
                           raw, {"cwe_class": p_hat}, a.lam, common)
            rec = build_rec(src, c["state"], qs, {"cwe_class": lab_dict(keys, t)}, m, f"nvd_cwe/{c['cve']}",
                            LICENSES["nvd_cwe"], rec_id=f"{src}-{sha16(c['cve'])}")
        out[sp].append(rec)
    train_counts = write_corpus(Path(a.out) / "nvd_cwe", {k: out[k] for k in ("train", "val")}, "nvd_cwe__")
    eval_counts = write_corpus(Path(a.gde1) / "nvd_cwe", {"eval": out["eval"]}, "nvd_cwe__")
    hold = ("HOLD: not for the zero-shot training line until the user rules. research/realworld_data.md (sections 2 and "
            "6) put NVD / KEV / EPSS out of training as 'security triage', a forbidden family near typed-decisions' "
            "security_incidents workflow. CVE -> CWE classification is not alert triage and passes "
            "synth_workflows.exclusion's code checks (qid cwe_class), but the domain is the same; every record carries "
            "meta.td_workflow = 'security'. The GDE-1 eval copy (/tmp/gde1/nvd_cwe, panel E) is unaffected.\n")
    (Path(a.out) / "nvd_cwe" / "HOLD_SECURITY_DOMAIN.txt").write_text(hold)
    manifest = {"builder": BUILDER, "built": datetime.now().isoformat(timespec="seconds"), "seed": a.seed,
                "lambda": a.lam, "license": LICENSES["nvd_cwe"], "license_evidence": LICENSE_EVIDENCE["nvd_cwe"],
                "source": {"history": gb.NVD_HIST + " (eventName=New CVE Received)", "cves": NVD_CVES, "cwe": CWE_XML,
                           "cwe_list": cwe_ver, "events_read": len(events), "current_records": len(cves),
                           "window": [NVD_FROM, NVD_TO]},
                "options": {k: criteria[k] for k in keys}, "train_option_frequency_nvd": dict(freq.most_common(40)),
                "splits": {"train": f"NVD publication {NVD_FROM} .. {TRAIN_END} (exclusive)",
                           "val": f"{TRAIN_END} .. {VAL_END}", "eval": f"{VAL_END} .. {NVD_TO} (GDE-1 side panel E)"},
                "caps": CWE_CAPS, "build_report": dict(sorted(rep.items())), "label_model": lrep,
                "coverage_by_publication_month": coverage, "zero_shot_line": "HOLD (" + hold.strip()[:200] + " ...)"}
    (Path(a.out) / "nvd_cwe" / "manifest.json").write_text(json.dumps(
        {**manifest, "role": "TRAINING corpus (train / val) -- HOLD", "counts": train_counts}, indent=1, default=str))
    (Path(a.gde1) / "nvd_cwe").mkdir(parents=True, exist_ok=True)
    (Path(a.gde1) / "nvd_cwe" / "manifest.json").write_text(json.dumps(
        {**manifest, "set": "nvd_cwe", "domain": "E", "eval_only": True, "counts": eval_counts,
         "splits_stats": {"eval": gb.split_stats(out["eval"])},
         "dedup_vs_training": {"status": "NOT RUN: run `build_jev_rawsrc2.py gate`"}}, indent=1, default=str))
    log(f"nvd_cwe: train {train_counts.get('train')}, val {train_counts.get('val')}, eval {eval_counts.get('eval')}")


# =========================================================================== Supreme Court Database (eval only: NC)
SCDB_RELEASE = "SCDB_2026_01"
SCDB_FILES = {"csv": ("SCDB_2026_01_caseCentered_Citation.csv.zip",
                      "https://scdb.la.psu.edu/?jet_download=69e2a3e7da8f9782e65de282ff4742e4116bae7e"),
              "dta": ("SCDB_2026_01_caseCentered_Citation.dta.zip",
                      "https://scdb.la.psu.edu/?jet_download=fcbe51fd96c53811fafbc3e3b92d08876d06ac3f")}
SCDB_KEYS = ["affirmed", "reversed", "vacated_remanded", "other"]
SCDB_OF = {"2": "affirmed", "3": "reversed", "4": "reversed", "5": "vacated_remanded", "8": "vacated_remanded"}
SCDB_CRITERIA = {
    "affirmed": "Affirmed: the Court upholds the judgment under review (including affirmed as modified).",
    "reversed": "Reversed: the Court reverses the judgment under review, with or without sending the case back.",
    "vacated_remanded": "Vacated: the Court vacates the judgment under review, usually sending the case back for "
                        "further proceedings.",
    "other": "Any other disposition: affirmed in part and reversed or vacated in part, the petition or appeal "
             "dismissed (including as improvidently granted), a stay or motion granted, a certification, or no "
             "disposition.",
}
SCDB_INSTR = "How will the U.S. Supreme Court dispose of the judgment under review in this case?"
SCDB_DEV_TERMS = (2015, 2024)
# fields never read into a state: everything the decision produced or that the SCDB codes from the opinion
SCDB_NEVER = ("caseDisposition", "caseDispositionUnusual", "partyWinning", "precedentAlteration", "declarationUncon",
              "decisionDirection", "decisionDirectionDissent", "authorityDecision1", "authorityDecision2", "lawType",
              "lawSupp", "lawMinor", "majOpinWriter", "majOpinAssigner", "splitVote", "majVotes", "minVotes",
              "voteUnclear", "decisionType", "dateDecision", "usCite", "sctCite", "ledCite", "lexisCite", "certReason",
              "lcDisagreement", "lcDispositionDirection", "issue")
SCDB_OUTCOME_RX = re.compile(r"\b(?:the (?:supreme )?court (?:affirm|revers|vacat|remand)\w*|per curiam|dissent\w*|"
                             r"majority opinion|concurr\w*|\d-\d (?:vote|decision))\b", re.I)


def mdy(s):
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(s or "").strip())
    return date(int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else None


def scdb_fetch(cache):
    import csv
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for _, (fn, url) in SCDB_FILES.items():
        if not (cache / fn).exists():
            (cache / fn).write_bytes(polite_get(url, 2.0, accept="application/zip"))
    vl = cache / "value_labels.json"
    if not vl.exists():
        import pandas as pd   # only to read the Stata value labels once
        z = zipfile.ZipFile(cache / SCDB_FILES["dta"][0])
        with pd.io.stata.StataReader(io.BytesIO(z.read(z.namelist()[0]))) as r:
            labs, lbl, names = r.value_labels(), r._lbllist, r._varlist
        vl.write_text(json.dumps({"var_to_label": {n: l for n, l in zip(names, lbl) if l},
                                  "labels": {k: {str(int(c)): v for c, v in d.items()} for k, d in labs.items()}}))
    z = zipfile.ZipFile(cache / SCDB_FILES["csv"][0])
    rows = list(csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode("utf-8"))))
    return rows, json.loads(vl.read_text())


def scdb_state(r, vl):
    def lab(var):
        v = str(r.get(var) or "").strip()
        if not v:
            return None
        ln = vl["var_to_label"].get(var)
        return (vl["labels"].get(ln) or {}).get(v, f"code {v}")
    term = int(r["term"])
    arg = [x for x in (mdy(r.get("dateArgument")), mdy(r.get("dateRearg"))) if x]
    lines = ["SUPREME COURT CASE (information available before the decision)",
             f"Term: October Term {term} (Chief Justice {r.get('chief')})",
             f"Case: {r.get('caseName')}" + (f" (docket {r['docket']})" if r.get("docket") else ""),
             f"Route to the Court: {lab('jurisdiction') or 'not recorded'}"]
    for side in ("petitioner", "respondent"):
        s = lab(side + "State")
        lines.append(f"{side.capitalize()}: {lab(side) or 'not recorded'}" + (f" (state: {s})" if s else ""))
    src, srcs = lab("caseSource"), lab("caseSourceState")
    org, orgs = lab("caseOrigin"), lab("caseOriginState")
    lines.append(f"Court whose decision is under review: {src or 'not recorded'}" + (f" ({srcs})" if srcs else ""))
    if org:
        lines.append(f"Court where the case began: {org}" + (f" ({orgs})" if orgs else ""))
    lc_ = lab("lcDisposition")
    if lc_:
        lines.append(f"What the court below did: {lc_}")
    adm = lab("adminAction")
    if adm:
        a_s = lab("adminActionState")
        lines.append(f"Agency action involved before the case reached the courts: {adm}" + (f" ({a_s})" if a_s else ""))
    if str(r.get("threeJudgeFdc")) == "1":
        lines.append("Heard below by a three-judge district court: yes")
    ia = lab("issueArea")
    if ia:
        lines.append(f"Issue area (SCDB coding): {ia}")
    lines.append("Oral argument: " + (", ".join(x.isoformat() for x in arg) if arg else
                                      "none recorded (decided without oral argument)"))
    return "\n".join(lines), arg


def cmd_scdb(a):
    cache = Path(a.work) / "cache" / "scdb"
    rows, vl = scdb_fetch(cache)
    rep = Counter()
    out = {"eval": [], "dev": []}
    for r in rows:
        term = int(r["term"])
        decided = mdy(r.get("dateDecision"))
        if decided is None:
            rep["drop/no_decision_date"] += 1
            continue
        if decided >= date.fromisoformat(EVAL_FROM):
            sp = "eval"
        elif SCDB_DEV_TERMS[0] <= term <= SCDB_DEV_TERMS[1]:
            sp = "dev"
        else:
            rep["skip/outside_terms"] += 1
            continue
        state, arg = scdb_state(r, vl)
        if SCDB_OUTCOME_RX.search(state):
            rep["drop/outcome_wording_in_state"] += 1
            continue
        k = SCDB_OF.get(str(r.get("caseDisposition") or "").strip(), "other")
        decision = max(arg).isoformat() if arg else f"{term}-10-01"
        if decision > decided.isoformat():
            decision = f"{term}-10-01"
        meta = gde1_meta("scdb", "O", {
            "label_provenance": "outcome (SCDB caseDisposition, the Court's disposition of the judgment under review)",
            "decision_date": decision, "info_date": decision,
            "resolution_date": {"case_disposition": decided.isoformat()},
            "raw_outcome": {"case_disposition": k, "caseDisposition": r.get("caseDisposition")},
            "scdb_case_id": r["caseId"], "term": term, "scdb_release": SCDB_RELEASE, "do_not_train": True,
            "time_slice": (f"post_{EVAL_FROM}" if sp == "eval" else
                           f"dev: PRE-CUTOFF terms {SCDB_DEV_TERMS[0]}-{SCDB_DEV_TERMS[1]} (knowledge-contaminated; "
                           "probe / calibration only)"),
            "td_workflow": None})
        rec = build_rec("scdb/case_disposition", state,
                        {"case_disposition": choice_q(SCDB_INSTR, dict(SCDB_CRITERIA))},
                        {"case_disposition": onehot(SCDB_KEYS, k)}, meta, f"scdb/{r['caseId']}", LICENSES["scdb"],
                        rec_id=f"gde1-scdb-{r['caseId']}")
        out[sp].append(rec)
        rep[f"{sp}/{k}"] += 1
    for sp in list(out):
        out[sp], nd = gb.dedup_within(out[sp])
        rep[f"{sp}/within_set_identical_states_dropped"] = nd
    counts = write_corpus(Path(a.gde1) / "scdb", out, "scdb__")
    (Path(a.gde1) / "scdb" / "DO_NOT_TRAIN.txt").write_text(
        "Supreme Court Database (SCDB_2026_01) derived eval files. Licence: CC BY-NC 3.0 US (the last licence the SCDB "
        "published); NON-COMMERCIAL. Never place these files, or anything derived from them, under a training "
        "directory. dev/ is PRE-CUTOFF (terms 2015-2024): knowledge-contaminated, for probes and calibration only.\n")
    manifest = {"set": "scdb", "domain": "O", "builder": BUILDER, "built": datetime.now().isoformat(timespec="seconds"),
                "license": LICENSES["scdb"], "license_evidence": LICENSE_EVIDENCE["scdb"], "eval_only": True,
                "do_not_train": True,
                "source": {"release": SCDB_RELEASE, "release_date": "2026-09-07", "terms": "1946-2025",
                           "files": {k: v[1] for k, v in SCDB_FILES.items()}, "rows": len(rows)},
                "splits": {"eval": f"decided on / after {EVAL_FROM} (OT2025)",
                           "dev": f"terms {SCDB_DEV_TERMS[0]}-{SCDB_DEV_TERMS[1]} decided before {EVAL_FROM} -- "
                                  "PRE-CUTOFF, contaminated; not a headline"},
                "label": "caseDisposition 2 -> affirmed; 3, 4 -> reversed; 5, 8 -> vacated_remanded; else other",
                "state_fields": "term, chief, case name, docket, jurisdiction, petitioner / respondent (+ state), "
                                "source and origin courts (+ state), lcDisposition, adminAction (+ state), "
                                "threeJudgeFdc, issueArea, argument / reargument dates",
                "never_in_state": list(SCDB_NEVER),
                "caveat_issue_area": "SCDB assigns issue / issueArea after reading the opinion; the broad area is "
                                     "knowable from the petition, the specific `issue` code is not used",
                "not_available": "no petition text, questions presented, briefs or opinion text are in the SCDB; "
                                 "joining opinion text would leak the disposition",
                "build_report": dict(sorted(rep.items())), "counts": counts,
                "splits_stats": {sp: gb.split_stats(v) for sp, v in out.items()}}
    (Path(a.gde1) / "scdb" / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"scdb: {counts}")


# =========================================================================== chufangao/CTO probe (nothing written to corpora)
CTO_REPO, CTO_REV = "chufangao/CTO", "6b080abb5a510e8a6573085f4b843e08ac3b6204"
CTO_FILE = "human_labels_2020_2024/human_labels_2020_2024.csv"


def cmd_cto(a):
    import csv
    csv.field_size_limit(1 << 30)
    cache = Path(a.work) / "cache" / "cto"
    cache.mkdir(parents=True, exist_ok=True)
    for f in ("README.md", CTO_FILE):
        dest = cache / Path(f).name
        if not dest.exists():
            dest.write_bytes(polite_get(f"https://huggingface.co/datasets/{CTO_REPO}/resolve/{CTO_REV}/{f}", 2.0,
                                        accept="*/*"))
    rows = list(csv.DictReader(open(cache / Path(CTO_FILE).name, encoding="utf-8")))
    by = Counter((r["overall_status"], r["labels"]) for r in rows)
    comp = [r for r in rows if r["overall_status"] == "COMPLETED"]
    rep = {"rows": len(rows), "status_x_label": {f"{s}/{l}": n for (s, l), n in sorted(by.items())},
           "completion_year": dict(Counter(r["completion_year"] for r in rows)),
           "phase": dict(Counter(r["phase"] for r in rows)),
           "completed_rows": len(comp),
           "completed_without_posted_results": sum(1 for r in comp if not r["results_first_posted_date"]),
           "rule_labelled_non_completed": sum(1 for r in rows if r["overall_status"] != "COMPLETED"),
           "per_row_human_flag": "none: the file does not mark which rows the three experts labelled",
           "licence": {"hf_card_yaml": "license: mit (huggingface.co/datasets/chufangao/CTO @ " + CTO_REV + ")",
                       "github_chufangao_CTO": "empty repository (size 0, no LICENSE); its description points to "
                                               "github.com/chufangao/CTOD",
                       "github_chufangao_CTOD": "LICENSE: 'MIT License / Copyright (c) 2025 Chufan Gao' (GitHub API "
                                                "licence: mit)",
                       "upstream": "CTTI AACT copy of ClinicalTrials.gov (CTTI.zip); ClinicalTrials.gov terms apply"},
           "paper": "arXiv 2406.10292 (HTML): 'we manually annotated over 2,500 challenging cases ... We applied the "
                    "same rule-based termination criteria and p-value threshold of less than 0.05 as used in the "
                    "automated labeling step to filter out the easier cases. For the remaining unlabeled trials, we "
                    "engaged three clinical trial experts'"}
    ours = {}
    for sp in ("train", "val"):
        for r in read_dir(Path(a.out) / "ctgov", (sp,)).get(sp, []):
            ours[r["meta"]["nct_id"]] = sp
    for r in read_dir(Path(a.gde1) / "ctgov", ("eval",)).get("eval", []):
        ours[r["meta"]["nct_id"]] = "eval"
    if ours:
        rep["overlap_with_ctgov_build"] = dict(Counter(ours.get(r["nct_id"], "not_selected") for r in rows))
    (Path(a.work) / "cto_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# =========================================================================== gate: ext2 dedup, both directions
OWN_SETS = ("ctgov", "nvd_cwe", "scdb")
NCT_ANY = re.compile(r"\bNCT\d{8}\b")
CVE_ANY = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
SCDB_TEMPLATE = re.compile(r"^(?!Case: ).*$")


def content_view(rec):
    """The SOURCE text of a state without this builder's (or part B's) template lines, for dedup (part B's rule: our
    own boilerplate must never make two states look alike)."""
    m = rec.get("meta") or {}
    src = str(rec.get("source") or "")
    gset = m.get("gde1_set")
    if gset in ("kev", "forecastbench"):
        return gb.content_view(rec)
    lines = str(rec["state"]).split("\n")
    if src.startswith("nvd_cwe/") or gset == "nvd_cwe":
        keep = [ln[2:] if ln.startswith("- ") else ln for ln in lines if ln.strip() and not gb.KEV_TEMPLATE.match(ln)]
    elif src.startswith("ctgov/") or gset == "ctgov":
        keep = [re.sub(r"^(Title|Official title|Lead sponsor): ", "", ln) for ln in lines
                if ln.strip() and not CTG_TEMPLATE.match(ln)]
    elif src.startswith("scdb/") or gset == "scdb":
        keep = [ln[len("Case: "):] for ln in lines if ln.startswith("Case: ")]
    else:
        return rec["state"]
    return "\n".join(keep)


def content_probes(content):
    return [ln.strip() for ln in content.split("\n") if len(ln.split()) >= 5]


SHINGLE_RULES = {
    # build_jev_ext2: >= 3 shared 8-word shingles with ONE fingerprint state
    "ext2": {"max_df": None, "min_shared": 3, "min_contain": 0.0},
    # build_jev_openreview / build_gde1_a constants: >= 5 shared shingles covering >= 10% of the shorter text, shingles
    # held by > 20 fingerprint states ignored (stock phrases). Registry texts (eligibility boilerplate) need it: the
    # plain ext2 rule dropped 36% of the CT.gov train split on phrases like "the purpose of this study is to determine"
    "calibrated": {"max_df": 20, "min_shared": 5, "min_contain": 0.10},
}


def make_gate(rule="ext2"):
    """build_jev_ext2.Index with its exact and leaf rules unchanged and the shingle rule chosen by `rule`. Under
    'calibrated', a match the plain ext2 rule would have made is kept in .plain_hits (reported, not dropped)."""
    import numpy as np
    import build_jev_ext2 as x2
    cfg = SHINGLE_RULES[rule]

    class Gate(x2.Index):
        plain_hit = None

        def freeze(self):
            super().freeze()
            self.NS = np.bincount(self.S, minlength=len(self.ST)) if len(self.S) else np.zeros(len(self.ST), int)

        def _shingle(self, nf):
            self.plain_hit = None
            if rule == "ext2":
                return super()._shingle(nf)
            plain = super()._shingle(nf)
            if plain is None:
                return None
            hs = x2.shingle_hashes(nf)
            keys = np.fromiter(hs, dtype=np.int64, count=len(hs))
            lo, hi = np.searchsorted(self.H, keys, "left"), np.searchsorted(self.H, keys, "right")
            nz = np.nonzero((hi > lo) & (hi - lo <= cfg["max_df"]))[0]
            if len(nz):
                ids = np.concatenate([self.S[lo[i]:hi[i]] for i in nz])
                u, c = np.unique(ids, return_counts=True)
                need = np.maximum(cfg["min_shared"], cfg["min_contain"] * np.minimum(self.NS[u], len(hs)))
                sel = c >= need
                if sel.any():
                    sid, n = min(zip(u[sel].tolist(), c[sel].tolist()),
                                 key=lambda x: (self.klass[self.ST[x[0]]] != "eval", -x[1]))
                    shared = [hs[int(keys[i])] for i in nz if (self.S[lo[i]:hi[i]] == sid).any()][:2]
                    return ("shingle", int(self.ST[sid]), n, shared)
            self.plain_hit = plain
            return None
    return Gate()


def gde1_set_dirs(root):
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and any((d / sp).is_dir() for sp in ("eval", "dev")):
            yield d


def cmd_gate(a):
    from types import SimpleNamespace
    import build_jev_ext2 as x2
    t0 = time.time()
    # ---------------------------------------------------------------- 1. eval fingerprints
    idx = make_gate(a.shingle_rule)
    ids_in_eval = {"nct": defaultdict(set), "cve": defaultdict(set)}
    coverage = {}

    def note_ids(tag, st):
        txt = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
        for m in NCT_ANY.findall(txt):
            ids_in_eval["nct"][m.upper()].add(tag)
        for m in CVE_ANY.findall(txt):
            ids_in_eval["cve"][m.upper()].add(tag)
    for d in gde1_set_dirs(a.gde1):
        tid = idx.tag(f"gde1:{d.name}", "eval")
        n = 0
        for sp in ("eval", "dev"):
            for p in sorted((d / sp).glob("*.jsonl")):
                for line in p.open(encoding="utf-8"):
                    r = json.loads(line)
                    n += bool(idx.add(tid, content_view(r)))
                    note_ids(f"gde1:{d.name}", r["state"])
                    for k in ("nct_id", "cve_id"):
                        if (r.get("meta") or {}).get(k):
                            ids_in_eval["nct" if k == "nct_id" else "cve"][r["meta"][k].upper()].add(f"gde1:{d.name}")
        coverage[f"gde1:{d.name}"] = n
    for spec in a.eval:
        tag, _, p = spec.partition("=")
        n = idx.add_rows(f"given:{tag}", "eval", x2.jsonl_states(p), p)
        for st in x2.jsonl_states(p):
            note_ids(f"given:{tag}", st)
        coverage[f"given:{tag}"] = n
    if a.fetch_evals:
        xa = SimpleNamespace(hf_token=None, cache=str(Path(a.work) / "ext2_url_cache"))
        for fam, fn in x2.EVAL_FETCHERS.items():
            per = Counter()
            try:
                for tag, st in fn(xa):
                    per[tag] += bool(idx.add(idx.tag(f"eval:{tag}", "eval"), st))
                    note_ids(f"eval:{tag}", st)
                coverage[fam] = dict(per)
                log(f"gate: eval {fam}: {sum(per.values())} states")
            except Exception as e:  # noqa: BLE001
                coverage[fam] = f"FAILED {type(e).__name__}: {str(e)[:300]}"
                log(f"gate: eval {fam} FAILED: {e}")
    missing = [f for f, v in coverage.items() if isinstance(v, str)]
    if missing and not a.allow_missing_evals:
        raise SystemExit(f"gate: eval families not covered: {missing} (pass --allow-missing-evals to accept)")
    idx.freeze()
    # ---------------------------------------------------------------- 2. training corpora vs eval
    train_report = {}
    for src_dir in sorted(p for p in Path(a.out).iterdir() if p.is_dir()):
        mpath = src_dir / "manifest.json"
        res = {"by_file": {}, "fingerprint_states": dict(idx.sizes), "coverage": coverage,
               "shingle_rule": {a.shingle_rule: SHINGLE_RULES[a.shingle_rule]},
               "when": datetime.now().isoformat(timespec="seconds"), "dry_run": bool(a.dry_run)}
        for sp in ("train", "val"):
            for f in sorted((src_dir / sp).glob("*.jsonl")):
                recs = [json.loads(l) for l in f.open(encoding="utf-8")]
                kept, drops, ex, plain = [], Counter(), [], Counter()
                for r in recs:
                    m = r.get("meta") or {}
                    hit = None
                    for k, kind in (("nct_id", "nct"), ("cve_id", "cve")):
                        v = str(m.get(k) or "").upper()
                        if v and v in ids_in_eval[kind]:
                            hit = ("id", sorted(ids_in_eval[kind][v])[0], None, [v])
                    if hit is None:
                        c = content_view(r)
                        hit = idx.check(c, content_probes(c))
                        if hit is None and idx.plain_hit:
                            plain[f"ext2_rule_only:{idx.names[idx.plain_hit[1]]}"] += 1
                    if hit:
                        drops[f"{hit[0]}:{hit[1]}"] += 1
                        if len(ex) < 6:
                            ex.append({"id": r["id"], "kind": hit[0], "tag": hit[1], "n_shared": hit[2],
                                       "sample": hit[3]})
                        continue
                    kept.append(r)
                res["by_file"][f"{sp}/{f.name}"] = {"before": len(recs), "after": len(kept), "dropped": dict(drops),
                                                    "reported_not_dropped": dict(plain), "examples": ex}
                if not a.dry_run:
                    tmp = f.with_suffix(".jsonl.part")
                    with tmp.open("w", encoding="utf-8") as fh:
                        for r in kept:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    tmp.replace(f)
        if mpath.exists() and not a.dry_run:
            man = json.loads(mpath.read_text())
            man.setdefault("dedup_vs_eval_runs", []).append(res)
            man["dedup_vs_eval"] = {"status": f"run (ext2 gate, shingle rule {a.shingle_rule}" +
                                    "); training corpora jev_data / jev_ext_strict / jev_ext2 / jev_ext3 not on the "
                                    "laptop: corpus-side check pending on a node" if not a.fingerprint else "run",
                                    "dropped": sum(sum(v["dropped"].values()) for v in res["by_file"].values())}
            mpath.write_text(json.dumps(man, indent=1, default=str))
        train_report[src_dir.name] = {k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                                      for k, v in res["by_file"].items()}
        for k, v in res["by_file"].items():
            if v["examples"]:
                train_report[src_dir.name][k]["examples"] = v["examples"][:3]
    # ---------------------------------------------------------------- 3. new eval sets vs training corpora
    # this build's own train / val were already separated from these eval sets in step 2 (and by construction: NCT /
    # CVE ids, time); indexing them again would only re-fire the asymmetric stock-phrase filter and thin the eval
    # sets, so step 3 checks the OTHER training corpora (--fingerprint)
    tidx = make_gate(a.shingle_rule)
    tsizes = {}
    for spec in a.fingerprint:
        tag, _, p = spec.partition("=")
        tsizes[tag] = tidx.add_rows(f"corpus:{tag}", "corpus", gb.iter_fingerprint_states(p), p)
        log(f"gate: training fingerprint {tag}: {tsizes[tag]} states")
    tidx.freeze()
    eval_report = {}
    for gset in OWN_SETS:
        root = Path(a.gde1) / gset
        if not root.exists():
            continue
        res = {"fingerprints": tsizes, "shingle_rule": {a.shingle_rule: SHINGLE_RULES[a.shingle_rule]}, "by_file": {}}
        for sp in ("eval", "dev"):
            for f in sorted((root / sp).glob("*.jsonl")):
                recs = [json.loads(l) for l in f.open(encoding="utf-8")]
                kept, drops, ex, plain = [], Counter(), [], Counter()
                for r in recs:
                    c = content_view(r)
                    hit = tidx.check(c, content_probes(c))
                    if hit is None and tidx.plain_hit:
                        plain[f"ext2_rule_only:{tidx.names[tidx.plain_hit[1]]}"] += 1
                    if hit:
                        drops[f"{hit[0]}:{hit[1]}"] += 1
                        if len(ex) < 4:
                            ex.append({"id": r["id"], "kind": hit[0], "tag": hit[1], "n_shared": hit[2],
                                       "sample": hit[3]})
                        continue
                    kept.append(r)
                res["by_file"][f"{sp}/{f.name}"] = {"before": len(recs), "after": len(kept), "dropped": dict(drops),
                                                    "reported_not_dropped": dict(plain), "examples": ex}
                if not a.dry_run:
                    tmp = f.with_suffix(".jsonl.part")
                    with tmp.open("w", encoding="utf-8") as fh:
                        for r in kept:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    tmp.replace(f)
        mpath = root / "manifest.json"
        if mpath.exists() and not a.dry_run:
            man = json.loads(mpath.read_text())
            man.setdefault("dedup_vs_training_runs", []).append(res)
            man["dedup_vs_training"] = {
                "status": ("PROXY ONLY: against the laptop's raw-source training corpora " + str(sorted(tsizes)) +
                           "; the authoritative run against jev_data / jev_ext_strict / jev_ext2 / jev_ext3 is pending "
                           "on a node (`build_jev_rawsrc2.py gate --fingerprint ...`)"),
                "dropped": sum(sum(v["dropped"].values()) for v in res["by_file"].values())}
            mpath.write_text(json.dumps(man, indent=1, default=str))
        eval_report[gset] = res["by_file"]
    out = {"train_vs_eval": train_report, "eval_vs_training": eval_report, "eval_coverage": coverage,
           "ids_in_eval_fingerprints": {k: len(v) for k, v in ids_in_eval.items()},
           "seconds": round(time.time() - t0, 1)}
    (Path(a.work) / f"gate_report{'_dry' if a.dry_run else ''}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:20000])


# =========================================================================== check
SCANS = {"ctgov": STATUS_RX, "nvd_cwe": CWE_ID_RX, "scdb": SCDB_OUTCOME_RX}
NEVER_FIELDS_RX = {
    "ctgov": re.compile(r"overall ?status|why stopped|status verified|enrol+ment: \d|actual enrol+ment|"
                        r"(?:primary )?completion date|results first|last update", re.I),
    "nvd_cwe": re.compile(r"Weakness \(CWE\)|cwe\.mitre\.org|vulnStatus|NVD analysis", re.I),
    "scdb": re.compile(r"decided on|date of decision|majority|minority votes|\bwinning\b|precedent", re.I),
}
LEAK_QS = {"ctgov": [("final_status", FS_KEYS), ("results_posted_12m", ["yes", "no"])]}


def readback_dir(root, splits):
    cf = gb.load_cf()
    units, per = Counter(), Counter()
    lens = []
    for u in cf.iter_units(str(root), list(splits), set(), None):
        assert abs(sum(u["target"]) - 1) < 1e-6 and 2 <= len(u["target"]) <= 52, u["record_id"]
        units[f"{u['split']}/{u['qtype']}/{u['label_kind']}"] += 1
        per[f"{u['split']}/{u['source']}"] += 1
        lens.append(len(u["state"]))
        nc = bool(cf.NC_LICENCE.search(u["license"] or ""))
        per[f"{u['split']}/nc_licence={nc}"] += 1
    lens.sort()
    return {"iter_units": dict(units), "units_by_source": dict(per), "total_units": sum(units.values()),
            "state_chars": ({"median": lens[len(lens) // 2], "p95": lens[int(len(lens) * 0.95)], "max": lens[-1],
                             "over_12000": sum(1 for x in lens if x > 12000)} if lens else None)}


def leak_probe_cwe(recs, n=6000, seed=0):
    """build_jev_fda.leak_probe on single-gold CWE records (raw outcome = the mapped option)."""
    from build_jev_fda import leak_probe
    keys = option_keys(recs[0]["questions"]["cwe_class"])
    rows = []
    for r in recs:
        mp = r["meta"]["raw_outcome"].get("mapped") or {}
        if len(mp) != 1:
            continue
        rows.append({"state": r["state"], "meta": {"raw_outcome": {"cwe_class": next(iter(mp))},
                                                   "base_rate": r["meta"]["base_rate"]}})
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    return leak_probe(rows, "cwe_class", keys)


def cmd_check(a):
    from build_jev_fda import leak_probe, run_exclusion
    report = {}
    for src in ("ctgov", "nvd_cwe"):
        root = Path(a.out) / src
        if not root.exists():
            continue
        recs = read_dir(root, ("train", "val"))
        rep = {"readback": readback_dir(root, ("train", "val")), "label_stats": label_stats(recs)}
        for v in recs.values():
            for r in v:
                validate(r)
        hits = Counter()
        for v in recs.values():
            for r in v:
                for rx in (SCANS[src], NEVER_FIELDS_RX[src]):
                    for m in rx.finditer(r["state"]):
                        hits[m.group(0).lower()] += 1
        rep["outcome_word_scan"] = dict(hits) or "0 hits"
        ids = [r["id"] for v in recs.values() for r in v]
        groups = {sp: {r["base_id"] for r in v} for sp, v in recs.items()}
        ev = read_dir(Path(a.gde1) / src, ("eval",)).get("eval", [])
        groups["eval"] = {r["base_id"] for r in ev}
        st = {sp: {norm(r["state"]) for r in v} for sp, v in recs.items()}
        st["eval"] = {norm(r["state"]) for r in ev}
        rep["separation"] = {"duplicate_ids": len(ids) - len(set(ids)),
                             "groups_train_val": len(groups.get("train", set()) & groups.get("val", set())),
                             "groups_train_eval": len(groups.get("train", set()) & groups["eval"]),
                             "groups_val_eval": len(groups.get("val", set()) & groups["eval"]),
                             "identical_states_train_eval": len(st.get("train", set()) & st["eval"]),
                             "identical_states_val_eval": len(st.get("val", set()) & st["eval"])}
        rep["exclusion_errors"] = run_exclusion(f"rawsrc2_{src}", {
            src: sorted({q for v in recs.values() for r in v for q in r["questions"]})})
        train = recs.get("train", [])
        if train and not a.no_probe:
            if src == "ctgov":
                for qid, keys in LEAK_QS["ctgov"]:
                    sub = [r for r in train if qid in r["questions"]]
                    sub = random.Random(0).sample(sub, min(a.probe_n, len(sub)))
                    if len(sub) > 200:
                        log(f"check: leak probe {src}/{qid} on {len(sub)} records")
                        rep[f"leak_probe_{qid}"] = leak_probe(sub, qid, keys)
            else:
                log(f"check: leak probe {src}/cwe_class")
                rep["leak_probe_cwe_class"] = leak_probe_cwe(train, n=a.probe_n)
        report[f"train/{src}"] = rep
        samples = []
        for sp in ("train", "val"):
            for q in sorted({next(iter(r["questions"])) for r in recs.get(sp, [])}):
                pool = [r for r in recs[sp] if q in r["questions"]]
                if pool:
                    s = random.Random(len(samples)).choice(pool)
                    samples.append({**s, "state": s["state"][:1800] + (" ..." if len(s["state"]) > 1800 else "")})
        (root / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
        (root / "check_report.json").write_text(json.dumps(rep, indent=1, default=str))
    for gset in OWN_SETS:
        root = Path(a.gde1) / gset
        if not root.exists():
            continue
        recs = read_dir(root, ("eval", "dev"))
        bad = Counter()
        for sp, v in recs.items():
            for r in v:
                validate(r)
                m = r["meta"]
                if not r.get("license"):
                    bad["missing_license"] += 1
                if m.get("gde1_set") != gset:
                    bad["bad_gde1_set"] += 1
                if m.get("domain") not in ({"nvd_cwe": ("E",)}.get(gset, ("O",))):
                    bad["bad_domain"] += 1
                if not m.get("decision_date") or not m.get("info_date"):
                    bad["missing_dates"] += 1
                if str(m.get("info_date")) > str(m.get("decision_date")):
                    bad["info_after_decision"] += 1
                if m.get("domain") == "O":
                    rd = m.get("resolution_date") or {}
                    if set(rd) != set(r["questions"]):
                        bad["missing_resolution_date"] += 1
                    elif any(x < m["decision_date"] for x in rd.values()):
                        bad["resolution_before_decision"] += 1
                    elif sp == "eval" and any(x < EVAL_FROM for x in rd.values()):
                        bad["eval_resolution_before_eval_from"] += 1
        hits = Counter()
        for v in recs.values():
            for r in v:
                for rx in (SCANS[gset], NEVER_FIELDS_RX[gset]):
                    for mm in rx.finditer(r["state"]):
                        hits[mm.group(0).lower()] += 1
        ids = [r["id"] for v in recs.values() for r in v]
        rep = {"readback": readback_dir(root, sorted(recs)), "schema_problems": dict(bad) or "none",
               "duplicate_ids": len(ids) - len(set(ids)), "outcome_word_scan": dict(hits) or "0 hits",
               "stats": {sp: gb.split_stats(v) for sp, v in recs.items()}, "label_stats": label_stats(recs)}
        report[f"gde1/{gset}"] = rep
        samples = []
        for v in recs.values():
            for src in sorted({r["source"] for r in v}):
                pool = [r for r in v if r["source"] == src]
                s = pool[len(pool) // 3]
                samples.append({**s, "state": s["state"][:1800] + (" ..." if len(s["state"]) > 1800 else "")})
        (root / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
        (root / "check_report.json").write_text(json.dumps(rep, indent=1, default=str))
    (Path(a.work) / "check_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("label_stats", "stats")}
                      for k, v in report.items()}, indent=1, default=str)[:30000])


# =========================================================================== self-test (offline fixtures)
def self_test():
    import tempfile
    end = date(2026, 10, 1)
    assert dlo("2024-12") == date(2024, 12, 1) and dhi("2024-12") == date(2024, 12, 31) and dhi("2024-02") == date(2024, 2, 29)
    assert split_of_day(date(2024, 12, 31), end) == "train" and split_of_day(date(2025, 1, 1), end) == "val"
    assert split_of_day(date(2025, 7, 1), end) == "eval" and split_of_day(date(2026, 10, 2), end) is None

    def row(nct, st, comp=(None, None), pc=(None, None), fp="2018-01-10", ver="2024-06", lu="2024-06-30", rfp=None):
        return {"nct": nct, "status": st, "verified": ver, "start": [None, None], "pc": list(pc), "comp": list(comp),
                "first_submit": fp, "first_post": fp, "results_first_submit": None, "results_first_post": rfp,
                "last_update_post": lu, "has_results": bool(rfp)}
    r1 = row("NCT00000001", "COMPLETED", comp=("2023-05-01", "ACTUAL"), pc=("2023-03-15", "ACTUAL"), rfp="2023-12-01")
    assert ctg_final_status_resolution(r1, end)[:3] == ("train", date(2023, 5, 1), False)
    assert ctg_results_resolution(r1, end)[:3] == ("train", date(2024, 3, 14), True)
    r2 = row("NCT00000002", "COMPLETED", comp=("2024-12", "ACTUAL"), pc=("2024-11", "ACTUAL"))
    assert ctg_final_status_resolution(r2, end)[0] == "train"        # month precision inside train
    r3 = row("NCT00000003", "TERMINATED", comp=("2025-08-20", "ACTUAL"), pc=("2023-06-01", "ACTUAL"))
    assert ctg_final_status_resolution(r3, end)[0] is None, "eval window but registered before the cutoff"
    assert ctg_results_resolution(r3, end)[0] == "train"
    r3b = row("NCT00000033", "TERMINATED", comp=("2026-02-20", "ACTUAL"), pc=("2026-01-10", "ACTUAL"), fp="2025-07-20")
    assert ctg_final_status_resolution(r3b, end)[:3] == ("eval", date(2026, 2, 20), False)
    r4 = row("NCT00000004", "COMPLETED", comp=("2019-01-01", "ACTUAL"), fp="2020-02-02")
    assert ctg_final_status_resolution(r4, end)[0] is None, "retrospective registration"
    r5 = row("NCT00000005", "WITHDRAWN", ver="2024-03", lu="2025-02-01")
    assert ctg_final_status_resolution(r5, end)[0] is None, "halted train candidate updated after the train cut"
    r6 = row("NCT00000006", "WITHDRAWN", ver="2025-08", lu="2025-08-20", fp="2025-07-02")
    assert ctg_final_status_resolution(r6, end)[:3] == ("eval", date(2025, 8, 1), True)
    r7 = row("NCT00000007", "COMPLETED", comp=("2025-09-30", "ACTUAL"), pc=("2025-08-01", "ACTUAL"), rfp="2026-07-01")
    assert ctg_results_resolution(r7, end)[:3] == ("eval", date(2026, 8, 1), True)
    r8 = row("NCT00000008", "COMPLETED", comp=("2024-10-01", "ACTUAL"), pc=("2024-09-01", "ACTUAL"))
    assert ctg_results_resolution(r8, end)[0] is None, "deadline in the val/eval gap"
    r9 = row("NCT00000009", "COMPLETED", comp=("2026-09-01", "ACTUAL"), pc=("2026-01-01", "ACTUAL"))
    assert ctg_results_resolution(r9, end)[0] is None, "deadline after the data date"
    r10 = row("NCT00000010", "COMPLETED", comp=("2026-03-01", "ACTUAL"), pc=("2025-08-01", "ACTUAL"), fp="2025-07-03",
              rfp="2025-06-01")
    assert ctg_results_resolution(r10, end)[0] is None, "results posted before primary completion: inconsistent"
    r11 = row("NCT00000011", "COMPLETED", comp=("2024-12-01", "ACTUAL"), pc=("2024-02-01", "ACTUAL"), fp="2023-01-05")
    cand, picked, rep = ctg_select([r1, r2, r3, r3b, r4, r5, r6, r7, r8, r9, r10, r11], end,
                                   caps={k: 10 for k in CTG_CAPS}, seed=1)
    assert cand["NCT00000003"][0] == "train" and set(cand["NCT00000003"][2]) == {"results_posted"}
    assert cand["NCT00000011"][0] == "val" and set(cand["NCT00000011"][2]) == {"results_posted"}, "one split per trial"
    assert rep["final_status/dropped_trial_in_later_split_val"] == 1
    assert set(picked[("final_status", "train")]) == {"NCT00000001", "NCT00000002", "NCT00000008"}
    # state: status sentences and trial ids removed, a status word in the title drops the record
    st = {"protocolSection": {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Drug X for Condition Y (see NCT01234567)"},
        "statusModule": {"studyFirstPostDateStruct": {"date": "2018-01-10"}, "overallStatus": "TERMINATED"},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Pharma", "class": "INDUSTRY"}},
        "oversightModule": {"oversightHasDmc": True},
        "descriptionModule": {"briefSummary": "This study tests drug X. The study was terminated early due to slow "
                                              "accrual. Patients receive X daily."},
        "conditionsModule": {"conditions": ["Condition Y"]},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"], "enrollmentInfo": {"count": 3,
                         "type": "ACTUAL"}, "designInfo": {"allocation": "RANDOMIZED", "primaryPurpose": "TREATMENT",
                                                           "maskingInfo": {"masking": "NONE"}}},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "X"}]},
        "eligibilityModule": {"sex": "ALL", "minimumAge": "18 Years", "healthyVolunteers": False,
                              "eligibilityCriteria": "Inclusion Criteria:\n* Adults with Y\nExclusion Criteria:\n* Pregnancy"}}}
    rep2 = Counter()
    text, info = ctg_state(st, rep2)
    assert text and "terminated" not in text.lower() and "NCT01234567" not in text and "NCT[masked]" in text
    assert "Pregnancy" in text and "sex all" in text and "actual" not in text.lower() and "Enrol" not in text
    assert "Patients receive X daily." in text and rep2["status_sentence_removed/summary"] == 1
    assert info["phase"] == "PHASE2" and info["sponsor_class"] == "INDUSTRY"
    st2 = json.loads(json.dumps(st))
    st2["protocolSection"]["identificationModule"]["briefTitle"] = "Study terminated early: drug X"
    assert ctg_state(st2, Counter())[0] is None
    cv = content_view({"source": "ctgov/final_status", "state": text, "meta": {}})
    assert "Design:" not in cv and "Acme Pharma" in cv and "Drug X for Condition Y" in cv
    # CWE: gold provenance, mapping up CWE-1000 ChildOf links, equal split
    parents = {"122": ["787"], "787": ["119"], "80": ["79"], "79": ["74"], "121": ["787", "788"]}
    opts = {"cwe_787", "cwe_79", "cwe_74"}
    assert cwe_map("CWE-122", opts, parents) == ["cwe_787"] and cwe_map("CWE-80", opts, parents) == ["cwe_79"]
    assert cwe_map("CWE-119", opts, parents) == ["other"] and cwe_map("NVD-CWE-Other", opts, parents) == ["other"]
    assert cwe_map("NVD-CWE-noinfo", opts, parents) is None
    w = [{"source": "psirt@x", "type": "Secondary", "values": ["CWE-20"]},
         {"source": NVD_SOURCE, "type": "Primary", "values": ["CWE-79"]}]
    assert cwe_gold(w) == (["CWE-79"], "nvd") and cwe_gold(w[:1]) == (["CWE-20"], "cna")
    assert cwe_gold([{"source": NVD_SOURCE, "values": ["NVD-CWE-noinfo"]}, w[0]]) == (None, "nvd_noinfo")
    dist = cwe_dist(["CWE-79", "CWE-122"], opts, parents)
    assert dist == {"cwe_79": 0.5, "cwe_787": 0.5}
    ev = {"description": "XSS (CWE-79) in Foo via CVE-2024-0001 parameter, see https://cwe.mitre.org/data/definitions/79.html .", "cvss": {"V3.1": "AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N"},
          "cwe": ["Foo Inc. CWE-79"], "refs": ["https://cwe.mitre.org/data/definitions/79.html", "https://foo.example/a"],
          "source": "psirt@foo", "cna_name": "Foo Inc."}
    s = nvd_state(ev, "CVE-2024-0001", "2024-03-01")
    assert "Weakness (CWE)" not in s and "CWE-79" not in s and "CWE-[masked]" in s and "cwe.mitre.org" not in s
    assert "CVE-[this]" in s and "foo.example" in s and "base score 6.1" in s
    assert content_view({"source": "nvd_cwe/cwe_class", "state": s, "meta": {}}).startswith("XSS (CWE-[masked]) in Foo")
    # SCDB: pre-decision state, disposition mapping
    vl = {"var_to_label": {"jurisdiction": "J", "petitioner": "P", "respondent": "P", "caseSource": "C",
                           "lcDisposition": "L", "issueArea": "I", "respondentState": "S"},
          "labels": {"J": {"1": "cert"}, "P": {"28": "State", "137": "person accused of crime"},
                     "C": {"300": "State Supreme Court"}, "L": {"2": "affirmed"}, "I": {"1": "Criminal Procedure"},
                     "S": {"29": "Mississippi"}}}
    srow = {"caseId": "2025-001", "term": "2025", "chief": "Roberts", "caseName": "PITTS v. MISSISSIPPI",
            "docket": "24-1159", "jurisdiction": "1", "petitioner": "137", "respondent": "28", "respondentState": "29",
            "caseSource": "300", "lcDisposition": "2", "issueArea": "1", "dateArgument": "", "caseDisposition": "4",
            "dateDecision": "11/24/2025", "majVotes": "9", "partyWinning": "1"}
    stt, arg = scdb_state(srow, vl)
    assert "PITTS v. MISSISSIPPI" in stt and "What the court below did: affirmed" in stt and not arg
    assert "decided without oral argument" in stt and "9" not in stt.replace("docket 24-1159", "")
    assert SCDB_OF.get("4") == "reversed" and SCDB_OF.get("6", "other") == "other"
    assert not SCDB_OUTCOME_RX.search(stt)
    # records -> write -> iter_units read-back
    with tempfile.TemporaryDirectory() as td:
        lab = lab_dict(FS_KEYS, [0.6, 0.3, 0.05, 0.05])
        tr = build_rec("ctgov/final_status", text, {"final_status": choice_q(FS_INSTR, dict(FS_CRITERIA))},
                       {"final_status": lab}, train_meta("x", "nct:1", "2018-01", "2018-01", {}, "train", {}, {},
                                                         {"final_status": "completed"}, {"final_status": lab}, 0.5),
                       "ctgov/NCT00000001", LICENSES["ctgov"].format(version=CTG_DATA_VERSION))
        evr = build_rec("ctgov/final_status", text + "\nx", {"final_status": choice_q(FS_INSTR, dict(FS_CRITERIA))},
                        {"final_status": onehot(FS_KEYS, "terminated")},
                        gde1_meta("ctgov", "O", {"decision_date": "2018-01-10", "info_date": "2018-01-10",
                                                 "resolution_date": {"final_status": "2025-08-20"}}),
                        "ctgov/NCT00000003", LICENSES["ctgov"].format(version=CTG_DATA_VERSION))
        write_corpus(Path(td) / "c", {"train": [tr]}, "ctgov__")
        write_corpus(Path(td) / "g", {"eval": [evr]}, "ctgov__")
        rb = readback_dir(Path(td) / "c", ("train",))
        assert rb["total_units"] == 1 and rb["units_by_source"].get("train/nc_licence=False") == 1, rb
        cf = gb.load_cf()
        assert not cf.NC_LICENCE.search(LICENSES["ctgov"]) and not cf.NC_LICENCE.search(LICENSES["nvd_cwe"])
        assert cf.NC_LICENCE.search(LICENSES["scdb"]), "SCDB must read as non-commercial to the trainer"
        assert readback_dir(Path(td) / "g", ("eval",))["total_units"] == 1
        # the gate: a training record whose summary sentence sits in an eval set is dropped; our template is not
        try:
            import numpy  # noqa: F401
            g = make_gate("ext2")
            tid = g.tag("eval:toy", "eval")
            g.add(tid, "Unrelated eval item. This study tests drug X in adults with condition Y over twelve weeks.")
            g.add(tid, content_view({"source": "ctgov/final_status", "state": "CLINICAL TRIAL REGISTRATION (design as "
                                     "registered; no status, dates or enrolment)\nFirst posted on ClinicalTrials.gov: "
                                     "2025-08\nPhase: phase 3\nDesign: allocation randomized; model parallel; primary "
                                     "purpose treatment; masking none", "meta": {}}))
            g.freeze()
            c1 = content_view(tr)
            assert g.check(c1, content_probes(c1)) is None, "template lines must not match"
            t2 = dict(tr, state=tr["state"] + "\nBrief summary:\nThis study tests drug X in adults with condition Y over "
                                              "twelve weeks.")
            c2 = content_view(t2)
            assert g.check(c2, content_probes(c2)) is not None
            # calibrated rule: a 10-word stock phrase (3 shingles) is reported, not dropped; a copied paragraph drops
            g3 = make_gate("calibrated")
            tid = g3.tag("eval:toy", "eval")
            para = ("Participants will receive the study vaccine at weeks zero and four and attend follow-up visits "
                    "at the clinic for blood draws and symptom diaries through month twelve after enrolment.")
            g3.add(tid, "Eligible adults must be able to understand and sign the consent form. " + para)
            g3.freeze()
            stock = "Some unrelated trial text here. Eligible adults must be able to understand and sign the consent."
            assert g3.check(stock) is None and g3.plain_hit is not None, "stock phrase: reported only"
            assert g3.check("Another summary. " + para) is not None, "copied paragraph: dropped"
        except ImportError:
            print("  (numpy missing: gate check skipped)")
    print("self-test OK")


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--work", default=WORK)
        p.add_argument("--out", default=OUT)
        p.add_argument("--gde1", default=GDE1)
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--lam", type=float, default=LAMBDA)
    for name in ("ctgov", "nvd_cwe"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--fetch-only", action="store_true")
    for name in ("scdb", "cto"):
        common(sub.add_parser(name))
    p = sub.add_parser("gate")
    common(p)
    p.add_argument("--fingerprint", action="append", default=[], metavar="TAG=PATH", help="training corpus (repeatable)")
    p.add_argument("--eval", action="append", default=[], metavar="TAG=PATH", help="extra eval jsonl (repeatable)")
    p.add_argument("--no-fetch-evals", dest="fetch_evals", action="store_false")
    p.add_argument("--allow-missing-evals", action="store_true")
    p.add_argument("--shingle-rule", choices=sorted(SHINGLE_RULES), default="calibrated",
                   help="ext2 = build_jev_ext2's plain >= 3-shingle rule; calibrated = openreview / part A containment rule "
                        "(the plain-rule matches are still reported)")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("check")
    common(p)
    p.add_argument("--probe-n", type=int, default=6000)
    p.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    cmds = {"ctgov": cmd_ctgov, "nvd_cwe": cmd_nvd_cwe, "scdb": cmd_scdb, "cto": cmd_cto, "gate": cmd_gate,
            "check": cmd_check}
    if a.cmd not in cmds:
        ap.print_help()
        return
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
