#!/usr/bin/env python3
"""
build_jev_safety.py -- pipeline P4 (research/realworld_data.md "### P4"): safety narratives with expert codes and
downstream outcomes, in the jev_data record schema (build_jev_data.record / build_jev_ext2.make_record; read back
through trainers/cache_features.iter_units). Write-up, licences, statistics: research/safety_pipeline.md.

Sources (all US federal works; per-record `license` strings in LICENSES):
  osha_sir/coded       OSHA Severe Injury Reports (2015-01 .. 2025-11). State = the final narrative (injury-outcome
                       sentences KEPT, as the design says: they are evidence for the codes), industry, US state, month.
                       Questions: injury_event_group (choice 7), injury_source_group (choice 6), injury_nature_group
                       (choice 7), injured_body_region (choice 6) = OIICS codes assigned by OSHA coders (E);
                       onsite_inspection_opened (noul) = O.
  osha_sir/severity    The same reports with every injury-outcome clause REMOVED (mechanism only). Questions:
                       amputation_reported, inpatient_hospitalization_reported (noul, O, 29 CFR 1904.39 definitions).
                       The design does not ask these on the full narrative because they leak; here they are asked
                       on the mechanism-only text and each is kept only if it passes the leak gate (see below).
  ntsb/aviation        NTSB avall.mdb (2008-). State = the FACTUAL narrative (narr_accp) with the opening
                       boilerplate and every damage / injury sentence removed, sentences sharing a 6-word shingle
                       with the probable cause removed, registration marks masked; plus aircraft, operation, phase of
                       flight, weather. Never narr_accf (analysis) or narr_cause (probable cause). Questions:
                       cause_area (choice 5, E), aircraft_damage_level (score 3, E), highest_injury_level (score 4, E);
                       the last two only if they pass the leak gate.
  nhtsa/investigations NHTSA ODI defect investigations (PE / EA / RQ, opened 1995-) joined to consumer complaints
                       filed BEFORE the opening date for the same make / model / model year and top-level component.
                       Never SUMMARY, CDATE or CAMPNO in the state. Questions: investigation_ends_in_recall (noul, O),
                       next_regulatory_step (choice 3, PE only, O).
  osha_enforcement/fatality_investigations   DOL OSHA accident investigation summaries (abstracts, 2012-) linked by
                       accident_injury.rel_insp_nr to their inspection and its violations.
  osha_enforcement/construction_inspections  NAICS 23 inspections (2012-) without an abstract: structured state.
                       Questions (both): most_serious_citation_type (choice 5), initial_penalty_band (score 5),
                       penalty_reduced_after_issuance (noul; train / val only: censored on EVAL) -- all O, from the
                       violation table. Inspections whose first citation is dated > 400 days after opening are
                       dropped (OSH Act 9(c) six-month rule; later dates are amendments).

Labels (meta.label_source[qid] = expert | outcome; meta.label_method[qid] = the exact construction):
  train / val  t = LAMBDA * onehot(code or outcome) + (1 - LAMBDA) * p_hat, p_hat a hierarchical Dirichlet-smoothed
               base rate over fields the state shows (e.g. global > NAICS sector > NAICS subsector), computed
               LEAVE-ONE-OUT over train + val items so an item's own label never enters its own base rate; each
               level's prior strength is chosen by LOO log score. The base-rate model's LOO log loss / Brier / ECE
               and its score on EVAL are in the manifest (realworld_data.md 3.2: "if it is not calibrated on
               held-out folds, it is not a label").
  eval         the one-hot realised code / outcome (label_kind hard); meta.labels_smoothed and meta.base_rate hold
               the smoothed target and the train-only base rate.
Splits: EVAL = decided on or after 2025-07-01 (SIR: event date; NTSB: final-report approval date, event 2024 or later;
NHTSA: closing date; DOL: first citation issuance, else case close). Train = decided before. Val = VAL_PCT% of train
GROUPS by hash. Groups (establishment, NTSB event, NHTSA PE->EA chain, accident) never straddle splits: a group with
an EVAL item has its earlier items dropped (counted as held_out_group). EVAL states identical to a train / val state,
or sharing > 50% of their rare 8-word shingles with one, are dropped.
Leakage controls: per-source outcome-word removal on every state (counts in the manifest; `check` re-scans and
must find 0), the as-of field rule above, a cross-validated bag-of-words leak probe vs a structured-field model per
question (top features audited against a leak lexicon; questions marked GATED are written only if the audit passes),
eval-vs-train exact and 8-word-shingle overlap checks. PII: no person, employer or establishment names, no street
addresses, cities, ZIP codes, registration marks, VINs; e-mails / phones / "Mr X" are masked in free text.

Usage (laptop; stdlib + numpy + scipy; mdbtools is run through `nix shell nixpkgs#mdbtools` if not on PATH):
  python3 build_jev_safety.py fetch --cache /tmp/safety_jev_cache            # ~4.2 GB download, small extracts
  mkdir -p /tmp/jevstub && echo 'AutoTokenizer = None' > /tmp/jevstub/transformers.py \\
      && echo 'class Tensor: pass' > /tmp/jevstub/torch.py
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_safety.py build --cache /tmp/safety_jev_cache --out /tmp/safety_jev
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_safety.py check --out /tmp/safety_jev
  PYTHONPATH=/tmp/jevstub .venv/bin/python build_jev_safety.py self-test
Output: <out>/{train,val,eval}/<source with / -> __>.jsonl, <out>/manifest.json, <out>/samples.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import random
import re
import shutil
import subprocess
import sys
import time
import types
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_jev_data import choice_q, noul_q, option_keys, record, score_q, validate  # noqa: E402

csv.field_size_limit(1 << 30)

# --------------------------------------------------------------------------- constants
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36")   # osha.gov's CloudFront rejects non-browser agents (HTTP 403)
URLS = {
    "osha_sir.zip": "https://www.osha.gov/sites/default/files/January2015toNovember2025.zip",
    "ntsb_avall.zip": "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.zip",
    "nhtsa_FLAT_INV.zip": "https://static.nhtsa.gov/odi/ffdd/inv/FLAT_INV.zip",
    "nhtsa_FLAT_CMPL.zip": "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip",
    "naics_2017.xlsx": "https://www.census.gov/naics/2017NAICS/2-6%20digit_2017_Codes.xlsx",
    "naics_2022.xlsx": "https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx",
}
DOL_TABLES = ["accident", "accident_abstract", "accident_injury", "inspection", "violation"]
for _t in DOL_TABLES:
    URLS[f"dol_OSHA_{_t}.zip"] = f"https://data.dol.gov/data-catalog/OSHA/{_t}/OSHA_{_t}.zip"
NTSB_TABLES = ["events", "aircraft", "narratives", "Findings", "NTSB_Admin", "Events_Sequence",
               "eADMSPUB_DataDictionary", "ct_seqevt"]

LICENSES = {
    "osha_sir": ("US federal government work, public domain (17 U.S.C. 105): OSHA Severe Injury Reports "
                 "(https://www.osha.gov/severe-injury-reports, data 2015-01-01..2025-11-30; OIICS codes assigned by "
                 "OSHA). No employer names, addresses, cities, ZIP codes or coordinates are kept."),
    "ntsb": ("US federal government work, public domain (17 U.S.C. 105): NTSB aviation accident database avall.mdb "
             "(https://data.ntsb.gov/avdata). NTSB reports are not admissible as evidence in civil suits (49 U.S.C. "
             "1154(b)); no registration marks or names are kept."),
    "nhtsa": ("US federal government work, public domain (17 U.S.C. 105): NHTSA ODI investigations (FLAT_INV) and "
              "consumer complaints (FLAT_CMPL), https://static.nhtsa.gov/odi/ffdd/. Complaint narratives are written "
              "by members of the public and published by NHTSA; no complainant fields (city, VIN, dealer, operator "
              "name) are kept and e-mails / phones / names are masked."),
    "osha_enforcement": ("US federal government work, public domain (17 U.S.C. 105): DOL OSHA enforcement data "
                         "(inspection, violation, accident, accident_abstract, accident_injury tables) from "
                         "https://data.dol.gov/data-catalog/OSHA/. Not endorsed or certified by the US Department of "
                         "Labor. No establishment names or addresses are kept."),
}
DATASETS = {
    "osha_sir/coded": URLS["osha_sir.zip"], "osha_sir/severity": URLS["osha_sir.zip"],
    "ntsb/aviation": URLS["ntsb_avall.zip"],
    "nhtsa/investigations": URLS["nhtsa_FLAT_INV.zip"] + " + " + URLS["nhtsa_FLAT_CMPL.zip"],
    "osha_enforcement/fatality_investigations": "https://data.dol.gov/data-catalog/OSHA/ (accident, accident_abstract, "
                                                "accident_injury, inspection, violation)",
    "osha_enforcement/construction_inspections": "https://data.dol.gov/data-catalog/OSHA/ (inspection, violation)",
}

EVAL_START = "2025-07-01"
LAMBDA = 0.5
VAL_PCT = 10
GRID = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
NARR_CHARS = 1400
SURFACE_JSON_PCT, SURFACE_FORM_PCT = 55, 25            # the rest is the memo surface
CAPS = {  # per source and split; construction kept at >= CONSTRUCTION_MIN_SHARE of a capped split (see cap_items)
    "osha_sir/coded": {"train": 30000, "val": 3000, "eval": None},
    "osha_sir/severity": {"train": 15000, "val": 1500, "eval": None},
    "ntsb/aviation": {"train": 15000, "val": 1500, "eval": None},
    "nhtsa/investigations": {"train": None, "val": None, "eval": None},
    "osha_enforcement/fatality_investigations": {"train": 15000, "val": 1500, "eval": None},
    "osha_enforcement/construction_inspections": {"train": 15000, "val": 1500, "eval": 4000},
}
BENCH_QIDS = {"action", "needs_review", "outcome", "risk", "urgency", "category", "churn_risk", "needs_human",
              "discrepancy_severity", "disposition", "duplicate", "matches_order", "credential_compromise",
              "severity", "true_positive"}

US_STATES = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
             "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
             "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
             "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
             "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
             "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
             "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
             "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
             "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
             "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico", "GU": "Guam",
             "VI": "U.S. Virgin Islands", "AS": "American Samoa", "MP": "Northern Mariana Islands"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- text helpers
def norm_ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def norm_key(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def h100(s):
    return int(hashlib.sha1(str(s).encode()).hexdigest()[:8], 16) % 100


def gid(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


SENT_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def sentences(text):
    return [s for s in SENT_END.split(norm_ws(text)) if s.strip()]


def clip(text, n=NARR_CHARS):
    """Clip at a sentence boundary if one falls in the last 40%, else at a word boundary."""
    text = norm_ws(text)
    if len(text) <= n:
        return text
    cut = text[:n]
    m = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if m >= int(n * 0.6):
        return cut[:m + 1]
    return cut[:cut.rfind(" ")].rstrip(",;:") + " ..."


PII = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
    (re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"), "[phone]"),
    (re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)?"), "[name]"),
    (re.compile(r"\b(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-HJ-NPR-Z])[A-HJ-NPR-Z0-9]{17}\b"), "[vin]"),
]


URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)


def scrub_pii(text, names=()):
    text = URL.sub("[link]", text)
    for pat, rep in PII:
        text = pat.sub(rep, text)
    for nm in names:   # employer / establishment names and street addresses, if they occur verbatim
        nm = norm_ws(nm)
        if len(nm) >= 6 and not nm.isdigit():
            text = re.sub(re.escape(nm), "[employer]" if "[" not in nm else "", text, flags=re.I)
    return text


def drop_sentences(text, pattern):
    """Remove every sentence matching `pattern`; returns (text, n_removed)."""
    kept, n = [], 0
    for s in sentences(text):
        if pattern.search(s):
            n += 1
        else:
            kept.append(s)
    return " ".join(kept), n


def shingles(text, k=8):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


# --------------------------------------------------------------------------- dates
def parse_date(s):
    s = str(s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(s[:25] if "%z" in fmt else s, fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def ym(d):
    return f"{d.year:04d}-{d.month:02d}" if d else None


# --------------------------------------------------------------------------- fetch
def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"cached {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    log(f"GET {url}")
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            tmp.rename(dest)
            log(f"  -> {dest.name} {dest.stat().st_size / 1e6:.1f} MB")
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"  {type(e).__name__}: {e}; retry {attempt + 1}")
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"download failed: {url}")


def zip_csv_rows(path, numeric_chunks=True):
    """Stream the CSV members of a zip (DOL ships LOAD..._chunk_<n>.csv members) as dicts."""
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
    if numeric_chunks:
        names.sort(key=lambda s: int(re.search(r"(\d+)\.\w+$", s).group(1)) if re.search(r"(\d+)\.\w+$", s) else 0)
    for n in names:
        with z.open(n) as f:
            yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline=""))


def mdb_export_cmd():
    if shutil.which("mdb-export"):
        return ["mdb-export"]
    if shutil.which("nix"):
        return ["nix", "shell", "nixpkgs#mdbtools", "-c", "mdb-export"]
    raise SystemExit("mdb-export not found (install mdbtools, or have nix available)")


def xlsx_rows(path):
    z = zipfile.ZipFile(path)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    tag = "{%s}" % ns["m"]
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            ss.append("".join(t.text or "" for t in si.iter(tag + "t")))
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(tag + "row"):
        out = {}
        for c in row.findall("m:c", ns):
            col = re.match(r"([A-Z]+)", c.get("r")).group(1)
            v = c.find("m:v", ns)
            if v is None:
                isel = c.find("m:is", ns)
                out[col] = "".join(x.text or "" for x in isel.iter(tag + "t")) if isel is not None else ""
            else:
                out[col] = ss[int(v.text)] if c.get("t") == "s" else v.text
        yield out


def write_jsonl_gz(path, rows):
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fetch(a):
    cache = Path(a.cache)
    raw, ex = cache / "raw", cache / "extracts"
    raw.mkdir(parents=True, exist_ok=True)
    ex.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        download(url, raw / name)
    # NAICS titles (Census; public domain) -> naics.json
    naics = {}
    for p in ("naics_2017.xlsx", "naics_2022.xlsx"):
        for r in xlsx_rows(raw / p):
            code, title = str(r.get("B", "")).strip(), str(r.get("C", "")).strip()
            if re.fullmatch(r"\d{2,6}(-\d{2})?", code):
                naics[code] = re.sub(r"T$", "", title).strip()
    (ex / "naics.json").write_text(json.dumps(naics, sort_keys=True))
    log(f"naics titles: {len(naics)}")
    extract_ntsb(raw, ex)
    extract_nhtsa(raw, ex)
    extract_dol(raw, ex)
    log("fetch done")


def extract_ntsb(raw, ex):
    out = ex / "ntsb"
    if all((out / f"{t}.csv").exists() for t in NTSB_TABLES):
        log("ntsb extracts cached")
        return
    out.mkdir(exist_ok=True)
    mdb = ex / "avall.mdb"
    with zipfile.ZipFile(raw / "ntsb_avall.zip") as z:
        z.extract("avall.mdb", ex)
    cmd = mdb_export_cmd()
    for t in NTSB_TABLES:
        with open(out / f"{t}.csv", "wb") as f:
            subprocess.run(cmd + ["-D", "%Y-%m-%d", str(mdb), t], stdout=f, check=True)
        log(f"ntsb {t}: {(out / f'{t}.csv').stat().st_size / 1e6:.1f} MB")
    mdb.unlink()


INV_FIELDS = ["action", "make", "model", "year", "compname", "mfr", "odate", "cdate", "campno", "subject", "summary"]
CMPL_COLS = {"odino": 1, "make": 3, "model": 4, "year": 5, "crash": 6, "fire": 8, "injured": 9, "deaths": 10,
             "compdesc": 11, "ldate": 16, "cdescr": 19}


def comp_top(s):
    return norm_ws(str(s).split(":")[0]).upper()


def extract_nhtsa(raw, ex):
    inv_p, cmp_p = ex / "nhtsa_inv.jsonl.gz", ex / "nhtsa_complaints.jsonl.gz"
    if inv_p.exists() and cmp_p.exists():
        log("nhtsa extracts cached")
        return
    acts = defaultdict(list)
    with zipfile.ZipFile(raw / "nhtsa_FLAT_INV.zip") as z, z.open(z.namelist()[0]) as f:
        for line in io.TextIOWrapper(f, encoding="latin-1"):
            c = [x.strip() for x in line.rstrip("\r\n").split("\t")]
            if len(c) < 11:
                continue
            acts[c[0]].append(dict(zip(INV_FIELDS, c[:11])))
    rows = []
    keys = defaultdict(list)   # (make, model, year) -> [(top component, odate)]
    for a, rs in acts.items():
        r0 = rs[0]
        if a[:2] not in ("PE", "EA", "RQ") or r0["odate"] < "19950101":
            continue
        rows.append({"action": a, "rows": [{k: r[k] for k in ("make", "model", "year", "compname")} for r in rs],
                     "mfr": r0["mfr"], "odate": r0["odate"], "cdate": r0["cdate"], "campno": r0["campno"],
                     "subject": r0["subject"], "summary": r0["summary"]})
        for r in rs:
            keys[(r["make"].upper(), r["model"].upper(), r["year"])].append((comp_top(r["compname"]), r0["odate"]))
    log(f"nhtsa actions kept: {write_jsonl_gz(inv_p, rows)}")
    kept = 0

    def gen():
        nonlocal kept
        with zipfile.ZipFile(raw / "nhtsa_FLAT_CMPL.zip") as z, z.open(z.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="latin-1"):
                c = line.rstrip("\r\n").split("\t")
                if len(c) < 20:
                    continue
                k = (c[3].strip().upper(), c[4].strip().upper(), c[5].strip())
                if k not in keys:
                    continue
                top, ld = comp_top(c[11]), c[16].strip()
                if not any(top == t and ld and ld < od for t, od in keys[k]):
                    continue
                kept += 1
                yield {n: c[i].strip() for n, i in CMPL_COLS.items()} | {"cdescr": c[19].strip()[:1200]}
    log(f"nhtsa complaints kept: {write_jsonl_gz(cmp_p, gen())}")


HOST_KEY = re.compile(r"^[A-Z]\d{3}\s+\d{9}$")


def dol_group(r):
    """Establishment key: OSHA's HOST_EST_KEY when it is a real key ("N022    000000096"); the file also holds
    placeholders ("", "000000000", "HOST_EST_KEY_VALUE") shared by ~1M rows, which fall back to name + state + ZIP."""
    k = (r.get("HOST_EST_KEY") or "").strip()
    if HOST_KEY.match(k) and not set(k[-9:]) <= {"0"}:
        return "h" + gid(k)
    return "n" + gid(norm_key(r.get("ESTAB_NAME")), r.get("SITE_STATE", ""), (r.get("SITE_ZIP") or "")[:5])


def extract_dol(raw, ex):
    outs = [ex / f for f in ("dol_accidents.jsonl.gz", "dol_inspections.jsonl.gz", "dol_violations.jsonl.gz",
                             "dol_history.jsonl.gz")]
    if all(p.exists() for p in outs):
        log("dol extracts cached")
        return
    acc = {}
    for r in zip_csv_rows(raw / "dol_OSHA_accident.zip"):
        d = parse_date(r["EVENT_DATE"])
        if d and d >= date(2010, 1, 1):
            acc[r["SUMMARY_NR"]] = {"summary_nr": r["SUMMARY_NR"], "event_date": d.isoformat(),
                                    "event_desc": r["EVENT_DESC"], "const_end_use": r["CONST_END_USE"],
                                    "build_stories": r["BUILD_STORIES"], "nonbuild_ht": r["NONBUILD_HT"],
                                    "project_cost": r["PROJECT_COST"], "project_type": r["PROJECT_TYPE"],
                                    "fatality": r["FATALITY"], "lines": [], "insp": Counter(), "degree": Counter()}
    for r in zip_csv_rows(raw / "dol_OSHA_accident_abstract.zip"):
        a = acc.get(r["SUMMARY_NR"])
        if a is not None:
            try:
                ln = int(float(r["LINE_NR"] or 0))
            except ValueError:
                ln = 0
            a["lines"].append((ln, r["ABSTRACT_TEXT"] or ""))
    for r in zip_csv_rows(raw / "dol_OSHA_accident_injury.zip"):
        a = acc.get(r["SUMMARY_NR"])
        if a is not None:
            if r["REL_INSP_NR"]:
                a["insp"][str(int(float(r["REL_INSP_NR"])))] += 1
            a["degree"][(r["DEGREE_OF_INJ"] or "").split(".")[0]] += 1
    linked = set()
    acc_rows = []
    for a in acc.values():
        text = " ".join(t for _, t in sorted(a.pop("lines"), key=lambda x: x[0]))
        if not norm_ws(text) or not a["insp"]:
            continue
        a["abstract"] = norm_ws(text)
        a["insp"] = dict(a["insp"])
        a["degree"] = dict(a["degree"])
        linked.update(a["insp"])
        acc_rows.append(a)
    log(f"dol accidents with abstract + inspection: {write_jsonl_gz(ex / 'dol_accidents.jsonl.gz', acc_rows)}")

    keep_fields = ["ACTIVITY_NR", "REPORTING_ID", "SITE_STATE", "NAICS_CODE", "SIC_CODE", "OWNER_TYPE", "ADV_NOTICE",
                   "SAFETY_HLTH", "INSP_TYPE", "INSP_SCOPE", "UNION_STATUS", "NR_IN_ESTAB", "OPEN_DATE",
                   "CLOSE_CONF_DATE", "CLOSE_CASE_DATE", "MIGRANT"]
    sel, hist, n_host = {}, [], 0
    for r in zip_csv_rows(raw / "dol_OSHA_inspection.zip"):
        d = parse_date(r["OPEN_DATE"])
        if not d or d < date(2007, 1, 1):
            continue
        g = dol_group(r)
        n_host += g.startswith("h")
        hist.append({"a": r["ACTIVITY_NR"], "g": g, "o": d.isoformat()})
        if d >= date(2012, 1, 1) and (r["NAICS_CODE"].startswith("23") or r["ACTIVITY_NR"] in linked):
            sel[r["ACTIVITY_NR"]] = {k.lower(): r[k] for k in keep_fields} | {"group": g, "accident_linked":
                                                                               r["ACTIVITY_NR"] in linked}
    log(f"dol history rows (2007+): {len(hist)}  host_est_key share {n_host / max(1, len(hist)):.2f}  "
        f"selected inspections: {len(sel)}")
    agg = {}
    serious_hist = {}
    vt = Counter()
    for r in zip_csv_rows(raw / "dol_OSHA_violation.zip"):
        a = r["ACTIVITY_NR"]
        if (r["DELETE_FLAG"] or "").strip():
            continue
        t = (r["VIOL_TYPE"] or "").strip()
        iss = parse_date(r["ISSUANCE_DATE"])
        if a in sel:
            vt[t] += 1
            g = agg.setdefault(a, {"types": Counter(), "n": 0, "init": 0.0, "cur": 0.0, "n_pen": 0, "n_final": 0,
                                   "n_contest": 0, "issue": None, "standards": Counter()})
            g["types"][t] += 1
            g["n"] += 1
            ip = float(r["INITIAL_PENALTY"] or 0)
            g["init"] += ip
            g["cur"] += float(r["CURRENT_PENALTY"] or 0)
            g["n_pen"] += ip > 0
            g["n_final"] += bool((r["FINAL_ORDER_DATE"] or "").strip())
            g["n_contest"] += bool((r["CONTEST_DATE"] or "").strip())
            g["standards"][(r["STANDARD"] or "").strip()[:4]] += 1
            if iss and (g["issue"] is None or iss.isoformat() < g["issue"]):
                g["issue"] = iss.isoformat()
        if t in ("S", "W", "R") and iss:
            prev = serious_hist.get(a)
            if prev is None or iss.isoformat() < prev:
                serious_hist[a] = iss.isoformat()
    log(f"dol violation types among selected inspections: {dict(vt)}")
    rows = []
    for a, g in agg.items():
        g["types"] = dict(g["types"])
        g["standards"] = dict(g["standards"].most_common(5))
        rows.append({"a": a, **g})
    write_jsonl_gz(ex / "dol_violations.jsonl.gz", rows)
    write_jsonl_gz(ex / "dol_inspections.jsonl.gz", sel.values())
    for h in hist:
        if h["a"] in serious_hist:
            h["s"] = serious_hist[h["a"]]
    write_jsonl_gz(ex / "dol_history.jsonl.gz", hist)
    log(f"dol inspections with violations: {len(rows)}")


# --------------------------------------------------------------------------- base rates (hierarchical, LOO)
def loo_rates(y, levels, in_counts, K):
    """Hierarchical Dirichlet-smoothed base rates, leave-one-out for items that are in the counts.

    y: [n] option index; levels: [n] tuples of cell keys, coarse -> fine (global is implicit); in_counts: [n] bool
    (train + val items; EVAL items are scored with the full train counts). Level l's rate is
    (n_cell - self + a_l * p_{l-1}) / (N_cell - self + a_l), a_l chosen by LOO log score over in-count items.
    Returns (p [n, K], info)."""
    import numpy as np
    n = len(y)
    y = np.asarray(y)
    inc = np.asarray(in_counts, bool)
    E = np.zeros((n, K))
    E[np.arange(n), y] = 1.0
    Es = E * inc[:, None]
    tot = Es.sum(0)
    p = (tot[None, :] - Es + 1.0) / (tot.sum() - inc[:, None] + K)
    info = {"levels": [], "n_counts": int(inc.sum())}

    def ll(q):
        return float(-np.log(q[np.arange(n), y][inc] + 1e-12).mean()) if inc.any() else float("nan")

    info["global_ll"] = ll(p)
    L = len(levels[0]) if n else 0
    for li in range(L):
        keys = [lv[li] for lv in levels]
        idx = {k: j for j, k in enumerate(dict.fromkeys(keys))}
        ki = np.array([idx[k] for k in keys])
        C = np.zeros((len(idx), K))
        np.add.at(C, ki[inc], E[inc])
        Ci = C[ki] - Es
        ni = Ci.sum(1, keepdims=True)
        best = None
        for a in GRID:
            q = (Ci + a * p) / (ni + a)
            s = ll(q)
            if best is None or s < best[0]:
                best = (s, a, q)
        p = best[2]
        info["levels"].append({"cells": len(idx), "a": best[1], "loo_ll": round(best[0], 4)})
    return p, info


def calib_report(p, y, mask):
    """Log loss, Brier, accuracy and 15-bin top-label ECE of base-rate predictions p on items `mask`."""
    import numpy as np
    y = np.asarray(y)
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    P, Y = p[m], y[m]
    n, K = P.shape
    oh = np.zeros_like(P)
    oh[np.arange(n), Y] = 1
    conf, pred = P.max(1), P.argmax(1)
    corr = (pred == Y).astype(float)
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(corr[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(15) if (bins == b).any())
    return {"n": int(n), "log_loss": round(float(-np.log(P[np.arange(n), Y] + 1e-12).mean()), 4),
            "brier": round(float(((P - oh) ** 2).sum(1).mean()), 4), "accuracy": round(float(corr.mean()), 4),
            "ece_top15": round(float(ece), 4)}


def tidy(d):
    """Round to 6 places and make the values sum to exactly 1 (validate checks 1e-6)."""
    keys = list(d)
    vals = [max(0.0, round(float(d[k]), 6)) for k in keys]
    s = sum(vals)
    vals = [v / s for v in vals]
    vals = [round(v, 6) for v in vals]
    j = max(range(len(vals)), key=lambda i: vals[i])
    vals[j] = round(vals[j] + 1.0 - sum(vals), 6)
    return dict(zip(keys, vals))


# --------------------------------------------------------------------------- question specs
class Q:
    """One question schema: qid, type, instructions, option keys (choice: alphabetical) / levels, texts."""

    def __init__(self, qid, qtype, instructions, options, source_tag, method, yes_no=None):
        self.qid, self.type, self.instructions, self.source_tag, self.method = qid, qtype, instructions, source_tag, method
        if qtype == "choice":
            self.keys = sorted(options)
            self.criteria = {k: options[k] for k in self.keys}
        elif qtype == "score":
            self.keys = list(options)
        else:
            self.keys = ["yes", "no"]
            self.yes_no = yes_no
        assert qid not in BENCH_QIDS, qid

    def render(self):
        if self.type == "choice":
            return choice_q(self.instructions, dict(self.criteria))
        if self.type == "score":
            return score_q(self.instructions, self.keys)
        q = noul_q(self.instructions)
        if self.yes_no:
            q["criteria"] = {"yes": self.yes_no[0], "no": self.yes_no[1]}
        return q


def bench_overlap_errors(qsets):
    """realworld_data.md 3.8.2: no benchmark qid, and no schema sharing >= 2 qids with a benchmark workflow."""
    errs = []
    try:
        from synth_workflows.exclusion import EXCLUDED_QIDS, EXCLUDED_SCHEMAS
    except Exception:  # noqa: BLE001
        EXCLUDED_QIDS, EXCLUDED_SCHEMAS = sorted(BENCH_QIDS), []
    bench = defaultdict(set)
    for w, q, *_ in EXCLUDED_SCHEMAS:
        bench[w].add(q)
    for src, qids in qsets.items():
        for q in qids:
            if q in EXCLUDED_QIDS:
                errs.append(f"{src}: qid {q} is a benchmark qid")
        for w, ids in bench.items():
            if len(ids & set(qids)) >= 2:
                errs.append(f"{src}: shares {sorted(ids & set(qids))} with benchmark workflow {w}")
    return errs


# --------------------------------------------------------------------------- state surfaces
def render_state(title, sections, surface):
    """json: compact sorted-key JSON (the benchmark's form); form: labelled fields; memo: prose header + text."""
    if surface == "json":
        return json.dumps(sections, sort_keys=True, ensure_ascii=False)
    lines = [title.upper()] if surface == "form" else [title + "."]
    for sec, fields in sections.items():
        if surface == "form":
            lines.append(sec.replace("_", " ").capitalize())
            for k, v in fields.items():
                if v is None or v == "":
                    continue
                lines.append(f"  {k.replace('_', ' ').capitalize()}: {v if not isinstance(v, (list, dict)) else json.dumps(v, ensure_ascii=False)}")
        else:
            parts = []
            long = []
            for k, v in fields.items():
                if v is None or v == "":
                    continue
                if isinstance(v, str) and len(v) > 160:
                    long.append((k, v))
                elif isinstance(v, list):
                    long.append((k, " | ".join(str(x) for x in v)))
                else:
                    parts.append(f"{k.replace('_', ' ')}: {v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)}")
            if parts:
                lines.append(f"{sec.replace('_', ' ').capitalize()} -- " + "; ".join(parts) + ".")
            for k, v in long:
                lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")
    return "\n".join(lines) if surface == "form" else "\n\n".join(lines)


def pick_surface(key):
    b = h100("surface|" + key)
    return "json" if b < SURFACE_JSON_PCT else "form" if b < SURFACE_JSON_PCT + SURFACE_FORM_PCT else "memo"


# --------------------------------------------------------------------------- leak probe
def _tok(text):
    return re.findall(r"[a-z][a-z0-9']+", text.lower())


def bow(texts, vocab=None, min_df=5, max_feat=20000):
    import numpy as np
    import scipy.sparse as sp
    docs = [set(_tok(t)) for t in texts]
    if vocab is None:
        df = Counter(w for d in docs for w in d)
        vocab = {w: i for i, (w, c) in enumerate(x for x in df.most_common(max_feat) if x[1] >= min_df)}
    rows, cols = [], []
    for i, d in enumerate(docs):
        for w in d:
            j = vocab.get(w)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), len(vocab))), vocab


def onehot_feats(feats, vocab=None):
    import numpy as np
    import scipy.sparse as sp
    if vocab is None:
        vocab = {f: i for i, f in enumerate(sorted({f for fs in feats for f in fs}))}
    rows, cols = [], []
    for i, fs in enumerate(feats):
        for f in fs:
            j = vocab.get(f)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(feats), len(vocab))), vocab


def fit_logreg(X, y, K, l2=1.0, iters=200):
    import numpy as np
    from scipy.optimize import minimize
    n, d = X.shape
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1

    def f(w):
        W = w.reshape(d + 1, K)
        Z = X @ W[:-1] + W[-1]
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        loss = -np.sum(Y * np.log(P + 1e-12)) / n + 0.5 * l2 * np.sum(W[:-1] ** 2) / n
        G = (P - Y) / n
        g = np.vstack([X.T @ G + l2 * W[:-1] / n, G.sum(0, keepdims=True)])
        return loss, g.ravel()

    res = minimize(f, np.zeros((d + 1) * K), jac=True, method="L-BFGS-B", options={"maxiter": iters})
    return res.x.reshape(d + 1, K)


def predict(W, X):
    import numpy as np
    Z = X @ W[:-1] + W[-1]
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def auc(scores, labels):
    import numpy as np
    s, l = np.asarray(scores), np.asarray(labels).astype(bool)
    if l.all() or (~l).all():
        return None
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def leak_probe(texts, struct, y, K, keys, groups, audit=None, max_n=12000, seed=0):
    """5-fold (by group) bag-of-words logistic on the state text vs the same model on structured fields only.

    Returns metrics for both, the text model's top features per option, and `flagged`: top features matching
    the leak-audit lexicon (a non-empty list fails a GATED question)."""
    import numpy as np
    rng = random.Random(seed)
    idx = list(range(len(y)))
    if len(idx) > max_n:
        idx = sorted(rng.sample(idx, max_n))
    texts = [texts[i] for i in idx]
    struct = [struct[i] for i in idx]
    y = np.asarray([y[i] for i in idx])
    folds = np.array([h100("fold|" + groups[i]) % 5 for i in idx])
    out = {"n": len(idx)}
    for name, (X, vocab) in (("bow", bow(texts)), ("structured", onehot_feats(struct))):
        # L2 strength picked on fold 0 (train on folds 1-4), then 5-fold out-of-fold predictions
        best_l2, best_ll = 1.0, None
        for l2 in (1.0, 4.0, 16.0, 64.0):
            tr, te = folds != 0, folds == 0
            if not (te.any() and tr.any()):
                break
            Pv = predict(fit_logreg(X[tr], y[tr], K, l2=l2), X[te])
            llv = float(-np.log(Pv[np.arange(te.sum()), y[te]] + 1e-12).mean())
            if best_ll is None or llv < best_ll:
                best_l2, best_ll = l2, llv
        P = np.zeros((len(y), K))
        for f in range(5):
            tr, te = folds != f, folds == f
            if te.any() and tr.any():
                W = fit_logreg(X[tr], y[tr], K, l2=best_l2)
                P[te] = predict(W, X[te])
        m = {"l2": best_l2, "log_loss": round(float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean()), 4),
             "accuracy": round(float((P.argmax(1) == y).mean()), 4)}
        if K == 2:
            a_ = auc(P[:, 0], y == 0)
            m["auc"] = round(a_, 4) if a_ is not None else None
        out[name] = m
        if name == "bow":
            W = fit_logreg(X, y, K, l2=best_l2)
            inv = {j: w for w, j in vocab.items()}
            Wc = W[:-1] - W[:-1].mean(1, keepdims=True)
            top = {}
            for k in range(K):
                order = np.argsort(-Wc[:, k])[:15 if K > 2 else 25]
                top[keys[k]] = [inv[j] for j in order]
                if K == 2:
                    break
            out["top_features"] = top
    flagged = []
    if audit is not None:
        for k, words in out.get("top_features", {}).items():
            flagged += [w for w in words if audit.search(w)]
    out["flagged"] = sorted(set(flagged))
    out["passes"] = not flagged
    return out


# --------------------------------------------------------------------------- splitting and capping
def assign_splits(items, stats):
    """EVAL = decision_date >= EVAL_START; groups with an EVAL item drop their earlier items; val by group hash."""
    eval_groups = {it["group"] for it in items if it["decision_date"] >= EVAL_START}
    out = []
    for it in items:
        if it["decision_date"] >= EVAL_START:
            it["split"], it["split_reason"] = "eval", f"eval: decided {it['decision_date']} (>= {EVAL_START})"
        elif it["group"] in eval_groups:
            stats["held_out_group"] += 1
            continue
        elif h100("val|" + it["group"]) < VAL_PCT:
            it["split"], it["split_reason"] = "val", f"val: decided before {EVAL_START}; {VAL_PCT}% group hash"
        else:
            it["split"], it["split_reason"] = "train", f"train: decided before {EVAL_START}"
        out.append(it)
    return out


CONSTRUCTION_MIN_SHARE = 0.4


def cap_items(items, cap, rng):
    """Random sample down to `cap`, with construction (NAICS 23, the largest SIR sector and the design's priority)
    at max(its natural share, CONSTRUCTION_MIN_SHARE) of the cap, as far as construction items are available."""
    if cap is None or len(items) <= cap:
        return items
    cons = [it for it in items if it.get("construction")]
    rest = [it for it in items if not it.get("construction")]
    n_cons = min(len(cons), max(round(cap * len(cons) / len(items)), round(cap * CONSTRUCTION_MIN_SHARE)))
    n_cons = max(n_cons, cap - len(rest))
    return rng.sample(cons, n_cons) + rng.sample(rest, cap - n_cons)


# --------------------------------------------------------------------------- NAICS helpers
class Naics:
    SECTOR = {"31": "31-33", "32": "31-33", "33": "31-33", "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"}

    def __init__(self, path):
        self.t = json.loads(Path(path).read_text()) if Path(path).exists() else {}

    def sector_code(self, code):
        c2 = str(code)[:2]
        return self.SECTOR.get(c2, c2)

    def title(self, code):
        code = str(code)
        for n in (6, 5, 4, 3):
            if len(code) >= n and code[:n] in self.t:
                return self.t[code[:n]]
        return None

    def describe(self, code):
        code = re.sub(r"\D", "", str(code or ""))
        if len(code) < 2 or code.startswith("00") or code in ("999999",):
            return None
        sec = self.t.get(self.sector_code(code))
        if not sec:
            return None
        d = {"sector": f"{sec} (NAICS {self.sector_code(code)})"}
        sub = self.t.get(code[:3])
        if sub:
            d["subsector"] = f"{sub} (NAICS {code[:3]})"
        ind = self.title(code) if len(code) >= 4 else None
        if ind and ind != sub:
            d["industry"] = f"{ind} (NAICS {code})"
        return d


# --------------------------------------------------------------------------- OSHA SIR
SIR_EVENT = {"1": "violence_or_animal", "2": "transportation_incident", "3": "fire_or_explosion",
             "4": "fall_slip_or_trip", "5": "harmful_substance_or_environment",
             "6": "contact_with_object_or_equipment", "7": "overexertion_or_bodily_reaction"}
SIR_EVENT_CRIT = {
    "contact_with_object_or_equipment": "Contact with objects and equipment: struck by or against an object, caught in "
        "or compressed by running machinery, shifting objects or collapsing material, or rubbed or abraded (OIICS "
        "event division 6).",
    "fall_slip_or_trip": "Falls, slips, trips: a fall to a lower level or on the same level, a jump to a lower level, "
        "or a slip or trip without a fall (OIICS division 4).",
    "fire_or_explosion": "Fires and explosions: the injury was produced by a fire or an explosion, such as ignited "
        "vapors, a flash fire or a ruptured pressure vessel (OIICS division 3).",
    "harmful_substance_or_environment": "Exposure to harmful substances or environments: electricity, hot or cold "
        "objects and substances, environmental heat or cold, harmful chemicals inhaled or on the skin, oxygen "
        "deficiency, noise or radiation (OIICS division 5).",
    "overexertion_or_bodily_reaction": "Overexertion and bodily reaction: lifting, pushing, pulling, carrying, "
        "bending, twisting, reaching or climbing, without being struck or caught (OIICS division 7).",
    "transportation_incident": "Transportation incidents: a vehicle or powered industrial vehicle in motion was "
        "involved -- collisions, overturns, pedestrians struck by a vehicle, an occupant caught between a vehicle "
        "and an object (OIICS division 2).",
    "violence_or_animal": "Violence and other injuries by persons or animals: injuries inflicted by another person "
        "(assault, shooting, restraint) or by an animal or insect (bites, kicks, stings) (OIICS division 1).",
}
SIR_SOURCE = {"3": "machinery", "4": "part_or_material", "6": "structure_or_surface", "7": "tool_or_equipment",
              "8": "vehicle", "1": "other_source", "2": "other_source", "5": "other_source", "9": "other_source"}
SIR_SOURCE_CRIT = {
    "machinery": "Machinery: conveyors, presses, saws, food-processing and woodworking machines, mixers, hoists and "
        "cranes, earthmoving and other powered or unpowered machines (OIICS source division 3).",
    "part_or_material": "Parts and materials: building materials, pipes and ducts, machine and vehicle parts, "
        "trailers, power lines and electrical parts, metal stock, lumber and scrap (OIICS division 4).",
    "structure_or_surface": "Structures and surfaces: floors, walkways and ground surfaces, roofs, stairs, scaffolds, "
        "openings and other structural elements (OIICS division 6).",
    "tool_or_equipment": "Tools, instruments and equipment: hand and power tools, ladders, welding and cutting "
        "equipment and other non-machine equipment (OIICS division 7).",
    "vehicle": "Vehicles: trucks, tractor-trailers and cars, forklifts and other powered industrial vehicles, pallet "
        "jacks, rail, water and air vehicles (OIICS division 8).",
    "other_source": "Other sources: chemicals; containers, furniture and fixtures (boxes, pallets, tanks, reels); "
        "persons, animals, plants and minerals (including the worker's own bodily motion, patients, trees); "
        "environmental heat, water and other sources (OIICS divisions 1, 2, 5 and 9).",
}
SIR_NATURE_CRIT = {
    "amputation_or_avulsion": "Amputations, avulsions and enucleations: loss of all or part of a body part, "
        "including fingertip amputations.",
    "burn_or_electrical": "Burns and corrosions (heat, chemical, electrical) and electrocutions or electric shocks.",
    "fracture": "Fractures: broken bones, including chipped and crushed bones.",
    "head_spinal_or_internal": "Intracranial injuries (concussions, brain hemorrhages), spinal cord injuries and "
        "internal injuries to organs or blood vessels.",
    "heat_cold_or_toxic_effect": "Effects of environmental conditions or substances: heat exhaustion and heat "
        "stroke, frostbite and hypothermia, poisoning, toxic or allergic reactions, asphyxiation and drowning.",
    "open_wound_or_bruise": "Open wounds and surface injuries: cuts, lacerations, punctures, gunshot wounds, "
        "abrasions, bruises and contusions.",
    "other_injury": "Other natures: sprains, strains, tears, dislocations and hernias; crushing injuries and "
        "nonspecific pain or soreness; multiple injury types of different kinds; diseases such as heart attacks.",
}
SIR_BODY_CRIT = {
    "arm_or_shoulder": "Upper extremities other than the hand: shoulders, upper arms, elbows, forearms and wrists, "
        "or the arm together with the hand.",
    "hand_or_fingers": "Hands and fingers, including fingertips, thumbs and fingernails.",
    "head_or_neck": "Head and neck: skull, brain, scalp, face, eyes, ears, jaw and neck.",
    "lower_extremity": "Lower extremities: legs, thighs, knees, ankles, feet and toes.",
    "multiple_or_body_system": "Multiple body parts in different regions, or a body system (for example the "
        "circulatory or respiratory system, or the body as a whole).",
    "trunk": "Trunk: chest and ribs, back and spine, abdomen, and the pelvic region including the hips.",
}


def sir_nature(title):
    t = title.lower().strip()
    if (t.startswith("multiple") or re.search(r"\bmultiple (types|traumatic|severe|surface|soft|effects|injuries|"
                                              r"intracranial|poison|nonspecified|symptoms|diseases)", t)
            or re.match(r"(fractures|burns|severe wounds|soft tissue injuries|sprains|injuries to the brain, spinal "
                        r"cord) and ", t) or "and other injuries" in t):
        return "other_injury"
    if re.search(r"amputat|avulsion|enucleat", t):
        return "amputation_or_avulsion"
    if "fractur" in t and "cartilage" not in t and "skull fracture and" not in t and "intracranial" not in t:
        return "fracture"
    if re.search(r"burn|electrocut|electric shock|electrical", t):
        return "burn_or_electrical"
    if re.search(r"intracranial|concussion|brain|cerebral|internal (injur|organ)|closed trauma|spinal cord|paraly", t):
        return "head_spinal_or_internal"
    if re.search(r"heat|frostbite|hypotherm|reduced temperature|poison|toxic|allergenic|asphyx|drown|anaphyla|"
                 r"caisson|smoke", t):
        return "heat_cold_or_toxic_effect"
    if re.search(r"\bcut|lacerat|punct|gunshot|open wound|abrasion|bruis|contusion|surface|scratch|blister|stab", t):
        return "open_wound_or_bruise"
    if t in ("nonclassifiable", "nature  unspecified", "nature, unspecified"):
        return None
    return "other_injury"


def sir_body(code, title):
    c, t = str(code).strip(), str(title).lower()
    d = c[:1]
    if d in ("1", "2"):
        return "head_or_neck"
    if d == "3":
        return "trunk"
    if d == "4":
        if c.startswith("44") or re.search(r"finger|hand\(s\)(, unspecified|  unspecified)?$|thumb|knuckle", t):
            return "hand_or_fingers" if "arm" not in t and "wrist" not in t else "arm_or_shoulder"
        return "arm_or_shoulder"
    if d == "5":
        return "trunk" if "hip" in t else "lower_extremity"
    if d in ("6", "8"):
        return "multiple_or_body_system"
    return None


SIR_INSPECTION_LEAK = re.compile(r"\b(OSHA|inspect\w*|CSHO|compliance (?:safety and health )?officer|citation\w*|"
                                 r"cited|area office|investigat\w*)\b", re.I)
BODY = (r"(?:\S+'s\s+)?(?:(?:left|right|lower|upper|lower-back)\s+)*(?:arms?|legs?|ankles?|wrists?|hips?|ribs?|"
        r"heels?|knees?|kneecaps?|foot|feet|toes?|hands?|fingers?|thumbs?|clavicles?|collarbones?|femurs?|tibias?|"
        r"fibulas?|humerus|vertebrae?|pelvis|back|neck|nose|jaw|skull|shoulders?|elbows?|forearms?|bones?|"
        r"shins?|ankle|spine)\b")
SEVERITY_WORDS = re.compile(
    r"\b(amputat\w*|sever(?:ed|ing|s)?|severe\w*|cut off|cut (?:his|her|their|through (?:his|her))|fingertips?|"
    r"tip of (?:his|her|their|the|a|an|one|two|three|several)\b|lost (?:the |his |her |their |a |an |part |parts |"
    r"portions? |one |two |three )?(?:of )?(?:the |his |her )?(?:tip|tips|finger|fingers|thumb|toe|toes|hand|arm|leg|"
    r"foot|eye|digit|digits)|loss of|hospital\w*|admit\w*|overnight|surger\w*|surgical\w*|treat(?:ed|ment|ing)|"
    r"transported (?:to|by)|was transported|taken to|ambulance|ems|paramedic\w*|medical\w*|clinic\w*|emergency "
    r"(?:room|department|services)|er|icu|intensive care|observation|stitch\w*|sutur\w*|fractur\w*|"
    r"burn(?:s|ed|ing)?|lacerat\w*|injur\w*|wound\w*|crush(?:ed|ing|es)?|dislocat\w*|concuss\w*|bruis\w*|"
    r"contus\w*|died|death|dead|fatal\w*|kill\w*|pain|pains|painful|hurt\w*|bleed\w*|blood\w*|deglov\w*|"
    r"avuls\w*|enucleat\w*|reattach\w*|amputee|stable condition|critical condition|degree|diagnos\w*|"
    r"x-?rays?|splint\w*|released (?:from|the following|the next|later)|discharged|doctor\w*|physician\w*|nurse\w*|"
    r"trauma\w*|lost consciousness|"
    r"unconscious|pronounced|recover\w*|swollen|swelling|bones?|tendon\w*|ligament\w*|stitches|"
    r"suffer\w*|sustain\w*|(?:broke|broken|breaking|break|breaks)\s+(?:\S+\s+){0,4}?" + BODY + r"|"
    r"(?:arm|leg|wrist|ankle|hip|rib|bone|finger|toe|foot|hand|back|neck|nose|jaw|pelvis|femur|shoulder|elbow|"
    r"skull|heel|knee|vertebra)(?:s|e)? (?:was |were )?(?:broke|broken))\b", re.I)
SEVERITY_AUDIT = re.compile(r"^(amput\w*|sever\w*|hospital\w*|admit\w*|overnight|surg\w*|treat\w*|ambulance|"
                            r"medical\w*|clinic|fractur\w*|broke\w*|burn\w*|lacerat\w*|injur\w*|crush\w*|"
                            r"died|dead|death|kill\w*|tip|tips|lost|loss|doctor|nurse|bone\w*|released|"
                            r"discharg\w*|stitch\w*|condition|observation|er|ems|cast)$")
CLAUSE_SPLIT = re.compile(r",\s+|;\s+|\s+(?=(?:and|which|resulting|causing|that resulted|where (?:he|she|they)|"
                          r"while (?:he|she|they) (?:was|were) being)\b)")


def mechanism_only(text):
    """Cut each sentence at its first clause that names an injury, treatment or outcome; drop it if that is the
    first clause. Returns (text, n_sentences_cut)."""
    text = re.sub(r"\b(the )?injured (employee|worker|operator|driver|individual|person)\b",
                  lambda m: (m.group(1) or "") + m.group(2), text, flags=re.I)
    out, n_cut = [], 0
    for s in sentences(text):
        bounds = [0] + [m.end() for m in CLAUSE_SPLIT.finditer(s)]
        cut = None
        for i, b in enumerate(bounds):
            e = bounds[i + 1] if i + 1 < len(bounds) else len(s)
            if SEVERITY_WORDS.search(s[b:e]):
                cut = b
                break
        if cut is None:
            out.append(s)
            continue
        n_cut += 1
        if cut == 0:
            continue
        kept = re.sub(r"(,|;|\s)+(and|which|resulting|causing|that resulted|where|while)?\s*$", "", s[:cut].rstrip())
        kept = kept.rstrip(",;: ")
        if len(kept.split()) >= 4:
            out.append(kept + ".")
    return " ".join(out), n_cut


def sir_questions():
    lam = LAMBDA
    m_e = (f"{lam}*onehot(OIICS code assigned by OSHA, grouped) + {1 - lam}*LOO Dirichlet base rate "
           f"(global > NAICS sector > NAICS subsector)")
    return {
        "injury_event_group": Q("injury_event_group", "choice",
                                "Which OIICS event or exposure division did OSHA's coder assign to this injury? The "
                                "event is the manner in which the injury was produced or inflicted.",
                                SIR_EVENT_CRIT, "expert", m_e),
        "injury_source_group": Q("injury_source_group", "choice",
                                 "Which OIICS source division did OSHA's coder assign? The source is the object, "
                                 "substance, person, motion or exposure that directly produced the injury.",
                                 SIR_SOURCE_CRIT, "expert", m_e),
        "injury_nature_group": Q("injury_nature_group", "choice",
                                 "Which group does the OIICS nature-of-injury code assigned by OSHA's coder fall in? "
                                 "The nature is the principal physical characteristic of the injury.",
                                 SIR_NATURE_CRIT, "expert", m_e + " (nature grouped from the code TITLE: the file "
                                                                  "mixes OIICS 1.01 and 2.01 nature codes)"),
        "injured_body_region": Q("injured_body_region", "choice",
                                 "Which body region does the OIICS part-of-body code assigned by OSHA's coder fall "
                                 "in? The part of body is the part directly affected by the injury.",
                                 SIR_BODY_CRIT, "expert", m_e),
        "onsite_inspection_opened": Q("onsite_inspection_opened", "noul",
                                      "OSHA opens an inspection of the establishment in response to this report.",
                                      None, "outcome",
                                      f"{lam}*onehot(SIR 'Inspection' number present) + {1 - lam}*LOO Dirichlet base "
                                      f"rate (global > sector > sector x event group > sector x event group x US "
                                      f"state)",
                                      yes_no=("An OSHA inspection is opened and linked to this report.",
                                              "No inspection is linked to the report: OSHA handles it without one, for "
                                              "example by asking the employer to investigate and report back.")),
        "amputation_reported": Q("amputation_reported", "noul",
                                 "The employer's report of this incident includes an amputation.", None, "outcome",
                                 f"{lam}*onehot(SIR Amputation >= 1) + {1 - lam}*LOO Dirichlet base rate (global > "
                                 f"sector > sector x event group)",
                                 yes_no=("At least one worker suffered an amputation as 29 CFR 1904.39(b)(11) defines "
                                         "it: the traumatic loss of a limb or other external body part, severed "
                                         "completely or partially, including fingertip amputations with or without "
                                         "bone loss and parts later reattached.",
                                         "No amputation: the report is for an in-patient hospitalization or the loss "
                                         "of an eye. Avulsions, enucleations, deglovings, scalpings, severed ears and "
                                         "broken or chipped teeth are not amputations.")),
        "inpatient_hospitalization_reported": Q(
            "inpatient_hospitalization_reported", "noul",
            "At least one worker hurt in this incident is admitted to a hospital as an in-patient.", None, "outcome",
            f"{lam}*onehot(SIR Hospitalized >= 1) + {1 - lam}*LOO Dirichlet base rate (global > sector > sector x "
            f"event group)",
            yes_no=("At least one worker was formally admitted to the in-patient service of a hospital or clinic for "
                    "care or treatment (29 CFR 1904.39(b)(9)).",
                    "No in-patient admission was reported: the report is for an amputation or the loss of an eye "
                    "without admission. Admission only for observation or diagnostic testing does not count "
                    "(1904.39(b)(10)).")),
    }


def load_sir(cache, naics, stats):
    rows = []
    with zipfile.ZipFile(Path(cache) / "raw" / "osha_sir.zip") as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")))
    items = []
    for r in rows:
        d = parse_date(r["EventDate"])
        narr = norm_ws(r["Final Narrative"])
        if not d or len(narr) < 30:
            stats["sir_no_date_or_narrative"] += 1
            continue
        naics_code = re.sub(r"\D", "", r["Primary NAICS"])
        ind = naics.describe(naics_code)
        if not ind:
            stats["sir_no_naics"] += 1
            continue
        narr = scrub_pii(narr, [r["Employer"], r["Address1"], r["Address2"]])
        coded_narr, n_insp = drop_sentences(narr, SIR_INSPECTION_LEAK)
        stats["sir_inspection_sentences_removed"] += n_insp
        mech, n_cut = mechanism_only(coded_narr)
        stats["sir_severity_sentences_cut"] += n_cut
        ev = SIR_EVENT.get(r["Event"].strip()[:1])
        src = SIR_SOURCE.get(r["Source"].strip()[:1]) if not r["SourceTitle"].strip().lower().startswith(
            "nonclassifiable") else None
        nat = sir_nature(r["NatureTitle"])
        body = sir_body(r["Part of Body"], r["Part of Body Title"])
        st = r["State"].strip().title()
        sector = naics.sector_code(naics_code)
        items.append({
            "id": r["ID"], "upa": r["UPA"], "date": d, "decision_date": d.isoformat(), "month": ym(d),
            "group": "sir" + gid(norm_key(r["Employer"]), r["Zip"][:5]), "naics": naics_code, "sector": sector,
            "sub": naics_code[:3], "industry": ind, "us_state": st, "federal": r["FederalState"].strip() != "0",
            "narr": clip(coded_narr), "mech": clip(mech), "construction": sector == "23",
            "y": {"injury_event_group": ev, "injury_source_group": src, "injury_nature_group": nat,
                  "injured_body_region": body,
                  "onsite_inspection_opened": "yes" if r["Inspection"].strip() else "no",
                  "amputation_reported": "yes" if float(r["Amputation"] or 0) >= 1 else "no",
                  "inpatient_hospitalization_reported": "yes" if float(r["Hospitalized"] or 0) >= 1 else "no"},
            "raw": {"event": norm_ws(r["EventTitle"]), "source": norm_ws(r["SourceTitle"]),
                    "nature": norm_ws(r["NatureTitle"]), "part_of_body": norm_ws(r["Part of Body Title"]),
                    "hospitalized": r["Hospitalized"], "amputation": r["Amputation"],
                    "inspection_linked": bool(r["Inspection"].strip())},
        })
    log(f"sir rows {len(rows)} -> items {len(items)}")
    return items


def sir_state(it, variant):
    sections = {
        "establishment": {**it["industry"], "us_state": it["us_state"],
                          "jurisdiction": "federal OSHA" if it["federal"] else "state-plan state (reported to "
                                                                                 "federal OSHA)"},
        "incident": {"month": it["month"], "narrative": it["narr"] if variant == "coded" else it["mech"]},
        "report": {"type": "severe injury report under 29 CFR 1904.39 (amputation, in-patient hospitalization or loss "
                           "of an eye, reported by the employer)" if variant == "coded" else
                           "severe injury report under 29 CFR 1904.39; the injury-outcome sentences of the narrative "
                           "are withheld"},
    }
    return sections


# --------------------------------------------------------------------------- NTSB
NTSB_CAUSE = {"01": "aircraft", "02": "personnel", "03": "environment", "04": "organizational", "05": "not_determined"}
NTSB_CAUSE_CRIT = {
    "aircraft": "Aircraft: findings about the aircraft -- its systems, structure, powerplant and fluids, or its "
        "operation, performance and control parameters (NTSB codes 'directional control not maintained' and "
        "'airspeed not maintained' here).",
    "environment": "Environmental issues: weather and light conditions (wind, visibility, density altitude, icing), "
        "terrain, obstacles, runway or airport conditions, and birds or wildlife.",
    "not_determined": "Not determined: the NTSB's finding category 'Not determined' -- the investigation could not "
        "establish the reason.",
    "organizational": "Organizational issues: policies, procedures, oversight or training of the operator, the "
        "maintenance organization, another company, or regulatory oversight.",
    "personnel": "Personnel issues: actions, decisions, experience, knowledge, or physical or psychological condition "
        "of the pilot, crew, maintenance personnel, controllers or others involved.",
}
NTSB_DAMAGE = {"MINR": 0, "SUBS": 1, "DEST": 2}
NTSB_DAMAGE_LEVELS = [
    "minor: damage that does not meet the definition of substantial",
    "substantial: damage or failure that adversely affects the structural strength, performance or flight "
    "characteristics and would normally require major repair or replacement of the affected component (49 CFR 830.2)",
    "destroyed: the aircraft is destroyed or damaged beyond economic repair",
]
NTSB_INJURY = {"NONE": 0, "MINR": 1, "SERS": 2, "FATL": 3}
NTSB_INJURY_LEVELS = [
    "none: nobody aboard or on the ground was injured",
    "minor: the worst injury is neither serious nor fatal",
    "serious: the worst injury required hospitalization for more than 48 hours, or involved a fracture (except simple "
    "fractures of fingers, toes or nose), severe bleeding, nerve, muscle or tendon damage, an internal organ, or "
    "second- or third-degree burns or burns over more than 5% of the body (49 CFR 830.2)",
    "fatal: at least one person died of injuries within 30 days of the accident (49 CFR 830.2)",
]
NTSB_OUTCOME = re.compile(r"\b(injur\w*|fatal\w*|kill\w*|died|death\w*|deceased|dead|damag\w*|destroy\w*|"
                          r"substantial\w*|uninjured|hospital\w*|surviv\w*|wreck\w*|burn\w*|consumed|fire\w*|"
                          r"postcrash|post-crash|postaccident fire|autopsy|toxicolog\w*|medical examiner|coroner|"
                          r"fractur\w*|trauma\w*|ambulance|rescue\w*)\b", re.I)
NTSB_CAUSE_WORDS = re.compile(r"\b(probable cause|contribut\w* to (?:the|this) (?:accident|incident)|NTSB "
                              r"determin\w*|was caused by|the cause of the (?:accident|incident)|reason for the "
                              r"(?:accident|incident) could not)\b", re.I)
NTSB_AUDIT = re.compile(r"^(injur\w*|fatal\w*|kill\w*|died|death|dead|damag\w*|destroy\w*|substantial\w*|"
                        r"wreck\w*|burn\w*|fire\w*|consumed|postcrash|autopsy|toxicology|coroner|fractur\w*|"
                        r"survivor\w*|ambulance|hospital\w*|rescue\w*|serious\w*|minor)$")
REG_MARK = re.compile(r"\bN\d{1,5}[A-Z]{0,2}\b")
NTSB_FOREIGN = re.compile(r"jurisdiction of|further information|may be obtained from|information pertaining to|"
                          r"\bTel(?:ephone)?\b|\bFax\b|Apartado|Postal|C[oó]digo|e-?mail|\[link\]|"
                          r"accident investigation (?:authority|board|bureau)|this report is for informational", re.I)


def ntsb_questions():
    lam = LAMBDA
    return {
        "cause_area": Q("cause_area", "choice",
                        "Which area does the first finding cited in the NTSB's probable cause for this accident "
                        "fall in?", NTSB_CAUSE_CRIT, "expert",
                        f"{lam}*onehot(category of the first cause finding: Findings.Cause_Factor='C', else the first "
                        f"finding with cm_inPC; aircraft 1) + {1 - lam}*LOO Dirichlet base rate (global > aircraft "
                        f"category > category x operation (FAR part) > ... x phase of flight)"),
        "aircraft_damage_level": Q("aircraft_damage_level", "score",
                                   "How severe is the damage to the aircraft, as the NTSB records it?",
                                   NTSB_DAMAGE_LEVELS, "expert",
                                   f"{lam}*onehot(aircraft.damage, aircraft 1) + {1 - lam}*LOO Dirichlet base rate "
                                   f"(global > category > category x phase of flight)"),
        "highest_injury_level": Q("highest_injury_level", "score",
                                  "What is the highest level of injury in this accident, as the NTSB records it?",
                                  NTSB_INJURY_LEVELS, "expert",
                                  f"{lam}*onehot(events.ev_highest_injury) + {1 - lam}*LOO Dirichlet base rate "
                                  f"(global > category > category x operation)"),
    }


def ntsb_dict(ex):
    dd = defaultdict(dict)
    for r in csv.DictReader(open(ex / "ntsb" / "eADMSPUB_DataDictionary.csv", encoding="utf-8", errors="replace")):
        if r.get("code_iaids"):
            dd[r["Column"]][r["code_iaids"].strip()] = r["meaning"].strip()
    phase = {}
    for r in csv.DictReader(open(ex / "ntsb" / "ct_seqevt.csv", encoding="utf-8", errors="replace")):
        phase[r["code"].strip()] = r["meaning"].strip()
    return dd, phase


MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"


def ntsb_mask(narr, cause_text, stats):
    s_cause = shingles(cause_text, 6)
    out = []
    # report section headers are glued to the next sentence ("HISTORY OF FLIGHTOn April 29, ..."): split them off
    narr = re.sub(r"\b([A-Z][A-Z&/,]+(?:\s+[A-Z&/,]+){1,6})(?=[A-Z][a-z])", r"\1. ", narr)
    sents = sentences(narr)
    first = True
    for s in sents:
        if not re.search(r"[a-z]", s):          # a header line
            continue
        if first:
            first = False
            if re.search(r"\b(19|20)\d\d\b|" + MONTHS, s):
                stats["ntsb_opening_sentence_removed"] += 1
                continue
        if NTSB_CAUSE_WORDS.search(s) or (s_cause and shingles(s, 6) & s_cause):
            stats["ntsb_cause_sentences_removed"] += 1
            continue
        if NTSB_OUTCOME.search(s):
            stats["ntsb_outcome_sentences_removed"] += 1
            continue
        if NTSB_FOREIGN.search(s):
            stats["ntsb_foreign_authority_boilerplate_removed"] += 1
            continue
        if re.search(r"was operated (as|under)|Title 14 Code of Federal Regulations|Part \d{2,3} .{0,40}flight\.$", s):
            stats["ntsb_operation_boilerplate_removed"] += 1
            continue
        out.append(REG_MARK.sub("[registration]", s))
    return " ".join(out)


def load_ntsb(cache, stats):
    ex = Path(cache) / "extracts"
    dd, phase = ntsb_dict(ex)

    def rows(t):
        return csv.DictReader(open(ex / "ntsb" / f"{t}.csv", encoding="utf-8", errors="replace", newline=""))

    ev = {r["ev_id"]: r for r in rows("events")}
    ac = {r["ev_id"]: r for r in rows("aircraft") if r["Aircraft_Key"] == "1"}
    adm = {r["ev_id"]: r for r in rows("NTSB_Admin")}
    nar = {r["ev_id"]: r for r in rows("narratives") if r["Aircraft_Key"] == "1"}
    fnd = defaultdict(list)
    for r in rows("Findings"):
        if r["Aircraft_Key"] == "1":
            fnd[r["ev_id"]].append(r)
    first_phase = {}
    for r in rows("Events_Sequence"):
        if r["Aircraft_Key"] != "1":
            continue
        k = int(r["Occurrence_No"] or 99)
        if r["ev_id"] not in first_phase or k < first_phase[r["ev_id"]][0]:
            first_phase[r["ev_id"]] = (k, phase.get(r["phase_no"].strip(), ""))
    items = []
    for e, r in ev.items():
        if r["ev_type"] != "ACC":
            stats["ntsb_not_accident"] += 1
            continue
        n, a, ad = nar.get(e), ac.get(e), adm.get(e)
        if not n or not a or not ad or not n["narr_accp"].strip() or not n["narr_cause"].strip():
            stats["ntsb_no_factual_or_cause"] += 1
            continue
        appr = parse_date(ad["approval_date"])
        if not appr:
            stats["ntsb_no_approval_date"] += 1
            continue
        fs = sorted(fnd.get(e, []), key=lambda x: int(x["finding_no"] or 0))
        c = [x for x in fs if x["Cause_Factor"].strip() == "C"] or [x for x in fs if x.get("cm_inPc") == "T"]
        cause = NTSB_CAUSE.get(c[0]["category_no"].strip()) if c else None
        narr = ntsb_mask(norm_ws(n["narr_accp"]), norm_ws(n["narr_cause"]), stats)
        narr = scrub_pii(narr)
        if len(narr) < 150:
            stats["ntsb_narrative_too_short_after_masking"] += 1
            continue
        evd = parse_date(r["ev_date"])
        if appr.isoformat() >= EVAL_START and (not evd or evd.year < 2024):
            stats["ntsb_revised_report_of_old_event_dropped"] += 1   # approval re-dated; first published long ago
            continue
        cat = dd["acft_category"].get(a["acft_category"].strip(), a["acft_category"].strip() or "unknown")
        far = a["far_part"].strip()
        ph = first_phase.get(e, (0, ""))[1].title()
        items.append({
            "id": e, "group": "ntsb" + e, "decision_date": appr.isoformat(), "month": ym(evd),
            "category": cat, "far": far, "phase": ph or "Unknown", "construction": False,
            "narr": clip(narr),
            "aircraft": {"category": cat, "make": a["acft_make"].strip().title(), "model": a["acft_model"].strip(),
                         "amateur_built": a["homebuilt"].strip() == "Y", "engines": a["num_eng"].strip() or None},
            "flight": {"operation": f"14 CFR Part {far.lstrip('0')}" if far.isdigit() else dd["far_part"].get(far, far),
                       "purpose": dd["type_fly"].get(a["type_fly"].strip(), a["type_fly"].strip()) or None,
                       "phase_of_flight_at_first_event": ph or None},
            "conditions": {"light": dd["light_cond"].get(r["light_cond"].strip(), r["light_cond"].strip()) or None,
                           "weather": {"VMC": "visual meteorological conditions",
                                       "IMC": "instrument meteorological conditions"}.get(
                               r["wx_cond_basic"].strip(), None),
                           "wind_speed_kts": r["wind_vel_kts"].strip() or None,
                           "temperature_c": r["wx_temp"].strip() or None},
            "location": {"country": r["ev_country"].strip() or None, "us_state": r["ev_state"].strip() or None},
            "y": {"cause_area": cause,
                  "aircraft_damage_level": NTSB_DAMAGE_LEVELS[NTSB_DAMAGE[a["damage"].strip()]]
                  if a["damage"].strip() in NTSB_DAMAGE else None,
                  "highest_injury_level": NTSB_INJURY_LEVELS[NTSB_INJURY[r["ev_highest_injury"].strip()]]
                  if r["ev_highest_injury"].strip() in NTSB_INJURY else None},
            "raw": {"first_cause_finding": c[0]["finding_description"].strip() if c else None,
                    "damage": a["damage"].strip(), "highest_injury": r["ev_highest_injury"].strip(),
                    "ntsb_no": r["ntsb_no"]},
        })
    log(f"ntsb accidents -> items {len(items)}")
    return items


def ntsb_state(it):
    return {"aircraft": {k: v for k, v in it["aircraft"].items() if v not in (None, "")},
            "flight": {k: v for k, v in it["flight"].items() if v},
            "conditions": {k: v for k, v in it["conditions"].items() if v},
            "event": {"month": it["month"], **{k: v for k, v in it["location"].items() if v},
                      "factual_narrative": it["narr"],
                      "note": "opening summary, damage / injury descriptions and probable-cause text withheld"}}


# --------------------------------------------------------------------------- NHTSA
NHTSA_TYPE = {"PE": "Preliminary Evaluation (PE)", "EA": "Engineering Analysis (EA)",
              "RQ": "Recall Query (RQ): a review of the adequacy of an earlier recall"}
NHTSA_RECALL_WORDS = re.compile(r"\b(recall\w*|campaign|\d{2}V-?\d{3}|safety notice|bulletin)\b", re.I)


def nhtsa_questions():
    lam = LAMBDA
    return {
        "investigation_ends_in_recall": Q(
            "investigation_ends_in_recall", "noul", "This defect investigation ends with a safety recall.", None,
            "outcome", f"{lam}*onehot(FLAT_INV CAMPNO present) + {1 - lam}*LOO Dirichlet base rate (global > "
                       f"investigation type > type x top-level component)",
            yes_no=("A recall campaign is recorded as resulting from the investigation.",
                    "The investigation closes with no recall recorded as its result.")),
        "next_regulatory_step": Q(
            "next_regulatory_step", "choice", "How does this preliminary evaluation end?",
            {"recall": "A recall campaign is recorded as resulting from the preliminary evaluation.",
             "escalate_to_engineering_analysis": "No recall at this stage; ODI upgrades the preliminary evaluation "
                                                 "to an engineering analysis (EA) for further study.",
             "close_without_action": "ODI closes the preliminary evaluation with no recall and no upgrade."},
            "outcome", f"{lam}*onehot(CAMPNO -> recall; else an EA referenced in the PE closing summary or an EA "
                       f"summary citing this PE -> escalate; else close) + {1 - lam}*LOO Dirichlet base rate "
                       f"(global > top-level component)"),
    }


def load_nhtsa(cache, stats):
    ex = Path(cache) / "extracts"
    acts = list(read_jsonl_gz(ex / "nhtsa_inv.jsonl.gz"))
    by_id = {a["action"]: a for a in acts}
    cmpl = defaultdict(list)
    for c in read_jsonl_gz(ex / "nhtsa_complaints.jsonl.gz"):
        cmpl[(c["make"].upper(), c["model"].upper(), c["year"])].append(c)

    def pe_ref(a):   # "PE05017" -> regex for "PE05-017" / "PE 05-017" / "PE05017"
        return re.compile(rf"\bPE\s?{a[2:4]}-?{a[4:7]}\b", re.I)
    ea_refs = {a["action"]: a["summary"] for a in acts if a["action"].startswith("EA")}
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    escalated = set()
    for a in acts:
        if not a["action"].startswith("PE"):
            continue
        pat = pe_ref(a["action"])
        for ea, summ in ea_refs.items():
            if pat.search(summ or ""):
                escalated.add(a["action"])
                parent[find(ea)] = find(a["action"])
        if re.search(r"upgrad\w* (?:this |the )?(?:PE |investigation |preliminary evaluation )?(?:to|into) (?:an? )?"
                     r"(?:EA|engineering analysis)|\bEA\s?\d{2}-?\d{3}\b", a["summary"] or "", re.I):
            escalated.add(a["action"])
            for m in re.finditer(r"\bEA\s?(\d{2})-?(\d{3})\b", a["summary"] or "", re.I):
                ea = f"EA{m.group(1)}{m.group(2)}"
                if ea in by_id:
                    parent[find(ea)] = find(a["action"])
    items = []
    for a in acts:
        cd, od = parse_date(a["cdate"]), parse_date(a["odate"])
        if not cd or not od:
            stats["nhtsa_open_or_undated"] += 1
            continue
        typ = a["action"][:2]
        comps = Counter(comp_top(r["compname"]) for r in a["rows"])
        top = comps.most_common(1)[0][0] if comps else "UNKNOWN"
        tops = set(comps)
        seen, matched = set(), []
        for r in a["rows"]:
            for c in cmpl.get((r["make"].upper(), r["model"].upper(), r["year"]), []):
                if c["odino"] in seen or comp_top(c["compdesc"]) not in tops or not c["ldate"] or c["ldate"] >= a["odate"]:
                    continue
                seen.add(c["odino"])
                matched.append(c)
        mm = defaultdict(set)
        for r in a["rows"]:
            if r["year"].isdigit() and r["year"] != "9999":
                mm[(r["make"].title(), r["model"])].add(int(r["year"]))
            else:
                mm[(r["make"].title(), r["model"])]
        vehicles = []
        for (mk, mo), ys in sorted(mm.items(), key=lambda kv: -len(kv[1]))[:8]:
            yr = f" ({min(ys)}-{max(ys)})" if len(ys) > 1 else f" ({min(ys)})" if ys else ""
            vehicles.append(f"{mk} {mo}{yr}".strip())
        if len(mm) > 8:
            vehicles.append(f"... and {len(mm) - 8} more make/model combinations")
        examples = []
        for c in sorted(matched, key=lambda c: c["ldate"], reverse=True):
            t = norm_ws(c["cdescr"])
            if not t or NHTSA_RECALL_WORDS.search(t):
                continue
            examples.append(f"{c['ldate'][:4]}-{c['ldate'][4:6]}, {c['compdesc'].title()}: " +
                            clip(scrub_pii(t), 320))
            if len(examples) >= 3:
                break
        subj = norm_ws(a["subject"])
        if NHTSA_RECALL_WORDS.search(subj):
            stats["nhtsa_subject_mentions_recall_dropped"] += 1
            continue
        if a["campno"]:
            nxt = "recall"
        elif a["action"] in escalated:
            nxt = "escalate_to_engineering_analysis"
        else:
            nxt = "close_without_action"
        items.append({
            "id": a["action"], "group": "nhtsa" + find(a["action"]), "decision_date": cd.isoformat(),
            "construction": False, "type": typ, "top": top,
            "sections": {
                "investigation": {"type": NHTSA_TYPE[typ], "opened": ym(od), "subject": subj,
                                  "component": norm_ws(a["rows"][0]["compname"]).title() if a["rows"] else None,
                                  "manufacturer": norm_ws(a["mfr"]).title() or None},
                "vehicles": {"make_model_years": vehicles, "make_model_year_rows": len(a["rows"])},
                "complaints_before_opening": {
                    "count": len(matched),
                    "involving_crash": sum(c["crash"] == "Y" for c in matched),
                    "involving_fire": sum(c["fire"] == "Y" for c in matched),
                    "persons_injured": sum(int(c["injured"] or 0) for c in matched if c["injured"].isdigit()),
                    "deaths": sum(int(c["deaths"] or 0) for c in matched if c["deaths"].isdigit()),
                    "latest_examples": examples},
            },
            "y": {"investigation_ends_in_recall": "yes" if a["campno"] else "no",
                  "next_regulatory_step": nxt if typ == "PE" else None},
            "raw": {"action": a["action"], "campno": a["campno"], "cdate": a["cdate"]},
        })
    log(f"nhtsa items {len(items)}")
    return items


# --------------------------------------------------------------------------- DOL OSHA enforcement
DOL_INSP_TYPE = {"A": "accident", "B": "complaint", "C": "referral", "D": "monitoring", "E": "variance",
                 "F": "follow-up", "G": "unprogrammed related", "H": "planned (programmed)", "I": "programmed related",
                 "J": "unprogrammed other", "K": "programmed other", "L": "other", "M": "fatality / catastrophe",
                 "N": "unprogrammed emphasis"}
DOL_OWNER = {"A": "private", "B": "local government", "C": "state government", "D": "federal"}
DOL_CITE_CRIT = {
    "none_cited": "No citation is issued: the inspection closes with no violation cited.",
    "other_than_serious": "The most severe item is other-than-serious: a violation directly related to job safety and "
                          "health that probably would not cause death or serious physical harm.",
    "serious": "The most severe item is serious: a substantial probability that death or serious physical harm could "
               "result from a condition the employer knew or should have known about (OSH Act section 17(k)).",
    "repeat": "At least one item is a repeat violation and none is willful: the employer was cited before, under a "
              "final order, for the same or a substantially similar condition.",
    "willful": "At least one item is willful: the employer intentionally and knowingly violated a requirement, or "
               "acted with plain indifference to employee safety and health.",
}
PENALTY_LEVELS = ["$0: no penalty is proposed", "$1 to $4,999", "$5,000 to $14,999", "$15,000 to $49,999",
                  "$50,000 or more"]
PENALTY_EDGES = [0.0, 5000.0, 15000.0, 50000.0]
DOL_LEAK = re.compile(r"\b(cit(?:ed|ation\w*)|violat\w*|penalt\w*|willful\w*|repeat\w*|serious violation|"
                      r"29 CFR|1926\.\d+|1910\.\d+|fined|proposed|settle\w*|contest\w*|abate\w*|informal conference)\b",
                      re.I)
DOL_AUDIT = re.compile(r"^(cit\w*|violat\w*|penalt\w*|willful|repeat\w*|cfr|fined|settle\w*|contest\w*|abate\w*)$")


def penalty_band(x):
    if x <= 0:
        return 0
    return 1 + sum(x >= e for e in PENALTY_EDGES[1:])


def dol_questions():
    lam = LAMBDA
    base = f"{1 - lam}*LOO Dirichlet base rate (global > NAICS sector > NAICS subsector > subsector x inspection type)"
    return {
        "most_serious_citation_type": Q("most_serious_citation_type", "choice",
                                        "What is the most severe classification among the citation items OSHA issues "
                                        "from this inspection?", DOL_CITE_CRIT, "outcome",
                                        f"{lam}*onehot(max of non-deleted violation types W > R > S > O; none if no "
                                        f"item; skipped if any item is 'U' unclassified and none willful) + " + base),
        "initial_penalty_band": Q("initial_penalty_band", "score",
                                  "How large is the total penalty OSHA initially proposes for this inspection?",
                                  PENALTY_LEVELS, "outcome",
                                  f"{lam}*onehot(band of the sum of non-deleted INITIAL_PENALTY; nominal dollars) + "
                                  + base),
        "penalty_reduced_after_issuance": Q(
            "penalty_reduced_after_issuance", "noul",
            "The penalty is reduced after the citations are issued, before they become final.", None, "outcome",
            f"{lam}*onehot(sum CURRENT_PENALTY < sum INITIAL_PENALTY; only inspections with a proposed penalty whose "
            f"penalized items all have a final order date) + " + base,
            yes_no=("The final total penalty is lower than the initially proposed total (an informal settlement, a "
                    "formal settlement or a contest reduced it).",
                    "The final total penalty equals (or exceeds) the initially proposed total.")),
    }


def size_bucket(n):
    try:
        n = int(float(n))
    except (TypeError, ValueError):
        return None
    for hi, lab in ((10, "1-10"), (25, "11-25"), (100, "26-100"), (250, "101-250"), (1000, "251-1,000")):
        if n <= hi:
            return lab
    return "more than 1,000"


def load_dol(cache, naics, stats):
    ex = Path(cache) / "extracts"
    insp = {r["activity_nr"]: r for r in read_jsonl_gz(ex / "dol_inspections.jsonl.gz")}
    viol = {r["a"]: r for r in read_jsonl_gz(ex / "dol_violations.jsonl.gz")}
    import bisect
    hist_o, hist_s = defaultdict(list), defaultdict(list)
    for h in read_jsonl_gz(ex / "dol_history.jsonl.gz"):
        hist_o[h["g"]].append(h["o"])
        if h.get("s"):
            hist_s[h["g"]].append((h["o"], h["s"]))
    for g in hist_o.values():
        g.sort()
    for g in hist_s.values():
        g.sort()
    big = {g for g, v in hist_o.items() if len(v) > 300}
    stats["dol_history_groups_over_300_inspections_ignored"] = len(big)
    acc_by_insp = {}
    for a in read_jsonl_gz(ex / "dol_accidents.jsonl.gz"):
        best = max(a["insp"].items(), key=lambda kv: (kv[1], -int(kv[0])))[0]
        if best in insp and parse_date(a["event_date"]) >= date(2012, 1, 1):
            prev = acc_by_insp.get(best)
            if prev is None or a["event_date"] > prev["event_date"]:
                acc_by_insp[best] = a
    data_end = date(2026, 9, 30)
    items = []
    for an, r in insp.items():
        od = parse_date(r["open_date"])
        if not od or r["insp_scope"] == "D":
            stats["dol_no_inspection_conducted"] += 1
            continue
        v = viol.get(an)
        a = acc_by_insp.get(an)
        kind = "fatality" if a else ("construction" if r["naics_code"].startswith("23") else None)
        if kind is None:
            continue
        close = parse_date(r["close_case_date"])
        if v and v.get("issue"):
            decision = v["issue"]
            if (parse_date(decision) - od).days > 400:   # OSH Act 9(c): citations within 6 months; later = amended
                stats["dol_first_issuance_over_400_days_after_opening_dropped"] += 1
                continue
        elif close and (data_end - od).days > 200:
            decision = close.isoformat()
        else:
            stats["dol_outcome_not_yet_known"] += 1
            continue
        types = (v or {}).get("types", {})
        if types.get("W"):
            cite = "willful"
        elif types.get("U"):
            cite = None
            stats["dol_unclassified_items_question_skipped"] += 1
        elif types.get("R"):
            cite = "repeat"
        elif types.get("S"):
            cite = "serious"
        elif types.get("O"):
            cite = "other_than_serious"
        elif not v or not v["n"]:
            cite = "none_cited"
        else:
            cite = None
            stats["dol_unknown_violation_types"] += 1
        init = (v or {}).get("init", 0.0)
        reduced = None
        if v and v["n_pen"] > 0 and v["n_final"] >= v["n_pen"]:
            reduced = "yes" if v["cur"] < v["init"] - 0.5 else "no"
            if decision >= EVAL_START:
                # censored: by the data date only quickly-finalised EVAL cases have an answer (contested ones run
                # for years), so the eval answer distribution is biased; not asked on EVAL items
                reduced = None
                stats["dol_penalty_reduction_withheld_on_eval_censored"] += 1
        g = r["group"]
        lo, hi = (od - timedelta(days=5 * 365)).isoformat(), od.isoformat()
        if g in big:
            n_prior = n_prior_serious = None
        else:
            os_ = hist_o.get(g, [])
            n_prior = bisect.bisect_left(os_, hi) - bisect.bisect_left(os_, lo)
            n_prior_serious = sum(1 for o, s in hist_s.get(g, []) if lo <= o < hi and s < hi)
        naics_code = re.sub(r"\D", "", r["naics_code"])
        ind = naics.describe(naics_code)
        if not ind:
            stats["dol_no_naics"] += 1
            continue
        sector = naics.sector_code(naics_code)
        est = {**ind, "us_state": US_STATES.get(r["site_state"], r["site_state"]) or None,
               "employees": size_bucket(r["nr_in_estab"]), "ownership": DOL_OWNER.get(r["owner_type"]),
               "union_represented": "yes" if r["union_status"] in ("Y", "U", "A") else "no"}
        inspection = {"type": DOL_INSP_TYPE.get(r["insp_type"], "unknown"),
                      "focus": {"S": "safety", "H": "health"}.get(r["safety_hlth"]),
                      "opened": ym(od), "prior_inspections_5y": n_prior,
                      "prior_inspections_with_serious_willful_or_repeat_5y": n_prior_serious,
                      "advance_notice": {"Y": "yes", "N": "no"}.get(r["adv_notice"])}
        sections = {"employer": {k: x for k, x in est.items() if x}, "inspection": {k: x for k, x in inspection.items()
                                                                                   if x is not None}}
        if a:
            text, n_rm = drop_sentences(a["abstract"], DOL_LEAK)
            stats["dol_abstract_sentences_removed"] += n_rm
            text = scrub_pii(text)
            if len(text) < 80:
                stats["dol_abstract_too_short"] += 1
                continue
            deg = a["degree"]
            inc = {"month": ym(parse_date(a["event_date"])), "narrative": clip(text),
                   "workers_killed": int(deg.get("1", 0)), "workers_hospitalized": int(deg.get("2", 0)),
                   "workers_injured_not_hospitalized": int(deg.get("3", 0))}
            proj = {k: a[k] for k in ("build_stories", "nonbuild_ht") if a.get(k) not in (None, "", "0", "0.0")}
            if proj:
                inc["construction_project"] = {("stories" if k == "build_stories" else "height_ft"):
                                               int(float(x)) for k, x in proj.items()}
            sections = {"incident": inc, **sections}
        items.append({
            "id": an, "group": "dol" + g, "decision_date": decision, "kind": kind, "construction": sector == "23",
            "sector": sector, "sub": naics_code[:3], "insp_type": r["insp_type"], "sections": sections,
            "y": {"most_serious_citation_type": cite, "initial_penalty_band": PENALTY_LEVELS[penalty_band(init)],
                  "penalty_reduced_after_issuance": reduced},
            "raw": {"types": types, "initial_penalty": round(init, 2), "current_penalty": round((v or {}).get("cur", 0), 2),
                    "issuance": (v or {}).get("issue"), "accident_summary_nr": a["summary_nr"] if a else None},
        })
    log(f"dol items {len(items)} ({Counter(i['kind'] for i in items)})")
    return items


# --------------------------------------------------------------------------- generic labelling + writing
def label_items(items, qspecs, levels_fn, stats_q):
    """Fill it['labels'][qid] / it['kinds'][qid] / meta for every item with a y for that qid."""
    import numpy as np
    for qid, q in qspecs.items():
        sel = [it for it in items if it["y"].get(qid) is not None]
        if not sel:
            continue
        K = len(q.keys)
        yi = [q.keys.index(it["y"][qid]) for it in sel]
        lv = [levels_fn(qid, it) for it in sel]
        inc = [it["split"] != "eval" for it in sel]
        p, info = loo_rates(yi, lv, inc, K)
        stats_q[qid] = {"n": len(sel), "K": K, "base_rate_fit": info,
                        "base_rate_calibration_trainval_loo": calib_report(p, yi, inc),
                        "base_rate_on_eval": calib_report(p, yi, [not x for x in inc]),
                        "label_distribution": {k: sum(1 for x in yi if x == j) for j, k in enumerate(q.keys)}}
        for it, y, row in zip(sel, yi, p):
            oh = np.zeros(K)
            oh[y] = 1
            smooth = tidy(dict(zip(q.keys, LAMBDA * oh + (1 - LAMBDA) * row)))
            base = tidy(dict(zip(q.keys, row)))
            if it["split"] == "eval":
                it.setdefault("labels", {})[qid] = tidy(dict(zip(q.keys, oh)))
                it.setdefault("kinds", {})[qid] = "hard"
                it.setdefault("smoothed", {})[qid] = smooth
            else:
                it.setdefault("labels", {})[qid] = smooth
                it.setdefault("kinds", {})[qid] = "soft"
            it.setdefault("base", {})[qid] = base


def make_rec(source, family, state, it, qspecs, qids, extra_meta):
    qs = {qid: qspecs[qid].render() for qid in qids}
    labs = {qid: it["labels"][qid] for qid in qids}
    kinds = {qid: it["kinds"][qid] for qid in qids}
    rec = record(source, state, qs, labs, kinds, None)
    rec["base_id"] = rec["id"]
    rec["license"] = LICENSES[family]
    meta = {"dataset": DATASETS[source], "group": it["group"], "as_of": extra_meta.pop("as_of", None),
            "decision_date": it["decision_date"], "split": it["split"], "split_reason": it["split_reason"],
            "label_source": {q: qspecs[q].source_tag for q in qids},
            "label_method": {q: (qspecs[q].method if it["split"] != "eval" else
                                 "onehot(realised code / outcome); smoothed target in labels_smoothed")
                             for q in qids},
            "lambda": LAMBDA if it["split"] != "eval" else None,
            "base_rate": {q: it["base"][q] for q in qids},
            "realised": {q: it["y"][q] for q in qids},
            "construction": bool(it.get("construction")), **extra_meta}
    if it["split"] == "eval":
        meta["labels_smoothed"] = {q: it["smoothed"][q] for q in qids}
    rec["meta"] = meta
    validate(rec)
    return rec


class Out:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        self.counts = Counter()

    def write(self, rec):
        sp = rec["meta"]["split"]
        key = (sp, rec["source"])
        if key not in self.files:
            p = self.root / sp / (rec["source"].replace("/", "__") + ".jsonl")
            p.parent.mkdir(parents=True, exist_ok=True)
            self.files[key] = p.open("w", encoding="utf-8")
        self.files[key].write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.counts[key] += 1

    def close(self):
        for f in self.files.values():
            f.close()


def probe_and_gate(name, items, qspecs, qids, text_fn, struct_fn, audit, gated, manifest):
    """Run the leak probe on train + val items for each question; return the qids allowed to be written."""
    allowed = []
    for qid in qids:
        sel = [it for it in items if it["y"].get(qid) is not None and it["split"] != "eval"]
        if len(sel) < 200:
            allowed.append(qid)
            continue
        keys = qspecs[qid].keys
        res = leak_probe([text_fn(it) for it in sel], [struct_fn(it) for it in sel],
                         [keys.index(it["y"][qid]) for it in sel], len(keys), keys, [it["group"] for it in sel],
                         audit=audit)
        res["gated"] = qid in gated
        manifest.setdefault("leak_probe", {}).setdefault(name, {})[qid] = res
        log(f"  probe {name}/{qid}: bow {res['bow']} | structured {res['structured']} | flagged {res['flagged']}")
        if qid in gated and not res["passes"]:
            log(f"  GATE: {name}/{qid} FAILS the leak audit -> not written")
            continue
        allowed.append(qid)
    return allowed


def emit_source(source, family, items, qspecs, qids, state_fn, rng, out, manifest, min_q=1, as_of_fn=None):
    by_split = defaultdict(list)
    for it in items:
        if sum(1 for q in qids if q in it.get("labels", {})) >= min_q:
            by_split[it["split"]].append(it)
    stats = Counter()
    seen_states = set()
    sh_df, sh_idx, sh_docs = Counter(), defaultdict(set), []
    for sp in ("train", "val", "eval"):
        if sp == "eval":   # index the train + val free text once, for the near-duplicate rule below
            for d in sh_docs:
                sh_df.update(d)
            for i, d in enumerate(sh_docs):
                for x in d:
                    if sh_df[x] <= 20:
                        sh_idx[x].add(i)
        chosen = cap_items(sorted(by_split[sp], key=lambda x: x["id"]), CAPS[source][sp], rng)
        for it in chosen:
            surface = pick_surface(source + it["id"])
            title, sections = state_fn(it)
            state = render_state(title, sections, surface)
            if sp == "eval" and state in seen_states:
                stats[("eval", "dropped_identical_to_train_state")] += 1
                continue
            sh = shingles(state_text_fields(state))
            if sp == "eval" and sh:
                cnt = Counter(i for x in sh for i in sh_idx.get(x, ()))
                if cnt and max(cnt.values()) / len(sh) > 0.5:
                    stats[("eval", "dropped_near_duplicate_of_train_state")] += 1
                    continue
            elif sp != "eval":
                sh_docs.append(sh)
            seen_states.add(state)
            qs = [q for q in qids if q in it.get("labels", {})]
            rec = make_rec(source, family, state, it, qspecs, qs,
                           {"as_of": as_of_fn(it) if as_of_fn else None, "surface": surface,
                            "source_id": it["id"], "raw": it.get("raw"), "sector": it.get("sector")})
            out.write(rec)
            stats[(sp, "records")] += 1
            stats[(sp, "construction")] += bool(it.get("construction"))
            stats[(sp, "units")] += len(qs)
    manifest.setdefault("sources", {})[source] = {
        "available": {sp: len(v) for sp, v in by_split.items()},
        "written": {sp: {"records": stats[(sp, "records")], "units": stats[(sp, "units")],
                         "construction_records": stats[(sp, "construction")]} for sp in ("train", "val", "eval")},
        "eval_dropped_identical_to_train_state": stats[("eval", "dropped_identical_to_train_state")],
        "eval_dropped_near_duplicate_of_train_state": stats[("eval", "dropped_near_duplicate_of_train_state")],
        "questions": qids, "caps": CAPS[source]}
    log(f"{source}: " + ", ".join(f"{sp} {stats[(sp, 'records')]}" for sp in ("train", "val", "eval")))


# --------------------------------------------------------------------------- build
def build(a):
    cache, outdir = Path(a.cache), Path(a.out)
    if outdir.exists():
        for sp in ("train", "val", "eval"):
            shutil.rmtree(outdir / sp, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    naics = Naics(cache / "extracts" / "naics.json")
    out = Out(outdir)
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "eval_start": EVAL_START, "lambda": LAMBDA,
                "val_pct": VAL_PCT, "licenses": LICENSES, "datasets": DATASETS, "questions": {}, "stats": {}}
    stats = Counter()
    qsets = {}

    # ---- OSHA SIR
    if "osha_sir" in a.sources:
        sq = sir_questions()
        items = assign_splits(load_sir(cache, naics, stats), stats)

        def lv(qid, it):
            if qid == "onsite_inspection_opened":
                ev = it["y"]["injury_event_group"] or "unknown"
                return (it["sector"], it["sector"] + "|" + ev, it["sector"] + "|" + ev + "|" + it["us_state"])
            if qid in ("amputation_reported", "inpatient_hospitalization_reported"):
                ev = it["y"]["injury_event_group"] or "unknown"
                return (it["sector"], it["sector"] + "|" + ev)
            return (it["sector"], it["sub"])
        qstats = {}
        label_items(items, sq, lv, qstats)
        manifest["questions"]["osha_sir"] = qstats

        def struct(it):
            return [f"sector={it['sector']}", f"sub={it['sub']}", f"state={it['us_state']}",
                    f"month={it['month'][5:]}", f"fed={it['federal']}"]
        coded_q = ["injury_event_group", "injury_source_group", "injury_nature_group", "injured_body_region",
                   "onsite_inspection_opened"]
        sev_q = ["amputation_reported", "inpatient_hospitalization_reported"]
        coded_ok = probe_and_gate("osha_sir/coded", items, sq, coded_q, lambda it: it["narr"], struct,
                                  SIR_INSPECTION_LEAK_AUDIT, {"onsite_inspection_opened"}, manifest)
        sev_items = [it for it in items if len(it["mech"].split()) >= 8]
        stats["sir_severity_too_short_after_cut"] = len(items) - len(sev_items)
        sev_ok = probe_and_gate("osha_sir/severity", sev_items, sq, sev_q, lambda it: it["mech"], struct,
                                SEVERITY_AUDIT, set(sev_q), manifest)

        def coded_state(it):
            return "OSHA severe injury report", sir_state(it, "coded")

        def sev_state(it):
            return "OSHA severe injury report (outcome withheld)", sir_state(it, "severity")
        emit_source("osha_sir/coded", "osha_sir", items, sq, coded_ok, coded_state, rng, out, manifest, min_q=3,
                    as_of_fn=lambda it: it["decision_date"])
        if sev_ok:
            emit_source("osha_sir/severity", "osha_sir", sev_items, sq, sev_ok, sev_state, rng, out, manifest,
                        as_of_fn=lambda it: it["decision_date"])
        qsets["osha_sir/coded"], qsets["osha_sir/severity"] = coded_ok, sev_ok

    # ---- NTSB
    if "ntsb" in a.sources:
        nq = ntsb_questions()
        items = assign_splits(load_ntsb(cache, stats), stats)

        def lv(qid, it):
            if qid == "cause_area":
                return (it["category"], it["category"] + "|" + it["far"], it["category"] + "|" + it["far"] + "|" + it["phase"])
            if qid == "aircraft_damage_level":
                return (it["category"], it["category"] + "|" + it["phase"])
            return (it["category"], it["category"] + "|" + it["far"])
        qstats = {}
        label_items(items, nq, lv, qstats)
        manifest["questions"]["ntsb"] = qstats

        def struct(it):
            return [f"cat={it['category']}", f"far={it['far']}", f"phase={it['phase']}",
                    f"wx={it['conditions']['weather']}", f"light={it['conditions']['light']}",
                    f"purpose={it['flight']['purpose']}", f"amateur={it['aircraft']['amateur_built']}"]
        ok = probe_and_gate("ntsb/aviation", items, nq, list(nq), lambda it: it["narr"], struct, NTSB_AUDIT,
                            {"aircraft_damage_level", "highest_injury_level"}, manifest)
        emit_source("ntsb/aviation", "ntsb", items, nq, ok, lambda it: ("NTSB aviation accident record",
                                                                         ntsb_state(it)), rng, out, manifest,
                    as_of_fn=lambda it: it["month"])
        qsets["ntsb/aviation"] = ok

    # ---- NHTSA
    if "nhtsa" in a.sources:
        hq = nhtsa_questions()
        items = assign_splits(load_nhtsa(cache, stats), stats)

        def lv(qid, it):
            if qid == "investigation_ends_in_recall":
                return (it["type"], it["type"] + "|" + it["top"])
            return (it["top"],)
        qstats = {}
        label_items(items, hq, lv, qstats)
        manifest["questions"]["nhtsa"] = qstats
        emit_source("nhtsa/investigations", "nhtsa", items, hq, list(hq),
                    lambda it: ("NHTSA defect investigation at opening", it["sections"]), rng, out, manifest,
                    as_of_fn=lambda it: it["sections"]["investigation"]["opened"])
        qsets["nhtsa/investigations"] = list(hq)

    # ---- DOL OSHA enforcement
    if "dol" in a.sources:
        dq = dol_questions()
        items = assign_splits(load_dol(cache, naics, stats), stats)

        def lv(qid, it):
            return (it["sector"], it["sub"], it["sub"] + "|" + it["insp_type"])
        qstats = {}
        label_items(items, dq, lv, qstats)
        manifest["questions"]["osha_enforcement"] = qstats
        fat = [it for it in items if it["kind"] == "fatality"]
        con = [it for it in items if it["kind"] == "construction"]

        def struct(it):
            ins = it["sections"]["inspection"]
            return [f"sector={it['sector']}", f"sub={it['sub']}", f"type={it['insp_type']}",
                    f"size={it['sections']['employer'].get('employees')}",
                    f"prior={min(ins.get('prior_inspections_5y') or 0, 5)}",
                    f"priorS={min(ins.get('prior_inspections_with_serious_willful_or_repeat_5y') or 0, 3)}"]
        ok_f = probe_and_gate("osha_enforcement/fatality_investigations", fat, dq, list(dq),
                              lambda it: it["sections"]["incident"]["narrative"], struct, DOL_AUDIT, set(dq), manifest)
        emit_source("osha_enforcement/fatality_investigations", "osha_enforcement", fat, dq, ok_f,
                    lambda it: ("OSHA fatality / catastrophe investigation at opening", it["sections"]), rng, out,
                    manifest, as_of_fn=lambda it: it["sections"]["inspection"]["opened"])
        emit_source("osha_enforcement/construction_inspections", "osha_enforcement", con, dq, list(dq),
                    lambda it: ("OSHA construction inspection at opening", it["sections"]), rng, out, manifest,
                    as_of_fn=lambda it: it["sections"]["inspection"]["opened"])
        qsets["osha_enforcement/fatality_investigations"] = ok_f
        qsets["osha_enforcement/construction_inspections"] = list(dq)

    out.close()
    errs = bench_overlap_errors(qsets)
    manifest["exclusion_errors"] = errs
    if errs:
        raise SystemExit(f"exclusion errors: {errs}")
    manifest["stats"] = dict(stats)
    manifest["files"] = {f"{sp}/{src.replace('/', '__')}.jsonl": n for (sp, src), n in sorted(out.counts.items())}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"build done: {sum(out.counts.values())} records; stats {dict(stats)}")


SIR_INSPECTION_LEAK_AUDIT = re.compile(r"^(osha|inspect\w*|csho|citation\w*|cited|investigat\w*|officer|"
                                       r"compliance|rri|referral)$")


# --------------------------------------------------------------------------- check
def load_cache_features():
    for mod in ("torch", "transformers"):
        try:
            importlib.import_module(mod)
        except ImportError:
            stub = types.ModuleType(mod)
            stub.AutoTokenizer = None
            stub.Tensor = type("Tensor", (), {})
            sys.modules[mod] = stub
    spec = importlib.util.spec_from_file_location("cache_features", HERE / "trainers" / "cache_features.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


OUTCOME_SCANS = {  # what must never appear in a state, per source (re-scanned by `check`)
    "osha_sir/coded": SIR_INSPECTION_LEAK,
    "osha_sir/severity": SEVERITY_WORDS,
    "ntsb/aviation": re.compile(NTSB_OUTCOME.pattern + "|" + NTSB_CAUSE_WORDS.pattern + r"|\bN\d{2,5}[A-Z]{0,2}\b",
                                re.I),
    "nhtsa/investigations": re.compile(r"\b(recall\w*|campaign|\d{2}V-?\d{3})\b", re.I),
    "osha_enforcement/fatality_investigations": DOL_LEAK,
    "osha_enforcement/construction_inspections": DOL_LEAK,
}


def state_text_fields(state):
    """The free-text leaves of a state (JSON) or the whole text surface."""
    try:
        obj = json.loads(state)
    except ValueError:   # form / memo surfaces: the labelled free-text fields
        parts = re.findall(r"(?:^|\n)\s*(?:Narrative|Factual narrative|Latest examples)\s*:\s*(.+)", state)
        parts += re.findall(r"\b[Ss]ubject: ([^;\n]+)", state)
        return " ".join(parts)
    out = []

    def walk(x, k=None):
        if isinstance(x, dict):
            for kk, v in x.items():
                walk(v, kk)
        elif isinstance(x, list):
            for v in x:
                walk(v, k)
        elif isinstance(x, str) and k in ("narrative", "factual_narrative", "latest_examples", "subject"):
            out.append(x)
    walk(obj)
    return " ".join(out)


def check(a):
    import numpy as np
    outdir = Path(a.out)
    cf = load_cache_features()
    recs = defaultdict(list)
    for sp in ("train", "val", "eval"):
        for p in sorted((outdir / sp).glob("*.jsonl")):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                validate(r)
                for qid, q in r["questions"].items():
                    keys, texts, target = cf.question_options(q, r["labels"][qid])
                    assert keys == option_keys(q), (r["id"], qid)
                    assert all(abs(t - r["labels"][qid][k]) < 1e-12 for k, t in zip(keys, target)), (r["id"], qid)
                    assert len(texts) == len(keys) and all(str(t).strip() for t in texts)
                recs[(r["source"], sp)].append(r)
    units = Counter()
    stats_by = defaultdict(lambda: {"maxp": [], "ent": []})
    for u in cf.iter_units(str(outdir), ["train", "val", "eval"], set(), None):
        t = np.asarray(u["target"], float)
        assert abs(t.sum() - 1) < 1e-6 and 2 <= len(t) <= 52, u["record_id"]
        units[(u["source"], u["split"], u["qtype"])] += 1
        ent = float(-(t[t > 0] * np.log(t[t > 0])).sum())
        for key in ((u["split"], u["qtype"]), (u["split"], "all"), (u["source"], u["split"], "all")):
            stats_by[key]["maxp"].append(float(t.max()))
            stats_by[key]["ent"].append(ent)
    label_stats = {"|".join(k): {"n": len(v["maxp"]), "mean_max_p": round(float(np.mean(v["maxp"])), 3),
                                 "mean_entropy_nats": round(float(np.mean(v["ent"])), 3)}
                   for k, v in sorted(stats_by.items())}
    # leakage audits
    leaks = {}
    for (src, sp), rs in recs.items():
        pat = OUTCOME_SCANS.get(src)
        hits = Counter()
        for r in rs:
            text = state_text_fields(r["state"])
            # the severity builder cuts per clause; scan the same units (an object that "broke ... struck the
            # employee's arm" across a comma is mechanism, not outcome)
            units_ = [c for snt in sentences(text) for c in CLAUSE_SPLIT.split(snt)] if src == "osha_sir/severity" \
                else [text]
            for u_ in units_:
                for m in pat.finditer(u_) if pat else []:
                    hits[m.group(0).lower()] += 1
        leaks[f"{src}|{sp}"] = {"records": len(rs), "outcome_word_hits": sum(hits.values()),
                                "top_hits": hits.most_common(8)}
    overlap = {}
    for src in sorted({s for s, _ in recs}):
        tr = recs.get((src, "train"), []) + recs.get((src, "val"), [])
        ev = recs.get((src, "eval"), [])
        tr_states = {r["state"] for r in tr}
        tr_groups = {r["meta"]["group"] for r in tr}
        df = Counter()
        tr_sh = []
        for r in tr:
            s = shingles(state_text_fields(r["state"]))
            tr_sh.append(s)
            df.update(s)
        idx = defaultdict(set)
        for i, s in enumerate(tr_sh):
            for x in s:
                if df[x] <= 20:
                    idx[x].add(i)
        near = half = 0
        for r in ev:
            cnt = Counter()
            es = shingles(state_text_fields(r["state"]))
            for x in es:
                for i in idx.get(x, ()):
                    cnt[i] += 1
            best = max(cnt.values()) if cnt else 0
            near += best >= 3
            half += bool(es) and best / len(es) > 0.5
        overlap[src] = {"eval": len(ev), "identical_state_in_train": sum(r["state"] in tr_states for r in ev),
                        "group_in_train": sum(r["meta"]["group"] in tr_groups for r in ev),
                        "eval_with_>=3_shared_rare_8gram_shingles_with_one_train_state": near,
                        "eval_sharing_>50pct_of_its_8gram_shingles_with_one_train_state": half}
    # sector table
    sectors = defaultdict(Counter)
    for (src, sp), rs in recs.items():
        for r in rs:
            sec = r["meta"].get("sector") or ("23" if r["meta"].get("construction") else None)
            if sec:
                sectors[(src, sp)]["construction (NAICS 23)" if sec == "23" else sec] += 1
    pii = Counter()
    for rs in recs.values():
        for r in rs:
            st = r["state"]
            pii["zip_like"] += bool(re.search(r"\b\d{5}(-\d{4})?\b", st) and re.search(r"\b(zip|postal)\b", st, re.I))
            pii["email"] += bool(re.search(r"\b[\w.+-]+@[\w-]+\.\w+", st))
            pii["phone"] += bool(re.search(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b", st))
    report = {"records": {f"{s}|{sp}": len(v) for (s, sp), v in sorted(recs.items())},
              "units": {"|".join(k): v for k, v in sorted(units.items())},
              "label_stats": label_stats,
              "gold_reference": {"typed_decisions_mean_max_p": 0.659, "typed_decisions_mean_entropy_nats": 0.767,
                                 "by_type_entropy": {"choice": 0.864, "noul": 0.454, "score": 0.930}},
              "leak_scan": leaks, "eval_train_overlap": overlap,
              "sectors": {"|".join(k): dict(v.most_common(12)) for k, v in sorted(sectors.items())},
              "pii_scan": dict(pii),
              "size_mb": round(sum(p.stat().st_size for p in outdir.rglob("*.jsonl")) / 1e6, 1)}
    (outdir / "check_report.json").write_text(json.dumps(report, indent=1))
    rng = random.Random(7)
    samples = []
    for src in sorted({s for s, _ in recs}):
        pool = recs.get((src, "train")) or recs.get((src, "eval"))
        samples.append(rng.choice(pool))
    with open(outdir / "samples.jsonl", "w", encoding="utf-8") as f:
        for r in samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("records", "size_mb", "pii_scan", "eval_train_overlap")}, indent=1))
    bad = {k: v for k, v in leaks.items() if v["outcome_word_hits"]}
    print("leak-scan hits:", json.dumps(bad, indent=1) if bad else "none")
    print("label stats (split|type):", json.dumps({k: v for k, v in label_stats.items() if k.count("|") == 1},
                                                  indent=1))
    return report


# --------------------------------------------------------------------------- self-test
def self_test():
    import numpy as np
    # LOO base rate equals a brute-force recomputation without the item
    rng = random.Random(0)
    y = [rng.randrange(3) for _ in range(60)]
    lv = [(str(rng.randrange(3)),) for _ in range(60)]
    inc = [True] * 50 + [False] * 10
    p, info = loo_rates(y, lv, inc, 3)
    a_ = info["levels"][0]["a"]
    for i in (0, 7, 55):
        ys = [y[j] for j in range(50) if j != i]
        g = np.array([sum(1 for v in ys if v == k) for k in range(3)], float)
        p0 = (g + 1) / (g.sum() + 3)
        cell = np.array([sum(1 for j in range(50) if j != i and lv[j] == lv[i] and y[j] == k) for k in range(3)], float)
        exp = (cell + a_ * p0) / (cell.sum() + a_)
        assert np.allclose(p[i], exp), (i, p[i], exp)
    assert np.allclose(p.sum(1), 1)
    # mechanism-only cut
    t, n = mechanism_only("An employee's fingers entered a running machine and contacted the blade, resulting in "
                          "amputations of parts of two fingers. He was hospitalized overnight.")
    assert "amput" not in t.lower() and "hospital" not in t.lower() and "running machine" in t, t
    assert not SEVERITY_WORDS.search(t)
    t, _ = mechanism_only("A worker was painting a wall when the scaffold plank broke. He fell 12 feet to the ground.")
    assert "painting" in t and "fell 12 feet" in t, t
    # nature / body mapping
    assert sir_nature("Fractures") == "fracture" and sir_nature("Amputations") == "amputation_or_avulsion"
    assert sir_nature("Fractures and dislocations") == "other_injury"
    assert sir_nature("Heat (thermal) burns, unspecified") == "burn_or_electrical"
    assert sir_nature("Effects of heat and light, unspecified") == "heat_cold_or_toxic_effect"
    assert sir_body("4422", "Fingertip(s)") == "hand_or_fingers" and sir_body("51", "Hip joint(s)") == "trunk"
    # tidy + question round trip through cache_features
    cf = load_cache_features()
    for qs in (sir_questions(), ntsb_questions(), nhtsa_questions(), dol_questions()):
        for qid, q in qs.items():
            lab = tidy({k: 1.0 / len(q.keys) + (0.01 if i == 0 else 0) for i, k in enumerate(q.keys)})
            assert abs(sum(lab.values()) - 1) < 1e-9
            keys, texts, target = cf.question_options(q.render(), lab)
            assert keys == q.keys and target == [lab[k] for k in keys], qid
            if q.type == "choice":
                assert keys == sorted(keys)
    assert not bench_overlap_errors({"x": list(sir_questions()) + list(ntsb_questions()) + list(nhtsa_questions())
                                     + list(dol_questions())})
    # surfaces
    secs = {"b": {"z": 1, "a": "x"}, "a": {"narrative": "Some text " * 30}}
    js = render_state("T", secs, "json")
    assert js == json.dumps(secs, sort_keys=True) and ", " in js and ": " in js
    assert "Narrative:" in render_state("T", secs, "memo") and "NARRATIVE" not in render_state("T", secs, "form")
    # PII scrub
    s = scrub_pii("Call Mr. John Smith at 555-123-4567 or js@x.com about ACME ROOFING LLC", ["Acme Roofing LLC"])
    assert "Smith" not in s and "555" not in s and "@" not in s and "ACME" not in s, s
    # leak probe smoke
    texts = ["the cat sat amputated"] * 50 + ["the dog ran"] * 50
    res = leak_probe(texts, [["a"]] * 100, [0] * 50 + [1] * 50, 2, ["yes", "no"], [str(i) for i in range(100)],
                     audit=SEVERITY_AUDIT, max_n=100)
    assert not res["passes"] and "amputated" in res["flagged"], res
    # penalty bands
    assert [penalty_band(x) for x in (0, 1, 4999, 5000, 14999, 15000, 49999, 50000)] == [0, 1, 1, 2, 2, 3, 3, 4]
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--cache", default="/tmp/safety_jev_cache")
    b = sub.add_parser("build")
    b.add_argument("--cache", default="/tmp/safety_jev_cache")
    b.add_argument("--out", default="/tmp/safety_jev")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sources", default="osha_sir,ntsb,nhtsa,dol")
    c = sub.add_parser("check")
    c.add_argument("--out", default="/tmp/safety_jev")
    sub.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a)
    elif a.cmd == "build":
        a.sources = {s.strip() for s in a.sources.split(",") if s.strip()}
        build(a)
    elif a.cmd == "check":
        check(a)
    else:
        self_test()


if __name__ == "__main__":
    main()
