#!/usr/bin/env python3
"""
build_wikidata_jev.py -- Wikidata / Wikipedia decision datasets for jev, in the jev_data record schema
(build_jev_data.record / validate; every record reads back through trainers/cache_features.iter_units).

Two builders share this file, each in its own section with its own subcommands:
  shapes 1-2  edit-history outcomes; Wikidata deletion / property-proposal decisions   (section "SHAPES 1-2")
  shapes 3-4  exact comparisons computed from Wikidata claims; classification of masked
              English Wikipedia leads into fixed Wikidata taxonomies                   (section "SHAPES 3-4")
Docs: research/wikidata_pipeline.md.

Layout: /tmp/wikidata_jev/<source>/{train,val,eval,...}/<source>.jsonl + <source>/manifest.json.

How the sections plug in: each section appends a registrar to REGISTRARS (it adds its subparsers and sets
`func` on each) and an offline self-test to SELF_TESTS. `--self-test` runs every registered self-test.
Keep the `if __name__ == "__main__"` block LAST so every section is registered before main() runs.

  python3 build_wikidata_jev.py --self-test          # offline fixtures, every section
  python3 build_wikidata_jev.py <subcommand> -h
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))

OUT_ROOT = "/tmp/wikidata_jev"
REGISTRARS = []    # callables(subparsers) -> None; each sub-parser sets `func` via set_defaults(func=...)
SELF_TESTS = []    # callables() -> None; raise / assert on failure


# =========================================================================== SHAPES 1-2
# Shape 1 (outcomes over time from Wikidata edit history) and shape 2 (real decisions on Wikidata's own process pages).
# Owner: the shapes 1-2 builder. Everything here is prefixed S12_ / s12_ / s1_ / s2_ so the sections cannot collide.
#
# Subcommands (python3 build_wikidata_jev.py <cmd> -h):
#   s1-office-end     "According to Wikidata, will <person>'s tenure as <position> end in (T1, T2]?"       P39 end time
#   s1-athlete-team   "... will <athlete>'s spell at <team> end in (T1, T2]?"                              P54 end time
#   s1-org-dissolved  "... will <organisation> get a dissolution date (P576) in (T1, T2]?"   case-control, weighted
#   s1-iucn-status    "... what IUCN status (P141) will <taxon> have on T2?" + "will it change?"
#   s2-rfd            Wikidata:Requests for deletions archives: deleted / kept / merged
#   s2-property-proposal  Wikidata:Property proposal pages: created / not done
#   s12-check         read-back (iter_units), leakage scans, cross-split checks, bag-of-words leak probe (numpy+scipy)
#   s12-dedup         dedup against training corpora (build_jev_ext2.Index via build_gde1_b), as GDE-1 part B
#
# Data access (2026-10-01; research/wikidata_pipeline.md §2):
#   * Wikimedia's 2026 API rate limits give a client whose User-Agent carries no contact address ~10 requests / min.
#     No contact is sent (by instruction), so every MediaWiki request is paced (default 7.5 s, shared across this
#     section's processes through a lock file) and 429 Retry-After is honoured.
#   * Revisions AS OF a date come from Special:Export (offset=<date>, dir=desc, limit=1), which returns the revision in
#     force at that date for ~80 items per request; the action API's rvstart works one page at a time.
#   * WDQS (query.wikidata.org) is in an outage with a 1 request / min rule and the usual 60 s runtime limit: used only
#     for candidate pools (random statement samples) and bulk label lookups (VALUES), >= 70 s apart, truncation checked.
#   * maxlag is NOT sent: during the WDQS outage Wikidata's maxlag reports the query-service lag (~500 s), which throttles
#     edit bots, not readers; reads poll the DB replication lag instead and pause while it exceeds 5 s.
import fcntl  # noqa: E402
import gzip  # noqa: E402
import http.client  # noqa: E402
import hashlib  # noqa: E402
import html  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import date, datetime, timedelta, timezone  # noqa: E402

S12_BUILDER = "build_wikidata_jev.py shapes 1-2 v1 (2026-10-01)"
S12_UA = ("jev-wikidata-decision-builder/0.1 (jev research project: decision datasets from public Wikidata revision "
          "history and Wikidata process pages; serial requests, low rate) python-urllib")
S12_API = "https://www.wikidata.org/w/api.php"
S12_INDEX = "https://www.wikidata.org/w/index.php"
S12_WDQS = "https://query.wikidata.org/sparql"
S12_WORK = "/tmp/wikidata_a_work"
S12_SEED = 2026
S12_EVAL_WINDOW = ("2025-06-30", "2026-09-30")            # T1 <= 2025-06-30, T2 >= 2026-06
S12_TRAIN_WINDOWS = [("2019-06-30", "2020-09-30"), ("2020-06-30", "2021-09-30"), ("2021-06-30", "2022-09-30"),
                     ("2022-06-30", "2023-09-30"), ("2023-06-30", "2024-09-30")]   # T1 and T2 both before 2025-01
S12_VAL_PCT = 5
S12_S2_EVAL_FROM = "2025-07-01"        # shape 2: eval = closed on/after
S12_S2_TRAIN_BEFORE = "2025-01-01"     # shape 2: train / val = closed before
S12_CAP_TRAIN, S12_CAP_EVAL = 20000, 8000   # units per source
S12_PROBE_STATE = ("KNOWLEDGE PROBE: the record (state) is withheld on purpose. Answer from the question alone, using "
                   "only what you already know.")
S12_LICENSING_REV = "Wikidata:Licensing rev 2548950862 (2026-09-24)"
S12_LIC_CC0 = ("CC0-1.0 (Wikidata structured data; " + S12_LICENSING_REV + ": 'All structured data (i.e. the main, "
               "Property, Lexeme, and EntitySchema namespaces) is released into the public domain under Creative "
               "Commons Zero.')")
S12_LIC_BYSA = ("CC-BY-SA-4.0 (Wikidata project-namespace text; " + S12_LICENSING_REV + ": 'Text in other namespaces is "
                "copyrighted and available under the Creative Commons Attribution-ShareAlike 4.0 License.' "
                "Attribution: the contributors to {page} (see its history, {url}); derivatives that are shared must "
                "carry the same licence. Item / property IDs and structured values: CC0-1.0)")
S12_LICENSE_EVIDENCE = {
    "Wikidata:Licensing": {"revid": 2548950862, "timestamp": "2026-09-24T03:42:07Z",
                           "url": "https://www.wikidata.org/wiki/Wikidata:Licensing",
                           "quote": "All structured data (i.e. the main, Property, Lexeme, and EntitySchema namespaces) is "
                                    "released into the public domain under Creative Commons Zero. Text in other "
                                    "namespaces is copyrighted and available under the Creative Commons "
                                    "Attribution-ShareAlike 4.0 License."},
    "Wikidata:Copyright": {"revid": 1523614187, "timestamp": "2021-11-07T09:18:36Z",
                           "url": "https://www.wikidata.org/wiki/Wikidata:Copyright",
                           "quote": "All structured data from the main, Property, Lexeme, and EntitySchema namespaces is "
                                    "available under the Creative Commons CC0 License; text in the other namespaces is "
                                    "available under the Creative Commons Attribution-ShareAlike License; additional "
                                    "terms may apply.",
                           "note": "this older page links the BY-SA 3.0 text; Wikidata:Licensing (2026) and the "
                                   "Wikimedia Terms of Use (2023-06) say 4.0"},
}


def s12_log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def s12_sha(s, n=16):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:n]


def s12_hash01(*parts):
    """Uniform [0, 1) from the parts: seeded, order-independent selection."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:13], 16) / float(16 ** 13)


def s12_jsonl_read(path):
    p = Path(path)
    if not p.exists():
        return []
    op = gzip.open if p.name.endswith(".gz") else open
    out = []
    try:     # tolerant of a run killed mid-append: a partial last line / gzip member is skipped (re-fetched on resume)
        with op(p, "rt", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    try:
                        out.append(json.loads(ln))
                    except ValueError:
                        s12_log(f"  {p.name}: skipped a partial line")
    except (EOFError, OSError) as e:
        s12_log(f"  {p.name}: truncated at the end ({type(e).__name__}); {len(out)} rows kept")
    return out


def s12_jsonl_append(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if p.name.endswith(".gz") else open
    with op(p, "at", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================================== HTTP (MediaWiki; paced, shared lock)
class S12Http:
    """Serial MediaWiki client. One request in flight across every process of this section (an flock'd stamp file holds
    the time of the last request), >= min_interval s apart, 429 Retry-After honoured, 5xx / network errors retried with
    backoff, DB replication lag polled every 200 requests. Counts requests, bytes, 429s for the manifests."""

    def __init__(self, min_interval=7.5, stamp=None):
        self.min_interval = float(min_interval)
        self.stamp = stamp or os.path.join(S12_WORK, ".mw_stamp")
        Path(self.stamp).parent.mkdir(parents=True, exist_ok=True)
        self.n_req = self.n_bytes = self.n_429 = self.n_retry = self.n_lag_pauses = 0

    def _wait(self):
        with open(self.stamp, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                s = f.read().strip()
                last = float(s) if s else 0.0
                w = last + self.min_interval - time.time()
                if w > 0:
                    time.sleep(w)
                f.seek(0)
                f.truncate()
                f.write(str(time.time()))
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _penalty(self, seconds):
        """After a 429: push the shared stamp forward so every process waits."""
        with open(self.stamp, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                f.truncate()
                f.write(str(time.time() + seconds - self.min_interval))
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def request(self, url, data=None, tries=10, timeout=180):
        last = None
        for t in range(tries):
            self._wait()
            req = urllib.request.Request(url, data=data, headers={"User-Agent": S12_UA})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    body = r.read()
                self.n_req += 1
                self.n_bytes += len(body)
                return body
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 429:
                    self.n_429 += 1
                    ra = e.headers.get("Retry-After")
                    wait = (float(ra) if ra and ra.strip().isdigit() else 30.0) + 5.0
                    self._penalty(wait)
                    s12_log(f"  429 (rate limited): waiting {wait:.0f}s")
                    continue
                if e.code not in (500, 502, 503, 504):
                    raise
                wait = min(300.0, 10.0 * (2 ** t))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException,
                    ValueError) as e:      # IncompleteRead / broken chunked streams included
                last = e
                wait = min(300.0, 10.0 * (2 ** t))
            self.n_retry += 1
            s12_log(f"  retry {t + 1}/{tries} in {wait:.0f}s ({str(last)[:120]})")
            time.sleep(wait)
        raise RuntimeError(f"giving up on {url[:160]}: {last}")

    def _lag_gate(self):
        if self.n_req % 200 != 0:
            return
        for _ in range(40):
            try:
                d = json.loads(self.request(S12_API + "?" + urllib.parse.urlencode(
                    {"action": "query", "meta": "siteinfo", "siprop": "dbrepllag", "format": "json",
                     "formatversion": "2"})))
                lag = max(float(x.get("lag", 0)) for x in d["query"]["dbrepllag"])
            except Exception:  # noqa: BLE001
                return
            if lag <= 5:
                return
            self.n_lag_pauses += 1
            s12_log(f"  DB replication lag {lag:.1f}s > 5s: pausing 60s")
            time.sleep(60)

    def api(self, params):
        self._lag_gate()
        p = {"format": "json", "formatversion": "2", **params}
        d = json.loads(self.request(S12_API + "?" + urllib.parse.urlencode(p)))
        if isinstance(d, dict) and d.get("error"):
            raise RuntimeError(f"API error {d['error'].get('code')}: {str(d['error'].get('info'))[:200]}")
        return d

    def export(self, titles, offset=None):
        """{title: {'revid', 'ts', 'model', 'text'}} for the revision of each page in force at `offset` (ISO time; the
        newest revision older than it), or the current revision when offset is None. Pages that did not exist yet are
        absent from the result."""
        self._lag_gate()
        data = {"title": "Special:Export", "pages": "\n".join(titles), "action": "submit"}
        if offset:
            data.update({"offset": offset, "limit": "1", "dir": "desc"})
        else:
            data["curonly"] = "1"
        return s12_parse_export(self.request(S12_INDEX, urllib.parse.urlencode(data).encode()))

    def page_texts(self, titles):
        """Current wikitext of up to 50 pages per request: {title: (timestamp, revid, text)}."""
        out = {}
        for i in range(0, len(titles), 50):
            d = self.api({"action": "query", "prop": "revisions", "titles": "|".join(titles[i:i + 50]),
                          "rvprop": "content|timestamp|ids", "rvslots": "main"})
            for p in (d.get("query") or {}).get("pages", []):
                r = (p.get("revisions") or [None])[0]
                if r:
                    out[p["title"]] = (r["timestamp"], r["revid"], ((r.get("slots") or {}).get("main") or {}).get("content"))
        return out

    def allpages(self, prefix, ns=4):
        titles, cont = [], {}
        while True:
            d = self.api({"action": "query", "list": "allpages", "apnamespace": str(ns), "apprefix": prefix,
                          "aplimit": "500", **cont})
            titles += [p["title"] for p in d["query"]["allpages"]]
            if not d.get("continue"):
                return titles
            cont = {"apcontinue": d["continue"]["apcontinue"]}

    def search_random(self, query, n, max_calls=None):
        """Up to n distinct item ids from repeated random-sorted CirrusSearch calls (500 each) + totalhits. A union of
        independent uniform draws is uniform, so the inclusion probability is |sample| / totalhits."""
        got, seen, total, calls, stale = [], set(), None, 0, 0
        max_calls = max_calls or (n // 500 + 1) * 3 + 3
        while len(got) < n and calls < max_calls:
            d = self.api({"action": "query", "list": "search", "srsearch": query, "srsort": "random", "srlimit": "500",
                          "srprop": "", "srinfo": "totalhits", "srnamespace": "0"})
            calls += 1
            total = d["query"]["searchinfo"]["totalhits"]
            new = 0
            for x in d["query"]["search"]:
                if x["title"] not in seen:
                    seen.add(x["title"])
                    got.append(x["title"])
                    new += 1
            if len(got) >= (total or 0):
                break
            stale = stale + 1 if new < 25 else 0
            if stale >= 3:
                break
        return got[:n], total

    def stats(self):
        return {"requests": self.n_req, "bytes": self.n_bytes, "http_429": self.n_429, "retries": self.n_retry,
                "lag_pauses": self.n_lag_pauses, "min_interval_s": self.min_interval}


def _s12_local(tag):
    return tag.rsplit("}", 1)[-1]


def s12_parse_export(body):
    """Special:Export XML -> {title: {'revid', 'ts', 'model', 'text'}} (the newest revision present for each page)."""
    import xml.etree.ElementTree as ET
    out = {}
    for _, el in ET.iterparse(io.BytesIO(body), events=("end",)):
        if _s12_local(el.tag) != "page":
            continue
        title, best = None, None
        for ch in el:
            n = _s12_local(ch.tag)
            if n == "title":
                title = ch.text
            elif n == "revision":
                rev = {}
                for x in ch:
                    m = _s12_local(x.tag)
                    if m == "id":
                        rev["revid"] = int(x.text)
                    elif m == "timestamp":
                        rev["ts"] = x.text
                    elif m == "model":
                        rev["model"] = x.text
                    elif m == "text":
                        rev["text"] = x.text or ""
                if best is None or rev.get("ts", "") > best.get("ts", ""):
                    best = rev
        if title and best:
            out[title] = best
        el.clear()
    return out


# =========================================================================== WDQS (outage rule: 1 / min, 60 s runtime)
class S12Truncated(RuntimeError):
    pass


class S12Wdqs:
    def __init__(self, spacing=70.0, stamp=None):
        self.spacing = float(spacing)
        self.stamp = stamp or os.path.join(S12_WORK, ".wdqs_stamp")
        Path(self.stamp).parent.mkdir(parents=True, exist_ok=True)
        self.n_req = 0

    def _wait(self):
        with open(self.stamp, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                s = f.read().strip()
                w = (float(s) if s else 0.0) + self.spacing - time.time()
                if w > 0:
                    time.sleep(w)
                f.seek(0)
                f.truncate()
                f.write(str(time.time()))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def query(self, q, tries=4):
        last = None
        for _ in range(tries):
            self._wait()
            req = urllib.request.Request(S12_WDQS, data=urllib.parse.urlencode({"query": q}).encode(),
                                         headers={"User-Agent": S12_UA, "Accept": "text/tab-separated-values"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                self.n_req += 1
                return s12_parse_tsv(body)
            except urllib.error.HTTPError as e:
                last = e
                ra = e.headers.get("Retry-After")
                s12_log(f"  WDQS HTTP {e.code} (retry-after {ra})")
                time.sleep((float(ra) if ra and ra.isdigit() else 60.0) + 5)
            except S12Truncated as e:
                raise e
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, ValueError) as e:
                last = e          # IncompleteRead on a large chunked response: retry the query
                s12_log(f"  WDQS read failed ({type(e).__name__}): retrying")
                time.sleep(30)
        raise RuntimeError(f"WDQS failed: {last}")

    def labels(self, ids, chunk=4000):
        """{id: English label (else 'mul')} for item / property ids, via VALUES lookups (current labels)."""
        out = {}
        ids = sorted(set(ids))
        for i in range(0, len(ids), chunk):
            vals = " ".join("wd:" + x for x in ids[i:i + chunk])
            rows = self.query("SELECT ?x ?l ?lang WHERE { VALUES ?x { %s } ?x rdfs:label ?l . "
                              "BIND(LANG(?l) AS ?lang) FILTER(?lang = 'en' || ?lang = 'mul') }" % vals)
            for r in rows:
                if r["x"] and r["l"] and (r["lang"] == "en" or r["x"] not in out):
                    out[r["x"]] = r["l"]
        return out

    def labels_any(self, ids, chunk=4000):
        """{id: (label, lang)} in any language, for ids without an English / 'mul' label."""
        out = {}
        ids = sorted(set(ids))
        for i in range(0, len(ids), chunk):
            vals = " ".join("wd:" + x for x in ids[i:i + chunk])
            rows = self.query("SELECT ?x (SAMPLE(?l) AS ?lab) (SAMPLE(LANG(?l)) AS ?lang) WHERE { VALUES ?x { %s } "
                              "?x rdfs:label ?l . } GROUP BY ?x" % vals)
            for r in rows:
                if r.get("x") and r.get("lab"):
                    out[r["x"]] = (r["lab"], r.get("lang"))
        return out


_S12_ENT = re.compile(r"^<http://www\.wikidata\.org/(?:entity|prop/direct|prop)/(?:statement/)?([^>]+)>$")


def s12_tsv_value(v):
    v = v.strip()
    if not v:
        return None
    if v.startswith("<") and v.endswith(">"):
        m = _S12_ENT.match(v)
        return m.group(1) if m else v[1:-1]
    if v.startswith('"'):
        i, out = 1, []
        while i < len(v):
            c = v[i]
            if c == "\\" and i + 1 < len(v):
                out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}.get(v[i + 1], v[i + 1]))
                i += 2
                continue
            if c == '"':
                break
            out.append(c)
            i += 1
        return "".join(out)
    return v


def s12_parse_tsv(body):
    text = body.decode("utf-8", "replace")
    tail = text[-3000:]
    if "java." in tail or "Exception" in tail or "TimeoutException" in tail:
        raise S12Truncated("WDQS response ends in a server exception (60 s timeout?): result truncated")
    lines = text.split("\n")
    if not lines or not lines[0].startswith("?"):
        raise RuntimeError("WDQS: no TSV header: " + text[:200])
    hdr = [h.strip().lstrip("?") for h in lines[0].split("\t")]
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        rows.append({h: s12_tsv_value(parts[i]) if i < len(parts) else None for i, h in enumerate(hdr)})
    return rows


# =========================================================================== Wikibase JSON
def wb_value(snak):
    if not snak:
        return ["none"]
    st = snak.get("snaktype")
    if st == "somevalue":
        return ["some"]
    if st == "novalue":
        return ["no"]
    dv = snak.get("datavalue") or {}
    t, v = dv.get("type"), dv.get("value")
    if t == "wikibase-entityid":
        return ["item", v.get("id") or (("Q" if v.get("entity-type") == "item" else "P") + str(v.get("numeric-id")))]
    if t == "time":
        return ["time", v.get("time"), int(v.get("precision", 11))]
    if t == "quantity":
        u = str(v.get("unit") or "")
        return ["qty", v.get("amount"), u.rsplit("/", 1)[-1] if u.startswith("http") else None]
    if t == "string":
        return ["str", v]
    if t == "monolingualtext":
        return ["mono", v.get("text"), v.get("language")]
    if t == "globecoordinate":
        return ["coord", v.get("latitude"), v.get("longitude")]
    return ["other", t]


_S12_TIME = re.compile(r"^([+-]\d+)-(\d\d)-(\d\d)T")


def s12_interval(tv):
    """['time', '+2001-03-00T00:00:00Z', precision] -> (lo, hi) inclusive dates, or None. 11 day, 10 month, 9 year,
    8 decade, 7 century."""
    if not tv or tv[0] != "time" or not tv[1]:
        return None
    m = _S12_TIME.match(tv[1])
    if not m:
        return None
    y, mo, dd, p = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(tv[2])
    if y < 1 or y > 9999:
        return None
    try:
        if p >= 11 and mo and dd:
            return date(y, mo, dd), date(y, mo, dd)
        if p >= 10 and mo:
            return date(y, mo, 1), date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1)
        if p >= 9:
            return date(y, 1, 1), date(y, 12, 31)
        if p == 8:
            y0 = y - y % 10
            return date(max(1, y0), 1, 1), date(y0 + 9, 12, 31)
        if p == 7:
            y0 = (y - 1) // 100 * 100 + 1
            return date(max(1, y0), 1, 1), date(y0 + 99, 12, 31)
    except ValueError:
        return None
    return None


def s12_window(iv, t1, t2):
    """Where a dated event falls relative to (T1, T2]: before (<= T1) / in / after (> T2) / straddle / unknown."""
    if iv is None:
        return "unknown"
    lo, hi = iv
    t1, t2 = date.fromisoformat(t1), date.fromisoformat(t2)
    if hi <= t1:
        return "before"
    if lo > t2:
        return "after"
    if lo > t1 and hi <= t2:
        return "in"
    return "straddle"


def s12_fmt_time(tv):
    iv = s12_interval(tv)
    if iv is None:
        return str(tv[1] if tv and len(tv) > 1 else "?")
    lo, p = iv[0], int(tv[2])
    if p >= 11:
        return lo.isoformat()
    if p == 10:
        return lo.strftime("%Y-%m")
    if p == 9:
        return str(lo.year)
    if p == 8:
        return f"{lo.year}s"
    return f"{(lo.year - 1) // 100 + 1}th century"


def s12_compact(ent, ptypes=None, max_per_prop=40):
    """A revision's entity JSON -> a compact dict (labels in en/mul, up to 5 aliases, claims without references, external
    identifiers reduced to counts, sitelink count + enwiki title). Redirect revisions -> {'redirect': target}."""
    if ent is None:
        return None
    if ent.get("redirect"):
        return {"redirect": ent.get("redirect"), "id": ent.get("entity")}

    def lang(field):
        d = ent.get(field) or {}
        for L in ("en", "mul"):
            v = d.get(L)
            if v and v.get("value"):
                return v["value"]
        return None
    c = {"id": ent.get("id"), "label": lang("labels"), "desc": lang("descriptions"),
         "aliases": [a.get("value") for a in ((ent.get("aliases") or {}).get("en") or [])][:5],
         "n_sitelinks": len(ent.get("sitelinks") or {}),
         "enwiki": ((ent.get("sitelinks") or {}).get("enwiki") or {}).get("title"), "claims": {}, "xids": {}}
    for pid, sts in (ent.get("claims") or {}).items():
        if ptypes and ptypes.get(pid) == "ExternalId":
            c["xids"][pid] = len(sts)
            continue
        out = []
        for s in sts[:max_per_prop]:
            q = {}
            for qp, xs in (s.get("qualifiers") or {}).items():
                q[qp] = [wb_value(x) for x in xs[:6]]
            out.append({"id": s.get("id"), "rank": s.get("rank", "normal"), "v": wb_value(s.get("mainsnak")), "q": q})
        c["claims"][pid] = out
        if len(sts) > max_per_prop:
            c["claims"][pid].append({"more": len(sts) - max_per_prop})
    return c


def s12_stmts(c, pid, live=True):
    """Statements of a compact entity (non-deprecated when live)."""
    out = [s for s in ((c or {}).get("claims") or {}).get(pid, []) if "more" not in s]
    return [s for s in out if s.get("rank") != "deprecated"] if live else out


def s12_truthy(c, pid):
    st = s12_stmts(c, pid)
    pref = [s for s in st if s.get("rank") == "preferred"]
    return pref or st


def s12_items(c, pid):
    return [s["v"][1] for s in s12_stmts(c, pid) if s["v"][0] == "item"]


def s12_ref_ids(c, limit_props=None):
    """Every item / property id a rendered state could name (for label lookups)."""
    ids = set()
    for pid, sts in ((c or {}).get("claims") or {}).items():
        ids.add(pid)
        for s in sts:
            if "more" in s:
                continue
            if s["v"][0] == "item":
                ids.add(s["v"][1])
            if s["v"][0] == "qty" and s["v"][2]:
                ids.add(s["v"][2])
            for qp, vs in s.get("q", {}).items():
                ids.add(qp)
                for v in vs:
                    if v[0] == "item":
                        ids.add(v[1])
    return {x for x in ids if re.match(r"^[QP]\d+$", str(x))}


# --------------------------------------------------------------------------- state rendering
S12_ORDER = {
    "person": ["P31", "P21", "P27", "P569", "P570", "P19", "P20", "P106", "P39", "P54", "P102", "P413", "P641", "P1532",
               "P118", "P69", "P108", "P463", "P166", "P2048", "P2067", "P1559", "P735", "P734"],
    "org": ["P31", "P17", "P131", "P159", "P571", "P576", "P112", "P169", "P488", "P452", "P1128", "P2139", "P2403",
            "P749", "P355", "P127", "P1454", "P740", "P136", "P264", "P527", "P463", "P1366", "P1365", "P856"],
    "taxon": ["P31", "P225", "P105", "P171", "P141", "P183", "P1843", "P9714", "P2043", "P2067", "P181", "P1420"],
}
S12_QUAL_SKIP = {"P1545"}   # series ordinal: noise
S12_MAX_STATE = 6000


def s12_val_text(v, L):
    k = v[0]
    if k == "item":
        return L.get(v[1]) or v[1]
    if k == "time":
        return s12_fmt_time(v)
    if k == "qty":
        a = str(v[1] or "").lstrip("+")
        return a + (" " + (L.get(v[2]) or v[2]) if v[2] and v[2] != "1" else "")
    if k == "str":
        return str(v[1])[:200]
    if k == "mono":
        return str(v[1])[:200]
    if k == "coord":
        try:
            return f"{float(v[1]):.3f}, {float(v[2]):.3f}"
        except (TypeError, ValueError):
            return "?"
    if k == "some":
        return "unknown value"
    if k == "no":
        return "no value"
    return "?"


def s12_stmt_text(s, L, max_q=6):
    t = s12_val_text(s["v"], L)
    qs = []
    for qp, vs in s.get("q", {}).items():
        if qp in S12_QUAL_SKIP:
            continue
        qs.append(f"{L.get(qp) or qp}: " + ", ".join(s12_val_text(v, L) for v in vs[:3]))
        if len(qs) >= max_q:
            break
    return t + (f" ({'; '.join(qs)})" if qs else "")


def s12_render(c, as_of, L, kind, max_vals=10, max_chars=S12_MAX_STATE, known_props=None, focus=None):
    """The item as it stood on `as_of`, as plain text: label, description, aliases, statements (priority properties
    first, then by property number, up to a length budget), external-identifier and sitelink counts."""
    lines = [f"WIKIDATA ITEM {c.get('id')} AS IT STOOD ON {as_of}"]
    lines.append(f"Label (en): {c.get('label') or '(none)'}")
    if c.get("desc"):
        lines.append(f"Description (en): {c['desc']}")
    if c.get("aliases"):
        lines.append("Also known as: " + "; ".join(a for a in c["aliases"] if a))
    lines.append("Statements:")
    claims = c.get("claims") or {}
    order = [p for p in S12_ORDER.get(kind, []) if p in claims]
    order += sorted((p for p in claims if p not in order), key=lambda p: int(p[1:]) if p[1:].isdigit() else 10 ** 9)
    budget = max_chars - sum(len(x) + 1 for x in lines) - 200
    omitted = 0
    for pid in order:
        if known_props is not None and pid not in known_props and not L.get(pid):
            continue          # a property deleted since (no datatype, no label): nothing meaningful to show
        sts = [s for s in claims[pid] if "more" not in s and s.get("rank") != "deprecated"]
        more = sum(s.get("more", 0) for s in claims[pid] if "more" in s)
        if not sts:
            continue
        cap_line, nv = 900, max_vals
        if focus and pid == focus[0]:      # the question's statement first, never cut off
            sts = sorted(sts, key=lambda x: x.get("id") != focus[1])
            cap_line, nv = 2500, 20
        vals = [s12_stmt_text(s, L) for s in sts[:nv]]
        extra = len(sts) - len(vals) + more
        ln = f"- {L.get(pid) or pid}: " + "; ".join(vals) + (f"; ... and {extra} more" if extra > 0 else "")
        if len(ln) > cap_line:
            ln = ln[:cap_line - 20] + " ..."
        if len(ln) + 1 > budget:
            omitted += 1
            continue
        lines.append(ln)
        budget -= len(ln) + 1
    if omitted:
        lines.append(f"- ({omitted} further properties not shown)")
    nx = sum((c.get("xids") or {}).values())
    if nx:
        lines.append(f"External identifiers: {nx} (in {len(c['xids'])} databases)")
    lines.append(f"Wikimedia sitelinks: {c.get('n_sitelinks', 0)}" +
                 (f"; English Wikipedia article: {c['enwiki']}" if c.get("enwiki") else ""))
    return "\n".join(lines)


# =========================================================================== shared output: records, splits, writing
def s12_window_of(qid, seed, eval_share):
    """('eval'|'train'|'val', T1, T2) for an item: the pool is a hash of the item, so an item is never in both."""
    if s12_hash01(seed, "pool", qid) < eval_share:
        return ("eval",) + S12_EVAL_WINDOW
    w = S12_TRAIN_WINDOWS[int(s12_hash01(seed, "year", qid) * len(S12_TRAIN_WINDOWS))]
    sp = "val" if s12_hash01(seed, "val", qid) < S12_VAL_PCT / 100 else "train"
    return (sp,) + w


def s12_record(source, state, qs, labs, meta, base_id, licence, rec_id):
    from build_jev_data import record, validate
    kinds = {q: ("hard" if sum(1 for v in labs[q].values() if v > 0) == 1 else "soft") for q in qs}
    r = record(source, state, qs, labs, kinds, None)
    r["id"] = rec_id
    r["base_id"] = base_id
    r["license"] = licence
    r["meta"] = meta
    validate(r)
    return r


def s12_noul(p):
    p = 1.0 if p else 0.0
    return {"yes": p, "no": 1.0 - p}


def s12_cap(recs, cap, seed, tag):
    """Seeded uniform sample of records down to `cap` units (records here are small: 1-2 questions)."""
    n = sum(len(r["questions"]) for r in recs)
    if not cap or n <= cap:
        return recs, {"units_before_cap": n, "cap": cap}
    recs = sorted(recs, key=lambda r: s12_hash01(seed, tag, r["id"]))
    out, u = [], 0
    for r in recs:
        if u + len(r["questions"]) > cap:
            continue
        out.append(r)
        u += len(r["questions"])
    return out, {"units_before_cap": n, "cap": cap, "units_after_cap": u}


def s12_split_stats(recs):
    by_q, yes, wyes, kinds, opts = Counter(), defaultdict(list), defaultdict(lambda: [0.0, 0.0]), Counter(), defaultdict(Counter)
    dec, res, info = [], [], []
    for r in recs:
        m = r["meta"]
        w = float(m.get("sample_weight") or 1.0)
        dec.append(str(m.get("decision_date"))[:10])
        info.append(str(m.get("info_date"))[:10])
        for q, lab in r["labels"].items():
            by_q[q] += 1
            kinds[r["label_kind"][q]] += 1
            top = max(lab, key=lab.get)
            opts[q][top] += 1
            if "yes" in lab:
                yes[q].append(lab["yes"])
                wyes[q][0] += w * lab["yes"]
                wyes[q][1] += w
            rd = (m.get("resolution_date") or {}).get(q)
            if rd:
                res.append(rd)
    out = {"records": len(recs), "units": sum(by_q.values()), "units_by_qid": dict(by_q), "label_kind": dict(kinds),
           "label_counts_by_qid": {q: dict(c.most_common()) for q, c in opts.items()},
           "yes_rate_by_qid": {q: round(sum(v) / len(v), 4) for q, v in yes.items() if v},
           "decision_date_range": [min(dec), max(dec)] if dec else None,
           "info_date_range": [min(info), max(info)] if info else None,
           "resolution_date_range": [min(res), max(res)] if res else None}
    if any(float(r["meta"].get("sample_weight") or 1) != 1.0 for r in recs):
        out["weighted_yes_rate_by_qid"] = {q: round(a / b, 5) for q, (a, b) in wyes.items() if b}
        wd = defaultdict(Counter)
        for r in recs:
            w = float(r["meta"].get("sample_weight") or 1.0)
            for q, lab in r["labels"].items():
                for k, v in lab.items():
                    wd[q][k] += w * v
        out["weighted_label_dist_by_qid"] = {q: {k: round(v / sum(c.values()), 5) for k, v in c.most_common()}
                                             for q, c in wd.items()}
        out["population_size_estimate"] = round(sum(float(r["meta"].get("sample_weight") or 1.0) for r in recs))
    return out


def s12_probe_copy(r):
    p = json.loads(json.dumps(r))
    p["id"] = r["id"] + "-probe"
    p["state"] = S12_PROBE_STATE
    p["meta"] = {**p["meta"], "probe": "state_removed", "probe_of": r["id"]}
    return p


def s12_gde1_copy(r, gset):
    g = json.loads(json.dumps(r))
    g["id"] = "gde1-" + r["id"]
    g["meta"] = {**g["meta"], "gde1_set": gset, "domain": "O", "gde1_part": "wikidata", "eval_only": True,
                 "td_workflow": None}
    return g


def s12_write(out_root, folder, gset, by_split, manifest):
    """<out>/<folder>/{train,val,eval,probe}/<source>.jsonl + manifest.json; GDE-1 copies of eval (+ probe) under
    <out>/gde1/<gset>/{eval,probe}/ with their own manifest."""
    root = Path(out_root) / folder
    for sp, recs in by_split.items():
        d = root / sp
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jsonl"):
            old.unlink()
        fh = {}
        for r in recs:
            fn = r["source"].replace("/", "__") + ".jsonl"
            if fn not in fh:
                fh[fn] = (d / fn).open("w", encoding="utf-8")
            fh[fn].write(json.dumps(r, ensure_ascii=False) + "\n")
        for f in fh.values():
            f.close()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
    g = Path(out_root) / "gde1" / gset
    gm = {"set": gset, "domain": "O", "builder": S12_BUILDER, "built": manifest.get("built"),
          "license": manifest.get("license"), "license_evidence": S12_LICENSE_EVIDENCE, "eval_only": True,
          "derived_from": str(root), "splits": {}, "notes": manifest.get("gde1_notes"),
          "dedup_vs_training": {"status": "NOT RUN: `build_wikidata_jev.py s12-dedup --fingerprint TAG=PATH ...` on a "
                                          "node against jev_data, jev_ext_strict, jev_ext2, jev_ext3"}}
    for sp in ("eval", "probe"):
        recs = [s12_gde1_copy(r, gset) for r in by_split.get(sp, [])]
        d = g / sp
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jsonl"):
            old.unlink()
        if recs:
            with (d / (recs[0]["source"].replace("/", "__") + ".jsonl")).open("w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gm["splits"][sp] = s12_split_stats(recs) if recs else {"records": 0}
    (g / "manifest.json").write_text(json.dumps(gm, indent=1, ensure_ascii=False, default=str))
    s12_log(f"{folder}: wrote " + ", ".join(f"{sp} {len(v)}" for sp, v in by_split.items()) + f"; GDE-1 copy {g}")


# =========================================================================== SHAPE 1: outcomes over time (edit history)
class S12Labels:
    """Current English labels of items / properties (WDQS VALUES lookups), cached in <work>/labels.jsonl, plus property
    datatypes (<work>/props.json). Labels of REFERENCED entities are current, not as of T1 (a documented caveat)."""

    def __init__(self, work, wdqs):
        self.path = Path(work) / "labels.jsonl"
        self.ppath = Path(work) / "props.json"
        self.wdqs = wdqs
        self.L, self.anydone = {}, set()
        for r in s12_jsonl_read(self.path):
            if r["id"] in self.L and self.L[r["id"]] and not r.get("l"):
                continue
            self.L[r["id"]] = r["l"]
            if r.get("any"):
                self.anydone.add(r["id"])
        self.ptypes = {}
        if self.ppath.exists():
            self.ptypes = json.loads(self.ppath.read_text())

    def ensure_props(self):
        if self.ptypes:
            return
        rows = self.wdqs.query("SELECT ?p ?t ?l WHERE { ?p wikibase:propertyType ?t . OPTIONAL { ?p rdfs:label ?l . "
                               "FILTER(LANG(?l) = 'en') } }")
        self.ptypes = {r["p"]: str(r["t"]).rsplit("#", 1)[-1] for r in rows if r.get("p")}
        new = [{"id": r["p"], "l": r["l"]} for r in rows if r.get("p") and r.get("l") and r["p"] not in self.L]
        for x in new:
            self.L[x["id"]] = x["l"]
        s12_jsonl_append(self.path, new)
        self.ppath.write_text(json.dumps(self.ptypes))
        s12_log(f"properties: {len(self.ptypes)} datatypes, {len(new)} labels")

    def ensure(self, ids):
        miss = sorted({i for i in ids if i not in self.L})
        if miss:
            s12_log(f"labels: fetching {len(miss)} via WDQS ({math.ceil(len(miss) / 4000)} queries, >= 70 s apart)")
            for j in range(0, len(miss), 4000):        # persisted query by query (a failure loses one chunk only)
                part = miss[j:j + 4000]
                got = self.wdqs.labels(part)
                rows = [{"id": i, "l": got.get(i)} for i in part]
                for r in rows:
                    self.L[r["id"]] = r["l"]
                s12_jsonl_append(self.path, rows)
        # second pass: any-language labels for ids with no English / 'mul' label (e.g. Catalan-only items)
        none = sorted({i for i in ids if not self.L.get(i) and i not in self.anydone})
        if none:
            got = self.wdqs.labels_any(none)
            rows = [{"id": i, "l": (got[i][0] if i in got else None), "lang": (got[i][1] if i in got else None),
                     "any": True} for i in none]
            for r in rows:
                self.L[r["id"]] = r["l"]
                self.anydone.add(r["id"])
            s12_jsonl_append(self.path, rows)
        return len(miss) + len(none)

    def get(self, i):
        return self.L.get(i)

    def view(self):
        return {k: v for k, v in self.L.items() if v}


# --------------------------------------------------------------------------- source specs
S12_NATIONAL = re.compile(r"\bnational\b|\bunder[- ]?\d\d\b|\bU-?\d\d\b|olympic|\bselecci[oó]n\b|nationalmannschaft|"
                          r"\bnazionale\b|\bs[ée]lection\b|\bequipe de france\b|reprezentacja|\bB team\b",
                          re.I)
# the 8 IUCN Red List categories as Wikidata P141 values (checked on WDQS 2026-10-01: Q96377276 'endangered' is the
# P141 value, 17,257 uses; Q11394 is not used); 'not evaluated' (25), 'Lower Risk' (2), 'conservation dependent' (1)
# are dropped
S12_IUCN = {"Q211005": ("LC", "Least Concern"), "Q719675": ("NT", "Near Threatened"),
            "Q278113": ("VU", "Vulnerable"), "Q96377276": ("EN", "Endangered"),
            "Q219127": ("CR", "Critically Endangered"), "Q239509": ("EW", "Extinct in the Wild"),
            "Q237350": ("EX", "Extinct"), "Q3245245": ("DD", "Data Deficient")}
S12_ORG_CLASSES = [("Q7278", "political party"), ("Q215380", "musical group"), ("Q4830453", "business")]


def s1_tenure_spec(name, pid, years, noun, kind):
    return {"name": name, "pid": pid, "years": years, "noun": noun, "kind": "person", "family": "tenure", "qid": "tenure_ends",
            "athlete": kind == "athlete"}


S1_SPECS = {
    "office_end": s1_tenure_spec("office_end", "P39", 12, "position", "office"),
    "athlete_team": s1_tenure_spec("athlete_team", "P54", 8, "team", "athlete"),
    "org_dissolved": {"name": "org_dissolved", "pid": "P576", "kind": "org", "family": "org", "qid": "dissolved"},
    "iucn_status": {"name": "iucn_status", "pid": "P141", "kind": "taxon", "family": "iucn", "qid": "status_at_t2"},
}


def s1_drop(reason):
    return ("drop", reason, None)


def s1_ok(info):
    return ("ok", None, info)


# ---- tenure (P39 office, P54 team)
def s1_tenure_t1(spec, c1, qid, t1, seed):
    if not c1:
        return s1_drop("absent_at_T1")
    if c1.get("redirect"):
        return s1_drop("redirect_at_T1")
    if "Q5" not in s12_items(c1, "P31"):
        return s1_drop("not_human_at_T1")
    if s12_stmts(c1, "P570"):
        return s1_drop("date_of_death_recorded_at_T1")
    t1d = date.fromisoformat(t1)
    if spec["athlete"]:
        b = [s12_interval(s["v"]) for s in s12_stmts(c1, "P569") if s["v"][0] == "time"]
        b = [x for x in b if x]
        if b and b[0][0].year < t1d.year - 45:
            return s1_drop("born_more_than_45y_before_T1")
    lo_bound = date(t1d.year - spec["years"], t1d.month, t1d.day)
    el = []
    for s in s12_stmts(c1, spec["pid"]):
        if s["v"][0] != "item" or any(k in s.get("q", {}) for k in ("P582", "P1366", "P1534")):
            continue
        st = [v for v in s.get("q", {}).get("P580", []) if v[0] == "time"]
        iv = s12_interval(st[0]) if st else None
        if not iv or iv[1] > t1d or iv[0] < lo_bound:
            continue
        el.append((s, st[0], iv))
    if not el:
        return s1_drop(f"no_open_{spec['pid']}_started_within_{spec['years']}y_before_T1")
    if spec["athlete"]:   # the current team = the latest start; ties by hash
        el.sort(key=lambda x: (-x[2][0].toordinal(), s12_hash01(seed, qid, x[0]["id"])))
    else:                 # several offices can run at once: one by hash
        el.sort(key=lambda x: s12_hash01(seed, qid, x[0]["id"]))
    return s1_ok({"cands": [{"stmt": s["id"], "value": s["v"][1], "start": st} for s, st, _ in el],
                  "n_open_in_window": len(el)})


def s1_tenure_refine(spec, info, L):
    for c in info["cands"]:
        if spec["athlete"] and S12_NATIONAL.search(L.get(c["value"]) or ""):
            continue
        return s1_ok({**c, "n_open_in_window": info["n_open_in_window"]})
    return s1_drop("only_national_teams_open" if spec["athlete"] else "no_candidate")


def s1_tenure_t2(spec, info, c2, t1, t2):
    if not c2:
        return s1_drop("absent_at_T2")
    if c2.get("redirect"):
        return s1_drop("merged_into_another_item_by_T2")
    s = [x for x in s12_stmts(c2, spec["pid"], live=False) if x["id"] == info["stmt"]]
    if not s:
        return s1_drop("statement_removed_by_T2")
    s = s[0]
    if s.get("rank") == "deprecated":
        return s1_drop("statement_deprecated_by_T2")
    if s["v"][0] != "item" or s["v"][1] != info["value"]:
        return s1_drop("statement_value_changed_by_T2")
    ends = s.get("q", {}).get("P582", [])
    if not ends:
        if s.get("q", {}).get("P1366") or s.get("q", {}).get("P1534"):
            return s1_drop("successor_or_end_cause_without_end_time_at_T2")
        return s1_ok({"yes": False, "end": None, "class": "none"})
    if ends[0][0] != "time":
        return s1_drop("end_time_unknown_value_at_T2")
    w = s12_window(s12_interval(ends[0]), t1, t2)
    if w == "in":
        return s1_ok({"yes": True, "end": s12_fmt_time(ends[0]), "class": w})
    if w == "after":
        return s1_ok({"yes": False, "end": s12_fmt_time(ends[0]), "class": w})
    if w == "before":
        return s1_drop("premise_false_end_dated_on_or_before_T1")
    return s1_drop(f"end_time_{w}")


def s1_tenure_question(spec, c1, info, L, t1, t2):
    who = c1.get("label") or c1.get("id")
    what = L.get(info["value"]) or info["value"]
    start = s12_fmt_time(info["start"])
    if spec["athlete"]:
        lead = (f"According to Wikidata, will {who} ({c1['id']})'s spell as a member of {what} (member since {start}; no "
                f"end date recorded on {t1}) end after {t1} and on or before {t2}?")
        obj = "member-of-sports-team statement"
    else:
        lead = (f"According to Wikidata, will {who} ({c1['id']})'s tenure as {what} (held since {start}; no end date "
                f"recorded on {t1}) end after {t1} and on or before {t2}?")
        obj = "position-held statement"
    return {spec["qid"]: {"type": "noul", "instructions": lead + f" The state is the Wikidata item as it stood on {t1}.",
                          "criteria": {"yes": f"Wikidata's revision in force on {t2} gives this {obj} an end time dated "
                                              f"after {t1} and on or before {t2}.",
                                       "no": f"Wikidata's revision in force on {t2} records no end time for this {obj}, "
                                             f"or one dated after {t2}."}}}


# ---- organisations (P576), case-control
def s1_org_t1(spec, c1, qid, t1, seed, row=None):
    if not c1:
        return s1_drop("absent_at_T1")
    if c1.get("redirect"):
        return s1_drop("redirect_at_T1")
    cls = (row or {}).get("cls")
    if cls and cls not in s12_items(c1, "P31"):
        return s1_drop("class_not_in_P31_at_T1")
    if s12_stmts(c1, "P576"):
        return s1_drop("dissolution_date_already_recorded_at_T1")
    if s12_stmts(c1, "P582"):
        return s1_drop("end_time_already_recorded_at_T1")
    return s1_ok({"cls": cls})


def s1_org_t2(spec, info, c2, t1, t2):
    if not c2:
        return s1_drop("absent_at_T2")
    if c2.get("redirect"):
        return s1_drop("merged_into_another_item_by_T2")
    vals = [s["v"] for s in s12_stmts(c2, "P576")]
    if not vals:
        return s1_ok({"yes": False, "class": "none", "date": None})
    if any(v[0] != "time" for v in vals):
        return s1_drop("dissolution_date_unknown_value_at_T2")
    ws = [s12_window(s12_interval(v), t1, t2) for v in vals]
    if "before" in ws:
        return s1_drop("premise_false_dissolution_dated_on_or_before_T1")
    if "straddle" in ws or "unknown" in ws:
        return s1_drop("dissolution_date_straddles_window")
    if "in" in ws:
        return s1_ok({"yes": True, "class": "in", "date": s12_fmt_time(vals[ws.index("in")])})
    return s1_ok({"yes": False, "class": "after", "date": s12_fmt_time(vals[0])})


def s1_org_question(spec, c1, info, L, t1, t2):
    who = c1.get("label") or c1.get("id")
    return {spec["qid"]: {"type": "noul",
                          "instructions": (f"According to Wikidata, will {who} ({c1['id']}) get a dissolution date "
                                           f"(property P576 'dissolved, abolished or demolished date') dated after {t1} and "
                                           f"on or before {t2}? The state is the Wikidata item as it stood on {t1}, when it "
                                           f"had no dissolution date."),
                          "criteria": {"yes": f"Wikidata's revision in force on {t2} records a dissolution date for this "
                                              f"item dated after {t1} and on or before {t2}.",
                                       "no": f"Wikidata's revision in force on {t2} records no dissolution date for it, or "
                                             f"one dated after {t2}."}}}


# ---- IUCN status (P141)
def s1_iucn_t1(spec, c1, qid, t1, seed):
    if not c1:
        return s1_drop("absent_at_T1")
    if c1.get("redirect"):
        return s1_drop("redirect_at_T1")
    v = [s["v"][1] for s in s12_truthy(c1, "P141") if s["v"][0] == "item"]
    if len(set(v)) != 1:
        return s1_drop(f"T1_status_count_{min(len(set(v)), 2)}")
    if v[0] not in S12_IUCN:
        return s1_drop("T1_status_outside_IUCN_categories")
    return s1_ok({"status": v[0]})


def s1_iucn_t2(spec, info, c2, t1, t2):
    if not c2:
        return s1_drop("absent_at_T2")
    if c2.get("redirect"):
        return s1_drop("merged_into_another_item_by_T2")
    v = [s["v"][1] for s in s12_truthy(c2, "P141") if s["v"][0] == "item"]
    if len(set(v)) != 1:
        return s1_drop(f"T2_status_count_{min(len(set(v)), 2)}")
    if v[0] not in S12_IUCN:
        return s1_drop("T2_status_outside_IUCN_categories")
    return s1_ok({"status_t2": v[0], "yes": v[0] != info["status"]})


def s1_iucn_question(spec, c1, info, L, t1, t2):
    who = c1.get("label") or c1.get("id")
    code, name = S12_IUCN[info["status"]]
    crit = {S12_IUCN[q][0]: f"{S12_IUCN[q][1]} ({S12_IUCN[q][0]})" for q in S12_IUCN}
    return {"status_at_t2": {"type": "choice",
                             "instructions": (f"According to Wikidata, what IUCN conservation status (property P141) will "
                                              f"{who} ({c1['id']}) have on {t2}? On {t1} Wikidata gave {name} ({code}). "
                                              f"The state is the Wikidata item as it stood on {t1}."),
                             "criteria": crit},
            "status_changes": {"type": "noul",
                               "instructions": (f"According to Wikidata, will the IUCN conservation status of {who} "
                                                f"({c1['id']}) on {t2} differ from {name} ({code}), its status on {t1}?"),
                               "criteria": {"yes": f"Wikidata's revision in force on {t2} gives a different IUCN status.",
                                            "no": f"Wikidata's revision in force on {t2} still gives {name} ({code})."}}}


S1_FUNCS = {"tenure": (s1_tenure_t1, s1_tenure_t2, s1_tenure_question),
            "org": (s1_org_t1, s1_org_t2, s1_org_question),
            "iucn": (s1_iucn_t1, s1_iucn_t2, s1_iucn_question)}


def s1_labels_of(spec, info, res):
    if spec["family"] == "iucn":
        keys = [S12_IUCN[q][0] for q in S12_IUCN]
        tgt = S12_IUCN[res["status_t2"]][0]
        return {"status_at_t2": {k: (1.0 if k == tgt else 0.0) for k in keys}, "status_changes": s12_noul(res["yes"])}
    return {spec["qid"]: s12_noul(res["yes"])}


def s1_t2_keep(spec):
    return {"tenure": [spec.get("pid", "P39"), "P570"], "org": ["P576", "P582", "P31"], "iucn": ["P141"]}[spec["family"]]


# --------------------------------------------------------------------------- candidates
S12_Q_TENURE = """SELECT ?item ?st ?val ?start WHERE {
  SERVICE bd:sample { ?item p:%(pid)s ?st . bd:serviceParam bd:sample.limit %(n)d ; bd:sample.sampleType "RANDOM" . }
  ?st pq:P580 ?start .
  FILTER(?start >= "%(lo)s"^^xsd:dateTime && ?start <= "%(hi)s"^^xsd:dateTime)
  ?st ps:%(pid)s ?val .
}"""


def s1_candidates_tenure(spec, a, http, wdqs, wd):
    """Persons with a P39 / P54 statement whose start time lies in [earliest T1 - years, latest T1]: unions of WDQS random
    statement samples (bd:sample; each query sized to finish well inside the 60 s limit)."""
    path = wd / "candidates.jsonl"
    if path.exists():
        return
    lo = f"{int(S12_TRAIN_WINDOWS[0][0][:4]) - spec['years']}-01-01T00:00:00Z"
    hi = f"{S12_EVAL_WINDOW[0]}T23:59:59Z"
    items, rep = {}, Counter()
    for i in range(a.wdqs_queries):
        try:
            rows = wdqs.query(S12_Q_TENURE % {"pid": spec["pid"], "n": a.wdqs_sample, "lo": lo, "hi": hi})
        except S12Truncated:
            rep["wdqs_truncated_queries_discarded"] += 1
            s12_log("  WDQS result truncated (60 s limit): discarded; lower --wdqs-sample")
            continue
        rep["wdqs_rows"] += len(rows)
        for r in rows:
            if r.get("item") and re.match(r"^Q\d+$", r["item"]):
                items.setdefault(r["item"], set()).add(r["st"])
        s12_log(f"  WDQS sample {i + 1}/{a.wdqs_queries}: {len(rows)} rows, {len(items)} distinct items")
    out = []
    for q in sorted(items):
        sp, t1, t2 = s12_window_of(q, a.seed, a.eval_share)
        out.append({"qid": q, "split": sp, "t1": t1, "t2": t2, "n_sampled_statements": len(items[q])})
    s12_jsonl_append(path, out)
    (wd / "candidates_report.json").write_text(json.dumps({**rep, "items": len(out), "query": S12_Q_TENURE % {
        "pid": spec["pid"], "n": a.wdqs_sample, "lo": lo, "hi": hi}, "queries": a.wdqs_queries}, indent=1))


def s1_candidates_iucn(spec, a, http, wdqs, wd):
    path = wd / "candidates.jsonl"
    if path.exists():
        return
    ids, total = http.search_random("haswbstatement:P141", a.search_n)
    out = []
    for q in sorted(ids):
        sp, t1, t2 = s12_window_of(q, a.seed, a.eval_share)
        out.append({"qid": q, "split": sp, "t1": t1, "t2": t2, "sample_weight": 1.0})
    s12_jsonl_append(path, out)
    (wd / "candidates_report.json").write_text(json.dumps({"search": "haswbstatement:P141", "totalhits": total,
                                                           "sampled": len(ids)}, indent=1))


S12_Q_ORG_POS = """SELECT ?item ?d WHERE {
  ?item wdt:P31 wd:%(cls)s ; wdt:P576 ?d .
  FILTER(?d >= "%(lo)s"^^xsd:dateTime && ?d < "%(hi)s"^^xsd:dateTime)
  %(exclude)s
}"""


def s1_candidates_org(spec, a, http, wdqs, wd):
    """Case-control by class c (disjoint: an item counts under the first listed class it has). Stratum POS(w) = the class
    members whose CURRENT P576 lies in window w (all taken, from WDQS); REST = every other member, sampled uniformly
    (CirrusSearch random sort over the whole class; members of POS(w) drawn this way are dropped from REST). Inclusion
    probability of a REST member in window w: n_rest(w) / (N_class * share(w) - |POS(w)|); meta.sample_weight = 1 / it.
    The label comes from the T2 revision, so POS members can turn out 'no' (date recorded after T2)."""
    path = wd / "candidates.jsonl"
    if path.exists():
        return
    rep, out = {}, []
    windows = [("eval",) + S12_EVAL_WINDOW] + [("train",) + w for w in S12_TRAIN_WINDOWS]
    share = {S12_EVAL_WINDOW: a.eval_share}
    for w in S12_TRAIN_WINDOWS:
        share[w] = (1 - a.eval_share) / len(S12_TRAIN_WINDOWS)
    prev = []
    for cls, cname in S12_ORG_CLASSES:
        excl = " ".join(f"FILTER NOT EXISTS {{ ?item wdt:P31 wd:{p} . }}" for p in prev)
        try:
            rows = wdqs.query(S12_Q_ORG_POS % {"cls": cls, "exclude": excl, "lo": "2019-01-01T00:00:00Z",
                                               "hi": "2100-01-01T00:00:00Z"})
        except S12Truncated:      # the 60 s limit: one query per year instead
            s12_log(f"  {cname}: P576 query truncated; retrying year by year")
            rows = []
            for y in range(2019, 2028):
                rows += wdqs.query(S12_Q_ORG_POS % {"cls": cls, "exclude": excl, "lo": f"{y}-01-01T00:00:00Z",
                                                    "hi": f"{y + 1}-01-01T00:00:00Z" if y < 2027 else
                                                    "2100-01-01T00:00:00Z"})
        pos_date = {}
        for r in rows:
            if not r.get("item") or not r.get("d"):
                continue
            m = re.match(r"^(\d{4})-(\d\d)-(\d\d)", r["d"])
            if m:
                pos_date.setdefault(r["item"], []).append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        query = f"haswbstatement:P31={cls} " + " ".join(f"-haswbstatement:P31={p}" for p in prev)
        sample, total = http.search_random(query.strip(), a.search_n)
        prev.append(cls)
        crep = {"class": cname, "N_class": total, "wdqs_P576_rows_since_2019": len(rows), "rest_drawn": len(sample),
                "windows": {}}
        for (sp0, t1, t2) in windows:
            def in_w(q):
                sp, a1, a2 = s12_window_of(q, a.seed, a.eval_share)
                return (a1, a2) == (t1, t2)
            t1d, t2d = date.fromisoformat(t1), date.fromisoformat(t2)
            # WDQS returns P576 at day precision (year-precision dates become Jan 1): a coarse screen; the label is exact
            pos = sorted(q for q, ds in pos_date.items() if in_w(q) and any(t1d < d <= t2d for d in ds))
            rest = sorted(q for q in sample if in_w(q) and q not in set(pos))
            N_w = (total or 0) * share[(t1, t2)]
            p_rest = min(1.0, len(rest) / max(1.0, N_w - len(pos)))
            crep["windows"][f"{t1}..{t2}"] = {"pos": len(pos), "rest_sampled": len(rest), "N_class_in_window_pool": round(N_w),
                                              "rest_inclusion_prob": round(p_rest, 6)}
            for q in pos:
                sp, _, _ = s12_window_of(q, a.seed, a.eval_share)
                out.append({"qid": q, "split": sp, "t1": t1, "t2": t2, "cls": cls, "stratum": "pos_current_P576_in_window",
                            "incl_prob": 1.0, "sample_weight": 1.0})
            for q in rest:
                sp, _, _ = s12_window_of(q, a.seed, a.eval_share)
                out.append({"qid": q, "split": sp, "t1": t1, "t2": t2, "cls": cls, "stratum": "rest",
                            "incl_prob": p_rest, "sample_weight": round(1.0 / p_rest, 6) if p_rest > 0 else None})
        rep[cls] = crep
        s12_log(f"  {cname}: N={total}, P576 since 2019: {len(pos_date)} items, rest drawn {len(sample)}")
    seen, uniq = set(), []
    for r in out:
        if r["qid"] not in seen:
            seen.add(r["qid"])
            uniq.append(r)
    s12_jsonl_append(path, uniq)
    (wd / "candidates_report.json").write_text(json.dumps(rep, indent=1))


S1_CANDIDATES = {"tenure": s1_candidates_tenure, "org": s1_candidates_org, "iucn": s1_candidates_iucn}


# --------------------------------------------------------------------------- fetch stages
def s1_export_stage(http, wd, fname, todo, offset_of, ptypes, keep=None, batch=80):
    """Export the revision in force at each item's offset (grouped by offset, `batch` items per request) and append the
    compact form to <wd>/<fname>. Raw revisions are never written."""
    done = {r["qid"] for r in s12_jsonl_read(wd / fname)}
    groups = defaultdict(list)
    for q in todo:
        if q not in done:
            groups[offset_of(q)].append(q)
    n_total = sum(len(v) for v in groups.values())
    n = 0
    for off, qs in sorted(groups.items()):
        for i in range(0, len(qs), batch):
            chunk = qs[i:i + batch]
            got = http.export(chunk, f"{off}T23:59:59Z")
            rows = []
            for q in chunk:
                g = got.get(q)
                if not g:
                    rows.append({"qid": q, "at": off, "missing": True})
                    continue
                try:
                    ent = json.loads(g.get("text") or "null")
                except ValueError:
                    ent = None
                c = s12_compact(ent, ptypes)
                if c is not None and keep is not None and not c.get("redirect"):
                    c = {**c, "claims": {p: v for p, v in c["claims"].items() if p in keep}, "xids": {}}
                rows.append({"qid": q, "at": off, "rev": g.get("revid"), "ts": g.get("ts"), "c": c})
            s12_jsonl_append(wd / fname, rows)
            n += len(chunk)
            s12_log(f"  {fname}: {n}/{n_total} (offset {off}); requests so far {http.n_req}, 429s {http.n_429}")


def s1_select(spec, cand, t1rows, seed, L=None):
    """Per item: T1 eligibility -> chosen target (tenure: refined with labels when L is given)."""
    f_t1 = S1_FUNCS[spec["family"]][0]
    sel, drops = {}, Counter()
    for q, row in cand.items():
        t = t1rows.get(q)
        if t is None:
            continue
        if t.get("missing"):
            drops[f"{row['split']}/item_created_after_T1"] += 1
            continue
        if spec["family"] == "org":
            st, why, info = f_t1(spec, t["c"], q, row["t1"], seed, row)
        else:
            st, why, info = f_t1(spec, t["c"], q, row["t1"], seed)
        if st == "ok" and spec["family"] == "tenure" and L is not None:
            st, why, info = s1_tenure_refine(spec, info, L)
        if st != "ok":
            drops[f"{row['split']}/{why}"] += 1
            continue
        sel[q] = info
    return sel, drops


S12_Q_P54_END = """SELECT DISTINCT ?item WHERE {
  VALUES ?item { %(vals)s }
  ?item p:%(pid)s ?st . ?st pq:P582 ?e .
  FILTER(?e > "%(t1)sT23:59:59Z"^^xsd:dateTime && ?e <= "%(t2)sT23:59:59Z"^^xsd:dateTime)
}"""


def s1_pos_screen(spec, cand, wdqs, wd, chunk=4000):
    """Case-control screen for a tenure source: which candidates have, in CURRENT data, a statement of the target
    property ending inside their own window. Used only for the sampling design (all such items are taken, the rest
    is sampled); the label still comes from the T2 revision alone. Cached in <wd>/pos_screen.json."""
    path = wd / "pos_screen.json"
    if path.exists():
        return json.loads(path.read_text())
    by_w = defaultdict(list)
    for q, r in cand.items():
        by_w[(r["t1"], r["t2"])].append(q)
    pos = {}
    for (t1, t2), qs in sorted(by_w.items()):
        qs = sorted(qs)
        for i in range(0, len(qs), chunk):
            rows = wdqs.query(S12_Q_P54_END % {"vals": " ".join("wd:" + x for x in qs[i:i + chunk]), "pid": spec["pid"],
                                               "t1": t1, "t2": t2})
            for r in rows:
                if r.get("item"):
                    pos[r["item"]] = True
        s12_log(f"  pos screen {t1}..{t2}: {len(qs)} candidates, {sum(1 for q in qs if q in pos)} with a current end in the window")
    out = {q: bool(pos.get(q)) for q in cand}
    path.write_text(json.dumps(out))
    return out


def s1_fetch(spec, a):
    wd = Path(a.work) / "s1" / spec["name"]
    wd.mkdir(parents=True, exist_ok=True)
    http, wdqs = S12Http(a.min_interval), S12Wdqs()
    lab = S12Labels(a.work, wdqs)
    lab.ensure_props()
    S1_CANDIDATES[spec["family"]](spec, a, http, wdqs, wd)
    cand = {r["qid"]: r for r in s12_jsonl_read(wd / "candidates.jsonl")}
    # budget: the first N candidates of each pool in hash order (seeded); orgs keep every POS member
    pick = []
    for pool, n in (("eval", a.eval_items), ("trainval", a.train_items)):
        qs = [q for q, r in cand.items() if (r["split"] == "eval") == (pool == "eval")]
        if spec["family"] == "org":
            pos = [q for q in qs if cand[q]["stratum"] != "rest"]
            rest = sorted((q for q in qs if cand[q]["stratum"] == "rest"), key=lambda q: s12_hash01(a.seed, "take", q))
            pick += pos + rest[:max(0, n - len(pos))]
        else:
            pick += sorted(qs, key=lambda q: s12_hash01(a.seed, "take", q))[:n]
    if spec["family"] == "tenure" and getattr(a, "case_control", False):
        screen = s1_pos_screen(spec, cand, wdqs, wd)
        extra = sorted(q for q, v in screen.items() if v and q not in set(pick))
        s12_log(f"{spec['name']}: case-control adds {len(extra)} candidates whose current {spec['pid']} ends in their window")
        pick += extra
    (wd / "picked.json").write_text(json.dumps(sorted(pick)))
    s12_log(f"{spec['name']}: {len(cand)} candidates, {len(pick)} picked for T1 export")
    s1_export_stage(http, wd, "t1.jsonl.gz", pick, lambda q: cand[q]["t1"], lab.ptypes)
    t1rows = {r["qid"]: r for r in s12_jsonl_read(wd / "t1.jsonl.gz")}
    sel, _ = s1_select(spec, {q: cand[q] for q in pick}, t1rows, a.seed)
    ids = set()
    for q in sel:
        ids |= s12_ref_ids(t1rows[q]["c"])
    ids |= set(S12_IUCN)
    lab.ensure(ids)
    sel, _ = s1_select(spec, {q: cand[q] for q in pick}, t1rows, a.seed, lab.view())
    s1_export_stage(http, wd, "t2.jsonl.gz", sorted(sel), lambda q: cand[q]["t2"], lab.ptypes, keep=set(s1_t2_keep(spec)))
    st = json.loads((wd / "http_stats.json").read_text()) if (wd / "http_stats.json").exists() else {}
    for k, v in http.stats().items():
        st[k] = st.get(k, 0) + v if k != "min_interval_s" else v
    st["wdqs_requests"] = st.get("wdqs_requests", 0) + wdqs.n_req
    (wd / "http_stats.json").write_text(json.dumps(st, indent=1))


def s1_position_kind(label):
    """'Mayor of Lyon' -> 'mayor'; "conseiller municipal d'Oudrenne" -> 'conseiller municipal'; 'member of the
    Bundestag' -> 'member' (combined with the country it names a cluster)."""
    t = str(label).strip().lower()
    t = re.split(r"\s+(?:of|for|in|at|de|du|des|d'|del|della|di|von|van|der|för|i)\s+|\s+d'", t)[0]
    return t[:60] or "?"


def s1_composition(recs, qid, top=12):
    c = defaultdict(lambda: [0, 0.0])
    for r in recs:
        k = r["meta"].get("cluster")
        if k:
            c[k][0] += 1
            c[k][1] += r["labels"][qid]["yes"]
    return [{"cluster": k, "n": n, "yes_rate": round(y / n, 3)} for k, (n, y) in
            sorted(c.items(), key=lambda kv: -kv[1][0])[:top]]


def s1_build(spec, a):
    wd = Path(a.work) / "s1" / spec["name"]
    lab = S12Labels(a.work, None)
    L = lab.view()
    cand = {r["qid"]: r for r in s12_jsonl_read(wd / "candidates.jsonl")}
    pick = json.loads((wd / "picked.json").read_text())
    t1rows = {r["qid"]: r for r in s12_jsonl_read(wd / "t1.jsonl.gz")}
    t2rows = {r["qid"]: r for r in s12_jsonl_read(wd / "t2.jsonl.gz")}
    sel, drops = s1_select(spec, {q: cand[q] for q in pick}, t1rows, a.seed, L)
    _, f_t2, f_q = S1_FUNCS[spec["family"]]
    folder, gset, source = f"s1_{spec['name']}", f"wd_{spec['name']}", f"wikidata/{spec['name']}"
    by = defaultdict(list)
    funnel = Counter()
    # org case-control: the fetch budget may keep only part of a (class, window) REST sample; the realised inclusion
    # probability is the drawn probability times the share of drawn REST candidates that were picked (a uniform hash
    # subsample), so the weight is rescaled per (class, window)
    rest_scale, tenure_cc = {}, None
    if spec["family"] == "tenure" and (wd / "pos_screen.json").exists() and getattr(a, "case_control", False):
        # strata over the candidate pool of each window: POS = a current end in the window (all taken); REST = the
        # others, of which the budget took a uniform hash sample: weight = |REST| / |REST picked|
        screen = json.loads((wd / "pos_screen.json").read_text())
        pset = set(pick)
        tot, took = Counter(), Counter()
        for q, r in cand.items():
            if not screen.get(q):
                tot[r["t1"]] += 1
                took[r["t1"]] += q in pset
        tenure_cc = {"screen": screen, "rest_weight": {t: (tot[t] / took[t]) if took[t] else None for t in tot},
                     "pos": Counter(cand[q]["t1"] for q, v in screen.items() if v)}
    if spec["family"] == "org":
        drawn, taken = Counter(), Counter()
        pset = set(pick)
        for q, r in cand.items():
            if r.get("stratum") == "rest":
                drawn[(r["cls"], r["t1"])] += 1
                taken[(r["cls"], r["t1"])] += q in pset
        rest_scale = {k: taken[k] / drawn[k] for k in drawn if drawn[k]}
    for q in pick:
        funnel[f"{cand[q]['split']}/picked"] += 1
        funnel[f"{cand[q]['split']}/t1_eligible"] += q in sel
    leak = Counter()
    for q in sorted(sel):
        row, info, t1r, t2r = cand[q], sel[q], t1rows[q], t2rows.get(q)
        if t2r is None:
            drops[f"{row['split']}/t2_not_fetched"] += 1
            continue
        st, why, res = f_t2(spec, info, None if t2r.get("missing") else t2r["c"], row["t1"], row["t2"])
        if st != "ok":
            drops[f"{row['split']}/{why}"] += 1
            continue
        c1 = t1r["c"]
        # leakage guards (asserted): the state is the T1 revision; the tenure target carries no end / successor
        assert t1r["ts"] <= f"{row['t1']}T23:59:59Z", (q, t1r["ts"])
        if spec["family"] == "tenure":
            tgt = [s for s in s12_stmts(c1, spec["pid"]) if s["id"] == info["stmt"]]
            assert tgt and not any(k in tgt[0]["q"] for k in ("P582", "P1366", "P1534")), q
        if spec["family"] == "org":
            assert not s12_stmts(c1, "P576") and not s12_stmts(c1, "P582"), q
        state = s12_render(c1, row["t1"], L, spec["kind"], known_props=set(lab.ptypes) or None,
                           focus=(spec["pid"], info["stmt"]) if spec["family"] == "tenure" else None)
        late = [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", state) if d > row["t1"]]
        leak["states_with_a_full_date_after_T1"] += bool(late)
        qs = f_q(spec, c1, info, L, row["t1"], row["t2"])
        labs = s1_labels_of(spec, info, res)
        w = float(row.get("sample_weight") or 1.0)
        incl = row.get("incl_prob")
        if spec["family"] == "org" and row.get("stratum") == "rest":
            incl = float(row["incl_prob"]) * rest_scale[(row["cls"], row["t1"])]
            w = round(1.0 / incl, 6)
        if tenure_cc is not None:
            if tenure_cc["screen"].get(q):
                w, incl = 1.0, 1.0
            else:
                w = round(tenure_cc["rest_weight"][row["t1"]], 6)
                incl = round(1.0 / w, 6)
        meta = {"dataset": "wikidata", "group": q, "wikidata_item": q, "t1": row["t1"], "t2": row["t2"],
                "rev_t1": t1r["rev"], "ts_t1": t1r["ts"], "rev_t2": t2r["rev"], "ts_t2": t2r["ts"],
                "decision_date": row["t1"], "info_date": t1r["ts"][:10],
                "resolution_date": {k: row["t2"] for k in qs}, "horizon_days": (date.fromisoformat(row["t2"]) -
                                                                                date.fromisoformat(row["t1"])).days,
                "label_provenance": "outcome: the Wikidata revision in force on T2 (MediaWiki Special:Export)",
                "time_slice": "post_2025-06 (outcome after the model cutoff window)" if row["split"] == "eval" else
                "pre_2025-01 (T1 and T2 before 2025-01)",
                "sample_weight": w, "outcome": {k: v for k, v in res.items() if k != "yes"}, "selection": info}
        if spec["family"] == "org":
            meta["sampling"] = {"class": row.get("cls"), "stratum": row.get("stratum"), "inclusion_prob": incl,
                                "drawn_inclusion_prob": row.get("incl_prob")}
        if tenure_cc is not None:
            meta["sampling"] = {"stratum": "pos_current_end_in_window" if tenure_cc["screen"].get(q) else "rest",
                                "inclusion_prob": incl, "population": "the WDQS candidate pool of the window"}
        if spec["family"] == "tenure":
            meta["n_statements_sampled_by_wdqs"] = row.get("n_sampled_statements")
        base_id = f"wikidata/{q}"
        if spec["name"] == "office_end":
            # common shocks: outcomes of one office type in one country move together (an election ends every French
            # municipal mandate in March 2026), so the bootstrap cluster is (country, position kind), not the person
            ctry = next((L.get(v) or v for v in s12_items(c1, "P27")), "?")
            kind = s1_position_kind(L.get(info["value"]) or info["value"])
            meta["country"], meta["position_kind"] = ctry, kind
            meta["cluster"] = f"{ctry} / {kind}"
            base_id = f"wikidata-office/{s12_sha(meta['cluster'], 12)}"
        r = s12_record(source, state, qs, labs, meta, base_id=base_id, licence=S12_LIC_CC0,
                       rec_id=f"wd-{spec['name']}-{q}-{row['t1']}")
        r["split"] = row["split"]
        by[row["split"]].append(r)
    caps, out = {}, {}
    uncapped = None
    if spec["name"] == "office_end":
        # stratified design: at most N records per (country, position kind) cluster and split (seeded hash), so one
        # election calendar (French municipal mandates, ~half the uncapped eval) cannot dominate; unweighted
        uncapped = {sp: s1_composition(by.get(sp, []), spec["qid"]) for sp in ("train", "eval")}
        for sp in ("train", "val", "eval"):
            k = a.cluster_cap_eval if sp == "eval" else a.cluster_cap_train
            grp = defaultdict(list)
            for r in by.get(sp, []):
                grp[r["meta"]["cluster"]].append(r)
            kept = []
            for cl, rs in grp.items():
                rs.sort(key=lambda r: s12_hash01(a.seed, "cluster-cap", r["id"]))
                kept += rs[:k]
                drops[f"{sp}/cluster_cap"] += max(0, len(rs) - k)
            by[sp] = kept
    for sp in ("train", "val", "eval"):
        recs = sorted(by.get(sp, []), key=lambda r: r["id"])
        recs, caps[sp] = s12_cap(recs, S12_CAP_EVAL if sp == "eval" else S12_CAP_TRAIN, a.seed, f"{spec['name']}/{sp}")
        out[sp] = recs
    out["probe"] = [s12_probe_copy(r) for r in out["eval"]]
    windows = {"eval": {"t1": S12_EVAL_WINDOW[0], "t2": S12_EVAL_WINDOW[1]},
               "train_val": [{"t1": t1, "t2": t2} for t1, t2 in S12_TRAIN_WINDOWS]}
    http_stats = json.loads((wd / "http_stats.json").read_text()) if (wd / "http_stats.json").exists() else {}
    cand_rep = json.loads((wd / "candidates_report.json").read_text()) if (wd / "candidates_report.json").exists() else {}
    manifest = {"source": source, "shape": 1, "builder": S12_BUILDER, "built": datetime.now().isoformat(timespec="seconds"),
                "license": S12_LIC_CC0, "license_evidence": S12_LICENSE_EVIDENCE, "time_windows": windows,
                "split_rule": "item pool by sha1(seed|pool|QID) (eval share %.2f); train-pool items get one of the five "
                              "train windows by hash; val = %d%% of the train pool by hash: an item is in exactly one "
                              "split" % (a.eval_share, S12_VAL_PCT),
                "candidates": cand_rep, "funnel": dict(funnel), "drops": dict(sorted(drops.items())),
                "caps": caps, "http": http_stats, "leakage_scan": dict(leak),
                "leakage_controls": S1_LEAKAGE[spec["family"]], "label_rule": S1_LABEL_RULE[spec["family"]],
                "caveats": S1_CAVEATS, "splits": {sp: s12_split_stats(v) for sp, v in out.items()},
                "composition_top_clusters": ({sp: s1_composition(out[sp], spec["qid"]) for sp in ("train", "eval")}
                                             if spec["name"] == "office_end" else None),
                "composition_before_cluster_cap": uncapped,
                "case_control": ({"rest_weight_by_t1": tenure_cc["rest_weight"], "pos_candidates_by_t1": dict(tenure_cc["pos"]),
                                  "rule": "POS = candidates whose CURRENT statement of the property ends in their window "
                                          "(WDQS screen, all taken); REST = the others, uniformly sampled by the budget; "
                                          "labels from the T2 revision only"} if tenure_cc else None),
                "cluster_cap": ({"eval": a.cluster_cap_eval, "train_val": a.cluster_cap_train,
                                 "cluster": "(country of citizenship at T1, position kind = the position label before "
                                            "' of / de / d' ...')", "weights": "none: the population is cluster-capped"}
                                if spec["name"] == "office_end" else None),
                "gde1_notes": "eval = items whose T1 is 2025-06-30 and T2 2026-09-30; probe = the same questions with "
                              "the state withheld (knowledge probe: run it to see what the model knows without the "
                              "record)"}
    s12_write(a.out, folder, gset, out, manifest)
    return manifest


S1_LEAKAGE = {
    "tenure": ["state = the item revision in force on T1 (Special:Export offset=T1, dir=desc, limit=1), asserted ts <= T1",
               "the target statement is open at T1: no end time (P582), successor (P1366) or end cause (P1534) "
               "qualifier, asserted at build time",
               "the label is read only from the revision in force on T2",
               "items whose T2 revision dates the end on or before T1 are dropped (the tenure had already ended: the "
               "question's premise was false), symmetric across splits",
               "labels of referenced items are CURRENT labels (WDQS), not as of T1"],
    "org": ["state = the item revision in force on T1, asserted to carry no P576 and no P582",
            "label = P576 in the T2 revision; items whose T2 revision dates the dissolution on or before T1 are dropped",
            "the sampling strata use CURRENT data (P576 now in the window) but only through meta.sample_weight; the label "
            "never reads current data"],
    "iucn": ["state = the item revision in force on T1 (it shows the T1 status: that is the anchor, not a leak)",
             "label = the truthy P141 value in the T2 revision"],
}
S1_LABEL_RULE = {
    "tenure": "yes iff the T2 revision gives the target statement an end time in (T1, T2]; no iff it has none (or one "
              "after T2); dropped if the end is dated <= T1, straddles a window edge at its precision, is 'unknown "
              "value', or if the statement was removed / changed / deprecated, or carries a successor without an end",
    "org": "yes iff the T2 revision has a P576 value in (T1, T2]; no iff it has none (or one after T2); dropped if a "
           "value is <= T1 (premise false), straddles, or is 'unknown value'",
    "iucn": "status_at_t2 = the T2 revision's single truthy P141 value among the 8 IUCN categories (one-hot); "
            "status_changes = it differs from the T1 value; items with 0 or several truthy values, or other values, "
            "are dropped",
}
S1_CAVEATS = ["Labels are what Wikidata RECORDED by T2, not the real world: an event that happened in the window but was "
              "entered after T2 counts as 'no' (recording lag), identically in every split",
              "MEMORISATION: every pretrained model has seen Wikidata dumps that include the train windows' outcomes; only "
              "the eval split (outcomes recorded 2025-07 .. 2026-09) is after the knowledge window, and the probe split "
              "measures what a model answers without the state"]


def s1_cmd(name):
    def run(a):
        spec = S1_SPECS[name]
        if not a.no_fetch:
            s1_fetch(spec, a)
        m = s1_build(spec, a)
        print(json.dumps({k: m["splits"][k] for k in m["splits"]}, indent=1)[:4000])
        return 0
    return run


# =========================================================================== SHAPE 2: decisions on Wikidata's process pages
S2_MONTHS = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                             "September", "October", "November", "December"])}
S2_SIG_TS = re.compile(r"(\d{1,2}):(\d\d), (\d{1,2}) (January|February|March|April|May|June|July|August|September|"
                       r"October|November|December) (\d{4}) \(UTC\)")


def s2_ts(line):
    """Every signature timestamp in a line, as 'YYYY-MM-DDTHH:MM'."""
    return [f"{int(y):04d}-{S2_MONTHS[mo]:02d}-{int(d):02d}T{int(h):02d}:{mi}"
            for h, mi, d, mo, y in S2_SIG_TS.findall(line)]


def s2_users(line):
    return re.findall(r"\[\[\s*(?:User|Benutzer|Utilisateur|Usuario|Utente)\s*:\s*([^|\]/#]+)", line, re.I)


def s2_clean(t):
    """Wikitext -> readable text: signatures become '- user (YYYY-MM-DD HH:MM UTC)', item / property templates their
    ids, vote templates words, other templates / HTML / link markup removed."""
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = S2_SIG_TS.sub(lambda m: f"({int(m.group(5)):04d}-{S2_MONTHS[m.group(4)]:02d}-{int(m.group(3)):02d} "
                                f"{int(m.group(1)):02d}:{m.group(2)} UTC)", t)
    t = re.sub(r"\{\{\s*(?:Q|Item|Q'|Qid)\s*\|\s*Q?(\d+)\s*\}\}", r"Q\1", t, flags=re.I)
    t = re.sub(r"\{\{\s*(?:P|Property|P')\s*\|\s*P?(\d+)\s*\}\}", r"P\1", t, flags=re.I)
    t = re.sub(r"\{\{\s*(?:L|Lexeme)\s*\|\s*L?(\d+)\s*\}\}", r"L\1", t, flags=re.I)
    votes = {"support": "Support", "oppose": "Oppose", "o": "Oppose", "s": "Support", "neutral": "Neutral",
             "comment": "Comment:", "keep": "Keep (vote)", "vk": "Keep (vote)", "delete": "Delete (vote)",
             "vd": "Delete (vote)", "weak support": "Weak support", "weak oppose": "Weak oppose",
             "strong support": "Strong support", "strong oppose": "Strong oppose", "on hold": "On hold:",
             "question": "Question:", "info": "Info:", "merge": "Merge (vote)", "vm": "Merge (vote)",
             "agree": "Agree", "disagree": "Disagree", "int:talkpagelinktext": "talk", "vote keep": "Keep (vote)",
             "vote delete": "Delete (vote)", "wait": "Wait:", "int motivation": "Motivation", "int:motivation": "Motivation",
             "int:talk": "Discussion", "int discussion": "Discussion", "int:discussion": "Discussion"}
    t = re.sub(r"\{\{\s*([^{}|]+?)\s*\}\}", lambda m: votes.get(m.group(1).strip().lower(), "{{" + m.group(1) + "}}"), t)
    t = re.sub(r"\{\{\s*(?:ping|re|reply to|u|replyto|noping|user)\s*\|([^{}]*)\}\}",
               lambda m: "@" + ", @".join(x.strip() for x in m.group(1).split("|") if x.strip() and "=" not in x), t, flags=re.I)
    t = re.sub(r"\{\{\s*(?:Statement)\s*\|([^{}]*)\}\}", lambda m: " ".join(x.strip() for x in m.group(1).split("|")), t)
    for _ in range(3):
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\(?\s*\[\[\s*(?:User[ _]talk|Benutzer[ _]Diskussion|Discussion[ _]utilisateur)\s*:[^\]]*\]\]\s*\)?", "", t,
               flags=re.I)
    t = re.sub(r"\[\[\s*(?:User|Special:Contributions|Benutzer|Utilisateur)\s*:\s*([^|\]]+)\|[^\]]*\]\]",
               lambda m: m.group(1).split("/")[-1].strip(), t, flags=re.I)
    t = re.sub(r"\[\[\s*:?\s*(?:d:)?([^|\]]+)\|([^\]]*)\]\]", r"\2", t)
    t = re.sub(r"\[\[\s*:?\s*([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]", r"\2 (\1)", t)
    t = re.sub(r"\[(https?://[^\s\]]+)\]", r"\1", t)
    t = re.sub(r"</?(?:span|sup|sub|small|big|s|u|b|i|font|div|code|tt|del|ins|nowiki|br|p|center)\b[^>]*>", "", t, flags=re.I)
    t = re.sub(r"'''?", "", t)
    t = html.unescape(t).replace("\u200e", "").replace("\u200f", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def s2_indent(line):
    m = re.match(r"^([:*#]+)\s*", line)
    return (len(m.group(1)), line[m.end():]) if m else (0, line)


# --------------------------------------------------------------------------- Requests for deletions
S2_RFD_CLOSE = re.compile(r"\{\{\s*(deleted|done|not ?done|not ?deleted|nd|kept|merged|redirected|withdrawn|already deleted)"
                          r"\s*(?:\||\}\})", re.I)
S2_RFD_LEAK = re.compile(
    r"\{\{\s*(?:deleted|done|not ?done|not ?deleted|nd|kept|merged|redirected|withdrawn)\b|"
    r"\bI(?:'ve| have)? (?:just )?(?:deleted|merged|redirected|restored)\b|"
    r"\b(?:item|it|this|these|both|all)\b[^.]{0,30}\b(?:has|have|was|were|is|are) (?:now |already |been )*"
    r"(?:deleted|merged|redirected|kept)\b|\balready (?:deleted|merged|redirected)\b|"
    r"\bredirect(?:ed)? (?:created|to Q)|\bmerged (?:into|with|to) Q\d+|\bclos(?:ed|ing) (?:as|this)\b|"
    r"\bnot done\b|\bnot deleted\b|\bkept\b", re.I)
S2_MERGE_REQUEST = re.compile(r"\b(?:be|can|should|could|will|would|to|please|must|need|needs|might|may|shall)\s+$")


def s2_rfd_leak(text):
    """The outcome detector for deletion requests: S2_RFD_LEAK, plus 'merged' used as a done deed ('items merged', 'I
    have merged', 'has been merged', 'merged with Q..') rather than a request ('should be merged', 'can be merged'),
    plus 'redirect restored'."""
    if S2_RFD_LEAK.search(text) or re.search(r"\bredirect(?:ion)? (?:was |has been )?restored\b", text, re.I):
        return True
    for m in re.finditer(r"\bmerged\b", text, re.I):
        if not S2_MERGE_REQUEST.search(text[max(0, m.start() - 30):m.start()].lower()):
            return True
    return False


S2_RFD_CRIT = {"deleted": "Deleted: the nominated item(s) were deleted.",
               "kept": "Kept: the request was declined (not done / kept) and the item(s) stayed.",
               "merged": "Merged: the item was merged into another item and left as a redirect instead of being deleted."}


def s2_rfd_outcome(line):
    m = S2_RFD_CLOSE.search(line)
    if not m:
        return None
    k = m.group(1).lower().replace(" ", "")
    if k in ("deleted", "alreadydeleted"):
        return "deleted"
    if k in ("merged", "redirected"):
        return "merged"
    if k in ("notdone", "kept", "notdeleted", "nd"):
        return "kept"
    if k == "withdrawn":
        return "withdrawn"
    if re.search(r"redirect|merg", line, re.I):
        return "merged"
    return "deleted"   # {{done}} on a deletion request


def s2_rfd_sections(text):
    parts = re.split(r"(?m)^==(?!=)\s*(.*?)\s*==\s*$", text)
    return [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def s2_rfd_parse(heading, body, page):
    """One archived request -> dict(items, filed, close_ts, outcome, nomination lines, pre-close lines, ...) or
    (None, reason)."""
    def ids_in(t):
        return set(re.findall(r"\b[QL]\d+\b", t)) | {"Q" + n for n in re.findall(r"\{\{\s*Q\s*\|\s*Q?(\d+)", t, re.I)}
    items = ids_in(heading) | set(re.findall(r"rfd links\s*\|\s*(?:1\s*=\s*)?(?:Lexeme:)?([QL]\d+)", body, re.I))
    for g in re.findall(r"\{\{\s*Rfd group(?:/1)?\s*\|([^}]*)\}\}", body, re.I):
        items |= ids_in(g)
    lines = [ln for ln in body.split("\n")]
    sig = [i for i, ln in enumerate(lines) if s2_ts(ln)]
    if not sig:
        return None, "no_signature"
    i0 = sig[0]
    if not items:      # bulk requests list their items in the nomination ({{Q|..}} lines)
        items = ids_in("\n".join(lines[:i0 + 1]))
    items = sorted(items, key=lambda x: (x[0], int(x[1:])))
    if not items:
        return None, "no_item_id"
    if s2_rfd_outcome(lines[i0]):
        return None, "unsigned_nomination"
    nom = [ln for ln in lines[:i0 + 1] if ln.strip() and not re.match(r"^\s*\{\{\s*(rfd links|Rfd group)", ln, re.I)]
    rationale = " ".join(re.findall(r"rfd links\s*\|[^}]*?2\s*=\s*([^}]*)\}\}", body, re.I))
    if not nom:
        return None, "empty_nomination"
    filed = s2_ts(lines[i0])[0]
    disc = lines[i0 + 1:]
    closes = [(j, s2_rfd_outcome(ln)) for j, ln in enumerate(disc) if s2_rfd_outcome(ln)]
    if not closes:
        return None, "no_close"
    outs = {o for _, o in closes}
    if "withdrawn" in outs:
        return None, "withdrawn"
    if len(outs) > 1:
        return None, "conflicting_closes"
    j_close, outcome = closes[0]
    cts = s2_ts(disc[j_close])
    if not cts:
        later = [s2_ts(ln) for ln in disc[j_close + 1:] if s2_ts(ln)]
        cts = later[0] if later else []
    if not cts:
        return None, "close_without_timestamp"
    close_ts = cts[0]
    if close_ts < filed:
        return None, "close_before_filing"
    pre, cut_leak = [], 0
    for ln in disc[:j_close]:
        if s2_rfd_leak(ln):
            cut_leak = 1
            break
        ts = s2_ts(ln)
        if ts and max(ts) > close_ts:
            continue          # a comment inserted above the close but written after it
        if ln.strip():
            pre.append(ln)
    closer = (s2_users(disc[j_close]) or [None])[-1]
    return {"items": items, "filed": filed, "close_ts": close_ts, "outcome": outcome, "nomination": nom, "pre": pre,
            "rationale": rationale.strip(), "nominator": (s2_users(lines[i0]) or [None])[-1], "closer": closer,
            "page": page, "heading": heading, "cut_at_leak_line": cut_leak, "n_pre_comments": len(pre)}, "ok"


def s2_render_lines(lines):
    out = []
    for ln in lines:
        d, rest = s2_indent(ln)
        c = s2_clean(rest)
        if c:
            out.append("  " * d + ("- " if d else "") + c)
    return out


def s2_rfd_state(p):
    s = [f"WIKIDATA REQUEST FOR DELETION (Wikidata:Requests for deletions), filed {p['filed'].replace('T', ' ')} UTC",
         "Item(s) nominated: " + ", ".join(p["items"]), "", "Nomination:"]
    s += ["  " + x for x in s2_render_lines(p["nomination"])]
    s += ["", "Discussion before the close:" if p["pre"] else "Discussion before the close: (none)"]
    s += s2_render_lines(p["pre"])
    return "\n".join(s)[:S12_MAX_STATE]


def s2_rfd_fetch(a, http, wd):
    """Archive day pages (one wikitext per day) for the eval window and a hash-sampled set of train days."""
    tpath = wd / "rfd_titles.json"
    if tpath.exists():
        titles = json.loads(tpath.read_text())
    else:
        titles = []
        for y in range(2019, 2027):
            titles += [t for t in http.allpages(f"Requests for deletions/Archive/{y}/")
                       if re.search(r"/Archive/\d{4}/\d\d/\d\d$", t)]
        tpath.write_text(json.dumps(titles))
    def day(t):
        return "-".join(t.rsplit("/", 3)[-3:])
    eval_days = [t for t in titles if day(t) >= "2025-06-25"]
    train_all = [t for t in titles if day(t) <= "2025-01-07"]
    train_days = sorted(train_all, key=lambda t: s12_hash01(a.seed, "rfd-day", t))[:a.train_days]
    want = sorted(set(eval_days + train_days))
    done = {r["title"] for r in s12_jsonl_read(wd / "rfd_pages.jsonl.gz")}
    todo = [t for t in want if t not in done]
    s12_log(f"rfd: {len(titles)} archive days listed; eval days {len(eval_days)}, train days {len(train_days)} of "
            f"{len(train_all)}; to fetch {len(todo)}")
    for i in range(0, len(todo), 40):
        got = http.page_texts(todo[i:i + 40])
        s12_jsonl_append(wd / "rfd_pages.jsonl.gz", [{"title": t, "ts": v[0], "revid": v[1], "text": v[2]}
                                                   for t, v in got.items()])
        s12_log(f"  rfd pages {min(i + 40, len(todo))}/{len(todo)}")
    (wd / "rfd_sample.json").write_text(json.dumps({"n_days_listed": len(titles), "eval_days": len(eval_days),
                                                     "train_days_sampled": len(train_days),
                                                     "train_days_available": len(train_all)}))


def s2_split_of(close_date, key, seed):
    if close_date >= S12_S2_EVAL_FROM:
        return "eval"
    if close_date < S12_S2_TRAIN_BEFORE:
        return "val" if s12_hash01(seed, "val", key) < S12_VAL_PCT / 100 else "train"
    return None


def s2_rfd_build(a):
    wd = Path(a.work) / "s2" / "rfd"
    pages = s12_jsonl_read(wd / "rfd_pages.jsonl.gz")
    drops, by = Counter(), defaultdict(list)
    seen, seen_ids = set(), set()
    for pg in sorted(pages, key=lambda p: p["title"]):
        for heading, body in s2_rfd_sections(pg["text"] or ""):
            p, why = s2_rfd_parse(heading, body, pg["title"])
            if p is None:
                drops[why] += 1
                continue
            sp = s2_split_of(p["close_ts"][:10], p["items"][0], a.seed)
            if sp is None:
                drops["closed_2025-01_to_2025-06_gap"] += 1
                continue
            state = s2_rfd_state(p)
            if s2_rfd_leak(state):
                drops["outcome_detector_hit_after_cut"] += 1
                continue
            key = (tuple(p["items"]), p["filed"])
            if key in seen:
                drops["duplicate_request"] += 1
                continue
            seen.add(key)
            items = ", ".join(p["items"][:6]) + (f" and {len(p['items']) - 6} more" if len(p["items"]) > 6 else "")
            qs = {"outcome": {"type": "choice",
                              "instructions": (f"This is a request on Wikidata:Requests for deletions, filed "
                                               f"{p['filed'][:10]} for {items}. Given the nomination and the discussion "
                                               f"before the close, how was the request closed?"),
                              "criteria": dict(S2_RFD_CRIT)}}
            labs = {"outcome": {k: (1.0 if k == p["outcome"] else 0.0) for k in S2_RFD_CRIT}}
            url = "https://www.wikidata.org/wiki/" + pg["title"].replace(" ", "_")
            meta = {"dataset": "wikidata", "group": p["items"][0], "page": pg["title"], "page_revid": pg["revid"],
                    "items": p["items"], "filed": p["filed"], "close_ts": p["close_ts"], "nominator": p["nominator"],
                    "closer": p["closer"], "decision_date": p["close_ts"][:10],
                    "info_date": max([p["filed"]] + [t for ln in p["pre"] for t in s2_ts(ln)])[:10],
                    "resolution_date": {"outcome": p["close_ts"][:10]}, "label_provenance": "recorded close of the "
                    "request (closing template / DeltaBot deletion notice), archive page", "raw_outcome": p["outcome"],
                    "cut_at_leak_line": p["cut_at_leak_line"], "n_pre_comments": p["n_pre_comments"],
                    "time_slice": "closed >= 2025-07-01" if sp == "eval" else "closed < 2025-01-01", "sample_weight": 1.0}
            r = s12_record("wikidata/rfd", state, qs, labs, meta,
                           base_id=f"wikidata-rfd/{s12_sha((p['nominator'] or '') + '|' + (p['rationale'] or '') + '|' + p['filed'][:10])}",
                           licence=S12_LIC_BYSA.format(page=pg["title"], url=url),
                           rec_id=f"wd-rfd-{s12_sha(pg['title'] + '|' + heading + '|' + p['filed'])}")
            if r["id"] in seen_ids:
                drops["duplicate_record_id"] += 1
                continue
            seen_ids.add(r["id"])
            r["split"] = sp
            by[sp].append(r)
    # an item never in both: drop train / val requests that name an eval item
    ev_items = {q for r in by["eval"] for q in r["meta"]["items"]}
    for sp in ("train", "val"):
        keep = [r for r in by[sp] if not set(r["meta"]["items"]) & ev_items]
        drops[f"{sp}_shares_an_item_with_eval"] += len(by[sp]) - len(keep)
        by[sp] = keep
    return s2_finish(a, "rfd", by, drops, {
        "pages": len(pages), "sample": json.loads((wd / "rfd_sample.json").read_text()) if (wd / "rfd_sample.json").exists() else None,
        "question": "outcome: choice deleted / kept / merged",
        "label_rule": "the first closing template in the section ({{deleted}}, DeltaBot's {{deleted|admin=..}}, {{done}} "
                      "= deleted unless the line says redirect / merge, {{not done}} / {{kept}} = kept, {{merged}} / "
                      "{{redirected}} / {{done}} + 'redirect' = merged); sections with conflicting closes, withdrawn "
                      "requests or no close are dropped",
        "leakage_controls": ["the state is the nomination plus the discussion lines BEFORE the first closing line",
                             "pre-close lines that carry a signature dated after the close are dropped (comments "
                             "inserted above the close later)",
                             "the discussion is cut at the first line the outcome detector matches (S2_RFD_LEAK: closing "
                             "templates, 'I deleted / merged', 'has been deleted / merged / kept', 'already deleted', "
                             "'redirect created', 'not done', 'kept'); a state the detector still matches is dropped",
                             "the nominated items' content is NOT included: deleted items cannot be read, so including "
                             "it only for kept items would leak the label",
                             "train / val requests naming an item that has an eval request are dropped"]})


# --------------------------------------------------------------------------- Property proposals
S2_PP_SKIP = re.compile(r"/(Archive|Overview|Header|navbar|Generic|Natural science|Authority control|Creative work|"
                        r"Place|Person|Organization|Sports|Transportation|Lexemes|Sister projects|Wikimedia|Computing|"
                        r"Commons|Event|Unsorted|Proposal Template|Template|Mentions|Statistics|Sandbox)(/|$)", re.I)


def s2_template(text, name):
    """The first {{name ...}} template (balanced braces) -> (start, end) or None."""
    m = re.search(r"\{\{\s*" + name + r"\s*[\n|]", text, re.I)
    if not m:
        return None
    i, depth = m.start(), 0
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                return m.start(), i
            continue
        i += 1
    return None


def s2_params(tpl):
    inner = tpl[2:-2]
    parts, depth, cur, i = [], 0, [], 0
    while i < len(inner):
        two = inner[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            cur.append(two)
            i += 2
            continue
        if two in ("}}", "]]"):
            depth -= 1
            cur.append(two)
            i += 2
            continue
        if inner[i] == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(inner[i])
        i += 1
    parts.append("".join(cur))
    out = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def s2_translate_en(v):
    m = re.search(r"\|\s*en\s*=\s*(.*?)(?=\n\s*\||\}\}|$)", v or "", re.S)
    return s2_clean(m.group(1)) if m else s2_clean(v or "")


def s2_pp_body(title, text, close_ts, pid, close_rx=None):
    """The state parts of a proposal page text: name, template fields, the lines after the template up to the first
    closing / outcome line (and without lines signed after close_ts), the proposal time. Used on the revision in force
    just before the close (the state) and, for the outcome and close time, on the current page."""
    span = s2_template(text or "", r"Property[ _]proposal")
    if not span:
        return None
    prm = s2_params(text[span[0]:span[1]])
    head = text[:span[0]]
    name = s2_translate_en(head) if "TranslateThis" in head else ""
    name = re.sub(r"^[=\s]+|[=\s]+$", "", name).strip() or title.split("/", 1)[-1]
    after = text[span[1]:].split("\n")
    sig_all = [t for ln in text.split("\n") for t in s2_ts(ln)]
    cut_rx = re.compile(r"\{\{\s*(?:done|not ?done|withdrawn)\s*\}\}|\bcreated as\b|has been created|\bnot done\b|"
                        r"\bwithdraw" + (r"|\{\{\s*P\s*\|\s*P?%s\s*\}\}|\b%s\b" % (pid[1:], pid) if pid else ""), re.I)
    j_close = None
    if close_rx is not None:
        j_close = next((j for j, ln in enumerate(after) if close_rx.search(ln) and s2_ts(ln)), None)
        if j_close is None:
            j_close = next((j for j, ln in enumerate(after) if close_rx.search(ln)), None)
    j_cut = next((j for j, ln in enumerate(after) if cut_rx.search(ln)), None)
    stop = min(x for x in (j_close, j_cut, len(after)) if x is not None)
    pre = []
    for ln in after[:stop]:
        ts = s2_ts(ln)
        if close_ts and ts and max(ts) > close_ts:
            continue
        pre.append(ln)
    return {"name": name, "params": prm, "pre": pre, "after": after, "j_close": j_close,
            "proposed": min(sig_all) if sig_all else None,
            "cut_before_close": int(j_cut is not None and (j_close is None or j_cut < j_close))}


def s2_pp_parse(title, text):
    """Outcome (status field), created property id, close time (signature of the first closing line) and the state
    parts of the CURRENT page text."""
    span = s2_template(text or "", r"Property[ _]proposal")
    if not span:
        return None, "no_proposal_template"
    prm = s2_params(text[span[0]:span[1]])
    status = re.sub(r"<!--.*?-->", "", prm.get("status", ""), flags=re.S).strip().lower()
    if re.fullmatch(r"p?\d+", status):
        outcome, pid = "created", "P" + status.lstrip("p")
    elif status.replace(" ", "") in ("notdone",):
        outcome, pid = "not_done", None
    elif status.startswith("withdrawn"):
        return None, "withdrawn"
    else:
        return None, f"open_or_other_status:{status[:20] or 'empty'}"
    if outcome == "created":
        n = pid[1:]
        close_rx = re.compile(r"\{\{\s*P\s*\|\s*P?%s\s*\}\}|\bP%s\b|created|\{\{\s*done\s*\}\}" % (n, n), re.I)
    else:
        close_rx = re.compile(r"\{\{\s*not ?done\s*\}\}|\bnot done\b|no consensus|\bclos(?:ed|ing)\b", re.I)
    b = s2_pp_body(title, text, None, pid, close_rx)
    after, j_close = b["after"], b["j_close"]
    close_ts = None
    if j_close is not None:
        ts = s2_ts(after[j_close]) or next((s2_ts(ln) for ln in after[j_close + 1:] if s2_ts(ln)), [])
        close_ts = ts[0] if ts else None
    if close_ts:      # re-cut with the close time known (drops lines inserted above the close but signed after it)
        b = s2_pp_body(title, text, close_ts, pid, close_rx)
    return {"title": title, "name": b["name"], "outcome": outcome, "pid": pid, "params": b["params"], "pre": b["pre"],
            "close_ts": close_ts, "proposed": b["proposed"],
            "closer": (s2_users(after[j_close]) or [None])[-1] if j_close is not None else None,
            "cut_before_close": b["cut_before_close"], "state_source": "current page"}, "ok"


S2_PP_SKIP_PARAMS = {"status", "subpage", "topic", "translations"}


def s2_pp_ids(p):
    """Item / property ids named in the proposal's fields (for label lookups; never the created property's id)."""
    ids = set()
    for k, v in p["params"].items():
        if k not in S2_PP_SKIP_PARAMS:
            ids |= set(re.findall(r"\b[QP]\d+\b", s2_clean(v)))
            ids |= {"Q" + n for n in re.findall(r"\{\{\s*Q\s*\|\s*Q?(\d+)", v, re.I)}
            ids |= {"P" + n for n in re.findall(r"\{\{\s*P\s*\|\s*P?(\d+)", v, re.I)}
    return ids - ({p["pid"]} if p.get("pid") else set())


def s2_pp_state(p, L=None):
    L = L or {}

    def lab(t):
        return re.sub(r"\b([QP]\d+)\b", lambda m: f"{L[m.group(1)]} ({m.group(1)})" if L.get(m.group(1)) else m.group(1), t)
    s = [f"WIKIDATA PROPERTY PROPOSAL (Wikidata:Property proposal): {p['name']}",
         f"Proposed: {(p['proposed'] or '?').replace('T', ' ')} UTC", ""]
    for k, v in p["params"].items():
        if k in S2_PP_SKIP_PARAMS or not v.strip():
            continue
        val = s2_translate_en(v) if "translatethis" in v.lower() else s2_clean(v)
        if val:
            s.append(f"{k}: {lab(val)[:600]}")
    body = []
    for x in s2_render_lines(p["pre"]):
        h = re.match(r"^=+\s*(.*?)\s*=+$", x)
        if h:
            if h.group(1):
                body.append(f"[{h.group(1)}]")
            continue
        body.append(x)
    s += ["", "Motivation and discussion before the close:"] + (body or ["(none)"])
    return "\n".join(s)[:S12_MAX_STATE]


def s2_parse_history(body):
    """Special:Export with history -> {title: [(revid, timestamp, text), ...]} (oldest first)."""
    import xml.etree.ElementTree as ET
    out = {}
    for _, el in ET.iterparse(io.BytesIO(body), events=("end",)):
        if _s12_local(el.tag) != "page":
            continue
        title, revs = None, []
        for ch in el:
            n = _s12_local(ch.tag)
            if n == "title":
                title = ch.text
            elif n == "revision":
                rv = {}
                for x in ch:
                    m = _s12_local(x.tag)
                    if m in ("id", "timestamp", "text"):
                        rv[m] = x.text
                revs.append((int(rv.get("id") or 0), rv.get("timestamp") or "", rv.get("text") or ""))
        if title:
            out[title] = sorted(revs, key=lambda r: r[1])
        el.clear()
    return out


def s2_pp_close(p, created):
    """(close time 'YYYY-MM-DDTHH:MM', source) of a parsed proposal: its closing signature, else (created ones) the
    property's creation time."""
    if p["close_ts"]:
        return p["close_ts"], "closing comment"
    if p["outcome"] == "created":
        ct = created.get("Property:" + p["pid"])
        if ct:
            return ct[:16], "property_creation_time"
    return None, None


def s2_pp_fetch(a, http, wd):
    tpath = wd / "pp_titles.json"
    if tpath.exists():
        titles = json.loads(tpath.read_text())
    else:
        titles = http.allpages("Property proposal/")
        tpath.write_text(json.dumps(titles))
    titles = [t for t in titles if not S2_PP_SKIP.search(t)]
    done = {r["title"] for r in s12_jsonl_read(wd / "pp_pages.jsonl.gz")}
    todo = [t for t in titles if t not in done]
    s12_log(f"property proposals: {len(titles)} candidate pages, {len(todo)} to fetch ({math.ceil(len(todo) / 50)} requests)")
    for i in range(0, len(todo), 50):
        got = http.page_texts(todo[i:i + 50])
        s12_jsonl_append(wd / "pp_pages.jsonl.gz", [{"title": t, "ts": v[0], "revid": v[1], "text": v[2]}
                                                  for t, v in got.items()])
        missing = [t for t in todo[i:i + 50] if t not in got]
        if missing:
            s12_jsonl_append(wd / "pp_pages.jsonl.gz", [{"title": t, "missing": True} for t in missing])
        if (i // 50) % 10 == 0:
            s12_log(f"  proposals {min(i + 50, len(todo))}/{len(todo)}; requests {http.n_req}, 429s {http.n_429}")
    # created proposals whose close line has no timestamp: the property's creation time (first revision)
    pages = s12_jsonl_read(wd / "pp_pages.jsonl.gz")
    need = []
    for pg in pages:
        if pg.get("missing"):
            continue
        p, why = s2_pp_parse(pg["title"], pg["text"])
        if p and p["outcome"] == "created" and not p["close_ts"]:
            need.append("Property:" + p["pid"])
    ids = set()
    for pg in pages:
        if not pg.get("missing"):
            p, _ = s2_pp_parse(pg["title"], pg["text"])
            if p:
                ids |= s2_pp_ids(p)
    lab = S12Labels(a.work, S12Wdqs())
    lab.ensure(ids)
    have = {r["title"] for r in s12_jsonl_read(wd / "pp_created.jsonl")}
    need = sorted(set(need) - have)
    for i in range(0, len(need), 80):
        data = {"title": "Special:Export", "pages": "\n".join(need[i:i + 80]), "action": "submit", "dir": "asc",
                "limit": "1", "offset": "1"}
        got = s12_parse_export(http.request(S12_INDEX, urllib.parse.urlencode(data).encode()))
        s12_jsonl_append(wd / "pp_created.jsonl", [{"title": t, "created": (got.get(t) or {}).get("ts")}
                                                   for t in need[i:i + 80]])
    # the STATE comes from the revision in force just before the close: page histories (Special:Export dir=asc, up to
    # 200 revisions per page, ~40 pages per request) for every eval proposal and a hash sample of train / val ones
    created = {r["title"]: r["created"] for r in s12_jsonl_read(wd / "pp_created.jsonl")}
    tg = {"eval": [], "tv": []}
    for pg in pages:
        if pg.get("missing"):
            continue
        p, _ = s2_pp_parse(pg["title"], pg["text"])
        if not p:
            continue
        ct, _ = s2_pp_close(p, created)
        sp = s2_split_of(ct[:10], pg["title"], a.seed) if ct else None
        if sp:
            tg["eval" if sp == "eval" else "tv"].append((pg["title"], ct))
    tv = sorted(tg["tv"], key=lambda x: s12_hash01(a.seed, "pp-hist", x[0]))[:a.pp_train]
    want = dict(tg["eval"] + tv)
    have = {r["title"] for r in s12_jsonl_read(wd / "pp_atclose.jsonl.gz")}
    todo = sorted(t for t in want if t not in have)
    s12_log(f"property proposals: pre-close revisions for {len(want)} ({len(tg['eval'])} eval, {len(tv)} of "
            f"{len(tg['tv'])} train/val); {len(todo)} to fetch")
    for i in range(0, len(todo), 40):
        chunk = todo[i:i + 40]
        data = {"title": "Special:Export", "pages": "\n".join(chunk), "action": "submit", "dir": "asc", "offset": "1",
                "limit": "200"}
        hist = s2_parse_history(http.request(S12_INDEX, urllib.parse.urlencode(data).encode()))
        rows = []
        for t in chunk:
            revs = hist.get(t) or []
            before = [r for r in revs if r[1][:16] < want[t]]
            if before:
                rv = before[-1]
                rows.append({"title": t, "close_ts": want[t], "revid": rv[0], "ts": rv[1], "text": rv[2],
                             "n_revisions": len(revs), "history_truncated": len(revs) >= 200})
            else:
                rows.append({"title": t, "close_ts": want[t], "none": True, "n_revisions": len(revs)})
        s12_jsonl_append(wd / "pp_atclose.jsonl.gz", rows)
        if (i // 40) % 10 == 0:
            s12_log(f"  pre-close revisions {min(i + 40, len(todo))}/{len(todo)}; requests {http.n_req}, 429s {http.n_429}")


def s2_pp_build(a):
    wd = Path(a.work) / "s2" / "property_proposal"
    pages = s12_jsonl_read(wd / "pp_pages.jsonl.gz")
    created = {r["title"]: r["created"] for r in s12_jsonl_read(wd / "pp_created.jsonl")}
    atclose = {r["title"]: r for r in s12_jsonl_read(wd / "pp_atclose.jsonl.gz")}
    L = S12Labels(a.work, None).view()
    drops, by = Counter(), defaultdict(list)
    for pg in sorted(pages, key=lambda x: x["title"]):
        if pg.get("missing"):
            drops["page_missing"] += 1
            continue
        p, why = s2_pp_parse(pg["title"], pg["text"])
        if p is None:
            drops[why.split(":")[0]] += 1
            continue
        p["close_ts"], approx = s2_pp_close(p, created)
        if not p["close_ts"]:
            drops["no_close_timestamp"] += 1
            continue
        at = atclose.get(pg["title"])
        if at is None:
            drops["pre_close_revision_not_sampled"] += 1
            continue
        if at.get("none"):
            drops["no_revision_before_the_close"] += 1
            continue
        b = s2_pp_body(pg["title"], at["text"], p["close_ts"], p["pid"])
        if b is None:
            drops["no_proposal_template_before_the_close"] += 1
            continue
        p.update({k: b[k] for k in ("name", "params", "pre", "proposed", "cut_before_close")})
        p["state_source"] = {"revid": at["revid"], "ts": at["ts"], "rule": "the last revision before the close"}
        if p["proposed"] and p["close_ts"] < p["proposed"]:
            drops["close_before_proposal"] += 1
            continue
        sp = s2_split_of(p["close_ts"][:10], pg["title"], a.seed)
        if sp is None:
            drops["closed_2025-01_to_2025-06_gap"] += 1
            continue
        state = s2_pp_state(p, L)
        leak = re.compile(r"\{\{\s*(?:done|not ?done|withdrawn)\s*\}\}|\bcreated as\b|has been created|\bnot done\b" +
                          (r"|\b%s\b" % p["pid"] if p["pid"] else ""), re.I)
        if leak.search(state):
            drops["outcome_detector_hit_after_cut"] += 1
            continue
        qs = {"property_created": {"type": "noul",
                                   "instructions": (f"This is a Wikidata property proposal ('{p['name']}', proposed "
                                                    f"{(p['proposed'] or '?')[:10]}). Given the proposal and its "
                                                    f"discussion before the close, was the property created?"),
                                   "criteria": {"yes": "Created: the proposal was closed by creating the property.",
                                                "no": "Not done: the proposal was closed without creating the property."}}}
        labs = {"property_created": s12_noul(p["outcome"] == "created")}
        url = "https://www.wikidata.org/wiki/" + pg["title"].replace(" ", "_")
        meta = {"dataset": "wikidata", "group": pg["title"], "page": pg["title"], "page_revid": pg["revid"],
                "proposed": p["proposed"], "close_ts": p["close_ts"], "close_ts_source": approx or "closing comment",
                "closer": p["closer"], "created_property": p["pid"], "state_source": p["state_source"],
                "decision_date": p["close_ts"][:10],
                "info_date": max([p["proposed"] or p["close_ts"]] + [t for ln in p["pre"] for t in s2_ts(ln)])[:10],
                "resolution_date": {"property_created": p["close_ts"][:10]},
                "label_provenance": "the proposal's status field (property id = created; 'not done')",
                "raw_outcome": p["outcome"], "cut_before_close": p["cut_before_close"],
                "time_slice": "closed >= 2025-07-01" if sp == "eval" else "closed < 2025-01-01", "sample_weight": 1.0}
        r = s12_record("wikidata/property_proposal", state, qs, labs, meta, base_id=f"wikidata-pp/{s12_sha(pg['title'])}",
                       licence=S12_LIC_BYSA.format(page=pg["title"], url=url), rec_id=f"wd-pp-{s12_sha(pg['title'])}")
        r["split"] = sp
        by[sp].append(r)
    return s2_finish(a, "property_proposal", by, drops, {
        "pages": len(pages), "question": "property_created: yes (created) / no (not done)",
        "label_rule": "status field of {{Property proposal}}: a property id = created, 'not done' = no; 'withdrawn' "
                      "and open / ready / on-hold proposals are dropped",
        "close_time": "signature timestamp of the first closing line (the line naming the new property for created ones, "
                      "'not done' / 'no consensus' / 'closing' for the rest); created proposals without one use the "
                      "property's creation time (its first revision, Special:Export dir=asc)",
        "leakage_controls": ["the state is built from the page revision in force just BEFORE the close (Special:Export "
                             "history; the last revision whose minute precedes the closing signature), so later edits "
                             "to the page cannot leak; outcome and close time come from the current page",
                             "the status field (which holds the created property id) is removed",
                             "the discussion is cut at the first closing line, or earlier at the first line with a "
                             "done / not done / withdrawn template, 'created as', 'has been created', 'not done' or the "
                             "new property id",
                             "pre-close lines signed after the close time are dropped",
                             "a state the outcome detector still matches is dropped"]})


def s2_case_control(recs, cap, seed, tag):
    """When the records exceed the cap: keep every record whose outcome is not the majority class and a hash-ranked
    simple random sample of the majority class to fill the cap; meta.sample_weight = 1 / inclusion probability (1 for
    the minority classes), as KEV does, so weighted scores estimate the full population."""
    if len(recs) <= cap:
        return recs, {"case_control": False, "records": len(recs)}
    lab = lambda r: max(next(iter(r["labels"].values())).items(), key=lambda kv: kv[1])[0]   # noqa: E731
    counts = Counter(lab(r) for r in recs)
    maj = counts.most_common(1)[0][0]
    minor = [r for r in recs if lab(r) != maj]
    major = sorted((r for r in recs if lab(r) == maj), key=lambda r: s12_hash01(seed, tag, "cc", r["id"]))
    n_maj = max(0, cap - len(minor))
    keep_maj = major[:n_maj]
    p = n_maj / len(major) if major else 1.0
    for r in minor:
        r["meta"]["sample_weight"] = 1.0
        r["meta"]["sampling"] = {"stratum": "minority_outcome", "inclusion_prob": 1.0}
    for r in keep_maj:
        r["meta"]["sample_weight"] = round(1.0 / p, 6)
        r["meta"]["sampling"] = {"stratum": f"majority_outcome:{maj}", "inclusion_prob": round(p, 6)}
    return sorted(minor + keep_maj, key=lambda r: r["id"]), {
        "case_control": True, "population_counts": dict(counts), "majority": maj, "majority_inclusion_prob": round(p, 6),
        "records": len(minor) + len(keep_maj), "seed": seed}


def s2_finish(a, name, by, drops, extra):
    out = {}
    caps = {}
    for sp in ("train", "val", "eval"):
        recs = sorted(by.get(sp, []), key=lambda r: r["id"])
        if sp == "eval":    # case-control, weighted (KEV-style): every minority outcome kept
            recs, caps[sp] = s2_case_control(recs, S12_CAP_EVAL, a.seed, f"{name}/{sp}")
        else:               # natural rates for training: a seeded uniform sample down to the cap
            recs, caps[sp] = s12_cap(recs, S12_CAP_TRAIN, a.seed, f"{name}/{sp}")
        out[sp] = recs
    out["probe"] = [s12_probe_copy(r) for r in out["eval"]]
    wd = Path(a.work) / "s2" / name
    http_stats = json.loads((wd / "http_stats.json").read_text()) if (wd / "http_stats.json").exists() else {}
    manifest = {"source": f"wikidata/{name}", "shape": 2, "builder": S12_BUILDER,
                "built": datetime.now().isoformat(timespec="seconds"),
                "license": "CC-BY-SA-4.0 (discussion text) + CC0-1.0 (ids); per-record strings name the page",
                "license_evidence": S12_LICENSE_EVIDENCE,
                "time_windows": {"eval": f"closed on/after {S12_S2_EVAL_FROM}", "train_val": f"closed before {S12_S2_TRAIN_BEFORE}"},
                "drops": dict(sorted(drops.items())), "caps": caps, "http": http_stats, **extra,
                "caveats": ["MEMORISATION: archived discussions up to the training cutoffs are in pretraining corpora, "
                            "outcome included; only eval (closed 2025-07 .. 2026-09) is after the knowledge window; the "
                            "probe split withholds the state",
                            "share-alike: the text is CC BY-SA 4.0; a published derivative must carry the licence and "
                            "attribution"],
                "splits": {sp: s12_split_stats(v) for sp, v in out.items()},
                "gde1_notes": "eval = closed on/after 2025-07-01; probe = the same questions with the state withheld"}
    s12_write(a.out, f"s2_{name}", f"wd_{name}", out, manifest)
    return manifest


def s2_cmd(name):
    def run(a):
        wd = Path(a.work) / "s2" / name
        wd.mkdir(parents=True, exist_ok=True)
        if not a.no_fetch:
            http = S12Http(a.min_interval)
            (s2_rfd_fetch if name == "rfd" else s2_pp_fetch)(a, http, wd)
            st = json.loads((wd / "http_stats.json").read_text()) if (wd / "http_stats.json").exists() else {}
            for k, v in http.stats().items():
                st[k] = st.get(k, 0) + v if k != "min_interval_s" else v
            (wd / "http_stats.json").write_text(json.dumps(st, indent=1))
        m = (s2_rfd_build if name == "rfd" else s2_pp_build)(a)
        print(json.dumps({"drops": m["drops"], "splits": m["splits"]}, indent=1)[:4000])
        return 0
    return run


# =========================================================================== shapes 1-2: check, leak probe, dedup
S12_FOLDERS = ["s1_office_end", "s1_athlete_team", "s1_org_dissolved", "s1_iucn_status", "s2_rfd", "s2_property_proposal"]
S12_MAIN_QID = {"s1_office_end": "tenure_ends", "s1_athlete_team": "tenure_ends", "s1_org_dissolved": "dissolved",
                "s1_iucn_status": "status_changes", "s2_rfd": "outcome", "s2_property_proposal": "property_created"}


def s12_leak_probe(recs, qid, folds=5, dim=1 << 15, seed=0, weighted=False):
    """Cross-validated hashed bag-of-words (+ bigrams) multinomial logistic regression, state -> label, against the
    label prior. Far above the prior with outcome words among the top features would mean a leak; evidence words
    (notability, sources, office type) are what the model is supposed to use. Needs numpy + scipy."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.sparse import csr_matrix
    recs = [r for r in recs if qid in r["labels"]]
    if len(recs) < 50:
        return {"n": len(recs), "skipped": "fewer than 50 records"}
    keys = list(recs[0]["labels"][qid])
    vocab, rows, ys, ws = {}, [], [], []
    for r in recs:
        toks = re.findall(r"[a-z][a-z0-9]+", r["state"].lower())
        feats = set()
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % dim
            feats.add(h)
            vocab.setdefault(h, t.replace("_", " "))
        rows.append(sorted(feats))
        lab = r["labels"][qid]
        ys.append(keys.index(max(lab, key=lab.get)))
        ws.append(float(r["meta"].get("sample_weight") or 1.0) if weighted else 1.0)
    n, K = len(ys), len(keys)
    X = csr_matrix((np.ones(sum(len(f) for f in rows)), np.concatenate([np.array(f, dtype=np.int64) for f in rows]),
                    np.cumsum([0] + [len(f) for f in rows])), shape=(n, dim))
    y, w = np.array(ys), np.array(ws)

    def fit(Xt, yt, wt, lam=1.0):
        Y = np.eye(K)[yt]

        def f(v):
            W, b = v[:-K].reshape(dim, K), v[-K:]
            Z = Xt @ W + b
            Z -= Z.max(1, keepdims=True)
            lse = np.log(np.exp(Z).sum(1))
            P = np.exp(Z - lse[:, None])
            obj = -(wt * ((Z * Y).sum(1) - lse)).sum() + 0.5 * lam * (W ** 2).sum()
            G = Xt.T @ ((P - Y) * wt[:, None])
            return obj / wt.sum(), np.concatenate([(G + lam * W).ravel(), ((P - Y) * wt[:, None]).sum(0)]) / wt.sum()
        return minimize(f, np.zeros(dim * K + K), jac=True, method="L-BFGS-B", options={"maxiter": 150}).x
    fold = np.random.default_rng(seed).integers(0, folds, n)
    P = np.zeros((n, K))
    Pr = np.zeros((n, K))
    for k in range(folds):
        tr, te = fold != k, fold == k
        v = fit(X[tr], y[tr], w[tr])
        Z = X[te] @ v[:-K].reshape(dim, K) + v[-K:]
        Z -= Z.max(1, keepdims=True)
        P[te] = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        prior = (np.bincount(y[tr], weights=w[tr], minlength=K) + 0.5) / (w[tr].sum() + 0.5 * K)
        Pr[te] = prior
    eps = 1e-12
    ll = lambda Q: float(-(w * np.log(Q[np.arange(n), y] + eps)).sum() / w.sum())   # noqa: E731
    acc = lambda Q: float((w * (Q.argmax(1) == y)).sum() / w.sum())                 # noqa: E731
    v = fit(X, y, w)
    W = v[:-K].reshape(dim, K)
    top = {keys[j]: [vocab.get(int(ix), str(ix)) for ix in np.argsort(-(W[:, j] - W.mean(1)))[:15]] for j in range(K)}
    return {"n": n, "classes": keys, "label_counts": {keys[j]: int((y == j).sum()) for j in range(K)},
            "bow_cv_logloss": round(ll(P), 4), "prior_cv_logloss": round(ll(Pr), 4),
            "bow_cv_accuracy": round(acc(P), 4), "prior_accuracy": round(acc(Pr), 4),
            "logloss_skill_vs_prior": round(1 - ll(P) / ll(Pr), 4) if ll(Pr) > 0 else None, "top_features": top}


def s12_load_cf():
    import build_gde1_b
    return build_gde1_b.load_cf()


def s12_check_folder(out, folder, cf, probe=True):
    from build_jev_data import option_keys, validate
    root = Path(out) / folder
    rep = {"folder": folder, "built": (json.loads((root / "manifest.json").read_text()) or {}).get("built")
           if (root / "manifest.json").exists() else None}
    recs = defaultdict(list)
    bad = Counter()
    for spd in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(spd.glob("*.jsonl")):
            for ln in f.open(encoding="utf-8"):
                r = json.loads(ln)
                validate(r)
                m = r["meta"]
                if not r.get("license"):
                    bad["missing_license"] += 1
                if not (m.get("decision_date") and m.get("info_date")):
                    bad["missing_dates"] += 1
                if str(m.get("info_date")) > str(m.get("decision_date")):
                    bad["info_after_decision"] += 1
                if set((m.get("resolution_date") or {})) != set(r["questions"]):
                    bad["resolution_date_keys"] += 1
                elif any(str(v) < str(m["decision_date"]) for v in m["resolution_date"].values()):
                    bad["resolution_before_decision"] += 1
                for qid, q in r["questions"].items():
                    keys, texts, tgt = cf.question_options(q, r["labels"][qid])
                    assert keys == option_keys(q) and abs(sum(tgt) - 1) < 1e-6, (r["id"], qid)
                recs[spd.name].append(r)
    units = Counter()
    for u in cf.iter_units(str(root), sorted(recs), set(), None):
        units[f"{u['split']}/{u['qtype']}/{u['label_kind']}"] += 1
    main = [sp for sp in ("train", "val", "eval") if sp in recs]
    ids = [r["id"] for sp in recs for r in recs[sp]]
    base = {sp: {r["base_id"] for r in recs[sp]} for sp in main}
    items = {sp: {q for r in recs[sp] for q in (r["meta"].get("items") or [r["meta"].get("wikidata_item")]) if q}
             for sp in main}
    states = {sp: {r["state"] for r in recs[sp]} for sp in main}
    tv = lambda d: (d.get("train", set()) | d.get("val", set()))   # noqa: E731
    lens = sorted(len(r["state"]) for sp in main for r in recs[sp])
    rep.update({"records": {sp: len(v) for sp, v in recs.items()}, "iter_units": dict(units),
                "schema_problems": dict(bad) or "none", "duplicate_ids": len(ids) - len(set(ids)),
                "base_id_overlap_trainval_eval": len(tv(base) & base.get("eval", set())),
                "item_overlap_trainval_eval": len(tv(items) & items.get("eval", set())),
                "identical_states_trainval_eval": len(tv(states) & states.get("eval", set())),
                "state_chars": {"median": lens[len(lens) // 2], "p95": lens[int(len(lens) * 0.95)], "max": lens[-1],
                                "over_12000": sum(x > 12000 for x in lens)} if lens else None})
    if "probe" in recs:
        rep["probe"] = {"records": len(recs["probe"]), "all_states_withheld": all(r["state"] == S12_PROBE_STATE
                                                                                  for r in recs["probe"]),
                        "same_questions_as_eval": {r["meta"]["probe_of"] for r in recs["probe"]} ==
                        {r["id"] for r in recs.get("eval", [])}}
    det = {"s2_rfd": S2_RFD_LEAK,
           "s2_property_proposal": re.compile(r"\{\{\s*(?:done|not ?done|withdrawn)\s*\}\}|\bcreated as\b|"
                                              r"has been created|\bnot done\b", re.I)}.get(folder)
    if det:
        hits = Counter()
        for sp in main:
            for r in recs[sp]:
                for h in {m.group(0).lower() for m in det.finditer(r["state"])}:
                    hits[f"{sp}:{h}"] += 1
        rep["outcome_detector_hits_in_states"] = dict(hits) or "0 hits"
        if folder == "s2_rfd":
            rep["rfd_leak_function_hits"] = {sp: sum(s2_rfd_leak(r["state"]) for r in recs[sp]) for sp in main}
    if folder.startswith("s1_"):
        late = Counter()
        for sp in main:
            for r in recs[sp]:
                t1 = r["meta"]["t1"]
                if any(d > t1 for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", r["state"])):
                    late[sp] += 1
                assert r["meta"]["ts_t1"] <= t1 + "T23:59:59Z", r["id"]
                assert r["meta"]["ts_t2"] <= r["meta"]["t2"] + "T23:59:59Z", r["id"]
        rep["states_naming_a_full_date_after_T1"] = dict(late) or "0"
        rep["revision_time_asserts"] = "ok: every ts_t1 <= T1 and ts_t2 <= T2"
    if probe:
        try:
            qid = S12_MAIN_QID[folder]
            rep["leak_probe"] = {}
            for sp in ("train", "eval"):
                if recs.get(sp):
                    rep["leak_probe"][sp] = s12_leak_probe(recs[sp], qid, weighted=folder == "s1_org_dissolved")
        except ImportError as e:
            rep["leak_probe"] = f"skipped ({e}); run with .venv/bin/python"
    samples = []
    for sp in ("train", "eval", "probe"):
        if recs.get(sp):
            r = recs[sp][len(recs[sp]) // 3]
            samples.append({**r, "state": r["state"][:2500] + (" ..." if len(r["state"]) > 2500 else "")})
    gset = "wd_" + folder.split("_", 1)[1]
    g = Path(out) / "gde1" / gset
    if g.exists():
        gu = Counter(f"{u['split']}/{u['qtype']}" for u in cf.iter_units(str(g), ["eval", "probe"], set(), None))
        rep["gde1_copy_iter_units"] = dict(gu)
    if folder == "s1_office_end":
        rep["base_id_note"] = ("office_end clusters by (country, position kind), so cluster ids recur across splits by "
                               "design; items never do (item_overlap_trainval_eval)")
    (root / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    (root / "check_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def s12_check(a):
    cf = s12_load_cf()
    folders = [f for f in (a.sources.split(",") if a.sources else S12_FOLDERS) if (Path(a.out) / f).exists()]
    allrep = {f: s12_check_folder(a.out, f, cf, probe=not a.no_leak_probe) for f in folders}
    print(json.dumps(allrep, indent=1, default=str)[:20000])
    return 0


S12_TEMPLATE_LINE = re.compile(r"^(WIKIDATA ITEM .* AS IT STOOD ON .*|Statements:|Wikimedia sitelinks: .*|External identifiers: .*|"
                               r"WIKIDATA REQUEST FOR DELETION .*|Nomination:|Discussion before the close:.*|"
                               r"WIKIDATA PROPERTY PROPOSAL .*|Motivation and discussion before the close:|Proposed: .*|"
                               r"Item\(s\) nominated: .*)$")


def s12_content_view(r):
    return "\n".join(ln for ln in r["state"].split("\n") if ln.strip() and not S12_TEMPLATE_LINE.match(ln.strip()))


def s12_dedup(a):
    """Drop records whose content overlaps a training corpus (build_jev_ext2.Index through build_gde1_b.build_index:
    exact state, >= 5-word leaf lines, >= 3 shared 8-word shingles), in place, for the given splits; probe copies of
    dropped eval records go too. The authoritative run needs jev_data / jev_ext_strict / jev_ext2 / jev_ext3 (nodes)."""
    import build_gde1_b as gb
    idx, sizes = gb.build_index(a.fingerprint)
    res = {}
    for folder in [f for f in (a.sources.split(",") if a.sources else S12_FOLDERS) if (Path(a.out) / f).exists()]:
        root = Path(a.out) / folder
        dropped_ids = set()
        info = {}
        for sp in a.splits.split(","):
            for f in sorted((root / sp).glob("*.jsonl")):
                recs = [json.loads(ln) for ln in f.open(encoding="utf-8")]
                kept, d = [], Counter()
                for r in recs:
                    c = s12_content_view(r)
                    probes = [ln for ln in c.split("\n") if len(ln.split()) >= 5]
                    hit = idx.check(c, probes)
                    if hit:
                        d[f"{hit[0]}:{hit[1]}"] += 1
                        dropped_ids.add(r["id"])
                    else:
                        kept.append(r)
                info[f"{sp}/{f.name}"] = {"before": len(recs), "after": len(kept), "dropped": dict(d)}
                if not a.dry_run:
                    with f.open("w", encoding="utf-8") as fh:
                        for r in kept:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not a.dry_run and dropped_ids:
            for pf in (root / "probe").glob("*.jsonl"):
                recs = [json.loads(ln) for ln in pf.open(encoding="utf-8")]
                with pf.open("w", encoding="utf-8") as fh:
                    for r in recs:
                        if r["meta"].get("probe_of") not in dropped_ids:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            gset = "wd_" + folder.split("_", 1)[1]
            for gf in (Path(a.out) / "gde1" / gset).glob("*/*.jsonl"):
                recs = [json.loads(ln) for ln in gf.open(encoding="utf-8")]
                with gf.open("w", encoding="utf-8") as fh:
                    for r in recs:
                        src = r["meta"].get("probe_of") or r["id"][len("gde1-"):]
                        if src not in dropped_ids:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        mp = root / "manifest.json"
        if mp.exists() and not a.dry_run:
            m = json.loads(mp.read_text())
            m.setdefault("dedup_vs_training_runs", []).append({"kind": a.kind, "fingerprints": sizes, "by_file": info,
                                                               "when": datetime.now().isoformat(timespec="seconds")})
            mp.write_text(json.dumps(m, indent=1, ensure_ascii=False, default=str))
        res[folder] = info
    print(json.dumps({"fingerprints": sizes, "results": res}, indent=1))
    return 0


# =========================================================================== shapes 1-2: offline self-test
def s12_self_test():
    import tempfile
    # time precision and windows
    assert s12_interval(["time", "+2023-05-00T00:00:00Z", 10]) == (date(2023, 5, 1), date(2023, 5, 31))
    assert s12_interval(["time", "+2024-00-00T00:00:00Z", 9]) == (date(2024, 1, 1), date(2024, 12, 31))
    assert s12_window(s12_interval(["time", "+2025-08-01T00:00:00Z", 11]), "2025-06-30", "2026-09-30") == "in"
    assert s12_window(s12_interval(["time", "+2025-00-00T00:00:00Z", 9]), "2025-06-30", "2026-09-30") == "straddle"
    assert s12_window(s12_interval(["time", "+2025-06-30T00:00:00Z", 11]), "2025-06-30", "2026-09-30") == "before"
    assert s12_window(s12_interval(["time", "+2027-01-00T00:00:00Z", 10]), "2025-06-30", "2026-09-30") == "after"
    # WDQS TSV parsing and truncation detection
    rows = s12_parse_tsv(b'?item\t?st\t?start\n<http://www.wikidata.org/entity/Q5>\t<http://www.wikidata.org/entity/'
                         b'statement/Q5-ab>\t"2011-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>\n'
                         b'<http://www.wikidata.org/entity/Q6>\t\t"say \\"hi\\""@en\n')
    assert rows[0] == {"item": "Q5", "st": "Q5-ab", "start": "2011-01-01T00:00:00Z"} and rows[1]["start"] == 'say "hi"'
    try:
        s12_parse_tsv(b"?a\n<x>\n\tat java.lang.Thread.run(Thread.java:750)\n")
        raise AssertionError("truncation not detected")
    except S12Truncated:
        pass

    # entities: a person at T1 / T2 in Wikibase revision JSON
    def snak(pid, kind, val):
        dv = {"item": {"type": "wikibase-entityid", "value": {"entity-type": "item", "id": val}},
              "time": {"type": "time", "value": {"time": val[0], "precision": val[1]}}}[kind]
        return {"snaktype": "value", "property": pid, "datavalue": dv}

    def stmt(sid, pid, kind, val, quals=None, rank="normal"):
        q = {}
        for qp, k, v in quals or []:
            q.setdefault(qp, []).append(snak(qp, k, v))
        return {"id": sid, "rank": rank, "mainsnak": snak(pid, kind, val), "qualifiers": q}

    def person(q, p39, extra=None):
        return {"type": "item", "id": q, "labels": {"en": {"language": "en", "value": "Ada Example"}},
                "descriptions": {"en": {"value": "politician"}},
                "claims": {"P31": [stmt(q + "$a", "P31", "item", "Q5")], "P39": p39, **(extra or {})},
                "sitelinks": {"enwiki": {"title": "Ada Example"}, "dewiki": {"title": "Ada Example"}}}
    st = ("+2021-05-01T00:00:00Z", 11)
    t1e = person("Q100", [stmt("Q100$m", "P39", "item", "Q1939555", [("P580", "time", st), ("P768", "item", "Q64")]),
                          stmt("Q100$old", "P39", "item", "Q7", [("P580", "time", ("+2001-00-00T00:00:00Z", 9))])])
    c1 = s12_compact(t1e, {"P214": "ExternalId"})
    spec = S1_SPECS["office_end"]
    s, why, info = s1_tenure_t1(spec, c1, "Q100", "2025-06-30", 1)
    assert s == "ok" and [x["stmt"] for x in info["cands"]] == ["Q100$m"], (s, why, info)
    s, why, info = s1_tenure_refine(spec, info, {})
    L = {"Q1939555": "member of the Bundestag", "Q64": "Berlin", "P39": "position held", "P580": "start time",
         "P768": "electoral district", "P31": "instance of", "Q5": "human"}
    state = s12_render(c1, "2025-06-30", L, "person")
    assert "member of the Bundestag (start time: 2021-05-01; electoral district: Berlin)" in state, state
    assert "AS IT STOOD ON 2025-06-30" in state and "English Wikipedia article: Ada Example" in state
    fs = s12_render(c1, "2025-06-30", L, "person", focus=("P39", "Q100$old"))
    assert fs.index("Q7") < fs.index("member of the Bundestag"), "the focus statement comes first"
    qs = s1_tenure_question(spec, c1, info, L, "2025-06-30", "2026-09-30")
    assert s1_position_kind("Mayor of Lyon") == "mayor" and s1_position_kind("conseiller municipal d'Oudrenne") == \
        "conseiller municipal" and s1_position_kind("member of the Bundestag") == "member"
    assert "member of the Bundestag" in qs["tenure_ends"]["instructions"] and "Ada Example (Q100)" in qs["tenure_ends"]["instructions"]

    def t2_with(end):
        quals = [("P580", "time", st)] + ([("P582", "time", end)] if end else [])
        return s12_compact(person("Q100", [stmt("Q100$m", "P39", "item", "Q1939555", quals)]))
    assert s1_tenure_t2(spec, info, t2_with(("+2025-10-14T00:00:00Z", 11)), "2025-06-30", "2026-09-30")[2]["yes"] is True
    assert s1_tenure_t2(spec, info, t2_with(None), "2025-06-30", "2026-09-30")[2]["yes"] is False
    assert s1_tenure_t2(spec, info, t2_with(("+2024-03-01T00:00:00Z", 11)), "2025-06-30", "2026-09-30")[1] == \
        "premise_false_end_dated_on_or_before_T1"
    assert s1_tenure_t2(spec, info, t2_with(("+2025-00-00T00:00:00Z", 9)), "2025-06-30", "2026-09-30")[1] == "end_time_straddle"
    assert s1_tenure_t2(spec, info, s12_compact({"entity": "Q100", "redirect": "Q9"}), "2025-06-30", "2026-09-30")[1] == \
        "merged_into_another_item_by_T2"
    closed = s12_compact(person("Q100", [stmt("Q100$m", "P39", "item", "Q1939555", [("P580", "time", st),
                                                                                    ("P582", "time", ("+2024-01-01T00:00:00Z", 11))])]))
    assert s1_tenure_t1(spec, closed, "Q100", "2025-06-30", 1)[0] == "drop", "an end time at T1 is not eligible"
    dead = s12_compact(person("Q100", t1e["claims"]["P39"], {"P570": [stmt("x", "P570", "time", ("+2024-01-01T00:00:00Z", 11))]}))
    assert s1_tenure_t1(spec, dead, "Q100", "2025-06-30", 1)[1] == "date_of_death_recorded_at_T1"
    aspec = S1_SPECS["athlete_team"]
    ath = s12_compact(person("Q200", [], {"P54": [stmt("Q200$n", "P54", "item", "Q11", [("P580", "time", ("+2024-07-01T00:00:00Z", 11))]),
                                                  stmt("Q200$c", "P54", "item", "Q12", [("P580", "time", ("+2023-07-01T00:00:00Z", 11))])]}))
    s, _, ai = s1_tenure_t1(aspec, ath, "Q200", "2025-06-30", 1)
    assert s == "ok" and ai["cands"][0]["stmt"] == "Q200$n", "latest start first"
    s, _, ai = s1_tenure_refine(aspec, ai, {"Q11": "Brazil national under-20 football team", "Q12": "FC Example"})
    assert ai["stmt"] == "Q200$c", "national teams are skipped"
    assert not S12_NATIONAL.search("Team Sky") and not S12_NATIONAL.search("Olympique Lyonnais")
    assert S12_NATIONAL.search("Germany national under-21 football team") and S12_NATIONAL.search("Selección de fútbol")
    # organisations
    ospec = S1_SPECS["org_dissolved"]
    org = lambda claims: s12_compact({"id": "Q300", "labels": {"en": {"value": "Band X"}}, "claims": claims})  # noqa: E731
    o1 = org({"P31": [stmt("a", "P31", "item", "Q215380")]})
    assert s1_org_t1(ospec, o1, "Q300", "2025-06-30", 1, {"cls": "Q215380"})[0] == "ok"
    assert s1_org_t1(ospec, o1, "Q300", "2025-06-30", 1, {"cls": "Q7278"})[1] == "class_not_in_P31_at_T1"
    o2 = org({"P31": [stmt("a", "P31", "item", "Q215380")], "P576": [stmt("b", "P576", "time", ("+2026-02-00T00:00:00Z", 10))]})
    assert s1_org_t2(ospec, {}, o2, "2025-06-30", "2026-09-30")[2]["yes"] is True
    assert s1_org_t2(ospec, {}, o2, "2022-06-30", "2023-09-30")[2]["yes"] is False
    assert s1_org_t2(ospec, {}, o2, "2026-06-30", "2027-09-30")[1] == "premise_false_dissolution_dated_on_or_before_T1"
    # IUCN
    ispec = S1_SPECS["iucn_status"]
    tax = lambda v: s12_compact({"id": "Q400", "labels": {"en": {"value": "Panthera x"}},  # noqa: E731
                                 "claims": {"P141": [stmt("s", "P141", "item", v)]}})
    s, _, ii = s1_iucn_t1(ispec, tax("Q278113"), "Q400", "2025-06-30", 1)
    _, _, ir = s1_iucn_t2(ispec, ii, tax("Q96377276"), "2025-06-30", "2026-09-30")
    labs = s1_labels_of(ispec, ii, ir)
    assert labs["status_at_t2"]["EN"] == 1.0 and labs["status_changes"]["yes"] == 1.0
    assert set(s1_iucn_question(ispec, tax("Q278113"), ii, {}, "2025-06-30", "2026-09-30")["status_at_t2"]["criteria"]) == \
        {v[0] for v in S12_IUCN.values()}
    # Special:Export parsing
    xml = (b'<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"><page><title>Q100</title><ns>0</ns><id>1</id>'
           b'<revision><id>55</id><timestamp>2025-06-20T10:00:00Z</timestamp><model>wikibase-item</model>'
           b'<text bytes="2">{"id":"Q100"}</text></revision></page></mediawiki>')
    ex = s12_parse_export(xml)
    assert ex["Q100"]["revid"] == 55 and json.loads(ex["Q100"]["text"])["id"] == "Q100"

    # RfD sections
    sig = "[[User:Nom|Nom]] ([[User talk:Nom|talk]]) 21:43, 10 August 2025 (UTC)"
    adm = "--[[User:DeltaBot|DeltaBot]] ([[User talk:DeltaBot|talk]]) 03:11, 12 August 2025 (UTC)"
    page = ("{{Archive}}\n\n== [[Q1]] ==\n\n{{rfd links|1=Q1|2=Not notable.}}\n\nNot notable. " + sig + "\n"
            ": {{on hold}} This item is linked from 1 other. " + adm.replace("03:11, 12", "21:50, 10") + "\n"
            ":{{Keep}} it has sources. [[User:B|B]] 22:00, 10 August 2025 (UTC)\n"
            ": {{deleted|admin=X}} " + adm + "\n:: Thanks! [[User:C|C]] 05:00, 12 August 2025 (UTC)\n\n"
            "== [[Q2]] ==\n{{rfd links|1=Q2|2=dup of Q3}}\nDuplicate of {{Q|3}} " + sig + "\n"
            ":Merge them. [[User:D|D]] 23:00, 10 August 2025 (UTC)\n"
            ":Late remark inserted above the close. [[User:E|E]] 10:00, 13 August 2025 (UTC)\n"
            ": {{done}} Redirect created by [[User:A]] " + adm + "\n\n"
            "== [[Q4]] ==\n{{rfd links|1=Q4|2=x}}\nx " + sig + "\n:I have merged it into Q5. [[User:F|F]] 23:00, 10 August 2025 (UTC)\n"
            ":{{not done}} " + adm + "\n\n== [[Q6]] ==\nno signature here\n")
    secs = s2_rfd_sections(page)
    assert [h for h, _ in secs] == ["[[Q1]]", "[[Q2]]", "[[Q4]]", "[[Q6]]"]
    p1, _ = s2_rfd_parse(*secs[0], "P")
    assert p1["outcome"] == "deleted" and p1["close_ts"] == "2025-08-12T03:11" and p1["filed"] == "2025-08-10T21:43"
    st1 = s2_rfd_state(p1)
    assert "Thanks" not in st1 and "deleted" not in st1.lower() and "Keep (vote) it has sources" in st1, st1
    assert "On hold: This item is linked from 1 other." in st1 and "Nom (2025-08-10 21:43 UTC)" in st1, st1
    p2, _ = s2_rfd_parse(*secs[1], "P")
    assert p2["outcome"] == "merged" and "Late remark" not in s2_rfd_state(p2) and "Duplicate of Q3" in s2_rfd_state(p2)
    p4, _ = s2_rfd_parse(*secs[2], "P")
    assert p4["outcome"] == "kept" and p4["cut_at_leak_line"] == 1 and "merged" not in s2_rfd_state(p4).lower()
    assert s2_rfd_parse(*secs[3], "P") == (None, "no_signature")
    assert s2_rfd_leak("Items merged, same with the Cebuano articles.") and s2_rfd_leak("I have thus merged it with Q5.")
    assert s2_rfd_leak("Link to redirect restored, and items merged.") and s2_rfd_leak("It has been merged.")
    assert not s2_rfd_leak("Date should be merged!") and not s2_rfd_leak("I suggest they be merged & this one kept"[:30])
    assert not s2_rfd_leak("Seems can merged.") and not s2_rfd_leak("Deleted on jawiki. Not notable.")
    # property proposals
    pp = ("== {{TranslateThis | anchor = en\n| en = Example film ID\n}} ==\n{{Property proposal\n|status = 14435\n"
          "|description = {{TranslateThis\n | en = identifier for a film in Example DB\n }}\n|datatype = external-id\n"
          "|example 1 = {{Statement|Q1|→|abc}}\n}}\n\n=== Motivation ===\nUseful. " + sig + "\n\n=== Discussion ===\n"
          "* {{Support}} [[User:Y|Y]] 01:12, 11 August 2025 (UTC)\n"
          "* {{re|Nom}} {{done}} as {{P|14435}}. " + adm + "\n* {{Oppose}} too late [[User:Z|Z]] 09:00, 13 August 2025 (UTC)\n")
    pr, _ = s2_pp_parse("Wikidata:Property proposal/Example film ID", pp)
    assert pr["outcome"] == "created" and pr["pid"] == "P14435" and pr["close_ts"] == "2025-08-12T03:11", pr
    ps = s2_pp_state(pr)
    assert "14435" not in ps and "status" not in ps and "too late" not in ps and "Support" in ps, ps
    assert "description: identifier for a film in Example DB" in ps and "Example film ID" in ps, ps
    nd = pp.replace("|status = 14435", "|status = not done <!-- leave empty -->").replace(
        "{{done}} as {{P|14435}}.", "{{not done}}, no consensus.")
    pn, _ = s2_pp_parse("Wikidata:Property proposal/X", nd)
    assert pn["outcome"] == "not_done" and pn["close_ts"] == "2025-08-12T03:11" and "not done" not in s2_pp_state(pn).lower()
    assert s2_pp_parse("x", pp.replace("|status = 14435", "|status = withdrawn"))[1] == "withdrawn"
    assert s2_pp_parse("x", pp.replace("|status = 14435", "|status = ready"))[1].startswith("open_or_other_status")
    assert s2_split_of("2025-07-01", "k", 1) == "eval" and s2_split_of("2025-03-01", "k", 1) is None
    # the state of a proposal comes from the revision before the close (history export; same-minute edits excluded)
    hist = (b'<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"><page><title>T</title><revision><id>2</id>'
            b'<timestamp>2025-08-12T03:11:30Z</timestamp><text>b</text></revision><revision><id>1</id>'
            b'<timestamp>2025-08-10T21:43:00Z</timestamp><text>a</text></revision></page></mediawiki>')
    hh = s2_parse_history(hist)
    assert [r[0] for r in hh["T"]] == [1, 2] and [r for r in hh["T"] if r[1][:16] < "2025-08-12T03:11"][-1][0] == 1
    pre_close = pp.split("* {{re|Nom}}")[0].replace("|status = 14435", "|status = ready")
    b = s2_pp_body("Wikidata:Property proposal/Example film ID", pre_close, "2025-08-12T03:11", "P14435")
    st_b = s2_pp_state({**pr, **{k: b[k] for k in ("name", "params", "pre", "proposed")}})
    assert "14435" not in st_b and "ready" not in st_b and "Support" in st_b and "Useful." in st_b, st_b
    assert s2_pp_close({"close_ts": None, "outcome": "created", "pid": "P9"}, {"Property:P9": "2024-01-02T03:04:05Z"}) == \
        ("2024-01-02T03:04", "property_creation_time")

    # records -> write -> iter_units read-back, GDE-1 copies, probe
    cf = s12_load_cf()
    with tempfile.TemporaryDirectory() as td:
        recs = {"train": [], "val": [], "eval": []}
        for i, (sp, t1, t2, yes) in enumerate([("train", "2022-06-30", "2023-09-30", True), ("val", "2021-06-30", "2022-09-30", False),
                                              ("eval", "2025-06-30", "2026-09-30", False)]):
            q = s1_tenure_question(spec, c1, info, L, t1, t2)
            m = {"dataset": "wikidata", "group": f"Q{i}", "wikidata_item": f"Q{i}", "t1": t1, "t2": t2, "ts_t1": t1 + "T00:00:00Z",
                 "ts_t2": t2 + "T00:00:00Z", "decision_date": t1, "info_date": t1, "resolution_date": {"tenure_ends": t2},
                 "sample_weight": 1.0}
            r = s12_record("wikidata/office_end", state + f"\n#{i}", q, {"tenure_ends": s12_noul(yes)}, m, f"wikidata/Q{i}",
                           S12_LIC_CC0, f"wd-office_end-Q{i}-{t1}")
            recs[sp].append(r)
        recs["probe"] = [s12_probe_copy(r) for r in recs["eval"]]
        s12_write(td, "s1_office_end", "wd_office_end", recs, {"built": "t", "license": S12_LIC_CC0})
        rep = s12_check_folder(td, "s1_office_end", cf, probe=False)
        assert rep["schema_problems"] == "none" and rep["duplicate_ids"] == 0 and rep["item_overlap_trainval_eval"] == 0, rep
        assert rep["probe"]["all_states_withheld"] and rep["probe"]["same_questions_as_eval"], rep["probe"]
        assert rep["iter_units"] == {"eval/noul/hard": 1, "probe/noul/hard": 1, "train/noul/hard": 1, "val/noul/hard": 1}, rep
        assert rep["gde1_copy_iter_units"] == {"eval/noul": 1, "probe/noul": 1}, rep
        units = list(cf.iter_units(str(Path(td) / "gde1" / "wd_office_end"), ["eval"], set(), None))
        assert units[0]["license"].startswith("CC0-1.0") and units[0]["noul_criteria"]["yes"].startswith("Wikidata's revision")
        g = json.loads((Path(td) / "gde1" / "wd_office_end" / "eval" / "wikidata__office_end.jsonl").read_text())
        assert g["meta"]["domain"] == "O" and g["meta"]["gde1_set"] == "wd_office_end" and g["meta"]["eval_only"]
    print("shapes 1-2 self-test OK")


# =========================================================================== shapes 1-2: subcommands
S1_DEFAULTS = {"office_end": {"eval_items": 3500, "train_items": 6500, "wdqs_sample": 400000, "wdqs_queries": 2, "search_n": 0},
               "athlete_team": {"eval_items": 3500, "train_items": 6500, "wdqs_sample": 150000, "wdqs_queries": 3, "search_n": 0},
               "org_dissolved": {"eval_items": 3500, "train_items": 6000, "wdqs_sample": 0, "wdqs_queries": 0, "search_n": 6000},
               "iucn_status": {"eval_items": 3000, "train_items": 5000, "wdqs_sample": 0, "wdqs_queries": 0, "search_n": 8000}}
S1_HELP = {"office_end": "SHAPE 1: will a person's position held (P39) get an end time in (T1, T2]?",
           "athlete_team": "SHAPE 1: will an athlete's sports-team membership (P54) end in (T1, T2]?",
           "org_dissolved": "SHAPE 1: will an organisation get a dissolution date (P576) in (T1, T2]? (case-control)",
           "iucn_status": "SHAPE 1: a taxon's IUCN status (P141) on T2, and whether it changes"}


def s12_register(sub):
    def common(p):
        p.add_argument("--out", default=OUT_ROOT)
        p.add_argument("--work", default=S12_WORK, help="scratch: compact revisions, labels, page texts (no raw dumps)")
        p.add_argument("--seed", type=int, default=S12_SEED)
        p.add_argument("--min-interval", type=float, default=7.5,
                       help="seconds between MediaWiki requests (shared lock across processes; ~10/min is the limit "
                            "for clients without contact details in the User-Agent)")
        p.add_argument("--no-fetch", action="store_true", help="rebuild from the work cache only")
    for name, d in S1_DEFAULTS.items():
        p = sub.add_parser("s1-" + name.replace("_", "-"), help=S1_HELP[name])
        common(p)
        p.add_argument("--eval-share", type=float, default=0.4, help="share of items in the eval pool (by hash)")
        p.add_argument("--eval-items", type=int, default=d["eval_items"], help="eval-pool items exported at T1")
        p.add_argument("--train-items", type=int, default=d["train_items"], help="train/val-pool items exported at T1")
        p.add_argument("--wdqs-sample", type=int, default=d["wdqs_sample"], help="bd:sample size per WDQS query")
        p.add_argument("--wdqs-queries", type=int, default=d["wdqs_queries"])
        p.add_argument("--search-n", type=int, default=d["search_n"], help="CirrusSearch random sample size")
        p.add_argument("--cluster-cap-eval", type=int, default=75, help="office_end: max records per (country, "
                       "position kind) cluster in eval")
        p.add_argument("--cluster-cap-train", type=int, default=150, help="office_end: the same for train and val")
        p.add_argument("--case-control", action=argparse.BooleanOptionalAction, default=(name == "athlete_team"),
                       help="tenure sources: add every candidate whose current statement ends in its window (WDQS "
                            "screen) and weight the rest (default on for athlete_team, whose natural rate is ~4%%)")
        p.set_defaults(func=s1_cmd(name))
    p = sub.add_parser("s2-rfd", help="SHAPE 2: Wikidata:Requests for deletions -> deleted / kept / merged")
    common(p)
    p.add_argument("--train-days", type=int, default=700, help="archive days sampled before 2025-01 (by hash)")
    p.set_defaults(func=s2_cmd("rfd"))
    p = sub.add_parser("s2-property-proposal", help="SHAPE 2: Wikidata:Property proposal -> created / not done")
    common(p)
    p.add_argument("--pp-train", type=int, default=6000,
                   help="train / val proposals whose pre-close revision is fetched (hash sample; every eval one is)")
    p.set_defaults(func=s2_cmd("property_proposal"))
    p = sub.add_parser("s12-check", help="shapes 1-2: read-back, leakage scans, cross-split checks, leak probe")
    p.add_argument("--out", default=OUT_ROOT)
    p.add_argument("--sources", default="", help="folders, comma-separated (default: every shapes 1-2 folder)")
    p.add_argument("--no-leak-probe", action="store_true")
    p.set_defaults(func=s12_check)
    p = sub.add_parser("s12-dedup", help="shapes 1-2: drop records overlapping training corpora (ext2 Index)")
    p.add_argument("--out", default=OUT_ROOT)
    p.add_argument("--sources", default="")
    p.add_argument("--splits", default="eval")
    p.add_argument("--fingerprint", action="append", default=[], required=True, metavar="TAG=PATH")
    p.add_argument("--kind", choices=["built", "upstream-proxy"], default="built")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=s12_dedup)


REGISTRARS.append(s12_register)
SELF_TESTS.append(s12_self_test)


# =========================================================================== SHAPES 3-4
# Shape 3  wd_compare   exact comparisons computed from REFERENCED Wikidata claims (knowledge side panel "K" for
#                       GDE-1; an open-book variant with the facts in the state is optional training data).
# Shape 4  wd_classify  classification of a masked English Wikipedia lead into fixed Wikidata taxonomies
#                       (instance-of type, country, occupation group, industry sector, event era).
# Subcommands: wd-compare, wd-classify (fetch into a compact cache, then build), wd34-gate (dedup), wd34-check.
# Docs: research/wikidata_pipeline.md, sections "Shape 3" and "Shape 4".
#
# Network etiquette: query.wikidata.org (SPARQL) and the en.wikipedia / www.wikidata.org action APIs, serial requests
# with a minimum spacing per host, retries with backoff on 429/5xx (Retry-After honoured), a descriptive User-Agent
# that names the project and carries NO personal contact. A 403 stops the run (S34Refused) instead of retrying.
# Caches are compact jsonl.gz of the extracted fields only (no raw responses) under S34_WORK.
import gzip as _s34_gzip  # noqa: E402
import hashlib as _s34_hashlib  # noqa: E402
import json as _s34_json  # noqa: E402
import math as _s34_math  # noqa: E402
import random as _s34_random  # noqa: E402
import re as _s34_re  # noqa: E402
import time as _s34_time  # noqa: E402
import unicodedata as _s34_ud  # noqa: E402
import urllib.error as _s34_uerr  # noqa: E402
import urllib.parse as _s34_uparse  # noqa: E402
import urllib.request as _s34_ureq  # noqa: E402
from collections import Counter as _S34Counter, defaultdict as _s34_dd  # noqa: E402
from datetime import datetime as _s34_dt  # noqa: E402

S34_WORK = "/tmp/wikidata_b_work/s34"
S34_UA = "jev-wikidata-build/0.1 (jev decision-model research dataset build; low-volume polite queries)"
S34_SPARQL = "https://query.wikidata.org/sparql"
S34_WP_API = "https://en.wikipedia.org/w/api.php"
S34_WD_API = "https://www.wikidata.org/w/api.php"
# en.wikipedia.org / www.wikidata.org sit behind the Wikimedia API gateway, which since 2026 allows 10 requests per
# minute to a client whose User-Agent has no contact information (mediawiki.org/wiki/Wikimedia_APIs/Rate_limits, read
# 2026-10-01; 200/min needs an email or URL in the UA, which this build may not send). One request per 10 s keeps under
# it, and the bulk text comes from a single pinned Hugging Face shard instead. WDQS is not behind that gateway.
S34_SPACING = {"query.wikidata.org": 1.0, "en.wikipedia.org": 10.0, "www.wikidata.org": 10.0, "huggingface.co": 1.0}
S34_HF_REPO = "wikimedia/wikipedia"
S34_HF_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"      # lastModified 2024-01-09; config 20231101.en
S34_HF_FILE = "20231101.en/train-00028-of-00041.parquet"       # 156,288 articles, page ids 55,614,901-57,373,918
S34_HF_SHA256 = "10589a39188af404fa458da252df7dcf22c6f6395a1501d9243ef56ce4c3c148"
S34_SEED = 3434
S34_POPULAR, S34_MID = 30, 8        # sitelink strata: popular >= 30, mid 8-29, obscure <= 7
# First English Wikipedia article created on 2025-07-01 ("St. Ives, NSW", 2025-07-01T00:00:15Z, logid 170708192, read
# from list=logevents on 2026-10-01). Page ids are assigned at creation, so pageid >= this = article created on or after
# 2025-07-01 (conservative: a page created as a redirect earlier and expanded later keeps its old id).
S34_FRESH_PAGEID = 80326871
S34_FRESH_FROM = "2025-07-01"
S34_STATE_REMOVED = "(state removed: knowledge probe)"
S34_REF_PROPS = ("P854", "P248")    # a "real" reference: reference URL or stated in (not P143 imported-from-Wikimedia)

S34_LIC_WD_QUOTE = ("All structured data from the main, Property, Lexeme, and EntitySchema namespaces is available under "
                    "the Creative Commons CC0 License")
S34_LIC_WD = ("CC0-1.0 (Wikidata claims: '" + S34_LIC_WD_QUOTE + "' -- https://www.wikidata.org/wiki/Wikidata:Copyright)")
S34_LIC_WP_QUOTE = ("Permission is granted to copy, distribute and/or modify Wikipedia's text under the terms of the "
                    "Creative Commons Attribution-ShareAlike 4.0 International License and, unless otherwise noted, the "
                    "GNU Free Documentation License")
S34_LICENSE_EVIDENCE = {
    "wikidata": {"url": "https://www.wikidata.org/wiki/Wikidata:Copyright",
                 "quote": S34_LIC_WD_QUOTE + " ...",
                 "api_rightsinfo": "All structured data from the main and property namespace is available under the "
                                   "Creative Commons CC0 License; text in the other namespaces is available under the "
                                   "Creative Commons Attribution-ShareAlike License; additional terms may apply.",
                 "read": "2026-10-01"},
    "wikipedia": {"url": "https://en.wikipedia.org/wiki/Wikipedia:Copyrights (revision 1369466282, 2026-08-15)",
                  "quote": S34_LIC_WP_QUOTE + " ...",
                  "attribution": "Wikipedia content can be copied, modified, and redistributed if and only if the copied "
                                 "version is made available on the same terms to others and acknowledgment of the authors "
                                 "of the Wikipedia article used is included (a link back to the article is generally "
                                 "thought to satisfy the attribution requirement ...)",
                  "api_rightsinfo": "Creative Commons Attribution-Share Alike 4.0 "
                                    "(https://creativecommons.org/licenses/by-sa/4.0/deed.en)",
                  "read": "2026-10-01"},
}


def s34_lic_wp(title, revid, qid):
    return (f"CC-BY-SA-4.0 (text: English Wikipedia article '{title}', revision {revid}, "
            f"https://en.wikipedia.org/w/index.php?oldid={revid}, by Wikipedia contributors; modified: lead section only, "
            f"plain text, the subject's names masked as [ENTITY]; share-alike applies to redistributed copies) + "
            f"CC0-1.0 (labels: Wikidata {qid} claims)")


S34_LIC_DUMP_EVIDENCE = {
    "url": "https://huggingface.co/datasets/wikimedia/wikipedia (revision b04c8d1c, lastModified 2024-01-09)",
    "card_yaml_license": ["cc-by-sa-3.0", "gfdl"],
    "note": "the 2023-11-01 dump postdates Wikipedia's switch of its text licence to CC BY-SA 4.0 (2023-06); older "
            "contributions are CC BY-SA 3.0 + GFDL, so records say CC BY-SA 3.0/4.0 + GFDL",
    "read": "2026-10-01"}


def s34_lic_dump(title, pageid, qid):
    return (f"CC-BY-SA-3.0/4.0 + GFDL (text: English Wikipedia article '{title}', https://en.wikipedia.org/wiki?curid="
            f"{pageid}, as of the 2023-11-01 dump via huggingface.co/datasets/{S34_HF_REPO} 20231101.en @ "
            f"{S34_HF_REV[:8]} (card: cc-by-sa-3.0, gfdl), by Wikipedia contributors; modified: lead only, plain text, "
            f"the subject's names masked as [ENTITY]; share-alike applies to redistributed copies) + "
            f"CC0-1.0 (labels: Wikidata {qid} claims)")


def s34_log(msg):
    print(f"[{_s34_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def s34_sha(*parts):
    return _s34_hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def s34_u(*parts):
    """A uniform number in [0, 1) from the parts: seeded, order-independent."""
    return int(_s34_hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:13], 16) / float(16 ** 13)


def s34_stratum(sl, pageid=None):
    if pageid is not None and int(pageid) >= S34_FRESH_PAGEID:
        return "fresh"
    sl = int(sl or 0)
    return "popular" if sl >= S34_POPULAR else "mid" if sl >= S34_MID else "obscure"


def s34_qid(uri):
    return str(uri).rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- network
class S34Refused(Exception):
    """The service refused us (HTTP 403). The brief: stop and report; never add personal contact details."""


_s34_last = {}


def s34_http(url, data=None, accept="application/json", timeout=150, tries=6):
    host = _s34_uparse.urlsplit(url).netloc
    last = None
    for t in range(tries):
        wait = _s34_last.get(host, 0.0) + S34_SPACING.get(host, 1.0) - _s34_time.time()
        if wait > 0:
            _s34_time.sleep(wait)
        _s34_last[host] = _s34_time.time()
        req = _s34_ureq.Request(url, data=data, headers={"User-Agent": S34_UA, "Accept": accept})
        try:
            with _s34_ureq.urlopen(req, timeout=timeout) as r:
                return r.read()
        except _s34_uerr.HTTPError as e:
            body = e.read(400).decode("utf-8", "replace")
            if e.code == 403:
                raise S34Refused(f"HTTP 403 from {host}: {body[:300]}") from None
            last = RuntimeError(f"HTTP {e.code} from {host}: {body[:200]}")
            if e.code not in (429, 500, 502, 503, 504) or t == tries - 1 or "TimeoutException" in body:
                raise last from None     # a query timeout is deterministic: fail fast so the caller's fallback runs
            ra = e.headers.get("Retry-After") if e.headers else None
            pause = max(float(ra) if ra and ra.isdigit() else 0.0, 5.0 * 2 ** t)
        except (_s34_uerr.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            if t == tries - 1:
                raise
            pause = 5.0 * 2 ** t
        s34_log(f"  retry {t + 1}/{tries - 1} for {host} in {pause:.0f}s ({last})")
        _s34_time.sleep(pause)
    raise last


def s34_sparql(query, timeout=150):
    """Rows as {var: value} (entity URIs shortened to Q-ids)."""
    data = _s34_uparse.urlencode({"query": query}).encode()
    # strict=False: some labels carry raw control characters, which strict JSON parsing rejects
    js = _s34_json.loads(s34_http(S34_SPARQL, data=data, accept="application/sparql-results+json", timeout=timeout),
                         strict=False)
    out = []
    for b in js["results"]["bindings"]:
        row = {}
        for k, v in b.items():
            val = v["value"]
            if v["type"] == "uri" and val.startswith("http://www.wikidata.org/entity/"):
                val = s34_qid(val)
            row[k] = val
        out.append(row)
    return out


def s34_api(base, params):
    params = dict(params, format="json", formatversion="2")
    js = _s34_json.loads(s34_http(base + "?" + _s34_uparse.urlencode(params)))
    if "error" in js:
        raise RuntimeError(f"API error from {base}: {str(js['error'])[:300]}")
    return js


def s34_wjsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with _s34_gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(_s34_json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def s34_rjsonl(path):
    with _s34_gzip.open(path, "rt", encoding="utf-8") as f:
        return [_s34_json.loads(line) for line in f if line.strip()]


def s34_labels(qids, langs="en"):
    """{qid: (label, description)} in English, via SPARQL VALUES batches."""
    out, qids = {}, sorted(set(qids))
    for i in range(0, len(qids), 400):
        vals = " ".join("wd:" + q for q in qids[i:i + 400])
        rows = s34_sparql(f"""SELECT ?i ?l ?d WHERE {{ VALUES ?i {{ {vals} }}
  OPTIONAL {{ ?i rdfs:label ?l . FILTER(LANG(?l) = "{langs}") }}
  OPTIONAL {{ ?i schema:description ?d . FILTER(LANG(?d) = "{langs}") }} }}""")
        for r in rows:
            out[r["i"]] = (r.get("l"), r.get("d"))
    return out


# --------------------------------------------------------------------------- common record helpers
def s34_record(source, rec_id, state, qs, labs, meta, base_id, licence):
    from build_jev_data import record, validate
    kinds = {q: ("hard" if sum(1 for v in labs[q].values() if v > 0) == 1 else "soft") for q in qs}
    r = record(source, state, qs, labs, kinds, None)
    r["id"] = rec_id
    r["base_id"] = base_id
    r["license"] = licence
    r["meta"] = meta
    validate(r)
    return r


def s34_onehot(keys, gold):
    assert gold in keys, (gold, keys)
    return {k: (1.0 if k == gold else 0.0) for k in keys}


def s34_noul_q(instructions, yes, no):
    return {"type": "noul", "instructions": instructions, "criteria": {"yes": yes, "no": no}}


def s34_write(out_root, source, splits, manifest):
    """<out_root>/<source>/<split>/<file>.jsonl (one file per record `source`) + manifest.json. Only this source's own
    jsonl files are replaced; nothing else is removed."""
    root = Path(out_root) / source
    for sp, recs in splits.items():
        d = root / sp
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jsonl"):
            old.unlink()
        fh = {}
        for r in recs:
            fn = r["source"].replace("/", "__") + ".jsonl"
            if fn not in fh:
                fh[fn] = (d / fn).open("w", encoding="utf-8")
            fh[fn].write(_s34_json.dumps(r, ensure_ascii=False) + "\n")
        for f in fh.values():
            f.close()
    (root / "manifest.json").write_text(_s34_json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
    s34_log(f"{source}: wrote " + ", ".join(f"{sp} {len(v)} records / {sum(len(r['questions']) for r in v)} units"
                                           for sp, v in splits.items()))


def s34_units(recs):
    return sum(len(r["questions"]) for r in recs)


def s34_label_stats(recs):
    """Per question id: counts of the gold option (one-hot) and the record count by stratum / family."""
    gold, strata, fam = _s34_dd(_S34Counter), _S34Counter(), _S34Counter()
    for r in recs:
        m = r.get("meta") or {}
        strata[m.get("stratum")] += 1
        fam[m.get("wd_family") or m.get("gold_type")] += 1
        for q, lab in r["labels"].items():
            g = max(lab, key=lab.get)
            gold[q][g] += 1
    return {"records": len(recs), "units": s34_units(recs), "by_stratum": dict(strata), "by_family": dict(fam),
            "gold_by_question": {q: dict(c.most_common()) for q, c in gold.items()}}


def s34_union_find(pairs):
    """Connected components over entity sets: {entity: root}. Used as base_id so a bootstrap clusters records that
    share an entity."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for ents in pairs:
        ents = list(ents)
        for e in ents[1:]:
            ra, rb = find(ents[0]), find(e)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {e: find(e) for e in list(parent)}


# =========================================================================== SHAPE 3: wd_compare
S34_CMP = {
    "inception": {"prop": "P571", "kind": "time", "plabel": "inception",
                  "classes": {"Q3918": "university", "Q4830453": "business", "Q476028": "association football club",
                              "Q33506": "museum", "Q11032": "newspaper", "Q7278": "political party",
                              "Q215380": "musical group"}, "sample": 25000},
    "elevation": {"prop": "P2044", "kind": "qty", "plabel": "elevation above sea level",
                  "classes": {"Q8502": "mountain", "Q8072": "volcano"}, "sample": 80000},
    "length": {"prop": "P2043", "kind": "qty", "plabel": "length", "classes": {"Q4022": "river"}, "sample": 120000},
    "height": {"prop": "P2048", "kind": "qty", "plabel": "height",
               "classes": {"Q11303": "skyscraper", "Q12518": "tower"}, "sample": 60000},
    "area": {"prop": "P2046", "kind": "qty", "plabel": "area", "classes": {"Q23442": "island", "Q23397": "lake"},
             "sample": 80000},
    "population": {"prop": "P1082", "kind": "pop", "plabel": "population",
                   "classes": {"Q515": "city", "Q3957": "town", "Q1549591": "big city", "Q3624078": "sovereign state",
                               "Q484170": "commune of France", "Q262166": "municipality in Germany",
                               "Q747074": "comune of Italy", "Q1093829": "city in the United States"},
                   "sample": 6000},
    "country": {"prop": "P17", "kind": "item", "plabel": "country",
                "classes": {"Q515": "city", "Q3957": "town", "Q8502": "mountain", "Q23397": "lake",
                            "Q3918": "university", "Q12280": "bridge", "Q16970": "church building",
                            "Q23413": "castle"}, "sample": 40000},
}
S34_CMP_EVAL_UNITS = {"inception": 1800, "population": 1500, "elevation": 600, "length": 600, "height": 600,
                      "area": 600, "country": 1800}                      # 7,500 <= 8,000
S34_CMP_TRAIN = {"inception": 3500, "population": 3500, "elevation": 900, "length": 900, "height": 900, "area": 900,
                 "country": 2500}                                        # ~13k records
S34_CMP_VAL_FRAC = 0.06
S34_POPULAR_ROWS = 25000  # row cap on the (unordered) popular scans
S34_POP_TOL = 0.005      # an item's statements at one date must agree within 0.5%
S34_QTY_TOL = 0.01       # an item's (normalized) quantity statements must agree within 1%
S34_NEAR_TIE = 0.02      # skip pairs within 2%
S34_DESC_LEAK = _s34_re.compile(r"\d|tallest|highest|longest|largest|biggest|smallest|shortest|lowest|oldest|youngest|"
                                r"newest|earliest|first|founded|established|\bsince\b|populous|capital|metropol|"
                                r"megacity|largest|major", _s34_re.I)
S34_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]


def s34_cmp_query(fam, cls, mode, limit):
    sp = S34_CMP[fam]
    p, kind = sp["prop"], sp["kind"]
    if mode == "sample":
        head = (f'SERVICE bd:sample {{ ?item wdt:P31 wd:{cls} . bd:serviceParam bd:sample.limit {limit} ; '
                f'bd:sample.sampleType "RANDOM" . }}\n  ?item wikibase:sitelinks ?sl .')
    elif mode == "popular_sample":
        head = (f'SERVICE bd:sample {{ ?item wdt:P31 wd:{cls} . bd:serviceParam bd:sample.limit {limit} ; '
                f'bd:sample.sampleType "RANDOM" . }}\n  ?item wikibase:sitelinks ?sl . FILTER(?sl >= {S34_POPULAR})')
    else:
        head = f"?item wdt:P31 wd:{cls} ; wikibase:sitelinks ?sl . FILTER(?sl >= {S34_POPULAR})"
    if kind == "time":
        val, vs = (f"?st psv:{p} ?v . ?v wikibase:timeValue ?t ; wikibase:timePrecision ?prec ; "
                   f"wikibase:timeCalendarModel ?cal ."), "?t ?prec ?cal"
    elif kind == "qty":
        val, vs = (f"?st psv:{p} ?v . ?v wikibase:quantityAmount ?amt ; wikibase:quantityUnit ?unit . "
                   f"OPTIONAL {{ ?st psn:{p}/wikibase:quantityAmount ?namt . }}"), "?amt ?unit ?namt"
    elif kind == "pop":
        val, vs = (f"?st psv:{p}/wikibase:quantityAmount ?amt . OPTIONAL {{ ?st pqv:P585 ?pv . ?pv wikibase:timeValue "
                   f"?pit ; wikibase:timePrecision ?pprec . }} OPTIONAL {{ ?item wdt:P17 ?cty . }}"), "?amt ?pit ?pprec ?cty"
    else:
        val, vs = f"?st ps:{p} ?c .", "?c"
    refs = " ".join("pr:" + x for x in S34_REF_PROPS)
    return f"""SELECT ?item ?sl ?st ?rank {vs} (SAMPLE(?refp) AS ?rp) WHERE {{
  {head}
  ?item p:{p} ?st . ?st wikibase:rank ?rank . FILTER(?rank != wikibase:DeprecatedRank)
  {val}
  OPTIONAL {{ ?st prov:wasDerivedFrom ?ref . ?ref ?refp ?rv . FILTER(?refp IN ({refs.replace(' ', ', ')})) }}
}} GROUP BY ?item ?sl ?st ?rank {vs}
{"" if mode == "sample" else "LIMIT " + str(S34_POPULAR_ROWS)}"""


def s34_cmp_rows_to_items(fam, cls, rows, items):
    """Aggregate statement rows into compact items: {qid: {qid, cls, sl, stmts: {sid: {...}}, cty: [...]}}."""
    kind = S34_CMP[fam]["kind"]
    for r in rows:
        q = r["item"]
        it = items.setdefault(q, {"qid": q, "fam": fam, "cls": cls, "sl": int(r.get("sl") or 0), "stmts": {}, "cty": []})
        sid = r["st"].rsplit("/", 1)[-1]
        s = it["stmts"].setdefault(sid, {"id": sid, "rank": "p" if r["rank"].endswith("PreferredRank") else "n",
                                         "ref": 0})
        if r.get("rp"):
            s["ref"] = 1
        if kind == "time":
            s.update(t=r["t"], prec=int(r["prec"]), cal=s34_qid(r.get("cal", "")))
        elif kind == "qty":
            s.update(amt=float(r["amt"]), unit=s34_qid(r["unit"]), namt=float(r["namt"]) if r.get("namt") else None)
        elif kind == "pop":
            s.update(amt=float(r["amt"]))
            if r.get("pit"):
                s.update(pit=r["pit"][:10], pprec=int(r.get("pprec") or 0))
            if r.get("cty") and r["cty"] not in it["cty"]:
                it["cty"].append(r["cty"])
        else:
            s.update(c=r["c"])
    return items


def s34_compare_fetch(a):
    """SPARQL -> compact cache: compare_<family>.jsonl.gz (items with their non-deprecated statements), compare_meta.json
    (English labels / descriptions of every item, unit and country; sovereign states with continents)."""
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    report = {"snapshot": _s34_dt.now().strftime("%Y-%m-%d"), "queries": []}
    allq = set()
    for fam, sp in S34_CMP.items():
        path = work / f"compare_{fam}.jsonl.gz"
        if path.exists() and not a.refetch:
            items = {it["qid"]: it for it in s34_rjsonl(path)}
            s34_log(f"compare {fam}: cached {len(items)} items")
        else:
            items = {}
            for cls in sp["classes"]:
                for mode, limit in (("sample", sp["sample"]), ("popular", 0)):
                    t0 = _s34_time.time()
                    try:
                        rows = s34_sparql(s34_cmp_query(fam, cls, mode, limit))
                    except S34Refused:
                        raise
                    except Exception as e:  # noqa: BLE001  timeout on a huge class: fall back to a big random sample
                        if mode != "popular":
                            raise
                        s34_log(f"  {fam}/{cls} popular scan failed ({str(e)[:120]}); bd:sample fallback")
                        try:
                            rows = s34_sparql(s34_cmp_query(fam, cls, "popular_sample", 60000))
                        except S34Refused:
                            raise
                        except Exception as e2:  # noqa: BLE001
                            s34_log(f"  {fam}/{cls} popular fallback failed too ({str(e2)[:120]}); skipped")
                            report["queries"].append({"family": fam, "class": cls, "mode": mode,
                                                      "error": str(e2)[:200]})
                            continue
                    n0 = len(items)
                    s34_cmp_rows_to_items(fam, cls, rows, items)
                    report["queries"].append({"family": fam, "class": cls, "mode": mode, "rows": len(rows),
                                              "new_items": len(items) - n0, "s": round(_s34_time.time() - t0, 1)})
                    s34_log(f"compare {fam}/{cls}/{mode}: {len(rows)} rows, {len(items) - n0} new items "
                            f"({_s34_time.time() - t0:.0f}s)")
            for it in items.values():
                it["stmts"] = list(it["stmts"].values())
            s34_wjsonl(path, list(items.values()))
        vrep = _S34Counter()
        for it in items.values():          # labels only for entities that pass the per-item rules
            if s34_item_value(fam, it, vrep) is None:
                continue
            allq.add(it["qid"])
            for s in it["stmts"]:
                for k in ("unit", "c"):
                    if s.get(k):
                        allq.add(s[k])
            allq |= set(it.get("cty") or [])
    mpath = work / "compare_meta.json"
    meta = _s34_json.loads(mpath.read_text()) if mpath.exists() and not a.refetch else {}
    if "states" not in meta:
        rows = s34_sparql("""SELECT ?c ?cont WHERE { ?c wdt:P31 wd:Q3624078 .
  FILTER NOT EXISTS { ?c wdt:P31 wd:Q3024240 } OPTIONAL { ?c wdt:P30 ?cont } }""")
        st = _s34_dd(list)
        for r in rows:
            st[r["c"]].append(r.get("cont"))
        meta["states"] = {c: sorted({x for x in v if x}) for c, v in st.items()}
    need = sorted((allq | set(meta["states"])) - set(meta.get("labels", {})))
    if need:
        s34_log(f"compare: fetching English labels for {len(need)} entities")
        lab = s34_labels(need)
        meta.setdefault("labels", {}).update({q: list(v) for q, v in lab.items()})
    meta["fetch_report"] = report if report["queries"] else meta.get("fetch_report")
    mpath.write_text(_s34_json.dumps(meta, ensure_ascii=False))
    return meta


def s34_year(t):
    """Signed year from a WDQS time literal ('1636-01-01T00:00:00Z', '-0500-01-01T00:00:00Z')."""
    m = _s34_re.match(r"^([+-]?)(\d+)-(\d\d)-(\d\d)", t)
    if not m:
        return None
    y = int(m.group(2))
    return -y if m.group(1) == "-" else y


def s34_best(stmts):
    pref = [s for s in stmts if s["rank"] == "p"]
    return pref or [s for s in stmts if s["rank"] == "n"]


def s34_item_value(fam, it, rep):
    """The item's single, referenced, non-conflicting value for a family, or None (reason counted in rep).
    time -> {'year', 'date', 'prec', 'cal', 'sid'}; qty -> {'v' (normalized), 'amt', 'unit', 'sid'};
    item -> {'c', 'sid'}; pop -> {date: {'v', 'prec', 'sid'}} for every valid point in time."""
    kind = S34_CMP[fam]["kind"]
    if kind == "pop":
        by = _s34_dd(list)
        for s in it["stmts"]:
            if s.get("pit"):
                by[(s["pit"], s.get("pprec", 0))].append(s)
        out = {}
        for (d, prec), ss in by.items():
            vals = [s["amt"] for s in ss]
            if prec < 9 or min(vals) <= 0:
                rep["pop_date/bad_precision_or_value"] += 1
                continue
            if max(vals) / min(vals) > 1 + S34_POP_TOL:
                rep["pop_date/conflicting_values"] += 1
                continue
            if not any(s["ref"] for s in ss):
                rep["pop_date/unreferenced"] += 1
                continue
            out[f"{d}|{prec}"] = {"v": vals[0], "prec": prec, "date": d, "sid": ss[0]["id"]}
        if not out:
            rep["item/no_valid_population_date"] += 1
            return None
        return out
    best = s34_best(it["stmts"])
    if not best:
        rep["item/no_statement"] += 1
        return None
    if not any(s["ref"] for s in best):
        rep["item/unreferenced"] += 1
        return None
    if kind == "time":
        years = {s34_year(s["t"]) for s in best}
        if len(years) != 1 or None in years:
            rep["item/conflicting_statements"] += 1
            return None
        if min(s["prec"] for s in best) < 9:
            rep["item/precision_below_year"] += 1
            return None
        s = next(s for s in best if s["ref"])
        return {"year": years.pop(), "date": s["t"][:10], "prec": s["prec"], "cal": s.get("cal"), "sid": s["id"]}
    if kind == "qty":
        vals = [s["namt"] for s in best if s.get("namt") is not None]
        if len(vals) != len(best):
            rep["item/not_normalizable_unit"] += 1
            return None
        if min(vals) <= 0:
            rep["item/non_positive_value"] += 1
            return None
        if max(vals) / min(vals) > 1 + S34_QTY_TOL:
            rep["item/conflicting_statements"] += 1
            return None
        s = next(s for s in best if s["ref"])
        return {"v": s["namt"], "amt": s["amt"], "unit": s["unit"], "sid": s["id"]}
    cs = {s["c"] for s in best}
    if len(cs) != 1:
        rep["item/conflicting_statements"] += 1
        return None
    s = next(s for s in best if s["ref"])
    return {"c": s["c"], "sid": s["id"]}


def s34_fmt_num(x):
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def s34_fmt_date(d, prec, cal=None):
    y = s34_year(d)
    ys = f"{-y} BCE" if y is not None and y <= 0 else str(y)
    m = _s34_re.match(r"^[+-]?\d+-(\d\d)-(\d\d)", d)
    if prec >= 11 and m:
        s = f"{int(m.group(2))} {S34_MONTHS[int(m.group(1)) - 1]} {ys}"
    elif prec == 10 and m:
        s = f"{S34_MONTHS[int(m.group(1)) - 1]} {ys}"
    else:
        s = ys
    return s + (" (Julian calendar)" if cal == "Q1985786" else "")


S34_SI = {"qty_len": "metre", "qty_area": "square metre"}


def s34_unit_text(unit, labels):
    if unit in ("1", "", None):
        return ""
    return (labels.get(unit) or [None])[0] or unit


def s34_card(it, labels, cls_label, show_desc=True):
    lab, desc = (labels.get(it["qid"]) or [None, None])[:2]
    desc = desc if (show_desc and desc and not S34_DESC_LEAK.search(desc)) else None
    return f"- {lab} [{it['qid']}] ({cls_label})" + (f": {desc}" if desc else "")


def s34_cmp_build(a):
    meta = s34_compare_fetch(a) if not a.no_fetch else _s34_json.loads((Path(a.work) / "compare_meta.json").read_text())
    labels, states = meta["labels"], meta["states"]
    rng = _s34_random.Random(f"{a.seed}:wd_compare")
    rep = _S34Counter()
    snapshot = (meta.get("fetch_report") or {}).get("snapshot") or "2026-10-01"
    split_of = lambda q: ("eval" if s34_u(a.seed, "wd34split", q) < 0.30 else  # noqa: E731
                          "val" if s34_u(a.seed, "wd34split", q) < 0.30 + S34_CMP_VAL_FRAC else "train")
    out = {"train": [], "val": [], "eval": []}
    fam_report = {}

    def ok_label(q):
        lab = (labels.get(q) or [None])[0]
        return bool(lab) and not _s34_re.fullmatch(r"Q\d+", lab)

    for fam, sp in S34_CMP.items():
        path = Path(a.work) / f"compare_{fam}.jsonl.gz"
        if not path.exists():
            rep[f"{fam}/missing_cache"] += 1
            continue
        frep = _S34Counter()
        items = {}
        for it in s34_rjsonl(path):
            v = s34_item_value(fam, it, frep)
            if v is None:
                continue
            if not ok_label(it["qid"]):
                frep["item/no_english_label"] += 1
                continue
            if fam == "country" and v["c"] not in states:
                frep["item/country_not_a_current_sovereign_state"] += 1
                continue
            it = dict(it, val=v, split=split_of(it["qid"]), stratum=s34_stratum(it["sl"]))
            items[it["qid"]] = it
        frep["items_valid"] = len(items)
        cands = s34_cmp_pairs(fam, items, labels, states, rng, frep)
        fam_report[fam] = {"build": dict(frep), "candidates_by_split_stratum": dict(_S34Counter(
            f"{c['split']}/{c['stratum']}" for c in cands))}
        for sp_name in ("eval", "train", "val"):
            pool = [c for c in cands if c["split"] == sp_name]
            if sp_name == "eval":
                pick = s34_stratified(pool, S34_CMP_EVAL_UNITS[fam], rng)
            elif sp_name == "train":
                pick = s34_stratified(pool, S34_CMP_TRAIN[fam], rng, natural=True)
            else:
                pick = s34_stratified(pool, max(60, int(S34_CMP_TRAIN[fam] * S34_CMP_VAL_FRAC)), rng, natural=True)
            for c in pick:
                out[sp_name].append(s34_cmp_record(fam, c, labels, snapshot, open_book=(sp_name != "eval"), rng=rng))
            fam_report[fam][f"selected_{sp_name}"] = len(pick)
    # bootstrap clusters: connected components of shared entities, per split
    for sp_name, recs in out.items():
        comp = s34_union_find([r["meta"]["qids"] for r in recs])
        for r in recs:
            r["base_id"] = f"wd_compare/{comp[r['meta']['qids'][0]]}"
    probes = {"eval_state_removed": [s34_state_removed(r) for r in out["eval"]]}
    man = {"source": "wd_compare", "shape": 3, "builder": "build_wikidata_jev.py wd-compare (shapes 3-4 section)",
           "built": _s34_dt.now().isoformat(timespec="seconds"), "wikidata_snapshot": snapshot, "seed": a.seed,
           "license": S34_LIC_WD, "license_evidence": {"wikidata": S34_LICENSE_EVIDENCE["wikidata"]},
           "panel": "eval = K side panel (closed-book knowledge; scored per set, never in GDE-1's V/P/O/X headline); "
                    "train / val = OPEN-BOOK variant (the facts are in the state; exact computation)",
           "rules": {"reference": f"every compared claim has a reference with {S34_REF_PROPS} (P143 'imported from "
                                  f"Wikimedia project' does not count)",
                     "rank": "deprecated statements ignored; preferred rank used when present, else normal",
                     "conflict": f"an item is skipped when its best-rank statements disagree (time: different years; "
                                 f"quantity: > {S34_QTY_TOL:.0%} apart; population at one date: > {S34_POP_TOL:.1%} apart; "
                                 f"country: more than one)",
                     "near_tie": f"pairs within {S34_NEAR_TIE:.0%} (quantities, populations) or in the same year "
                                 f"(inception) are skipped",
                     "precision": "time values need at least year precision",
                     "split": "by entity: sha1(seed, qid) -> eval 30% / val 6% / train 64%; every entity of a record "
                              "is in the record's split, so no eval entity appears in train",
                     "strata": f"by sitelink count of the LEAST popular entity of a record: popular >= {S34_POPULAR}, "
                               f"mid {S34_MID}-{S34_POPULAR - 1}, obscure <= {S34_MID - 1}",
                     "state_removed_probe": "eval_state_removed/ = the eval records with the state replaced by "
                                            f"'{S34_STATE_REMOVED}' (options = the entities' English labels)"},
           "families": {f: {k: v for k, v in sp.items() if k != "sample"} for f, sp in S34_CMP.items()},
           "family_report": fam_report, "build_report": dict(rep),
           "fetch_report": meta.get("fetch_report"),
           "splits": {sp: s34_label_stats(v) for sp, v in {**out, **probes}.items()}}
    s34_write(a.out, "wd_compare", {**out, **probes}, man)
    return 0


def s34_cmp_pairs(fam, items, labels, states, rng, frep):
    """Candidate records (dicts) for a family; each entity is used in at most 2 records."""
    kind = S34_CMP[fam]["kind"]
    uses = _S34Counter()
    seen = set()
    cands = []
    lab = lambda q: (labels.get(q) or [""])[0]  # noqa: E731

    def free(q):
        return uses[q] < 2

    if kind == "item":                                   # 4-way: which one is in country Z
        pools = _s34_dd(list)
        for it in items.values():
            pools[(it["cls"], it["split"], it["stratum"])].append(it)
        for key, pool in sorted(pools.items()):
            rng.shuffle(pool)
            for t in pool:
                if not free(t["qid"]):
                    continue
                z = t["val"]["c"]
                if s34_names_country(lab(t["qid"]), lab(z)):
                    frep["record/target_label_names_the_country"] += 1
                    continue
                zc = set(states.get(z) or [])
                others = [d for d in pool if d["qid"] != t["qid"] and free(d["qid"]) and d["val"]["c"] != z
                          and lab(d["qid"]).lower() != lab(t["qid"]).lower()]
                same = [d for d in others if zc & set(states.get(d["val"]["c"]) or [])]
                same_ids = {d["qid"] for d in same}
                pick, cs, ls = [], {z}, {lab(t["qid"]).lower()}
                for d in same + [d for d in others if d["qid"] not in same_ids]:
                    if d["val"]["c"] in cs or lab(d["qid"]).lower() in ls:
                        continue
                    if s34_names_country(lab(d["qid"]), lab(z)) or s34_names_country(lab(d["qid"]), lab(d["val"]["c"])):
                        frep["distractor/label_names_a_country"] += 1
                        continue
                    pick.append(d)
                    cs.add(d["val"]["c"])
                    ls.add(lab(d["qid"]).lower())
                    if len(pick) == 3:
                        break
                if len(pick) < 3:
                    frep["record/not_enough_distractors"] += 1
                    continue
                ents = [t] + pick
                k = tuple(sorted(e["qid"] for e in ents))
                if k in seen:
                    continue
                seen.add(k)
                for e in ents:
                    uses[e["qid"]] += 1
                cands.append({"fam": fam, "ents": ents, "gold": t["qid"], "country": z, "split": t["split"],
                              "stratum": key[2], "cls": key[0],
                              "same_continent_distractors": sum(1 for d in pick if d["qid"] in same_ids)})
        return cands

    if kind == "pop":
        pools = _s34_dd(list)
        for it in items.values():
            cty = it.get("cty") or []
            if it["cls"] == "Q3624078":
                ckey = "*"
            elif len(cty) == 1:
                ckey = cty[0]
            else:
                frep["item/population_without_single_country"] += 1
                continue
            for dk in it["val"]:
                pools[(dk, ckey, it["split"], it["stratum"])].append(it)
        order = sorted(pools, key=lambda k: -len(pools[k]))
    else:
        pools = _s34_dd(list)
        for it in items.values():
            pools[(it["cls"], it["split"], it["stratum"])].append(it)
        order = sorted(pools)
    for key in order:
        pool = [it for it in pools[key]]
        for _pass in range(2):
            rng.shuffle(pool)
            avail = [it for it in pool if free(it["qid"])]
            i = 0
            while i < len(avail) - 1:
                x = avail[i]
                if not free(x["qid"]):
                    i += 1
                    continue
                for j in range(i + 1, min(i + 6, len(avail))):
                    y = avail[j]
                    if not free(y["qid"]) or lab(x["qid"]).lower() == lab(y["qid"]).lower():
                        continue
                    k = tuple(sorted((x["qid"], y["qid"]))) + ((key[0],) if kind == "pop" else ())
                    if k in seen:
                        continue
                    if kind == "time":
                        vx, vy = x["val"]["year"], y["val"]["year"]
                        if vx == vy:
                            frep["pair/same_year"] += 1
                            continue
                    else:
                        vx = x["val"][key[0]]["v"] if kind == "pop" else x["val"]["v"]
                        vy = y["val"][key[0]]["v"] if kind == "pop" else y["val"]["v"]
                        if abs(vx - vy) / max(vx, vy) < S34_NEAR_TIE:
                            frep["pair/near_tie"] += 1
                            continue
                    seen.add(k)
                    uses[x["qid"]] += 1
                    uses[y["qid"]] += 1
                    c = {"fam": fam, "ents": [x, y], "split": key[-2], "stratum": key[-1],
                         "cls": x["cls"] if kind != "pop" else None}
                    if kind == "time":
                        c["gold"] = x["qid"] if vx < vy else y["qid"]
                        c["gap_years"] = abs(vx - vy)
                    elif kind == "qty":
                        c["gold"] = x["qid"] if vx > vy else y["qid"]
                        c["ratio"] = round(max(vx, vy) / min(vx, vy), 4)
                    else:
                        c.update(date_key=key[0], country=key[1], gold=x["qid"] if vx > vy else y["qid"],
                                 ratio=round(max(vx, vy) / min(vx, vy), 4))
                    cands.append(c)
                    break
                i += 1
    return cands


def s34_names_country(label, country_label):
    """True if an entity's label contains a country's English name (a give-away in the 4-way country question)."""
    c = (country_label or "").lower()
    return len(c) >= 4 and c in (label or "").lower()


def s34_stratified(pool, n, rng, natural=False):
    """Up to n candidates: equal shares per stratum (shortfalls refilled from the others), or a uniform sample when
    natural=True. Seeded."""
    pool = list(pool)
    rng.shuffle(pool)
    if natural or len(pool) <= n:
        return pool[:n]
    by = _s34_dd(list)
    for c in pool:
        by[c["stratum"]].append(c)
    keys = sorted(by)
    out, quota = [], {k: n // len(keys) for k in keys}
    for k in keys:
        out += by[k][:quota[k]]
    rest = [c for k in keys for c in by[k][quota[k]:]]
    rng.shuffle(rest)
    return out + rest[:n - len(out)]


def s34_cmp_record(fam, c, labels, snapshot, open_book, rng):
    sp = S34_CMP[fam]
    kind, prop, plabel = sp["kind"], sp["prop"], sp["plabel"]
    lab = lambda q: (labels.get(q) or [""])[0]  # noqa: E731
    ents = list(c["ents"])
    rng.shuffle(ents)
    qids = [e["qid"] for e in ents]
    cls_lab = lambda e: (sp["classes"].get(e["cls"]) or "entity")  # noqa: E731
    crit = {e["qid"]: lab(e["qid"]) for e in ents}
    if kind == "item":
        z = c["country"]
        zl = lab(z)
        lines = ["Four entities from Wikidata (English label [item id] (class)):"]
        lines += [s34_card(e, labels, cls_lab(e), show_desc=False) for e in ents]
        if open_book:
            lines += ["", "Facts from Wikidata (referenced 'country' (P17) claims):"]
            lines += [f"- {lab(e['qid'])}: country = {lab(e['val']['c'])}" for e in ents]
        instr = (f"Exactly one of these four is located in {zl}, according to Wikidata's referenced 'country' (P17) "
                 f"claims; each of the other three is in a different country. Which one is in {zl}?")
        if open_book:
            instr = "Using only the facts in the state: " + instr
        qs = {"which_in_country": {"type": "choice", "instructions": instr, "criteria": crit}}
        labs = {"which_in_country": s34_onehot(list(crit), c["gold"])}
        values = {e["qid"]: {"country": e["val"]["c"], "statement": e["val"]["sid"]} for e in ents}
        extra = {"country": z, "same_continent_distractors": c["same_continent_distractors"]}
    else:
        lines = ["Two entities from Wikidata (English label [item id] (class): description):"]
        lines += [s34_card(e, labels, cls_lab(e) if kind != "pop" else (sp["classes"].get(e["cls"]) or "place"))
                  for e in ents]
        if kind == "time":
            if open_book:
                lines += ["", "Facts from Wikidata (referenced 'inception' (P571) claims):"]
                lines += [f"- {lab(e['qid'])}: inception {s34_fmt_date(e['val']['date'], e['val']['prec'], e['val']['cal'])}"
                          for e in ents]
            instr = ("Which of these two was founded, formed or established earlier, according to Wikidata's "
                     "referenced 'inception' (P571) claims? The two inception years differ.")
            values = {e["qid"]: {"year": e["val"]["year"], "date": e["val"]["date"], "precision": e["val"]["prec"],
                                 "statement": e["val"]["sid"]} for e in ents}
            extra = {"gap_years": c["gap_years"]}
            qid_q = "founded_earlier"
        elif kind == "qty":
            if open_book:
                lines += ["", f"Facts from Wikidata (referenced '{plabel}' ({prop}) claims, in the units as stored):"]
                lines += [f"- {lab(e['qid'])}: {plabel} {s34_fmt_num(e['val']['amt'])} "
                          f"{s34_unit_text(e['val']['unit'], labels)}".rstrip() for e in ents]
            word = {"elevation": "the higher elevation above sea level", "length": "the greater length",
                    "height": "the greater height", "area": "the larger area"}[fam]
            instr = (f"Which of these two has {word}, according to Wikidata's referenced '{plabel}' ({prop}) claims? "
                     f"The two values differ by more than {S34_NEAR_TIE:.0%}.")
            values = {e["qid"]: {"normalized_si": e["val"]["v"], "amount": e["val"]["amt"], "unit": e["val"]["unit"],
                                 "statement": e["val"]["sid"]} for e in ents}
            extra = {"ratio": c["ratio"]}
            qid_q = {"elevation": "higher", "length": "longer", "height": "taller", "area": "larger_area"}[fam]
        else:
            dk = c["date_key"]
            v0 = ents[0]["val"][dk]
            dtext = s34_fmt_date(v0["date"], v0["prec"])
            dtext = ("the year " + dtext) if v0["prec"] == 9 else dtext
            if open_book:
                lines += ["", "Facts from Wikidata (referenced 'population' (P1082) claims with their 'point in time'):"]
                for e in ents:
                    ks = sorted(e["val"], key=lambda k: (abs(s34_year(e["val"][k]["date"]) - s34_year(v0["date"])), k))[:4]
                    ks = sorted(ks)
                    lines.append(f"- {lab(e['qid'])}: " + "; ".join(
                        f"{s34_fmt_num(e['val'][k]['v'])} ({s34_fmt_date(e['val'][k]['date'], e['val'][k]['prec'])})"
                        for k in ks))
            A, B = lab(ents[0]["qid"]), lab(ents[1]["qid"])
            instr = (f"According to Wikidata's referenced population (P1082) figures for {dtext}, did {A} have a "
                     f"larger population than {B}? The two figures differ by more than {S34_NEAR_TIE:.0%}.")
            if open_book:
                instr = "Using only the facts in the state: " + instr
            qs = {"population_larger": s34_noul_q(instr, f"{A} had the larger population ({dtext})",
                                                  f"{B} had the larger population ({dtext})")}
            yes = 1.0 if c["gold"] == ents[0]["qid"] else 0.0
            labs = {"population_larger": {"yes": yes, "no": 1.0 - yes}}
            values = {e["qid"]: {"population": e["val"][dk]["v"], "point_in_time": e["val"][dk]["date"],
                                 "precision": e["val"][dk]["prec"], "statement": e["val"][dk]["sid"]} for e in ents}
            extra = {"ratio": c["ratio"], "point_in_time": v0["date"], "country_pool": c["country"]}
        if kind != "pop":
            if open_book:
                instr = "Using only the facts in the state: " + instr
            qs = {qid_q: {"type": "choice", "instructions": instr, "criteria": crit}}
            labs = {qid_q: s34_onehot(list(crit), c["gold"])}
    state = "\n".join(lines)
    source = f"wd_compare/{fam}"
    rec_id = f"wd_compare-{fam}-{'ob' if open_book else 'cb'}-{s34_sha(*qids, c.get('date_key', ''))}"
    meta = {"wd_family": fam, "prop": prop, "qids": qids, "gold_qid": c["gold"], "values": values,
            "stratum": c["stratum"], "sitelinks": {e["qid"]: e["sl"] for e in ents},
            "classes": {e["qid"]: e["cls"] for e in ents}, "split": c["split"], "open_book": open_book,
            "label_provenance": f"computed from referenced Wikidata {prop} claims (snapshot {snapshot})",
            "wikidata_snapshot": snapshot, **extra}
    if not open_book:
        meta.update(gde1_set="wd_compare", domain="K", panel="K (knowledge side panel; never in the V/P/O/X headline)",
                    decision_date=snapshot, info_date=snapshot,
                    time_slice="none (Wikidata facts, mostly in every model's pretraining data)")
    return s34_record(source, rec_id, state, qs, labs, meta, base_id=rec_id, licence=S34_LIC_WD)


def s34_state_removed(r):
    """The knowledge probe: the same questions with the state removed."""
    p = _s34_json.loads(_s34_json.dumps(r))
    p["id"] = r["id"] + "-sr"
    p["state"] = S34_STATE_REMOVED
    p["meta"] = dict(p["meta"], probe="state_removed", probe_of=r["id"])
    return p


# =========================================================================== SHAPE 4: wd_classify
S34_TYPES = {   # type key -> (option text, direct P31 classes); verified English labels 2026-10-01 (see the doc)
    "person": ("a person (a human being)", ["Q5"]),
    "settlement": ("a city, town, village or municipality",
                   ["Q515", "Q3957", "Q532", "Q486972", "Q1549591", "Q1093829", "Q747074", "Q484170", "Q262166",
                    "Q2039348", "Q15284", "Q5119", "Q200250", "Q1637706", "Q7930989", "Q3184121", "Q2074737",
                    "Q15127012"]),
    "admin_region": ("a country, state, province, county or other administrative region",
                     ["Q6256", "Q3624078", "Q47168", "Q35657", "Q34876", "Q10864048", "Q28575", "Q1115575"]),
    "company": ("a company or business",
                ["Q4830453", "Q783794", "Q891723", "Q6881511", "Q1589009", "Q167037", "Q18388277", "Q1058914",
                 "Q210167", "Q1137109", "Q46970", "Q2085381", "Q18127", "Q1762059", "Q507619"]),
    "organization": ("a non-commercial organization (political party, NGO, agency, association, foundation, union)",
                     ["Q43229", "Q163740", "Q7278", "Q48204", "Q484652", "Q15911314", "Q327333", "Q7210356",
                      "Q2659904", "Q708676", "Q157031", "Q4438121", "Q1530022", "Q31855", "Q178790"]),
    "school": ("a school, college or university",
               ["Q3918", "Q875538", "Q902104", "Q189004", "Q9842", "Q9826", "Q3914", "Q159334", "Q1371037",
                "Q15936437", "Q62078547"]),
    "sports_team": ("a sports team or sports club",
                    ["Q476028", "Q847017", "Q12973014", "Q13393265", "Q17156793", "Q4498974", "Q6979593"]),
    "film": ("a film", ["Q11424", "Q24869", "Q202866", "Q506240", "Q93204", "Q24862"]),
    "tv_series": ("a television series or programme",
                  ["Q5398426", "Q581714", "Q1259759", "Q15416", "Q526877", "Q63952888"]),
    "music_release": ("an album, single, song or other musical work",
                      ["Q482994", "Q134556", "Q7366", "Q208569", "Q209939", "Q222910", "Q169930", "Q105543609"]),
    "musical_group": ("a band or other musical group",
                      ["Q215380", "Q5741069", "Q2088357", "Q9212979", "Q216337", "Q641066"]),
    "written_work": ("a book, novel, poem, comic, magazine, newspaper or journal",
                     ["Q571", "Q7725634", "Q47461344", "Q8261", "Q5185279", "Q49084", "Q1004", "Q14406742",
                      "Q21198342", "Q277759", "Q41298", "Q11032", "Q1002697", "Q5633421", "Q737498"]),
    "software_game": ("a video game, software or programming language", ["Q7889", "Q7397", "Q9143"]),
    "building_structure": ("a building or structure (incl. bridges, stations, roads, dams, stadiums)",
                           ["Q41176", "Q11303", "Q16970", "Q12280", "Q23413", "Q483110", "Q1248784", "Q55488",
                            "Q2977", "Q32815", "Q44539", "Q16560", "Q57821", "Q12518", "Q39614", "Q1137809", "Q3947",
                            "Q1021645", "Q24398318", "Q108325", "Q928830", "Q34442", "Q12323", "Q44377", "Q11315"]),
    "natural_feature": ("a natural geographic feature (mountain, river, lake, island, ...)",
                        ["Q8502", "Q4022", "Q23397", "Q23442", "Q54050", "Q46831", "Q355304", "Q8514", "Q34763",
                         "Q185113", "Q35666", "Q8072", "Q39594", "Q34038", "Q1322134", "Q165", "Q150784", "Q39816",
                         "Q33837", "Q133056"]),
    "taxon": ("a biological taxon (species, genus, family, ...)", ["Q16521"]),
    "event": ("an event (battle, war, election, disaster, sports event or season, ...)",
              ["Q178561", "Q198", "Q40231", "Q8065", "Q3839081", "Q16510064", "Q18608583", "Q27020041", "Q1656682",
               "Q13406554", "Q645883", "Q124757", "Q273120", "Q350604", "Q744913", "Q7944", "Q500834", "Q15275719",
               "Q1079023", "Q1190554"]),
    "vehicle_vessel": ("a ship, aircraft model, car model or other vehicle type",
                       ["Q11446", "Q2811", "Q174736", "Q3231690", "Q15056993", "Q15056995", "Q19832486", "Q17205",
                        "Q182531", "Q161705"]),
}
S34_COUNTRIES = {"us": ("United States", "Q30"), "gb": ("United Kingdom", "Q145"), "ca": ("Canada", "Q16"),
                 "au": ("Australia", "Q408"), "in": ("India", "Q668"), "fr": ("France", "Q142"),
                 "de": ("Germany", "Q183"), "it": ("Italy", "Q38"), "es": ("Spain", "Q29"),
                 "nl": ("Netherlands", "Q55"), "se": ("Sweden", "Q34"), "pl": ("Poland", "Q36"),
                 "ru": ("Russia", "Q159"), "jp": ("Japan", "Q17"), "cn": ("China", "Q148"), "br": ("Brazil", "Q155")}
S34_DEMONYMS = {"us": ["American"], "gb": ["British", "English", "Scottish", "Welsh"], "ca": ["Canadian"],
                "au": ["Australian"], "in": ["Indian"], "fr": ["French"], "de": ["German"], "it": ["Italian"],
                "es": ["Spanish"], "nl": ["Dutch"], "se": ["Swedish"], "pl": ["Polish"], "ru": ["Russian"],
                "jp": ["Japanese"], "cn": ["Chinese"], "br": ["Brazilian"]}
S34_OCC = {
    "politics": ("politics, diplomacy or government service", ["Q82955", "Q193391", "Q372436", "Q212238"]),
    "sport": ("sport (athlete, player, coach, racing driver)",
              ["Q937857", "Q3665646", "Q10833314", "Q10871364", "Q19204627", "Q11513337", "Q11774891", "Q12299841",
               "Q13141064", "Q2066131", "Q10843402", "Q13382576", "Q2309784", "Q11338576", "Q10349745", "Q13474373",
               "Q14089670", "Q628099", "Q41583", "Q15117302", "Q3014296", "Q13381863", "Q10873124", "Q4144610",
               "Q2730732", "Q13561328", "Q11607585", "Q378622", "Q17516936", "Q10866633", "Q13381753", "Q18515558",
               "Q19841381", "Q15306067", "Q16029547", "Q13414980", "Q12840545", "Q21141381"]),
    "acting_entertainment": ("acting, comedy, modelling or presenting",
                             ["Q33999", "Q10800557", "Q10798782", "Q2405480", "Q2259451", "Q245068", "Q4610556",
                              "Q947873", "Q2722764"]),
    "music": ("music (singer, musician, composer, conductor, producer)",
              ["Q177220", "Q639669", "Q36834", "Q488205", "Q753110", "Q855091", "Q486748", "Q130857", "Q183945",
               "Q158852", "Q2252262", "Q1259917", "Q386854", "Q806349", "Q1075651", "Q584301", "Q12800682",
               "Q1198887"]),
    "writing_journalism": ("writing or journalism",
                           ["Q36180", "Q49757", "Q6625963", "Q1930187", "Q482980", "Q28389", "Q214917", "Q4853732",
                            "Q11774202", "Q333634", "Q18844224", "Q1607826", "Q15980158", "Q18814623"]),
    "science_academia": ("science, engineering or academia",
                         ["Q901", "Q1622272", "Q169470", "Q593644", "Q170790", "Q864503", "Q2374149", "Q350979",
                          "Q1650915", "Q188094", "Q201788", "Q4964182", "Q11063", "Q3055126", "Q520549", "Q81096",
                          "Q205375", "Q1225716", "Q2919046", "Q4773904", "Q3745071", "Q14467526", "Q16267607"]),
    "visual_arts_design": ("visual arts, architecture or design",
                           ["Q1028181", "Q1281618", "Q33231", "Q42973", "Q483501", "Q644687", "Q1114448", "Q3391743",
                            "Q5322166", "Q627325", "Q329439", "Q10862983", "Q15296811", "Q1925963"]),
    "film_tv_production": ("film or television directing and production",
                           ["Q2526255", "Q3282637", "Q578109", "Q3455803", "Q222344", "Q7042855", "Q2059704",
                            "Q1053574"]),
    "military": ("the military", ["Q189290", "Q47064", "Q4991371", "Q1402561", "Q11545923"]),
    "religion": ("religion (clergy, theologian)",
                 ["Q250867", "Q42603", "Q1234713", "Q611644", "Q152002", "Q133485", "Q1423891", "Q15995642",
                  "Q49476"]),
    "business": ("business or finance", ["Q43845", "Q131524", "Q484876", "Q806798", "Q2961975", "Q557880", "Q1979607"]),
    "law": ("law (lawyer, judge, jurist)", ["Q40348", "Q16533", "Q185351"]),
    "medicine": ("medicine (physician, surgeon, nurse)", ["Q39631", "Q774306", "Q186360", "Q12013238", "Q211346"]),
}
# industry (P452) values are mapped by keywords on their English labels (first matching sector wins per value)
S34_INDUSTRY = [
    ("finance", "banking, finance or insurance",
     r"bank|financ|insur|invest|asset management|stock exchange|credit|payment|fintech|accounting"),
    ("media_entertainment", "media, publishing, entertainment or video games",
     r"media|publish|broadcast|television|radio|film|motion picture|entertainment|music|record label|video game|"
     r"newspaper|news|advertis|animation"),
    ("technology", "software, internet, computing, electronics or telecommunications",
     r"software|internet|computer|information technology|semiconductor|electronic|telecom|digital|data|"
     r"artificial intelligence|e-commerce|online"),
    ("transport_automotive", "automotive, aerospace, airlines, rail, shipping or logistics",
     r"automo|car manufactur|vehicle|aerospace|aviation|airline|aircraft|rail|transport|shipping|logistic|"
     r"freight|motorcycle|truck|bus"),
    ("energy_resources", "energy, oil and gas, mining or utilities",
     r"oil|petroleum|natural gas|energy|electric power|electricity|mining|coal|utility|renewable|nuclear|metal"),
    ("health_pharma", "pharmaceuticals, biotechnology or health care",
     r"pharma|biotech|health|medical|hospital|drug|life science"),
    ("food_beverage", "food, beverages or agriculture",
     r"food|beverage|brew|beer|wine|drink|agricult|dairy|confection|bakery|restaurant|coffee|spirits|tobacco"),
    ("retail_consumer", "retail, consumer goods, fashion or hospitality",
     r"retail|supermarket|fashion|clothing|apparel|consumer|cosmetic|luxury|hotel|hospitality|toy|furniture|"
     r"department store|footwear"),
    ("industrial", "manufacturing, construction, engineering, chemicals or defence",
     r"manufactur|construction|engineering|steel|chemical|machinery|industrial|defen[cs]e|arms|weapon|"
     r"real estate|building materials"),
]
S34_ERA = ["before 1800", "1800-1899", "1900-1945", "1946-1999", "2000-2019", "2020 or later"]
S34_TYPE_QUOTA_TRAIN = 1400
S34_EVAL_PER_STRATUM = 1100
S34_TRAIN_MAX = 20000
S34_EVAL_UNITS = 8000
S34_VAL_MAX = 1000
S34_GENERIC = set("""university college school academy institute club football association company corporation group
records band film album series season battle election station bridge island park street airport party society
museum hospital hotel castle tower building centre center hall house church cathedral river lake mount mountain
county city town village district province state national international royal united kingdom republic north south
east west new old great little upper lower saint lord king queen prince princess duke count baron emperor pope
first second third general war the and of for de la le del der van von los las san santa fort port bay sea ocean
valley desert glacier cape point beach hill hills range forest canyon falls creek road highway line railway limited
inc ltd plc ag gmbh records studios studio entertainment media network radio television news times daily
magazine journal review press books book novel poem song single airlines airways bank motors systems
technologies software games game team sport sports league cup championship tournament olympic olympics world
music orchestra choir ensemble quartet trio duo brothers sisters family people museum gallery library
school high primary secondary community federal central regional municipal american british english french
german italian spanish""".split())
S34_STOP = set("""a an the in on at is it as and or of to by for from with this that these those he she they his her its
was were be been has have had not but also after before during since until when where which who whom what there here
one two three new old all some many most other such only more less about over under between into through born died
no yes we you i my our your their them him us so if then than very can may will would could should must out up down
off per via de da di do du el la le les los las del der den des die das von van ten ter het een und et y e o a""".split())
S34_DERIV = r"(?:'s|’s|s|es|ian|ians|an|ans|ese|ish|ic|ite|ites|er|ers|ism|ist|ists|esque)?"
S34_CJK = _s34_re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")
S34_NONLATIN = _s34_re.compile(r"[Ͱ-ϿЀ-ԯ԰-֏֐-ۿ܀-ࣿऀ-෿"
                               r"฀-໿ༀ-࿿က-႟Ⴀ-ჿᄀ-ᇿሀ-፿"
                               r"぀-ヿ㐀-鿿가-힯]")
S34_IPA = _s34_re.compile(r"[ˈˌːəɪʊʃʒθðŋɛɔæɐʁɲʎʔɾɹɻʂʐɕʑɨʉɯɤʌɑɒøœɵɘɜɞʏ̩̯̃]")
S34_PAREN_DROP = _s34_re.compile(r"pronounc|listen|\blit\.|literally|romani[sz]ed|translit|transl\.|\bn[ée]e\b|"
                                 r"\b(?:[A-Z][a-z]+ ){0,2}[A-Z][a-z]+:\s", _s34_re.I)


def s34_strip_accents(s):
    return "".join(ch for ch in _s34_ud.normalize("NFKD", s) if not _s34_ud.combining(ch))


def s34_core_title(t):
    t = _s34_re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    return t


def s34_clean_lead(extract, max_chars=1500, min_chars=300):
    """The lead's first paragraph(s) as plain text: pronunciation / native-name parentheticals removed (they mostly hold
    the subject's names in other languages or scripts), truncated at a sentence end."""
    paras = [p.strip() for p in str(extract or "").split("\n") if p.strip()]
    out = ""
    for p in paras:
        if out and len(out) >= min_chars:
            break
        out = (out + "\n" + p) if out else p
    t = out
    t = _s34_re.sub(r"\s*\[[^\[\]]*\]", lambda m: "" if S34_IPA.search(m.group(0)) or S34_NONLATIN.search(m.group(0))
                    else m.group(0), t)
    for _ in range(4):                        # innermost parentheses first; nested ones get exposed on later passes
        t2 = _s34_re.sub(r"\s*\(([^()]*)\)", lambda m: "" if (S34_IPA.search(m.group(1)) or S34_NONLATIN.search(
            m.group(1)) or S34_PAREN_DROP.search(m.group(1)) or not m.group(1).strip(" ,;:")) else m.group(0), t)
        if t2 == t:
            break
        t = t2
    t = _s34_re.sub(r"[ \t]+", " ", t)
    t = _s34_re.sub(r" ([,.;:])", r"\1", t).strip()
    if len(t) > max_chars:
        cut = t[:max_chars]
        k = max(cut.rfind(". "), cut.rfind(".\n"))
        t = cut[:k + 1] if k > 200 else cut.rstrip() + " ..."
    return t


def s34_name_variants(names, title, label, person_parts=()):
    """Strings to mask: every label / alias / sitelink title (all languages) + the title and label without the
    disambiguator, their accent-stripped forms, and the distinctive tokens of multi-word names (a person's given and
    family names; 'Harvard' of 'Harvard University'). Returns (multiword_ci, single_cs, cjk)."""
    base = set()
    for n in list(names) + [title, label]:
        if not n:
            continue
        n = str(n).strip()
        for v in (n, s34_core_title(n), n.split(",")[0].strip(), s34_core_title(n).split(",")[0].strip()):
            if v and len(v) <= 120:
                base.add(v)
    toks = set()
    for v in [title, label, s34_core_title(title or ""), *list(person_parts)]:
        for tk in _s34_re.split(r"[\s\-–,/]+", str(v or "")):
            tk = tk.strip(".'\"()")
            if len(tk) >= 4 and tk[:1].isupper() and tk.lower() not in S34_GENERIC:
                toks.add(tk)
    for p in person_parts:
        if p and len(p) >= 3:
            toks.add(p)
    allv = set()
    for v in base | toks:
        allv.add(v)
        sa = s34_strip_accents(v)
        if sa != v:
            allv.add(sa)
    multi, single, cjk = set(), set(), set()
    own = {str(x).strip() for x in (title, label, s34_core_title(title or ""), s34_core_title(label or "")) if x}
    for v in allv:
        if not S34_CJK.search(v) and " " not in v.strip() and v not in own and (
                v.lower() in S34_STOP or v.lower() in S34_GENERIC or (len(v) <= 3 and not v.isupper())):
            continue
        if S34_CJK.search(v):
            if len(v) >= 2:
                cjk.add(v)
        elif " " in v.strip():
            if len(v) >= 4:
                multi.add(v)
        elif len(v) >= 2 and not (len(v) <= 3 and v.lower() == v):
            single.add(v)
    return multi, single, cjk


def s34_mask(text, variants):
    """Replace every variant with [ENTITY] (multi-word names case-insensitively, single tokens case-sensitively, CJK
    as substrings); derivative suffixes ('s, -ian, -ese, ...) go with the name. Returns (masked text, n replaced)."""
    multi, single, cjk = variants
    n = 0
    for group, flags in ((sorted(multi, key=len, reverse=True), _s34_re.I), (sorted(single, key=len, reverse=True), 0)):
        for i in range(0, len(group), 200):
            alt = "|".join(_s34_re.escape(v) for v in group[i:i + 200])
            if not alt:
                continue
            rx = _s34_re.compile(r"(?<![\w\[])(?:" + alt + r")(" + S34_DERIV + r")(?![\w\]])", flags)
            text, k = rx.subn(lambda m: "[ENTITY]" + ("'s" if m.group(1) in ("'s", "’s") else ""), text)
            n += k
    for v in sorted(cjk, key=len, reverse=True):
        k = text.count(v)
        if k:
            text = text.replace(v, "[ENTITY]")
            n += k
    text = _s34_re.sub(r"\[ENTITY\](?:[\s\-]*\[ENTITY\])+", "[ENTITY]", text)
    text = _s34_re.sub(r"\[ENTITY\]'s(?:[\s\-]*\[ENTITY\])+", "[ENTITY]", text)
    return text, n


S34_ALT = _s34_re.compile(
    r"(\[ENTITY\](?:'s)?[^.;\[\]]{0,40}?\b(?:also|formerly|originally|commonly|better|popularly|previously|"
    r"alternatively|sometimes|or|and)\s+(?:known|called|named|referred to|spelled|spelt|styled)\s+(?:as\s+)?"
    r"(?:the\s+)?)([\"\u201c']?[A-Z][^,.;()\[\]]{1,80})")


def s34_mask_alt(text):
    """Alternate names given inline right after the subject ('[ENTITY], also known as the Battle of Yashan, ...'): the
    span up to the next , . ; ( is masked. Only spans within 40 characters of an [ENTITY] mention count, so 'native to
    the Neotropics, also called ...' elsewhere in the lead is left alone. Returns (text, n masked)."""
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        return m.group(1) + "[ENTITY]"
    return S34_ALT.sub(sub, text), n


def s34_leak(masked, variants, title):
    """Residual leakage of the subject's names in a masked lead.
      word       a Latin-script name variant (>= 4 chars) still present as a whole word, case-insensitively
                 (a masking failure: such records are dropped)
      substring  a Latin-script name variant (>= 5 chars) embedded inside a longer word, case-insensitively
                 (compounds / derivatives, e.g. 'Einsteinium', 'Ikrampur'; reported, kept). A lowercase common word
                 that happens to be part of a name ('medical research' for 'Raine Medical Research Foundation') is
                 not counted: the case-sensitive single-token rule leaves those on purpose
      title      the core English title present as a substring, case-insensitively"""
    multi, single, _ = variants
    low = masked.lower()
    lowa = s34_strip_accents(low)
    lat_m = [v for v in multi if not S34_NONLATIN.search(v) and len(v) >= 4]
    lat_s = [v for v in single if not S34_NONLATIN.search(v) and len(v) >= 4 and v.lower() not in S34_GENERIC]
    word = False
    if lat_m:
        word = bool(_s34_re.search(r"(?<!\w)(?:" + "|".join(_s34_re.escape(v) for v in lat_m) + r")(?!\w)", masked,
                                   _s34_re.I))
    if not word and lat_s:      # a single-token name counts only where it is written like a name (capitalised)
        rx = _s34_re.compile(r"(?<!\w)(?:" + "|".join(_s34_re.escape(v) for v in lat_s) + r")(?!\w)", _s34_re.I)
        word = any(m.group(0)[:1].isupper() for m in rx.finditer(masked))
    sub = False                 # a variant embedded in a longer word: compounds / derivatives ("Ikrampur", "Einsteinium")
    for v in lat_m + lat_s:
        if len(v) < 5 or v.lower() in S34_STOP:
            continue
        for hay, needle in ((low, v.lower()), (lowa, s34_strip_accents(v.lower()))):
            k = hay.find(needle)
            while k != -1 and not sub:
                e = k + len(needle)
                if (k > 0 and hay[k - 1].isalpha()) or (e < len(hay) and hay[e].isalpha()):
                    sub = True
                k = hay.find(needle, k + 1)
        if sub:
            break
    ct = s34_core_title(title or "").lower()
    return {"word": bool(word), "substring": bool(sub), "title": bool(ct and len(ct) >= 3 and ct in low)}


def s34_era_bin(y):
    if y < 1800:
        return S34_ERA[0]
    if y < 1900:
        return S34_ERA[1]
    if y < 1946:
        return S34_ERA[2]
    if y < 2000:
        return S34_ERA[3]
    if y < 2020:
        return S34_ERA[4]
    return S34_ERA[5]


def s34_type_index():
    idx = {}
    for t, (_, cls) in S34_TYPES.items():
        for c in cls:
            idx[c] = t
    return idx


def s34_occ_index():
    idx = {}
    for g, (_, occ) in S34_OCC.items():
        for o in occ:
            idx.setdefault(o, g)
    return idx


def s34_industry_group(label):
    for key, _, rx in S34_INDUSTRY:
        if _s34_re.search(rx, label or "", _s34_re.I):
            return key
    return None


def s34_gold(d, industry_labels, rep):
    """Gold answers for one item from its Wikidata details: {'type': key, 'country': key, 'occupation': key,
    'industry': key, 'era': level} (absent = not asked). None when the type is unmapped or conflicting."""
    tix, oix = s34_type_index(), s34_occ_index()
    types = {tix[c] for c in d.get("P31", []) if c in tix}
    if not types:
        rep["type/unmapped_p31"] += 1
        return None
    if len(types) > 1:
        rep["type/conflicting_p31_types"] += 1
        return None
    g = {"type": types.pop()}
    cprop = "P27" if g["type"] == "person" else "P17"
    cs = set(d.get(cprop, []))
    inv = {q: k for k, (_, q) in S34_COUNTRIES.items()}
    if len(cs) == 1 and next(iter(cs)) in inv:
        g["country"] = inv[next(iter(cs))]
    elif cs:
        rep["country/not_single_or_outside_set"] += 1
    if g["type"] == "person":
        groups = {oix[o] for o in d.get("P106", []) if o in oix}
        if len(groups) == 1:
            g["occupation"] = groups.pop()
        elif len(groups) > 1:
            rep["occupation/conflicting_groups"] += 1
        else:
            rep["occupation/unmapped"] += 1
    if g["type"] == "company":
        secs = {s34_industry_group(industry_labels.get(v)) for v in d.get("P452", [])}
        secs.discard(None)
        if len(secs) == 1:
            g["industry"] = secs.pop()
        elif len(secs) > 1:
            rep["industry/conflicting_sectors"] += 1
    if g["type"] == "event":
        dates = d.get("dates") or {}
        for prop in ("P585", "P580"):
            ds = [x for x in dates.get(prop, []) if x[1] >= 9]
            if ds:
                bins = {s34_era_bin(s34_year(x[0])) for x in ds}
                if len(bins) == 1:
                    g["era"] = bins.pop()
                    g["era_year"] = s34_year(ds[0][0])
                else:
                    rep["era/conflicting_dates"] += 1
                break
    return g


def s34_lit(t):
    return '"' + str(t).replace("\\", "\\\\").replace('"', '\\"') + '"@en'


def s34_shard_leads(a, work):
    """{title: {pageid, title, lead}} for every article of the pinned shard: the first block of the article text (up to
    the first blank line, i.e. the lead), cached as compact jsonl.gz. The parquet is deleted afterwards unless
    --keep-shard. Needs pyarrow (run under `uv run --no-project --with pyarrow --with numpy python ...`)."""
    path = work / "classify_shard_leads.jsonl.gz"
    if path.exists() and not a.refetch:
        return {r["title"]: r for r in s34_rjsonl(path)}, None
    pq_path = Path(a.hf_dir) / Path(S34_HF_FILE).name
    if not pq_path.exists():
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{S34_HF_REPO}/resolve/{S34_HF_REV}/{S34_HF_FILE}"
        s34_log(f"downloading {url}")
        tmp = pq_path.with_name(pq_path.name + ".part")
        tmp.write_bytes(s34_http(url, accept="*/*", timeout=900))
        tmp.replace(pq_path)
    sha = _s34_hashlib.sha256(pq_path.read_bytes()).hexdigest()
    if sha != S34_HF_SHA256:
        raise SystemExit(f"{pq_path}: sha256 {sha} != pinned {S34_HF_SHA256}")
    import pyarrow.parquet as pq
    f = pq.ParquetFile(pq_path)
    rows = []
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=["id", "title", "text"])
        for pid, title, text in zip(t.column("id").to_pylist(), t.column("title").to_pylist(),
                                    t.column("text").to_pylist()):
            rows.append({"pageid": int(pid), "title": title, "lead": str(text or "").split("\n\n", 1)[0][:4000]})
    s34_wjsonl(path, rows)
    info = {"repo": S34_HF_REPO, "revision": S34_HF_REV, "file": S34_HF_FILE, "sha256": sha, "rows": len(rows),
            "pageid_range": [min(r["pageid"] for r in rows), max(r["pageid"] for r in rows)]}
    (work / "classify_shard_info.json").write_text(_s34_json.dumps(info, indent=1))
    if not a.keep_shard:
        pq_path.unlink()
        s34_log(f"deleted {pq_path} (compact leads kept in {path})")
    return {r["title"]: r for r in rows}, info


def s34_title_map(titles, work, a, name):
    """{title: {qid, sl, p31: [...]}} for English Wikipedia titles, via WDQS sitelink names (batches of 250)."""
    path = work / f"classify_titlemap_{name}.jsonl.gz"
    have = {r["title"]: r for r in s34_rjsonl(path)} if path.exists() and not a.refetch else {}
    need = [t for t in sorted(set(titles)) if t not in have]
    s34_log(f"title map {name}: {len(have)} cached, {len(need)} to map")
    for i in range(0, len(need), 250):
        batch = need[i:i + 250]
        rows = s34_sparql(f"""SELECT ?t ?item ?sl ?c WHERE {{ VALUES ?t {{ {" ".join(s34_lit(t) for t in batch)} }}
  ?article schema:name ?t ; schema:isPartOf <https://en.wikipedia.org/> ; schema:about ?item .
  ?item wikibase:sitelinks ?sl . OPTIONAL {{ ?item wdt:P31 ?c }} }}""")
        for t in batch:
            have[t] = {"title": t, "qid": None, "sl": 0, "p31": []}
        for r in rows:
            h = have[r["t"]]
            h["qid"], h["sl"] = r["item"], int(r["sl"])
            if r.get("c") and r["c"] not in h["p31"]:
                h["p31"].append(r["c"])
        if (i // 250) % 40 == 0:
            s34_log(f"  title map {name} {i + len(batch)}/{len(need)}")
            s34_wjsonl(path, list(have.values()))
    s34_wjsonl(path, list(have.values()))
    return have


def s34_fresh_titles(a, work):
    """Titles of English Wikipedia articles created on or after 2025-07-01, from creation-log windows spread over
    2025-07-01 .. 2026-09-14 (action API, at most one request per 10 s)."""
    path = work / "classify_fresh_log.jsonl.gz"
    if path.exists() and not a.refetch:
        return s34_rjsonl(path)
    d0, rows = _s34_dt(2025, 7, 1), {}
    for k in range(0, 440, 440 // a.fresh_windows):
        st = _s34_dt.fromordinal(d0.toordinal() + k).replace(hour=(k * 7) % 24).strftime("%Y-%m-%dT%H:00:00Z")
        js = s34_api(S34_WP_API, {"action": "query", "list": "logevents", "letype": "create", "lenamespace": 0,
                                  "lestart": st, "ledir": "newer", "lelimit": 500, "leprop": "title|ids|timestamp"})
        for e in js["query"]["logevents"]:
            if e.get("pageid") and int(e["pageid"]) >= S34_FRESH_PAGEID:
                rows[e["title"]] = {"title": e["title"], "pageid": int(e["pageid"]), "created": e["timestamp"]}
    out = list(rows.values())
    s34_wjsonl(path, out)
    s34_log(f"fresh: {len(out)} pages created on/after {S34_FRESH_FROM} from the creation log")
    return out


def s34_classify_details(qids, work, a, event_qids=()):
    """{qid: {P31: [...], P17, P27, P106, P452, dates: {P585: [[t, prec]], P580: ...}, names: [...], person_parts,
    label, sl}} via light SPARQL VALUES batches of 100 (core claims; name parts; dates for events only); cached
    incrementally. (One 200-item query with every branch timed out on the 2026 WDQS.)"""
    path = work / "classify_details.jsonl.gz"
    have = {d["qid"]: d for d in s34_rjsonl(path)} if path.exists() and not a.refetch else {}
    need = [q for q in sorted(set(qids)) if q not in have]
    ev = set(event_qids)
    s34_log(f"details: {len(have)} cached, {len(need)} to fetch")
    for i in range(0, len(need), 100):
        batch = need[i:i + 100]
        vals = " ".join("wd:" + q for q in batch)
        core = s34_sparql(f"""SELECT ?item ?p ?v WHERE {{ VALUES ?item {{ {vals} }}
  {{ ?item wdt:P31 ?v . BIND("P31" AS ?p) }} UNION {{ ?item wdt:P17 ?v . BIND("P17" AS ?p) }}
  UNION {{ ?item wdt:P27 ?v . BIND("P27" AS ?p) }} UNION {{ ?item wdt:P106 ?v . BIND("P106" AS ?p) }}
  UNION {{ ?item wdt:P452 ?v . BIND("P452" AS ?p) }} UNION {{ ?item wikibase:sitelinks ?v . BIND("sl" AS ?p) }} }}""")
        nm = s34_sparql(f"""SELECT ?item ?p ?v WHERE {{ VALUES ?item {{ {vals} }}
  {{ ?item wdt:P734|wdt:P735 ?x . ?x rdfs:label ?v . FILTER(LANG(?v) = "en") BIND("part" AS ?p) }}
  UNION {{ ?item wdt:P1448|wdt:P1705|wdt:P1477|wdt:P1813|wdt:P1449|wdt:P742|wdt:P2561 ?v . BIND("name" AS ?p) }}
  UNION {{ ?item rdfs:label ?v . FILTER(LANG(?v) = "en") BIND("label" AS ?p) }} }}""")
        evb = [q for q in batch if q in ev]
        dates = []
        if evb:
            # every pattern inside its UNION branch: a shared `?s wikibase:rank ?rank` outside the UNION lets the
            # planner start from all statements (the 504s of the first run)
            dates = s34_sparql(f"""SELECT ?item ?p ?t ?prec ?rank WHERE {{ VALUES ?item {{ {" ".join("wd:" + q for q in evb)} }}
  {{ ?item p:P585 ?s . ?s wikibase:rank ?rank ; psv:P585 ?tv . ?tv wikibase:timeValue ?t ; wikibase:timePrecision ?prec .
     BIND("P585" AS ?p) }}
  UNION
  {{ ?item p:P580 ?s . ?s wikibase:rank ?rank ; psv:P580 ?tv . ?tv wikibase:timeValue ?t ; wikibase:timePrecision ?prec .
     BIND("P580" AS ?p) }} }}""")
            dates = [r for r in dates if not str(r.get("rank", "")).endswith("DeprecatedRank")]
        for q in batch:
            have[q] = {"qid": q, "P31": [], "P17": [], "P27": [], "P106": [], "P452": [], "names": [],
                       "person_parts": [], "dates": {}, "label": None, "sl": 0}
        for r in core + nm:
            d, p, v = have[r["item"]], r["p"], r["v"]
            if p in ("P31", "P17", "P27", "P106", "P452"):
                if v not in d[p]:
                    d[p].append(v)
            elif p == "sl":
                d["sl"] = int(v)
            elif p == "label":
                d["label"] = v
            elif p == "part":
                if v not in d["person_parts"]:
                    d["person_parts"].append(v)
            elif p == "name" and v not in d["names"]:
                d["names"].append(v)
        for r in dates:
            have[r["item"]]["dates"].setdefault(r["p"], []).append([r["t"], int(r["prec"])])
        if (i // 100) % 20 == 0:
            s34_log(f"  details {i + len(batch)}/{len(need)}")
            s34_wjsonl(path, list(have.values()))
    s34_wjsonl(path, list(have.values()))
    return have


def s34_popular_candidates(a, work):
    """Random POPULAR items (>= 30 sitelinks) with an English article, for every type: bd:sample over the type's first
    three P31 classes, filtered on sitelinks, cached. Their leads come from the live API (eval only)."""
    path = work / "classify_popular.jsonl.gz"
    if path.exists() and not a.refetch:
        return s34_rjsonl(path)
    out, rng = {}, _s34_random.Random(f"{a.seed}:wd34popular")
    for t, (_, classes) in S34_TYPES.items():
        for cls in classes[:3]:
            lim = 300000 if cls in ("Q5", "Q16521") else 40000
            try:
                rows = s34_sparql(f"""SELECT ?item ?sl ?article WHERE {{
  SERVICE bd:sample {{ ?item wdt:P31 wd:{cls} . bd:serviceParam bd:sample.limit {lim} ; bd:sample.sampleType "RANDOM" . }}
  ?item wikibase:sitelinks ?sl . FILTER(?sl >= {S34_POPULAR})
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}""")
            except S34Refused:
                raise
            except Exception as e:  # noqa: BLE001
                s34_log(f"  popular {t}/{cls} failed ({str(e)[:100]}); skipped")
                continue
            rng.shuffle(rows)
            for r in rows[:250]:
                out.setdefault(r["item"], {"qid": r["item"], "title": s34_title_of(r["article"]), "sl": int(r["sl"]),
                                           "p31": [cls], "pageid": None, "source": "popular_live", "created": None})
            s34_log(f"popular {t}/{cls}: {len(rows)} rows")
    rows = list(out.values())
    s34_wjsonl(path, rows)
    return rows


def s34_title_of(url):
    return _s34_uparse.unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " ")


def s34_classify_names(qids, work, a):
    """{qid: [distinct labels, aliases (all languages) and sitelink titles (all wikis)]} via WDQS (batches of 80)."""
    path = work / "classify_names.jsonl.gz"
    have = {d["qid"]: d["names"] for d in s34_rjsonl(path)} if path.exists() and not a.refetch else {}
    need = [q for q in sorted(set(qids)) if q not in have]
    s34_log(f"names: {len(have)} cached, {len(need)} to fetch")
    for i in range(0, len(need), 50):
        batch = need[i:i + 50]
        rows = s34_sparql(f"""SELECT ?item ?n WHERE {{ VALUES ?item {{ {" ".join("wd:" + q for q in batch)} }}
  {{ ?item rdfs:label ?n }} UNION {{ ?item skos:altLabel ?n }} UNION {{ ?s schema:about ?item ; schema:name ?n }} }}""")
        acc = _s34_dd(set)
        for r in rows:
            v = r.get("n")
            if v and len(v) <= 120:
                acc[r["item"]].add(s34_core_title(v) if r.get("n") else v)
                acc[r["item"]].add(v)
        for q in batch:
            have[q] = sorted(acc.get(q, ()))
        if (i // 50) % 80 == 0:
            s34_log(f"  names {i + len(batch)}/{len(need)}")
            s34_wjsonl(path, [{"qid": k, "names": v} for k, v in have.items()])
    s34_wjsonl(path, [{"qid": k, "names": v} for k, v in have.items()])
    return have


def s34_classify_leads(titles, work, a):
    """{title: {title, pageid, revid, extract, disambiguation, qid}} via the TextExtracts API (20 intros per call)."""
    path = work / "classify_leads.jsonl.gz"
    have = {d["title"]: d for d in s34_rjsonl(path)} if path.exists() and not a.refetch else {}
    need = [t for t in sorted(set(titles)) if t not in have]
    s34_log(f"leads: {len(have)} cached, {len(need)} to fetch")
    for i in range(0, len(need), 20):
        batch = need[i:i + 20]
        js = s34_api(S34_WP_API, {"action": "query", "prop": "extracts|info|pageprops", "exintro": 1,
                                  "explaintext": 1, "exsectionformat": "plain", "exlimit": 20,
                                  "ppprop": "wikibase_item|disambiguation", "titles": "|".join(batch)})
        norm = {n["from"]: n["to"] for n in js["query"].get("normalized", [])}
        got = {}
        for p in js["query"].get("pages", []):
            pp = p.get("pageprops") or {}
            got[p["title"]] = {"title": p["title"], "pageid": p.get("pageid"), "revid": p.get("lastrevid"),
                               "extract": p.get("extract") or "", "disambiguation": "disambiguation" in pp,
                               "qid": pp.get("wikibase_item"), "missing": bool(p.get("missing"))}
        for t in batch:
            have[t] = got.get(norm.get(t, t)) or {"title": t, "missing": True, "extract": ""}
        if (i // 20) % 100 == 0:
            s34_log(f"  leads {i + len(batch)}/{len(need)}")
            s34_wjsonl(path, list(have.values()))
    s34_wjsonl(path, list(have.values()))
    return have


def s34_type_of(p31s):
    tix = s34_type_index()
    ts = {tix[c] for c in p31s if c in tix}
    return ts.pop() if len(ts) == 1 else ("conflict" if ts else None)


def s34_classify_select(pool, a, rep):
    """pool: [{qid, title, sl, p31, pageid, source: shard|fresh, created}] -> {split: [item + type + stratum]}. Split by
    QID hash (fresh -> eval only); eval stratified by popularity with types round-robin; train capped per type. An
    oversampling margin covers records lost later (no lead, masking, details)."""
    rng = _s34_random.Random(f"{a.seed}:wd34select")
    items, seen = [], set()
    for c in pool:
        if not c.get("qid") or c["qid"] in seen:
            rep["select/unmapped_title_or_duplicate_item"] += 1
            continue
        seen.add(c["qid"])
        t = s34_type_of(c.get("p31") or [])
        if t is None:
            rep["select/type_unmapped_p31"] += 1
            continue
        if t == "conflict":
            rep["select/type_conflicting_p31"] += 1
            continue
        st = "fresh" if c["source"] == "fresh" else s34_stratum(c["sl"])
        u = s34_u(a.seed, "wd34cls", c["qid"])
        live = c["source"] in ("fresh", "popular_live")       # live-API leads: eval only
        split = "eval" if live else ("eval" if u < 0.14 else "val" if u < 0.19 else "train")
        items.append(dict(c, type=t, stratum=st, split=split))
    rep["select/typed_items"] = len(items)
    rep["select/typed_by_stratum"] = dict(_S34Counter(f"{it['split']}/{it['stratum']}" for it in items))
    # the live popular add-on is appended after the shuffle, in hash order, so it never perturbs the other selections
    main = [it for it in items if it["source"] != "popular_live"]
    popl = sorted((it for it in items if it["source"] == "popular_live"), key=lambda it: s34_u(a.seed, "pop", it["qid"]))
    rng.shuffle(main)
    items = main + popl
    out = {"eval": [], "val": [], "train": []}
    by_st = _s34_dd(list)
    for it in items:
        if it["split"] == "eval":
            by_st[it["stratum"]].append(it)
    for st, lst in sorted(by_st.items()):
        by_t = _s34_dd(list)
        for it in lst:
            by_t[it["type"]].append(it)
        pick, want = [], int(S34_EVAL_PER_STRATUM * a.oversample)
        while len(pick) < want and any(by_t.values()):
            for t in sorted(by_t):
                if by_t[t] and len(pick) < want:
                    pick.append(by_t[t].pop())
        out["eval"] += pick
    for sp, cap in (("train", S34_TRAIN_MAX), ("val", S34_VAL_MAX)):
        per_t = _S34Counter()
        quota = S34_TYPE_QUOTA_TRAIN if sp == "train" else max(40, S34_TYPE_QUOTA_TRAIN * S34_VAL_MAX // S34_TRAIN_MAX)
        for it in items:
            if it["split"] != sp or per_t[it["type"]] >= quota * a.oversample:
                continue
            per_t[it["type"]] += 1
            out[sp].append(it)
        out[sp] = out[sp][:int(cap * a.oversample)]
    return out


def s34_classify_questions(g, mode, rng):
    """Questions + labels for an item's gold. mode: 'lead' (the masked lead is the state), 'name' (only the entity's
    English label is given), 'none' (state removed)."""
    pre = {"lead": "The state is the opening of an English Wikipedia article about one entity, with the entity's own "
                   "names masked as [ENTITY]. ",
           "name": "The state gives the English name of one entity, called [ENTITY] below. ",
           "none": "The state was removed: answer about an unknown entity, called [ENTITY] below. "}[mode]
    qs, labs = {}, {}

    def choice(qid, instr, opts, gold):
        keys = list(opts)
        rng.shuffle(keys)
        qs[qid] = {"type": "choice", "instructions": pre + instr, "criteria": {k: opts[k] for k in keys}}
        labs[qid] = s34_onehot(keys, gold)

    choice("type", "What kind of thing is [ENTITY]? (Gold: its Wikidata 'instance of' (P31) claims, grouped into "
                   "these categories.)", {k: v[0] for k, v in S34_TYPES.items()}, g["type"])
    if "country" in g:
        what = "country of citizenship (P27)" if g["type"] == "person" else "country (P17)"
        choice("country", f"Which country is [ENTITY] associated with, according to its Wikidata {what} claim? Exactly "
                          f"one of these countries is recorded for it.", {k: v[0] for k, v in S34_COUNTRIES.items()},
               g["country"])
    if "occupation" in g:
        choice("occupation", "Which occupation group does [ENTITY] belong to, according to its Wikidata 'occupation' "
                             "(P106) claims grouped into these categories?", {k: v[0] for k, v in S34_OCC.items()},
               g["occupation"])
    if "industry" in g:
        choice("industry", "Which sector is [ENTITY] in, according to its Wikidata 'industry' (P452) claims grouped "
                           "into these sectors?", {k: d for k, d, _ in S34_INDUSTRY}, g["industry"])
    if "era" in g:
        qs["era"] = {"type": "score", "instructions": pre + "When did [ENTITY] take place, according to its Wikidata "
                     "'point in time' (P585) or 'start time' (P580) claim?", "levels": list(S34_ERA)}
        labs["era"] = s34_onehot(list(S34_ERA), g["era"])
    return qs, labs


def s34_evidence(masked, g):
    """Is the gold label's own word in the masked lead? (an easiness measure, not leakage: classification from text
    is SUPPOSED to find its evidence there). Country: name or demonym; type: a keyword of the type."""
    low = masked.lower()
    out = {}
    if "country" in g:
        name = S34_COUNTRIES[g["country"]][0]
        words = [name] + S34_DEMONYMS[g["country"]]
        out["country"] = any(_s34_re.search(r"\b" + _s34_re.escape(w.lower()) + r"\b", low) for w in words)
    kw = {"person": r"\b(born|was an?|is an?)\b", "settlement": r"\b(city|town|village|municipality|commune|comune)\b",
          "admin_region": r"\b(country|state|province|county|region|district)\b", "company": r"\b(company|corporation)\b",
          "organization": r"\b(organi[sz]ation|party|association|agency|foundation|union)\b",
          "school": r"\b(school|college|university)\b", "sports_team": r"\b(club|team)\b", "film": r"\bfilm\b",
          "tv_series": r"\b(television|tv|series)\b", "music_release": r"\b(album|single|song)\b",
          "musical_group": r"\b(band|group|duo)\b", "written_work": r"\b(book|novel|poem|comic|magazine|newspaper|journal)\b",
          "software_game": r"\b(video game|software|programming language|game)\b",
          "building_structure": r"\b(building|bridge|station|church|tower|castle|stadium|road|dam)\b",
          "natural_feature": r"\b(mountain|river|lake|island|hill|bay|peninsula|glacier)\b",
          "taxon": r"\b(species|genus|family)\b", "event": r"\b(battle|war|election|earthquake|season|tournament)\b",
          "vehicle_vessel": r"\b(ship|aircraft|car|locomotive|submarine|destroyer)\b"}
    out["type"] = bool(_s34_re.search(kw[g["type"]], low))
    return out


def s34_cls_build(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rep = _S34Counter()
    snapshot = _s34_dt.now().strftime("%Y-%m-%d")
    shard, shard_info = s34_shard_leads(a, work)
    if shard_info is None and (work / "classify_shard_info.json").exists():
        shard_info = _s34_json.loads((work / "classify_shard_info.json").read_text())
    s34_log(f"shard: {len(shard)} articles")
    tmap = s34_title_map(list(shard), work, a, "shard")
    pool = [{**tmap[t], "pageid": shard[t]["pageid"], "source": "shard", "created": None} for t in shard if t in tmap]
    fresh_log = [] if a.no_fresh else s34_fresh_titles(a, work)
    if fresh_log:
        fmap = s34_title_map([r["title"] for r in fresh_log], work, a, "fresh")
        pool += [{**fmap[r["title"]], "pageid": r["pageid"], "source": "fresh", "created": r["created"]}
                 for r in fresh_log if r["title"] in fmap]
    rep["pool/shard_titles"] = len(shard)
    rep["pool/shard_titles_mapped_to_items"] = sum(1 for t in shard if (tmap.get(t) or {}).get("qid"))
    rep["pool/fresh_log_pages"] = len(fresh_log)
    if not a.no_popular:
        shard_q = {c["qid"] for c in pool if c.get("qid")}
        pop = [c for c in s34_popular_candidates(a, work) if c["qid"] not in shard_q]
        rep["pool/popular_live_candidates"] = len(pop)
        pool += pop
    sel = s34_classify_select(pool, a, rep)
    allsel = [it for v in sel.values() for it in v]
    s34_log("selected: " + ", ".join(f"{k} {len(v)}" for k, v in sel.items()))
    details = s34_classify_details([it["qid"] for it in allsel], work, a,
                                   event_qids={it["qid"] for it in allsel if it["type"] == "event"})
    ipath = work / "classify_industry_labels.json"
    ind = _s34_json.loads(ipath.read_text()) if ipath.exists() else {}
    need = sorted({v for d in details.values() for v in d.get("P452", [])} - set(ind))
    if need:
        ind.update({q: (v[0] or "") for q, v in s34_labels(need).items()})
        ipath.write_text(_s34_json.dumps(ind, ensure_ascii=False))
    names = s34_classify_names([it["qid"] for it in allsel], work, a)
    live_sel = [it for it in sel["eval"] if it["source"] in ("fresh", "popular_live")]
    live = s34_classify_leads([it["title"] for it in live_sel], work, a) if live_sel else {}
    out = {"train": [], "val": [], "eval": []}
    leak_before, evid, mask_frac = _S34Counter(), _s34_dd(_S34Counter), []
    from claude_overlap import norm as _norm
    seen_states = set()     # exact masked-state dedup across splits, eval first (templated stubs collide once masked)
    for sp in ("eval", "val", "train"):
        cap = {"eval": None, "val": S34_VAL_MAX, "train": S34_TRAIN_MAX}[sp]
        per_st = _S34Counter()
        for it in sel[sp]:
            d = details.get(it["qid"])
            if not d:
                rep["details/missing"] += 1
                continue
            g = s34_gold(d, ind, rep)
            if not g:
                continue
            if g["type"] != it["type"]:
                rep["details/type_changed_since_title_map"] += 1
                continue
            if it["source"] in ("fresh", "popular_live"):
                L = live.get(it["title"]) or {}
                if L.get("missing") or not L.get("extract") or L.get("disambiguation"):
                    rep[f"lead/{it['source']}_missing_or_disambiguation"] += 1
                    continue
                if L.get("qid") and L["qid"] != it["qid"]:
                    rep["lead/sitelink_qid_mismatch"] += 1
                    continue
                raw, title, pageid, revid = L["extract"], L["title"], L.get("pageid"), L.get("revid")
                lic = s34_lic_wp(title, revid, it["qid"])
                text_src = f"live English Wikipedia (action API extracts), revision {revid}"
            else:
                S = shard[it["title"]]
                raw, title, pageid, revid = S["lead"], S["title"], S["pageid"], None
                lic = s34_lic_dump(title, pageid, it["qid"])
                text_src = f"Wikipedia 2023-11-01 dump via {S34_HF_REPO} 20231101.en @ {S34_HF_REV[:8]}"
            if title.lower().startswith(("list of", "lists of", "index of")):
                rep["lead/list_article"] += 1
                continue
            if sp == "eval" and per_st[it["stratum"]] >= S34_EVAL_PER_STRATUM:
                rep[f"eval/over_stratum_quota_{it['stratum']}"] += 1
                continue
            lead = s34_clean_lead(raw)
            if len(lead) < 120:
                rep["lead/too_short"] += 1
                continue
            variants = s34_name_variants(names.get(it["qid"], []) + d.get("names", []), title, d.get("label"),
                                         d.get("person_parts", []) if g["type"] == "person" else ())
            masked, n = s34_mask(lead, variants)
            masked, n_alt = s34_mask_alt(masked)
            rep["mask/alternate_name_after_subject_masked"] += bool(n_alt)
            lk = s34_leak(masked, variants, title)
            for k, v in lk.items():
                leak_before[k] += v
            leak_before["n"] += 1
            leak_before["no_name_found"] += (n == 0)
            if lk["word"]:
                rep["mask/residual_whole_word_name_dropped"] += 1
                continue
            frac = masked.count("[ENTITY]") * 8 / max(1, len(masked))
            if frac > 0.25:
                rep["mask/too_much_masked"] += 1
                continue
            ns = _norm(masked)
            if ns in seen_states:
                rep[f"dedup/identical_masked_lead_{sp}"] += 1
                continue
            seen_states.add(ns)
            mask_frac.append(frac)
            rng = _s34_random.Random(f"{a.seed}:{it['qid']}")
            qs, labs = s34_classify_questions(g, "lead", rng)
            ev = s34_evidence(masked, g)
            for k, v in ev.items():
                evid[k]["present"] += v
                evid[k]["n"] += 1
            meta = {"qid": it["qid"], "title": title, "pageid": pageid, "revid": revid, "text_source": text_src,
                    "sitelinks": d.get("sl") or it["sl"], "stratum": it["stratum"], "gold_type": g["type"], "gold": g,
                    "article_created": it.get("created"), "split": sp,
                    "mask": {"n_replaced": n, "n_alternate_names": n_alt, "leak_substring": lk["substring"],
                             "leak_title": lk["title"],
                             "n_variants": sum(len(x) for x in variants)},
                    "evidence_in_lead": ev, "wikidata_snapshot": snapshot,
                    "label_provenance": "Wikidata truthy claims (P31 / P17 / P27 / P106 / P452 / P585 / P580) mapped "
                                        "to fixed taxonomies (build_wikidata_jev.py S34_*)"}
            if sp == "eval":
                meta.update(gde1_set="wd_classify", domain="K", panel="K (knowledge side panel; never in the V/P/O/X "
                            "headline)", decision_date=snapshot, info_date=snapshot,
                            time_slice=("fresh: article created on or after " + S34_FRESH_FROM) if it["stratum"] ==
                            "fresh" else ("none (current article of a popular entity, in pretraining data)"
                                          if it["source"] == "popular_live" else
                                          "none (articles of 2017-18, in pretraining data)"))
            rec_id = f"wd_classify-{it['qid']}"
            out[sp].append(s34_record("wd_classify/lead", rec_id, masked, qs, labs, meta,
                                      base_id=f"wd_classify/{it['qid']}", licence=lic))
            per_st[it["stratum"]] += 1
            if cap and len(out[sp]) >= cap:
                break
    # eval cap: <= S34_EVAL_UNITS units, removing seeded-random records from the largest stratum first
    erng = _s34_random.Random(f"{a.seed}:wd34evalcap")
    erng.shuffle(out["eval"])
    n_units = s34_units(out["eval"])
    while n_units > S34_EVAL_UNITS:
        cnt = _S34Counter(r["meta"]["stratum"] for r in out["eval"])
        big = cnt.most_common(1)[0][0]
        k = next(i for i, r in enumerate(out["eval"]) if r["meta"]["stratum"] == big)
        n_units -= len(out["eval"].pop(k)["questions"])
        rep["eval/dropped_for_unit_cap"] += 1
    out["eval"].sort(key=lambda r: r["id"])
    probes = {"eval_state_removed": [], "eval_name_only": []}
    for r in out["eval"]:
        g = r["meta"]["gold"]
        for mode, psp in (("none", "eval_state_removed"), ("name", "eval_name_only")):
            rng = _s34_random.Random(f"{a.seed}:{r['meta']['qid']}")
            qs, labs = s34_classify_questions(g, mode, rng)
            name = (details.get(r["meta"]["qid"]) or {}).get("label") or s34_core_title(r["meta"]["title"])
            st = S34_STATE_REMOVED if mode == "none" else f"Entity: {name}"
            probes[psp].append(s34_record(
                "wd_classify/lead", r["id"] + ("-sr" if mode == "none" else "-name"), st, qs, labs,
                dict(r["meta"], probe="state_removed" if mode == "none" else "name_only", probe_of=r["id"]),
                base_id=r["base_id"], licence=S34_LIC_WD))
    lb = {k: round(leak_before[k] / max(1, leak_before["n"]), 4) for k in ("word", "substring", "title", "no_name_found")}
    la = {}
    for sp, recs in out.items():
        if recs:
            la[sp] = {"leak_word": 0.0,
                      "leak_substring": round(sum(r["meta"]["mask"]["leak_substring"] for r in recs) / len(recs), 4),
                      "leak_title": round(sum(r["meta"]["mask"]["leak_title"] for r in recs) / len(recs), 4)}
    mask_frac.sort()
    man = {"source": "wd_classify", "shape": 4, "builder": "build_wikidata_jev.py wd-classify (shapes 3-4 section)",
           "built": _s34_dt.now().isoformat(timespec="seconds"), "wikidata_snapshot": snapshot, "seed": a.seed,
           "license": "per record: CC BY-SA (English Wikipedia lead; 2023-11 dump or live revision) + CC0-1.0 "
                      "(Wikidata labels); see license_evidence",
           "license_evidence": {**S34_LICENSE_EVIDENCE, "hf_dump": S34_LIC_DUMP_EVIDENCE},
           "text_sources": {"shard": shard_info, "fresh": "live action API extracts (exintro, explaintext), at most one "
                            "request per 10 s (Wikimedia API gateway limit for clients without contact info)"},
           "panel": "eval = K side panel (classification from text; models saw Wikipedia, see the name-only probe); "
                    "train / val = general classification practice",
           "taxonomies": {"type": {k: {"text": v[0], "p31_classes": v[1]} for k, v in S34_TYPES.items()},
                          "country": {k: {"text": v[0], "qid": v[1]} for k, v in S34_COUNTRIES.items()},
                          "occupation": {k: {"text": v[0], "p106_values": v[1]} for k, v in S34_OCC.items()},
                          "industry": {k: {"text": d, "p452_label_regex": rx} for k, d, rx in S34_INDUSTRY},
                          "era": S34_ERA},
           "rules": {"items": "every article of one pinned Hugging Face shard (wikimedia/wikipedia 20231101.en, page ids "
                              "55.6M-57.4M: articles created about 2017-10 .. 2018-04), mapped to Wikidata by sitelink "
                              "title on WDQS; plus English articles created on or after 2025-07-01 (creation-log "
                              "windows) as the FRESH eval stratum",
                     "type": "one-hot; an item whose truthy P31 values map to two types is skipped; unmapped P31 values "
                             "beside a mapped one are allowed",
                     "country": "asked only when the item has exactly one truthy P27 (people) / P17 (others) value and "
                                "it is in the fixed set of 16",
                     "occupation": "people only; asked when every mapped P106 value falls in ONE group (closed-world: "
                                   "absence of a claim is not read as a 'no')",
                     "industry": "companies only; P452 values mapped by keyword on their English labels; one sector",
                     "era": "events only; P585 else P580 at year precision or better, one era bin",
                     "split": "QID hash: eval 14% / val 5% / train 81%; fresh articles are eval only",
                     "eval_strata": f"popular (>= {S34_POPULAR} sitelinks), mid ({S34_MID}-{S34_POPULAR - 1}), obscure "
                                   f"(<= {S34_MID - 1}), fresh; up to {S34_EVAL_PER_STRATUM} records each, types "
                                   "round-robin within a stratum",
                     "masking": "the subject's labels and aliases in every language and its sitelink titles in every "
                                "wiki (WDQS), official / native / birth / short names, nicknames, pseudonyms (P1448, "
                                "P1705, P1477, P1813, P1449, P742, P2561), the title without its disambiguator, given "
                                "and family names of people, distinctive tokens of multi-word names, accent-stripped "
                                "forms; derivative suffixes go with the name ('s is kept); pronunciation and "
                                "other-language / other-script parentheticals are removed first",
                     "probes": "eval_state_removed/ (state removed) and eval_name_only/ (state = the English label "
                               "only): the gap between name-only and lead accuracy measures how much is recall"},
           "leakage": {"before_drop": lb, "after_drop": la,
                       "definition": "word = a Latin-script name variant (>= 4 chars) still present as a capitalised "
                                     "whole word (records dropped); substring = a variant (>= 5 chars) embedded in a "
                                     "longer word, e.g. compounds and derivatives (kept); title = the core English title "
                                     "as a substring (kept); no_name_found = nothing was masked (the lead never names "
                                     "the subject)",
                       "masked_char_fraction": {"median": mask_frac[len(mask_frac) // 2] if mask_frac else None,
                                                "p95": mask_frac[int(len(mask_frac) * 0.95)] if mask_frac else None}},
           "gold_label_word_in_lead": {k: round(v["present"] / max(1, v["n"]), 4) for k, v in evid.items()},
           "build_report": dict(rep),
           "splits": {sp: s34_label_stats(v) for sp, v in {**out, **probes}.items()}}
    s34_write(a.out, "wd_classify", {**out, **probes}, man)
    return 0


# =========================================================================== dedup gate and check
S34_TEMPLATE = _s34_re.compile(r"^(Two|Four) entities from Wikidata .*|^Facts from Wikidata .*$")


def s34_content(rec):
    """The record's own text for the gate: wd_compare states without our template lines; wd_classify masked leads."""
    return "\n".join(ln for ln in rec["state"].split("\n") if ln.strip() and not S34_TEMPLATE.match(ln))


def s34_read_split(root, sp):
    recs = []
    for f in sorted((Path(root) / sp).glob("*.jsonl")):
        recs += [_s34_json.loads(line) for line in f.open(encoding="utf-8")]
    return recs


def s34_rewrite(root, sp, keep_ids):
    for f in sorted((Path(root) / sp).glob("*.jsonl")):
        rows = [line for line in f.open(encoding="utf-8") if _s34_json.loads(line)["id"] in keep_ids]
        tmp = f.with_name(f.name + ".part")
        tmp.write_text("".join(rows), encoding="utf-8")
        tmp.replace(f)


def s34_jevbench_states(work):
    """jev-bench v0.1.1 test states (all 22 configs; BoolQ / FEVER / StrategyQA hold Wikipedia text), cached as a compact
    jsonl.gz of states only."""
    path = Path(work) / "fp_jevbench_test_states.jsonl.gz"
    if path.exists():
        return [r["s"] for r in s34_rjsonl(path)]
    import build_jev_ext2 as x2
    cx = x2.cx                                   # trainers/convert_extevals.py: the jev-bench pins
    rows = []
    for cfg in x2.JEVBENCH_CONFIGS:
        url = f"https://huggingface.co/datasets/{cx.JEVBENCH}/resolve/{cx.JEVBENCH_V011}/data/{cfg}/test.jsonl"
        data = s34_http(url, accept="*/*")
        for line in data.decode("utf-8").split("\n"):        # not splitlines(): JSON strings may hold U+2028
            if line.strip():
                st = _s34_json.loads(line).get("state")
                if st is not None:
                    rows.append({"s": st, "t": cfg})
    s34_wjsonl(path, rows)
    return [r["s"] for r in rows]


def s34_gate(a):
    """(1) train / val records vs EVAL fingerprints (build_jev_ext2.Index: exact, >= 5-word leaf, >= 3 shared 8-word
    shingles): jev-bench v0.1.1 test, every local GDE-1 set (/tmp/gde1/*), and this builder's own eval states;
    (2) eval records vs the TRAINING-corpus proxy (build_gde1_a.Gate over /tmp/gde1_a_work/fp: the upstream sources of
    jev_data, jev_ext_strict, jev_ext2, jev_ext3). Drops in place and records the run in each manifest."""
    import build_jev_ext2 as x2
    sources = [s for s in a.sources.split(",") if s]
    res = {}
    # ---- (1) train/val vs eval fingerprints
    idx = x2.Index()
    sizes = {}
    t_jb = idx.tag("eval:jevbench_v011_test", "eval")
    sizes["jevbench_v011_test"] = sum(bool(idx.add(t_jb, s)) for s in s34_jevbench_states(a.work))
    for d in sorted(Path(a.gde1).glob("*/")):
        tg = idx.tag(f"eval:gde1/{d.name}", "eval")
        n = 0
        for f in sorted(d.glob("*/*.jsonl")):
            for line in f.open(encoding="utf-8"):
                n += bool(idx.add(tg, _s34_json.loads(line)["state"]))
        sizes[f"gde1/{d.name}"] = n
    idx.freeze()
    # this builder's own eval states: exact / leaf only. Entities are split, so a shingle overlap between a train and an
    # eval record is a shared template (Wikidata descriptions such as "lake in Mason County, Washington", stub leads such
    # as "... is a village in the administrative district of Gmina ..."), not a shared item (dry run: 931 of 12,096
    # wd_compare train records would have gone on templated descriptions alone).
    own = x2.Index()
    own_eval = {}
    for src in sources:
        tg = own.tag(f"eval:{src}", "eval")
        recs = s34_read_split(Path(a.out) / src, "eval")
        own_eval[src] = recs
        sizes[f"own/{src}/eval (exact + leaf only)"] = sum(bool(own.add(tg, s34_content(r))) for r in recs)
    own.freeze()
    for src in sources:
        root = Path(a.out) / src
        mpath = root / "manifest.json"
        man = _s34_json.loads(mpath.read_text())
        run = {"when": _s34_dt.now().isoformat(timespec="seconds"), "fingerprints": sizes, "by_split": {}}
        for sp in ("train", "val"):
            recs = s34_read_split(root, sp)
            keep, drops, ex = set(), _S34Counter(), []
            for r in recs:
                c = s34_content(r)
                probes = [ln for ln in c.split("\n") if len(ln.split()) >= 5]
                hit = idx.check(c, probes)
                if not hit:
                    h2 = own.check(c, probes)
                    hit = h2 if h2 and h2[0] in ("exact", "leaf") else None
                if hit:
                    drops[f"{hit[0]}:{hit[1]}"] += 1
                    if len(ex) < 5:
                        ex.append({"id": r["id"], "kind": hit[0], "tag": hit[1], "n": hit[2], "sample": hit[3]})
                    continue
                keep.add(r["id"])
            if not a.dry_run:
                s34_rewrite(root, sp, keep)
            run["by_split"][sp] = {"before": len(recs), "after": len(keep), "dropped": dict(drops), "examples": ex}
        man["dedup_train_vs_eval"] = run
        mpath.write_text(_s34_json.dumps(man, indent=1, ensure_ascii=False, default=str))
        res[src] = {"train_vs_eval": run["by_split"]}
    # ---- (2) eval vs the training-corpus proxy
    if a.proxy and Path(a.proxy, "fp").exists():
        import build_gde1_a as ga
        items, owner = [], {}
        for src in sources:
            for r in own_eval[src]:
                c = s34_content(r)
                items.append({"key": r["id"], "state": c, "qs": {q: {"split": "eval"} for q in r["questions"]},
                              "fields": [ln for ln in c.split("\n") if ln.strip()]})
                owner[r["id"]] = src
        hits, read = ga.gate_items(items, lambda: ga.proxy_stream(a.proxy))
        drop = {k for k, h in hits.items() if ga.dropping(h)}
        for src in sources:
            root = Path(a.out) / src
            mpath = root / "manifest.json"
            man = _s34_json.loads(mpath.read_text())
            mine = [k for k in drop if owner.get(k) == src]
            ex = [{"id": k, "hits": [list(map(str, h))[:3] for h in hits[k][:2]]} for k in mine[:5]]
            reported = sum(1 for k in hits if owner.get(k) == src)
            keep = {r["id"] for r in own_eval[src]} - drop
            if not a.dry_run:
                s34_rewrite(root, "eval", keep)
                for psp in [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("eval_")]:
                    s34_rewrite(root, psp, {i for i in (r["id"] for r in s34_read_split(root, psp))
                                            if i.rsplit("-", 1)[0] in keep})
            man["dedup_eval_vs_training"] = {
                "status": "PROXY ONLY: raw upstream copies of the training sources (build_gde1_a.py fingerprints in "
                          f"{a.proxy}/fp); the authoritative run against the built corpora is pending on a node",
                "rule": "build_gde1_a.Gate defaults (exact; leaf; calibrated 8-word shingles)",
                "training_states_read": sum(read.values()), "items": sum(1 for k in owner.values() if k == src),
                "dropped": len(mine), "reported_not_dropped": reported - len(mine), "examples": ex,
                "when": _s34_dt.now().isoformat(timespec="seconds")}
            mpath.write_text(_s34_json.dumps(man, indent=1, ensure_ascii=False, default=str))
            res.setdefault(src, {})["eval_vs_training_proxy"] = {"dropped": len(mine), "reported": reported}
    else:
        res["eval_vs_training_proxy"] = f"SKIPPED: no proxy at {a.proxy}/fp"
        for src in sources:
            mpath = Path(a.out) / src / "manifest.json"
            man = _s34_json.loads(mpath.read_text())
            man["dedup_eval_vs_training"] = {
                "status": f"NOT RUN: no training-corpus proxy at {a.proxy}/fp (build_gde1_a.py fingerprints; the copy "
                          "used during development was deleted from /tmp by another process on 2026-10-01). Run on a "
                          "node against the built corpora: .venv/bin/python build_wikidata_jev.py wd34-gate "
                          f"--sources {src} --proxy <dir holding fp/ from build_gde1_a.py fingerprints>",
                "when": _s34_dt.now().isoformat(timespec="seconds")}
            if not a.dry_run:
                mpath.write_text(_s34_json.dumps(man, indent=1, ensure_ascii=False, default=str))
    print(_s34_json.dumps(res, indent=1, default=str))
    return 0


def s34_check(a):
    """Schema + iter_units read-back of every split, label / stratum stats, leakage scan, samples.json."""
    from build_jev_data import option_keys, validate
    from build_gde1_b import load_cf
    cf = load_cf()
    allrep = {}
    for src in [s for s in a.sources.split(",") if s]:
        root = Path(a.out) / src
        rep = {"source": src, "splits": {}}
        bad = _S34Counter()
        samples = []
        for spd in sorted(p for p in root.iterdir() if p.is_dir()):
            recs = s34_read_split(root, spd.name)
            ids = [r["id"] for r in recs]
            for r in recs:
                validate(r)
                if not r.get("license"):
                    bad["missing_license"] += 1
                if spd.name == "eval" and (r["meta"].get("domain") != "K" or not r["meta"].get("decision_date")):
                    bad["eval_without_K_meta"] += 1
                for qid, q in r["questions"].items():
                    keys, texts, tgt = cf.question_options(q, r["labels"][qid])
                    assert keys == option_keys(q) and abs(sum(tgt) - 1) < 1e-6
                if src == "wd_classify" and spd.name in ("train", "val", "eval"):
                    t = r["meta"]["title"]
                    if s34_core_title(t).lower() in r["state"].lower() and len(s34_core_title(t)) >= 4:
                        rep.setdefault("title_substring_in_masked_state", _S34Counter())[spd.name] += 1
            units = _S34Counter()
            lens = []
            for u in cf.iter_units(str(root), [spd.name], set(), None):
                units[f"{u['source']}/{u['qid']}/{u['qtype']}/K={len(u['option_keys'])}"] += 1
                lens.append(len(u["state"]))
            lens.sort()
            rep["splits"][spd.name] = {"records": len(recs), "units": sum(units.values()), "iter_units": dict(units),
                                       "duplicate_ids": len(ids) - len(set(ids)),
                                       "state_chars": {"median": lens[len(lens) // 2], "max": lens[-1]} if lens else None,
                                       **{k: v for k, v in s34_label_stats(recs).items() if k not in ("records", "units")}}
            if recs and spd.name in ("train", "eval", "eval_state_removed", "eval_name_only"):
                r = recs[len(recs) // 3]
                samples.append({**r, "state": r["state"][:1200]})
        ev = {r["id"] for r in s34_read_split(root, "eval")}
        tr = s34_read_split(root, "train")
        if src == "wd_compare":
            evq = {q for r in s34_read_split(root, "eval") for q in r["meta"]["qids"]}
            rep["train_eval_shared_entities"] = sum(1 for r in tr if set(r["meta"]["qids"]) & evq)
        else:
            evq = {r["meta"]["qid"] for r in s34_read_split(root, "eval")}
            rep["train_eval_shared_entities"] = sum(1 for r in tr if r["meta"]["qid"] in evq)
        rep["eval_ids"] = len(ev)
        rep["schema_problems"] = dict(bad) or "none"
        (root / "check_report.json").write_text(_s34_json.dumps(rep, indent=1, default=str))
        (root / "samples.json").write_text(_s34_json.dumps(samples, indent=1, ensure_ascii=False))
        allrep[src] = rep
    print(_s34_json.dumps({s: {k: v for k, v in r.items() if k != "splits"} | {
        "splits": {sp: {"records": x["records"], "units": x["units"]} for sp, x in r["splits"].items()}}
        for s, r in allrep.items()}, indent=1, default=str))
    return 0


# =========================================================================== self-test (offline fixtures)
def s34_self_test():
    import tempfile
    from build_gde1_b import load_cf
    # --- shape 3: item values, conflicts, pairing, records
    rep = _S34Counter()
    st = lambda sid, rank="n", ref=1, **kw: {"id": sid, "rank": rank, "ref": ref, **kw}  # noqa: E731
    mk = lambda q, sl, stmts, cls="Q3918", cty=(): {"qid": q, "cls": cls, "sl": sl, "stmts": stmts, "cty": list(cty)}  # noqa: E731
    it_ok = mk("Q1", 50, [st("s1", t="1636-01-01T00:00:00Z", prec=9, cal="Q1985727")])
    it_conf = mk("Q2", 50, [st("s2", t="1700-01-01T00:00:00Z", prec=9), st("s3", t="1701-01-01T00:00:00Z", prec=9)])
    it_pref = mk("Q3", 50, [st("s4", rank="p", t="1701-01-01T00:00:00Z", prec=9), st("s5", t="1650-01-01T00:00:00Z", prec=9)])
    it_noref = mk("Q4", 50, [st("s6", ref=0, t="1800-01-01T00:00:00Z", prec=9)])
    it_dec = mk("Q5", 50, [st("s7", t="1900-01-01T00:00:00Z", prec=8)])
    assert s34_item_value("inception", it_ok, rep)["year"] == 1636
    assert s34_item_value("inception", it_conf, rep) is None and rep["item/conflicting_statements"] == 1
    assert s34_item_value("inception", it_pref, rep)["year"] == 1701, "preferred rank wins over normal"
    assert s34_item_value("inception", it_noref, rep) is None and rep["item/unreferenced"] == 1
    assert s34_item_value("inception", it_dec, rep) is None and rep["item/precision_below_year"] == 1
    assert s34_year("-0500-01-01T00:00:00Z") == -500 and s34_fmt_date("-0500-01-01T00:00:00Z", 9) == "500 BCE"
    q_ok = mk("Q10", 40, [st("a", amt=14505, unit="Q3710", namt=4421.124), st("b", amt=4421, unit="Q11573", namt=4421.0)], cls="Q8502")
    q_conf = mk("Q11", 40, [st("c", amt=100, unit="Q11573", namt=100.0), st("d", amt=120, unit="Q11573", namt=120.0)], cls="Q8502")
    assert abs(s34_item_value("elevation", q_ok, rep)["v"] - 4421.124) < 1e-6
    assert s34_item_value("elevation", q_conf, rep) is None
    p_ok = mk("Q20", 5, [st("e", amt=1000, pit="2020-01-01", pprec=9), st("f", amt=1002, pit="2020-01-01", pprec=9),
                         st("g", amt=900, pit="2010-01-01", pprec=9, ref=0)], cls="Q515", cty=["Q142"])
    pv = s34_item_value("population", p_ok, rep)
    assert set(pv) == {"2020-01-01|9"} and rep["pop_date/unreferenced"] == 1, pv
    p_conf = mk("Q21", 5, [st("h", amt=1000, pit="2020-01-01", pprec=9), st("i", amt=1100, pit="2020-01-01", pprec=9)], cls="Q515")
    assert s34_item_value("population", p_conf, rep) is None and rep["pop_date/conflicting_values"] == 1
    # pairing: near ties and same years are skipped; gold is the earlier / larger one
    labels = {f"Q{i}": [f"Entity {i}", "a thing" if i % 2 else "university founded in 1900"] for i in range(100, 140)}
    labels.update({"Q142": ["France", ""], "Q183": ["Germany", ""], "Q38": ["Italy", ""], "Q29": ["Spain", ""],
                   "Q3710": ["foot", ""], "Q11573": ["metre", ""]})
    items = {}
    for i, y in zip(range(100, 108), [1800, 1800, 1801, 1850, 1900, 1950, 1951, 1990]):
        it = mk(f"Q{i}", 3, [st(f"t{i}", t=f"{y}-01-01T00:00:00Z", prec=9)])
        items[it["qid"]] = dict(it, val=s34_item_value("inception", it, rep), split="eval", stratum="obscure")
    rng = _s34_random.Random(0)
    fr = _S34Counter()
    pairs = s34_cmp_pairs("inception", items, labels, {}, rng, fr)
    assert pairs and all(p["ents"][0]["val"]["year"] != p["ents"][1]["val"]["year"] for p in pairs)
    for p in pairs:
        ys = {e["qid"]: e["val"]["year"] for e in p["ents"]}
        assert ys[p["gold"]] == min(ys.values())
    use = _S34Counter(e["qid"] for p in pairs for e in p["ents"])
    assert max(use.values()) <= 2
    r = s34_cmp_record("inception", pairs[0], labels, "2026-10-01", open_book=False, rng=rng)
    assert r["meta"]["domain"] == "K" and "founded in" not in r["state"], r["state"]
    assert r["labels"]["founded_earlier"][pairs[0]["gold"]] == 1.0 and "inception" not in r["state"].split("\n")[-1]
    ob = s34_cmp_record("inception", pairs[0], labels, "2026-10-01", open_book=True, rng=rng)
    assert "Facts from Wikidata" in ob["state"] and "domain" not in ob["meta"]
    sr = s34_state_removed(r)
    assert sr["state"] == S34_STATE_REMOVED and sr["labels"] == r["labels"] and sr["id"] != r["id"]
    # 4-way country with distinct-country distractors
    states = {"Q142": ["Q46"], "Q183": ["Q46"], "Q38": ["Q46"], "Q29": ["Q46"]}
    citems = {}
    for i, c in zip(range(120, 128), ["Q142", "Q142", "Q183", "Q38", "Q29", "Q183", "Q38", "Q29"]):
        it = mk(f"Q{i}", 3, [st(f"c{i}", c=c)], cls="Q515")
        citems[it["qid"]] = dict(it, val=s34_item_value("country", it, rep), split="train", stratum="obscure")
    cp = s34_cmp_pairs("country", citems, labels, states, _s34_random.Random(1), _S34Counter())
    assert cp and all(len({e["val"]["c"] for e in p["ents"]}) == 4 for p in cp)
    cr = s34_cmp_record("country", cp[0], labels, "2026-10-01", open_book=True, rng=rng)
    assert sum(cr["labels"]["which_in_country"].values()) == 1 and len(cr["questions"]["which_in_country"]["criteria"]) == 4
    # quantity in raw units: open-book shows feet vs metres, gold is on the normalized value
    qitems = {}
    for i, (amt, unit, namt) in zip(range(130, 134), [(14505, "Q3710", 4421.1), (4000, "Q11573", 4000.0),
                                                     (3000, "Q11573", 3000.0), (9000, "Q3710", 2743.2)]):
        it = mk(f"Q{i}", 3, [st(f"q{i}", amt=amt, unit=unit, namt=namt)], cls="Q8502")
        qitems[it["qid"]] = dict(it, val=s34_item_value("elevation", it, rep), split="train", stratum="obscure")
    qp = s34_cmp_pairs("elevation", qitems, labels, {}, _s34_random.Random(2), _S34Counter())
    for p in qp:
        vs = {e["qid"]: e["val"]["v"] for e in p["ents"]}
        assert vs[p["gold"]] == max(vs.values())
    qr = s34_cmp_record("elevation", qp[0], labels, "2026-10-01", open_book=True, rng=rng)
    assert "foot" in qr["state"] or "metre" in qr["state"]
    # population pair record (noul)
    pitems = {}
    for i, v in zip(range(134, 138), [1000, 1010, 2000, 5000]):
        it = mk(f"Q{i}", 3, [st(f"p{i}", amt=v, pit="2021-01-01", pprec=9)], cls="Q515", cty=["Q142"])
        pitems[it["qid"]] = dict(it, val=s34_item_value("population", it, rep), split="eval", stratum="obscure")
    pp = s34_cmp_pairs("population", pitems, labels, {}, _s34_random.Random(3), _S34Counter())
    assert pp and all(p["ratio"] >= 1 + S34_NEAR_TIE for p in pp)
    pr = s34_cmp_record("population", pp[0], labels, "2026-10-01", open_book=False, rng=rng)
    assert "the year 2021" in pr["questions"]["population_larger"]["instructions"]
    assert set(pr["questions"]["population_larger"]["criteria"]) == {"yes", "no"}
    print("ok  shape 3: values, conflicts, ranks, references, near ties, pairing, records, probe")

    # --- shape 4: gold mapping, cleaning, masking, leakage
    rep = _S34Counter()
    g = s34_gold({"P31": ["Q5"], "P27": ["Q183"], "P106": ["Q169470", "Q1622272"]}, {}, rep)
    assert g == {"type": "person", "country": "de", "occupation": "science_academia"}, g
    assert s34_gold({"P31": ["Q5"], "P27": ["Q183", "Q30"], "P106": ["Q33999", "Q177220"]}, {}, rep) == {"type": "person"}
    assert rep["occupation/conflicting_groups"] == 1 and rep["country/not_single_or_outside_set"] == 1
    assert s34_gold({"P31": ["Q11424", "Q5"]}, {}, rep) is None and rep["type/conflicting_p31_types"] == 1
    assert s34_gold({"P31": ["Q999999"]}, {}, rep) is None
    gi = s34_gold({"P31": ["Q4830453"], "P17": ["Q30"], "P452": ["Q1", "Q2"]}, {"Q1": "banking", "Q2": "investment banking"}, rep)
    assert gi["industry"] == "finance" and gi["country"] == "us"
    ge = s34_gold({"P31": ["Q178561"], "dates": {"P585": [["1757-06-23T00:00:00Z", 11]]}}, {}, rep)
    assert ge["era"] == "before 1800" and ge["type"] == "event"
    lead = ("Munich (German: München [ˈmʏnçn̩] , Bavarian: Minga [ˈmɪŋ(ː)ɐ] ) is the capital and most populous city "
            "of the German state of Bavaria. Munich's population was 1.5 million. Muenchen and the Münchner area; "
            "the Munich-based firm grew. Parisian visitors and the parish church.")
    clean = s34_clean_lead(lead)
    assert "German:" not in clean and "ˈ" not in clean and clean.startswith("Munich is the capital"), clean
    v = s34_name_variants(["München", "Muenchen", "Minga", "Munich"], "Munich", "Munich")
    m, n = s34_mask(clean, v)
    assert "Munich" not in m and "Muenchen" not in m and "[ENTITY] is the capital" in m and n >= 4, m
    assert "Münchner" in m, "an unknown derivative is not masked by the rules"
    lk = s34_leak(m, v, "Munich")
    assert not lk["word"] and lk["substring"] is False and not lk["title"]
    vp = s34_name_variants(["Albert Einstein"], "Albert Einstein", "Albert Einstein", person_parts=["Einstein", "Albert"])
    mp, _ = s34_mask("Albert Einstein was a physicist. Einstein's theory; Einsteinian ideas. Einsteinium is named "
                     "after him. Eduard Einstein was his son.", vp)
    assert "Einstein " not in mp and "[ENTITY]'s theory" in mp and mp.count("[ENTITY]") >= 4, mp
    assert s34_leak(mp, vp, "Albert Einstein")["substring"], "Einsteinium is a substring leak (reported, kept)"
    vt = s34_name_variants(["Apple Inc.", "Apple"], "Apple Inc.", "Apple Inc.")
    mt, _ = s34_mask("Apple Inc. is a company; an apple a day. APPLE was founded.", vt)
    assert "an apple a day" in mt and "[ENTITY] is a company" in mt, mt
    vc = s34_name_variants(["東京都", "Tokyo"], "Tokyo", "Tokyo")
    mc, _ = s34_mask("Tokyo, officially the Tokyo Metropolis, 東京都庁 is here.", vc)
    assert "東京都" not in mc and "Tokyo" not in mc, mc
    assert s34_core_title("Mercury (planet)") == "Mercury"
    ma, na = s34_mask_alt("The [ENTITY], also known as the Battle of Yashan, was fought in 1279. It is native to the "
                          "Neotropics, also called the New World tropics.")
    assert na == 1 and "Yashan" not in ma and "also called the New World tropics" in ma, ma
    rng = _s34_random.Random(0)
    qs, labs = s34_classify_questions(ge, "lead", rng)
    assert qs["era"]["type"] == "score" and labs["era"]["before 1800"] == 1.0 and len(qs["type"]["criteria"]) == len(S34_TYPES)
    qs2, labs2 = s34_classify_questions(g, "name", _s34_random.Random(0))
    assert set(qs2) == {"type", "country", "occupation"} and qs2["type"]["instructions"].startswith("The state gives")
    with tempfile.TemporaryDirectory() as td:
        recs = [s34_record("wd_classify/lead", "x1", m, *s34_classify_questions(g, "lead", rng), {"qid": "Q1726",
                           "title": "Munich", "stratum": "popular", "gold_type": "person", "gold": g,
                           "mask": {"leak_substring": False, "leak_title": False}},
                           base_id="wd_classify/Q1726", licence=s34_lic_wp("Munich", 1, "Q1726")),
                r, pr, cr]
        s34_write(td, "toy", {"eval": recs[:1], "train": recs[1:]}, {"x": 1})
        cf = load_cf()
        units = list(cf.iter_units(str(Path(td) / "toy"), ["eval", "train"], set(), None))
        assert len(units) == 3 + 3 and all(abs(sum(u["target"]) - 1) < 1e-9 for u in units), len(units)
        assert any(u["qtype"] == "noul" and u["noul_criteria"] for u in units)
        assert all(u["license"] for u in units)
    print("ok  shape 4: gold mapping, lead cleaning, masking (aliases, derivatives, case, CJK), leakage flags")
    print("ok  shapes 3-4: records load through trainers/cache_features.iter_units")


def s34_register(sub):
    def common(p):
        p.add_argument("--out", default=OUT_ROOT)
        p.add_argument("--work", default=S34_WORK, help="compact caches (jsonl.gz of extracted fields only)")
        p.add_argument("--seed", type=int, default=S34_SEED)
        p.add_argument("--refetch", action="store_true", help="ignore the compact caches")
    p = sub.add_parser("wd-compare", help="shape 3: exact comparisons from referenced Wikidata claims (K panel + "
                                          "open-book train)")
    common(p)
    p.add_argument("--no-fetch", action="store_true", help="build from the caches only")
    p.set_defaults(func=s34_cmp_build)
    p = sub.add_parser("wd-classify", help="shape 4: masked English Wikipedia leads -> fixed Wikidata taxonomies")
    common(p)
    p.add_argument("--oversample", type=float, default=1.3, help="items selected per wanted record")
    p.add_argument("--hf-dir", default="/tmp/wikidata_b_work/hf", help="where the pinned shard is downloaded")
    p.add_argument("--keep-shard", action="store_true", help="keep the parquet after extracting the leads")
    p.add_argument("--no-fresh", action="store_true", help="skip the fresh (post-2025-07) stratum")
    p.add_argument("--no-popular", action="store_true", help="skip the live popular (>= 30 sitelinks) eval stratum")
    p.add_argument("--fresh-windows", type=int, default=22, help="creation-log windows (one API request each)")
    p.set_defaults(func=s34_cls_build)
    p = sub.add_parser("wd34-gate", help="shapes 3-4 dedup: train vs eval fingerprints; eval vs training proxy")
    p.add_argument("--out", default=OUT_ROOT)
    p.add_argument("--work", default=S34_WORK)
    p.add_argument("--sources", default="wd_compare,wd_classify")
    p.add_argument("--gde1", default="/tmp/gde1", help="local GDE-1 sets used as eval fingerprints")
    p.add_argument("--proxy", default="/tmp/gde1_a_work", help="build_gde1_a.py work dir holding fp/ (training proxy)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=s34_gate)
    p = sub.add_parser("wd34-check", help="shapes 3-4: schema + iter_units read-back, stats, samples")
    p.add_argument("--out", default=OUT_ROOT)
    p.add_argument("--sources", default="wd_compare,wd_classify")
    p.set_defaults(func=s34_check)


REGISTRARS.append(s34_register)
SELF_TESTS.append(s34_self_test)


# =========================================================================== main (keep last)
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run every section's offline self-test")
    sub = ap.add_subparsers(dest="cmd")
    for reg in REGISTRARS:
        reg(sub)
    a = ap.parse_args(argv)
    if a.self_test:
        for t in SELF_TESTS:
            t()
        print(f"self-test passed ({len(SELF_TESTS)} section tests)")
        return 0
    if not getattr(a, "func", None):
        ap.print_help()
        return 2
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())
