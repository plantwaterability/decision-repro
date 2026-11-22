#!/usr/bin/env python3
"""
build_jev_physics.py -- real-world physics / astronomy / earth-science decisions whose labels are PUBLISHED CALIBRATED
PROBABILITIES (or an expert's stated probability), in the jev_data record schema (build_jev_data.record /
build_jev_ext2.make_record conventions; read back through trainers/cache_features.iter_units).

Design and licences: research/physics_pipeline.md. Conventions: research/realworld_data.md §2-3.

Sources (one `source` per family):
  physics/gw_alert     LIGO / Virgo / KAGRA public alerts (GraceDB public superevents + their GCN VOEvent notices),
                       O3 + O4, CBC candidates. Licence: CC BY 4.0 (GWOSC terms; see doc for the caveat).
        state     what the alert recipient knows when the notice arrives: observing run, month, detection pipeline
                  and search, detectors, FAR, the significance flag, alert latency, external (GRB) coincidence, and
                  the notice's own sky map summarised (90% / 50% credible area, distance mean and s.d., the BAYESTAR
                  log Bayes factors). Never: the notice's Classification / Properties groups, superevent labels,
                  GraceID, event page, GW name, GWSkyNet score, exact event time.
        labels    source_class      choice  BNS / NSBH / BBH / terrestrial (+ mass gap in O3)  = the notice's p_astro
                  has_neutron_star  noul    HasNS        (em_bright, conditioned on astrophysical)
                  has_remnant       noul    HasRemnant   (em_bright)
                  has_mass_gap      noul    HasMassGap   (em_bright, O4 only)
                  alert_retracted   noul    OUTCOME: a Retraction notice followed (significant alerts only)
                  in_gw_catalog     noul    OUTCOME: the candidate is in the final GWTC catalogue (O3, O4a, O4b)
  physics/pager        USGS PAGER (Prompt Assessment of Global Earthquakes for Response), 2013-, public domain.
        state     the earthquake (magnitude, depth, epicentre, local time, max shaking), PAGER's population and
                  economic exposure by shaking intensity, the largest nearby cities and their intensity, the building
                  stock comment, nearby past earthquakes. Never: alert colours, impact comments, loss estimates,
                  ComCat alert / sig / felt / cdi, version or elapsed time.
        labels    fatality_alert_level   score 4  green / yellow / orange / red  = PAGER's fatality bin probabilities
                  economic_alert_level   score 4                                  = PAGER's economic-loss bins
                  estimated_fatalities   score 7  0 / 1-9 / ... / 100,000+         = the 7 published bins
  physics/spc_mcd      NOAA SPC Mesoscale Discussions (via the Iowa Environmental Mesonet), public domain (NWS).
        state     the discussion text with every watch sentence, the probability line and the forecaster's peak
                  hazard estimates removed; area affected; issue month and hour.
        labels    watch_issuance        noul  the forecaster's stated "Probability of Watch Issuance" (expert, n=1)
                  watch_issued_outcome  noul  OUTCOME: an SPC severe-thunderstorm / tornado watch whose outline
                                              overlaps the discussion area is issued within 3 h

Splits: EVAL = events on or after --eval-from (default 2025-07-01); train / val = earlier, val a 10% hash sample by
group (GW superevent, earthquake event id, MD year+number). Outcome questions use lam*onehot + (1-lam)*p_hat on
train / val (p_hat a leave-one-out smoothed cell rate) and the one-hot outcome on EVAL, as build_jev_courts.py does.

Usage (laptop is fine; ~6 GB is streamed for the GW sky maps, ~1 GB for PAGER; the cache keeps small extracts):
  python3 build_jev_physics.py fetch --cache /tmp/physics_jev/cache --sources gw,pager,spc
  python3 build_jev_physics.py build --cache /tmp/physics_jev/cache --out /tmp/physics_jev
  python3 build_jev_physics.py check --out /tmp/physics_jev     # iter_units read-back, scans, stats, leak probe
  python3 build_jev_physics.py --self-test                       # no network
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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_jev_data import choice_q, noul_q, option_keys, record, score_q, validate  # noqa: E402

UA = "jev-physics-research/0.1 (research dataset build; public data, polite rate)"
GDB = "https://gracedb.ligo.org/api/"
GWOSC_ALL = "https://gwosc.org/eventapi/json/allevents/"
USGS_Q = "https://earthquake.usgs.gov/fdsnws/event/1/query"
IEM = "https://mesonet.agron.iastate.edu"

LICENSES = {
    "physics/gw_alert": ("CC-BY-4.0: LIGO/Virgo/KAGRA public alerts (GraceDB public superevents and GCN notices) and "
                         "GWOSC event catalogue; GWOSC: 'Data from this web site are licensed under a Creative "
                         "Commons Attribution 4.0 International License'. Attribute the LIGO Scientific, Virgo and "
                         "KAGRA Collaborations. GraceDB pages carry no separate licence (see research/physics_pipeline.md)."),
    "physics/pager": ("US public domain: USGS PAGER and ComCat ('USGS-authored or produced data and information are "
                      "considered to be in the U.S. Public Domain'); PAGER code CC0. Population exposure derives from "
                      "LandScan (ORNL; CC BY 4.0 since 2022, earlier releases under licence) -- attribute ORNL LandScan."),
    "physics/spc_mcd": ("US public domain: NOAA / NWS Storm Prediction Center text products (17 U.S.C. 105), retrieved "
                        "through the Iowa Environmental Mesonet (Iowa State University) archive; credit IEM."),
}
ORIGINS = {
    "physics/gw_alert": "gracedb.ligo.org public superevents + VOEvent notices; gwosc.org eventapi (GWTC catalogues)",
    "physics/pager": "earthquake.usgs.gov ComCat losspager products (pager.xml, alerts.json, exposures.json)",
    "physics/spc_mcd": "mesonet.agron.iastate.edu SPC MCD archive (api/1/nws/spc_mcd, AFOS SWOMCD text, watch outlines)",
}

EVAL_FROM = "2025-07-01"
VAL_PCT = 10
LAMBDA = 0.5

_tls = threading.local()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(obj):
    return json.dumps(obj, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)


def hkey(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class HttpError(Exception):
    def __init__(self, code, url):
        super().__init__(f"HTTP {code} for {url}")
        self.code = code


def http_get(url, headers=None, timeout=180, tries=7, max_bytes=None):
    """GET with retries and backoff. 404 / 410 raise HttpError at once (a missing file is an answer, not a fault)."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if max_bytes:
                    data = r.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        raise HttpError(413, url)
                    return data
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404, 409, 410):   # 409 = ComCat event deleted
                raise HttpError(e.code, url) from None
            last = e
        except HttpError:
            raise
        except Exception as e:  # noqa: BLE001  timeouts, resets
            last = e
        time.sleep(min(90, 2 ** i + random.random()))
    raise RuntimeError(f"giving up on {url}: {last}")


def read_gz(path, tolerant=True):
    """Rows of a gz JSONL; with `tolerant`, a file still being appended to ends at its last complete row."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        while True:
            try:
                line = f.readline()
            except (EOFError, OSError):
                if not tolerant:
                    raise
                log(f"{Path(path).name}: incomplete gzip tail (fetch still running?) -- read up to it")
                return
            if not line:
                return
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if not tolerant:
                        raise
                    return


class GzAppender:
    """Append-only gz JSONL with resume: `done` holds the keys already written (one gzip member per open)."""

    def __init__(self, path, key):
        self.path, self.key, self.lock = Path(path), key, threading.Lock()
        self.done = set()
        if self.path.exists():
            try:
                for r in read_gz(self.path):
                    self.done.add(r[key])
            except (EOFError, OSError, json.JSONDecodeError):
                log(f"{self.path.name}: truncated tail ignored ({len(self.done)} complete rows kept)")
                rows = []
                try:
                    with gzip.open(self.path, "rt", encoding="utf-8") as f:
                        for line in f:
                            rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
                with gzip.open(self.path, "wt", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                self.done = {r[key] for r in rows}
        self.f = gzip.open(self.path, "at", encoding="utf-8")
        self.n = 0

    def add(self, row):
        with self.lock:
            self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.done.add(row[self.key])
            self.n += 1
            if self.n % 200 == 0:
                self.f.flush()

    def close(self):
        self.f.close()


def run_pool(items, fn, workers, sink, what, every=250):
    t0, n, errs = time.time(), 0, Counter()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fu in as_completed(futs):
            n += 1
            try:
                row = fu.result()
                if row is not None:
                    sink.add(row)
            except Exception as e:  # noqa: BLE001
                errs[type(e).__name__ + ":" + str(e)[:80]] += 1
            if n % every == 0:
                log(f"{what}: {n}/{len(futs)} done, {time.time() - t0:.0f}s, errors {sum(errs.values())}")
    log(f"{what}: finished {n} in {time.time() - t0:.0f}s; errors {dict(errs.most_common(8))}")
    return errs


# --------------------------------------------------------------------------- FITS (sky maps; no astropy needed)
def _fits_value(raw):
    raw = raw.strip()
    if raw.startswith("'"):
        out, i = [], 1
        while i < len(raw):
            if raw[i] == "'":
                if i + 1 < len(raw) and raw[i + 1] == "'":
                    out.append("'")
                    i += 2
                    continue
                break
            out.append(raw[i])
            i += 1
        return "".join(out).rstrip()
    v = raw.split("/")[0].strip()
    if v in ("T", "F"):
        return v == "T"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def fits_hdus(buf):
    """[(header dict, data bytes)] for every HDU in a FITS byte string."""
    pos, hdus = 0, []
    while pos + 2880 <= len(buf):
        hdr, end = {}, False
        while not end:
            block = buf[pos:pos + 2880]
            pos += 2880
            if len(block) < 2880:
                return hdus
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    end = True
                    break
                if card[8:10] == "= " and key not in hdr:
                    hdr[key] = _fits_value(card[10:])
        naxis = int(hdr.get("NAXIS", 0) or 0)
        size = 0
        if naxis:
            size = abs(int(hdr.get("BITPIX", 8))) // 8
            for k in range(1, naxis + 1):
                size *= int(hdr.get(f"NAXIS{k}", 0))
            size += int(hdr.get("PCOUNT", 0) or 0)
        hdus.append((hdr, buf[pos:pos + size]))
        pos += ((size + 2879) // 2880) * 2880
    return hdus


_TFORM = {"K": ">i8", "J": ">i4", "I": ">i2", "D": ">f8", "E": ">f4", "B": "u1", "L": "u1"}


def fits_table(hdr, data):
    import numpy as np
    fields = []
    for i in range(1, int(hdr["TFIELDS"]) + 1):
        m = re.match(r"(\d*)([A-Z])", str(hdr[f"TFORM{i}"]).strip())
        rep, code = int(m.group(1) or 1), m.group(2)
        dt = _TFORM[code]
        fields.append((str(hdr.get(f"TTYPE{i}", f"COL{i}")).strip(), dt) if rep == 1 else
                      (str(hdr.get(f"TTYPE{i}", f"COL{i}")).strip(), dt, (rep,)))
    dtype = np.dtype(fields)
    assert dtype.itemsize == int(hdr["NAXIS1"]), (dtype.itemsize, hdr["NAXIS1"])
    return np.frombuffer(data, dtype=dtype, count=int(hdr["NAXIS2"]))


def skymap_summary(buf):
    """Credible areas and distance summary of a LIGO/Virgo/KAGRA sky map (multi-order NUNIQ or flat HEALPix)."""
    import numpy as np
    if buf[:2] == b"\x1f\x8b":
        buf = gzip.decompress(buf)
    hdus = fits_hdus(buf)
    tab = next(((h, d) for h, d in hdus if h.get("XTENSION") == "BINTABLE"), None)
    if tab is None:
        raise ValueError("no BINTABLE in sky map")
    hdr, data = tab
    t = fits_table(hdr, data)
    names = t.dtype.names
    sr_to_deg2 = (180.0 / math.pi) ** 2
    if "UNIQ" in names:
        uniq = t["UNIQ"].astype(np.int64)
        order = (np.floor(np.log2(uniq.astype(np.float64))).astype(np.int64) - 2) // 2
        area = 4 * math.pi / (12.0 * 4.0 ** order)
        dens = t["PROBDENSITY"].astype(np.float64)
    else:
        nside = int(hdr["NSIDE"])
        col = "PROB" if "PROB" in names else names[0]
        p = t[col].astype(np.float64).ravel()
        area = np.full(p.shape, 4 * math.pi / (12.0 * nside * nside))
        dens = p / area
    prob = dens * area
    tot = float(prob.sum())
    idx = np.argsort(-dens, kind="stable")
    cum = np.cumsum(prob[idx]) / tot
    a_cum = np.cumsum(area[idx])

    def credible(level):
        j = int(np.searchsorted(cum, level))
        return float(a_cum[min(j, len(a_cum) - 1)] * sr_to_deg2)
    out = {"area90_deg2": round(credible(0.9), 2), "area50_deg2": round(credible(0.5), 2),
           "prob_total": round(tot, 4)}
    for k in ("DISTMEAN", "DISTSTD", "LOGBCI", "LOGBSN", "CREATOR", "INSTRUME", "DATE", "DATE-OBS", "RUNTIME",
              "MOCORDER", "NSIDE"):
        if k in hdr:
            out[k.lower().replace("-", "_")] = hdr[k]
    return out


# --------------------------------------------------------------------------- GW fetch
def parse_voevent(xml_bytes):
    root = ET.fromstring(xml_bytes)
    ns_strip = lambda tag: tag.split("}")[-1]  # noqa: E731
    what = next((el for el in root if ns_strip(el.tag) == "What"), None)
    params, groups, skymap = {}, {}, None
    if what is not None:
        for el in what:
            tag = ns_strip(el.tag)
            if tag == "Param":
                params[el.get("name")] = el.get("value")
            elif tag == "Group":
                g = {}
                for p in el:
                    if ns_strip(p.tag) == "Param":
                        g[p.get("name")] = p.get("value")
                        if p.get("name") == "skymap_fits":
                            skymap = p.get("value")
                groups[el.get("name") or el.get("type")] = g
    iso = None
    for el in root.iter():
        if ns_strip(el.tag) == "ISOTime":
            iso = (el.text or "").strip()
            break
    who_date = None
    for el in root.iter():
        if ns_strip(el.tag) == "Date":
            who_date = (el.text or "").strip()
            break
    return {"params": params, "groups": groups, "skymap_url": skymap, "event_time": iso, "notice_time": who_date,
            "role": root.get("role")}


def fetch_gw_list(cache):
    path = cache / "gw_superevents.jsonl.gz"
    if path.exists():
        log(f"{path.name} exists; delete it to refresh the superevent list")
        return list(read_gz(path))
    out, start = [], 0
    while True:
        url = f"{GDB}superevents/?query=category%3A+Production&count=250&start={start}&format=json"
        d = json.loads(http_get(url))
        ss = d.get("superevents") or []
        for s in ss:
            pe = s.get("preferred_event_data") or {}
            out.append({"superevent_id": s["superevent_id"], "created": s.get("created"), "t_0": s.get("t_0"),
                        "far": s.get("far"), "labels": s.get("labels"), "gw_id": s.get("gw_id"),
                        "pref": {k: pe.get(k) for k in ("group", "pipeline", "search", "instruments", "far",
                                                        "reporting_latency", "offline")},
                        "pipelines": sorted((s.get("pipeline_preferred_events") or {}).keys())})
        start += len(ss)
        log(f"superevent list: {start}/{d.get('numRows')}")
        if not ss or start >= int(d.get("numRows") or 0):
            break
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def gw_notice(url, with_skymap=True):
    nt = parse_voevent(http_get(url))
    if with_skymap and nt["skymap_url"]:
        try:
            nt["skymap"] = skymap_summary(http_get(nt["skymap_url"], max_bytes=120_000_000))
        except Exception as e:  # noqa: BLE001
            nt["skymap_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return nt


def gw_one(sid):
    d = json.loads(http_get(f"{GDB}superevents/{sid}/voevents/?format=json"))
    rows = []
    for v in d.get("voevents") or []:
        links = v.get("links") or {}
        rows.append({"N": int(v["N"]), "type": v.get("voevent_type"), "created": v.get("created"),
                     "filename": v.get("filename"), "file": links.get("file")})
    rows.sort(key=lambda r: r["N"])
    out = {"superevent_id": sid, "voevents": [{k: r[k] for k in ("N", "type", "created", "filename")} for r in rows],
           "retracted": any(r["type"] == "RE" for r in rows)}
    first_re = min((r["N"] for r in rows if r["type"] == "RE"), default=None)
    first = None
    for r in rows:
        if r["type"] != "PR" or not r["file"]:
            continue
        try:
            nt = gw_notice(r["file"], with_skymap=False)
        except HttpError as e:      # a few O3 notices are listed but their file is gone (404)
            out.setdefault("missing_notice_files", []).append({"N": r["N"], "code": e.code})
            continue
        if first is None:
            out["first_notice_any"] = {"N": r["N"], **{k: nt[k] for k in ("params", "groups", "event_time",
                                                                           "notice_time")}}
        if nt["groups"].get("Classification") and nt["skymap_url"]:
            if nt["skymap_url"]:
                try:
                    nt["skymap"] = skymap_summary(http_get(nt["skymap_url"], max_bytes=120_000_000))
                except Exception as e:  # noqa: BLE001
                    nt["skymap_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            first = {"N": r["N"], "type": "PR", **nt}
            break
    out["first_classified"] = first
    later = [r for r in rows if r["type"] in ("IN", "UP") and r["file"] and (first_re is None or r["N"] < first_re)]
    if later:
        for r in reversed(later):
            try:
                nt = gw_notice(r["file"])
            except HttpError as e:
                out.setdefault("missing_notice_files", []).append({"N": r["N"], "code": e.code})
                continue
            out["last_update"] = {"N": r["N"], "type": r["type"], **nt}
            break
    return out


def fetch_gw(a, cache):
    events = fetch_gw_list(cache)
    log(f"{len(events)} production superevents")
    raw = http_get(GWOSC_ALL)
    (cache / "gwosc_allevents.json").write_bytes(raw)
    sink = GzAppender(cache / "gw_notices.jsonl.gz", "superevent_id")
    todo = [e["superevent_id"] for e in events if e["superevent_id"] not in sink.done]
    if a.limit:
        todo = todo[: a.limit]
    log(f"GW notices: {len(sink.done)} cached, {len(todo)} to fetch with {a.gw_workers} workers")
    try:
        run_pool(todo, gw_one, a.gw_workers, sink, "gw")
    finally:
        sink.close()


# --------------------------------------------------------------------------- PAGER fetch
def parse_pager_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = {"root": dict(root.attrib)}
    ev = root.find("event")
    out["event"] = dict(ev.attrib) if ev is not None else {}
    alerts = {}
    for al in root.iter("alert"):
        alerts[al.get("type")] = {"level": al.get("level"), "summary": al.get("summary"), "units": al.get("units"),
                                  "bins": [dict(b.attrib) for b in al.iter("bin")]}
    out["alerts_xml"] = alerts
    out["exposure"] = [dict(x.attrib) for x in root.iter("exposure")]
    cities = [dict(c.attrib) for c in root.iter("city")]
    for c in cities:
        try:
            c["_pop"] = int(float(c.get("population") or 0))
        except ValueError:
            c["_pop"] = 0
    top_pop = sorted(cities, key=lambda c: -c["_pop"])[:10]
    top_mmi = sorted(cities, key=lambda c: -float(c.get("mmi") or 0))[:5]
    keep, seen = [], set()
    for c in top_pop + top_mmi:
        k = (c.get("name"), c.get("lat"), c.get("lon"))
        if k not in seen:
            seen.add(k)
            keep.append({kk: c.get(kk) for kk in ("name", "population", "mmi", "iscapital", "lat", "lon")})
    out["cities"] = keep
    out["n_cities"] = len(cities)
    for tag in ("structcomment", "alertcomment", "impact_comment", "secondary_effects"):
        el = root.find(tag)
        out[tag] = (el.text or "").strip() if el is not None and el.text else None
    return out


def pager_one(feat):
    eid = feat["id"]
    det = json.loads(http_get(f"{USGS_Q}?eventid={urllib.parse.quote(eid)}&format=geojson"))
    pr = det["properties"]
    prods = (pr.get("products") or {}).get("losspager") or []
    if not prods:
        return {"id": eid, "error": "no_losspager"}
    p = prods[0]
    cont = p.get("contents") or {}

    def url(*names):
        for n in names:
            if n in cont:
                return cont[n]["url"]
        return None
    out = {"id": eid, "summary": {k: pr.get(k) for k in ("mag", "magType", "place", "time", "tz", "tsunami", "mmi",
                                                         "alert", "sig", "felt", "cdi", "net", "type", "title",
                                                         "status")},
           "geometry": det.get("geometry", {}).get("coordinates"),
           "product": {"properties": p.get("properties"), "updateTime": p.get("updateTime"), "status": p.get("status"),
                       "n_versions_in_feed": len(prods)},
           "shakemap_props": ((pr.get("products") or {}).get("shakemap") or [{}])[0].get("properties")}
    xu = url("pager.xml")
    if not xu:
        out["error"] = "no_pager_xml"
        return out
    out["pager"] = parse_pager_xml(http_get(xu))
    au = url("json/alerts.json", "alerts.json")
    if au:
        try:
            out["alerts_json"] = json.loads(http_get(au))
        except Exception as e:  # noqa: BLE001
            out["alerts_json_error"] = str(e)[:100]
    eu = url("json/exposures.json", "exposures.json")
    if eu:
        try:
            ex = json.loads(http_get(eu))
            pe = ex.get("population_exposure") or {}
            ee = ex.get("economic_exposure") or {}
            out["exposures_json"] = {"population": pe, "economic_aggregated": ee.get("aggregated_exposure"),
                                     "economic_mmi": ee.get("mmi")}
        except Exception as e:  # noqa: BLE001
            out["exposures_json_error"] = str(e)[:100]
    return out


def fetch_pager(a, cache):
    lst = cache / "pager_events.json"
    if not lst.exists():
        raw = http_get(f"{USGS_Q}?format=geojson&producttype=losspager&starttime=2013-01-01&orderby=time-asc",
                       timeout=600)
        lst.write_bytes(raw)
    feats = json.loads(lst.read_text())["features"]
    feats = [f for f in feats if (f["properties"].get("type") or "earthquake") == "earthquake"
             and f["properties"]["time"] >= 1356998400000]   # 2013-01-01
    sink = GzAppender(cache / "pager_events.jsonl.gz", "id")
    todo = [f for f in feats if f["id"] not in sink.done]
    if a.limit:
        todo = todo[: a.limit]
    log(f"PAGER: {len(feats)} events since 2013, {len(sink.done)} cached, {len(todo)} to fetch, {a.usgs_workers} workers")
    try:
        run_pool(todo, pager_one, a.usgs_workers, sink, "pager")
    finally:
        sink.close()


# --------------------------------------------------------------------------- SPC fetch (via IEM)
def month_starts(y0, y1_excl_month):
    y, m = y0, 1
    while (y, m) < y1_excl_month:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def spc_month(ym):
    """One calendar month: the MCD index (watch confidence + polygon), the MCD texts, and the SAW watch outlines."""
    y, m = ym
    nxt = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=timezone.utc)
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    hours = int((nxt - start).total_seconds() // 3600)
    idx = json.loads(http_get(f"{IEM}/api/1/nws/spc_mcd.geojson?valid={nxt:%Y-%m-%dT%H:%M:%SZ}&hours={hours}"))
    mcds = []
    for f in idx.get("features") or []:
        p = f["properties"]
        mcds.append({**{k: p.get(k) for k in ("product_id", "year", "num", "issue", "expire", "watch_confidence",
                                              "concerning")}, "geometry": f.get("geometry")})
    sd, ed = f"{start:%Y-%m-%d}", f"{nxt:%Y-%m-%d}"
    txt = http_get(f"{IEM}/cgi-bin/afos/retrieve.py?pil=SWOMCD&fmt=text&sdate={sd}&edate={ed}&limit=9999",
                   timeout=300).decode("utf-8", "replace")
    saws = []
    for k in range(10):
        s = http_get(f"{IEM}/cgi-bin/afos/retrieve.py?pil=SAW{k}&fmt=text&sdate={sd}&edate={ed}&limit=9999",
                     timeout=300).decode("utf-8", "replace")
        saws.append(s)
    return {"month": f"{y:04d}-{m:02d}", "mcds": mcds, "mcd_text": txt, "saw_text": "\x03".join(saws)}


def fetch_spc(a, cache):
    sink = GzAppender(cache / "spc_months.jsonl.gz", "month")
    now = datetime.now(timezone.utc)
    months = [ym for ym in month_starts(a.spc_from, (now.year, now.month)) if f"{ym[0]:04d}-{ym[1]:02d}" not in sink.done]
    if a.limit:
        months = months[: a.limit]
    log(f"SPC: {len(sink.done)} months cached, {len(months)} to fetch, {a.iem_workers} workers")
    try:
        run_pool(months, spc_month, a.iem_workers, sink, "spc", every=12)
    finally:
        sink.close()


# =========================================================================== BUILD
# --------------------------------------------------------------------------- shared
def parse_iso(s):
    if not s:
        return None
    s = str(s).strip().replace(" UTC", "").replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def lab_dict(keys, vals):
    """Non-negative, rounded to 6 dp, summing to exactly 1 (the last key absorbs the rounding)."""
    v = [max(0.0, float(x)) for x in vals]
    s = sum(v)
    if s <= 0:
        raise ValueError("empty distribution")
    v = [round(x / s, 6) for x in v]
    j = max(range(len(v)), key=lambda i: v[i])
    v[j] = round(v[j] + (1.0 - sum(v)), 6)
    return dict(zip(keys, v))


def noul_lab(p):
    p = min(max(float(p), 0.0), 1.0)
    return lab_dict(["yes", "no"], [p, 1 - p])


def kind_of(lab):
    return "hard" if sum(1 for v in lab.values() if v > 0) == 1 else "soft"


def entropy(vals):
    return -sum(x * math.log(x) for x in vals if x > 0)


def split_for(day, group, eval_from):
    if day >= eval_from:
        return "eval"
    return "val" if hkey("val", group) % 100 < VAL_PCT else "train"


def surface_for(group):
    return "json" if hkey("surface", group) % 100 < 60 else "memo"


def render_memo(state, title):
    lines = [title]
    for sec, body in state.items():
        lines.append("")
        lines.append(sec.replace("_", " ").upper() + ":")
        if isinstance(body, dict):
            for k, v in body.items():
                if isinstance(v, list):
                    lines.append(f"- {k.replace('_', ' ')}:")
                    for it in v:
                        lines.append("    * " + (", ".join(f"{kk.replace('_', ' ')} {vv}" for kk, vv in it.items())
                                                 if isinstance(it, dict) else str(it)))
                else:
                    lines.append(f"- {k.replace('_', ' ')}: {v}")
        elif isinstance(body, list):
            for it in body:
                lines.append("- " + (", ".join(f"{kk.replace('_', ' ')} {vv}" for kk, vv in it.items())
                                     if isinstance(it, dict) else str(it)))
        else:
            lines.append(str(body))
    return "\n".join(lines)


def render(state, surface, title):
    return jdump(state) if surface == "json" else render_memo(state, title)


def shuffled_criteria(criteria, rng):
    items = list(criteria.items())
    rng.shuffle(items)
    return dict(items)


def blend(y, p_hat, lam):
    """lam * onehot(y) + (1 - lam) * p_hat for a binary outcome -> P(yes)."""
    return lam * float(y) + (1 - lam) * float(p_hat)


def loo_cell_rates(cells, y, fit, kappas=(0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)):
    """Leave-one-out Beta-smoothed rate per cell, shrunk to the (LOO) global rate; kappa by LOO log score.

    An item's own outcome never enters its own rate; items outside `fit` (EVAL) never enter any rate."""
    n_fit = sum(1 for f in fit if f)
    s_fit = sum(yy for yy, f in zip(y, fit) if f)
    cnt, pos = Counter(), Counter()
    for c, yy, f in zip(cells, y, fit):
        if f:
            cnt[c] += 1
            pos[c] += yy

    def rate(i, k):
        c, own = cells[i], fit[i]
        n, s = cnt[c] - own, pos[c] - (y[i] if own else 0)
        g = (s_fit - (y[i] if own else 0)) / max(1, n_fit - own)
        return min(max((s + k * g) / (n + k), 1e-3), 1 - 1e-3)

    def score(k):
        return -sum(math.log(rate(i, k) if y[i] else 1 - rate(i, k)) for i in range(len(y)) if fit[i])
    best = min(kappas, key=score) if n_fit else kappas[0]
    return [rate(i, best) for i in range(len(y))], best


def make_rec(source, state, qs, labs, meta, group, kinds=None):
    kinds = {q: (kinds or {}).get(q) or kind_of(labs[q]) for q in qs}
    rec = record(source, state, qs, labs, kinds, None)
    rec["base_id"] = rec["id"]
    rec["license"] = LICENSES[source]
    rec["meta"] = {"dataset": ORIGINS[source], "group": str(group), **meta}
    validate(rec)
    return rec


# --------------------------------------------------------------------------- outcome-word scans (per source)
SCANS = {
    "physics/gw_alert": re.compile(
        r"\b(bns|nsbh|bbh|terrestrial|neutron[ -]star|black[ -]hole|mass[ -]gap|retract\w*|hasns|hasremnant|"
        r"hasmassgap|p_?astro|classification|em[_ -]?bright|gw\d{6}\w*|s\d{6}[a-z]{1,3}|advno|advok|gwskynet)\b",
        re.I),
    "physics/pager": re.compile(
        r"\b((green|yellow|orange|red) alert|alert level|pager alert|casualties (are|is)|damage (is|are) "
        r"(likely|probable|possible|unlikely)|losses (are|is)|of gdp|level response|estimated (economic )?losses|"
        r"fatalities (are|is))\b", re.I),
    "physics/spc_mcd": re.compile(r"\b(watch\w*|ww|wws|issuance|probability of|headlines?)\b", re.I),
}


def scan(source, text):
    return sorted({m.group(0).lower() for m in SCANS[source].finditer(text)})


# --------------------------------------------------------------------------- GW alerts
GW_RUNS = [("O3a", "2019-04-01", "2019-10-02"), ("O3b", "2019-11-01", "2020-03-28"),
           ("ER15 (engineering run before O4)", "2023-04-20", "2023-05-24"), ("O4a", "2023-05-24", "2024-01-17"),
           ("ER16 (engineering run before O4b)", "2024-03-15", "2024-04-10"), ("O4b", "2024-04-10", "2025-01-28"),
           ("O4c", "2025-01-28", "2025-11-19")]
# Final catalogue per run (GWOSC eventapi shortName). GWTC-4.1 is a superset of GWTC-4.0 (129 + 11 events); the
# O4a / O4b catalogues also contain candidates from the engineering runs just before them (from 2023-05-17, 2024-04-06).
GW_CATALOGS = {"O3a": ["GWTC-2.1-confident"], "O3b": ["GWTC-3-confident"],
               "ER15 (engineering run before O4)": ["GWTC-4.0", "GWTC-4.1"], "O4a": ["GWTC-4.0", "GWTC-4.1"],
               "ER16 (engineering run before O4b)": ["GWTC-5.0"], "O4b": ["GWTC-5.0"]}
DETECTOR = {"H1": "LIGO Hanford (H1)", "L1": "LIGO Livingston (L1)", "V1": "Virgo (V1)", "K1": "KAGRA (K1)"}
SEARCH = {"AllSky": "all-sky compact-binary search", "BBH": "targeted binary-black-hole search",
          "EarlyWarning": "early-warning search (detection before merger)", "SSM": "sub-solar-mass search",
          "IMBH": "intermediate-mass black-hole search"}
GW_CLASS4 = {
    "bns": "Binary neutron star merger: both objects lighter than 3 solar masses",
    "nsbh": "Neutron star - black hole merger: one object lighter than 3 solar masses, the other heavier",
    "bbh": "Binary black hole merger: both objects heavier than 3 solar masses",
    "terrestrial": "Terrestrial: not astrophysical; a background noise fluctuation or an instrumental glitch",
}
GW_CLASS5 = {   # O3 definitions (LVK O3 user guide): MassGap was its own class and BBH meant both > 5 Msun
    "bns": "Binary neutron star merger: both objects lighter than 3 solar masses",
    "nsbh": "Neutron star - black hole merger: one object lighter than 3 solar masses, the other heavier than 5",
    "bbh": "Binary black hole merger: both objects heavier than 5 solar masses",
    "mass_gap": "Mass-gap merger: at least one object between 3 and 5 solar masses",
    "terrestrial": "Terrestrial: not astrophysical; a background noise fluctuation or an instrumental glitch",
}
GW_KEYMAP = {"BNS": "bns", "NSBH": "nsbh", "BBH": "bbh", "MassGap": "mass_gap", "Terrestrial": "terrestrial"}
GW_Q = {
    "source_class": ("What produced this gravitational-wave candidate? Give the probability of each source class, "
                     "reading the alert's detection statistics and sky map the way the low-latency source "
                     "classification does (classes use the component masses of the merging objects)."),
    "has_neutron_star": ("Assume the candidate is a real compact-binary merger. Does at least one of the two objects "
                         "have a mass below 3 solar masses, i.e. could it be a neutron star?"),
    "has_remnant": ("Assume the candidate is a real compact-binary merger. Was a nonzero amount of neutron-star "
                    "matter left outside the final compact object (ejecta or a disk, the material that could power an "
                    "electromagnetic counterpart such as a kilonova)?"),
    "has_mass_gap": ("Assume the candidate is a real compact-binary merger. Does at least one of the two objects have "
                     "a mass between 3 and 5 solar masses?"),
    "alert_retracted": ("Will this alert be retracted, i.e. will the collaborations' follow-up vetting conclude the "
                        "candidate is not astrophysical (noise or an instrumental artefact) and send a Retraction "
                        "notice?"),
    "in_gw_catalog": ("Will this candidate be confirmed in the collaborations' final Gravitational-Wave Transient "
                      "Catalog (GWTC) for its observing run, i.e. found by the offline reanalysis with a probability "
                      "of astrophysical origin of at least 0.5?"),
}


def run_of(dt):
    d = dt.strftime("%Y-%m-%d")
    for name, a, b in GW_RUNS:
        if a <= d < b:
            return name
    return "outside an observing run"


def far_text(far_hz):
    py = far_hz * 3.15576e7
    if py >= 1:
        return f"{py:.3g} per year"
    return f"1 per {1 / py:.3g} years" if py > 0 else "0"


def gw_state(nt, kind):
    p = nt["params"]
    et, ntime = parse_iso(nt["event_time"]), parse_iso(nt["notice_time"])
    run = run_of(et)
    far = float(p["FAR"])
    det = [d.strip() for d in (p.get("Instruments") or "").split(",") if d.strip()]
    alert = {"notice": {"PR": "preliminary (automated, before human vetting)",
                        "IN": "initial (issued after human vetting)",
                        "UP": "update (after human vetting, with refined analyses)"}[kind],
             "observing_run": run, "month": f"{et:%Y-%m}"}
    if ntime:
        alert["minutes_after_signal"] = round((ntime - et).total_seconds() / 60.0, 1)
    detection = {"analysis_group": {"CBC": "compact binary coalescence (modelled waveforms)",
                                    "Burst": "unmodelled burst"}.get(p.get("Group"), p.get("Group")),
                 "pipeline": (p.get("Pipeline") or "").lower(),
                 "search": SEARCH.get(p.get("Search"), p.get("Search")),
                 "detectors": [DETECTOR.get(d, d) for d in det],
                 "false_alarm_rate_hz": float(f"{far:.4g}"),
                 "false_alarm_rate": far_text(far)}
    if "Significant" in p:
        detection["significant"] = ("yes (passes the public significance threshold)" if p["Significant"] == "1"
                                    else "no (low-significance alert)")
    state = {"alert": alert, "detection": detection}
    sk = nt.get("skymap")
    if sk:
        sm = {"area_90pct_deg2": round(sk["area90_deg2"], 1), "area_50pct_deg2": round(sk["area50_deg2"], 1)}
        if sk.get("creator"):
            sm = {"made_by": sk["creator"], **sm}
        if sk.get("distmean") is not None and math.isfinite(float(sk["distmean"])) and float(sk["distmean"]) > 0:
            sm["distance_mpc"] = f"{float(sk['distmean']):.0f} +/- {float(sk['diststd']):.0f}"
        if sk.get("logbsn") is not None:
            sm["log_bayes_factor_signal_vs_noise"] = round(float(sk["logbsn"]), 2)
        if sk.get("logbci") is not None:
            sm["log_bayes_factor_coherent_vs_incoherent"] = round(float(sk["logbci"]), 2)
        state["sky_map"] = sm
    ext = {}
    for gname, g in nt["groups"].items():
        if gname in ("GW_SKYMAP", "Classification", "Properties"):
            continue
        for k, v in g.items():
            if re.search(r"(ivorn|url|skymap|_id$|^id$|gcn_notice)", k, re.I) or v in (None, ""):
                continue
            ext[f"{gname} {k}".replace(" ", "_").lower()] = v
    if ext:
        state["external_coincidence"] = ext
    return state, run, et


def load_gwtc(cache):
    raw = json.loads((cache / "gwosc_allevents.json").read_text())["events"]
    by_cat = defaultdict(list)
    for v in raw.values():
        if v.get("GPS") is not None:
            by_cat[v["catalog.shortName"]].append(v)
    return by_cat


def gw_catalog_match(by_cat, run, gps):
    cats = GW_CATALOGS.get(run)
    if not cats or gps is None:
        return None, None
    best = None
    for c in cats:
        for v in by_cat.get(c, []):
            dt = abs(float(v["GPS"]) - float(gps))
            if dt < 1.0 and (best is None or dt < best[0]):
                best = (dt, v)
    if best is None:
        return False, None
    pa = best[1].get("p_astro")
    return (pa is None or float(pa) >= 0.5), {"name": best[1].get("commonName"), "catalog": best[1].get("catalog.shortName"),
                                             "p_astro": pa, "dt_s": round(best[0], 4)}


def build_gw(a, cache):
    sups = {r["superevent_id"]: r for r in read_gz(cache / "gw_superevents.jsonl.gz")}
    by_cat = load_gwtc(cache)
    rows = list(read_gz(cache / "gw_notices.jsonl.gz"))
    cands, drops = [], Counter()
    for r in rows:
        sid = r["superevent_id"]
        sup = sups.get(sid, {})
        for kind_key in ("first_classified", "last_update"):
            nt = r.get(kind_key)
            if not nt:
                if kind_key == "first_classified":
                    drops["no_classified_notice(burst or unclassified)"] += 1
                continue
            p = nt["params"]
            if p.get("HardwareInj") == "1":
                drops["hardware_injection"] += 1
                continue
            cls = nt["groups"].get("Classification")
            if not cls:
                drops[f"{kind_key}:no_classification"] += 1
                continue
            if not nt.get("skymap"):
                drops[f"{kind_key}:no_skymap({(nt.get('skymap_error') or '')[:30]})"] += 1
                continue
            cands.append((r, sup, nt, "PR" if kind_key == "first_classified" else nt["type"]))
    # outcome base rates (fit on pre-EVAL first notices only)
    eval_from = a.eval_from
    items = []
    for r, sup, nt, kind in cands:
        state, run, et = gw_state(nt, kind)
        day = et.strftime("%Y-%m-%d")
        sig = nt["params"].get("Significant", "1" if run.startswith("O3") else "0") == "1"
        far_bin = min(2, max(-8, math.floor(math.log10(max(float(nt["params"]["FAR"]) * 3.15576e7, 1e-12)))))
        inc, cat = gw_catalog_match(by_cat, run, sup.get("t_0"))
        items.append({"r": r, "sup": sup, "nt": nt, "kind": kind, "state": state, "run": run, "et": et, "day": day,
                      "sig": sig, "far_bin": far_bin, "in_cat": inc, "cat": cat,
                      "split": split_for(day, r["superevent_id"], eval_from)})
    ret_idx = [i for i, it in enumerate(items) if it["kind"] == "PR" and it["sig"]]
    ret_cells = [(items[i]["run"][:2], items[i]["far_bin"]) for i in ret_idx]
    ret_y = [int(items[i]["r"]["retracted"]) for i in ret_idx]
    ret_fit = [items[i]["split"] != "eval" for i in ret_idx]
    ret_p, ret_k = loo_cell_rates(ret_cells, ret_y, ret_fit)
    ret_of = dict(zip(ret_idx, ret_p))
    cat_idx = [i for i, it in enumerate(items) if it["in_cat"] is not None]
    cat_cells = [(items[i]["run"], items[i]["sig"], items[i]["far_bin"], items[i]["kind"] == "PR") for i in cat_idx]
    cat_y = [int(items[i]["in_cat"]) for i in cat_idx]
    cat_fit = [items[i]["split"] != "eval" for i in cat_idx]
    cat_p, cat_k = loo_cell_rates(cat_cells, cat_y, cat_fit)
    cat_of = dict(zip(cat_idx, cat_p))
    recs = []
    for i, it in enumerate(items):
        nt, kind, sid = it["nt"], it["kind"], it["r"]["superevent_id"]
        rng = random.Random(hkey("gw", sid, kind))
        qs, labs, lsrc, lmeth, kinds = {}, {}, {}, {}, {}
        cls = {GW_KEYMAP[k]: float(v) for k, v in nt["groups"]["Classification"].items() if k in GW_KEYMAP}
        crit = GW_CLASS5 if "mass_gap" in cls else GW_CLASS4
        qid = "source_class_o3" if "mass_gap" in cls else "source_class"
        if set(cls) != set(crit):
            drops["classification_keys_unexpected"] += 1
            continue
        crit = shuffled_criteria(crit, rng)
        qs[qid] = choice_q(GW_Q["source_class"], crit)
        labs[qid] = lab_dict(list(crit), [cls[k] for k in crit])
        lsrc[qid] = "published_probability:lvk_p_astro"
        lmeth[qid] = (f"the notice's Classification group = the preferred pipeline's ({nt['params'].get('Pipeline')}) "
                      "low-latency multi-component p_astro, normalised")
        props = nt["groups"].get("Properties") or {}
        for key, q in (("HasNS", "has_neutron_star"), ("HasRemnant", "has_remnant"), ("HasMassGap", "has_mass_gap")):
            if props.get(key) not in (None, ""):
                qs[q] = noul_q(GW_Q[q])
                labs[q] = noul_lab(float(props[key]))
                lsrc[q] = "published_probability:lvk_em_bright"
                lmeth[q] = f"the notice's Properties.{key} (em_bright; conditioned on an astrophysical merger)"
        outcome = {}
        if i in ret_of:
            y = int(it["r"]["retracted"])
            outcome["alert_retracted"] = bool(y)
            qs["alert_retracted"] = noul_q(GW_Q["alert_retracted"])
            if it["split"] == "eval":
                labs["alert_retracted"] = noul_lab(y)
                lmeth["alert_retracted"] = "one-hot realised outcome (a Retraction notice exists) -- EVAL"
            else:
                labs["alert_retracted"] = noul_lab(blend(y, ret_of[i], a.lam))
                lmeth["alert_retracted"] = (f"lam*onehot(retracted) + (1-lam)*p_hat, lam={a.lam}; p_hat = LOO "
                                            f"Beta-smoothed rate over run family x log10 FAR/yr cells (kappa={ret_k}), "
                                            "fit on pre-EVAL significant preliminary alerts")
            lsrc["alert_retracted"] = "outcome"
            kinds["alert_retracted"] = "hard" if it["split"] == "eval" else "soft"
        if i in cat_of:
            y = int(it["in_cat"])
            outcome["in_gw_catalog"] = bool(y)
            qs["in_gw_catalog"] = noul_q(GW_Q["in_gw_catalog"])
            if it["split"] == "eval":
                labs["in_gw_catalog"] = noul_lab(y)
                lmeth["in_gw_catalog"] = "one-hot: matched (|dt| < 1 s) to a GWTC confident event with p_astro >= 0.5"
            else:
                labs["in_gw_catalog"] = noul_lab(blend(y, cat_of[i], a.lam))
                lmeth["in_gw_catalog"] = (f"lam*onehot(in GWTC, |dt| < 1 s, p_astro >= 0.5) + (1-lam)*p_hat, "
                                          f"lam={a.lam}; p_hat = LOO Beta-smoothed rate over run x significant x "
                                          f"FAR cells (kappa={cat_k})")
            lsrc["in_gw_catalog"] = "outcome"
            kinds["in_gw_catalog"] = "hard" if it["split"] == "eval" else "soft"
        surface = surface_for(sid + kind)
        text = render(it["state"], surface, "GRAVITATIONAL-WAVE CANDIDATE ALERT")
        # The search NAME is an alert-time input ("BBH" = the targeted heavy-binary search); everything else is scanned.
        hits = scan("physics/gw_alert", text.replace(str(it["state"]["detection"]["search"]), ""))
        if hits:
            drops["outcome_word_hit:" + ",".join(hits)] += 1
            continue
        meta = {"label_source": lsrc, "label_method": lmeth, "n_raters": None,
                "as_of": nt.get("notice_time"), "event_day": it["day"], "observing_run": it["run"],
                "notice": {"type": kind, "N": nt["N"], "superevent_id": sid},
                "significant": it["sig"], "surface": surface, "split_reason": f"{it['split']}:event {it['day']}",
                "outcome": outcome, "gwtc_match": it["cat"],
                "pipelines_final": it["sup"].get("pipelines"), "gw_id": it["sup"].get("gw_id")}
        rec = make_rec("physics/gw_alert", text, qs, labs, meta, sid, kinds)
        recs.append((it["split"], rec))
    rep = {"candidates": len(cands), "records": len(recs), "drops": dict(drops),
           "retraction": {"n": len(ret_y), "retracted": sum(ret_y), "kappa": ret_k},
           "catalog": {"n": len(cat_y), "in_catalog": sum(cat_y), "kappa": cat_k}}
    return recs, rep


# --------------------------------------------------------------------------- PAGER
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
MMI_WORD = {1: "not felt", 2: "weak", 3: "weak", 4: "light", 5: "moderate", 6: "strong", 7: "very strong",
            8: "severe", 9: "violent", 10: "extreme"}
FAT_LEVELS = ["green: no shaking-related deaths expected (estimate below 1)",
              "yellow: 1 to 99 shaking-related deaths",
              "orange: 100 to 999 shaking-related deaths",
              "red: 1,000 or more shaking-related deaths"]
ECO_LEVELS = ["green: economic losses below US$1 million",
              "yellow: US$1 million to under US$100 million",
              "orange: US$100 million to under US$1 billion",
              "red: US$1 billion or more"]
FAT_BINS = ["0", "1-9", "10-99", "100-999", "1,000-9,999", "10,000-99,999", "100,000 or more"]
PAGER_Q = {
    "fatality_alert_level": ("How many people are likely to be killed by the shaking from this earthquake? Choose the "
                             "fatality alert level (US Geological Survey PAGER scale)."),
    "economic_alert_level": ("How large are the economic losses from the shaking likely to be? Choose the "
                             "economic-loss alert level (US Geological Survey PAGER scale)."),
    "estimated_fatalities": "How many shaking-related deaths will this earthquake cause? Choose the range.",
}
COLOR_IDX = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


def mmi_label(x):
    k = int(round(float(x)))
    k = min(max(k, 1), 10)
    return f"{ROMAN[k]} ({MMI_WORD[k]})"


def pager_bins(r, which):
    """7 bin probabilities (ascending) and colours: alerts.json (full precision) else pager.xml (rounded)."""
    src, bins = None, None
    aj = (r.get("alerts_json") or {}).get(which)
    if aj and len(aj.get("bins") or []) == 7:
        bins, src = aj["bins"], "alerts.json"
    else:
        ax = (r["pager"]["alerts_xml"] or {}).get(which)
        if ax and len(ax.get("bins") or []) == 7:
            bins, src = ax["bins"], "pager.xml"
    if not bins:
        return None, None, None
    bins = sorted(bins, key=lambda b: float(b["min"]))
    probs = [float(b["probability"]) for b in bins]
    if src == "pager.xml" and sum(probs) > 1.5:     # integer percentages (old products)
        probs = [p / 100.0 for p in probs]
    return probs, [b.get("color") for b in bins], src


def pager_floor_key(r, which):
    aj = (r.get("alerts_json") or {}).get(which) or {}
    g = aj.get("gvalue")
    return (str(r["pager"]["root"].get("version", ""))[:4], which, None if g is None else round(float(g), 4))


def pager_vec_key(probs):
    return tuple(round(p, 3) for p in probs)


def pager_floor_table(rows, min_count=20):
    """Modal 7-bin vector per (software version, loss type, gvalue) -> the version's minimum-loss clamp.

    PAGER v2 publishes a lognormal over the bins; when the estimated median loss is negligible the median is clamped,
    so every such event in a (version, gvalue) cell gets the SAME vector (e.g. v2.0b fatality, gvalue 2.5:
    0.653 / 0.302 / 0.043 / 0.001). Those vectors are a modelling convention, not a calibrated estimate."""
    cells = defaultdict(Counter)
    for r in rows:
        for which in ("fatality", "economic"):
            probs, _, src = pager_bins(r, which)
            if probs and src == "alerts.json":
                cells[pager_floor_key(r, which)][pager_vec_key(probs)] += 1
    table = {}
    for k, c in cells.items():
        vec, n = c.most_common(1)[0]
        if n >= min_count and vec[0] >= 0.5:
            table[k] = {"vector": vec, "count": n, "cell_size": sum(c.values())}
    # The same clamp vector recurs across gvalue cells (and across loss types) of one software version, including
    # cells too small to show it as a mode: a vector that is a clamp anywhere in a version is a clamp everywhere in it.
    by_version = defaultdict(set)
    for (ver, _, _), v in table.items():
        by_version[ver].add(v["vector"])
    return table, dict(by_version)


def pager_implied_median(probs, gvalue):
    """Median of the lognormal PAGER fitted, recovered from P(bin 0 = below 1 unit) and its sigma (gvalue)."""
    from statistics import NormalDist
    p0 = min(max(float(probs[0]), 1e-9), 1 - 1e-9)
    return math.exp(-float(gvalue) * NormalDist().inv_cdf(p0))


def pager_is_floor(r, which, probs, floors_by_version):
    """True when the bins are PAGER's stand-in for a ZERO point estimate rather than a fitted loss distribution.

    * any vector that is a version's modal clamp (pager_floor_table);
    * fatality: point estimates are whole deaths, so a fitted distribution has median >= 1; an implied median below
      0.75 means the estimate was 0 and the bins are the default (e.g. median 0.5 at gvalue 1.441: 0.685 / 0.296);
    * economic: the zero-estimate default has median US$0.5 million."""
    ver, _, g = pager_floor_key(r, which)
    if pager_vec_key(probs) in floors_by_version.get(ver, set()):
        return True
    if g is None:
        return False
    m = pager_implied_median(probs, g)
    return m < 0.75 if which == "fatality" else abs(m - 0.5) < 0.02


def pager_state(r):
    pg = r["pager"]
    ev = pg["event"]
    lat, lon = float(ev["lat"]), float(ev["lon"])
    et = parse_iso(ev.get("event_timestamp"))
    quake = {"magnitude": round(float(ev["magnitude"]), 1), "depth_km": round(float(ev["depth"]), 1),
             "epicentre": ev.get("event_description"), "latitude": round(lat, 2), "longitude": round(lon, 2),
             "month": f"{et:%Y-%m}" if et else None}
    lt = (ev.get("localtime") or "").strip()
    m = re.search(r"(\d{1,2}):(\d{2})", lt[-8:] if lt else "")
    if m:
        quake["local_time"] = f"{int(m.group(1)):02d}:{m.group(2)}"
    if ev.get("maxmmi"):
        quake["max_shaking_intensity"] = "MMI " + mmi_label(ev["maxmmi"])
    tsu = pg["root"].get("tsunami") or (r["summary"] or {}).get("tsunami")
    if str(tsu) == "1":
        quake["tsunami_message_issued"] = "yes (a tsunami warning centre message exists for this region)"
    exposure = {}
    for x in pg["exposure"]:
        if "dmin" not in x or "dmax" not in x:      # some products wrap the bins in an attribute-less <exposure>
            continue
        k = int(round((float(x["dmin"]) + float(x["dmax"])) / 2))
        n = int(float(x["exposure"] or 0))
        if k >= 2 and n > 0:
            exposure["MMI " + mmi_label(k)] = f"{n:,} people" + ("" if x.get("rangeInsideMap", "1") == "1"
                                                                  else " (map edge: partial count)")
    state = {"earthquake": quake, "population_exposed_by_shaking_intensity": exposure or "none above MMI II"}
    ej = r.get("exposures_json") or {}
    eco = ej.get("economic_aggregated")
    if eco and len(eco) == 10:
        tot5 = sum(float(v or 0) for v in eco[4:])
        tot7 = sum(float(v or 0) for v in eco[6:])
        state["economic_exposure"] = {"usd_exposed_at_mmi_v_or_more": f"${tot5 / 1e9:,.2f} billion",
                                      "usd_exposed_at_mmi_vii_or_more": f"${tot7 / 1e9:,.2f} billion"}
    # PAGER's code "UK" is "unknown / no country" (open ocean), not the United Kingdom: leave it out
    cc = [c for c in (ej.get("population") or {}).get("country_exposures") or [] if c.get("country_code") != "UK"]
    if len(cc) > 1:
        state["countries_exposed"] = [{"country": c.get("country_code"),
                                       "people_at_mmi_vi_or_more": f"{int(sum(c['exposure'][5:])):,}"}
                                      for c in cc if sum(c.get("exposure") or [0]) > 0][:5]
    cities = [c for c in pg.get("cities") or [] if int(float(c.get("population") or 0)) > 0]
    cities = sorted(cities, key=lambda c: -int(float(c["population"])))[:6]
    if cities:
        state["largest_nearby_cities"] = [{"city": c["name"], "population": f"{int(float(c['population'])):,}",
                                           "shaking": "MMI " + mmi_label(c["mmi"]),
                                           **({"capital": "yes"} if str(c.get("iscapital")) == "1" else {})}
                                          for c in cities]
    ctx = {}
    if pg.get("structcomment"):
        ctx["building_stock"] = re.sub(r"\s+", " ", pg["structcomment"]).strip()
    if pg.get("alertcomment"):
        ctx["past_earthquakes_nearby"] = re.sub(r"\s+", " ", pg["alertcomment"]).strip()
    if pg.get("secondary_effects"):
        ctx["secondary_hazards"] = re.sub(r"\s+", " ", pg["secondary_effects"]).strip()
    if ctx:
        state["regional_context"] = ctx
    return state, et


def build_pager(a, cache):
    recs, drops = [], Counter()
    rows = [r for r in read_gz(cache / "pager_events.jsonl.gz")]
    floors, floors_by_version = pager_floor_table([r for r in rows if r.get("pager")])
    n_floor = Counter()
    for r in rows:
        if r.get("error") or not r.get("pager"):
            drops[f"fetch:{r.get('error')}"] += 1
            continue
        if not r.get("alerts_json"):
            # PAGER 1.0 products (2013 - mid 2017): integer-percent bins and a different minimum-loss convention
            # (99 / 1 / 0 ...), so the same kind of event gets a different label by software era. Dropped.
            drops["pager_v1_product(integer percentages, older loss convention)"] += 1
            continue
        try:
            state, et = pager_state(r)
        except (KeyError, ValueError, TypeError) as e:
            drops[f"state:{type(e).__name__}"] += 1
            continue
        if et is None:
            drops["no_event_time"] += 1
            continue
        fat, fcol, fsrc = pager_bins(r, "fatality")
        eco, ecol, esrc = pager_bins(r, "economic")
        if fat is None or eco is None or sum(fat) <= 0 or sum(eco) <= 0:
            drops["no_alert_bins"] += 1
            continue
        qs, labs, lsrc, lmeth = {}, {}, {}, {}

        def by_colour(p, col):
            v = [0.0] * 4
            for x, c in zip(p, col):
                v[COLOR_IDX.get(c, 0)] += x
            return v
        eid = r["id"]
        f_floor = pager_is_floor(r, "fatality", fat, floors_by_version)
        e_floor = pager_is_floor(r, "economic", eco, floors_by_version)
        keep_floor = hkey("pager_floor", eid) % 100 < a.pager_floor_keep_pct
        floor_q = {}
        if not f_floor or keep_floor:
            qs["fatality_alert_level"] = score_q(PAGER_Q["fatality_alert_level"], FAT_LEVELS)
            labs["fatality_alert_level"] = lab_dict(FAT_LEVELS, by_colour(fat, fcol))
            qs["estimated_fatalities"] = score_q(PAGER_Q["estimated_fatalities"], FAT_BINS)
            labs["estimated_fatalities"] = lab_dict(FAT_BINS, fat)
            floor_q.update({"fatality_alert_level": f_floor, "estimated_fatalities": f_floor})
        if not e_floor or keep_floor:
            qs["economic_alert_level"] = score_q(PAGER_Q["economic_alert_level"], ECO_LEVELS)
            labs["economic_alert_level"] = lab_dict(ECO_LEVELS, by_colour(eco, ecol))
            floor_q["economic_alert_level"] = e_floor
        if not qs:
            drops["minimum_loss_clamp_only(not kept in the floor sample)"] += 1
            continue
        for q in qs:
            src = esrc if q == "economic_alert_level" else fsrc
            lsrc[q] = "published_probability:usgs_pager" + ("_floor" if floor_q[q] else "")
            lmeth[q] = (f"PAGER v{r['pager']['root'].get('version', '?')[:4]} loss-model probabilities ({src}; "
                        "7 order-of-magnitude bins" + ("; summed by alert colour)" if q != "estimated_fatalities" else ")")
                        + ("; THIS VECTOR IS THE VERSION'S MINIMUM-LOSS CLAMP (identical for every negligible-loss "
                           "event with the same gvalue), kept in a tagged sample" if floor_q[q] else ""))
            n_floor[(q, floor_q[q])] += 1
        surface = surface_for(eid)
        text = render(state, surface, "EARTHQUAKE IMPACT ASSESSMENT REQUEST")
        hits = scan("physics/pager", text)
        if hits:
            drops["outcome_word_hit:" + ",".join(hits)] += 1
            continue
        day = et.strftime("%Y-%m-%d")
        sp = split_for(day, eid, a.eval_from)
        pv = r["pager"]["event"]
        meta = {"label_source": lsrc, "label_method": lmeth, "n_raters": None,
                "as_of": r["pager"]["root"].get("process_timestamp"), "event_day": day,
                "pager_version": pv.get("number"), "elapsed_after_event": r["pager"]["root"].get("elapsed"),
                "official_alert": {"fatality": (r["pager"]["alerts_xml"].get("fatality") or {}).get("level"),
                                   "economic": (r["pager"]["alerts_xml"].get("economic") or {}).get("level"),
                                   "comcat": (r["summary"] or {}).get("alert")},
                "usgs_event_id": eid, "surface": surface, "split_reason": f"{sp}:event {day}",
                "probability_source": {"fatality": fsrc, "economic": esrc},
                "pager_software": r["pager"]["root"].get("version"),
                "pager_floor": {"fatality": f_floor, "economic": e_floor},
                "gvalue": {w: ((r.get("alerts_json") or {}).get(w) or {}).get("gvalue") for w in ("fatality", "economic")}}
        recs.append((sp, make_rec("physics/pager", text, qs, labs, meta, eid)))
    return recs, {"records": len(recs), "drops": dict(drops),
                  "questions_by_floor_flag": {f"{q}:{'floor' if f else 'model'}": n for (q, f), n in sorted(n_floor.items())},
                  "floor_vectors_by_version": {k: sorted(v) for k, v in floors_by_version.items()},
                  "floor_table": {f"{k[0]}|{k[1]}|g={k[2]}": v for k, v in sorted(floors.items(), key=str)}}


# --------------------------------------------------------------------------- SPC mesoscale discussions
SPC_Q = ("Will the Storm Prediction Center issue a severe thunderstorm or tornado watch for this area as a result of "
         "the situation described (within the next few hours)?")
WATCH_SENT = re.compile(r"\b(watch\w*|ww|wws|issuance|headlines?|box(es)?|svr|sel\d?)\b", re.I)
SAW_HEAD = re.compile(r"^WW (\d+) (TORNADO|SEVERE TSTM|TEST)\b.*?(\d{6})Z - (\d{6})Z", re.M)


def split_products(txt, head):
    starts = [m.start() for m in re.finditer(head, txt, re.M)]
    return [txt[s:e] for s, e in zip(starts, starts[1:] + [len(txt)])]


def ll_points(block):
    pts = []
    for tok in re.findall(r"\b(\d{8})\b", block):
        la, lo = int(tok[:4]) / 100.0, int(tok[4:]) / 100.0
        if lo < 50:
            lo += 100.0
        pts.append((-lo, la))
    return pts


def parse_saws(month, txt):
    y, m = map(int, month.split("-"))
    out = []
    for prod in split_products(txt, r"^WWUS30 KWNS \d{6}"):
        hm = re.match(r"WWUS30 KWNS (\d{2})(\d{2})(\d{2})", prod)
        h = SAW_HEAD.search(prod)
        if not hm or not h or h.group(2) == "TEST":
            continue
        try:
            t = datetime(y, m, int(hm.group(1)), int(hm.group(2)), int(hm.group(3)), tzinfo=timezone.utc)
        except ValueError:
            continue
        ll = re.search(r"LAT\.\.\.LON([\d\s]+)", prod)
        pts = ll_points(ll.group(1)) if ll else []
        if len(pts) >= 3:
            out.append({"num": int(h.group(1)), "type": "TOR" if h.group(2) == "TORNADO" else "SVR", "issued": t,
                        "poly": pts})
    return out


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside


def poly_overlap_frac(a_poly, b_poly, step=0.1):
    """Share of polygon A's grid points (step deg) that fall inside polygon B."""
    xs, ys = [p[0] for p in a_poly], [p[1] for p in a_poly]
    bx, by = [p[0] for p in b_poly], [p[1] for p in b_poly]
    if max(xs) < min(bx) or min(xs) > max(bx) or max(ys) < min(by) or min(ys) > max(by):
        return 0.0
    n_in = n_both = 0
    y = min(ys)
    while y <= max(ys):
        x = min(xs)
        while x <= max(xs):
            if point_in_poly(x, y, a_poly):
                n_in += 1
                if point_in_poly(x, y, b_poly):
                    n_both += 1
            x += step
        y += step
    return n_both / n_in if n_in else 0.0


def clean_sentences(text):
    t = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<!\.)\.(?!\.)\s+", t)
    keep, dropped = [], 0
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if WATCH_SENT.search(s):
            dropped += 1
            continue
        keep.append(s if s.endswith(".") else s + ".")
    return " ".join(keep), dropped


def parse_mcd(prod):
    out = {}
    m = re.search(r"Mesoscale Discussion (\d+)", prod, re.I)
    if not m:
        return None
    out["num"] = int(m.group(1))
    m = re.search(r"^\s*\d{3,4} (AM|PM) [A-Z]{3,4} \w{3} (\w{3}) (\d{1,2}) (\d{4})", prod, re.M | re.I)
    out["year"] = int(m.group(4)) if m else None
    m = re.search(r"^Areas affected\.\.\.(.+?)\n\s*\n", prod, re.M | re.I | re.S)
    out["areas"] = re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    m = re.search(r"^Concerning\.\.\.(.+)$", prod, re.M | re.I)
    out["concerning_raw"] = m.group(1).strip() if m else None
    m = re.search(r"Probability of Watch Issuance\.\.\.\s*(\d+) percent", prod, re.I)
    out["prob_line"] = int(m.group(1)) if m else None
    m = re.search(r"SUMMARY\.\.\.(.+?)\n\s*\n", prod, re.S | re.I)
    out["summary"] = m.group(1) if m else ""
    m = re.search(r"DISCUSSION\.\.\.(.+?)(?:\n\s*\n\s*\.\.|\n\s*\n\s*\.\.\.Please|\n\s*MOST PROBABLE|\n\s*ATTN|"
                  r"\n\s*LAT\.\.\.LON|$)", prod, re.S | re.I)
    out["discussion"] = m.group(1) if m else ""
    m = re.search(r"^\s*\.\.([^\n]+?)\.\.\s+\d{2}/\d{2}/\d{4}", prod, re.M)
    out["n_forecasters"] = len([x for x in m.group(1).split("/") if x.strip()]) if m else None
    return out


def build_spc(a, cache):
    months = list(read_gz(cache / "spc_months.jsonl.gz"))
    watches = {}
    for mo in months:
        for w in parse_saws(mo["month"], mo["saw_text"]):
            watches.setdefault((w["issued"].year, w["num"]), w)
    wlist = sorted(watches.values(), key=lambda w: w["issued"])
    w_times = [w["issued"] for w in wlist]
    import bisect
    recs, drops, calib = [], Counter(), defaultdict(lambda: [0, 0])
    for mo in months:
        texts = {}
        for prod in split_products(mo["mcd_text"], r"^ACUS11 KWNS \d{6}"):
            p = parse_mcd(prod)
            if p and p["year"]:
                texts.setdefault((p["year"], p["num"]), p)
        for md in mo["mcds"]:
            wc = md.get("watch_confidence")
            if wc is None:
                drops["no_watch_probability"] += 1
                continue
            key = (int(md["year"]), int(md["num"]))
            p = texts.get(key)
            if p is None:
                drops["text_not_found"] += 1
                continue
            if p["prob_line"] is not None and int(p["prob_line"]) != int(float(wc)):
                drops["probability_mismatch_index_vs_text"] += 1
                continue
            t0 = parse_iso(md["issue"])
            geom = md.get("geometry") or {}
            if geom.get("type") != "Polygon" or not t0:
                drops["no_polygon"] += 1
                continue
            poly = [tuple(x) for x in geom["coordinates"][0]]
            j0 = bisect.bisect_right(w_times, t0)
            j1 = bisect.bisect_right(w_times, t0 + timedelta(hours=3))
            hit = None
            for w in wlist[j0:j1]:
                f = poly_overlap_frac(poly, w["poly"])
                if f >= 0.10:
                    hit = {"num": w["num"], "type": w["type"], "minutes_after": round((w["issued"] - t0).total_seconds() / 60),
                           "overlap": round(f, 2)}
                    break
            summ, d1 = clean_sentences(p["summary"])
            disc, d2 = clean_sentences(p["discussion"])
            if len(disc) + len(summ) < 120:
                drops["too_little_text_after_cleaning"] += 1
                continue
            conc = p["concerning_raw"] or ""
            conc = "...".join(x for x in conc.split("...") if x.strip() and not WATCH_SENT.search(x)).strip()
            if len(disc) > 1700:
                disc = disc[:1700].rsplit(" ", 1)[0] + " [...]"
            state = {"mesoscale_discussion": {
                "issued": f"{t0:%Y-%m}, {t0:%H} UTC", "areas_affected": p["areas"],
                **({"concerning": conc} if conc else {}), "summary": summ, "discussion": disc}}
            group = f"{key[0]}-{key[1]:04d}"
            surface = surface_for(group)
            text = render(state, surface, "STORM PREDICTION CENTER MESOSCALE DISCUSSION (forecaster's analysis)")
            hits = scan("physics/spc_mcd", text)
            if hits:
                drops["outcome_word_hit:" + ",".join(hits)] += 1
                continue
            prob = float(wc) / 100.0
            qs = {"watch_issuance": noul_q(SPC_Q)}
            labs = {"watch_issuance": noul_lab(prob)}
            day = t0.strftime("%Y-%m-%d")
            sp = split_for(day, group, a.eval_from)
            calib[int(float(wc))][0] += 1
            calib[int(float(wc))][1] += int(hit is not None)
            meta = {"label_source": {"watch_issuance": "expert_probability:spc_forecaster"},
                    "label_method": {"watch_issuance": "the MD's 'Probability of Watch Issuance' (one of 5/20/40/60/80/95 "
                                                       "percent), stated by the SPC forecaster(s) who wrote it"},
                    "n_raters": p["n_forecasters"], "as_of": md["issue"], "event_day": day,
                    "mcd": {"year": key[0], "num": key[1], "product_id": md["product_id"]},
                    "outcome": {"watch_within_3h_overlapping": hit is not None, "watch": hit},
                    "sentences_removed": d1 + d2, "surface": surface, "split_reason": f"{sp}:issued {day}"}
            recs.append((sp, make_rec("physics/spc_mcd", text, qs, labs, meta, group,
                                      {"watch_issuance": "soft"})))
    rep = {"records": len(recs), "drops": dict(drops), "watches_parsed": len(wlist),
           "calibration_prob_vs_watch_within_3h": {k: {"n": n, "watch_rate": round(h / n, 3)}
                                                  for k, (n, h) in sorted(calib.items())}}
    return recs, rep


def build(a):
    cache, outdir = Path(a.cache), Path(a.out)
    srcs = {s.strip() for s in a.sources.split(",") if s.strip()}
    stems = {"gw": "physics__gw_alert", "pager": "physics__pager", "spc": "physics__spc_mcd"}
    for sp in ("train", "val", "eval"):      # clear only the sources being rebuilt
        for name in srcs:
            p = outdir / sp / f"{stems[name]}.jsonl"
            if p.exists():
                p.unlink()
    mpath = outdir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.update({"built": datetime.now().isoformat(timespec="seconds"), "eval_from": a.eval_from,
                     "lambda": a.lam, "val_pct": VAL_PCT, "pager_floor_keep_pct": a.pager_floor_keep_pct,
                     "licenses": LICENSES, "origins": ORIGINS})
    files = {}

    def write(sp, rec):
        key = (sp, rec["source"])
        if key not in files:
            p = outdir / sp / (rec["source"].replace("/", "__") + ".jsonl")
            p.parent.mkdir(parents=True, exist_ok=True)
            files[key] = p.open("w", encoding="utf-8")
        files[key].write(json.dumps(rec, ensure_ascii=False) + "\n")
    for name, fn, need in (("gw", build_gw, "gw_notices.jsonl.gz"), ("pager", build_pager, "pager_events.jsonl.gz"),
                           ("spc", build_spc, "spc_months.jsonl.gz")):
        if name not in srcs:
            continue
        if not (cache / need).exists():
            log(f"{name}: {need} missing, skipped")
            continue
        recs, rep = fn(a, cache)
        seen = set()
        n = Counter()
        for sp, rec in recs:
            if rec["id"] in seen:
                rep.setdefault("duplicate_ids_dropped", 0)
                rep["duplicate_ids_dropped"] += 1
                continue
            seen.add(rec["id"])
            write(sp, rec)
            n[sp] += 1
        rep["by_split"] = dict(n)
        manifest[name] = rep
        log(f"{name}: {dict(n)}  drops {rep.get('drops')}")
    for fh in files.values():
        fh.close()
    mpath.write_text(json.dumps(manifest, indent=1, default=str))


# =========================================================================== CHECK
def load_cf():
    import importlib.util
    import types
    for mod in ("torch", "transformers"):
        try:
            __import__(mod)
        except ImportError:
            stub = types.ModuleType(mod)
            stub.AutoTokenizer = None
            stub.Tensor = type("Tensor", (), {})
            sys.modules[mod] = stub
    spec = importlib.util.spec_from_file_location("cache_features", HERE / "trainers" / "cache_features.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


def calib_report(P, y):
    import numpy as np
    P = np.clip(np.asarray(P, dtype=float), 1e-9, 1)
    y = np.asarray(y)
    n = len(y)
    ll = float(-np.log(P[np.arange(n), y]).mean())
    acc = float((P.argmax(1) == y).mean())
    Y = np.eye(P.shape[1])[y]
    brier = float(((P - Y) ** 2).sum(1).mean())
    conf = P.max(1)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = float(sum(abs((P.argmax(1) == y)[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                    for b in range(10) if (bins == b).any()))
    return {"n": n, "log_loss": round(ll, 4), "accuracy": round(acc, 4), "brier": round(brier, 4), "ece": round(ece, 4)}


def leak_probe(recs, qid, folds=5, dim=1 << 15, seed=0):
    """Cross-validated hashed bag-of-words multinomial logistic: state -> argmax of the label. Compared with the
    label-prior baseline; the top features are what to read for leaked phrases."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.sparse import csr_matrix
    keys = None
    rows, ys = [], []
    for r in recs:
        lab = r["labels"][qid]
        if keys is None:
            keys = sorted(lab)
        if sorted(lab) != keys:
            continue
        y = keys.index(max(lab, key=lab.get))
        toks = re.findall(r"[a-z][a-z0-9]+", r["state"].lower())
        feats = {int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % dim for t in toks}
        feats |= {int(hashlib.md5((x + "_" + z).encode()).hexdigest()[:8], 16) % dim for x, z in zip(toks, toks[1:])}
        rows.append(sorted(feats))
        ys.append(y)
    n, K = len(ys), len(keys)
    X = csr_matrix((np.ones(sum(len(f) for f in rows)), np.concatenate([np.array(f, dtype=np.int64) for f in rows]),
                    np.cumsum([0] + [len(f) for f in rows])), shape=(n, dim))
    y = np.array(ys)
    fold = np.random.default_rng(seed).integers(0, folds, n)
    P = np.zeros((n, K))
    res = None
    for k in range(folds):
        tr, te = fold != k, fold == k
        Xt, Y = X[tr], np.eye(K)[y[tr]]

        def f(w):
            W, b = w[:-K].reshape(dim, K), w[-K:]
            Z = Xt @ W + b
            Z -= Z.max(1, keepdims=True)
            lse = np.log(np.exp(Z).sum(1))
            Pp = np.exp(Z - lse[:, None])
            obj = -((Z * Y).sum(1) - lse).sum() + 0.5 * (W ** 2).sum()
            g = np.concatenate([(Xt.T @ (Pp - Y) + W).ravel(), (Pp - Y).sum(0)])
            return obj / len(Y), g / len(Y)
        res = minimize(f, np.zeros(dim * K + K), jac=True, method="L-BFGS-B", options={"maxiter": 150})
        W, b = res.x[:-K].reshape(dim, K), res.x[-K:]
        Z = X[te] @ W + b
        Z -= Z.max(1, keepdims=True)
        P[te] = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    prior = np.bincount(y, minlength=K) / n
    vocab = {}
    for r in recs:
        toks = re.findall(r"[a-z][a-z0-9]+", r["state"].lower())
        for t in toks:
            vocab.setdefault(int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % dim, t)
        for x, z in zip(toks, toks[1:]):
            vocab.setdefault(int(hashlib.md5((x + "_" + z).encode()).hexdigest()[:8], 16) % dim, x + " " + z)
    W = res.x[:-K].reshape(dim, K)
    top = {keys[j][:24]: [vocab.get(int(ix), "<bigram>") for ix in np.argsort(-(W[:, j] - W.mean(1)))[:10]]
           for j in range(K) if (y == j).sum() >= 5}
    return {"qid": qid, "n": n, "bow_cv": calib_report(P, y), "label_prior": calib_report(np.tile(prior, (n, 1)), y),
            "top_features": top}


def check(a):
    import numpy as np
    outdir = Path(a.out)
    cf = load_cf()
    report = {}
    recs = defaultdict(list)
    for sp in ("train", "val", "eval"):
        for p in sorted((outdir / sp).glob("physics__*.jsonl")):
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                validate(r)
                for qid, q in r["questions"].items():
                    keys, texts, target = cf.question_options(q, r["labels"][qid])
                    assert keys == option_keys(q), (r["id"], qid)
                    assert [r["labels"][qid][k] for k in keys] == target, (r["id"], qid)
                    assert abs(sum(target) - 1) < 1e-6 and min(target) >= 0, (r["id"], qid)
                recs[sp].append(r)
    units, kinds = Counter(), Counter()
    for u in cf.iter_units(str(outdir), ["train", "val", "eval"], set(), None):
        assert abs(sum(u["target"]) - 1) < 1e-6 and len(u["target"]) == len(u["option_texts"])
        units[(u["split"], u["source"], u["qtype"])] += 1
        kinds[(u["split"], u["source"], u["label_kind"])] += 1
    report["iter_units"] = {"units": {"/".join(k): v for k, v in sorted(units.items())},
                            "label_kind": {"/".join(k): v for k, v in sorted(kinds.items())},
                            "total_units": sum(units.values()),
                            "records": {sp: Counter(r["source"] for r in v) for sp, v in recs.items()}}
    ids = [r["id"] for v in recs.values() for r in v]
    groups = {sp: {(r["source"], r["meta"]["group"]) for r in v} for sp, v in recs.items()}
    states = {sp: {r["state"] for r in v} for sp, v in recs.items()}
    report["dedup"] = {"duplicate_ids": len(ids) - len(set(ids)),
                       "groups_train_and_eval": len(groups.get("train", set()) & groups.get("eval", set())),
                       "groups_train_and_val": len(groups.get("train", set()) & groups.get("val", set())),
                       "groups_val_and_eval": len(groups.get("val", set()) & groups.get("eval", set())),
                       "identical_states_train_and_eval": len(states.get("train", set()) & states.get("eval", set())),
                       "identical_states_train_and_val": len(states.get("train", set()) & states.get("val", set()))}
    hits = Counter()
    for v in recs.values():
        for r in v:
            txt = r["state"]
            if r["source"] == "physics/gw_alert":
                txt = re.sub(r"targeted binary-black-hole search", "", txt)
            for h in scan(r["source"], txt):
                hits[(r["source"], h)] += 1
    report["outcome_word_scan_on_written_states"] = {f"{s}:{h}": n for (s, h), n in hits.items()} or "0 hits"
    # label statistics
    dist = {}
    for sp, v in recs.items():
        acc = defaultdict(lambda: [0.0, 0.0, 0, 0])
        for r in v:
            for qid, lab in r["labels"].items():
                vals = list(lab.values())
                tag = r["meta"]["label_source"].get(qid, "?")
                for key in (r["source"], f"{r['source']}:{qid}", f"{r['source']}|{tag}", "ALL"):
                    acc[key][0] += max(vals)
                    acc[key][1] += entropy(vals)
                    acc[key][2] += 1
                    acc[key][3] += int(sum(1 for x in vals if x > 0) == 1)
        dist[sp] = {k: {"units": n, "mean_max_p": round(m / n, 4), "mean_entropy_nats": round(e / n, 4),
                        "share_zero_entropy": round(z / n, 4)} for k, (m, e, n, z) in sorted(acc.items())}
    report["label_distribution"] = dist
    report["typed_decisions_gold_reference"] = {"mean_max_p": 0.659, "mean_entropy_nats": 0.767}
    # subsets that matter for GW: significant vs low-significance alerts
    sub = defaultdict(lambda: [0.0, 0.0, 0])
    for r in recs["train"] + recs["val"]:
        if r["source"] != "physics/gw_alert":
            continue
        tag = "significant" if r["meta"]["significant"] else "low_significance"
        for qid, lab in r["labels"].items():
            vals = list(lab.values())
            sub[tag][0] += max(vals)
            sub[tag][1] += entropy(vals)
            sub[tag][2] += 1
    report["gw_train_val_by_significance"] = {k: {"units": n, "mean_max_p": round(m / n, 4),
                                                  "mean_entropy_nats": round(e / n, 4)} for k, (m, e, n) in sub.items()}
    # calibration of the published probabilities against later outcomes
    cal = {}
    gw = [r for v in recs.values() for r in v if r["source"] == "physics/gw_alert" and "in_gw_catalog" in r["meta"]["outcome"]]
    if gw:
        P, y = [], []
        for r in gw:
            q = "source_class" if "source_class" in r["labels"] else "source_class_o3"
            pa = 1 - r["labels"][q]["terrestrial"]
            P.append([1 - pa, pa])
            y.append(int(r["meta"]["outcome"]["in_gw_catalog"]))
        cal["gw_online_p_astro_vs_in_gwtc"] = calib_report(P, y)
        rel = defaultdict(lambda: [0, 0.0, 0])
        for (q0, q1), yy in zip(P, y):
            b = min(int(q1 * 5), 4)
            rel[b][0] += 1
            rel[b][1] += q1
            rel[b][2] += yy
        cal["gw_reliability_p_astro_quintiles"] = {f"{b / 5:.1f}-{(b + 1) / 5:.1f}": {"n": n, "mean_p": round(s / n, 3),
                                                                                    "in_gwtc_rate": round(c / n, 3)}
                                                   for b, (n, s, c) in sorted(rel.items())}
    gwr = [r for v in recs.values() for r in v if r["source"] == "physics/gw_alert" and "alert_retracted" in r["meta"]["outcome"]]
    if gwr:
        P, y = [], []
        for r in gwr:
            q = "source_class" if "source_class" in r["labels"] else "source_class_o3"
            pt = r["labels"][q]["terrestrial"]
            P.append([1 - pt, pt])
            y.append(int(r["meta"]["outcome"]["alert_retracted"]))
        cal["gw_online_p_terrestrial_vs_retracted"] = calib_report(P, y)
    spc = [r for v in recs.values() for r in v if r["source"] == "physics/spc_mcd"]
    if spc:
        P = [[1 - r["labels"]["watch_issuance"]["yes"], r["labels"]["watch_issuance"]["yes"]] for r in spc]
        y = [int(r["meta"]["outcome"]["watch_within_3h_overlapping"]) for r in spc]
        cal["spc_forecaster_probability_vs_watch_within_3h"] = calib_report(P, y)
        rel = defaultdict(lambda: [0, 0])
        for (q0, q1), yy in zip(P, y):
            rel[round(q1, 2)][0] += 1
            rel[round(q1, 2)][1] += yy
        cal["spc_reliability"] = {str(k): {"n": n, "watch_rate": round(c / n, 3)} for k, (n, c) in sorted(rel.items())}
    report["calibration_vs_outcomes"] = cal
    # exclusion list (qids)
    try:
        from synth_workflows.exclusion import EXCLUDED_QIDS
        allq = sorted({q for v in recs.values() for r in v for q in r["questions"]})
        report["qids"] = allq
        report["benchmark_qid_collisions"] = sorted(set(allq) & set(EXCLUDED_QIDS)) or "none"
    except Exception as e:  # noqa: BLE001
        report["benchmark_qid_collisions"] = f"could not run: {type(e).__name__}: {e}"
    # leak probes
    rng = random.Random(0)
    probes = {}
    for src, qid in (("physics/gw_alert", "source_class"), ("physics/pager", "fatality_alert_level"),
                     ("physics/spc_mcd", "watch_issuance")):
        pool = [r for r in recs["train"] if r["source"] == src and qid in r["labels"]]
        if len(pool) >= 50:
            probes[f"{src}:{qid}"] = leak_probe(rng.sample(pool, min(a.probe_n, len(pool))), qid)
    report["leak_probe"] = probes
    report["bytes_on_disk"] = sum(p.stat().st_size for p in outdir.rglob("physics__*.jsonl"))
    (outdir / "check_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    samples = []
    for sp, src, cond in (("train", "physics/gw_alert", lambda r: r["meta"]["significant"]),
                          ("eval", "physics/gw_alert", lambda r: True),
                          ("train", "physics/pager", lambda r: r["meta"]["label_source"].get("fatality_alert_level")
                           == "published_probability:usgs_pager"),
                          ("eval", "physics/pager", lambda r: r["meta"]["label_source"].get("economic_alert_level")
                           == "published_probability:usgs_pager"),
                          ("train", "physics/spc_mcd", lambda r: True)):
        pool = [r for r in recs.get(sp, []) if r["source"] == src and cond(r)]
        if pool:
            samples.append(random.Random(len(samples)).choice(pool))
    (outdir / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))


# =========================================================================== self-test (no network)
def self_test():
    import numpy as np
    # FITS: a tiny two-pixel-order multi-order map built by hand
    def card(k, v):
        s = f"{k:<8}= {v:>20}" if not isinstance(v, str) else f"{k:<8}= '{v:<8}'"
        return s.ljust(80).encode()

    def block(cards):
        b = b"".join(cards) + b"END".ljust(80)
        return b + b" " * ((2880 - len(b) % 2880) % 2880)
    uniq = np.array([4 * 4 ** 1 + i for i in range(48)], dtype=">i8")          # order 1: 48 pixels
    dens = np.zeros(48, dtype=">f8")
    pix_area = 4 * math.pi / 48
    dens[:4] = 0.225 / pix_area     # 90% of the probability in 4 pixels
    dens[4:] = (0.1 / 44) / pix_area
    tab = np.zeros(48, dtype=[("UNIQ", ">i8"), ("PROBDENSITY", ">f8")])
    tab["UNIQ"], tab["PROBDENSITY"] = uniq, dens
    data = tab.tobytes()
    data += b"\0" * ((2880 - len(data) % 2880) % 2880)
    hdr0 = block([card("SIMPLE", "T"), card("BITPIX", 8), card("NAXIS", 0)])
    hdr1 = block([card("XTENSION", "BINTABLE"), card("BITPIX", 8), card("NAXIS", 2), card("NAXIS1", 16),
                  card("NAXIS2", 48), card("PCOUNT", 0), card("GCOUNT", 1), card("TFIELDS", 2),
                  card("TTYPE1", "UNIQ"), card("TFORM1", "K"), card("TTYPE2", "PROBDENSITY"), card("TFORM2", "D"),
                  card("DISTMEAN", 100.5)])
    s = skymap_summary(hdr0 + hdr1 + data)
    assert abs(s["area90_deg2"] - 4 * pix_area * (180 / math.pi) ** 2) < 0.01, s
    assert abs(s["prob_total"] - 1) < 1e-9 and s["distmean"] == 100.5
    # labels
    d = lab_dict(["a", "b", "c"], [0.1234567, 0.2, 0.6765433])
    assert abs(sum(d.values()) - 1) < 1e-12 and kind_of(noul_lab(1.0)) == "hard" and kind_of(noul_lab(0.3)) == "soft"
    # LOO rates never use an item's own outcome
    p, k = loo_cell_rates(["x", "x", "y", "y"], [1, 0, 0, 0], [True, True, True, False])
    assert p[0] < 0.5 and p[3] == p[2] or p[3] > 0
    # SPC text cleaning removes every watch sentence
    txt = ("Storms will intensify. A watch is likely by 22Z. Large hail possible. New WW will be needed soon. "
           "Severe weather headlines may be required.")
    c, n = clean_sentences(txt)
    assert c == "Storms will intensify. Large hail possible." and n == 3, (c, n)
    assert ll_points("32549449 30010211") == [(-94.49, 32.54), (-102.11, 30.01)]
    sq = [(0, 0), (0, 1), (1, 1), (1, 0)]
    assert poly_overlap_frac(sq, [(0.5, -1), (0.5, 2), (3, 2), (3, -1)], step=0.05) > 0.4
    assert poly_overlap_frac(sq, [(5, 5), (5, 6), (6, 6)], step=0.05) == 0
    # scans
    assert scan("physics/gw_alert", "pipeline gstlal, BNS candidate") == ["bns"]
    assert scan("physics/pager", "Orange alert for shaking-related fatalities") == ["orange alert"]
    assert scan("physics/pager", "resulting in a reported 7,079 fatalities.") == []
    assert scan("physics/spc_mcd", "Trends will be monitored for watch issuance") == ["issuance", "watch"]
    assert far_text(1 / 3.15576e7) == "1 per year" or far_text(1 / 3.15576e7).startswith("1")
    assert run_of(datetime(2023, 6, 1, tzinfo=timezone.utc)) == "O4a"
    print("self-test OK")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fetch")
    f.add_argument("--cache", default="/tmp/physics_jev/cache")
    f.add_argument("--sources", default="gw,pager,spc")
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--gw-workers", type=int, default=4)
    f.add_argument("--usgs-workers", type=int, default=6)
    f.add_argument("--iem-workers", type=int, default=3)
    f.add_argument("--spc-from", type=int, default=2012, help="first year of SPC mesoscale discussions")
    b = sub.add_parser("build")
    b.add_argument("--cache", default="/tmp/physics_jev/cache")
    b.add_argument("--out", default="/tmp/physics_jev")
    b.add_argument("--sources", default="gw,pager,spc")
    b.add_argument("--eval-from", default=EVAL_FROM, help="events on/after this day are EVAL")
    b.add_argument("--lam", type=float, default=LAMBDA, help="outcome blend weight on the one-hot (train/val)")
    b.add_argument("--pager-floor-keep-pct", type=int, default=10,
                   help="share (by event-id hash) of PAGER minimum-loss-clamp questions kept, tagged _floor")
    c = sub.add_parser("check")
    c.add_argument("--out", default="/tmp/physics_jev")
    c.add_argument("--probe-n", type=int, default=3000)
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.cmd == "fetch":
        cache = Path(a.cache)
        cache.mkdir(parents=True, exist_ok=True)
        srcs = {s.strip() for s in a.sources.split(",") if s.strip()}
        if "gw" in srcs:
            fetch_gw(a, cache)
        if "pager" in srcs:
            fetch_pager(a, cache)
        if "spc" in srcs:
            fetch_spc(a, cache)
    elif a.cmd == "build":
        build(a)
    elif a.cmd == "check":
        check(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
