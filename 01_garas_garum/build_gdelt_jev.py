#!/usr/bin/env python3
"""
build_gdelt_jev.py -- GDELT 2.0 event panel + forecasting decision questions for jev (train / val / test / test_recent).

Docs: research/gdelt.md. Licence: GDELT terms -- "unlimited and unrestricted use for any academic, commercial, or
governmental use of any kind without fee"; "Any use or redistribution of the data must include a citation to the GDELT
Project and a link to this website (https://www.gdeltproject.org/)." -> CLEAN line, with attribution in every record.
Only GDELT's structured fields are used. The linked news articles (SOURCEURL) are never fetched.

STAGES (subcommands)
  panel      download the GDELT 2.0 English export stream (15-minute files in masterfilelist.txt, md5-checked) and aggregate
             each UTC day by DATEADDED into: country x EventRootCode x QuadClass (ActionGeo_CountryCode, FIPS 10-4) and
             directed actor-country dyads (Actor1CountryCode -> Actor2CountryCode, CAMEO). Dedup by GLOBALEVENTID within
             the day; ids kept per day for the cross-day check. Streams: no raw file is kept. Resumable per day.
             CPU node only (x11 head), nice -n 10, <= half the cores.
  merge      cross-day GLOBALEVENTID duplicate check (+ re-aggregation of affected days without the duplicates),
             concatenation into the final parquet panel, top-200 dyads (by train-period volume), GDELT lookups, manifest.
  questions  the four question families from the merged panel (numpy core; pandas only to read parquet):
               F1 gdelt_f1_count     score: next-week event count of root code r in country c, ~10 trailing-year decile bins
               F2 gdelt_f2_exceed    noul : next-week count > trailing-52-week 90th percentile
               F3 gdelt_f3_goldstein choice: next 28 days' mean Goldstein (country or dyad) up / same / down, stated tolerance
               F4 gdelt_f4_tone      score: next-week mean AvgTone minus the previous 4 weeks' mean, 5 trailing-quantile bins
             Forecast origins are Mondays 00:00 UTC; a state uses ONLY weeks strictly before the origin; labels are the
             realised outcome (hard); baselines (climatology, seasonal naive, persistence, negative binomial / Gaussian)
             are computed from pre-origin data only and stored in meta.baselines.
             Splits by origin, with the whole resolution window inside the split:
               train <= 2025-06-30 | val 2025-07-01..2025-12-31 | test 2026-01-01..2026-05-31 | test_recent 2026-06-01..
             (test and test_recent are disjoint; their union is the full 2026 test period.)
  validate   read-back through trainers/cache_features.question_options (torch / transformers stubbed if missing).

Targets are "what GDELT will record" (machine-coded counts from news coverage), not ground truth about the world.

  python3 build_gdelt_jev.py panel --root /data/local/gdelt --start 20150218 --end 20261003 --workers 4
  python3 build_gdelt_jev.py merge --root /data/local/gdelt
  python3 build_gdelt_jev.py questions --panel /data/local/gdelt/panel --out /data/local/gdelt/jev_gdelt_v1
  python3 build_gdelt_jev.py validate --data /data/local/gdelt/jev_gdelt_v1
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VERSION = "gdelt_jev v1 (2026-10-04)"
SEED = 20261004

LICENSE = "GDELT terms (unrestricted use; cite GDELT + link)"
CITATION = ("The GDELT Project (https://www.gdeltproject.org/). GDELT 2.0 Event Database, English export stream "
            "(data.gdeltproject.org/gdeltv2/), accessed 2026-10-04. Leetaru, K. and Schrodt, P. A. (2013). GDELT: Global "
            "Data on Events, Location and Tone, 1979-2012. ISA Annual Convention.")
LICENSE_QUOTE = ("All datasets released by the GDELT Project are available for unlimited and unrestricted use for any "
                 "academic, commercial, or governmental use of any kind without fee. ... Any use or redistribution of the "
                 "data must include a citation to the GDELT Project and a link to this website "
                 "(https://www.gdeltproject.org/). [gdeltproject.org/about.html]")

MASTER_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
LOOKUP_URLS = {n: f"https://www.gdeltproject.org/data/lookups/{n}" for n in
               ("CAMEO.eventcodes.txt", "FIPS.country.txt", "CAMEO.country.txt", "CAMEO.goldsteinscale.txt",
                "CAMEO.type.txt", "CAMEO.knowngroup.txt")}

# GDELT 2.0 export columns we read (0-based; the file has 61 tab-separated columns, no header)
EXPORT_COLS = {0: "id", 1: "sqldate", 7: "a1cc", 17: "a2cc", 25: "isroot", 28: "root", 29: "quad", 30: "gold",
               31: "mentions", 32: "sources", 33: "articles", 34: "tone", 53: "cc", 59: "dateadded"}
N_EXPORT_COLS = 61

ROOT_NAMES = {   # CAMEO root codes (CAMEO.eventcodes.txt); fallback when the lookup file is absent
    "01": "Make public statement", "02": "Appeal", "03": "Express intent to cooperate", "04": "Consult",
    "05": "Engage in diplomatic cooperation", "06": "Engage in material cooperation", "07": "Provide aid", "08": "Yield",
    "09": "Investigate", "10": "Demand", "11": "Disapprove", "12": "Reject", "13": "Threaten", "14": "Protest",
    "15": "Exhibit force posture", "16": "Reduce relations", "17": "Coerce", "18": "Assault", "19": "Fight",
    "20": "Use unconventional mass violence"}
QUAD_NAMES = {1: "Verbal Cooperation", 2: "Material Cooperation", 3: "Verbal Conflict", 4: "Material Conflict"}
# GDELT 2.0 QuadClass by root code, as observed in the data (merge stats.root_quad_counts): 01-05 verbal cooperation,
# 06-09 material cooperation (09 Investigate included), 10-14 verbal conflict (14 Protest included), 15-20 material conflict
ROOT_QUAD = {f"{i:02d}": (1 if i <= 5 else 2 if i <= 9 else 3 if i <= 14 else 4) for i in range(1, 21)}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sha(s, n=16):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:n]


def hash01(*parts):
    """Uniform [0, 1) from the parts: seeded, order-independent selection."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:13], 16) / float(16 ** 13)


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# =========================================================================== PANEL (node)
def read_masterlist(path):
    """{YYYYMMDD: [(ts14, size, md5, https_url), ...]} for the English export stream."""
    days = defaultdict(list)
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) != 3 or not p[2].endswith(".export.CSV.zip"):
                continue
            name = p[2].rsplit("/", 1)[1]
            ts = name.split(".")[0]
            if len(ts) != 14 or not ts.isdigit():
                continue
            days[ts[:8]].append((ts, int(p[0]), p[1], p[2].replace("http://", "https://", 1)))
    for d in days:
        days[d].sort()
    return days


def _fetch(session, url, md5, tries=4):
    import requests  # noqa: F401  (node only)
    last = None
    for k in range(tries):
        try:
            r = session.get(url, timeout=90)
            if r.status_code == 404:
                return None, "404"
            r.raise_for_status()
            b = r.content
            if hashlib.md5(b).hexdigest() != md5:
                last = "md5"
                time.sleep(2 * (k + 1))
                continue
            return b, "ok"
        except Exception as e:  # noqa: BLE001 -- network: retry then record
            last = type(e).__name__
            time.sleep(2 * (k + 1))
    return None, last or "fail"


def parse_export(raw_zip):
    """One 15-minute export zip -> (DataFrame of EXPORT_COLS, raw line count)."""
    import csv
    import zipfile

    import pandas as pd
    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    data = z.read(z.namelist()[0])
    n_lines = data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)
    if not data.strip():
        return None, 0
    cols = sorted(EXPORT_COLS)
    df = pd.read_csv(io.BytesIO(data), sep="\t", header=None, usecols=cols, dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip",
                     engine="c")
    df.columns = [EXPORT_COLS[c] for c in cols]
    return df, n_lines


def aggregate_day(df, day):
    """A day's deduplicated events -> (country rows, dyad rows, stats). Pure pandas; used by the node and the tests."""
    import pandas as pd
    out = {}
    for c in ("gold", "tone", "mentions", "sources", "articles"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["quad"] = pd.to_numeric(df["quad"], errors="coerce").fillna(0).astype("int8")
    df["isroot"] = (df["isroot"] == "1").astype("int32")
    for c in ("mentions", "sources", "articles"):
        df[c] = df[c].fillna(0)
    df["gold_ok"] = df["gold"].notna().astype("int32")
    df["tone_ok"] = df["tone"].notna().astype("int32")
    df["gold0"] = df["gold"].fillna(0.0)
    df["tone0"] = df["tone"].fillna(0.0)
    df["gold2"] = df["gold0"] ** 2
    df["tone2"] = df["tone0"] ** 2
    agg = dict(n=("id", "size"), n_root=("isroot", "sum"), mentions=("mentions", "sum"), sources=("sources", "sum"),
               articles=("articles", "sum"), gold_n=("gold_ok", "sum"), gold_sum=("gold0", "sum"),
               gold_sq=("gold2", "sum"), tone_n=("tone_ok", "sum"), tone_sum=("tone0", "sum"), tone_sq=("tone2", "sum"))
    g = df.groupby(["cc", "root", "quad"], sort=True).agg(**agg).reset_index()
    g.insert(0, "date", int(day))
    out["country"] = g
    dd = df[(df["a1cc"] != "") & (df["a2cc"] != "")].copy()
    for q in (1, 2, 3, 4):
        dd[f"q{q}"] = (dd["quad"] == q).astype("int32")
    agg2 = dict(agg)
    agg2.pop("sources"); agg2.pop("articles")
    agg2.update({f"n_q{q}": (f"q{q}", "sum") for q in (1, 2, 3, 4)})
    h = dd.groupby(["a1cc", "a2cc"], sort=True).agg(**agg2).reset_index()
    h.insert(0, "date", int(day))
    out["dyad"] = h
    return out


def process_day(day, files, root, threads=8, exclude_ids=None):
    """Download + parse + dedup + aggregate one UTC day; writes days/<YYYY>/<day>.{country,dyad}.parquet, ids, cov json."""
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root)
    ddir = root / "days" / day[:4]
    ddir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=threads, pool_maxsize=threads))
    s.headers["User-Agent"] = "jev-gdelt-panel/1.0 (research; structured event fields only)"

    def get(f):
        ts, size, md5, url = f
        b, st = _fetch(s, url, md5)
        return f, b, st

    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(get, files))
    frames, file_stats = [], []
    n_lines = 0
    nbytes = 0
    for i, (f, b, st) in enumerate(res):
        ts, size, md5, url = f
        fs = {"ts": ts, "bytes": size, "status": st, "rows": 0, "id_min": None, "id_max": None, "lines": 0}
        if b is not None:
            nbytes += len(b)
            try:
                df, nl = parse_export(b)
            except Exception as e:  # noqa: BLE001 -- a corrupt file is recorded, not fatal
                fs["status"] = f"parse:{type(e).__name__}"
                df, nl = None, 0
            fs["lines"] = nl
            n_lines += nl
            if df is not None and len(df):
                df["fileidx"] = i
                ids = pd.to_numeric(df["id"], errors="coerce")
                df = df[ids.notna()].copy()
                df["id"] = ids[ids.notna()].astype("int64")
                fs["rows"] = int(len(df))
                fs["id_min"], fs["id_max"] = int(df["id"].min()), int(df["id"].max())
                frames.append(df)
        file_stats.append(fs)
    cov = {"date": day, "files_listed": len(files), "files_ok": sum(1 for x in file_stats if x["status"] == "ok"),
           "file_status": dict(Counter(x["status"] for x in file_stats)), "bytes_zip": nbytes, "lines_raw": n_lines}
    if frames:
        df = pd.concat(frames, ignore_index=True)
        cov["rows_parsed"] = int(len(df))
        df = df.sort_values(["fileidx"], kind="stable").drop_duplicates("id", keep="first")
        cov["dup_within_day"] = cov["rows_parsed"] - int(len(df))
        if exclude_ids is not None and len(exclude_ids):
            m = df["id"].isin(set(int(x) for x in exclude_ids))
            cov["excluded_cross_day"] = int(m.sum())
            df = df[~m]
        cov["rows_dedup"] = int(len(df))
        cov["dateadded_mismatch"] = int((df["dateadded"].str[:8] != day).sum())
        sq = pd.to_datetime(df["sqldate"], format="%Y%m%d", errors="coerce")
        dday = pd.Timestamp(datetime.strptime(day, "%Y%m%d"))
        lag = (dday - sq).dt.days
        cov["sqldate_lag_gt1"] = int((lag > 1).sum())
        cov["sqldate_lag_gt30"] = int((lag > 30).sum())
        cov["sqldate_future"] = int((lag < 0).sum())
        cov["sqldate_bad"] = int(sq.isna().sum())
        cov["cc_empty"] = int((df["cc"] == "").sum())
        cov["quad_root_mismatch"] = int(sum(1 for r, q in zip(df["root"], df["quad"])
                                            if ROOT_QUAD.get(r) is not None and str(ROOT_QUAD[r]) != q))
        ids = np.sort(df["id"].to_numpy(dtype=np.int64))
        cov["id_min"], cov["id_max"] = (int(ids[0]), int(ids[-1])) if len(ids) else (None, None)
        if len(ids):
            dif = np.diff(ids)
            np.savez_compressed(ddir / f"{day}.ids.npz", first=ids[:1], diff=dif.astype(np.uint32)
                                if (len(dif) == 0 or dif.max() < 2 ** 32) else dif)
        agg = aggregate_day(df, day)
        agg["country"].to_parquet(ddir / f"{day}.country.parquet", index=False)
        agg["dyad"].to_parquet(ddir / f"{day}.dyad.parquet", index=False)
        cov["country_rows"], cov["dyad_rows"] = int(len(agg["country"])), int(len(agg["dyad"]))
    else:
        cov.update(rows_parsed=0, rows_dedup=0, dup_within_day=0)
    cov["secs"] = round(time.time() - t0, 2)
    cov["files"] = file_stats
    tmp = ddir / f"{day}.cov.json.tmp"
    tmp.write_text(json.dumps(cov))
    os.replace(tmp, ddir / f"{day}.cov.json")       # the done-marker: written last, atomically
    return {k: v for k, v in cov.items() if k != "files"}


def _day_job(args):
    day, files, root, threads, excl = args
    try:
        return process_day(day, files, root, threads, excl)
    except Exception as e:  # noqa: BLE001 -- logged; the day stays undone and a re-run retries it
        return {"date": day, "error": f"{type(e).__name__}: {e}"}


def cmd_panel(a):
    from multiprocessing import Pool
    root = Path(a.root)
    (root / "days").mkdir(parents=True, exist_ok=True)
    ml = root / "masterfilelist.txt"
    if a.refresh_master or not ml.exists():
        import requests
        r = requests.get(MASTER_URL, timeout=300)
        r.raise_for_status()
        ml.write_bytes(r.content)
    days = read_masterlist(ml)
    want = sorted(d for d in days if a.start <= d <= a.end)
    todo = [d for d in want if not (root / "days" / d[:4] / f"{d}.cov.json").exists()]
    if a.limit_days:
        todo = todo[: a.limit_days]
    log(f"panel: {len(want)} days in [{a.start}, {a.end}], {len(todo)} to do; "
        f"{sum(len(days[d]) for d in todo)} files, {sum(f[1] for d in todo for f in days[d]) / 1e9:.2f} GB")
    t0 = time.time()
    n_done, nbytes, nrows = 0, 0, 0
    with Pool(a.workers) as pool:
        for cov in pool.imap_unordered(_day_job, [(d, days[d], str(root), a.threads, None) for d in todo]):
            n_done += 1
            if "error" in cov:
                log(f"  ERROR {cov['date']}: {cov['error']}")
                continue
            nbytes += cov.get("bytes_zip", 0)
            nrows += cov.get("rows_dedup", 0)
            if n_done % 25 == 0 or cov["files_ok"] < cov["files_listed"]:
                el = time.time() - t0
                log(f"  {n_done}/{len(todo)} days  last {cov['date']} files {cov['files_ok']}/{cov['files_listed']} "
                    f"rows {cov.get('rows_dedup')}  total {nbytes / 1e9:.2f} GB {nrows / 1e6:.1f}M rows  "
                    f"{el / 60:.1f} min  eta {(len(todo) - n_done) * el / max(n_done, 1) / 60:.1f} min")
    log(f"panel done: {n_done} days, {nbytes / 1e9:.2f} GB downloaded, {nrows / 1e6:.1f}M rows, "
        f"{(time.time() - t0) / 60:.1f} min")
    (root / "logs").mkdir(exist_ok=True)
    with open(root / "logs" / "panel_runs.jsonl", "a") as f:
        f.write(json.dumps({"start": a.start, "end": a.end, "days": n_done, "bytes": nbytes, "rows": nrows,
                            "secs": round(time.time() - t0, 1), "workers": a.workers, "threads": a.threads,
                            "finished": datetime.now(timezone.utc).isoformat()}) + "\n")


# =========================================================================== MERGE (node)
TRAIN_END = date(2025, 6, 30)
COUNTRY_SCHEMA = {"date": "int32", "cc": "str", "root": "str", "quad": "int8", "n": "int64", "n_root": "int64",
                  "mentions": "float64", "sources": "float64", "articles": "float64", "gold_n": "int64",
                  "gold_sum": "float64", "gold_sq": "float64", "tone_n": "int64", "tone_sum": "float64",
                  "tone_sq": "float64"}
DYAD_SCHEMA = {"date": "int32", "a1cc": "str", "a2cc": "str", "n": "int64", "n_root": "int64", "mentions": "float64",
               "gold_n": "int64", "gold_sum": "float64", "gold_sq": "float64", "tone_n": "int64", "tone_sum": "float64",
               "tone_sq": "float64", "n_q1": "int64", "n_q2": "int64", "n_q3": "int64", "n_q4": "int64"}


def load_ids(path):
    z = np.load(path)
    return np.concatenate([z["first"].astype(np.int64), z["first"][0] + np.cumsum(z["diff"].astype(np.int64))]) \
        if len(z["first"]) else np.zeros(0, np.int64)


def cross_day_dups(root, days):
    """{day: ids also present in an EARLIER day} -- GLOBALEVENTIDs are near-monotone, so only ids <= the running max of
    earlier days can be duplicates, and only days whose [min, max] range covers them need loading."""
    root = Path(root)
    ranges, cache, out = [], {}, {}
    run_max = -1

    def ids_of(d):
        if d not in cache:
            p = root / "days" / d[:4] / f"{d}.ids.npz"
            cache[d] = load_ids(p) if p.exists() else np.zeros(0, np.int64)
            if len(cache) > 40:
                cache.pop(next(iter(cache)))
        return cache[d]

    for d in days:
        ids = ids_of(d)
        if not len(ids):
            continue
        cand = ids[ids <= run_max]
        if len(cand):
            lo, hi = cand.min(), cand.max()
            hit = []
            for (e, emin, emax) in ranges:
                if emax >= lo and emin <= hi:
                    hit.append(np.intersect1d(cand, ids_of(e), assume_unique=True))
            hit = np.unique(np.concatenate(hit)) if hit else np.zeros(0, np.int64)
            if len(hit):
                out[d] = hit
        ranges.append((d, int(ids.min()), int(ids.max())))
        run_max = max(run_max, int(ids.max()))
    return out


def _cast(df, schema):
    for c, t in schema.items():
        df[c] = df[c].astype(t if t != "str" else object)
    return df[list(schema)]


def _concat_parquet(paths, out_path, schema, keep=None):
    import pandas as pd  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq
    w, n = None, 0
    for p in paths:
        if not p.exists():
            continue
        df = _cast(pq.read_table(p).to_pandas(), schema)
        if keep is not None:
            df = keep(df)
        t = pa.Table.from_pandas(df, preserve_index=False)
        if w is None:
            w = pq.ParquetWriter(out_path, t.schema, compression="zstd")
        w.write_table(t.cast(w.schema))
        n += len(df)
    if w is not None:
        w.close()
    return n


def read_lookup(path):
    """GDELT lookup txt -> {code: label}; skips a header row."""
    out = {}
    if not Path(path).exists():
        return out
    for i, ln in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 2 or not p[0].strip():
            continue
        if i == 0 and (p[0].upper().startswith("CODE") or p[0].upper().startswith("CAMEO")):
            continue
        out[p[0].strip()] = p[1].strip()
    return out


def cmd_merge(a):
    import pandas as pd
    import requests
    t0 = time.time()
    root = Path(a.root)
    out = root / "panel"
    (out / "lookups").mkdir(parents=True, exist_ok=True)
    ml = read_masterlist(root / "masterfilelist.txt")
    want = sorted(d for d in ml if a.start <= d <= a.end)
    missing = [d for d in want if not (root / "days" / d[:4] / f"{d}.cov.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} days not done yet (first {missing[:5]}); re-run `panel`")
    # ---- cross-day duplicate GLOBALEVENTIDs: drop the LATER copy by re-aggregating that day without it
    dups = cross_day_dups(root, want)
    n_cross = int(sum(len(v) for v in dups.values()))
    log(f"cross-day duplicate ids: {n_cross} in {len(dups)} days")
    if dups and not a.no_fix:
        for d, ids in sorted(dups.items()):
            cov = process_day(d, ml[d], root, a.threads, exclude_ids=ids)
            log(f"  re-aggregated {d} without {len(ids)} cross-day duplicates -> {cov.get('rows_dedup')} rows")
        left = cross_day_dups(root, want)
        log(f"cross-day duplicates after the fix: {int(sum(len(v) for v in left.values()))}")
    covs = [json.loads((root / "days" / d[:4] / f"{d}.cov.json").read_text()) for d in want]
    files = []
    for c in covs:
        for f in c.pop("files"):
            files.append({"date": int(c["date"]), **f})
        c["file_status"] = json.dumps(c["file_status"], sort_keys=True)
    cov = pd.DataFrame(covs)
    # the panel run's per-day quad_root_mismatch used an earlier, wrong root->QuadClass map (06-08 / 09-13 / 14-20); the
    # real mapping is read off root_quad_counts below, so the stale column is dropped rather than published
    cov = cov.drop(columns=[c for c in ("quad_root_mismatch",) if c in cov.columns])
    cov["date"] = cov["date"].astype("int32")
    cov.to_parquet(out / "coverage_daily.parquet", index=False)
    pd.DataFrame(files).to_parquet(out / "coverage_files.parquet", index=False)
    # ---- country x root x quad
    cpaths = [root / "days" / d[:4] / f"{d}.country.parquet" for d in want]
    n_country = _concat_parquet(cpaths, out / "country_daily.parquet", COUNTRY_SCHEMA)
    log(f"country_daily: {n_country} rows")
    # ---- dyads: full, then the top 200 by TRAIN-period (<= 2025-06-30) volume
    dpaths = [root / "days" / d[:4] / f"{d}.dyad.parquet" for d in want]
    n_dyad = _concat_parquet(dpaths, out / "dyad_daily_full.parquet", DYAD_SCHEMA)
    tot = Counter()
    tot_all = Counter()
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(out / "dyad_daily_full.parquet")
    tr_end = int(TRAIN_END.strftime("%Y%m%d"))
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=["date", "a1cc", "a2cc", "n"]).to_pandas()
        g = df.groupby(["a1cc", "a2cc"])["n"].sum()
        tot_all.update(dict(g))
        g2 = df[df["date"] <= tr_end].groupby(["a1cc", "a2cc"])["n"].sum()
        tot.update(dict(g2))
    top = [k for k, _ in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[: a.top_dyads]]
    topset = set(top)
    n_top = _concat_parquet(dpaths, out / "dyad_daily_top200.parquet", DYAD_SCHEMA,
                            keep=lambda df: df[[(x, y) in topset for x, y in zip(df["a1cc"], df["a2cc"])]])
    # ---- lookups
    for name, url in LOOKUP_URLS.items():
        p = out / "lookups" / name
        if not p.exists():
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                p.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                log(f"  lookup {name}: {type(e).__name__}")
    cameo = read_lookup(out / "lookups" / "CAMEO.country.txt")
    pd.DataFrame([{"rank": i + 1, "a1cc": x, "a2cc": y, "a1_name": cameo.get(x, x), "a2_name": cameo.get(y, y),
                   "n_train_period": int(tot[(x, y)]), "n_all": int(tot_all[(x, y)])}
                  for i, (x, y) in enumerate(top)]).to_csv(out / "dyad_top200.csv", index=False)
    log(f"dyads: {n_dyad} rows full ({len(tot_all)} pairs), top {len(top)} -> {n_top} rows")
    # ---- summary statistics for the manifest / research doc
    cd = pq.read_table(out / "country_daily.parquet", columns=["date", "cc", "root", "quad", "n"]).to_pandas()
    cd["year"] = cd["date"] // 10000
    rq = cd.groupby(["root", "quad"])["n"].sum().reset_index()
    by_year = cd.groupby("year")["n"].sum().to_dict()
    by_cc = cd[cd["cc"] != ""].groupby("cc")["n"].sum().sort_values(ascending=False)
    fips = read_lookup(out / "lookups" / "FIPS.country.txt")
    runs = [json.loads(x) for x in (root / "logs" / "panel_runs.jsonl").read_text().splitlines() if x.strip()] \
        if (root / "logs" / "panel_runs.jsonl").exists() else []
    stats = {
        "days": len(want), "date_range": [want[0], want[-1]],
        "files_listed": int(cov["files_listed"].sum()), "files_ok": int(cov["files_ok"].sum()),
        "file_status": dict(sum((Counter(json.loads(s)) for s in cov["file_status"]), Counter())),
        "bytes_zip_downloaded": int(cov["bytes_zip"].sum()),
        "rows_parsed": int(cov["rows_parsed"].sum()), "rows_dedup": int(cov["rows_dedup"].sum()),
        "dup_within_day": int(cov["dup_within_day"].sum()), "dup_cross_day_removed": n_cross,
        "lines_raw": int(cov["lines_raw"].sum()),
        "dateadded_mismatch": int(cov["dateadded_mismatch"].fillna(0).sum()),
        "sqldate_lag_gt1": int(cov["sqldate_lag_gt1"].fillna(0).sum()),
        "sqldate_lag_gt30": int(cov["sqldate_lag_gt30"].fillna(0).sum()),
        "sqldate_future": int(cov["sqldate_future"].fillna(0).sum()),
        "actiongeo_cc_empty": int(cov["cc_empty"].fillna(0).sum()),
        "days_files_lt_90pct": [str(x) for x in cov.loc[cov["files_ok"] < 0.9 * 96, "date"].tolist()],
        "events_by_year": {str(k): int(v) for k, v in by_year.items()},
        "n_country_codes": int((by_cc > 0).sum()),
        "top_countries": [{"cc": k, "name": fips.get(k, k), "n": int(v)} for k, v in by_cc.head(25).items()],
        "root_quad_counts": [{"root": r, "quad": int(q), "n": int(n)} for r, q, n in rq.itertuples(index=False)],
        "root_quad_off_map": int(sum(int(n) for r, q, n in rq.itertuples(index=False) if ROOT_QUAD.get(r) != int(q))),
        "panel_runs": runs,
    }
    manifest = {"version": VERSION, "built": datetime.now(timezone.utc).isoformat(), "license": LICENSE,
                "license_quote": LICENSE_QUOTE, "citation": CITATION, "stats": stats,
                "files": {}, "rows": {"country_daily": n_country, "dyad_daily_full": n_dyad,
                                      "dyad_daily_top200": n_top, "coverage_daily": len(cov),
                                      "coverage_files": len(files)},
                "merge_secs": None}
    (out / "README.md").write_text(PANEL_README.format(citation=CITATION, license=LICENSE, quote=LICENSE_QUOTE,
                                                       start=want[0], end=want[-1]))
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest["files"][str(p.relative_to(out))] = {"bytes": p.stat().st_size, "md5": md5_file(p)}
    manifest["merge_secs"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"merge done in {manifest['merge_secs']} s -> {out}")


PANEL_README = """# GDELT 2.0 daily event panel (jev, v1)

Built by `build_gdelt_jev.py panel` + `merge` from the GDELT 2.0 Event Database English export stream
(15-minute files listed in https://data.gdeltproject.org/gdeltv2/masterfilelist.txt), UTC days {start}..{end}.

**Citation (required by the GDELT terms):** {citation}
**Licence:** {license}. "{quote}"

Only GDELT's structured fields are used (codes, counts, Goldstein scale, tone). No news article was fetched.
Counts measure what GDELT recorded from news coverage, not what happened in the world.

## Files
- `country_daily.parquet`: per UTC day (by DATEADDED) x ActionGeo_CountryCode (FIPS 10-4; '' = no geolocation) x
  EventRootCode x QuadClass: n events (deduplicated by GLOBALEVENTID), n_root (IsRootEvent=1), sums of NumMentions /
  NumSources / NumArticles, gold_n / gold_sum / gold_sq (GoldsteinScale), tone_n / tone_sum / tone_sq (AvgTone).
  Means = sum / n; variances from the squares.
- `dyad_daily_full.parquet`: per day x directed (Actor1CountryCode, Actor2CountryCode) (CAMEO codes, both non-empty):
  the same sums plus n_q1..n_q4 (events per QuadClass).
- `dyad_daily_top200.parquet`, `dyad_top200.csv`: the 200 pairs with the most events in the TRAIN period
  (<= 2025-06-30), so the selection uses no validation / test data.
- `coverage_daily.parquet`: per day, files listed / downloaded (md5-checked) / failed, bytes, raw lines, parsed rows,
  within-day duplicates, rows whose DATEADDED day differs from the file day, SQLDATE lag counts, id range.
- `coverage_files.parquet`: per 15-minute file: status, rows, id range.
- `lookups/`: GDELT's CAMEO / FIPS lookup tables (https://www.gdeltproject.org/data/lookups/).
- `manifest.json`: row counts, date range, md5s, download statistics, wall time.
"""


# =========================================================================== WEEKLY PANEL (numpy core: no pandas)
WEEK0 = date(2015, 2, 16)        # the Monday on / before the first GDELT 2.0 file (2015-02-18 23:00 UTC)
FILES_PER_DAY = 96               # one export file every 15 minutes
GOOD_WEEK_COV, GOOD_DAY_COV = 0.9, 0.5   # a "good" week: >= 90% of its files arrived and no day under 50%
ROOTS = [f"{i:02d}" for i in range(1, 21)]
LOOK = 56                        # history weeks an origin needs (52-week window + the 4 weeks before its first week)
SPLITS = (("train", None, date(2025, 6, 30)), ("val", date(2025, 7, 1), date(2025, 12, 31)),
          ("test", date(2026, 1, 1), date(2026, 5, 31)), ("test_recent", date(2026, 6, 1), None))
FAMILIES = {"f1": "gdelt_f1_count", "f2": "gdelt_f2_exceed", "f3": "gdelt_f3_goldstein", "f4": "gdelt_f4_tone"}
HORIZON = {"f1": 1, "f2": 1, "f3": 4, "f4": 1}       # weeks
# eligibility (pre-origin data only)
F12_MIN_MEAN, F12_MIN_NZ = 10.0, 26                   # series mean >= 10 events / week, nonzero in >= 26 of 52 weeks
F3C_MIN_MEAN, F3C_MIN_4W = 50.0, 200                  # country: >= 50 events / week over 52 weeks, >= 200 in the last 4
F3D_MIN_MEAN, F3D_MIN_4W = 20.0, 80                   # dyad
F4_MIN_MEAN, F4_MIN_4W = 50.0, 200
MIN_GOOD_52 = 48                                      # good weeks among the trailing 52
F3_MIN_NEXT, F4_MIN_NEXT = 20, 10                     # events needed in the resolution window (else dropped, counted)
UNIFORM_MIX = 0.01                                    # parametric baselines: 1% uniform so log scores stay finite
DEFAULT_TARGETS = {   # questions per (split, family); F3 is half countries, half dyads
    "train": {"f1": 50000, "f2": 40000, "f3": 30000, "f4": 30000},
    "val": {"f1": 1500, "f2": 1200, "f3": 1000, "f4": 1000},
    "test": {"f1": 2000, "f2": 1600, "f3": 1000, "f4": 1000},
    "test_recent": {"f1": 1600, "f2": 1300, "f3": 800, "f4": 800},
}


def ymd(d):
    """int YYYYMMDD -> date"""
    d = int(d)
    return date(d // 10000, d // 100 % 100, d % 100)


class WeeklyPanel:
    """Dense weekly arrays; time is the LAST axis everywhere. Week w covers UTC days [week0 + 7w, week0 + 7w + 7).

    A    cov / cov_daymin / good / global_n          [W]
         cr_n                                        [C, 20, W]  country x root-code counts
         c_n, c_gold_sum, c_gold_n, c_tone_sum, c_tone_n   [C, W]
         c_quad                                      [C, 4, W]
         d_n, d_gold_sum, d_gold_n, d_tone_sum, d_tone_n   [D, W]   directed actor-country dyads
         d_quad                                      [D, 4, W]
    """

    def __init__(self, week0, A, ccs, dyads, names, end_day):
        self.week0, self.A, self.ccs, self.dyads, self.names = week0, A, list(ccs), list(dyads), names
        self.cc_index = {c: i for i, c in enumerate(self.ccs)}
        self.dy_index = {d: i for i, d in enumerate(self.dyads)}
        self.W = int(A["global_n"].shape[-1])
        self.end_day = end_day

    def week_start(self, w):
        return self.week0 + timedelta(days=7 * int(w))

    def view(self, lo, hi):
        """Arrays restricted to weeks [lo, hi), keeping ABSOLUTE week indexing for lo = 0 (a history view)."""
        return {k: v[..., lo:hi] for k, v in self.A.items()}

    def cc_name(self, c):
        return self.names.get("fips", {}).get(c) or c

    def actor_name(self, a):
        return self.names.get("cameo", {}).get(a) or a


def weekly_panel_from_columns(country, dyad, coverage, end_day, names=None, week0=WEEK0):
    """Daily panel columns (dicts of equal-length arrays) -> WeeklyPanel. Only COMPLETE weeks up to end_day are kept.

    country:  date (int YYYYMMDD), cc, root, quad, n, gold_sum, gold_n, tone_sum, tone_n
    dyad:     date, a1cc, a2cc, n, gold_sum, gold_n, tone_sum, tone_n, n_q1..n_q4
    coverage: date, files_ok
    """
    names = names or {}
    W = ((end_day - week0).days + 1) // 7

    def widx(dates):
        u, inv = np.unique(np.asarray(dates, dtype=np.int64), return_inverse=True)
        wu = np.array([(ymd(x) - week0).days // 7 for x in u], dtype=np.int64)
        du = np.array([(ymd(x) - week0).days for x in u], dtype=np.int64)
        return wu[inv], du[inv]

    A = {}
    cw, cd = widx(coverage["date"])
    ok = (cd >= 0) & (cd < W * 7)
    files = _acc((W * 7,), (cd[ok],), np.asarray(coverage["files_ok"], dtype=np.float64)[ok])
    daily = (files / FILES_PER_DAY).reshape(W, 7)
    A["cov"] = daily.mean(1)
    A["cov_daymin"] = daily.min(1)
    A["good"] = (A["cov"] >= GOOD_WEEK_COV) & (A["cov_daymin"] >= GOOD_DAY_COV)

    w, _ = widx(country["date"])
    cc = np.asarray(country["cc"]).astype(str)
    keep = (w >= 0) & (w < W)
    A["global_n"] = _acc((W,), (w[keep],), np.asarray(country["n"], dtype=np.float64)[keep])
    keep &= cc != ""
    ccs = sorted(set(cc[keep].tolist()))
    ci = {c: i for i, c in enumerate(ccs)}
    C = len(ccs)
    uc, inv = np.unique(cc, return_inverse=True)
    cidx = np.array([ci.get(x, -1) for x in uc.tolist()], dtype=np.int64)[inv]
    ridx = _root_index(np.asarray(country["root"]).astype(str))
    quad = np.asarray(country["quad"], dtype=np.int64)
    n = np.asarray(country["n"], dtype=np.float64)
    for k in ("n", "gold_sum", "gold_n", "tone_sum", "tone_n"):
        A[f"c_{k}"] = _acc((C, W), (cidx[keep], w[keep]), np.asarray(country[k], dtype=np.float64)[keep])
    kr = keep & (ridx >= 0)
    A["cr_n"] = _acc((C, len(ROOTS), W), (cidx[kr], ridx[kr], w[kr]), n[kr])
    kq = keep & (quad >= 1) & (quad <= 4)
    A["c_quad"] = _acc((C, 4, W), (cidx[kq], quad[kq] - 1, w[kq]), n[kq])

    dw, _ = widx(dyad["date"]) if len(dyad["date"]) else (np.zeros(0, np.int64), None)
    pairs = list(zip(np.asarray(dyad["a1cc"]).astype(str).tolist(), np.asarray(dyad["a2cc"]).astype(str).tolist()))
    dyads = sorted(set(pairs))
    di = {d: i for i, d in enumerate(dyads)}
    D = len(dyads)
    didx = np.array([di[p] for p in pairs], dtype=np.int64)
    dk = (dw >= 0) & (dw < W)
    for k in ("n", "gold_sum", "gold_n", "tone_sum", "tone_n"):
        A[f"d_{k}"] = _acc((D, W), (didx[dk], dw[dk]), np.asarray(dyad[k], dtype=np.float64)[dk])
    A["d_quad"] = np.stack([_acc((D, W), (didx[dk], dw[dk]), np.asarray(dyad[f"n_q{q + 1}"], dtype=np.float64)[dk])
                            for q in range(4)], 1) if D else np.zeros((0, 4, W))
    last = week0 + timedelta(days=7 * W - 1)
    return WeeklyPanel(week0, A, ccs, dyads, names, last)


def _acc(shape, idx, vals):
    """Dense sum of vals at the index tuple (np.bincount on the raveled index: far faster than np.add.at)."""
    size = int(np.prod(shape))
    if size == 0:
        return np.zeros(shape)
    flat = np.ravel_multi_index(tuple(np.asarray(i, dtype=np.int64) for i in idx), shape) if len(idx[0]) else \
        np.zeros(0, np.int64)
    return np.bincount(flat, weights=np.asarray(vals, dtype=np.float64), minlength=size).reshape(shape)


def _root_index(root):
    u, inv = np.unique(root, return_inverse=True)
    m = np.array([ROOTS.index(r) if r in ROOTS else -1 for r in u], dtype=np.int64)
    return m[inv]


def load_panel(panel_dir):
    """Merged parquet panel -> WeeklyPanel (pandas / pyarrow; node)."""
    import pyarrow.parquet as pq
    p = Path(panel_dir)
    cd = pq.read_table(p / "country_daily.parquet",
                       columns=["date", "cc", "root", "quad", "n", "gold_sum", "gold_n", "tone_sum", "tone_n"]).to_pandas()
    dy = pq.read_table(p / "dyad_daily_top200.parquet",
                       columns=["date", "a1cc", "a2cc", "n", "gold_sum", "gold_n", "tone_sum", "tone_n",
                                "n_q1", "n_q2", "n_q3", "n_q4"]).to_pandas()
    cov = pq.read_table(p / "coverage_daily.parquet", columns=["date", "files_ok"]).to_pandas()
    names = {"fips": read_lookup(p / "lookups" / "FIPS.country.txt"),
             "cameo": read_lookup(p / "lookups" / "CAMEO.country.txt")}
    end_day = ymd(int(cov["date"].max()))
    return weekly_panel_from_columns({k: cd[k].to_numpy() for k in cd.columns}, {k: dy[k].to_numpy() for k in dy.columns},
                                     {k: cov[k].to_numpy() for k in cov.columns}, end_day, names)


# =========================================================================== QUESTION FAMILIES
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _mix(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0, None)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
    return (1 - UNIFORM_MIX) * p + UNIFORM_MIX / len(p)


def _laplace(counts):
    c = np.asarray(counts, dtype=np.float64) + 0.5
    return c / c.sum()


def _dist(keys, p):
    return {k: round(float(v), 6) for k, v in zip(keys, p)}


def nb_fit(y):
    """Negative binomial from history only: mean = last 4 weeks; size k by moments of the residuals of each of the trailing
    52 weeks about the mean of its own previous 4 weeks (var = m + m^2 / k; k -> 1e6 = Poisson when not over-dispersed)."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    m = np.array([y[s - 4:s].mean() for s in range(T - 52, T)])
    res = y[T - 52:] - m
    mbar = max(float(m.mean()), 0.5)
    v = float(np.mean(res ** 2))
    k = mbar ** 2 / (v - mbar) if v > mbar * 1.0001 else 1e6
    return max(float(y[-4:].mean()), 0.5), float(np.clip(k, 0.2, 1e6))


def nb_cdf(x, mu, k):
    from scipy.stats import nbinom
    return nbinom.cdf(x, k, k / (k + mu))


def count_cuts(y52):
    """Integer cut points c_1 < c_2 < ... from the trailing-year deciles: bin j = [c_j, c_{j+1} - 1], c_0 = 0."""
    q = np.quantile(np.asarray(y52, dtype=np.float64), np.arange(1, 10) / 10.0)
    return sorted(set(int(math.floor(v)) + 1 for v in q))


def count_levels(cuts):
    f = "{:,}".format
    lv = [f"0 to {f(cuts[0] - 1)}" if cuts[0] > 1 else "0"]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        lv.append(f(lo) if hi - 1 == lo else f"{f(lo)} to {f(hi - 1)}")
    lv.append(f"{f(cuts[-1])} or more")
    return lv


def count_bin(cuts, y):
    return int(np.searchsorted(np.asarray(cuts), y, side="right"))


def nb_bin_probs(cuts, mu, k):
    cdf = np.concatenate([[0.0], nb_cdf(np.asarray(cuts, dtype=np.float64) - 1, mu, k), [1.0]])   # P(Y <= c_j - 1)
    return np.diff(cdf)


def gauss_bin_probs(abs_edges, ref, mu, sigma):
    """P(next in each bin) for next - ref ~ N(mu, sigma^2); abs_edges are the bins' absolute lower edges 1..K-1."""
    z = [0.0] + [_norm_cdf((e - ref - mu) / sigma) for e in abs_edges] + [1.0]
    return np.diff(np.asarray(z))


def _mean(s, n):
    s, n = float(np.sum(s)), float(np.sum(n))
    return s / n if n > 0 else float("nan")


def _fmt_week_list(P, t0, vals, fmt, good, per_line=4):
    items = []
    for i, v in enumerate(vals):
        w = t0 + i
        star = "" if good[w] else "*"
        items.append(f"{P.week_start(w).isoformat()}{star}: {fmt(v)}")
    return "\n".join(" | ".join(items[i:i + per_line]) for i in range(0, len(items), per_line))


def _header(P, t):
    o = P.week_start(t)
    return (f"GDELT 2.0 Event Database: events machine-coded (CAMEO) from English-language news by the GDELT Project "
            f"(https://www.gdeltproject.org/). Forecast date: Monday {o.isoformat()} 00:00 UTC; the data below end on "
            f"Sunday {(o - timedelta(days=1)).isoformat()}. Weeks run Monday to Sunday (UTC) by the date GDELT added each "
            f"event; each event (GLOBALEVENTID) counts once.")


def _country_context(P, H, c, t):
    """Shared country lines: volume, top event types, QuadClass mix, Goldstein and tone, worldwide volume."""
    name = P.cc_name(P.ccs[c])
    n = H["c_n"][c]
    cr = H["cr_n"][c]
    top = np.argsort(-cr[:, t - 4:t].sum(1), kind="stable")[:6]
    tops = "; ".join(f"{ROOT_NAMES[ROOTS[i]]} {cr[i, t - 4:t].sum() / 4:,.0f}" for i in top if cr[i, t - 4:t].sum() > 0)
    q4, q52 = H["c_quad"][c][:, t - 4:t].sum(1), H["c_quad"][c][:, t - 52:t].sum(1)
    mix = ", ".join(f"{QUAD_NAMES[i + 1]} {100 * q4[i] / max(q4.sum(), 1):.0f}% ({100 * q52[i] / max(q52.sum(), 1):.0f}%)"
                    for i in range(4))
    g4, g52 = _mean(H["c_gold_sum"][c][t - 4:t], H["c_gold_n"][c][t - 4:t]), \
        _mean(H["c_gold_sum"][c][t - 52:t], H["c_gold_n"][c][t - 52:t])
    o4, o52 = _mean(H["c_tone_sum"][c][t - 4:t], H["c_tone_n"][c][t - 4:t]), \
        _mean(H["c_tone_sum"][c][t - 52:t], H["c_tone_n"][c][t - 52:t])
    gl = H["global_n"]
    return [f"All events located in {name}: last 4 weeks {n[t - 4:t].mean():,.0f} per week (trailing 52 weeks "
            f"{n[t - 52:t].mean():,.0f} per week).",
            f"Top event types in {name}, last 4 weeks (events per week): {tops}.",
            f"Event mix in {name} by QuadClass, last 4 weeks (trailing 52 weeks): {mix}.",
            f"Mean Goldstein score of all events in {name}: last 4 weeks {g4:+.2f}, trailing 52 weeks {g52:+.2f}. "
            f"Mean tone: {o4:+.2f} and {o52:+.2f}.",
            f"All GDELT events worldwide: last week {gl[t - 1]:,.0f}; trailing 52 weeks {gl[t - 52:t].mean():,.0f} per week."]


def _coverage_note(H, t, lo):
    return ["(* = a week with incomplete GDELT update files: under 90% of the 15-minute files arrived, or a day under "
            "50%.)"] if not H["good"][lo:t].all() else []


def _common_meta(P, fam, t, split, entity, cluster):
    o = P.week_start(t)
    h = HORIZON[fam]
    return {"family": FAMILIES[fam], "split": split, "origin": o.isoformat(), "horizon_days": 7 * h,
            "resolution": [o.isoformat(), (o + timedelta(days=7 * h - 1)).isoformat()],
            "history_end": (o - timedelta(days=1)).isoformat(), "entity": entity, "cluster": cluster,
            "license": LICENSE, "citation": CITATION, "builder": VERSION,
            "gdelt": "2.0 Event export (English), by DATEADDED (UTC), deduplicated by GLOBALEVENTID",
            "label_basis": "what GDELT recorded (machine-coded news events), not ground truth"}


def make_record(fam, qid, state, q, label_key, meta, base_id):
    from_keys = option_keys(q)
    src = FAMILIES[fam]
    rid = f"{src}-{sha(state + '|' + json.dumps(q, sort_keys=True))}"
    return {"id": rid, "source": src, "license": LICENSE, "base_id": base_id, "state": state, "questions": {qid: q},
            "labels": {qid: {k: (1.0 if k == label_key else 0.0) for k in from_keys}},
            "label_kind": {qid: "hard"}, "label_source": {qid: "outcome"}, "meta": meta}


def option_keys(q):
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["yes", "no"]
    return list(q["levels"])


def _count_state(P, H, c, r, t):
    name, cc, rn = P.cc_name(P.ccs[c]), P.ccs[c], ROOT_NAMES[ROOTS[r]]
    y = H["cr_n"][c, r]
    y52 = y[t - 52:t]
    share4 = y[t - 4:t].sum() / max(H["c_n"][c][t - 4:t].sum(), 1)
    share52 = y52.sum() / max(H["c_n"][c][t - 52:t].sum(), 1)
    lines = [_header(P, t), "",
             f"Location: {name} (GDELT ActionGeo country code {cc}).",
             f'Event type: CAMEO root code {ROOTS[r]} "{rn}".', "",
             f'Weekly "{rn}" events located in {name}, last 26 weeks (week start: count):',
             _fmt_week_list(P, t - 26, y[t - 26:t], lambda v: f"{v:,.0f}", H["good"]),
             f"Trailing 52 weeks ({P.week_start(t - 52).isoformat()} to {(P.week_start(t) - timedelta(days=1)).isoformat()}): "
             f"mean {y52.mean():,.1f}, median {np.median(y52):,.1f}, 10th percentile {np.quantile(y52, 0.1):,.1f}, "
             f"90th percentile {np.quantile(y52, 0.9):,.1f}, min {y52.min():,.0f}, max {y52.max():,.0f}; nonzero in "
             f"{int((y52 > 0).sum())} of 52 weeks.",
             f"Last 4 weeks: mean {y[t - 4:t].mean():,.1f} per week. The same week one year before the forecast week "
             f"(week of {P.week_start(t - 52).isoformat()}): {y[t - 52]:,.0f}.",
             f'"{rn}" share of all events in {name}: last 4 weeks {100 * share4:.1f}%, trailing 52 weeks '
             f"{100 * share52:.1f}%."]
    lines += _country_context(P, H, c, t)
    lines += _coverage_note(H, t, t - 26)
    return "\n".join(lines)


def q_f1(P, c, r, t, split):
    """Ordered bins for next week's count of root code r in country c."""
    H, F = P.view(0, t), P.view(t, t + 1)
    y = H["cr_n"][c, r]
    cuts = count_cuts(y[t - 52:t])
    if len(cuts) < 3:
        return None, "few_bins"
    levels = count_levels(cuts)
    target = int(round(F["cr_n"][c, r, 0]))
    k = count_bin(cuts, target)
    mu, kk = nb_fit(y[:t])
    base = {"climatology": _laplace(np.bincount([count_bin(cuts, v) for v in y[t - 52:t]], minlength=len(levels))),
            "negbin": _mix(nb_bin_probs(cuts, mu, kk)),
            "seasonal_naive": _mix(nb_bin_probs(cuts, max(float(y[t - 52]), 0.5), kk)),
            "persistence": _mix(nb_bin_probs(cuts, max(float(y[t - 1]), 0.5), kk))}
    o = P.week_start(t)
    name, rn = P.cc_name(P.ccs[c]), ROOT_NAMES[ROOTS[r]]
    instr = (f'How many "{rn}" events (CAMEO root code {ROOTS[r]}) located in {name} will GDELT 2.0 record in the week of '
             f"Monday {o.isoformat()} to Sunday {(o + timedelta(days=6)).isoformat()} (UTC, by the date GDELT added each "
             f"event)? Choose the range that will contain the count. The ranges are the deciles of the previous 52 weekly "
             f"counts.")
    q = {"type": "score", "instructions": instr, "levels": levels}
    qid = f"gdelt_f1/{P.ccs[c]}_{ROOTS[r]}_{o.isoformat()}"
    meta = _common_meta(P, "f1", t, split, {"kind": "country", "code": P.ccs[c], "code_system": "FIPS 10-4",
                                            "name": name}, f"fips:{P.ccs[c]}")
    meta.update({"root_code": ROOTS[r], "root_name": rn, "cuts": cuts, "target_value": target, "target_bin": k,
                 "baselines": {b: _dist(levels, p) for b, p in base.items()},
                 "baseline_params": {"negbin": {"mu": round(mu, 4), "size": round(kk, 4)},
                                     "seasonal_naive_mu": float(y[t - 52]), "persistence_mu": float(y[t - 1])}})
    state = _count_state(P, H, c, r, t)
    return make_record("f1", qid, state, q, levels[k], meta, f"gdelt/{P.ccs[c]}_{ROOTS[r]}/{o.isoformat()}"), "ok"


def q_f2(P, c, r, t, split):
    """Will next week's count exceed the trailing-52-week 90th percentile?"""
    H, F = P.view(0, t), P.view(t, t + 1)
    y = H["cr_n"][c, r]
    y52 = y[t - 52:t]
    p90 = float(np.quantile(y52, 0.9))
    thr = int(math.floor(p90))
    target = int(round(F["cr_n"][c, r, 0]))
    yes = target > thr
    mu, kk = nb_fit(y[:t])

    def nb_yes(m):
        return float(1.0 - nb_cdf(thr, m, kk))

    base = {"climatology": _laplace([(y52 > thr).sum(), (y52 <= thr).sum()]),
            "negbin": _mix([nb_yes(mu), 1 - nb_yes(mu)]),
            "seasonal_naive": _mix([nb_yes(max(float(y[t - 52]), 0.5)), 1 - nb_yes(max(float(y[t - 52]), 0.5))]),
            "persistence": _mix([nb_yes(max(float(y[t - 1]), 0.5)), 1 - nb_yes(max(float(y[t - 1]), 0.5))])}
    o = P.week_start(t)
    name, rn = P.cc_name(P.ccs[c]), ROOT_NAMES[ROOTS[r]]
    instr = (f'Will GDELT 2.0 record more than {thr:,} "{rn}" events (CAMEO root code {ROOTS[r]}) located in {name} in the '
             f"week of Monday {o.isoformat()} to Sunday {(o + timedelta(days=6)).isoformat()} (UTC, by the date GDELT "
             f"added each event)? {thr:,} is the 90th percentile of the previous 52 weekly counts ({p90:,.1f}), rounded "
             f"down.")
    q = {"type": "noul", "instructions": instr,
         "criteria": {"yes": f"More than {thr:,} events", "no": f"{thr:,} events or fewer"}}
    qid = f"gdelt_f2/{P.ccs[c]}_{ROOTS[r]}_{o.isoformat()}"
    meta = _common_meta(P, "f2", t, split, {"kind": "country", "code": P.ccs[c], "code_system": "FIPS 10-4",
                                            "name": name}, f"fips:{P.ccs[c]}")
    meta.update({"root_code": ROOTS[r], "root_name": rn, "threshold": thr, "p90": round(p90, 4),
                 "target_value": target, "baselines": {b: _dist(["yes", "no"], p) for b, p in base.items()},
                 "baseline_params": {"negbin": {"mu": round(mu, 4), "size": round(kk, 4)}}})
    state = _count_state(P, H, c, r, t)
    return make_record("f2", qid, state, q, "yes" if yes else "no", meta,
                       f"gdelt/{P.ccs[c]}_{ROOTS[r]}/{o.isoformat()}"), "ok"


def _series(A, kind, e, what):
    p = "c" if kind == "country" else "d"
    return A[f"{p}_{what}_sum"][e], A[f"{p}_{what}_n"][e], A[f"{p}_n"][e]


def _window_mean(s, n, lo, hi):
    return _mean(s[lo:hi], n[lo:hi])


def _gauss_fit(ds, xs, m_ref, ref, sn):
    """sigma of the trailing changes; mean-reversion slope beta (changes on gaps to the 52-week mean); seasonal naive mu."""
    ds, xs = np.asarray(ds), np.asarray(xs)
    sigma = max(float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0, 0.01)
    beta = float(np.clip((xs * ds).sum() / (xs ** 2).sum(), 0.0, 1.0)) if (xs ** 2).sum() > 0 else 0.0
    return {"sigma": sigma, "beta": beta, "mu_rw": 0.0, "mu_mr": beta * (m_ref - ref),
            "mu_sn": float(sn) if sn is not None and not math.isnan(sn) else 0.0}


def q_f3(P, kind, e, t, split):
    """Direction (down / same / up) of the next 28 days' mean Goldstein vs the previous 28 days, with a stated tolerance."""
    H, F = P.view(0, t), P.view(t, t + 4)
    gs, gn, n = _series(H, kind, e, "gold")
    prev = _window_mean(gs, gn, t - 4, t)
    ref = round(prev, 2)
    m52 = _window_mean(gs, gn, t - 52, t)
    ds, xs = [], []
    for s in range(t - 52, t - 3):
        a, b = _window_mean(gs, gn, s, s + 4), _window_mean(gs, gn, s - 4, s)
        if not (math.isnan(a) or math.isnan(b)):
            ds.append(a - b)
            xs.append(m52 - b)
    if len(ds) < 20:
        return None, "few_history_windows"
    tol = max(0.02, round(0.5 * float(np.median(np.abs(ds))), 2))
    lo, hi = round(ref - tol, 2), round(ref + tol, 2)
    fs, fn, fnn = _series(F, kind, e, "gold")
    if fn.sum() < F3_MIN_NEXT:
        return None, "few_next_events"
    nxt = _mean(fs, fn)
    lab = "up" if nxt > hi else "down" if nxt < lo else "same"
    sn_a, sn_b = _window_mean(gs, gn, t - 52, t - 48), _window_mean(gs, gn, t - 56, t - 52)
    g = _gauss_fit(ds, xs, m52, ref, sn_a - sn_b)
    keys = ["down", "same", "up"]
    dsa = np.asarray(ds)
    base = {"climatology": _laplace([(dsa < -tol).sum(), ((dsa >= -tol) & (dsa <= tol)).sum(), (dsa > tol).sum()])}
    for nm, mu in (("gauss_rw", g["mu_rw"]), ("mean_revert", g["mu_mr"]), ("seasonal_naive", g["mu_sn"])):
        base[nm] = _mix(gauss_bin_probs([lo, hi], ref, mu, g["sigma"]))
    o = P.week_start(t)
    end = o + timedelta(days=27)
    if kind == "country":
        code = P.ccs[e]
        name = P.cc_name(code)
        subj = f"events located in {name}"
        entity = {"kind": "country", "code": code, "code_system": "FIPS 10-4", "name": name}
        cluster, ek = f"fips:{code}", f"cty_{code}"
    else:
        a1, a2 = P.dyads[e]
        n1, n2 = P.actor_name(a1), P.actor_name(a2)
        subj = f"events with an Actor1 from {n1} (CAMEO {a1}) and an Actor2 from {n2} (CAMEO {a2})"
        entity = {"kind": "dyad", "code": f"{a1}->{a2}", "code_system": "CAMEO country", "name": f"{n1} -> {n2}"}
        cluster, ek = f"cameo:{a1}", f"dyad_{a1}_{a2}"
    instr = (f"Will the mean Goldstein score of {subj} that GDELT 2.0 records in the next 28 days (Monday {o.isoformat()} "
             f"to Sunday {end.isoformat()}, UTC, by the date GDELT added each event) be lower, about the same, or higher "
             f"than the previous 28 days' mean of {ref:+.2f}? Tolerance {tol:.2f}: 'about the same' means from "
             f"{lo:+.2f} to {hi:+.2f} inclusive.")
    q = {"type": "choice", "instructions": instr,
         "criteria": {"down": f"Lower: below {lo:+.2f}", "same": f"About the same: {lo:+.2f} to {hi:+.2f}",
                      "up": f"Higher: above {hi:+.2f}"}}
    qid = f"gdelt_f3/{ek}_{o.isoformat()}"
    meta = _common_meta(P, "f3", t, split, entity, cluster)
    meta.update({"reference": ref, "tolerance": tol, "bounds": [lo, hi], "target_value": round(nxt, 6),
                 "target_delta": round(nxt - ref, 6), "next_events": int(fn.sum()),
                 "baselines": {b: _dist(keys, p) for b, p in base.items()},
                 "baseline_params": {k: round(v, 6) for k, v in g.items()}})
    state = _goldstein_state(P, H, kind, e, t, gs, gn, n, prev, m52, ds)
    return make_record("f3", qid, state, q, lab, meta, f"gdelt/{ek}/{o.isoformat()}"), "ok"


def _goldstein_state(P, H, kind, e, t, gs, gn, n, prev, m52, ds):
    wk = [(_mean(gs[w], gn[w]), n[w]) for w in range(t - 26, t)]
    blocks = [_window_mean(gs, gn, s, s + 4) for s in range(t - 52, t, 4)]
    wm = np.array([_mean(gs[w], gn[w]) for w in range(t - 52, t)])
    wm = wm[~np.isnan(wm)]
    dsa = np.asarray(ds)
    lines = [_header(P, t), ""]
    if kind == "country":
        code = P.ccs[e]
        name = P.cc_name(code)
        lines.append(f"Location: {name} (GDELT ActionGeo country code {code}). Events located in {name}.")
    else:
        a1, a2 = P.dyads[e]
        lines.append(f"Directed dyad: events whose Actor1 is from {P.actor_name(a1)} (CAMEO {a1}) and whose Actor2 is from "
                     f"{P.actor_name(a2)} (CAMEO {a2}); Actor1 is the source, Actor2 the target.")
    lines += ["Goldstein scale: -10 (most conflictual) to +10 (most cooperative), from the CAMEO event code.", "",
              "Weekly mean Goldstein score (event count in parentheses), last 26 weeks:",
              _fmt_week_list(P, t - 26, wk, lambda v: (f"{v[0]:+.2f} ({v[1]:,.0f})" if not math.isnan(v[0])
                                                       else "none (0)"), H["good"], per_line=4),
              "28-day means over the trailing 52 weeks, oldest first: "
              + ", ".join("n/a" if math.isnan(b) else f"{b:+.2f}" for b in blocks) + ".",
              f"Previous 28 days ({P.week_start(t - 4).isoformat()} to {(P.week_start(t) - timedelta(days=1)).isoformat()}):"
              f" mean {prev:+.2f} over {n[t - 4:t].sum():,.0f} events. Trailing 52 weeks: mean {m52:+.2f}; standard "
              f"deviation of the weekly means {np.std(wm):.2f}.",
              f"Change between consecutive 28-day windows (stepped weekly over the trailing year): median absolute "
              f"{np.median(np.abs(dsa)):.2f}; 10th / 50th / 90th percentiles {np.quantile(dsa, 0.1):+.2f} / "
              f"{np.quantile(dsa, 0.5):+.2f} / {np.quantile(dsa, 0.9):+.2f}."]
    if kind == "country":
        lines += _country_context(P, H, e, t)
    else:
        q4, q52 = H["d_quad"][e][:, t - 4:t].sum(1), H["d_quad"][e][:, t - 52:t].sum(1)
        lines.append("Event mix by QuadClass, last 4 weeks (trailing 52 weeks): " + ", ".join(
            f"{QUAD_NAMES[i + 1]} {100 * q4[i] / max(q4.sum(), 1):.0f}% ({100 * q52[i] / max(q52.sum(), 1):.0f}%)"
            for i in range(4)) + ".")
        to4 = _mean(H["d_tone_sum"][e][t - 4:t], H["d_tone_n"][e][t - 4:t])
        to52 = _mean(H["d_tone_sum"][e][t - 52:t], H["d_tone_n"][e][t - 52:t])
        lines.append(f"Mean tone of the dyad's events: last 4 weeks {to4:+.2f}, trailing 52 weeks {to52:+.2f}. Events per "
                     f"week: last 4 weeks {n[t - 4:t].mean():,.0f}, trailing 52 weeks {n[t - 52:t].mean():,.0f}.")
        rev = P.dy_index.get((P.dyads[e][1], P.dyads[e][0]))
        if rev is not None:
            rg = _mean(H["d_gold_sum"][rev][t - 4:t], H["d_gold_n"][rev][t - 4:t])
            lines.append(f"Reverse direction ({P.dyads[e][1]} -> {P.dyads[e][0]}), last 4 weeks: "
                         f"{H['d_n'][rev][t - 4:t].sum():,.0f} events, mean Goldstein "
                         + ("n/a" if math.isnan(rg) else f"{rg:+.2f}") + ".")
        gl = H["global_n"]
        lines.append(f"All GDELT events worldwide: last week {gl[t - 1]:,.0f}; trailing 52 weeks "
                     f"{gl[t - 52:t].mean():,.0f} per week.")
    lines += _coverage_note(H, t, t - 26)
    return "\n".join(lines)


def tone_cuts(ds):
    q = np.quantile(np.asarray(ds), [0.1, 0.3, 0.7, 0.9])
    cuts = []
    for v in q:
        v = round(float(v), 2)
        if cuts and v <= cuts[-1]:
            v = round(cuts[-1] + 0.01, 2)
        cuts.append(v)
    return cuts


def q_f4(P, c, t, split):
    """Ordered bins for next week's mean tone in country c relative to the previous 4 weeks' mean."""
    H, F = P.view(0, t), P.view(t, t + 1)
    ts_, tn, n = _series(H, "country", c, "tone")
    prev = _window_mean(ts_, tn, t - 4, t)
    ref = round(prev, 2)
    m52 = _window_mean(ts_, tn, t - 52, t)
    ds, xs = [], []
    for s in range(t - 52, t):
        a, b = _mean(ts_[s], tn[s]), _window_mean(ts_, tn, s - 4, s)
        if not (math.isnan(a) or math.isnan(b)):
            ds.append(a - b)
            xs.append(m52 - b)
    if len(ds) < 40:
        return None, "few_history_weeks"
    cuts = tone_cuts(ds)
    abs_edges = [round(ref + x, 2) for x in cuts]
    fs, fn, _ = _series(F, "country", c, "tone")
    if fn.sum() < F4_MIN_NEXT:
        return None, "few_next_events"
    nxt = _mean(fs, fn)
    k = int(np.searchsorted(np.asarray(abs_edges), nxt, side="right"))
    levels = [f"below {abs_edges[0]:+.2f} (change below {cuts[0]:+.2f})"]
    for j in range(3):
        levels.append(f"{abs_edges[j]:+.2f} to {abs_edges[j + 1]:+.2f} (change {cuts[j]:+.2f} to {cuts[j + 1]:+.2f})")
    levels.append(f"{abs_edges[3]:+.2f} or more (change {cuts[3]:+.2f} or more)")
    sn = _mean(ts_[t - 52], tn[t - 52]) - _window_mean(ts_, tn, t - 56, t - 52)
    g = _gauss_fit(ds, xs, m52, ref, sn)
    base = {"climatology": _laplace(np.bincount([int(np.searchsorted(np.asarray(cuts), d, side="right")) for d in ds],
                                                minlength=5))}
    for nm, mu in (("gauss_rw", g["mu_rw"]), ("mean_revert", g["mu_mr"]), ("seasonal_naive", g["mu_sn"])):
        base[nm] = _mix(gauss_bin_probs(abs_edges, ref, mu, g["sigma"]))
    o = P.week_start(t)
    code = P.ccs[c]
    name = P.cc_name(code)
    instr = (f"What will the mean tone (GDELT AvgTone averaged over events) of events located in {name} be in the week of "
             f"Monday {o.isoformat()} to Sunday {(o + timedelta(days=6)).isoformat()} (UTC, by the date GDELT added each "
             f"event), compared with the previous 4 weeks' mean of {ref:+.2f}? Each range includes its lower end; the "
             f"change cut points are the 10th / 30th / 70th / 90th percentiles of the past year's weekly changes.")
    q = {"type": "score", "instructions": instr, "levels": levels}
    qid = f"gdelt_f4/{code}_{o.isoformat()}"
    meta = _common_meta(P, "f4", t, split, {"kind": "country", "code": code, "code_system": "FIPS 10-4", "name": name},
                        f"fips:{code}")
    meta.update({"reference": ref, "cuts": cuts, "abs_edges": abs_edges, "target_value": round(nxt, 6),
                 "target_delta": round(nxt - ref, 6), "target_bin": k, "next_events": int(fn.sum()),
                 "baselines": {b: _dist(levels, p) for b, p in base.items()},
                 "baseline_params": {kk: round(v, 6) for kk, v in g.items()}})
    state = _tone_state(P, H, c, t, ts_, tn, n, prev, m52, ds)
    return make_record("f4", qid, state, q, levels[k], meta, f"gdelt/tone_{code}/{o.isoformat()}"), "ok"


def _tone_state(P, H, c, t, ts_, tn, n, prev, m52, ds):
    code = P.ccs[c]
    name = P.cc_name(code)
    wk = [(_mean(ts_[w], tn[w]), n[w]) for w in range(t - 26, t)]
    dsa = np.asarray(ds)
    lines = [_header(P, t), "",
             f"Location: {name} (GDELT ActionGeo country code {code}). Events located in {name}.",
             "Tone: GDELT's AvgTone, the average sentiment of the articles that mention an event (about -10 to +10; "
             "news tone is usually negative).", "",
             "Weekly mean tone (event count in parentheses), last 26 weeks:",
             _fmt_week_list(P, t - 26, wk, lambda v: (f"{v[0]:+.2f} ({v[1]:,.0f})" if not math.isnan(v[0])
                                                      else "none (0)"), H["good"]),
             f"Previous 4 weeks ({P.week_start(t - 4).isoformat()} to {(P.week_start(t) - timedelta(days=1)).isoformat()}):"
             f" mean tone {prev:+.2f} over {n[t - 4:t].sum():,.0f} events. Trailing 52 weeks: mean {m52:+.2f}.",
             "Each week's mean tone minus the mean of the 4 weeks before it, over the trailing year: 10th / 30th / 50th / "
             "70th / 90th percentiles " + " / ".join(f"{np.quantile(dsa, q):+.2f}" for q in (0.1, 0.3, 0.5, 0.7, 0.9))
             + "."]
    lines += _country_context(P, H, c, t)
    lines += _coverage_note(H, t, t - 26)
    return "\n".join(lines)


# =========================================================================== CANDIDATES, SPLITS, SAMPLING
def split_origins(P, fam, split):
    """Week indices t whose origin is in the split and whose whole resolution window lies inside it (and in the data)."""
    h = HORIZON[fam]
    _, lo, hi = next(s for s in SPLITS if s[0] == split)
    hi = min(hi, P.end_day) if hi else P.end_day
    out = []
    for t in range(LOOK, P.W - h + 1):
        o = P.week_start(t)
        if (lo is None or o >= lo) and o + timedelta(days=7 * h - 1) <= hi:
            out.append(t)
    return out


def _csum(x):
    return np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, -1)], -1)


def candidates(P, fam, split):
    """Arrays of eligible (entity..., t) from PRE-origin data only, plus the resolution window's coverage."""
    ts = split_origins(P, fam, split)
    h = HORIZON[fam]
    good = P.A["good"].astype(np.float64)
    g52 = _csum(good)
    ok_t = [t for t in ts if g52[t] - g52[t - 52] >= MIN_GOOD_52 and good[t:t + h].all()]
    out = []
    if fam in ("f1", "f2"):
        y = P.A["cr_n"]
        cs, nz = _csum(y), _csum((y > 0).astype(np.float64))
        for t in ok_t:
            m = ((cs[..., t] - cs[..., t - 52]) / 52 >= F12_MIN_MEAN) & (nz[..., t] - nz[..., t - 52] >= F12_MIN_NZ)
            c, r = np.nonzero(m)
            out += [("country", int(a), int(b), t) for a, b in zip(c, r)]
    elif fam == "f3":
        for kind, arr, mm, m4 in (("country", P.A["c_n"], F3C_MIN_MEAN, F3C_MIN_4W),
                                  ("dyad", P.A["d_n"], F3D_MIN_MEAN, F3D_MIN_4W)):
            cs = _csum(arr)
            for t in ok_t:
                m = ((cs[:, t] - cs[:, t - 52]) / 52 >= mm) & (cs[:, t] - cs[:, t - 4] >= m4)
                out += [(kind, int(e), -1, t) for e in np.nonzero(m)[0]]
    else:
        cs = _csum(P.A["c_n"])
        for t in ok_t:
            m = ((cs[:, t] - cs[:, t - 52]) / 52 >= F4_MIN_MEAN) & (cs[:, t] - cs[:, t - 4] >= F4_MIN_4W)
            out += [("country", int(e), -1, t) for e in np.nonzero(m)[0]]
    return out


def _cand_keys(P, cand):
    kind, e, r, t = cand
    if kind == "country":
        g = f"fips:{P.ccs[e]}"
        sub = f"{g}/{ROOTS[r]}" if r >= 0 else g
    else:
        g = f"cameo:{P.dyads[e][0]}"
        sub = f"dyad:{P.dyads[e][0]}->{P.dyads[e][1]}"
    return g, sub


def balanced_order(P, cands, seed):
    """Round-robin priority: across clusters (countries; dyads by Actor1 country), then across series inside a cluster,
    then seeded-random origins inside a series. Taking a prefix gives water-filled quotas (each cluster the same number
    until it runs out). Deterministic: every hash is sha1 of codes, origins and the seed."""
    if not cands:
        return []
    keys = [_cand_keys(P, c) for c in cands]
    gs = sorted(set(k[0] for k in keys))
    subs = sorted(set(k[1] for k in keys))
    gh = {g: hash01(seed, "g", g) for g in gs}
    sh = {s: hash01(seed, "s", s) for s in subs}
    item = np.array([hash01(seed, "i", k[1], P.week_start(c[3]).isoformat()) for k, c in zip(keys, cands)])
    gmap = {g: i for i, g in enumerate(gs)}
    gid = np.array([gmap[k[0]] for k in keys])
    smap = {s: i for i, s in enumerate(subs)}
    sid = np.array([smap[k[1]] for k in keys])
    o1 = np.lexsort((item, sid))                       # by series, then item hash
    rank_s = np.empty(len(cands), np.int64)
    ss = sid[o1]
    start = np.r_[0, np.nonzero(np.diff(ss))[0] + 1]
    rank_s[o1] = np.arange(len(o1)) - np.repeat(start, np.diff(np.r_[start, len(o1)]))
    shv = np.array([sh[k[1]] for k in keys])
    o2 = np.lexsort((shv, rank_s, gid))                # inside a cluster: series round-robin
    rank_g = np.empty(len(cands), np.int64)
    gg = gid[o2]
    start = np.r_[0, np.nonzero(np.diff(gg))[0] + 1]
    rank_g[o2] = np.arange(len(o2)) - np.repeat(start, np.diff(np.r_[start, len(o2)]))
    ghv = np.array([gh[k[0]] for k in keys])
    return list(np.lexsort((ghv, rank_g)))             # across clusters: round-robin


def build_family_split(P, fam, split, n_target, seed):
    cands = candidates(P, fam, split)
    stats = {"candidates": len(cands), "drops": Counter()}
    recs = []
    if fam == "f3":
        parts = [[c for c in cands if c[0] == "country"], [c for c in cands if c[0] == "dyad"]]
        quotas = [n_target - n_target // 2, n_target // 2]
    else:
        parts, quotas = [cands], [n_target]
    for part, quota in zip(parts, quotas):
        got = 0
        for i in balanced_order(P, part, f"{seed}/{fam}/{split}"):
            if got >= quota:
                break
            kind, e, r, t = part[i]
            if fam == "f1":
                rec, why = q_f1(P, e, r, t, split)
            elif fam == "f2":
                rec, why = q_f2(P, e, r, t, split)
            elif fam == "f3":
                rec, why = q_f3(P, kind, e, t, split)
            else:
                rec, why = q_f4(P, e, t, split)
            if rec is None:
                stats["drops"][why] += 1
                continue
            recs.append(rec)
            got += 1
    recs.sort(key=lambda r: (r["meta"]["origin"], next(iter(r["questions"]))))
    stats["drops"] = dict(stats["drops"])
    return recs, stats


def family_summary(recs):
    s = {"n": len(recs)}
    if not recs:
        return s
    cl = Counter(r["meta"]["cluster"] for r in recs)
    s["clusters"] = len(cl)
    s["top_clusters"] = [[k, v] for k, v in cl.most_common(5)]
    s["max_cluster_share"] = round(cl.most_common(1)[0][1] / len(recs), 4)
    s["origins"] = [min(r["meta"]["origin"] for r in recs), max(r["meta"]["origin"] for r in recs)]
    s["n_origins"] = len(set(r["meta"]["origin"] for r in recs))
    lab = Counter()
    for r in recs:
        qid = next(iter(r["questions"]))
        keys = option_keys(r["questions"][qid])
        k = max(range(len(keys)), key=lambda i: r["labels"][qid][keys[i]])
        lab[k if r["questions"][qid]["type"] == "score" else keys[k]] += 1
    s["label_counts"] = {str(k): v for k, v in sorted(lab.items(), key=lambda kv: str(kv[0]))}
    sl = [len(r["state"]) for r in recs]
    s["state_chars"] = {"mean": round(float(np.mean(sl)), 1), "max": int(max(sl))}
    if recs[0]["meta"]["entity"]["kind"] in ("country", "dyad"):
        s["entity_kinds"] = dict(Counter(r["meta"]["entity"]["kind"] for r in recs))
    return s


def write_jsonl(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_corpus(P, out, targets=None, seed=SEED, families=None, splits=None):
    out = Path(out)
    targets = targets or DEFAULT_TARGETS
    manifest = {"version": VERSION, "seed": seed, "license": LICENSE, "license_quote": LICENSE_QUOTE,
                "citation": CITATION, "panel_weeks": [P.week_start(0).isoformat(), P.end_day.isoformat()],
                "splits": {s: {"from": lo.isoformat() if lo else None, "to": hi.isoformat() if hi else None}
                           for s, lo, hi in SPLITS},
                "targets": targets, "families": {}, "files": {}}
    for split, _, _ in SPLITS:
        if splits and split not in splits:
            continue
        for fam in FAMILIES:
            if families and fam not in families:
                continue
            n = targets.get(split, {}).get(fam, 0)
            if not n:
                continue
            t0 = time.time()
            recs, st = build_family_split(P, fam, split, n, seed)
            p = out / split / f"{FAMILIES[fam]}.jsonl"
            write_jsonl(p, recs)
            summ = family_summary(recs)
            summ.update(st)
            summ["secs"] = round(time.time() - t0, 1)
            manifest["families"].setdefault(split, {})[FAMILIES[fam]] = summ
            log(f"  {split:11s} {FAMILIES[fam]:20s} {len(recs):6d} / {n} (candidates {st['candidates']}, drops "
                f"{st['drops']}) {summ['secs']} s")
    for p in sorted(out.rglob("*.jsonl")):
        manifest["files"][str(p.relative_to(out))] = {"bytes": p.stat().st_size, "md5": md5_file(p),
                                                      "lines": sum(1 for _ in open(p))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.md").write_text(CORPUS_README.format(citation=CITATION, license=LICENSE, quote=LICENSE_QUOTE))
    return manifest


CORPUS_README = """# jev_gdelt_v1: forecasting decision questions from the GDELT 2.0 event stream

Layout: `<split>/<family>.jsonl` for split in train / val / test / test_recent and family in
gdelt_f1_count (score: next-week count bins), gdelt_f2_exceed (noul: count above the trailing-52-week 90th percentile),
gdelt_f3_goldstein (choice: next 28 days' mean Goldstein down / same / up), gdelt_f4_tone (score: next-week mean tone bins).
One question per record, in the jev record schema (id, source, license, base_id, state, questions, labels, label_kind,
label_source, meta). Built by build_gdelt_jev.py; docs in research/gdelt.md.

**Citation (required by the GDELT terms):** {citation}
**Licence:** {license}. "{quote}"

Only GDELT's structured fields are used; no news article text. Labels are what GDELT recorded (machine-coded counts,
Goldstein scores and tone from news coverage), not ground truth. Forecast origins are Mondays 00:00 UTC; each state uses
only data before its origin; each question's resolution window lies inside its split. test and test_recent are disjoint
(origins 2026-01-01..05-31 and from 2026-06-01). meta.baselines holds climatology / seasonal-naive / persistence /
negative-binomial (counts) or Gaussian (Goldstein, tone) predictive distributions over the same options, from pre-origin
data only; meta.target_value holds the realised number for numeric readouts.
"""


# =========================================================================== VALIDATE
def load_cache_features(repo=HERE):
    """trainers/cache_features.py, with torch / transformers stubbed when they are not installed (it imports both at top
    level; question_options / iter_units need neither)."""
    import importlib.util
    import tempfile
    stub = None
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        stub = tempfile.mkdtemp(prefix="gdelt_cf_stubs_", dir=str(Path(repo) / "tmp") if (Path(repo) / "tmp").is_dir()
                                else None)
        Path(stub, "transformers.py").write_text("AutoTokenizer = None\n")
        Path(stub, "torch.py").write_text("class Tensor:\n    pass\n")
        sys.path.insert(0, stub)
    dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True          # no __pycache__ inside the stub dir, so it can be removed file by file
    try:
        spec = importlib.util.spec_from_file_location("cache_features_gdelt", str(Path(repo) / "trainers/cache_features.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = dwb
        if stub:
            sys.path.remove(stub)
            for f in ("transformers.py", "torch.py"):
                Path(stub, f).unlink(missing_ok=True)
            try:
                os.rmdir(stub)
            except OSError:
                pass


def load_build_jev_data(repo=HERE):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_jev_data_gdelt", str(Path(repo) / "build_jev_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_corpus(data, repo=HERE):
    """Every record through build_jev_data.validate and cache_features.iter_units / question_options / is_noncommercial;
    qids unique corpus-wide; every resolution window inside its split; every date in a state before the origin."""
    import re
    cf = load_cache_features(repo)
    bjd = load_build_jev_data(repo)
    data = Path(data)
    rep = {"units": {}, "errors": []}
    seen = set()
    win = {s: (lo, hi) for s, lo, hi in SPLITS}
    date_re = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
    for split in [s for s, _, _ in SPLITS]:
        if not (data / split).is_dir():
            continue
        for p in sorted((data / split).glob("*.jsonl")):
            for ln in open(p, encoding="utf-8"):
                r = json.loads(ln)
                bjd.validate(r)
                qid = next(iter(r["questions"]))
                if qid in seen:
                    rep["errors"].append(f"duplicate qid {qid}")
                seen.add(qid)
                m = r["meta"]
                o = date.fromisoformat(m["origin"])
                r0, r1 = (date.fromisoformat(x) for x in m["resolution"])
                lo, hi = win[split]
                if m["split"] != split or (lo and r0 < lo) or (hi and r1 > hi) or r0 != o:
                    rep["errors"].append(f"{qid}: resolution {m['resolution']} outside {split}")
                for d in date_re.findall(r["state"]):
                    if date.fromisoformat(d) > o or (date.fromisoformat(d) == o and f"Forecast date: Monday {d}"
                                                     not in r["state"]):
                        rep["errors"].append(f"{qid}: state mentions {d} >= origin {o}")
                        break
                if r["license"] != LICENSE or "GDELT" not in m.get("citation", ""):
                    rep["errors"].append(f"{qid}: licence / citation")
            units = list(cf.iter_units(str(data), [split], {p.stem}))
            for u in units:
                if cf.is_noncommercial(u):
                    rep["errors"].append(f"{u['qid']}: flagged non-commercial")
                if len(u["option_keys"]) > 52:
                    rep["errors"].append(f"{u['qid']}: K > 52")
            rep["units"][f"{split}/{p.stem}"] = len(units)
    rep["n_errors"] = len(rep["errors"])
    rep["errors"] = rep["errors"][:50]
    return rep


def cmd_questions(a):
    t0 = time.time()
    P = load_panel(a.panel)
    log(f"weekly panel: {P.W} weeks {P.week_start(0)}..{P.end_day}, {len(P.ccs)} countries, {len(P.dyads)} dyads, "
        f"good weeks {int(P.A['good'].sum())} ({time.time() - t0:.0f} s)")
    targets = json.loads(a.targets) if a.targets else DEFAULT_TARGETS
    if a.scale != 1.0:
        targets = {s: {f: max(1, int(round(n * a.scale))) for f, n in d.items()} for s, d in targets.items()}
    man = build_corpus(P, a.out, targets, a.seed, set(a.families.split(",")) if a.families else None,
                       set(a.splits.split(",")) if a.splits else None)
    man["build_secs"] = round(time.time() - t0, 1)
    (Path(a.out) / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"questions done in {man['build_secs']} s -> {a.out}")


def cmd_validate(a):
    rep = validate_corpus(a.data, Path(a.repo))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 1 if rep["n_errors"] else 0


# =========================================================================== CLI (keep LAST)
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("panel", help="download + aggregate the GDELT 2.0 export stream per UTC day (node)")
    p.add_argument("--root", default="/data/local/gdelt")
    p.add_argument("--start", default="20150218")
    p.add_argument("--end", default="20261003", help="last COMPLETE UTC day")
    p.add_argument("--workers", type=int, default=4, help="day processes (<= half the cores)")
    p.add_argument("--threads", type=int, default=8, help="download connections per day process")
    p.add_argument("--limit-days", type=int, default=0)
    p.add_argument("--refresh-master", action="store_true")
    p.set_defaults(func=cmd_panel)
    p = sub.add_parser("merge", help="cross-day dedup check, final parquet panel, top-200 dyads, lookups, manifest")
    p.add_argument("--root", default="/data/local/gdelt")
    p.add_argument("--start", default="20150218")
    p.add_argument("--end", default="20261003")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--top-dyads", type=int, default=200)
    p.add_argument("--no-fix", action="store_true", help="report cross-day duplicates without re-aggregating")
    p.set_defaults(func=cmd_merge)
    p = sub.add_parser("questions", help="build the four question families from the merged panel")
    p.add_argument("--panel", default="/data/local/gdelt/panel")
    p.add_argument("--out", default="/data/local/gdelt/jev_gdelt_v1")
    p.add_argument("--seed", default=str(SEED))
    p.add_argument("--targets", default="", help="JSON {split: {f1: n, ...}}; default DEFAULT_TARGETS")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every target (smoke runs)")
    p.add_argument("--families", default="", help="comma list of f1,f2,f3,f4 (default all)")
    p.add_argument("--splits", default="", help="comma list (default all)")
    p.set_defaults(func=cmd_questions)
    p = sub.add_parser("validate", help="schema / leakage / licence read-back of a built corpus")
    p.add_argument("--data", required=True)
    p.add_argument("--repo", default=str(HERE), help="repo root holding trainers/cache_features.py, build_jev_data.py")
    p.add_argument("--out", default="", help="also write the JSON report here")
    p.set_defaults(func=cmd_validate)
    for reg in REGISTRARS:
        reg(sub)
    a = ap.parse_args(argv)
    return a.func(a) or 0


REGISTRARS = globals().get("REGISTRARS", [])

if __name__ == "__main__":
    raise SystemExit(main())
