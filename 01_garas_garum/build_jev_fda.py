#!/usr/bin/env python3
"""
build_jev_fda.py -- P1 of research/realworld_data.md: FDA recall classification records built from the openFDA
enforcement bulk downloads (food, drug, device) plus FDA's device root-cause categories (device/recall), written in the
jev_data record schema (build_jev_data.record / validate) and read back through trainers/cache_features.iter_units.
Design and its reasons: research/fda_dmhc_pipeline.md.

Unit and group
  One product row of an enforcement report (`recall_number`). Group = `event_id`: the products of one recall event
  share a class, so every rate, cap and split works on whole events. In-event duplicates (same description, reason and
  distribution text) collapse to one row; at most --per-event rows per event are written.
  sources: fda_recalls/food, fda_recalls/drug, fda_recalls/device -> <out>/<split>/fda_recalls__<centre>.jsonl

Questions (qids are not benchmark qids; `check` runs synth_workflows.exclusion.exclusion_errors)
  recall_class        choice 3   class_i / class_ii / class_iii; criteria paraphrase FDA's definitions (21 CFR 7.3(m))
  distribution_reach  score 4    local < regional < multi-state < nationwide or international (a reading question)
  classified_quickly  noul       FDA classifies the recall within 60 days of its initiation
  root_cause_group    choice 5   device only: FDA's root-cause category, grouped
  hazard_kind         choice 5   TEACHER (T) question: written unlabelled to <out>/teacher_fill/{train,val}/ for the (48)
                                 labeller (trainers/claude_verbal_label.py); `merge-fill` attaches its labels later.
                                 Never for eval: real-world eval labels do not come from an LLM.

Labels (meta.label_source[qid] is the provenance tag; meta.label_method[qid] the exact construction)
  recall_class        expert. train / val: 0.5 * onehot(FDA class) + 0.5 * p_hat. p_hat is a hierarchical Dirichlet-
                      smoothed class rate over global > centre > centre x product type > ... x hazard group > ... x
                      initiator > ... x initiation year, computed LEAVE-EVENT-OUT (no product of the item's own event is
                      counted) on train only; val rows get train-only rates. Each level's prior strength is chosen by
                      leave-event-out log score. eval: one-hot; the smoothed target and the base rate sit in meta.
  distribution_reach  rule_exact: a parser over the distribution text exactly as shown in the state. Not asked when the
                      text names no place. `check` writes a 200-row sample for the manual agreement audit (>= 98%).
  classified_quickly  outcome: center_classification_date - recall_initiation_date <= 60 days; blended like
                      recall_class with a leave-event-out rate (global > centre > centre x year > centre x year x hazard);
                      one-hot in eval. Asked only when the record's split membership cannot depend on the answer:
                      initiated >= its split's first classification date and >= 365 days before the split's last one
                      (train: initiated <= 2023-12-31; eval: initiated 2025-07-01 .. export - 365 d; val: none).
  root_cause_group    expert: FDA root_cause_description (device/recall, joined on recall_number) mapped to 5 groups;
                      one-hot; 'Under Investigation by firm', 'Unknown/Undetermined by firm', 'Pending', missing -> not asked.

As-of rule (the state holds what was known when the firm initiated the recall)
  kept   centre, product type, product description, quantity, reason for recall, distribution pattern, initiator
         (voluntary / FDA mandated), how consignees were first notified, firm location (US state or country), initiation
         date, and the firm's recall history from EARLIER initiations only (distinct events in the prior 24 months, and
         those in the same hazard group; firms matched on a normalised `recalling_firm`).
  never  classification, center_classification_date, report_date, status, termination_date, the firm's name and
         address, code_info / lot numbers, openFDA annotations, GMDN fields.
  Classification wording: a row whose state text mentions a recall or device class ('Class I/II/III/1/2/3') or FDA's
  class definitions is dropped; `check` asserts that no written state matches CLASS_WORDS.

Splits (by FDA's decision, center_classification_date; whole events; an event spanning two windows is dropped)
  train  classified <= 2024-12-31     val  2025-01-01 .. 2025-06-30     eval  classified >= 2025-07-01

Usage (stdlib + numpy + scipy; laptop is fine: ~390 MB is downloaded, slim extracts ~40 MB, corpus ~25 MB)
  python3 build_jev_fda.py fetch --cache /tmp/fda_jev/cache          # keyless bulk zips -> slim extracts (zips deleted)
  python3 build_jev_fda.py build --cache /tmp/fda_jev/cache --out /tmp/fda_jev
  python3 build_jev_fda.py check --out /tmp/fda_jev                  # read-back, scans, leak probe, label stats
  python3 build_jev_fda.py merge-fill --out /tmp/fda_jev --fill-labelled <labeller output dir>
  python3 build_jev_fda.py --self-test
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import importlib.util
import io
import json
import math
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_jev_data import choice_q, noul_q, option_keys, record, score_q, sha, validate  # noqa: E402

UA = "jev-fda-research/0.1 (research dataset build; openFDA bulk downloads, keyless)"
DOWNLOAD_JSON = "https://api.fda.gov/download.json"
ENDPOINTS = {"food": ("food", "enforcement"), "drug": ("drug", "enforcement"), "device": ("device", "enforcement"),
             "device_recall": ("device", "recall")}
CENTRES = ("food", "drug", "device")
LICENSE = ("openFDA, CC0 1.0 (https://open.fda.gov/license/); no GMDN fields. Enforcement reports (food, drug, device) "
           "and device recalls, bulk downloads of 2026-09-28.")
LAMBDA = 0.5
TRAIN_END, VAL_START, VAL_END, EVAL_START = date(2024, 12, 31), date(2025, 1, 1), date(2025, 6, 30), date(2025, 7, 1)
QUICK_DAYS, QUICK_HORIZON_DAYS, HISTORY_DAYS = 60, 365, 730
TYPED_GOLD = {"mean_max_p": 0.659, "mean_entropy_nats": 0.767}


# --------------------------------------------------------------------------- small utils (shared with build_jev_dmhc)
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hkey(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def jdump(obj):
    """The benchmark's state surface: compact JSON, keys sorted at every level, ', ' / ': ' separators."""
    return json.dumps(obj, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)


def clip(text, n):
    """Clip free text at a word boundary (ellipsis marks the cut)."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
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
    tmp.rename(path)
    return n


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


class _DefaultPortRedirect(urllib.request.HTTPRedirectHandler):
    """Drop an explicit default port from a redirect target. data.chhs.ca.gov redirects to a presigned S3 URL written as
    https://s3.amazonaws.com:443/...; urllib would send 'Host: s3.amazonaws.com:443', which breaks the host-signed
    signature (403). curl normalises the port away; so do we."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newurl = re.sub(r"^(https://[^/:?#]+):443(?=[/?#]|$)", r"\1", newurl)
        newurl = re.sub(r"^(http://[^/:?#]+):80(?=[/?#]|$)", r"\1", newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_DefaultPortRedirect)


def http_get(url, dest=None, timeout=1800, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with _OPENER.open(req, timeout=timeout) as r:
                if dest is None:
                    return r.read()
                tmp = Path(str(dest) + ".part")
                with open(tmp, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                tmp.rename(dest)
                return dest
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            log(f"  retry {k + 1} {url}: {type(e).__name__}: {e}")
            time.sleep(5 * (k + 1))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iter_results(zpath, chunk=1 << 20):
    """Stream the objects of the top-level "results" array of an openFDA download zip without loading the whole file
    (device/recall is 1.5 GB of JSON)."""
    z = zipfile.ZipFile(zpath)
    dec = json.JSONDecoder()
    with z.open(z.namelist()[0]) as raw:
        f = io.TextIOWrapper(raw, encoding="utf-8")
        buf = ""
        while True:
            more = f.read(chunk)
            if not more:
                raise ValueError(f"{zpath}: no results array")
            buf += more
            i = buf.find('"results"')
            j = buf.find("[", i) if i >= 0 else -1
            if j >= 0:
                buf = buf[j + 1:]
                break
        eof = False
        while True:
            k, n = 0, len(buf)
            while k < n and buf[k] in " \t\r\n,":
                k += 1
            if k < n and buf[k] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, k)
            except ValueError:
                if eof:
                    raise
                more = f.read(chunk)
                eof = not more
                buf = buf[k:] + more
                continue
            yield obj
            buf = buf[end:]


def ymd(s):
    """openFDA YYYYMMDD (or YYYY-MM-DD) -> date; None if missing or implausible."""
    s = re.sub(r"\D", "", str(s or ""))
    if len(s) != 8:
        return None
    try:
        d = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None
    return d if date(1990, 1, 1) <= d <= date(2030, 1, 1) else None


# --------------------------------------------------------------------------- labels: hierarchical leave-group-out rates
def encode_levels(keys_by_level):
    import numpy as np
    out = []
    for keys in keys_by_level:
        d = {}
        out.append(np.array([d.setdefault(k, len(d)) for k in keys], dtype=np.int64))
    return out


def group_out_rates(level_ids, y, K, fit_mask, groups, kappas):
    """Hierarchical Dirichlet-smoothed class rates, p_L = (n_L + kappa_L p_{L-1}) / (N_L + kappa_L), level 0 coarsest.

    Only `fit_mask` items with y >= 0 are counted. Every count an item sees has its whole GROUP removed (leave-group-out),
    so neither its own outcome nor a sibling product's from the same event enters its rate. Items outside the fit set
    (val, eval) belong to groups with no counted members and so get the fit-set rates unchanged. Returns [N, K]."""
    import numpy as np
    N = len(y)
    own = np.zeros((N, K))
    idx = np.where(fit_mask & (y >= 0))[0]
    own[idx, y[idx]] = 1.0
    G = int(groups.max()) + 1
    p = np.full((N, K), 1.0 / K)
    for L, ids in enumerate(level_ids):
        cnt = np.zeros((int(ids.max()) + 1, K))
        np.add.at(cnt, ids[idx], own[idx])
        _, inv = np.unique(ids * G + groups, return_inverse=True)
        gcnt = np.zeros((int(inv.max()) + 1, K))
        np.add.at(gcnt, inv[idx], own[idx])
        n = cnt[ids] - gcnt[inv]
        p = (n + kappas[L] * p) / (n.sum(1, keepdims=True) + kappas[L])
    return p


def fit_kappas(level_ids, y, K, fit_mask, groups, grid=(0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 2000)):
    """Coordinate search of each level's prior strength by mean leave-group-out log score on the fit set."""
    import numpy as np
    kap = [20.0] * len(level_ids)
    idx = np.where(fit_mask & (y >= 0))[0]

    def score(k):
        p = group_out_rates(level_ids, y, K, fit_mask, groups, k)
        return float(np.mean(np.log(np.clip(p[idx, y[idx]], 1e-12, 1))))
    best = score(kap)
    for _ in range(2):
        for L in range(len(kap)):
            for g in grid:
                trial = kap[:L] + [float(g)] + kap[L + 1:]
                s = score(trial)
                if s > best + 1e-9:
                    best, kap = s, trial
    return kap, best


def calib_report(p, y):
    """Brier (multiclass), log loss, accuracy and 15-bin top-label ECE of probabilities p [n, K] against classes y."""
    import numpy as np
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    K = p.shape[1]
    oh = np.eye(K)[y]
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(np.mean(conf[bins == b]) - np.mean(pred[bins == b] == y[bins == b])) * np.mean(bins == b)
              for b in range(15) if np.any(bins == b))
    return {"n": int(len(y)), "brier": round(float(np.mean(np.sum((p - oh) ** 2, 1))), 5),
            "log_loss": round(float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))), 5),
            "accuracy": round(float(np.mean(pred == y)), 5), "ece15_top_label": round(float(ece), 5)}


def blend(onehot_idx, p, K, lam):
    import numpy as np
    t = (1 - lam) * np.asarray(p, dtype=float)
    t[onehot_idx] += lam
    return t / t.sum()


def lab_dict(keys, vec):
    """A label dict over `keys` that sums to 1 after rounding to 6 decimals (the last key absorbs the rounding)."""
    vals = [max(0.0, float(v)) for v in vec]
    s = sum(vals)
    vals = [round(v / s, 6) for v in vals]
    rest = round(1.0 - sum(vals[:-1]), 6)
    if rest < 0:
        vals = [v / sum(vals) for v in vals]
    else:
        vals[-1] = rest
    return dict(zip(keys, vals))


def onehot_dict(keys, k):
    return {x: (1.0 if x == k else 0.0) for x in keys}


def entropy(v):
    return -sum(x * math.log(x) for x in v if x > 0)


# --------------------------------------------------------------------------- checks shared with build_jev_dmhc
def load_cf():
    """trainers/cache_features.py as every downstream script imports it (torch / transformers must be importable;
    stub modules on PYTHONPATH are enough: iter_units and question_options use neither)."""
    spec = importlib.util.spec_from_file_location("cache_features", HERE / "trainers" / "cache_features.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


def read_corpus(outdir, prefix, splits=("train", "val", "eval")):
    recs = defaultdict(list)
    for sp in splits:
        for p in sorted((Path(outdir) / sp).glob(f"{prefix}*.jsonl")):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    recs[sp].append(json.loads(line))
    return recs


def readback(outdir, recs, splits=("train", "val", "eval")):
    """Validate every record, round-trip every label through cf.question_options, and enumerate cf.iter_units."""
    cf = load_cf()
    roundtrip = 0
    for sp, v in recs.items():
        for r in v:
            validate(r)
            for qid, q in r["questions"].items():
                keys, texts, target = cf.question_options(q, r["labels"][qid])
                assert keys == option_keys(q), (r["id"], qid)
                assert [r["labels"][qid][k] for k in keys] == target, (r["id"], qid)
                assert abs(sum(target) - 1) < 1e-6 and all(t >= 0 for t in target), (r["id"], qid)
                assert len(texts) == len(keys) and all(str(t).strip() for t in texts), (r["id"], qid)
                roundtrip += 1
    units, kinds, per_src = Counter(), Counter(), Counter()
    for u in cf.iter_units(str(outdir), list(splits), set(), None):
        assert abs(sum(u["target"]) - 1) < 1e-6 and 2 <= len(u["target"]) <= 52, u["record_id"]
        units[(u["split"], u["qtype"])] += 1
        kinds[(u["split"], u["label_kind"])] += 1
        per_src[(u["split"], u["source"])] += 1
    return {"labels_round_tripped": roundtrip,
            "iter_units_total": sum(units.values()),
            "units_by_split_type": {f"{s}/{t}": n for (s, t), n in sorted(units.items())},
            "units_by_split_source": {f"{s}/{x}": n for (s, x), n in sorted(per_src.items())},
            "label_kind": {f"{s}/{k}": n for (s, k), n in sorted(kinds.items())},
            "records": {sp: len(v) for sp, v in recs.items()}}


def label_stats(recs):
    """Mean max-p and entropy (nats) per split and per question, to set beside typed-decisions' gold (0.659 / 0.767)."""
    out = {}
    for sp, v in recs.items():
        mx = en = 0.0
        n = 0
        perq = defaultdict(lambda: [0.0, 0.0, 0, Counter()])
        for r in v:
            for qid, lab in r["labels"].items():
                vals = list(lab.values())
                mx += max(vals)
                en += entropy(vals)
                n += 1
                pq = perq[qid]
                pq[0] += max(vals)
                pq[1] += entropy(vals)
                pq[2] += 1
                pq[3][r["meta"]["label_source"][qid]] += 1
        if n:
            out[sp] = {"units": n, "mean_max_p": round(mx / n, 4), "mean_entropy_nats": round(en / n, 4),
                       "per_question": {q: {"n": c, "mean_max_p": round(s / c, 4), "mean_entropy_nats": round(e / c, 4),
                                            "label_source": dict(ls)}
                                        for q, (s, e, c, ls) in sorted(perq.items())}}
    out["typed_decisions_gold_reference"] = TYPED_GOLD
    return out


def dedup_report(recs):
    ids = [r["id"] for v in recs.values() for r in v]
    groups = {sp: {r["meta"]["group"] for r in v} for sp, v in recs.items()}
    states = {sp: {r["state"] for r in v} for sp, v in recs.items()}
    out = {"duplicate_ids": len(ids) - len(set(ids))}
    for a, b in (("train", "val"), ("train", "eval"), ("val", "eval")):
        out[f"groups_{a}_and_{b}"] = len(groups.get(a, set()) & groups.get(b, set()))
        out[f"identical_states_{a}_and_{b}"] = len(states.get(a, set()) & states.get(b, set()))
    out["duplicate_states_within_train"] = len(recs.get("train", [])) - len(states.get("train", set()))
    return out


def run_exclusion(workflow, schema_qids):
    """synth_workflows.exclusion.exclusion_errors on the schemas this builder writes (qid sets per schema)."""
    try:
        from synth_workflows.exclusion import exclusion_errors

        class _S:
            def __init__(self, name, qs):
                self.name, self._qs = name, qs

            def questions(self):
                return self._qs

        class _W:
            def __init__(self, name, schemas):
                self.name, self.schemas = name, schemas
        wf = _W(workflow, [_S(n, {q: None for q in qs}) for n, qs in schema_qids.items()])
        return exclusion_errors(wf) or "none"
    except Exception as e:  # noqa: BLE001
        return f"could not run: {type(e).__name__}: {e}"


def leak_probe(recs, qid, keys, folds=5, dim=1 << 15, seed=0, text_of=None):
    """Cross-validated hashed bag-of-words (+ bigrams) multinomial logistic, state -> realised label, against the
    train-only base rate stored in meta. A probe far above the base rate whose top features are outcome words means a
    leak; hazard / request words are evidence, not leakage."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.sparse import csr_matrix
    text_of = text_of or (lambda r: r["state"])

    def feats(text):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        f = {int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % dim for t in toks}
        f |= {int(hashlib.md5((a + "_" + b).encode()).hexdigest()[:8], 16) % dim for a, b in zip(toks, toks[1:])}
        return sorted(f), toks
    rows, ys, base, vocab = [], [], [], {}
    for r in recs:
        f, toks = feats(text_of(r))
        rows.append(f)
        for t in toks:
            vocab.setdefault(int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % dim, t)
        for a_, b_ in zip(toks, toks[1:]):
            vocab.setdefault(int(hashlib.md5((a_ + "_" + b_).encode()).hexdigest()[:8], 16) % dim, a_ + " " + b_)
        ys.append(keys.index(r["meta"]["raw_outcome"][qid]))
        br = (r["meta"].get("base_rate") or {}).get(qid)
        base.append([br[k] for k in keys] if br else None)
    n, K = len(ys), len(keys)
    X = csr_matrix((np.ones(sum(len(f) for f in rows)), np.concatenate([np.array(f, dtype=np.int64) for f in rows]),
                    np.cumsum([0] + [len(f) for f in rows])), shape=(n, dim))
    y = np.array(ys)
    fold = np.random.default_rng(seed).integers(0, folds, n)
    P = np.zeros((n, K))

    def fit(Xt, yt, lam=1.0):
        Y = np.eye(K)[yt]

        def f(w):
            W, b = w[:-K].reshape(dim, K), w[-K:]
            Z = Xt @ W + b
            Z -= Z.max(1, keepdims=True)
            lse = np.log(np.exp(Z).sum(1))
            Pp = np.exp(Z - lse[:, None])
            obj = -((Z * Y).sum(1) - lse).sum() + 0.5 * lam * (W ** 2).sum()
            G = Xt.T @ (Pp - Y)
            return obj / len(yt), np.concatenate([(G + lam * W).ravel(), (Pp - Y).sum(0)]) / len(yt)
        return minimize(f, np.zeros(dim * K + K), jac=True, method="L-BFGS-B", options={"maxiter": 200}).x
    for k in range(folds):
        tr, te = fold != k, fold == k
        w = fit(X[tr], y[tr])
        Z = X[te] @ w[:-K].reshape(dim, K) + w[-K:]
        Z -= Z.max(1, keepdims=True)
        P[te] = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    have = np.array([b is not None for b in base])
    B = np.array([b if b is not None else [1.0 / K] * K for b in base])
    prior = np.bincount(y, minlength=K) / n
    w = fit(X, y)
    W = w[:-K].reshape(dim, K)
    return {"n": n, "bow_cv": calib_report(P, y),
            "base_rate_train_only": calib_report(B[have], y[have]) if have.any() else None,
            "global_prior": calib_report(np.tile(prior, (n, 1)), y),
            "top_features": {keys[j]: [vocab.get(int(ix), str(ix)) for ix in np.argsort(-(W[:, j] - W.mean(1)))[:15]]
                             for j in range(K)}}


def write_fill(outdir, sp, source_file, fill_rec):
    p = Path(outdir) / "teacher_fill" / sp / source_file
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fill_rec, ensure_ascii=False) + "\n")


def fill_record(rec, sp, questions):
    """An unlabelled teacher-fill record for the (48) labeller: the main record's state and the T questions."""
    return {"id": rec["id"] + "-T", "base_id": rec["id"], "source": rec["source"], "license": rec["license"],
            "split": sp, "unlabelled": True, "state": rec["state"], "questions": questions,
            "meta": {"main_id": rec["id"], "group": rec["meta"]["group"],
                     "label_source": {q: "teacher_verbal:pending" for q in questions}}}


def merge_fill(outdir, labelled_dir, prefix):
    """Attach teacher labels (claude_verbal_label.py output, same layout as <out>/teacher_fill) to the main records."""
    lab = {}
    for p in Path(labelled_dir).rglob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("labels"):
                lab[r["meta"]["main_id"]] = r
    stats = Counter()
    for sp in ("train", "val", "eval"):
        for p in sorted((Path(outdir) / sp).glob(f"{prefix}*.jsonl")):
            out = []
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                t = lab.get(r["id"])
                if t:
                    k = max((len(v.get("samples", {}).get(q, [])) for v in t["meta"].get("labeller", {}).get("teachers", {}).values()
                             for q in t["questions"]), default=0)
                    for q, qd in t["questions"].items():
                        r["questions"][q] = qd
                        r["labels"][q] = t["labels"][q]
                        r["label_kind"][q] = "soft"
                        r["meta"]["label_source"][q] = f"teacher_verbal:{k}"
                        r["meta"].setdefault("label_method", {})[q] = "(48) teacher mean distribution (meta.teacher_fill)"
                    r["meta"]["teacher_fill"] = {"fill_id": t["id"], "labeller": t["meta"].get("labeller")}
                    validate(r)
                    stats[f"{sp}_merged"] += 1
                else:
                    stats[f"{sp}_no_fill_label"] += 1
                out.append(r)
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    log(f"merge-fill: {dict(stats)}")
    return stats


# --------------------------------------------------------------------------- vocabulary: questions
CLASS_KEYS = ["class_i", "class_ii", "class_iii"]
CLASS_OF = {"Class I": "class_i", "Class II": "class_ii", "Class III": "class_iii"}
CLASS_CRITERIA = {
    "class_i": "Class I: there is a reasonable probability that using or being exposed to the product will cause "
               "serious adverse health consequences or death.",
    "class_ii": "Class II: using or being exposed to the product may cause temporary or medically reversible adverse "
                "health consequences, or the probability of serious adverse health consequences is remote.",
    "class_iii": "Class III: using or being exposed to the product is not likely to cause adverse health consequences.",
}
CLASS_INSTR = ("Which FDA recall class fits this product's hazard? Use FDA's class definitions given in the options.")
REACH_LEVELS = [
    "local: a single US state or territory, or one place within it",
    "regional: two to five US states, and no other country",
    "multi-state: six or more US states, not described as nationwide, and no other country",
    "nationwide or international: described as nationwide or throughout the US, or reaching any country outside the US",
]
REACH_INSTR = "According to the distribution description, how far did the recalled product reach?"
QUICK_INSTR = "FDA classifies this recall within 60 days of the date the firm initiated it."
QUICK_CRITERIA = {"yes": "FDA's classification of the recall is dated no more than 60 days after the recall initiation "
                         "date.",
                  "no": "FDA classifies the recall more than 60 days after it was initiated."}
ROOT_KEYS = ["component_or_supplier", "device_design", "labelling_or_other", "manufacturing_process", "software"]
ROOT_CRITERIA = {
    "component_or_supplier": "A purchased part or material: nonconforming or contaminated components or materials, "
                             "component design or selection, component or vendor change control, counterfeit parts.",
    "device_design": "The device's own design: an inadequate device or package design or specification.",
    "labelling_or_other": "Labelling, regulatory status or another cause: labelling errors, design or mix-ups, a missing "
                          "or wrong expiry date, use error, marketing without clearance or approval, radiation-control "
                          "requirements, or causes FDA records only as 'other'.",
    "manufacturing_process": "How the device was made, packaged, stored or released: process control or design, process "
                             "or packaging change control, employee error, equipment maintenance, environmental control, "
                             "storage, material mix-ups.",
    "software": "Software: its design, a software change, its deployment in manufacturing, or its behaviour in the use "
                "environment.",
}
ROOT_INSTR = "Which group does the root cause FDA recorded for this device recall belong to?"
ROOT_GROUP = {
    "Device Design": "device_design", "Package design/selection": "device_design",
    "Process control": "manufacturing_process", "Process design": "manufacturing_process",
    "Process change control": "manufacturing_process", "Employee error": "manufacturing_process",
    "Equipment maintenance": "manufacturing_process", "Environmental control": "manufacturing_process",
    "Storage": "manufacturing_process", "Packaging": "manufacturing_process",
    "Packaging process control": "manufacturing_process", "Packaging change control": "manufacturing_process",
    "Mixed-up of materials/components": "manufacturing_process", "Manufacturing material removal": "manufacturing_process",
    "Reprocessing Controls": "manufacturing_process", "Finished device change control": "manufacturing_process",
    "Release of Material/Component prior to receiving test results": "manufacturing_process",
    "Nonconforming Material/Component": "component_or_supplier", "Component design/selection": "component_or_supplier",
    "Component change control": "component_or_supplier", "Vendor change control": "component_or_supplier",
    "Material/Component Contamination": "component_or_supplier", "Counterfeit": "component_or_supplier",
    "Software design": "software", "Software Design Change": "software", "Software change control": "software",
    "Software Manufacturing/Software Deployment": "software", "Software design (manufacturing process)": "software",
    "Software in the Use Environment": "software",
    "Error in labeling": "labelling_or_other", "Labeling design": "labelling_or_other",
    "Labeling Change Control": "labelling_or_other", "Labeling mix-ups": "labelling_or_other",
    "Labeling False and Misleading": "labelling_or_other", "Incorrect or no expiration date": "labelling_or_other",
    "Use error": "labelling_or_other", "No Marketing Application": "labelling_or_other", "PMA": "labelling_or_other",
    "Radiation Control for Health and Safety Act": "labelling_or_other", "Other": "labelling_or_other",
}
ROOT_NOT_ASKED = {"Under Investigation by firm", "Unknown/Undetermined by firm", "Pending", None, ""}
HAZARD_KEYS = ["foreign_material", "labelling_or_process", "microbial", "potency_or_quality", "undeclared_allergen"]
HAZARD_CRITERIA = {
    "foreign_material": "Foreign material: pieces of metal, glass, plastic, rubber or other matter that should not be in "
                        "the product, or a physical defect that can injure.",
    "labelling_or_process": "Labelling or process: wrong, missing or misleading labelling or instructions, or a "
                            "manufacturing, storage or regulatory-process failure without a specific contaminant.",
    "microbial": "Microbial: contamination or possible contamination with bacteria, moulds, viruses or other microbes, "
                 "or a lack of sterility assurance.",
    "potency_or_quality": "Potency or quality: the product does not perform or meet its specification (strength, purity, "
                          "impurities, dissolution, malfunction, inaccurate results).",
    "undeclared_allergen": "Undeclared allergen or ingredient: the product contains an allergen or active ingredient its "
                           "label does not declare.",
}
HAZARD_INSTR = "What kind of hazard led to this recall?"

CLASS_WORDS = re.compile(
    r"\bclass\s*-?\s*(?:i{1,3}|iv|[123])\b|\bclass(?:es)?\s+(?:one|two|three)\b|serious adverse health consequences|"
    r"medically reversible|not likely to cause adverse|remote probability|\brecall class", re.I)

# --------------------------------------------------------------------------- derived pre-decision fields
FIRM_SUFFIX = re.compile(r"\b(?:inc|incorporated|llc|l l c|ltd|limited|corp|corporation|co|company|lp|llp|plc|sa|s a|"
                         r"gmbh|ag|bv|nv|the|dba|d b a)\b")


def firm_key(name):
    s = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower())
    return re.sub(r"\s+", " ", FIRM_SUFFIX.sub(" ", s)).strip()


_HZ = {
    "food": [
        ("listeria", r"listeria"), ("salmonella", r"salmonell"),
        ("e_coli", r"\be\.?\s?coli\b|\bstec\b|o157|shiga"),
        ("botulism_or_underprocessing", r"botul|clostridium|c\.\s?bot|under-?process|low.acid|acidified|swell|bloat"),
        ("other_microbial", r"bacteri|microb|\bmou?ld|yeast|cronobacter|bacillus|staph|vibrio|hepatitis|norovirus|"
                            r"cyclospora|burkholderia|pathogen|fung"),
        ("undeclared_allergen", r"undeclared|allergen|\bmilk\b|\begg|peanut|tree nut|\bnuts?\b|almond|cashew|pecan|walnut|"
                                r"\bsoy|wheat|gluten|sesame|shellfish|crustacean"),
        ("foreign_material", r"foreign|metal|glass|plastic|rubber|\bwood|stone|fragment|piece|shard|wire|bone"),
        ("chemical", r"\blead\b|arsenic|pesticide|chemical|toxin|aflatoxin|histamine|scombro|sulfite|cleaning|"
                     r"cadmium|mercury|patulin|ammonia"),
        ("labelling", r"label|misbrand|mislabel|net weight|nutrition facts"),
    ],
    "drug": [
        ("sterility", r"steril|microb|bacteri|fung|mou?ld|yeast|contaminat\w* with|endotoxin|burkholderia"),
        ("unapproved_or_undeclared", r"unapproved|undeclared|sildenafil|tadalafil|hidden|not approved|without an approved"),
        ("impurity", r"impurit|ndma|nitrosamine|benzene|degrad|n-nitroso"),
        ("potency", r"potency|subpotent|super-?potent|assay"),
        ("dissolution_or_spec", r"dissolution|specification|\boos\b|stability|uniformity"),
        ("particulate_or_foreign", r"particulate|foreign|glass|metal|particle|fiber"),
        ("cgmp", r"cgmp|good manufacturing|gmp"),
        ("labelling", r"label|mislabel|misbrand|incorrect (?:strength|instructions)|mix-?up|package insert"),
        ("packaging_or_container", r"packag|container|seal|leak|crack|defective"),
    ],
    "device": [
        ("software", r"software|firmware|algorithm"),
        ("sterility", r"steril|contaminat|bacteri|endotoxin|bioburden|fung|mou?ld"),
        ("labelling", r"label|instructions for use|\bifu\b|mislabel|expir"),
        ("unapproved", r"without (?:a |an )?(?:510|clearance|approval|premarket)|unapproved|not cleared|not been cleared"),
        ("electrical_or_thermal", r"electric|shock|overheat|fire|burn|thermal|battery|short circuit|spark|smoke"),
        ("mechanical_failure", r"break|broken|crack|fractur|detach|leak|separat|tear|rupture|loose"),
        ("inaccurate_result", r"inaccura|incorrect result|false (?:positive|negative)|erroneous|calibrat|measurement"),
        ("malfunction", r"fail|malfunction|may not|does not|inoperab|alarm|stop|error"),
        ("component_or_material", r"component|material|supplier|vendor|part"),
    ],
}
HAZARD_RX = {c: [(g, re.compile(p, re.I)) for g, p in v] for c, v in _HZ.items()}


def hazard_group(centre, reason):
    for g, rx in HAZARD_RX[centre]:
        if rx.search(reason or ""):
            return g
    return "other"


def product_type(centre, r):
    desc = (r.get("product_description") or "") + " " + (r.get("reason_for_recall") or "")
    if centre == "food":
        if re.search(r"supplement", desc, re.I):
            return "dietary supplement"
        if re.search(r"\b(?:dogs?|cats?|pets?|animal feed|livestock|horses?|cattle|poultry feed|bird ?seed|kibble|"
                     r"animal food|feed for)\b", desc, re.I):
            return "animal food"
        return "human food"
    if centre == "drug":
        ot = (r.get("openfda_product_type") or "").upper()
        if re.search(r"compound|outsourcing facility|503b|pharmacy", desc, re.I):
            return "compounded drug"
        if re.search(r"homeopath|\bhpus\b", desc, re.I):
            return "homeopathic drug"
        if "PRESCRIPTION" in ot or re.search(r"\brx\b|rx only|prescription", desc, re.I):
            return "prescription drug"
        if "OTC" in ot or re.search(r"\botc\b|drug facts|over[- ]the[- ]counter", desc, re.I):
            return "over-the-counter drug"
        return "drug (type not stated)"
    spec = (r.get("medical_specialty") or "").strip()
    return f"medical device: {spec.lower()}" if spec else "medical device (specialty not recorded)"


def firm_location(r):
    c = (r.get("country") or "").strip()
    st = (r.get("state") or "").strip()
    if c in ("United States", "US", "USA", ""):
        return f"United States ({st})" if st else "United States"
    return c


# --------------------------------------------------------------------------- distribution reach (rule-exact parser)
US = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
      "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
      "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
      "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
      "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
      "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
      "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia", "PR": "Puerto Rico", "GU": "Guam",
      "VI": "Virgin Islands", "AS": "American Samoa", "MP": "Northern Mariana Islands"}
_NAMES = sorted(US.values(), key=len, reverse=True)
STATE_NAME_RX = re.compile(r"\b(" + "|".join(re.escape(n) for n in _NAMES) + r")\b", re.I)
STATE_CODE_RX = re.compile(r"(?<![A-Za-z.])(" + "|".join(US) + r")(?![A-Za-z])")
AMBIGUOUS_CODES = {"IN", "OR", "ME", "OK", "HI", "OH", "DE", "LA", "AL", "CO", "ID", "MO", "NE", "PA", "MA", "MD", "VA",
                   "WA", "SC", "AS", "GA", "MS", "MT", "UT"}
COUNTRIES = (
    "Afghanistan|Albania|Algeria|Andorra|Angola|Antigua|Argentina|Armenia|Aruba|Australia|Austria|Azerbaijan|Bahamas|"
    "Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bermuda|Bhutan|Bolivia|Bosnia|Botswana|Brazil|Brunei|"
    "Bulgaria|Burkina|Burundi|Cambodia|Cameroon|Canada|Cayman|China|Colombia|Comoros|Congo|Costa Rica|Croatia|"
    "Curacao|Cyprus|Czech|Czechia|Denmark|Djibouti|Dominica|Dominican Republic|Ecuador|Egypt|El Salvador|"
    "Eritrea|Estonia|Eswatini|Ethiopia|Europe|European Union|Fiji|Finland|France|Gabon|Gambia|Germany|Ghana|Greece|"
    "Grenada|Guatemala|Guinea|Guyana|Haiti|Honduras|Hong Kong|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|"
    "Italy|Ivory Coast|Jamaica|Japan|Kazakhstan|Kenya|Korea|Kosovo|Kuwait|Kyrgyzstan|Laos|Latvia|Lesotho|"
    "Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Macau|Macao|Madagascar|Malawi|Malaysia|Maldives|Malta|"
    "Martinique|Mauritania|Mauritius|Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nepal|"
    "Netherlands|New Zealand|Nicaragua|Niger|Nigeria|North Macedonia|Macedonia|Norway|Oman|Pakistan|Palestine|"
    "Paraguay|Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Saudi Arabia|Scotland|Senegal|Serbia|"
    "Seychelles|Sierra Leone|Singapore|Slovakia|Slovenia|Somalia|South Africa|Spain|Sri Lanka|Sudan|Suriname|Sweden|"
    "Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Togo|Tonga|Trinidad|Tunisia|Turkiye|Turkmenistan|"
    "Uganda|Ukraine|United Arab Emirates|UAE|United Kingdom|Great Britain|Uruguay|Uzbekistan|Venezuela|Vietnam|"
    "Viet Nam|Wales|Yemen|Zambia|Zimbabwe|Asia|Africa|South America|Central America|Caribbean|Middle East|Latin America|"
    "Canadian|Overseas")
FOREIGN_RX = re.compile(
    r"\b(?:world\s?-?wide|international\w*|foreign|overseas|export\w*|outside (?:of )?(?:the )?(?:U\.?S|united states|"
    r"country)|countries|OUS|O\.U\.S)\b|(?<!New )\bMexico\b|\bPanama\b(?! City)|\bUK\b|\bU\.K\b|\b(?:" + COUNTRIES + r")\b",
    re.I)
# names that are also US places or common words ("New England", "Holland, MI", "Peru, IN", "turkey", a firm called
# Jordan): matched only capitalised, and England not after "New"
FOREIGN_CASED_RX = re.compile(r"(?<!New )(?<!NEW )\bEngland\b|\b(?:Turkey|Jordan|Peru|Cuba|Chile|Mali)\b(?!,? (?:IN|NY|ME|MO|IL|IA|VT|OH|KS)\b)")
NATION_RX = re.compile(
    r"\bnation\s?-?wide\b|\bnationally\b|\bthroughout (?:the )?(?:entire )?(?:U\.?S\.?A?|United States|country|nation|"
    r"states)\b|\ball (?:50|fifty) states\b|\bacross (?:the )?(?:U\.?S\.?A?|United States|country|nation)\b|"
    r"\b(?:U\.?S\.?A?|United States) ?-?wide\b|\bcoast to coast\b|\bcountry\s?-?wide\b", re.I)
# negated foreign / international mentions ("No foreign distribution", "FOREIGN: NONE", "International: None reported",
# "There are no Canada accounts") are removed before the foreign test; found by the first 200-row manual audit
NEG_FOREIGN_RX = re.compile(
    r"\b(?:foreign|international|OUS|overseas|export)\w*(?: distribution| consignees?| accounts?| customers?)?\s*[:\-\u2013]?\s*"
    r"(?:none|n/?a|no|0)\b(?: reported| known)?|"
    r"\b(?:no|not|none|without|zero)\b[^.;:]{0,40}?\b(?:foreign|international\w*|overseas|OUS|export\w*|outside (?:of )?"
    r"(?:the )?(?:U\.?S\.?A?|United States)|Canad\w*|Mexic\w*|countries)\b[^.;]{0,30}", re.I)
N_STATES_RX = re.compile(r"\b(\d{1,2})\s+(?:US |U\.S\. |different |additional )?states\b", re.I)
NOT_STATE_RX = re.compile(r"\bVA (?:hospital|medical|facilit|center|centre|system|pharmac|clinic)|\bM\.?D\.? (?:office|s\b)",
                          re.I)


def reach_level(text):
    """0 local, 1 regional (2-5 states), 2 multi-state (6+), 3 nationwide or international; None if no place is named.

    The rule is the one the score levels state, applied to the distribution text exactly as the state shows it."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return None, {}
    t2 = re.sub(r"\bWashington,?\s*D\.?\s?C\.?", " DC ", t)
    t2 = re.sub(r"\bP\.\s?R\.(?=\W|$)", " PR ", t2)
    t2 = NOT_STATE_RX.sub(" ", t2)
    t2 = NEG_FOREIGN_RX.sub(" ", t2)
    foreign = FOREIGN_RX.search(t2) or FOREIGN_CASED_RX.search(t2)
    nation = NATION_RX.search(t2)
    states = {next(c for c, n in US.items() if n.lower() == m.group(1).lower()) for m in STATE_NAME_RX.finditer(t2)}
    # "New Mexico", "West Virginia" etc. are matched whole (longest first); avoid the bare "Virginia" inside them
    letters = [ch for ch in t2 if ch.isalpha()]
    caps = sum(ch.isupper() for ch in letters) / max(1, len(letters))
    for m in STATE_CODE_RX.finditer(t2):
        code = m.group(1)
        if caps > 0.6 and code in AMBIGUOUS_CODES:
            before, after = t2[max(0, m.start() - 3):m.start()], t2[m.end():m.end() + 3]
            if not (re.search(r"[,;/&]\s*$|\band\s*$", before, re.I) or re.match(r"\s*[,;/&]|\s+and\b", after, re.I)):
                continue
        states.add(code)
    stated = max([int(m.group(1)) for m in N_STATES_RX.finditer(t2)] or [0])
    n = max(len(states), stated)
    info = {"states": sorted(states), "stated_count": stated or None,
            "foreign": foreign.group(0) if foreign else None, "nationwide": nation.group(0) if nation else None}
    if foreign or nation:
        return 3, info
    if n >= 6:
        return 2, info
    if n >= 2:
        return 1, info
    if n == 1:
        return 0, info
    return None, info


# --------------------------------------------------------------------------- fetch
ENF_FIELDS = ["recall_number", "event_id", "product_type", "classification", "recalling_firm", "state", "country",
              "voluntary_mandated", "initial_firm_notification", "distribution_pattern", "product_description",
              "product_quantity", "reason_for_recall", "recall_initiation_date", "center_classification_date",
              "report_date", "status", "termination_date"]


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def slim(name, r):
    if name == "device_recall":
        of = r.get("openfda") or {}
        return {"recall_number": r.get("product_res_number"), "event_id": r.get("res_event_number"),
                "root_cause_description": r.get("root_cause_description"),
                "medical_specialty": _first(of.get("medical_specialty_description")),
                "event_date_initiated": r.get("event_date_initiated")}
    out = {k: r.get(k) for k in ENF_FIELDS}
    if name == "drug":
        of = r.get("openfda") or {}
        out["openfda_product_type"] = _first(of.get("product_type"))
    return out


def fetch(a):
    cache = Path(a.cache)
    raw = Path(a.raw_dir) if a.raw_dir else cache / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    dj = json.loads(http_get(DOWNLOAD_JSON))
    prov = {"fetched": datetime.now().isoformat(timespec="seconds"), "download_json": DOWNLOAD_JSON,
            "download_json_meta": dj.get("meta"), "endpoints": {}}
    for name, (cat, ep) in ENDPOINTS.items():
        info = dj["results"][cat][ep]
        rows, files = [], []
        for part in info["partitions"]:
            url = part["file"]
            dest = raw / url.rsplit("/", 1)[1]
            if not dest.exists():
                log(f"download {url} ({part.get('size_mb')} MB)")
                http_get(url, dest)
            digest = file_sha256(dest)
            n0 = len(rows)
            for r in iter_results(dest):
                rows.append(slim(name, r))
            files.append({"url": url, "size_mb": part.get("size_mb"), "records_declared": part.get("records"),
                          "records_read": len(rows) - n0, "sha256": digest})
            if not a.keep_raw:
                dest.unlink()
        n = write_gz(cache / f"{name}.jsonl.gz", rows)
        prov["endpoints"][name] = {"export_date": info.get("export_date"), "total_records": info.get("total_records"),
                                   "rows_extracted": n, "files": files}
        log(f"{name}: {n:,} rows (export {info.get('export_date')})")
    (cache / "fetch_provenance.json").write_text(json.dumps(prov, indent=1))


# --------------------------------------------------------------------------- build
def render_memo(state):
    lines = []
    for sec in ("product", "recall", "firm_history", "timing"):
        if sec not in state:
            continue
        lines.append(sec.replace("_", " ").upper() + ":")
        for k, v in state[sec].items():
            lines.append(f"- {k.replace('_', ' ')}: {v}")
    return "\n".join(lines)


def render_form(state):
    p, r, h, t = state["product"], state["recall"], state["firm_history"], state["timing"]
    return "\n".join([
        f"FDA recall entry ({p['centre']} product)",
        f"Product type: {p['product_type']}",
        f"Product: {p['description']}",
        f"Quantity: {p['quantity']}",
        f"Reason for recall: {r['reason_for_recall']}",
        f"Distribution: {r['distribution_pattern']}",
        f"Initiated: {t['initiated']} ({r['initiated_by']}); first consignee notification: {r['consignee_notification']}",
        f"Firm location: {r['firm_location']}",
        f"Firm's other recall events initiated in the prior 24 months: {h['prior_recalls_24m']} "
        f"({h['prior_recalls_same_hazard_24m']} with the same kind of hazard)",
    ])


def render(state, surface):
    return jdump(state) if surface == "json" else render_memo(state) if surface == "memo" else render_form(state)


def split_of_event(dates):
    lo, hi = min(dates), max(dates)
    if hi <= TRAIN_END:
        return "train", "train: classified <= 2024-12-31"
    if lo >= VAL_START and hi <= VAL_END:
        return "val", "val: classified 2025-01-01 .. 2025-06-30"
    if lo >= EVAL_START:
        return "eval", "eval: classified >= 2025-07-01 (decided after 2025-06)"
    return None, "event classified across two split windows (dropped)"


def quick_window(sp, export):
    """Initiation dates for which a record's split cannot depend on whether it was classified within 60 days."""
    if sp == "train":
        return None, TRAIN_END - timedelta(days=QUICK_HORIZON_DAYS)
    if sp == "val":
        return VAL_START, VAL_END - timedelta(days=QUICK_HORIZON_DAYS)        # empty: 6 months < 365 days
    return EVAL_START, export - timedelta(days=QUICK_HORIZON_DAYS)


def stratified_take(items, cap, floor, key, rng):
    """Proportional stratified sample (by `key`) to about `cap` items, every stratum kept up to `floor`.
    Returns (taken, weight per stratum = population share / sample share)."""
    by = defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    N = len(items)
    if not cap or N <= cap:
        return items, {k: 1.0 for k in by}
    taken, weights = [], {}
    for k, v in sorted(by.items()):
        want = min(len(v), max(min(len(v), floor), round(cap * len(v) / N)))
        rng.shuffle(v)
        taken.extend(v[:want])
        weights[k] = want
    T = len(taken)
    weights = {k: round((len(by[k]) / N) / (w / T), 4) for k, w in weights.items()}
    return taken, weights


def build(a):
    import numpy as np
    cache, outdir = Path(a.cache), Path(a.out)
    prov = json.loads((cache / "fetch_provenance.json").read_text())
    export = max(date.fromisoformat(v["export_date"]) for v in prov["endpoints"].values())
    for sp in ("train", "val", "eval"):
        for d in (outdir / sp, outdir / "teacher_fill" / sp):
            if d.exists():
                for p in d.glob("fda_recalls__*.jsonl"):
                    p.unlink()
    stats = Counter()
    root = {r["recall_number"]: r for r in read_gz(cache / "device_recall.jsonl.gz")}
    rows = []
    for centre in CENTRES:
        for r in read_gz(cache / f"{centre}.jsonl.gz"):
            stats[f"read_{centre}"] += 1
            r["centre"] = centre
            if r.get("classification") not in CLASS_OF:
                stats["drop_not_classified"] += 1
                continue
            r["t0"], r["t1"] = ymd(r.get("recall_initiation_date")), ymd(r.get("center_classification_date"))
            if not r["t0"] or not r["t1"] or r["t0"] < date(2000, 1, 1):
                stats["drop_bad_dates"] += 1
                continue
            if centre == "device":
                rr = root.get(r["recall_number"])
                r["medical_specialty"] = rr.get("medical_specialty") if rr else None
                r["root_cause_description"] = rr.get("root_cause_description") if rr else None
                stats["device_linked_to_device_recall" if rr else "device_not_linked"] += 1
            r["hazard"] = hazard_group(centre, r.get("reason_for_recall"))
            r["ptype"] = product_type(centre, r)
            r["firm"] = firm_key(r.get("recalling_firm"))
            rows.append(r)
    log(f"rows with a class and valid dates: {len(rows):,}")
    # firm history: distinct events, by earliest initiation date, from every classified row (as-of t0 per row)
    ev_first, ev_firm, ev_haz = {}, {}, defaultdict(Counter)
    for r in rows:
        e = r["event_id"]
        ev_first[e] = min(ev_first.get(e, r["t0"]), r["t0"])
        ev_firm.setdefault(e, r["firm"])
        ev_haz[e][r["hazard"]] += 1
    firm_events, firm_haz_events = defaultdict(list), defaultdict(list)
    for e, d in ev_first.items():
        f = ev_firm[e]
        if not f:
            continue
        hz = ev_haz[e].most_common(1)[0][0]
        firm_events[f].append((d, e))
        firm_haz_events[(f, hz)].append((d, e))
    for v in list(firm_events.values()) + list(firm_haz_events.values()):
        v.sort()

    def prior(lst, t, e):
        lo = bisect.bisect_left(lst, (t - timedelta(days=HISTORY_DAYS), ""))
        hi = bisect.bisect_left(lst, (t, ""))
        return sum(1 for d, x in lst[lo:hi] if x != e)
    # class-wording filter, then in-event duplicate collapse
    kept, seen = [], set()
    for r in rows:
        txt = " ".join(str(r.get(k) or "") for k in ("product_description", "reason_for_recall", "distribution_pattern",
                                                      "product_quantity"))
        if CLASS_WORDS.search(txt):
            stats[f"drop_class_wording_{r['centre']}"] += 1
            continue
        dk = (r["event_id"], re.sub(r"\W+", " ", (r.get("product_description") or "").lower()).strip(),
              re.sub(r"\W+", " ", (r.get("reason_for_recall") or "").lower()).strip(),
              re.sub(r"\W+", " ", (r.get("distribution_pattern") or "").lower()).strip())
        if dk in seen:
            stats["collapse_in_event_duplicate"] += 1
            continue
        seen.add(dk)
        f = r["firm"]
        r["hist"] = {"prior_recalls_24m": prior(firm_events.get(f, []), r["t0"], r["event_id"]) if f else 0,
                     "prior_recalls_same_hazard_24m": prior(firm_haz_events.get((f, r["hazard"]), []), r["t0"],
                                                            r["event_id"]) if f else 0}
        kept.append(r)
    log(f"after class-wording filter and in-event collapse: {len(kept):,}")
    # splits by event
    ev_dates = defaultdict(list)
    for r in kept:
        ev_dates[r["event_id"]].append(r["t1"])
    ev_split = {e: split_of_event(ds) for e, ds in ev_dates.items()}
    items = []
    for r in kept:
        sp, why = ev_split[r["event_id"]]
        if sp is None:
            stats["drop_event_spans_split_windows"] += 1
            continue
        r["split"], r["split_reason"] = sp, why
        items.append(r)
    log(f"in a split: {len(items):,}  {Counter(r['split'] for r in items)}")
    # base rates over every split-assigned row (fit = train), before any cap
    N = len(items)
    gid = {e: i for i, e in enumerate(sorted({r["event_id"] for r in items}))}
    groups = np.array([gid[r["event_id"]] for r in items])
    fit = np.array([r["split"] == "train" for r in items])
    y = np.array([CLASS_KEYS.index(CLASS_OF[r["classification"]]) for r in items])
    cen = [r["centre"] for r in items]
    pt = [r["centre"] + "|" + r["ptype"] for r in items]
    hz = [p + "|" + r["hazard"] for p, r in zip(pt, items)]
    ini = [h + "|" + (r.get("voluntary_mandated") or "") for h, r in zip(hz, items)]
    yr = [i + "|" + str(r["t0"].year) for i, r in zip(ini, items)]
    lv_class = encode_levels([["all"] * N, cen, pt, hz, ini, yr])
    log("recall_class: fitting leave-event-out hierarchical rates (global > centre > product type > hazard > initiator > year)")
    kap, ll = fit_kappas(lv_class, y, 3, fit, groups)
    p_class = group_out_rates(lv_class, y, 3, fit, groups, kap)
    prior_cls = np.bincount(y[fit], minlength=3) / fit.sum()
    rep = {"recall_class_label_model": {
        "levels": ["global", "centre", "centre x product type", "... x hazard group", "... x initiator",
                   "... x initiation year"], "kappas": kap, "leave_event_out_log_score_train": round(ll, 5),
        "class_rates_train": dict(zip(CLASS_KEYS, prior_cls.round(4).tolist())),
        "leave_event_out_on_train": calib_report(p_class[fit], y[fit]),
        "global_rate_on_train": calib_report(np.tile(prior_cls, (int(fit.sum()), 1)), y[fit])}}
    for sp in ("val", "eval"):
        s = np.array([r["split"] == sp for r in items])
        if s.any():
            rep["recall_class_label_model"][f"train_only_rates_on_{sp}"] = calib_report(p_class[s], y[s])
            rep["recall_class_label_model"][f"global_rate_on_{sp}"] = calib_report(np.tile(prior_cls, (int(s.sum()), 1)), y[s])
    log(f"  kappas {kap}; {rep['recall_class_label_model']['leave_event_out_on_train']}")
    # classified_quickly: eligibility windows, outcome, rates
    lag = np.array([(r["t1"] - r["t0"]).days for r in items])
    yq = np.where(lag <= QUICK_DAYS, 0, 1)                     # 0 = yes (within 60 days), 1 = no
    elig = np.zeros(N, dtype=bool)
    for i, r in enumerate(items):
        lo, hi = quick_window(r["split"], export)
        elig[i] = lag[i] >= 0 and (lo is None or r["t0"] >= lo) and r["t0"] <= hi
    lv_q = encode_levels([["all"] * N, cen, [c + "|" + str(r["t0"].year) for c, r in zip(cen, items)],
                          [c + "|" + str(r["t0"].year) + "|" + r["hazard"] for c, r in zip(cen, items)]])
    yq_fit = np.where(elig, yq, -1)
    kq, llq = fit_kappas(lv_q, yq_fit, 2, fit & elig, groups)
    p_quick = group_out_rates(lv_q, yq_fit, 2, fit & elig, groups, kq)
    fq = fit & elig
    rep["classified_quickly_label_model"] = {
        "levels": ["global", "centre", "centre x initiation year", "... x hazard group"], "kappas": kq,
        "leave_event_out_log_score_train": round(llq, 5), "eligible_by_split": {
            sp: int((elig & np.array([r["split"] == sp for r in items])).sum()) for sp in ("train", "val", "eval")},
        "rate_within_60d_train": round(float((yq[fq] == 0).mean()), 4),
        "leave_event_out_on_train": calib_report(p_quick[fq], yq[fq])}
    ev = elig & np.array([r["split"] == "eval" for r in items])
    if ev.any():
        rep["classified_quickly_label_model"]["train_only_rates_on_eval"] = calib_report(p_quick[ev], yq[ev])
        rep["classified_quickly_label_model"]["rate_within_60d_eval"] = round(float((yq[ev] == 0).mean()), 4)
        # residual censoring: share of eval-window initiations still unclassified at export, from the train lag law
        tl = defaultdict(list)
        for i in np.where(fit)[0]:
            tl[cen[i]].append(lag[i])
        for c in tl:
            tl[c] = np.sort(np.array(tl[c]))
        miss = [1.0 - np.searchsorted(tl[cen[i]], (export - items[i]["t0"]).days, side="right") / len(tl[cen[i]])
                for i in np.where(ev)[0]]
        rep["classified_quickly_label_model"]["eval_expected_missing_unclassified_share"] = round(float(np.mean(miss)), 4)
    # per-event cap, then caps per split (proportional by centre x class with a floor)
    rng = random.Random(a.seed)
    by_ev = defaultdict(list)
    for i, r in enumerate(items):
        by_ev[r["event_id"]].append(i)
    keep_idx = []
    for e, v in sorted(by_ev.items()):
        v = sorted(v, key=lambda i: hkey("pick", a.seed, items[i]["recall_number"]))
        if a.per_event and len(v) > a.per_event:
            stats["drop_per_event_cap"] += len(v) - a.per_event
            v = v[:a.per_event]
        keep_idx.extend(v)
    by_split = defaultdict(list)
    for i in keep_idx:
        by_split[items[i]["split"]].append(i)
    caps = {"train": a.max_train, "val": a.max_val, "eval": a.max_eval}
    chosen, weights = {}, {}
    for sp, v in by_split.items():
        v.sort()
        take, w = stratified_take(v, caps[sp], a.cell_floor if sp == "train" else 0,
                                  lambda i: (items[i]["centre"], items[i]["classification"]), rng)
        chosen[sp], weights[sp] = sorted(take), w
        stats[f"cap_{sp}_dropped"] += len(v) - len(take)
    # records
    files = {}
    out_counts, train_states = Counter(), set()
    for sp in ("train", "val", "eval"):
        for i in chosen.get(sp, []):
            r = items[i]
            centre = r["centre"]
            state = {
                "product": {"centre": centre, "product_type": r["ptype"],
                            "description": clip(r.get("product_description"), a.max_desc),
                            "quantity": clip(r.get("product_quantity") or "not stated", 160)},
                "recall": {"reason_for_recall": clip(r.get("reason_for_recall"), a.max_reason),
                           "distribution_pattern": clip(r.get("distribution_pattern") or "not stated", a.max_dist),
                           "initiated_by": r.get("voluntary_mandated") or "not stated",
                           "consignee_notification": r.get("initial_firm_notification") or "not stated",
                           "firm_location": firm_location(r)},
                "firm_history": r["hist"],
                "timing": {"initiated": r["t0"].isoformat()},
            }
            s100 = int(hkey("surface", r["recall_number"])[:4], 16) % 100
            surface = "json" if s100 < 60 else ("memo" if s100 < 85 else "form")
            text = render(state, surface)
            if CLASS_WORDS.search(text):
                stats["drop_class_wording_rendered"] += 1
                continue
            if sp == "train":
                train_states.add(text)
            elif text in train_states:
                stats[f"{sp}_dropped_state_identical_to_train"] += 1
                continue
            yk = CLASS_OF[r["classification"]]
            qs = {"recall_class": choice_q(CLASS_INSTR, dict(CLASS_CRITERIA))}
            t_cls = blend(y[i], p_class[i], 3, a.lam)
            labs = {"recall_class": onehot_dict(CLASS_KEYS, yk) if sp == "eval" else lab_dict(CLASS_KEYS, t_cls)}
            src = {"recall_class": "expert"}
            meth = {"recall_class": ("onehot(FDA classification)" if sp == "eval" else
                                     f"{a.lam}*onehot(FDA classification) + {1 - a.lam}*leave-event-out hierarchical base "
                                     "rate (global > centre > product type > hazard > initiator > initiation year; "
                                     "train-only fit)")}
            raw = {"recall_class": yk, "days_to_classification": int(lag[i])}
            lvl, rinfo = reach_level(state["recall"]["distribution_pattern"])
            if lvl is not None:
                qs["distribution_reach"] = score_q(REACH_INSTR, REACH_LEVELS)
                labs["distribution_reach"] = onehot_dict(REACH_LEVELS, REACH_LEVELS[lvl])
                src["distribution_reach"] = "rule_exact"
                meth["distribution_reach"] = "reach_level() parser over the distribution text shown in the state"
                raw["distribution_reach"] = REACH_LEVELS[lvl].split(":")[0]
            else:
                stats["reach_not_asked_no_place"] += 1
            if elig[i]:
                qk = "yes" if yq[i] == 0 else "no"
                qs["classified_quickly"] = {**noul_q(QUICK_INSTR), "criteria": dict(QUICK_CRITERIA)}
                pq = p_quick[i]
                labs["classified_quickly"] = (onehot_dict(["yes", "no"], qk) if sp == "eval" else
                                              lab_dict(["yes", "no"], blend(yq[i], pq, 2, a.lam)))
                src["classified_quickly"] = "outcome"
                meth["classified_quickly"] = ("onehot(days to classification <= 60)" if sp == "eval" else
                                              f"{a.lam}*onehot(days to classification <= 60) + {1 - a.lam}*leave-event-out "
                                              "base rate (global > centre > centre x year > ... x hazard; train-only)")
                raw["classified_quickly"] = qk
            rc = r.get("root_cause_description")
            if centre == "device":
                raw["root_cause_description"] = rc
                if rc not in ROOT_NOT_ASKED and rc in ROOT_GROUP:
                    qs["root_cause_group"] = choice_q(ROOT_INSTR, dict(ROOT_CRITERIA))
                    labs["root_cause_group"] = onehot_dict(ROOT_KEYS, ROOT_GROUP[rc])
                    src["root_cause_group"] = "expert"
                    meth["root_cause_group"] = "onehot(FDA root_cause_description mapped by ROOT_GROUP)"
                    raw["root_cause_group"] = ROOT_GROUP[rc]
                else:
                    stats["root_cause_not_asked"] += 1
            kinds = {q: ("hard" if sum(1 for v in labs[q].values() if v > 0) == 1 else "soft") for q in qs}
            source = f"fda_recalls/{centre}"
            rec = record(source, text, qs, labs, kinds, None)
            sig = json.dumps({k: option_keys(v) for k, v in qs.items()}, sort_keys=True)
            rec["id"] = f"{source}-{sha(r['recall_number'] + text + sig)}"
            rec["base_id"] = rec["id"]
            rec["license"] = LICENSE
            stratum = f"{centre}|{r['classification']}"
            rec["meta"] = {
                "dataset": f"openFDA {centre} enforcement (export {prov['endpoints'][centre]['export_date']})"
                           + (" + device/recall root causes" if centre == "device" else ""),
                "group": f"fda_event:{r['event_id']}", "recall_number": r["recall_number"], "event_id": r["event_id"],
                "as_of": r["t0"].isoformat(), "decision_date": r["t1"].isoformat(), "split_reason": r["split_reason"],
                "surface": surface, "label_source": src, "label_method": meth, "raw_outcome": raw,
                "lambda": a.lam if sp != "eval" else None,
                "base_rate": {"recall_class": lab_dict(CLASS_KEYS, p_class[i]),
                              **({"classified_quickly": lab_dict(["yes", "no"], p_quick[i])} if elig[i] else {})},
                "smoothed_target": ({"recall_class": lab_dict(CLASS_KEYS, t_cls),
                                     **({"classified_quickly": lab_dict(["yes", "no"], blend(yq[i], p_quick[i], 2, a.lam))}
                                        if elig[i] else {})} if sp == "eval" else None),
                "base_rate_cell": yr[i], "hazard_group": r["hazard"], "product_type": r["ptype"],
                "reach_parse": rinfo, "stratum": stratum, "sample_weight": weights[sp].get((centre, r["classification"]), 1.0),
                "teacher_fill": "pending: hazard_kind in teacher_fill/" if sp != "eval" else None,
            }
            validate(rec)
            key = (sp, source)
            if key not in files:
                pth = outdir / sp / (source.replace("/", "__") + ".jsonl")
                pth.parent.mkdir(parents=True, exist_ok=True)
                files[key] = pth.open("w", encoding="utf-8")
            files[key].write(json.dumps(rec, ensure_ascii=False) + "\n")
            if sp != "eval":     # real-world eval labels never come from an LLM (realworld_data.md §5)
                write_fill(outdir, sp, source.replace("/", "__") + ".jsonl",
                           fill_record(rec, sp, {"hazard_kind": choice_q(HAZARD_INSTR, dict(HAZARD_CRITERIA))}))
            out_counts[(sp, centre)] += 1
    for fh in files.values():
        fh.close()
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "seed": a.seed, "lambda": a.lam,
                "export_date": export.isoformat(), "fetch": prov,
                "splits": {"train": "classified <= 2024-12-31", "val": "classified 2025-01-01 .. 2025-06-30",
                           "eval": "classified >= 2025-07-01 (one-hot labels for expert / outcome questions)"},
                "caps": {"per_event": a.per_event, "train": a.max_train, "val": a.max_val, "eval": a.max_eval,
                         "train_cell_floor": a.cell_floor},
                "sample_weights": {sp: {f"{k[0]}|{k[1]}": v for k, v in w.items()} for sp, w in weights.items()},
                "records": {f"{sp}/{c}": n for (sp, c), n in sorted(out_counts.items())},
                "records_by_split": {sp: sum(n for (s, _), n in out_counts.items() if s == sp) for sp in ("train", "val", "eval")},
                "stats": dict(stats), **rep}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"wrote {manifest['records_by_split']} records to {outdir}")


# --------------------------------------------------------------------------- check
def check(a):
    import numpy as np  # noqa: F401
    outdir = Path(a.out)
    recs = read_corpus(outdir, "fda_recalls__")
    report = {"readback": readback(outdir, recs)}
    report["dedup"] = dedup_report(recs)
    hits = Counter()
    for v in recs.values():
        for r in v:
            for m in CLASS_WORDS.finditer(r["state"]):
                hits[m.group(0).lower()] += 1
    report["class_wording_in_written_states"] = dict(hits) or "0 hits"
    report["label_stats"] = label_stats(recs)
    rc = defaultdict(Counter)
    for sp, v in recs.items():
        for r in v:
            for q, k in r["meta"]["raw_outcome"].items():
                if q in ("recall_class", "classified_quickly", "root_cause_group", "distribution_reach"):
                    rc[f"{sp}/{q}"][k] += 1
            rc[f"{sp}/surface"][r["meta"]["surface"]] += 1
    report["realised_labels"] = {k: dict(c) for k, c in sorted(rc.items())}
    lens = sorted(len(r["state"]) for v in recs.values() for r in v)
    report["state_chars"] = {"min": lens[0], "p10": lens[len(lens) // 10], "median": lens[len(lens) // 2],
                             "p90": lens[9 * len(lens) // 10], "max": lens[-1]}
    report["exclusion_errors"] = run_exclusion("fda_recall_classification", {
        "food_recall": ["recall_class", "distribution_reach", "classified_quickly", "hazard_kind"],
        "drug_recall": ["recall_class", "distribution_reach", "classified_quickly", "hazard_kind"],
        "device_recall": ["recall_class", "distribution_reach", "classified_quickly", "root_cause_group", "hazard_kind"]})
    rng = random.Random(0)
    train = recs.get("train", [])
    if train:
        sample = rng.sample(train, min(a.probe_n, len(train)))
        report["leak_probe_recall_class"] = leak_probe(sample, "recall_class", CLASS_KEYS)
    # 200-row sample for the manual distribution_reach audit (and agreement if the manual labels exist)
    pool = [r for v in recs.values() for r in v if "distribution_reach" in r["questions"]]
    done = {line.split("\t", 1)[0] for m in outdir.glob("reach_audit_manual_s*.tsv")
            if not m.name.endswith(f"_s{a.audit_seed}.tsv") for line in m.read_text(encoding="utf-8").splitlines()[1:]}
    fresh = [r for r in pool if r["id"] not in done]        # a new audit never reuses rows an earlier audit labelled
    aud = random.Random(a.audit_seed).sample(fresh, min(200, len(fresh)))
    with (outdir / f"reach_audit_sample_s{a.audit_seed}.tsv").open("w", encoding="utf-8") as f:
        f.write("record_id\tparser_level\tdistribution_text\n")
        for r in aud:
            txt = json.loads(r["state"])["recall"]["distribution_pattern"] if r["meta"]["surface"] == "json" else None
            if txt is None:
                m = re.search(r"(?:distribution pattern|Distribution): (.*)", r["state"])
                txt = m.group(1) if m else ""
            f.write(f"{r['id']}\t{r['meta']['raw_outcome']['distribution_reach']}\t{txt}\n")
    report["reach_manual_audit"] = {}
    for man in sorted(outdir.glob("reach_audit_manual_s*.tsv")):
        lab = {}
        for line in man.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                lab[parts[0]] = parts[1].strip()
        by_id = {r["id"]: r for r in pool}
        got = [(i, by_id[i]["meta"]["raw_outcome"]["distribution_reach"]) for i in lab if i in by_id]
        agree = sum(1 for i, p in got if lab[i] == p)
        report["reach_manual_audit"][man.name] = {
            "n": len(got), "agree": agree, "agreement": round(agree / max(1, len(got)), 4),
            "disagreements": [(i, p, lab[i]) for i, p in got if lab[i] != p]}
    fills = sum(1 for p in (outdir / "teacher_fill").rglob("*.jsonl") for _ in p.open())
    report["teacher_fill_records_pending"] = fills
    report["bytes_on_disk"] = sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file() and "cache" not in p.parts)
    (outdir / "check_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    samples, want = [], [("train", "fda_recalls/food"), ("train", "fda_recalls/drug"), ("train", "fda_recalls/device"),
                         ("val", "fda_recalls/device"), ("eval", "fda_recalls/food")]
    for sp, src in want:
        pool = [r for r in recs.get(sp, []) if r["source"] == src and len(r["questions"]) >= 3]
        if pool:
            samples.append(random.Random(len(samples)).choice(pool))
    (outdir / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- self-test (no network)
def self_test():
    import tempfile

    import numpy as np
    # streaming reader on a small zip
    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "x.json.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("x.json", json.dumps({"meta": {"a": 1}, "results": [{"i": i, "s": "x" * (i * 7)} for i in range(50)]}))
        got = list(iter_results(zp, chunk=64))
        assert [g["i"] for g in got] == list(range(50)), got[:3]
    # leave-group-out: an item's own group never enters its rate
    lv = encode_levels([["all"] * 6, ["a", "a", "a", "b", "b", "b"]])
    g = np.array([0, 0, 1, 2, 2, 3])
    fit = np.ones(6, dtype=bool)
    p1 = group_out_rates(lv, np.array([0, 0, 1, 1, 0, 1]), 2, fit, g, [1.0, 1.0])
    p2 = group_out_rates(lv, np.array([1, 1, 1, 1, 0, 1]), 2, fit, g, [1.0, 1.0])
    assert abs(p1[0] - p2[0]).max() < 1e-12 and abs(p1[1] - p2[1]).max() < 1e-12, "own group moved its rate"
    assert abs(p1[2] - p2[2]).max() > 1e-6, "other groups must count"
    # uncounted (val / eval) items never move anyone's rate
    m = np.array([True, True, True, True, True, False])
    p3 = group_out_rates(lv, np.array([0, 0, 1, 1, 0, 1]), 2, m, g, [1.0, 1.0])
    p4 = group_out_rates(lv, np.array([0, 0, 1, 1, 0, 0]), 2, m, g, [1.0, 1.0])
    assert abs(p3 - p4).max() < 1e-12
    t = blend(1, [0.7, 0.2, 0.1], 3, 0.5)
    assert abs(t.sum() - 1) < 1e-12 and abs(t[1] - 0.6) < 1e-12
    assert abs(sum(lab_dict(["x", "y", "z"], [0.2, 0.3, 0.5]).values()) - 1) < 1e-9
    # class wording
    for s in ("Class II Recall - US distribution", "Class 1 lasers", "serious adverse health consequences", "CLASS III"):
        assert CLASS_WORDS.search(s), s
    for s in ("Labeling: Not Elsewhere Classified", "classic flavor", "first-class mail", "classification pending"):
        assert not CLASS_WORDS.search(s), s
    # distribution reach rule
    cases = [("NY", 0), ("Distributed throughout FL.", 0), ("P.R. consignees only.", 0), ("Within the U.S in FL and CA.", 1),
             ("Product was distributed to 6 major distributors", None), ("Nationwide", 3), ("US Nationwide distribution.", 3),
             ("CO, NJ, England, South Africa,", 3), ("Products distributed to Florida, Nevada and the Dominican Republic", 3),
             ("Nationwide in US: Arizona, California, Georgia", 3), ("AL, AZ, CA, CO, FL, GA, IL", 2),
             ("Distributed in NJ, NY, PA and CT", 1), ("Distributed in New Mexico and Texas", 1),
             ("Distributors in 44 states", 2), ("Distributed to VA hospitals in Ohio", 0), ("New York and West Virginia", 1),
             ("DISTRIBUTED IN THE STATES OF OHIO AND INDIANA", 1), ("Washington, DC and Maryland", 1),
             ("Panama City, FL", 0), ("Canada", 3), ("worldwide", 3), ("New England states: MA, CT, RI", 1),
             ("Holland, MI", 0), ("Peru, IN and Fort Wayne, IN", 0), ("Turkey and Jordan", 3),
             ("Mexican grocery stores in TX and CA", 1), ("U.S. distribution to the following: CA No foreign distribution.", 0),
             ("DOMESTIC: MI, PA, OH, IN, IL FOREIGN: NONE", 1), ("OH. There are no Canada accounts for this recall.", 0),
             ("National: AZ, CA, CT, FL, IN, ME, MI International: None reported", 2), ("Domestic: NY; foreign: Japan", 3),
             ("Nationwide. No foreign distribution.", 3)]
    for txt, want in cases:
        got, info = reach_level(txt)
        assert got == want, (txt, got, want, info)
    # derived fields contain no class wording; question vocab passes the scan
    for v in list(CLASS_CRITERIA.values()):
        assert CLASS_WORDS.search(v)       # the criteria quote the classes on purpose; they are not state text
    for v in REACH_LEVELS + list(ROOT_CRITERIA.values()) + list(HAZARD_CRITERIA.values()) + [REACH_INSTR, QUICK_INSTR]:
        assert not CLASS_WORDS.search(v), v
    assert firm_key("Acme Foods, Inc.") == firm_key("ACME FOODS INC") == "acme foods"
    assert hazard_group("food", "Product may be contaminated with Listeria monocytogenes") == "listeria"
    assert hazard_group("food", "undeclared milk") == "undeclared_allergen"
    assert set(ROOT_GROUP.values()) == set(ROOT_KEYS)
    st = {"product": {"centre": "food", "product_type": "human food", "description": "x", "quantity": "1"},
          "recall": {"reason_for_recall": "r", "distribution_pattern": "NY", "initiated_by": "v", "consignee_notification": "c",
                     "firm_location": "United States (NY)"},
          "firm_history": {"prior_recalls_24m": 0, "prior_recalls_same_hazard_24m": 0}, "timing": {"initiated": "2020-01-01"}}
    for s in ("json", "memo", "form"):
        assert "NY" in render(st, s)
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["fetch", "build", "check", "merge-fill", "all"])
    ap.add_argument("--cache", default="/tmp/fda_jev/cache")
    ap.add_argument("--raw-dir", default=None, help="where the bulk zips are downloaded (default <cache>/raw)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the bulk zips after extraction")
    ap.add_argument("--out", default="/tmp/fda_jev")
    ap.add_argument("--lam", type=float, default=LAMBDA, help="weight on the realised label in blended targets")
    ap.add_argument("--per-event", type=int, default=3, help="max product rows written per recall event (0 = all)")
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--cell-floor", type=int, default=200, help="train rows kept per centre x class stratum, at least")
    ap.add_argument("--max-val", type=int, default=1500)
    ap.add_argument("--max-eval", type=int, default=0, help="0 = every eval row (after the per-event cap)")
    ap.add_argument("--max-desc", type=int, default=450)
    ap.add_argument("--max-reason", type=int, default=600)
    ap.add_argument("--max-dist", type=int, default=600)
    ap.add_argument("--probe-n", type=int, default=8000)
    ap.add_argument("--audit-seed", type=int, default=7, help="seed of the 200-row distribution_reach audit sample")
    ap.add_argument("--fill-labelled", default=None, help="merge-fill: labeller output directory")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.stage in ("fetch", "all"):
        fetch(a)
    if a.stage in ("build", "all"):
        build(a)
    if a.stage in ("check", "all"):
        check(a)
    if a.stage == "merge-fill":
        merge_fill(a.out, a.fill_labelled, "fda_recalls__")


if __name__ == "__main__":
    main()
