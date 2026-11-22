#!/usr/bin/env python3
"""
build_jev_rd_alpha.py -- OUTCOME-LABEL SMOOTHING VARIANTS of a real-decision corpus (jev_rd2 / jev_rd3 layout), 2026-10-04.

The real-data builders label many questions as  t = 0.5 * onehot(realised value) + 0.5 * LOO base rate  (rd1_pipeline.md
section 4: "kept as built"). Under cross-entropy the optimum on such a target is q* = 0.5 p(y|x) + 0.5 b(y): a fixed shrinkage
toward the base rate. This script rewrites ONLY those labels with another weight alpha on the base rate:

    t_alpha = (1 - alpha) * e + alpha * b,        e = the realised value (one-hot; nvd_cwe: the builder's gold split),
                                                  b = the builder's leave-one-out base rate.

b is not re-estimated. Every affected published label satisfies t_0.5 = 0.5 e + 0.5 b, so b = 2 t_0.5 - e and

    t_alpha = 2 alpha * t_0.5 + (1 - 2 alpha) * e          (computed per option, then rounded to 9 decimals)

-- a function of the published label and the realised value only. alpha = 0.5 returns the published value itself
(2 * 0.5 = 1.0 and 1 - 2 * 0.5 = 0.0 exactly in floating point; round(x, 9) is the identity on the builders' 6-decimal
values), so the alpha = 0.5 variant must reproduce the input byte for byte: `verify` proves it.

WHICH LABELS (AFFECTED): a question is rewritten iff its builder's label_method is the FIXED-LAMBDA blend of a realised value
and a leave-one-out base rate, lambda = 0.5:
  '0.5*onehot(...) + 0.5*...'  (outcomes2: afd, aidev, swe_rebench, swiss; ctgov; dmhc outcome; fda; safety)
  '0.5*gold (...) + 0.5*...'   (nvd_cwe: gold = the builder's equal split over mapped CWE options, meta.raw_outcome.mapped)
  'O:lam*onehot+(1-lam)*...' / 'V3:lam*onehot(split)+(1-lam)*...' with meta.lambda = 0.5 (courts dispositions, court_divided,
                               panel_dissent: the realised "at least one dissenter", smoothed like an outcome)
  'lam*onehot(...) + (1-lam)*p_hat, lam=0.5'  (physics alert_retracted, in_gw_catalog)
  'derived:sum_disposition_mass_by_change_level' (courts degree_of_change: the disposition label summed by change level, so
                               itself 0.5 onehot(realised level) + 0.5 (summed base rate))
NOT rewritten: rater / vote posteriors (OpenReview shares, ClinVar / CIViC Dirichlet posteriors, DMHC Beta-binomial reviewer
posteriors and panel splits, state-court justice votes), published model / forecaster probabilities (LVK p_astro / em_bright,
USGS PAGER, SPC), rule / exact / expert one-hot labels, and outcome labels that are already one-hot (folktables, wikidata).

CHECKS on every affected question (published label t, independent of alpha; any failure aborts):
  lambda = 0.5 (meta.lambda where present, else the method string); e is a distribution over the label's keys; t[e] >= 0.5 for
  a one-hot e; b = 2 t - e has min >= -2e-6 and sums to 1 within 1e-5; where the record stores meta.base_rate[q], |b - stored|
  <= 2e-5 per option (the builders round both to 6 decimals).
label_kind of a rewritten question follows the builders' rule (hard iff exactly one option > 0), so alpha = 0 one-hot labels are
'hard'. Nothing else in a record changes (meta.label_method / lambda / smoothed_target still describe the published label);
manifest.json and gate_report.json are copied unchanged; label_variant.json (new) records alpha, the rule and the counts.

    python3 build_jev_rd_alpha.py transform --src jev_rd2 --out jev_rd2_a000 --alpha 0
    python3 build_jev_rd_alpha.py verify --base jev_rd2 --variant jev_rd2_a050 [--expect-identical]
    python3 build_jev_rd_alpha.py decompose --src jev_rd2 --split val --out val_eb.jsonl   (e, b per affected unit, for selection)
    python3 build_jev_rd_alpha.py teacher --src jev_rd2 --oof oof_train.jsonl --out jev_rd2_xfit_bow   (cross-fitted teacher labels)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

VERSION = "rd_alpha 1.0 (2026-10-04)"
CHANGE_OF = {"affirm": 0, "dismiss": 0, "affirm_in_part": 1, "remand_only": 1, "reverse_or_vacate": 2,
             "reverse": 2, "reverse_and_remand": 2}          # build_jev_courts.CHANGE_OF (degree_of_change levels)
ROUND = 9
BR_TOL = 2e-5


def _get(d, q):
    return d.get(q) if isinstance(d, dict) else d


def method_of(r, q):
    m = r.get("meta") or {}
    x = _get(m.get("label_method"), q)
    return "" if x is None else str(x)


def lam_of(r, q, meth):
    """The builder's lambda for question q: meta.lambda when present (a number or a per-question dict), else parsed from the
    method string ('0.5*onehot', '0.5*gold', 'lam=0.5')."""
    m = r.get("meta") or {}
    if "lambda" in m and m["lambda"] is not None:
        v = _get(m["lambda"], q)
        if isinstance(v, (int, float)):
            return float(v)
    mm = re.match(r"^([0-9.]+)\*(onehot|gold)", meth) or re.search(r"lam=([0-9.]+)", meth)
    return float(mm.group(1)) if mm else None


def kind_of_method(meth):
    """'blend' (a fixed-lambda realised-value + LOO base-rate blend), 'derived' (courts degree_of_change) or None."""
    if meth == "derived:sum_disposition_mass_by_change_level":
        return "derived"
    if re.match(r"^0\.5\*(onehot|gold)\b", meth) or meth.startswith("O:lam*onehot") or meth.startswith("V3:lam*onehot") \
            or (meth.startswith("lam*onehot(") and "lam=0.5" in meth):
        return "blend"
    return None


def realised(r, q, keys):
    """The realised value of an affected question as a distribution over the label keys {key: mass} (one-hot except nvd_cwe's
    gold split), from the builder's own meta fields."""
    m = r.get("meta") or {}
    src = str(r.get("source") or "")
    if q == "cwe_class":                                        # nvd_cwe: the gold split over mapped options
        raw = m.get("raw_outcome") or {}
        mp = raw.get("mapped") if isinstance(raw, dict) else None
        if not isinstance(mp, dict) or not mp:
            raise ValueError(f"{r['id']} {q}: no raw_outcome.mapped")
        s = sum(float(v) for v in mp.values())
        return {k: float(v) / s for k, v in mp.items()}
    if q == "degree_of_change":                                 # courts: raw_outcome is the record's disposition dict
        v = m.get("raw_outcome")
    elif "raw_outcome" in m and isinstance(m["raw_outcome"], dict) and q in m["raw_outcome"]:
        v = m["raw_outcome"][q]
    elif "realised" in m and isinstance(m["realised"], dict) and q in m["realised"]:
        v = m["realised"][q]
    elif "outcome" in m and isinstance(m["outcome"], dict) and q in m["outcome"]:
        v = m["outcome"][q]
    else:
        raise ValueError(f"{r['id']} ({src}) {q}: no realised value in meta (raw_outcome / realised / outcome)")
    if q == "degree_of_change":                                 # the realised disposition's change level
        if not isinstance(v, dict):
            raise ValueError(f"{r['id']} {q}: raw_outcome is not the disposition dict")
        disp = v.get("appellate_disposition") or v.get("state_supreme_disposition")
        return {keys[CHANGE_OF[disp]]: 1.0}
    if isinstance(v, bool):
        v = "yes" if v else "no"
    v = str(v)
    if v not in keys:
        raise ValueError(f"{r['id']} {q}: realised {v!r} not among the label keys {keys}")
    return {v: 1.0}


def classify(r, q):
    """None (not affected) or (e: {key: mass}, kind) for an affected question; raises on an affected question whose published
    label is not the documented blend."""
    meth = method_of(r, q)
    kind = kind_of_method(meth)
    if kind is None:
        return None
    lab = r["labels"][q]
    keys = list(lab)
    lam = lam_of(r, q, meth) if kind == "blend" else float((r.get("meta") or {}).get("lambda", 0.5))
    if lam is None or abs(lam - 0.5) > 1e-12:
        raise ValueError(f"{r['id']} {q}: lambda {lam} != 0.5 ({meth[:80]})")
    e = realised(r, q, keys)
    if any(k not in lab for k in e):
        raise ValueError(f"{r['id']} {q}: realised keys {sorted(e)} not in label keys")
    return e, kind


def decompose(r, q, e):
    """b = 2 t - e and the checks; returns (b, max |b - stored base rate| or None)."""
    lab = r["labels"][q]
    b = {k: 2.0 * float(lab[k]) - e.get(k, 0.0) for k in lab}
    if min(b.values()) < -2e-6 or abs(sum(b.values()) - 1.0) > 1e-5:
        raise ValueError(f"{r['id']} {q}: published label is not 0.5 e + 0.5 b (b min {min(b.values()):.3g}, sum "
                         f"{sum(b.values()):.8f})")
    if len(e) == 1 and float(lab[next(iter(e))]) < 0.5:
        raise ValueError(f"{r['id']} {q}: realised option has mass {lab[next(iter(e))]} < 0.5")
    br = _get((r.get("meta") or {}).get("base_rate"), q) if isinstance((r.get("meta") or {}).get("base_rate"), dict) else None
    err = None
    if isinstance(br, dict):
        err = max(abs(b[k] - float(br.get(k, 0.0))) for k in lab)
        if err > BR_TOL:
            raise ValueError(f"{r['id']} {q}: implied base rate differs from meta.base_rate by {err:.3g}")
    return b, err


def kind_of(lab):
    return "hard" if sum(1 for v in lab.values() if v > 0) == 1 else "soft"


def relabel(lab, e, alpha):
    c1, c0 = 2.0 * alpha, 1.0 - 2.0 * alpha
    return {k: round(c1 * float(v) + c0 * e.get(k, 0.0), ROUND) for k, v in lab.items()}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def files_of(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())


# --------------------------------------------------------------------------- transform
def transform(src, out, alpha):
    src, out = Path(src), Path(out)
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit("--alpha in [0, 1]")
    if out.exists():
        raise SystemExit(f"{out} exists")
    stats = collections.OrderedDict()           # (split, source file, qid, label_source, method kind) -> counters
    brerr = 0.0
    for split in ("train", "val"):
        (out / split).mkdir(parents=True)
        for p in sorted((src / split).glob("*.jsonl")):
            n_rec = n_changed = 0
            with open(p, encoding="utf-8") as f, open(out / split / p.name, "w", encoding="utf-8") as g:
                for line in f:
                    r = json.loads(line)
                    n_rec += 1
                    changed = False
                    for q in list(r["labels"]):
                        c = classify(r, q)
                        if c is None:
                            continue
                        e, kind = c
                        _, err = decompose(r, q, e)
                        if err is not None:
                            brerr = max(brerr, err)
                        old = r["labels"][q]
                        new = relabel(old, e, alpha)
                        nk = kind_of(new)
                        key = (split, p.stem, q, r["label_source"].get(q), kind)
                        st = stats.setdefault(key, collections.Counter())
                        st["units"] += 1
                        st["maxp_before"] += max(float(v) for v in old.values())
                        st["maxp_after"] += max(new.values())
                        st["kind_changed"] += int(nk != r["label_kind"][q])
                        if new != old or nk != r["label_kind"][q]:
                            r["labels"][q] = new
                            r["label_kind"][q] = nk
                            changed = True
                    n_changed += int(changed)
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{split}/{p.name}: {n_rec} records, {n_changed} with a rewritten label", flush=True)
    for f in ("manifest.json", "gate_report.json"):
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
    table = [{"split": k[0], "file": k[1], "qid": k[2], "label_source": k[3], "method": k[4], "units": v["units"],
              "mean_maxp_published": round(v["maxp_before"] / v["units"], 6), "mean_maxp_variant": round(v["maxp_after"] / v["units"], 6),
              "label_kind_changed": v["kind_changed"]} for k, v in stats.items()]
    rep = {"version": VERSION, "alpha": alpha, "rule": "t_alpha = (1 - alpha) * e + alpha * b = 2 alpha t_0.5 + (1 - 2 alpha) e, "
                                                         "rounded to 9 decimals; e realised value, b builder LOO base rate",
           "source_dir": str(src), "max_abs_err_implied_vs_stored_base_rate": brerr,
           "affected_units": {s: sum(x["units"] for x in table if x["split"] == s) for s in ("train", "val")},
           "affected": table,
           "files_md5": {f: md5(out / f) for f in files_of(out)}}
    (out / "label_variant.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("alpha", "affected_units", "max_abs_err_implied_vs_stored_base_rate")}), flush=True)


# --------------------------------------------------------------------------- verify
def verify(base, variant, expect_identical=False):
    """Every file of `base` is in `variant`; jsonl files have the same records in the same order and differ at most in
    labels[q] / label_kind[q] of AFFECTED questions; everything else equal. With --expect-identical: byte-identical files."""
    base, variant = Path(base), Path(variant)
    fb, fv = files_of(base), files_of(variant)
    extra = sorted(set(fv) - set(fb))
    missing = sorted(set(fb) - set(fv))
    if missing:
        raise SystemExit(f"FAIL: files missing from the variant: {missing}")
    rep = {"files": len(fb), "extra_in_variant": extra, "byte_identical": 0, "differ": [], "records": 0,
           "records_changed": 0, "labels_changed": collections.Counter(), "nonaffected_label_changes": 0,
           "other_field_changes": 0}
    for f in fb:
        a, b = base / f, variant / f
        if a.read_bytes() == b.read_bytes():
            rep["byte_identical"] += 1
        else:
            rep["differ"].append(f)
        if not f.endswith(".jsonl"):
            continue
        with open(a, encoding="utf-8") as fa, open(b, encoding="utf-8") as fb_:
            for la, lb in zip(fa, fb_):
                rep["records"] += 1
                if la == lb:
                    continue
                ra, rb = json.loads(la), json.loads(lb)
                rep["records_changed"] += 1
                for k in set(ra) | set(rb):
                    if k in ("labels", "label_kind"):
                        continue
                    if ra.get(k) != rb.get(k):
                        rep["other_field_changes"] += 1
                if set(ra["labels"]) != set(rb["labels"]):
                    rep["other_field_changes"] += 1
                for q in ra["labels"]:
                    if ra["labels"][q] != rb["labels"][q] or ra["label_kind"][q] != rb["label_kind"][q]:
                        if classify(ra, q) is None:
                            rep["nonaffected_label_changes"] += 1
                        else:
                            rep["labels_changed"][f"{f.split('/')[0]}/{Path(f).stem}/{q}"] += 1
                        if list(ra["labels"][q]) != list(rb["labels"][q]):
                            rep["other_field_changes"] += 1       # option keys / order must be kept
            if sum(1 for _ in open(a, encoding="utf-8")) != sum(1 for _ in open(b, encoding="utf-8")):
                raise SystemExit(f"FAIL: {f}: record counts differ")
    rep["labels_changed"] = dict(sorted(rep["labels_changed"].items()))
    ok = rep["nonaffected_label_changes"] == 0 and rep["other_field_changes"] == 0
    if expect_identical:
        ok = ok and rep["byte_identical"] == rep["files"]
    rep["ok"] = ok
    print(json.dumps(rep, indent=1))
    if not ok:
        raise SystemExit("FAIL")
    return rep


# --------------------------------------------------------------------------- decompose (for the alpha selection)
def decompose_cmd(src, split, out):
    """One line per affected unit of `split`: {file, id, qid, keys, t (published), e, b} (b = 2 t - e)."""
    n = 0
    with open(out, "w") as g:
        for p in sorted((Path(src) / split).glob("*.jsonl")):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    for q in r["labels"]:
                        c = classify(r, q)
                        if c is None:
                            continue
                        e, kind = c
                        b, _ = decompose(r, q, e)
                        keys = list(r["labels"][q])
                        g.write(json.dumps({"file": p.stem, "id": r["id"], "qid": q, "label_source": r["label_source"].get(q),
                                            "keys": keys, "t": [float(r["labels"][q][k]) for k in keys],
                                            "e": [e.get(k, 0.0) for k in keys], "b": [max(0.0, b[k]) for k in keys],
                                            "state": r["state"] if isinstance(r["state"], str) else json.dumps(r["state"],
                                                                                                             ensure_ascii=False)},
                                           ensure_ascii=False) + "\n")
                        n += 1
    print(f"{n} affected {split} units -> {out}")


# --------------------------------------------------------------------------- teacher (cross-fitted conditional-probability labels)
def teacher(src, oof, out, name):
    """Affected labels <- a cross-fitted teacher's out-of-fold probability p_oof (one line per affected TRAIN unit: {file, id, qid,
    p_oof} in the label's key order, as tmp/datavar/scripts/alpha_select.py writes it); val labels and every other label unchanged.
    The realised value stays in meta (raw_outcome / outcome / realised), so it remains available for evaluation."""
    src, out = Path(src), Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    P = {}
    with open(oof) as f:
        for line in f:
            d = json.loads(line)
            P[(d["file"], d["id"], d["qid"])] = d["p_oof"]
    used = 0
    stats = collections.Counter()
    for split in ("train", "val"):
        (out / split).mkdir(parents=True)
        for p in sorted((src / split).glob("*.jsonl")):
            with open(p, encoding="utf-8") as f, open(out / split / p.name, "w", encoding="utf-8") as g:
                for line in f:
                    r = json.loads(line)
                    if split == "train":
                        for q in list(r["labels"]):
                            c = classify(r, q)
                            if c is None:
                                continue
                            decompose(r, q, c[0])                    # the same checks as the alpha variants
                            v = P.get((p.stem, r["id"], q))
                            if v is None or len(v) != len(r["labels"][q]):
                                raise SystemExit(f"no teacher prediction for {p.stem} {r['id']} {q}")
                            s = sum(v)
                            new = {k: round(float(x) / s, ROUND) for k, x in zip(r["labels"][q], v)}
                            r["labels"][q] = new
                            r["label_kind"][q] = kind_of(new)
                            used += 1
                            stats[f"{p.stem}/{q}"] += 1
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
    if used != len(P):
        raise SystemExit(f"{len(P)} teacher predictions, {used} used")
    for f in ("manifest.json", "gate_report.json"):
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
    rep = {"version": VERSION, "variant": name, "rule": "affected TRAIN labels <- the cross-fitted teacher's out-of-fold probability "
                                                          "(renormalised, 9 decimals); val and all other labels as published",
           "teacher_predictions_md5": md5(oof), "affected_train_units": used, "by_question": dict(sorted(stats.items())),
           "files_md5": {f: md5(out / f) for f in files_of(out)}}
    (out / "label_variant.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("variant", "affected_train_units", "teacher_predictions_md5")}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    te = sub.add_parser("teacher")
    te.add_argument("--src", required=True)
    te.add_argument("--oof", required=True)
    te.add_argument("--out", required=True)
    te.add_argument("--name", default="xfit_teacher")
    t = sub.add_parser("transform")
    t.add_argument("--src", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--alpha", type=float, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--base", required=True)
    v.add_argument("--variant", required=True)
    v.add_argument("--expect-identical", action="store_true")
    d = sub.add_parser("decompose")
    d.add_argument("--src", required=True)
    d.add_argument("--split", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "transform":
        transform(a.src, a.out, a.alpha)
    elif a.cmd == "teacher":
        teacher(a.src, a.oof, a.out, a.name)
    elif a.cmd == "verify":
        verify(a.base, a.variant, a.expect_identical)
    else:
        decompose_cmd(a.src, a.split, a.out)


if __name__ == "__main__":
    sys.exit(main())
