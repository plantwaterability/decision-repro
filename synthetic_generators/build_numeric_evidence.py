#!/usr/bin/env python3
"""build_numeric_evidence.py -- the (101) PRIOR-OFFSET data (research/p101_likelihood_ratio_head.md): the (94) numeric corpus and the
Bayesian-consistency suite's items rendered for a readout that applies the prior by arithmetic,

    q_i = softmax_i( log pi_i + l_i(D) ),   pi_i = the stated prior's mass in bin i (meta.log_prior_bins, exact, folded end bins),

so the network's option-letter logits are trained to be the EVIDENCE l_i(D) (a log-likelihood up to a constant). Everything is
build_numeric_jev.py's own code (imported: renderer option `variant`, log_bin_masses); nothing about the items changes.

  numeric   ROOT/numeric_ev  EVIDENCE-ONLY prompts: the prior sentence replaced by "a prior ... will be applied separately"; the same
                             items, axes, labels (the exact posterior under the STATED prior), ids and qids as numeric_v1; meta adds
                             log_prior_bins, prompt_variant, offset_exact (False for normal_pred / ab_diff: there the target is not the
                             parameter the prior is on, so a prior-free l is an approximation -- meta.offset_note), and in val / test
                             ev_prompt_in_train (the same evidence text occurs in train, with another prior)
            ROOT/numeric_pt  the default prompts (prior IN the text, byte-identical states) + the same meta: the double-counting
                             control (prior in text AND offsets) and the "caller does both" check
            --compare-v1 DIR checks every record against an existing numeric_v1 build: identical once the (101) keys (and, for
                             numeric_ev, the state) are set aside
  suite     the consistency suite's records (default types prior_only, prior_swap; build_bayes_consistency.py's output, read only)
            with meta.log_prior_bins from meta.prior over meta.bin_edges (meta.bc.unit_map) and, --variant evidence_only, the prior
            line (line 2 of intro / prior / data) replaced -- one template per GROUP, so a prior_swap group's members have IDENTICAL
            prompts and differ only in their offsets. Same ids / qids / labels: eval_bayes_consistency.py reads the npz unchanged.
            A record is converted only if its prior line is unambiguous and its exact target is reproduced from meta (else skipped,
            counted in the manifest).
  floor     the BINNING FLOORS of a prior-free bin-level evidence vector (pre-registered bounds): (train) KL(post || softmax(log pi +
            l_flat)), l_flat = the likelihood averaged over each bin (flat measure; folded end bins: the flat-density mass / bin
            width); (swap) KL(post' || softmax(log pi' + l*)), l* = log post - log pi under the stated prior, re-used under another
            prior pi' (numeric in-family: strength x3 / x(1/3), mean +-1 prior sd; suite: the prior_swap members). Per family.

  .venv/bin/python build_numeric_evidence.py numeric --out tmp/p101/data --compare-v1 tmp/numeric_v1/numeric_v1
  .venv/bin/python build_numeric_evidence.py numeric --axes v2 --out tmp/numv2/numeric_p101_v2 --compare tmp/numv2/numeric_v2
                                                   # (v2 axes, research/numeric_v2_axes.md: the axis no longer carries the prior)
  .venv/bin/python build_numeric_evidence.py suite --data tmp/bc/bayes_consistency_v1 --out tmp/p101/data/bc_ev --variant evidence_only
  .venv/bin/python build_numeric_evidence.py floor --data tmp/p101/data/numeric_ev --suite tmp/p101/data/bc_ev
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _import_bn():
    """build_numeric_jev.py from beside this file (repo root, or a tarball's tools/), else one level up."""
    for d in (HERE, HERE.parent):
        if (d / "build_numeric_jev.py").exists():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import build_numeric_jev
            return build_numeric_jev
    raise ImportError("build_numeric_jev.py not found beside build_numeric_evidence.py")


bn = _import_bn()
VERSION = "build_numeric_evidence.py v1 (2026-10-04)"
P101_KEYS = ("prompt_variant", "log_prior_bins", "log_prior_bins_nominal", "offset_exact", "offset_note", "ev_prompt_in_train")
CONVENTIONS = {"folded": "log_prior_bins", "nominal": "log_prior_bins_nominal"}
SUITE_TYPES = ("prior_only", "prior_swap")
PRIOR_WORDS = re.compile(r"(Prior|prior|Before any data|belief)")
DIST_WORDS = re.compile(r"(Beta|Gamma|Normal|normal|uniform)")
STATED_WORDS = re.compile(r"(prior|posterior)", re.I)


# ============================================================================ helpers shared with apply_prior.py / the tests
def unit_map(meta):
    """(scale, offset) of display = natural x scale + offset: the suite's meta.bc.unit_map, else meta.axis.scale and 0."""
    um = (meta.get("bc") or {}).get("unit_map")
    if um:
        return float(um.get("scale", 1.0)), float(um.get("offset", 0.0))
    return float(meta["axis"]["scale"]), 0.0


def log_masses_on_axis(dist, meta, fold=True):
    """bn.log_bin_masses on a record's own axis (meta.bin_edges, unit map; folded ends, or fold=False the nominal intervals)."""
    scale, off = unit_map(meta)
    return bn.log_bin_masses(dist, np.asarray(meta["bin_edges"], float) - off, scale, fold)


def prior_from_spec(spec):
    """'beta:A,B' | 'gamma:SHAPE,RATE' | 'normal:MEAN,SD' (natural units) -> a build_numeric_jev distribution."""
    kind, _, args = str(spec).partition(":")
    v = [float(x) for x in args.split(",") if x.strip()]
    if kind == "beta" and len(v) == 2 and min(v) > 0:
        return bn.BetaD(*v)
    if kind == "gamma" and len(v) == 2 and min(v) > 0:
        return bn.GammaD(*v)
    if kind == "normal" and len(v) == 2 and v[1] > 0:
        return bn.NormalD(*v)
    raise ValueError(f"prior spec {spec!r}: beta:A,B | gamma:SHAPE,RATE | normal:MEAN,SD with positive scale parameters")


def data_stats(meta):
    """The sufficient statistics of a parameter-family record, in either schema (numeric corpus / suite):
    coin {n, h}; rate {tot, T}; normal_mean {n, xbar, sigma}. None for other families."""
    fam = (meta.get("bc") or {}).get("family") or (meta.get("stress") or {}).get("base_family") or meta["family"]   # (v2) stress
    d = meta.get("data") or {}
    if fam in ("coin", "coin_seq", "probe"):
        return "coin", {"n": int(d["n"]), "h": int(d["h"])}
    if fam == "rate":
        tot = d.get("total", d.get("tot"))
        T = d.get("exposure", d.get("T"))
        return "rate", {"tot": int(tot), "T": float(T)}
    if fam == "normal_mean":
        sig = d.get("sigma")
        if sig is None:
            sig = ((meta.get("prior") or {}).get("stated") or {}).get("sigma")
        return "normal_mean", {"n": int(d["n"]), "xbar": None if d.get("xbar") is None else float(d["xbar"]), "sigma": float(sig)}
    return None, None


def conjugate_posterior(meta, prior):
    """The exact posterior of a parameter-family record's data under ANOTHER prior of the same conjugate family (natural units);
    None when the record is not a parameter family or the prior is of the wrong kind."""
    fam, st = data_stats(meta)
    kind = type(prior).__name__                       # by name: build_numeric_jev may be loaded twice (tests, tools/)
    if fam == "coin" and kind == "BetaD":
        return bn.BetaD(prior.a + st["h"], prior.b + st["n"] - st["h"])
    if fam == "rate" and kind == "GammaD":
        return bn.GammaD(prior.k + st["tot"], prior.r + st["T"])
    if fam == "normal_mean" and kind == "NormalD":
        if not st["n"]:
            return bn.NormalD(prior.m, prior.s)
        prec = 1 / prior.s ** 2 + st["n"] / st["sigma"] ** 2
        return bn.NormalD((prior.m / prior.s ** 2 + st["n"] * st["xbar"] / st["sigma"] ** 2) / prec, math.sqrt(1 / prec))
    return None


def flat_likelihood(meta):
    """The likelihood of a parameter-family record as a normalised density in the parameter (the flat-prior posterior): coin
    Beta(h + 1, n - h + 1), rate Gamma(tot + 1, T), normal N(xbar, sigma / sqrt(n)); None with no data (L == 1) or another family."""
    fam, st = data_stats(meta)
    if fam == "coin":
        return bn.BetaD(st["h"] + 1, st["n"] - st["h"] + 1) if st["n"] else None
    if fam == "rate":
        return bn.GammaD(st["tot"] + 1, st["T"]) if st["T"] > 0 else None
    if fam == "normal_mean":
        return bn.NormalD(st["xbar"], st["sigma"] / math.sqrt(st["n"])) if st["n"] else None
    return None


def l_flat(meta):
    """A PRIOR-FREE bin-level evidence vector: the likelihood averaged over each interior bin (flat measure); a folded end bin gets
    the flat-density mass over its range divided by the axis's bin width. Zeros with no data. [K] (any constant cancels)."""
    K = len(meta["bin_edges"]) - 1
    L = flat_likelihood(meta)
    if L is None:
        return np.zeros(K)
    e = np.asarray(meta["bin_edges"], float)
    scale, _ = unit_map(meta)
    lm = log_masses_on_axis(L, meta)
    width = np.diff(e) / scale                            # natural-unit widths
    w = np.full(K, float(meta["axis"]["width"]) / scale)
    ax = meta["axis"]
    inner = np.ones(K, dtype=bool)
    inner[0], inner[-1] = not ax.get("fold_low", False), not ax.get("fold_high", False)
    return lm - np.log(np.where(inner, width, w))


def softmax_log(v):
    v = np.asarray(v, float)
    m = np.max(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0
    p = np.exp(v - m)
    return p / p.sum()


def kl(p, q, eps=1e-300):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], eps)))).sum())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def write_jsonl(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================================ numeric
def compare_with_v1(root, v1, variant):
    """Every record of `root` against the numeric_v1 build `v1`: the same files, lines, ids and bytes once the (101) meta keys (and,
    evidence-only, the state) are set aside. -> report."""
    rep = {"files": 0, "records": 0, "mismatch": [], "state_changed": 0}
    for split in ("train", "val", "test"):
        for p in sorted((Path(v1) / split).glob("*.jsonl")):
            q = Path(root) / split / p.name
            if not q.exists():
                rep["mismatch"].append(f"missing {split}/{p.name}")
                continue
            rep["files"] += 1
            with open(p, encoding="utf-8") as fa, open(q, encoding="utf-8") as fb:
                for i, (la, lb) in enumerate(zip(fa, fb, strict=True)):
                    a, b = json.loads(la), json.loads(lb)
                    if variant == "evidence_only":
                        rep["state_changed"] += a["state"] != b["state"]
                        b["state"] = a["state"]
                    for k in P101_KEYS:
                        b["meta"].pop(k, None)
                    if json.dumps(b, ensure_ascii=False) + "\n" != la:
                        rep["mismatch"].append(f"{split}/{p.name}:{i}")
                    rep["records"] += 1
    rep["n_mismatch"] = len(rep["mismatch"])
    rep["mismatch"] = rep["mismatch"][:20]
    return rep


def build_numeric(out, scale=1.0, ns=bn.NS, compare_v1="", log=print, axes="v1"):
    """axes 'v2': the same two renderings of numeric_v2's items (compare against a numeric_v2 build with --compare)."""
    out = Path(out)
    man = {"version": VERSION, "generator": bn.VERSION if axes == "v1" else bn.VERSION_V2, "scale": scale, "seed_namespace": ns,
           "corpora": {}}
    if axes != "v1":
        man["axes"] = axes
    for name, variant in (("numeric_ev", "evidence_only"), ("numeric_pt", "prior_text")):
        t0 = time.time()
        m = bn.build(out / name, scale, ns, log=lambda *a: None, variant=variant, log_prior=True, axes=axes)
        rep = bn.check(out / name, sample_recompute=600)
        ent = {"variant": variant, "build_secs": round(time.time() - t0, 1), "sizes": m["sizes"], "check_errors": rep["n_errors"],
               "check": {k: rep[k] for k in ("label_sum_maxdev", "recompute_maxdiff", "log_prior_records", "log_prior_recompute_maxdiff")
                         if k in rep}}
        if "ev_prompt_in_train" in m["stats"]:
            ent["ev_prompt_in_train"] = m["stats"]["ev_prompt_in_train"]
        if compare_v1:
            ent["compare_v1"] = compare_with_v1(out / name, compare_v1, variant)
        man["corpora"][name] = ent
        log(f"{name}: {json.dumps(ent)[:600]} ({time.time() - t0:.0f} s)")
        if rep["n_errors"]:
            raise SystemExit(f"{name}: check errors {rep['errors'][:5]}")
        if compare_v1 and ent["compare_v1"]["n_mismatch"]:
            raise SystemExit(f"{name}: differs from {compare_v1}: {ent['compare_v1']['mismatch'][:5]}")
    copy_tools(out)
    (out / "manifest_p101.json").write_text(json.dumps(man, indent=1))
    return man


def copy_tools(out):
    """The code that made / reads the data, beside it (provenance; the fleet job runs tools/eval_numeric.py, apply_prior.py)."""
    t = Path(out) / "tools"
    t.mkdir(parents=True, exist_ok=True)
    for src in (HERE / "build_numeric_jev.py", HERE / "build_numeric_evidence.py", HERE / "analysis" / "eval_numeric.py",
                HERE / "analysis" / "apply_prior.py"):
        if src.exists():
            shutil.copy(src, t / src.name)


# ============================================================================ suite
def convert_suite_record(r, variant):
    """One consistency-suite record -> (record with meta.log_prior_bins [+ evidence-only state], None) or (None, skip reason)."""
    m = r["meta"]
    (qid, q), = r["questions"].items()
    if q.get("type") != "score" or "bin_edges" not in m or (m.get("bc") or {}).get("q", {}).get("target") != "param":
        return None, "not a parameter-bins question"
    if m["prior"].get("dist") not in ("beta", "gamma", "normal"):
        return None, f"prior {m['prior'].get('dist')}"
    pd = bn.dist_from_params(m["prior"])
    ex = m.get("exact") or {}
    if ex.get("dist") not in ("beta", "gamma", "normal"):
        return None, "exact posterior not parametric"
    lab = np.array([r["labels"][qid][k] for k in q["levels"]], float)
    rec_lab = np.exp(log_masses_on_axis(bn.dist_from_params(ex), m))
    if np.abs(rec_lab - lab).max() > 2e-8:                # the axis transform reproduces the record's own labels
        return None, "labels not reproduced from meta"
    out = json.loads(json.dumps(r))
    om = out["meta"]
    om["prompt_variant"] = variant
    for fold, key in ((True, "log_prior_bins"), (False, "log_prior_bins_nominal")):
        lp = np.maximum(log_masses_on_axis(pd, m, fold), -1e4)
        om[key] = [round(float(x), bn.LOGP_DP) for x in lp]
    om["offset_exact"] = True
    if variant == "evidence_only":
        lines = r["state"].split("\n")
        if len(lines) < 3 or not PRIOR_WORDS.search(lines[1]) or not DIST_WORDS.search(lines[1]):
            return None, "prior line not found at line 2"
        if any(STATED_WORDS.search(x) for i, x in enumerate(lines) if i != 1):
            return None, "another line states a prior / posterior"
        fam = m["bc"]["family"]
        lines[1] = bn.evidence_line(fam, m["bc"]["group"])   # one template per group: a group's members share the prompt
        out["state"] = "\n".join(lines)
    return out, None


def convert_suite(data, out, types=SUITE_TYPES, variant="evidence_only", log=print):
    data, out = Path(data), Path(out)
    rep = {"version": VERSION, "source": str(data), "variant": variant, "types": {}, "skipped": {}}
    src_man = data / "manifest.json"
    if src_man.exists():
        try:
            sm = json.loads(src_man.read_text())
            rep["source_manifest"] = {k: sm.get(k) for k in ("version", "seed_namespace", "build_secs") if k in sm}
        except ValueError:
            pass
    for t in types:
        p = data / "test" / f"bc_{t}.jsonl"
        if not p.exists():
            raise SystemExit(f"suite type {t}: {p} not found")
        recs, skip = [], Counter()
        for r in read_jsonl(p):
            o, why = convert_suite_record(r, variant)
            if o is None:
                skip[why] += 1
            else:
                recs.append(o)
        write_jsonl(out / "test" / f"bc_{t}.jsonl", recs)
        groups = defaultdict(set)
        for o in recs:
            groups[o["meta"]["bc"]["group"]].add(o["state"] + "\n" + next(iter(o["questions"].values()))["instructions"])
        rep["types"][t] = {"records": len(recs), "groups": len(groups),
                           "groups_one_prompt": sum(len(v) == 1 for v in groups.values())}
        rep["skipped"][t] = dict(skip)
        log(f"suite {t}: {len(recs)} records ({dict(skip) or 'none skipped'}); groups {len(groups)}, with one shared prompt "
            f"{rep['types'][t]['groups_one_prompt']}")
    (out / "manifest_p101.json").write_text(json.dumps(rep, indent=1))
    return rep


# ============================================================================ floors
def swap_priors(prior):
    """Alternative priors of the same conjugate family: strength x3 and x1/3 (same mean), mean +-1 prior sd (same strength)."""
    out = {}
    kind = type(prior).__name__
    if kind == "BetaD":
        s, m = prior.a + prior.b, prior.mean
        sd = math.sqrt(prior.var)
        out["strength_x3"], out["strength_x1/3"] = bn.BetaD(3 * prior.a, 3 * prior.b), bn.BetaD(prior.a / 3, prior.b / 3)
        for sgn, nm in ((1, "mean_+1sd"), (-1, "mean_-1sd")):
            m2 = min(max(m + sgn * sd, 0.02), 0.98)
            out[nm] = bn.BetaD(m2 * s, (1 - m2) * s)
    elif kind == "GammaD":
        out["strength_x3"], out["strength_x1/3"] = bn.GammaD(3 * prior.k, 3 * prior.r), bn.GammaD(prior.k / 3, prior.r / 3)
        for sgn, nm in ((1, "mean_+1sd"), (-1, "mean_-1sd")):
            m2 = prior.mean + sgn * math.sqrt(prior.var)
            if m2 > 0:
                out[nm] = bn.GammaD(prior.k, prior.k / m2)
    elif kind == "NormalD":
        out["strength_x3"], out["strength_x1/3"] = bn.NormalD(prior.m, prior.s / math.sqrt(3)), bn.NormalD(prior.m, prior.s * math.sqrt(3))
        out["mean_+1sd"], out["mean_-1sd"] = bn.NormalD(prior.m + prior.s, prior.s), bn.NormalD(prior.m - prior.s, prior.s)
    return out


def _summ(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())} if len(v) else {"n": 0}


def floors(data, splits=("test",), suite="", max_per_family=0):
    """The binning floors (module docstring), for BOTH offset conventions (folded / nominal end bins). numeric, per family:
    (train) KL(post || softmax(lp + l_flat)) and the log-score skill it caps; (swap) KL(post' || softmax(lp' + l*)), l* = lq - lp
    under the stated prior, over swap_priors; (swap_flat) the same with l_flat. suite: KL(member || softmax(lp_member + l*_base))
    over the prior_swap groups, and for prior_only no-data members KL(prior || softmax(lp)) (the flat-evidence answer)."""
    res = {"numeric": {}, "suite": {}}
    acc = defaultdict(list)
    ent = defaultdict(list)
    for split in splits:
        for p in sorted((Path(data) / split).glob("*.jsonl")):
            n_f = 0
            for r in read_jsonl(p):
                m = r["meta"]
                fam = m["family"]
                if fam not in ("coin", "coin_seq", "probe", "rate", "normal_mean"):
                    continue
                if max_per_family and n_f >= max_per_family:
                    break
                n_f += 1
                prior, post = bn.dist_from_params(m["prior"]), bn.dist_from_params(m["exact"])
                lq = log_masses_on_axis(post, m)
                pe = np.exp(lq)
                ent[fam].append(float(-(pe[pe > 0] * lq[pe > 0]).sum()))
                lf = l_flat(m)
                nodata = flat_likelihood(m) is None
                for conv, fold in (("folded", True), ("nominal", False)):
                    lp = log_masses_on_axis(prior, m, fold)
                    v = kl(pe, softmax_log(lp + lf))
                    acc[(fam, conv, "train")].append(v)
                    acc[(fam, conv, "train_nodata" if nodata else "train_data")].append(v)
                    for nm, pr2 in swap_priors(prior).items():
                        post2 = conjugate_posterior(m, pr2)
                        p2 = np.exp(log_masses_on_axis(post2, m))
                        lp2 = log_masses_on_axis(pr2, m, fold)
                        acc[(fam, conv, "swap")].append(kl(p2, softmax_log(lp2 + lq - lp)))
                        acc[(fam, conv, "swap_flat")].append(kl(p2, softmax_log(lp2 + lf)))
    lk = math.log(bn.K)
    for fam in sorted(ent):
        H = float(np.mean(ent[fam]))
        ent_f = {"n": len(ent[fam]), "exact_entropy": H}
        for conv in CONVENTIONS:
            k = float(np.mean(acc[(fam, conv, "train")]))
            ent_f[conv] = {nm: _summ(acc[(fam, conv, nm)]) for nm in ("train", "train_data", "train_nodata", "swap", "swap_flat")}
            ent_f[conv]["skill_cap"] = (lk - H - k) / (lk - H) if lk > H else float("nan")
        res["numeric"][fam] = ent_f
    if suite:
        out = defaultdict(list)
        p = Path(suite) / "test" / "bc_prior_swap.jsonl"
        if p.exists():
            groups = defaultdict(dict)
            for r in read_jsonl(p):
                groups[r["meta"]["bc"]["group"]][r["meta"]["bc"]["member"]] = r
            for g, mem in groups.items():
                if "base" not in mem:
                    continue
                b = mem["base"]["meta"]
                lqb = log_masses_on_axis(bn.dist_from_params(b["exact"]), b)
                for conv, key in CONVENTIONS.items():
                    lstar = lqb - np.asarray(b[key], float)
                    for nm in ("strength", "mean"):
                        if nm in mem:
                            mm = mem[nm]["meta"]
                            pe = np.exp(log_masses_on_axis(bn.dist_from_params(mm["exact"]), mm))
                            out[(mm["bc"]["family"], conv, f"swap_{nm}")].append(
                                kl(pe, softmax_log(np.asarray(mm[key], float) + lstar)))
        p = Path(suite) / "test" / "bc_prior_only.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                mm = r["meta"]
                if mm["bc"]["member"] != "nodata":
                    continue
                pe = np.exp(log_masses_on_axis(bn.dist_from_params(mm["exact"]), mm))
                for conv, key in CONVENTIONS.items():
                    out[(mm["bc"]["family"], conv, "prior_only_flat")].append(kl(pe, softmax_log(np.asarray(mm[key], float))))
        res["suite"] = {f"{f}/{c}/{nm}": _summ(v) for (f, c, nm), v in sorted(out.items())}
    return res


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("numeric")
    a1.add_argument("--out", required=True)
    a1.add_argument("--scale", type=float, default=1.0)
    a1.add_argument("--ns", default=bn.NS)
    a1.add_argument("--compare-v1", "--compare", dest="compare_v1", default="",
                    help="the default-rendering corpus of the same axes (numeric_v1 / numeric_v2) to compare record by record")
    a1.add_argument("--axes", choices=bn.AXES, default="v1")
    a2 = sub.add_parser("suite")
    a2.add_argument("--data", required=True)
    a2.add_argument("--out", required=True)
    a2.add_argument("--types", default=",".join(SUITE_TYPES))
    a2.add_argument("--variant", choices=bn.PROMPT_VARIANTS, default="evidence_only")
    a3 = sub.add_parser("floor")
    a3.add_argument("--data", required=True)
    a3.add_argument("--splits", default="test")
    a3.add_argument("--suite", default="")
    a3.add_argument("--max-per-family", type=int, default=0)
    a3.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "numeric":
        build_numeric(a.out, a.scale, a.ns, a.compare_v1, axes=a.axes)
        return 0
    if a.cmd == "suite":
        convert_suite(a.data, a.out, [t for t in a.types.split(",") if t], a.variant)
        return 0
    res = floors(a.data, tuple(s for s in a.splits.split(",") if s), a.suite, a.max_per_family)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
