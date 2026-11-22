#!/usr/bin/env python3
"""Regression tests for REVIEW_2026-10-13_2105 findings 1, 2 and 5 against the CURRENT Stage 0 code (fabricated rows only).
  panel_overlap_repro : two families, ten parents each, one shared ancestry component -> dev / confirmation component-disjoint,
                        the shared component quarantined, validator clean; a fabricated overlap makes the validator refuse.
  family_count_repro  : two clusters of five linked families, minimum 5 clusters per panel -> the run STOPS (exit 3) reporting
                        1 cluster per panel; an explicit approved amendment lets it run and it reports 1 independent cluster.
  missing_overlap_repro: sd0_families refuses without a scan manifest, with an incomplete entry, or with a missing hits file;
                        a COMPLETE entry with a valid EMPTY hits file runs (zero hits != missing scan).
Run: python3 tests/test_pools_review_2105.py   (needs pandas + pyarrow)"""
import gzip
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

import pandas as pd

CODE = pathlib.Path(__file__).resolve().parents[1]
AREAS = ["knowledge", "language", "retrieval", "tools", "arts"]
FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def cfg(minc, approved=None):
    c = dict(seed=1, panel_bad_basis=[], panel_allow=[], panel_exclude_mechanisms=[], min_parents=5, panel_parents=10,
             area_weights={a: int(a == "knowledge") for a in AREAS}, panel_family_cap=10,
             min_clusters_per_area={s: {a: minc if a == "knowledge" else 0 for a in AREAS} for s in ["dev_select", "confirm_seal"]},
             min_clusters_total={}, max_sib=4, confirm_reserve_frac=0, comp_max_parents=50, m0_recipeA_roots={}, m1_shares={},
             head_fit_decisions=0, head_fit_frac=.1, cal_parents=0, cal_frac=.03, discovery_parents=0, discovery_frac=.1, discovery_bad_basis=[])
    if approved:
        c["approved_amendments"] = approved
    return c


def unit(fam, j, comp):
    u = f"{fam}_{j}"
    return dict(uid=u, pool="synthetic", parent_key=u, canonical_family=fam, comp=comp, excl="", upstream_source=fam, root=fam,
                m0_root=False, m1_eligible=True, m1_category="test", di_ancestry=False, label_basis="human_annotation", area="knowledge")


def run_pools(p, rows, fams, c):
    pd.DataFrame(rows).to_parquet(p / "u.parquet")
    pd.DataFrame(fams).to_parquet(p / "f.parquet")
    (p / "cfg.json").write_text(json.dumps(c))
    return subprocess.run([sys.executable, str(CODE / "sd0_pools.py"), "--units", str(p / "u.parquet"), "--families", str(p / "f.parquet"),
                           "--config", str(p / "cfg.json"), "--out", str(p / "out")], capture_output=True, text=True)


def fam_row(fam, cluster, n):
    return dict(canonical_family=fam, cluster=cluster, parents=n, di=False, m0=False, m1=True, mech="", basis="human_annotation", area="knowledge")


def panel_overlap():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)
        rows, fams = [], []
        for c in range(2):
            fam = f"fam{c}_0"
            fams.append(fam_row(fam, f"cluster{c}", 10))
            for j in range(10):
                rows.append(unit(fam, j, 0 if j == 0 else len(rows) + 1))
        r = run_pools(p, rows, fams, cfg(1))
        check(r.returncode == 0, f"panel_overlap: builder runs (rc {r.returncode}) {r.stderr[-300:]}")
        if r.returncode:
            return
        out = pd.read_parquet(p / "out/units_pools.parquet")
        rep = json.load(open(p / "out/pools_report.json"))
        dv = set(out.loc[out.pool_assign.str.startswith("dev_select"), "comp"])
        cv = set(out.loc[out.pool_assign.str.startswith("confirm_seal"), "comp"])
        check(not (dv & cv), f"panel_overlap: DEV_CONFIRM_SHARED_COMPONENTS {sorted(dv & cv)} is empty")
        check(rep.get("cross_panel_components") == 1, "panel_overlap: the shared component is quarantined before parent selection")
        check(set(out.loc[out.comp == 0, "pool_assign"]) == {"cross_panel_bridge"}, "panel_overlap: both sides of component 0 are quarantined")
        check((rep.get("component_disjointness") or {}).get("intersections") == 0, "panel_overlap: publication validator reports 0 intersections")
        sys.path.insert(0, str(CODE))
        import importlib
        sp = importlib.import_module("sd0_pools")
        bad = out.copy()
        i = bad.index[bad.pool_assign == "dev_select"][0]
        j = bad.index[bad.pool_assign == "confirm_seal"][0]
        bad.loc[j, "comp"] = bad.loc[i, "comp"]
        try:
            sp.validate_disjoint(bad)
            check(False, "panel_overlap: validator refuses a fabricated dev/confirm overlap")
        except SystemExit as e:
            check("DISJOINTNESS FAILED" in str(e), "panel_overlap: validator refuses a fabricated dev/confirm overlap")


def family_count():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)
        rows, fams = [], []
        for c in range(2):
            for f in range(5):
                fam = f"fam{c}_{f}"
                fams.append(fam_row(fam, f"cluster{c}", 10))
                for j in range(10):
                    rows.append(unit(fam, j, len(rows)))
        r = run_pools(p, rows, fams, cfg(5))
        rep = json.load(open(p / "out/pools_report.json"))
        check(r.returncode == 3 and "STOPPED" in rep, f"family_count: 5 linked families in 1 cluster do not satisfy a 5-cluster minimum (rc {r.returncode})")
        un = (rep.get("STOPPED") or {}).get("unmet", {})
        check(un.get("dev_select/knowledge", {}).get("clusters") == 1 and un.get("confirm_seal/knowledge", {}).get("clusters") == 1,
              f"family_count: shortfall reports 1 cluster per panel {un}")
        check(not (p / "out/units_pools.parquet").exists(), "family_count: no pools written when stopped")
        r = run_pools(p, rows, fams, cfg(5, {"dev_select/knowledge": 1, "confirm_seal/knowledge": 1}))
        rep = json.load(open(p / "out/pools_report.json"))
        check(r.returncode == 0 and rep.get("panel_independent_clusters") == {"dev_select": 1, "confirm_seal": 1},
              f"family_count: with an explicit amendment it runs and reports 1 independent cluster per panel ({rep.get('panel_independent_clusters')})")
        check(len(rep["panel_families"]["dev_select"]["knowledge"]) == 5, "family_count: family names reported separately (5 in the dev cluster)")


def missing_overlap():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)
        (p / "hits").mkdir()
        (p / "dec").mkdir()
        row = dict(uid="synthetic|a|0|q", base_ok=True, base_reason="", rk="rk0", canonical_family="independent", parent_key="p0",
                   area="knowledge", upstream_source="source0", pool="synthetic", m0_root=False, m1_eligible=True, m1_category="human_grounded",
                   di_ancestry=False, label_basis="human_annotation", generator_mechanism="")
        pd.DataFrame([row, {**row, "uid": "synthetic|a|1|q", "rk": "rk1", "base_ok": False, "base_reason": "bad_license"}]).to_parquet(p / "cand.parquet")
        pd.DataFrame([dict(rk="rk0", comp=0), dict(rk="rk1", comp=1)]).to_parquet(p / "comp.parquet")
        with gzip.open(p / "dec/dec_synthetic.jsonl.gz", "wt") as f:
            f.write(json.dumps(dict(uid=row["uid"], dec="g", dec_nogold="v")) + "\n")
            f.write(json.dumps(dict(uid="synthetic|a|1|q", dec="x", dec_nogold="x")) + "\n")
        base = [sys.executable, str(CODE / "sd0_families.py"), "--cand", str(p / "cand.parquet"), "--comp", str(p / "comp.parquet"),
                "--dec", str(p / "dec"), "--pools", "synthetic", "--hits", str(p / "hits"), "--out", str(p / "out")]
        r = subprocess.run(base, capture_output=True, text=True)
        check(r.returncode != 0, "missing_overlap: refuses without a scan manifest")
        (p / "sm.json").write_text(json.dumps({"pools": {}}))
        r = subprocess.run(base + ["--scan-manifest", str(p / "sm.json")], capture_output=True, text=True)
        check(r.returncode != 0 and "protected-overlap gate" in r.stderr, "missing_overlap: refuses a pool with no scan-manifest entry")
        (p / "sm.json").write_text(json.dumps({"pools": {"synthetic": {"status": "INCOMPLETE", "reasons": ["missing hits_local"]}}}))
        r = subprocess.run(base + ["--scan-manifest", str(p / "sm.json")], capture_output=True, text=True)
        check(r.returncode != 0, "missing_overlap: refuses an INCOMPLETE scan")
        with gzip.open(p / "hits/hits_synthetic.jsonl.gz", "wt"):
            pass                                   # a valid EMPTY hits artifact
        m = hashlib.md5(open(p / "hits/hits_synthetic.jsonl.gz", "rb").read()).hexdigest()
        (p / "sm.json").write_text(json.dumps({"pools": {"synthetic": {"status": "COMPLETE", "reasons": [], "hits": {"md5_local": m}}}}))
        (p / "hits/hits_synthetic.jsonl.gz").rename(p / "hits/moved.gz")
        r = subprocess.run(base + ["--scan-manifest", str(p / "sm.json")], capture_output=True, text=True)
        check(r.returncode != 0 and "hits file missing" in r.stderr, "missing_overlap: refuses a COMPLETE entry whose hits file is missing")
        (p / "hits/moved.gz").rename(p / "hits/hits_synthetic.jsonl.gz")
        r = subprocess.run(base + ["--scan-manifest", str(p / "sm.json")], capture_output=True, text=True)
        ok = r.returncode == 0 and json.load(open(p / "out/family_links.json"))["eligible"] == 1
        check(ok, f"missing_overlap: a COMPLETE scan with a valid empty hits file runs (zero hits) {r.stderr[-300:]}")


if __name__ == "__main__":
    panel_overlap()
    family_count()
    missing_overlap()
    print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
    sys.exit(1 if FAIL else 0)
