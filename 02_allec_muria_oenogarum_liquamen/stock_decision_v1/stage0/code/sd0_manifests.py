#!/usr/bin/env python3
"""Stage 0 manifests and registries (stock_decision_v1). Reads the frozen outputs of sd0_families / sd0_pools / sd0_pilot /
sd0_reserve / sd0_render (and their S3 keys) and writes:
  manifests/{M0,M1,head_fit,cal_select,dev_select,confirm_seal,discovery}.manifest.json
  manifests/pilot_{C0,B0,C1,B1}.manifest.json        (one per factorial arm; C and B of a mixture share the exposure list)
  source_inventory.json, split_registry.json, overlap_report_internal.json (merged into overlap_report.json by the caller)
The confirmation manifest carries counts, families, digests and keys only (no items, labels or texts).

  python3 sd0_manifests.py --work DIR --s3 S3PREFIX --out DIR
"""
import argparse
import collections
import gzip
import hashlib
import json
import os

import pandas as pd

ARMS = {"C0": ("C", "M0"), "B0": ("B", "M0"), "C1": ("C", "M1"), "B1": ("B", "M1")}
ATTN = {"C": {"global": "causal_same_document", "local": "causal_same_document_window_W"},
        "B": {"global": "bidirectional_same_document", "local": "causal_same_document_window_W"}}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def strata(df):
    return (df.kind.astype(str) + "|" + df.k_bucket.astype(str) + "|" + df.len_bucket.astype(str) + "|" +
            df.hard.map({True: "hard", False: "soft"})).value_counts().sort_index().to_dict()


def comp_summary(df):
    return {"units": int(len(df)), "parents": int(df.groupby(["pool", "parent_key"]).ngroups), "families": int(df.canonical_family.nunique()),
            "upstream_sources": int(df.upstream_source.nunique()), "components": int(df.comp.nunique()),
            "by_area": df.groupby("area").size().to_dict(), "by_kind": df.groupby("kind").size().to_dict(),
            "k_buckets": df.groupby("k_bucket").size().to_dict(), "len_buckets": df.groupby("len_bucket").size().to_dict(),
            "hard_share": round(float(df.hard.mean()), 4), "di_ancestry_units": int(df.di_ancestry.sum())}


def files_block(work, s3, tag, sealed=False):
    out = {}
    for kind in ("decisions", "exposures"):
        p = os.path.join(work, "render", f"{kind}_{tag}.jsonl.gz")
        out[kind] = {"key": f"{s3}/{'confirm_seal_SEALED' if sealed else 'render'}/{kind}_{tag}.jsonl.gz", "sha256": sha_file(p), "md5": md5_file(p)}
    hdr = json.load(open(os.path.join(work, "render", f"render_{tag}.json")))
    out["render_header"] = {"key": f"{s3}/{'confirm_seal_SEALED' if sealed else 'render'}/render_{tag}.json",
                            "sha256": sha_file(os.path.join(work, "render", f"render_{tag}.json"))}
    return out, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W, S3 = a.work, a.s3.rstrip("/")
    os.makedirs(os.path.join(a.out, "manifests"), exist_ok=True)
    u = pd.read_parquet(os.path.join(W, "units_pools.parquet"))
    # ---- publication gates (refuse to write anything if one fails)
    sm = json.load(open(os.path.join(W, "scan_manifest.json")))
    if not sm.get("all_complete") or set(sm["pools"]) != set(u.pool.unique()):
        raise SystemExit("publication gate: protected-overlap scan manifest incomplete or does not cover every pool")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sd0_pools", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd0_pools.py"))
    SP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SP)
    disjoint = SP.validate_disjoint(u)
    train_uids = set()
    for f in ("pilot/pilot_M0.parquet", "pilot/pilot_M1.parquet", "reserve/m1_reserve.parquet"):
        train_uids |= set(pd.read_parquet(os.path.join(W, f), columns=["uid"]).uid)
    pa = u.set_index("uid").pool_assign
    off = [x for x in train_uids if pa.get(x) != "pilot"]
    if off:
        raise SystemExit(f"publication gate: {len(off)} training uids outside the pilot pool, e.g. {off[:3]}")
    TOOLS_NOTE = ("tools: 1 independent cluster per panel (Tasksource shell_safety_v2 / agent_action_safety), transfer unresolved; "
                  "the tools area does not count toward any broad-transfer claim (amendment A2, sha256 146ea1a7...)")
    prep = json.load(open(os.path.join(W, "pools_report.json")))
    pil = json.load(open(os.path.join(W, "pilot", "pilot_report.json")))
    res = json.load(open(os.path.join(W, "reserve", "m1_reserve_report.json")))
    common = {"stock_tokenizer": "google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475", "seed": 20261013,
              "publication_gates": {"protected_scan_manifest": {"key": f"{S3}/work/scan_manifest.json", "sha256": sha_file(os.path.join(W, "scan_manifest.json")), "all_complete": True},
                                    "component_disjointness": disjoint, "training_uids_in_pilot_pool": True},
              "units_pools": {"key": f"{S3}/work/units_pools.parquet", "sha256": sha_file(os.path.join(W, "units_pools.parquet"))}}
    # ---------------- mixtures
    mix = {}
    for m in ("M0", "M1"):
        sel = pd.read_parquet(os.path.join(W, "pilot", f"pilot_{m}.parquet"))
        fb, hdr = files_block(W, S3, f"pilot_{m}")
        e = [json.loads(l) for l in gzip.open(os.path.join(W, "render", f"exposures_pilot_{m}.jsonl.gz"), "rt")]
        ntok = sum(x["n_tokens"] for x in e)
        ex = pd.DataFrame(e)
        body = {"schema": "stock-decision-mixture-manifest-v1", "mixture": m, **common, "files": fb,
                "selection": {"key": f"{S3}/pilot/pilot_{m}.parquet", "sha256": sha_file(os.path.join(W, "pilot", f"pilot_{m}.parquet"))},
                "renderer_sha256": hdr["renderer_sha256"], "renderer": hdr["renderer"], "option_order_rule": hdr["option_order_rule"],
                "label_semantics": hdr.get("label_semantics"),
                "n_exposures": len(sel), "unique_decisions": int(sel.uid.nunique()), "exposure_repetition": {"1": len(sel)},
                "rendered_tokens_real": int(ntok), "tokens_by_len_bucket_real": hdr["tokens_by_len_bucket"],
                "strata": strata(sel), "summary": comp_summary(sel),
                "by_upstream_source": sel.groupby("upstream_source").size().sort_values(ascending=False).to_dict(),
                "by_family": sel.groupby("canonical_family").size().sort_values(ascending=False).to_dict(),
                "by_generator": sel.dropna(subset=["generator_mechanism"]).groupby("generator_mechanism").size().to_dict(),
                "max_siblings_per_parent": int(sel.groupby(["pool", "parent_key"]).size().max())}
        if m == "M0":
            body["definition"] = ("frozen, cleaned, capped sample of the current mixture: garas_gemma_vbll_v2 freeze_r2 recipe A roots "
                                  "(curated emo_pool_v1 + numeric block), quotas = recipe A root shares scaled to 38,400, identity checks rebuilt")
            body["caps_used"] = pil.get("m0_caps_used")
            body["cap_amendment"] = ("source cap raised from 2% (768) to the smallest feasible on the ladder; recipe A has ~48 upstream "
                                     "sources, so 2% caps cannot reach 38,400 exposures")
            body["recipeA_vs_M0_by_root"] = {r: {"recipeA_share": round(n / 120000, 5), "M0_share": round(int((sel.root == r).sum()) / len(sel), 5)}
                                             for r, n in sorted(pil["config"]["m0_recipeA_roots"].items())}
            bp = ex[ex.decision_uid.isin(set(sel.uid[sel.root == "bayes_pref_flights"]))]
            body["bayes_pref_exposures"] = {"root": "bayes_pref_flights", "n": int(len(bp)), "positions": sorted(int(p) for p in bp.pos),
                                            "strata": strata(sel[sel.root == "bayes_pref_flights"])}
        else:
            body["definition"] = "source-balanced mixture of stock-independent data with the (possibly amended) category shares"
            body["category_targets"] = pil["cat_tot"]
            body["category_achieved"] = sel.groupby("m1_category").size().to_dict()
            body["caps_used"] = pil["config"]["m1_caps"]
            body["reserve"] = {"key": f"{S3}/reserve/m1_reserve.parquet", "sha256": sha_file(os.path.join(W, "reserve", "m1_reserve.parquet")), **res}
        body["matching"] = {"fields": ["kind", "K_bucket", "length_bucket", "hard_soft", "exposure_repetition"],
                            "strata_equal_to_other_mixture": pil["strata_equal"], "token_balance": pil["token_balance"]}
        mix[m] = (body, ntok)
        json.dump(body, open(os.path.join(a.out, "manifests", f"{m}.manifest.json"), "w"), indent=1, sort_keys=True, default=str)
    t0, t1 = mix["M0"][1], mix["M1"][1]
    for arm, (att, m) in ARMS.items():
        b = mix[m][0]
        arm_m = {"schema": "stock-decision-pilot-arm-manifest-v1", "arm": arm, "attention_mode": att, "attention": ATTN[att],
                 "mixture": m, "exposures": b["files"]["exposures"], "decisions": b["files"]["decisions"],
                 "renderer_sha256": b["renderer_sha256"], "n_exposures": b["n_exposures"], "rendered_tokens_real": b["rendered_tokens_real"],
                 "shared_option_permutations_with": [x for x, (aa, mm) in ARMS.items() if mm == m and x != arm],
                 "seed0_order": ["C0", "B1", "B0", "C1"], "real_token_gap_M0_vs_M1": round((t0 - t1) / t1, 6),
                 "steps": 600, "global_decisions_per_step": 64}
        json.dump(arm_m, open(os.path.join(a.out, "manifests", f"pilot_{arm}.manifest.json"), "w"), indent=1, sort_keys=True)
    # ---------------- panels
    for tag, sealed in (("head_fit", False), ("cal_select", False), ("dev_select", False), ("discovery", False), ("confirm_seal", True)):
        s = u[u.pool_assign == tag]
        fb, hdr = files_block(W, S3, tag, sealed)
        body = {"schema": "stock-decision-panel-manifest-v1", "panel": tag, **common, "files": fb, "renderer_sha256": hdr["renderer_sha256"],
                "label_semantics": hdr.get("label_semantics"),
                "rendered_tokens_real": hdr["counts"].get("tokens"), "decisions": int(len(s)), "summary": comp_summary(s),
                "uid_set_sha256": hashlib.sha256("\n".join(sorted(s.uid)).encode()).hexdigest()}
        if tag == "confirm_seal":
            r_ = u[u.pool_assign == "confirm_seal_reserve"]
            fr, _ = files_block(W, S3, "confirm_seal_reserve", True)
            body.update(sealed=True, families=sorted(s.canonical_family.unique()),
                        reserve={"files": fr, "decisions": int(len(r_)), "parents": int(r_.groupby(["pool", "parent_key"]).ngroups),
                                 "uid_set_sha256": hashlib.sha256("\n".join(sorted(r_.uid)).encode()).hexdigest()},
                        access="sealed: one final read after the candidate, head, renderer and inference settings are frozen; never sent to teachers")
            body["summary"] = {k: v for k, v in body["summary"].items() if k in ("units", "parents", "families", "components", "by_area")}
        else:
            body["families"] = s.groupby("canonical_family").size().sort_values(ascending=False).to_dict()
        if tag in ("dev_select", "confirm_seal"):
            body["independent_clusters"] = prep["panel_independent_clusters"][tag]
            body["cluster_counts_by_area"] = prep["panel_cluster_counts"][tag]
            body["tools_note"] = TOOLS_NOTE
        if tag == "dev_select":
            body["families_count"] = int(s.canonical_family.nunique())
            body["panel_families_by_area"] = prep["panel_families"]["dev_select"]
        if tag == "discovery":
            body["area_allocation"] = {"initial": prep["discovery_alloc_initial"], "final": prep["discovery_alloc_final"], "achieved": prep["discovery"]}
        json.dump(body, open(os.path.join(a.out, "manifests", f"{tag}.manifest.json"), "w"), indent=1, sort_keys=True, default=str)
    # ---------------- registries
    live = u[u.excl == ""]
    inv = {}
    for src, g in u.groupby("upstream_source"):
        gl = g[g.excl == ""]
        inv[src] = {"pools": sorted(g.pool.unique()), "roots": sorted(set(map(str, g.root.unique()))), "family": g.canonical_family.iloc[0],
                    "generator_mechanisms": sorted(set(str(x) for x in g.generator_mechanism.dropna().unique())), "area": g.area.iloc[0],
                    "skill": g.skill.iloc[0], "domain": g.domain.iloc[0], "label_basis": g.label_basis.iloc[0],
                    "m1_category": g.m1_category.iloc[0], "m1_eligible": bool(g.m1_eligible.any()), "m0_root": bool(g.m0_root.any()),
                    "di_ancestry": bool(g.di_ancestry.any()), "train_units": int(len(g)), "eligible_units": int(len(gl)),
                    "eligible_parents": int(gl.groupby(["pool", "parent_key"]).ngroups),
                    "exclusions": g.loc[g.excl != "", "excl"].value_counts().to_dict(), "assignment": g.pool_assign.value_counts().to_dict(),
                    "eligible_kind": gl.groupby("kind").size().to_dict(), "eligible_k_buckets": gl.groupby("k_bucket").size().to_dict(),
                    "eligible_len_buckets": gl.groupby("len_bucket").size().to_dict(), "eligible_hard_share": round(float(gl.hard.mean()), 4) if len(gl) else None}
    json.dump({"schema": "stock-decision-source-inventory-v1", **common, "sources": inv,
               "totals": {"train_units": int(len(u)), "eligible": int(len(live)), "sources": len(inv),
                          "families": int(u.canonical_family.nunique())}}, open(os.path.join(a.out, "source_inventory.json"), "w"), indent=1, sort_keys=True, default=str)
    fam = u.groupby("canonical_family").agg(reserved_for=("reserved_for", "first"), area=("area", "first"))
    split = {"schema": "stock-decision-split-registry-v1", **common,
             "rules": {"panels": "whole family clusters reserved for dev_select / confirm_seal; other records of reserved families unusable; "
                                 "records of other families sharing an ancestry component with a reserved family removed ('bridge')",
                       "training_pools": "component-disjoint carving: head_fit (decisions), cal_select (parents), discovery (parents); rest pilot",
                       "max_siblings_per_parent": 4},
             "families": {f: {"split": (r.reserved_for or "train"), "area": r.area} for f, r in fam.iterrows()},
             "pool_assign_counts": u.pool_assign.value_counts().to_dict(),
             "panel_families": prep["panel_families"], "panel_shortfalls_by_area": prep["panel_shortfalls_by_area"],
             "panel_cluster_eligibility": prep["panel_cluster_eligibility"], "bridge_units": prep["bridge_units"],
             "components_too_big_to_carve": prep["components_too_big_to_carve"],
             "pools_report": {"key": f"{S3}/work/pools_report.json", "sha256": sha_file(os.path.join(W, "pools_report.json"))}}
    json.dump(split, open(os.path.join(a.out, "split_registry.json"), "w"), indent=1, sort_keys=True, default=str)
    print("manifests written")


if __name__ == "__main__":
    main()
