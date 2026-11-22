#!/usr/bin/env python3
"""Stage 0 licence registry (stock_decision_v1): one entry per upstream source of every candidate pool.

Fields: licence strings as recorded on the records (with counts), resolved licence, evidence, commercial_use (yes / no / unresolved),
attribution_required, share_alike (derivatives / generated descendants inherit the terms), generation_permission (may we generate
descendants from it for training: yes / yes_share_alike / no / unresolved), line (clean / NC / quarantine), DI ancestry,
model-labelled flag, M0 / M1 use. Rule: anything not resolvable to a commercial-use licence goes to quarantine (not trained, not
used for generation); NC sources are the NC line (never in M0 / M1 / panels here). Tasksource rows keep their per-row upstream
licence metadata; the filtered-full export admits only license_use == commercial rows.

  python3 sd0_licences.py --core licence_strings_core.json --map source_map_core_v1.json [--ts tasksource_sources_v1.json] --out license_registry.json
"""
import argparse
import json
import re

RESOLVED = {   # licences recorded as "(verify)" on the records, resolved from the dataset cards (HF API cardData.license, fetched 2026-10-13)
    "banking77": ("CC-BY-4.0", "huggingface.co/datasets/PolyAI/banking77 @90d4e2ee5521 cardData.license cc-by-4.0"),
    "clinc_oos": ("CC-BY-3.0", "huggingface.co/datasets/clinc/clinc_oos @155b9c710419 cardData.license cc-by-3.0"),
    "go_emotions": ("Apache-2.0", "huggingface.co/datasets/google-research-datasets/go_emotions @add492243ff9 cardData.license apache-2.0"),
    "gsm8k": ("MIT", "huggingface.co/datasets/openai/gsm8k @740312add88f cardData.license mit"),
    "measuring_hate_speech": ("CC-BY-4.0", "huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech @5468f6e11839 cardData.license cc-by-4.0"),
}
SA = re.compile(r"BY-SA|share-alike|ShareAlike|GFDL|GPL", re.I)
NC = re.compile(r"\bNC\b|non-?commercial|research terms", re.I)
PD = re.compile(r"public domain|CC0|17 U\.?S\.?C\.? 105|no use restrictions|unrestricted|\(ours\)", re.I)
ATTR = re.compile(r"CC-?BY|Apache|MIT|GPL|GFDL|cite", re.I)


def classify(strings):
    s = " || ".join(x for x, _ in strings)
    if NC.search(s):
        return "no", "NC"
    if "(verify)" in s:
        return "unresolved", "quarantine"
    if PD.search(s) or ATTR.search(s):
        return "yes", "clean"
    return "unresolved", "quarantine"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--ts", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    core = json.load(open(a.core))
    smap = json.load(open(a.map))["roots"]
    smap.setdefault("swiss", {**smap["courts"], "upstream_source": "swiss_federal_court"})
    reg = {}
    for root, strings in sorted(core.items()):
        key = root if root != "bf" else "bayes_families"
        m = smap.get(root) or {}
        if root == "bf":
            m = {"upstream_source": "bayes_families", "label_basis": "exact_posterior", "m1_eligible": True}
        commercial, line = classify(strings)
        resolved, evidence = None, "record licence field (converter-recorded, with source evidence in the string)"
        if root in RESOLVED:
            resolved, evidence = RESOLVED[root]
            commercial, line = "yes", "clean"
        sa = bool(SA.search(" ".join(x for x, _ in strings)))
        gen_override = None
        if root == "ctgov":     # US NLM registry data; terms page is JS-rendered and was not re-fetched in this pass
            commercial, line = "yes", "clean"
            evidence = "v1 curation adjudication (US NLM ClinicalTrials.gov registry data); terms page not re-verified in this pass"
            gen_override = "unresolved"
        model_lab = m.get("label_basis") == "model_or_unverified_synthetic"
        gen = gen_override or ("no" if line != "clean" else ("yes_share_alike" if sa else "yes"))
        reg[key] = {"upstream_source": m.get("upstream_source", key), "record_licence_strings": strings, "resolved_licence": resolved,
                    "evidence": evidence, "commercial_use": commercial, "attribution_required": bool(ATTR.search(" ".join(x for x, _ in strings))),
                    "share_alike": sa, "generation_permission": gen, "line": line, "model_labelled": model_lab,
                    "di_ancestry": (m.get("m1_reason") or "").startswith("Decision Index"),
                    "m1_eligible": bool(m.get("m1_eligible")), "m1_reason": m.get("m1_reason", "")}
    if a.ts:
        ts = json.load(open(a.ts))
        for src, v in sorted((ts.get("sources") or {}).items()):
            lic = v.get("licences") or []
            strings = [[str(x.get("license")), int(x.get("rows") or 0)] for x in lic]
            uses = {str(x.get("license_use")) for x in lic}
            commercial = "yes" if uses <= {"commercial"} and uses else "unresolved"
            custom = any(re.search(r"\bcustom\b|\bother\b|unknown", x[0], re.I) for x in strings)
            sa = bool(SA.search(" ".join(x for x, _ in strings)))
            reg[f"tasksource:{src}"] = {"upstream_source": v.get("upstream_source") or v.get("upstream_dataset") or src, "record_licence_strings": strings,
                                        "resolved_licence": None, "evidence": "tasksource filtered-full per-row license metadata (license_use == commercial)",
                                        "commercial_use": commercial, "attribution_required": True, "share_alike": sa,
                                        "custom_or_other_terms": custom,
                                        "generation_permission": ("unresolved" if custom else "yes_share_alike" if sa else "yes") if commercial == "yes" else "unresolved",
                                        "line": "clean" if commercial == "yes" else "quarantine", "model_labelled": False,
                                        "di_ancestry": bool(v.get("di_root")), "m1_eligible": bool(v.get("m1_eligible")) and commercial == "yes",
                                        "m1_reason": v.get("m1_reason") or ("" if commercial == "yes" else "licence use unresolved")}
    out = {"schema": "stock-decision-license-registry-v1", "rules": __doc__.strip(), "entries": reg,
           "counts": {k: sum(1 for v in reg.values() if v["line"] == k) for k in ("clean", "NC", "quarantine")}}
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(out["counts"]), sorted(k for k, v in reg.items() if v["line"] != "clean"))


if __name__ == "__main__":
    main()
