#!/usr/bin/env python3
"""registry_v3 = registry_v2's 59 populations (pins unchanged) + the protected material created since 10-12 (metadata only)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = [
    {"population": "confirm_p1_index", "kind": "opaque_index", "required": True, "expected_md5": "25950115cea0cb543a042685efe1cf12",
     "s3_candidates": ["sealed/confirm_p1/index/confirm_p1_exclusion_index_v1.json.gz"],
     "note": "CONFIRM-P1 exclusion index (ids, state_sha1, clusters, entity keys of bf_v2 / synth_ev_v1 / rw_v1). Items, salts and keys are "
             "never fetched (sealed/confirm_p1/*/items.tgz.enc and the node-side .keys stay untouched)"},
    {"population": "render1_dev_v1", "kind": "tarball", "required": True, "tarball_md5": "5d73b18e13145e85f7ed587d83195098",
     "s3_candidates": ["results_render1/data/render1_dev_v1.tgz"], "member_regex": r"^(?!.*/\._).*\.jsonl$",
     "note": "RENDER1 dev bundle (P_def, cval, didev, egdev, caldev): read panels of the renderer decision"},
    {"population": "render0_pop_v1", "kind": "tarball", "required": True, "tarball_md5": "9069c7c2bd0a9cc19beb956dd9287f9d",
     "s3_candidates": ["results_render0/data/render0_pop_v1.tgz"], "member_regex": r"^(?!.*/\._).*\.jsonl$",
     "note": "RENDER0 stage-0 scoring population (curated val + DIDEV RAGTruth / PhishNChips)"},
    {"population": "fullft_devP_rows", "kind": "jsonl_gz", "required": True, "file_kind": "rendered",
     "s3_candidates": ["results_fullft/render/devP/rows.jsonl.gz"],
     "note": "fullft pilot DEV devP (11,963 units: pilot-val, caldev, EG val, didev, bfdev) as RENDERED prompts (field: prompt)"},
    {"population": "fullft_devB_rows", "kind": "jsonl_gz", "required": True, "file_kind": "rendered",
     "s3_candidates": ["results_fullft/render/devB/rows.jsonl.gz"],
     "note": "fullft LR-probe DEV devB as RENDERED prompts (field: prompt)"},
    {"population": "di03_suite_rows", "kind": "side_raw_rows", "required": True, "after": "di03_gsm8k_v3",
     "globs": ["suite-0.3/selected-rows.jsonl.gz", "suite-0.3/added-rows.jsonl.gz", "suite-0.3/gsm8k-rows.jsonl.gz"],
     "convert_decision_index_md5": "dc22d68350c28a6c23ba3a1adcc8c75a",
     "note": "Decision Index 0.3 public suite rows (every benchmark; built by the kit's own sha256-gated suite03 step during di03_gsm8k_v3 "
             "staging); converted where convert_decision_index knows the catalog, else every raw state + question field kept"},
]


def main():
    reg = json.loads((HERE / "registry_v2.json").read_text())
    names = {P["population"] for P in reg["populations"]}
    assert not names & {P["population"] for P in NEW}
    reg["schema"] = "stock-decision-protected-registry-v3"
    reg["derived_from"] = {"registry_v2_sha256": hashlib.sha256((HERE / "registry_v2.json").read_bytes()).hexdigest()}
    reg["populations"] = reg["populations"] + NEW
    reg["not_included"] = (reg.get("sealed_not_indexable") or []) + [
        {"name": "CONFIRM-P1 items (bf_v2, synth_ev_v1, rw_v1)", "reason": "sealed; covered by confirm_p1_index (hashes / ids only)"},
        {"name": "SCREEN4F / SCREEN4M panels", "reason": "no separate panel object found: SCREEN4F/M read the SCREEN4 panels (screen4_dev) "
                                                         "and s4_mixed_v1.tgz is a training mixture, not an evaluation panel"},
        {"name": "Tasksource heldouts (default/full validation, test)", "reason": "not part of this line's protected set; Tasksource's own "
                                                                                "filter already protected them for filtered-full"},
        {"name": "this line's new dev_select / cal_select / confirm_seal panels", "reason": "built later by the parent from candidate "
                                                                                           "pools; split by ancestry components, not by this index"}]
    out = HERE / "registry_v3.json"
    out.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    print(len(reg["populations"]), hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
