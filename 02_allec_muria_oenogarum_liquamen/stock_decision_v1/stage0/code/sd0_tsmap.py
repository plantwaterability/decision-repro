#!/usr/bin/env python3
"""Stage 0 Tasksource source map (stock_decision_v1), from the TASKSOURCE inventory (tasksource_sources_v1.json) with these fixed
decisions: upstream source = upstream dataset, aliased to our curated source names where it is the same upstream (HelpSteer2,
go_emotions, social_i_qa, cosmos_qa, e-CARE, medmcqa, scruples, procedural-typed-decisions); canonical family = the proposed family;
rule-generated synthetic reasoning corpora (bAbI, RuleTaker, ParaRule-Plus, StepGame, SpartQA, LogicalEntailment, Corr2Cause, ToMi,
MindGames, FOL-NLI, probability_words_nli, SpaceNLI, ConceptRules v2) are M1 'verified_synthetic' with one generator mechanism per
corpus (labels come from the corpus generator / prover, not annotators); procedural-typed-decisions is a Typed-Decisions workflow
imitation (excluded from M1 and panels, as for the curated copy); UltraFeedback-paired and WildGuardMix carry model-produced labels
(kept in the Tasksource share, excluded from panels and discovery). Everything else: 'tasksource_filtered_full', upstream labels
retained by the GLM-5.3 review (label basis upstream_annotation_llm_reviewed), EXCEPT sources whose upstream human annotation is
documented on the dataset card (tasksource_label_evidence_v1.json): label basis human_annotation with the card as evidence. WANLI and CommonsenseQA 2.0 are not the DI
benchmarks ANLI / CommonsenseQA (substring matches only)."""
import json
import sys

RULE_SYN = {"babi", "ruletaker", "pararule_plus", "stepgame", "spartqa", "logical_entailment", "corr2cause", "tomi_nli", "mindgames",
            "fol_nli", "probability_words_nli", "spacenli", "conceptrules_v2"}
MODEL_LAB = {"ultrafeedback_paired", "wildguardmix_cleaned"}
ALIAS_UP = {"nvidia/HelpSteer2": "helpsteer2", "google-research-datasets/go_emotions": "go_emotions", "tasksource/social_i_qa": "social_i_qa",
            "Samsoup/cosmos_qa": "cosmos_qa", "12ml/e-CARE": "e_care", "openlifescienceai/medmcqa": "medmcqa", "tasksource/scruples": "scruples"}
inv = json.load(open(sys.argv[1]))
LEV = json.load(open(sys.argv[3]))["sources"] if len(sys.argv) > 3 else {}    # tasksource_label_evidence_v1.json
out = {}
for s, v in inv["sources"].items():
    fam = v["proposed_family"]
    up = v.get("upstream_dataset") or f"(unknown:{s})"
    up = ALIAS_UP.get(up, up)
    mech = None
    cat, ok, why, basis = "tasksource_filtered_full", True, "", "upstream_annotation_llm_reviewed"
    if fam == "procedural_typed_decisions":
        up, mech, cat, ok, why, basis = "procedural_typed_decisions", "procedural/" + s.split("/")[-1], None, False, \
            "synthetic imitation of Typed-Decisions workflows (zero-shot policy)", "rule_synthetic"
    elif fam in RULE_SYN:
        mech, cat, basis = "ts_rule/" + fam, "verified_synthetic", "rule_oracle"
    elif fam in MODEL_LAB:
        basis = "model_labelled_upstream"
    elif (LEV.get(s) or {}).get("label_basis") == "human_annotation":
        basis = "human_annotation"                     # documented human annotation upstream (dataset card): tier source_gold
    out[s] = {"root": "ts:" + s, "upstream_source": up, "canonical_family": fam, "generator_mechanism": mech,
              "area": v["proposed_area"], "skill": v.get("proposed_skill"), "domain": v.get("proposed_domain"), "label_basis": basis,
              "m1_category": cat if ok else None, "m1_eligible": ok, "m1_reason": why, "upstream_dataset": v.get("upstream_dataset"),
              "upstream_revisions": v.get("upstream_revisions"), "licences": v.get("licences"), "variants": v.get("variants"),
              "di_root": False, "label_evidence": (LEV.get(s) or {}).get("label_evidence") if basis == "human_annotation" else None}
json.dump({"schema": "stock-decision-tasksource-map-v1", "doc": __doc__, "sources": out}, open(sys.argv[2], "w"), indent=1, sort_keys=True)
import collections
c = collections.Counter((v["m1_category"], v["area"]) for v in out.values())
print(len(out), sorted(c.items(), key=lambda x: str(x)))
