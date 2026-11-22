#!/usr/bin/env python3
"""Stage 0 source map for the NON-Tasksource candidate pools (curated emo_pool_v1, garas synth supply, MIRACL / QASPER /
ContractNLI / HANS / BLiMP, bayes_families_v1 train). One entry per v2 ROOT (or synthetic generator), stating:
  upstream source (2% cap unit), canonical family (6% cap unit), generator mechanism (2% cap unit, synthetic only),
  Decision Index area (knowledge / language / retrieval / tools / arts) with the skill that puts it there, domain,
  label basis, M0 / M1 eligibility and the reason.
Rules fixed here, before any selection:
  * DI area by SKILL, following the DI 0.3 area table (research/decision_index.md): forecasting realised outcomes -> arts (ForecastBench,
    Habermas sit in arts); NLI / commonsense / reading comprehension -> language (ANLI, WinoGrande, HellaSwag, NLI4CT);
    classification / relevance / fact verification -> retrieval; tool / agent action -> tools; math / science / chess / facts -> knowledge.
  * M1 is stock-independent: no model-labelled or teacher-labelled sources (mghafiri v1 is Claude-labelled, jebadiah is half
    Qwen-teacher, kaivoss is distillation data, vagmi unverified), no synthetic imitations of Typed-Decisions workflows
    (procedural_typed_decisions), no Decision Index benchmark roots (training on a scored benchmark's train split makes it
    'fitted'; Tasksource filtered-full applies the same holdout), no NC. M0 keeps the current mixture's sources (cleaned/capped).
  * Synthetic generator mechanisms are grouped by the underlying model, not by package or template: Beta-Bernoulli posterior
    predictive (garas beta_rate, bayes_families coin / rate) is ONE mechanism (2% generator cap across packages).
  * For synthetic data the 'upstream source' (2% cap) is the generator within its package; the canonical family (6% cap) is the
    document collection: garas comparative_cost / grounded_answer / linked_facts / retrieval_rank / scope_discrimination share one
    synthetic fact world (one ancestry component of 51k lineages), so they form ONE family.
"""
import json
import sys

A = {}


def put(root, source, family, area, skill, domain, basis, m1_cat, m1_ok, why="", mech=None, pools=("emo_pool_v1",)):
    A[root] = {"upstream_source": source, "canonical_family": family, "generator_mechanism": mech, "area": area, "skill": skill,
               "domain": domain, "label_basis": basis, "m1_category": m1_cat if m1_ok else None, "m1_eligible": bool(m1_ok),
               "m1_reason": why, "pools": list(pools)}


# ---- curated emo_pool_v1 real-outcome (rd3 / gdelt): realised outcomes, stock-independent
for r, src, area, skill, dom in [
        ("afd", "wikipedia_afd", "arts", "forecast_community_decision", "wikipedia_governance"),
        ("aidev", "aidev_pr_outcomes", "arts", "forecast_software_outcome", "software"),
        ("courts", "courtlistener_swiss_courts", "arts", "forecast_legal_outcome", "law"),
        ("ctgov", "clinicaltrials_gov", "arts", "forecast_trial_outcome", "medicine"),
        ("dmhc", "ca_dmhc_imr", "arts", "forecast_review_outcome", "health_insurance"),
        ("fda", "openfda", "arts", "forecast_regulatory_outcome", "regulation"),
        ("folktables", "folktables_acs", "arts", "forecast_tabular_outcome", "census"),
        ("openreview", "openreview", "arts", "forecast_peer_review_outcome", "science_review"),
        ("safety", "osha_nhtsa", "arts", "forecast_safety_outcome", "safety_regulation"),
        ("swe_rebench", "swe_rebench", "arts", "forecast_software_outcome", "software"),
        ("gdelt", "gdelt", "arts", "forecast_event_statistics", "news_events"),
        ("genomics", "genomics_outcomes", "knowledge", "scientific_outcome", "biology"),
        ("physics", "physics_outcomes", "knowledge", "scientific_outcome", "physics"),
        ("lichess", "lichess", "knowledge", "chess_outcome", "games"),
        ("wiki_census", "wikipedia_census", "knowledge", "quantity_estimation", "demographics"),
        ("wikidata", "wikidata", "knowledge", "factual_knowledge", "encyclopedic"),
        ("nvd_cwe", "nvd_cwe", "retrieval", "classification", "security")]:
    put(r, src, src, area, skill, dom, "realised_outcome", "human_grounded", True)
put("open_jev", "open_jev", "open_jev", "tools", "web_agent_action", "web", "recorded_agent_action", "human_grounded", False,
    "action labels come from recorded agent trajectories, not verified human / oracle decisions: review before M1")
put("esci", "amazon_esci", "amazon_esci", "retrieval", "relevance", "ecommerce", "human_annotation", None, False, "DI benchmark root (37_esci)")
# ---- curated human-labelled public
put("pubmedqa", "pubmedqa", "pubmedqa", "language", "reading_comprehension", "biomedicine", "human_annotation", "human_grounded", True)
put("medmcqa", "medmcqa", "medmcqa", "knowledge", "exam_qa", "medicine", "human_annotation", "human_grounded", True)
put("code_contests", "code_contests", "code_contests", "knowledge", "program_correctness", "software", "execution_verified", "human_grounded", True)
put("cosmos_qa", "cosmos_qa", "cosmos_qa", "language", "commonsense_reading", "narrative", "human_annotation", "human_grounded", True)
put("e_care", "e_care", "e_care", "language", "causal_commonsense", "general", "human_annotation", "human_grounded", True)
put("social_i_qa", "social_i_qa", "social_i_qa", "language", "social_commonsense", "general", "human_annotation", "human_grounded", True)
put("bbq", "bbq", "bbq", "language", "bias_qa", "social", "template_rule", "human_grounded", True)
put("casino", "casino", "casino", "arts", "negotiation_outcome", "dialogue", "realised_outcome", "human_grounded", True)
put("helpsteer2", "helpsteer2", "nvidia_helpsteer", "arts", "response_quality_rating", "assistant_responses", "human_rating", "human_grounded", True)
for r, src, fam, skill, dom in [("go_emotions", "go_emotions", "go_emotions", "emotion_distribution", "social_media"),
                                ("measuring_hate_speech", "measuring_hate_speech", "measuring_hate_speech", "annotator_distribution", "social_media"),
                                ("stanford_politeness", "stanford_politeness", "stanford_politeness", "politeness_distribution", "online_requests"),
                                ("ibm_argq", "ibm_argq", "ibm_argument_quality", "argument_quality_distribution", "debate"),
                                ("scruples", "scruples", "scruples", "moral_judgment_distribution", "everyday_ethics")]:
    put(r, src, fam, "arts", skill, dom, "human_distribution", "probabilistic_disagreement", True)
# ---- curated DI-ancestry roots: M0 only
for r, src, area, skill in [("banking77", "banking77", "retrieval", "intent_classification"), ("clinc_oos", "clinc150", "retrieval", "intent_classification"),
                            ("gsm8k", "gsm8k", "knowledge", "math_word_problems"), ("circa", "circa", "language", "indirect_answers"),
                            ("commonsense_qa", "commonsense_qa", "language", "commonsense_qa"), ("winogrande", "winogrande", "language", "coreference_commonsense"),
                            ("tabfact", "tabfact", "language", "table_fact_verification"),
                            ("contract_nli", "contract_nli", "language", "document_nli"), ("blimp", "blimp", "language", "grammaticality")]:
    put(r, src, src, area, skill, "general", "human_annotation", None, False, "Decision Index benchmark root (fitted if trained)",
        pools=("contractnli_v1",) if r == "contract_nli" else ("blimp_v1",) if r == "blimp" else ("emo_pool_v1",))
put("ms_marco", "ms_marco", "ms_marco", "retrieval", "relevance", "web", "human_annotation", None, False, "NC licence line")
# ---- curated model-labelled / workflow-imitation: M0 only
for r, why in [("mghafiri", "Claude-labelled (v1)"), ("jebadiah", "half its teacher is Qwen3.8-27B"), ("kaivoss", "distillation data (system-one-270m)"),
               ("vagmi", "community synthetic, labels unverified by us")]:
    put(r, r, r, "tools", "policy_tool_decision", "business_workflows", "model_or_unverified_synthetic", None, False, why)
put("procedural_typed_decisions", "procedural_typed_decisions", "procedural_typed_decisions", "language", "procedural_state_tracking",
    "typed_decision_workflows", "rule_synthetic", None, False, "synthetic imitation of Typed-Decisions workflows (zero-shot policy)")
# ---- real sources acquired for v2
put("miracl", "miracl", "miracl", "retrieval", "multilingual_relevance", "wikipedia", "human_annotation", "human_grounded", True, pools=("miracl_v1",))
put("qasper", "qasper", "qasper", "language", "scientific_document_qa", "nlp_papers", "human_annotation", "human_grounded", True, pools=("qasper_v1",))
put("hans", "hans", "hans", "language", "nli_heuristics", "general", "template_rule", "human_grounded", True, pools=("hans_v1",))
# ---- exact probabilistic (oracle) and verified synthetic: mechanism-grouped
put("bayes_pref_flights", "bayes_pref", "bayes_pref", "knowledge", "preference_posterior", "travel", "exact_posterior",
    "probabilistic_disagreement", True, mech="preference_posterior_bayes_pref")
put("synth_beta_rate", "garas_synth/beta_rate", "garas_synth/beta_rate", "knowledge", "posterior_predictive", "synthetic", "exact_posterior",
    "probabilistic_disagreement", True, mech="beta_bernoulli_posterior", pools=("synth_supply_v1", "emo_pool_v1"))
put("synth_binary_evidence", "garas_synth/binary_evidence", "garas_synth/binary_evidence", "knowledge", "likelihood_ratio_update", "synthetic", "exact_posterior",
    "probabilistic_disagreement", True, mech="binary_likelihood_ratio", pools=("synth_supply_v1", "emo_pool_v1"))
put("synth_comparative_cost", "garas_synth/comparative_cost", "garas_synth/fact_world", "knowledge", "expected_loss_action", "synthetic", "exact_action",
    "verified_synthetic", True, mech="comparative_cost_expected_loss", pools=("synth_supply_v1",))
put("synth_grounded_answer", "garas_synth/grounded_answer", "garas_synth/fact_world", "language", "grounded_answerability", "synthetic", "rule_oracle",
    "verified_synthetic", True, mech="grounded_answer_rules", pools=("synth_supply_v1",))
put("synth_linked_facts", "garas_synth/linked_facts", "garas_synth/fact_world", "language", "multi_hop_linked_evidence", "synthetic", "rule_oracle",
    "verified_synthetic", True, mech="linked_facts_graph", pools=("synth_supply_v1",))
put("synth_retrieval_rank", "garas_synth/retrieval_rank", "garas_synth/fact_world", "retrieval", "relevance_ranking", "synthetic", "rule_oracle",
    "verified_synthetic", True, mech="synthetic_fact_retrieval", pools=("synth_supply_v1",))
put("synth_scope_discrimination", "garas_synth/scope_discrimination", "garas_synth/fact_world", "language", "scope_discrimination", "synthetic", "rule_oracle",
    "verified_synthetic", True, mech="scope_discrimination_rules", pools=("synth_supply_v1",))
for g, mech, skill in [("coin", "beta_bernoulli_posterior", "posterior_predictive"), ("rate", "beta_bernoulli_posterior", "posterior_predictive"),
                       ("normal_mean", "normal_mean_posterior", "posterior_predictive"), ("lr", "binary_likelihood_ratio", "likelihood_ratio_update")]:
    put(f"bf_{g}", f"bayes_families/{g}", f"bayes_families/{g}", "knowledge", skill, "synthetic", "exact_posterior", "probabilistic_disagreement", True,
        mech=mech, pools=("bayes_families_v1",))
put("bf_ret_fact", "bayes_families/ret_fact", "bayes_families/ret_fact", "retrieval", "evidence_retrieval", "synthetic", "rule_oracle", "verified_synthetic",
    True, mech="synthetic_fact_retrieval", pools=("bayes_families_v1",))

# ---- gap_supply_v1 (PILOT-DATA patched sdv1_gap_generators_pd; deterministic oracle families aimed at Decision Index task TYPES,
#      not DI data). M2 Bucket 3 only (not M1). synth_tool_policy is an agent tool-routing decision = a Typed-Decisions workflow
#      domain: flagged td_workflow_imitation, excluded from the strict zero-shot line (allowed in M2 by coordinator decision).
GAP = [("synth_tool_policy", "tools", "tool_call_policy", "agent_tools", ["When2Call", "BFCL", "API-Bank"], "agent_tool_routing"),
       ("synth_appliance_sata_case", "tools", "constraint_scheduling", "home_energy", ["Home Appliance Simulator", "SATA-Bench"], None),
       ("synth_causal_scm_cladder", "knowledge", "causal_inference", "synthetic", ["CLadder", "GPQA", "BBH"], None),
       ("synth_music_harmony_pop909", "arts", "chord_recognition", "music", ["POP909-CL", "high-K menus 33-52"], None),
       ("synth_pyexec_cruxeval", "knowledge", "program_execution", "software", ["CRUXEval", "GSM8K"], None)]
for fam, area, skill, dom, di, td in GAP:
    put(fam, f"gap_supply/{fam}", f"gap_supply/{fam}", area, skill, dom, "rule_oracle", "gap_supply", False,
        "M2 bucket 3 gap supply (not part of M1)", mech=f"gap/{fam}", pools=("gap_supply_v1", "gap_supply_v2"))
    A[fam]["di_target_benchmarks"] = di
    A[fam]["td_workflow_imitation"] = td
    A[fam]["m2_bucket"] = "3_di03_gap_closers_deterministic_oracles"

# ---- long_knowledge_v1 (PILOT-DATA build_long_knowledge.py): verified long-context supply (2k-8k token states, absence, retrieval
#      grounding, large menus). NOT in M1 / M2: candidate supply for M3. CUAD / MAUD underlie LegalBench test tasks (protected index);
#      TyDi QA underlies MIRACL queries; both left to the protected scan + component check.
LK = [("lk_records_ctgov", "clinicaltrials_gov", "lk_records/clinicaltrials_gov", "knowledge", "multi_record_lookup", "medicine", "rule_oracle"),
      ("lk_records_nhtsa", "osha_nhtsa", "lk_records/nhtsa_odi", "knowledge", "multi_record_lookup", "safety_regulation", "rule_oracle"),
      ("lk_records_ntsb", "osha_nhtsa", "lk_records/ntsb_aviation", "knowledge", "multi_record_lookup", "safety_regulation", "rule_oracle"),
      ("lk_records_osha_sir", "osha_nhtsa", "lk_records/osha_sir", "knowledge", "multi_record_lookup", "safety_regulation", "rule_oracle"),
      ("lk_records_fjc", "courtlistener", "lk_records/fjc_appeals", "knowledge", "multi_record_lookup", "law", "rule_oracle"),
      ("lk_records_sscdp", "courtlistener", "lk_records/sscdp", "knowledge", "multi_record_lookup", "law", "rule_oracle"),
      ("lk_cuad", "cuad", "cuad", "language", "contract_clause_presence", "law", "human_annotation"),
      ("lk_maud", "maud", "maud", "language", "merger_agreement_qa", "law", "human_annotation"),
      ("lk_musique", "musique", "musique", "knowledge", "multi_hop_answerability", "wikipedia", "human_annotation"),
      ("lk_tydi", "tydiqa", "tydiqa", "retrieval", "passage_selection_or_none", "wikipedia", "human_annotation")]
for root, src, fam, area, skill, dom, basis in LK:
    put(root, src, fam, area, skill, dom, basis, None, False, "long_knowledge_v1 supply (M3 candidate; not M1 / M2)",
        mech=f"lk/{root[3:]}", pools=("long_knowledge_v1",))
    A[root]["supply"] = "long_knowledge_v1"


if __name__ == "__main__":
    out = {"schema": "stock-decision-source-map-core-v1", "doc": __doc__.strip(), "roots": A}
    json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
    print(len(A), sum(v["m1_eligible"] for v in A.values()))
