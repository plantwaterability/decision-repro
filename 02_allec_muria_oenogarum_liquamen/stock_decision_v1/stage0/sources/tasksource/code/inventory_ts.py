#!/usr/bin/env python3
"""stock_decision_v1 Stage 0: per-source / per-upstream-dataset inventory of the converted Tasksource filtered-full pool, with a
PROPOSED canonical family (derivatives counted under their upstream family), Decision Index area (knowledge / language /
retrieval / tools / arts, DI 0.3 area semantics), skill and domain, alias candidates against our other pools, and DI benchmark
root flags. Proposals only: PILOT-DATA freezes families / areas.

  python3 inventory_ts.py UNITS.jsonl.gz TOKENS.jsonl.gz SOURCES_HIST.yaml OUT.json
"""
import collections
import gzip
import hashlib
import json
import re
import sys

import yaml

UNITS, TOKS, SY, OUT = sys.argv[1:5]

# DI 0.2.1 / 0.3 scored benchmark roots (decision_index_v021 file stems) + tasksource's own benchmark_source_holdout blocked_roots
DI_ROOTS = ["bfcl", "toolret", "apibank", "banking77", "clinc150", "clincoos", "routerbench", "homeappliances", "sgd", "sgdx",
            "schemaguideddstc8", "contractnli", "anli", "bpomp", "humicroedit", "pop909", "cfcolor", "mmlu", "mmlupro", "gpqa",
            "gpqadiamond", "arc", "arceasy", "arcchallenge", "ai2arc", "winogrande", "hellaswag", "gsm8k", "chessbench", "musr",
            "satabench", "simplebench", "bright", "esci", "acos", "finentity", "isarcasmeval", "vast", "nli4ct", "nli4ctsemeval2024",
            "cruxeval", "cladder", "hle", "forecastbench", "habermas", "phishnchips", "bbh", "bigbench", "ragtruth", "hover",
            "hover3way", "when2call", "newyorker", "blimp", "commonsenseqa", "cose", "openbookqa", "truthfulqa", "judgebench",
            "jevbench", "belebele", "financialphrasebank", "itsupporttickets", "circa", "tabfact"]
# our other pools' roots (garas v2 source_roots_v2 roots + real sources + bayes_families)
OUR_ROOTS = {"afd", "aidev", "banking77", "bayes_pref_flights", "bbq", "blimp", "casino", "circa", "clinc_oos", "code_contests",
             "commonsense_qa", "contract_nli", "cosmos_qa", "courts", "ctgov", "dmhc", "e_care", "esci", "fda", "folktables", "gdelt",
             "genomics", "go_emotions", "gsm8k", "hans", "helpsteer2", "ibm_argq", "jebadiah", "kaivoss", "lichess",
             "measuring_hate_speech", "medmcqa", "mghafiri", "miracl", "ms_marco", "nvd_cwe", "open_jev", "openreview", "physics",
             "procedural_typed_decisions", "pubmedqa", "qasper", "safety", "scruples", "social_i_qa", "stanford_politeness",
             "swe_rebench", "synth_beta_rate", "synth_binary_evidence", "synth_comparative_cost", "synth_grounded_answer",
             "synth_linked_facts", "synth_retrieval_rank", "synth_scope_discrimination", "tabfact", "vagmi", "wiki_census",
             "wikidata", "winogrande", "bayes_families_v1"}
# explicit alias findings (tasksource source prefix -> our root), checked by hand against the source lists
ALIASES = {"HelpSteer2": "helpsteer2 (curated jev_data helpsteer2: same nvidia/HelpSteer2 upstream)",
           "go_emotions": "go_emotions (curated jev_data go_emotions)", "social_i_qa": "social_i_qa (curated)",
           "cosmos_qa": "cosmos_qa (curated)", "e-CARE": "e_care (curated)", "medmcqa": "medmcqa (curated)",
           "scruples": "scruples (curated)",
           "procedural-typed-decisions": "procedural_typed_decisions (curated jev_ext_strict procedural_* stems: same generator family)"}

# canonical family: derivatives under their upstream family; key = source prefix (before first '/') unless listed
FAMILY = {"HelpSteer": "nvidia_helpsteer", "HelpSteer2": "nvidia_helpsteer", "HelpSteer3": "nvidia_helpsteer",
          "hh-rlhf": "anthropic_hh_rlhf", "babi_nli": "babi", "balanced-copa": "copa", "codah": "codah",
          "snli": "snli", "breaking_nli": "snli", "UNLI": "snli", "ethics": "hendrycks_ethics", "utilitarianism": "hendrycks_ethics",
          "robust_nli": "mnli", "robust_nli_is_sd": "mnli", "robust_nli_li_ts": "mnli", "syntactic-augmentation-nli": "mnli",
          "fever-evidence-related": "fever", "math_qa": "aqua_rat", "wiki_hop": "qangaroo_wikihop",
          "prompt-injection-dataset": "s_labs_prompt_injection", "Prompt-injection-dataset": "neuralchemy_prompt_injection",
          "cycic_classification": "cycic", "cycic_multiplechoice": "cycic", "spartqa-mchoice": "spartqa", "spartqa-yn": "spartqa",
          "procedural-typed-decisions": "procedural_typed_decisions"}
FAMILY_SUB = {"gen_debiased_nli/snli": "snli", "gen_debiased_nli/mnli": "mnli", "defeasible-nli/snli": "snli",
              "defeasible-nli/atomic": "atomic", "defeasible-nli/social": "social_chem_101",
              "multilingual/xcopa": "copa", "multilingual/xcsr/X-CODAH": "codah", "multilingual/xstory_cloze": "story_cloze",
              "multilingual/disrpt": "disrpt", "multilingual/AfriSenti": "afrisenti", "multilingual/NusaX": "nusax",
              "multilingual/masakhanews": "masakhanews", "multilingual/klue": "klue", "multilingual/offenseval_dravidian":
              "offenseval_dravidian", "multilingual/exams": "exams", "multilingual/MLMA": "mlma_hate_speech",
              "multilingual/americas_nli": "xnli_americasnli", "multilingual/massive": "massive", "multilingual/mtop": "mtop",
              "multilingual/miam": "miam", "multilingual/multilingual-sentiments": "multilingual_sentiments",
              "multilingual/tweet_sentiment_multilingual": "tweet_sentiment_multilingual", "multilingual/universal-joy":
              "universal_joy", "multilingual/x-stance": "x_stance"}
DERIVED_NOTE = {"breaking_nli": "SNLI premises", "UNLI": "u-SNLI: SNLI pairs re-annotated with probabilities",
                "gen_debiased_nli": "model-generated hypotheses on SNLI / MNLI premises", "robust_nli": "MNLI-derived stress tests",
                "robust_nli_is_sd": "MNLI-derived", "robust_nli_li_ts": "MNLI-derived", "syntactic-augmentation-nli": "MNLI augmentation",
                "defeasible-nli": "premises from SNLI / ATOMIC / Social-Chem-101", "multilingual/xcopa": "COPA translated",
                "multilingual/xcsr": "CODAH translated", "multilingual/xstory_cloze": "StoryCloze translated",
                "balanced-copa": "COPA mirrored", "math_qa": "AQuA-RAT annotated", "WANLI": "MNLI-pattern generation + human labels (new text)",
                "MSciNLI": "extends SciNLI to more domains (new text)", "fever-evidence-related": "FEVER claims / evidence",
                "vitaminc": "Wikipedia revisions; FEVER-style claims", "commonsense_qa_2.0": "CSQA2 (Talmor 2021), NOT CommonsenseQA"}


def family_of(src):
    for k in sorted(FAMILY_SUB, key=len, reverse=True):
        if src.startswith(k):
            return FAMILY_SUB[k]
    p = src.split("/")[0]
    return FAMILY.get(p, re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_"))


LANG = ("language", "NLI / entailment, commonsense plausibility, sentiment, emotion, stance, sarcasm, toxicity, discourse, linguistic"
        " judgement: DI language (ANLI, ContractNLI, NLI4CT, HellaSwag, WinoGrande, ACOS, FinEntity, iSarcasmEval, VAST, RAGTruth)")
KNOW = ("knowledge", "exam / domain QA, math, formal logic, causal / theory-of-mind / spatial reasoning: DI knowledge (MMLU-Pro, GPQA,"
        " GSM8K, BBH, MuSR, CLadder, CRUXEval, HLE)")
RETR = ("retrieval", "intent / topic routing, relevance / duplicate matching, screening (spam, injection, harm), fact verification over"
        " evidence: DI retrieval (BANKING77, CLINC150, BRIGHT, ESCI, PhishNChips, HoVer)")
TOOL = ("tools", "agent / tool-action decisions, procedural policy over records: DI tools (BFCL, ToolRet, API-Bank, When2Call)")
ARTS = ("arts", "subjective quality / taste / value judgements: DI arts (Humicroedit, New Yorker, Habermas, cfcolor)")
KNOW_SRC = ("MedQA", "medmcqa", "head_qa", "multilingual/exams", "wikimedqa", "qasc", "quartz", "CREAK", "commonsense_qa_2.0",
            "numer_sense", "cycic", "math_qa", "prm800k", "lex_glue/case_hold", "FOL-nli", "logical-entailment", "folio", "ruletaker",
            "PARARULE", "conceptrules", "corr2cause", "babi_nli", "probability_words_nli", "spartqa", "stepgame", "tomi-nli",
            "mindgames", "dgen", "proto_qa")
RETR_SRC = ("dnd_style_intents", "multilingual/massive", "multilingual/mtop", "snips_built_in_intents", "dbpedia_14",
            "multilingual/masakhanews", "multilingual/klue/ynat", "open_question_type", "hyperpartisan_news", "lex_glue/ledgar",
            "lex_glue/scotus", "lex_glue/unfair_tos", "sms_spam", "prompt-injection", "Prompt-injection", "PromptShield",
            "prompt-injections", "wildguardmix", "fever-evidence-related", "vitaminc", "fool-me-twice", "health_fact", "wice",
            "hlgd", "github-issue-similarity", "patent-phrase-similarity", "wiki_hop")
TOOL_SRC = ("agent_action_safety", "shell-safety-v2", "procedural-typed-decisions")
ARTS_SRC = ("HelpSteer", "hh-rlhf", "UltraFeedback", "scruples", "ethics/", "utilitarianism", "moral_stories", "Touche23",
            "stackoverflow-questions")


def area_of(src):
    for pref, a in ((KNOW_SRC, KNOW), (RETR_SRC, RETR), (TOOL_SRC, TOOL), (ARTS_SRC, ARTS)):
        if any(src.startswith(p) for p in pref):
            return a
    return LANG


SKILL_RULES = [("nli", "nli"), ("entail", "nli"), ("sentiment", "sentiment"), ("senti", "sentiment"), ("emo", "emotion"),
               ("joy", "emotion"), ("hate", "toxicity"), ("toxic", "toxicity"), ("offens", "toxicity"), ("civil_comments", "toxicity"),
               ("stance", "stance"), ("args_me", "stance"), ("rumoureval", "stance"), ("intent", "intent_routing"),
               ("massive", "intent_routing"), ("mtop", "intent_routing"), ("injection", "safety_screening"),
               ("PromptShield", "safety_screening"), ("wildguard", "safety_screening"), ("spam", "safety_screening"),
               ("safety", "agent_safety"), ("HelpSteer", "response_quality"), ("hh-rlhf", "preference"), ("UltraFeedback", "preference"),
               ("prm800k", "math_verification"), ("math", "math"), ("MedQA", "medical_qa"), ("medmcqa", "medical_qa"),
               ("head_qa", "medical_qa"), ("wikimedqa", "medical_qa"), ("exams", "exam_qa"), ("fever", "fact_verification"),
               ("vitaminc", "fact_verification"), ("fool-me", "fact_verification"), ("health_fact", "fact_verification"),
               ("wice", "fact_verification"), ("similar", "similarity"), ("sts", "similarity"), ("hlgd", "similarity"),
               ("disrpt", "discourse"), ("discovery", "discourse"), ("copa", "commonsense"), ("codah", "commonsense"),
               ("story_cloze", "commonsense"), ("cosmos", "reading_comprehension"), ("social_i_qa", "commonsense"),
               ("e-CARE", "causal_commonsense"), ("babi", "reasoning"), ("logic", "logic"), ("FOL", "logic"), ("folio", "logic"),
               ("rule", "logic"), ("corr2cause", "causal_reasoning"), ("spartqa", "spatial_reasoning"), ("stepgame", "spatial_reasoning"),
               ("tomi", "theory_of_mind"), ("mindgames", "theory_of_mind"), ("ethics", "moral_judgement"), ("scruples", "moral_judgement"),
               ("moral", "moral_judgement"), ("procedural", "procedural_records"), ("dbpedia", "topic_classification"),
               ("masakhanews", "topic_classification"), ("ynat", "topic_classification"), ("lex_glue", "legal_classification"),
               ("wnut", "ner"), ("docred", "relation_extraction"), ("sem_eval", "relation_extraction"), ("acceptab", "acceptability")]


def skill_of(src):
    s = src.lower()
    for k, v in SKILL_RULES:
        if k.lower() in s:
            return v
    return "classification_other"


def domain_of(src, q):
    s = (src + " " + (q or "")).lower()
    for keys, d in ((("med", "health", "sdoh", "usmle"), "medical"), (("lex_glue", "legal", "scotus", "case_hold", "ledgar", "tos"), "legal"),
                    (("financ",), "finance"), (("sci", "patent"), "science_technical"), (("tweet", "twitter", "reddit", "civil_comments",
                     "hate", "blog", "emoji", "emo2019"), "social_media"), (("news", "headline", "rumour"), "news"),
                    (("helpsteer", "hh-rlhf", "ultrafeedback", "prm800k", "injection", "promptshield", "wildguard", "agent", "shell"),
                     "llm_assistant"), (("stackoverflow", "github", "shell"), "software"), (("procedural",), "synthetic_records"),
                    (("math",), "math")):
        if any(k in s for k in keys):
            return d
    return "multilingual" if src.startswith("multilingual/") else "general"


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).casefold())


def kb(k):
    return "2" if k == 2 else "3-5" if k <= 5 else "6-16" if k <= 16 else "17-32" if k <= 32 else "33-52" if k <= 52 else ">52"


def lb(t):
    return "<=512" if t <= 512 else "513-2048" if t <= 2048 else "2049-8192" if t <= 8192 else ">8192"


sy = yaml.safe_load(open(SY))["sources"]
tok = {}
with gzip.open(TOKS, "rt") as f:
    for line in f:
        r = json.loads(line)
        tok[r["uid"]] = r["tok_r2"]
per = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
parents = collections.defaultdict(set)
lic = collections.defaultdict(collections.Counter)
q0 = {}
missing_tok = 0
with gzip.open(UNITS, "rt") as f:
    for line in f:
        r = json.loads(line)
        src = r["tasksource_source"]
        P = per[src]
        P["n"]["rows"] += 1
        P["kind"][r["kind"]] += 1
        P["variant"][r.get("variant")] += 1
        P["K"][kb(r["K"])] += 1
        P["label"][r["label_kind"]] += 1
        t = tok.get(r["uid"])
        if t is None:
            missing_tok += 1
        else:
            P["len"][lb(t)] += 1
            P["n"]["tok_r2"] += t
            if r["K"] <= 52 and t <= 8192 - 16:
                P["n"]["eligible_K52_8176"] += 1
        parents[src].add(r["parent_key"])
        lic[src][(r["license"], r.get("license_use"))] += 1
        if r.get("loader_error"):
            P["n"]["loader_error"] += 1
        if r.get("nc"):
            P["n"]["nc_reader_flag"] += 1
srcs = {}
for src, P in sorted(per.items()):
    d = sy.get(src) or {}
    ds = d.get("dataset") or (d.get("data_files_from") or [None])[0]
    names = {norm(src.split("/")[0]), norm(src)} | ({norm(ds.split("/")[-1])} if ds else set()) | {norm(x.split("/")[-1]) for x in (d.get("originals") or [])}
    di_exact = sorted(r for r in DI_ROOTS if r in names)
    di_sub = sorted(r for r in DI_ROOTS if len(r) >= 4 and any(r in n for n in names) and r not in di_exact)
    fam = family_of(src)
    area, why = area_of(src)
    pref = src.split("/")[0]
    alias = ALIASES.get(pref) or next((v for k, v in ALIASES.items() if src.startswith(k)), None)
    srcs[src] = {"rows": P["n"]["rows"], "parents_group_state": len(parents[src]), "kinds": dict(P["kind"]), "variants": dict(P["variant"]),
                 "K_buckets": dict(P["K"]), "r2_length_buckets": dict(P["len"]), "label_kind": dict(P["label"]),
                 "tok_r2_total": P["n"]["tok_r2"], "eligible_K_le_52_tok_le_8176": P["n"]["eligible_K52_8176"],
                 "loader_errors": P["n"]["loader_error"], "nc_reader_flag": P["n"]["nc_reader_flag"],
                 "licences": [{"license": a, "license_use": b, "rows": n} for (a, b), n in lic[src].most_common()],
                 "upstream_dataset": ds, "upstream_originals": d.get("originals"), "upstream_revisions": d.get("revisions"),
                 "upstream_config": d.get("config"), "generated": d.get("generated"), "card_licenses": d.get("card_licenses"),
                 "dpi_licenses": d.get("dpi_licenses"),
                 "proposed_family": fam, "derived_note": next((v for k, v in DERIVED_NOTE.items() if src.startswith(k)), None),
                 "proposed_area": area, "area_rationale": why, "proposed_skill": skill_of(src), "proposed_domain": domain_of(src, ""),
                 "alias_our_pools": alias, "di_root_exact": di_exact, "di_root_substring_review": di_sub,
                 "flags": [x for x in (["synthetic_TD_workflow_imitation (zero-shot rule: excluded from the zero-shot line)"]
                           if src.startswith("procedural-typed-decisions") else []) +
                           (["upstream dataset id not in sources.yaml 'dataset' (data_files_from / originals used)"] if not d.get("dataset") else [])]}
fams = collections.defaultdict(lambda: {"sources": [], "rows": 0, "parents": 0, "eligible": 0, "areas": collections.Counter()})
for s, v in srcs.items():
    F = fams[v["proposed_family"]]
    F["sources"].append(s)
    F["rows"] += v["rows"]
    F["parents"] += v["parents_group_state"]
    F["eligible"] += v["eligible_K_le_52_tok_le_8176"]
    F["areas"][v["proposed_area"]] += v["rows"]
ups = collections.defaultdict(lambda: {"sources": [], "rows": 0})
for s, v in srcs.items():
    U = ups[v["upstream_dataset"] or f"(unknown:{s})"]
    U["sources"].append(s)
    U["rows"] += v["rows"]
area_tot = collections.Counter()
for v in srcs.values():
    area_tot[v["proposed_area"]] += v["eligible_K_le_52_tok_le_8176"]
out = {"schema": "stock-decision-tasksource-inventory-v1", "pool": "tasksource_ff_v1", "status": "PROPOSALS (families, areas, skills,"
       " domains) for PILOT-DATA to freeze", "units_file_md5": hashlib.md5(open(UNITS, "rb").read()).hexdigest(),
       "tokens_file_md5": hashlib.md5(open(TOKS, "rb").read()).hexdigest(), "missing_token_rows": missing_tok,
       "n_sources": len(srcs), "n_proposed_families": len(fams), "n_upstream_datasets": len(ups),
       "eligible_rows_by_proposed_area": dict(area_tot),
       "alias_candidates": {s: v["alias_our_pools"] for s, v in srcs.items() if v["alias_our_pools"]},
       "di_root_exact": {s: v["di_root_exact"] for s, v in srcs.items() if v["di_root_exact"]},
       "di_root_substring_review": {s: v["di_root_substring_review"] for s, v in srcs.items() if v["di_root_substring_review"]},
       "families": {k: {**v, "areas": dict(v["areas"])} for k, v in sorted(fams.items())},
       "upstream_datasets": dict(sorted(ups.items())), "sources": srcs,
       "eligibility_rule_used_for_counts": "K <= 52 and untruncated R2 prompt tokens <= 8192 - 16 (fullft_render margin)"}
json.dump(out, open(OUT, "w"), indent=1, sort_keys=True, default=str)
print(json.dumps({k: out[k] for k in ("n_sources", "n_proposed_families", "n_upstream_datasets", "eligible_rows_by_proposed_area",
                                       "missing_token_rows", "di_root_exact", "di_root_substring_review", "alias_candidates")}, default=str))
