#!/usr/bin/env python3
"""
build_jev_dmhc.py -- P2 of research/realworld_data.md: California DMHC Independent Medical Review (IMR) determinations
as typed-decision records (jev_data schema: build_jev_data.record / validate; read back through
trainers/cache_features.iter_units). Design and its reasons: research/fda_dmhc_pipeline.md.

Source: data.ca.gov package independent-medical-review-imr-determinations-trend (CKAN resource
1e9714f7-ad88-4d73-adb3-72bd65c8aa77; the CSV is served from data.chhs.ca.gov), licence_id "cc-by". One row per IMR
case (`ReferenceID`); 42,749 rows, 2001-2026. source name: dmhc_imr -> <out>/<split>/dmhc_imr.jsonl

State (strict truncation; the outcome is written into `Findings` in ~78% of rows)
  request        diagnosis and treatment categories (DMHC's coding), review track (standard / expedited), report year
  patient        DMHC's AgeRange and PatientGender, verbatim: clinical covariates (screening and treatment windows,
                 sex-specific procedures, pregnancy care, paediatric vs adult standards). Kept and AUDITED, not blinded:
                 `check` writes fairness_audit.{json,tsv} (reversal rates by age range x gender) and `build` writes the
                 counterfactual-flip split eval_cf/ (see below).
  case_summary   `Findings` cut BEFORE the first reviewer marker (Findings:, Final Result:, Credentials/, "The physician
                 reviewer", "Two of the three ...", "Physician 2", found that, determined that, conclud...), then cut
                 again sentence by sentence: a sentence with evaluative or outcome wording (STOP) ends the summary. Where
                 the whole text is the reviewers' own writing (the 2004-06 narratives with no marker, and every 2018+
                 record wrapped as "Findings: The physician reviewer found that ..."), only request / denial sentences
                 are kept ("narrow" cut); where DMHC's own "Nature of Statutory Criteria/Case Summary" or the request
                 paragraph precedes the reviewers' findings, factual sentences are kept too ("broad" cut).
  plan_position  the summary's sentences attributed to the health plan (its denial and stated reason).
  never          Determination, Type (asked as review_basis), DaysToReview, DaysToAdopt, ReferenceID. Free text keeps
                 its own age / sex wording (`--neutralise-text` restores the old neutraliser; off by default).
  Leak controls  the design's outcome detector (OUTCOME_DETECTOR) and a stricter audit list (AUDIT_RX) must not match a
                 written state: a hit in the summary drops the record, a hit in a plan sentence drops that sentence;
                 `check` re-scans every state, runs a cross-validated bag-of-words leak probe against the base rate, and
                 writes a 200-row sample for the manual audit (summary_audit_manual_s<seed>.tsv is read back).

Questions (qids are not benchmark qids; `check` runs synth_workflows.exclusion.exclusion_errors)
  denial_reversed_on_review  noul      Independent medical review reverses the health plan's denial.
  panel_split                choice 4  experimental / investigational cases with a parsed 3-reviewer vote:
                                       all_three_reverse / all_three_uphold / two_reverse / two_uphold
  review_basis               choice 3  experimental_investigational / medical_necessity / urgent_care_reimbursement
  evidence_strength_for_request, harm_if_delayed   score 4   TEACHER (T) fill, written unlabelled to
                                       <out>/teacher_fill/{train,val}/ for the (48) labeller; never for eval.

Labels (meta.label_source / meta.label_method per question)
  denial_reversed_on_review
    votes:n   experimental / investigational cases whose Findings give the reviewers' split (c of n reviewers would
              reverse): g = (c + a m) / (n + a), the Beta-binomial posterior mean of a reviewer's reversal propensity.
              m = hierarchical leave-one-case-out per-reviewer reversal rate (global > treatment category > treatment
              subcategory > subcategory x report year), fitted on train; a by Beta-binomial marginal likelihood on
              train. Used in every split (eval too: the soft gold is the smoothed share, a and m from train only).
    outcome   every other case: 0.5 * onehot(Determination) + 0.5 * p_hat, p_hat a hierarchical leave-one-case-out
              case-level reversal rate (global > Type > Type x treatment category > Type x treatment subcategory >
              ... x report year), train-only fit, prior strengths by LOO log score. eval: one-hot.
  panel_split   votes:3  Beta-binomial posterior predictive for a fresh 3-reviewer panel on the same case given the
              realised c of 3 (same a, m). The realised split itself is in meta.raw_outcome.
  review_basis  expert (DMHC `Type`), one-hot.

Splits (group = ReferenceID; identical duplicate rows collapse). ReportYear has no month, so the decision time is
estimated from the ReferenceID sequence (a running case counter, contiguous across report years): within a year the
case's rank among that year's sequence numbers is mapped to a fraction of the year.
  train  ReportYear <= 2024     val  2025, within-year rank < 0.40 (est. Jan-May 2025)
  eval   decided after 2025-06: ReportYear 2026, or 2025 with rank >= 0.60 (est. Aug-Dec 2025)
  2025 rows with rank 0.40-0.60 are a buffer around the 2025-06-30 cut and are not written.
  eval_cf  counterfactual-flip copies of eligible eval cases, for measuring prediction shift (never scored here):
           per case an `original`, a `gender_swap` (Female <-> Male) and an `age_shift` (one AgeRange bin, direction by
           hash, clamped at the ends) record, sharing meta.cf_pair_id (= base_id). Eligible: gender Female / Male, age range
           recorded, no SEX_AGE_SPECIFIC_RX match in the categories or free text, and no explicit age / sex / pronoun /
           parent wording in the free text (CF_TEXT_ATTR_RX), so the swap cannot contradict the text. Labels are the
           original's; the counterfactual outcome is unknown.

Usage (stdlib + numpy + scipy; laptop is fine: the CSV is 85 MB)
  python3 build_jev_dmhc.py fetch --cache /tmp/dmhc_jev/cache
  python3 build_jev_dmhc.py build --cache /tmp/dmhc_jev/cache --out /tmp/dmhc_jev
  python3 build_jev_dmhc.py check --out /tmp/dmhc_jev
  python3 build_jev_dmhc.py merge-fill --out /tmp/dmhc_jev --fill-labelled <labeller output dir>
  python3 build_jev_dmhc.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_jev_data import choice_q, noul_q, option_keys, record, score_q, sha, validate  # noqa: E402
from build_jev_fda import (  # noqa: E402  shared label / check machinery
    blend, calib_report, clip, dedup_report, encode_levels, file_sha256, fill_record, fit_kappas, group_out_rates,
    hkey, http_get, jdump, lab_dict, label_stats, leak_probe, log, merge_fill, onehot_dict, read_corpus, readback,
    run_exclusion, write_fill)

csv.field_size_limit(1 << 30)

PACKAGE_ID = "independent-medical-review-imr-determinations-trend"
RESOURCE_ID = "1e9714f7-ad88-4d73-adb3-72bd65c8aa77"
PACKAGE_SHOW = f"https://data.ca.gov/api/3/action/package_show?id={PACKAGE_ID}"
CHHS_PACKAGE_SHOW = "https://data.chhs.ca.gov/api/3/action/package_show?id=b79b3447-4c10-4ae6-84e2-1076f83bb24e"
LICENSE = ("California DMHC, IMR Determinations Trend, CC BY (data.ca.gov package metadata license_id 'cc-by', "
           "'Creative Commons Attribution', version not stated). The CSV is served by the CalHHS Open Data Portal "
           "(data.chhs.ca.gov), which states no licence and applies its Terms of Use (non-exclusive, revocable licence to "
           "use and distribute; attribution and citation required; modified data must be marked as modified). "
           "Attribution / citation: California Department of Managed Health Care, 'DMHC IMR Data, 2001 - Current', "
           "https://data.ca.gov/dataset/independent-medical-review-imr-determinations-trend (metadata modified "
           "2026-09-28). MODIFIED, not official government data: Findings truncated before the reviewers' conclusions; "
           "fields dropped (Determination, Type, DaysToReview, DaysToAdopt).")
LAMBDA = 0.5
TRAIN_MAX_YEAR, SPLIT_YEAR = 2024, 2025
VAL_RANK_MAX, EVAL_RANK_MIN = 0.40, 0.60
SOURCE = "dmhc_imr"

# --------------------------------------------------------------------------- text normalisation and the cutter
def normalize(t):
    t = html.unescape(t or "").replace("‑", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("\xa0", " ").replace("¿", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # markers glued to the previous sentence: "patientFinal Result:" -> "patient. Final Result:"
    t = re.sub(r"(?<=[a-z0-9)\]])\s*(?=(?:Final Result|Findings|Credentials/Qualifications|Nature of Statutory Criteria)"
               r"\s*:)", ". ", t)
    # a full stop glued to the next sentence ("thigh).PET/CT is ...", "Tymlos.A review ..."; third manual audit)
    t = re.sub(r"(?<=[a-z0-9)\]])\.(?=[A-Z])", ". ", t)
    return t


LEAD = re.compile(r"^\W*(?:DO NOT POST[^.]*\.\s*)?(?:Standard Review|Expedited Review|Gender\s*:\s*[MF](?=[A-Z]))?\W*",
                  re.I)
WRAP = re.compile(r"^\W*Findings\s*:\s*(?:The|Two of the three|Three of the three|All three)?\s*(?:physician |medical "
                  r"professional )?reviewers?\s+(?:found|determined)\s+that\s*", re.I)
NSC = re.compile(r"Nature of Statutory Criteria\s*/\s*Case Summary\s*:?\s*", re.I)
PHYS1 = re.compile(r"^\W*Physician\s*#?\s*1\s*:?\s*", re.I)
END = re.compile(
    r"\bFindings\s*:|\bFinal Result\s*:|\bCredentials\s*/|\bPhysician\s*#?\s*[2-9]\b|\bReviewer\s*#?\s*\d|"
    r"\b(?:The|A|One|Two|Three|All three|Both)(?: (?:of|out of) the (?:three|3))?(?: physician| medical professional)? "
    r"reviewers?\b|\bfound that\b|\bdetermined that\b|\bconclud\w*|\bI have determined\b|\bIn my (?:opinion|judgment)\b",
    re.I)
# evaluative / reviewer-voice wording; the first build's leak probe surfaced "The submitted documentation supports the
# medical necessity of ..." (top bag-of-words feature for reversal), so these are in STOP and AUDIT_RX
EVALUATIVE = ("|medical necessity|\\b(?:supports?|supported|supporting|supportive) (?:the|a|an|this|its|their|use|medical|that|for the)\\b|\\b(?:does|do|did) not support\\b|\\bunsupported\\b|\\b(?:documentation|records?|evidence|information|notes?) (?:\\w+ ){0,2}(?:fails?|failed) to\\b|\\bfails? to (?:demonstrate|document|establish|show|support|meet)\\b|\\bshould\\b|last resort|\\bnot recommended\\b|\\brecommended against\\b|\\bexperts?\\b|\\bauthors?\\b|\\bresearchers?\\b|\\bcolleagues\\b|\\bAmerican (?:College|Academy|Society|Association)\\b|\\bNCCN\\b|\\bpeer[- ]reviewed\\b|\\btrials?\\b|\\brandomi[sz]ed\\b|meta-analys|\\binsufficient\\b|\\bno (?:clinical )?(?:evidence|indication|documentation)\\b"
              # added after the first 200-row manual audit: guideline / approval context kept after a request
              r"|\b(?:is|are|was|were) recommended\b|\bapproved (?:for|by)\b|\bFDA[- ]approved\b|\bindicated (?:for|in)\b|"
              r"\bappropriate patients\b|\bpharmacotherapy\b"
              # added after the second manual audit: guideline bodies and evaluative descriptions
              r"|\brecommendations?\b|\brecommends\b|\brecommending\b|\b(?:Society|Association|College|Academy|"
              r"Institute|Committee|Council|Centers for Disease)\b|\b(?:CDC|ACIP|AHS|ACOG|ACR|USPSTF|NICE)\b|\bhighly\b|"
              r"\bprecise\b|\bminimi[sz]e\w*|\badvantage\w*|\bsafer\b|\bsuperior\b|\bbetter than\b")
STOP = re.compile(
    r"overturn|uphold|upheld|medically necessary|likely to (?:be|have been) more beneficial|more beneficial|should be|"
    r"\b(?:is|are|was|were) (?:not )?(?:medically )?(?:indicated|appropriate|warranted|supported|justified|reasonable)\b|"
    r"standard (?:of care|therapy|treatment)|literature|\bstud(?:y|ies)\b|evidence|guideline|research|according to|"
    r"\bcorrect\b|\bincorrect\b|\bagree|\bdisagree|\btherefore\b|\bthus\b|\bconsequently\b|\bhowever\b|\balthough\b|"
    r"\bcriteria\b|\bmeets?\b|\bnot supported\b|efficacy|effective|safety|\bproven\b|\bunproven\b|\bbenefi"
    + EVALUATIVE, re.I)
# what the PLAN asserts in its denial; allowed only inside a sentence attributed to the plan
PLAN_CLAIM = re.compile(r"\b(?:is |are |was |were )?not medically necessary\b|\bmedical necessity\b|\bexperimental\b|"
                        r"\binvestigational\b|\bnot (?:a )?covered\b|\bcriteria\b|\b(?:does|did) not meet\b|\bnot meet\b",
                        re.I)
PLAN_SENT = re.compile(
    r"\b(?:health plan|the plan|plan's)\b.{0,160}\b(?:den(?:ied|ies|ial|y)|declin\w*|modif\w*|delay\w*|deem\w*|"
    r"consider\w*|determin\w*|stat\w*|indicat\w*|report\w*|found|asserts?|contends?|maintains?)\b|"
    r"\b(?:den(?:ied|ial)|declin\w*)\b.{0,80}\b(?:health plan|the plan)\b", re.I)
# "recommend" is NOT a narrow-mode request word: in reviewer-written text it introduced guideline sentences (CDC / ACIP
# recommendations, "Pharmacotherapy is recommended ...") in the second manual audit; broad mode still keeps it via FACT
NARROW = re.compile(r"\b(?:request\w*|seek\w*|sought|reimburse\w*|authoriz\w*|coverage|den(?:ied|ial|ies|y)|declined|"
                    r"health plan|the plan)\b", re.I)
FACT = re.compile(
    r"\b(?:request\w*|seek\w*|sought|reimburse\w*|authoriz\w*|coverage|cover|den(?:ied|ial|ies|y)|declin\w*|modif\w*|"
    r"health plan|the plan|history|diagnos\w*|presents?|presented|underw\w*|status post|prescrib\w*|recommend\w*|"
    r"(?:the|this|an?) (?:patient|enrollee|member|child|infant) (?:is|was|has|had)|suffer\w*|treated|treatment of|"
    r"considers?|experimental|investigational)\b", re.I)
AT_ISSUE = re.compile(r"^(?:At issue|The issue|The question)\b[^.]{0,40}\bwhether\b", re.I)

# the design's outcome detector, verbatim (realworld_data.md P2)
OUTCOME_DETECTOR = re.compile(r"overturn|uphold|upheld|is medically necessary|is not medically necessary|"
                              r"likely to be more beneficial|should be covered", re.I)
# a stricter audit list: verdict words, reviewer attributions and evaluative phrases
AUDIT_RX = re.compile(
    r"overturn\w*|uphold\w*|upheld|revers\w*|\bmedically necessary\b|more beneficial|should (?:be|have been)\b|"
    r"\bmust (?:cover|authorize|pay|provide|reimburse)\b|in (?:the )?favou?r of|denial (?:stands|is|was) "
    r"(?:correct|appropriate|wrong|incorrect)|\breviewers?\b|found that|determined that|final result|\bfindings?\s*:|"
    r"credentials|board[- ]certified|two of the three|all three|three of the three|majority|\bwe find\b|\bI have\b|"
    r"\bis (?:not )?(?:appropriate|indicated|warranted|justified)\b|\bnot supported\b" + EVALUATIVE, re.I)


def sentences(t):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=["(A-Z])', t) if s.strip()]


def plan_ok(s):
    """A plan-attributed sentence may state the plan's own claim; nothing else evaluative or verdict-like."""
    rest = PLAN_CLAIM.sub(" ", s)
    return not STOP.search(rest) and not AUDIT_RX.search(rest) and not OUTCOME_DETECTOR.search(s)


def cut_findings(findings, max_scan=12):
    """Strict truncation of an IMR `Findings` text -> dict(fmt, mode, summary [sentences], plan [sentences], why)."""
    t = LEAD.sub("", normalize(findings))
    wrapped = False
    w = WRAP.match(t)
    if w:
        t, wrapped = t[w.end():], True
    fmt = "plain"
    m = NSC.search(t)
    if m:
        if END.search(t[:m.start()]):
            wrapped = True
        t, fmt = t[m.end():], "nsc"
    else:
        m = PHYS1.search(t)
        if m:
            t, fmt = t[m.end():], "physician_n"
    e = END.search(t)
    closed = bool(e and re.match(r"Findings\s*:", t[e.start():], re.I))
    sec = t[:e.start()] if e else t
    broad = (fmt == "physician_n") or (closed and not wrapped)
    kept, why = [], "end_of_section"
    for i, s in enumerate(sentences(sec)):
        if i >= max_scan:
            why = "scan_limit"
            break
        if AT_ISSUE.search(s):              # restates the question under review; never kept, never ends the summary
            continue
        is_plan = bool(PLAN_SENT.search(s))
        if broad:
            if not (plan_ok(s) if is_plan else not STOP.search(s)):
                why = "stop_word"
                break
            if not (FACT.search(s) or NARROW.search(s)):
                why = "not_fact"
                break
            kept.append((s, is_plan))
        else:
            if NARROW.search(s):
                if not (plan_ok(s) if is_plan else not STOP.search(s)):
                    why = "stop_word"
                    break
                kept.append((s, is_plan))
            elif kept:
                why = "after_request"
                break
            elif STOP.search(s) and not is_plan:
                why = "stop_before_request"
                break
    summ = [s for s, p in kept if not p]
    plan = [s for s, p in kept if p]
    if summ and summ[0][:1].islower():
        summ[0] = summ[0][0].upper() + summ[0][1:]
    return {"fmt": fmt + ("/wrapped" if wrapped else ""), "mode": "broad" if broad else "narrow",
            "summary": summ, "plan": plan, "why": why}


# --------------------------------------------------------------------------- protected attributes (age, sex)
NUMW = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty|thirty|forty|fourty|fifty|sixty|seventy|eighty|ninety|hundred)")
AGE_RX = re.compile(
    r"\b(?:\d{1,3}(?:\.\d)?|" + NUMW + r"(?:[- ](?:" + NUMW + r"|and|a|half)){0,3})(?:[- ]{1,2}(?:year|month|week|day)s?)+[- ]{1,2}old\b|"
    r",?\s*\b(?:aged?|age)\s+\d{1,3}(?:\s*(?:years|yrs)(?: old)?)?\b|\b\d{1,3}\s*(?:years|yrs)\s+(?:of age|old)\b|"
    r"\bin (?:his|her|their) (?:early |mid-?|late )?(?:teens|twenties|thirties|forties|fifties|sixties|seventies|"
    r"eighties|nineties|\d0s)\b|"
    # truncated forms seen in the audits: "a then 1-year- enrollee", "a 3-year, enrollee"
    r"\b\d{1,3}[- ](?:year|month|week|day)s?\b[,-]?(?=\s+(?:old\s+)?(?:male\s+|female\s+)?(?:enrollee|patient|member|child|"
    r"infant|boy|girl|man|woman|person)\b)", re.I)
PERSON_NOUN = (r"(?:patient|enrollee|member|child|infant|adolescent|teen\w*|toddler|newborn|baby|student|veteran|"
               r"individual|person|minor|adult|dependent|subscriber|beneficiary)")
SEX_ADJ_RX = re.compile(r"\b(?:male|female)\s+(?=" + PERSON_NOUN + r"\b)", re.I)
SEX_NOUN_RX = re.compile(r"\b(a|an|the|this|that)\s+(?:male|female)\b(?!\s+(?:external|pattern|infertility|genital|"
                         r"reproductive|breast|pelvic|condom|catheter|urinary|hormone|to|-))", re.I)
PERSON_SEX = [(re.compile(r"\b(?:man|woman|gentleman|lady)\b", re.I), "person"),
              (re.compile(r"\b(?:men|women)\b", re.I), "people"),
              (re.compile(r"\b(?:boy|girl)\b", re.I), "child"), (re.compile(r"\b(?:boys|girls)\b", re.I), "children"),
              (re.compile(r"\b(?:mother|father)\b", re.I), "parent"), (re.compile(r"\b(?:mothers|fathers)\b", re.I), "parents"),
              (re.compile(r"\b(?:son|daughter)\b", re.I), "child"), (re.compile(r"\b(?:sons|daughters)\b", re.I), "children"),
              (re.compile(r"\b(?:wife|husband)\b", re.I), "spouse"),
              (re.compile(r"\b(?:brother|sister)\b", re.I), "sibling"),
              (re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?\s+(?=[A-Z])"), "")]
_PRON = {"he": "they", "she": "they", "him": "them", "his": "their", "hers": "theirs", "himself": "themself",
         "herself": "themself"}
PRON_RX = re.compile(r"\b(he|she|him|his|hers|himself|herself)\b", re.I)
HER_RX = re.compile(r"\bher\b(\s*)(\S*)", re.I)
HER_OBJ_NEXT = {"to", "and", "or", "with", "for", "in", "on", "at", "by", "from", "that", "as", "about", "after", "before",
                "because", "if", "when", "since", "but", "so", "than", "into", "until", "while", ""}
AGREE = [(re.compile(r"\b(they)\s+has\b", re.I), r"\1 have"), (re.compile(r"\b(they)\s+is\b", re.I), r"\1 are"),
         (re.compile(r"\b(they)\s+was\b", re.I), r"\1 were"), (re.compile(r"\b(they)\s+does\b", re.I), r"\1 do")]
AGE_BINS = ["0 to 10", "11 to 20", "21 to 30", "31 to 40", "41 to 50", "51 to 64", "65+"]
GENDER_SWAP = {"Female": "Male", "Male": "Female"}
# Conservative list for the counterfactual flip: a case is NOT eligible when its diagnosis / treatment categories or its
# free text match any of these. Sex- or reproduction-specific anatomy, care and drugs; gender identity; and age-specific
# care (paediatric, developmental, growth hormone, infant RSV, immunisation, geriatric conditions), plus osteoporosis
# (sex- and age-skewed) and its drugs. Built from DMHC's own category vocabulary (1,125 values, read 2026-10-01).
SEX_AGE_SPECIFIC_RX = re.compile(
    r"\b(?:male|female|men|women|man|woman|boys?|girls?|gender\w*|transgender|trans|feminiz\w*|masculiniz\w*|"
    r"phalloplasty|vaginoplasty|breasts?|mammo\w*|mastect\w*|mastopexy|gynecomast\w*|prostat\w*|psa|testic\w*|"
    r"testost\w*|erectile|impotence|penile|penis|scrot\w*|vasect\w*|circumcis\w*|pregnan\w*|prenatal|perinatal|"
    r"post-?partum|obstet\w*|ob-?gyn|gyn\w*|matern\w*|midwife|l&d|labor & delivery|cesarean|miscarriage|abortion|"
    r"birth|childbirth|fertil\w*|infertil\w*|ivf|art reprod|embryo\w*|oocyte\w*|sperm|uter\w*|fibroids?|leimyomata|"
    r"ovar\w*|ovary|pcos|cervix|pap smear|vagin\w*|vulv\w*|endometri\w*|menstru\w*|menorrhagia|dysmenorrhea|"
    r"menopaus\w*|peri-menopausal|hysterect\w*|d & c|tubal ligation|contracept\w*|estrogen|estradiol|progest\w*|"
    r"lactat\w*|pelvic|repro\w*|gestational|fem genital|hormones?|brca\w*|hpv|osteoporo\w*|osteopen\w*|prolia|"
    r"evenity|forteo|tymlos|teriparatide|abaloparatide|denosumab|romosozumab|reclast|alendronate|veozah|addyi|"
    r"intrarosa|orilissa|myfembree|oriahnn|makena|nexplanon|mirena|viagra|cialis|levitra|sildenafil|tadalafil|"
    r"androderm|androgel|xyosted|jatenzo|"
    r"pediatric\w*|paediatric\w*|child\w*|infant\w*|newborn\w*|neonat\w*|nicu|toddler\w*|adolescen\w*|juvenile|"
    r"teen\w*|youth|school|premature|failure to thrive|short stature|growth horm\w*|somatropin|genotropin|"
    r"norditropin|omnitrope|saizen|serostim|nutropin|humatrope|zomacton|skytrofa|sogroya|ngenla|increlex|puberty|"
    r"precocious|autis\w*|asperger\w*|perv develop|pervasive develop\w*|develop\w* delay|delayed speech|speech delay|"
    r"appld behav|applied behav\w*|adaptive behav|aba|floor-time|play therapy|social skills|sensory integration|"
    r"cleft|congenital|down'?s syndrome|rett'?s?|prader-willi|charge syndrome|hunter syndrome|pku|phenylketonuria|"
    r"cranial helmet|tongue tie|tubes in ears|rsv|synagis|beyfortus|palivizumab|nirsevimab|immuni[sz]\w*|vaccin\w*|"
    r"adhd|add|attention deficit|oppositional defiant|attachment disorder|tourette\w*|serious emot\w*|sev emot\w*|"
    r"adult|geriatric|elder\w*|dementia|alzheimer\w*|macular degeneration|cataracts?|medicare)\b", re.I)
# explicit age / sex / relationship wording about the patient in the free text: a swapped field would contradict it
CF_TEXT_ATTR_RX = re.compile(r"\b(?:\d{1,3}|[a-z]+)[- ]{1,2}(?:years?|months?|weeks?|days?)[- ]{1,2}old\b|\baged? \d|"
                             r"\b(?:he|she|him|his|her|hers|himself|herself|male|female|mother|father|son|daughter|wife|"
                             r"husband|parents?|guardian|conservator|minor|dependent)\b", re.I)
RESIDUAL_RX = re.compile(r"\b(?:\d{1,3}|[a-z]+)[- ]years?[- ]old\b|\b(?:he|she|him|his|her|hers|himself|herself|man|woman|"
                         r"boy|girl|mother|father|son|daughter|wife|husband)\b|\b(?:a|an|the|this) (?:male|female)\b", re.I)


def _case(src, rep):
    return rep[:1].upper() + rep[1:] if src[:1].isupper() else rep


def neutralise(text):
    """Remove ages and sex descriptors of the person, and make pronouns neutral. Returns (text, n_edits).
    Clinical terms that imply sex (prostate, mammogram, 'female infertility', product names) are left as written."""
    n = 0

    def sub(rx, rep, t):
        nonlocal n
        t2, k = rx.subn(rep, t)
        n += k
        return t2
    t = sub(AGE_RX, "", text)
    t = sub(SEX_ADJ_RX, "", t)
    t = sub(SEX_NOUN_RX, r"\1 patient", t)
    for rx, rep in PERSON_SEX:
        t = sub(rx, rep, t)

    def pron(m):
        nonlocal n
        n += 1
        return _case(m.group(1), _PRON[m.group(1).lower()])
    t = PRON_RX.sub(pron, t)

    def her(m):
        nonlocal n
        n += 1
        nxt = re.sub(r"\W", "", m.group(2)).lower()
        rep = "them" if (nxt in HER_OBJ_NEXT or not m.group(1) or re.match(r"[.,;:)]", m.group(2) or "")) else "their"
        return _case(m.group(0), rep) + m.group(1) + m.group(2)
    t = HER_RX.sub(her, t)
    for rx, rep in AGREE:
        t = rx.sub(rep, t)
    # articles and spacing after removals ("a  enrollee" -> "an enrollee", "The parent of a , enrollee" -> ...)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r"\b(a|an|A|An)\s+(?:,\s*)?(?=\w)", lambda m: m.group(0), t)
    t = re.sub(r"\b([Aa])n?\s+(?=([aeiouAEIOU]))", lambda m: m.group(1) + "n ", t)
    t = re.sub(r"\b([Aa])n\s+(?=[^aeiouAEIOU\W])", lambda m: m.group(1) + " ", t)
    t = re.sub(r"\bThe patient is a patient (whose|who)\b", r"The patient, \1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t, n


# --------------------------------------------------------------------------- reviewer votes
COUNT_RX = re.compile(
    r"\b(?P<k>one|two|three|all three|all 3|both)(?: (?:of|out of) (?:the )?(?P<n>three|3|two|2))?"
    r"(?: (?:physician|medical professional|independent))? reviewers?\b(?= (?:found|determined|concluded|agreed|felt|"
    r"noted|opined|stated|believe\w*|indicated|recommend\w*|also))", re.I)
DIR_UPHOLD = re.compile(r"not (?:likely|expected) to (?:be|have been) (?:more beneficial|superior|of greater benefit)|"
                        r"\b(?:is|are|was|were) not medically necessary|should be upheld|not likely to be superior", re.I)
DIR_REVERSE = re.compile(r"(?<!not )\b(?:is|are|was|were|would be) (?:more )?likely to (?:be|have been) "
                         r"(?:more beneficial|superior|of greater benefit)|(?<!not )\b(?:is|are|was|were) medically "
                         r"necessary|should be overturned", re.I)
# hedged reasoning ("fails to demonstrate the superior efficacy", "lack of evidence ... likely to be more beneficial")
HEDGE = re.compile(r"lack of|\blacks\b|\bfail\w*|ha(?:s|ve) not been|not been (?:shown|established|demonstrated)|"
                   r"insufficient|\bno (?:evidence|data)|yet to be|not established|unproven|do(?:es)? not support|"
                   r"although|whether", re.I)
KWORD = {"one": 1, "two": 2, "three": 3, "all three": 3, "all 3": 3, "both": 2}


def direction(s):
    """'uphold' / 'reverse' from a conclusion-style sentence, None when the sentence states no clear conclusion."""
    if DIR_UPHOLD.search(s):
        return "uphold"
    if DIR_REVERSE.search(s) and not HEDGE.search(s):
        return "reverse"
    return None


def panel_size(findings):
    i = findings.find("Credentials")
    cred = findings[i:] if i >= 0 else ""
    if not cred:
        return None
    if re.search(r"\b(?:all three|three (?:physician |medical professional )?reviewers|reviewers? 1,? 2,? and 3|"
                 r"reviewer\s*#?\s*3|third reviewer)\b", cred, re.I):
        return 3
    if re.search(r"\btwo (?:physician |medical professional )?reviewers (?:are|is)\b", cred, re.I) and \
            re.search(r"\bone (?:physician |medical professional )?reviewer (?:is|are)\b", cred, re.I):
        return 3
    if re.search(r"\bboth (?:physician |medical professional )?reviewers\b", cred, re.I) or \
            (re.search(r"\breviewers? 1 and 2\b", cred, re.I) and not re.search(r"\b3\b", cred)):
        return 2
    return None


def parse_votes(findings, overturned):
    """(n, c, how) with c = reviewers for reversing the denial, or (None, None, why-not). The text is normalised, cut
    before the credentials section (whose 'All three reviewers are board certified' is not a vote), and every count
    statement must agree with the determination; per-physician opinions (2004-06) are counted one by one."""
    t = normalize(findings)
    i = t.find("Credentials")
    body = t[:i] if i >= 0 else t
    secs = re.split(r"\bPhysician\s*#?\s*([1-9])\s*:?", body)
    if len(secs) >= 7:                       # "Physician 1: ... Physician 2: ... Physician 3: ..."
        votes = []
        for j in range(1, len(secs) - 1, 2):
            sec = secs[j + 1]
            tail = " ".join(sentences(sec)[-3:])
            d = direction(tail)
            if d is None:
                return None, None, "per_physician_direction_unclear"
            votes.append(d)
        n, c = len(votes), sum(1 for v in votes if v == "reverse")
        if n != 3:
            return None, None, f"per_physician_n={n}"
        if (c >= 2) != overturned:
            return None, None, "per_physician_majority_contradicts_determination"
        return n, c, "per_physician_opinions"
    stmts = []
    for m in COUNT_RX.finditer(body):
        k = KWORD[m.group("k").lower()]
        nn = m.group("n")
        n_stated = {"three": 3, "3": 3, "two": 2, "2": 2}.get((nn or "").lower())
        end = body.find(".", m.end())
        s = body[m.start(): end if end > 0 else len(body)]
        stmts.append((k, n_stated, direction(s), s))
    if not stmts:
        return None, None, "no_count_statement"
    n = panel_size(t) or 3
    if any(ns == 2 for _, ns, _, _ in stmts) or any(k == 2 and "both" in s.lower()[:6] for k, _, _, s in stmts):
        n = 2
    maj = "reverse" if overturned else "uphold"
    ks = set()
    for k, _, d, s in stmts:
        if k == 1:                           # a minority opinion: it must oppose the determination
            if d is not None and d == maj:
                return None, None, "minority_statement_agrees_with_determination"
            ks.add("split")
        elif k == n:
            if d is not None and d != maj:
                return None, None, "unanimous_statement_contradicts_determination"
            ks.add("unanimous")
        elif k == 2 and n == 3:
            if d is not None and d != maj:
                return None, None, "majority_statement_contradicts_determination"
            ks.add("split")
        else:
            return None, None, f"count_{k}_of_{n}"
    if ks == {"split"}:
        if n != 3:
            return None, None, "split_on_two_member_panel"
        return 3, (2 if overturned else 1), "count_statements_split"
    if ks == {"unanimous"}:
        return n, (n if overturned else 0), "count_statements_unanimous"
    return None, None, "split_and_unanimous_statements"


# --------------------------------------------------------------------------- questions
REVERSE_INSTR = "Independent medical review reverses the health plan's denial."
REVERSE_CRITERIA = {"yes": "The reviewers find the requested service medically necessary, or likely to be more beneficial "
                           "than standard care, and the plan must cover it.",
                    "no": "The reviewers agree with the plan, and its denial stands."}
SPLIT_KEYS = ["all_three_reverse", "all_three_uphold", "two_reverse", "two_uphold"]
SPLIT_CRITERIA = {
    "all_three_reverse": "All three physician reviewers find in the patient's favour and would reverse the denial.",
    "all_three_uphold": "All three physician reviewers agree with the plan and would let the denial stand.",
    "two_reverse": "Two of the three reviewers would reverse the denial and one would let it stand.",
    "two_uphold": "Two of the three reviewers would let the denial stand and one would reverse it.",
}
SPLIT_INSTR = ("Three physician reviewers each judge this request independently. How will their opinions divide?")
SPLIT_OF_C = {0: "all_three_uphold", 1: "two_uphold", 2: "two_reverse", 3: "all_three_reverse"}
BASIS_KEYS = ["experimental_investigational", "medical_necessity", "urgent_care_reimbursement"]
BASIS_CRITERIA = {
    "experimental_investigational": "Experimental or investigational: the plan denied the treatment as experimental or "
                                    "investigational for the patient's condition.",
    "medical_necessity": "Medical necessity: the plan denied, delayed or modified the service as not medically necessary "
                         "for the patient's condition.",
    "urgent_care_reimbursement": "Urgent or emergency care: the plan refused to pay for urgent or emergency care the "
                                 "patient received from a provider outside the plan.",
}
BASIS_OF = {"Experimental/Investigational": "experimental_investigational", "Medical Necessity": "medical_necessity",
            "Urgent Care": "urgent_care_reimbursement"}
BASIS_INSTR = "On what grounds is this denial being reviewed?"
EVID_LEVELS = ["weak: standard guidelines do not support the requested service for this situation",
               "mixed: guidelines or studies disagree, or support it only for other situations",
               "moderate: guidelines support it for situations like this one, with conditions or caveats",
               "strong: standard guidelines clearly recommend it for this situation"]
EVID_INSTR = "How strongly do standard clinical guidelines support the requested service for this patient?"
HARM_LEVELS = ["none: delaying or forgoing the service would not affect the patient's health",
               "minor: delay would cause discomfort or inconvenience but no lasting harm",
               "significant: delay could worsen the condition or cause harm that is hard to reverse",
               "serious: delay could threaten the patient's life or cause severe, lasting harm"]
HARM_INSTR = "How much harm could the patient suffer if the requested service is delayed or not provided?"


# --------------------------------------------------------------------------- fetch
def fetch(a):
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    pkg = json.loads(http_get(PACKAGE_SHOW))["result"]
    (cache / "package_show.json").write_text(json.dumps({"result": pkg}, indent=1))
    res = next(r for r in pkg["resources"] if r["id"] == RESOURCE_ID)
    chhs = None
    try:
        chhs = json.loads(http_get(CHHS_PACKAGE_SHOW))["result"]
        (cache / "chhs_package_show.json").write_text(json.dumps({"result": chhs}, indent=1))
    except Exception as e:  # noqa: BLE001
        log(f"CHHS package_show failed ({type(e).__name__}); data.ca.gov metadata is the licence record")
    dest = cache / "imr.csv"           # http_get writes <dest>.part and renames, so a failed download keeps the old file
    log(f"download {res['url']}")
    http_get(res["url"], dest)
    dd = next((r for r in pkg["resources"] if r.get("format") == "PDF"), None)
    if dd:
        http_get(dd["url"], cache / "imrdatadictionary.pdf")
    with open(dest, encoding="utf-8-sig", newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    prov = {"fetched": datetime.now().isoformat(timespec="seconds"), "package_show": PACKAGE_SHOW,
            "license_id": pkg.get("license_id"), "license_title": pkg.get("license_title"),
            "license_url": pkg.get("license_url"), "metadata_modified": pkg.get("metadata_modified"),
            "organization": (pkg.get("organization") or {}).get("title"), "resource_id": RESOURCE_ID,
            "csv_url": res["url"], "csv_sha256": file_sha256(dest), "csv_rows": n,
            "chhs_license_id": (chhs or {}).get("license_id"), "chhs_metadata_modified": (chhs or {}).get("metadata_modified")}
    (cache / "fetch_provenance.json").write_text(json.dumps(prov, indent=1))
    log(f"IMR: {n:,} rows; licence {prov['license_id']} ({prov['license_title']})")


# --------------------------------------------------------------------------- label models
def binom_group_out_rate(level_ids, c, n, fit_mask, kappas):
    """Hierarchical per-reviewer rate m (c of n reviewers per case), leave-one-case-out on the fit set:
    m_L = (C_L - c_i + kappa_L m_{L-1}) / (T_L - n_i + kappa_L); cases outside the fit set get fit-set rates."""
    import numpy as np
    cf, nf = np.where(fit_mask, c, 0.0), np.where(fit_mask, n, 0.0)
    m = np.full(len(c), 0.5)
    for L, ids in enumerate(level_ids):
        C = np.bincount(ids, weights=cf, minlength=int(ids.max()) + 1)
        T = np.bincount(ids, weights=nf, minlength=int(ids.max()) + 1)
        m = (C[ids] - cf + kappas[L] * m) / (T[ids] - nf + kappas[L])
    return np.clip(m, 1e-6, 1 - 1e-6)


def fit_binom_kappas(level_ids, c, n, fit_mask, grid=(2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000)):
    import numpy as np
    kap = [100.0] * len(level_ids)
    s_ = fit_mask & (n > 0)

    def score(k):
        m = binom_group_out_rate(level_ids, c, n, fit_mask, k)
        return float(np.sum(c[s_] * np.log(m[s_]) + (n[s_] - c[s_]) * np.log(1 - m[s_])) / max(1.0, n[s_].sum()))
    best = score(kap)
    for _ in range(2):
        for L in range(len(kap)):
            for g in grid:
                trial = kap[:L] + [float(g)] + kap[L + 1:]
                sc = score(trial)
                if sc > best + 1e-12:
                    best, kap = sc, trial
    return kap, best


def fit_betabinom_a(c, n, m):
    """Concentration a of theta ~ Beta(a m, a (1 - m)) maximising the Beta-binomial marginal likelihood of c of n."""
    import numpy as np
    from scipy.special import betaln
    best, best_a = -np.inf, None
    for a in np.exp(np.linspace(np.log(0.05), np.log(5000), 120)):
        ll = float(np.sum(betaln(c + a * m, n - c + a * (1 - m)) - betaln(a * m, a * (1 - m))))
        if ll > best:
            best, best_a = ll, float(a)
    return best_a, best


def betabinom_pmf(j, n, al, be):
    from scipy.special import betaln, comb
    return float(comb(n, j) * math.exp(betaln(al + j, be + n - j) - betaln(al, be)))


# --------------------------------------------------------------------------- build
def split_of(year, rank):
    if year <= TRAIN_MAX_YEAR:
        return "train", f"train: ReportYear <= {TRAIN_MAX_YEAR}"
    if year == SPLIT_YEAR:
        if rank < VAL_RANK_MAX:
            return "val", f"val: ReportYear 2025, within-year sequence rank {rank:.3f} < {VAL_RANK_MAX} (est. Jan-May 2025)"
        if rank >= EVAL_RANK_MIN:
            return "eval", (f"eval: ReportYear 2025, within-year sequence rank {rank:.3f} >= {EVAL_RANK_MIN} "
                            "(est. decided after 2025-06)")
        return None, f"buffer: ReportYear 2025, rank {rank:.3f} in [{VAL_RANK_MAX}, {EVAL_RANK_MIN}) around 2025-06-30"
    return "eval", f"eval: ReportYear {year} (decided after 2025-06)"


def render_memo(state):
    rq = state["request"]
    lines = ["REQUEST:"] + [f"- {k.replace('_', ' ')}: {v}" for k, v in rq.items()]
    if "patient" in state:
        lines += ["PATIENT:"] + [f"- {k.replace('_', ' ')}: {v}" for k, v in state["patient"].items()]
    lines += ["CASE SUMMARY:", state["case_summary"]]
    if "plan_position" in state:
        lines += ["PLAN POSITION:", state["plan_position"]]
    return "\n".join(lines)


def render_form(state):
    rq = state["request"]
    out = [f"Independent Medical Review application ({rq['review_track'].lower()} review, {rq['report_year']})"]
    if "patient" in state:
        out.append(f"Patient: age range {state['patient']['age_range']}; gender {state['patient']['gender']}")
    out += [f"Diagnosis: {rq['diagnosis']}", f"Treatment requested: {rq['treatment']}",
            f"Summary: {state['case_summary']}"]
    if "plan_position" in state:
        out.append(f"Health plan's position: {state['plan_position']}")
    return "\n".join(out)


def render(state, surface):
    return jdump(state) if surface == "json" else render_memo(state) if surface == "memo" else render_form(state)


def cf_ineligible(r, state):
    """Why an eval case cannot be flipped (None if it can)."""
    g, ar = state["patient"]["gender"], state["patient"]["age_range"]
    if g not in GENDER_SWAP:
        return "skip_gender_not_female_or_male"
    if ar not in AGE_BINS:
        return "skip_age_range_not_recorded"
    cats = " ".join([state["request"]["diagnosis"], state["request"]["treatment"]])
    free = state["case_summary"] + " " + state.get("plan_position", "")
    if SEX_AGE_SPECIFIC_RX.search(cats):
        return "skip_sex_or_age_specific_category"
    if SEX_AGE_SPECIFIC_RX.search(free):
        return "skip_sex_or_age_specific_free_text"
    if CF_TEXT_ATTR_RX.search(free):
        return "skip_age_sex_or_parent_wording_in_free_text"
    return None


def age_shift(age_range, key):
    """One AgeRange bin up or down (direction by hash of the case, clamped at the ends)."""
    i = AGE_BINS.index(age_range)
    up = int(hkey("cf_age", key)[:2], 16) % 2 == 0
    j = i + 1 if (up and i < len(AGE_BINS) - 1) or i == 0 else i - 1
    return AGE_BINS[j]


def counterfactual_records(rec, state, surface, r):
    """original / gender_swap / age_shift copies of one eval record, same questions and labels, shared pair id."""
    pair = f"{SOURCE}-cf-{r['ReferenceID']}"
    g, ar = state["patient"]["gender"], state["patient"]["age_range"]
    out = []
    for role, patient, change in (
            ("original", dict(state["patient"]), None),
            ("gender_swap", {**state["patient"], "gender": GENDER_SWAP[g]},
             {"field": "gender", "from": g, "to": GENDER_SWAP[g]}),
            ("age_shift", {**state["patient"], "age_range": age_shift(ar, r["ReferenceID"])},
             {"field": "age_range", "from": ar, "to": age_shift(ar, r["ReferenceID"])})):
        st = {**state, "patient": patient}
        text = render(st, surface)
        assert not OUTCOME_DETECTOR.search(text)
        cf = {**rec, "state": text, "questions": rec["questions"], "labels": rec["labels"],
              "label_kind": rec["label_kind"]}
        cf["id"] = f"{SOURCE}-cf-{sha(pair + role + text)}"
        cf["base_id"] = pair
        cf["meta"] = {**rec["meta"], "patient": patient, "split_reason": "eval_cf: counterfactual copy of an eval case",
                      "cf_pair_id": pair, "cf_role": role, "cf_change": change, "cf_of_record": rec["id"],
                      "cf_note": "labels copied from the original case; the counterfactual outcome is unknown, so score "
                                 "the prediction shift between roles, not accuracy"}
        validate(cf)
        out.append(cf)
    return out


def cat(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    return f"{a} / {b}" if a and b and a != b else (a or b or "not recorded")


def build(a):
    import numpy as np
    from scipy.special import betaln
    cache, outdir = Path(a.cache), Path(a.out)
    prov = json.loads((cache / "fetch_provenance.json").read_text()) if (cache / "fetch_provenance.json").exists() else {}
    for sp in ("train", "val", "eval", "eval_cf"):
        for d in (outdir / sp, outdir / "teacher_fill" / sp):
            if d.exists():
                for p in d.glob(f"{SOURCE}*.jsonl"):
                    p.unlink()
    stats = Counter()
    with open(cache / "imr.csv", encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    stats["rows_read"] = len(raw)
    seen, rows = {}, []
    for r in raw:
        k = r["ReferenceID"].strip()
        if k in seen:
            stats["duplicate_identical_collapsed" if seen[k] == r else "duplicate_conflicting_dropped"] += 1
            continue
        seen[k] = r
        rows.append(r)
    conflicts = {r["ReferenceID"].strip() for r in raw if seen[r["ReferenceID"].strip()] != r}
    rows = [r for r in rows if r["ReferenceID"].strip() not in conflicts]
    # within-year rank of the case number (the decision-time estimate)
    by_year = defaultdict(list)
    for r in rows:
        r["_year"] = int(r["ReportYear"])
        r["_seq"] = int(r["ReferenceID"].split("-")[1])
        by_year[r["_year"]].append(r["_seq"])
    for y in by_year:
        by_year[y].sort()
    for r in rows:
        v = by_year[r["_year"]]
        lo, hi = np.searchsorted(v, r["_seq"], "left"), np.searchsorted(v, r["_seq"], "right")
        r["_rank"] = float((lo + hi - 1) / 2 / max(1, len(v) - 1)) if len(v) > 1 else 0.5
    # cut and audit (the free text keeps its own age / sex wording unless --neutralise-text)
    items = []
    leak_raw = Counter()
    for r in rows:
        f = r["Findings"]
        leak_raw["design_detector_hit_in_raw_findings"] += bool(OUTCOME_DETECTOR.search(f))
        c = cut_findings(f)
        stats[f"cut_fmt_{c['fmt']}|{c['mode']}"] += 1
        summ, plan = list(c["summary"]), []
        for s in c["plan"]:
            if OUTCOME_DETECTOR.search(s) or not plan_ok(s):
                stats["plan_sentence_dropped_audit"] += 1
                continue
            plan.append(s)
        if not summ:
            stats["drop_no_summary_isolated" if not plan else "drop_summary_only_plan_sentences"] += 1
            continue
        summary_text = " ".join(summ)
        if OUTCOME_DETECTOR.search(summary_text):
            stats["drop_outcome_detector_in_summary"] += 1
            continue
        if AUDIT_RX.search(summary_text):
            stats["drop_audit_phrase_in_summary"] += 1
            continue
        if a.neutralise_text:
            summary_text, e1 = neutralise(summary_text)
            plan_text, e2 = neutralise(" ".join(plan)) if plan else ("", 0)
        else:
            plan_text, e1, e2 = " ".join(plan), 0, 0
        summary_text = clip(summary_text, a.max_summary)
        plan_text = clip(plan_text, a.max_plan) if plan_text else ""
        if OUTCOME_DETECTOR.search(summary_text + " " + plan_text) or AUDIT_RX.search(summary_text):
            stats["drop_detector_after_clip"] += 1
            continue
        r["_cut"], r["_summary"], r["_plan"], r["_edits"] = c, summary_text, plan_text, e1 + e2
        sp, why = split_of(r["_year"], r["_rank"])
        if sp is None:
            stats["drop_buffer_2025_midyear"] += 1
            continue
        r["_split"], r["_why"] = sp, why
        items.append(r)
    log(f"cut: {len(items):,} of {len(rows):,} cases kept  {Counter(r['_split'] for r in items)}")
    # labels: case-level reversal rate
    N = len(items)
    y = np.array([0 if r["Determination"].startswith("Overturned") else 1 for r in items])   # 0 = reversed (yes)
    fit = np.array([r["_split"] == "train" for r in items])
    groups = np.arange(N)
    T_ = [r["Type"] for r in items]
    tc = [t + "|" + r["TreatmentCategory"] for t, r in zip(T_, items)]
    ts = [t + "|" + r["TreatmentSubCategory"] for t, r in zip(tc, items)]
    ty = [t + "|" + str(r["_year"]) for t, r in zip(ts, items)]
    lv = encode_levels([["all"] * N, T_, tc, ts, ty])
    log("reversal: fitting hierarchical LOO case rates (global > Type > x treatment category > x subcategory > x year)")
    kap, ll = fit_kappas(lv, y, 2, fit, groups)
    p_case = group_out_rates(lv, y, 2, fit, groups, kap)
    prior = np.bincount(y[fit], minlength=2) / fit.sum()
    rep = {"case_rate_label_model": {
        "levels": ["global", "Type", "Type x treatment category", "... x treatment subcategory", "... x report year"],
        "kappas": kap, "loo_log_score_train": round(ll, 5),
        "reversal_rate_train": round(float(prior[0]), 4),
        "loo_on_train": calib_report(p_case[fit], y[fit]),
        "global_rate_on_train": calib_report(np.tile(prior, (int(fit.sum()), 1)), y[fit])}}
    for sp in ("val", "eval"):
        s = np.array([r["_split"] == sp for r in items])
        if s.any():
            rep["case_rate_label_model"][f"train_only_rates_on_{sp}"] = calib_report(p_case[s], y[s])
            rep["case_rate_label_model"][f"global_rate_on_{sp}"] = calib_report(np.tile(prior, (int(s.sum()), 1)), y[s])
            rep["case_rate_label_model"][f"reversal_rate_{sp}"] = round(float((y[s] == 0).mean()), 4)
    # votes (experimental / investigational)
    vn, vc, vhow = np.zeros(N), np.zeros(N), [None] * N
    why_not = Counter()
    for i, r in enumerate(items):
        if r["Type"] != "Experimental/Investigational":
            continue
        n_, c_, how = parse_votes(r["Findings"], y[i] == 0)
        if n_:
            vn[i], vc[i], vhow[i] = n_, c_, how
        else:
            why_not[how] += 1
    has_v = vn > 0
    ei = np.array([r["Type"] == "Experimental/Investigational" for r in items])
    lv_v = encode_levels([["all"] * N, [r["TreatmentCategory"] for r in items], [r["TreatmentSubCategory"] for r in items],
                          [r["TreatmentSubCategory"] + "|" + str(r["_year"]) for r in items]])
    fv = fit & has_v
    kap_m, llm = fit_binom_kappas(lv_v, vc, vn, fv)
    m = binom_group_out_rate(lv_v, vc, vn, fv, kap_m)
    a_hat, a_ll = fit_betabinom_a(vc[fv], vn[fv], m[fv])
    g = (vc + a_hat * m) / np.maximum(vn + a_hat, 1e-9)
    al, be = vc + a_hat * m, (vn - vc) + a_hat * (1 - m)
    split_dist = np.zeros((N, 4))
    for i in np.where(has_v & (vn == 3))[0]:
        split_dist[i] = [betabinom_pmf(j, 3, al[i], be[i]) for j in range(4)]   # j = reviewers reversing
    prior_pred = lambda i: [betabinom_pmf(j, 3, a_hat * m[i], a_hat * (1 - m[i])) for j in range(4)]  # noqa: E731
    # a check of the vote parse: realised splits vs the prior predictive on held-out splits
    sp3 = {sp: np.where(has_v & (vn == 3) & np.array([r["_split"] == sp for r in items]))[0] for sp in ("train", "val", "eval")}
    rep["vote_label_model"] = {
        "ei_cases_kept": int(ei.sum()), "ei_with_parsed_votes": int(has_v.sum()),
        "ei_with_parsed_votes_by_split": {sp: int((has_v & np.array([r["_split"] == sp for r in items])).sum())
                                          for sp in ("train", "val", "eval")},
        "parse_method": dict(Counter(h for h in vhow if h)), "not_parsed_reasons": dict(why_not.most_common()),
        "panel_sizes": dict(Counter(int(x) for x in vn[has_v])),
        "kappas_per_reviewer_rate": kap_m, "per_reviewer_reversal_rate_train": round(float(vc[fv].sum() / vn[fv].sum()), 4),
        "a": round(a_hat, 4), "beta_binomial_loglik_train": round(a_ll, 3),
        "realised_splits": {sp: dict(Counter(SPLIT_OF_C[int(vc[i])] for i in idx)) for sp, idx in sp3.items()},
    }
    for sp in ("val", "eval"):
        idx = sp3[sp]
        if len(idx):
            P = np.array([prior_pred(i) for i in idx])
            rep["vote_label_model"][f"prior_predictive_split_on_{sp}"] = calib_report(P, vc[idx].astype(int))
    log(f"  votes: {int(has_v.sum())} EI cases parsed; a={a_hat:.3g}; per-reviewer rate "
        f"{rep['vote_label_model']['per_reviewer_reversal_rate_train']}")
    # records
    rng = random.Random(a.seed)
    order = list(range(N))
    rng.shuffle(order)
    by_split = defaultdict(list)
    for i in order:
        by_split[items[i]["_split"]].append(i)
    caps = {"train": a.max_train, "val": a.max_val, "eval": a.max_eval}
    files, counts, train_states, cf_stats = {}, Counter(), set(), Counter()
    for sp in ("train", "val", "eval"):
        pool = by_split.get(sp, [])
        if caps[sp] and len(pool) > caps[sp]:
            stats[f"cap_{sp}_dropped"] += len(pool) - caps[sp]
            pool = pool[:caps[sp]]
        for i in sorted(pool):
            r = items[i]
            patient = {"age_range": (r["AgeRange"] or "").strip() or "not recorded",
                       "gender": (r["PatientGender"] or "").strip() or "not recorded"}
            state = {"case_summary": r["_summary"], "patient": patient,
                     "request": {"diagnosis": cat(r["DiagnosisCategory"], r["DiagnosisSubCategory"]),
                                 "report_year": r["_year"], "review_track": r["IMRType"] or "not recorded",
                                 "treatment": cat(r["TreatmentCategory"], r["TreatmentSubCategory"])}}
            if r["_plan"]:
                state["plan_position"] = r["_plan"]
            s100 = int(hkey("surface", r["ReferenceID"])[:4], 16) % 100
            surface = "json" if s100 < 60 else ("memo" if s100 < 85 else "form")
            text = render(state, surface)
            if OUTCOME_DETECTOR.search(text):
                stats["drop_detector_rendered"] += 1
                continue
            if sp == "train":
                train_states.add(text)
            elif text in train_states:
                stats[f"{sp}_dropped_state_identical_to_train"] += 1
                continue
            yes = bool(y[i] == 0)
            qs = {"denial_reversed_on_review": {**noul_q(REVERSE_INSTR), "criteria": dict(REVERSE_CRITERIA)},
                  "review_basis": choice_q(BASIS_INSTR, dict(BASIS_CRITERIA))}
            src, meth = {}, {}
            raw = {"denial_reversed_on_review": "yes" if yes else "no", "determination": r["Determination"],
                   "review_basis": BASIS_OF[r["Type"]]}
            if has_v[i]:
                gi = float(g[i])
                labs = {"denial_reversed_on_review": lab_dict(["yes", "no"], [gi, 1 - gi])}
                src["denial_reversed_on_review"] = f"votes:{int(vn[i])}"
                meth["denial_reversed_on_review"] = (f"Beta-binomial posterior mean (c + a m) / (n + a): c={int(vc[i])} of "
                                                     f"n={int(vn[i])} reviewers reverse ({vhow[i]}); m = LOO per-reviewer "
                                                     f"cell rate, a={a_hat:.4g} (train marginal likelihood)")
                raw["reviewers_reversing"] = f"{int(vc[i])} of {int(vn[i])}"
                if vn[i] == 3:
                    qs["panel_split"] = choice_q(SPLIT_INSTR, dict(SPLIT_CRITERIA))
                    sd = {SPLIT_OF_C[j]: split_dist[i][j] for j in range(4)}
                    labs["panel_split"] = lab_dict(SPLIT_KEYS, [sd[k] for k in SPLIT_KEYS])
                    src["panel_split"] = "votes:3"
                    meth["panel_split"] = ("Beta-binomial posterior predictive split of a fresh 3-reviewer panel given "
                                           f"the realised {int(vc[i])} of 3 reversing (a={a_hat:.4g}, m as above)")
                    raw["panel_split"] = SPLIT_OF_C[int(vc[i])]
            else:
                t = blend(y[i], p_case[i], 2, a.lam)
                labs = {"denial_reversed_on_review": (onehot_dict(["yes", "no"], "yes" if yes else "no") if sp == "eval"
                                                      else lab_dict(["yes", "no"], t))}
                src["denial_reversed_on_review"] = "outcome"
                meth["denial_reversed_on_review"] = ("onehot(Determination)" if sp == "eval" else
                                                     f"{a.lam}*onehot(Determination) + {1 - a.lam}*LOO hierarchical case "
                                                     "rate (global > Type > treatment category > subcategory > report "
                                                     "year; train-only fit)")
            labs["review_basis"] = onehot_dict(BASIS_KEYS, BASIS_OF[r["Type"]])
            src["review_basis"] = "expert"
            meth["review_basis"] = "onehot(DMHC Type)"
            kinds = {q: ("hard" if sum(1 for v in labs[q].values() if v > 0) == 1 else "soft") for q in qs}
            rec = record(SOURCE, text, qs, labs, kinds, None)
            sig = json.dumps({k: option_keys(v) for k, v in qs.items()}, sort_keys=True)
            rec["id"] = f"{SOURCE}-{sha(r['ReferenceID'] + text + sig)}"
            rec["base_id"] = rec["id"]
            rec["license"] = LICENSE
            rec["meta"] = {
                "dataset": f"DMHC IMR Determinations Trend (data.ca.gov, modified {prov.get('metadata_modified', '?')})",
                "group": f"dmhc_imr:{r['ReferenceID']}", "reference_id": r["ReferenceID"], "report_year": r["_year"],
                "as_of": f"{r['_year']} (request as filed; decision month not published)",
                "decision_time_estimate": {"report_year": r["_year"], "within_year_sequence_rank": round(r["_rank"], 4),
                                           "estimated": r["_year"] == SPLIT_YEAR},
                "split_reason": r["_why"], "surface": surface, "label_source": src, "label_method": meth,
                "raw_outcome": raw, "n_raters": int(vn[i]) if has_v[i] else None,
                "lambda": a.lam if (sp != "eval" and not has_v[i]) else None,
                "base_rate": {"denial_reversed_on_review": lab_dict(["yes", "no"], p_case[i]),
                              **({"per_reviewer_reversal_rate": round(float(m[i]), 6)} if has_v[i] else {}),
                              **({"panel_split_prior_predictive": lab_dict(SPLIT_KEYS, [dict(zip(
                                  [SPLIT_OF_C[j] for j in range(4)], prior_pred(i)))[k] for k in SPLIT_KEYS])}
                                 if has_v[i] and vn[i] == 3 else {})},
                "smoothed_target": ({"denial_reversed_on_review": lab_dict(["yes", "no"], blend(y[i], p_case[i], 2, a.lam))}
                                    if sp == "eval" and not has_v[i] else None),
                "base_rate_cell": ty[i], "cut": {"format": r["_cut"]["fmt"], "mode": r["_cut"]["mode"],
                                                 "stopped_by": r["_cut"]["why"]},
                "patient": patient, "free_text_neutralised": bool(a.neutralise_text),
                "protected_attribute_edits": r["_edits"],
                "excluded_fields": ["Type", "Determination", "DaysToReview", "DaysToAdopt"],
                "teacher_fill": "pending: evidence_strength_for_request, harm_if_delayed in teacher_fill/" if sp != "eval" else None,
            }
            validate(rec)
            if (sp, SOURCE) not in files:
                pth = outdir / sp / f"{SOURCE}.jsonl"
                pth.parent.mkdir(parents=True, exist_ok=True)
                files[(sp, SOURCE)] = pth.open("w", encoding="utf-8")
            files[(sp, SOURCE)].write(json.dumps(rec, ensure_ascii=False) + "\n")
            if sp != "eval":     # real-world eval labels never come from an LLM (realworld_data.md §5)
                write_fill(outdir, sp, f"{SOURCE}.jsonl", fill_record(rec, sp, {
                    "evidence_strength_for_request": score_q(EVID_INSTR, EVID_LEVELS),
                    "harm_if_delayed": score_q(HARM_INSTR, HARM_LEVELS)}))
            counts[sp] += 1
            if sp == "eval":
                why_not = cf_ineligible(r, state)
                if why_not:
                    cf_stats[why_not] += 1
                else:
                    for cf in counterfactual_records(rec, state, surface, r):
                        if ("eval_cf", SOURCE) not in files:
                            pth = outdir / "eval_cf" / f"{SOURCE}.jsonl"
                            pth.parent.mkdir(parents=True, exist_ok=True)
                            files[("eval_cf", SOURCE)] = pth.open("w", encoding="utf-8")
                        files[("eval_cf", SOURCE)].write(json.dumps(cf, ensure_ascii=False) + "\n")
                        cf_stats["records_" + cf["meta"]["cf_role"]] += 1
                    cf_stats["eligible_cases"] += 1
    for fh in files.values():
        fh.close()
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "seed": a.seed, "lambda": a.lam, "fetch": prov,
                "splits": {"train": f"ReportYear <= {TRAIN_MAX_YEAR}",
                           "val": f"ReportYear 2025 with within-year ReferenceID rank < {VAL_RANK_MAX}",
                           "eval": f"ReportYear 2026, or 2025 with rank >= {EVAL_RANK_MIN} (decided after 2025-06)",
                           "buffer_not_written": f"ReportYear 2025 with rank in [{VAL_RANK_MAX}, {EVAL_RANK_MIN})"},
                "caps": {"train": a.max_train, "val": a.max_val, "eval": a.max_eval},
                "records_by_split": dict(counts), "leak_in_raw_findings": {
                    "rows": len(rows), "design_detector_hits": leak_raw["design_detector_hit_in_raw_findings"],
                    "share": round(leak_raw["design_detector_hit_in_raw_findings"] / len(rows), 4)},
                "stats": dict(stats), "eval_cf": {
                    "description": "counterfactual-flip copies of eligible eval cases (original / gender_swap / age_shift); "
                                   "labels are the original's; score prediction shift only",
                    "eligibility": "gender Female/Male, age range recorded, no SEX_AGE_SPECIFIC_RX match in categories or "
                                   "free text, no CF_TEXT_ATTR_RX match in free text",
                    "stats": dict(cf_stats)}, **rep}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"wrote {dict(counts)} records to {outdir}")


# --------------------------------------------------------------------------- check
# an excluded field rendered as a field (JSON key or memo / form label), not the bare word: a treatment can be called
# "Self-Determination Theory"
EXCLUDED_FIELD_RX = re.compile(r'"(?:Determination|DaysToReview|DaysToAdopt|Type|determination|type)"\s*:|'
                               r'^(?:- )?(?:determination|days to review|days to adopt|review type)\s*:', re.I | re.M)
PATIENT_FIELD_RX = re.compile(r'"patient": \{"age_range": "[^"]+", "gender": "[^"]+"\}|^PATIENT:\n- age range: .+\n- gender: .+$|'
                              r'^Patient: age range .+; gender .+$', re.M)


def fairness_tables(recs, splits=("train", "eval")):
    """Reversal rates by AgeRange x PatientGender (and margins): the realised outcome (Determination) over all cases,
    and the per-reviewer reversal share sum(c) / sum(n) over cases with parsed votes. Also the mean training label."""
    out = {}
    for sp in splits:
        cells = defaultdict(lambda: {"n": 0, "reversed": 0, "vote_cases": 0, "votes_reverse": 0, "votes": 0,
                                     "label_yes_sum": 0.0})
        for r in recs.get(sp, []):
            pa = r["meta"]["patient"]
            keys = [(pa["age_range"], pa["gender"]), (pa["age_range"], "all"), ("all", pa["gender"]), ("all", "all")]
            ro = r["meta"]["raw_outcome"]
            for k in keys:
                c = cells[k]
                c["n"] += 1
                c["reversed"] += ro["denial_reversed_on_review"] == "yes"
                c["label_yes_sum"] += r["labels"]["denial_reversed_on_review"]["yes"]
                if "reviewers_reversing" in ro:
                    cr, nr = (int(x) for x in ro["reviewers_reversing"].split(" of "))
                    c["vote_cases"] += 1
                    c["votes_reverse"] += cr
                    c["votes"] += nr
        rows = []
        for (age, gen), c in cells.items():
            rows.append({"age_range": age, "gender": gen, "n": c["n"],
                         "outcome_reversal_rate": round(c["reversed"] / c["n"], 4) if c["n"] else None,
                         "vote_cases": c["vote_cases"],
                         "per_reviewer_reversal_share": round(c["votes_reverse"] / c["votes"], 4) if c["votes"] else None,
                         "mean_label_yes": round(c["label_yes_sum"] / c["n"], 4) if c["n"] else None})
        order = {a: i for i, a in enumerate(AGE_BINS + ["not recorded", "all"])}
        gorder = {"Female": 0, "Male": 1, "Other": 2, "not recorded": 3, "all": 4}
        rows.sort(key=lambda x: (order.get(x["age_range"], 99), gorder.get(x["gender"], 99)))
        out[sp] = rows
    return out


def state_parts(r):
    """(case_summary, plan_position) of a written state, on any surface."""
    s, sf = r["state"], r["meta"]["surface"]
    if sf == "json":
        o = json.loads(s)
        return o["case_summary"], o.get("plan_position", "")
    if sf == "memo":
        m = re.search(r"CASE SUMMARY:\n(.*?)(?:\nPLAN POSITION:\n(.*))?$", s, re.S)
    else:
        m = re.search(r"Summary: (.*?)(?:\nHealth plan's position: (.*))?$", s, re.S)
    return m.group(1), (m.group(2) or "")


def state_text(r):
    """The free text of a state (summary + plan position) on any surface."""
    a, b = state_parts(r)
    return (a + " " + b).strip()


def check(a):
    outdir = Path(a.out)
    allrecs = read_corpus(outdir, SOURCE, splits=("train", "val", "eval", "eval_cf"))
    report = {"readback": readback(outdir, allrecs, splits=("train", "val", "eval", "eval_cf"))}
    cfrecs = allrecs.pop("eval_cf", [])
    recs = allrecs
    report["dedup"] = dedup_report(recs)
    det, aud, res = Counter(), Counter(), Counter()
    patient_present = Counter()
    for v in list(recs.values()) + [cfrecs]:
        for r in v:
            for mm in OUTCOME_DETECTOR.finditer(r["state"]):
                det[mm.group(0).lower()] += 1
            summ, plan = state_parts(r)
            txt = (summ + " " + plan).strip()
            for mm in AUDIT_RX.finditer(summ):
                aud[mm.group(0).lower()] += 1
            for mm in AUDIT_RX.finditer(PLAN_CLAIM.sub(" ", plan)):
                aud["plan:" + mm.group(0).lower()] += 1
            for mm in RESIDUAL_RX.finditer(txt):
                res[mm.group(0).lower()] += 1
            assert not EXCLUDED_FIELD_RX.search(r["state"]), (r["id"], EXCLUDED_FIELD_RX.search(r["state"]).group(0))
            patient_present["with_patient_fields" if PATIENT_FIELD_RX.search(r["state"]) else "MISSING"] += 1
    report["patient_fields_in_states"] = dict(patient_present)
    report["outcome_detector_hits_in_written_states"] = dict(det) or "0 hits"
    report["audit_phrase_hits_in_written_free_text"] = dict(aud.most_common(20)) or "0 hits"
    report["age_sex_wording_in_free_text_kept"] = dict(res.most_common(20)) or "0 hits"
    # fairness audit (age range and gender are kept as clinical covariates; audited, not blinded)
    fair = fairness_tables(recs)
    report["fairness_audit"] = fair
    (outdir / "fairness_audit.json").write_text(json.dumps(fair, indent=1))
    with (outdir / "fairness_audit.tsv").open("w", encoding="utf-8") as f:
        f.write("split\tage_range\tgender\tn\toutcome_reversal_rate\tvote_cases\tper_reviewer_reversal_share\t"
                "mean_label_yes\n")
        for sp, rows in fair.items():
            for x in rows:
                f.write("\t".join(str(v) for v in (sp, x["age_range"], x["gender"], x["n"], x["outcome_reversal_rate"],
                                                    x["vote_cases"], x["per_reviewer_reversal_share"],
                                                    x["mean_label_yes"])) + "\n")
    # eval_cf integrity: complete trios, one change per copy, labels equal to the original's
    pairs = defaultdict(dict)
    for r in cfrecs:
        pairs[r["meta"]["cf_pair_id"]][r["meta"]["cf_role"]] = r
    complete = [p for p in pairs.values() if set(p) == {"original", "gender_swap", "age_shift"}]
    for p in complete:
        o = p["original"]
        for role in ("gender_swap", "age_shift"):
            x = p[role]
            assert x["labels"] == o["labels"] and x["questions"] == o["questions"], x["id"]
            ch = x["meta"]["cf_change"]
            assert o["meta"]["patient"][ch["field"]] == ch["from"] and x["meta"]["patient"][ch["field"]] == ch["to"]
            other = "age_range" if ch["field"] == "gender" else "gender"
            assert o["meta"]["patient"][other] == x["meta"]["patient"][other]
    evo = {r["id"]: r for r in recs.get("eval", [])}
    assert all(p["original"]["state"] == evo[p["original"]["meta"]["cf_of_record"]]["state"] for p in complete)
    report["eval_cf"] = {"records": len(cfrecs), "pairs": len(pairs), "complete_trios": len(complete),
                         "eval_cases": len(recs.get("eval", [])),
                         "gender_swaps": dict(Counter(f"{p['gender_swap']['meta']['cf_change']['from']}->"
                                                      f"{p['gender_swap']['meta']['cf_change']['to']}" for p in complete)),
                         "age_shifts": dict(Counter(f"{p['age_shift']['meta']['cf_change']['from']}->"
                                                    f"{p['age_shift']['meta']['cf_change']['to']}" for p in complete)),
                         "label_source": dict(Counter(p["original"]["meta"]["label_source"]["denial_reversed_on_review"]
                                                      for p in complete))}
    report["label_stats"] = label_stats(recs)
    rc = defaultdict(Counter)
    for sp, v in recs.items():
        for r in v:
            for q in ("denial_reversed_on_review", "panel_split", "review_basis"):
                if q in r["meta"]["raw_outcome"]:
                    rc[f"{sp}/{q}"][r["meta"]["raw_outcome"][q]] += 1
            rc[f"{sp}/label_source_reversal"][r["meta"]["label_source"]["denial_reversed_on_review"]] += 1
            rc[f"{sp}/surface"][r["meta"]["surface"]] += 1
            rc[f"{sp}/cut_mode"][r["meta"]["cut"]["mode"]] += 1
            rc[f"{sp}/report_year"][r["meta"]["report_year"]] += 1
    report["realised_labels"] = {k: dict(sorted(c.items(), key=lambda x: str(x[0]))) for k, c in sorted(rc.items())}
    for sp, v in recs.items():
        lens = sorted(len(r["state"]) for r in v)
        sl = sorted(len(state_text(r)) for r in v)
        if lens:
            report.setdefault("state_chars", {})[sp] = {"min": lens[0], "median": lens[len(lens) // 2],
                                                        "p90": lens[9 * len(lens) // 10], "max": lens[-1],
                                                        "free_text_median": sl[len(sl) // 2]}
    report["exclusion_errors"] = run_exclusion("medical_review_determinations", {
        "experimental_investigational_review": ["denial_reversed_on_review", "panel_split", "review_basis",
                                                "evidence_strength_for_request", "harm_if_delayed"],
        "medical_necessity_review": ["denial_reversed_on_review", "review_basis", "evidence_strength_for_request",
                                     "harm_if_delayed"]})
    # leak probe on the realised outcome (train) against the train-only base rate
    rng = random.Random(0)
    train = recs.get("train", [])
    if train:
        sample = rng.sample(train, min(a.probe_n, len(train)))
        report["leak_probe_reversal_free_text_only"] = leak_probe(sample, "denial_reversed_on_review", ["yes", "no"],
                                                                  text_of=state_text)
        report["leak_probe_reversal_whole_state"] = leak_probe(sample, "denial_reversed_on_review", ["yes", "no"])
    ev = recs.get("eval", [])
    if ev:
        report["leak_probe_reversal_eval_free_text_only"] = leak_probe(ev, "denial_reversed_on_review", ["yes", "no"],
                                                                       text_of=state_text)
    # 200-row manual audit sample (all splits; eval over-represented: its cut is the one that matters most)
    pool = [r for v in recs.values() for r in v]
    done = {line.split("\t", 1)[0] for mf in outdir.glob("summary_audit_manual_s*.tsv")
            if not mf.name.endswith(f"_s{a.audit_seed}.tsv") for line in mf.read_text(encoding="utf-8").splitlines()[1:]}
    pool = [r for r in pool if r["id"] not in done]
    evp = [r for r in pool if r["id"] in {x["id"] for x in ev}]
    rs = random.Random(a.audit_seed)
    aud_rows = rs.sample(evp, min(60, len(evp)))
    rest = [r for r in pool if r not in aud_rows]
    aud_rows += rs.sample(rest, min(200 - len(aud_rows), len(rest)))
    with (outdir / f"summary_audit_sample_s{a.audit_seed}.tsv").open("w", encoding="utf-8") as f:
        f.write("record_id\tsplit_reason\tcut_mode\tfree_text\n")
        for r in aud_rows:
            f.write(f"{r['id']}\t{r['meta']['split_reason'].split(':')[0]}\t{r['meta']['cut']['mode']}\t"
                    f"{state_text(r).replace(chr(9), ' ')}\n")
    report["summary_manual_audit"] = {}
    for mf in sorted(outdir.glob("summary_audit_manual_s*.tsv")):
        lines = mf.read_text(encoding="utf-8").splitlines()[1:]
        verdicts = Counter(line.split("\t")[1] for line in lines if "\t" in line)
        report["summary_manual_audit"][mf.name] = {
            "n": sum(verdicts.values()), "verdicts": dict(verdicts),
            "flagged": [line.split("\t") for line in lines if "\t" in line and line.split("\t")[1] != "clean"]}
    fills = sum(1 for p in (outdir / "teacher_fill").rglob("*.jsonl") for _ in p.open())
    report["teacher_fill_records_pending"] = fills
    report["bytes_on_disk"] = sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file() and "cache" not in p.parts)
    (outdir / "check_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    samples = []
    for sp, pred in (("train", lambda r: "panel_split" in r["questions"]), ("train", lambda r: "panel_split" not in r["questions"]),
                     ("val", lambda r: True), ("eval", lambda r: "panel_split" in r["questions"]),
                     ("eval", lambda r: "panel_split" not in r["questions"])):
        pool = [r for r in recs.get(sp, []) if pred(r)]
        if pool:
            samples.append(random.Random(len(samples)).choice(pool))
    (outdir / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- self-test (no network)
def self_test():
    import numpy as np
    f = ("Nature of Statutory Criteria/Case Summary: The parent of a 13-year-old male enrollee has requested "
         "authorization and coverage for Xifaxan for treatment of his small intestinal bowel overgrowth. "
         "The Health Plan denied the request as not medically necessary. Findings: The physician reviewer found that "
         "Xifaxan is medically necessary. Final Result: Therefore, the Health Plan's denial should be overturned.")
    c = cut_findings(f)
    assert c["mode"] == "broad" and len(c["summary"]) == 1 and len(c["plan"]) == 1, c
    assert not OUTCOME_DETECTOR.search(" ".join(c["summary"])), c
    t, n = neutralise(c["summary"][0])
    assert "13" not in t and "male" not in t.lower() and " his " not in t and n >= 3, t
    assert t.startswith("The parent of an enrollee has requested"), t
    w = ("Findings: The physician reviewer found that a patient has requested authorization and coverage for left total "
         "knee arthroplasty (TKA). The authors of a study state that pain should be reported. Therefore TKA is not "
         "medically necessary.Final Result: The reviewer determined ... upheld.")
    c = cut_findings(w)
    assert c["mode"] == "narrow" and c["summary"] == ["A patient has requested authorization and coverage for left total "
                                                      "knee arthroplasty (TKA)."], c
    assert not cut_findings("Nature of Statutory Criteria/Case Summary: The patient requested Acthar Gel. The submitted "
                            "documentation supports the medical necessity of the requested medication. Findings: x")["summary"][1:]
    assert cut_findings("DO NOT POST - X BLANK.An enrollee has requested IMRT for cancer. Findings: y")["summary"] == [
        "An enrollee has requested IMRT for cancer."]
    assert cut_findings("Findings: The physician reviewer found that a patient has requested Wegovy. Pharmacotherapy is "
                        "recommended for weight loss. Final Result: x")["summary"] == ["A patient has requested Wegovy."]
    assert cut_findings("Findings: The physician reviewer found that the parent of a patient has requested Beyfortus. The "
                        "current CDC ACIP recommendation for Beyfortus is for infants. Final Result: x")["summary"] == [
        "The parent of a patient has requested Beyfortus."]
    assert cut_findings("Findings: The physician reviewer found that a patient has requested a PET/CT scan (skull to "
                        "thigh).PET/CT is most often used to evaluate disease. Final Result: x")["summary"] == [
        "A patient has requested a PET/CT scan (skull to thigh)."]
    assert not cut_findings("At issue is whether the scan was medically necessary. Findings: The reviewer found that "
                            "...")["summary"]
    # neutraliser
    for s, want in [("She reports pain in her knee.", "They reports pain in their knee."),
                    ("The plan denied her request.", "The plan denied their request."),
                    ("The provider referred her to a surgeon.", "The provider referred them to a surgeon."),
                    ("A 45-year-old female enrollee has requested surgery.", "An enrollee has requested surgery."),
                    ("He has requested a PureWick male external catheter.", "They have requested a PureWick male external catheter."),
                    ("A five-year-old boy with autism.", "A child with autism."),
                    ("A 49-year-year-old female has requested surgery.", "A patient has requested surgery."),
                    ("The patient is a young woman with cancer.", "The patient is a young person with cancer."),
                    ("The parent of a then 1-year- old female enrollee has requested care.",
                     "The parent of a then enrollee has requested care."),
                    ("The parent of a 3-year, enrollee has requested IVIG.", "The parent of an enrollee has requested IVIG.")]:
        got, _ = neutralise(s)
        assert got == want, (s, got, want)
    # votes
    ov = "Two of the three physician reviewers found that the test is likely to be more beneficial. Credentials/Qualifications: All three reviewers are board certified."
    assert parse_votes(ov, True) == (3, 2, "count_statements_split"), parse_votes(ov, True)
    assert parse_votes(ov, False)[0] is None
    un = "Three of the three reviewers found that the treatment is not likely to be more beneficial. Credentials/Qualifications: Reviewer 1 ..."
    assert parse_votes(un, False) == (3, 0, "count_statements_unanimous")
    old = ("A patient requested Botox. Findings: Two physician reviewers found that Botox is not likely to be superior. "
           "One physician reviewer found that Botox is likely to be more beneficial.")
    assert parse_votes(old, False) == (3, 1, "count_statements_split"), parse_votes(old, False)
    pp = ("Physician 1: The patient is a 45-year-old male. I have determined the procedure is not likely to be more "
          "beneficial. Therefore, the denial should be upheld. Physician 2: The patient ... is likely to be more beneficial. "
          "Physician 3: ... I have determined the requested procedure is not likely to be more beneficial for treatment.")
    assert parse_votes(pp, False) == (3, 1, "per_physician_opinions"), parse_votes(pp, False)
    two = "Both reviewers found that X is likely to be more beneficial. Credentials/Qualifications: Both reviewers are board certified."
    assert parse_votes(two, True) == (2, 2, "count_statements_unanimous"), parse_votes(two, True)
    # label maths
    lv = encode_levels([["all"] * 4, ["a", "a", "b", "b"]])
    m1 = binom_group_out_rate(lv, np.array([3.0, 0, 0, 0]), np.array([3.0] * 4), np.ones(4, dtype=bool), [10.0, 10.0])
    m2 = binom_group_out_rate(lv, np.array([0.0, 0, 0, 0]), np.array([3.0] * 4), np.ones(4, dtype=bool), [10.0, 10.0])
    assert abs(m1[0] - m2[0]) < 1e-12 and abs(m1[1] - m2[1]) > 1e-6
    p = [betabinom_pmf(j, 3, 2.5, 1.5) for j in range(4)]
    assert abs(sum(p) - 1) < 1e-9
    for sp, want in ((split_of(2024, 0.9)[0], "train"), (split_of(2025, 0.1)[0], "val"), (split_of(2025, 0.5)[0], None),
                     (split_of(2025, 0.7)[0], "eval"), (split_of(2026, 0.0)[0], "eval")):
        assert sp == want
    # patient covariates, renderers and the counterfactual flip
    st = {"case_summary": "A patient has requested authorization and coverage for Repatha.",
          "patient": {"age_range": "51 to 64", "gender": "Female"},
          "request": {"diagnosis": "Cardiac/Circ Problem / High Cholesterol", "report_year": 2025,
                      "review_track": "Standard", "treatment": "Pharmacy / Cardiac Rx"}}
    for sf in ("json", "memo", "form"):
        txt = render(st, sf)
        assert PATIENT_FIELD_RX.search(txt) and not EXCLUDED_FIELD_RX.search(txt), (sf, txt)
        fake = {"state": txt, "meta": {"surface": sf}}
        assert state_parts(fake)[0] == st["case_summary"], (sf, state_parts(fake))
    assert cf_ineligible(None, st) is None
    assert cf_ineligible(None, {**st, "case_summary": "A patient has requested a screening mammogram."}) == \
        "skip_sex_or_age_specific_free_text"
    assert cf_ineligible(None, {**st, "case_summary": "The patient has requested Repatha for her cholesterol."}) == \
        "skip_age_sex_or_parent_wording_in_free_text"
    assert cf_ineligible(None, {**st, "request": {**st["request"], "treatment": "Pharmacy / Osteoporosis Rx"}}) == \
        "skip_sex_or_age_specific_category"
    assert cf_ineligible(None, {**st, "patient": {"age_range": "51 to 64", "gender": "Other"}}) == \
        "skip_gender_not_female_or_male"
    assert age_shift("0 to 10", "x") == "11 to 20" and age_shift("65+", "x") == "51 to 64"
    assert all(abs(AGE_BINS.index(age_shift(b, k)) - AGE_BINS.index(b)) == 1 for b in AGE_BINS for k in "abcdef")
    for v in list(BASIS_CRITERIA.values()) + list(SPLIT_CRITERIA.values()) + EVID_LEVELS + HARM_LEVELS:
        assert v.strip()
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["fetch", "build", "check", "merge-fill", "all"])
    ap.add_argument("--cache", default="/tmp/dmhc_jev/cache")
    ap.add_argument("--out", default="/tmp/dmhc_jev")
    ap.add_argument("--lam", type=float, default=LAMBDA, help="weight on the realised outcome in O labels")
    ap.add_argument("--max-train", type=int, default=0, help="0 = every train case")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--max-summary", type=int, default=1200)
    ap.add_argument("--max-plan", type=int, default=500)
    ap.add_argument("--probe-n", type=int, default=12000)
    ap.add_argument("--audit-seed", type=int, default=7)
    ap.add_argument("--neutralise-text", action="store_true",
                    help="neutralise ages, sex words and pronouns in the free text (the pre-2026-10-01 behaviour; off)")
    ap.add_argument("--fill-labelled", default=None)
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
        merge_fill(a.out, a.fill_labelled, SOURCE)


if __name__ == "__main__":
    main()
