"""IRB protocol review: how an institutional review board (human research protection office) should handle a
research protocol, a consent document, an amendment, an annual renewal, or a recruitment and payment plan.

Every institution, study, drug code, rule number and person here is fictional. The review rules rendered in the
states ("HRPP-SOP 7" and so on) are invented institutional rules written for this generator: simplified,
plausible, and deliberately NOT a statement of any real regulation.

Each schema draws a `regime` first. Borderline cases sit on a written line (a 460 mL blood draw against a
450 mL limit, a consent form at grade 8.4 against a grade-8 target); conflicting cases pit the investigator's
own classification against the facts ("exempt, anonymous" for a survey that stores names), or a clean report
against one alarming deviation.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    money,
    near,
    noisy,
    noul,
    org,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    regime,
    score,
    show,
    table,
)


def _inst(rng):
    return org(rng, "school")


def _protocol_no(rng, d):
    return f"HRP-{d.year}-{rng.randint(100, 1999):04d}"


def _n(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def _pi(rng):
    return pick(rng, ["Dr. ", "Dr. ", "Prof. "]) + person(rng)


# =========================================================================== schema 1: review pathway
DESIGNS = {
    "online_survey": dict(titles=["Sleep, screen time and mood in {g}", "Everyday stress and coping in {g}"],
                          procedure="a 20-minute online questionnaire", contact=True),
    "interviews": dict(titles=["Experiences of arranging support: interviews with {g}",
                               "How {g} make decisions about money"],
                       procedure="one 60-minute audio-recorded interview", contact=True),
    "secondary_records": dict(titles=["Readmission patterns after hip fracture, 2019-2024",
                                      "Course grades and tutoring use in introductory chemistry"],
                              procedure="review of existing records; no contact with participants", contact=False),
    "blood_draw": dict(titles=["Inflammation markers during exercise training in {g}",
                               "Vitamin levels in {g}"],
                       procedure="venous blood samples at each visit", contact=True),
    "behavioral_intervention": dict(titles=["A sleep-restriction program for insomnia in {g}",
                                            "Graded exercise sessions for fatigue in {g}"],
                                    procedure="an 8-week supervised program with weekly sessions", contact=True),
    "deception_task": dict(titles=["Feedback and persistence on a timed puzzle task in {g}",
                                   "Perceived group opinion and risk choices in {g}"],
                           procedure="a 40-minute lab task with a cover story, followed by debriefing", contact=True),
    "quality_improvement": dict(titles=["Reducing missed follow-up calls in the cardiology clinic",
                                        "Shortening discharge paperwork on ward 6B"],
                                procedure="changes to a clinic workflow, with before/after audit of charts", contact=False),
    "public_dataset": dict(titles=["Regional trends in bicycle injuries from a published open dataset",
                                   "Secondary analysis of a public national time-use survey"],
                           procedure="analysis of a downloaded public dataset", contact=False),
}
POPULATIONS = {
    "adults": "adults aged 18 and over from the community",
    "own_students": "students enrolled in the investigator's own course",
    "minors": "children aged 12-17, with parental permission",
    "impaired_adults": "adults with moderate memory impairment, some unable to consent for themselves",
    "prisoners": "adults held at a county detention facility",
    "employees": "staff employed in the investigator's own department",
}
GROUP_WORDS = {"adults": "community adults", "own_students": "first-year students", "minors": "adolescents",
               "impaired_adults": "adults with memory impairment", "prisoners": "people in county detention",
               "employees": "department staff"}
TOPICS = {None: None, "substance_use": "recent drug and alcohol use", "mental_health": "symptoms of depression and self-harm",
          "workplace_conflict": "conflicts with supervisors", "sexual_health": "sexual behaviour"}
PATHWAY_RULES = ("HRPP-SOP 7 (summary). Not human-subjects research: no interaction with living people and no "
                 "identifiable private information, OR no aim to produce generalisable knowledge. Exempt: minimal risk; "
                 "adult participants; surveys, interviews, benign tasks or existing data; identifiable sensitive "
                 "information only if held in institution-approved secure storage. Expedited: minimal risk, and every "
                 "procedure is on the institutional list (includes blood draws totalling no more than 450 mL in any "
                 "8-week period, recordings, noninvasive measures). Full board: anything above minimal risk, and any "
                 "study enrolling prisoners. Minors and adults unable to consent cannot be exempt except for anonymous "
                 "educational tests.")


class InitialReviewPathway(Schema):
    name = "initial_review_pathway"
    description = "A new protocol at intake: which level of IRB review it needs, and what the office must ask for first."
    surfaces = {"json": 0.5, "submission_form": 0.25, "analyst_precheck": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        design = pick(rng, list(DESIGNS))
        pop = pick_w(rng, {"adults": 5, "own_students": 2, "minors": 2, "impaired_adults": 1, "prisoners": 1,
                           "employees": 1})
        ident_level = pick_w(rng, {"anonymous": 3, "coded": 4, "identified": 3})
        topic = pick_w(rng, {None: 4, "substance_use": 2, "mental_health": 2, "workplace_conflict": 1, "sexual_health": 1})
        storage = pick_w(rng, {"institutional_encrypted": 5, "commercial_platform": 3, "personal_laptop": 1})
        blood_ml = None
        intervention = "none"
        generalisable = True
        requested = pick_w(rng, {"exempt": 4, "expedited": 3, "full_board": 1, "not_human_subjects": 1})
        advisor_signed = None
        variant = None
        student_pi = chance(rng, 0.35)
        if reg == "clear":
            if chance(rng, 0.5):
                design, pop, ident_level, topic, storage, requested = ("online_survey", "adults", "anonymous", None,
                                                                       "institutional_encrypted", "exempt")
            else:
                design, pop, ident_level, requested = "behavioral_intervention", pick(rng, ["impaired_adults", "prisoners"]), "identified", "full_board"
                intervention = "elevated"
        elif reg == "borderline":
            b = pick(rng, ["blood_volume", "coded_sensitive", "qi_publication", "own_students"])
            if b == "blood_volume":
                design, pop = "blood_draw", pick(rng, ["adults", "employees"])
                requested = "expedited"
            elif b == "coded_sensitive":
                design, ident_level = pick(rng, ["online_survey", "interviews"]), "coded"
                topic = pick(rng, ["substance_use", "mental_health", "sexual_health"])
                storage = pick(rng, ["commercial_platform", "institutional_encrypted"])
                requested = "exempt"
            elif b == "qi_publication":
                design, requested = "quality_improvement", "not_human_subjects"
                generalisable = pick(rng, [True, False, None])
            else:
                design, pop, requested = pick(rng, ["online_survey", "deception_task"]), "own_students", "exempt"
        else:
            variant = pick(rng, ["exempt_claim_identified", "minors_online", "public_data_linked", "low_risk_label_on_intervention"])
            if variant == "exempt_claim_identified":
                design, requested, ident_level = pick(rng, ["online_survey", "interviews"]), "exempt", "identified"
                topic = pick(rng, ["substance_use", "mental_health"])
                storage = pick(rng, ["commercial_platform", "personal_laptop"])
            elif variant == "minors_online":
                design, pop, requested, ident_level = "online_survey", "minors", "exempt", pick(rng, ["anonymous", "coded"])
            elif variant == "public_data_linked":
                design, requested, ident_level = "public_dataset", "not_human_subjects", "coded"
            else:
                design, requested, intervention = "behavioral_intervention", "expedited", pick(rng, ["low", "elevated"])
        if design == "blood_draw":
            blood_ml = near(rng, 450, 90, lo=40, integer=True) if reg != "clear" else rng.randint(40, 200)
        if design == "behavioral_intervention" and intervention == "none":
            intervention = pick(rng, ["low", "elevated"])
        if not DESIGNS[design]["contact"] and pop in ("minors", "impaired_adults", "prisoners") and chance(rng, 0.5):
            pop = "adults"
        if topic == "workplace_conflict" and pop not in ("adults", "employees"):
            topic = None
        if student_pi:
            advisor_signed = pick_w(rng, {True: 3, False: 1, None: 1})
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "design": design, "title": pick(rng, DESIGNS[design]["titles"]).format(g=GROUP_WORDS[pop]),
            "population": pop, "identifiability": ident_level, "sensitive_topic": topic, "storage": storage,
            "blood_ml_8wk": blood_ml, "intervention_risk": intervention, "generalisable_aim": generalisable,
            "requested_level": requested, "student_pi": student_pi, "advisor_signed": advisor_signed,
            "sample_n": pick(rng, [12, 30, 60, 150, 400, 2500]), "funding": pick(rng, ["unfunded", "federal grant",
                                                                                     "industry sponsor", "internal pilot fund"]),
            "pi": _pi(rng) if not student_pi else person(rng), "advisor": _pi(rng), "institution": _inst(rng),
            "department": pick(rng, ["Psychology", "Nursing", "Kinesiology", "Public Health", "Education",
                                     "Internal Medicine", "Sociology", "Agricultural Economics"]),
            "received": received, "protocol": _protocol_no(rng, received),
            "pi_risk_statement": pick(rng, ["no risks beyond everyday life", "minimal risk", "some discomfort possible",
                                            "risks are described in section 8"]),
        }

    def _ident_text(self, f):
        if not DESIGNS[f["design"]]["contact"]:
            return {"anonymous": "records extracted without names or record numbers",
                    "coded": "records extracted under a study code; the key to names is kept by the team",
                    "identified": "extracted records keep names and record numbers"}[f["identifiability"]]
        return {"anonymous": "no names or contact details collected; IP logging switched off",
                "coded": "responses stored under a study code; the key linking codes to names is kept by the team",
                "identified": "names and email addresses stored with the responses"}[f["identifiability"]]

    def _storage_text(self, f):
        if not DESIGNS[f["design"]]["contact"] and f["storage"] == "commercial_platform":
            return "a personal cloud-storage account held by the investigator"
        return {"institutional_encrypted": "institution-managed encrypted research drive",
                "commercial_platform": "a commercial survey platform account held by the investigator",
                "personal_laptop": "the investigator's own laptop, password protected"}[f["storage"]]

    def _procedures(self, f):
        p = [DESIGNS[f["design"]]["procedure"]]
        if f["blood_ml_8wk"]:
            p.append(f"total blood drawn about {f['blood_ml_8wk']} mL over 8 weeks")
        if f["sensitive_topic"] and DESIGNS[f["design"]]["contact"]:
            p.append(f"questions include items on {TOPICS[f['sensitive_topic']]}")
        elif f["sensitive_topic"] and f["design"] != "public_dataset":
            p.append(f"extracted fields include notes on {TOPICS[f['sensitive_topic']]}")
        if f["intervention_risk"] == "elevated":
            p.append("sessions push participants to their tolerance limit, with a clinician on call")
        elif f["intervention_risk"] == "low":
            p.append("sessions are mild and can be stopped at any time")
        if f["design"] == "quality_improvement":
            p.append({True: "results will be written up for a national conference", False: "results are for internal use only",
                      None: "plans for sharing results not stated"}[f["generalisable_aim"]])
        if f["design"] == "public_dataset" and f["identifiability"] != "anonymous":
            p.append("the team will link the dataset to hospital admission records using date of birth and postcode")
        return p

    def render(self, f, rng, surface):
        pop = POPULATIONS[f["population"]]
        req = {"exempt": "exempt", "expedited": "expedited", "full_board": "full board",
               "not_human_subjects": "not-human-subjects determination"}[f["requested_level"]]
        pi_line = f["pi"] + (f" (student; advisor {f['advisor']})" if f["student_pi"] else "")
        adv = (None if not f["student_pi"] else {True: "signed", False: "not signed", None: "missing"}[f["advisor_signed"]])
        if surface == "json":
            return dumps(prune({
                "submission": {"protocol": f["protocol"], "title": f["title"], "received": f["received"].isoformat(),
                               "investigator": pi_line, "department": f["department"], "funding": f["funding"],
                               "review_level_requested": req, "advisor_signature": adv},
                "study": {"population": pop, "planned_n": f["sample_n"], "procedures": self._procedures(f),
                          "investigator_risk_statement": f["pi_risk_statement"]},
                "data": {"identifiability": self._ident_text(f), "storage": self._storage_text(f)},
                "rules": PATHWAY_RULES,
            }))
        if surface == "submission_form":
            rows = [("Protocol", f["protocol"]), ("Title", f["title"]), ("PI", pi_line),
                    ("Department", f["department"]), ("Funding", f["funding"]), ("Participants", pop),
                    ("Target enrolment", f["sample_n"]), ("Review type requested", req.upper()),
                    ("Faculty advisor signature", adv), ("Identifiers", self._ident_text(f)),
                    ("Data storage", self._storage_text(f)), ("Risks (PI)", f["pi_risk_statement"])]
            return ("NEW STUDY SUBMISSION -- " + f["institution"] + " IRB\n" + kv_lines(rows, rng) +
                    "\n\nProcedures:\n" + bullets(self._procedures(f), rng) + "\n\n" + PATHWAY_RULES)
        # analyst_precheck
        analyst = person(rng)
        obs = [f"{f['protocol']} ({f['department']}) requests {req}" + ("" if "determination" in req else " review")
               + f" for \"{f['title']}\".",
               f"Population: {pop}; n = {f['sample_n']}.",
               "Procedures: " + "; ".join(self._procedures(f)) + ".",
               f"Data: {self._ident_text(f)}, kept on {self._storage_text(f)}.",
               f"PI describes the risk as {f['pi_risk_statement']}."]
        if f["student_pi"] and f["advisor_signed"] is not True:
            obs.append("Advisor sign-off is not in the packet.")
        return (f"PRE-REVIEW CHECK by {analyst}, {fmt_date(f['received'] + timedelta(days=rng.randint(1, 6)), rng)}\n\n"
                + noisy(join_sentences(obs), rng, 0.12) + "\n\nReference: " + PATHWAY_RULES)

    def questions(self):
        return {
            "review_pathway": choice("Which level of IRB review does this protocol need?", {
                "exempt": "Exempt review: a single reviewer confirms it fits an exempt category, with no continuing review.",
                "expedited": "Expedited review by one experienced reviewer, because it is minimal risk and every procedure is on the list.",
                "full_board": "Review by the convened board, because risk or the population takes it outside the lighter routes.",
                "not_human_subjects": "No IRB review; the project does not meet the definition of human-subjects research.",
            }),
            "minimal_risk_study": noul(
                "The study involves no more than minimal risk to the people taking part.",
                yes="No session pushes people to their tolerance limit, any blood drawn stays within 450 mL in 8 weeks, "
                    "and identifiable or coded sensitive answers are kept in institution-approved secure storage.",
                no="Sessions push participants to their tolerance limit, blood drawn exceeds 450 mL in 8 weeks, or "
                   "identifiable or coded sensitive answers are kept outside approved secure storage."),
            "disclosure_harm": score("How much harm could participants suffer if their data were linked back to them?", [
                "Negligible: nothing collected would matter if disclosed.",
                "Minor: some embarrassment, no lasting consequence.",
                "Serious: damage to reputation, employment or relationships.",
                "Severe: legal exposure or lasting harm to health, livelihood or safety.",
            ]),
            "population_safeguards": score("How much added protection does the participant population need?", [
                "None beyond standard consent.",
                "Light: attention to how people are approached so they feel free to decline.",
                "Substantial: permission, assent or a representative's consent must be built in.",
                "Extensive: a population with formally restricted choice, needing specific board findings.",
            ]),
            "intake_request": choice("What should the IRB office ask the investigator for before the protocol goes forward?", {
                "consent_or_assent_materials": "Consent, parental permission or assent materials suited to the population.",
                "data_protection_plan": "A data-protection plan: where identifiable data live, who holds the key, when it is destroyed.",
                "nothing_further": "Nothing further; the packet is sufficient for the review it needs.",
                "procedure_justification": "A justification of the procedure's size or intensity, such as the blood volume or program load.",
                "recruitment_separation": "A recruitment plan that keeps the investigator away from people they teach or supervise.",
            }),
        }


# =========================================================================== schema 2: consent document
CONSENT_STUDIES = {
    "phase1_drug": dict(label="a first-in-human dose-escalation study of the investigational drug TRX-517",
                        risks=["liver injury seen in animal studies", "infusion reactions", "risks that are not yet known"],
                        pages=(14, 28), pops={"general_adults": 3, "clinic_patients": 2}),
    "mindfulness_app": dict(label="a six-week study of a mindfulness phone app for stress",
                            risks=["emotional discomfort from reflection exercises", "loss of confidentiality of app data"],
                            pages=(4, 9), pops={"general_adults": 3, "older_adults": 1}),
    "biobank": dict(label="a blood donation for a research biobank with future unspecified use",
                    risks=["bruising or fainting from the blood draw", "re-identification from genetic data",
                           "use of samples in research the donor may object to"], pages=(6, 12),
                    pops={"general_adults": 2, "clinic_patients": 2, "parents": 1, "older_adults": 1}),
    "family_genomics": dict(label="whole-genome sequencing of families with an inherited heart condition",
                            risks=["findings that reveal unexpected family relationships",
                                   "incidental findings about other disease risks", "anxiety from uncertain results"],
                            pages=(10, 20), pops={"clinic_patients": 3, "parents": 2}),
    "memory_mri": dict(label="a research MRI study of memory in older adults",
                       risks=["discomfort in a confined space", "incidental findings on the scan", "loud noise during scanning"],
                       pages=(6, 12), pops={"older_adults": 1}),
    "wearable_sensor": dict(label="a feasibility study of a wearable glucose sensor",
                            risks=["skin irritation under the adhesive", "wrong readings if the sensor malfunctions"],
                            pages=(5, 10), pops={"clinic_patients": 3, "general_adults": 1, "parents": 1}),
}
REQUIRED_ELEMENTS = ["whom to contact with questions or concerns", "that taking part is voluntary and can be stopped",
                     "how long identifiable records are kept", "what happens if a participant is injured",
                     "whether data may be shared for future research", "alternatives to taking part"]
CONSENT_POPS = {"general_adults": "adults from the general community", "parents": "parents consenting for children aged 7-11",
                "older_adults": "adults over 65, some with early memory loss", "clinic_patients": "patients of the study clinic"}


class ConsentDocumentReview(Schema):
    name = "consent_document_review"
    description = "A draft informed-consent document checked against the protocol and the institution's consent standards."
    surfaces = {"json": 0.5, "reviewer_checklist": 0.25, "consent_excerpt": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        st = pick(rng, list(CONSENT_STUDIES))
        s = CONSENT_STUDIES[st]
        pop = pick_w(rng, s["pops"])
        grade = round(between(rng, 6.0, 13.5), 1)
        key_info = pick_w(rng, {"first_page": 4, "buried": 2, "absent": 1})
        missing = [e for e in REQUIRED_ELEMENTS if chance(rng, 0.12)]
        omitted = [r for r in s["risks"] if chance(rng, 0.2)]
        lep = pick(rng, [0, 2, 5, 12, 25, 40])
        translation = pick_w(rng, {True: 2, False: 2, None: 1}) if lep >= 5 else None
        coercive = None
        withheld = chance(rng, 0.1)
        assent = (pick_w(rng, {True: 3, False: 1}) if pop == "parents" else None)
        capacity_plan = (pick_w(rng, {True: 2, False: 2}) if pop == "older_adults" else None)
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                grade, key_info, missing, omitted, withheld = round(between(rng, 6.0, 7.6), 1), "first_page", [], [], False
                lep, translation = pick(rng, [0, 2]), None
                assent = True if pop == "parents" else None
                capacity_plan = True if pop == "older_adults" else None
            else:
                grade, key_info = round(between(rng, 11.5, 14.0), 1), "absent"
                missing = rng.sample(REQUIRED_ELEMENTS, 3)
                omitted = s["risks"][:2]
        elif reg == "borderline":
            grade = round(near(rng, 8.0, 1.2, lo=5.5), 1)
            missing = rng.sample(REQUIRED_ELEMENTS, pick(rng, [0, 0, 1]))
            omitted = rng.sample(s["risks"], pick(rng, [0, 0, 1]))
            lep = pick(rng, [4, 5, 6, 12])
            translation = pick(rng, [False, None]) if lep >= 5 else None
        else:
            variant = pick(rng, ["polished_but_risk_gap", "plain_but_pressure", "complete_but_capacity"])
            if variant == "polished_but_risk_gap":
                grade, key_info, missing = round(between(rng, 6.5, 8.0), 1), "first_page", []
                omitted = [s["risks"][0]]
            elif variant == "plain_but_pressure":
                grade, missing, omitted = round(between(rng, 6.0, 8.0), 1), [], []
                coercive = pick(rng, ["Your doctor strongly recommends you join this study.",
                                      "This study offers you the best possible care.",
                                      "Most patients like you agree to take part."])
                st = pick(rng, ["phase1_drug", "family_genomics", "wearable_sensor", "biobank"])
                s = CONSENT_STUDIES[st]
                pop = "clinic_patients"
            else:
                grade, missing, omitted = round(between(rng, 7.0, 9.0), 1), [], []
                st = pick(rng, ["memory_mri", "biobank", "mindfulness_app"])
                s = CONSENT_STUDIES[st]
                pop, capacity_plan = "older_adults", False
        pages = rng.randint(*s["pages"])
        if pop != "parents":
            assent = None
        if pop != "older_adults":
            capacity_plan = None
        return {
            "regime": reg, "variant": variant, "study_type": st, "study_label": s["label"], "population": pop,
            "grade_level": grade, "grade_target": 8, "pages": pages, "key_info_summary": key_info,
            "missing_elements": missing, "protocol_risks": s["risks"], "risks_omitted": omitted,
            "limited_english_pct": lep, "translation_planned": translation, "pressure_phrase": coercive,
            "payment_withheld_if_withdraw": withheld, "assent_form_included": assent,
            "capacity_assessment_plan": capacity_plan, "reviewer": person(rng), "institution": _inst(rng),
            "protocol": _protocol_no(rng, rand_date(rng)), "version": f"v{rng.randint(1, 4)}.{rng.randint(0, 3)}",
        }

    def _risks_listed(self, f):
        return [r for r in f["protocol_risks"] if r not in f["risks_omitted"]]

    def render(self, f, rng, surface):
        pop = CONSENT_POPS[f["population"]]
        key = {"first_page": "on page 1", "buried": f"on page {max(2, f['pages'] // 2)}", "absent": None}[f["key_info_summary"]]
        lep = f"{f['limited_english_pct']}% of the expected participants have limited English"
        trans = {True: "translated form planned", False: "no translation planned", None: "translation not addressed"}.get(
            f["translation_planned"]) if f["limited_english_pct"] >= 5 else None
        if surface == "json":
            return dumps(prune({
                "document": {"protocol": f["protocol"], "version": f["version"], "study": f["study_label"],
                             "pages": f["pages"], "reading_grade_level": f["grade_level"],
                             "key_information_summary": key or "not present"},
                "content_check": {"elements_not_found": f["missing_elements"] or None,
                                  "risks_listed_in_form": self._risks_listed(f),
                                  "risks_listed_in_protocol": f["protocol_risks"],
                                  "quoted_language_of_concern": f["pressure_phrase"],
                                  "payment_clause": ("payment for completed visits is forfeited if the participant withdraws"
                                                     if f["payment_withheld_if_withdraw"] else None)},
                "population": {"description": pop, "language": lep, "translation": trans,
                               "child_assent_form": f["assent_form_included"],
                               "capacity_assessment_described": f["capacity_assessment_plan"]},
                "standard": f"{f['institution']} consent standard: written at or below grade {f['grade_target']}, key information "
                            f"first; a translated form is required before enrolment when 10% or more of expected "
                            f"participants have limited English.",
            }))
        if surface == "reviewer_checklist":
            yn = lambda b: "Y" if b else "N"  # noqa: E731
            rows = [["Reading level <= grade 8", f"{yn(f['grade_level'] <= 8)} ({f['grade_level']})"],
                    ["Key information up front", key or "no summary"],
                    ["All required elements", "Y" if not f["missing_elements"] else "N: " + "; ".join(f["missing_elements"])],
                    ["Risks match protocol", "Y" if not f["risks_omitted"] else "N: missing " + "; ".join(f["risks_omitted"])],
                    ["Language access", f"{f['limited_english_pct']}%" + (f", {trans}" if trans else "")]]
            if f["assent_form_included"] is not None:
                rows.append(["Child assent form", yn(f["assent_form_included"])])
            if f["capacity_assessment_plan"] is not None:
                rows.append(["Capacity assessment", yn(f["capacity_assessment_plan"])])
            comments = []
            if f["pressure_phrase"]:
                comments.append(f"Form says: \"{f['pressure_phrase']}\"")
            if f["payment_withheld_if_withdraw"]:
                comments.append("Payment for completed visits is lost if the participant withdraws.")
            return (f"CONSENT REVIEW CHECKLIST -- {f['protocol']} {f['version']} ({f['pages']} pp.)\nStudy: {f['study_label']}\n"
                    f"Participants: {pop}\nReviewer: {f['reviewer']}\n\n" + table(["item", "finding"], rows, rng)
                    + ("\n\nComments:\n" + bullets(comments, rng) if comments else "")
                    + "\n\nStandard: grade 8 or below; key information first; translated form before enrolment when "
                      "10% or more of participants have limited English.")
        # consent_excerpt
        para = [f"You are being asked to take part in {f['study_label']}."]
        if f["pressure_phrase"]:
            para.append(f["pressure_phrase"])
        para.append("Possible risks include " + ", ".join(self._risks_listed(f)) + ".")
        if f["payment_withheld_if_withdraw"]:
            para.append("If you leave the study early you will not be paid for the visits you completed.")
        note = [f"Excerpt from {f['protocol']} consent {f['version']}, {f['pages']} pages, grade level {f['grade_level']}.",
                ("No key-information summary." if key is None else f"Key-information summary {key}."),
                (f"Could not find: {'; '.join(f['missing_elements'])}." if f["missing_elements"] else ""),
                f"Protocol section 8 lists: {'; '.join(f['protocol_risks'])}.", f"{lep}" + (f"; {trans}." if trans else "."),
                ({True: "Child assent form attached.", False: "No child assent form attached.", None: ""}[f["assent_form_included"]]),
                ({True: "Capacity assessment described.", False: "No capacity assessment described.",
                  None: ""}[f["capacity_assessment_plan"]])]
        return ("\"" + noisy(join_sentences(para), rng, 0.08) + "\"\n\nReviewer notes: " + join_sentences(note)
                + "\nStandard: a translated form is needed before enrolment when 10% or more of participants have "
                  "limited English.")

    def questions(self):
        return {
            "consent_form_outcome": choice("What should happen to this consent document?", {
                "approve": "Approve the document as written.",
                "approve_pending_minor_edits": "Approve once specific minor edits are made and checked by IRB staff.",
                "require_revision_and_rereview": "Send it back for substantial rewriting and a fresh review.",
                "rethink_consent_process": "Ask for a different consent process altogether, such as a representative, "
                                           "assent or a translated form, not just edits to this text.",
            }),
            "comprehension_barrier": score("How hard would the expected participants find it to understand this form?", [
                "Easy: plain language, key facts first.",
                "Manageable: a few passages need simplifying.",
                "Difficult: long or technical enough that many would miss key points.",
                "Prohibitive: most participants could not understand what they are agreeing to.",
            ]),
            "risk_disclosure_gap": score("How far does the form's risk section fall short of what the protocol describes?", [
                "No gap: every protocol risk appears in plain words.",
                "Small: a minor risk is missing or vaguely worded.",
                "Material: a risk a reasonable person would weigh is missing.",
                "Serious: the most important risk is missing or downplayed.",
            ]),
            "informed_choice_possible": noul(
                "As written, the form lets a participant make a free and informed choice.",
                yes="No required element or protocol risk is missing, there is no pressure wording or withdrawal "
                    "penalty, and any child assent form or capacity assessment the population needs is in place.",
                no="A required element or protocol risk is missing, the form pressures people or penalises "
                   "withdrawal, or a needed child assent form or capacity assessment is absent."),
            "translation_needed_first": noul(
                "A translated consent form is needed before enrolment begins.",
                yes="10% or more of the expected participants have limited English, which the consent standard says "
                    "requires a translated form first.",
                no="Fewer than 10% of the expected participants have limited English."),
        }


# =========================================================================== schema 3: amendment
AMENDMENT_TYPES = {
    "add_procedure": "add a research MRI scan at the final visit",
    "extra_blood": "add two extra blood draws (about 30 mL each) for a new biomarker",
    "increase_sample": "increase the target enrolment",
    "add_adolescents": "extend eligibility to 16- and 17-year-olds",
    "new_safety_info": "report new safety information received from the sponsor",
    "new_site": "add a second recruitment site at a partner clinic",
    "pi_change": "replace the principal investigator, who is leaving the institution",
    "remove_procedure": "drop the optional follow-up interview",
    "share_with_company": "share coded data with a commercial analytics partner",
    "remote_consent": "allow consent by video call instead of in person",
}


class AmendmentReview(Schema):
    name = "amendment_review"
    description = "A modification to an approved study: how it should be routed, and what it means for enrolled participants."
    surfaces = {"json": 0.5, "amendment_form": 0.25, "investigator_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick(rng, list(AMENDMENT_TYPES))
        orig = pick_w(rng, {"exempt": 1, "expedited": 4, "full_board": 3})
        enrolled = rng.randint(0, 180)
        active = min(enrolled, rng.randint(0, 60))
        increase_pct = rng.randint(10, 150) if kind == "increase_sample" else None
        implemented = chance(rng, 0.35)
        consent_updated = pick_w(rng, {True: 4, False: 2, None: 1})
        activation_days = pick(rng, [None, None, 5, 14, 30, 60])
        tracked = chance(rng, 0.8)
        severity = None
        variant = None
        if kind == "new_safety_info":
            severity = pick_w(rng, {"minor": 2, "moderate": 2, "serious": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                kind, implemented, consent_updated, tracked = pick(rng, ["remove_procedure", "pi_change"]), False, True, True
                severity = None
            else:
                kind, severity, implemented = "new_safety_info", "serious", False
                enrolled = max(enrolled, rng.randint(15, 60))
                active = rng.randint(10, enrolled)
                orig = "full_board"
        elif reg == "borderline":
            kind = pick(rng, ["increase_sample", "extra_blood", "new_site", "remote_consent", "share_with_company"])
            if kind == "increase_sample":
                increase_pct = near(rng, 50, 25, lo=5, integer=True)
            severity = None
        else:
            variant = pick(rng, ["already_in_use", "minor_label_major_change", "safety_info_late"])
            if variant == "already_in_use":
                implemented = True
                kind = pick(rng, ["remote_consent", "add_procedure", "new_site"])
            elif variant == "minor_label_major_change":
                kind = pick(rng, ["add_adolescents", "share_with_company", "extra_blood"])
            else:
                kind, severity = "new_safety_info", pick(rng, ["moderate", "serious"])
                activation_days = None
        if kind == "increase_sample" and increase_pct is None:
            increase_pct = rng.randint(10, 150)
        if kind != "new_safety_info":
            severity = None
        # how the paperwork hints at early use: an explicit note, a subtle slip, preparation that only looks like use
        if implemented:
            early_signal = pick_w(rng, {"explicit": 3, "subtle": 2})
        else:
            early_signal = pick_w(rng, {"none": 3, "ambiguous_prep": 1})
        submitted = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "amendment_type": kind, "original_review_level": orig,
            "enrolled_to_date": enrolled, "currently_active": active, "sample_increase_pct": increase_pct,
            "safety_signal": severity, "implemented_before_approval": implemented, "early_use_signal": early_signal,
            "consent_updated": consent_updated, "sponsor_activation_in_days": activation_days,
            "tracked_changes_attached": tracked, "pi_labels_it": pick_w(rng, {"minor": 4, "major": 1, None: 1}),
            "submitted": submitted, "protocol": _protocol_no(rng, submitted), "pi": _pi(rng),
            "amendment_no": f"AM-{rng.randint(1, 12):02d}", "coordinator": person(rng), "institution": _inst(rng),
            "signal_detail": pick(rng, ["two cases of raised liver enzymes at the highest dose",
                                        "a higher-than-expected rate of fainting after infusions",
                                        "one participant with a severe allergic reaction at another site"]),
        }

    def _change_text(self, f):
        t = AMENDMENT_TYPES[f["amendment_type"]]
        if f["amendment_type"] == "increase_sample":
            t += f" by {f['sample_increase_pct']}%"
        if f["safety_signal"]:
            t += f": {f['signal_detail']}"
        return t

    def render(self, f, rng, surface):
        change = self._change_text(f)
        what = "the revised consent form" if f["amendment_type"] == "new_safety_info" else "the change"
        used = {"explicit": f"The site coordinator notes {what} has been in use since "
                            f"{fmt_date(f['submitted'] - timedelta(days=rng.randint(7, 40)), rng)}.",
                "subtle": pick(rng, ["Early feedback from participants on the new arrangement has been positive.",
                                     "The two most recent visits already follow the revised schedule."]),
                "ambiguous_prep": pick(rng, ["Staff have been trained on the new procedure and materials are printed, ready to go.",
                                             "The partner site has completed its onboarding checklist."]),
                "none": None}[f["early_use_signal"]]
        cons = {True: "revised consent attached", False: "consent form not revised", None: "consent impact not stated"}[f["consent_updated"]]
        label = {"minor": "minor", "major": "major", None: None}[f["pi_labels_it"]]
        if surface == "json":
            return dumps(prune({
                "amendment": {"number": f["amendment_no"], "protocol": f["protocol"], "submitted": f["submitted"].isoformat(),
                              "summary": change, "investigator_classification": label, "consent": cons,
                              "tracked_changes_attached": f["tracked_changes_attached"],
                              "sponsor_activation_in_days": f["sponsor_activation_in_days"], "site_note": used},
                "study": {"original_review_level": f["original_review_level"].replace("_", " "),
                          "enrolled_to_date": f["enrolled_to_date"], "participants_currently_on_study": f["currently_active"],
                          "investigator": f["pi"]},
            }))
        if surface == "amendment_form":
            rows = [("Protocol", f["protocol"]), ("Amendment", f["amendment_no"]), ("PI", f["pi"]),
                    ("Submitted", fmt_date(f["submitted"], rng)),
                    ("Originally approved via", f["original_review_level"].replace("_", " ")),
                    ("Change requested", change), ("PI classification", show(label, rng)),
                    ("Consent documents", cons), ("Tracked-changes protocol", "attached" if f["tracked_changes_attached"] else "not attached"),
                    ("Enrolled / on study", f"{f['enrolled_to_date']} / {f['currently_active']}"),
                    ("Sponsor activation date", show(f["sponsor_activation_in_days"], rng, lambda v: f"in {v} days"))]
            return "MODIFICATION REQUEST\n" + kv_lines(rows, rng, skip_none=False) + (f"\n\nNote: {used}" if used else "")
        # investigator_email
        body = [f"Please find attached {f['amendment_no']} for {f['protocol']}. We would like to {change}."]
        if label == "minor":
            body.append("We see this as a minor change.")
        if f["sponsor_activation_in_days"]:
            body.append(f"The sponsor would like this active within {f['sponsor_activation_in_days']} days.")
        body.append(f"So far {f['enrolled_to_date']} people have enrolled and {f['currently_active']} are still on study.")
        if f["amendment_type"] == "new_safety_info":
            body.append({True: "A revised consent form with the new information is attached.",
                         False: "We have not yet revised the consent form; the sponsor's letter is attached.",
                         None: ""}[f["consent_updated"]])
        else:
            body.append({True: "A revised consent form is attached.", False: "We don't think the consent form needs to change.",
                         None: ""}[f["consent_updated"]])
        if f["early_use_signal"] == "explicit":
            body.append(pick(rng, ["The team has already started doing this at the main site to keep things moving.",
                                   "To be transparent, we began this a few weeks ago before realising it needed approval."]))
        elif used:
            body.append(used)
        if not f["tracked_changes_attached"]:
            body.append("I'll send the tracked-changes protocol separately.")
        return (f"Subject: {f['protocol']} {f['amendment_no']} -- modification\nFrom: {f['pi']}\n"
                f"To: irb-office@{f['institution'].split()[0].lower()}.example.edu\n\n"
                + noisy(join_sentences(body), rng, 0.15) + f"\n\n{f['pi']}")

    def questions(self):
        return {
            "amendment_route": choice("How should the IRB office route this modification?", {
                "acknowledge_administratively": "Record it administratively; it does not affect participants or the approved research.",
                "approve_as_minor": "Approve it through the minor-change route with a single reviewer.",
                "refer_to_convened_board": "Send it to the convened board, because it changes risk, population or consent materially.",
                "return_incomplete": "Return it to the investigator as incomplete before any reviewer sees it.",
            }),
            "reconsent_required": noul(
                "Participants already enrolled must be re-consented because of this change.",
                yes="The change alters risks, procedures or data use in a way people still on study would want to know.",
                no="The change does not affect people already enrolled, or none remain on study."),
            "risk_change": score("How much does the modification change what participants face?", [
                "It reduces risk or burden.",
                "It leaves risk and burden essentially unchanged.",
                "It adds modest risk or burden.",
                "It adds substantial risk or a new kind of harm.",
            ]),
            "implemented_early": noul(
                "The change appears to have been put into practice before the IRB approved it.",
                yes="The paperwork shows the change already in use: a note that it is in use, visits already following "
                    "the new schedule, participants' feedback on it, or the team saying it has started.",
                no="Nothing shows it in use; staff training, printed materials or a completed onboarding checklist is "
                   "preparation, not use."),
            "turnaround_need": score("How quickly does the IRB need to act on this modification?", [
                "At the normal pace; nothing depends on the date.",
                "Within a couple of weeks, to meet a planned activation.",
                "Before the next convened meeting, because participants' information or safety is involved.",
                "Right away; people on study may be exposed to a newly identified risk now.",
            ]),
        }


# =========================================================================== schema 4: continuing review
RENEWAL_RULE = ("HRPP rule 9.3: a board-approved study may be renewed by a single reviewer only when enrolment is "
                "permanently closed, all participants have finished study interventions and no new risks have been "
                "identified; otherwise the convened board renews it. Research may not continue past the expiry date.")


class ContinuingReview(Schema):
    name = "continuing_review"
    description = "An annual renewal report for an approved study, due before its approval expires."
    surfaces = {"json": 0.5, "progress_report": 0.25, "analyst_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        status = pick_w(rng, {"enrolling": 5, "follow_up_only": 3, "analysis_only": 2})
        target = pick(rng, [40, 60, 120, 300])
        frac_time = round(between(rng, 0.3, 0.95), 2)
        enrolled = int(target * min(1.2, max(0.05, rng.gauss(frac_time, 0.25))))
        withdrawals = rng.randint(0, max(1, enrolled // 6))
        wd_reason = pick_w(rng, {"moved_or_lost": 3, "visit_burden": 2, "side_effects": 1, "unstated": 2})
        minor_dev = rng.randint(0, 6)
        major_dev = pick_w(rng, {0: 6, 1: 2, 2: 1})
        major_kind = pick(rng, ["consented with an outdated form version", "enrolled one person who did not meet eligibility",
                                "missed a scheduled safety lab", "stored identifiable files on an unapproved drive"])
        complaints = pick_w(rng, {0: 6, 1: 2, 2: 1})
        new_risk = chance(rng, 0.12)
        orig = pick_w(rng, {"expedited": 4, "full_board": 4})
        days_to_expiry = pick(rng, [4, 9, 16, 28, 45, 70])
        next_meeting = rng.randint(3, 30)
        gaps = [g for g in ["enrolment table", "deviation log", "current consent version", "summary of findings"] if chance(rng, 0.1)]
        pi_status = pick_w(rng, {"active": 7, "on_leave": 1, "left_institution": 1})
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                major_dev, complaints, new_risk, gaps, pi_status, days_to_expiry = 0, 0, False, [], "active", pick(rng, [45, 70])
                withdrawals = min(withdrawals, 2)
            else:
                major_dev, gaps, days_to_expiry, pi_status = 2, ["deviation log", "enrolment table"], pick(rng, [4, 9]), "left_institution"
                complaints = pick(rng, [1, 2])
        elif reg == "borderline":
            days_to_expiry = near(rng, next_meeting, 6, lo=2, integer=True)
            status = pick(rng, ["follow_up_only", "analysis_only", "enrolling"])
            major_dev = pick(rng, [0, 1])
        else:
            variant = pick(rng, ["clean_report_one_serious", "analysis_only_but_new_risk", "strong_enrolment_high_withdrawal"])
            if variant == "clean_report_one_serious":
                minor_dev, major_dev, complaints = 0, 1, 0
                major_kind = pick(rng, ["stored identifiable files on an unapproved drive", "consented with an outdated form version"])
            elif variant == "analysis_only_but_new_risk":
                status, new_risk, orig = "analysis_only", True, "full_board"
            else:
                enrolled = int(target * rng.uniform(0.8, 1.1))
                withdrawals = int(enrolled * rng.uniform(0.25, 0.4))
                wd_reason = pick(rng, ["visit_burden", "side_effects"])
        if status != "enrolling":
            enrolled = max(enrolled, int(target * rng.uniform(0.6, 1.0)))
        title = pick(rng, ["Home blood-pressure coaching after stroke", "Peer mentoring for first-generation students",
                           "A low-dose iron supplement in pregnancy", "Balance training for adults over 70",
                           "Text-message reminders for asthma inhaler use"])
        if wd_reason == "side_effects" and title not in ("A low-dose iron supplement in pregnancy",
                                                         "Balance training for adults over 70"):
            wd_reason = "visit_burden"
        received = rand_date(rng)
        return {
            "regime": reg, "variant": variant, "status": status, "target_n": target, "enrolled": enrolled,
            "fraction_of_period_elapsed": frac_time, "withdrawals": min(withdrawals, enrolled),
            "withdrawal_reason": wd_reason, "minor_deviations": minor_dev, "major_deviations": major_dev,
            "major_deviation_kind": major_kind if major_dev else None, "complaints": complaints,
            "complaint_text": pick(rng, ["repeated reminder calls after saying no", "rude treatment at a study visit",
                                         "payment arrived months late"]) if complaints else None,
            "new_risk_information": new_risk, "original_review_level": orig, "days_to_expiry": days_to_expiry,
            "next_board_meeting_days": next_meeting, "report_gaps": gaps, "pi_status": pi_status,
            "received": received, "protocol": _protocol_no(rng, received), "pi": _pi(rng), "analyst": person(rng),
            "title": title,
        }

    def _wd_text(self, f):
        se = "strains or minor falls in sessions" if f["title"].startswith("Balance") else "side effects"
        return {"moved_or_lost": "moved away or lost to follow-up", "visit_burden": "too many visits",
                "side_effects": se, "unstated": "no reason recorded"}[f["withdrawal_reason"]]

    def render(self, f, rng, surface):
        expiry = f["received"] + timedelta(days=f["days_to_expiry"])
        meeting = f["received"] + timedelta(days=f["next_board_meeting_days"])
        status = {"enrolling": "open to enrolment", "follow_up_only": "closed to enrolment; participants in follow-up visits",
                  "analysis_only": "closed to enrolment; data analysis only"}[f["status"]]
        pi = {"active": None, "on_leave": "PI on sabbatical; co-investigator running day-to-day",
              "left_institution": "PI has left the institution; no replacement named"}[f["pi_status"]]
        if surface == "json":
            return dumps(prune({
                "study": {"protocol": f["protocol"], "title": f["title"], "investigator": f["pi"], "status": status,
                          "approved_via": f["original_review_level"].replace("_", " "), "investigator_status": pi},
                "progress": {"enrolled": f["enrolled"], "target": f["target_n"], "withdrawn": f["withdrawals"],
                             "withdrawal_reasons": self._wd_text(f) if f["withdrawals"] else None,
                             "minor_deviations": f["minor_deviations"], "major_deviations": f["major_deviations"],
                             "major_deviation_detail": f["major_deviation_kind"], "complaints": f["complaints"],
                             "complaint_detail": f["complaint_text"],
                             "new_risk_information": ("new risk information reported this year" if f["new_risk_information"] else "none")},
                "timeline": {"report_received": f["received"].isoformat(), "approval_expires": expiry.isoformat(),
                             "next_board_meeting": meeting.isoformat()},
                "report_missing": f["report_gaps"] or None,
                "rule": RENEWAL_RULE,
            }))
        if surface == "progress_report":
            rows = [("Protocol", f["protocol"]), ("Title", f["title"]), ("PI", f["pi"]), ("Status", status),
                    ("Approved via", f["original_review_level"].replace("_", " ")),
                    ("Enrolled / target", f"{f['enrolled']} / {f['target_n']}"),
                    ("Withdrawals", f"{f['withdrawals']}" + (f" ({self._wd_text(f)})" if f["withdrawals"] else "")),
                    ("Deviations (minor / major)", f"{f['minor_deviations']} / {f['major_deviations']}"),
                    ("Major deviation", f["major_deviation_kind"]), ("Complaints", f["complaints"]),
                    ("New risk information", "yes" if f["new_risk_information"] else "no"),
                    ("Approval expires", fmt_date(expiry, rng))]
            missing = ("\nSections left blank: " + ", ".join(f["report_gaps"])) if f["report_gaps"] else ""
            return ("CONTINUING REVIEW -- PROGRESS REPORT\n" + kv_lines(rows, rng) + missing
                    + (f"\nComplaint: {f['complaint_text']}" if f["complaint_text"] else "") + (f"\nNote: {pi}" if pi else "")
                    + f"\nNext board meeting: {fmt_date(meeting, rng)}\n\n{RENEWAL_RULE}")
        para = [f"{f['protocol']} ({f['title']}) is {status}; originally approved via "
                f"{f['original_review_level'].replace('_', ' ')}.",
                f"Report received {fmt_date(f['received'], rng)}; approval expires {fmt_date(expiry, rng)}, "
                f"next board meeting {fmt_date(meeting, rng)}.",
                f"Enrolment {f['enrolled']} of {f['target_n']}, {f['withdrawals']} withdrawn"
                + (f" (mostly {self._wd_text(f)})." if f["withdrawals"] else "."),
                f"{f['minor_deviations']} minor deviation(s)" + (f" and {f['major_deviations']} major: {f['major_deviation_kind']}."
                                                                if f["major_deviations"] else "."),
                (f"{f['complaints']} complaint(s): {f['complaint_text']}." if f["complaints"] else ""),
                ("New risk information was reported during the year." if f["new_risk_information"] else ""),
                (pi + "." if pi else ""),
                (f"Missing from the packet: {', '.join(f['report_gaps'])}." if f["report_gaps"] else "")]
        return (f"MEMO -- continuing review pre-check ({f['analyst']})\n\n" + noisy(join_sentences(para), rng, 0.1)
                + f"\n\n{RENEWAL_RULE}")

    def questions(self):
        return {
            "renewal_decision": choice("What should the IRB decide on this renewal?", {
                "approve_renewal": "Renew approval for another year as the study stands.",
                "approve_with_conditions": "Renew, with conditions such as a corrective plan or a missing document within a set time.",
                "pause_new_enrolment": "Renew follow-up of current participants but stop new enrolment until concerns are resolved.",
                "refer_noncompliance_review": "Refer the study for a noncompliance review before any renewal is granted.",
            }),
            "approval_lapse_risk": score("How likely is the study to pass its expiry date without a completed renewal?", [
                "Unlikely: a complete packet with more than three weeks to expiry.",
                "Possible: three weeks or less to expiry, but the packet is complete and the reviewer it needs can act "
                "in time.",
                "Likely: the packet has gaps, or the convened board must renew it and its next meeting falls after "
                "expiry.",
                "Near certain: a week or less to expiry, with gaps in the packet or a board renewal that cannot happen "
                "before then.",
            ]),
            "welfare_signal": score("How strongly do withdrawals, complaints and deviations point to problems for participants?", [
                "Not at all: numbers look ordinary for this kind of study.",
                "Faintly: one item deserves a question.",
                "Clearly: a pattern suggests burden, harm or disrespect.",
                "Strongly: participants' safety or rights may already have been compromised.",
            ]),
            "single_reviewer_renewal": noul(
                "Institutional rules allow this renewal to be handled by a single reviewer.",
                yes="The study was approved via the lighter route, or has reached a stage the renewal rule lets one reviewer handle.",
                no="The rule requires the convened board, for example because interventions continue or new risks appeared."),
            "investigator_oversight_adequate": noul("The investigator is maintaining adequate oversight of the study."),
        }


# =========================================================================== schema 5: recruitment and payment
RECRUIT_POPS = {"undergraduates": "undergraduate students", "low_income": "adults in a low-income neighbourhood",
                "unhoused": "adults using a homeless shelter", "clinic_patients": "patients of the investigator's clinic",
                "general": "adults from the general public", "employees": "employees of a partner company"}
ROUTES = {"flyers": "posters on public notice boards", "social_ads": "social media ads targeted at people who follow condition-related pages",
          "own_patients": "the investigator invites their own patients during clinic visits",
          "own_class": "the instructor announces the study in their own lecture",
          "registry": "letters to people in a volunteer registry who agreed to be contacted",
          "records_cold_call": "research staff phone people identified from medical records",
          "manager_email": "an email from participants' line managers",
          "shelter_staff": "shelter case workers mention the study to residents they support"}
ROUTES_FOR = {"undergraduates": ["flyers", "social_ads", "own_class"], "low_income": ["flyers", "social_ads", "registry"],
              "unhoused": ["flyers", "shelter_staff"], "clinic_patients": ["own_patients", "records_cold_call", "flyers", "registry"],
              "general": ["flyers", "social_ads", "registry"], "employees": ["manager_email", "flyers"]}
PROCEDURES = {
    "questionnaires": dict(text="questionnaires only", hours=(1, 3), visits=(1, 2)),
    "sleep_lab": dict(text="an overnight stay in a sleep lab", hours=(12, 16), visits=(1, 2)),
    "mri_cognitive": dict(text="an MRI scan and memory tests", hours=(2, 4), visits=(1, 2)),
    "blood_fitness": dict(text="blood draws and a treadmill fitness test", hours=(3, 6), visits=(2, 4)),
    "investigational_medicine": dict(text="taking an investigational medicine for 8 weeks, with clinic check-ups",
                                     hours=(12, 30), visits=(5, 10)),
    "exercise_program": dict(text="a supervised 6-week exercise program", hours=(10, 20), visits=(6, 12)),
}
PAY_GUIDE = ("Institutional payment guidance: $20-$40 per hour of participant time; payment should accrue per visit, with "
             "no more than 25% held back as a completion bonus; recruitment material must not make payment the main message.")


class RecruitmentCompensationReview(Schema):
    name = "recruitment_compensation_review"
    description = "A recruitment and participant-payment plan, with its advert text, checked for undue influence and pressure."
    surfaces = {"json": 0.5, "flyer_and_schedule": 0.25, "reviewer_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        pop = pick(rng, list(RECRUIT_POPS))
        route = pick(rng, list(ROUTES))
        proc = pick(rng, list(PROCEDURES))
        rate = between(rng, 12, 70)
        bonus_share = pick(rng, [0, 0, 10, 20, 25, 30, 50])
        claim = pick_w(rng, {None: 4, "treatment": 1, "cash_headline": 2})
        lottery = chance(rng, 0.15)
        variant = None
        if reg == "clear":
            if chance(rng, 0.5):
                pop, route, rate, bonus_share, claim = "general", pick(rng, ["flyers", "registry"]), between(rng, 22, 35), pick(rng, [0, 10]), None
                proc = pick(rng, ["questionnaires", "mri_cognitive", "blood_fitness"])
            else:
                pop, rate, bonus_share, claim = pick(rng, ["unhoused", "low_income"]), between(rng, 90, 160), 60, "cash_headline"
                proc = pick(rng, ["investigational_medicine", "sleep_lab"])
        elif reg == "borderline":
            rate = near(rng, pick(rng, [20, 40]), 5, lo=8)
            bonus_share = near(rng, 25, 8, lo=0, integer=True)
        else:
            variant = pick(rng, ["fair_pay_coercive_route", "high_pay_low_burden", "modest_pay_big_claims"])
            if variant == "fair_pay_coercive_route":
                rate, route = between(rng, 22, 38), pick(rng, ["own_patients", "own_class", "manager_email"])
                pop = {"own_patients": "clinic_patients", "own_class": "undergraduates", "manager_email": "employees"}[route]
            elif variant == "high_pay_low_burden":
                rate, proc = between(rng, 60, 110), "questionnaires"
                pop = pick(rng, ["low_income", "undergraduates"])
            else:
                rate, claim, proc = between(rng, 15, 30), "treatment", pick(rng, ["investigational_medicine", "exercise_program"])
        if route not in ROUTES_FOR[pop]:
            route = pick(rng, ROUTES_FOR[pop])
        if claim == "treatment" and proc not in ("investigational_medicine", "exercise_program"):
            claim = pick(rng, [None, "cash_headline"])
        p = PROCEDURES[proc]
        hours = rng.randint(*p["hours"])
        visits = rng.randint(*p["visits"])
        credit_alt = pick_w(rng, {True: 1, False: 1}) if pop == "undergraduates" else None
        total = max(10, round(rate * hours / 5) * 5)
        return {
            "regime": reg, "variant": variant, "population": pop, "route": route, "procedure_key": proc,
            "procedures": p["text"], "hours": hours, "visits": visits,
            "hourly_equivalent": round(total / hours, 2), "total_payment": total, "completion_bonus_pct": bonus_share,
            "ad_claim": claim, "lottery": lottery, "course_credit_alternative": credit_alt,
            "protocol": _protocol_no(rng, rand_date(rng)), "pi": _pi(rng), "reviewer": person(rng),
            "study_name": pick(rng, ["The PATHWAYS study", "Project Stride", "The BRIGHT study", "The Kestrel study",
                                     "The CALM-2 study", "The Lantern study"]),
            "contact": f"study-team@{pick(rng, ['research', 'trials', 'hrpp'])}.example.edu",
        }

    def _headline(self, f):
        pay = money(f["total_payment"], None, style="sym")
        treat = ("Be among the first to try a breakthrough new medicine" if f["procedure_key"] == "investigational_medicine"
                 else "Get FREE expert coaching that could transform your health")
        return {None: f"Volunteers needed for {f['study_name'].replace('The ', 'the ')}",
                "treatment": treat, "cash_headline": f"EARN UP TO {pay}!"}[f["ad_claim"]]

    def _ad_body(self, f):
        return (f"{f['study_name']} is looking for {RECRUIT_POPS[f['population']]}. Taking part involves {f['procedures']} "
                f"over {_n(f['visits'], 'visit')}, about {_n(f['hours'], 'hour')} in total.")

    def _rate(self, f):
        return f"{money(f['total_payment'], None, style='sym')} in total, about ${f['hourly_equivalent']:.0f} per hour of participant time"

    def _schedule_rows(self, f):
        per_visit = int(f["total_payment"] * (1 - f["completion_bonus_pct"] / 100) / f["visits"])
        bonus = f["total_payment"] - per_visit * f["visits"]
        rows = [[f"Visit {i + 1}", money(per_visit, None, style="sym")] for i in range(min(f["visits"], 4))]
        if f["visits"] > 4:
            rows.append([f"Visits 5-{f['visits']}", f"{money(per_visit, None, style='sym')} each"])
        if bonus:
            rows.append(["Completion bonus", money(bonus, None, style="sym")])
        return rows

    def render(self, f, rng, surface):
        pop, route = RECRUIT_POPS[f["population"]], ROUTES[f["route"]]
        credit = {True: "course credit available with an equal-effort alternative assignment",
                  False: "course credit offered, no alternative assignment", None: None}[f["course_credit_alternative"]]
        lottery = "entry into a draw for a tablet computer" if f["lottery"] else None
        if surface == "json":
            return dumps(prune({
                "plan": {"protocol": f["protocol"], "study": f["study_name"], "population": pop, "recruitment_route": route,
                         "procedures": f["procedures"], "time_commitment_hours": f["hours"], "visits": f["visits"]},
                "payment": {"total_usd": f["total_payment"], "per_hour_usd": round(f["hourly_equivalent"]),
                            "completion_bonus_percent": f["completion_bonus_pct"],
                            "extras": [x for x in (credit, lottery) if x] or None},
                "advert": {"headline": self._headline(f), "body": self._ad_body(f), "contact": f["contact"]},
                "guidance": PAY_GUIDE,
            }))
        if surface == "flyer_and_schedule":
            flyer = [self._headline(f), "", f"{f['study_name']} is looking for {pop}.",
                     f"Taking part involves {f['procedures']} over {_n(f['visits'], 'visit')}, about {_n(f['hours'], 'hour')} in total.",
                     f"Contact {f['contact']}."]
            return ("=== ADVERT (draft) ===\n" + "\n".join(flyer) + "\n\n=== PAYMENT SCHEDULE ===\n"
                    + table(["item", "amount"], self._schedule_rows(f), rng)
                    + (f"\nAlternatively: {credit}" if credit else "") + (f"\nAlso: {lottery}" if lottery else "")
                    + f"\nThat is {self._rate(f)}."
                    + f"\n\nRecruitment route: {route}.\n{PAY_GUIDE}")
        notes = [f"{f['protocol']} ({f['study_name']}), PI {f['pi']}.",
                 f"Recruiting {pop}. Route: {route}.",
                 f"Burden: {f['procedures']}; {_n(f['visits'], 'visit')}, ~{f['hours']} h.",
                 f"Pays {self._rate(f)}, {f['completion_bonus_pct']}% of it only on completion.",
                 f"Ad headline reads \"{self._headline(f)}\"; body: \"{self._ad_body(f)}\"",
                 (credit[0].upper() + credit[1:] + "." if credit else ""), (lottery.capitalize() + "." if lottery else "")]
        return (f"REVIEWER NOTES -- {f['reviewer']}\n\n" + noisy(join_sentences(notes), rng, 0.12) + f"\n\n{PAY_GUIDE}")

    def questions(self):
        return {
            "recruitment_plan_verdict": choice("What should the IRB require of this recruitment and payment plan?", {
                "approve": "Approve as submitted: a neutral advert, pay within $20-$40 an hour with no more than 25% held "
                           "to completion, and an approach free of pressure or intrusion.",
                "approve_with_wording_edits": "Approve once the advert headline stops promising treatment or making "
                                              "money the main message; payment and approach are within guidance.",
                "restructure_payment": "Require a different payment structure: more than $40 an hour, or more than 25% "
                                       "held back as a completion bonus.",
                "change_recruitment_route": "Require a different approach: the person's own teacher, clinician or "
                                            "manager does the asking, or staff cold-call people found in medical records.",
            }),
            "undue_influence": score("How likely is the payment to push people into taking part against their better judgement?", [
                "Not at all: token or no payment.",
                "Unlikely: modest payment in line with the time asked.",
                "Slight: generous, but unlikely to override anyone's judgement.",
                "Real: the amount or its structure could sway people who would otherwise decline.",
                "Strong: for this population the payment could override concern about the risks.",
            ]),
            "advert_misleading": noul(
                "The advert overstates the benefits of taking part or understates what it involves.",
                yes="The headline promises a breakthrough treatment, free health benefits, or easy money.",
                no="The headline simply asks for volunteers and the body states the procedures and time involved."),
            "pressure_from_authority": noul(
                "The way people are approached puts them under pressure from someone with power over them.",
                yes="A teacher, treating clinician or manager is doing the asking, with no buffer between them.",
                no="People are approached by someone with no hold over their grades, care or job."),
            "contact_intrusiveness": score("How intrusive is the way people are identified and first contacted?", [
                "Not intrusive: people come forward on their own.",
                "Mildly: contact is based on a prior agreement to be approached.",
                "Noticeably: people are singled out using information about them.",
                "Highly: private records are used to make unsolicited contact.",
            ]),
        }


WORKFLOW = Workflow(
    name="irb_protocol_review",
    title="IRB protocol review",
    description="How an institutional review board should handle human-research protocols: review level, consent "
                "documents, modifications, annual renewals and recruitment and payment plans.",
    domain="research ethics / human research protection",
    schemas=[InitialReviewPathway(), ConsentDocumentReview(), AmendmentReview(), ContinuingReview(),
             RecruitmentCompensationReview()],
    notes="Institutions, studies, drug codes and rules are fictional; the rules are invented institutional policy, "
          "not regulation. No eligibility screening, adverse-event or deviation adjudication (clinical_trial_screening "
          "covers those from the sponsor side).",
)
