"""Professional licence discipline: complaints against licensed nurses, engineers and contractors, judged by a
FICTIONAL licensing board against its written practice act and sanctioning guidelines.

Six schemas: an investigated nursing complaint, an engineer's sealed-work complaint, a contractor's consumer
complaint, the board staff's intake screen of a new complaint, a negotiated consent agreement put to the board,
and a probationer's compliance review. Every state quotes the rules it is judged against (guideline steps,
deposit caps, abandonment periods, time limits, early-termination fractions), and those rules are sampled per
case, so each answer comes from comparing facts with the rule.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'). Borderline cases sit near a stated
line (a deposit a few dollars over the cap, a project idle 43 days against a 45-day abandonment rule); conflicting
cases pair a sympathetic licensee with damning evidence, or a confident respondent with a weak file. `intended(f)`
records the answer logic a careful board reviewer would apply to the RENDERED facts; the builder does not use it.

All boards, jurisdictions, licensees, complainants and firms are fictional. Outcomes turn on conduct, evidence,
harm, history and remediation only: no protected attribute (age, sex, race, religion, disability, nationality) is
a factor, and health conditions of licensees are never the basis of a decision.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    ORG_A,
    REAL_NAME_CLASHES,
    Schema,
    Workflow,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    lognormal_amount,
    money,
    near,
    noisy,
    noul,
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

JURIS = ["Province of Kelvara", "State of Marlbridge", "Commonwealth of Ostwick", "State of Penhallow",
         "Territory of Greywater", "Province of Rookhaven", "State of Ambermoor", "State of Corrandale",
         "Province of Wexmoor", "State of Lanthorne", "Commonwealth of Haverleigh", "Territory of Brackenridge"]
BOARD = {"nursing": "Board of Nursing", "engineering": "Board of Professional Engineers",
         "contractors": "Contractors' Licensing Board", "pharmacy": "Board of Pharmacy",
         "home_inspection": "Home Inspector Licensing Board"}
ACT = {"nursing": "Nurse Practice Act", "engineering": "Engineering Practice Act",
       "contractors": "Residential Contractors Act", "pharmacy": "Pharmacy Practice Act",
       "home_inspection": "Home Inspection Licensing Act"}


def _short(j):
    return j.split(" of ")[-1]


def _board(rng, kind):
    j = pick(rng, JURIS)
    return {"jurisdiction": j, "board": f"{_short(j)} {BOARD[kind]}", "act": f"{_short(j)} {ACT[kind]}",
            "sec": pick(rng, ["s. ", "§ ", "Rule "]), "base": rng.randint(4, 31)}


def _cite(b, k):
    return f"{b['sec']}{b['base'] + k}"


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _yrs(n):
    return f"{n} year{'' if n == 1 else 's'}"


FIRM_B = ["Builders", "Construction", "Home Improvement", "Renovations", "Roofing", "Contracting", "Remodeling"]


def _firm(rng, suffixes=FIRM_B):
    a, i = pick(rng, ORG_A), rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        name = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if name not in REAL_NAME_CLASHES:
            return name
    return name


PRIOR = {"none": None, "letter": "letter of concern", "probation": "probation (completed)"}
OUTCOME_STEPS = ["dismiss_complaint", "letter_of_concern", "probation_with_terms", "suspension"]


# =========================================================================== schema 1
# allegation -> (text, allowed culpability, cue group, harm category, allowed harm levels)
NURSE_ALLEGATIONS = {
    "wrong_dose": ("gave {drug} at {mult} times the prescribed dose", ["inadvertent", "at_risk", "reckless"], "med", "clinical", [1, 2, 3]),
    "wrong_patient": ("gave {drug} to the wrong patient", ["inadvertent", "at_risk", "reckless"], "med", "clinical", [1, 2, 3]),
    "pump_rate": ("ran an infusion of {drug} at the wrong rate", ["inadvertent", "at_risk", "reckless"], "med", "clinical", [1, 2, 3]),
    "beyond_scope": ("changed a patient's {drug} dose without an order", ["at_risk", "reckless"], "med", "clinical", [1, 2, 3]),
    "missed_reassessment": ("did not reassess a post-operative patient for {hours} hours when hourly checks were ordered", ["at_risk", "reckless"], "assess", "clinical", [1, 2, 3]),
    "failure_to_escalate": ("recorded falling oxygen levels but did not call the physician or the rapid-response team", ["at_risk", "reckless"], "assess", "clinical", [1, 2, 3]),
    "improper_delegation": ("delegated sterile wound care to an aide who had not been trained in it", ["at_risk", "reckless"], "assess", "clinical", [1, 2]),
    "left_shift": ("left the unit mid-shift without handing over {n} patients", ["reckless"], "assess", "clinical", [1, 2, 3]),
    "falsified_charting": ("charted {care} that badge and camera records show were never done", ["intentional"], "records", "clinical", [0, 1, 2]),
    "late_documentation": ("completed a whole shift's charting two days later from memory", ["at_risk", "reckless"], "records", "records", [0, 1]),
    "unwitnessed_waste": ("recorded {n} unwitnessed wastes of {opioid} and left count discrepancies unresolved", ["reckless", "intentional"], "records", "records", [0, 1]),
    "lapsed_licence": ("worked {n} shifts after the licence lapsed for non-renewal", ["inadvertent", "at_risk"], "licence", "records", [0]),
    "privacy_post": ("posted a photo of a patient's wound on a public social-media account", ["at_risk", "reckless"], "conduct", "dignity", [1]),
    "rough_handling": ("handled a resident roughly and spoke to them abusively, as two aides witnessed", ["reckless", "intentional"], "conduct", "dignity", [1, 2]),
}
DRUGS = ["insulin", "heparin", "morphine", "hydromorphone", "potassium chloride", "enoxaparin", "digoxin",
         "metoprolol", "vancomycin", "magnesium sulphate", "warfarin", "labetalol"]
OPIOIDS = ["hydromorphone", "fentanyl", "oxycodone", "morphine"]
CARE = ["two-hourly repositioning", "wound dressing changes", "blood-glucose checks", "fall-risk rounds",
        "catheter care", "neurological observations"]
# (cue text, does it point at a system failure rather than the nurse?)
CUES = {
    ("med", "inadvertent"): [("a look-alike vial was stored in the same drawer", True),
                             ("the order had been transcribed wrongly before it reached the nurse", True),
                             ("the nurse was covering an unfamiliar unit that night", False),
                             ("the pump's drug library had not been updated for the new concentration", True)],
    ("med", "at_risk"): [("the nurse skipped the second-nurse check because the unit was two staff short", False),
                         ("the nurse overrode the pump alert, saying alerts fire constantly", False),
                         ("the nurse did not scan the wristband, a step colleagues said was often skipped", False)],
    ("med", "reckless"): [("the nurse had been coached twice on the same check in the past year", False),
                          ("the pharmacist queried the dose and the nurse gave it anyway", False),
                          ("a colleague warned the nurse at the time", False)],
    ("assess", "at_risk"): [("the nurse said the ward was too busy to keep up with checks", False),
                            ("the nurse assumed the previous shift had already escalated", False),
                            ("the nurse relied on an aide's report without checking", False)],
    ("assess", "reckless"): [("an aide told the nurse twice that the patient looked unwell", False),
                             ("the unit log shows the nurse on a personal call during the gap", False),
                             ("the nurse had a written warning about the same lapse last year", False)],
    ("records", "at_risk"): [("the nurse said end-of-shift charting was normal practice on the unit", False),
                             ("the electronic record was down for part of the shift", True)],
    ("records", "reckless"): [("the nurse had been told in writing to stop batching entries", False),
                              ("the wastes were left for the next shift to co-sign", False)],
    ("records", "intentional"): [("the nurse later altered entries to hide the gap", False),
                                 ("the nurse asked a colleague to co-sign a waste the colleague did not see", False),
                                 ("entries are timed after the nurse had badged out of the building", False)],
    ("licence", "inadvertent"): [("the renewal reminder went to an address the nurse had moved from", True),
                                 ("the online renewal failed and the nurse believed it had gone through", True)],
    ("licence", "at_risk"): [("the nurse knew renewal was overdue but kept working while the paperwork was pending", False)],
    ("conduct", "at_risk"): [("the nurse thought blurring the face made the photo anonymous", False),
                            ("the nurse said the account was meant for colleagues only", False)],
    ("conduct", "reckless"): [("the nurse said it was a joke among colleagues", False),
                              ("the nurse had completed the employer's training on this the month before", False),
                              ("a manager had warned the nurse about the same behaviour", False)],
    ("conduct", "intentional"): [("the nurse carried on after a colleague intervened", False),
                                 ("the nurse deleted messages before the employer's interview", False)],
}
HARM_TEXT = {
    ("clinical", 0): ["no patient was harmed", "the patients involved showed no ill effects on review"],
    ("clinical", 1): ["the patient was monitored and showed no ill effects", "the error was caught at the next check and the patient was unharmed",
                      "the patient's observations stayed normal"],
    ("clinical", 2): ["the patient needed {rescue} and recovered within a day", "the patient's stay was extended by two days",
                      "the patient was moved to a monitored bed overnight and recovered"],
    ("clinical", 3): ["the patient was transferred to intensive care and has lasting kidney damage",
                      "the patient fell while unattended and fractured a hip",
                      "the patient had a cardiac arrest and was resuscitated, with lasting effects"],
    ("records", 0): ["no patient care was affected", "no patient was harmed"],
    ("records", 1): ["the next shift relied on the record and nearly repeated a dose", "a patient's pain relief was delayed while counts were reconciled"],
    ("dignity", 1): ["the patient was distressed but not physically hurt", "the resident was frightened but uninjured"],
    ("dignity", 2): ["the resident had bruising on both forearms", "the resident needed treatment for a skin tear"],
}
RESCUE = ["a reversal agent", "an emergency glucose infusion", "extra monitoring and fluids", "a blood transfusion"]
SETTINGS = ["a surgical ward at a regional hospital", "a long-term care home", "a home-health agency", "a day-surgery centre",
            "an emergency department", "a rehabilitation unit", "a school-health service", "a staffing agency placement",
            "a hospice", "a mental-health inpatient unit", "a dialysis clinic", "a cardiac step-down unit"]
EVIDENCE = {"strong": ["the medication record, the dispensing-cabinet audit trail and two witness statements",
                       "camera footage, badge logs and the patient record", "the patient record, the pump log and the nurse's own admission"],
            "moderate": ["the complainant's account and partial records", "one witness statement and the shift roster",
                         "the patient record, though the key entry is ambiguous"],
            "weak": ["an anonymous complaint and a shift log the nurse disputes", "a family member's account with no supporting records",
                     "a second-hand report from a former colleague"]}
NURSE_RESPONSE = {"admits_remediated": "admits the conduct, completed a relevant course unprompted and wrote a reflective statement",
                  "admits": "admits the conduct and apologises",
                  "partial": "accepts partial responsibility and points to short staffing",
                  "denies": "denies the allegation",
                  "no_response": "did not respond to two letters from the board"}


class NursingComplaintDisposition(Schema):
    name = "nursing_complaint_disposition"
    description = "An investigated complaint against a nurse, put to the board for an outcome under its sanctioning guidelines."
    surfaces = {"json": 0.5, "investigator_report": 0.25, "board_agenda_item": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        b = _board(rng, "nursing")
        key = pick(rng, list(NURSE_ALLEGATIONS))
        _, culps, _, _, harms = NURSE_ALLEGATIONS[key]
        culp = pick(rng, culps, [3, 2, 1][:len(culps)])
        harm = pick(rng, harms, [{0: 3, 1: 4, 2: 3, 3: 1}[h] for h in harms])
        evidence = pick_w(rng, {"strong": 4, "moderate": 4, "weak": 1})
        prior = pick_w(rng, {"none": 7, "letter": 2, "probation": 1})
        response = pick(rng, list(NURSE_RESPONSE))
        if reg == "clear":
            if chance(rng, 0.5):
                key = pick(rng, ["falsified_charting", "unwitnessed_waste", "left_shift", "rough_handling", "wrong_dose"])
                _, culps, _, _, harms = NURSE_ALLEGATIONS[key]
                culp, harm, evidence = culps[-1], harms[-1], "strong"
                response = pick(rng, ["denies", "no_response"])
            else:
                key = pick(rng, ["wrong_dose", "lapsed_licence", "pump_rate", "late_documentation"])
                _, culps, _, _, harms = NURSE_ALLEGATIONS[key]
                culp, harm = culps[0], harms[0]
                evidence = pick(rng, ["weak", "moderate", "moderate"])
                prior, response = "none", pick(rng, ["admits_remediated", "admits"])
        elif reg == "conflicting":
            twist = pick(rng, ["sympathetic_serious", "confident_weak", "clean_record_deliberate"])
            if twist == "sympathetic_serious":
                culp, harm, evidence, response = pick(rng, [c for c in culps if c != "intentional"] or culps), harms[-1], "strong", "admits_remediated"
            elif twist == "confident_weak":
                evidence, response = "weak", "denies"
            else:
                key = pick(rng, ["falsified_charting", "unwitnessed_waste", "privacy_post"])
                _, culps, _, _, harms = NURSE_ALLEGATIONS[key]
                culp, harm, prior, response = culps[-1], pick(rng, harms), "none", "admits"
        group = NURSE_ALLEGATIONS[key][2]
        cue, system = pick(rng, CUES.get((group, culp)) or CUES[(group, NURSE_ALLEGATIONS[key][1][0])])
        hcat = NURSE_ALLEGATIONS[key][3]
        lic = pick_w(rng, {"registered nurse": 5, "practical nurse": 3, "nurse practitioner": 1})
        return {"regime": reg, "board": b, "allegation": key, "culpability": culp, "harm": harm, "cue": cue,
                "system_factor": system, "evidence": evidence, "evidence_text": pick(rng, EVIDENCE[evidence]),
                "prior": prior, "prior_year": rng.randint(2016, 2024), "response": response,
                "licence_type": lic, "years_licensed": pick(rng, [1, 3, 6, 11, 19, 27]), "setting": pick(rng, SETTINGS),
                "employer_action": pick(rng, ["dismissed the nurse", "retrained the nurse and kept the nurse on", "took no action",
                                              "accepted the nurse's resignation before its review"]),
                "complainant": pick(rng, ["the employer", "a patient's daughter", "a co-worker", "the unit manager",
                                          "a patient", "the pharmacy department"]),
                "drug": pick(rng, DRUGS), "opioid": pick(rng, OPIOIDS), "care": pick(rng, CARE),
                "mult": pick(rng, [2, 5, 10]), "hours": pick(rng, [3, 4, 6]), "n": pick(rng, [2, 3, 4, 6, 9]),
                "rescue": pick(rng, RESCUE), "harm_text_i": rng.randrange(3),
                "licensee": person(rng), "case_id": ident(rng, "NB", 5, "-"),
                "licence_no": f"{'RN' if lic != 'practical nurse' else 'PN'}{rng.randint(100000, 999999)}",
                "investigator": person(rng), "meeting": rand_date(rng)}

    def _allegation_text(self, f):
        return NURSE_ALLEGATIONS[f["allegation"]][0].format(**f)

    def _harm_text(self, f):
        pool = HARM_TEXT[(NURSE_ALLEGATIONS[f["allegation"]][3], f["harm"])]
        return pool[f["harm_text_i"] % len(pool)].format(**f)

    def _guidelines(self, f):
        b = f["board"]
        return [f"{_cite(b, 0)} A complaint not proved by clear evidence is dismissed.",
                f"{_cite(b, 1)} An honest error or a shortcut that caused no lasting harm, by a nurse with no prior discipline: letter of concern.",
                f"{_cite(b, 2)} Reckless conduct, a pattern, or conduct causing temporary harm: probation of 12-24 months with conditions.",
                f"{_cite(b, 3)} Deliberate conduct, or conduct causing serious harm: suspension.",
                f"{_cite(b, 4)} Prior discipline moves the outcome one step up; candour and completed remediation may move it one "
                "step down for errors and shortcuts, never below a letter of concern once a breach is proved."]

    def render(self, f, rng, surface):
        b = f["board"]
        alleg = self._allegation_text(f)
        prior = (f"{PRIOR[f['prior']]} in {f['prior_year']}" if f["prior"] != "none" else "none")
        finding = noisy(join_sentences([
            f"On a shift at {f['setting']}, the nurse {alleg}.", f"Investigators found that {f['cue']}.",
            f"Outcome: {self._harm_text(f)}."]), rng, 0.12)
        if surface == "json":
            return dumps(prune({
                "case": {"id": f["case_id"], "board": b["board"], "complainant": f["complainant"]},
                "licensee": {"name": f["licensee"], "licence": f"{f['licence_type']}, {f['licence_no']}",
                             "years_licensed": f["years_licensed"], "prior_discipline": prior},
                "investigation": {"findings": finding, "evidence": f["evidence_text"],
                                  "respondent_position": NURSE_RESPONSE[f["response"]],
                                  "employer_action": f["employer_action"]},
                "guidelines": {"source": f"{b['act']} sanctioning guidelines", "steps": self._guidelines(f)},
            }))
        if surface == "investigator_report":
            head = kv_lines([("Case", f["case_id"]), ("Respondent", f"{f['licensee']} ({f['licence_type']}, {f['licence_no']})"),
                             ("Years licensed", f["years_licensed"]), ("Complainant", f["complainant"]),
                             ("Prior discipline", prior), ("Investigator", f["investigator"])], rng)
            body = join_sentences([finding, f"Evidence relied on: {f['evidence_text']}.",
                                   f"The respondent {NURSE_RESPONSE[f['response']]}.",
                                   f"The employer {f['employer_action']}."])
            return (f"{b['board'].upper()} - INVESTIGATION REPORT\n{head}\n\nSummary:\n{body}\n\n"
                    f"Sanctioning guidelines ({b['act']}):\n" + "\n".join(self._guidelines(f)))
        rows = [["Respondent", f"{f['licensee']}, {f['licence_type']}"], ["Allegation", alleg],
                ["Setting", f["setting"]], ["Context found", f["cue"]], ["Patient outcome", self._harm_text(f)],
                ["Evidence", f["evidence_text"]], ["Respondent's position", NURSE_RESPONSE[f["response"]]],
                ["Prior discipline", prior], ["Employer", f["employer_action"]]]
        return (f"Agenda item {f['case_id']} - {b['board']} meeting {fmt_date(f['meeting'], rng)}\n\n"
                + table(["Item", "Detail"], rows, rng) + "\n\nGuidelines:\n" + bullets(self._guidelines(f), rng))

    def questions(self):
        return {
            "board_outcome": choice("What outcome should the board impose on this complaint?", {
                "dismiss_complaint": "Dismiss: the complaint is not proved, or the conduct met the standard.",
                "letter_of_concern": "Issue a non-disciplinary letter of concern recorded on file.",
                "probation_with_terms": "Place the licence on probation with conditions for a fixed period.",
                "suspension": "Suspend the licence for a period before any return to practice.",
            }),
            "practice_condition": choice("Which condition, if any, should go with the board's outcome?", {
                "controlled_drug_restriction": "Bar access to controlled drugs and require random count audits.",
                "no_further_conditions": "None beyond the outcome itself.",
                "supervised_practice": "Require a practice monitor and no unsupervised shifts.",
                "targeted_coursework": "Require a course on the specific failing, such as medication safety or documentation.",
            }),
            "patient_harm_level": score("How much harm did the conduct cause patients?", [
                "None: no patient was affected.",
                "Exposure only: a patient was put at risk but showed no ill effects.",
                "Temporary harm: a patient needed extra treatment and recovered.",
                "Serious harm: lasting injury, intensive care or a fall with fracture.",
            ]),
            "culpability": score("How blameworthy was the nurse's state of mind?", [
                "Inadvertent: an honest slip, often set up by the system around the nurse.",
                "At-risk: a shortcut taken without seeing the danger.",
                "Reckless: a known risk consciously disregarded.",
                "Deliberate: the nurse knowingly did wrong or hid it.",
            ]),
            "standard_breach_proved": noul(
                "The evidence proves that the nurse departed from the standard of care.",
                yes="Reliable evidence shows the conduct happened and fell below what a reasonable nurse would do.",
                no="The evidence is too thin, or the event came from a system failure rather than the nurse's own practice."),
        }

    def _breached(self, f):
        return f["evidence"] != "weak" and not (f["system_factor"] and f["culpability"] == "inadvertent")

    def _step(self, f):
        if not self._breached(f):
            return 0
        culp, harm = f["culpability"], f["harm"]
        step = {"inadvertent": 1, "at_risk": 1, "reckless": 2, "intentional": 3}[culp]
        if harm >= 2:
            step = max(step, 2)
        if harm == 3:
            step = 3
        if f["prior"] != "none":
            step += 1
        if f["response"] == "admits_remediated" and culp in ("inadvertent", "at_risk"):
            step -= 1
        return _clamp(step, 1, 3)

    def intended(self, f):
        step = self._step(f)
        culp = f["culpability"]
        group = NURSE_ALLEGATIONS[f["allegation"]][2]
        if step == 0:
            cond = "no_further_conditions"
        elif step == 1:
            cond = "no_further_conditions" if f["response"] in ("admits_remediated", "admits") else "targeted_coursework"
        elif f["allegation"] == "unwitnessed_waste":
            cond = "controlled_drug_restriction"
        elif group in ("records", "licence", "conduct"):
            cond = "supervised_practice" if step == 3 else "targeted_coursework"
        elif f["harm"] == 3 or culp == "reckless":
            cond = "supervised_practice"
        else:
            cond = "targeted_coursework"
        return {"board_outcome": OUTCOME_STEPS[step], "practice_condition": cond,
                "patient_harm_level": f["harm"],
                "culpability": ["inadvertent", "at_risk", "reckless", "intentional"].index(culp),
                "standard_breach_proved": "yes" if self._breached(f) else "no"}


# =========================================================================== schema 2
# (work, disciplines competent to seal it, public-safety weight 0-3)
ENG_WORK = [
    ("a 3.5 m retaining wall behind a school car park", ["structural", "geotechnical", "civil"], 3),
    ("a timber deck on a two-storey house", ["structural", "civil"], 1),
    ("the HVAC replacement for a public library", ["mechanical"], 1),
    ("the fire-sprinkler layout for a storage warehouse", ["fire protection", "mechanical"], 3),
    ("a stormwater detention pond for a 40-lot subdivision", ["civil"], 2),
    ("an electrical service upgrade for a medical clinic", ["electrical"], 2),
    ("rooftop solar racking on an older warehouse roof", ["structural"], 2),
    ("foundation repairs for a settled house", ["structural", "geotechnical"], 2),
    ("a load rating for a single-lane farm bridge", ["structural", "civil"], 3),
    ("slope stabilisation above a county road", ["geotechnical", "civil"], 3),
    ("lighting for a surface car park", ["electrical"], 0),
    ("a steel canopy over a bus platform", ["structural"], 2),
    ("the standby-generator design for a care home", ["electrical"], 3),
    ("grading and drainage for a sports field", ["civil"], 1),
    ("a mezzanine floor in a retail unit", ["structural"], 2),
    ("ventilation for an underground parking garage", ["mechanical"], 2),
]
ENG_DISCIPLINES = ["civil", "structural", "mechanical", "electrical", "geotechnical", "fire protection"]
ENG_ISSUES = {"competence": "sealed work outside the engineer's field",
              "stamping": "sealed drawings prepared by an unlicensed drafting firm",
              "errors": "design errors in sealed documents"}
RC_EVIDENCE = {"strong": ["the file holds the engineer's own calculations and marked-up drafts",
                          "emails show the engineer directing the drafter and reviewing three revisions"],
               "partial": ["the engineer reviewed the final set but no calculations of the engineer's own are in the file",
                           "the engineer visited once; the drafter's calculations were accepted unchecked"],
               "none": ["the drafting firm's invoices list a per-sheet 'seal fee' paid to the engineer",
                        "the engineer could not produce any calculations, markups or correspondence about the project"]}
PEER = {0: "the board's peer reviewer found the design sound", 1: "the peer reviewer found minor drafting and labelling errors",
        2: "the peer reviewer found load or capacity errors that needed redesign", 3: "the peer reviewer found errors that made the design unsafe as drawn"}
CONSEQ = {"as_designed": "the work was built as designed and is in use",
          "caught": "the errors were caught at plan review and corrected before construction",
          "built_fixed": "the work was built and has since been retrofitted",
          "distress": "the built work showed cracking and deflection and is under monitoring",
          "failure": "part of the built work failed; no one was hurt"}


class EngineerSealedWorkComplaint(Schema):
    name = "engineer_sealed_work_complaint"
    description = "A complaint that a licensed engineer sealed deficient or improperly supervised work, judged against the practice act."
    surfaces = {"json": 0.5, "peer_review_summary": 0.25, "complaint_and_response": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        b = _board(rng, "engineering")
        work, fields, safety = pick(rng, ENG_WORK)
        issue = pick_w(rng, {"competence": 3, "stamping": 4, "errors": 3})
        discipline = pick(rng, fields) if chance(rng, 0.45) else pick(rng, [d for d in ENG_DISCIPLINES if d not in fields])
        similar = pick(rng, [0, 0, 1, 2, 4, 8])
        rc = "strong" if issue != "stamping" else pick_w(rng, {"strong": 1, "partial": 2, "none": 3})
        err = pick_w(rng, {0: 3, 1: 3, 2: 3, 3: 1})
        conseq = pick(rng, list(CONSEQ)) if err >= 2 else "caught"
        stamped = pick(rng, [1, 2, 4, 6, 14]) if issue == "stamping" else 1
        prior = pick_w(rng, {"none": 8, "letter": 2, "probation": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                discipline, issue, rc, err, conseq, prior = pick(rng, fields), "errors", "strong", pick(rng, [0, 1]), "caught", "none"
                similar = pick(rng, [4, 8])
            else:
                issue, rc, stamped, err = "stamping", "none", pick(rng, [14, 30]), pick(rng, [2, 3])
                conseq = pick(rng, ["distress", "failure", "built_fixed"])
        elif reg == "borderline":
            # a partly supervised stamp near the two-project line; outside the licence's field but with some
            # experience; or errors right at the redesign line
            if issue == "stamping":
                rc, stamped = "partial", pick(rng, [1, 1, 2, 3])
            elif chance(rng, 0.5):
                discipline = pick(rng, [d for d in ENG_DISCIPLINES if d not in fields])
                similar = pick(rng, [2, 3, 4])
                issue = "competence"
            else:
                err = pick(rng, [1, 2])
        else:
            twist = pick(rng, ["sound_but_stamped", "in_field_unsafe"])
            if twist == "sound_but_stamped":
                issue, rc, err, conseq = "stamping", pick(rng, ["partial", "none"]), 0, "caught"
                stamped = pick(rng, [1, 3, 6, 14])
            else:
                discipline, issue, rc, err = pick(rng, fields), "errors", "strong", 3
                conseq = pick(rng, ["distress", "failure"])
        if issue != "stamping":
            rc, stamped = "strong", 1
        if err == 0:
            conseq = "as_designed"
        elif err == 1:
            conseq = "caught"
        return {"regime": reg, "board": b, "work": work, "competent_fields": fields, "safety_weight": safety,
                "issue": issue, "licensed_discipline": discipline, "similar_projects": similar,
                "rc_evidence": rc, "rc_text": pick(rng, RC_EVIDENCE[rc]), "error_level": err, "consequence": conseq,
                "projects_sealed_for_firm_12m": stamped, "prior": prior, "prior_year": rng.randint(2015, 2024),
                "engineer": person(rng), "licence_no": f"PE-{rng.randint(10000, 99999)}",
                "complainant": pick(rng, ["the county building official", "the project owner", "another engineer",
                                          "a neighbouring owner", "the school district", "the general contractor"]),
                "drafting_firm": _firm(rng, ["Drafting", "Design Services", "CAD Studio", "Plans"]),
                "response": pick(rng, ["candid and has corrected the drawings", "defensive and says the reviewer is too strict",
                                       "blames the drafter", "did not respond in time"]),
                "case_id": ident(rng, "PE", 4, "-"), "filed": rand_date(rng)}

    def _in_competence(self, f):
        return f["licensed_discipline"] in f["competent_fields"] or f["similar_projects"] >= 3

    def _rc_met(self, f):
        if f["rc_evidence"] == "strong":
            return True
        if f["rc_evidence"] == "partial":
            return f["projects_sealed_for_firm_12m"] == 1     # a one-off final review, not a sealing business
        return False

    def _rules(self, f):
        b = f["board"]
        return [f"{_cite(b, 0)} An engineer may seal only work within the engineer's competence by education or "
                "experience, whatever discipline the licence names.",
                f"{_cite(b, 1)} An engineer may seal documents only if prepared by the engineer or under the engineer's "
                "responsible charge: direct control, personal review of the design and its calculations.",
                f"{_cite(b, 2)} Guidelines: minor errors without risk: letter of concern; competence or supervision "
                "breaches, or design errors needing redesign: probation with practice review; unsafe work that was "
                "built, or repeated plan-stamping: suspension. Prior discipline moves one step up."]

    def render(self, f, rng, surface):
        b = f["board"]
        prior = f"{PRIOR[f['prior']]} in {f['prior_year']}" if f["prior"] != "none" else "none"
        facts = [f"Project: {f['work']}.", f"Licence: {f['licensed_discipline']} engineering ({f['licence_no']}).",
                 f"Similar projects in the engineer's record: {f['similar_projects']}.",
                 f"Supervision evidence: {f['rc_text']}.", f"Peer review: {PEER[f['error_level']]}.",
                 f"Construction status: {CONSEQ[f['consequence']]}."]
        if f["issue"] == "stamping":
            facts.append(f"Sealed {f['projects_sealed_for_firm_12m']} project(s) for {f['drafting_firm']} in the past 12 months.")
        if surface == "json":
            return dumps(prune({
                "complaint": {"id": f["case_id"], "board": b["board"], "filed_by": f["complainant"],
                              "alleges": ENG_ISSUES[f["issue"]], "project": f["work"]},
                "engineer": {"name": f["engineer"], "licensed_discipline": f["licensed_discipline"],
                             "similar_projects_on_record": f["similar_projects"], "prior_discipline": prior,
                             "response": f["response"]},
                "evidence": {"supervision": f["rc_text"], "peer_review": PEER[f["error_level"]],
                             "construction_status": CONSEQ[f["consequence"]],
                             "projects_sealed_for_drafting_firm_12m": (f["projects_sealed_for_firm_12m"]
                                                                       if f["issue"] == "stamping" else None)},
                "rules": {"act": b["act"], "provisions": self._rules(f)},
            }))
        if surface == "peer_review_summary":
            return (f"PEER REVIEW AND INVESTIGATION SUMMARY - {f['case_id']}\nRespondent: {f['engineer']}\n"
                    f"Complaint: {ENG_ISSUES[f['issue']]} (filed by {f['complainant']})\n\n" + bullets(facts, rng)
                    + f"\n\nPrior discipline: {prior}. Respondent's reply was {f['response']}.\n\n" + "\n".join(self._rules(f)))
        letter = noisy(join_sentences([
            f"I am writing as {f['complainant']} about the documents {f['engineer']} sealed for {f['work']}.",
            {"competence": "I question whether this engineer is qualified for this kind of design.",
             "stamping": f"The drawings appear to have been produced by {f['drafting_firm']}, who are not licensed.",
             "errors": "The design contains errors that concern me."}[f["issue"]],
            {"as_designed": "", "caught": "", "built_fixed": "Remedial work has already been needed.",
             "distress": "We are now seeing cracks.", "failure": "Part of it has already failed."}[f["consequence"]]]), rng)
        return (f"Complaint to the {b['board']}, received {fmt_date(f['filed'], rng)}\n\n{letter}\n\n"
                f"[Board investigator notes]\n" + "\n".join(facts) + f"\nPrior discipline: {prior}.\n"
                f"Engineer's reply: {f['response']}.\n\n" + "\n".join(self._rules(f)))

    def questions(self):
        return {
            "engineer_sanction": choice("What should the board do about this engineer?", {
                "dismiss_complaint": "Dismiss: the work and the sealing practice met the act.",
                "letter_of_concern": "Send a letter of concern about minor shortcomings.",
                "probation_with_practice_review": "Place the licence on probation with outside review of the engineer's sealed work.",
                "suspension": "Suspend the licence.",
            }),
            "public_safety_exposure": score("How far did the sealed work expose the public to danger?", [
                "Negligible: low-consequence work, or nothing wrong with it.",
                "Limited: defects in modest work, caught or easily fixed.",
                "Significant: defects in work people rely on for safety.",
                "Grave: unsafe life-safety work that was built or failed.",
            ]),
            "design_shortfall": score("How far did the design fall short of competent engineering practice?", [
                "Not at all: the design is sound.",
                "Slightly: drafting and presentation errors only.",
                "Substantially: capacity or performance errors requiring redesign.",
                "Severely: the design was unsafe as sealed.",
            ]),
            "responsible_charge_shown": noul(
                "The engineer had responsible charge of the documents the engineer sealed.",
                yes="The engineer prepared or directly controlled and personally checked the design and its calculations.",
                no="The engineer sealed others' work without meaningful control or review."),
            "within_competence": noul("The sealed work lay within the engineer's competence by education or experience."),
        }

    def intended(self, f):
        rc = self._rc_met(f)
        comp = self._in_competence(f)
        err = f["error_level"]
        gap = _clamp(err + (0 if comp or err == 0 else 1)) if err else (1 if not comp else 0)
        built_bad = f["consequence"] in ("distress", "failure")
        exposure = f["safety_weight"] if err >= 2 else (1 if err == 1 and f["safety_weight"] >= 2 else 0)
        if built_bad:
            exposure = max(exposure, 2) + (1 if f["safety_weight"] >= 2 else 0)
        elif f["consequence"] == "caught" and err >= 2:
            exposure = min(exposure, 2)
        exposure = _clamp(exposure)
        if (err == 3 and built_bad) or (not rc and f["projects_sealed_for_firm_12m"] >= 6):
            step = 3
        elif err >= 2 or not rc or not comp:
            step = 2
        elif err == 1:
            step = 1
        else:
            step = 0
        if f["prior"] != "none" and step > 0:
            step = min(3, step + 1)
        keys = ["dismiss_complaint", "letter_of_concern", "probation_with_practice_review", "suspension"]
        return {"engineer_sanction": keys[step], "public_safety_exposure": exposure, "design_shortfall": gap,
                "responsible_charge_shown": "yes" if rc else "no", "within_competence": "yes" if comp else "no"}


# =========================================================================== schema 3
# (project, defects an inspector might find on it)
PROJECTS = [
    ("a kitchen remodel", ["new circuits were run without junction boxes", "the range hood vents into the ceiling void", "cabinets are not fixed to studs"]),
    ("a roof replacement", ["the flashing was left incomplete and leaks", "underlayment is missing on the north slope", "ridge vents were never installed"]),
    ("a rear deck", ["joists are undersized and spaced too widely", "the ledger is nailed rather than bolted to the house", "guardrails are below code height"]),
    ("a basement finish", ["new wiring was run without boxes", "no egress window was provided for the bedroom", "framing sits directly on the damp slab"]),
    ("a single-room addition", ["the footing is shallower than the frost line", "headers over the windows are undersized", "the roof tie-in leaks"]),
    ("an in-ground pool", ["the pool bonding grid is incomplete", "the barrier fence has no self-closing gate", "the shell has cracked"]),
    ("a bathroom renovation", ["tiles are cracking because no backer board was used", "the shower pan was not waterproofed", "the fan vents into the attic"]),
    ("replacement windows", ["windows were set without flashing tape", "two units were installed out of plumb", "sills slope inward and let water in"]),
    ("a detached garage", ["the slab was poured without the required vapour barrier", "anchor bolts are missing on one wall", "the roof trusses are not braced"]),
    ("a new porch and steps", ["stair risers vary by more than the code allows", "posts sit on bare soil", "there is no handrail"]),
    ("foundation waterproofing", ["the drain tile was not connected to the sump", "the membrane stops short of the footing"]),
    ("a heat-pump installation", ["the outdoor unit sits on an unlevel pad", "the refrigerant lines are not insulated", "the breaker is oversized"]),
    ("a garage conversion", ["the converted room has no heat source", "the fire separation to the house was removed", "the new window is too small for egress"]),
    ("siding replacement", ["house wrap was omitted behind the siding", "trim was installed without flashing above windows"]),
]
LICENCE_CLASS = ["general building contractor", "residential remodelling contractor", "home-improvement contractor"]


def _licence_class(rng, project):
    if "roof" in project:
        return pick(rng, ["roofing contractor", "general building contractor"])
    if "pool" in project:
        return "pool and spa contractor"
    if "heat-pump" in project:
        return "mechanical contractor"
    return pick(rng, LICENCE_CLASS)
EXCUSES = ["says the homeowner kept changing the plans", "says materials were delayed by the supplier",
           "says the homeowner stopped paying", "says a family emergency stopped work",
           "offers to return and finish within two weeks", "did not respond to the board"]


class ContractorConsumerComplaint(Schema):
    name = "contractor_consumer_complaint"
    description = "A homeowner's complaint against a licensed contractor: deposits, abandonment, permits and workmanship, against the contractors act."
    surfaces = {"json": 0.5, "homeowner_complaint": 0.25, "investigator_summary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        b = _board(rng, "contractors")
        project, defect_pool = pick(rng, PROJECTS)
        price = int(round(lognormal_amount(rng, 32000, 0.6), -2))
        cap_pct = pick(rng, [10, 15, 20, 33])
        cap_abs = pick(rng, [None, 1000, 2500, 5000])
        cap = price * cap_pct / 100 if cap_abs is None else min(price * cap_pct / 100, cap_abs)
        dep_ratio = pick(rng, [0.4, 0.8, 1.0, 1.6, 2.5])
        abandon_days = pick(rng, [30, 45, 60, 90])
        idle = pick(rng, [0, 0, 10, 25, 70, 150])
        done = pick(rng, [0.0, 0.15, 0.4, 0.7, 0.95])
        permit = pick_w(rng, {"pulled": 5, "not_pulled": 3, "not_required": 1})
        defects = pick(rng, [0, 0, 1, 2])
        responsive = pick(rng, EXCUSES)
        prior = pick_w(rng, {"none": 6, "letter": 2, "probation": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                dep_ratio, idle, permit, defects, done, prior = 0.6, 0, "pulled", 0, 0.95, "none"
                responsive = pick(rng, EXCUSES[:3])
            else:
                dep_ratio, idle, permit, done = pick(rng, [1.6, 2.5]), abandon_days + pick(rng, [40, 120]), "not_pulled", pick(rng, [0.0, 0.15])
                responsive = "did not respond to the board"
        elif reg == "borderline":
            if chance(rng, 0.5):
                dep_ratio = near(rng, 1.0, 0.06, lo=0.85)
            else:
                idle = near(rng, abandon_days, 6, lo=1, integer=True)
        else:
            twist = pick(rng, ["finished_but_rule_breaks", "excuse_for_abandonment"])
            if twist == "finished_but_rule_breaks":
                done, defects, dep_ratio, permit = 0.95, 0, pick(rng, [1.6, 2.5]), "not_pulled"
            else:
                idle, responsive = abandon_days + pick(rng, [10, 30]), pick(rng, ["says materials were delayed by the supplier",
                                                                                   "offers to return and finish within two weeks"])
        defects = min(defects, len(defect_pool))
        deposit = int(round(cap * dep_ratio, -1))
        paid = int(max(deposit, round(price * min(1.0, done + pick(rng, [0.0, 0.1, 0.3])), -2)))
        repair = int(round(sum(lognormal_amount(rng, 2600, 0.6) for _ in range(defects)), -1))
        return {"regime": reg, "board": b, "project": project, "licence_class": _licence_class(rng, project),
                "price": price, "cap_pct": cap_pct, "cap_abs": cap_abs, "cap": round(cap, 2), "deposit": deposit,
                "paid": paid, "done_fraction": done, "abandon_days": abandon_days, "idle_days": idle, "permit": permit,
                "defects": rng.sample(defect_pool, defects), "repair_estimate": repair, "contractor_reply": responsive,
                "prior": prior, "prior_year": rng.randint(2016, 2024), "insolvent": chance(rng, 0.2),
                "contractor": _firm(rng), "owner": person(rng), "homeowner": person(rng),
                "case_id": ident(rng, "CLB", 5, "-"), "contract_date": rand_date(rng)}

    def _loss(self, f):
        return max(0.0, f["paid"] - f["price"] * f["done_fraction"]) + f["repair_estimate"]

    def _breaches(self, f):
        out = []
        if f["deposit"] > f["cap"] + 0.005:
            out.append("deposit")
        if f["idle_days"] >= f["abandon_days"] and f["done_fraction"] < 0.9 and f["contractor_reply"] not in (
                "says the homeowner stopped paying",):
            out.append("abandonment")
        if f["permit"] == "not_pulled":
            out.append("permit")
        if f["defects"]:
            out.append("code")
        return out

    def _rules(self, f):
        b = f["board"]
        cap = f"{f['cap_pct']}% of the contract price" + (f" or {money(f['cap_abs'])}, whichever is less," if f["cap_abs"] else "")
        return [f"{_cite(b, 0)} A contractor may not take a down payment larger than {cap} before work starts.",
                f"{_cite(b, 1)} Stopping work for {f['abandon_days']} consecutive days without lawful excuse (such as "
                "non-payment by the owner) is abandonment.",
                f"{_cite(b, 2)} All required permits must be obtained before work begins; work must meet the building code.",
                f"{_cite(b, 3)} Disputes over taste or price alone are civil matters for the courts. The board may cite and "
                "fine, order restitution, impose probation or suspend; claims against an insolvent or absent contractor "
                "may be paid from the recovery fund."]

    def render(self, f, rng, surface):
        b = f["board"]
        start = f["contract_date"]
        defects = "; ".join(f["defects"]) if f["defects"] else None
        prior = f"{PRIOR[f['prior']]} in {f['prior_year']}" if f["prior"] != "none" else "none"
        permit = {"pulled": "permit obtained", "not_pulled": "no permit obtained", "not_required": "no permit required for this work"}[f["permit"]]
        if surface == "json":
            return dumps(prune({
                "complaint": {"id": f["case_id"], "board": b["board"], "homeowner": f["homeowner"], "project": f["project"]},
                "contractor": {"business": f["contractor"], "licence_class": f["licence_class"], "prior_discipline": prior,
                               "reply": f["contractor_reply"], "business_status": "closed its office; phone disconnected" if f["insolvent"] else None},
                "money": {"contract_price": f["price"], "down_payment": f["deposit"], "total_paid": f["paid"]},
                "work": {"completed_share": f"{round(f['done_fraction'] * 100)}%", "days_since_last_work": f["idle_days"],
                         "permit": permit, "inspector_defects": defects,
                         "repair_estimate": f["repair_estimate"] or None},
                "rules": {"act": b["act"], "provisions": self._rules(f)},
            }))
        if surface == "homeowner_complaint":
            last = start + timedelta(days=rng.randint(20, 90))
            text = noisy(join_sentences([
                f"We hired {f['contractor']} on {fmt_date(start, rng)} for {f['project']} at {money(f['price'], rng)}.",
                f"They asked for {money(f['deposit'], rng)} up front and we have paid {money(f['paid'], rng)} in total.",
                (f"Nobody has been on site since {fmt_date(last, rng)}, {f['idle_days']} days ago." if f["idle_days"]
                 else "They are still coming to site."),
                f"An inspector we hired says about {round(f['done_fraction'] * 100)}% of the work is done.",
                (f"Problems found: {defects}; fixing them is quoted at {money(f['repair_estimate'], rng)}." if defects else ""),
                ("As far as we can tell no permit was ever pulled." if f["permit"] == "not_pulled" else "")]), rng)
            tail = kv_lines([("Contractor's reply", f["contractor_reply"]), ("Prior discipline", prior),
                             ("Business status", "office closed, phone disconnected" if f["insolvent"] else "trading")], rng)
            return (f"COMPLAINT FORM - {b['board']}\nComplainant: {f['homeowner']}\n\n{text}\n\n[staff notes]\n{tail}\n\n"
                    + "\n".join(self._rules(f)))
        rows = [("Case", f["case_id"]), ("Contractor", f"{f['contractor']} ({f['licence_class']})"),
                ("Project", f["project"]), ("Contract price", money(f["price"], rng)),
                ("Down payment", money(f["deposit"], rng)), ("Paid to date", money(f["paid"], rng)),
                ("Work complete", f"{round(f['done_fraction'] * 100)}%"), ("Days since last work", f["idle_days"]),
                ("Permit", permit), ("Defects", show(defects, rng) if f["defects"] else "none found"),
                ("Repair estimate", money(f["repair_estimate"], rng) if f["repair_estimate"] else "n/a"),
                ("Contractor reply", f["contractor_reply"]), ("Prior discipline", prior),
                ("Business status", "office closed" if f["insolvent"] else "trading")]
        return "INVESTIGATOR SUMMARY\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._rules(f))

    def questions(self):
        return {
            "contractor_board_action": choice("What action should the board take on this complaint?", {
                "citation_and_fine": "Issue a citation with a fine for a rule breach that caused little loss.",
                "dismiss_as_civil_dispute": "Close it: no breach of the act, only a private disagreement.",
                "restitution_order_with_probation": "Order the contractor to repay or put things right, under probation.",
                "suspend_licence": "Suspend the licence because of serious or repeated breaches.",
            }),
            "homeowner_recovery_route": choice("How should the homeowner's loss be made good, if at all?", {
                "contractor_completes_work": "Have the contractor return and finish or correct the work under supervision.",
                "contractor_refund": "Order the contractor to refund money paid for work not done or done badly.",
                "no_recovery_owed": "Nothing: the homeowner has received what was paid for.",
                "recovery_fund_claim": "Pay the homeowner from the recovery fund because the contractor cannot or will not.",
            }),
            "consumer_loss": score("How large is the homeowner's financial loss?", [
                "None: payments match the value of the work.",
                "Small: under about $2,500 out of pocket.",
                "Substantial: roughly $2,500 to $10,000.",
                "Severe: over $10,000.",
            ]),
            "statutory_breaches": score("How serious are the contractor's breaches of the act?", [
                "None found.",
                "One technical breach.",
                "Two breaches, or one with real consequences.",
                "Several breaches, including abandonment or unsafe work.",
            ]),
            "down_payment_over_cap": noul(
                "The contractor took a larger down payment than the act allows.",
                yes="The down payment exceeds the statutory cap computed for this contract.",
                no="The down payment is at or under the cap."),
        }

    def intended(self, f):
        loss = self._loss(f)
        br = self._breaches(f)
        lvl = 0 if loss < 300 else 1 if loss < 2500 else 2 if loss < 10000 else 3
        sev = len(br)
        if "abandonment" in br or ("code" in br and len(f["defects"]) >= 2):
            sev = max(sev, 2) + (1 if sev >= 2 else 0)
        sev = _clamp(sev)
        unresponsive = f["contractor_reply"] == "did not respond to the board" or f["insolvent"]
        if sev == 0 and lvl <= 1:
            act = "dismiss_as_civil_dispute"
        elif ("abandonment" in br and unresponsive) or sev >= 3 or (f["prior"] == "probation" and sev >= 1):
            act = "suspend_licence"
        elif lvl >= 2:
            act = "restitution_order_with_probation"
        else:
            act = "citation_and_fine"
        if lvl == 0:
            route = "no_recovery_owed"
        elif unresponsive:
            route = "recovery_fund_claim"
        elif f["contractor_reply"] == "offers to return and finish within two weeks" or (f["done_fraction"] >= 0.7 and f["defects"]):
            route = "contractor_completes_work"
        else:
            route = "contractor_refund"
        return {"contractor_board_action": act, "homeowner_recovery_route": route, "consumer_loss": lvl,
                "statutory_breaches": sev, "down_payment_over_cap": "yes" if "deposit" in br else "no"}


# =========================================================================== schema 4
# board kind -> [(complaint text, within the board's jurisdiction, body to refer to or None, risk if true 0-3)]
INTAKE_ITEMS = {
    "nursing": [
        ("a night nurse at {place} sleeps on shift and leaves residents unchecked", True, None, 3),
        ("a nurse gave my father the wrong medication and nobody told us", True, None, 2),
        ("a home-health nurse is charting visits she never makes", True, None, 2),
        ("a nurse posted pictures of patients on social media", True, None, 1),
        ("a nurse is still working although her licence lapsed months ago", True, None, 1),
        ("the hospital overcharged us for a three-day stay", False, "the health-insurance ombudsman", 0),
        ("the surgeon misdiagnosed my mother", False, "the medical board", 2),
        ("the care home never has enough staff at weekends", False, "the health department's facility inspectors", 2),
        ("a co-worker is rude to me in the break room", False, None, 0),
    ],
    "engineering": [
        ("the engineer's report missed an obvious foundation crack in the house I bought", True, None, 1),
        ("a retaining wall designed by an engineer is bulging over a public footpath", True, None, 3),
        ("an engineer seals drawings for a drafting firm she has never visited", True, None, 2),
        ("the engineer who inspected the shopping-centre parking deck ignored visible corrosion", True, None, 3),
        ("my engineer's invoice is twice the quote", False, None, 0),
        ("a man who is not licensed advertises engineering services", False, "the consumer-affairs unlicensed-practice unit", 1),
        ("the planning department rejected my permit unfairly", False, "the permit appeals board", 0),
    ],
    "contractors": [
        ("a contractor took a deposit and never started the job", True, None, 1),
        ("the contractor did electrical work without a permit and the outlets spark", True, None, 3),
        ("my roofer left the roof open for weeks", True, None, 2),
        ("the contractor's crew damaged my neighbour's wall and walked off the job", True, None, 1),
        ("I don't like the colour the painter used", False, None, 0),
        ("a big-box store will not honour the warranty on my water heater", False, "the consumer-protection division", 0),
        ("an unlicensed handyman took my money", False, "the consumer-affairs unlicensed-practice unit", 1),
    ],
    "pharmacy": [
        ("a pharmacy technician filled my child's prescription with the wrong strength", True, None, 2),
        ("the pharmacist on duty refuses to counsel patients and the dispensary is filthy", True, None, 1),
        ("my insurer will not cover my prescription", False, "the insurance regulator", 0),
    ],
    "home_inspection": [
        ("the home inspector missed a failing roof that was obvious from the street", True, None, 1),
        ("the inspector's report left out live wiring in the attic", True, None, 2),
        ("the seller hid water damage from me", False, "the real-estate commission", 0),
    ],
}
PLACES = ["a care home on Mill Road", "the county nursing facility", "a rehabilitation centre", "a group home",
          "the hospital's stroke unit", "a respite centre"]
COMPLAINANTS = ["a patient's relative", "a co-worker", "an anonymous caller", "a client", "a building official",
                "a neighbour", "an employer", "a former employee"]


class ComplaintIntakeScreen(Schema):
    name = "complaint_intake_screen"
    description = "A new complaint at a licensing board's intake desk: open, expedite, refer, close or ask for more."
    surfaces = {"json": 0.5, "online_complaint_form": 0.25, "intake_triage_sheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind = pick_w(rng, {"nursing": 4, "engineering": 3, "contractors": 3, "pharmacy": 1, "home_inspection": 1})
        b = _board(rng, kind)
        text, within, refer, risk = pick(rng, INTAKE_ITEMS[kind], [2 if x[1] else 1 for x in INTAKE_ITEMS[kind]])
        ident_level = pick_w(rng, {"licence_number": 3, "name_only": 4, "unidentified": 2})
        date_given = pick_w(rng, {"exact": 4, "approximate": 3, "none": 2})
        docs = chance(rng, 0.45)
        limit = pick(rng, [2, 3, 5])
        years_ago = pick(rng, [0, 1, 2, 3, 4, 6, 8, 10])
        concealed = chance(rng, 0.12)
        ongoing = pick_w(rng, {True: 4, False: 3, None: 2})
        if reg == "clear":
            if chance(rng, 0.5):
                text, within, refer, risk = pick(rng, [x for x in INTAKE_ITEMS[kind] if x[1]] )
                ident_level, date_given, docs, years_ago = "licence_number", "exact", True, 0
            else:
                text, within, refer, risk = pick(rng, [x for x in INTAKE_ITEMS[kind] if not x[1]])
        elif reg == "borderline":
            years_ago = pick(rng, [limit - 1, limit, limit + 1])
            if chance(rng, 0.5):
                ident_level, docs = "name_only", False
        else:
            twist = pick(rng, ["grave_but_vague", "old_but_concealed", "detailed_but_outside"])
            if twist == "grave_but_vague":
                text, within, refer, risk = pick(rng, [x for x in INTAKE_ITEMS[kind] if x[1]], )
                ident_level, date_given, docs = "unidentified", pick(rng, ["approximate", "none"]), False
            elif twist == "old_but_concealed":
                years_ago, concealed = limit + pick(rng, [1, 3]), True
                text, within, refer, risk = pick(rng, [x for x in INTAKE_ITEMS[kind] if x[1]])
            else:
                text, within, refer, risk = pick(rng, [x for x in INTAKE_ITEMS[kind] if not x[1]])
                ident_level, date_given, docs = "licence_number", "exact", True
        return {"regime": reg, "board": b, "kind": kind, "complaint": text.format(place=pick(rng, PLACES)),
                "within_jurisdiction": within, "referral_body": refer, "risk_if_true": risk,
                "identification": ident_level, "date_given": date_given, "documents_attached": docs,
                "limit_years": limit, "years_since_conduct": years_ago, "concealed": concealed,
                "still_practising": ongoing, "complainant": pick(rng, COMPLAINANTS), "complainant_name": person(rng),
                "licensee": person(rng), "licence_no": f"{kind[:2].upper()}-{rng.randint(10000, 99999)}",
                "received": rand_date(rng), "intake_id": ident(rng, "INT", 6, "-")}

    def _specificity(self, f):
        s = {"licence_number": 2, "name_only": 1, "unidentified": 0}[f["identification"]]
        s += {"exact": 1, "approximate": 0, "none": -1}[f["date_given"]] + (1 if f["documents_attached"] else 0)
        return _clamp(s)

    def _timely(self, f):
        return f["years_since_conduct"] <= f["limit_years"] or f["concealed"]

    def _priority(self, f):
        if not f["within_jurisdiction"] or not self._timely(f):
            return 0
        p = f["risk_if_true"]
        if f["still_practising"] is False:
            p -= 1
        elif f["still_practising"] is True and p >= 2 and f["years_since_conduct"] == 0:
            p += 1
        return _clamp(p)

    def _rules(self, f):
        b = f["board"]
        return [f"{_cite(b, 0)} The board hears complaints about the professional conduct of its own licensees. Fee "
                "disputes and matters of taste are civil; conduct of unlicensed persons or other professions belongs to "
                "the body that regulates them.",
                f"{_cite(b, 1)} A complaint must be made within {_yrs(f['limit_years'])} of the conduct, unless the "
                "conduct was concealed or is continuing.",
                f"{_cite(b, 2)} Complaints alleging a present danger to the public are assigned the same day; others "
                "join the queue in order, with a 180-day target."]

    def _when(self, f):
        if f["years_since_conduct"] == 0:
            return {"exact": "last month, on a date given", "approximate": "a few weeks ago", "none": "date not stated"}[f["date_given"]]
        return {"exact": f"{_yrs(f['years_since_conduct'])} ago, on a date given",
                "approximate": f"about {_yrs(f['years_since_conduct'])} ago", "none": "date not stated"}[f["date_given"]]

    def render(self, f, rng, surface):
        b = f["board"]
        who = {"licence_number": f"{f['licensee']} (licence {f['licence_no']})", "name_only": f["licensee"],
               "unidentified": "name not known"}[f["identification"]]
        still = {True: "still working in the same role", False: "no longer works there", None: None}[f["still_practising"]]
        conceal = "The complainant says the licensee hid the conduct and it only came to light this year." if f["concealed"] else ""
        if surface == "json":
            return dumps(prune({
                "complaint": {"id": f["intake_id"], "received": fmt_date(f["received"], style="iso"), "board": b["board"],
                              "from": f["complainant"], "about": who, "allegation": f["complaint"],
                              "when": self._when(f), "documents_attached": f["documents_attached"],
                              "notes": conceal or None},
                "licensee_status": {"current_practice": still},
                "intake_rules": {"act": b["act"], "provisions": self._rules(f)},
            }))
        if surface == "online_complaint_form":
            body = noisy(join_sentences([f"I want to report that {f['complaint']}.",
                                         f"The person is {who}." if f["identification"] != "unidentified" else "I don't know their name.",
                                         (("This happened last month." if f["years_since_conduct"] == 0 else
                                           f"This happened about {_yrs(f['years_since_conduct'])} ago.")
                                          if f["date_given"] != "none" else ""),
                                         conceal,
                                         ("I have attached photos and paperwork." if f["documents_attached"] else "")]), rng)
            meta = kv_lines([("Form ref", f["intake_id"]), ("Submitted", fmt_date(f["received"], rng)),
                             ("Complainant", f"{f['complainant_name']} ({f['complainant']})"),
                             ("Licensee status (staff lookup)", show(still, rng))], rng)
            return f"ONLINE COMPLAINT - {b['board']}\n{meta}\n\nWhat happened:\n{body}\n\n" + "\n".join(self._rules(f))
        rows = [("Intake", f["intake_id"]), ("Board", b["board"]), ("Source", f["complainant"]), ("Subject", who),
                ("Allegation", f["complaint"]), ("When", self._when(f)),
                ("Documents", "attached" if f["documents_attached"] else "none"),
                ("Licensee status", show(still, rng)), ("Notes", conceal or "-")]
        return "INTAKE TRIAGE SHEET\n" + kv_lines(rows, rng) + "\n\nRules:\n" + bullets(self._rules(f), rng)

    def questions(self):
        return {
            "intake_route": choice("Where should board staff send this complaint?", {
                "close_without_action": "Close it: it is outside the board's authority or time limit and no other body needs it.",
                "expedite_investigation": "Open an investigation and assign it at once because of a present danger.",
                "open_investigation": "Open an investigation in the normal queue.",
                "refer_to_other_body": "Pass it to the agency or board that regulates the matter.",
                "request_more_information": "Ask the complainant for the details needed before anything can be opened.",
            }),
            "allegation_specificity": score("How specific and checkable is the complaint as filed?", [
                "Vague: no identifiable person, date or supporting material.",
                "Thin: some detail, but key facts are missing.",
                "Workable: the person and the events can be identified.",
                "Precise: named licensee, dates and documents.",
            ]),
            "investigation_priority": score("How quickly should the board's investigators act on this complaint?", [
                "No faster than the standard queue, or not at all.",
                "Ahead of the queue, within 60 days.",
                "Within the next two weeks.",
                "Today: someone may be put at risk tomorrow.",
            ]),
            "within_board_jurisdiction": noul(
                "The complaint concerns the professional conduct of a licensee of this board.",
                yes="It alleges that one of the board's licensees practised below the act's standards.",
                no="It is a fee or taste dispute, concerns an unlicensed person or another profession, or targets an institution."),
            "filed_within_time_limit": noul("The complaint was made within the act's time limit or falls under an exception to it."),
        }

    def intended(self, f):
        pr = self._priority(f)
        spec = self._specificity(f)
        if not f["within_jurisdiction"]:
            route = "refer_to_other_body" if f["referral_body"] else "close_without_action"
        elif not self._timely(f):
            route = "close_without_action"
        elif pr >= 3:
            route = "expedite_investigation"
        elif f["identification"] == "unidentified" or spec == 0:
            route = "request_more_information"
        else:
            route = "open_investigation"
        return {"intake_route": route, "allegation_specificity": spec, "investigation_priority": pr,
                "within_board_jurisdiction": "yes" if f["within_jurisdiction"] else "no",
                "filed_within_time_limit": "yes" if self._timely(f) else "no"}


# =========================================================================== schema 5
LADDER = ["letter of concern", "public reprimand", "12 months' probation", "24 months' probation",
          "36 months' probation", "90-day suspension", "1-year suspension", "revocation"]
# (profession, violation, guideline range lo..hi on the ladder, condition the guideline requires or None)
VIOLATIONS = [
    ("nursing", "documentation failures without patient harm", 0, 2, None),
    ("nursing", "a medication error causing temporary harm", 2, 3, "practice_monitor"),
    ("nursing", "unwitnessed controlled-drug wastage", 3, 5, "practice_restriction"),
    ("nursing", "practising on a lapsed licence", 0, 1, None),
    ("nursing", "failure to escalate a deteriorating patient", 2, 4, "practice_monitor"),
    ("engineering", "sealing work outside the engineer's competence", 2, 4, "competency_assessment"),
    ("engineering", "sealing drawings prepared without responsible charge", 3, 6, "practice_restriction"),
    ("engineering", "a negligent design that needed redesign", 2, 3, "practice_monitor"),
    ("contractors", "taking an excessive down payment", 0, 2, "practice_restriction"),
    ("contractors", "abandoning a project with consumer loss", 3, 5, "practice_restriction"),
    ("contractors", "unpermitted work that failed inspection", 1, 3, "competency_assessment"),
    ("pharmacy", "dispensing errors that reached patients", 2, 3, "practice_monitor"),
    ("home_inspection", "missing major defects in inspection reports", 1, 3, "competency_assessment"),
]
CONDITION_TEXT = {
    "practice_monitor": {"nursing": "a practice monitor reviewing the nurse's work and reporting quarterly",
                         "engineering": "an outside engineer reviewing every sealed set before issue",
                         "contractors": "third-party inspection of every job before final payment",
                         "pharmacy": "a supervising pharmacist checking every fill",
                         "home_inspection": "a senior inspector co-signing every report"},
    "practice_restriction": {"nursing": "no access to controlled drugs",
                             "engineering": "no sealing of documents prepared by outside firms",
                             "contractors": "no contracts above $50,000 and all deposits held in escrow",
                             "pharmacy": "no unsupervised dispensing",
                             "home_inspection": "no inspections of commercial buildings"},
    "competency_assessment": "an independent competency assessment before unrestricted practice",
    "coursework": "a course on the subject of the violation"}
PROFESSION_LABEL = {"nursing": "nurse", "engineering": "professional engineer", "contractors": "contractor",
                    "pharmacy": "pharmacy technician", "home_inspection": "home inspector"}


def _cond_text(key, profession):
    t = CONDITION_TEXT[key]
    return t[profession] if isinstance(t, dict) else t
AGGRAVATING = ["the conduct was repeated over several months", "the person harmed was especially vulnerable",
               "the licensee was not candid with investigators", "the licensee gained financially",
               "several patients or clients were affected", "the licensee continued after being warned"]
MITIGATING = ["the licensee self-reported", "remediation was completed before the hearing",
              "a long record without complaints", "full cooperation with the investigation",
              "restitution has already been paid", "the employer's systems contributed"]


class ConsentAgreementReview(Schema):
    name = "consent_agreement_review"
    description = "A settlement negotiated by board staff and a licensee, put to the board for acceptance against the sanctioning guidelines."
    surfaces = {"json": 0.5, "draft_consent_order": 0.25, "staff_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        prof, viol, lo, hi, req = pick(rng, VIOLATIONS)
        b = _board(rng, prof)
        n_agg, n_mit = pick(rng, [0, 1, 1, 2, 3]), pick(rng, [0, 1, 1, 2, 3])
        prior = pick_w(rng, {None: 5, 0: 1, 1: 1, 2: 2, 3: 2, 4: 1})
        offset = pick_w(rng, {-2: 1, -1: 2, 0: 4, 1: 2, 2: 1})
        has_req = req is None or chance(rng, 0.45)
        if reg == "clear":
            if chance(rng, 0.5):
                offset, has_req, prior = 0, True, None
                n_agg, n_mit = pick(rng, [(0, 1), (1, 1), (1, 2)])
            else:
                offset, n_agg, n_mit = pick(rng, [-3, 3]), *pick(rng, [(3, 0), (0, 3)])
        elif reg == "borderline":
            offset = pick(rng, [-1, 1])
        else:
            twist = pick(rng, ["prior_ignored", "heavy_aggravation_low_end", "missing_condition"])
            if twist == "prior_ignored":
                prior, offset = lo, 0
            elif twist == "heavy_aggravation_low_end":
                n_agg, n_mit, offset = 3, 0, 0
            else:
                has_req, offset = False, 0
                req = req or "practice_monitor"
        mid = (lo + hi) // 2
        if offset == 0:
            proposed = pick(rng, list(range(lo, hi + 1)))
            if reg == "conflicting" and n_agg == 3:
                proposed = lo
        elif offset < 0:
            proposed = lo + offset
        else:
            proposed = hi + offset
        proposed = max(0, min(len(LADDER) - 1, proposed))
        if prior is not None:
            prior = min(prior, len(LADDER) - 2)
        conds = []
        if req and has_req:
            conds.append(req)
        if chance(rng, 0.6):
            conds.append("coursework")
        if chance(rng, 0.5):
            conds.append("fine")
        return {"regime": reg, "board": b, "profession": prof, "violation": viol, "range_lo": lo, "range_hi": hi,
                "range_mid": mid, "required_condition": req, "proposed": proposed, "conditions": conds,
                "fine_amount": pick(rng, [250, 500, 1000, 2500, 5000]), "prior_rung": prior,
                "prior_year": rng.randint(2014, 2023), "aggravating": rng.sample(AGGRAVATING, n_agg),
                "mitigating": rng.sample(MITIGATING, n_mit), "licensee": person(rng), "staff_attorney": person(rng),
                "case_id": ident(rng, "CA", 5, "-"), "meeting": rand_date(rng)}

    def _balance(self, f):
        d = len(f["aggravating"]) + (1 if f["prior_rung"] is not None else 0) - len(f["mitigating"])
        return 0 if d <= -2 else 1 if d <= 0 else 2 if d == 1 else 3

    def _prior_ok(self, f):
        return f["prior_rung"] is None or f["proposed"] > f["prior_rung"]

    def _window(self, f):
        bal, lo, hi = self._balance(f), f["range_lo"], f["range_hi"]
        if bal >= 2:
            return max(lo, (lo + hi + 1) // 2), hi + (1 if bal == 3 else 0)
        return lo - (1 if bal == 0 else 0), hi

    def _fit(self, f):
        p, lo, hi = f["proposed"], f["range_lo"], f["range_hi"]
        return 0 if p < lo - 1 else 1 if p == lo - 1 else 2 if p <= hi else 3 if p == hi + 1 else 4

    def _terms(self, f):
        out = [LADDER[f["proposed"]]]
        for c in f["conditions"]:
            out.append(f"a fine of ${f['fine_amount']:,}" if c == "fine" else _cond_text(c, f["profession"]))
        return out

    def _guideline(self, f):
        b = f["board"]
        req = (f" Required condition: {_cond_text(f['required_condition'], f['profession'])}."
               if f["required_condition"] else "")
        return [f"{_cite(b, 0)} Guideline for {f['violation']}: from {LADDER[f['range_lo']]} to {LADDER[f['range_hi']]}.{req}",
                f"{_cite(b, 1)} Aggravating factors point to the upper part of the range and, where they clearly dominate, "
                "one step above it; mitigating factors point to the lower part and, where they clearly dominate, one step below it.",
                f"{_cite(b, 2)} A licensee disciplined before for any conduct must receive a heavier sanction than last time."]

    def render(self, f, rng, surface):
        b = f["board"]
        prior = (f"{LADDER[f['prior_rung']]} in {f['prior_year']}" if f["prior_rung"] is not None else "none")
        terms = self._terms(f)
        if surface == "json":
            return dumps(prune({
                "agreement": {"case": f["case_id"], "board": b["board"], "respondent": f["licensee"],
                              "profession": PROFESSION_LABEL[f["profession"]], "violation_admitted": f["violation"],
                              "proposed_terms": terms},
                "factors": {"aggravating": f["aggravating"] or ["none identified"],
                            "mitigating": f["mitigating"] or ["none identified"], "prior_discipline": prior},
                "guidelines": {"act": b["act"], "provisions": self._guideline(f)},
            }))
        if surface == "draft_consent_order":
            para = noisy(join_sentences([
                f"The respondent, {f['licensee']}, admits {f['violation']}.",
                f"The parties agree to the following: {'; '.join(terms)}.",
                f"Prior discipline: {prior}.",
                (f"In aggravation: {'; '.join(f['aggravating'])}." if f["aggravating"] else "No aggravating factors were identified."),
                (f"In mitigation: {'; '.join(f['mitigating'])}." if f["mitigating"] else "No mitigating factors were offered.")]), rng, 0.1)
            return (f"DRAFT CONSENT ORDER - {b['board']} - {f['case_id']}\n\n{para}\n\n"
                    f"This order takes effect only if the board accepts it.\n\n" + "\n".join(self._guideline(f)))
        body = kv_lines([("To", f"{b['board']} members"), ("From", f"{f['staff_attorney']}, staff counsel"),
                         ("Meeting", fmt_date(f["meeting"], rng)), ("Case", f"{f['case_id']} ({f['licensee']})"),
                         ("Violation", f["violation"]), ("Proposed", "; ".join(terms)), ("Prior discipline", prior)], rng)
        return (f"STAFF MEMO - proposed consent agreement\n{body}\n\nAggravating:\n"
                + (bullets(f["aggravating"], rng) if f["aggravating"] else "- none") + "\nMitigating:\n"
                + (bullets(f["mitigating"], rng) if f["mitigating"] else "- none") + "\n\n" + "\n".join(self._guideline(f)))

    def questions(self):
        return {
            "board_vote": choice("How should the board vote on the proposed consent agreement?", {
                "accept_as_proposed": "Accept the agreement as drafted.",
                "accept_with_amendments": "Accept it only with specific changes to the sanction or conditions.",
                "reject_too_harsh": "Reject it: the sanction is well beyond what the guidelines and factors support.",
                "reject_too_lenient": "Reject it: the sanction falls well short of what the guidelines and factors require.",
            }),
            "missing_condition": choice("Which condition, if any, does the agreement leave out that the guidelines require?", {
                "competency_assessment": "An independent competency assessment.",
                "none_missing": "Nothing: every required condition is included.",
                "practice_monitor": "A practice monitor reviewing the licensee's work.",
                "practice_restriction": "A restriction on the licensee's scope of practice.",
            }),
            "guideline_fit": score("Where does the proposed sanction sit against the guideline range for this violation?", [
                "Well below the range.",
                "One step below the range.",
                "Within the range.",
                "One step above the range.",
                "Well above the range.",
            ]),
            "aggravation_balance": score("How do the aggravating and mitigating factors balance out?", [
                "Mitigation clearly outweighs aggravation.",
                "They are roughly balanced.",
                "Aggravation somewhat outweighs mitigation.",
                "Aggravation clearly dominates.",
            ]),
            "prior_discipline_respected": noul(
                "The proposed sanction respects the rule on earlier discipline.",
                yes="The licensee has no earlier discipline, or the proposed sanction is heavier than the earlier one.",
                no="The licensee was disciplined before and the proposed sanction is no heavier than last time."),
        }

    def intended(self, f):
        lo_w, hi_w = self._window(f)
        p = f["proposed"]
        below = max(0, lo_w - p)
        above = max(0, p - hi_w)
        if not self._prior_ok(f):
            below = max(below, 1)
        missing = f["required_condition"] if f["required_condition"] and f["required_condition"] not in f["conditions"] else None
        if below >= 2:
            vote = "reject_too_lenient"
        elif above >= 2:
            vote = "reject_too_harsh"
        elif below or above or missing:
            vote = "accept_with_amendments"
        else:
            vote = "accept_as_proposed"
        return {"board_vote": vote, "missing_condition": missing or "none_missing", "guideline_fit": self._fit(f),
                "aggravation_balance": self._balance(f),
                "prior_discipline_respected": "yes" if self._prior_ok(f) else "no"}


# =========================================================================== schema 6
PROBATION_CASES = {
    "nursing": ["a medication error that harmed a patient", "falsified charting on a rehabilitation unit",
                "unwitnessed controlled-drug wastage", "failing to escalate a deteriorating patient"],
    "engineering": ["sealing a design outside the engineer's competence", "plan-stamping for a drafting firm",
                    "a negligent canopy design"],
    "contractors": ["abandoning two remodelling jobs", "unpermitted electrical work", "taking excessive deposits"],
    "pharmacy": ["repeated dispensing errors", "poor controlled-drug record-keeping"],
}
MONITOR = {"satisfactory": "work reviewed without concerns", "minor": "minor concerns, addressed when raised",
           "serious": "serious concerns about judgement and record-keeping"}
COURSE = {"done": "completed on time", "done_late": "completed after the due date", "overdue": "not completed; due date passed",
          "not_due": "not yet due"}
FINE = {"paid": "paid in full", "instalments": "instalment plan, payments current", "overdue": "two instalments missed"}
NEW_COMPLAINT = {"none": "none", "dismissed": "one complaint received and dismissed as unfounded",
                 "pending": "one new complaint under investigation", "substantiated": "a new complaint has been substantiated"}


class ProbationComplianceReview(Schema):
    name = "probation_compliance_review"
    description = "A licensee's compliance with the terms of board probation, reviewed for early termination, extension or revocation."
    surfaces = {"json": 0.5, "compliance_officer_memo": 0.25, "monitor_report_digest": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        prof = pick_w(rng, {"nursing": 4, "engineering": 2, "contractors": 3, "pharmacy": 1})
        b = _board(rng, prof)
        term = pick(rng, [12, 24, 36, 48])
        frac = pick(rng, [0.5, 0.67, 0.75])
        elapsed = max(3, int(term * rng.uniform(0.3, 0.95)))
        off = pick_w(rng, {0: 4, 2: 1, 4: 1, 7: 1})
        tolls = chance(rng, 0.65)
        missing = pick_w(rng, {0: 5, 1: 3, 2: 1})
        late = pick_w(rng, {0: 4, 1: 3, 2: 2})
        monitor = pick_w(rng, {"satisfactory": 3, "minor": 4, "serious": 2})
        course = pick_w(rng, {"done": 4, "done_late": 2, "overdue": 1, "not_due": 2})
        fine = pick_w(rng, {"paid": 4, "instalments": 3, "overdue": 1})
        checkins = pick_w(rng, {0: 5, 1: 3, 2: 2, 4: 1})
        complaint = pick_w(rng, {"none": 5, "dismissed": 2, "pending": 2, "substantiated": 1})
        request = chance(rng, 0.45)
        if reg == "clear":
            if chance(rng, 0.5):
                missing, late, monitor, course, fine, checkins, complaint, request, off = 0, 0, "satisfactory", "done", "paid", 0, "none", True, 0
                elapsed = int(term * rng.uniform(frac + 0.05, 0.95)) + 1
            else:
                missing, checkins, fine = 2, 4, "overdue"
                monitor, course = pick(rng, ["serious", "serious", "minor"]), "overdue"
                complaint = pick(rng, ["substantiated", "pending", "none"])
        elif reg == "borderline":
            # just over or under the early-termination line, or one small lapse
            request = True
            elapsed = max(3, int(round(term * frac)) + pick(rng, [-2, -1, 0, 1, 2]) + (off if tolls else 0))
            if chance(rng, 0.7):
                late, missing, checkins, complaint = pick(rng, [0, 0, 1]), 0, 0, pick(rng, ["none", "dismissed"])
                monitor = pick(rng, ["satisfactory", "satisfactory", "minor"])
                course, fine = pick(rng, ["done", "done", "done_late"]), pick(rng, ["paid", "instalments"])
        else:
            twist = pick(rng, ["perfect_but_complaint", "glowing_monitor_missed_terms", "time_off_counts"])
            request = True
            if twist == "perfect_but_complaint":
                missing, late, checkins, course, fine, monitor = 0, 0, 0, "done", "paid", "satisfactory"
                complaint = pick(rng, ["pending", "dismissed"])
                elapsed = int(term * rng.uniform(frac + 0.05, 0.95)) + 1
            elif twist == "glowing_monitor_missed_terms":
                monitor, missing, checkins, fine = "satisfactory", pick(rng, [1, 2]), pick(rng, [1, 2]), pick(rng, ["overdue", "instalments"])
            else:
                missing, late, checkins, course, fine, monitor, complaint = 0, 0, 0, "done", "paid", "satisfactory", "none"
                off, tolls = pick(rng, [4, 7]), True
                elapsed = int(round(term * frac)) + pick(rng, [1, 2])
        elapsed = min(elapsed, term - 1)
        off = min(off, max(0, elapsed - 2))
        return {"regime": reg, "board": b, "profession": prof, "original_violation": pick(rng, PROBATION_CASES[prof]),
                "term_months": term, "early_fraction": frac, "elapsed_months": elapsed, "months_not_practising": off,
                "tolling_rule": tolls, "reports_missing": missing, "reports_late": late, "monitor": monitor,
                "course": course, "fine": fine, "missed_checkins": checkins, "new_complaint": complaint,
                "early_termination_requested": request, "licensee": person(rng), "officer": person(rng),
                "order_id": ident(rng, "PO", 5, "-"), "review_date": rand_date(rng)}

    def _counted(self, f):
        return f["elapsed_months"] - (f["months_not_practising"] if f["tolling_rule"] else 0)

    def _compliance(self, f):
        pts = 2 * f["reports_missing"] + f["reports_late"] + f["missed_checkins"]
        pts += {"done": 0, "done_late": 1, "overdue": 2, "not_due": 0}[f["course"]]
        pts += {"paid": 0, "instalments": 0, "overdue": 2}[f["fine"]]
        return 3 if pts == 0 else 2 if pts <= 2 else 1 if pts <= 5 else 0

    def _concern(self, f):
        c = {"satisfactory": 0, "minor": 1, "serious": 2}[f["monitor"]]
        if f["new_complaint"] == "pending":
            c = max(c, 2)
        if f["new_complaint"] == "substantiated":
            c = 3
        return c

    def _eligible(self, f):
        return (self._counted(f) >= f["early_fraction"] * f["term_months"] and self._compliance(f) == 3
                and f["new_complaint"] not in ("pending", "substantiated"))

    def _rules(self, f):
        b = f["board"]
        pct = {0.5: "half", 0.67: "two thirds", 0.75: "three quarters"}[f["early_fraction"]]
        toll = ("Months in which the licensee does not practise do not count toward the term."
                if f["tolling_rule"] else "The term runs by calendar month whether or not the licensee practises.")
        return [f"{_cite(b, 0)} A probationer may ask for early termination after serving {pct} of the term, if every "
                "condition has been met on time and no complaint is pending. " + toll,
                f"{_cite(b, 1)} Non-compliance may lead to an extension of the term or, if serious or repeated, a "
                "petition to revoke probation and suspend the licence."]

    def render(self, f, rng, surface):
        b = f["board"]
        due = f["elapsed_months"] // 3
        reports = f"{due - f['reports_missing']} of {due} quarterly reports received" + (f", {f['reports_late']} late" if f["reports_late"] else "")
        req = "requested by the licensee" if f["early_termination_requested"] else "not requested"
        off = (f"{f['months_not_practising']} months without practising during the term" if f["months_not_practising"]
               else "practising throughout")
        if surface == "json":
            return dumps(prune({
                "order": {"id": f["order_id"], "board": b["board"], "licensee": f["licensee"],
                          "profession": PROFESSION_LABEL[f["profession"]], "for": f["original_violation"],
                          "term_months": f["term_months"],
                          "months_elapsed": f["elapsed_months"]},
                "compliance": {"employer_reports": reports, "practice_monitor": MONITOR[f["monitor"]],
                               "coursework": COURSE[f["course"]], "fine": FINE[f["fine"]],
                               "missed_check_in_calls": f["missed_checkins"], "practice_during_term": off,
                               "new_complaints": NEW_COMPLAINT[f["new_complaint"]]},
                "early_termination": req,
                "rules": {"act": b["act"], "provisions": self._rules(f)},
            }))
        if surface == "compliance_officer_memo":
            para = noisy(join_sentences([
                f"{f['licensee']} is {f['elapsed_months']} months into a {f['term_months']}-month probation ordered for "
                f"{f['original_violation']}.", f"Reports: {reports}.", f"The practice monitor reports {MONITOR[f['monitor']]}.",
                f"Coursework: {COURSE[f['course']]}. Fine: {FINE[f['fine']]}.",
                (f"Missed check-in calls: {f['missed_checkins']}." if f["missed_checkins"] else "No check-in calls missed."),
                f"The licensee has had {off}." if f["months_not_practising"] else "",
                f"New complaints: {NEW_COMPLAINT[f['new_complaint']]}.", f"Early termination is {req}."]), rng, 0.1)
            return (f"MEMO - probation review, order {f['order_id']}\nFrom: {f['officer']}, compliance officer\n\n{para}\n\n"
                    + "\n".join(self._rules(f)))
        rows = [["Term", f"{f['term_months']} months, {f['elapsed_months']} elapsed"], ["Practice", off],
                ["Quarterly reports", reports], ["Monitor", MONITOR[f["monitor"]]], ["Coursework", COURSE[f["course"]]],
                ["Fine", FINE[f["fine"]]], ["Missed check-ins", str(f["missed_checkins"])],
                ["New complaints", NEW_COMPLAINT[f["new_complaint"]]], ["Early termination", req]]
        return (f"Probation digest - {f['licensee']} ({PROFESSION_LABEL[f['profession']]}), order {f['order_id']}\n\n"
                + table(["Condition", "Status"], rows, rng) + "\n\n" + "\n".join(self._rules(f)))

    def questions(self):
        return {
            "probation_decision": choice("What should the board decide at this probation review?", {
                "continue_as_ordered": "Let the probation continue on its current terms.",
                "extend_probation": "Lengthen the probation because conditions were not fully met or concerns remain.",
                "grant_early_termination": "End the probation early as the licensee requests.",
                "petition_to_revoke_probation": "Start proceedings to revoke probation and suspend the licence.",
            }),
            "terms_compliance": score("How fully has the licensee complied with the probation terms?", [
                "Poorly: many conditions missed or overdue.",
                "Patchily: several lapses, some still outstanding.",
                "Mostly: a minor lapse or two, since put right.",
                "Fully: every condition met on time.",
            ]),
            "ongoing_practice_concern": score("How much concern does the licensee's current practice raise?", [
                "None: monitors report sound practice and no new complaints.",
                "Slight: minor points raised and dealt with.",
                "Real: serious monitor concerns or a new complaint under investigation.",
                "Grave: a new complaint has been substantiated.",
            ]),
            "early_termination_eligible": noul(
                "The licensee meets the rule's conditions for early termination of probation.",
                yes="Enough of the term has been served counting only qualifying months, every condition was met on time, "
                    "and no complaint is pending.",
                no="Too little qualifying time has been served, a condition was missed or late, or a complaint is pending."),
            "time_off_excluded": noul("Some of the elapsed months do not count toward completing the probation term."),
        }

    def intended(self, f):
        comp, conc, elig = self._compliance(f), self._concern(f), self._eligible(f)
        if f["early_termination_requested"] and elig and conc == 0:
            dec = "grant_early_termination"
        elif comp == 0 or conc == 3:
            dec = "petition_to_revoke_probation"
        elif comp == 1 or conc == 2:
            dec = "extend_probation"
        else:
            dec = "continue_as_ordered"
        return {"probation_decision": dec, "terms_compliance": comp, "ongoing_practice_concern": conc,
                "early_termination_eligible": "yes" if elig else "no",
                "time_off_excluded": "yes" if f["tolling_rule"] and f["months_not_practising"] > 0 else "no"}


WORKFLOW = Workflow(
    name="professional_license_discipline",
    title="Professional licence discipline",
    description="How a licensing board should handle complaints against licensed nurses, engineers, contractors and "
                "similar professionals under its written practice act and sanctioning guidelines.",
    domain="public administration / occupational licensing",
    schemas=[NursingComplaintDisposition(), EngineerSealedWorkComplaint(), ContractorConsumerComplaint(),
             ComplaintIntakeScreen(), ConsentAgreementReview(), ProbationComplianceReview()],
    notes="Boards, jurisdictions, licensees and firms are fictional; every state quotes the rule it is judged against. "
          "Outcomes turn on conduct, evidence, harm, history and remediation; no protected attribute or health "
          "condition of a licensee is a factor. No helpdesk threads, security alerts or bills.",
)
