"""Sport TUE and eligibility review: therapeutic use exemptions and anti-doping eligibility decisions under a
written, FICTIONAL anti-doping code.

Six schemas: a prospective TUE application, a retroactive TUE after an adverse finding, recognition of another
body's TUE for an event, a whereabouts-failure review, a reinstatement request at the end of a sanction or a
retirement, and a sanctioned athlete's request to take part in an activity. Every state quotes the code articles
it is judged against, and their variable parts are sampled per case (filing windows, notice periods, the
return-to-training window, deadlines, the prize-money line), so each answer comes from comparing facts with rules.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'). Borderline cases sit on a line (an
application filed on the last permitted day, a third failure 11.5 months after the first); conflicting cases pair a
strong medical file with a dose the condition does not need, or a glowing explanation with no records behind it.
`intended(f)` records the answer logic a careful committee member would apply to the RENDERED facts; the builder
does not use it.

Agencies, federations, clubs, athletes and physicians are fictional; medication names are generic and their status
comes only from the fictional code quoted in the state. TUEs are judged on clinical evidence, alternatives, dose and
enhancement; eligibility on sanctions, credit, testing availability and costs. No protected attribute (age, sex,
race, religion, disability status, nationality) decides anything, and there are no sex-category or national-team
eligibility cases.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
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

CODES = ["Continental Sport Integrity Code", "Unified Clean Sport Code", "Inter-Federation Anti-Doping Code",
         "Common Code on Doping in Sport"]
AGENCIES = ["Northmark Anti-Doping Agency", "Valdren Sport Integrity Office", "Kestmoor Clean Sport Authority",
            "Orsenne Anti-Doping Commission", "Halvard Sport Integrity Unit", "Brisaland Anti-Doping Office",
            "Tarrow Isles Clean Sport Agency", "Ellmark Doping Control Authority"]
# (sport, category used for enhancement judgements)
SPORTS = [("road cycling", "endurance"), ("marathon running", "endurance"), ("rowing", "endurance"),
          ("cross-country skiing", "endurance"), ("swimming", "endurance"), ("triathlon", "endurance"),
          ("weightlifting", "strength"), ("sprint athletics", "power"), ("canoe sprint", "power"),
          ("speed skating", "power"), ("artistic gymnastics", "power"), ("judo", "combat"), ("boxing", "combat"),
          ("wrestling", "combat"), ("archery", "precision"), ("pistol shooting", "precision"), ("biathlon", "precision"),
          ("curling", "precision"), ("football", "team"), ("rugby union", "team"), ("basketball", "team"),
          ("field hockey", "team"), ("volleyball", "team"), ("water polo", "team")]
FED_B = ["Federation", "Union", "Association", "Confederation"]
REGIONS = ["Northmark", "Valdren", "Kestmoor", "Orsenne", "Halvard", "Brisaland", "Tarrow Isles", "Ellmark"]
LEVELS = {"international": "international (registered testing pool)",
          "national": "national (national testing pool)",
          "regional": "regional (no testing pool)"}


def _code(rng):
    return {"code": f"{pick(rng, CODES)} ({pick(rng, [2024, 2025, 2026])} edition)", "art": rng.randint(4, 9)}


def _art(c, n, sub=None):
    return f"Art. {c['art']}.{n}" + (f"({sub})" if sub else "")


GOVERNING = {"road cycling": "Cycling", "marathon running": "Athletics", "sprint athletics": "Athletics",
             "rowing": "Rowing", "cross-country skiing": "Ski", "swimming": "Swimming", "triathlon": "Triathlon",
             "weightlifting": "Weightlifting", "canoe sprint": "Canoe", "speed skating": "Skating",
             "artistic gymnastics": "Gymnastics", "judo": "Judo", "boxing": "Boxing", "wrestling": "Wrestling",
             "archery": "Archery", "pistol shooting": "Shooting", "biathlon": "Biathlon", "curling": "Curling",
             "football": "Football", "rugby union": "Rugby", "basketball": "Basketball", "field hockey": "Hockey",
             "volleyball": "Volleyball", "water polo": "Aquatics"}


def _fed(rng, sport):
    return f"{pick(rng, REGIONS)} {GOVERNING[sport]} {pick(rng, FED_B)}"


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _mo(n):
    return f"{n} month{'' if n == 1 else 's'}"


# (condition, medication, code class, status, enhancement weight by sport category, permitted alternative or None,
#  diagnostic tests a specialist would hold)
TUE_PAIRS = [
    ("attention-deficit hyperactivity disorder", "methylphenidate", "stimulants", "in_competition",
     {"precision": 2, "default": 1}, "a non-stimulant medication", "a structured psychiatric assessment with rating scales"),
    ("narcolepsy", "modafinil", "stimulants", "in_competition", {"default": 1}, "scheduled sleep and behavioural measures",
     "an overnight sleep study and daytime sleep-latency test"),
    ("type 1 diabetes", "insulin", "metabolic modulators", "at_all_times", {"default": 0}, None,
     "C-peptide and islet-antibody tests"),
    ("primary hypogonadism", "testosterone", "anabolic agents", "at_all_times",
     {"strength": 3, "power": 3, "combat": 3, "default": 2}, None, "two morning testosterone levels with LH and FSH"),
    ("severe asthma", "inhaled salbutamol at a high daily dose", "beta-2 agonists",
     "at_all_times", {"endurance": 2, "default": 1}, "standard-dose inhalers within the ceiling",
     "spirometry with a bronchial challenge test"),
    ("a flare of ulcerative colitis", "oral prednisolone", "glucocorticoids", "in_competition", {"default": 1},
     "mesalazine", "colonoscopy with biopsy"),
    ("chronic sinusitis with nasal polyps", "an oral glucocorticoid course", "glucocorticoids", "in_competition",
     {"default": 1}, "nasal steroid sprays", "a CT scan of the sinuses"),
    ("hypertension", "a beta-blocker", "beta-blockers", "particular_sports", {"precision": 3, "default": 0},
     "an ACE inhibitor", "24-hour blood-pressure monitoring"),
    ("atrial fibrillation", "a beta-blocker", "beta-blockers", "particular_sports", {"precision": 3, "default": 0},
     "a calcium-channel blocker", "an ECG and a 48-hour heart monitor"),
    ("anaemia of chronic kidney disease", "an erythropoiesis-stimulating agent", "peptide hormones", "at_all_times",
     {"endurance": 3, "default": 2}, "iron infusions", "haemoglobin trends and kidney-function tests"),
    ("adult growth-hormone deficiency", "somatropin", "peptide hormones", "at_all_times",
     {"strength": 3, "power": 3, "default": 2}, None, "a growth-hormone stimulation test"),
    ("pain after shoulder surgery", "oxycodone", "narcotics", "in_competition", {"combat": 2, "default": 1},
     "paracetamol and anti-inflammatories", "operation notes and the surgeon's discharge letter"),
    ("fluid loss from severe gastroenteritis", "an intravenous infusion above 100 mL", "prohibited methods",
     "at_all_times", {"endurance": 2, "default": 1}, "oral rehydration", "emergency-department notes and blood tests"),
    ("Crohn's disease", "oral budesonide", "glucocorticoids", "in_competition", {"default": 1}, "a biologic infusion",
     "endoscopy and an MRI of the small bowel"),
]
STATUS_TEXT = {"in_competition": "prohibited in competition only", "at_all_times": "prohibited at all times",
               "particular_sports": "prohibited in particular sports, including precision sports"}


def _enh_base(pair, cat):
    w = pair[4]
    return w.get(cat, w["default"])


# =========================================================================== schema 1
class ProspectiveTueApplication(Schema):
    name = "prospective_tue_application"
    description = "An athlete applies ahead of competition for an exemption to use a prohibited medication; the TUE committee rules."
    surfaces = {"json": 0.5, "committee_worksheet": 0.25, "physician_letter": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        c = _code(rng)
        sport, cat = pick(rng, SPORTS)
        pair = pick(rng, TUE_PAIRS)
        evidence = pick_w(rng, {"specialist": 4, "gp_letter": 3, "self_report": 2})
        test_age = pick(rng, [0, 1, 2, 5, 8])
        valid_years = pick(rng, [4, 4, 6])
        severity = pick_w(rng, {"severe": 3, "moderate": 4, "mild": 3})
        alt = "none_exists" if pair[5] is None else pick_w(rng, {"tried_failed": 4, "unsuitable_documented": 2, "not_tried": 3})
        dose = pick_w(rng, {"standard": 6, "above_standard": 2, "unstated": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                pair = pick(rng, [p for p in TUE_PAIRS if _enh_base(p, cat) <= 1])
                evidence, test_age, severity, dose = "specialist", pick(rng, [0, 1]), "severe", "standard"
                alt = "none_exists" if pair[5] is None else "tried_failed"
            else:
                evidence, dose, severity = "self_report", pick(rng, ["above_standard", "unstated"]), "mild"
                alt = "none_exists" if pair[5] is None else "not_tried"
        else:
            # one criterion is in doubt and the rest are sound: which one is the case's question
            focus = pick_w(rng, {"diagnosis": 4, "impairment": 2, "enhancement": 2, "alternative": 2, "none": 2})
            evidence, test_age, severity, dose = "specialist", pick(rng, [0, 1, 2]), pick(rng, ["severe", "moderate"]), "standard"
            alt = pick_w(rng, {"tried_failed": 4, "unsuitable_documented": 3, "not_tried": 1})
            if focus == "diagnosis":
                evidence = pick_w(rng, {"gp_letter": 2, "self_report": 1, "specialist": 2})
                if evidence == "specialist":
                    test_age = valid_years + pick(rng, [0, 1, 3])
            elif focus == "impairment":
                severity = "mild"
            elif focus == "enhancement":
                if chance(rng, 0.5):
                    dose = "above_standard"
                pair = pick(rng, [p for p in TUE_PAIRS if _enh_base(p, cat) >= 2] or TUE_PAIRS)
            elif focus == "alternative":
                pair = pick(rng, [p for p in TUE_PAIRS if p[5] is not None])
                alt = "not_tried"
            if reg == "borderline" and focus == "none":
                test_age = pick(rng, [valid_years - 1, valid_years])      # on the edge of the code's test-age limit
        if pair[5] is None:
            alt = "none_exists"
        return {"regime": reg, "code": c, "sport": sport, "sport_category": cat, "condition": pair[0],
                "medication": pair[1], "drug_class": pair[2], "status": pair[3], "alternative": pair[5],
                "tests": pair[6], "evidence": evidence, "test_age_years": test_age, "valid_years": valid_years,
                "severity": severity, "alt_history": alt, "dose": dose, "enh_base": _enh_base(pair, cat),
                "level": pick(rng, ["international", "national"]), "athlete": person(rng), "physician": person(rng),
                "agency": pick(rng, AGENCIES), "federation": _fed(rng, sport),
                "days_to_event": pick(rng, [12, 25, 40, 75, 140]), "requested_months": pick(rng, [12, 24, 48]),
                "app_id": ident(rng, "TUE", 5, "-"), "submitted": rand_date(rng)}

    def _dx(self, f):
        if f["evidence"] == "self_report":
            return 0
        if f["evidence"] == "gp_letter":
            return 1
        return 3 if f["test_age_years"] < f["valid_years"] else 2

    def _enh(self, f):
        e = f["enh_base"] + (1 if f["dose"] == "above_standard" else 0)
        if f["status"] == "particular_sports" and f["sport_category"] != "precision":
            e = 0
        return _clamp(e)

    def _alt_ok(self, f):
        return f["alt_history"] in ("tried_failed", "unsuitable_documented", "none_exists")

    def _criteria(self, f):
        c = f["code"]
        return [f"{_art(c, 2)} A TUE is granted only if all four hold: (a) the athlete has a diagnosed medical condition, "
                f"supported by tests no older than {f['valid_years']} years; (b) without the treatment the athlete's health "
                "would be significantly impaired; (c) the treatment is unlikely to improve performance beyond a return to "
                "the athlete's normal health; (d) no reasonable permitted alternative exists.",
                f"{_art(c, 3)} The committee may limit validity or dose, or ask for missing evidence, before deciding."]

    def _alt_text(self, f):
        if f["alt_history"] == "none_exists":
            return "no permitted alternative is available for this condition"
        return {"tried_failed": f"{f['alternative']} was tried for several months without adequate control",
                "unsuitable_documented": f"{f['alternative']} is documented as unsuitable because of side effects",
                "not_tried": f"{f['alternative']} has not been tried"}[f["alt_history"]]

    def _evidence_text(self, f):
        if f["evidence"] == "specialist":
            return f"specialist report with {f['tests']}, performed {f['test_age_years']} year(s) ago" if f["test_age_years"] \
                else f"specialist report with {f['tests']}, performed this year"
        return {"gp_letter": "a family-doctor letter describing symptoms; no test results attached",
                "self_report": "the athlete's own statement and a pharmacy receipt"}[f["evidence"]]

    def render(self, f, rng, surface):
        sev = {"severe": "symptoms stop the athlete training and disturb daily life",
               "moderate": "symptoms limit training on most days",
               "mild": "symptoms are occasional and mostly managed"}[f["severity"]]
        dose = {"standard": "a standard therapeutic dose", "above_standard": "a dose above the usual therapeutic range",
                "unstated": None}[f["dose"]]
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["app_id"], "athlete": f["athlete"], "level": LEVELS[f["level"]],
                                "sport": f["sport"], "federation": f["federation"], "agency": f["agency"],
                                "first_competition_in_days": f["days_to_event"],
                                "validity_requested_months": f["requested_months"]},
                "medical": {"condition": f["condition"], "medication": f["medication"], "dose": dose,
                            "severity": sev, "diagnostic_evidence": self._evidence_text(f),
                            "alternatives": self._alt_text(f)},
                "code": {"name": f["code"]["code"], "medication_status": f"{f['drug_class']}: {STATUS_TEXT[f['status']]}",
                         "articles": self._criteria(f)},
            }))
        if surface == "committee_worksheet":
            rows = [["Athlete", f"{f['athlete']} ({f['sport']}, {f['level']})"], ["Condition", f["condition"]],
                    ["Medication", f"{f['medication']} ({f['drug_class']}, {STATUS_TEXT[f['status']]})"],
                    ["Dose", show(dose, rng)], ["Severity", sev], ["Evidence", self._evidence_text(f)],
                    ["Alternatives", self._alt_text(f)], ["Validity asked", f"{f['requested_months']} months"],
                    ["First event", f"in {f['days_to_event']} days"]]
            return (f"TUE COMMITTEE WORKSHEET - {f['agency']} - {f['app_id']}\n\n" + table(["Item", "Entry"], rows, rng)
                    + f"\n\n{f['code']['code']}:\n" + "\n".join(self._criteria(f)))
        letter = noisy(join_sentences([
            f"I have treated {f['athlete']} for {f['condition']}.", f"At present {sev}.",
            f"I propose {f['medication']}" + (f" at {dose}." if dose else "."),
            f"Diagnostic basis: {self._evidence_text(f)}.", f"On alternatives: {self._alt_text(f)}.",
            pick(rng, ["I do not believe this treatment gives any competitive advantage.",
                       "The treatment simply allows normal health.", ""])]), rng, 0.12)
        return (f"Physician letter attached to TUE application {f['app_id']} ({f['agency']})\n"
                f"From: Dr {f['physician']}\n\n{letter}\n\n[application form]\n"
                + kv_lines([("Sport", f["sport"]), ("Level", LEVELS[f["level"]]),
                            ("Medication class", f"{f['drug_class']} - {STATUS_TEXT[f['status']]}"),
                            ("Validity requested", f"{f['requested_months']} months")], rng)
                + "\n\n" + "\n".join(self._criteria(f)))

    def questions(self):
        return {
            "tue_ruling": choice("How should the TUE committee rule on this application?", {
                "deny": "Refuse the exemption; the code's conditions are clearly not met.",
                "grant": "Grant the exemption for the period and dose requested.",
                "grant_limited": "Grant it with a shorter validity, a dose cap or a review date.",
                "request_more_information": "Hold the decision and ask for missing medical evidence.",
            }),
            "weakest_criterion": choice("Which of the code's conditions is least well supported by the file?", {
                "all_conditions_met": "None: every condition is adequately supported.",
                "diagnosis": "The diagnosis and its supporting tests.",
                "health_impairment": "That health would be significantly impaired without the treatment.",
                "no_enhancement": "That the treatment would do no more than restore normal health.",
                "no_permitted_alternative": "That no reasonable permitted alternative exists.",
            }),
            "diagnostic_evidence_strength": score("How strong is the evidence for the diagnosis?", [
                "Absent: the athlete's word and a prescription.",
                "Thin: a doctor's description without tests.",
                "Adequate but dated: proper tests, older than the code accepts.",
                "Strong: a specialist diagnosis with current tests.",
            ]),
            "enhancement_concern": score("How much concern is there that the treatment would enhance performance in this sport?", [
                "None: the substance offers no advantage in this sport at this dose.",
                "Slight: a theoretical advantage, well controlled by the dose.",
                "Real: the substance can help performance in this kind of sport.",
                "High: a potent performance aid in this sport, or a dose beyond therapeutic need.",
            ]),
            "alternative_ruled_out": noul(
                "The file shows that no reasonable permitted alternative would do.",
                yes="Permitted alternatives were tried and failed, are documented as unsuitable, or do not exist.",
                no="A permitted alternative exists and has not been tried or ruled out."),
        }

    def intended(self, f):
        dx, enh, alt = self._dx(f), self._enh(f), self._alt_ok(f)
        if dx <= 2:
            weak = "diagnosis"
        elif not alt:
            weak = "no_permitted_alternative"
        elif enh >= 3:
            weak = "no_enhancement"
        elif f["severity"] == "mild":
            weak = "health_impairment"
        else:
            weak = "all_conditions_met"
        if dx == 0 or (enh >= 3 and f["dose"] == "above_standard"):
            ruling = "deny"
        elif dx <= 2 or not alt or f["dose"] == "unstated":
            ruling = "request_more_information"
        elif enh >= 2 or f["dose"] == "above_standard" or f["severity"] == "mild":
            ruling = "grant_limited"
        else:
            ruling = "grant"
        return {"tue_ruling": ruling, "weakest_criterion": weak, "diagnostic_evidence_strength": dx,
                "enhancement_concern": enh, "alternative_ruled_out": "yes" if alt else "no"}


# =========================================================================== schema 2
LAB_NAME = {"inhaled salbutamol at a high daily dose": "salbutamol above the urinary threshold",
            "an oral glucocorticoid course": "a glucocorticoid", "oral prednisolone": "prednisolone",
            "oral budesonide": "budesonide"}
EVENTS = {"international": ["a World Cup round", "the continental championships", "a world-ranking final", "a Grand Prix meeting"],
          "national": ["the national championships", "a national selection trial", "a national league match", "a national cup final"],
          "regional": ["a regional qualifier", "a county league match", "a club open meet", "a regional championship"]}
TREATMENT_CONTEXT = {
    "emergency": ["were treated in an emergency department for {condition}", "were given {medication} by paramedics at the roadside",
                  "were admitted overnight to hospital with {condition}"],
    "urgent_short_notice": ["were prescribed {medication} by a doctor two days before the test, with no time to apply",
                            "started {medication} the evening before the test when symptoms suddenly worsened"],
    "planned": ["had been taking {medication} for several weeks on a doctor's advice",
                "began a planned course of {medication} a month before the test"],
}


class RetroactiveTueAfterFinding(Schema):
    name = "retroactive_tue_after_finding"
    description = "After an adverse analytical finding, the athlete applies for a retroactive TUE; the committee decides whether one can be granted."
    surfaces = {"json": 0.5, "results_management_email": 0.25, "committee_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        c = _code(rng)
        sport, cat = pick(rng, SPORTS)
        pair = pick(rng, [p for p in TUE_PAIRS if p[3] != "particular_sports" and p[2] != "prohibited methods"])
        context = pick_w(rng, {"emergency": 3, "urgent_short_notice": 2, "planned": 4})
        level = pick_w(rng, {"international": 4, "national": 3, "regional": 3})
        records = pick_w(rng, {"full": 5, "partial": 3, "missing": 2})
        consistency = pick_w(rng, {"consistent": 5, "borderline": 3, "inconsistent": 2})
        deadline = pick(rng, [15, 21, 30])
        filed = pick(rng, [3, 9, 14, 20, 26, 40])
        severity = pick_w(rng, {"serious": 4, "moderate": 4, "minor": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                context, records, consistency, severity = "emergency", "full", "consistent", "serious"
                filed = rng.randint(2, deadline - 2)
            else:
                context, level, records, consistency = "planned", "international", "partial", "inconsistent"
        elif reg == "borderline":
            filed = near(rng, deadline, 3, lo=1, integer=True)
            consistency = pick(rng, ["consistent", "borderline"])
        else:
            twist = pick(rng, ["strong_case_wrong_route", "good_route_no_records"])
            if twist == "strong_case_wrong_route":
                context, level, records, consistency, severity = "planned", "international", "full", "consistent", "serious"
            else:
                context, records = pick(rng, ["emergency", "urgent_short_notice"]), "missing"
        return {"regime": reg, "code": c, "sport": sport, "condition": pair[0], "medication": pair[1],
                "drug_class": pair[2], "status": pair[3], "context": context,
                "context_text": pick(rng, TREATMENT_CONTEXT[context]).format(condition=pair[0], medication=pair[1]),
                "level": level, "records": records, "consistency": consistency, "severity": severity,
                "deadline_days": deadline, "filed_after_days": filed, "athlete": person(rng),
                "lab_name": LAB_NAME.get(pair[1], pair[1]),
                "agency": pick(rng, AGENCIES), "case_id": ident(rng, "RM", 5, "-"), "test_date": rand_date(rng),
                "test_type": pick(rng, ["in-competition", "out-of-competition"]), "event": pick(rng, EVENTS[level])}

    def _ground(self, f):
        return f["context"] in ("emergency", "urgent_short_notice") or f["level"] == "regional"

    def _justification(self, f):
        j = {"full": 2, "partial": 1, "missing": 0}[f["records"]] + {"serious": 1, "moderate": 0, "minor": -1}[f["severity"]]
        return _clamp(j)

    def _consistency(self, f):
        return {"consistent": 3, "borderline": 2, "inconsistent": 0}[f["consistency"]] - (1 if f["records"] == "missing" and f["consistency"] != "inconsistent" else 0)

    def _articles(self, f):
        c = f["code"]
        return [f"{_art(c, 5)} A TUE may be granted retroactively only if: (a) emergency or urgent treatment was needed; "
                "(b) there was not enough time or opportunity to apply before the sample was collected; or (c) the athlete "
                "was a regional-level competitor not required to apply in advance. The conditions of the code's TUE "
                "criteria must also be met.",
                f"{_art(c, 6)} The application must be filed within {f['deadline_days']} days of notice of the finding. In "
                "exceptional cases where none of the grounds applies, a fairness panel may still consider it.",
                f"{_art(c, 7)} The reported treatment must explain the substance, concentration and timing found."]

    def render(self, f, rng, surface):
        cons = {"consistent": "the concentration and timing fit the reported treatment",
                "borderline": "the concentration is at the upper edge of what the reported dose could produce",
                "inconsistent": "the concentration is far above what the reported treatment could produce"}[f["consistency"]]
        recs = {"full": "complete hospital and pharmacy records", "partial": "a discharge summary but no prescribing record",
                "missing": "no medical records yet; the athlete says they will follow"}[f["records"]]
        sev = {"serious": "a serious episode", "moderate": "a moderate episode", "minor": "a minor complaint"}[f["severity"]]
        if surface == "json":
            return dumps(prune({
                "finding": {"case": f["case_id"], "substance": f"{f['lab_name']}; the code lists it under {f['drug_class']}, {STATUS_TEXT[f['status']]}",
                            "sample": f"{f['test_type']}, {f['event']}", "date": fmt_date(f["test_date"], style="iso"),
                            "laboratory_comment": cons},
                "athlete": {"name": f["athlete"], "sport": f["sport"], "level": LEVELS[f["level"]]},
                "application": {"filed_days_after_notice": f["filed_after_days"], "account": f"The athlete says they {f['context_text']}.",
                                "medical_records": recs, "episode": sev},
                "code": {"name": f["code"]["code"], "articles": self._articles(f)},
            }))
        if surface == "results_management_email":
            body = noisy(join_sentences([
                f"Following the {f['test_type']} test at {f['event']} on {fmt_date(f['test_date'], rng)}, the laboratory "
                f"reported {f['lab_name']} in {f['athlete']}'s sample.",
                f"The athlete ({f['sport']}, {LEVELS[f['level']]}) has now applied for a retroactive exemption, "
                f"{f['filed_after_days']} days after being notified.", f"The athlete says they {f['context_text']} ({sev}).",
                f"Records supplied: {recs}.", f"Lab comment: {cons}."]), rng, 0.12)
            return (f"Subject: {f['case_id']} - retroactive TUE application\nFrom: results management, {f['agency']}\n\n"
                    f"{body}\n\n" + "\n".join(self._articles(f)))
        rows = [("Case", f["case_id"]), ("Athlete", f"{f['athlete']} - {f['sport']}"), ("Level", LEVELS[f["level"]]),
                ("Substance found", f"{f['lab_name']}, listed under {f['drug_class']}"), ("Athlete's account", f"they {f['context_text']}; {sev}"),
                ("Records", recs), ("Laboratory", cons),
                ("Filed", f"{f['filed_after_days']} days after notice (limit {f['deadline_days']})")]
        return "COMMITTEE MEMO - retroactive TUE\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._articles(f))

    def questions(self):
        return {
            "retroactive_ruling": choice("How should the committee deal with the retroactive TUE application?", {
                "deny_and_continue_case": "Refuse it; results management of the finding continues.",
                "grant_retroactive": "Grant the exemption with effect from before the sample, closing the case.",
                "refer_to_fairness_panel": "Send it to the fairness panel because no listed ground applies but the medical case is strong.",
                "request_medical_records": "Ask for the missing medical records before ruling.",
            }),
            "medical_justification": score("How well is the athlete's treatment medically justified on the evidence?", [
                "Not shown: no records and a trivial complaint.",
                "Weakly: partial records or a minor complaint.",
                "Reasonably: records support a genuine need.",
                "Clearly: full records of a serious episode.",
            ]),
            "explains_finding": score("How well does the reported treatment explain what the laboratory found?", [
                "Not at all: the concentration or timing does not fit.",
                "Doubtfully: the fit depends on unrecorded details.",
                "Plausibly: it fits, though near the edge.",
                "Fully: substance, amount and timing all fit.",
            ]),
            "retroactive_ground_applies": noul(
                "One of the code's grounds for a retroactive exemption applies to this athlete.",
                yes="The treatment was an emergency or too urgent to apply for in time, or the athlete is a regional-level competitor.",
                no="The treatment was planned with time to apply, and the athlete was required to apply in advance."),
            "filed_on_time": noul("The retroactive application was filed within the code's deadline."),
        }

    def intended(self, f):
        ground, just, cons = self._ground(f), self._justification(f), self._consistency(f)
        timely = f["filed_after_days"] <= f["deadline_days"]
        if not timely:
            ruling = "deny_and_continue_case"
        elif f["records"] == "missing" and (ground or f["severity"] == "serious"):
            ruling = "request_medical_records"
        elif cons <= 1:
            ruling = "deny_and_continue_case"
        elif ground and just >= 2:
            ruling = "grant_retroactive"
        elif not ground and just >= 2:
            ruling = "refer_to_fairness_panel"
        else:
            ruling = "deny_and_continue_case"
        return {"retroactive_ruling": ruling, "medical_justification": just, "explains_finding": _clamp(cons),
                "retroactive_ground_applies": "yes" if ground else "no", "filed_on_time": "yes" if timely else "no"}


# =========================================================================== schema 3
GRANTING_BODIES = {"national_agency": "the athlete's national anti-doping agency",
                   "other_federation": "another sport's international federation",
                   "team_doctor": "the athlete's club doctor, by letter (not an anti-doping body)"}


class TueRecognitionRequest(Schema):
    name = "tue_recognition_request"
    description = "An athlete asks an event's federation to recognise a TUE granted by another body, ahead of an international event."
    surfaces = {"json": 0.5, "recognition_form": 0.25, "federation_email_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        c = _code(rng)
        sport, cat = pick(rng, SPORTS)
        pair = pick(rng, TUE_PAIRS)
        body = pick_w(rng, {"national_agency": 6, "other_federation": 2, "team_doctor": 1})
        docs = {"diagnosis": chance(rng, 0.8), "tests": chance(rng, 0.7), "specialist": chance(rng, 0.7),
                "translation": chance(rng, 0.75)}
        days_to_event = pick(rng, [5, 9, 18, 35, 70, 120])
        event_len = pick(rng, [2, 4, 9, 14])
        expiry_offset = pick(rng, [-20, -3, 0, 2, 30, 200])  # days between TUE expiry and the event's last day
        dose_change = pick_w(rng, {"same": 6, "lower": 1, "higher": 2})
        committee_in = pick(rng, [7, 14, 21, 30])
        if reg == "clear":
            if chance(rng, 0.5):
                body, docs, expiry_offset, dose_change = "national_agency", {k: True for k in docs}, 200, "same"
            else:
                body = "team_doctor"
                docs = {k: chance(rng, 0.3) for k in docs}
        elif reg == "borderline":
            expiry_offset = pick(rng, [-2, -1, 0, 1, 3])
        else:
            twist = pick(rng, ["complete_but_higher_dose", "incomplete_but_urgent"])
            if twist == "complete_but_higher_dose":
                docs, dose_change, body = {k: True for k in docs}, "higher", "national_agency"
            else:
                docs["tests"], docs["specialist"], days_to_event = False, False, pick(rng, [5, 9])
        return {"regime": reg, "code": c, "sport": sport, "condition": pair[0], "medication": pair[1],
                "drug_class": pair[2], "granting_body": body, "docs": docs, "days_to_event": days_to_event,
                "event_days": event_len, "expiry_offset": expiry_offset, "dose_change": dose_change,
                "committee_in_days": committee_in, "athlete": person(rng), "federation": _fed(rng, sport),
                "national_agency": pick(rng, AGENCIES), "event": pick(rng, ["continental championships", "a World Series final",
                                                                           "the inter-federation games", "a world-ranking event",
                                                                           "an Olympic-style multi-sport festival"]),
                "request_id": ident(rng, "REC", 4, "-"), "today": rand_date(rng)}

    def _completeness(self, f):
        if f["granting_body"] == "team_doctor":
            return min(1, sum(f["docs"].values()) // 2)
        return _clamp(sum([f["docs"]["diagnosis"], f["docs"]["tests"], f["docs"]["specialist"]]) + (0 if f["docs"]["translation"] else -1))

    def _covers(self, f):
        return f["expiry_offset"] >= 0

    def _urgency(self, f):
        d = f["days_to_event"]
        if d <= 7:
            return 3
        if d < f["committee_in_days"]:
            return 2
        return 1 if d <= 60 else 0

    def _dates(self, f):
        start = f["today"] + timedelta(days=f["days_to_event"])
        end = start + timedelta(days=f["event_days"] - 1)
        expiry = end + timedelta(days=f["expiry_offset"])
        return start, end, expiry

    def _articles(self, f):
        c = f["code"]
        return [f"{_art(c, 8)} A federation recognises a TUE granted by a national anti-doping agency or another "
                "federation if the file shows the code's criteria are met; a document from a club or team doctor is not "
                "a TUE. Recognition never extends beyond the TUE's own expiry or above its approved dose.",
                f"{_art(c, 9)} Files that are incomplete, untranslated or disputed go to the federation's TUE committee, "
                f"which meets in {f['committee_in_days']} days."]

    def render(self, f, rng, surface):
        start, end, expiry = self._dates(f)
        have = [k for k, v in f["docs"].items() if v]
        lack = [k for k, v in f["docs"].items() if not v]
        dose = {"same": "the same dose as approved", "lower": "a lower dose than approved",
                "higher": "a higher dose than the original TUE approved"}[f["dose_change"]]
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "received": fmt_date(f["today"], style="iso"),
                            "federation": f["federation"], "athlete": f["athlete"],
                            "sport": f["sport"], "event": f["event"], "event_dates":
                            f"{fmt_date(start, style='iso')} to {fmt_date(end, style='iso')}"},
                "original_tue": {"granted_by": GRANTING_BODIES[f["granting_body"]], "condition": f["condition"],
                                 "medication": f"{f['medication']} ({f['drug_class']})", "expires": fmt_date(expiry, style="iso"),
                                 "current_use": dose},
                "file": {"documents_supplied": have or None, "documents_missing": lack or None},
                "code": {"name": f["code"]["code"], "articles": self._articles(f)},
            }))
        if surface == "recognition_form":
            rows = [("Request", f["request_id"]), ("Received", fmt_date(f["today"], rng)),
                    ("Athlete", f"{f['athlete']} ({f['sport']})"), ("Event", f["event"]),
                    ("Event dates", f"{fmt_date(start, rng)} - {fmt_date(end, rng)}"),
                    ("Original TUE granted by", GRANTING_BODIES[f["granting_body"]]),
                    ("Condition / medication", f"{f['condition']} / {f['medication']}"),
                    ("Original TUE expires", fmt_date(expiry, rng)), ("Current use", dose),
                    ("Documents supplied", ", ".join(have) or "none"), ("Missing", ", ".join(lack) or "none")]
            return "TUE RECOGNITION REQUEST\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._articles(f))
        msgs = [(f["athlete"], f"I'm {pick(rng, ['competing', 'entered', 'selected'])} at {f['event']} from {fmt_date(start, rng)} and need my TUE for "
                               f"{f['medication']} recognised. It was issued by {GRANTING_BODIES[f['granting_body']]}."),
                ("Federation medical office", f"Thanks. Our record shows it expires {fmt_date(expiry, rng)}. "
                                              f"Documents received: {', '.join(have) or 'none'}."),
                (f["athlete"], f"I'm now on {dose}." + (" I can send the rest of the file later." if lack else ""))]
        thread = "\n\n".join(f"From: {w}\n{noisy(t, rng, 0.1)}" for w, t in msgs)
        return (f"Subject: {f['request_id']} TUE recognition ({f['federation']})\nFirst message: {fmt_date(f['today'], rng)}\n\n"
                f"{thread}\n\n" + "\n".join(self._articles(f)))

    def questions(self):
        return {
            "recognition_ruling": choice("How should the federation respond to the recognition request?", {
                "recognise": "Recognise the TUE for the event as it stands.",
                "recognise_to_expiry_only": "Recognise it only up to its expiry and tell the athlete to renew for the remaining days.",
                "refer_to_committee": "Send the file to the federation's TUE committee for a full review.",
                "refuse": "Refuse recognition: there is no valid TUE to recognise.",
            }),
            "file_gap": choice("What is the main problem with the file, if any?", {
                "dose_exceeds_approval": "The athlete now uses more than the original TUE approved.",
                "expiry_before_event_ends": "The original TUE runs out before the event finishes.",
                "medical_evidence_missing": "The diagnosis evidence, tests or translation are missing.",
                "no_problem": "Nothing material: the file supports recognition.",
            }),
            "file_completeness": score("How complete is the medical file behind the original TUE?", [
                "Missing: little or nothing beyond a note.",
                "Partial: key documents are absent.",
                "Mostly complete: one secondary item missing.",
                "Complete: diagnosis, tests, specialist report and translation.",
            ]),
            "decision_time_pressure": score("How pressing is a decision, given the event dates and the committee's schedule?", [
                "Not pressing: months to go.",
                "Moderate: the event is weeks away, after the committee meets.",
                "Pressing: the event starts before the committee next meets.",
                "Critical: the event starts within a week.",
            ]),
            "covers_event_dates": noul(
                "The original TUE stays valid through the last day of the event.",
                yes="Its expiry date falls on or after the event's final day.",
                no="It expires before the event ends."),
        }

    def intended(self, f):
        comp, cov = self._completeness(f), self._covers(f)
        if f["granting_body"] == "team_doctor":
            ruling, gap = "refuse", "medical_evidence_missing"
        else:
            if comp <= 1:
                gap = "medical_evidence_missing"
            elif f["dose_change"] == "higher":
                gap = "dose_exceeds_approval"
            elif not cov:
                gap = "expiry_before_event_ends"
            else:
                gap = "no_problem"
            if comp <= 1 or f["dose_change"] == "higher":
                ruling = "refer_to_committee"
            elif not cov:
                ruling = "recognise_to_expiry_only"
            elif comp == 2:
                ruling = "refer_to_committee" if not f["docs"]["translation"] else "recognise"
            else:
                ruling = "recognise"
        return {"recognition_ruling": ruling, "file_gap": gap, "file_completeness": comp,
                "decision_time_pressure": self._urgency(f), "covers_event_dates": "yes" if cov else "no"}


def _add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12)
    return d.replace(year=d.year + y, month=mo + 1, day=min(d.day, 28))


# =========================================================================== schema 4
# (athlete's explanation, how negligent it shows the athlete to be 0-3, does an independent record back it?)
WB_EXPLANATIONS = {
    "missed_test": [("did not update the app after the training camp moved to another town", 3, False),
                    ("updated the one-hour slot 20 minutes after it had begun; the app log shows the time", 2, True),
                    ("was stuck on a delayed flight; the airline's delay notice is attached", 1, True),
                    ("was taken to hospital with a broken wrist; the emergency-department record is attached", 0, True),
                    ("says the doorbell was broken and was inside the whole hour", 2, False),
                    ("had a family emergency and left in a hurry, with no paperwork to show", 1, False),
                    ("overslept at a hotel after a late match", 3, False),
                    ("says a coach changed the schedule without telling the athlete", 2, False)],
    "filing_failure": [("forgot the quarterly deadline", 3, False),
                       ("the agency's app was down on the deadline day; the agency's outage notice confirms it", 0, True),
                       ("submitted on time but left the overnight address blank by mistake", 2, True),
                       ("was in hospital for the last week of the quarter; the discharge letter is attached", 0, True),
                       ("relied on an agent who did not file", 3, False),
                       ("believed the coach had filed for the whole squad", 2, False)],
}


class WhereaboutsFailureReview(Schema):
    name = "whereabouts_failure_review"
    description = "A reported missed test or filing failure is reviewed: confirm it or not, and whether it completes a charge under the code."
    surfaces = {"json": 0.5, "attempt_report": 0.25, "athlete_reply": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        c = _code(rng)
        sport, _ = pick(rng, SPORTS)
        window = pick(rng, [12, 12, 18])
        ftype = pick_w(rng, {"missed_test": 3, "filing_failure": 2})
        expl, negl, docs = pick(rng, WB_EXPLANATIONS[ftype])
        prior_count = pick_w(rng, {0: 1, 1: 2, 2: 5})
        prior_months = sorted(rng.sample([1, 3, 4, 6, 8, 10, 13, 17], prior_count))
        waited = pick_w(rng, {60: 4, 55: 3, 40: 2})
        call_rule = chance(rng, 0.5)
        called = chance(rng, 0.65)
        address_ok = chance(rng, 0.75)
        reminder = chance(rng, 0.7)
        if reg == "clear":
            if chance(rng, 0.5):
                expl, negl, docs = pick(rng, [e for e in WB_EXPLANATIONS[ftype] if e[1] == 3])
                waited, called, address_ok, reminder, prior_count = 60, True, True, True, 2
                prior_months = sorted(rng.sample([2, 4, 7], 2))
            else:
                expl, negl, docs = pick(rng, [e for e in WB_EXPLANATIONS[ftype] if e[1] == 0])
        elif reg == "borderline":
            # the first earlier failure sits right at the edge of the rolling window
            prior_count = 2
            prior_months = sorted([pick(rng, [3, 5, 8]), window + pick(rng, [-2, -1, 0])])
            if chance(rng, 0.7):
                expl, negl, docs = pick(rng, [e for e in WB_EXPLANATIONS[ftype] if e[1] >= 2])
                waited, address_ok = 60, True
            elif ftype == "missed_test":
                waited = pick(rng, [55, 60, 60])
        else:
            twist = pick(rng, ["careless_but_procedure_flawed", "sympathetic_without_records"])
            if twist == "careless_but_procedure_flawed":
                expl, negl, docs = pick(rng, [e for e in WB_EXPLANATIONS[ftype] if e[1] >= 2])
                waited, called, call_rule, reminder = pick(rng, [40, 55]), False, True, False
            else:
                expl, negl, docs = pick(rng, [e for e in WB_EXPLANATIONS[ftype] if not e[2]] or WB_EXPLANATIONS[ftype])
                negl = min(negl, 1) if chance(rng, 0.6) else negl
                if negl > 1:
                    expl, negl, docs = "had a family emergency and left in a hurry, with no paperwork to show", 1, False
        return {"regime": reg, "code": c, "sport": sport, "window_months": window, "failure_type": ftype,
                "explanation": expl, "negligence": negl, "documented": docs, "prior_months_before": prior_months,
                "dco_waited_min": waited, "call_rule": call_rule, "dco_called": called, "address_matched": address_ok,
                "reminder_sent": reminder, "athlete": person(rng), "dco": person(rng), "agency": pick(rng, AGENCIES),
                "case_id": ident(rng, "WF", 5, "-"), "attempt_date": rand_date(rng),
                "slot": pick(rng, ["06:00-07:00", "06:30-07:30", "07:00-08:00", "20:00-21:00", "21:00-22:00"]),
                "location": pick(rng, ["home address", "training centre", "team hotel", "university residence"])}

    def _procedure(self, f):
        if f["failure_type"] == "filing_failure":
            return 3 if f["reminder_sent"] else 1
        p = 3
        if f["dco_waited_min"] < 60:
            p -= 2 if f["dco_waited_min"] < 50 else 1
        if f["call_rule"] and not f["dco_called"]:
            p -= 1
        if not f["address_matched"]:
            p -= 1
        return _clamp(p)

    def _in_window(self, f):
        return sum(1 for m in f["prior_months_before"] if m < f["window_months"])

    def _confirmed(self, f):
        return self._procedure(f) >= 2 and f["negligence"] >= 2

    def _articles(self, f):
        c = f["code"]
        call = ("At the end of the hour the officer must phone the athlete." if f["call_rule"]
                else "The officer need not phone the athlete.")
        return [f"{_art(c, 11)} Three whereabouts failures (missed tests or filing failures, in any combination) within "
                f"any {f['window_months']}-month period are a violation.",
                f"{_art(c, 12)} A missed test is recorded only if the officer attempted the test at the filed location for "
                f"the full 60-minute slot. {call} A filing failure is recorded only after the athlete was sent a reminder "
                "and still did not file.",
                f"{_art(c, 13)} No failure is recorded if the athlete shows it was not caused by the athlete's negligence."]

    def _history(self, f, rng, style=None):
        style = style or pick(rng, ["iso", "long", "us", "eu"])
        return [fmt_date(f["attempt_date"] - timedelta(days=round(m * 30.4)), style=style) +
                f" ({pick(rng, ['missed test', 'filing failure'])})" for m in f["prior_months_before"]]

    def _attempt(self, f):
        if f["failure_type"] == "filing_failure":
            return (f"quarterly whereabouts filing not completed by the deadline; "
                    f"{'a reminder was sent' if f['reminder_sent'] else 'no reminder was sent'}")
        return (f"officer {f['dco']} attended the {f['location']} for the {f['slot']} slot and waited {f['dco_waited_min']} "
                f"minutes; {'phoned the athlete at the end' if f['dco_called'] else 'did not phone the athlete'}; "
                f"{'the location matched the filing' if f['address_matched'] else 'the address visited differs from the one filed'}")

    def render(self, f, rng, surface):
        hist = self._history(f, rng, "iso" if surface == "json" else None)
        if surface == "json":
            return dumps(prune({
                "case": {"id": f["case_id"], "agency": f["agency"], "athlete": f["athlete"], "sport": f["sport"]},
                "new_failure": {"type": f["failure_type"].replace("_", " "), "date": fmt_date(f["attempt_date"], style="iso"),
                                "record": self._attempt(f)},
                "earlier_failures": hist or ["none"],
                "athlete_explanation": f"The athlete {f['explanation']}.",
                "code": {"name": f["code"]["code"], "articles": self._articles(f)},
            }))
        if surface == "attempt_report":
            head = kv_lines([("Case", f["case_id"]), ("Athlete", f"{f['athlete']} ({f['sport']})"),
                             ("Date", fmt_date(f["attempt_date"], rng)), ("Type", f["failure_type"].replace("_", " "))], rng)
            return (f"WHEREABOUTS FAILURE REPORT - {f['agency']}\n{head}\n\nRecord: {self._attempt(f)}.\n\n"
                    f"Earlier recorded failures:\n" + (bullets(hist, rng) if hist else "- none") +
                    f"\n\nAthlete's explanation: the athlete {f['explanation']}.\n\n" + "\n".join(self._articles(f)))
        reply = noisy(join_sentences([
            f"I am writing about the {f['failure_type'].replace('_', ' ')} recorded on {fmt_date(f['attempt_date'], rng)}.",
            f"In short, I {f['explanation'].replace('the athlete', 'I')}.",
            pick(rng, ["I have always cooperated with testing.", "Please take my record into account.",
                       "I accept I should have been more careful.", ""])]), rng)
        return (f"Subject: {f['case_id']} - response to whereabouts failure\nFrom: {f['athlete']}\n\n{reply}\n\n"
                f"[agency file]\nRecord: {self._attempt(f)}.\nEarlier failures: {'; '.join(hist) or 'none'}.\n\n"
                + "\n".join(self._articles(f)))

    def questions(self):
        return {
            "failure_ruling": choice("How should the agency rule on the newly reported failure?", {
                "confirm_failure": "Confirm it and record it against the athlete.",
                "dismiss_no_negligence": "Do not record it: the athlete has shown it was not the athlete's fault.",
                "dismiss_procedural_defect": "Do not record it: the officer or agency did not follow the required procedure.",
                "request_athlete_evidence": "Ask the athlete for records supporting the explanation before ruling.",
            }),
            "procedure_followed": score("How fully were the code's procedural requirements met for this failure?", [
                "Barely: key steps were missed.",
                "Partly: a required step was skipped.",
                "Largely: a small shortfall that hardly matters.",
                "Fully: every required step was followed.",
            ]),
            "athlete_fault": score("How far was the failure the athlete's own fault?", [
                "Not at all: an emergency or a system outage outside the athlete's control.",
                "Slightly: bad luck the athlete could hardly avoid.",
                "Largely: a lapse the athlete should have prevented.",
                "Entirely: plain carelessness.",
            ]),
            "completes_violation": noul(
                "If this failure is recorded, the athlete will have three failures within the code's period.",
                yes="The new failure stands and at least two earlier failures fall inside the rolling period.",
                no="The new failure should not stand, or fewer than two earlier failures fall inside the period."),
            "explanation_backed_by_records": noul("The athlete's explanation is backed by an independent record."),
        }

    def intended(self, f):
        proc = self._procedure(f)
        if proc <= 1:
            ruling = "dismiss_procedural_defect"
        elif f["negligence"] <= 1 and f["documented"]:
            ruling = "dismiss_no_negligence"
        elif f["negligence"] <= 1:
            ruling = "request_athlete_evidence"
        else:
            ruling = "confirm_failure"
        complete = self._confirmed(f) and self._in_window(f) >= 2
        return {"failure_ruling": ruling, "procedure_followed": proc, "athlete_fault": f["negligence"],
                "completes_violation": "yes" if complete else "no",
                "explanation_backed_by_records": "yes" if f["documented"] else "no"}


# =========================================================================== schema 5
VIOLATIONS = ["a positive test for a stimulant", "refusing a sample collection", "a positive test for an anabolic agent",
              "three whereabouts failures", "possession of a prohibited substance", "tampering with a sample",
              "a positive test for a diuretic", "use of a prohibited method"]


class ReinstatementEligibility(Schema):
    name = "reinstatement_eligibility"
    description = "An athlete nearing the end of a period of ineligibility asks to return to competition on a chosen date."
    surfaces = {"json": 0.5, "reinstatement_request": 0.25, "eligibility_check": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        c = _code(rng)
        sport, _ = pick(rng, SPORTS)
        months = pick(rng, [12, 18, 24, 36, 48])
        prov = pick(rng, [0, 2, 3, 5, 8])
        respected = chance(rng, 0.8)
        decision = rand_date(rng)
        credit = prov if respected else 0
        end = _add_months(decision, months - credit)
        delta = pick(rng, [-250, -90, -20, 0, 10, 45])
        req_testing = pick(rng, [4, 6])
        avail_months = pick(rng, [0, 1, 2, 3, 4, 5, 6, 8])
        costs = pick_w(rng, {0: 5, 350: 1, 1200: 2, 4000: 1})
        plan = costs > 0 and chance(rng, 0.3)
        if reg == "clear":
            if chance(rng, 0.5):
                delta, avail_months, costs, plan = pick(rng, [0, 14, 30]), req_testing + 2, 0, False
            else:
                delta, avail_months = -pick(rng, [200, 300]), pick(rng, [0, 1])
        elif reg == "borderline":
            if chance(rng, 0.5):
                delta = pick(rng, [-12, -5, 0, 3])
            else:
                avail_months = req_testing + pick(rng, [-1, 0])
        else:
            twist = pick(rng, ["provisional_breach", "all_served_but_costs"])
            if twist == "provisional_breach":
                prov, respected = pick(rng, [3, 5, 8]), False
                credit = 0
                end = _add_months(decision, months)
                delta = 0
                requested = _add_months(decision, months - prov)
                avail_months = req_testing + 1
                return self._pack(reg, c, sport, months, prov, respected, decision, end, requested, req_testing,
                                  avail_months, costs, plan, rng)
            delta, avail_months, costs, plan = pick(rng, [0, 20]), req_testing + 1, pick(rng, [1200, 4000]), False
        requested = end + timedelta(days=delta)
        return self._pack(reg, c, sport, months, prov, respected, decision, end, requested, req_testing, avail_months,
                          costs, plan, rng)

    def _pack(self, reg, c, sport, months, prov, respected, decision, end, requested, req_testing, avail_months,
              costs, plan, rng):
        pool_entry = _add_months(requested, -avail_months) if avail_months else None
        return {"regime": reg, "code": c, "sport": sport, "violation": pick(rng, VIOLATIONS), "sanction_months": months,
                "provisional_months": prov, "provisional_respected": respected, "decision_date": decision,
                "eligible_from": end, "requested_return": requested, "testing_months_required": req_testing,
                "testing_months_available": avail_months, "pool_entry": pool_entry, "unpaid_costs": costs,
                "payment_plan": plan, "athlete": person(rng), "agency": pick(rng, AGENCIES),
                "federation": _fed(rng, sport), "case_id": ident(rng, "RI", 5, "-"),
                "breach_detail": pick(rng, ["entered a club time trial", "played two league matches",
                                            "trained with the national squad", "competed in an open meet"])}

    def _rem_days(self, f):
        return max(0, (f["eligible_from"] - f["requested_return"]).days)

    def _short_days(self, f):
        need = f["testing_months_required"] * 30
        have = (f["requested_return"] - f["pool_entry"]).days if f["pool_entry"] else 0
        return max(0, need - have)

    def _costs_block(self, f):
        return f["unpaid_costs"] > 0 and not f["payment_plan"]

    def _articles(self, f):
        c = f["code"]
        return [f"{_art(c, 15)} A period of ineligibility starts on the date of the final decision. Time served under a "
                "provisional suspension is credited, but only if the athlete respected it.",
                f"{_art(c, 16)} Before returning, the athlete must have been in the testing pool and available for testing "
                f"for at least the final {_mo(f['testing_months_required'])} of the period.",
                f"{_art(c, 17)} Costs and fines ordered must be paid, or a payment plan agreed, before reinstatement."]

    def render(self, f, rng, surface):
        prov = (f"provisional suspension of {_mo(f['provisional_months'])} before the decision"
                + ("" if f["provisional_respected"] else f"; during it the athlete {f['breach_detail']}")
                if f["provisional_months"] else "no provisional suspension")
        pool = (f"in the testing pool since {fmt_date(f['pool_entry'], style='iso')}" if f["pool_entry"]
                else "not yet in the testing pool")
        costs = ("none outstanding" if not f["unpaid_costs"] else
                 f"{f['unpaid_costs']:,} euros owed" + ("; payment plan agreed" if f["payment_plan"] else "; no plan agreed"))
        if surface == "json":
            return dumps(prune({
                "sanction": {"case": f["case_id"], "violation": f["violation"], "decision_date": fmt_date(f["decision_date"], style="iso"),
                             "period_months": f["sanction_months"], "before_decision": prov},
                "request": {"athlete": f["athlete"], "sport": f["sport"], "federation": f["federation"],
                            "requested_return": fmt_date(f["requested_return"], style="iso")},
                "status": {"testing_pool": pool, "costs": costs},
                "code": {"name": f["code"]["code"], "articles": self._articles(f)},
            }))
        if surface == "reinstatement_request":
            body = noisy(join_sentences([
                f"I am writing to ask to compete again from {fmt_date(f['requested_return'], rng)}.",
                f"My {_mo(f['sanction_months'])} period for {f['violation']} was imposed on {fmt_date(f['decision_date'], rng)}.",
                (f"I had already served {_mo(f['provisional_months'])} of provisional suspension, which I believe should count."
                 if f["provisional_months"] else ""),
                ("I have been " + pool + "." if f["pool_entry"] else "I have not yet rejoined the testing pool."),
                ("" if not f["unpaid_costs"] else ("I am paying the costs order by instalments." if f["payment_plan"]
                                                   else "I have not yet been able to pay the costs order."))]), rng)
            note = kv_lines([("Agency note", "provisional suspension " + ("respected" if f["provisional_respected"]
                                                                          else f"breached: athlete {f['breach_detail']}")
                              if f["provisional_months"] else "no provisional period")], rng)
            return (f"To: {f['federation']} / {f['agency']}\nFrom: {f['athlete']}\nRe: {f['case_id']} reinstatement\n\n"
                    f"{body}\n\n{note}\n\n" + "\n".join(self._articles(f)))
        rows = [["Case", f["case_id"]], ["Athlete", f"{f['athlete']} ({f['sport']})"], ["Violation", f["violation"]],
                ["Decision date", fmt_date(f["decision_date"], rng)], ["Period", _mo(f["sanction_months"])],
                ["Before decision", prov], ["Requested return", fmt_date(f["requested_return"], rng)],
                ["Testing pool", pool], ["Costs", costs]]
        return "ELIGIBILITY CHECK\n" + table(["Item", "Status"], rows, rng) + "\n\n" + "\n".join(self._articles(f))

    def questions(self):
        return {
            "reinstatement_ruling": choice("How should the agency answer the athlete's request to return?", {
                "reinstate_as_requested": "Allow the athlete to compete from the requested date.",
                "require_testing_period": "Refuse for now: the athlete must first complete the required time in the testing pool.",
                "set_later_return_date": "Refuse the date: the period of ineligibility has not run out by then.",
                "withhold_until_costs_settled": "Hold reinstatement until the costs are paid or a payment plan is agreed.",
            }),
            "binding_condition": choice("Which requirement, if any, stands between the athlete and the requested date?", {
                "ineligibility_not_served": "The period of ineligibility, as properly credited, has not ended.",
                "nothing_outstanding": "Nothing: every requirement is met.",
                "testing_time_short": "Too little time in the testing pool before the date.",
                "unpaid_costs": "Costs or fines are unpaid with no plan agreed.",
            }),
            "period_still_to_run": score("How much of the period of ineligibility would remain on the requested date?", [
                "None: it has ended by then.",
                "Less than a month.",
                "Between one and six months.",
                "More than six months.",
            ]),
            "testing_availability": score("How well has the athlete met the testing-pool requirement by the requested date?", [
                "Not at all: not in the pool.",
                "Partly: well short of the required time.",
                "Nearly: short by about a month or less.",
                "Fully: the required time is met.",
            ]),
            "provisional_credit_due": noul(
                "Time the athlete spent provisionally suspended counts toward the period.",
                yes="The athlete served a provisional suspension and respected it.",
                no="There was no provisional suspension, or the athlete broke it by taking part in sport."),
        }

    def intended(self, f):
        rem, short, costs = self._rem_days(f), self._short_days(f), self._costs_block(f)
        rem_lvl = 0 if rem == 0 else 1 if rem <= 31 else 2 if rem <= 183 else 3
        if short == 0:
            test_lvl = 3
        elif short <= 31:
            test_lvl = 2
        else:
            test_lvl = 1 if f["pool_entry"] else 0
        if rem > 0 and rem >= short:
            bind, ruling = "ineligibility_not_served", "set_later_return_date"
        elif short > 0:
            bind, ruling = "testing_time_short", "require_testing_period"
        elif costs:
            bind, ruling = "unpaid_costs", "withhold_until_costs_settled"
        else:
            bind, ruling = "nothing_outstanding", "reinstate_as_requested"
        credit = f["provisional_months"] > 0 and f["provisional_respected"]
        return {"reinstatement_ruling": ruling, "binding_condition": bind, "period_still_to_run": rem_lvl,
                "testing_availability": test_lvl, "provisional_credit_due": "yes" if credit else "no"}


# =========================================================================== schema 6
# (activity, organiser type, kind) -- organiser types bound by the code: member_club, federation, member_league
ACTIVITIES = [
    ("train with the senior squad at {club}", "member_club", "team_training"),
    ("use the gym at the national training centre", "federation", "team_training"),
    ("join {club}'s pre-season training camp", "member_club", "team_training"),
    ("coach the under-14 group at {club}", "member_club", "role"),
    ("referee youth matches in the regional league", "member_league", "role"),
    ("take an office job in the federation's events department", "federation", "role"),
    ("speak at the federation's anti-doping education sessions", "federation", "education"),
    ("train alone at a commercial gym", "commercial_gym", "solo_training"),
    ("work as a paid television commentator", "broadcaster", "non_sport_role"),
    ("enter a hospital's charity fun-run", "non_signatory", "compete"),
    ("race in a commercial ultra-distance series", "non_signatory", "compete"),
    ("play in a corporate five-a-side league", "non_signatory", "compete"),
    ("enter a professional road race run by a commercial promoter", "non_signatory", "compete"),
    ("enter an open invitational run by an independent promoter", "unknown", "compete"),
    ("compete in an exhibition event organised by a sponsor", "unknown", "compete"),
]
TOWNS = ["Riverside", "Northmark City", "Kestmoor", "Valdren", "Orsenne", "Halvard", "Ellmark", "Brisaland",
         "Old Harbour", "Westfield", "Larchmont", "Sallow Bay"]
CLUB_WORD = {"Athletics": "Athletic", "Aquatics": "Water Polo", "Ski": "Ski", "Skating": "Skating"}


def _club(rng, sport):
    word = CLUB_WORD.get(GOVERNING[sport], GOVERNING[sport])
    return f"{pick(rng, TOWNS)} {word} Club"


class IneligibilityActivityRequest(Schema):
    name = "ineligibility_activity_request"
    description = "An athlete serving a period of ineligibility asks whether a planned activity is allowed under the code."
    surfaces = {"json": 0.5, "athlete_query": 0.25, "integrity_officer_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        c = _code(rng)
        sport, _ = pick(rng, SPORTS)
        act, org_type, kind = pick(rng, ACTIVITIES, [{"team_training": 3, "compete": 2}.get(a[2], 1)
                                                      + (1 if a[1] == "unknown" else 0) for a in ACTIVITIES])
        total = pick(rng, [12, 18, 24, 48])
        window = pick(rng, [2, 3])
        left = pick(rng, [1, 3, 4, 6, 9, 14, 30])
        left = min(left, total)
        prize_line = pick(rng, [500, 1000, 2500])
        prize = pick(rng, [0, 0, 300, 800, 2000, 6000]) if kind == "compete" else 0
        national = kind == "compete" and chance(rng, 0.25)
        if reg == "clear":
            if chance(rng, 0.5):
                act, org_type, kind = pick(rng, [a for a in ACTIVITIES if a[1] in ("member_club", "federation", "member_league") and a[2] == "role"])
            else:
                act, org_type, kind = pick(rng, [a for a in ACTIVITIES if a[2] in ("solo_training", "non_sport_role", "education")])
        elif reg == "borderline":
            act, org_type, kind = pick(rng, [a for a in ACTIVITIES if a[2] in ("team_training", "compete")])
            left = window + pick(rng, [-1, 0, 1, 2])
            if kind == "compete":
                prize = pick(rng, [prize_line - 100, prize_line, prize_line + 200])
        else:
            twist = pick(rng, ["harmless_looking_role", "big_prize_outside_federation"])
            if twist == "harmless_looking_role":
                act, org_type, kind = pick(rng, [a for a in ACTIVITIES if a[2] == "role"])
                left = pick(rng, [1, 2])
            else:
                act, org_type, kind = pick(rng, [a for a in ACTIVITIES if a[1] == "non_signatory"
                                                 and ("commercial" in a[0])])
                prize, national = pick(rng, [2000, 6000]), chance(rng, 0.4)
        if kind != "compete":
            prize, national = 0, False
        return {"regime": reg, "code": c, "sport": sport, "activity": act.format(club=_club(rng, sport)),
                "organiser_type": org_type, "kind": kind, "sanction_months": total, "months_left": max(1, left),
                "window_months": window, "prize_line": prize_line, "prize_money": prize, "national_level": national,
                "athlete": person(rng), "officer": person(rng), "agency": pick(rng, AGENCIES),
                "case_id": ident(rng, "IA", 5, "-"), "asked": rand_date(rng)}

    def _bound(self, f):
        return f["organiser_type"] in ("member_club", "federation", "member_league")

    def _ruling(self, f):
        if f["organiser_type"] == "unknown":
            return "seek_organiser_details"
        if f["kind"] == "education":
            return "permit_now"
        if self._bound(f) and f["kind"] == "team_training":
            return "permit_now" if f["months_left"] <= f["window_months"] else "permit_from_training_window"
        if self._bound(f):
            return "refuse"
        if f["kind"] == "compete" and (f["prize_money"] > f["prize_line"] or f["national_level"]):
            return "refuse"
        return "permit_now"

    def _articles(self, f):
        c = f["code"]
        return [f"{_art(c, 20)} During ineligibility an athlete may not take part in any capacity in a competition or "
                "activity authorised or organised by a signatory, a member federation, or a member club or league, other "
                "than authorised anti-doping education or rehabilitation.",
                f"{_art(c, 21)} In the last {_mo(f['window_months'])} of the period, the athlete may train with a team or "
                "use the facilities of a member club or federation.",
                f"{_art(c, 22)} The athlete may not compete in events run by non-signatories that award national ranking "
                f"or prize money above {f['prize_line']:,} euros."]

    def _org_text(self, f):
        return {"member_club": "a member club of the federation", "federation": "the federation itself",
                "member_league": "a league affiliated to the federation", "commercial_gym": "a private commercial gym",
                "broadcaster": "a commercial broadcaster", "non_signatory": "an organiser outside the federation and the code",
                "unknown": "an organiser whose affiliation is not stated"}[f["organiser_type"]]

    def render(self, f, rng, surface):
        comp = ""
        if f["kind"] == "compete":
            comp = (f"Prize money: {f['prize_money']:,} euros" if f["prize_money"] else "No prize money") + \
                   ("; results count toward national ranking." if f["national_level"] else "; no ranking points.")
        if surface == "json":
            return dumps(prune({
                "athlete": {"name": f["athlete"], "sport": f["sport"], "ineligibility_months": f["sanction_months"],
                            "months_remaining": f["months_left"]},
                "request": {"id": f["case_id"], "wants_to": f["activity"], "organised_by": self._org_text(f),
                            "competition_details": comp or None},
                "code": {"name": f["code"]["code"], "articles": self._articles(f)},
            }))
        if surface == "athlete_query":
            body = noisy(join_sentences([
                f"I have {_mo(f['months_left'])} left of my {_mo(f['sanction_months'])} ban.",
                f"I would like to {f['activity']}, which is run by {self._org_text(f)}.", comp,
                pick(rng, ["Can you confirm this is OK?", "I don't want to risk anything, so please advise.",
                           "My coach says it should be fine."])]), rng)
            return (f"Subject: Query about activities during my ineligibility ({f['case_id']})\nFrom: {f['athlete']}\n"
                    f"To: {f['agency']}\n\n{body}\n\n" + "\n".join(self._articles(f)))
        rows = [("Query", f["case_id"]), ("Athlete", f"{f['athlete']} ({f['sport']})"),
                ("Period", f"{_mo(f['sanction_months'])}, {_mo(f['months_left'])} remaining"),
                ("Proposed activity", f["activity"]), ("Organiser", self._org_text(f)), ("Competition", comp or "n/a")]
        return (f"INTEGRITY OFFICER NOTE ({f['officer']})\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._articles(f)))

    def questions(self):
        return {
            "activity_ruling": choice("What should the agency tell the athlete about the proposed activity?", {
                "permit_from_training_window": "It is allowed only once the athlete enters the code's final return-to-training window.",
                "permit_now": "It is allowed now.",
                "refuse": "It is not allowed at any point during the period.",
                "seek_organiser_details": "The answer depends on who runs it; ask for the organiser's affiliation first.",
            }),
            "window_distance": score("How far is the athlete from the code's return-to-training window?", [
                "Already inside it.",
                "Within two months of it.",
                "Two to six months away.",
                "More than six months away.",
            ]),
            "breach_if_done_now": score("If the athlete did this now, how clearly would it breach the code?", [
                "Not at all: the activity lies outside the code's reach.",
                "Possibly: it depends on details not yet known, or sits near a line.",
                "Probably: it falls within a restriction, subject to the details.",
                "Certainly: plainly prohibited participation.",
            ]),
            "organiser_bound_by_code": noul(
                "The organiser of the activity is bound by the code.",
                yes="It is the federation, a member club or an affiliated league or event.",
                no="It is a commercial, charitable or other body outside the federation, or its status is unknown."),
            "involves_competing": noul("The proposed activity involves competing in an event."),
        }

    def intended(self, f):
        ruling = self._ruling(f)
        gap = f["months_left"] - f["window_months"]
        dist = 0 if gap <= 0 else 1 if gap <= 2 else 2 if gap <= 6 else 3
        if ruling == "permit_now":
            breach = 1 if f["kind"] == "compete" else 0
        elif ruling == "seek_organiser_details":
            breach = 1
        elif ruling == "permit_from_training_window":
            breach = 3
        else:
            breach = 3 if self._bound(f) else 2
        return {"activity_ruling": ruling, "window_distance": dist, "breach_if_done_now": breach,
                "organiser_bound_by_code": "yes" if self._bound(f) else "no",
                "involves_competing": "yes" if f["kind"] == "compete" else "no"}


WORKFLOW = Workflow(
    name="sports_tue_and_eligibility_review",
    title="Sport TUE and anti-doping eligibility review",
    description="How an anti-doping body should rule on therapeutic use exemptions, whereabouts failures and athletes' "
                "eligibility under a written anti-doping code.",
    domain="arts, entertainment and recreation / sport integrity",
    schemas=[ProspectiveTueApplication(), RetroactiveTueAfterFinding(), TueRecognitionRequest(),
             WhereaboutsFailureReview(), ReinstatementEligibility(), IneligibilityActivityRequest()],
    notes="Agencies, federations, clubs and athletes are fictional; medication status comes only from the fictional code "
          "quoted in each state. No sex-category or nationality eligibility cases; no protected attribute is a factor.",
)
