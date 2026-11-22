"""Financial aid appeals: special-circumstance appeals and professional-judgement requests at fictional colleges.

Five schemas: a fall in family income since the tax year on the aid application, a request to be assessed
independently of parents, an appeal to keep aid after failing academic-progress standards, a request to raise
the cost-of-attendance budget for an unusual expense, and a request to exclude one-off income from the
calculation. Each samples a `regime` first ('clear' / 'borderline' / 'conflicting') and lets it steer the
factors: an income drop just around the office's guideline, a lost job replaced last month at similar pay,
letters that are generic or missing, progress figures a hair under the standard, and one-off income that is
still sitting in savings.

Colleges, staff, students and families are fictional; amounts are invented. Decisions rest on income,
documentation, circumstances and academic records only. No applicant's race, religion, national origin, sex,
disability, age or marital status is stated or used, and mitigating events are limited to things like job
loss, bereavement, injury, housing loss or work hours.
"""
from __future__ import annotations

from .base import (
    FIRST,
    LAST,
    Schema,
    Workflow,
    chance,
    choice,
    dumps,
    fmt_date,
    join_sentences,
    kv_lines,
    lognormal_amount,
    maybe,
    money,
    near,
    noisy,
    noul,
    org,
    pick,
    pick_w,
    prune,
    rand_date,
    regime,
    score,
    table,
)

AWARD_YEAR = "2026-27"
TAX_YEAR = 2024


def _names(rng, n):
    """n fictional people with distinct first AND last names."""
    return [f"{a} {b}" for a, b in zip(rng.sample(FIRST, n), rng.sample(LAST, n))]


def _sid(rng):
    """A college student number (deliberately NOT shaped like a national identity number)."""
    return f"{pick(rng, ['A', 'B', 'T'])}{rng.randrange(10000000, 99999999)}"


def _r100(x):
    return int(round(x, -2))


def _college(rng):
    return org(rng, pick(rng, ["school", "school"]))


# =========================================================================== schema 1
class IncomeLossAppeal(Schema):
    name = "income_loss_appeal"
    description = "A request to recalculate aid because income has fallen since the tax year used on the aid application."
    surfaces = {"json": 0.5, "appeal_form": 0.25, "counselor_notes": 0.25}

    CAUSES = {"job_loss": "was laid off", "reduced_hours": "had working hours cut",
              "business_downturn": "saw the family business lose most of its main contract",
              "contract_ended": "had a fixed-term contract end without renewal",
              "death_of_earner": "lost the household's second earner, who died in the spring"}
    DOCS_FOR = {"job_loss": ["layoff letter", "unemployment benefit award letter"],
                "reduced_hours": ["employer letter confirming reduced hours", "three recent pay stubs"],
                "business_downturn": ["year-to-date profit and loss statement", "business bank statements"],
                "contract_ended": ["contract end notice", "unemployment benefit award letter"],
                "death_of_earner": ["death certificate", "survivor benefit letter"]}
    REEMP = {"not_yet": "not yet re-employed", "interviewing": "interviewing; no offer yet",
             "new_job_lower_pay": "started a new job last month at lower pay",
             "new_job_similar_pay": "started a new job last month at similar pay"}

    def sample_factors(self, rng):
        reg = regime(rng)
        filer = pick_w(rng, {"parents": 3, "independent_student": 1})
        cause = pick(rng, list(self.CAUSES))
        agi = _r100(lognormal_amount(rng, 64000, 0.45))
        guideline = pick(rng, [0.10, 0.15, 0.15, 0.20])
        drop = near(rng, guideline, 0.09, lo=0.01, hi=0.8)
        docs_status = pick_w(rng, {"complete": 3, "partial": 4, "statement_only": 2})
        reemp = pick_w(rng, {"not_yet": 3, "interviewing": 2, "new_job_lower_pay": 2, "new_job_similar_pay": 2})
        severance = None
        if cause in ("job_loss", "contract_ended") and chance(rng, 0.35):
            severance = _r100(agi * pick(rng, [0.08, 0.2, 0.4]))
        if reg == "clear":
            if chance(rng, 0.5):
                drop, docs_status, reemp, severance = round(rng.uniform(0.35, 0.6), 2), "complete", "not_yet", None
                cause = pick(rng, ["job_loss", "death_of_earner", "contract_ended"])
            else:
                drop, docs_status, reemp = round(rng.uniform(0.02, 0.07), 2), "statement_only", "new_job_similar_pay"
        elif reg == "conflicting":
            twist = pick(rng, ["replaced_job", "big_severance", "documented_but_small", "self_prepared_pnl"])
            if twist == "replaced_job":
                cause, drop, reemp = "job_loss", round(rng.uniform(0.25, 0.45), 2), "new_job_similar_pay"
            elif twist == "big_severance":
                cause, drop, reemp = "job_loss", round(rng.uniform(0.3, 0.5), 2), "not_yet"
                severance = _r100(agi * rng.uniform(0.35, 0.6))
            elif twist == "documented_but_small":
                drop, docs_status = round(guideline - rng.uniform(0.03, 0.06), 2), "complete"
            else:
                cause, drop, docs_status = "business_downturn", round(rng.uniform(0.3, 0.55), 2), "partial"
        drop = max(0.01, drop)
        if cause in ("reduced_hours", "business_downturn", "death_of_earner"):
            reemp = "not_applicable"
        if cause not in ("job_loss", "contract_ended"):
            severance = None
        projected = _r100(agi * (1 - drop))
        docs = self.DOCS_FOR[cause]
        supplied = docs if docs_status == "complete" else (docs[:1] if docs_status == "partial" else [])
        sai = int(round(lognormal_amount(rng, 9000, 0.6), -1))
        est = max(-1500, int(round(sai * (1 - min(0.95, drop * pick(rng, [1.4, 1.8, 2.2]))), -1)))
        student, earner, counselor = _names(rng, 3)
        return {"regime": reg, "filer": filer, "cause": cause, "tax_year_income": agi, "projected_income": projected,
                "drop": drop, "office_guideline_drop": guideline, "docs_status": docs_status, "docs_supplied": supplied,
                "docs_expected": docs, "reemployment": reemp, "severance": severance,
                "unemployment_weekly": pick(rng, [None, 380, 450, 520]) if cause in ("job_loss", "contract_ended") else None,
                "current_sai": sai, "estimated_sai_after": est, "student": student, "earner": earner,
                "counselor": counselor, "student_id": _sid(rng), "college": _college(rng),
                "submitted": rand_date(rng), "tone": pick(rng, ["formal", "worried", "brief"])}

    def _who(self, f):
        return "My parent" if f["filer"] == "parents" else "I"

    def _statement(self, f, rng):
        first = f["filer"] == "independent_student"
        subj = "I" if first else pick(rng, ["My father", "My mother", "My parent"])
        s = [f"{subj} {self.CAUSES[f['cause']]} earlier this year" + ("." if f["cause"] != "death_of_earner" else "")
             if f["cause"] != "death_of_earner" else "Our household lost its second earner, who died in the spring."]
        s.append(f"We expect about {money(f['projected_income'], rng)} this year, compared with "
                 f"{money(f['tax_year_income'], rng)} on the {TAX_YEAR} tax return.")
        if f["severance"]:
            s.append(pick(rng, [f"There was a severance payment of {money(f['severance'], rng)}.",
                                "There was some severance, which is going on bills."]))
        if f["reemployment"] in ("new_job_lower_pay", "new_job_similar_pay"):
            s.append({"new_job_lower_pay": "A new job started last month but pays less.",
                      "new_job_similar_pay": "Thankfully a new job started last month."}[f["reemployment"]])
        elif f["reemployment"] == "interviewing":
            s.append("Interviews are going on but nothing is confirmed.")
        if f["tone"] == "worried":
            s.append("Without more help I'm not sure I can stay enrolled.")
        return noisy(join_sentences(s), rng, 0.15)

    def render(self, f, rng, surface):
        stmt = self._statement(f, rng)
        if surface == "json":
            return dumps(prune({
                "appeal": {"student_id": f["student_id"], "award_year": AWARD_YEAR, "submitted": f["submitted"].isoformat(),
                           "type": "special circumstances: change in income", "family_statement": stmt},
                "income": {"tax_year": TAX_YEAR, "tax_year_income": f["tax_year_income"], "projected_current_year": f["projected_income"],
                           "projected_fall_pct": int(round(f["drop"] * 100)),
                           "severance_received": f["severance"],
                           "severance_in_months_of_old_pay": round(12 * f["severance"] / f["tax_year_income"]) if f["severance"] else None,
                           "unemployment_weekly_benefit": f["unemployment_weekly"],
                           "employment_now": self.REEMP.get(f["reemployment"])},
                "documents": {"supplied": f["docs_supplied"] or ["signed statement only"],
                              "not_yet_supplied": [d for d in f["docs_expected"] if d not in f["docs_supplied"]] or None},
                "aid_calculation": {"current_sai": f["current_sai"], "estimated_sai_if_projection_used": f["estimated_sai_after"]},
                "office_guideline": f"income changes are usually reviewed when projected income is at least "
                                    f"{int(f['office_guideline_drop'] * 100)}% below the tax-year figure",
            }))
        if surface == "appeal_form":
            rows = [("Student ID", f["student_id"]), ("Award year", AWARD_YEAR), ("Submitted", fmt_date(f["submitted"], rng)),
                    ("Appeal type", "change in income"),
                    (f"{TAX_YEAR} income (tax return)", money(f["tax_year_income"], rng)),
                    ("Estimated income this year", f"{money(f['projected_income'], rng)} "
                                                   f"({int(round(f['drop'] * 100))}% below the tax year)"),
                    ("Office review guideline", f"a fall of at least {int(f['office_guideline_drop'] * 100)}%"),
                    ("Severance / lump sums", f"{money(f['severance'], rng)} (about {round(12 * f['severance'] / f['tax_year_income'])} "
                                              f"months of old pay)" if f["severance"] else "none reported"),
                    ("Current employment", self.REEMP.get(f["reemployment"])),
                    ("Documents attached", ", ".join(f["docs_supplied"]) or "none")]
            return (f"SPECIAL CIRCUMSTANCES APPEAL -- {f['college']}\n" + kv_lines(rows, rng)
                    + "\n\nExplanation of circumstances:\n" + stmt)
        s = [f"{f['student_id']}: income appeal for {AWARD_YEAR}.",
             f"{TAX_YEAR} income {money(f['tax_year_income'], rng)}; family projects {money(f['projected_income'], rng)} "
             f"({int(round(f['drop'] * 100))}% lower; our guideline is {int(f['office_guideline_drop'] * 100)}%).",
             ("Documents in: " + ", ".join(f["docs_supplied"]) + "." if f["docs_supplied"] else "Only a signed statement so far."),
             (f"Still missing: {', '.join(d for d in f['docs_expected'] if d not in f['docs_supplied'])}."
              if f["docs_status"] != "complete" else ""),
             (f"Severance {money(f['severance'], rng)} (about {round(12 * f['severance'] / f['tax_year_income'])} "
              f"months of the earner's old pay) not counted in the projection." if f["severance"] else ""),
             (f"Employment: {self.REEMP[f['reemployment']]}." if f["reemployment"] in self.REEMP else ""),
             f"SAI {f['current_sai']:,} now; about {f['estimated_sai_after']:,} if we use their projection."]
        return f"Counselor notes ({f['counselor']})\n" + noisy(join_sentences(s), rng, 0.1)

    def questions(self):
        return {
            "pj_decision": choice("How should the aid office decide this income appeal?", {
                "adjust_conservatively": "Recalculate, but on a more cautious income figure than the family projects.",
                "adjust_as_requested": "Recalculate aid using the income the family projects for this year.",
                "deny_adjustment": "Keep the original calculation; the change does not justify an adjustment.",
                "hold_for_documents": "Hold the appeal and ask for specific documents before deciding.",
            }),
            "income_basis": choice("Which income figure best reflects the family's ability to pay this year?", {
                "annualised_current_pay": "Current earnings annualised from recent pay: a new job has started or hours "
                                          "were cut, so recent pay is a firmer figure than the projection.",
                "projection_as_given": "The family's own projection: income has fallen past the office guideline, "
                                       "nothing has replaced it and no lump sum came in.",
                "projection_plus_lump_sums": "The projection plus the severance or other lump sum received this year, "
                                             "which the projection leaves out.",
                "tax_year_figure": "The original tax-year income: the projected fall is smaller than the office's "
                                   "review guideline.",
            }),
            "loss_materiality": score("How material is the change in income for the family's ability to pay?", [
                "Immaterial: the fall is below the office's review guideline, or a new job at similar pay has "
                "replaced the income.",
                "Modest: past the guideline, but less than a quarter of tax-year income.",
                "Significant: a quarter to just under half of tax-year income.",
                "Severe: 45% or more of tax-year income has gone.",
            ]),
            "documentation_sufficiency": score("How well is the change in circumstances documented?", [
                "Undocumented: only the family's own signed statement.",
                "Thin: one of the expected documents is in, another is still missing.",
                "Adequate: all expected documents are in, but a figure that matters, such as a severance payment or "
                "new pay, has no document behind it.",
                "Thorough: all expected documents are in and no figure is left unsupported.",
            ]),
            "loss_likely_to_persist": noul(
                "The fall in income is likely to last through most of the award year.",
                yes="No new job at similar pay has started and no severance covers three or more months of the old pay.",
                no="A new job at similar pay has started, or a severance payment covers three or more months of the "
                   "old pay."),
        }


# =========================================================================== schema 2
class DependencyOverride(Schema):
    name = "dependency_override"
    description = "A student asks to be assessed without parental information because of a breakdown in the family."
    surfaces = {"json": 0.5, "student_statement": 0.25, "review_checklist": 0.25}

    CIRC = {"estrangement": "has had no contact with either parent since leaving home",
            "abandonment": "was left to fend for themselves when their parents moved away without a forwarding address",
            "left_after_conflict": "left the parental home after repeated serious conflict",
            "parents_unwilling": "says their parents refuse to fill in the aid application or contribute to costs"}
    LIVING = {"own_rental": "renting a room and paying their own way", "with_relative": "living with an aunt who provides some support",
              "friends": "staying with friends, moving every few months", "with_parents": "still living at the parental address"}
    SOURCES = {"school_counselor": "a former high-school counselor", "social_worker": "a social worker who knew the family",
               "teacher": "a former teacher", "employer": "the student's employer", "relative": "an aunt of the student",
               "therapist": "a licensed counselor the student sees"}
    QUALITY = {"specific": "gives dates and describes what the writer saw first-hand",
               "generic": "is short and general, without dates or first-hand detail",
               "unsigned": "is typed, unsigned and has no contact details"}

    def sample_factors(self, rng):
        reg = regime(rng)
        circ = pick_w(rng, {"estrangement": 3, "abandonment": 1, "left_after_conflict": 3, "parents_unwilling": 2})
        years = near(rng, 1.5, 1.4, lo=0.2, digits=1)
        living = pick_w(rng, {"own_rental": 3, "with_relative": 2, "friends": 2, "with_parents": 1})
        n_letters = pick(rng, [0, 1, 1, 2])
        sources = rng.sample(list(self.SOURCES), n_letters)
        quality = [pick_w(rng, {"specific": 3, "generic": 3, "unsigned": 1}) for _ in range(n_letters)]
        claimed_on_tax = maybe(rng, chance(rng, 0.25), 0.3)
        parents_pay_something = chance(rng, 0.2)
        prior_override = pick_w(rng, {"none": 4, "granted_here_last_year": 1, "granted_elsewhere": 1})
        work_hours = pick(rng, [0, 12, 20, 30])
        if reg == "clear":
            if chance(rng, 0.5):
                circ, years, living, n_letters = "estrangement", round(rng.uniform(2, 4), 1), "own_rental", 2
                sources, quality = rng.sample(["school_counselor", "social_worker", "therapist"], 2), ["specific", "specific"]
                claimed_on_tax, parents_pay_something = False, False
            else:
                circ, living, n_letters, sources, quality = "parents_unwilling", "with_parents", 0, [], []
                parents_pay_something, claimed_on_tax, prior_override = True, True, "none"
        elif reg == "conflicting":
            twist = pick(rng, ["strong_story_tax_claim", "good_letters_short_time", "prior_override_no_letter", "unwilling_framed_as_estranged"])
            if twist == "strong_story_tax_claim":
                circ, claimed_on_tax, n_letters = "estrangement", True, 1
                sources, quality = [pick(rng, ["school_counselor", "social_worker"])], ["specific"]
            elif twist == "good_letters_short_time":
                years, n_letters = round(rng.uniform(0.2, 0.6), 1), 2
                sources, quality = rng.sample(["school_counselor", "social_worker", "teacher"], 2), ["specific", "generic"]
            elif twist == "prior_override_no_letter":
                prior_override, n_letters, sources, quality = "granted_here_last_year", 0, [], []
            else:
                circ, living, n_letters = "parents_unwilling", "own_rental", 1
                sources, quality = ["relative"], ["generic"]
        if circ == "parents_unwilling" and living == "friends":
            living = "own_rental"
        if circ != "parents_unwilling" and living == "with_parents":
            living = "with_relative"
        student, reviewer = _names(rng, 2)
        return {"regime": reg, "circumstance": circ, "years_separated": years, "living": living,
                "letter_sources": sources, "letter_quality": quality, "parents_claimed_on_tax_return": claimed_on_tax,
                "parents_pay_phone_or_insurance": parents_pay_something, "prior_override": prior_override,
                "work_hours_per_week": work_hours, "student": student, "reviewer": reviewer, "student_id": _sid(rng),
                "college": _college(rng), "received": rand_date(rng)}

    def _letters(self, f):
        return [f"letter from {self.SOURCES[s]}: {self.QUALITY[q]}" for s, q in zip(f["letter_sources"], f["letter_quality"])]

    def _tax(self, f):
        v = f["parents_claimed_on_tax_return"]
        return {None: "not known whether parents still claim the student as a dependant",
                True: "parents' most recent tax return still lists the student as a dependant",
                False: "parents' most recent tax return does not list the student"}[v]

    def _prior(self, f):
        return {"none": None, "granted_here_last_year": "override granted by this office last year",
                "granted_elsewhere": "student says another college granted an override last year (no letter supplied)"}[f["prior_override"]]

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "request": {"student_id": f["student_id"], "award_year": AWARD_YEAR, "received": f["received"].isoformat(),
                            "stated_circumstance": f"Student {self.CIRC[f['circumstance']]}.",
                            "time_since_last_lived_with_parents_years": None if f["living"] == "with_parents" else f["years_separated"]},
                "student_situation": {"living_arrangement": self.LIVING[f["living"]], "work_hours_per_week": f["work_hours_per_week"]},
                "third_party_documentation": self._letters(f) or ["none received"],
                "other_information": {"tax_return": self._tax(f),
                                      "parents_pay_phone_or_insurance": f["parents_pay_phone_or_insurance"],
                                      "previous_override": self._prior(f)},
            }))
        if surface == "student_statement":
            circ_first = {"estrangement": "I have had no contact with either of my parents since I left home",
                          "abandonment": "My parents moved away without telling me where, and I have been on my own since",
                          "left_after_conflict": "I left my parents' home after things became impossible there",
                          "parents_unwilling": "My parents will not fill in the aid form or help with any costs"}[f["circumstance"]]
            since = (f" I moved out about {f['years_separated']} years ago." if f["circumstance"] == "parents_unwilling"
                     else f", about {f['years_separated']} years ago.")
            s = [circ_first + (since if f["living"] != "with_parents" else "."),
                 {"own_rental": "I rent a room and pay for everything myself.", "with_relative": "I live with my aunt, who helps where she can.",
                  "friends": "I have been staying with friends and moving around.", "with_parents": "For now I am still living at my parents' address."}[f["living"]],
                 (f"I work about {f['work_hours_per_week']} hours a week." if f["work_hours_per_week"] else ""),
                 ("My parents still pay my phone bill." if f["parents_pay_phone_or_insurance"] else ""),
                 "I can't get their tax information and I'm asking to be treated as independent."]
            att = self._letters(f)
            return "\n".join([f"Dependency override request -- {f['college']} -- {AWARD_YEAR}", f"Student: {f['student']} ({f['student_id']})", "",
                              noisy(join_sentences(s), rng, 0.2), "",
                              "Attachments:", *(f"  - {a}" for a in att), *([] if att else ["  (none)"]), "",
                              f"Office note: {self._tax(f)}." + (f" {self._prior(f).capitalize()}." if self._prior(f) else "")])
        rows = [("Student", f["student_id"]), ("Received", fmt_date(f["received"], rng)),
                ("Circumstance claimed", self.CIRC[f["circumstance"]]),
                ("Years since living with parents", None if f["living"] == "with_parents" else f["years_separated"]),
                ("Living arrangement", self.LIVING[f["living"]]), ("Work hours / week", f["work_hours_per_week"]),
                ("Third-party letters", "; ".join(self._letters(f)) or "none"), ("Tax return check", self._tax(f)),
                ("Parents pay phone / insurance", "yes" if f["parents_pay_phone_or_insurance"] else "no"),
                ("Previous override", self._prior(f) or "none")]
        return f"DEPENDENCY OVERRIDE REVIEW -- reviewer {f['reviewer']}\n" + kv_lines(rows, rng)

    def questions(self):
        return {
            "override_decision": choice("How should the office respond to the dependency override request?", {
                "deny_override": "Deny the override; parental information is required for any aid beyond the minimum.",
                "grant_override": "Grant the override and assess the student as independent for this award year.",
                "grant_provisionally": "Grant it for now, on condition that the missing corroboration arrives within a set period.",
                "request_third_party_letter": "Ask for a letter from a professional who knows the circumstances before deciding.",
                "unsubsidized_loan_only": "Decline the override but let the student borrow the basic unsubsidised loan without parental data.",
            }),
            "breakdown_evidence": score("How clearly do the circumstances show a genuine breakdown with the parents, not just an unwillingness to pay?", [
                "Not at all: the parents simply refuse to pay or share information, or the student still lives at the "
                "parental address.",
                "Weakly: a claimed break, but the parents still pay a bill or claim the student on their taxes, or it "
                "began less than six months ago.",
                "Clearly: a break of six months or more with no sign of parental support.",
                "Unmistakably: a break of two years or more with no support, confirmed by two specific letters from "
                "people who saw it first-hand.",
            ]),
            "corroboration_strength": score("How well do independent people corroborate the student's account?", [
                "Not at all: no third-party letter.",
                "Weakly: a letter that is generic, unsigned or from someone close to the student.",
                "Reasonably: one specific, signed letter from a professional who knows the situation.",
                "Strongly: several specific letters from professionals who saw the situation first-hand.",
            ]),
            "meets_override_standard": noul(
                "The student's situation meets the unusual-circumstances standard for a dependency override.",
                yes="The file shows a real breakdown or abandonment, not merely parents who will not pay or share information.",
                no="The situation is parental refusal or ordinary independence, or the claimed breakdown is contradicted."),
            "signs_of_parental_support": noul(
                "The file shows signs that the parents still support the student financially.",
                yes="The parents still pay a bill or insurance for the student, still claim the student on their tax "
                    "return, or the student still lives with them.",
                no="None of these: no payments, no tax claim, and the student lives apart from the parents."),
        }


# =========================================================================== schema 3
class SapAppeal(Schema):
    name = "sap_appeal"
    description = "An appeal to keep aid after failing the Satisfactory Academic Progress standards."
    surfaces = {"json": 0.45, "appeal_packet": 0.3, "committee_note": 0.25}

    CIRC = {"bereavement": "a grandparent who raised the student died mid-semester",
            "injury": "a broken leg from a road accident meant six weeks without getting to campus",
            "work_hours": "a second job taken on to cover rent meant working nights all term",
            "housing_loss": "the student lost their housing and moved three times in one term",
            "none_specific": "the student says they 'struggled with the transition' and lost motivation"}
    DOCS = {"bereavement": "funeral program", "injury": "emergency-room discharge papers",
            "work_hours": "employer schedule printouts", "housing_loss": "eviction notice", "none_specific": None}
    PLAN = {"advisor_signed": "academic plan signed by an advisor", "student_drafted": "plan drafted by the student, not yet reviewed",
            "none": "no academic plan submitted"}
    PROGRAM_CREDITS = 120
    MAX_ATTEMPTED = 180

    def sample_factors(self, rng):
        reg = regime(rng)
        n_terms = pick(rng, [3, 4, 5, 6])
        circ = pick_w(rng, {"bereavement": 2, "injury": 2, "work_hours": 2, "housing_loss": 2, "none_specific": 1})
        resolved = pick_w(rng, {"resolved": 3, "ongoing": 2, "unclear": 2})
        documented = circ != "none_specific" and chance(rng, 0.65)
        plan = pick_w(rng, {"advisor_signed": 3, "student_drafted": 3, "none": 2})
        prior_appeals = pick(rng, [0, 0, 0, 1, 1, 2])
        prior_major_credits = pick(rng, [0, 0, 30, 60, 90, 110])
        bad_severity = pick(rng, [0.3, 0.6, 1.0])
        if reg == "clear":
            if chance(rng, 0.5):
                circ, resolved, documented, plan, prior_appeals, prior_major_credits, bad_severity = (
                    pick(rng, ["bereavement", "injury"]), "resolved", True, "advisor_signed", 0, 0, 0.3)
            else:
                circ, resolved, documented, plan, prior_appeals, bad_severity = "none_specific", "unclear", False, "none", 2, 1.0
                n_terms = pick(rng, [5, 6])
        elif reg == "conflicting":
            twist = pick(rng, ["good_reason_repeat", "documented_but_ongoing", "fixable_but_timeframe", "no_docs_strong_plan"])
            if twist == "good_reason_repeat":
                documented, prior_appeals = True, 2
            elif twist == "documented_but_ongoing":
                circ, documented, resolved = pick(rng, ["work_hours", "housing_loss"]), True, "ongoing"
            elif twist == "fixable_but_timeframe":
                prior_major_credits, bad_severity, plan = pick(rng, [75, 90, 100]), 0.3, "advisor_signed"
                n_terms = pick(rng, [4, 5, 6])
            else:
                documented, plan = False, "advisor_signed"
        # build a term-by-term record; the last term is the one the circumstance hit
        terms, att_tot, earned_tot, qp = [], 0, 0, 0.0
        for i in range(n_terms):
            att = pick(rng, [12, 13, 15, 15, 16])
            if i == n_terms - 1:
                earned = int(round(att * rng.uniform(0.05, 0.5) * (1.3 - bad_severity)))
                gpa = round(max(0.0, rng.uniform(0.2, 1.3) * (1.3 - bad_severity)), 2)
            else:
                earned = int(round(att * rng.uniform(0.6, 1.0)))
                gpa = round(rng.uniform(1.6, 2.9), 2)
            terms.append({"term": i + 1, "attempted": att, "earned": min(att, earned), "gpa": gpa})
        for _ in range(8):   # the student must actually be failing a standard to be appealing
            att_tot = sum(t["attempted"] for t in terms) + prior_major_credits
            earned_tot = sum(t["earned"] for t in terms) + int(prior_major_credits * 0.8)
            qp = sum(t["gpa"] * t["attempted"] for t in terms)
            cum_gpa = round(qp / sum(t["attempted"] for t in terms), 2)
            completion = round(earned_tot / att_tot, 3)
            if cum_gpa < 2.0 or completion < 0.67 or att_tot > self.MAX_ATTEMPTED:
                break
            j = rng.randrange(n_terms)
            terms[j]["gpa"] = round(max(0.0, terms[j]["gpa"] - 0.35), 2)
            terms[j]["earned"] = max(0, terms[j]["earned"] - 3)
        applicable = int(round(prior_major_credits * 0.8 * pick(rng, [0.2, 0.4, 0.6, 0.9])))
        degree_credits = sum(t["earned"] for t in terms) + applicable
        student, reviewer = _names(rng, 2)
        return {"regime": reg, "circumstance": circ, "circumstance_resolved": resolved, "documented": documented,
                "earlier_credits_applying_to_degree": applicable, "credits_counting_toward_degree": degree_credits,
                "plan": plan, "prior_appeals": prior_appeals, "prior_major_credits": prior_major_credits,
                "terms": terms, "credits_attempted": att_tot, "credits_earned": earned_tot, "cumulative_gpa": cum_gpa,
                "completion_rate": completion, "student": student, "reviewer": reviewer, "student_id": _sid(rng),
                "college": _college(rng), "major": pick(rng, ["Biology", "Accounting", "Nursing", "History", "Computer Science",
                                                               "Social Work", "Mechanical Engineering"]),
                "deadline": rand_date(rng)}

    def _resolved_text(self, f):
        return {"resolved": "the student says the situation is now behind them",
                "ongoing": "the student says the situation is continuing this term",
                "unclear": "the appeal does not say whether the situation has changed"}[f["circumstance_resolved"]]

    def render(self, f, rng, surface):
        doc = self.DOCS[f["circumstance"]] if f["documented"] else None
        std = f"standard: cumulative GPA 2.00, completion 67%, maximum {self.MAX_ATTEMPTED} attempted credits for a {self.PROGRAM_CREDITS}-credit degree"
        need = self.PROGRAM_CREDITS - f["credits_counting_toward_degree"]
        proj = (f"{need} credits still needed; attempted so far plus still needed = "
                f"{f['credits_attempted'] + need} (aid limit {self.MAX_ATTEMPTED})")
        if surface == "json":
            return dumps(prune({
                "student": {"id": f["student_id"], "major": f["major"], "previous_sap_appeals": f["prior_appeals"],
                            "credits_attempted_in_earlier_major": f["prior_major_credits"] or None},
                "progress": {"cumulative_gpa": f["cumulative_gpa"], "completion_rate": f"{f['completion_rate'] * 100:.1f}%",
                              "credits_attempted": f["credits_attempted"], "credits_earned": f["credits_earned"],
                             "credits_counting_toward_current_degree": f["credits_counting_toward_degree"],
                             "terms": f["terms"], "policy": std, "credit_projection": proj},
                "appeal": {"circumstance": self.CIRC[f["circumstance"]], "status": self._resolved_text(f),
                           "documentation": doc or "none supplied", "academic_plan": self.PLAN[f["plan"]]},
            }))
        rows = [[f"Term {t['term']}", t["attempted"], t["earned"], f"{t['gpa']:.2f}"] for t in f["terms"]]
        tbl = table(["term", "attempted", "earned", "term GPA"], rows, rng)
        if surface == "appeal_packet":
            head = kv_lines([("Student", f"{f['student_id']} ({f['major']})"), ("Cumulative GPA", f"{f['cumulative_gpa']:.2f}"),
                             ("Completion rate", f"{f['completion_rate'] * 100:.1f}% ({f['credits_earned']}/{f['credits_attempted']})"),
                             ("Credits from earlier major", f"{f['prior_major_credits']} attempted, {f['earlier_credits_applying_to_degree']} apply to this degree"
                              if f["prior_major_credits"] else None),
                             ("Credits counting toward degree", f"{f['credits_counting_toward_degree']} of {self.PROGRAM_CREDITS}"),
                             ("Credit projection", proj),
                             ("Previous appeals", f["prior_appeals"]), ("Documentation", doc or "none"),
                             ("Academic plan", self.PLAN[f["plan"]])], rng)
            return "\n".join([f"SAP APPEAL -- {f['college']} -- {AWARD_YEAR}", head, "", tbl, "",
                              "Student's explanation: " + noisy(f"{self.CIRC[f['circumstance']][0].upper()}{self.CIRC[f['circumstance']][1:]}; "
                                                                f"{self._resolved_text(f)}.", rng, 0.2), "", f"({std})"])
        s = [f"{f['student_id']}, {f['major']}: cumulative {f['cumulative_gpa']:.2f}, completion {f['completion_rate'] * 100:.0f}% "
             f"({f['credits_earned']} of {f['credits_attempted']} attempted"
             + (f", including {f['prior_major_credits']} from an earlier major, {f['earlier_credits_applying_to_degree']} of which apply"
                if f["prior_major_credits"] else "") + f"); {f['credits_counting_toward_degree']} credits count toward the {self.PROGRAM_CREDITS}-credit degree.",
             f"Reason given: {self.CIRC[f['circumstance']]}; {self._resolved_text(f)}.",
             (f"Documented ({doc})." if doc else "Nothing documented."), f"{self.PLAN[f['plan']].capitalize()}.",
             (f"This is appeal number {f['prior_appeals'] + 1}." if f["prior_appeals"] else "First appeal.")]
        return "\n".join([f"Committee note -- {f['reviewer']}", noisy(join_sentences(s), rng, 0.1), "", tbl, "", std,
                          f"Credit projection: {proj}."])

    def questions(self):
        return {
            "sap_outcome": choice("How should the progress appeal be decided?", {
                "approve_on_probation": "Approve for one probationary term: a documented, resolved circumstance and "
                                        "numbers one solid term can fix.",
                "approve_with_plan": "Approve on the advisor-signed academic plan, with term-by-term targets, because "
                                     "recovery will take more than one term or the situation is still going on.",
                "deny_appeal": "Deny the appeal: no specific circumstance, a third appeal, or the student would pass "
                               "the maximum attempted credits before finishing.",
                "request_plan_or_documents": "Hold the appeal until the student supplies documents for the circumstance "
                                             "or an advisor-reviewed academic plan.",
            }),
            "recovery_feasibility": score("How realistic is it that the student will meet the standards within the plan's timeframe?", [
                "Unrealistic: the student would pass the maximum attempted credits before finishing the degree.",
                "Doubtful: cumulative GPA below 1.60 or completion below 50%, so near-perfect terms are needed.",
                "Plausible: GPA of at least 1.60 and completion of at least 50%, but short of the levels below.",
                "Likely: GPA of 1.85 or more and completion of 60% or more, so one solid term restores good standing.",
            ]),
            "circumstance_explains_record": score("How convincingly does the stated circumstance explain the poor record?", [
                "Not at all: no specific circumstance is given.",
                "Partly: a real event, but an earlier term already had a GPA below 1.70 before it.",
                "Largely: the event lines up with the one bad term, but it is undocumented or is ongoing work pressure.",
                "Fully: a documented bereavement, injury or housing loss lines up with the one bad term.",
            ]),
            "circumstance_resolved": noul(
                "The situation that caused the poor progress has been resolved or brought under control.",
                yes="The event is over or managed, so it should not keep affecting the student's studies.",
                no="The situation continues, or nothing says it has changed."),
            "within_maximum_timeframe": noul(
                "The student can still finish the degree without exceeding the maximum attempted credits aid will fund.",
                yes="Credits attempted so far plus the credits still needed for the degree come to no more than the "
                    "aid limit.",
                no="Credits attempted so far plus the credits still needed exceed the aid limit."),
        }


# =========================================================================== schema 4
class CostOfAttendanceIncrease(Schema):
    name = "cost_of_attendance_increase"
    description = "A request to raise a student's cost-of-attendance budget for an expense above the standard allowance."
    surfaces = {"json": 0.5, "request_form": 0.25, "student_email": 0.25}

    # key -> (description, median cost, office reasonable-cost cap or None, standard allowance line)
    EXPENSES = {
        "required_laptop": ("a new laptop for coursework", 1500, 1200, "books and supplies $1,100 per year"),
        "commuting_car_repair": ("repairs to the car used to reach an off-campus clinical placement", 1400, 1000, "transportation $1,800 per year"),
        "housing_above_allowance": ("rent above the off-campus housing allowance", 3600, None, "housing $10,800 per year"),
        "program_kit_fee": ("a required studio and equipment kit charged by the department", 900, None, "books and supplies $1,100 per year"),
        "study_abroad_travel": ("flights for a credit-bearing semester abroad", 1800, 1500, "transportation $1,800 per year"),
        "licensure_exam_fee": ("licensure exam and background-check fees due before graduation", 450, 500, "no allowance for exam fees"),
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        exp = pick(rng, list(self.EXPENSES))
        _, median, cap, _ = self.EXPENSES[exp]
        amount = _r100(near(rng, cap if cap else median, median * 0.4, lo=150)) if exp != "licensure_exam_fee" else int(near(rng, 450, 150, lo=150, integer=True))
        receipts = pick_w(rng, {"itemised": 3, "quote_only": 3, "none": 2})
        required = chance(rng, 0.6)
        prior = chance(rng, 0.35)
        grant_funds = pick_w(rng, {"available": 2, "limited": 3, "exhausted": 2})
        loan_left = pick(rng, [0, 0, 1500, 3500, 5500])
        need_gap = _r100(lognormal_amount(rng, 4000, 0.7))
        if reg == "clear":
            if chance(rng, 0.5):
                receipts, required, prior = "itemised", True, False
                amount = (min(amount, cap) if cap else amount)
                exp = pick(rng, ["program_kit_fee", "licensure_exam_fee", "required_laptop"])
                _, median, cap, _ = self.EXPENSES[exp]
                amount = _r100(median * rng.uniform(0.7, 0.95)) if exp != "licensure_exam_fee" else rng.randint(300, 480)
            else:
                receipts, required, prior = "none", False, True
                exp = "required_laptop"
                amount = _r100(rng.uniform(2600, 3400))
        elif reg == "conflicting":
            twist = pick(rng, ["required_but_over_cap", "reasonable_but_already_funded", "housing_choice", "approved_but_no_aid_left"])
            if twist == "required_but_over_cap":
                exp, required, receipts = "required_laptop", True, "itemised"
                amount = _r100(rng.uniform(1900, 2600))
            elif twist == "reasonable_but_already_funded":
                exp, prior, receipts = "required_laptop", True, "itemised"
                amount = _r100(rng.uniform(900, 1150))
            elif twist == "housing_choice":
                exp, required, receipts = "housing_above_allowance", False, "itemised"
            else:
                receipts, required, loan_left, grant_funds = "itemised", True, 0, "exhausted"
        if exp in ("housing_above_allowance", "study_abroad_travel", "commuting_car_repair") and prior:
            prior = False
        student, officer = _names(rng, 2)
        return {"regime": reg, "expense": exp, "amount": amount, "office_cap": self.EXPENSES[exp][2],
                "receipts": receipts, "required_by_programme": required, "funded_before": prior,
                "institutional_grant_funds": grant_funds, "unsubsidized_loan_remaining": loan_left,
                "unmet_need": need_gap, "student": student, "officer": officer, "student_id": _sid(rng),
                "college": _college(rng), "submitted": rand_date(rng),
                "enrolment": pick_w(rng, {"full-time": 4, "three-quarter time": 1, "half-time": 1})}

    RECEIPTS = {"itemised": "itemised receipts", "quote_only": "a quote, not yet paid", "none": "no receipts or quotes"}
    FUNDS = {"available": "institutional grant funds available", "limited": "institutional grant funds limited; priority to highest need",
             "exhausted": "institutional grant funds exhausted for the year"}

    def _required(self, f):
        e = f["expense"]
        if e == "housing_above_allowance":
            return ("student says no cheaper housing was available near the placement" if f["required_by_programme"]
                    else "student chose a single studio closer to campus")
        if e == "required_laptop":
            return ("the programme handbook requires a laptop of this specification" if f["required_by_programme"]
                    else "the handbook recommends a laptop but campus labs meet the requirement")
        return ("the programme handbook lists it as required" if f["required_by_programme"]
                else "not listed as a programme requirement")

    def _prior(self, f):
        if not f["funded_before"]:
            return None
        return "a laptop allowance was added to this student's budget two years ago" if f["expense"] == "required_laptop" \
            else "a similar adjustment was approved for this student last year"

    def render(self, f, rng, surface):
        desc, _, cap, allowance = self.EXPENSES[f["expense"]]
        if surface == "json":
            return dumps(prune({
                "request": {"student_id": f["student_id"], "award_year": AWARD_YEAR, "enrolment": f["enrolment"],
                            "expense": desc, "amount_requested": f["amount"], "evidence": self.RECEIPTS[f["receipts"]],
                            "programme_requirement": self._required(f)},
                "budget": {"standard_allowance": allowance, "office_reasonable_cost_cap": cap,
                           "typical_cost": self.EXPENSES[f["expense"]][1],
                           "earlier_adjustment": self._prior(f) or "none on file"},
                "aid_position": {"unmet_need_before_request": f["unmet_need"],
                                 "unsubsidized_loan_eligibility_remaining": f["unsubsidized_loan_remaining"],
                                 "institutional_grants": self.FUNDS[f["institutional_grant_funds"]]},
            }))
        if surface == "request_form":
            rows = [("Student ID", f["student_id"]), ("Submitted", fmt_date(f["submitted"], rng)), ("Enrolment", f["enrolment"]),
                    ("Expense", desc),
                    ("Amount", money(f["amount"], rng)), ("Evidence", self.RECEIPTS[f["receipts"]]),
                    ("Required by programme?", self._required(f)), ("Standard allowance", allowance),
                    ("Office cap for this item", money(cap, rng) if cap else "none set"),
                    ("Typical cost", money(self.EXPENSES[f["expense"]][1], rng)),
                    ("Earlier adjustment", self._prior(f) or "none on file"),
                    ("Unmet need", money(f["unmet_need"], rng)),
                    ("Unsubsidized loan still available", money(f["unsubsidized_loan_remaining"], rng) if f["unsubsidized_loan_remaining"] else "none"),
                    ("Grant funds", self.FUNDS[f["institutional_grant_funds"]])]
            return f"COST OF ATTENDANCE ADJUSTMENT REQUEST -- {f['college']}\n" + kv_lines(rows, rng)
        s = [f"I'm writing to ask for my budget to be increased by {money(f['amount'], rng)} for {desc}.",
             {"itemised": "I've attached the receipts.", "quote_only": "I've attached the quote; I haven't paid it yet.",
              "none": "I don't have the receipt to hand but can find it."}[f["receipts"]],
             ("It's required for my programme." if f["required_by_programme"] else "")
             if f["expense"] != "housing_above_allowance" else
             ("There was nothing cheaper near my placement." if f["required_by_programme"] else "I moved closer to campus this year."),
             pick(rng, ["I really can't cover this myself.", "Any help would make a big difference.", ""])]
        tail = [f"--- office lookup: allowance {allowance}; cap {money(cap, rng) if cap else 'none'}; "
                f"typical cost {money(self.EXPENSES[f['expense']][1], rng)}; "
                f"{self._prior(f) or 'no earlier adjustment'}; unmet need {money(f['unmet_need'], rng)}; "
                f"loan left {money(f['unsubsidized_loan_remaining'], rng) if f['unsubsidized_loan_remaining'] else 'none'}; "
                f"{self.FUNDS[f['institutional_grant_funds']]}."]
        return "\n".join([f"Subject: Budget increase request -- {f['student_id']}", f"From: {f['student']}", "To: financial aid", "",
                          noisy(join_sentences(s), rng, 0.2), "", f["student"].split()[0], ""] + tail)

    def questions(self):
        return {
            "coa_decision": choice("How should the aid office decide the budget increase?", {
                "approve_capped": "Approve, but only up to the office's reasonable-cost limit for this kind of expense.",
                "approve_full": "Approve the full amount requested.",
                "deny_increase": "Deny the increase; the expense does not qualify or is already provided for.",
                "request_receipts": "Hold the request until receipts or other proof of the cost are supplied.",
            }),
            "expense_necessity": score("How necessary is the expense for the student to attend and complete the programme?", [
                "Unnecessary: a personal choice the programme does not require.",
                "Helpful: it makes study easier but is not required.",
                "Important: hard to complete the programme without it.",
                "Essential: the programme requires it or the student cannot attend otherwise.",
            ]),
            "cost_reasonableness": score("How reasonable is the amount compared with what the item or service usually costs?", [
                "Excessive: more than half again above the office cap (or the typical cost where no cap is set).",
                "High: above the office cap or typical cost, by up to half again.",
                "Reasonable: at or up to 10% below the office cap or typical cost.",
                "Modest: more than 10% below the office cap or typical cost.",
            ]),
            "already_provided_for": noul(
                "The cost is already covered by the standard budget or by an earlier adjustment.",
                yes="An earlier adjustment already added this kind of expense to the student's budget.",
                no="No earlier adjustment is on file for it, and the standard allowance does not stretch to it."),
            "realistic_aid_effect": choice("If the increase were approved, what would it realistically make available?", {
                "additional_grant": "More grant aid from the college's own funds.",
                "emergency_fund_referral": "Nothing through regular aid; a referral to the student emergency fund is the only route.",
                "loan_eligibility_only": "Only additional loan eligibility, not grant money.",
                "no_additional_aid": "Nothing further, because the student has no remaining aid eligibility to unlock.",
            }),
        }


# =========================================================================== schema 5
class OneTimeIncomeExclusion(Schema):
    name = "one_time_income_exclusion"
    description = "A request to remove a one-off income item from the tax-year figures used to calculate aid."
    surfaces = {"json": 0.5, "pj_worksheet": 0.25, "family_letter": 0.25}

    ITEMS = {"retirement_withdrawal": "an early withdrawal from a retirement account",
             "severance_pay": "severance paid when a parent's job ended",
             "home_sale_gain": "the taxable gain on selling the family home",
             "back_pay_lump_sum": "back pay covering two earlier years, paid in one lump sum",
             "cancelled_debt": "cancelled credit-card debt reported as income",
             "lawsuit_settlement": "a taxable settlement for lost wages after a workplace injury"}
    SPENT = {"medical_bills": "paying hospital bills after a relative's surgery",
             "avoid_foreclosure": "clearing mortgage arrears to stop a foreclosure",
             "living_costs": "ordinary living costs while out of work",
             "moved_to_cheaper_home": "the deposit on a smaller, cheaper home",
             "paid_old_debts": "paying off older debts",
             "still_in_savings": "most of it is still in a savings account",
             "no_cash_received": "no cash was received; the debt was simply written off",
             "not_stated": None}
    SPENT_FITS = {"retirement_withdrawal": ["medical_bills", "avoid_foreclosure", "living_costs", "still_in_savings", "not_stated"],
                  "severance_pay": ["living_costs", "paid_old_debts", "still_in_savings", "not_stated"],
                  "home_sale_gain": ["moved_to_cheaper_home", "still_in_savings", "paid_old_debts"],
                  "back_pay_lump_sum": ["paid_old_debts", "living_costs", "still_in_savings", "not_stated"],
                  "cancelled_debt": ["no_cash_received"],
                  "lawsuit_settlement": ["medical_bills", "living_costs", "still_in_savings", "not_stated"]}

    def sample_factors(self, rng):
        reg = regime(rng)
        item = pick(rng, list(self.ITEMS))
        agi = _r100(lognormal_amount(rng, 70000, 0.4))
        share = pick(rng, [0.08, 0.15, 0.25, 0.4])
        spent = pick(rng, self.SPENT_FITS[item])
        recurs = chance(rng, 0.2) if item in ("retirement_withdrawal", "back_pay_lump_sum") else False
        docs = pick_w(rng, {"tax_form_and_statement": 3, "letter_only": 2, "none": 2})
        deadline = pick(rng, [3, 9, 20, 45, 90])
        grant_threshold = pick(rng, [6000, 7500, 9000])
        if reg == "clear":
            if chance(rng, 0.5):
                item, spent, recurs, docs, share = "retirement_withdrawal", pick(rng, ["medical_bills", "avoid_foreclosure"]), False, "tax_form_and_statement", 0.3
            else:
                item, spent, recurs, docs = "retirement_withdrawal", "still_in_savings", True, "none"
        elif reg == "conflicting":
            twist = pick(rng, ["one_off_but_saved", "spent_but_recurring", "cancelled_debt", "small_but_urgent"])
            if twist == "one_off_but_saved":
                item, spent, docs = pick(rng, ["home_sale_gain", "severance_pay"]), "still_in_savings", "tax_form_and_statement"
            elif twist == "spent_but_recurring":
                item, spent, recurs = "retirement_withdrawal", "medical_bills", True
            elif twist == "cancelled_debt":
                item, spent, docs = "cancelled_debt", "no_cash_received", pick(rng, ["tax_form_and_statement", "letter_only"])
            else:
                share, deadline = 0.08, pick(rng, [2, 4])
        if spent not in self.SPENT_FITS[item]:
            spent = pick(rng, self.SPENT_FITS[item])
        amount = _r100(agi * share)
        sai = int(round(agi * pick(rng, [0.1, 0.13, 0.16]), -1))
        sai_after = int(round(max(-1500, sai - amount * pick(rng, [0.2, 0.3, 0.4])), -1))
        student, officer = _names(rng, 2)
        return {"regime": reg, "item": item, "amount": amount, "tax_year_income": agi, "share_of_income": share,
                "spent_on": spent, "recurred_in_prior_years": recurs, "documents": docs, "days_to_payment_deadline": deadline,
                "current_sai": sai, "sai_if_excluded": sai_after, "grant_sai_threshold": grant_threshold,
                "enrolment_cancelled_if_unpaid": chance(rng, 0.5), "student": student, "officer": officer,
                "student_id": _sid(rng), "college": _college(rng), "received": rand_date(rng)}

    DOCS = {"tax_form_and_statement": "the tax form reporting the item and an account statement",
            "letter_only": "a letter from the family explaining it", "none": "nothing yet"}

    def _deadline(self, f):
        return (f"tuition payment due in {f['days_to_payment_deadline']} days"
                + ("; unpaid balances lead to cancellation of the student's classes" if f["enrolment_cancelled_if_unpaid"] else ""))

    def _recur(self, f):
        if f["item"] not in ("retirement_withdrawal", "back_pay_lump_sum"):
            return None
        return ("similar withdrawals appear on the two previous tax returns" if f["recurred_in_prior_years"]
                else "no similar item on the two previous tax returns")

    def render(self, f, rng, surface):
        spent = self.SPENT[f["spent_on"]]
        if surface == "json":
            return dumps(prune({
                "request": {"student_id": f["student_id"], "award_year": AWARD_YEAR, "received": f["received"].isoformat(),
                            "item": self.ITEMS[f["item"]], "amount": f["amount"], "family_says_it_was_used_for": spent,
                            "documents": self.DOCS[f["documents"]]},
                "tax_year": {"year": TAX_YEAR, "income_including_item": f["tax_year_income"], "prior_years": self._recur(f)},
                "calculation": {"current_sai": f["current_sai"], "sai_if_item_excluded": f["sai_if_excluded"],
                                "grant_eligibility_ends_above_sai": f["grant_sai_threshold"]},
                "timing": self._deadline(f),
            }))
        if surface == "pj_worksheet":
            rows = [("Student", f["student_id"]), ("Item", self.ITEMS[f["item"]]), ("Amount", money(f["amount"], rng)),
                    (f"{TAX_YEAR} income incl. item", money(f["tax_year_income"], rng)), ("Use of funds (per family)", spent),
                    ("Prior years", self._recur(f) or "n/a"), ("Documents", self.DOCS[f["documents"]]),
                    ("SAI now / if excluded", f"{f['current_sai']:,} / {f['sai_if_excluded']:,}"),
                    ("Grant cut-off SAI", f"{f['grant_sai_threshold']:,}"), ("Timing", self._deadline(f))]
            return f"PROFESSIONAL JUDGEMENT WORKSHEET -- {f['college']} -- analyst {f['officer']}\n" + kv_lines(rows, rng, skip_none=False)
        s = [f"Our {TAX_YEAR} income includes {self.ITEMS[f['item']]} of {money(f['amount'], rng)}.",
             ("It was spent on " + spent + "." if spent and f["spent_on"] not in ("still_in_savings", "no_cash_received")
              else ("To be honest " + spent + "." if f["spent_on"] == "still_in_savings" else
                    ("We never received any money; " + spent.split("; ")[1] + "." if f["spent_on"] == "no_cash_received" else ""))),
             "It will not happen again." if not f["recurred_in_prior_years"] else "We have had to do this before, but hope not again.",
             {"tax_form_and_statement": "The tax form and a statement are attached.", "letter_only": "", "none": ""}[f["documents"]],
             pick(rng, ["Please take it out of the calculation.", "We are asking for it to be excluded."])]
        return "\n".join([f"Subject: Request to exclude one-time income -- {f['student_id']}", "", "Dear financial aid office,", "",
                          noisy(join_sentences(s), rng, 0.2), "", "Sincerely,", f"Parent of {f['student']}", "",
                          f"[office: SAI {f['current_sai']:,} now, about {f['sai_if_excluded']:,} if excluded; grant cut-off "
                          f"{f['grant_sai_threshold']:,}; {self._deadline(f)}]"])

    def questions(self):
        return {
            "exclusion_decision": choice("How should the office treat the one-off income?", {
                "deny_exclusion": "Leave the income in the calculation.",
                "exclude_fully": "Remove the whole item from the income used to calculate aid.",
                "exclude_part": "Remove only the part spent on the stated necessary costs and count the rest.",
                "request_documents": "Ask for documents showing the item and how it was used before deciding.",
            }),
            "nonrecurrence_confidence": score("How confident can the office be that this income will not recur?", [
                "Not confident: the same kind of income has appeared before.",
                "Somewhat: nothing shows it recurring, but nothing rules it out.",
                "Fairly: it arose from a specific event unlikely to repeat.",
                "Fully: by its nature it cannot recur.",
            ]),
            "decision_urgency": score("How soon does the office need to decide this request?", [
                "No hurry: the next payment date is months away.",
                "In the normal queue of a few weeks.",
                "Before the tuition payment date in the next week or two.",
                "Within days: unpaid charges will soon cancel the student's classes.",
            ]),
            "used_for_necessities": noul(
                "The one-off income was used for necessary costs rather than kept as savings.",
                yes="The money went, or went mostly, on unavoidable costs such as medical bills or keeping the home.",
                no="Most of it remains in savings, its use is not stated, or no cash was received at all."),
            "changes_grant_eligibility": noul(
                "Excluding the item would change the student's grant eligibility, not only their loan amounts.",
                yes="The SAI is above the grant cut-off now and would fall to or below it if the item were excluded.",
                no="The SAI is on the same side of the grant cut-off either way."),
        }


WORKFLOW = Workflow(
    name="financial_aid_appeals",
    title="Financial aid appeals",
    description="How a college aid office should decide special-circumstance appeals and professional-judgement "
                "requests: income changes, dependency overrides, progress appeals, budget increases and one-off income.",
    domain="higher education / student financial aid",
    schemas=[IncomeLossAppeal(), DependencyOverride(), SapAppeal(), CostOfAttendanceIncrease(), OneTimeIncomeExclusion()],
    notes="Aid-office casework on income, documentation, circumstances and academic records at fictional colleges. "
          "No student-support conversation replies, no payment-fraud alerts, no bills. No protected attribute is "
          "stated or used.",
)
