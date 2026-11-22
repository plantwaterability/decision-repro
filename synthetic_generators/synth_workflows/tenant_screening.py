"""Tenant screening: rental applications assessed against a property's WRITTEN screening criteria.

Five schemas: a standard application, a guarantor offered to cover a shortfall, an adverse rental record that
needs individual assessment, irregular or inconsistent income documents, and a joint application by several
adult co-applicants. Each samples a `regime` first ('clear' / 'borderline' / 'conflicting') and lets it steer
the factors: income multiples a hair under the requirement, credit scores within a few points of the minimum,
references that contradict the numbers, and documents that are missing or do not quite add up.

Only lawful, criteria-based factors appear: verified income against a rent multiple, credit and rental-debt
history, landlord references, tenancy length, document authenticity and guarantees. No applicant's race,
religion, national origin, sex, disability, age or family status is stated or implied, household composition
is never a factor, and no income SOURCE is treated as disqualifying: income is weighed only on whether it is
documented and stable. Property managers, addresses and people are fictional.
"""
from __future__ import annotations

from datetime import timedelta

from .base import (
    FIRST,
    LAST,
    ORG_A,
    REAL_NAME_CLASHES,
    Schema,
    Workflow,
    chance,
    choice,
    dumps,
    fmt_date,
    fmt_ts,
    join_sentences,
    kv_lines,
    lognormal_amount,
    maybe,
    money,
    near,
    noisy,
    noul,
    pick,
    pick_w,
    prune,
    rand_date,
    rand_ts,
    regime,
    score,
    show,
    table,
)

STREETS = ["Alder", "Birch", "Canal", "Dover", "Elm", "Foundry", "Garnet", "Hollis", "Ivy", "Juniper", "Kestrel",
           "Linden", "Mill", "Nettle", "Orchard", "Quarry", "Rowan", "Sparrow", "Tanner", "Wharf"]
SUFFIX = ["Street", "Avenue", "Road", "Lane", "Court", "Terrace"]
MGR_KIND = ["Residential", "Property Management", "Lettings", "Homes", "Realty"]
# hand-built names are checked against base.REAL_NAME_CLASHES, like base.org's own output
EMPLOYERS = [e for e in ["Tidewater Cold Storage", "Granite Health Partners", "Copperline Systems", "Bluestem Provisions",
                         "Keystone Parcel Services", "Oakridge Clinic", "Meridian Analytics", "Northgate Works",
                         "Harbor Outfitters", "Pinecrest Labs", "Valewood Manufacturing", "Saltmarsh Technologies"]
             if e not in REAL_NAME_CLASHES]
EMP_TYPES = {"salaried": "salaried, permanent", "hourly": "hourly, permanent", "fixed_term_contract": "fixed-term contract",
             "self_employed": "self-employed", "commission": "base plus commission"}


def _names(rng, n):
    """n fictional people with distinct first AND last names."""
    return [f"{a} {b}" for a, b in zip(rng.sample(FIRST, n), rng.sample(LAST, n))]


def _address(rng):
    return f"{rng.randint(12, 980)} {pick(rng, STREETS)} {pick(rng, SUFFIX)}, Apt {rng.randint(1, 6)}{pick(rng, 'ABCDEF')}"


def _manager(rng):
    a, i = pick(rng, ORG_A), rng.randrange(len(MGR_KIND))
    for k in range(len(MGR_KIND)):          # skip any clash without an extra rng draw, as base.org does
        name = f"{a} {MGR_KIND[(i + k) % len(MGR_KIND)]}"
        if name not in REAL_NAME_CLASHES:
            return name
    return name


def _criteria(rng):
    """A property's written screening criteria (the thresholds the case is judged against)."""
    return {"income_multiple": pick(rng, [2.5, 2.75, 3.0, 3.0, 3.0, 3.5]),
            "min_credit": pick(rng, [580, 600, 620, 620, 650]),
            "rental_history_months": pick(rng, [12, 24, 24]),
            "lookback_years": pick(rng, [3, 5, 5, 7]),
            "guarantor_multiple": pick(rng, [4.0, 5.0, 5.0]),
            "decision_business_days": pick(rng, [2, 3, 3, 5])}


def _rent(rng, median=1750, sigma=0.35):
    return int(round(lognormal_amount(rng, median, sigma), -1))


def _r10(x):
    return int(round(x, -1))


def _x(v):
    return f"{v:.2f}x"


# =========================================================================== schema 1
class RentalApplication(Schema):
    name = "rental_application"
    description = "A single applicant's rental application checked against the property's written criteria."
    surfaces = {"json": 0.5, "screening_report": 0.25, "agent_email": 0.25}

    REF = {"positive": "paid on time, would rent to again", "lukewarm": "'no major issues', would not say more",
           "late_payments": "reported {n} late payment(s) in the last two years, all paid",
           "no_response": "did not respond to two requests", "relative_landlord": "reference is from the applicant's relative, who owned the unit",
           "first_tenancy": "none: first tenancy in the applicant's own name"}

    def sample_factors(self, rng):
        reg = regime(rng)
        crit = _criteria(rng)
        rent = _rent(rng)
        req, min_cr = crit["income_multiple"], crit["min_credit"]
        emp = pick_w(rng, {"salaried": 4, "hourly": 3, "fixed_term_contract": 1, "self_employed": 1, "commission": 1})
        doc = pick_w(rng, {"two_recent_paystubs": 5, "offer_letter": 1, "bank_statements": 2, "self_declared": 1})
        months_job = pick(rng, [1, 4, 9, 18, 40, 80])
        hist = pick(rng, [0, 8, 14, 26, 60])
        ref = pick_w(rng, {"positive": 4, "lukewarm": 2, "late_payments": 2, "no_response": 2, "relative_landlord": 1})
        late = pick(rng, [1, 2, 3]) if ref == "late_payments" else 0
        collections = pick(rng, [0, 0, 0, 180, 640, 2300])
        if reg == "clear":
            if chance(rng, 0.5):
                ratio, credit = round(rng.uniform(req + 0.6, req + 1.6), 2), min_cr + rng.randint(60, 150)
                doc, emp, months_job, hist, ref, late, collections = "two_recent_paystubs", "salaried", pick(rng, [18, 40, 80]), pick(rng, [26, 60]), "positive", 0, 0
            else:
                ratio, credit = round(rng.uniform(1.6, req - 0.6), 2), min_cr - rng.randint(40, 110)
                doc, ref, late = pick(rng, ["self_declared", "bank_statements"]), "late_payments", 3
        elif reg == "borderline":
            ratio = near(rng, req, 0.18, lo=1.5)
            credit = maybe(rng, near(rng, min_cr, 25, integer=True), 0.12)
        else:
            twist = pick(rng, ["rich_bad_ref", "short_income_great_history", "offer_letter_only", "thin_file_strong"])
            if twist == "rich_bad_ref":
                ratio, ref, late, credit = round(rng.uniform(req + 0.5, req + 1.5), 2), pick(rng, ["late_payments", "no_response"]), 3, min_cr + rng.randint(-20, 40)
            elif twist == "short_income_great_history":
                ratio, ref, hist, credit, late = round(rng.uniform(req - 0.45, req - 0.1), 2), "positive", 60, min_cr + rng.randint(80, 160), 0
            elif twist == "offer_letter_only":
                ratio, doc, months_job, emp = round(rng.uniform(req + 0.3, req + 1.2), 2), "offer_letter", 0, "salaried"
                credit = min_cr + rng.randint(-10, 60)
            else:
                ratio, credit, hist, ref = round(rng.uniform(req + 0.2, req + 1.0), 2), None, 0, "no_response"
            if ref == "late_payments" and not late:
                late = 2
        if hist == 0:
            ref, late = "first_tenancy", 0
        elif ref == "first_tenancy":
            ref = "no_response"
        if emp == "self_employed":
            months_job = max(months_job, pick(rng, [9, 18, 40]))
            if doc == "two_recent_paystubs":
                doc = "bank_statements"
        if doc == "offer_letter":
            months_job = 0
        income = _r10(rent * ratio)
        name, agent, pm = _names(rng, 3)
        return {"regime": reg, "criteria": crit, "rent": rent, "income_monthly": income, "income_ratio": round(income / rent, 2),
                "employment_type": emp, "income_document": doc, "months_at_employer": months_job,
                "employer": "own business" if emp == "self_employed" else pick(rng, EMPLOYERS), "credit_score": credit, "non_rental_collections": collections,
                "rental_history_months": hist, "landlord_reference": ref, "late_payments_24m": late,
                "file_complete_days_ago": pick(rng, [0, 1, 2, 3, 4]),
                "holding_deposit_hours_left": pick(rng, [None, None, 8, 24, 48, 96]),
                "competing_applications": pick(rng, [0, 0, 1, 2, 3]), "move_in_days": pick(rng, [5, 12, 21, 35, 60]),
                "applicant": name, "agent": agent, "property_manager": pm, "manager_co": _manager(rng),
                "address": _address(rng), "app_id": f"APP-{rng.randrange(10000, 99999)}",
                "agent_view": pick(rng, ["keen", "neutral", "doubtful"]), "submitted": rand_date(rng)}

    def _job_text(self, f):
        if f["employment_type"] == "self_employed":
            return f"self-employed, trading {f['months_at_employer']} months"
        if f["months_at_employer"] == 0:
            return f"{EMP_TYPES[f['employment_type']]} at {f['employer']}, starting next month"
        return f"{EMP_TYPES[f['employment_type']]} at {f['employer']}, {f['months_at_employer']} months"

    def _doc_text(self, f):
        return {"two_recent_paystubs": "two recent pay stubs", "offer_letter": "signed offer letter (start date next month)",
                "bank_statements": "three months of bank statements", "self_declared": "stated on the application only"}[f["income_document"]]

    def _ref_text(self, f):
        return self.REF[f["landlord_reference"]].format(n=f["late_payments_24m"])

    def render(self, f, rng, surface):
        c = f["criteria"]
        credit = f["credit_score"]
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["app_id"], "applicant": f["applicant"], "unit": f["address"], "monthly_rent": f["rent"],
                                "requested_move_in_days": f["move_in_days"]},
                "criteria": {"min_income_to_rent": c["income_multiple"], "min_credit_score": c["min_credit"],
                             "rental_history_months": c["rental_history_months"],
                             "decision_due_business_days_after_complete_file": c["decision_business_days"]},
                "income": {"monthly_gross": f["income_monthly"], "employment": self._job_text(f),
                           "evidence": self._doc_text(f)},
                "credit": {"score": credit if credit is not None else "no score (thin file)",
                           "non_rental_collections_usd": f["non_rental_collections"]},
                "rental_history": {"months_verified": f["rental_history_months"], "landlord_reference": self._ref_text(f)},
                "status": {"file_complete_days_ago": f["file_complete_days_ago"],
                           "holding_deposit_expires_in_hours": f["holding_deposit_hours_left"],
                           "other_applications_on_unit": f["competing_applications"]},
            }))
        if surface == "screening_report":
            lines = [f"TENANT SCREENING SUMMARY    {f['manager_co']}    report {f['app_id']}",
                     f"Unit: {f['address']}    Rent: {money(f['rent'], rng)}/mo", f"Applicant: {f['applicant']}", "-" * 60,
                     f"INCOME      {money(f['income_monthly'], rng)}/mo gross  ({_x(f['income_ratio'])} rent; required {c['income_multiple']}x)",
                     f"            {self._job_text(f)}; evidence: {self._doc_text(f)}",
                     f"CREDIT      {'score ' + str(credit) if credit is not None else 'NO SCORE (thin file)'} (minimum {c['min_credit']}); "
                     f"collections: {money(f['non_rental_collections'], rng) if f['non_rental_collections'] else 'none'}",
                     f"RENTAL      {f['rental_history_months']} months verified (required {c['rental_history_months']}); ref: {self._ref_text(f)}",
                     "-" * 60,
                     f"File complete {f['file_complete_days_ago']} day(s); decision due within {c['decision_business_days']} business days.",
                     f"Holding deposit: {'none taken' if f['holding_deposit_hours_left'] is None else str(f['holding_deposit_hours_left']) + 'h remaining'}."
                     f" Other applications on this unit: {f['competing_applications']}."]
            return "\n".join(lines)
        view = {"keen": pick(rng, ["They came across really well at the viewing and I'd like to get them in.",
                                   "Honestly I think they'd be a great tenant."]),
                "neutral": "No strong feelings either way from the viewing.",
                "doubtful": pick(rng, ["Something felt a bit off at the viewing, can't put my finger on it.",
                                       "They were vague when I asked about their current place."])}[f["agent_view"]]
        s = [f"Application {f['app_id']} for {f['address']} ({money(f['rent'], rng)}/mo) from {f['applicant']} is in.",
             f"Income is {money(f['income_monthly'], rng)} a month gross ({self._job_text(f)}), backed by {self._doc_text(f)}.",
             f"Credit {'score is ' + str(credit) if credit is not None else 'came back with no score, thin file'}"
             + (f", with {money(f['non_rental_collections'], rng)} in non-rental collections." if f["non_rental_collections"] else "."),
             (f"Rental history: {f['rental_history_months']} months; landlord {self._ref_text(f)}." if f["rental_history_months"]
              else "No previous tenancy in their own name."),
             f"File's been complete {f['file_complete_days_ago']} day(s)." if f["file_complete_days_ago"] else "File was completed today.",
             (f"Holding deposit runs out in {f['holding_deposit_hours_left']} hours." if f["holding_deposit_hours_left"] else ""),
             (f"There are {f['competing_applications']} other applications on the unit." if f["competing_applications"] else ""),
             view]
        return "\n".join([f"Subject: {f['app_id']} -- decision needed", f"From: {f['agent']} (leasing)",
                          f"To: {f['property_manager']}", "", f"Hi {f['property_manager'].split()[0]},", "",
                          noisy(join_sentences(s), rng, 0.2), "",
                          f"Our criteria: {c['income_multiple']}x rent, {c['min_credit']}+ credit, {c['rental_history_months']} months' history.",
                          "", f["agent"].split()[0]])

    def questions(self):
        return {
            "screening_decision": choice("What should the property manager decide on this application?", {
                "approve": "Offer the tenancy on the standard terms.",
                "approve_with_safeguard": "Offer it only with a safeguard the written criteria allow, such as a guarantor "
                                          "or the maximum permitted deposit.",
                "decline": "Decline the application under the written criteria and send the required adverse-action notice.",
                "hold_for_documents": "Keep the application open and ask for specific missing or unclear documents first.",
            }),
            "weakest_criterion": choice("On which written criterion is this application weakest?", {
                "credit": "Credit: the score or collections record is below or only just at the minimum.",
                "documentation": "Documentation: the evidence behind the figures is missing, unverified or unreliable.",
                "income": "Income: verified income is below or only just at the required multiple of the rent.",
                "meets_all_comfortably": "None: every written criterion is met with room to spare.",
                "rental_history": "Rental history: past tenancies are short, unverified or show problems.",
            }),
            "payment_reliability": score("How reliable does the applicant look as a rent payer on the evidence in the file?", [
                "Doubtful: the evidence points to trouble paying on time.",
                "Uncertain: the file is thin or its signals disagree.",
                "Probably reliable: minor gaps, nothing that points to late payment.",
                "Very reliable: stable, documented income and a clean payment record.",
            ]),
            "decision_timing": score("How soon does a decision on this file have to go out?", [
                "No pressure: nothing is waiting on it and the written deadline is days away.",
                "Within the usual few business days allowed by the written policy.",
                "Within a day or so, before the holding period lapses or another applicant is lost.",
                "Today: the written deadline or the holding deposit runs out now.",
            ]),
            "income_requirement_met": noul(
                "The applicant's documented income meets the property's written income requirement.",
                yes="Income backed by reliable documents reaches the stated multiple of the monthly rent.",
                no="Income falls short of the multiple, or the part that reaches it is not reliably documented."),
        }


# =========================================================================== schema 2
class GuarantorReview(Schema):
    name = "guarantor_review"
    description = "An applicant below the income criteria offers a guarantor; assess whether the guarantee is sufficient."
    surfaces = {"json": 0.5, "guarantor_worksheet": 0.25, "coordinator_email": 0.25}

    DOCS = {"tax_return_and_paystubs": "last year's tax return and two pay stubs", "paystubs_only": "two pay stubs",
            "bank_statement": "one bank statement", "self_declared": "income stated on the guarantor form only"}
    SCOPE = {"full_lease_term": "rent and damages for the full lease term", "capped_six_months": "capped at six months' rent",
             "rent_only": "rent only, not damages or fees"}
    SIGN = {"signed_witnessed": "signed and witnessed", "e_signed": "e-signed; identity check passed",
            "e_signed_no_id": "e-signed; identity check not completed", "not_signed": "not yet signed"}

    def sample_factors(self, rng):
        reg = regime(rng)
        crit = _criteria(rng)
        rent = _rent(rng)
        req, g_req = crit["income_multiple"], crit["guarantor_multiple"]
        a_ratio = round(rng.uniform(1.2, req - 0.1), 2)
        g_ratio = near(rng, g_req, 0.8, lo=1.5)
        g_credit = near(rng, 690, 50, integer=True)
        docs = pick_w(rng, {"tax_return_and_paystubs": 3, "paystubs_only": 3, "bank_statement": 2, "self_declared": 1})
        residence = pick_w(rng, {"same_state": 3, "other_state": 2})
        other_guarantees = pick(rng, [0, 0, 1, 2])
        scope = pick_w(rng, {"full_lease_term": 4, "capped_six_months": 2, "rent_only": 1})
        sign = pick_w(rng, {"signed_witnessed": 2, "e_signed": 3, "e_signed_no_id": 1, "not_signed": 2})
        a_hist = pick(rng, [0, 6, 14, 30])
        a_ref = pick_w(rng, {"positive": 3, "none": 2, "late_payments": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                g_ratio, g_credit, docs, residence, other_guarantees, scope, sign = round(g_req + rng.uniform(1.5, 4), 2), 760, "tax_return_and_paystubs", "same_state", 0, "full_lease_term", "signed_witnessed"
                a_ratio = round(rng.uniform(req - 0.8, req - 0.2), 2)
            else:
                g_ratio, g_credit, docs, scope, sign = round(g_req - rng.uniform(1.5, 2.5), 2), 590, "self_declared", "rent_only", "not_signed"
                a_ratio, a_ref = round(rng.uniform(1.1, 1.6), 2), "late_payments"
        elif reg == "conflicting":
            twist = pick(rng, ["wealthy_but_unsigned", "meets_but_overcommitted", "weak_g_strong_a"])
            if twist == "wealthy_but_unsigned":
                g_ratio, docs, sign = round(g_req + rng.uniform(1, 5), 2), pick(rng, ["bank_statement", "self_declared"]), pick(rng, ["not_signed", "e_signed_no_id"])
            elif twist == "meets_but_overcommitted":
                g_ratio, other_guarantees, scope = round(g_req + rng.uniform(0, 0.6), 2), pick(rng, [2, 3]), "capped_six_months"
            else:
                g_ratio, a_ratio, a_hist, a_ref = round(g_req - rng.uniform(0.3, 1.0), 2), round(req - rng.uniform(0.05, 0.25), 2), 30, "positive"
        if a_ref == "none":
            a_hist = 0
        elif a_hist == 0:
            a_ref = "none"
        a_income, g_income = _r10(rent * a_ratio), _r10(rent * g_ratio)
        applicant, guarantor, coord = _names(rng, 3)
        return {"regime": reg, "criteria": crit, "rent": rent, "applicant_income": a_income, "applicant_ratio": round(a_income / rent, 2),
                "guarantor_income": g_income, "guarantor_ratio": round(g_income / rent, 2), "guarantor_credit": g_credit,
                "guarantor_docs": docs, "guarantor_residence": residence, "guarantor_other_guarantees": other_guarantees,
                "guarantee_scope": scope, "signature": sign, "applicant_history_months": a_hist, "applicant_reference": a_ref,
                "applicant": applicant, "guarantor": guarantor, "coordinator": coord, "address": _address(rng),
                "manager_co": _manager(rng), "app_id": f"APP-{rng.randrange(10000, 99999)}",
                "applicant_credit": maybe(rng, near(rng, crit["min_credit"] + 20, 40, integer=True), 0.15),
                "lease_months": pick(rng, [12, 12, 18, 24])}

    def render(self, f, rng, surface):
        c = f["criteria"]
        ref = {"positive": "positive reference", "none": "no prior tenancy to reference",
               "late_payments": "reference mentions two late payments"}[f["applicant_reference"]]
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["app_id"], "unit": f["address"], "monthly_rent": f["rent"], "lease_months": f["lease_months"],
                                "applicant": {"name": f["applicant"], "monthly_income": f["applicant_income"],
                                              "income_to_rent": f["applicant_ratio"],
                                              "credit_score": f["applicant_credit"] if f["applicant_credit"] is not None else "not on file",
                                              "rental_history_months": f["applicant_history_months"],
                                              "reference": ref}},
                "criteria": {"applicant_income_to_rent": c["income_multiple"], "applicant_min_credit": c["min_credit"],
                             "guarantor_income_to_rent": c["guarantor_multiple"],
                             "guarantor_min_credit": 650, "guarantee_must_cover": "rent and damages for the full term"},
                "guarantor": {"name": f["guarantor"], "monthly_income": f["guarantor_income"], "credit_score": f["guarantor_credit"],
                              "income_evidence": self.DOCS[f["guarantor_docs"]],
                              "residence": "same state as the property" if f["guarantor_residence"] == "same_state" else "another state",
                              "other_leases_guaranteed": f["guarantor_other_guarantees"]},
                "guarantee": {"scope": self.SCOPE[f["guarantee_scope"]], "execution": self.SIGN[f["signature"]]},
            }))
        if surface == "guarantor_worksheet":
            rows = [("Application", f"{f['app_id']} / {f['address']}"), ("Rent", f"{money(f['rent'], rng)} x {f['lease_months']} months"),
                    ("Applicant income", f"{money(f['applicant_income'], rng)}/mo ({_x(f['applicant_ratio'])}; need {c['income_multiple']}x)"),
                    ("Applicant credit", f"{show(f['applicant_credit'], rng)} (min {c['min_credit']})"),
                    ("Applicant history", f"{f['applicant_history_months']} mo, {ref}"),
                    ("Guarantor", f["guarantor"]),
                    ("Guarantor income", f"{money(f['guarantor_income'], rng)}/mo ({_x(f['guarantor_ratio'])}; need {c['guarantor_multiple']}x)"),
                    ("Guarantor credit", f"{f['guarantor_credit']} (min 650)"), ("Evidence", self.DOCS[f["guarantor_docs"]]),
                    ("Guarantor lives in", "same state" if f["guarantor_residence"] == "same_state" else "another state"),
                    ("Other leases guaranteed", f["guarantor_other_guarantees"]),
                    ("Guarantee covers", self.SCOPE[f["guarantee_scope"]]), ("Execution", self.SIGN[f["signature"]])]
            return f"GUARANTOR WORKSHEET -- {f['manager_co']}\n" + kv_lines(rows, rng)
        a_cr = (f"credit {f['applicant_credit']} against our {c['min_credit']} minimum" if f["applicant_credit"] is not None
                else f"no credit score on file (our minimum is {c['min_credit']})")
        s = [f"{f['applicant']} ({f['app_id']}, {f['address']}) comes in at {money(f['applicant_income'], rng)} a month against "
             f"{money(f['rent'], rng)} rent ({_x(f['applicant_ratio'])}), so below our {c['income_multiple']}x, and has put forward "
             f"{f['guarantor']} as guarantor.",
             f"{f['guarantor'].split()[0]} earns {money(f['guarantor_income'], rng)} a month ({self.DOCS[f['guarantor_docs']]}), "
             f"credit {f['guarantor_credit']}, lives in {'the same state' if f['guarantor_residence'] == 'same_state' else 'another state'}.",
             (f"They already guarantee {f['guarantor_other_guarantees']} other lease(s)." if f["guarantor_other_guarantees"] else ""),
             f"The guarantee as drafted covers {self.SCOPE[f['guarantee_scope']]}; it is {self.SIGN[f['signature']]}.",
             f"Applicant history: {f['applicant_history_months']} months, {ref}; {a_cr}."]
        return "\n".join([f"Subject: Guarantor for {f['app_id']}", f"From: {f['coordinator']}", "To: screening team", "",
                          noisy(join_sentences(s), rng, 0.2), "",
                          f"Policy reminder: guarantor {c['guarantor_multiple']}x rent, 650+ credit, full-term rent and damages.",
                          f["coordinator"].split()[0]])

    def questions(self):
        return {
            "guarantor_decision": choice("How should the screening team treat the guarantor offer?", {
                "accept_guarantor": "Accept the guarantee and approve the application on it.",
                "accept_with_extra_security": "Accept it only with an added safeguard, such as a corrected guarantee "
                                              "form or the maximum permitted deposit.",
                "decline_application": "Treat the guarantee as insufficient and decline the application.",
                "request_guarantor_evidence": "Ask for further evidence of the guarantor's income, identity or "
                                              "commitments before deciding.",
            }),
            "guarantee_protection": score("How much protection would this guarantee give the landlord if rent went unpaid?", [
                "Little: it may not be collectable at all.",
                "Partial: it covers some of the exposure, or collecting would be hard.",
                "Good: it covers most of the exposure and should be collectable.",
                "Full: well-documented means, full scope and a properly executed agreement.",
            ]),
            "applicant_shortfall": score("How far short of the written criteria is the applicant on their own?", [
                "Marginal: income-to-rent no more than 0.5x below the required multiple, and no warning sign (a credit "
                "score under the property's minimum, or a reference reporting late payments).",
                "Moderate: income 0.5x to 1.0x below the multiple with no warning sign, or no more than 0.5x below "
                "with one warning sign.",
                "Substantial: income more than 1.0x below the multiple with no warning sign, or 0.5x to 1.0x below "
                "with one warning sign.",
                "Severe: income more than 1.0x below the multiple with a warning sign, or both warning signs present.",
            ]),
            "guarantor_meets_criteria": noul(
                "The guarantor meets the property's written guarantor criteria.",
                yes="Documented guarantor income reaches the required multiple and the credit and coverage terms are met.",
                no="The guarantor falls short on income, credit, evidence or the scope of the guarantee."),
            "guarantee_collectable": noul(
                "The guarantee could realistically be enforced if the tenancy went wrong."),
        }


# =========================================================================== schema 3
class AdverseHistoryReview(Schema):
    name = "adverse_history_review"
    description = "An application flagged for a past eviction case, rental debt or lease breach, for individual assessment."
    surfaces = {"json": 0.5, "review_memo": 0.25, "explanation_letter": 0.25}

    RECORD = {"eviction_filing_dismissed": "eviction case filed by a former landlord; dismissed",
              "eviction_judgement": "eviction judgement for possession and unpaid rent",
              "landlord_debt_collections": "former landlord's unpaid balance placed with a collection agency",
              "lease_violation_notices": "two written lease-violation notices (noise, unauthorised occupant)",
              "early_termination_balance": "early-termination balance owed to a former landlord"}
    # (third person, first person); None = no explanation offered
    EXPLAIN = {"repairs_dispute": ("They withheld rent while the heating was broken for six weeks, and the landlord then filed.",
                                   "I withheld rent while the heating was broken for six weeks, and the landlord then filed."),
               "job_loss_recovered": ("They lost a job that year and fell behind; they have been in steady work since.",
                                      "I lost my job that year and fell behind. I have been in steady work since."),
               "co_tenant_left": ("A co-tenant moved out mid-lease and stopped paying their share.",
                                  "My co-tenant moved out mid-lease and stopped paying their share."),
               "disputes_charges": ("They say they were charged for damage that was there at move-in.",
                                    "I was charged for damage that was already there when I moved in."),
               "neighbour_dispute": ("They had a long-running noise dispute with a downstairs neighbour.",
                                     "I had a long-running noise dispute with the neighbour downstairs."),
               "guest_not_occupant": ("A partner stayed for three weeks, which the landlord treated as an extra occupant.",
                                      "My partner stayed for three weeks and the landlord treated that as an extra occupant."),
               "job_relocation": ("They had to leave early for a job in another city.",
                                  "I had to leave early for a job in another city."),
               "none_given": None,
               "record_not_mine": ("They say the record belongs to someone with a similar name.",
                                   "I believe this record belongs to someone else with a similar name.")}
    FITS = {"eviction_filing_dismissed": ["repairs_dispute", "job_loss_recovered", "co_tenant_left", "none_given", "record_not_mine"],
            "eviction_judgement": ["job_loss_recovered", "co_tenant_left", "repairs_dispute", "none_given", "record_not_mine"],
            "landlord_debt_collections": ["co_tenant_left", "job_loss_recovered", "disputes_charges", "none_given", "record_not_mine"],
            "lease_violation_notices": ["neighbour_dispute", "guest_not_occupant", "none_given"],
            "early_termination_balance": ["job_relocation", "co_tenant_left", "none_given"]}
    SHORT = {"eviction_filing_dismissed": "an eviction case", "eviction_judgement": "an eviction judgement",
             "landlord_debt_collections": "a rental debt in collections", "lease_violation_notices": "two lease-violation notices",
             "early_termination_balance": "an early-termination balance"}
    STATUS = {"paid_in_full": "paid in full", "payment_plan_current": "on a payment plan, current",
              "unpaid": "unpaid", "disputed": "disputed by the applicant"}

    def sample_factors(self, rng):
        reg = regime(rng)
        crit = _criteria(rng)
        rent = _rent(rng)
        look = crit["lookback_years"]
        rec = pick(rng, list(self.RECORD))
        years_ago = near(rng, look, 1.6, lo=0.4, digits=1)
        amount = None if rec in ("eviction_filing_dismissed", "lease_violation_notices") else _r10(lognormal_amount(rng, 2400, 0.6))
        status = None if amount is None else pick_w(rng, {"paid_in_full": 2, "payment_plan_current": 2, "unpaid": 2, "disputed": 1})
        expl = pick_w(rng, {"repairs_dispute": 2, "job_loss_recovered": 3, "co_tenant_left": 2, "none_given": 2, "record_not_mine": 1})
        docs = pick_w(rng, {"court_order": 2, "payoff_letter": 2, "repair_complaints": 1, "none": 3})
        since = pick(rng, [0, 8, 18, 30, 48])
        cur_ref = pick_w(rng, {"positive": 3, "not_contacted": 2, "mixed": 1})
        ratio = round(rng.uniform(crit["income_multiple"] - 0.2, crit["income_multiple"] + 1.2), 2)
        credit = near(rng, crit["min_credit"] + 20, 45, integer=True)
        if reg == "clear":
            if chance(rng, 0.5):
                rec, years_ago, expl, docs, since, cur_ref = "eviction_filing_dismissed", round(look - rng.uniform(1, 2), 1), "repairs_dispute", "court_order", 30, "positive"
                amount, status = None, None
            else:
                rec, years_ago, since, cur_ref, expl, docs = "eviction_judgement", round(rng.uniform(0.5, 1.5), 1), 0, "not_contacted", "none_given", "none"
                amount, status = _r10(lognormal_amount(rng, 4200, 0.4)), "unpaid"
        elif reg == "conflicting":
            twist = pick(rng, ["recent_but_paid", "old_but_unpaid", "claims_not_mine", "dismissed_but_pattern"])
            if twist == "recent_but_paid":
                rec, years_ago, status, docs, since, cur_ref = "eviction_judgement", round(rng.uniform(1, 2.5), 1), "paid_in_full", "payoff_letter", 18, "positive"
                amount = amount or _r10(lognormal_amount(rng, 3000, 0.4))
            elif twist == "old_but_unpaid":
                rec, years_ago, status = "landlord_debt_collections", round(look + rng.uniform(0.2, 1.5), 1), "unpaid"
                amount = amount or _r10(lognormal_amount(rng, 2400, 0.5))
            elif twist == "claims_not_mine":
                expl, docs, cur_ref = "record_not_mine", "none", "not_contacted"
            else:
                rec, amount, status, since, cur_ref = "eviction_filing_dismissed", None, None, 8, "mixed"
        if rec in ("eviction_filing_dismissed", "lease_violation_notices"):
            amount, status = None, None
            if docs == "payoff_letter":
                docs = "none"
        elif amount is None:
            amount = _r10(lognormal_amount(rng, 2400, 0.6))
            status = status or "unpaid"
        if expl not in self.FITS[rec]:
            expl = pick(rng, self.FITS[rec])
        doc_ok = ({"court_order"} if rec in ("eviction_filing_dismissed", "eviction_judgement") else set()) | \
                 ({"payoff_letter"} if status in ("paid_in_full", "payment_plan_current") else set()) | \
                 ({"repair_complaints"} if expl == "repairs_dispute" else set()) | \
                 ({"move_in_photos"} if expl == "disputes_charges" else set()) | {"none"}
        if docs not in doc_ok:
            docs = pick(rng, sorted(doc_ok))
        since = min(since, max(0, int(years_ago * 12) - 4))
        if since == 0:
            cur_ref = "no_tenancy"
        applicant, analyst = _names(rng, 2)
        return {"regime": reg, "criteria": crit, "rent": rent, "record_type": rec, "years_ago": years_ago,
                "amount": amount, "debt_status": status, "explanation": expl, "documents": docs,
                "months_clean_since": since, "current_landlord_ref": cur_ref, "income_ratio": ratio, "credit_score": credit,
                "applicant": applicant, "analyst": analyst, "address": _address(rng), "manager_co": _manager(rng),
                "app_id": f"APP-{rng.randrange(10000, 99999)}", "flag_date": rand_date(rng)}

    DOCS = {"court_order": "a copy of the court's order in the case", "payoff_letter": "a letter from the creditor about the balance",
            "repair_complaints": "emails to the former landlord about repairs", "move_in_photos": "dated move-in photos",
            "none": None}

    def _record_line(self, f):
        amt = f" of {money(f['amount'])}" if f["amount"] else ""
        st = f" ({self.STATUS[f['debt_status']]})" if f["debt_status"] else ""
        return f"{self.RECORD[f['record_type']]}{amt}{st}, about {f['years_ago']} years ago"

    def render(self, f, rng, surface):
        c = f["criteria"]
        rule = (f"Judgements and unpaid landlord debt within {c['lookback_years']} years count against the applicant; "
                "filings without a judgement do not; every record gets an individual assessment.")
        cur = {"positive": "current landlord: on time throughout", "not_contacted": "current landlord not yet contacted",
               "mixed": "current landlord: 'mostly on time', one noise complaint",
               "no_tenancy": "no tenancy in the applicant's own name since"}[f["current_landlord_ref"]]
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["app_id"], "applicant": f["applicant"], "unit": f["address"], "monthly_rent": f["rent"],
                                "income_to_rent": f["income_ratio"], "credit_score": f["credit_score"]},
                "screening_hit": {"record": self.RECORD[f["record_type"]], "years_ago": f["years_ago"], "amount_usd": f["amount"],
                                  "balance_status": None if f["debt_status"] is None else self.STATUS[f["debt_status"]]},
                "applicant_response": {"explanation": (self.EXPLAIN[f["explanation"]] or (None,))[0],
                                       "documents_supplied": self.DOCS[f["documents"]]},
                "since_then": {"months_of_rent_paid_since": f["months_clean_since"], "current_tenancy": cur},
                "policy": {"rule": rule, "income_multiple_required": c["income_multiple"], "min_credit_score": c["min_credit"]},
            }))
        if surface == "review_memo":
            s = [f"Flag on {f['app_id']} ({f['applicant']}, {f['address']}): {self._record_line(f)}.",
                 (f"Applicant's explanation: {self.EXPLAIN[f['explanation']][0]}" if self.EXPLAIN[f["explanation"]] else "No explanation given yet."),
                 (f"Supplied {self.DOCS[f['documents']]}." if self.DOCS[f["documents"]] else "No supporting documents."),
                 f"Since then: {f['months_clean_since']} months of rent paid; {cur}.",
                 f"Otherwise: income {_x(f['income_ratio'])} rent (need {c['income_multiple']}x), credit {f['credit_score']} (min {c['min_credit']})."]
            return "\n".join([f"INDIVIDUAL ASSESSMENT -- {f['manager_co']}", f"Analyst: {f['analyst']}   Flagged: {fmt_date(f['flag_date'], rng)}",
                              "", noisy(join_sentences(s), rng, 0.15), "", "Policy: " + rule])
        expl = (self.EXPLAIN[f["explanation"]] or (None, None))[1]
        body = [f"To whom it may concern,", "",
                noisy(join_sentences([
                    f"I understand my application for {f['address']} was flagged for {self.SHORT[f['record_type']]} from about {f['years_ago']} years ago.",
                    (expl if expl else "I would rather not go into the background."),
                    ({"paid_in_full": "The balance has been paid in full.", "payment_plan_current": "I am paying it off on a plan and I'm up to date.",
                      "unpaid": "I haven't been able to clear it yet.", "disputed": "I dispute that I owe that amount."}.get(f["debt_status"], "")),
                    (f"I've attached {self.DOCS[f['documents']]}." if self.DOCS[f["documents"]] else ""),
                    (f"I have paid rent on time for the last {f['months_clean_since']} months." if f["months_clean_since"] else "")]), rng, 0.2),
                "", "Sincerely,", f["applicant"], "",
                f"--- screening note: income {_x(f['income_ratio'])} rent (need {c['income_multiple']}x); credit {f['credit_score']}; {cur}."]
        return "\n".join(body)

    def questions(self):
        return {
            "history_determination": choice("How should the adverse record be handled in deciding this application?", {
                "approve_with_conditions": "Approve, with a condition the policy allows such as a guarantor or the maximum permitted deposit.",
                "decline_on_history": "Decline because the record shows a rental risk the policy does not accept.",
                "disregard_record": "Set the record aside as outside the policy or not a fair reflection of the applicant, and assess the rest normally.",
                "request_evidence": "Ask for specific evidence about the record before deciding.",
            }),
            "record_weight": score("How much weight does the adverse record deserve now?", [
                "None: it falls outside the policy or does not reflect on the applicant.",
                "Little: it is old, explained or resolved.",
                "Some: it is relevant and only partly explained or resolved.",
                "Heavy: a recent, unresolved rental debt or judgement.",
            ]),
            "recent_record_offsets": score("How strongly does the applicant's more recent rental record offset the adverse item?", [
                "Not at all: there is no verified record since.",
                "Weakly: a short or unverified record since.",
                "Moderately: a verified record of a year or more, with minor blemishes.",
                "Strongly: a long, verified, clean record since.",
            ]),
            "record_counts_under_policy": noul(
                "The record counts against the applicant under the written screening criteria.",
                yes="It is a judgement or unpaid landlord debt inside the lookback period.",
                no="It was dismissed, falls outside the lookback, or was resolved in a way the criteria recognise."),
            "next_verification": choice("What should the screener obtain before finalising the decision?", {
                "court_disposition": "The court record showing how the case ended.",
                "creditor_confirmation": "Confirmation from the creditor of the balance and any payment plan.",
                "current_landlord_reference": "A reference from the applicant's current landlord.",
                "nothing_further": "Nothing more; the file is sufficient to decide.",
            }),
        }


# =========================================================================== schema 4
class IncomeVerification(Schema):
    name = "income_verification"
    description = "An application whose income documents are irregular, self-reported or inconsistent with what was declared."
    surfaces = {"json": 0.45, "verification_log": 0.3, "analyst_notes": 0.25}

    TYPES = {"self_employed": "self-employed (design contractor)", "commission": "base salary plus commission",
             "platform_gig": "delivery and rideshare platforms", "fixed_term_contract": "fixed-term contract, nine months left",
             "new_job_offer": "new job, starts in {d} days"}
    ANOMALY = {"none": None, "ytd_mismatch": "year-to-date totals on the two pay stubs do not add up to the pay periods shown",
               "font_mismatch": "the pay stub's amount fields are in a different font from the rest of the document",
               "free_email_employer": "the employer's HR contact uses a free webmail address",
               "rounded_deposits": "several deposits are round amounts from a personal account",
               "invoices_dont_match": "the invoices supplied do not match the deposits in timing or amount",
               "platform_summary_edited": "the platform earnings summary is a screenshot, not a downloaded statement"}
    ANOMALY_FITS = {"self_employed": ["rounded_deposits", "invoices_dont_match"], "platform_gig": ["platform_summary_edited", "rounded_deposits"],
                    "commission": ["ytd_mismatch", "font_mismatch", "free_email_employer"],
                    "fixed_term_contract": ["ytd_mismatch", "font_mismatch", "free_email_employer"],
                    "new_job_offer": ["free_email_employer", "font_mismatch"]}

    def sample_factors(self, rng):
        reg = regime(rng)
        crit = _criteria(rng)
        rent = _rent(rng)
        req = crit["income_multiple"]
        itype = pick(rng, list(self.TYPES))
        declared_ratio = round(rng.uniform(req, req + 1.3), 2)
        doc_share = pick_w(rng, {1.0: 3, 0.93: 2, 0.85: 2, 0.72: 2})
        months = pick(rng, [2, 3, 3, 6, 6])
        tax = pick_w(rng, {"filed_matches": 2, "filed_lower": 2, "not_provided": 2})
        employer_check = pick_w(rng, {"confirmed": 3, "no_response": 2, "number_mismatch": 1})
        anomaly = pick_w(rng, {"none": 5, "ytd_mismatch": 1, "font_mismatch": 1, "free_email_employer": 1, "rounded_deposits": 1})
        savings = pick(rng, [0, 1, 2, 4, 8])
        pattern = pick_w(rng, {"regular": 3, "irregular": 3, "one_large": 1})
        start_days = pick(rng, [10, 30, 60]) if itype == "new_job_offer" else None
        if itype in ("self_employed", "platform_gig"):
            employer_check = "not_applicable"
        if reg == "clear":
            if chance(rng, 0.5):
                doc_share, tax, anomaly, pattern, months = 1.0, "filed_matches", "none", "regular", 6
                employer_check = "confirmed" if employer_check != "not_applicable" else employer_check
                declared_ratio = round(rng.uniform(req + 0.6, req + 1.5), 2)
            else:
                doc_share, anomaly, tax = 0.72, pick(rng, ["ytd_mismatch", "font_mismatch"]), "not_provided"
                employer_check = "number_mismatch" if employer_check != "not_applicable" else employer_check
        elif reg == "borderline":
            declared_ratio = round(rng.uniform(req + 0.05, req + 0.5), 2)
            doc_share = pick(rng, [0.93, 0.85, 1.0])
        else:
            twist = pick(rng, ["high_but_unverifiable", "verified_but_short", "savings_cover", "one_anomaly_clean_else"])
            if twist == "high_but_unverifiable":
                declared_ratio, tax, months = round(rng.uniform(req + 1, req + 2.5), 2), "not_provided", 2
                employer_check = "no_response" if employer_check != "not_applicable" else employer_check
            elif twist == "verified_but_short":
                doc_share, tax, anomaly, declared_ratio = 0.85, "filed_matches", "none", round(rng.uniform(req, req + 0.25), 2)
            elif twist == "savings_cover":
                doc_share, savings = 0.72, pick(rng, [12, 18])
            else:
                anomaly, doc_share, tax = pick(rng, ["free_email_employer", "rounded_deposits"]), 1.0, "filed_matches"
        if anomaly != "none" and anomaly not in self.ANOMALY_FITS[itype]:
            anomaly = pick(rng, self.ANOMALY_FITS[itype])
        declared = _r10(rent * declared_ratio)
        documented = _r10(declared * doc_share)
        if itype == "new_job_offer":
            # the offer letter states the declared pay; the statements show the CURRENT, older job
            documented = _r10(declared * pick(rng, [0.62, 0.75, 0.9]))
            if tax == "filed_matches":
                tax = "filed_lower"
        applicant, analyst = _names(rng, 2)
        deposits = []
        for i in range(months):
            if pattern == "regular":
                v = documented * rng.uniform(0.94, 1.06)
            elif pattern == "irregular":
                v = documented * rng.uniform(0.45, 1.6)
            else:
                v = documented * (rng.uniform(2.2, 3.0) if i == months - 1 else rng.uniform(0.5, 0.8))
            deposits.append(_r10(v))
        return {"regime": reg, "criteria": crit, "rent": rent, "income_type": itype, "declared_monthly": declared,
                "documented_monthly": documented, "documented_share": doc_share, "statement_months": months,
                "tax_return": tax, "employer_check": employer_check, "document_anomaly": anomaly,
                "savings_months_of_rent": savings, "deposit_pattern": pattern, "monthly_deposits": deposits,
                "start_in_days": start_days, "applicant": applicant, "analyst": analyst, "address": _address(rng),
                "app_id": f"APP-{rng.randrange(10000, 99999)}", "opened": rand_ts(rng, hours=(8, 11))}

    def _type(self, f):
        return self.TYPES[f["income_type"]].format(d=f["start_in_days"])

    def _tax(self, f):
        return {"filed_matches": "last year's return supports roughly the declared level",
                "filed_lower": "last year's return shows about two-thirds of the declared level",
                "not_provided": None}[f["tax_return"]]

    def _offer(self, f):
        return (f"signed offer letter states {money(f['declared_monthly'])}/mo; statements show the current job"
                if f["income_type"] == "new_job_offer" else None)

    def _employer(self, f):
        return {"confirmed": "employer confirmed role and pay by phone", "no_response": "employer did not answer two calls or an email",
                "number_mismatch": "the phone number on the pay stub does not match the employer's listed number",
                "not_applicable": None}[f["employer_check"]]

    def render(self, f, rng, surface):
        c = f["criteria"]
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["app_id"], "applicant": f["applicant"], "unit": f["address"], "monthly_rent": f["rent"],
                                "required_income_to_rent": c["income_multiple"]},
                "declared": {"monthly_income": f["declared_monthly"], "source": self._type(f)},
                "documents": {"bank_statement_months": f["statement_months"], "monthly_deposits": f["monthly_deposits"],
                              "offer_letter": self._offer(f),
                              "tax_return": self._tax(f) or "not provided", "employer_check": self._employer(f),
                              "anomaly_noted": self.ANOMALY[f["document_anomaly"]]},
                "reserves": {"savings_in_months_of_rent": f["savings_months_of_rent"]},
            }))
        if surface == "verification_log":
            t = f["opened"]
            ev = [f"file {f['app_id']} opened; applicant {f['applicant']}; unit {f['address']}; rent {money(f['rent'], rng)}",
                  f"declared income {money(f['declared_monthly'], rng)}/mo ({self._type(f)})",
                  f"bank statements received: {f['statement_months']} month(s); deposits "
                  + ", ".join(money(d, rng, style='sym') for d in f["monthly_deposits"])]
            if self._offer(f):
                ev.append(self._offer(f))
            if self._tax(f):
                ev.append(f"tax return reviewed: {self._tax(f)}")
            else:
                ev.append("tax return requested; not received")
            if self._employer(f):
                ev.append(f"employment check: {self._employer(f)}")
            if self.ANOMALY[f["document_anomaly"]]:
                ev.append(f"NOTE: {self.ANOMALY[f['document_anomaly']]}")
            ev.append(f"reserves: savings approx. {f['savings_months_of_rent']} month(s) of rent")
            out = []
            for e in ev:
                t += timedelta(minutes=rng.randint(3, 55))
                out.append(f"{fmt_ts(t, style='log')}  {e}")
            return f"INCOME VERIFICATION LOG (policy: {c['income_multiple']}x rent, verified)\n" + "\n".join(out)
        rows = [[f"month {i + 1}", money(d, rng, style="sym")] for i, d in enumerate(f["monthly_deposits"])]
        s = [f"{f['applicant']} declared {money(f['declared_monthly'], rng)} a month from {self._type(f)}.",
             (f"The {self._offer(f)}." if self._offer(f) else ""),
             (f"Tax return: {self._tax(f)}." if self._tax(f) else "No tax return yet."),
             (f"{self._employer(f)[0].upper() + self._employer(f)[1:]}." if self._employer(f) else ""),
             (f"One thing bothers me: {self.ANOMALY[f['document_anomaly']]}." if self.ANOMALY[f["document_anomaly"]] else ""),
             f"Savings would cover about {f['savings_months_of_rent']} month(s) of rent." if f["savings_months_of_rent"] else "No meaningful savings shown."]
        return "\n".join([f"Analyst notes -- {f['app_id']} -- {f['analyst']}", f"Rent {money(f['rent'], rng)}; need {c['income_multiple']}x verified.",
                          "", "Deposits from statements:", table(["period", "deposits"], rows, rng), "",
                          noisy(join_sentences(s), rng, 0.2)])

    def questions(self):
        return {
            "verification_outcome": choice("What should the analyst do with the applicant's income evidence?", {
                "accept_declared": "Accept the declared income as verified and assess the application on it.",
                "assess_on_documented": "Assess the application on the lower figure the documents actually support.",
                "direct_verification": "Verify directly with the employer, accountant or platform before going further.",
                "refer_for_misrepresentation": "Refer the file for review because the documents may have been altered or misstated.",
                "request_more_records": "Ask the applicant for specific additional records before assessing income.",
            }),
            "income_stability": score("How stable is the applicant's income likely to be over the lease?", [
                "Unstable: large swings, gaps or a source that may stop.",
                "Variable: it moves month to month but has held up overall.",
                "Mostly steady: small variation around a reliable level.",
                "Steady: regular, documented income from an established source.",
            ]),
            "document_reliability": score("How far can the income documents be relied on?", [
                "Not at all: signs of alteration or claims that cannot be checked.",
                "Partly: gaps or inconsistencies that need explaining.",
                "Largely: minor gaps, nothing suspicious.",
                "Fully: consistent documents confirmed from an independent source.",
            ]),
            "documented_income_meets_multiple": noul(
                "The income the documents actually support reaches the rent multiple in the written criteria.",
                yes="The documented, recurring monthly income is at or above the required multiple of the rent.",
                no="Only the declared figure reaches the multiple, or even the documents fall short of it."),
            "discrepancy_is_innocent": noul(
                "Any gap between the declared and documented income looks like honest estimating rather than misstatement."),
        }


# =========================================================================== schema 5
class JointApplication(Schema):
    name = "joint_application"
    description = "Several adult co-applicants applying together for one unit, assessed as a group under written criteria."
    surfaces = {"json": 0.5, "applicant_table": 0.25, "agent_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        crit = _criteria(rng)
        rent = _rent(rng, 2600, 0.3)
        req, min_cr = crit["income_multiple"], crit["min_credit"]
        n = pick(rng, [2, 2, 3, 3, 4])
        combined_ratio = near(rng, req, 0.35, lo=1.8)
        weak = rng.randrange(n)
        weak_issue = pick_w(rng, {"low_credit": 3, "no_income_docs": 2, "late_payment_history": 2, "leaving_early": 2, "none": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                combined_ratio, weak_issue = round(req + rng.uniform(0.6, 1.4), 2), "none"
            else:
                combined_ratio, weak_issue = round(req - rng.uniform(0.6, 1.0), 2), pick(rng, ["low_credit", "late_payment_history"])
        elif reg == "conflicting":
            twist = pick(rng, ["rich_group_one_bad", "short_group_all_clean", "short_without_leaver"])
            if twist == "rich_group_one_bad":
                combined_ratio, weak_issue = round(req + rng.uniform(0.5, 1.2), 2), pick(rng, ["low_credit", "late_payment_history"])
            elif twist == "short_group_all_clean":
                combined_ratio, weak_issue = round(req - rng.uniform(0.1, 0.35), 2), "none"
            else:
                combined_ratio, weak_issue = round(req + rng.uniform(0.0, 0.3), 2), "leaving_early"
        shares = [rng.uniform(0.5, 1.5) for _ in range(n)]
        if weak_issue == "no_income_docs":
            shares[weak] = rng.uniform(0.3, 0.8)
        tot = sum(shares)
        incomes = [_r10(rent * combined_ratio * s / tot) for s in shares]
        names = _names(rng, n + 1)
        members = []
        for i in range(n):
            m = {"name": names[i], "monthly_income": incomes[i], "income_evidence": "pay stubs",
                 "credit_score": min_cr + rng.randint(15, 140), "rental_history": pick(rng, ["2 yrs, positive ref", "3 yrs, positive ref",
                                                                                             "18 mo, positive ref", "4 yrs, positive ref"]),
                 "plans_full_term": True}
            if i == weak:
                if weak_issue == "low_credit":
                    m["credit_score"] = near(rng, min_cr - 15, 25, integer=True)
                elif weak_issue == "no_income_docs":
                    m["income_evidence"] = pick(rng, ["stated only", "one bank statement"])
                elif weak_issue == "late_payment_history":
                    m["rental_history"] = pick(rng, ["2 yrs, 3 late payments", "1 yr, landlord reports arrears repaid"])
                elif weak_issue == "leaving_early":
                    m["plans_full_term"] = False
            members.append(m)
        return {"regime": reg, "criteria": crit, "rent": rent, "n_applicants": n, "members": members,
                "combined_income": sum(incomes), "combined_ratio": round(sum(incomes) / rent, 2), "weak_index": weak,
                "weak_issue": weak_issue, "joint_liability_signed": pick_w(rng, {True: 3, False: 1}),
                "agent": names[n], "address": _address(rng), "manager_co": _manager(rng),
                "app_id": f"GRP-{rng.randrange(1000, 9999)}", "lease_months": 12}

    def _member_line(self, m):
        extra = "" if m["plans_full_term"] else "; intends to move out after about six months"
        return (f"{m['name']}: {money(m['monthly_income'])}/mo ({m['income_evidence']}), credit {m['credit_score']}, "
                f"{m['rental_history']}{extra}")

    def render(self, f, rng, surface):
        c = f["criteria"]
        policy = (f"Combined verified income at least {c['income_multiple']}x rent; each co-applicant needs {c['min_credit']}+ credit "
                  f"and a rental record without repeated late payments, or a guarantor; every adult occupant signs and is "
                  f"jointly liable for the full rent.")
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["app_id"], "unit": f["address"], "monthly_rent": f["rent"],
                                "lease_months": f["lease_months"], "joint_and_several_clause_accepted": f["joint_liability_signed"]},
                "co_applicants": [{"name": m["name"], "monthly_income": m["monthly_income"], "income_evidence": m["income_evidence"],
                                   "credit_score": m["credit_score"], "rental_history": m["rental_history"],
                                   "note": None if m["plans_full_term"] else "intends to move out after about six months"}
                                  for m in f["members"]],
                "policy": policy,
            }))
        if surface == "applicant_table":
            rows = [[m["name"], money(m["monthly_income"], rng, style="sym"), m["income_evidence"], m["credit_score"], m["rental_history"],
                     "full term" if m["plans_full_term"] else "~6 months"] for m in f["members"]]
            return "\n".join([f"JOINT APPLICATION {f['app_id']} -- {f['address']} ({money(f['rent'], rng)}/mo, {f['lease_months']}-month lease)",
                              "", table(["applicant", "income/mo", "evidence", "credit", "history", "staying"], rows, rng), "",
                              f"Combined: {money(f['combined_income'], rng)}/mo. Joint-liability clause accepted: "
                              f"{'yes' if f['joint_liability_signed'] else 'not yet'}.", f"Policy: {policy}"])
        s = [f"Group application {f['app_id']} for {f['address']}, {f['n_applicants']} applicants, rent {money(f['rent'], rng)}."]
        s += [self._member_line(m) + "." for m in f["members"]]
        s.append(("Both have" if f["n_applicants"] == 2 else "All of them have") + " accepted the joint-liability clause."
                 if f["joint_liability_signed"] else "Not everyone has accepted the joint-liability clause yet.")
        s.append(pick(rng, ["Nice group, I'd like to get this done.", "Let me know what you need from them.", ""]))
        return "\n".join([f"Subject: {f['app_id']} group application", f"From: {f['agent']}", "To: screening", "",
                          noisy("\n".join(x for x in s if x), rng, 0.15), "", f"(Policy: {policy})"])

    def questions(self):
        return {
            "group_decision": choice("How should the screening team decide on the group application?", {
                "approve_group": "Approve all co-applicants on the standard joint lease.",
                "approve_with_guarantor": "Approve on condition that a guarantor backs the co-applicant who falls short.",
                "decline_group": "Decline the application as a whole under the written criteria.",
                "restructure_group": "Ask the group to restructure first, for example replacing or adding a co-applicant.",
            }),
            "combined_affordability": score("How comfortably does the group's verified income cover the rent?", [
                "Not covered: well below the required multiple.",
                "Just short: close to the multiple but under it.",
                "Covered: at or a little above the multiple.",
                "Comfortable: well above the multiple.",
            ]),
            "weakest_member_drag": score("How much does the weakest co-applicant weaken the application?", [
                "Not at all: every co-applicant is sound.",
                "Slightly: a minor gap the others easily carry.",
                "Noticeably: a real gap on credit, history, documents or commitment.",
                "Seriously: the gap puts the group's ability to pay the full term in doubt.",
            ]),
            "each_member_qualifies": noul(
                "Every co-applicant individually meets the credit and rental-history criteria.",
                yes="Each co-applicant has at least the minimum credit score and an acceptable rental record.",
                no="At least one co-applicant falls below the credit minimum or has a problem rental record."),
            "priority_issue": choice("Which issue most needs resolving before the group can be approved?", {
                "credit_or_history": "A co-applicant's credit or rental history.",
                "income_evidence": "Missing or weak income evidence for a co-applicant.",
                "nothing_outstanding": "Nothing; the application can be decided as it stands.",
                "term_commitment": "Whether everyone will stay and pay for the full lease term.",
            }),
        }


WORKFLOW = Workflow(
    name="tenant_screening",
    title="Tenant screening",
    description="Whether a rental application meets a property's written screening criteria, and on what terms.",
    domain="residential property management",
    schemas=[RentalApplication(), GuarantorReview(), AdverseHistoryReview(), IncomeVerification(), JointApplication()],
    notes="Criteria-based only: income multiple, credit and rental-debt history, references, document reliability and "
          "guarantees. No protected attribute, household composition or income source is a factor. No customer "
          "threads, no payment alerts, no bills.",
)
