"""Mortgage underwriting conditions: approve, condition, suspend, counter or decline a residential mortgage file.

Each case is one underwriting decision on a fictional lender's residential mortgage: the initial decision on
ratios, credit and reserves; the calculation of variable or self-employed income; the review of an appraisal;
the prior-to-close conditions check as closing nears; or the sourcing of a large deposit. The state carries what
an underwriter works from: the loan summary, worksheets, the conditions log, bank-statement lines, the
appraiser's grid, notes from the loan officer or processor, and the lender's program limits.

Fair-lending guard rails:
- Borrowers appear only as a loan number and "B1" / "B2". No name, race, religion, national origin, sex,
  marital or familial status, age or disability is rendered, sampled or used.
- Income types are employment, self-employment and rental income. Public-assistance, alimony and child-support
  income are left out entirely, so no case can discount them.
- A gift donor's relationship appears only because gift documentation requires it; it never drives an answer.

Structure follows it_change_approval.py: each schema draws a `regime` first and lets it steer the factors.
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
    lognormal_amount,
    maybe,
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

# (purpose, occupancy) -> program maximum LTV %, the lender's fictional "Core 30" matrix
MAX_LTV = {("purchase", "primary"): 95, ("rate_term_refi", "primary"): 90, ("cash_out_refi", "primary"): 80,
           ("purchase", "second_home"): 90, ("rate_term_refi", "second_home"): 85,
           ("cash_out_refi", "second_home"): 75, ("purchase", "investment"): 80,
           ("rate_term_refi", "investment"): 75, ("cash_out_refi", "investment"): 70}
PROGRAM = {"max_dti": 45, "max_dti_with_exception": 50, "min_score": 640, "min_score_above_45": 680,
           "reserves_months": 2, "reserves_months_above_45_or_investment": 6}
PROGRAM_TEXT = ("Core 30: max DTI 45%, or up to 50% with automated Accept, score >= 680 and >= 6 months reserves; "
                "min score 640; reserves 2 months PITIA (6 months if DTI > 45% or investment property)")
PURPOSE_TEXT = {"purchase": "purchase", "rate_term_refi": "rate/term refinance", "cash_out_refi": "cash-out refinance"}
OCC_TEXT = {"primary": "primary residence", "second_home": "second home", "investment": "investment property"}


def _loan_no(rng) -> str:
    return ident(rng, "LN", 8, "")


def _round_k(x: float) -> float:
    return round(x, -3)


def _r50(x: float) -> float:
    return float(round(x / 50) * 50)


# =========================================================================== schema 1
class InitialUnderwriteDecision(Schema):
    name = "initial_underwrite_decision"
    description = "A mortgage file at initial underwriting: ratios, credit, reserves and automated findings against program limits."
    surfaces = {"json": 0.5, "underwriter_worksheet": 0.25, "loan_officer_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        purpose = pick_w(rng, {"purchase": 5, "rate_term_refi": 2, "cash_out_refi": 2})
        occ = pick_w(rng, {"primary": 6, "second_home": 1, "investment": 2})
        max_ltv = MAX_LTV[(purpose, occ)]
        value = _round_k(lognormal_amount(rng, 420000, 0.45))
        ltv = near(rng, max_ltv - 4, 7, lo=40, hi=max_ltv + 3, digits=1)
        score_ = near(rng, 705, 50, lo=590, hi=820, integer=True)
        dti = near(rng, 41, 8, lo=18, hi=56, digits=1)
        reserves = pick(rng, [0.5, 1.5, 2.5, 4, 7, 14])
        aus = pick_w(rng, {"accept": 5, "refer": 3, "refer_with_caution": 1})
        derog = pick_w(rng, {"none": 6, "late_payment_12m": 2, "collection_account": 1, "bankruptcy_discharged_4y": 1})
        tenure = pick(rng, [0.6, 1.5, 3, 7, 15])
        shock = pick(rng, [5, 25, 60, 120, 170]) if purpose == "purchase" else None
        first_time = purpose == "purchase" and occ == "primary" and chance(rng, 0.5)
        co_borrower = chance(rng, 0.45)
        if reg == "clear":
            if chance(rng, 0.5):
                ltv, dti, score_, reserves, aus, derog = max_ltv - 12, near(rng, 31, 4, digits=1), 772, 9, "accept", "none"
                tenure, shock = pick(rng, [7, 15]), (15 if purpose == "purchase" else None)
            else:
                ltv, dti, score_, reserves, aus, derog = max_ltv + 2, near(rng, 53, 2, digits=1), 612, 0.5, \
                    "refer_with_caution", "collection_account"
        elif reg == "borderline":
            dti = near(rng, 45.5, 2.2, digits=1)
            reserves = pick(rng, [4, 5.5, 6, 6.5])
            score_ = near(rng, 680, 10, integer=True)
            ltv = near(rng, max_ltv - 0.5, 1.5, lo=40, digits=1)
            aus = pick(rng, ["accept", "accept", "refer"])
        else:
            if chance(rng, 0.5):
                # deep reserves, excellent credit and a long job, but DTI over the base limit and a Refer finding
                reserves, score_, tenure, derog = pick(rng, [14, 22]), near(rng, 790, 15, integer=True), 15, "none"
                dti, aus = near(rng, 48.5, 1.2, digits=1), "refer"
            else:
                # comfortable ratios, but thin reserves, a recent late payment and a big jump in housing cost
                dti, ltv, aus = near(rng, 34, 3, digits=1), max_ltv - 8, "accept"
                reserves, derog = 0.5, pick(rng, ["late_payment_12m", "bankruptcy_discharged_4y"])
                if purpose == "purchase":
                    shock = 170
        value = min(value, 950000.0)
        loan = _round_k(value * ltv / 100)
        # PITIA follows the loan (about 6.75% over 30 years, plus taxes and insurance); income follows from the DTI
        housing = round(loan * 0.00649 + value * 0.0019)
        other = pick(rng, [180, 420, 760, 1150, 1700])
        income = round((housing + other) / (dti / 100), -1)
        total_debt = housing + other
        return {"regime": reg, "loan_number": _loan_no(rng), "purpose": purpose, "occupancy": occ,
                "program_max_ltv": max_ltv, "value": value, "loan_amount": loan, "ltv": round(loan / value * 100, 1),
                "credit_score": score_, "dti": dti, "monthly_income": income, "housing_payment": housing,
                "other_debts": total_debt - housing, "reserves_months": reserves, "aus_finding": aus,
                "derogatory": derog, "job_tenure_years": tenure, "payment_shock_pct": shock,
                "first_time_buyer": first_time, "co_borrower": co_borrower,
                "property_type": pick(rng, ["single-family", "single-family", "condo", "townhouse", "2-unit"]),
                "lo": person(rng), "underwriter": person(rng), "date": rand_date(rng),
                "lo_tone": pick(rng, ["confident", "pushing", "neutral"])}

    DEROG = {"none": "no derogatory items", "late_payment_12m": "one 30-day late payment on an auto loan 7 months ago",
             "collection_account": "open medical collection of $2,340 and a 60-day late 10 months ago",
             "bankruptcy_discharged_4y": "Chapter 7 discharged 4 years ago; clean since"}
    AUS = {"accept": "Accept/Eligible", "refer": "Refer", "refer_with_caution": "Refer with Caution"}

    def _lo_note(self, f, rng):
        s = {"confident": "Solid borrowers, stable job, should be straightforward.",
             "pushing": ("Realtor is pressing for a quick approval, please look at this today if you can."
                         if f["purpose"] == "purchase" else
                         "Borrowers want this locked and approved before rates move, please look today if you can."),
             "neutral": "Let me know what you need."}[f["lo_tone"]]
        extra = []
        if f["job_tenure_years"] >= 7:
            extra.append(f"B1 has been with the same employer {f['job_tenure_years']:g} years.")
        elif f["job_tenure_years"] < 1:
            extra.append("B1 started a new job about 7 months ago, same line of work.")
        if f["first_time_buyer"]:
            extra.append("First-time buyers.")
        return noisy(join_sentences([s] + extra), rng)

    def render(self, f, rng, surface):
        pitia = money(f["housing_payment"], rng)
        if surface == "json":
            return dumps(prune({
                "loan": {"number": f["loan_number"], "purpose": PURPOSE_TEXT[f["purpose"]],
                         "occupancy": OCC_TEXT[f["occupancy"]], "property": f["property_type"],
                         "amount": f["loan_amount"], "value_or_price": f["value"],
                         "borrowers": "B1 + B2" if f["co_borrower"] else "B1"},
                "ratios": {"ltv_pct": f["ltv"], "dti_pct": f["dti"], "proposed_housing_payment": f["housing_payment"],
                           "other_monthly_debts": f["other_debts"], "qualifying_monthly_income": f["monthly_income"],
                           "payment_shock_pct": f["payment_shock_pct"]},
                "credit": {"representative_score": f["credit_score"], "history": self.DEROG[f["derogatory"]]},
                "assets": {"reserves_months_pitia": f["reserves_months"]},
                "automated_findings": self.AUS[f["aus_finding"]],
                "profile": {"job_tenure_years_b1": f["job_tenure_years"], "first_time_buyer": f["first_time_buyer"]},
                "program_limits": {"max_ltv_pct": f["program_max_ltv"], "rules": PROGRAM_TEXT},
                "loan_officer_note": self._lo_note(f, rng),
            }))
        if surface == "underwriter_worksheet":
            rows = [("Loan", f"{f['loan_number']} -- {PURPOSE_TEXT[f['purpose']]}, {OCC_TEXT[f['occupancy']]}, "
                             f"{f['property_type']}"),
                    ("Loan amount / value", f"{money(f['loan_amount'], rng)} / {money(f['value'], rng)}"),
                    ("LTV", f"{f['ltv']}% (program max {f['program_max_ltv']}%)"),
                    ("Qualifying income (monthly)", money(f["monthly_income"], rng)),
                    ("Proposed PITIA", pitia), ("Other debts", money(f["other_debts"], rng)),
                    ("DTI", f"{f['dti']}%"), ("Score", f["credit_score"]),
                    ("Credit history", self.DEROG[f["derogatory"]]),
                    ("Reserves", f"{f['reserves_months']:g} months"), ("AUS", self.AUS[f["aus_finding"]]),
                    ("Payment shock", show(f["payment_shock_pct"], rng, lambda v: f"{v}%")
                     if f["purpose"] == "purchase" else None),
                    ("Employment (B1)", f"{f['job_tenure_years']:g} yrs current employer")]
            return (f"UNDERWRITER WORKSHEET -- {f['underwriter']}, {fmt_date(f['date'], rng)}\n"
                    + kv_lines(rows, rng) + f"\n\nProgram: {PROGRAM_TEXT}\n\nLO comments: {self._lo_note(f, rng)}")
        lines = [f"Subject: {f['loan_number']} submitted to UW -- {PURPOSE_TEXT[f['purpose']]}", f"From: {f['lo']}",
                 f"To: {f['underwriter']}", "", "Hi,", "", self._lo_note(f, rng), "",
                 bullets([f"{money(f['loan_amount'], rng)} on {money(f['value'], rng)} ({f['ltv']}% LTV), "
                          f"{OCC_TEXT[f['occupancy']]}, {f['property_type']}",
                          f"DTI {f['dti']}% (income {money(f['monthly_income'], rng)}/mo, PITIA {pitia}, other debts "
                          f"{money(f['other_debts'], rng)})",
                          f"Score {f['credit_score']}; {self.DEROG[f['derogatory']]}",
                          f"Reserves {f['reserves_months']:g} months; AUS {self.AUS[f['aus_finding']]}",
                          f"B1 has {f['job_tenure_years']:g} years with the current employer"]
                         + ([f"Payment shock {f['payment_shock_pct']}% over current rent"]
                            if f["payment_shock_pct"] is not None else []), rng),
                 "", f"Program max LTV for this scenario is {f['program_max_ltv']}%.", "", "Thanks,",
                 f["lo"].split()[0]]
        return "\n".join(lines)

    def questions(self):
        return {
            "underwriting_decision": choice("What should the underwriter decide on this file?", {
                "approve_standard_conditions": "Approve with the usual conditions only.",
                "approve_added_conditions": "Approve, adding conditions aimed at the file's specific weak points.",
                "counteroffer_terms": "Offer different terms the file would support, such as a smaller loan or a "
                                      "different program.",
                "decline_file": "Decline; the file cannot be made to fit the program.",
                "suspend_for_documents": "Suspend the decision until specific missing information arrives.",
            }),
            "signoff_level": choice("Who needs to sign off on this decision?", {
                "credit_policy_exception": "Credit risk management, through a formal policy exception.",
                "not_approvable": "No one; the file cannot be approved in any form.",
                "senior_underwriter": "A senior underwriter as second signature, because the file uses an "
                                      "allowed higher limit or layers several risks.",
                "underwriter_authority": "The underwriter alone, within normal delegated authority.",
            }),
            "layered_risk": score("How much risk is layered in this file once ratios, credit and reserves are read "
                                  "together?", [
                "Low: every measure sits comfortably inside the limits.",
                "Moderate: one measure is stretched but the rest are sound.",
                "High: several measures are stretched at once.",
                "Excessive: the weaknesses compound beyond what the program accepts.",
            ]),
            "compensating_strength": score("How strong are the compensating factors that could offset the file's "
                                           "weak points?", [
                "None: reserves under 6 months, a score under 740 and under 7 years with the current employer.",
                "Thin: exactly one of reserves of 6+ months, a score of 740+, or 7+ years with the current employer.",
                "Solid: two of those three, or all three with reserves under 12 months.",
                "Strong: all three, with reserves of 12 months or more.",
            ]),
            "ratios_within_limits": noul(
                "The file's debt-to-income and loan-to-value ratios are within the program's limits for this "
                "borrower's profile.",
                yes="DTI is at or under 45%, or under 50% with every condition for the higher limit met, and LTV is "
                    "at or under the program maximum.",
                no="DTI or LTV is over its limit, or the higher DTI limit is used without meeting all of its "
                   "conditions."),
        }


# =========================================================================== schema 2
INCOME_TYPES = {"hourly_overtime": "hourly with overtime", "commission": "salaried plus commission",
                "bonus": "salaried plus annual bonus", "self_employed": "self-employed (sole owner of an LLC)"}


class IncomeCalculationReview(Schema):
    name = "income_calculation_review"
    description = "A processor's qualifying-income figure for variable or self-employed income, checked against the documents."
    surfaces = {"json": 0.5, "income_worksheet": 0.25, "processor_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        itype = pick(rng, list(INCOME_TYPES))
        base = _round_k(lognormal_amount(rng, 5600, 0.35)) if itype != "self_employed" else 0
        var_y1 = _round_k(lognormal_amount(rng, 26000 if itype != "self_employed" else 98000, 0.45))
        yoy = pick(rng, [-32, -18, -8, 4, 12, 25])
        history = pick(rng, [11, 16, 22, 26, 38, 60])
        ytd_trend = pick_w(rng, {"on_pace": 3, "below": 2, "above": 2})
        docs = pick_w(rng, {"complete": 4, "missing_prior_year": 1, "ytd_unverified": 2})
        gap = pick(rng, [0, 0, 0, 2, 7])
        letter = pick_w(rng, {"continues": 3, "reduced": 1, None: 3})
        debts = pick(rng, [240, 480, 820, 1250])
        if reg == "clear":
            if chance(rng, 0.5):
                yoy, history, ytd_trend, docs, gap, letter = pick(rng, [4, 12]), pick(rng, [38, 60]), "on_pace", \
                    "complete", 0, "continues"
            else:
                yoy, history, ytd_trend, docs, letter = -32, 11, "below", "missing_prior_year", "reduced"
        elif reg == "borderline":
            history = pick(rng, [22, 26])
            yoy = pick(rng, [-18, -8, 12])
            ytd_trend = pick(rng, ["on_pace", "below"])
        else:
            if chance(rng, 0.5):
                # long history, rising income, but this year's pace has collapsed and the letter says hours are cut
                history, yoy, ytd_trend, letter = 60, 25, "below", "reduced"
            else:
                # a falling year on paper, but the YTD pace and employer letter point firmly up
                history, yoy, ytd_trend, letter, docs = 38, -18, "above", "continues", "complete"
        var_y2 = _round_k(var_y1 * (1 + yoy / 100))
        ytd_factor = {"on_pace": 1.0, "below": rng.uniform(0.55, 0.8), "above": rng.uniform(1.15, 1.35)}[ytd_trend]
        months_ytd = pick(rng, [3, 5, 8, 10])
        ytd = _round_k(var_y2 * ytd_factor * months_ytd / 12)
        processor_basis = pick_w(rng, {"most_recent_year": 3, "ytd_annualised": 2, "two_year_average": 2})
        proc_var = {"most_recent_year": var_y2, "ytd_annualised": ytd * 12 / months_ytd,
                    "two_year_average": (var_y1 + var_y2) / 2}[processor_basis] / 12
        processor_income = round(base + proc_var)
        # the housing payment is sized so the processor's income lands near the 45% DTI limit a fair share of the time
        target_dti = near(rng, 43, 5, lo=28, hi=52, digits=1) if reg != "clear" else pick(rng, [33.0, 51.0])
        housing = max(1400, round(processor_income * target_dti / 100 - debts))
        return {"regime": reg, "loan_number": _loan_no(rng), "income_type": itype, "base_monthly": base,
                "variable_two_years_ago": var_y1, "variable_last_year": var_y2, "yoy_change_pct": yoy,
                "ytd_variable": ytd, "months_ytd": months_ytd, "variable_history_months": history,
                "ytd_trend": ytd_trend, "documents": docs, "employment_gap_months": gap,
                "employer_letter": letter, "other_debts": debts, "housing_payment": housing,
                "processor_basis": processor_basis, "processor_income": processor_income,
                "processor": person(rng), "underwriter": person(rng), "date": rand_date(rng),
                "industry_note": pick(rng, [None, None, "Employer's sector had layoffs this spring.",
                                            "Business moved to a larger unit last year."])}

    DOCS = {"hourly_overtime": {"complete": "2 years W-2s, 30 days pay stubs, written VOE with overtime breakdown",
                                "missing_prior_year": "last year's W-2 and pay stubs; prior-year W-2 not provided",
                                "ytd_unverified": "2 years W-2s; YTD pay stub only, no written VOE"},
            "commission": {"complete": "2 years W-2s, YTD pay stub, written VOE with commission history",
                           "missing_prior_year": "last year's W-2 only",
                           "ytd_unverified": "2 years W-2s; YTD commission per borrower's own spreadsheet"},
            "bonus": {"complete": "2 years W-2s, bonus letter, YTD pay stub",
                      "missing_prior_year": "last year's W-2 and bonus letter only",
                      "ytd_unverified": "2 years W-2s; bonus amount per borrower, no letter"},
            "self_employed": {"complete": "2 years personal and business returns, YTD P&L, 3 months business bank "
                                          "statements",
                              "missing_prior_year": "last year's returns and YTD P&L; prior year not filed with file",
                              "ytd_unverified": "2 years returns; YTD P&L prepared by borrower, unsigned"}}
    LETTER = {"continues": "employer / accountant letter: this income is expected to continue",
              "reduced": "employer letter: overtime/commission opportunities reduced from next quarter",
              None: None}
    BASIS = {"most_recent_year": "most recent year", "ytd_annualised": "YTD annualised",
             "two_year_average": "two-year average"}

    @staticmethod
    def _alternatives(f):
        """Qualifying income and DTI on each of the three usual bases, plus this year's pace against last year."""
        b, y1, y2 = f["base_monthly"], f["variable_two_years_ago"], f["variable_last_year"]
        inc = {"most recent year": round(b + y2 / 12), "YTD annualised": round(b + f["ytd_variable"] / f["months_ytd"]),
               "two-year average": round(b + (y1 + y2) / 24)}
        debt = f["housing_payment"] + f["other_debts"]
        alts = {k: (v, round(debt / v * 100, 1)) for k, v in inc.items()}
        pace = round(f["ytd_variable"] * 12 / f["months_ytd"] / y2 * 100 - 100)
        return alts, pace

    def _alt_text(self, f, rng):
        alts, pace = self._alternatives(f)
        return ("; ".join(f"{k}: {money(v, rng)}/mo, DTI {d}%" for k, (v, d) in alts.items())
                + f". This year's pace is {pace:+d}% against last year")

    def _dti(self, f, income):
        return round((f["housing_payment"] + f["other_debts"]) / income * 100, 1) if income else None

    def render(self, f, rng, surface):
        label = "net business income" if f["income_type"] == "self_employed" else "variable income"
        if surface == "json":
            return dumps(prune({
                "loan": f["loan_number"],
                "income": {"type": INCOME_TYPES[f["income_type"]], "base_monthly": f["base_monthly"] or None,
                           f"{label.replace(' ', '_')}_two_years_ago": f["variable_two_years_ago"],
                           f"{label.replace(' ', '_')}_last_year": f["variable_last_year"],
                           f"{label.replace(' ', '_')}_ytd": f"{money(f['ytd_variable'])} over {f['months_ytd']} months",
                           "history_months": f["variable_history_months"],
                           "employment_gap_last_2y_months": f["employment_gap_months"] or None},
                "documents": self.DOCS[f["income_type"]][f["documents"]],
                "letters": self.LETTER[f["employer_letter"]], "context": f["industry_note"],
                "processor_calculation": {"basis": self.BASIS[f["processor_basis"]],
                                          "qualifying_monthly_income": f["processor_income"],
                                          "dti_pct_at_this_income": self._dti(f, f["processor_income"])},
                "obligations": {"proposed_housing_payment": f["housing_payment"], "other_debts": f["other_debts"]},
                "alternative_bases": {k: {"qualifying_monthly_income": v, "dti_pct": d}
                                      for k, (v, d) in self._alternatives(f)[0].items()},
                "ytd_pace_vs_last_year_pct": self._alternatives(f)[1],
                "guideline": "variable/self-employed income needs a 24-month history; if it fell more than 15% "
                             "year on year, use the lower recent figure or document why it recovered; max DTI 45%",
            }))
        if surface == "income_worksheet":
            rows = [["Two years ago", money(f["variable_two_years_ago"], rng, style="sym")],
                    ["Last year", money(f["variable_last_year"], rng, style="sym") + f" ({f['yoy_change_pct']:+d}%)"],
                    [f"YTD ({f['months_ytd']} mo)", money(f["ytd_variable"], rng, style="sym")]]
            head = kv_lines([("Loan", f["loan_number"]), ("Income type", INCOME_TYPES[f["income_type"]]),
                             ("Base (monthly)", money(f["base_monthly"], rng) if f["base_monthly"] else None),
                             ("History of this income", f"{f['variable_history_months']} months"),
                             ("Gap in last 2 years", f"{f['employment_gap_months']} months"
                              if f["employment_gap_months"] else None)], rng)
            tail = kv_lines([("Documents", self.DOCS[f["income_type"]][f["documents"]]),
                             ("Letters", self.LETTER[f["employer_letter"]]),
                             ("Processor basis", self.BASIS[f["processor_basis"]]),
                             ("Processor qualifying income", money(f["processor_income"], rng) + "/mo"),
                             ("PITIA + other debts", f"{money(f['housing_payment'], rng)} + "
                                                     f"{money(f['other_debts'], rng)}"),
                             ("DTI at processor income", f"{self._dti(f, f['processor_income'])}%")], rng)
            return (f"INCOME CALCULATION WORKSHEET\n{head}\n\n{table(['Period', label], rows, rng)}\n\n{tail}"
                    + f"\n\nAlternative bases -- {self._alt_text(f, rng)}."
                    + (f"\n\nNote: {f['industry_note']}" if f["industry_note"] else ""))
        s = [(f"Base is {money(f['base_monthly'], rng)}/mo. " if f["base_monthly"] else "")
             + f"{label.capitalize()}: {money(f['variable_two_years_ago'], rng)} two years ago, "
               f"{money(f['variable_last_year'], rng)} last year, {money(f['ytd_variable'], rng)} so far this year "
               f"({f['months_ytd']} months).",
             f"They've had this income for {f['variable_history_months']} months.",
             f"I used the {self.BASIS[f['processor_basis']]}, which gives {money(f['processor_income'], rng)}/mo "
             f"and a DTI of {self._dti(f, f['processor_income'])}%.",
             "Docs: " + self.DOCS[f["income_type"]][f["documents"]] + ".",
             f"For comparison, {self._alt_text(f, rng)}."]
        if self.LETTER[f["employer_letter"]]:
            s.append(self.LETTER[f["employer_letter"]][0].upper() + self.LETTER[f["employer_letter"]][1:] + ".")
        if f["employment_gap_months"]:
            s.append(f"There was a {f['employment_gap_months']}-month gap in employment in the last two years.")
        if f["industry_note"]:
            s.append(f["industry_note"])
        return (f"Processor note -- {f['processor']}, {fmt_date(f['date'], rng)}\n"
                f"UW -- income on {f['loan_number']} ({INCOME_TYPES[f['income_type']]})\n\n"
                + noisy(join_sentences(s), rng, 0.15))

    def questions(self):
        return {
            "income_treatment": choice("How should the underwriter treat this income for qualifying?", {
                "average_two_years": "Qualify on the average of the last two years.",
                "cannot_determine_yet": "Hold off; the income cannot be set until more documents arrive.",
                "exclude_variable_income": "Leave the variable or business income out and qualify on anything "
                                           "else that remains.",
                "use_as_calculated": "Accept the processor's figure as calculated.",
                "use_lower_recent_figure": "Use the lower, more recent figure because the income is falling.",
            }),
            "income_stability": score("How stable and likely to continue is this income?", [
                "Unstable: under 12 months of history, an employer letter saying it is being reduced, or a fall of "
                "more than 15% last year with this year's pace also more than 10% down.",
                "Uncertain: a fall of more than 15% last year, this year's pace more than 10% down, or a recent "
                "employment gap.",
                "Reasonably stable: none of those, but under 24 months of history, a small fall, or no letter "
                "confirming the income continues.",
                "Stable: 24+ months, flat or rising, on pace this year, and a letter confirming it continues.",
            ]),
            "calculation_sensitivity": score("How much does qualifying depend on which income figure is used?", [
                "Not at all: DTI is 42% or less on all three bases (most recent year, YTD annualised, two-year "
                "average).",
                "Slightly: DTI is 45% or less on all three bases, but above 42% on at least one.",
                "Considerably: DTI is over 45% on exactly one of the three bases.",
                "Entirely: DTI is over 45% on two or all three bases.",
            ]),
            "history_requirement_met": noul(
                "The borrower has the income history the program requires for this kind of income.",
                yes="At least 24 months of this income is documented, without an unexplained recent gap.",
                no="The history is shorter than 24 months, a year is undocumented, or a recent gap breaks it."),
            "more_income_docs_needed": noul(
                "More income documentation is needed before this income can be used."),
        }


# =========================================================================== schema 3
REPAIRS = {"none": (None, "none"),
           "cosmetic": ("peeling paint on exterior trim and worn carpet", "cosmetic"),
           "handrail": ("missing handrail on the basement stairs", "safety"),
           "roof_leak": ("active roof leak with fresh ceiling staining in two rooms", "structural"),
           "furnace": ("furnace not operating at inspection", "habitability")}


class AppraisalReview(Schema):
    name = "appraisal_review"
    description = "An appraisal on a purchase loan reviewed for value support, comparables, condition and required repairs."
    surfaces = {"json": 0.5, "review_form": 0.25, "appraiser_comments": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        occ = pick_w(rng, {"primary": 6, "second_home": 1, "investment": 2})
        max_ltv = MAX_LTV[("purchase", occ)]
        price = _round_k(lognormal_amount(rng, 430000, 0.4))
        gap_pct = pick(rng, [-6, -3, -1, 0, 0, 1, 3])
        target_ltv = pick(rng, [max_ltv, max_ltv - 2, max_ltv - 5, max_ltv - 15])
        market = pick_w(rng, {"increasing": 2, "stable": 4, "declining": 2})
        n_comps = pick(rng, [3, 3, 4, 5])
        max_dist = pick(rng, [0.3, 0.8, 1.4, 3.5])
        oldest = pick(rng, [2, 4, 7, 11])
        net_adj = pick(rng, [4, 9, 14, 19, 27])
        condition = pick_w(rng, {2: 2, 3: 4, 4: 3, 5: 1})
        repair = pick_w(rng, {"none": 4, "cosmetic": 2, "handrail": 2, "roof_leak": 1, "furnace": 1})
        avm_pct = pick(rng, [-12, -5, 0, 4])
        ptype = pick(rng, ["single-family", "single-family", "condo", "2-unit", "townhouse"])
        if reg == "clear":
            if chance(rng, 0.5):
                gap_pct, market, max_dist, oldest, net_adj, condition, repair, avm_pct = 0, "stable", 0.3, 2, 4, \
                    2, "none", 0
                target_ltv = max_ltv - 5
            else:
                gap_pct, market, max_dist, oldest, net_adj, condition, repair, avm_pct = -6, "declining", 3.5, \
                    11, 27, 5, "roof_leak", -12
                target_ltv = max_ltv
        elif reg == "borderline":
            gap_pct = pick(rng, [-1, -3, 0])
            target_ltv = max_ltv - pick(rng, [0, 2])
            net_adj = pick(rng, [13, 14, 16])
            market = pick(rng, ["stable", "declining"])
            repair = pick(rng, ["none", "cosmetic", "handrail"])
        else:
            if chance(rng, 0.5):
                # the value comes in at price with tidy adjustments, but the comps are far and stale in a falling market
                gap_pct, net_adj, market, max_dist, oldest, avm_pct = 0, 6, "declining", 3.5, 11, -12
            else:
                # value well supported by close, recent comps, but a structural repair is required
                gap_pct, net_adj, max_dist, oldest, market = 1, 5, 0.3, 2, "stable"
                repair, condition = pick(rng, ["roof_leak", "furnace"]), 4
        value = _round_k(price * (1 + gap_pct / 100))
        loan = _round_k(price * target_ltv / 100)
        eff_max = max_ltv - (5 if market == "declining" else 0)
        return {"regime": reg, "loan_number": _loan_no(rng), "occupancy": occ, "program_max_ltv": max_ltv,
                "declining_market_max_ltv": eff_max, "contract_price": price, "appraised_value": value,
                "value_gap_pct": gap_pct, "loan_amount": loan, "ltv_on_price": round(loan / price * 100, 1),
                "market_trend": market, "comps": n_comps, "max_comp_distance_mi": max_dist,
                "oldest_comp_months": oldest, "largest_net_adjustment_pct": net_adj,
                "largest_gross_adjustment_pct": net_adj + pick(rng, [3, 6, 10]), "condition_rating": condition,
                "repair": repair, "as_is": repair in ("none", "cosmetic") and chance(rng, 0.8),
                "avm_value": _round_k(value * (1 + avm_pct / 100)), "property_type": ptype,
                "appraiser": person(rng), "reviewer": person(rng), "date": rand_date(rng),
                "area": pick(rng, ["suburban", "suburban", "urban", "rural"])}

    def _comment(self, f):
        s = []
        if f["value_gap_pct"] < 0:
            s.append("Adequate comparable support could not be found at the contract price.")
        if f["max_comp_distance_mi"] > 1:
            s.append(f"Limited recent sales in the subject's neighbourhood required comparables up to "
                     f"{f['max_comp_distance_mi']} miles away.")
        if f["oldest_comp_months"] > 6:
            s.append(f"One sale is {f['oldest_comp_months']} months old; it was the most similar in design.")
        if f["largest_net_adjustment_pct"] > 15:
            s.append("Adjustments exceed typical guidelines because of differences in size and condition.")
        rep = REPAIRS[f["repair"]][0]
        if rep:
            s.append(f"Noted: {rep}.")
        s.append({"increasing": "Values in the area are rising.", "stable": "Values are stable.",
                  "declining": "Marketing times have lengthened and values are softening."}[f["market_trend"]])
        return s

    def render(self, f, rng, surface):
        rep_txt, rep_kind = REPAIRS[f["repair"]]
        basis = "as is" if f["as_is"] else "subject to repairs"
        if surface == "json":
            return dumps(prune({
                "loan": {"number": f["loan_number"], "occupancy": OCC_TEXT[f["occupancy"]],
                         "amount": f["loan_amount"], "contract_price": f["contract_price"]},
                "appraisal": {"value": f["appraised_value"], "basis": basis, "property_type": f["property_type"],
                              "condition_rating": f"{f['condition_rating']} of 6 (1 = new)",
                              "noted_repairs": rep_txt, "market_trend": f["market_trend"], "area": f["area"],
                              "appraiser_commentary": " ".join(self._comment(f))},
                "comparables": {"count": f["comps"], "farthest_miles": f["max_comp_distance_mi"],
                                "oldest_sale_months": f["oldest_comp_months"],
                                "largest_net_adjustment_pct": f["largest_net_adjustment_pct"],
                                "largest_gross_adjustment_pct": f["largest_gross_adjustment_pct"]},
                "cross_checks": {"automated_valuation": f["avm_value"]},
                "guidelines": {"max_ltv_pct": f["program_max_ltv"],
                               "declining_market": "reduce max LTV by 5 points",
                               "ltv_basis": "lower of price and appraised value",
                               "comparables": "within 1 mile (suburban/urban) and 6 months; net adjustments <= 15%, "
                                              "gross <= 25%",
                               "repairs": "safety, structural or habitability items must be completed before "
                                          "closing"},
            }))
        if surface == "review_form":
            rows = [("Loan", f"{f['loan_number']} ({OCC_TEXT[f['occupancy']]}, {f['property_type']}, {f['area']})"),
                    ("Contract price", money(f["contract_price"], rng)), ("Loan amount", money(f["loan_amount"], rng)),
                    ("Appraised value", f"{money(f['appraised_value'], rng)} ({basis})"),
                    ("AVM cross-check", money(f["avm_value"], rng)),
                    ("Comparables", f"{f['comps']}; farthest {f['max_comp_distance_mi']} mi; oldest "
                                    f"{f['oldest_comp_months']} mo"),
                    ("Adjustments (largest)", f"net {f['largest_net_adjustment_pct']}%, gross "
                                              f"{f['largest_gross_adjustment_pct']}%"),
                    ("Condition", f"{f['condition_rating']} of 6"),
                    ("Repairs noted", f"{rep_txt} ({rep_kind})" if rep_txt else "none"),
                    ("Market", f["market_trend"])]
            return (f"APPRAISAL REVIEW -- {f['reviewer']}, {fmt_date(f['date'], rng)}\n" + kv_lines(rows, rng)
                    + f"\n\nProgram: max LTV {f['program_max_ltv']}% on the lower of price and value, 5 points less "
                      "in a declining market; comps within 1 mi / 6 mo, net adj <= 15%, gross <= 25%; safety, "
                      "structural and habitability repairs completed before closing.")
        grid = [[f"Comp {i + 1}", f"{between(rng, 0.1, f['max_comp_distance_mi'], digits=1) if i else f['max_comp_distance_mi']} mi",
                 f"{f['oldest_comp_months'] if i == 0 else rng.randint(1, max(1, f['oldest_comp_months']))} mo",
                 f"{f['largest_net_adjustment_pct'] if i == 1 else rng.randint(1, max(2, f['largest_net_adjustment_pct']))}%"]
                for i in range(f["comps"])]
        return (f"APPRAISER'S COMMENTS ({f['appraiser']}) -- {f['loan_number']}\n"
                f"Opinion of value: {money(f['appraised_value'], rng)} {basis}; contract price "
                f"{money(f['contract_price'], rng)}; loan {money(f['loan_amount'], rng)}\n\n"
                + noisy(join_sentences(self._comment(f)), rng, 0.1) + "\n\n"
                + table(["Sale", "Distance", "Age", "Net adj."], grid, rng)
                + f"\n\nLender guideline: max LTV {f['program_max_ltv']}% ({OCC_TEXT[f['occupancy']]}), -5 pts if "
                  "declining; LTV on lower of price/value.")

    def questions(self):
        return {
            "appraisal_action": choice("What should the underwriter do with this appraisal?", {
                "accept_appraisal": "Accept the appraisal and value as reported.",
                "accept_with_repair_condition": "Accept the value, but condition the loan on the listed repairs "
                                                "being completed and inspected.",
                "order_second_opinion": "Order a desk or field review by a second appraiser.",
                "reject_collateral": "Reject the property as security for this loan program.",
                "return_to_appraiser": "Send it back to the appraiser to support or correct specific points, such "
                                       "as comparable choice or adjustments.",
            }),
            "value_support": score("How well do the comparables and adjustments support the appraised value?", [
                "Poorly: distant, stale or heavily adjusted sales carry the value.",
                "Weakly: several guideline limits are stretched.",
                "Adequately: one point is stretched, with a reason given.",
                "Strongly: close, recent, lightly adjusted sales.",
            ]),
            "collateral_concern": score("How concerned should the lender be about the property as security for "
                                        "the loan?", [
                "Not concerned: sound property, well supported value, stable market.",
                "Slightly: a minor condition or market point to note.",
                "Considerably: value, condition or market could leave the loan under-secured.",
                "Seriously: the property may not cover the loan or is not in acceptable condition.",
            ]),
            "repairs_before_closing": noul(
                "Repairs must be completed and inspected before this loan can close.",
                yes="A noted item affects safety, structure or habitability, or the value is subject to repairs.",
                no="Nothing is noted, or only cosmetic wear that does not affect the property's soundness."),
            "ltv_over_limit_on_value": noul(
                "Measured on the lower of price and appraised value, the loan-to-value ratio exceeds the maximum "
                "that applies to this property."),
        }


# =========================================================================== schema 4
COND_POOL = [("paystubs", "Most recent 30 days of pay stubs for B1", "income"),
             ("inquiry_loe", "Letter explaining the recent credit inquiry", "credit"),
             ("hoi", "Homeowner's insurance binder naming the lender as mortgagee", "closing"),
             ("payoff", "Payoff statement for the existing first lien", "closing"),
             ("gift", "Signed gift letter and evidence of the donor's transfer", "assets"),
             ("vvoe", "Verbal verification of employment within 10 business days of closing", "income"),
             ("bank_stmt", "Updated bank statement covering the last 30 days", "assets"),
             ("hoa_cert", "HOA certification and budget", "collateral"),
             ("final_inspection", "Final inspection confirming the required repairs", "collateral")]
STATUS = ["cleared", "received - under review", "outstanding"]


class PriorToCloseConditions(Schema):
    name = "prior_to_close_conditions"
    description = "An approved file's prior-to-close conditions as closing and the rate-lock expiry approach."
    surfaces = {"json": 0.5, "conditions_log": 0.25, "closing_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        purpose = pick_w(rng, {"purchase": 3, "rate_term_refi": 1})
        drop = "payoff" if purpose == "purchase" else "gift"
        pool = [c for c in COND_POOL if c[0] != drop]
        conds = rng.sample(pool, rng.randint(4, 6))
        statuses = [pick_w(rng, {"cleared": 5, "received - under review": 2, "outstanding": 2}) for _ in conds]
        days_close = pick(rng, [1, 3, 5, 9, 16])
        lock_days = days_close + pick(rng, [-2, 0, 1, 4, 10])
        new_debt = pick_w(rng, {None: 5, 380: 1, 540: 1, 760: 1})
        vvoe = pick_w(rng, {"confirmed": 4, "pending": 3, "reduced_hours": 1, "on_leave": 1})
        balance_drop = pick_w(rng, {"none": 5, "minor": 2, "below_required": 1})
        approved_dti = near(rng, 40, 5, lo=25, hi=49.5, digits=1)
        income = _round_k(lognormal_amount(rng, 10500, 0.35))
        turnaround = pick(rng, [1, 2, 3])
        if reg == "clear":
            if chance(rng, 0.5):
                statuses = ["cleared"] * len(conds)
                days_close, lock_days, new_debt, vvoe, balance_drop = 16, 26, None, "confirmed", "none"
            else:
                statuses = [pick(rng, ["outstanding", "received - under review"]) for _ in conds]
                days_close, lock_days, new_debt, vvoe, balance_drop = 1, 0, 760, "on_leave", "below_required"
        elif reg == "borderline":
            days_close = pick(rng, [3, 5])
            lock_days = days_close + pick(rng, [-1, 0, 1])
            new_debt = pick(rng, [None, 380])
            approved_dti = near(rng, 43.5, 1.2, digits=1)
        else:
            if chance(rng, 0.5):
                # every document is in and the lock has room, but a new car loan appeared on the credit refresh
                statuses = ["cleared"] * len(conds)
                days_close, lock_days, new_debt, vvoe = 9, 19, pick(rng, [540, 760]), "confirmed"
                approved_dti = near(rng, 42.5, 1.5, digits=1)
            else:
                # the numbers are unchanged, but half the paperwork is outstanding and closing is days away
                new_debt, vvoe, balance_drop = None, "confirmed", "none"
                statuses = [pick(rng, ["outstanding", "received - under review", "cleared"]) for _ in conds]
                days_close, lock_days = 2, 3
        new_dti = round(approved_dti + (new_debt or 0) / income * 100, 1)
        return {"regime": reg, "loan_number": _loan_no(rng), "purpose": purpose, "conditions": [c[1] for c in conds],
                "condition_types": [c[2] for c in conds], "statuses": statuses,
                "days_to_closing": days_close, "lock_expires_in_days": lock_days, "new_monthly_debt": new_debt,
                "new_debt_kind": pick(rng, ["auto loan", "retail credit card", "furniture financing"])
                if new_debt else None,
                "vvoe_result": vvoe, "balance_change": balance_drop, "approved_dti": approved_dti,
                "dti_with_new_debt": new_dti, "monthly_income": income, "uw_turnaround_days": turnaround,
                "closer": person(rng), "lo": person(rng), "underwriter": person(rng), "date": rand_date(rng)}

    VVOE = {"confirmed": "employer confirmed active, full-time, same position",
            "pending": "left voicemail with HR, not yet confirmed",
            "reduced_hours": "employer says B1 moved to 30 hours a week last month",
            "on_leave": "employer says B1 is on unpaid leave until further notice"}
    BAL = {"none": "balance consistent with approval", "minor": "balance down about 8% since approval; reserves "
           "still above requirement", "below_required": "large withdrawal since approval; reserves now below the "
           "required 2 months"}

    def _events(self, f):
        e = []
        if f["new_monthly_debt"]:
            e.append(f"Credit refresh: new {f['new_debt_kind']} opened after approval, payment "
                     f"${f['new_monthly_debt']}/month (DTI {f['approved_dti']}% at approval, {f['dti_with_new_debt']}% "
                     "with it)")
        e.append("VVOE: " + self.VVOE[f["vvoe_result"]])
        e.append("Assets: " + self.BAL[f["balance_change"]])
        return e

    @staticmethod
    def _when(n):
        return "today" if n == 0 else f"in {n} day{'s' if n != 1 else ''}" if n > 0 else \
            f"{-n} day{'s' if n != -1 else ''} ago"

    def render(self, f, rng, surface):
        closing = f["date"] + timedelta(days=f["days_to_closing"])
        lock = f["date"] + timedelta(days=f["lock_expires_in_days"])
        if surface == "json":
            return dumps(prune({
                "loan": {"number": f["loan_number"], "purpose": PURPOSE_TEXT[f["purpose"]],
                         "approved_dti_pct": f["approved_dti"], "program_max_dti_pct": 45},
                "dates": {"today": f["date"].isoformat(), "scheduled_closing": closing.isoformat(),
                          "rate_lock_expires": lock.isoformat(), "days_to_closing": f["days_to_closing"],
                          "days_to_lock_expiry": f["lock_expires_in_days"],
                          "uw_review_turnaround_business_days": f["uw_turnaround_days"]},
                "conditions": [{"item": c, "status": s} for c, s in zip(f["conditions"], f["statuses"])],
                "since_approval": self._events(f),
            }))
        if surface == "conditions_log":
            rows = [[str(i + 1), c, s] for i, (c, s) in enumerate(zip(f["conditions"], f["statuses"]))]
            return (f"PRIOR-TO-CLOSE CONDITIONS -- {f['loan_number']}, {PURPOSE_TEXT[f['purpose']]} "
                    f"(as of {fmt_date(f['date'], rng)})\n"
                    f"Closing: {fmt_date(closing, rng)} ({self._when(f['days_to_closing'])}) | Lock expires: "
                    f"{fmt_date(lock, rng)} ({self._when(f['lock_expires_in_days'])}) | "
                    f"UW turnaround: {f['uw_turnaround_days']} business day(s)\n\n"
                    + table(["#", "Condition", "Status"], rows, rng) + "\n\nSince approval:\n"
                    + bullets(self._events(f), rng) + f"\n\nApproved DTI {f['approved_dti']}%; program max 45%.")
        open_items = [c for c, s in zip(f["conditions"], f["statuses"]) if s != "cleared"]
        msgs = [(f["closer"], f"Closing on {f['loan_number']} is set for {fmt_date(closing, rng)} "
                              f"({self._when(f['days_to_closing'])}) and the lock runs out {fmt_date(lock, rng)} "
                              f"({self._when(f['lock_expires_in_days'])}). Where are we on CTC?"),
                (f["lo"], ("Still waiting on: " + "; ".join(open_items) + ".") if open_items else
                 "Everything should be in now."),
                (f["underwriter"], " ".join(e + "." for e in self._events(f))
                 + f" Review turnaround is {f['uw_turnaround_days']} business day(s) once docs land.")]
        out = []
        for who, text in msgs:
            out.append(f"From: {who}\n{noisy(text, rng, 0.1)}")
        return f"Thread: {f['loan_number']} -- clear to close?\n\n" + "\n\n".join(out)

    def questions(self):
        return {
            "closing_readiness": choice("What should the underwriter do with this file now?", {
                "ctc_after_listed_items": "Clear to close once the open items are in and checked: conditions are "
                                          "still outstanding or under review, or the employment check is pending, "
                                          "and nothing found since approval breaks the approval.",
                "issue_clear_to_close": "Issue clear to close now: every condition is cleared, employment is "
                                        "confirmed, and any new debt leaves DTI at or under 45%.",
                "re_underwrite_file": "Send the file back through underwriting: hours were reduced, or new debt "
                                      "takes DTI above the 45% maximum.",
                "suspend_file": "Suspend: the borrower is on unpaid leave, or a withdrawal has left reserves below "
                                "the requirement.",
            }),
            "closing_clock": score("How tight is the timeline to close before the rate lock or the closing date "
                                   "is missed?", [
                "Comfortable: closing is more than 14 days away and the lock runs past it.",
                "Manageable: closing in 8 to 14 days, with the lock running past it.",
                "Tight: closing in 4 to 7 days, with the lock running past it.",
                "Critical: closing in 3 days or less, with the lock running past it and review of open items able "
                "to finish by then.",
                "Missed: the lock expires before the scheduled closing, or review of items still open takes more "
                "days than are left.",
            ]),
            "open_items_weight": score("How serious are the conditions and findings still open on this file?", [
                "None: everything is cleared and nothing new has turned up.",
                "Paperwork: routine documents that will not change the numbers.",
                "Numbers at stake: open items or new findings that could change income, debts or assets.",
                "Approval at stake: findings that could undo the loan approval.",
            ]),
            "material_change": noul(
                "Something has changed since approval that alters the borrower's qualification.",
                yes="New debt, a change in employment or a drop in assets moves a ratio or requirement against "
                    "the approval.",
                no="Nothing new, or the change is too small to move the ratios or requirements."),
            "lock_extension_needed": noul(
                "The rate lock will need to be extended for this loan to close."),
        }


# =========================================================================== schema 5
DEPOSIT_SOURCES = {
    "employer_bonus": "annual bonus from employer",
    "own_savings": "transfer from B1's savings account at another bank",
    "vehicle_sale": "sale of a second car",
    "cash_at_home": "cash kept at home, deposited in one go",
    "family_gift": "gift from B1's parent toward the down payment",
    "crypto_sale": "proceeds from selling crypto-assets",
    "business_account": "transfer from B1's business account",
}
EVIDENCE = {
    "employer_bonus": ["pay stub showing the bonus line", None],
    "own_savings": ["statement from the other account showing the withdrawal", None],
    "vehicle_sale": ["signed bill of sale and title transfer", "bill of sale only"],
    "cash_at_home": [None, "borrower's letter only"],
    "family_gift": ["signed gift letter and donor's bank statement showing the transfer", "signed gift letter only"],
    "crypto_sale": ["exchange statement showing the sale and withdrawal to B1's account", "screenshot of an exchange app"],
    "business_account": ["CPA letter that the withdrawal will not harm the business, plus business statements",
                         "business statements only"],
}


class LargeDepositSourcing(Schema):
    name = "large_deposit_sourcing"
    description = "A large deposit in the borrower's bank statements, to be sourced before the funds count toward closing."
    surfaces = {"json": 0.5, "statement_excerpt": 0.25, "asset_worksheet": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        income = round(lognormal_amount(rng, 9200, 0.35), -1)
        threshold = round(income * 0.5)
        src = pick(rng, list(DEPOSIT_SOURCES))
        deposit = _r50(income * pick(rng, [0.3, 0.45, 0.7, 1.5, 3.0]))
        evidence = pick(rng, EVIDENCE[src])
        seasoned_days = pick(rng, [12, 35, 58, 75])
        cash_to_close = _round_k(lognormal_amount(rng, 42000, 0.5))
        pitia = round(income * rng.uniform(0.24, 0.36))
        other_assets = _round_k(cash_to_close * pick(rng, [0.4, 0.8, 1.1, 1.5]))
        reserves_req = pick(rng, [2, 2, 6])
        split_pattern = False
        if reg == "clear":
            if chance(rng, 0.5):
                src, deposit = pick(rng, ["employer_bonus", "own_savings"]), _r50(income * 1.5)
                evidence, seasoned_days = EVIDENCE[src][0], 58
                other_assets = _round_k(cash_to_close * 1.5 + pitia * reserves_req)
            else:
                src, deposit, evidence = "cash_at_home", _r50(income * 3.0), "borrower's letter only"
                other_assets, seasoned_days = _round_k(cash_to_close * 0.4), 12
        elif reg == "borderline":
            deposit = _r50(near(rng, threshold, threshold * 0.12))
            other_assets = _round_k(near(rng, cash_to_close + pitia * reserves_req - deposit * 0.5,
                                         cash_to_close * 0.15))
        else:
            if chance(rng, 0.5):
                # well documented, but not needed: the borrower has plenty without it (or the reverse)
                src = pick(rng, ["crypto_sale", "family_gift", "business_account"])
                evidence = EVIDENCE[src][1]
                deposit = _r50(income * 1.5)
                other_assets = _round_k(cash_to_close * 1.6 + pitia * reserves_req)
            else:
                # each deposit is under the sourcing threshold, but there are several of them close together
                split_pattern, src, evidence = True, "cash_at_home", "borrower's letter only"
                deposit = _r50(threshold * rng.uniform(0.82, 0.95))
                other_assets = _round_k(cash_to_close * 0.8)
        n_deposits = 3 if split_pattern else 1
        total_verified = other_assets + deposit * n_deposits
        return {"regime": reg, "loan_number": _loan_no(rng), "monthly_income": income,
                "sourcing_threshold": threshold, "deposit_source": src, "deposit_amount": deposit,
                "deposit_count": n_deposits, "evidence": evidence, "days_in_account": seasoned_days,
                "cash_to_close": cash_to_close, "pitia": pitia, "reserves_required_months": reserves_req,
                "assets_excluding_deposit": other_assets, "total_assets": total_verified,
                "payroll_like": src == "employer_bonus", "underwriter": person(rng), "processor": person(rng),
                "date": rand_date(rng), "bank": pick(rng, ["Lakeshore Savings", "Copperline Credit Union",
                                                           "Granite Bank", "Valewood Bank"])}

    def _loe(self, f, rng):
        return noisy({"employer_bonus": "This was my yearly bonus from work, paid in March.",
                      "own_savings": "I moved money from my savings at my other bank to have it all in one place.",
                      "vehicle_sale": "We sold our second car since we won't need it after the move.",
                      "cash_at_home": ("I have saved cash at home over several years and took it to the branch a bit "
                                       "at a time." if f["deposit_count"] > 1 else
                                       "I have saved cash at home over several years and deposited it."),
                      "family_gift": "My parent gave us this money to help with the down payment. It does not have "
                                     "to be paid back.",
                      "crypto_sale": "I sold some crypto I bought a few years ago.",
                      "business_account": "I took a draw from my company account for the down payment."}[
            f["deposit_source"]], rng, 0.2)

    @staticmethod
    def _reserve_months(f):
        base = f["assets_excluding_deposit"] - f["cash_to_close"]
        return (round(base / f["pitia"], 1),
                round((base + f["deposit_amount"] * f["deposit_count"]) / f["pitia"], 1))

    def render(self, f, rng, surface):
        need = f["cash_to_close"] + f["pitia"] * f["reserves_required_months"]
        r_without, r_with = self._reserve_months(f)
        if surface == "json":
            return dumps(prune({
                "loan": f["loan_number"],
                "deposit": {"amount": f["deposit_amount"],
                            "occurrences": f"{f['deposit_count']} deposits of about this size within 3 weeks"
                            if f["deposit_count"] > 1 else None,
                            "days_in_account": f["days_in_account"], "borrower_explanation": self._loe(f, rng),
                            "evidence_provided": f["evidence"] or "none"},
                "qualifying": {"monthly_income": f["monthly_income"], "sourcing_threshold": f["sourcing_threshold"],
                               "sourcing_rule": "single deposits over 50% of monthly qualifying income must be "
                                                "sourced unless seasoned 60+ days or identifiable payroll"},
                "funds": {"cash_to_close": f["cash_to_close"], "monthly_pitia": f["pitia"],
                          "reserves_required_months": f["reserves_required_months"],
                          "verified_assets_excluding_deposit": f["assets_excluding_deposit"],
                          "reserves_after_closing_months": {"without_deposit": r_without, "with_deposit": r_with}},
            }))
        if surface == "statement_excerpt":
            start = f["date"] - timedelta(days=f["days_in_account"] + 10)
            bal = round(f["assets_excluding_deposit"] * rng.uniform(0.55, 0.75), 2)
            lines = []
            txns = [(start + timedelta(days=2), "PAYROLL DIRECT DEP", round(f["monthly_income"] * 0.36, 2)),
                    (start + timedelta(days=5), "CARD PURCHASE GROCERY", -round(rng.uniform(80, 240), 2)),
                    (start + timedelta(days=9), "ONLINE PMT UTILITY", -round(rng.uniform(90, 260), 2))]
            dep_label = {"employer_bonus": "PAYROLL DIRECT DEP - BONUS", "own_savings": "INCOMING TRANSFER",
                         "vehicle_sale": "MOBILE CHECK DEPOSIT", "cash_at_home": "BRANCH CASH DEPOSIT",
                         "family_gift": "INCOMING WIRE", "crypto_sale": "ACH CREDIT EXCHANGE",
                         "business_account": "TRANSFER FROM BUSINESS CHK"}[f["deposit_source"]]
            for k in range(f["deposit_count"]):
                txns.append((start + timedelta(days=10 + 6 * k), dep_label, float(f["deposit_amount"])))
            txns.sort(key=lambda t: t[0])
            for d, desc, amt in txns:
                bal = round(bal + amt, 2)
                lines.append(f"{d.strftime('%m/%d')}  {desc:<28} {amt:>11,.2f}  {bal:>12,.2f}")
            return (f"{f['bank']} -- statement excerpt (B1) -- loan {f['loan_number']}\n"
                    f"DATE   DESCRIPTION                  AMOUNT       BALANCE\n" + "\n".join(lines)
                    + f"\n\nLetter of explanation: \"{self._loe(f, rng)}\"\n"
                      f"Documents received: {f['evidence'] or 'none'}\n"
                      f"Qualifying income {money(f['monthly_income'], rng)}/mo; deposits over 50% of that "
                      f"({money(f['sourcing_threshold'], rng)}) must be sourced unless seasoned 60 days or payroll.\n"
                      f"Reserves after closing: {r_without:g} months without the deposit, {r_with:g} with it "
                      f"(required: {f['reserves_required_months']}).")
        rows = [("Loan", f["loan_number"]), ("Monthly qualifying income", money(f["monthly_income"], rng)),
                ("Deposit", money(f["deposit_amount"], rng)
                 + (f" (x{f['deposit_count']} within 3 weeks)" if f["deposit_count"] > 1 else "")),
                ("Days in account", f["days_in_account"]),
                ("Explanation", "cash kept at home, deposited over several branch visits" if f["deposit_count"] > 1
                 else DEPOSIT_SOURCES[f["deposit_source"]]),
                ("Evidence", show(f["evidence"], rng)), ("Cash to close", money(f["cash_to_close"], rng)),
                ("PITIA", money(f["pitia"], rng)), ("Reserves required", f"{f['reserves_required_months']} months"),
                ("Verified assets without the deposit", money(f["assets_excluding_deposit"], rng)),
                ("Needed (close + reserves)", money(need, rng)),
                ("Sourcing threshold (50% of income)", money(f["sourcing_threshold"], rng)),
                ("Reserves after closing", f"{r_without:g} months without the deposit, {r_with:g} with it")]
        return (f"ASSET WORKSHEET -- {f['processor']}\n" + kv_lines(rows, rng, skip_none=False)
                + "\n\nRule: single deposits > 50% of monthly qualifying income need sourcing unless seasoned 60+ "
                  "days or identifiable payroll.")

    def questions(self):
        return {
            "deposit_treatment": choice("How should the underwriter treat the deposit?", {
                "accept_as_sourced": "Accept it; the documents already trace it to an acceptable source.",
                "exclude_from_assets": "Leave it out of the borrower's usable assets.",
                "require_gift_documentation": "Require complete gift documentation before it can be used.",
                "require_paper_trail": "Require documents that trace the money to its source before it can be used.",
            }),
            "impact_if_excluded": choice("If the deposit could not be used, where would the file stand?", {
                "short_on_funds_to_close": "Short of the cash needed to close.",
                "short_on_reserves_only": "Enough to close, but short of the required reserves.",
                "still_qualifies": "Still enough for closing costs and reserves without it.",
            }),
            "sourcing_concern": score("How concerning is the origin of the deposited money?", [
                "Not at all: an ordinary, traceable source.",
                "Mildly: plausible, but the paper trail is incomplete.",
                "Considerably: hard to verify, or the pattern looks arranged.",
                "Seriously: the money could be an undisclosed loan or otherwise unacceptable.",
            ]),
            "reserves_strength": score("How strong are the borrower's reserves after closing, counting only funds "
                                       "that can be used today?", [
                "Deficient: below the required months (count the deposit only if it needs no sourcing or its "
                "documents already trace it to an acceptable source).",
                "Bare: from the requirement up to 2 months above it, counted the same way.",
                "Adequate: 2 to under 6 months above the requirement.",
                "Ample: 6 or more months above the requirement.",
            ]),
            "sourcing_required": noul(
                "Program rules require this deposit to be documented to its source.",
                yes="It is a gift, or the deposit (several similar deposits within three weeks taken together) "
                    "exceeds the sourcing threshold, has been in the account under 60 days and is not payroll.",
                no="It is not a gift and is at or under the threshold (also taken together), seasoned 60 days or "
                   "more, or identifiable payroll."),
        }


WORKFLOW = Workflow(
    name="mortgage_underwriting_conditions",
    title="Mortgage underwriting conditions",
    description="Whether a residential mortgage file should be approved, conditioned, suspended, countered or "
                "declined under a fictional lender's written program limits.",
    domain="lending / residential mortgage underwriting",
    schemas=[InitialUnderwriteDecision(), IncomeCalculationReview(), AppraisalReview(), PriorToCloseConditions(),
             LargeDepositSourcing()],
    notes="Credit decisions on the file's ratios, documents, collateral and assets only. Borrowers are loan numbers; "
          "no protected attribute is rendered or sampled, and no public-assistance, alimony or child-support "
          "income appears. No servicing hardship, no payment-fraud alerts, no bills.",
)
