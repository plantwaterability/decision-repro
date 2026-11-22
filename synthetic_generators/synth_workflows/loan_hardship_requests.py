"""Loan hardship requests: a servicer's loss-mitigation desk decides what relief, if any, a struggling borrower gets.

Inputs are an application (hardship letter, income and expense figures, documents received) plus the account's
servicing data, with the fictional programme's rules quoted in the state so every case is decidable in principle.
Not a support conversation: the borrower's letter is evidence inside an application, and no question asks how to
reply to anyone. No protected attribute drives any decision; health events appear only as the CAUSE of a hardship.
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


def _ms(rng):
    """One money style per rendered state (a real system does not mix them)."""
    return pick(rng, ["sym", "sym", "code"])


def _ds(rng):
    """One date style per rendered document."""
    return pick(rng, ["iso", "us", "long", "eu"])


def _letter_close(rng, name):
    return pick(rng, ["Thank you,", "Sincerely,", "Thanks for considering this,", "Regards,"]) + "\n" + name


# =========================================================================== schema 1
class MortgageForbearance(Schema):
    name = "mortgage_forbearance_request"
    description = "A mortgage borrower asks to pause or reduce payments after an income shock."
    surfaces = {"json": 0.45, "application_packet": 0.3, "servicer_note": 0.25}

    CAUSES = {
        "job_loss": "I was laid off when my employer closed the site",
        "reduced_hours": "my employer cut my hours from full time to about 24 a week",
        "medical_leave": "I have been on unpaid medical leave after surgery",
        "business_downturn": "my self-employment income dropped sharply this season",
        "household_earner_loss": "we lost the second income our household relied on",
        "major_repair": "we had to pay for an emergency roof replacement out of pocket",
    }
    CAUSES_3P = {
        "job_loss": "laid off when the employer closed the site",
        "reduced_hours": "hours cut from full time to about 24 a week",
        "medical_leave": "on unpaid medical leave after surgery",
        "business_downturn": "self-employment income down sharply this season",
        "household_earner_loss": "loss of the household's second income",
        "major_repair": "paid for an emergency roof replacement out of pocket",
    }
    EVIDENCE = {"recall_letter": "employer recall letter", "job_offer": "signed offer letter for a new job",
                "doctor_estimate": "physician's return-to-work estimate", "statement_only": "borrower's statement only"}

    def sample_factors(self, rng):
        reg = regime(rng)
        cause = pick(rng, list(self.CAUSES))
        income_before = round(between(rng, 3800, 11000), -1)
        drop_frac = rng.random()
        payment = round(income_before * between(rng, 0.2, 0.33), 0)
        other_debts = round(between(rng, 150, 1100), -1)
        delinquency = pick_w(rng, {0: 2, 30: 3, 60: 3, 90: 2})
        prior_fb = pick_w(rng, {0: 5, 3: 2, 6: 2, 9: 1})
        requested = pick(rng, [3, 6, 6, 9, 12])
        evidence = pick_w(rng, {"recall_letter": 2, "job_offer": 2, "doctor_estimate": 2, "statement_only": 3})
        expected_months = pick(rng, [2, 3, 4, 6, 9])
        income_docs = pick_w(rng, {"complete": 4, "partial": 3, "missing": 2})
        bank_months = pick_w(rng, {2: 4, 1: 3, 0: 2})
        reserves = round(between(rng, 0, 4), 1)
        occupancy = pick_w(rng, {"primary residence": 7, "second home": 2, "investment property": 1})
        lates = pick(rng, [0, 0, 0, 1, 2, 4])
        if reg == "clear":
            if chance(rng, 0.5):
                cause, evidence, expected_months, requested, prior_fb = "job_loss", "job_offer", 2, 3, 0
                income_docs, bank_months, occupancy, lates, delinquency = "complete", 2, "primary residence", 0, 30
            else:
                occupancy, prior_fb, requested, income_docs, bank_months = "investment property", 9, 9, "missing", 0
                evidence = "statement_only"
        elif reg == "borderline":
            # forbearance months used + requested sit right at the 12-in-24 cap; recovery evidence is soft
            prior_fb = pick(rng, [3, 6, 9])
            requested = 12 - prior_fb + pick(rng, [-3, 0, 0, 3])
            requested = max(3, requested)
            reserves = near(rng, 1.0, 0.6, lo=0.0, digits=1)
        else:  # conflicting: a clean history, but the "temporary" story and the documents disagree
            lates = 0
            if chance(rng, 0.5):
                cause, evidence, expected_months = "reduced_hours", "statement_only", 3
                income_docs = "complete"      # the employer letter in the file calls the schedule change permanent
            else:
                cause, evidence, occupancy = "medical_leave", "doctor_estimate", "second home"
                expected_months = requested + 3
        lo, hi = {"job_loss": (0.7, 1.0), "reduced_hours": (0.3, 0.5), "medical_leave": (0.6, 1.0),
                  "business_downturn": (0.3, 0.7), "household_earner_loss": (0.3, 0.55), "major_repair": (0.0, 0.0)}[cause]
        income_now = round(income_before * (1 - (lo + (hi - lo) * drop_frac)), -1)
        # income once the hardship ends, where the file says anything about it
        after = {"job_offer": round(income_before * between(rng, 0.75, 1.1), -1), "recall_letter": income_before,
                 "doctor_estimate": income_before, "statement_only": None}[evidence]
        employer_note = None
        if cause == "reduced_hours" and income_docs == "complete":
            employer_note = pick(rng, ["schedule change is permanent for the foreseeable future",
                                       "hours to be reviewed at the end of the quarter"])
        app_date = rand_date(rng)
        return {"regime": reg, "cause": cause, "income_before": income_before, "income_now": income_now,
                "income_after_expected": after, "payment": payment, "other_debts": other_debts,
                "days_delinquent": delinquency, "forbearance_months_used_24m": prior_fb,
                "months_requested": requested, "recovery_evidence": evidence,
                "borrower_expected_months": expected_months, "income_docs": income_docs,
                "bank_statement_months": bank_months, "reserves_months": reserves, "occupancy": occupancy,
                "late_payments_before_hardship_24m": lates, "employer_note": employer_note,
                "borrower": person(rng), "loan": ident(rng, "LN", 8, ""), "application_date": app_date,
                "in_foreclosure": delinquency >= 90 and chance(rng, 0.3), "servicer": org(rng, "bank")}

    RULES = ("Forbearance: owner-occupied homes only (primary residence or second home); no more than 12 months of "
             "forbearance in any 24-month period; hardship letter plus income documents or two months of bank "
             "statements required. Repayment plans may run up to 12 months. Payment deferral needs the hardship to be "
             "resolved.")

    @staticmethod
    def _dti(f, which):
        inc = f["income_now"] if which == "now" else f["income_after_expected"]
        if inc is None:
            return None
        return "no income" if inc <= 0 else f"{round(100 * (f['payment'] + f['other_debts']) / inc, 1)}%"

    def _letter(self, f, rng):
        s = [f"I am writing because {self.CAUSES[f['cause']]}."]
        if f["cause"] != "major_repair":
            s.append(f"Our household income went from about {money(f['income_before'])} to {money(f['income_now'])} a month.")
        if f["recovery_evidence"] == "job_offer":
            s.append(f"I have accepted a new position that starts in about {f['borrower_expected_months']} months.")
        elif f["recovery_evidence"] == "recall_letter":
            s.append(f"My employer expects to call us back within {f['borrower_expected_months']} months.")
        elif f["recovery_evidence"] == "doctor_estimate":
            s.append(f"My doctor expects me back at work in {f['borrower_expected_months']} months.")
        else:
            s.append(pick(rng, [f"I expect things to be back to normal in {f['borrower_expected_months']} months.",
                                "I am confident this is temporary."]))
        s.append(f"I am asking for {f['months_requested']} months of reduced or paused payments.")
        if f["late_payments_before_hardship_24m"] == 0:
            s.append("I have never missed a payment before this.")
        return noisy(join_sentences(s), rng)

    def render(self, f, rng, surface):
        ms = _ms(rng)
        m = lambda x: money(x, style=ms)  # noqa: E731
        docs = {"hardship_letter": "received",
                "income_documents": {"complete": "received", "partial": "pay stubs for 1 of 2 months",
                                     "missing": "not received"}[f["income_docs"]],
                "bank_statements": {2: "2 months", 1: "1 month", 0: "not received"}[f["bank_statement_months"]],
                "recovery_evidence": self.EVIDENCE[f["recovery_evidence"]]}
        if surface == "json":
            return dumps(prune({
                "application": {"id": ident(rng, "HRP", 6, "-"), "received": fmt_date(f["application_date"], style="iso"),
                                "months_requested": f["months_requested"], "hardship_letter": self._letter(f, rng)},
                "documents": docs,
                "finances": {"monthly_income_before": f["income_before"], "monthly_income_now": f["income_now"],
                             "expected_income_after_hardship": f["income_after_expected"],
                             "monthly_mortgage_payment": f["payment"], "other_monthly_debts": f["other_debts"],
                             "cash_reserves_months": f["reserves_months"], "employer_letter_note": f["employer_note"],
                             "payment_plus_debts_pct_of_income_now": self._dti(f, "now"),
                             "payment_plus_debts_pct_of_expected_income_after": self._dti(f, "after")},
                "loan": {"number": f["loan"], "occupancy": f["occupancy"], "days_delinquent": f["days_delinquent"],
                         "in_foreclosure": f["in_foreclosure"],
                         "late_payments_24m_before_hardship": f["late_payments_before_hardship_24m"],
                         "forbearance_months_used_last_24m": f["forbearance_months_used_24m"]},
                "programme_rules": self.RULES,
            }))
        if surface == "application_packet":
            letter = f"Dear {f['servicer']},\n\n{self._letter(f, rng)}\n\n{_letter_close(rng, f['borrower'])}"
            ws = table(["item", "amount / status"], [
                ["Income before hardship (monthly)", m(f["income_before"])],
                ["Income now (monthly)", m(f["income_now"])],
                ["Expected income after", show(f["income_after_expected"], rng, m)],
                ["Mortgage payment", m(f["payment"])], ["Other debts", m(f["other_debts"])],
                ["Payment + other debts, % of income now", self._dti(f, "now")],
                ["Payment + other debts, % of expected income after", self._dti(f, "after") or "no figure"],
                ["Savings (months of payments)", f["reserves_months"]],
                ["Income documents", docs["income_documents"]], ["Bank statements", docs["bank_statements"]],
                ["Recovery evidence", docs["recovery_evidence"]]], rng)
            head = kv_lines([("Loan", f["loan"]), ("Occupancy", f["occupancy"]),
                             ("Days delinquent", f["days_delinquent"]),
                             ("Forbearance used, last 24 months", f"{f['forbearance_months_used_24m']} months")], rng)
            extra = f"\nEmployer letter: {f['employer_note']}" if f["employer_note"] else ""
            return f"HARDSHIP LETTER\n{letter}\n\nWORKSHEET\n{ws}{extra}\n\n{head}\n\nProgramme: {self.RULES}"
        # servicer_note
        n = [f"Borrower {f['borrower']} ({f['loan']}, {f['occupancy']}) requests {f['months_requested']} months forbearance.",
             f"Cause per letter: {self.CAUSES_3P[f['cause']]}.",
             f"Income {m(f['income_before'])} -> {m(f['income_now'])}/mo; payment {m(f['payment'])}, other debts {m(f['other_debts'])}"
             f" ({self._dti(f, 'now')} of income now"
             + (f", {self._dti(f, 'after')} of expected income after)." if f["income_after_expected"] else
                "; no post-hardship income figure)."),
             f"Recovery support: {self.EVIDENCE[f['recovery_evidence']]}"
             + (f", expected income {m(f['income_after_expected'])}." if f["income_after_expected"] else "."),
             f"Docs: income {docs['income_documents']}, bank stmts {docs['bank_statements']}.",
             f"{f['days_delinquent']} days down; {f['late_payments_before_hardship_24m']} lates in the 24 months before.",
             f"Prior forbearance in last 24 months: {f['forbearance_months_used_24m']} months.",
             ("Loan is referred to foreclosure." if f["in_foreclosure"] else ""),
             (f"Employer letter says: {f['employer_note']}." if f["employer_note"] else "")]
        return "LOSS MITIGATION NOTE\n" + noisy(" ".join(x for x in n if x), rng, 0.1) + f"\n\nRules: {self.RULES}"

    def questions(self):
        return {
            "relief_recommendation": choice("Which form of relief should the loss-mitigation desk offer?", {
                "forbearance": "Pause or reduce payments for a set period: the hardship is temporary, the payment does "
                               "not fit current income, and the forbearance rules are met (owner-occupied, documents, "
                               "12-in-24 cap).",
                "repayment_plan": "Keep the regular payment and add an amount that clears the arrears: the hardship is "
                                  "temporary, the loan is no more than 30 days behind, and payment plus other debts "
                                  "are 40% or less of income now.",
                "loan_modification": "Permanently change the loan terms: the hardship is not shown to be temporary, so "
                                     "the payment has to fit a lower, lasting income.",
                "payment_deferral": "Move the missed payments to the end of the loan and resume the normal payment: "
                                    "the hardship was a one-off expense that is over, but a repayment plan does not fit.",
                "decline_and_refer": "Offer no relief and refer to housing counselling: the hardship is temporary but "
                                     "forbearance is barred (not owner-occupied, documents short of the rule, or the "
                                     "12-in-24 cap exceeded) and no repayment plan fits.",
            }),
            "hardship_is_temporary": noul(
                "The borrower's hardship is likely to be temporary.",
                yes="A third-party document gives an end point (employer recall letter, signed job offer or "
                    "physician's return-to-work estimate), or the hardship was a one-off expense with income unchanged.",
                no="Only the borrower's own statement supports a recovery, or an employer letter says the change is "
                   "permanent."),
            "evidence_completeness": score("How complete is the evidence for a decision under the programme?", [
                "Insufficient: the hardship or the income cannot be verified from the file.",
                "Thin: key items are missing and would have to be requested.",
                "Workable: minor gaps that do not block a decision.",
                "Complete: hardship, income and expenses are all documented.",
            ]),
            "post_hardship_affordability": score(
                "How well could the borrower afford the regular payment once the hardship ends?", [
                    "Not at all: payment plus other debts above 50% of the expected income after the hardship (or of "
                    "income now, where the file gives no post-hardship figure).",
                    "Barely: over 40% and up to 50% of that income.",
                    "Adequately: over 33% and up to 40% of that income.",
                    "Comfortably: 33% or less of that income.",
                ]),
            "meets_forbearance_rules": noul(
                "The loan and the request satisfy the programme's rules for forbearance."),
        }


# =========================================================================== schema 2
class ModificationReview(Schema):
    name = "modification_review"
    description = "A permanent-income-drop borrower is reviewed for a loan modification with ratios near the programme caps."
    surfaces = {"json": 0.45, "underwriting_worksheet": 0.3, "review_memo": 0.25}

    INCOME = {"salaried_long": "salaried, same employer 6+ years", "salaried_new": "salaried, new employer 4 months",
              "hourly_variable": "hourly with variable overtime", "self_employed": "self-employed, one year of returns",
              "benefits_fixed": "fixed pension and benefit income"}

    def sample_factors(self, rng):
        reg = regime(rng)
        cap = pick(rng, [43.0, 45.0])
        income = round(between(rng, 3200, 9000), -1)
        income_kind = pick(rng, list(self.INCOME))
        verified = pick_w(rng, {"verified": 5, "stated_only": 2, "partly_verified": 2})
        other_debts = round(between(rng, 100, 900), -1)
        target_dti = near(rng, cap, 4.0, lo=25, hi=60, digits=1)
        prior_mods = pick_w(rng, {0: 2, 1: 4, 2: 4})
        last_mod_years = round(between(rng, 0.5, 7.5), 1)
        delinquent = pick(rng, [60, 90, 120, 150, 180])
        failed_trial = chance(rng, 0.2)
        ltv = pick(rng, [72, 88, 97, 104, 118, 131])
        investor_limit = pick_w(rng, {"none": 5, "no principal forbearance": 2, "max term 40 years": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                target_dti, income_kind, verified, prior_mods, failed_trial = round(cap - 9, 1), "salaried_long", "verified", 0, False
            else:
                target_dti, verified, prior_mods, last_mod_years, failed_trial = round(cap + 11, 1), "stated_only", 2, 2.5, True
        elif reg == "borderline":
            target_dti = near(rng, cap, 0.8, digits=1)
            if prior_mods == 2:
                last_mod_years = near(rng, 5.0, 0.3, lo=4.2, digits=1)   # the 2-in-5-years rule is right at its edge
        else:  # conflicting: the ratio passes, but on shaky income; or strong income with a failed trial behind it
            if chance(rng, 0.5):
                target_dti, verified, income_kind = round(cap - between(rng, 1, 4), 1), "stated_only", pick(rng, ["self_employed", "hourly_variable"])
            else:
                target_dti, income_kind, failed_trial = round(cap + between(rng, 0.5, 2.5), 1), "salaried_long", True
        # the proposed payment follows from the target ratio, so every rendered figure agrees
        proposed = round(income * target_dti / 100 - other_debts, 0)
        proposed = max(proposed, 300.0)
        dti = round(100 * (proposed + other_debts) / income, 1)
        current_payment = round(proposed * between(rng, 1.15, 1.6), 0)
        app = rand_date(rng)
        loan_no = ident(rng, "LN", 8, "")
        borrower, reviewer = person(rng), person(rng)
        cause = pick(rng, ["permanent move to part-time work", "retirement on a smaller pension",
                           "a lasting pay cut after a role change", "loss of a second household income"])
        earlier = round(last_mod_years + between(rng, 0.5, 2.5), 1) if prior_mods == 2 else None
        return {"regime": reg, "dti_cap_pct": cap, "gross_monthly_income": income, "income_type": income_kind,
                "income_verification": verified, "other_monthly_debts": other_debts,
                "proposed_modified_payment": proposed, "resulting_dti_pct": dti,
                "current_payment": current_payment, "prior_modifications": prior_mods,
                "years_since_last_modification": last_mod_years if prior_mods else None,
                "years_since_earlier_modification": earlier,
                "days_delinquent": delinquent, "failed_prior_trial": failed_trial, "ltv_pct": ltv,
                "investor_restriction": investor_limit, "application_date": app, "loan": loan_no,
                "borrower": borrower, "reviewer": reviewer, "cause": cause}

    def _rules(self, f):
        return (f"Modification: resulting debt-to-income at or below {f['dti_cap_pct']:.0f}% on verified income; no "
                f"modification in the 3 years before the application and no more than 2 in any 5 years; at least 60 "
                f"days delinquent or in imminent default; a trial period of 3 payments precedes a permanent "
                f"modification.")

    def render(self, f, rng, surface):
        ms = _ms(rng)
        m = lambda x: money(x, style=ms)  # noqa: E731
        show_ratio = chance(rng, 0.6)          # some systems print the ratio, others leave the arithmetic to you
        ds = "iso" if surface == "json" else _ds(rng)
        last = (None if not f["prior_modifications"]
                else fmt_date(f["application_date"] - timedelta(days=int(365.25 * f["years_since_last_modification"])),
                              style=ds))
        earlier = (None if not f["years_since_earlier_modification"]
                   else fmt_date(f["application_date"]
                                 - timedelta(days=int(365.25 * f["years_since_earlier_modification"])), style=ds))
        if surface == "json":
            return dumps(prune({
                "application": {"loan": f["loan"], "received": fmt_date(f["application_date"], style="iso"),
                                "stated_cause": f["cause"], "days_delinquent": f["days_delinquent"]},
                "income": {"gross_monthly": f["gross_monthly_income"], "type": self.INCOME[f["income_type"]],
                           "verification": f["income_verification"].replace("_", " ")},
                "proposal": {"current_payment": f["current_payment"],
                             "proposed_payment": f["proposed_modified_payment"],
                             "other_monthly_debts": f["other_monthly_debts"],
                             "resulting_dti_pct": f["resulting_dti_pct"] if show_ratio else None},
                "history": {"prior_modifications": f["prior_modifications"], "last_modification_date": last,
                            "earlier_modification_date": earlier,
                            "failed_trial_plan": f["failed_prior_trial"], "loan_to_value_pct": f["ltv_pct"]},
                "investor": {"restriction": f["investor_restriction"]},
                "programme_rules": self._rules(f),
            }))
        if surface == "underwriting_worksheet":
            rows = [("Gross monthly income", m(f["gross_monthly_income"])),
                    ("Income type", self.INCOME[f["income_type"]]),
                    ("Verification", f["income_verification"].replace("_", " ")),
                    ("Current P&I+escrow", m(f["current_payment"])),
                    ("Proposed payment", m(f["proposed_modified_payment"])),
                    ("Other monthly debts", m(f["other_monthly_debts"])),
                    ("Back-end DTI", f"{f['resulting_dti_pct']}%" if show_ratio else "(not calculated)"),
                    ("Prior modifications", f["prior_modifications"]), ("Last modification", last),
                    ("Earlier modification", earlier),
                    ("Failed trial plan on record", "yes" if f["failed_prior_trial"] else "no"),
                    ("LTV", f"{f['ltv_pct']}%"), ("Investor restriction", f["investor_restriction"]),
                    ("Days delinquent", f["days_delinquent"])]
            return (f"MODIFICATION WORKSHEET  {f['loan']}  received {fmt_date(f['application_date'], style=ds)}\n"
                    + kv_lines(rows, rng, style=pick(rng, ["aligned", "colon"])) + f"\n\n{self._rules(f)}")
        memo = [f"{f['borrower']} applied for a modification after {f['cause']}.",
                f"Income: {m(f['gross_monthly_income'])} gross per month, {self.INCOME[f['income_type']]}; "
                f"verification {f['income_verification'].replace('_', ' ')}.",
                f"The proposed payment is {m(f['proposed_modified_payment'])} against {m(f['current_payment'])} today, "
                f"with {m(f['other_monthly_debts'])} of other debts"
                + (f", giving a back-end ratio of {f['resulting_dti_pct']}%." if show_ratio else "."),
                (f"This would be modification number {f['prior_modifications'] + 1}; the last one was on {last}"
                 + (f", and one before that on {earlier}." if earlier else ".")
                 if f["prior_modifications"] else "No earlier modifications."),
                ("A trial plan on this loan failed before." if f["failed_prior_trial"] else ""),
                f"LTV is about {f['ltv_pct']}%. Investor restriction: {f['investor_restriction']}.",
                pick(rng, ["Borrower seems committed.", "Borrower was hard to reach during collection.", ""])]
        return f"To: loss mitigation review\nFrom: {f['reviewer']}\nRe: {f['loan']}, application received " \
            f"{fmt_date(f['application_date'], style=ds)}\n\n" + noisy(join_sentences(memo), rng, 0.1) \
            + f"\n\n{self._rules(f)}"

    def questions(self):
        return {
            "modification_decision": choice("What should the reviewer do with this modification application?", {
                "approve_trial_plan": "Offer a trial plan on the proposed terms: within the modification limits, "
                                      "income fully verified and the resulting ratio at or under the cap.",
                "counter_offer_terms": "Offer a trial plan on different terms, such as a longer term: within the "
                                       "limits and verified, but the ratio is over the cap by 3 points or less.",
                "pend_for_documents": "Hold the file for income documents: within the limits, but the income is "
                                      "stated only or partly verified.",
                "deny_modification": "Deny: the prior-modification limits are not met, or the ratio on verified "
                                     "income is more than 3 points over the cap.",
            }),
            "ratio_within_cap": noul(
                "The post-modification debt-to-income ratio is within the programme cap.",
                yes="On verified income and all listed debts, the ratio is at or below the cap.",
                no="The ratio is above the cap, or it rests on income that has not been verified."),
            "income_dependability": score("How dependable is the income the modification would rely on?", [
                "Unreliable: irregular or unverified income.",
                "Uncertain: new or variable income with a short record.",
                "Steady: regular income with some recent change.",
                "Secure: long-standing, verified income.",
            ]),
            "redefault_likelihood": score(
                "How likely is the borrower to fall behind again within a year of a modification?", [
                    "Very low: a sound payment on stable income.",
                    "Low: some strain, but manageable.",
                    "Moderate: the budget is tight or the income is uncertain.",
                    "High: a thin margin plus a history of missed plans.",
                    "Very high: the payment is not sustainable on this income.",
                ]),
            "within_modification_limit": noul(
                "The borrower is within the programme's limit on prior modifications.",
                yes="No modification in the 3 years before the application, and at most one in the 5 years before it.",
                no="A modification within the last 3 years, or two within the last 5 years."),
        }


# =========================================================================== schema 3
class AutoLoanDeferral(Schema):
    name = "auto_loan_deferral"
    description = "An auto-loan borrower asks to defer payments; limits, collateral and the borrower's recovery all matter."
    surfaces = {"json": 0.45, "request_form": 0.3, "collector_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        behind = pick(rng, [1, 1, 2, 3])
        asked = pick(rng, [1, 2, 2, 3])
        used_12m = pick_w(rng, {0: 5, 1: 3, 2: 2})
        used_life = used_12m + pick(rng, [0, 0, 1, 2])
        balance = round(between(rng, 6000, 38000), -1)
        value = round(balance * between(rng, 0.7, 1.4), -2)
        payment = round(balance / between(rng, 30, 60), 0)
        insured = pick_w(rng, {"current": 6, "lapsed": 2, "unconfirmed": 2})
        cause = pick(rng, ["transmission repair kept me off work for two weeks", "short-term illness",
                           "a gap between jobs", "reduced shifts over the winter", "an unexpected medical bill"])
        income_back = pick_w(rng, {"back to normal": 4, "partly recovered": 3, "not yet": 3})
        needs_car_for_work = chance(rng, 0.7)
        loan_age = pick(rng, [4, 9, 18, 30, 44])
        docs = pick_w(rng, {"pay stub": 4, "none": 3, "doctor's note": 1, "repair invoice": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                asked, used_12m, used_life, insured, income_back, behind = 1, 0, 0, "current", "back to normal", 1
                value = round(balance * 1.25, -2)
            else:
                asked, used_12m, used_life, insured, income_back = 3, 2, 4, "lapsed", "not yet"
        elif reg == "borderline":
            used_12m = 1
            asked = pick(rng, [1, 2])          # policy: at most 2 deferred payments per 12 months
            used_life = pick(rng, [2, 3])      # and at most 4 over the life of the loan
            value = round(balance * near(rng, 1.0, 0.08), -2)
        else:
            if chance(rng, 0.5):
                income_back, docs, insured = "back to normal", "none", "unconfirmed"
            else:
                income_back, loan_age, used_12m = "not yet", 4, 0     # brand-new loan, early payment default
        used_life = max(used_life, used_12m)
        return {"regime": reg, "payments_behind": behind, "payments_requested": asked,
                "deferrals_used_12m": used_12m, "deferrals_used_life": used_life, "balance": balance,
                "vehicle_value": value, "monthly_payment": payment, "insurance": insured, "cause": cause,
                "income_status": income_back, "needs_car_for_work": needs_car_for_work, "loan_age_months": loan_age,
                "supporting_doc": docs, "borrower": person(rng), "account": ident(rng, "AUTO", 7, "-"),
                "vehicle": pick(rng, ["2019 compact sedan", "2021 small SUV", "2017 pickup", "2022 hatchback",
                                      "2018 minivan"]), "date": rand_date(rng)}

    RULES = ("Deferral policy: up to 2 payments per 12 months and 4 over the life of the loan; the account must "
             "not be more than 3 payments behind; proof of insurance must be current; the borrower must be able to "
             "resume regular payments.")

    def render(self, f, rng, surface):
        ms = _ms(rng)
        m = lambda x: money(x, style=ms)  # noqa: E731
        if surface == "json":
            return dumps(prune({
                "request": {"account": f["account"], "date": fmt_date(f["date"], style="iso"),
                            "payments_requested": f["payments_requested"], "reason": f["cause"],
                            "income_now": f["income_status"], "supporting_document": f["supporting_doc"],
                            "needs_vehicle_for_work": f["needs_car_for_work"]},
                "account": {"payments_behind": f["payments_behind"], "monthly_payment": f["monthly_payment"],
                            "balance": f["balance"], "loan_age_months": f["loan_age_months"],
                            "deferrals_last_12m": f["deferrals_used_12m"],
                            "deferrals_life_of_loan": f["deferrals_used_life"]},
                "collateral": {"vehicle": f["vehicle"], "estimated_value": f["vehicle_value"],
                               "insurance": f["insurance"]},
                "policy": self.RULES,
            }))
        if surface == "request_form":
            rows = [("Account", f["account"]), ("Date", fmt_date(f["date"], rng)),
                    ("Payments requested for deferral", f["payments_requested"]), ("Reason", f["cause"]),
                    ("Income now", f["income_status"]), ("Document attached", f["supporting_doc"]),
                    ("Vehicle needed for work", "yes" if f["needs_car_for_work"] else "no")]
            sys_rows = [("Payments behind", f["payments_behind"]), ("Monthly payment", m(f["monthly_payment"])),
                        ("Balance", m(f["balance"])), ("Vehicle", f"{f['vehicle']}, est. {m(f['vehicle_value'])}"),
                        ("Insurance", f["insurance"]), ("Loan age", f"{f['loan_age_months']} months"),
                        ("Deferrals (12m / life)", f"{f['deferrals_used_12m']} / {f['deferrals_used_life']}")]
            return ("PAYMENT DEFERRAL REQUEST (online form)\n" + kv_lines(rows, rng) + "\n\nSYSTEM DATA\n"
                    + kv_lines(sys_rows, rng) + f"\n\n{self.RULES}")
        note = [f"Spoke with {f['borrower'].split()[0]} re {f['account']}.",
                f"Wants to defer {f['payments_requested']} payment(s); {f['payments_behind']} behind now.",
                f"Reason: {f['cause']}. Says income is {f['income_status']}.",
                ("Needs the car to get to work." if f["needs_car_for_work"] else ""),
                f"Sent: {f['supporting_doc']}." if f["supporting_doc"] != "none" else "No documents sent.",
                {"current": "Insurance verified current.", "lapsed": "Insurance shows lapsed last month.",
                 "unconfirmed": "Could not confirm insurance."}[f["insurance"]],
                f"{f['vehicle']}, bal {m(f['balance'])} vs est value {m(f['vehicle_value'])}.",
                f"Prior deferrals: {f['deferrals_used_12m']} in last 12 mo, {f['deferrals_used_life']} total; "
                f"loan is {f['loan_age_months']} mo old."]
        return (f"{fmt_date(f['date'], rng)} collector note\n" + noisy(" ".join(x for x in note if x), rng, 0.15)
                + f"\n\n{self.RULES}")

    def questions(self):
        return {
            "deferral_decision": choice("How should the deferral request be answered?", {
                "grant_as_requested": "Defer every payment the borrower asked to defer.",
                "grant_fewer_payments": "Defer some of the payments, fewer than requested, within the policy limits.",
                "change_due_date": "Move the monthly due date to fit the borrower's pay cycle instead of deferring payments.",
                "decline_request": "Decline the deferral and continue normal collection activity.",
            }),
            "follow_up_needed": choice("What should the servicer ask for or do next on this account?", {
                "nothing_further": "Nothing further; the file is enough to act on.",
                "insurance_proof": "Ask the borrower for current proof of insurance on the vehicle.",
                "income_proof": "Ask for proof of the borrower's current income.",
                "specialist_review": "Pass the account to a collections specialist for a fuller review.",
            }),
            "collateral_exposure": score("How exposed is the lender's security in the vehicle?", [
                "Minimal: insured, and worth more than the balance.",
                "Modest: slight negative equity or a small coverage question.",
                "Significant: large negative equity or insurance in doubt.",
                "Severe: the vehicle may be uninsured and is worth far less than the balance.",
            ]),
            "hardship_support": score("How well does the file support the stated reason for the request?", [
                "Unsupported: nothing backs the reason given.",
                "Weakly: a plausible story, no documents.",
                "Reasonably: a document or the account history fits the story.",
                "Strongly: documents and history both confirm it.",
            ]),
            "can_resume_payments": noul(
                "The borrower will be able to resume regular payments after the deferral.",
                yes="Income is back or clearly returning, and the payment fits it.",
                no="Income has not recovered, or the file gives no reason to expect it will."),
        }


# =========================================================================== schema 4
class SmallBusinessRelief(Schema):
    name = "small_business_relief"
    description = "A small-business borrower asks for payment relief on a term loan after a revenue shock."
    surfaces = {"json": 0.45, "relief_application": 0.3, "credit_memo": 0.25}

    BUSINESSES = {"family restaurant": "Kitchen", "landscaping company": "Landscapes", "dental practice": "Dental",
                  "print shop": "Print & Copy", "clothing boutique": "Boutique", "auto repair garage": "Auto Repair",
                  "craft brewery": "Brewing Co.", "physiotherapy clinic": "Physiotherapy"}
    NAMES = ["Alder", "Basalt", "Cobalt", "Driftwood", "Ember", "Fernhill", "Juniper", "Lakeshore", "Oakridge",
             "Pinecrest", "Redfern", "Saltmarsh", "Valewood", "Westbrook", "Yarrow", "Ashgrove", "Bluestem", "Elmstead"]
    CAUSES = {"lost_client": "the loss of its largest client", "road_works": "a year of road works outside the premises",
              "seasonal": "a weaker-than-usual season", "supply_shock": "a sharp rise in the cost of key supplies",
              "owner_illness": "the owner's three-month absence through illness"}

    def sample_factors(self, rng):
        reg = regime(rng)
        cause = pick(rng, list(self.CAUSES))
        revenue_change = near(rng, -30, 18, lo=-75, hi=5, integer=True)
        dscr_before = round(between(rng, 1.15, 1.9), 2)
        dscr_now = round(max(0.2, dscr_before * (1 + revenue_change / 100 * 1.8)), 2)
        dscr_projected = near(rng, 1.0, 0.25, lo=0.3, hi=2.2)
        delinquent_days = pick(rng, [0, 0, 30, 60, 90])
        guarantor = pick_w(rng, {"personal guarantee, strong": 3, "personal guarantee, weak": 3, "none": 2})
        collateral = pick_w(rng, {"equipment, appraised above balance": 3, "equipment, appraised below balance": 3,
                                  "blanket lien only": 2})
        statements = pick_w(rng, {"interim P&L and tax return": 4, "tax return only": 3, "bank statements only": 2,
                                  "none yet": 1})
        cash_injection = pick_w(rng, {None: 5, 5000: 2, 20000: 1})
        other_lenders = pick_w(rng, {"granted relief": 2, "declined relief": 1, "not asked": 2, "unknown": 3})
        requested = pick(rng, ["6 months interest-only", "12-month term extension", "50% payment reduction for 6 months"])
        years = pick(rng, [2, 4, 7, 12, 20])
        if reg == "clear":
            if chance(rng, 0.5):
                dscr_projected, statements, guarantor, cause, years = 1.35, "interim P&L and tax return", "personal guarantee, strong", "road_works", 12
            else:
                dscr_projected, statements, guarantor, collateral, years = 0.55, "none yet", "none", "blanket lien only", 2
        elif reg == "borderline":
            dscr_projected = near(rng, 1.0, 0.08)
        else:
            if chance(rng, 0.5):
                dscr_projected, statements = round(between(rng, 1.15, 1.5), 2), "bank statements only"   # rosy, unbacked
            else:
                dscr_projected, other_lenders, guarantor = round(between(rng, 0.8, 0.95), 2), "granted relief", "personal guarantee, strong"
        business = pick(rng, list(self.BUSINESSES))
        return {"regime": reg, "business": business, "years_trading": years, "cause": cause,
                "revenue_change_6m_pct": revenue_change, "dscr_before": dscr_before, "dscr_now": dscr_now,
                "dscr_projected_12m": dscr_projected, "days_delinquent": delinquent_days, "guarantor": guarantor,
                "collateral": collateral, "financial_statements": statements, "owner_cash_injection": cash_injection,
                "other_lenders": other_lenders, "requested": requested,
                "balance": round(between(rng, 40000, 600000), -3),
                "borrower": f"{pick(rng, self.NAMES)} {self.BUSINESSES[business]}",
                "owner": person(rng), "officer": person(rng), "loan": ident(rng, "SBL", 6, "-"), "date": rand_date(rng)}

    RULES = ("Business relief: projected debt service coverage of at least 1.0x within 12 months, supported by "
             "financial statements no older than 6 months; relief of up to 6 months interest-only or a term extension of "
             "up to 24 months; exposures above $250,000 or without a guarantor go to the workout team.")

    def render(self, f, rng, surface):
        ms = _ms(rng)
        m = lambda x: money(x, style=ms)  # noqa: E731
        if surface == "json":
            return dumps(prune({
                "borrower": {"name": f["borrower"], "business": f["business"], "years_trading": f["years_trading"],
                             "owner": f["owner"]},
                "request": {"loan": f["loan"], "date": fmt_date(f["date"], style="iso"), "relief_requested": f["requested"],
                            "stated_cause": self.CAUSES[f["cause"]], "owner_cash_injection": f["owner_cash_injection"]},
                "performance": {"revenue_change_6m_pct": f["revenue_change_6m_pct"], "dscr_before": f["dscr_before"],
                                "dscr_now": f["dscr_now"], "dscr_projected_12m": f["dscr_projected_12m"],
                                "days_delinquent": f["days_delinquent"], "balance": f["balance"]},
                "support": {"guarantor": f["guarantor"], "collateral": f["collateral"],
                            "financial_statements": f["financial_statements"], "other_lenders": f["other_lenders"]},
                "policy": self.RULES,
            }))
        if surface == "relief_application":
            letter = noisy(join_sentences([
                f"{f['borrower']} is a {f['business']} that has traded for {f['years_trading']} years.",
                f"Revenue is down {abs(f['revenue_change_6m_pct'])}% over six months because of {self.CAUSES[f['cause']]}.",
                f"We are asking for {f['requested']}.",
                (f"I am putting in {m(f['owner_cash_injection'])} of my own money." if f["owner_cash_injection"] else ""),
                pick(rng, ["We expect to be back on track within the year.", "Bookings for next quarter look better.",
                           "We are cutting costs where we can."])]), rng)
            tbl = table(["measure", "value"], [["Debt service coverage (before)", f"{f['dscr_before']}x"],
                                               ["Debt service coverage (now)", f"{f['dscr_now']}x"],
                                               ["Projected in 12 months (borrower)", f"{f['dscr_projected_12m']}x"],
                                               ["Balance", m(f["balance"])], ["Days delinquent", f["days_delinquent"]],
                                               ["Statements provided", f["financial_statements"]]], rng)
            return (f"BUSINESS RELIEF APPLICATION {f['loan']}\n\n{letter}\n\n{f['owner']}, owner\n\n{tbl}\n\n"
                    f"Guarantor: {f['guarantor']}; collateral: {f['collateral']}; other lenders: {f['other_lenders']}.\n\n{self.RULES}")
        memo = [f"Borrower: {f['borrower']} ({f['business']}, {f['years_trading']} yrs). Loan {f['loan']}, balance {m(f['balance'])}.",
                f"Requests {f['requested']} after {self.CAUSES[f['cause']]}; revenue {f['revenue_change_6m_pct']}% over 6 months.",
                f"DSCR {f['dscr_before']}x -> {f['dscr_now']}x; borrower projects {f['dscr_projected_12m']}x in 12 months.",
                f"Statements on file: {f['financial_statements']}.",
                f"Guarantor: {f['guarantor']}. Collateral: {f['collateral']}.",
                f"Other lenders: {f['other_lenders']}.",
                (f"Owner offers {m(f['owner_cash_injection'])} cash injection." if f["owner_cash_injection"] else ""),
                f"Currently {f['days_delinquent']} days past due."]
        return f"CREDIT MEMO -- relief request\nPrepared by {f['officer']}\n\n" + bullets(memo, rng) + f"\n\nPolicy: {self.RULES}"

    def questions(self):
        return {
            "relief_structure": choice("Which relief structure should the lender offer?", {
                "interest_only_period": "A period of interest-only payments, then the original schedule resumes.",
                "term_extension": "Extend the loan term so the regular payment falls.",
                "temporary_reduction": "Cut the payment for a few months and recover the shortfall later.",
                "decline_relief": "Decline relief; the loan stays on its current terms.",
                "refer_to_workout": "Hand the loan to the workout team for a fuller restructuring.",
            }),
            "information_needed": choice("What should the lender obtain before finalising a decision?", {
                "none_needed": "Nothing further; the file supports a decision now.",
                "interim_financials": "Year-to-date profit and loss and a current balance sheet.",
                "cash_flow_forecast": "A month-by-month cash-flow forecast covering the relief period.",
                "guarantor_statement": "A current personal financial statement from the guarantor.",
            }),
            "business_viability": score("How viable does the business look over the next year?", [
                "Doubtful: projected coverage in 12 months is under 0.9x.",
                "Fragile: projected coverage of 0.9x to under 1.05x, or a higher projection backed only by bank "
                "statements or by nothing.",
                "Sound: projected 1.05x or more, backed by a tax return or interim accounts, without the Strong "
                "conditions.",
                "Strong: projected 1.25x or more, backed by a tax return or interim accounts, and coverage was 1.5x "
                "or more before the shock.",
            ]),
            "lender_protection": score("How well protected is the lender if the business fails despite relief?", [
                "Poorly: little collateral and no meaningful guarantee.",
                "Partly: collateral or a guarantee, but neither is strong.",
                "Well: solid collateral or a strong guarantee.",
                "Very well: collateral above the balance plus a strong guarantee.",
            ]),
            "projection_credible": noul(
                "The borrower's recovery projection is credible.",
                yes="The projected coverage is backed by recent statements and a plausible reason for recovery.",
                no="The projection is unsupported, stale, or at odds with the recent figures."),
        }


# =========================================================================== schema 5
class RepeatApplicant(Schema):
    name = "repeat_applicant_review"
    description = "A borrower who re-defaulted after earlier relief applies again while a foreclosure step is pending."
    surfaces = {"json": 0.45, "case_history": 0.3, "borrower_letter": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        prior = pick(rng, ["forbearance", "modification", "repayment plan", "forbearance then modification"])
        months_since = pick(rng, [5, 11, 17, 19, 26, 40])
        paid, due = rng.randint(2, 12), 12
        paid = min(paid, due)
        same_cause = chance(rng, 0.5)
        new_cause = pick(rng, ["a new job loss", "a spouse's serious illness", "storm damage to the house",
                               "a large rise in property tax and insurance"])
        days_to_sale = pick_w(rng, {None: 3, 20: 1, 35: 1, 40: 1, 75: 2})
        app_complete = pick_w(rng, {"complete": 3, "incomplete": 3})
        contact = pick_w(rng, {"responsive": 3, "intermittent": 3, "no contact until now": 2})
        income_trend = pick(rng, ["lower than at the last relief", "about the same", "higher than at the last relief"])
        partial_payments = pick(rng, [0, 0, 1, 3])
        if reg == "clear":
            if chance(rng, 0.5):
                same_cause, paid, contact, months_since, app_complete, days_to_sale = False, 12, "responsive", 26, "complete", 75
            else:
                same_cause, paid, contact, months_since, app_complete = True, 2, "no contact until now", 5, "incomplete"
        elif reg == "borderline":
            months_since = pick(rng, [17, 18, 19])             # the 18-month rule sits right at the edge
            days_to_sale = pick(rng, [35, 38, 40])            # and the 37-day protection window too
        else:
            if chance(rng, 0.5):
                same_cause, paid, partial_payments = True, 11, 3          # the same hardship, but a good-faith record
            else:
                same_cause, contact, paid = False, "no contact until now", 3
        return {"regime": reg, "prior_relief": prior, "months_since_prior_relief": months_since,
                "plan_payments_made": paid, "plan_payments_due": due, "same_cause_as_before": same_cause,
                "new_cause": None if same_cause else new_cause, "days_to_foreclosure_sale": days_to_sale,
                "application_status": app_complete, "contact_history": contact, "income_trend": income_trend,
                "partial_payments_since_default": partial_payments, "borrower": person(rng),
                "loan": ident(rng, "LN", 8, ""), "date": rand_date(rng), "prior_cause": pick(rng, [
                    "reduced hours", "a medical leave", "a business slowdown"])}

    RULES = ("Repeat relief: a new modification is not available within 18 months of the last one unless the default "
             "comes from a new hardship. A complete application received more than 37 days before a scheduled sale "
             "must be evaluated before the sale proceeds.")

    def render(self, f, rng, surface):
        ds = "iso" if surface == "json" else _ds(rng)
        sale = None if f["days_to_foreclosure_sale"] is None else fmt_date(
            f["date"] + timedelta(days=f["days_to_foreclosure_sale"]), style=ds)
        cause_now = (f"the same {f['prior_cause']} as before" if f["same_cause_as_before"] else f["new_cause"])
        if surface == "json":
            return dumps(prune({
                "application": {"loan": f["loan"], "received": fmt_date(f["date"], style="iso"),
                                "status": f["application_status"], "stated_cause": cause_now,
                                "income_vs_last_relief": f["income_trend"]},
                "prior_relief": {"type": f["prior_relief"], "months_ago": f["months_since_prior_relief"],
                                 "original_cause": f["prior_cause"],
                                 "plan_payments_made": f"{f['plan_payments_made']} of {f['plan_payments_due']}"},
                "collections": {"contact_history": f["contact_history"],
                                "partial_payments_since_default": f["partial_payments_since_default"]},
                "foreclosure": {"sale_date": sale} if sale else {"sale_scheduled": False},
                "programme_rules": self.RULES,
            }))
        if surface == "case_history":
            rows = [[f"{f['months_since_prior_relief']} months ago", f"{f['prior_relief']} granted ({f['prior_cause']})"],
                    ["during plan", f"{f['plan_payments_made']} of {f['plan_payments_due']} payments made"],
                    ["since default", f"{f['partial_payments_since_default']} partial payments; contact {f['contact_history']}"],
                    [fmt_date(f["date"], style=ds), f"new application received ({f['application_status']}); cause: {cause_now}"]]
            if sale:
                rows.append([sale, "foreclosure sale scheduled"])
            return (f"CASE HISTORY {f['loan']} -- {f['borrower']}\n" + table(["when", "event"], rows, rng)
                    + f"\n\nIncome compared with last relief: {f['income_trend']}.\n\n{self.RULES}")
        body = [f"I am applying again for help with my loan {f['loan']}.",
                (f"This time it is because of {f['new_cause']}, which is different from the {f['prior_cause']} last time."
                 if not f["same_cause_as_before"] else f"I am still dealing with the {f['prior_cause']} from before."),
                f"I made {f['plan_payments_made']} of the {f['plan_payments_due']} plan payments.",
                (f"I have sent {f['partial_payments_since_default']} partial payments since." if f["partial_payments_since_default"] else ""),
                f"My income is {f['income_trend']}.",
                (f"I understand a sale is scheduled for {sale}." if sale else ""),
                pick(rng, ["Please give us another chance.", "We want to keep our home.", "I will send anything you need."])]
        tail = kv_lines([("Application status", f["application_status"]), ("Prior relief", f["prior_relief"]),
                         ("Months since prior relief", f["months_since_prior_relief"]),
                         ("Contact history", f["contact_history"])], rng)
        return (f"Borrower letter received {fmt_date(f['date'], style=ds)}:\n\n" + noisy(join_sentences(body), rng)
                + f"\n\n{f['borrower']}\n\n--- servicer summary ---\n{tail}\n\n{self.RULES}")

    def questions(self):
        return {
            "repeat_relief_decision": choice("What should the servicer offer this returning applicant?", {
                "new_forbearance": "A new, short forbearance: the 18-month rule does not bar relief (18+ months, or a "
                                   "new hardship) and income is about the same as at the last relief.",
                "new_modification": "A new modification: the 18-month rule does not bar it and income is lower than "
                                    "at the last relief.",
                "short_repayment_plan": "A short repayment plan on the current terms: the 18-month rule does not bar "
                                        "relief and income is higher than at the last relief.",
                "decline_offer_exit_options": "Decline and set out exit options such as a sale: the same hardship "
                                              "within 18 months, with fewer than 9 of 12 plan payments made and fewer "
                                              "than 3 partial payments since.",
                "senior_review": "Send to a senior reviewer: the same hardship within 18 months, but 9 or more of 12 "
                                 "plan payments were made or 3 or more partial payments since default.",
            }),
            "hardship_is_new": noul(
                "The current default comes from a new hardship rather than the one the earlier relief addressed.",
                yes="The file describes a different event from the one behind the earlier relief.",
                no="It is the same difficulty, or the file does not show a new one."),
            "sale_must_pause": noul(
                "The scheduled foreclosure step must be paused while this application is evaluated.",
                yes="A sale is scheduled, the application is complete, and it arrived more than 37 days before the sale.",
                no="No sale is scheduled, the application is incomplete, or it arrived 37 days or fewer before the sale."),
            "plan_track_record": score("How well did the borrower keep to the earlier relief plan?", [
                "Poorly: most plan payments were missed.",
                "Unevenly: roughly half were made.",
                "Mostly: nearly all were made.",
                "Fully: every plan payment was made.",
            ]),
            "good_faith_evidence": score("How much does the file show the borrower acting in good faith?", [
                "Little: no contact or payments until the last moment.",
                "Some: occasional contact, few efforts to pay.",
                "Considerable: regular contact or partial payments.",
                "Strong: steady contact, partial payments and a complete application.",
            ]),
        }


# =========================================================================== schema 6
class DisasterAreaRelief(Schema):
    name = "disaster_area_relief"
    description = "A borrower in or near a declared disaster area asks for streamlined disaster forbearance."
    surfaces = {"json": 0.45, "intake_form": 0.3, "field_note": 0.25}

    EVENTS = ["river flooding", "a wildfire", "a hurricane", "a severe ice storm", "a tornado outbreak"]
    COUNTIES = ["Ashby", "Brookfield", "Carrow", "Dunmere", "Eastvale", "Fenwick", "Galloway", "Harwood"]

    def sample_factors(self, rng):
        reg = regime(rng)
        declared = rng.sample(self.COUNTIES, 3)
        others = [c for c in self.COUNTIES if c not in declared]
        location = pick_w(rng, {"inside": 5, "adjacent": 3, "split_zip": 2})
        county = pick(rng, declared) if location != "adjacent" else pick(rng, others)
        days_since = near(rng, 110, 60, lo=3, hi=200, integer=True)
        damage = pick_w(rng, {"none reported": 2, "minor": 3, "major": 3, "uninhabitable": 1})
        claim = pick_w(rng, {"filed": 4, "not filed": 3, "denied": 1})
        income_hit = pick_w(rng, {"workplace closed by the event": 3, "no change": 3, "reduced": 2})
        pre_delinquency = pick_w(rng, {0: 6, 30: 2, 60: 2})
        requested = pick(rng, [2, 3, 3, 6])
        occupancy = pick_w(rng, {"primary residence": 7, "rental property": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                location, damage, days_since, pre_delinquency, income_hit, occupancy = "inside", "major", 20, 0, "workplace closed by the event", "primary residence"
                county = declared[0]
            else:
                location, damage, income_hit, days_since, pre_delinquency = "adjacent", "none reported", "no change", 190, 60
                county = others[0]
        elif reg == "borderline":
            days_since = near(rng, 120, 6, lo=100, integer=True)       # the 120-day request window
            location = pick(rng, ["inside", "split_zip"])
            county = pick(rng, declared)
        else:
            if chance(rng, 0.5):
                location, damage, income_hit, county = "adjacent", "major", "workplace closed by the event", others[1]
            else:
                location, damage, income_hit, pre_delinquency = "inside", "none reported", "no change", 60
                county = declared[1]
        d = {"regime": reg, "event": pick(rng, self.EVENTS), "declared_counties": declared,
             "property_county": county, "location_status": location, "days_since_declaration": days_since,
             "damage": damage, "insurance_claim": claim, "income_impact": income_hit,
             "days_delinquent_at_declaration": pre_delinquency, "months_requested": requested,
             "occupancy": occupancy, "borrower": person(rng), "loan": ident(rng, "LN", 8, ""),
             "zip_note": ("the ZIP code straddles the declared boundary" if location == "split_zip" else None),
             "date": rand_date(rng)}
        # where the ZIP straddles the boundary, the parcel map settles which side the address is on
        d["parcel_side"] = pick(rng, ["declared", "undeclared"]) if location == "split_zip" else None
        if d["parcel_side"]:
            d["zip_note"] += ("; the parcel map puts this address inside the declared area" if d["parcel_side"] ==
                              "declared" else "; the parcel map puts this address outside the declared area")
        return d

    RULES = ("Streamlined disaster forbearance: up to 3 months with no documentation for owner-occupied homes in a "
             "declared county, if requested within 120 days of the declaration and the loan was less than 30 days "
             "delinquent on the declaration date. Other requests go to the standard hardship review.")

    def render(self, f, rng, surface):
        ds = "iso" if surface == "json" else _ds(rng)
        decl = fmt_date(f["date"] - timedelta(days=f["days_since_declaration"]), style=ds)
        if surface == "json":
            return dumps(prune({
                "declaration": {"event": f["event"], "declared_on": decl, "counties": f["declared_counties"]},
                "property": {"county": f["property_county"], "occupancy": f["occupancy"], "reported_damage": f["damage"],
                             "insurance_claim": f["insurance_claim"], "location_note": f["zip_note"]},
                "request": {"loan": f["loan"], "received": fmt_date(f["date"], style="iso"),
                            "months_requested": f["months_requested"], "income_impact": f["income_impact"],
                            "days_delinquent_at_declaration": f["days_delinquent_at_declaration"]},
                "programme_rules": self.RULES,
            }))
        if surface == "intake_form":
            rows = [("Loan", f["loan"]), ("Request received", fmt_date(f["date"], style=ds)),
                    ("Declared event", f"{f['event']} (declared {decl})"),
                    ("Declared counties", ", ".join(f["declared_counties"])),
                    ("Property county", f["property_county"] + (f" ({f['zip_note']})" if f["zip_note"] else "")),
                    ("Occupancy", f["occupancy"]), ("Damage reported", f["damage"]),
                    ("Insurance claim", f["insurance_claim"]), ("Effect on income", f["income_impact"]),
                    ("Days delinquent at declaration", f["days_delinquent_at_declaration"]),
                    ("Months requested", f["months_requested"])]
            return "DISASTER RELIEF INTAKE\n" + kv_lines(rows, rng) + f"\n\n{self.RULES}"
        note = [f"Call from {f['borrower']} ({f['loan']}) about {f['event']}.",
                f"Property is in {f['property_county']} County" + (f"; {f['zip_note']}" if f["zip_note"] else "") + ".",
                f"Declared counties: {', '.join(f['declared_counties'])} (declared {decl}).",
                f"Damage: {f['damage']}; insurance claim {f['insurance_claim']}.",
                f"Income: {f['income_impact']}.",
                f"Loan was {f['days_delinquent_at_declaration']} days down when the declaration was made.",
                f"Asks for {f['months_requested']} months. {f['occupancy'].capitalize()}."]
        return (f"FIELD NOTE -- call on {fmt_date(f['date'], style=ds)}\n" + noisy(" ".join(note), rng, 0.1)
                + f"\n\n{self.RULES}")

    def questions(self):
        return {
            "disaster_relief_path": choice("How should this disaster relief request be handled?", {
                "streamlined_forbearance": "Grant the streamlined forbearance: the event caused the difficulty and "
                                           "every streamlined condition is met (owner-occupied, inside the declared "
                                           "area, within 120 days, under 30 days behind at the declaration).",
                "standard_review": "Send it to the standard hardship review: the event caused the difficulty, a "
                                   "streamlined condition fails, and no filed claim on major damage is open.",
                "await_insurance_outcome": "Hold until the insurance claim is resolved: a streamlined condition "
                                           "fails, and a filed claim on major damage (or an uninhabitable home) is "
                                           "still open.",
                "no_disaster_relief": "Decline disaster relief: damage was minor or none and income did not change, "
                                      "so the difficulty does not arise from the event.",
            }),
            "inside_declared_area": noul(
                "The property lies within the area covered by the disaster declaration.",
                yes="The property's county is on the declaration list and the parcel map, where one is cited, puts it "
                    "inside the declared area.",
                no="The county is not on the list, or the parcel map puts the address outside the declared area."),
            "caused_by_disaster": noul(
                "The borrower's payment difficulty was caused by the declared event.",
                yes="The event did major damage or made the home uninhabitable, or closed the workplace or reduced "
                    "the borrower's income.",
                no="Damage was minor or none and income did not change."),
            "security_condition_concern": score(
                "How concerned should the servicer be about the property's condition as security for the loan?", [
                    "Not concerned: no damage reported.",
                    "Slightly: minor damage, likely covered.",
                    "Considerably: major damage, with coverage still open.",
                    "Seriously: the home is uninhabitable or the claim was denied.",
                ]),
            "supported_relief_length": score("How long a relief period does the file support?", [
                "None: the file supports no relief period.",
                "Short: one to three months.",
                "Medium: four to six months.",
                "Long: more than six months.",
            ]),
        }


WORKFLOW = Workflow(
    name="loan_hardship_requests",
    title="Loan hardship requests",
    description="Which relief, if any, a struggling borrower should get under a servicer's written hardship programme.",
    domain="consumer and small-business lending / loss mitigation",
    schemas=[MortgageForbearance(), ModificationReview(), AutoLoanDeferral(), SmallBusinessRelief(), RepeatApplicant(),
             DisasterAreaRelief()],
    notes="An application plus account data, not a support thread; no reply, churn or retention questions. No protected "
          "attribute drives a decision; health events appear only as the cause of a hardship.",
)
