"""Tax uncertain-position review: a group tax team decides how uncertain income-tax positions are recognised,
measured, updated, presented and disclosed in the financial statements.

Jurisdictions, tax statutes, authorities, companies and numbers are all FICTIONAL ("Valdoria", "Estmark", ...),
and every state quotes the written criteria that decide it: the group's uncertain-tax-position policy (a
more-likely-than-not recognition threshold, cumulative-probability measurement, settlement and statute rules,
presentation and disclosure rules) plus the fictional statutory rule in issue. Numbers are internally consistent:
probability tables sum to 100%, adjustments follow from the ranges shown, rollforwards add up (or the gap to the
ledger is a deliberate, computable difference).

No vendor bills, payment runs or procurement: the decisions are about a company's own tax provision.

Each schema carries `intended(f)`: the answer the written criteria point to for each question (choice key,
'yes'/'no', or a score level index). It is never rendered and never used as a label; it exists so the answer
spread can be checked for collapsed questions before a teacher labels the data.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

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
    lognormal_amount,
    near,
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    regime,
    score,
    table,
)

CURRENCIES = ["USD", "USD", "EUR", "GBP", "CAD", "AUD"]
SYM = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$"}
JURIS = ["Valdoria", "Estmark", "Corvania", "Lysandria", "Halvenia", "Ardenmoor", "Kestria", "Brevonia", "Solvane",
         "Marisca", "Tavenland", "Oskeria", "Quillon", "Drevana"]
AUTH_FORMS = ["{j} Revenue Service", "{j} Tax Agency", "Revenue Commission of {j}", "{j} Tax Administration"]
GROUP_A = ["Aldermoor", "Brackwater", "Castlebay", "Duncombe", "Elmhurst", "Farthing", "Galloway Point", "Heatherdale",
           "Inverleigh", "Juniper Hill", "Kingsbarn", "Lindenfield", "Marchmont", "Newhaven", "Otterburn", "Pendleton",
           "Queensferry", "Ravenscar", "Sandford", "Thornleigh", "Ulverston", "Vale Royal", "Whitcombe", "Yarrowby"]
GROUP_B = ["Group", "Holdings", "Industries", "plc", "International", "Brands", "Technologies"]


def _group(rng):
    return f"{pick(rng, GROUP_A)} {pick(rng, GROUP_B)}"


def _auth(rng, j):
    return pick(rng, AUTH_FORMS).format(j=j)


def kamt(k, cur, style):
    """An amount held in THOUSANDS, in one house style per state."""
    neg = "-" if k < 0 else ""
    k = abs(k)
    s = SYM.get(cur, cur + " ")
    if style == "m":
        return f"{neg}{s}{k / 1000:,.2f}m"
    if style == "k":
        return f"{neg}{s}{k:,.0f}k"
    return f"{neg}{s}{k * 1000:,.0f}"


def _st(rng):
    return pick(rng, ["m", "k", "full", "k"])


def _ds(rng):
    return pick(rng, ["iso", "long", "us", "eu", "iso"])


def _first(name):
    return name.split()[0]


def _r10(x):
    return int(round(x, -1)) if abs(x) >= 100 else int(round(x))


def _period_end(rng):
    y = pick(rng, [2025, 2026])
    m = pick(rng, [3, 6, 9, 12, 12])
    return date(y, m, calendar.monthrange(y, m)[1])


# =========================================================================== schema 1
ISSUES = {
    "participation_exemption": {
        "title": "exemption claimed on the gain from selling its shares in {sub}",
        "rule": "Gains on disposals of shares are exempt where the seller held at least 10% of the shares for at "
                "least 12 continuous months before the disposal.",
        "facts": {"clear": "The company held 35% of {sub} for four years before the sale.",
                  "arguable": "The company held 10.4% of {sub} for 12 months and 9 days; for six weeks during a rights "
                              "issue its holding was diluted to 9.7% until it took up its rights.",
                  "poor": "The company held 8.9% of {sub} directly and a further 2% through a fund it does not "
                          "control, for 14 months."}},
    "research_deduction": {
        "title": "150% research deduction claimed for its {proj} project",
        "rule": "Qualifying research spending is deductible at 150%. The work must seek an advance in science or "
                "technology by resolving a technological uncertainty; routine development with established methods "
                "does not qualify.",
        "facts": {"clear": "The project developed a new battery-cell chemistry; lab records document the uncertainties "
                           "and the approaches that failed.",
                  "arguable": "The project built a recommendation engine; some work used novel model designs but most "
                              "was integration with existing systems, and records are partial.",
                  "poor": "The project upgraded the company's customer-relationship system to a newer version of a "
                          "commercial package."}},
    "management_fees": {
        "title": "deduction of the intra-group management fee paid to its parent",
        "rule": "Charges for intra-group services are deductible only to the extent the services give the payer an "
                "economic benefit it would pay an independent party for; costs of the parent's own shareholder "
                "activities are not deductible.",
        "facts": {"clear": "The parent's shared payroll and IT team runs the company's payroll and systems; time "
                           "records and service reports exist.",
                  "arguable": "The fee covers 'strategic support' and group treasury; about half is backed by time "
                              "records, the rest is allocated by headcount.",
                  "poor": "The fee is 4% of turnover for 'brand oversight', mainly the cost of the parent's board and "
                          "investor relations."}},
    "interest_cap": {
        "title": "exclusion of certain finance costs from the interest-deductibility cap",
        "rule": "Net interest expense is deductible up to 30% of tax-adjusted EBITDA; any excess is carried forward "
                "for five years.",
        "facts": {"clear": "The excluded amounts are commitment fees on an undrawn facility, which the statute's "
                           "definition lists as not being interest.",
                  "arguable": "The excluded amounts are guarantee fees paid to the parent for guaranteeing the bank "
                              "loan; the statute is silent on guarantee fees.",
                  "poor": "The excluded amount is the discount on a zero-coupon note issued to an affiliate, "
                          "economically the same as interest."}},
    "taxable_presence": {
        "title": "position that it has no taxable presence in {j2}",
        "rule": "A foreign company has a taxable presence here if a person habitually concludes contracts on its "
                "behalf here, or if it has a fixed place of business here.",
        "facts": {"clear": "Two sales staff in {j2} generate leads; every contract is negotiated and signed at head "
                           "office.",
                  "arguable": "Sales staff in {j2} negotiate terms with customers; head office signs, but has never "
                              "changed terms the staff agreed.",
                  "poor": "The {j2} country manager signs customer contracts locally under a standing power of "
                          "attorney and works from a rented office."}},
    "royalty_withholding": {
        "title": "5% treaty rate applied to withholding tax on royalties paid to an affiliate",
        "rule": "Royalties paid abroad bear 20% withholding tax, reduced to 5% under the treaty if the recipient is "
                "the beneficial owner and not a conduit.",
        "facts": {"clear": "The recipient owns and develops the licensed technology and employs 40 engineers.",
                  "arguable": "The recipient holds the technology, employs three people and pays about 60% of what it "
                              "receives on to its own parent.",
                  "poor": "The recipient has no employees and passes 98% of the royalties to an affiliate in a "
                          "non-treaty country within days."}},
    "loss_continuity": {
        "title": "use of tax losses brought forward after its change of ownership",
        "rule": "Losses carried forward are forfeited after a change in ownership of more than 50% unless the "
                "company carries on the same business for 24 months after the change.",
        "facts": {"clear": "After the 60% ownership change the company carried on the same contract-catering business "
                           "with the same customers.",
                  "arguable": "After the change the company kept its catering business but moved half its revenue to "
                              "event management in month 20.",
                  "poor": "Six months after the change the company closed its catering operations and began letting "
                          "property."}},
    "first_year_allowance": {
        "title": "first-year allowance claimed on its new press line",
        "rule": "First-year allowances are available on equipment in use or ready for use before the year end.",
        "facts": {"clear": "The press line was installed, tested and produced saleable output in the last month of "
                           "the year.",
                  "arguable": "The press line was delivered and installed before the year end; commissioning tests "
                              "finished nine days after it.",
                  "poor": "The press line was paid for before the year end but was still at the supplier's factory "
                          "at the year end."}},
    "restructuring_provision": {
        "title": "deduction of its restructuring provision",
        "rule": "A provision is deductible only when the obligation is fixed and its amount can be determined with "
                "reasonable accuracy; general provisions are not deductible.",
        "facts": {"clear": "Redundancy notices had gone to named employees and amounts were agreed with the unions "
                           "before the year end.",
                  "arguable": "The plan was announced to staff before the year end; individual amounts were agreed "
                              "only after it.",
                  "poor": "The board approved a restructuring in principle; no one has been told and the amounts are "
                          "estimates."}},
    "hybrid_interest": {
        "title": "deduction of interest on the instrument issued to its sister company",
        "rule": "Interest is not deductible to the extent the corresponding receipt goes untaxed in the lender's "
                "jurisdiction because of how that jurisdiction classifies the instrument.",
        "facts": {"clear": "The lender is taxed on the interest in full at home; a tax certificate confirms it.",
                  "arguable": "The lender's country treats the instrument as debt, but the lender shelters the income "
                              "with group relief; no certificate has been obtained.",
                  "poor": "The lender's country treats the instrument as equity and exempts the receipts as "
                          "dividends."}},
    "bad_debt": {
        "title": "deduction for debts written off in the year",
        "rule": "Bad debts are deductible once written off, provided reasonable steps were taken to recover them.",
        "facts": {"clear": "The customer is in liquidation and the liquidator has confirmed unsecured creditors will "
                           "receive nothing.",
                  "arguable": "The debt was written off after two reminder letters; the customer still trades but "
                              "disputes the amount.",
                  "poor": "The debt is owed by a solvent affiliate and no recovery steps were taken."}},
    "software_costs": {
        "title": "immediate deduction of its software spending",
        "rule": "Costs of acquiring or creating an asset of enduring benefit are capital; costs of maintaining "
                "existing assets are deductible.",
        "facts": {"clear": "The costs were annual licence renewals and routine maintenance of existing systems.",
                  "arguable": "The costs configured a cloud ERP system the company does not own but will use for at "
                              "least five years.",
                  "poor": "The costs built a proprietary trading platform the company owns and will use for ten "
                          "years."}},
}
SUBS = ["Nordvik Components", "Teller Freight", "Halcyon Media", "Brightmoor Foods", "Corran Medical", "Velstra Retail",
        "Ostler Engineering", "Pellam Insurance Services"]
PROJECTS = ["Atlas", "Kestrel", "Bluefin", "Orchard", "Lantern", "Meridian", "Quartz", "Tamarisk"]


class PositionRecognitionReview(Schema):
    name = "position_recognition_review"
    description = ("A filing position taken on a tax return: is it more likely than not to be sustained on its "
                   "technical merits, and should any benefit be recognised?")
    surfaces = {"json": 0.5, "adviser_memo": 0.25, "provision_review_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        issue = pick(rng, list(ISSUES))
        fit = pick_w(rng, {"clear": 3, "arguable": 4, "poor": 3})
        case_law = pick_w(rng, {"supportive": 2, "mixed": 3, "adverse": 2, "none": 3})
        guidance = pick_w(rng, {"supportive": 2, "silent": 4, "contrary": 2})
        prior = pick_w(rng, {"accepted": 2, "challenged": 2, "none": 4})
        opinion = pick_w(rng, {"should": 2, "more_likely_than_not": 3, "reasonable_basis": 2, None: 2})
        if reg == "clear":
            if chance(rng, 0.5):
                fit, case_law, guidance, opinion = "clear", "supportive", pick(rng, ["supportive", "silent"]), "should"
            else:
                fit, case_law, guidance = "poor", pick(rng, ["adverse", "none"]), "contrary"
                opinion = pick(rng, ["reasonable_basis", None])
        elif reg == "conflicting":
            # a confident opinion against weak facts, or good facts with hostile guidance
            if chance(rng, 0.5):
                fit, opinion = "poor", "should"
            else:
                fit, guidance, case_law = "clear", "contrary", pick(rng, ["mixed", "adverse"])
        split = chance(rng, 0.38) and fit != "poor"
        j = pick(rng, JURIS)
        return {"regime": reg, "group": _group(rng), "entity": f"{pick(rng, GROUP_A)} {j} Ltd", "jurisdiction": j,
                "authority": _auth(rng, j), "other_jurisdiction": pick(rng, [x for x in JURIS if x != j]),
                "issue": issue, "statutory_fit": fit, "case_law": case_law, "guidance": guidance,
                "prior_examination": prior, "opinion": opinion,
                "opinion_relies_on_unverified_facts": opinion is not None and chance(rng, 0.3),
                "adviser_designed_structure": opinion is not None and chance(rng, 0.25),
                "detection_argument": chance(rng, 0.35), "two_year_split": split,
                "benefit_k": _r10(lognormal_amount(rng, 1800, 0.9)), "currency": pick(rng, CURRENCIES),
                "sub": pick(rng, SUBS), "proj": pick(rng, PROJECTS), "year": pick(rng, [2024, 2025, 2026]),
                "preparer": person(rng), "reviewer": person(rng), "adviser_firm": f"{pick(rng, GROUP_A)} Tax LLP",
                "ruling_year": rng.randint(2019, 2025)}

    @staticmethod
    def merit(f, fit=None):
        fit = fit or f["statutory_fit"]
        p = {"clear": 2, "arguable": 1, "poor": 0}[fit]
        p += {"supportive": 1, "mixed": 0, "adverse": -1, "none": 0}[f["case_law"]]
        p += {"supportive": 1, "silent": 0, "contrary": -1}[f["guidance"]]
        p += {"accepted": 1, "challenged": -1, "none": 0}[f["prior_examination"]]
        return 0 if p <= -1 else 1 if p == 0 else 2 if p == 1 else 3 if p == 2 else 4

    @staticmethod
    def reliance(f):
        if f["opinion"] is None:
            return 0
        r = {"should": 3, "more_likely_than_not": 2, "reasonable_basis": 1}[f["opinion"]]
        r -= 1 if f["opinion_relies_on_unverified_facts"] else 0
        r -= 1 if f["adviser_designed_structure"] else 0
        return max(0, r)

    def meets(self, f, m):
        # sound or strong merits meet the threshold; a balanced case is tipped over only by a reliable opinion
        return m >= 3 or (m == 2 and self.reliance(f) >= 2)

    def intended(self, f):
        m = self.merit(f)
        m2 = self.merit(f, "poor") if f["two_year_split"] else m
        ok = self.meets(f, m)
        if f["two_year_split"] and ok and not self.meets(f, m2):
            call = "split_by_year"
        elif ok:
            call = "recognise_measured_benefit"
        elif m == 2:
            call = "obtain_further_support"
        else:
            call = "recognise_no_benefit"
        return {"recognition_call": call, "technical_merit": m, "opinion_reliance": self.reliance(f),
                "threshold_met": "yes" if (ok and self.meets(f, m2)) else "no",
                "relies_on_detection_risk": "yes" if f["detection_argument"] else "no"}

    POLICY = {
        "UTP-1": "Recognise the benefit of a tax position only if it is more likely than not to be sustained on its "
                 "technical merits, assuming the authority examines it with full knowledge of all relevant facts.",
        "UTP-2": "The chance that the position will not be examined, or will be overlooked, must be ignored.",
        "UTP-3": "Weigh the statute, case law, the authority's published guidance and its past treatment of the "
                 "same position. A professional opinion is evidence, but carries less weight where it rests on facts "
                 "not verified or comes from the adviser who designed the arrangement.",
        "UTP-4": "Evaluate separately any part of a position (for example, a tax year) whose facts differ.",
    }

    def _texts(self, f):
        iss = ISSUES[f["issue"]]
        fmt = {"sub": f["sub"], "proj": f["proj"], "j2": f["other_jurisdiction"]}
        title = iss["title"].format(**fmt)
        facts = iss["facts"][f["statutory_fit"]].format(**fmt)
        if f["two_year_split"]:
            facts = (f"For the {f['year'] - 1} return: {facts} For the {f['year']} return the facts differ: "
                     + iss["facts"]["poor"].format(**fmt))
        a, j = f["authority"], f["jurisdiction"]
        law = {"supportive": f"A {j} appeal court allowed a closely similar claim in {f['ruling_year']}.",
               "mixed": "Two lower-court decisions on similar facts went opposite ways.",
               "adverse": f"The {j} supreme court rejected a similar claim in {f['ruling_year']}, on facts that "
                          f"differ in some respects.",
               "none": "There is no case law on the point."}[f["case_law"]]
        guid = {"supportive": f"The {a} has published guidance accepting this treatment.",
                "silent": f"The {a} has published nothing on the point.",
                "contrary": f"The {a}'s published manual takes the opposite view (manuals do not bind the courts)."}[f["guidance"]]
        prior = {"accepted": f"The {a} examined the same treatment in an earlier year and accepted it.",
                 "challenged": f"The {a} is challenging a similar position at another group company.",
                 "none": None}[f["prior_examination"]]
        if f["opinion"] is None:
            op = "No external opinion has been obtained."
        else:
            lvl = {"should": "'should' be sustained", "more_likely_than_not": "is 'more likely than not' to be sustained",
                   "reasonable_basis": "has 'a reasonable basis'"}[f["opinion"]]
            op = f"{f['adviser_firm']} concludes the position {lvl}"
            if f["opinion_relies_on_unverified_facts"]:
                op += ", relying on management representations it has not tested"
            op += "."
            if f["adviser_designed_structure"]:
                op += f" {f['adviser_firm']} also designed the arrangement."
        det = None
        if f["detection_argument"]:
            det = (f"The preparer notes the {a} 'has examined this kind of item only twice in ten years', so the "
                   f"risk is low.")
        return title, iss["rule"], facts, law, guid, prior, op, det

    def render(self, f, rng, surface):
        cur, st = f["currency"], _st(rng)
        title, rule, facts, law, guid, prior, op, det = self._texts(f)
        if surface == "json":
            return dumps(prune({
                "position": {"entity": f["entity"], "group": f["group"], "jurisdiction": f["jurisdiction"],
                             "description": title, "tax_benefit": kamt(f["benefit_k"], cur, st),
                             "return_year": f["year"]},
                "statute": rule, "facts": facts,
                "authorities": {"case_law": law, "guidance": guid, "examination_history": prior},
                "opinion": op, "preparer_note": det, "group_policy": self.POLICY,
            }))
        if surface == "adviser_memo":
            return "\n".join([
                f"{f['adviser_firm'].upper()} -- MEMORANDUM (extract)" if f["opinion"] else
                f"INTERNAL TAX MEMO -- {f['group']}",
                f"Re: {f['entity']} -- {title} ({f['year']} return); tax at stake {kamt(f['benefit_k'], cur, st)}", "",
                "Law: " + rule, "", "Facts: " + noisy(facts, rng, 0.05), "",
                "Authorities: " + join_sentences([law, guid, prior]), "",
                "Conclusion of adviser: " + op, "",
                ("Preparer's covering note: " + det) if det else "", "",
                "Group policy applied:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        return "\n".join([
            f"UTP REVIEW NOTE -- {f['group']} provision, {f['jurisdiction']}",
            f"Preparer: {f['preparer']}   Reviewer: {f['reviewer']}", "",
            kv_lines([("Entity", f["entity"]), ("Position", title), ("Return", f["year"]),
                      ("Benefit claimed", kamt(f["benefit_k"], cur, st)), ("Authority", f["authority"])], rng), "",
            bullets([f"Statute: {rule}", f"Facts: {facts}", law, guid, prior, op, det], rng), "",
            "Policy: " + " ".join(f"[{k}] {v}" for k, v in self.POLICY.items()),
        ])

    def questions(self):
        return {
            "recognition_call": choice("What should the provision recognise for this position?", {
                "obtain_further_support": "Nothing yet: the merits are finely balanced and no reliable opinion "
                                          "tips them, so obtain a proper technical opinion first.",
                "recognise_measured_benefit": "Recognise a benefit, measured under the group's measurement rules.",
                "recognise_no_benefit": "Recognise no benefit; reserve the full amount.",
                "split_by_year": "Treat the years separately: recognise a benefit for the year that meets the "
                                 "threshold and none for the other.",
            }),
            "technical_merit": score("How strong are the position's technical merits on the evidence presented?", [
                "Weak: unlikely to be sustained.",
                "Arguable: some support, but less likely than not.",
                "Balanced: about as likely to fail as to succeed.",
                "Sound: more likely than not to be sustained.",
                "Strong: should be sustained.",
            ]),
            "opinion_reliance": score("How much weight can the provision place on the external opinion?", [
                "None: there is no opinion, or it cannot be relied on.",
                "Little: a low-level or heavily qualified opinion.",
                "Moderate: a reasonable opinion with some limits.",
                "High: a strong, independent opinion on verified facts.",
            ]),
            "threshold_met": noul(
                "For every return it covers, the position is more likely than not to be sustained on its technical "
                "merits.",
                yes="On the statute, case law, guidance, examination history and any reliable opinion, success is "
                    "more likely than failure for each return covered.",
                no="For at least one return covered, that evidence leaves the position evenly balanced or likely to "
                   "fail."),
            "relies_on_detection_risk": noul(
                "The analysis in the file leans on the chance that the authority will not examine the position."),
        }


# =========================================================================== schema 2
FRACTIONS = [1.0, 0.85, 0.7, 0.6, 0.4, 0.3, 0.15, 0.0]
MEAS_ISSUES = ["the transfer price for parts sold to a sister company", "the useful life claimed for its network "
               "assets", "the allocation of head-office costs to its branch", "the valuation of patents moved to a "
               "group company", "the split of a settlement payment between deductible and non-deductible parts",
               "the amount of research spending treated as qualifying", "the market value used on an intra-group "
               "share transfer", "the timing of revenue on long-term contracts for tax purposes"]


class BenefitMeasurement(Schema):
    name = "benefit_measurement"
    description = ("An uncertain position's range of settlement outcomes: how much benefit to recognise under the "
                   "cumulative-probability rule, and what interest and penalties to accrue.")
    surfaces = {"json": 0.5, "measurement_workpaper": 0.25, "tax_email": 0.25}

    def _probs(self, rng, n, reg):
        while True:
            w = [rng.uniform(0.2, 1.0) for _ in range(n)]
            ps = [max(5, int(round(x / sum(w) * 20)) * 5) for x in w]
            diff = 100 - sum(ps)
            ps[ps.index(max(ps))] += diff
            if min(ps) < 5:
                continue
            cum, ok, near50 = 0, True, False
            for p in ps:
                cum += p
                if cum == 50:
                    ok = False
                if 50 < cum <= 60:
                    near50 = True
            if not ok:
                continue
            if reg == "borderline" and not near50 and rng.random() < 0.7:
                continue
            return ps

    def sample_factors(self, rng):
        reg = regime(rng)
        b = _r10(lognormal_amount(rng, 2500, 0.8))
        n = pick(rng, [3, 4, 4, 5])
        fr = sorted(rng.sample(FRACTIONS[1:-1], n - 2) + [1.0, pick(rng, [0.0, 0.0, 0.15])], reverse=True)
        fr = sorted(set(fr), reverse=True)
        while len(fr) < n:
            extra = pick(rng, [x for x in FRACTIONS if x not in fr])
            fr = sorted(set(fr + [extra]), reverse=True)
        probs = self._probs(rng, len(fr), reg)
        dom = pick_w(rng, {"none": 7, "full": 1.6, "nil": 1.4})
        if dom == "full" or (dom == "nil" and fr[-1] != 0.0):
            top = pick(rng, [55, 60, 65, 70])
            rest = self._probs(rng, len(fr) - 1, "clear")
            rest = [max(5, int(round(x * (100 - top) / 100 / 5)) * 5) for x in rest]
            probs = [top] + rest
            probs[1] += 100 - sum(probs)
        elif dom == "nil":
            top = pick(rng, [55, 60, 65])
            rest = self._probs(rng, len(fr) - 1, "clear")
            rest = [max(5, int(round(x * (100 - top) / 100 / 5)) * 5) for x in rest]
            probs = rest + [top]
            probs[0] += 100 - sum(probs)
        if min(probs) < 5:
            probs = self._probs(rng, len(fr), "clear")
        method = pick_w(rng, {"cumulative": 3, "expected_value": 2.5, "most_likely": 2, "full": 1.2, "nil": 0.6})
        basis = pick_w(rng, {"judgement": 2, "adviser": 3, "adviser_with_data": 3, "settlement_record": 2})
        n_settled = {"judgement": 0, "adviser": 0, "adviser_with_data": rng.randint(2, 6),
                     "settlement_record": rng.randint(6, 30)}[basis]
        if reg == "clear":
            method = "cumulative" if chance(rng, 0.6) else pick(rng, ["full", "expected_value"])
            basis, n_settled = ("settlement_record", rng.randint(12, 30)) if chance(rng, 0.6) else ("judgement", 0)
        disclosed = chance(rng, 0.55)
        reasonable = chance(rng, 0.6)
        j = pick(rng, JURIS)
        return {"regime": reg, "group": _group(rng), "jurisdiction": j, "authority": _auth(rng, j),
                "issue": pick(rng, MEAS_ISSUES), "benefit_k": b, "outcome_fractions": fr, "probabilities": probs,
                "preparer_method": method, "probability_basis": basis, "settled_cases": n_settled,
                "disclosed_on_return": disclosed, "reasonable_basis": reasonable,
                "interest_rate_pct": pick(rng, [4.0, 5.5, 6.0, 7.25, 8.0]), "years_outstanding": pick(rng, [1, 2, 3]),
                "penalty_full_pct": pick(rng, [20, 25, 30]), "penalty_reduced_pct": pick(rng, [5, 10]),
                "currency": pick(rng, CURRENCIES), "preparer": person(rng), "director": person(rng),
                "period_end": _period_end(rng)}

    @staticmethod
    def amounts(f):
        b = f["benefit_k"]
        outs = [_r10(b * x) for x in f["outcome_fractions"]]
        cum, correct, cum_at = 0, outs[-1], None
        for a, p in zip(outs, f["probabilities"]):
            cum += p
            if cum > 50:
                correct, cum_at = a, cum
                break
        ev = _r10(sum(a * p for a, p in zip(outs, f["probabilities"])) / 100)
        ml = max(zip(f["probabilities"], outs))[1]
        prop = {"cumulative": correct, "expected_value": ev, "most_likely": ml, "full": b, "nil": 0}[f["preparer_method"]]
        return {"outcomes": outs, "correct": correct, "cum_at": cum_at, "proposed": prop}

    @staticmethod
    def support(f):
        return {"judgement": 0, "adviser": 1, "adviser_with_data": 2,
                "settlement_record": 3 if f["settled_cases"] >= 10 else 2}[f["probability_basis"]]

    def intended(self, f):
        a = self.amounts(f)
        b = f["benefit_k"]
        share = a["correct"] / b
        lvl = 0 if a["correct"] == 0 else 1 if share < 0.25 else 2 if share < 0.5 else 3 if share < 0.75 else 4
        sup = self.support(f)
        if sup == 0 and a["cum_at"] is not None and a["cum_at"] <= 60:
            adj = "rebuild_probability_table"
        elif a["proposed"] == a["correct"]:
            adj = "accept_as_proposed"
        elif a["proposed"] > a["correct"]:
            adj = "increase_liability"
        else:
            adj = "decrease_liability"
        if a["correct"] >= b:
            pen = "no_accrual"
        elif f["disclosed_on_return"] and f["reasonable_basis"]:
            pen = "interest_only"
        elif f["disclosed_on_return"]:
            pen = "interest_and_reduced_penalty"
        else:
            pen = "interest_and_full_penalty"
        return {"measured_share": lvl, "probability_support": sup,
                "preparer_figure_correct": "yes" if a["proposed"] == a["correct"] else "no",
                "liability_adjustment": adj, "interest_penalty_accrual": pen}

    def _policy(self, f):
        return {
            "UTP-5": "Measure a recognised position at the largest amount of benefit whose cumulative probability of "
                     "being realised on settlement exceeds 50%, working down from the largest possible outcome.",
            "UTP-6": "Probabilities must be supported by evidence such as settlement history with the authority or "
                     "a specialist's analysis; unsupported judgement near the 50% line must be rebuilt.",
            "UTP-7": f"Accrue interest on any unrecognised benefit from the original due date. Statutory penalty: "
                     f"{f['penalty_full_pct']}% of the underpayment; {f['penalty_reduced_pct']}% if the position was "
                     f"disclosed on the return; nil if it was disclosed and had at least a reasonable basis.",
        }

    def _basis_text(self, f):
        return {"judgement": "Probabilities are the tax manager's own judgement.",
                "adviser": "Probabilities come from the external adviser's view; no settlement data was used.",
                "adviser_with_data": f"Adviser's estimate, informed by {f['settled_cases']} similar cases settled with "
                                     f"the {f['authority']}.",
                "settlement_record": f"Derived from the group's record of {f['settled_cases']} settlements of this "
                                     f"issue with the {f['authority']}."}[f["probability_basis"]]

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        a = self.amounts(f)
        pol = self._policy(f)
        rows = [[kamt(x, cur, st), f"{p}%"] for x, p in zip(a["outcomes"], f["probabilities"])]
        prop_txt = {"cumulative": "cumulative-probability approach", "expected_value": "probability-weighted average",
                    "most_likely": "single most likely outcome", "full": "full benefit, as filed",
                    "nil": "nil, pending further review"}[f["preparer_method"]]
        ret = (("Disclosed on the return" if f["disclosed_on_return"] else "Not disclosed on the return")
               + "; the adviser's view is that the position " +
               ("has at least a reasonable basis." if f["reasonable_basis"] else "does not reach a reasonable basis."))
        yrs = f["years_outstanding"]
        interest = (f"Interest rate {f['interest_rate_pct']}% a year; the position relates to returns filed "
                    f"{'a year' if yrs == 1 else str(yrs) + ' years'} ago.")
        if surface == "json":
            return dumps(prune({
                "position": {"group": f["group"], "jurisdiction": f["jurisdiction"], "issue": f["issue"],
                             "benefit_claimed": kamt(f["benefit_k"], cur, st), "period_end": f["period_end"].isoformat()},
                "settlement_outcomes": [{"benefit_sustained": kamt(x, cur, st), "probability_pct": p}
                                        for x, p in zip(a["outcomes"], f["probabilities"])],
                "probability_basis": self._basis_text(f),
                "proposal": {"recognised_benefit": kamt(a["proposed"], cur, st), "method": prop_txt,
                             "prepared_by": f["preparer"]},
                "return_and_penalties": {"filing": ret, "interest": interest},
                "group_policy": pol,
            }))
        if surface == "measurement_workpaper":
            return "\n".join([
                f"UTP MEASUREMENT -- {f['group']} / {f['jurisdiction']} -- period ended {fmt_date(f['period_end'], rng, ds)}",
                f"Issue: {f['issue']}. Benefit claimed: {kamt(f['benefit_k'], cur, st)}.", "",
                table(["Benefit sustained on settlement", "Probability"], rows, rng), "",
                self._basis_text(f),
                f"Proposed recognised benefit: {kamt(a['proposed'], cur, st)} ({prop_txt}) -- {f['preparer']}", "",
                ret, interest, "", "Policy:", "\n".join(f"  {k} {v}" for k, v in pol.items()),
            ])
        outs = "; ".join(f"{kamt(x, cur, st)} at {p}%" for x, p in zip(a["outcomes"], f["probabilities"]))
        return "\n".join([
            f"Subject: {f['jurisdiction']} UTP -- {f['issue']}", f"From: {f['preparer']}", f"To: {f['director']}", "",
            f"Hi {_first(f['director'])},", "",
            noisy(join_sentences([
                f"For the period ended {fmt_date(f['period_end'], rng, ds)} I've measured the position on {f['issue']} "
                f"(benefit claimed {kamt(f['benefit_k'], cur, st)}).",
                f"Possible settlement outcomes: {outs}.", self._basis_text(f),
                f"I'm proposing to recognise {kamt(a['proposed'], cur, st)}, using the {prop_txt}.", ret, interest]),
                rng, 0.12), "",
            "Policy for reference: " + " ".join(f"{k}: {v}" for k, v in pol.items()), "",
            _first(f["preparer"]),
        ])

    def questions(self):
        return {
            "measured_share": score("What share of the claimed benefit should be recognised under the "
                                    "measurement rule?", [
                "None of it.",
                "Less than a quarter.",
                "A quarter to under a half.",
                "A half to under three quarters.",
                "Three quarters or more.",
            ]),
            "probability_support": score("How well supported are the outcome probabilities?", [
                "Unsupported: one person's judgement.",
                "Thin: an adviser's view without settlement data.",
                "Reasonable: some settlement experience behind them.",
                "Strong: a substantial record of settlements on this issue.",
            ]),
            "preparer_figure_correct": noul(
                "The preparer's proposed recognised benefit is the amount the cumulative-probability rule gives.",
                yes="Working down from the largest outcome, the first amount whose cumulative probability passes 50% "
                    "equals the proposed figure.",
                no="The proposed figure is an average, the single most likely outcome, the full or nil amount, or "
                   "otherwise differs from what the rule gives."),
            "liability_adjustment": choice("What should happen to the proposed uncertain-tax liability?", {
                "accept_as_proposed": "Accept the proposed figure.",
                "decrease_liability": "Reduce the liability: more benefit should be recognised than proposed.",
                "increase_liability": "Increase the liability: less benefit should be recognised than proposed.",
                "rebuild_probability_table": "Send the probability table back to be rebuilt with evidence before "
                                             "settling any figure.",
            }),
            "interest_penalty_accrual": choice("What interest and penalties should be accrued on the position?", {
                "interest_and_full_penalty": "Interest plus the full statutory penalty on the unrecognised benefit.",
                "interest_and_reduced_penalty": "Interest plus the reduced penalty for positions disclosed on the return.",
                "interest_only": "Interest only; the penalty does not apply.",
                "no_accrual": "Nothing; the full benefit is recognised, so there is no underpayment to accrue on.",
            }),
        }


# =========================================================================== schema 3
EVENTS = {
    "statute_lapsed": "the statutory assessment period expired for {years_closed}",
    "exam_closed": "the {auth} closed its examination of {years_closed} with no adjustment",
    "settlement_agreed": "the company and the {auth} signed a settlement of the issue for {settle}",
    "ruling_favourable": "a {court} decided a similar case in another taxpayer's favour",
    "ruling_adverse": "a {court} decided a similar case against the taxpayer",
    "guidance_adverse": "the {auth} published guidance rejecting this treatment",
    "exam_opened": "the {auth} opened an examination of {years_open} that includes the issue",
    "settlement_offer": "the {auth} offered in writing to settle the issue for {settle}",
}


class StatuteAndSettlementUpdate(Schema):
    name = "statute_and_settlement_update"
    description = ("New information on an existing uncertain position (a statute lapse, closed examination, "
                   "settlement, ruling or new guidance): how should the reserve change, and in which period?")
    surfaces = {"json": 0.5, "utp_log": 0.25, "tax_director_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ev = pick_w(rng, {"statute_lapsed": 3, "exam_closed": 5.5, "settlement_agreed": 5.5, "ruling_favourable": 2,
                          "ruling_adverse": 2, "guidance_adverse": 1.5, "exam_opened": 2, "settlement_offer": 2})
        b = _r10(lognormal_amount(rng, 3000, 0.8))
        reserve = _r10(b * pick(rng, [0.35, 0.5, 0.6, 0.75, 0.9, 1.0]))
        years = pick(rng, [2, 3, 4])
        first = pick(rng, [2018, 2019, 2020, 2021])
        yrs = list(range(first, first + years))
        closed = rng.randint(1, years) if ev in ("statute_lapsed", "exam_closed") else 0
        if reg == "clear" and ev in ("statute_lapsed", "exam_closed"):
            closed = years
        per_year = _r10(reserve / years)
        settle = None
        if ev in ("settlement_agreed", "settlement_offer"):
            ratio = {"clear": pick(rng, [0.4, 1.4]), "borderline": near(rng, 1.0, 0.12),
                     "conflicting": rng.uniform(0.5, 1.5)}[reg]
            settle = min(_r10(reserve * ratio), b)       # a settlement never exceeds the full benefit claimed
        court = pick(rng, ["first-instance tribunal (appealable)", "court of appeal (further appeal possible)",
                           "supreme court (final)"])
        if reg == "clear" and ev in ("ruling_favourable", "ruling_adverse"):
            court = "supreme court (final)"
        appeal = chance(rng, 0.2) if ev in ("exam_closed", "settlement_agreed") else False
        reopen = pick_w(rng, {"remote": 4, "possible": 1}) if ev in ("exam_closed", "settlement_agreed") else None
        if reg == "conflicting" and ev in ("exam_closed", "settlement_agreed") and chance(rng, 0.5):
            if chance(rng, 0.5):
                appeal = True
            else:
                reopen = "possible"
        pe = _period_end(rng)
        after = chance(rng, 0.3)
        evd = pe + timedelta(days=rng.randint(5, 40)) if after else pe - timedelta(days=rng.randint(3, 80))
        j = pick(rng, JURIS)
        return {"regime": reg, "group": _group(rng), "jurisdiction": j, "authority": _auth(rng, j),
                "issue": pick(rng, MEAS_ISSUES), "event": ev, "benefit_k": b, "reserve_k": reserve,
                "years": yrs, "years_closed": closed, "reserve_per_year_k": per_year, "settlement_k": settle,
                "court": court, "company_will_appeal": appeal, "reopen_risk": reopen, "period_end": pe,
                "event_date": evd, "currency": pick(rng, CURRENCIES), "analyst": person(rng), "director": person(rng),
                "ref": ident(rng, "UTP", 3, "-")}

    def outcome(self, f):
        ev, r, b = f["event"], f["reserve_k"], f["benefit_k"]
        n, c = len(f["years"]), f["years_closed"]
        if ev in ("statute_lapsed", "exam_closed"):
            if ev == "exam_closed" and (f["company_will_appeal"] or f["reopen_risk"] == "possible"):
                return "keep_unchanged", r
            if c >= n:
                return "release_in_full", 0
            return "release_partially", _r10(r * (n - c) / n)
        if ev == "settlement_agreed":
            s = f["settlement_k"]
            if s < 0.95 * r:
                return "release_partially", s
            if s > 1.05 * r:
                return "increase_reserve", s
            return "keep_unchanged", r
        if ev == "settlement_offer":
            s = f["settlement_k"]
            return ("increase_reserve", s) if s > 1.05 * r else ("keep_unchanged", r)
        if ev == "ruling_favourable":
            if "final" in f["court"]:
                return "release_partially", _r10(r * 0.4)
            return "keep_unchanged", r
        if ev in ("ruling_adverse", "guidance_adverse"):
            if r >= 0.95 * b:
                return "keep_unchanged", r
            if ev == "ruling_adverse" and "final" not in f["court"] and "first-instance" in f["court"]:
                return "keep_unchanged", r
            return "increase_reserve", b
        return "keep_unchanged", r

    def intended(self, f):
        change, new_r = self.outcome(f)
        ev = f["event"]
        if ev == "exam_opened":
            w = 0
        elif ev == "ruling_favourable" and "final" not in f["court"]:
            w = 1
        elif ev in ("ruling_adverse",) and "first-instance" in f["court"]:
            w = 1
        elif ev in ("guidance_adverse", "settlement_offer", "ruling_favourable", "ruling_adverse"):
            w = 2 if "final" not in f["court"] or ev in ("guidance_adverse", "settlement_offer") else 3
        else:
            w = 3 if change != "keep_unchanged" else 2
        rem = new_r / f["benefit_k"]
        remaining = 0 if new_r == 0 else 1 if rem < 0.3 else 2 if rem < 0.7 else 3
        settled = (ev in ("exam_closed", "settlement_agreed") and not f["company_will_appeal"]
                   and f["reopen_risk"] == "remote")
        return {"reserve_change": change, "information_weight": w, "remaining_exposure": remaining,
                "effectively_settled": "yes" if settled else "no",
                "belongs_in_current_period": "yes" if f["event_date"] <= f["period_end"] else "no"}

    POLICY = {
        "UTP-8": "Derecognise or remeasure a position when new information changes the assessment: the assessment "
                 "period for a year expires, the position is effectively settled, a settlement is agreed, or a "
                 "ruling or published guidance changes the technical merits.",
        "UTP-9": "A position is effectively settled when the authority has completed its examination, the company "
                 "does not intend to appeal, and it is remote that the authority would re-examine it.",
        "UTP-10": "Changes are based on information available at the period end; information that arrives after "
                  "the period end is reflected in the period in which it arrives (and disclosed if material).",
    }

    def _event_text(self, f, cur, st):
        yc = f["years"][:f["years_closed"]]
        years_closed = (", ".join(str(y) for y in yc) if yc else "")
        years_open = ", ".join(str(y) for y in f["years"])
        t = EVENTS[f["event"]].format(auth=f["authority"], years_closed=years_closed, years_open=years_open,
                                      settle=kamt(f["settlement_k"], cur, st) if f["settlement_k"] else "",
                                      court=f["court"])
        extra = []
        if f["event"] in ("exam_closed", "settlement_agreed"):
            extra.append("The company intends to appeal part of the outcome." if f["company_will_appeal"]
                         else "The company does not intend to appeal.")
            extra.append({"remote": "Re-examination would require evidence of fraud, which there is no reason to expect.",
                          "possible": f"The {f['authority']} has said it may revisit the issue once a pending court "
                                      f"case is decided."}[f["reopen_risk"]])
        return t[0].upper() + t[1:] + ".", " ".join(extra)

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        ev, extra = self._event_text(f, cur, st)
        yrs = ", ".join(str(y) for y in f["years"])
        held = (f"Reserve held: {kamt(f['reserve_k'], cur, st)} against a claimed benefit of "
                f"{kamt(f['benefit_k'], cur, st)}, spread evenly over the {yrs} returns.")
        when = (f"Development dated {fmt_date(f['event_date'], rng, ds)}; period being closed ends "
                f"{fmt_date(f['period_end'], rng, ds)}.")
        if surface == "json":
            return dumps(prune({
                "position": {"ref": f["ref"], "group": f["group"], "jurisdiction": f["jurisdiction"],
                             "issue": f["issue"], "returns_affected": f["years"],
                             "benefit_claimed": kamt(f["benefit_k"], cur, st),
                             "reserve_held": kamt(f["reserve_k"], cur, st)},
                "development": {"what_happened": ev, "detail": extra or None,
                                "date": f["event_date"].isoformat()},
                "period_end": f["period_end"].isoformat(), "group_policy": self.POLICY,
            }))
        if surface == "utp_log":
            prev = f["event_date"] - timedelta(days=rng.randint(60, 200))
            return "\n".join([
                f"UTP REGISTER -- {f['ref']} -- {f['group']} ({f['jurisdiction']})",
                f"Issue: {f['issue']}; returns {yrs}", "",
                f"{fmt_date(prev, rng, ds)}  reserve reviewed, no change. {held}",
                f"{fmt_date(f['event_date'], rng, ds)}  {noisy(ev, rng, 0.05)} {extra}".rstrip(),
                f"{fmt_date(f['period_end'], rng, ds)}  period end", "",
                "Policy:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        return "\n".join([
            f"Subject: {f['ref']} {f['jurisdiction']} -- update for the close",
            f"From: {f['analyst']}", f"To: {f['director']}", "",
            f"{_first(f['director'])},", "",
            noisy(join_sentences([f"News on {f['issue']}: {ev[0].lower() + ev[1:]}", extra, held, when]), rng, 0.12),
            "", "Relevant policy: " + " ".join(f"{k}: {v}" for k, v in self.POLICY.items()), "",
            "How do you want to book it?", _first(f["analyst"]),
        ])

    def questions(self):
        return {
            "reserve_change": choice("What change to the reserve does this development call for?", {
                "increase_reserve": "Increase the reserve.",
                "keep_unchanged": "Leave the reserve as it is.",
                "release_in_full": "Release the whole reserve.",
                "release_partially": "Release part of the reserve.",
            }),
            "information_weight": score("How much does this development change the assessment of the position?", [
                "Not at all: it says nothing new about the merits or the amount.",
                "A little: relevant, but not decisive.",
                "Considerably: it shifts the likely outcome.",
                "Decisively: it settles or removes the exposure for the years it covers.",
            ]),
            "remaining_exposure": score("After this development, how much of the original exposure remains "
                                        "reserved?", [
                "None.",
                "A small part, under about a third.",
                "A substantial part, roughly a third to two thirds.",
                "Most or all of it.",
            ]),
            "effectively_settled": noul(
                "The position is effectively settled under the group's policy.",
                yes="The examination is complete or a settlement signed, the company will not appeal, and "
                    "re-examination is remote.",
                no="No completed examination or signed settlement, an appeal is intended, or the authority may "
                   "revisit the issue."),
            "belongs_in_current_period": noul(
                "The effect of this development belongs in the period now being closed.",
                yes="The development happened on or before the period end.",
                no="It happened after the period end, so it is reflected in the next period."),
        }


# =========================================================================== schema 4
TP_TYPES = {
    # (description, indicator, direction: 'low_bad' = result below range is the problem, base label)
    "distribution": ("buys finished goods from the group and resells them locally", "operating margin on sales",
                     "low_bad", "sales"),
    "contract_manufacturing": ("manufactures products for a group principal", "mark-up on total costs", "low_bad",
                               "total costs"),
    "contract_research": ("carries out research for the group's IP owner", "mark-up on total costs", "low_bad",
                          "total costs"),
    "shared_services": ("provides accounting and HR services to group companies", "mark-up on total costs", "low_bad",
                        "total costs"),
    "royalty": ("pays a royalty to a group company for trademarks", "royalty rate on sales", "high_bad", "sales"),
    "intra_group_loan": ("borrows from the group treasury company", "interest rate", "high_bad", "loan balance"),
}


class TransferPricingExposure(Schema):
    name = "transfer_pricing_exposure"
    description = ("An intra-group transaction tested against a benchmarking range: is the result arm's length, "
                   "what should be reserved, and what documentation work is needed?")
    surfaces = {"json": 0.5, "tp_workpaper": 0.25, "local_finance_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        tp = pick(rng, list(TP_TYPES))
        _, _, direction, _ = TP_TYPES[tp]
        q1, med, q3 = {"distribution": (1.6, 2.7, 3.9), "contract_manufacturing": (4.5, 6.8, 9.2),
                       "contract_research": (6.0, 8.5, 11.0), "shared_services": (4.0, 5.5, 7.5),
                       "royalty": (2.0, 3.5, 5.0), "intra_group_loan": (4.2, 5.4, 6.6)}[tp]
        jit = rng.uniform(0.85, 1.15)
        q1, med, q3 = round(q1 * jit, 1), round(med * jit, 1), round(q3 * jit, 1)
        pos = pick_w(rng, {"inside": 4, "edge": 3, "outside": 3})
        if reg == "clear":
            pos = pick(rng, ["inside", "outside"])
        bad_edge = q1 if direction == "low_bad" else q3
        sign = -1 if direction == "low_bad" else 1
        if pos == "inside":
            actual = round(rng.uniform(q1 + 0.15 * (q3 - q1), q3 - 0.15 * (q3 - q1)), 2)
        elif pos == "edge":
            actual = round(bad_edge + sign * rng.uniform(-0.25, 0.25), 2)
            if actual == bad_edge:
                actual = round(actual + sign * 0.05, 2)
        else:
            actual = round(bad_edge + sign * rng.uniform(0.4, 2.0), 2)
        base = _r10(lognormal_amount(rng, 60000, 0.8))
        rate = pick(rng, [18, 20, 22, 25, 28, 30])
        rule = pick(rng, ["median", "nearest_quartile"])
        study_age = pick_w(rng, {1: 4, 2: 3, 3: 2, 4: 1, 6: 0.5})
        comps = pick_w(rng, {4: 0.6, 7: 1, 9: 1, 12: 1, 15: 1, 21: 1})
        regional = pick_w(rng, {"same_region": 6, "other_region": 1})
        doc = chance(rng, 0.55)
        req = pick_w(rng, {"none": 3, "informal_query": 2, "formal_request": 4})
        days_left = (rng.randint(5, 20) if chance(rng, 0.5) else rng.randint(21, 60)) if req == "formal_request" else None
        if reg == "clear" and pos == "inside":
            study_age, comps, regional, doc = 1, 12, "same_region", chance(rng, 0.5)
            req = "none" if doc else req
        if reg == "conflicting":
            # a clean result on a stale study, or an outside-range result with a fresh, thorough file
            if chance(rng, 0.4):
                study_age = pick(rng, [4, 6])
            else:
                doc, study_age, comps = True, 1, 15
        recurring = chance(rng, 0.5)
        j = pick(rng, JURIS)
        return {"regime": reg, "group": _group(rng), "entity": f"{pick(rng, GROUP_A)} {j}", "jurisdiction": j,
                "authority": _auth(rng, j), "tp_type": tp, "direction": direction, "q1": q1, "median": med, "q3": q3,
                "actual": actual, "position_band": pos, "base_k": base, "tax_rate_pct": rate, "adjustment_rule": rule,
                "study_age_years": study_age, "comparables": comps, "comparables_region": regional,
                "local_file_exists": doc, "authority_request": req, "days_to_deadline": days_left,
                "outside_in_prior_years": recurring, "group_materiality_k": _r10(lognormal_amount(rng, 320, 0.5)),
                "doc_penalty_k": pick(rng, [25, 50, 75, 100, 150]), "currency": pick(rng, CURRENCIES),
                "analyst": person(rng), "local_cfo": person(rng), "fy": pick(rng, [2024, 2025, 2026])}

    @staticmethod
    def calc(f):
        a, q1, q3, med = f["actual"], f["q1"], f["q3"], f["median"]
        inside = q1 <= a <= q3
        if inside:
            adj_pts = 0.0
        else:
            target = med if f["adjustment_rule"] == "median" else (q1 if a < q1 else q3)
            adj_pts = abs(target - a)
        adj = _r10(f["base_k"] * adj_pts / 100)
        tax = _r10(adj * f["tax_rate_pct"] / 100)
        reliable = f["study_age_years"] <= 3 and f["comparables"] >= 6 and f["comparables_region"] == "same_region"
        return {"inside": inside, "adjustment_k": adj, "tax_k": tax, "reliable": reliable}

    def intended(self, f):
        c = self.calc(f)
        mat = f["group_materiality_k"]
        t = c["tax_k"]
        expo = 0 if t == 0 else 1 if t < 0.5 * mat else 2 if t < 1.5 * mat else 3
        req = f["authority_request"]
        press = {"none": 0, "informal_query": 1}.get(req)
        if press is None:
            press = 3 if f["days_to_deadline"] <= 20 else 2
        if f["local_file_exists"] and c["reliable"] and press > 0:
            press -= 1
        if not c["reliable"]:
            reserve = "refresh_study_first"
        elif not c["inside"]:
            reserve = "reserve_to_rule_point"
        elif not f["local_file_exists"]:
            reserve = "reserve_documentation_penalty"
        else:
            reserve = "no_reserve"
        if not f["local_file_exists"] and req == "formal_request":
            step = "prepare_local_file"
        elif not c["reliable"]:
            step = "update_benchmark_study"
        elif not c["inside"] and f["outside_in_prior_years"] and t >= 0.5 * mat:
            step = "seek_advance_pricing_agreement"
        elif not f["local_file_exists"]:
            step = "prepare_local_file"
        else:
            step = "no_further_action"
        return {"within_range": "yes" if c["inside"] else "no", "adjustment_exposure": expo,
                "documentation_deadline_pressure": press, "reserve_approach": reserve, "next_step": step}

    def _policy(self, f):
        to = ("the median of the range" if f["adjustment_rule"] == "median" else "the nearest edge of the range")
        return {
            "TP-1": f"{f['jurisdiction']} law: a result outside the interquartile range of the benchmarking study is "
                    f"adjusted to {to}.",
            "TP-2": f"{f['jurisdiction']} law: a local file must exist by the return filing date and be produced "
                    f"within the deadline in any formal request; failure carries a fixed penalty of "
                    f"{kamt(f['doc_penalty_k'], f['currency'], 'k')}.",
            "TP-3": "Group policy: a range can be relied on only if the study is no more than three years old and "
                    "has at least six comparables from the same region; otherwise refresh the study before "
                    "concluding.",
            "TP-4": "Group policy: reserve the tax on any adjustment the local rule would make; consider an advance "
                    "pricing agreement for recurring exposures above half of group materiality.",
        }

    def render(self, f, rng, surface):
        cur, st = f["currency"], _st(rng)
        desc, ind, _, base_lbl = TP_TYPES[f["tp_type"]]
        pol = self._policy(f)
        req = {"none": "No request from the authority.",
               "informal_query": f"The {f['authority']} has asked informally how the pricing was set.",
               "formal_request": f"The {f['authority']} has formally requested the transfer-pricing file; "
                                 f"{f['days_to_deadline']} days remain to produce it."}[f["authority_request"]]
        age = f["study_age_years"]
        study = (f"Benchmarking study {'one year' if age == 1 else str(age) + ' years'} old, {f['comparables']} comparables "
                 f"({'same region' if f['comparables_region'] == 'same_region' else 'mostly from another region'}).")
        docs = "Local file prepared for the year." if f["local_file_exists"] else "No local file has been prepared for the year."
        hist = ("The result was also outside the range in the two previous years." if f["outside_in_prior_years"]
                else None)
        if surface == "json":
            return dumps(prune({
                "transaction": {"entity": f["entity"], "group": f["group"], "jurisdiction": f["jurisdiction"],
                                "description": f"The entity {desc}.", "year": f["fy"]},
                "benchmark": {"indicator": ind, "lower_quartile_pct": f["q1"], "median_pct": f["median"],
                              "upper_quartile_pct": f["q3"], "study": study},
                "result": {"actual_pct": f["actual"], f"{base_lbl.replace(' ', '_')}": kamt(f["base_k"], cur, st),
                           "local_tax_rate_pct": f["tax_rate_pct"], "history": hist},
                "documentation": {"status": docs, "authority": req},
                "group_materiality": kamt(f["group_materiality_k"], cur, st), "rules": pol,
            }))
        if surface == "tp_workpaper":
            return "\n".join([
                f"TRANSFER PRICING REVIEW -- {f['entity']} ({f['jurisdiction']}) -- FY{f['fy']}",
                f"Transaction: the entity {desc}. Tested: {ind}.", "",
                table(["", "%"], [["Lower quartile", f"{f['q1']:.1f}"], ["Median", f"{f['median']:.1f}"],
                                  ["Upper quartile", f"{f['q3']:.1f}"], ["Actual result", f"{f['actual']:.2f}"]], rng),
                "", kv_lines([(base_lbl.capitalize(), kamt(f["base_k"], cur, st)),
                              ("Local tax rate", f"{f['tax_rate_pct']}%"),
                              ("Group materiality", kamt(f["group_materiality_k"], cur, st))], rng),
                "", bullets([study, docs, req, hist], rng), "",
                "Rules:", "\n".join(f"  {k} {v}" for k, v in pol.items()),
            ])
        return "\n".join([
            f"Subject: {f['entity']} transfer pricing -- FY{f['fy']} provision",
            f"From: {f['local_cfo']}", f"To: {f['analyst']}", "",
            noisy(join_sentences([
                f"Following up on the provision review. Our entity {desc}; the {ind} came out at {f['actual']:.2f}% "
                f"on {base_lbl} of {kamt(f['base_k'], cur, st)}.",
                f"The study gives a range of {f['q1']:.1f}% to {f['q3']:.1f}% (median {f['median']:.1f}%).", study,
                docs, req, hist, f"Our tax rate is {f['tax_rate_pct']}%."]), rng, 0.12), "",
            f"Group materiality for tax reserves is {kamt(f['group_materiality_k'], cur, st)}.",
            "Rules as I understand them: " + " ".join(f"{k}: {v}" for k, v in pol.items()), "",
            _first(f["local_cfo"]),
        ])

    def questions(self):
        return {
            "within_range": noul(
                "The tested result falls inside the arm's-length range of the benchmarking study.",
                yes="The actual result lies between the lower and upper quartiles.",
                no="The actual result lies below the lower quartile or above the upper quartile."),
            "adjustment_exposure": score("How large is the tax exposure from a local adjustment, relative to group "
                                         "materiality?", [
                "None: no adjustment would be made.",
                "Small: well under materiality.",
                "Around materiality.",
                "Large: well above materiality.",
            ]),
            "documentation_deadline_pressure": score("How pressing is the deadline for producing transfer-pricing "
                                                     "documentation?", [
                "No pressure: nothing has been asked for, or a sound file is ready.",
                "Some: an informal query, or a formal request with a usable file already in hand.",
                "High: a formal request with a few weeks left and work to do.",
                "Acute: a formal request due within about three weeks and no usable file.",
            ]),
            "reserve_approach": choice("How should the provision treat this transaction?", {
                "no_reserve": "No reserve: the result is inside a reliable range and documented.",
                "refresh_study_first": "Refresh the benchmarking study before concluding; the range cannot be "
                                       "relied on.",
                "reserve_documentation_penalty": "Reserve only the fixed documentation penalty; the pricing itself "
                                                 "is inside the range.",
                "reserve_to_rule_point": "Reserve the tax on the adjustment the local rule would make.",
            }),
            "next_step": choice("What is the most useful next step for the tax team?", {
                "no_further_action": "Nothing further this year.",
                "prepare_local_file": "Prepare the local transfer-pricing file.",
                "seek_advance_pricing_agreement": "Open talks with the authority on an advance pricing agreement.",
                "update_benchmark_study": "Commission an updated benchmarking study.",
            }),
        }


# =========================================================================== schema 5
class DisclosureAndPresentation(Schema):
    name = "disclosure_and_presentation"
    description = ("Year-end presentation and disclosure of unrecognised tax benefits: rollforward, balance-sheet "
                   "presentation, rate impact, indemnities and the draft note.")
    surfaces = {"json": 0.5, "rollforward_workpaper": 0.25, "draft_note_review": 0.25}

    ELEMENTS = {
        "rollforward": "a table reconciling opening to closing unrecognised benefits",
        "rate_effect": "the amount that would affect the effective tax rate if recognised",
        "interest_penalties": "the interest and penalty policy and amounts accrued",
        "open_years": "the tax years still open to examination in major jurisdictions",
        "twelve_month_change": "positions where a significant change within twelve months is reasonably possible",
    }

    def sample_factors(self, rng):
        reg = regime(rng)
        opening = _r10(lognormal_amount(rng, 6000, 0.7))
        add_cur = _r10(opening * rng.uniform(0.05, 0.35))
        add_prior = _r10(opening * rng.uniform(0, 0.15)) if chance(rng, 0.6) else 0
        red_prior = _r10(opening * rng.uniform(0, 0.12)) if chance(rng, 0.5) else 0
        settle = _r10(opening * rng.uniform(0.05, 0.25)) if chance(rng, 0.5) else 0
        lapse = _r10(opening * rng.uniform(0.03, 0.2)) if chance(rng, 0.6) else 0
        closing = opening + add_cur + add_prior - red_prior - settle - lapse
        diff_kind = pick_w(rng, {"none": 5, "lapse_not_booked": 2, "settlement_twice": 2, "addition_missing": 2})
        if diff_kind == "lapse_not_booked" and not lapse:
            diff_kind = "addition_missing"
        if diff_kind == "settlement_twice" and not settle:
            diff_kind = "addition_missing"
        if reg == "clear":
            diff_kind = "none" if chance(rng, 0.7) else diff_kind
        ledger = closing + {"none": 0, "lapse_not_booked": lapse, "settlement_twice": -settle,
                            "addition_missing": -add_cur}[diff_kind]
        perm_share = pick(rng, [0.0, 0.2, 0.35, 0.6, 0.8, 1.0, 1.0])
        permanent = _r10(closing * perm_share)
        nol_kind = pick_w(rng, {"covers_all": 3, "covers_part": 3, "restricted": 2, "none": 3})
        nol = {"covers_all": _r10(closing * rng.uniform(1.1, 3.0)), "covers_part": _r10(closing * rng.uniform(0.2, 0.8)),
               "restricted": _r10(closing * rng.uniform(1.1, 2.5)), "none": 0}[nol_kind]
        pay12 = chance(rng, 0.35)
        els = {k: chance(rng, 0.65) for k in self.ELEMENTS}
        if reg == "clear":
            full = chance(rng, 0.5)
            els = {k: full or chance(rng, 0.2) for k in els}
        ind = pick_w(rng, {"none": 4, "covered_creditworthy": 3, "covered_seller_insolvent": 1, "not_covered": 1})
        j = pick(rng, JURIS)
        return {"regime": reg, "group": _group(rng), "jurisdiction": j, "opening_k": opening, "additions_current_k": add_cur,
                "additions_prior_k": add_prior, "reductions_prior_k": red_prior, "settlements_k": settle,
                "lapses_k": lapse, "closing_k": closing, "ledger_k": ledger, "difference_kind": diff_kind,
                "permanent_k": permanent, "nol_kind": nol_kind, "nol_k": nol, "payment_within_12m": pay12,
                "note_elements": els, "indemnity": ind, "seller": f"{pick(rng, GROUP_A)} Capital",
                "currency": pick(rng, CURRENCIES), "year_end": _period_end(rng), "preparer": person(rng),
                "reviewer": person(rng)}

    def intended(self, f):
        c = f["closing_k"]
        if f["nol_kind"] == "covers_all":
            pres = "reduce_loss_carryforward_asset"
        elif f["nol_kind"] == "covers_part":
            pres = "split_asset_and_liability"
        elif f["payment_within_12m"]:
            pres = "current_liability"
        else:
            pres = "noncurrent_liability"
        share = f["permanent_k"] / c if c else 0
        etr = 0 if share == 0 else 1 if share < 0.5 else 2 if share < 1 else 3
        n = sum(f["note_elements"].values())
        comp = 0 if n <= 2 else 1 if n == 3 else 2 if n == 4 else 3
        return {"presentation_call": pres, "rate_effect_share": etr, "disclosure_completeness": comp,
                "rollforward_agrees": "yes" if f["ledger_k"] == c else "no",
                "indemnity_asset_required": "yes" if f["indemnity"] == "covered_creditworthy" else "no"}

    def _policy(self, f):
        return {
            "UTP-11": f"Present an unrecognised benefit as a reduction of a deferred tax asset for a loss "
                      f"carryforward in {f['jurisdiction']} to the extent that carryforward is available, under the "
                      f"tax law, to settle the extra tax that would arise; present any excess as a liability.",
            "UTP-12": "A liability is current only if payment is expected within twelve months.",
            "UTP-13": "Disclose: a rollforward table; the amount that would affect the effective tax rate; interest "
                      "and penalty policy and amounts; open tax years; and reasonably possible changes within "
                      "twelve months.",
            "UTP-14": "Where a seller indemnifies the group for a pre-acquisition position, record a separate "
                      "indemnification asset if the position is covered and recovery is probable; never net it "
                      "against the liability.",
        }

    def _texts(self, f, cur, st):
        nol = {"covers_all": f"Loss carryforward deferred tax asset in {f['jurisdiction']}: {kamt(f['nol_k'], cur, st)}, "
                             f"usable against the tax these positions would create.",
               "covers_part": f"Loss carryforward deferred tax asset in {f['jurisdiction']}: {kamt(f['nol_k'], cur, st)}, "
                              f"usable against the tax these positions would create.",
               "restricted": f"Loss carryforward deferred tax asset in {f['jurisdiction']}: {kamt(f['nol_k'], cur, st)}; "
                             f"the losses are ring-fenced to a trade that is not involved in these positions and "
                             f"cannot be set against the extra tax.",
               "none": f"No loss carryforwards in {f['jurisdiction']}."}[f["nol_kind"]]
        pay = ("The largest position is expected to be settled with the authority within the next six months."
               if f["payment_within_12m"] else "No examination is expected to conclude within the next year.")
        if f["permanent_k"] >= f["closing_k"]:
            perm = "All of the closing balance relates to permanent items (deductions or exemptions denied outright)."
        elif f["permanent_k"] == 0:
            perm = "All of the closing balance is timing: deductions taken earlier than the law allows."
        else:
            perm = (f"Of the closing balance, {kamt(f['permanent_k'], cur, st)} relates to permanent items; the rest "
                    f"is timing (deductions taken earlier than the law allows).")
        ind = {"none": None,
               "covered_creditworthy": f"{f['seller']}, which sold the subsidiary to the group, indemnifies the group "
                                       f"for these pre-acquisition positions; it is well capitalised and has paid "
                                       f"past claims promptly.",
               "covered_seller_insolvent": f"{f['seller']}, which sold the subsidiary, indemnifies the group for these "
                                           f"positions but entered insolvency proceedings this year.",
               "not_covered": f"{f['seller']}'s indemnity covers only pre-acquisition customs duties, not income "
                              f"tax positions."}[f["indemnity"]]
        return nol, pay, perm, ind

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        pol = self._policy(f)
        nol, pay, perm, ind = self._texts(f, cur, st)
        rf = [("Opening balance", f["opening_k"]), ("Additions -- current-year positions", f["additions_current_k"]),
              ("Additions -- prior-year positions", f["additions_prior_k"]),
              ("Reductions -- prior-year positions", -f["reductions_prior_k"]),
              ("Settlements", -f["settlements_k"]), ("Lapse of assessment periods", -f["lapses_k"])]
        covered = [v for k, v in self.ELEMENTS.items() if f["note_elements"][k]]
        missing = [v for k, v in self.ELEMENTS.items() if not f["note_elements"][k]]
        if surface == "json":
            return dumps(prune({
                "entity": {"group": f["group"], "year_end": f["year_end"].isoformat(), "jurisdiction": f["jurisdiction"]},
                "rollforward": {"amounts_in": f"{cur} thousands", "opening": f["opening_k"],
                                "additions_current_year": f["additions_current_k"],
                                "additions_prior_years": f["additions_prior_k"] or None,
                                "reductions_prior_years": f["reductions_prior_k"] or None,
                                "settlements": f["settlements_k"] or None, "lapses": f["lapses_k"] or None,
                                "closing_per_rollforward": f["closing_k"], "balance_per_ledger": f["ledger_k"]},
                "facts": [perm, nol, pay, ind],
                "draft_note": {"includes": covered or None, "omits": missing or None},
                "group_policy": pol,
            }))
        if surface == "rollforward_workpaper":
            rows = [[k, kamt(v, cur, st)] for k, v in rf if v != 0 or k == "Opening balance"]
            rows.append(["Closing balance (rollforward)", kamt(f["closing_k"], cur, st)])
            rows.append(["Balance per general ledger", kamt(f["ledger_k"], cur, st)])
            return "\n".join([
                f"UNRECOGNISED TAX BENEFITS -- {f['group']} -- {fmt_date(f['year_end'], rng, ds)}",
                f"Prepared {f['preparer']}; reviewer {f['reviewer']}", "",
                table(["Movement", "Amount"], rows, rng), "",
                bullets([perm, nol, pay, ind], rng), "",
                "Draft tax note covers: " + ("; ".join(covered) if covered else "nothing yet") + ".", "",
                "Policy:", "\n".join(f"  {k} {v}" for k, v in pol.items()),
            ])
        sent = {"rollforward": "A reconciliation of the beginning and ending amount of unrecognised tax benefits is "
                               "shown below.",
                "rate_effect": f"Of the total, {kamt(f['permanent_k'], cur, st)} would reduce the effective tax rate if "
                               f"recognised.",
                "interest_penalties": "The group classifies interest and penalties on uncertain positions within "
                                      "income tax expense.",
                "open_years": f"Tax years from {f['year_end'].year - 4} remain open in {f['jurisdiction']}.",
                "twelve_month_change": "It is reasonably possible that some positions will be settled within the "
                                       "next twelve months."}
        note = " ".join(sent[k] for k in self.ELEMENTS if f["note_elements"][k]) or \
            "The group recognises uncertain tax positions in accordance with its accounting policies."
        return "\n".join([
            f"DRAFT NOTE 9 -- Income taxes: uncertain positions ({f['group']}, {fmt_date(f['year_end'], rng, ds)})", "",
            "\"" + noisy(note, rng, 0.04) + "\"", "",
            f"Reviewer working notes ({f['reviewer']}):",
            bullets([f"Rollforward: opening {kamt(f['opening_k'], cur, st)}, + current-year additions "
                     f"{kamt(f['additions_current_k'], cur, st)}, + prior-year additions {kamt(f['additions_prior_k'], cur, st)}, "
                     f"- prior-year reductions {kamt(f['reductions_prior_k'], cur, st)}, - settlements "
                     f"{kamt(f['settlements_k'], cur, st)}, - lapses {kamt(f['lapses_k'], cur, st)}; closing "
                     f"{kamt(f['closing_k'], cur, st)}. Ledger shows {kamt(f['ledger_k'], cur, st)}.",
                     perm, nol, pay, ind], rng), "",
            "Policy: " + " ".join(f"[{k}] {v}" for k, v in pol.items()),
        ])

    def questions(self):
        return {
            "presentation_call": choice("How should the unrecognised tax benefits be presented in the balance sheet?", {
                "current_liability": "As a current liability.",
                "noncurrent_liability": "As a non-current liability.",
                "reduce_loss_carryforward_asset": "As a reduction of the loss-carryforward deferred tax asset.",
                "split_asset_and_liability": "Partly as a reduction of the loss-carryforward asset, with the excess "
                                             "as a liability.",
            }),
            "rate_effect_share": score("How much of the closing balance would affect the effective tax rate if "
                                       "recognised?", [
                "None of it.",
                "Less than half.",
                "Half or more, but not all.",
                "All of it.",
            ]),
            "disclosure_completeness": score("How complete is the draft disclosure against the policy's list?", [
                "Seriously incomplete: most required items are missing.",
                "Partial: two required items are missing.",
                "Nearly complete: one required item is missing.",
                "Complete: every required item is there.",
            ]),
            "rollforward_agrees": noul(
                "The rollforward of unrecognised tax benefits agrees with the balance in the general ledger.",
                yes="Opening plus additions less reductions, settlements and lapses equals the ledger balance.",
                no="The rollforward total and the ledger balance differ."),
            "indemnity_asset_required": noul(
                "A separate indemnification asset should be recorded for these positions."),
        }


WORKFLOW = Workflow(
    name="tax_uncertain_position_review",
    title="Tax uncertain-position review",
    description="A group tax team's judgments on uncertain income-tax positions: recognition, measurement, updates "
                "for settlements and statute lapses, transfer-pricing exposure, and presentation and disclosure, "
                "against written (fictional) statutes and group policy.",
    domain="tax / income-tax provision",
    schemas=[PositionRecognitionReview(), BenefitMeasurement(), StatuteAndSettlementUpdate(),
             TransferPricingExposure(), DisclosureAndPresentation()],
    notes="All jurisdictions, statutes, authorities and companies are fictional. Decisions concern a company's own "
          "tax provision; no vendor bills, payments or procurement.",
)
