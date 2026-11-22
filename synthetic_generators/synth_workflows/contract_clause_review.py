"""Contract clause review: how should a commercial deal team answer the other side's redlines?

Each case is one negotiation point in a commercial contract, where the company sells software and services: a
limitation-of-liability markup, an indemnity request, a customer's data-processing addendum, a request for a
termination-for-convenience right, or an inbound NDA on the other side's paper. The state carries the clause as
redlined, the deal context and the internal contracting playbook's thresholds. The questions ask for the
response, the exposure, who has to approve, and whether legal has to step in. Every party, product and person is
fictional.

Structure follows it_change_approval.py: each schema draws a `regime` first and lets it steer the factors, so
most cases sit near a playbook threshold, carry signals that disagree, or leave out a key fact.
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
    maybe,
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

CUSTOMER_SUFFIX = {
    "regional bank": ["Savings Bank", "Bank", "Credit Union"],
    "hospital network": ["Health", "Medical Group", "Hospital", "Care Partners"],
    "logistics company": ["Logistics", "Freight"],
    "university": ["University", "College", "Institute"],
    "retail chain": ["Outfitters", "Foods", "Supply Co.", "Home & Garden"],
    "insurance carrier": ["Mutual", "Assurance", "Insurance"],
    "county government": ["County"],
    "pharmaceutical distributor": ["Pharma Distribution", "Medical Supply"],
    "manufacturer": ["Manufacturing", "Works", "Industries"],
}
CUSTOMER_TYPES = list(CUSTOMER_SUFFIX)
# product -> the kinds of data it plausibly hosts (liability schema) and personal-data categories (DPA schema)
PRODUCT_DATA = {
    "field-service scheduling platform": ["personal_data", "business_data_only"],
    "contract analytics platform": ["business_data_only", "personal_data"],
    "payroll integration service": ["personal_data"],
    "patient scheduling portal": ["health_data", "personal_data"],
    "learning management system": ["personal_data"],
    "e-signature and document workflow service": ["personal_data", "business_data_only"],
    "IoT sensor monitoring platform": ["business_data_only"],
    "warehouse slotting optimiser": ["business_data_only"],
    "claims intake portal": ["health_data", "personal_data"],
    "subscription payments gateway": ["payment_data", "personal_data"],
}
PRODUCTS = list(PRODUCT_DATA)
SUBJECT_SCALE = {"regional bank": (40000, 900000), "hospital network": (60000, 1500000),
                 "logistics company": (3000, 60000), "university": (15000, 120000),
                 "retail chain": (100000, 4000000), "insurance carrier": (80000, 2500000),
                 "county government": (20000, 400000), "pharmaceutical distributor": (2000, 50000),
                 "manufacturer": (2000, 40000)}


def _customer(rng, kinds=None):
    kind = pick(rng, kinds or CUSTOMER_TYPES)
    return _safe_name(rng, CUSTOMER_SUFFIX[kind]), kind


def _safe_name(rng, suffixes) -> str:
    """ORG_A prefix + one of `suffixes`, skipping combinations that name a real business (base.REAL_NAME_CLASHES),
    the same way base.org does: deterministic, no extra rng draw."""
    a, i = rng.choice(ORG_A), rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        name = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if not any(name == c or name.startswith(c + " ") for c in REAL_NAME_CLASHES):
            return name
    return name


def _product_for(rng, data: str) -> str:
    return pick(rng, [p for p, ds in PRODUCT_DATA.items() if data in ds])


def _and_list(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _round_money(x: float) -> float:
    if x >= 1_000_000:
        return round(x, -5)
    if x >= 100_000:
        return round(x, -4)
    return round(x, -3)


def _slug(name: str) -> str:
    return "".join(c for c in name.lower().split()[0] if c.isalpha())


# =========================================================================== schema 1
STD_CARVE = ["gross negligence or wilful misconduct", "Supplier's IP infringement indemnity", "unpaid fees",
             "death or personal injury caused by negligence"]
NONSTD_CARVE = ["breach of confidentiality", "breach of the data protection schedule",
                "breach of the information security exhibit", "any violation of applicable law"]
LOL_RATIONALE = {
    "procurement_policy": "their procurement policy sets a minimum liability cap for software vendors",
    "regulator": "their regulator expects vendors that hold customer records to carry meaningful liability",
    "prior_incident": "a previous vendor of theirs had a data incident and the cap left them out of pocket",
    "board_directive": "their board issued a directive on third-party risk after an audit finding",
    "none_given": None,
}
DATA_LABEL = {"personal_data": "customer personal data", "health_data": "patient / health records",
              "payment_data": "payment card data", "business_data_only": "business data only (no personal data)"}


class LiabilityCapRedline(Schema):
    name = "liability_cap_redline"
    description = ("A customer's markup of the limitation-of-liability clause in a software deal: accept it, fall "
                   "back, compromise or hold the line.")
    surfaces = {"json": 0.5, "deal_desk_email": 0.25, "clause_comparison": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        customer, ctype = _customer(rng)
        acv = max(24000.0, _round_money(lognormal_amount(rng, 150000, 0.75)))
        data = pick_w(rng, {"personal_data": 4, "health_data": 2, "payment_data": 1, "business_data_only": 3})
        insurance = pick(rng, [2_000_000, 5_000_000, 5_000_000, 10_000_000, 20_000_000])
        cap_months = pick_w(rng, {12: 1, 18: 3, 24: 4, 36: 2, None: 1})     # None: general cap struck out
        floor = pick(rng, [None, None, None, 1_000_000, 2_000_000, 5_000_000])
        std = rng.sample(STD_CARVE, rng.randint(1, 3))
        nonstd = rng.sample(NONSTD_CARVE, pick(rng, [0, 0, 1, 1, 2]))
        consequential = pick_w(rng, {"retained": 5, "retained_except_data_loss": 3, "deleted": 2})
        super_cap = pick_w(rng, {None: 3, 2: 2, 3: 3, 5: 2, "unlimited": 1})   # data-protection claims
        fines_in_scope = chance(rng, 0.3)
        days_qe = pick(rng, [2, 5, 11, 24, 45, 70])
        commit = chance(rng, 0.4)
        strategic = chance(rng, 0.3)
        rep_view = pick_w(rng, {"push_accept": 3, "neutral": 3, "flag_concern": 2})
        precedent_claim = chance(rng, 0.2)
        rationale = pick(rng, list(LOL_RATIONALE))
        if reg == "clear":
            if chance(rng, 0.5):
                cap_months, floor, nonstd, consequential = 18, None, [], "retained"
                super_cap, days_qe, commit, precedent_claim = pick(rng, [None, 2]), pick(rng, [45, 70]), False, False
            else:
                cap_months, floor, consequential, super_cap = pick(rng, [None, 36]), None, "deleted", "unlimited"
                nonstd, insurance = rng.sample(NONSTD_CARVE, 2), 2_000_000
                data = pick(rng, ["personal_data", "health_data", "payment_data"])
        elif reg == "borderline":
            cap_months = 24
            gen = 2 * acv
            if chance(rng, 0.6):
                # a fixed floor that lifts the cap a little past the two-year fallback on this deal size
                floor = max(_round_money(near(rng, gen * 1.15, gen * 0.12)), 50_000)
            else:
                floor = None
            super_cap = pick(rng, [3, 3, 5, None])
            nonstd = rng.sample(NONSTD_CARVE, pick(rng, [0, 1]))
            peak = max(gen, floor or 0, (super_cap if isinstance(super_cap, int) else 0) * acv)
            insurance = max(500_000, _round_money(near(rng, peak, peak * 0.25)))
        else:
            if chance(rng, 0.5):
                # a friendly deal with a rep pushing to accept, carrying an uncapped data-protection carve-out
                strategic, rep_view, precedent_claim = True, "push_accept", True
                cap_months, consequential, super_cap = pick(rng, [12, 18]), "retained", "unlimited"
                nonstd = ["breach of the data protection schedule"] + (["breach of confidentiality"]
                                                                       if chance(rng, 0.4) else [])
                data = pick(rng, ["personal_data", "health_data"])
            else:
                # an alarming-looking markup on a small deal with little data and a big insurance tower
                consequential, cap_months = "deleted", pick(rng, [24, 36])
                acv = round(rng.uniform(20000, 60000), -3)
                data, insurance, rep_view = "business_data_only", 20_000_000, "flag_concern"
        gen_cap = None if cap_months is None else max(cap_months / 12 * acv, floor or 0)
        d = rand_date(rng)
        return {
            "regime": reg, "customer": customer, "customer_type": ctype, "product": _product_for(rng, data),
            "deal_id": ident(rng, "OPP", 6, "-"), "acv_usd": acv, "term_years": pick(rng, [1, 2, 3, 3]),
            "data_handled": data, "insurance_limit_usd": maybe(rng, insurance, 0.15),
            "cap_months": cap_months, "cap_floor_usd": floor, "general_cap_usd": gen_cap,
            "standard_carve_outs": std, "added_carve_outs": nonstd, "consequential_exclusion": consequential,
            "data_super_cap_x": super_cap, "fines_in_super_cap": fines_in_scope and super_cap is not None,
            "days_to_quarter_end": days_qe, "in_commit_forecast": commit, "strategic_logo": strategic,
            "rep": person(rng), "rep_view": rep_view, "precedent_claim": precedent_claim,
            "rationale": rationale, "other_open_points": pick(rng, [0, 1, 1, 2, 4]), "date": d,
        }

    # ------------------------------------------------------------------ text pieces
    @staticmethod
    def _cap_text(f, rng):
        if f["cap_months"] is None:
            return "no cap (customer struck the cap sentence)"
        base = f"{f['cap_months']} months of fees"
        if f["cap_floor_usd"]:
            return f"greater of {base} or {money(f['cap_floor_usd'], rng)}"
        return base

    @staticmethod
    def _super_text(f):
        s = f["data_super_cap_x"]
        if s is None:
            return None
        txt = "unlimited" if s == "unlimited" else f"{s}x annual fees"
        return txt + (" (including regulatory fines)" if f["fines_in_super_cap"] else "")

    def _rep_note(self, f, rng):
        s = []
        why = LOL_RATIONALE[f["rationale"]]
        s.append(f"Their counsel says {why}." if why else "They did not really explain the change.")
        s.append({"push_accept": pick(rng, ["Honestly I think we just take this, procurement there will not move.",
                                            "This is the last thing holding up signature, can we accept it?"]),
                  "neutral": pick(rng, ["Let me know what we can live with.", "Tell me where we land and I'll relay it."]),
                  "flag_concern": pick(rng, ["I told them this would be hard for us.",
                                             "The uncapped piece worries me given what we host for them."])}[f["rep_view"]])
        if f["precedent_claim"]:
            s.append(f"Legal agreed pretty much the same wording for {org(rng)} last quarter.")
        if f["strategic_logo"]:
            s.append("Marketing wants them as a reference logo.")
        return noisy(join_sentences(s), rng)

    def render(self, f, rng, surface):
        note = self._rep_note(f, rng)
        added = list(f["added_carve_outs"])
        uncapped = f["standard_carve_outs"] + added
        cons = {"retained": "mutual exclusion kept",
                "retained_except_data_loss": "exclusion kept, except loss or corruption of customer data",
                "deleted": "exclusion deleted"}[f["consequential_exclusion"]]
        sign_by = f["date"] + timedelta(days=max(1, f["days_to_quarter_end"] - 1))
        if surface == "json":
            return dumps(prune({
                "deal": {"id": f["deal_id"], "customer": f["customer"], "customer_type": f["customer_type"],
                         "product": f["product"], "annual_contract_value_usd": f["acv_usd"],
                         "term_years": f["term_years"], "data_hosted": f["data_handled"],
                         "reference_logo_target": f["strategic_logo"] or None,
                         "target_signature": sign_by.isoformat(), "days_to_quarter_end": f["days_to_quarter_end"],
                         "in_commit_forecast": f["in_commit_forecast"], "other_open_points": f["other_open_points"]},
                "redline": {"section": "12 Limitation of Liability", "general_cap": self._cap_text(f, rng),
                            "indirect_damages": cons, "uncapped_claims": uncapped,
                            "data_protection_cap": self._super_text(f)},
                "playbook": {"standard_cap": "12 months of fees", "fallback_cap": "up to 24 months of fees",
                             "data_protection_super_cap_max": "3x annual fees, fines excluded",
                             "indirect_damages": "exclusion stays; data-loss carve-back is a fallback",
                             "uncapped_claims_allowed": STD_CARVE},
                "insurance": {"tech_eo_and_cyber_limit_usd": f["insurance_limit_usd"]},
                "account_executive": {"name": f["rep"], "note": note},
            }))
        if surface == "deal_desk_email":
            changes = [f"12.2 cap: {self._cap_text(f, rng)} (our paper: 12 months of fees)",
                       f"12.1 indirect damages: {cons}"]
            if added:
                changes.append("12.3 uncapped list now also includes " + "; ".join(added))
            if self._super_text(f):
                changes.append(f"new 12.4: separate cap for data-protection claims, {self._super_text(f)}")
            lines = [
                f"Subject: {f['customer']} markup on section 12 -- {f['deal_id']}",
                f"From: {f['rep']}", f"To: dealdesk@{_slug(org(rng))}.example", "",
                "Hi deal desk,", "",
                f"{f['customer']} ({f['customer_type']}) sent back section 12 today. Summary of their changes:",
                bullets(changes, rng), "", note, "",
                kv_lines([("ACV", money(f["acv_usd"], rng)), ("Term", f"{f['term_years']} yr"),
                          ("Product", f["product"]), ("Data we host", DATA_LABEL[f["data_handled"]]),
                          ("Target signature", fmt_date(sign_by, rng)),
                          ("Quarter end", f"in {f['days_to_quarter_end']} days"
                           + (", deal is in commit" if f["in_commit_forecast"] else "")),
                          ("Other open points", f["other_open_points"])], rng),
                "", "Thanks,", f["rep"].split()[0], "",
                "-- CLM auto-attached: LoL playbook v4 -- standard cap 12m fees | fallback up to 24m | "
                "data super-cap max 3x ACV, no fines | keep indirect-damages exclusion (data-loss carve-back ok) | "
                "insurance tower: " + show(f["insurance_limit_usd"], rng, lambda v: money(v, rng)),
            ]
            return "\n".join(lines)
        # clause_comparison
        rows = [["General cap", "12 months of fees", self._cap_text(f, rng), "up to 24 months of fees"],
                ["Indirect damages", "excluded (mutual)", cons, "keep; data-loss carve-back ok"],
                ["Uncapped claims", "; ".join(f["standard_carve_outs"]), "; ".join(uncapped) or "none",
                 "standard list only"],
                ["Data-protection claims", "inside general cap", self._super_text(f) or "not addressed",
                 "super-cap up to 3x ACV, no fines"]]
        head = f"LIMITATION OF LIABILITY -- redline comparison ({f['deal_id']}, {f['customer']})"
        facts = kv_lines([("Annual contract value", money(f["acv_usd"], rng)),
                          ("Term", f"{f['term_years']} years"), ("Product", f["product"]),
                          ("Data hosted", DATA_LABEL[f["data_handled"]]),
                          ("Insurance (tech E&O + cyber)", show(f["insurance_limit_usd"], rng, lambda v: money(v, rng))),
                          ("Days to quarter end", f["days_to_quarter_end"]),
                          ("Commit forecast", "yes" if f["in_commit_forecast"] else "no"),
                          ("Other open points", f["other_open_points"])], rng, skip_none=False)
        return (f"{head}\n\n{table(['Term', 'Our paper', 'Customer markup', 'Playbook fallback'], rows, rng)}\n\n"
                f"{facts}\n\nAE note ({f['rep']}): {note}")

    def questions(self):
        return {
            "redline_response": choice("How should the deal team answer the customer's liability markup?", {
                "accept_markup": "Accept the customer's wording as it stands.",
                "counter_with_fallback": "Reply with the playbook's pre-approved fallback wording instead of the "
                                         "customer's.",
                "offer_bespoke_compromise": "Offer a tailored middle ground outside the playbook, for example a "
                                            "higher cap that applies only to certain claims.",
                "hold_standard_terms": "Restate the company's standard clause and explain why the markup cannot "
                                       "be taken.",
                "pause_for_principals_call": "Stop trading drafts and set up a call between the two sides' senior "
                                             "lawyers or executives.",
            }),
            "approval_authority": choice("Who has to approve the liability position before it goes back to the "
                                         "customer?", {
                "commercial_counsel": "A commercial lawyer on the legal team, because it goes beyond the "
                                      "pre-approved positions.",
                "deal_desk": "The deal desk under its standing playbook authority.",
                "general_counsel": "The general counsel personally, because it leaves a category of claims "
                                   "effectively unlimited.",
                "general_counsel_and_cfo": "The general counsel together with the finance chief, because the "
                                           "potential loss is large relative to the company's cover.",
            }),
            "exposure_level": score("How much financial exposure would the marked-up clause leave the company "
                                    "carrying?", [
                "Contained: no more than the standard cap.",
                "Moderate: above standard but inside what the playbook already allows.",
                "Elevated: a little past the fallback, but bounded and insurable.",
                "High: well past the fallback, or a significant claim type left without a cap.",
                "Severe: exposure that is effectively unlimited for claims that could realistically happen.",
            ]),
            "signing_clock": score("How tightly does the signing timeline constrain the negotiation of this "
                                   "clause?", [
                "Not at all: quarter end, the signing target, is more than six weeks away.",
                "Somewhat: quarter end is three to six weeks away.",
                "Tightly: quarter end is under two weeks away.",
                "Critically: quarter end is five days or less away, and the deal is in the commit forecast or "
                "this clause is the last open point.",
            ]),
            "beyond_insurance": noul(
                "The company's exposure under the marked-up clause could exceed its insurance cover.",
                yes="A cap, super-cap or uncapped claim type could produce a loss larger than the stated insurance "
                    "limit.",
                no="Every capped amount sits comfortably inside the insurance limit and nothing likely is left "
                   "unlimited."),
        }


# =========================================================================== schema 2
INDEM_TRIGGERS = {
    "ip_infringement": "any third-party claim alleging that the Service infringes or misappropriates that "
                       "party's intellectual property rights",
    "data_incident": "any third-party claim, regulatory inquiry or fine arising out of a Security Incident "
                     "involving Customer Data",
    "any_breach": "any and all losses, liabilities, costs and expenses arising out of or relating to Supplier's "
                  "breach of this Agreement",
    "personnel_conduct": "any claim arising from the negligent acts or omissions of Supplier personnel while on "
                         "Customer premises",
    "legal_noncompliance": "any fine, penalty or claim resulting from Supplier's failure to comply with laws "
                           "applicable to its provision of the Service",
}
TRIGGER_SHORT = {"ip_infringement": "IP infringement claims", "data_incident": "security-incident claims and fines",
                 "any_breach": "any breach of the agreement", "personnel_conduct": "on-site personnel negligence",
                 "legal_noncompliance": "fines for non-compliance with law"}


class IndemnityRequest(Schema):
    name = "indemnity_request"
    description = "A customer's drafted indemnity that the company, as supplier, is being asked to give."
    surfaces = {"json": 0.5, "counsel_memo": 0.25, "tracked_changes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        customer, ctype = _customer(rng)
        trig = pick_w(rng, {"ip_infringement": 3, "data_incident": 3, "any_breach": 2, "personnel_conduct": 1,
                            "legal_noncompliance": 2})
        first_party = chance(rng, 0.3)
        cap = pick_w(rng, {"uncapped": 3, "general_cap": 3, "super_cap_3x": 2})
        excl = pick_w(rng, {"retained": 3, "deleted": 1})
        defence = pick_w(rng, {"supplier_controls": 2, "supplier_controls_consent_to_settle": 3,
                               "customer_controls": 2})
        remedies = chance(rng, 0.5)
        mutual = chance(rng, 0.35)
        oss = pick_w(rng, {"clean": 3, "open_findings": 2, "not_run": 1})
        onsite = pick(rng, [0, 0, 2, 6, 15])
        prior_claims = pick(rng, [0, 0, 0, 1, 2])
        precedent = pick_w(rng, {"none": 3, "similar_but_capped": 3, "same_terms_granted": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                trig, first_party, cap, excl = "ip_infringement", False, "uncapped", "retained"
                defence, remedies, precedent, oss, prior_claims = ("supplier_controls_consent_to_settle", True,
                                                                   "same_terms_granted", "clean", 0)
            else:
                trig, first_party, cap, excl, defence = "any_breach", True, "uncapped", "deleted", "customer_controls"
                remedies, precedent, mutual = False, "none", False
        elif reg == "borderline":
            trig = pick(rng, ["data_incident", "legal_noncompliance", "personnel_conduct"])
            cap = pick(rng, ["super_cap_3x", "super_cap_3x", "general_cap"])
            defence = pick(rng, ["supplier_controls_consent_to_settle", "customer_controls"])
            excl, precedent = "retained", pick(rng, ["similar_but_capped", "none"])
        else:
            if chance(rng, 0.5):
                # the market-standard trigger, but with an open licence scan and the customer running the defence
                trig, oss, defence, first_party = "ip_infringement", "open_findings", "customer_controls", True
                precedent, remedies = "same_terms_granted", False
            else:
                # the broadest trigger, softened by mutuality, a cap, a small deal and prior acceptance
                trig, mutual, cap, precedent, first_party = "any_breach", True, "general_cap", "same_terms_granted", False
        if trig != "personnel_conduct":
            onsite = pick(rng, [0, 0, onsite])
        acv = max(20000.0, _round_money(lognormal_amount(rng, 120000, 0.7)))
        if reg == "conflicting" and trig == "any_breach":
            acv = round(rng.uniform(20000, 70000), -3)
        return {"regime": reg, "customer": customer, "customer_type": ctype, "product": pick(rng, PRODUCTS),
                "acv_usd": acv, "trigger": trig, "covers_first_party_losses": first_party, "cap_treatment": cap,
                "standard_exclusions": excl, "defence_control": defence, "remedies_clause": remedies,
                "mutual": mutual, "oss_licence_scan": maybe(rng, oss, 0.15) if trig == "ip_infringement" else None,
                "onsite_staff": onsite, "prior_indemnity_claims_5y": prior_claims, "precedent": precedent,
                "counsel": person(rng), "senior_counsel": person(rng), "matter": ident(rng, "LGL", 5, "-"),
                "customer_insists": chance(rng, 0.5), "date": rand_date(rng)}

    def _clause(self, f):
        s = [f"14.1 Supplier shall defend, indemnify and hold harmless Customer and its affiliates from and against "
             f"{INDEM_TRIGGERS[f['trigger']]}"
             + (", including losses suffered directly by Customer" if f["covers_first_party_losses"] else "") + "."]
        if f["standard_exclusions"] == "retained":
            s.append("14.2 Supplier has no obligation under 14.1 to the extent a claim results from Customer's "
                     "modification of the Service or its combination with items Supplier did not provide.")
        else:
            s.append("[14.2 exclusions deleted by Customer]")
        if f["remedies_clause"]:
            s.append("14.3 If the Service becomes, or in Supplier's opinion is likely to become, the subject of a "
                     "claim, Supplier may modify or replace it, or end the affected Service and refund unused "
                     "prepaid fees.")
        s.append({"supplier_controls": "14.4 Supplier shall have sole control of the defence and settlement of "
                                       "any claim.",
                  "supplier_controls_consent_to_settle": "14.4 Supplier controls the defence, but shall not settle "
                                                         "a claim that imposes any obligation on Customer without "
                                                         "Customer's written consent.",
                  "customer_controls": "14.4 Customer may conduct the defence and settle any claim with counsel of "
                                       "its choosing, at Supplier's cost."}[f["defence_control"]])
        s.append({"uncapped": "14.5 Obligations under this Section 14 are not subject to Section 12.",
                  "general_cap": "14.5 Obligations under this Section 14 are subject to Section 12.",
                  "super_cap_3x": "14.5 Obligations under this Section 14 are subject to a separate aggregate cap "
                                  "of three times (3x) the annual fees."}[f["cap_treatment"]])
        if f["mutual"]:
            s.append("14.6 Customer shall indemnify Supplier on equivalent terms for claims arising from Customer "
                     "Data or Customer's use of the Service in breach of this Agreement.")
        return s

    def _context(self, f, rng):
        c = []
        if f["trigger"] == "ip_infringement":
            c.append({"clean": "Latest open-source licence scan of the product came back clean.",
                      "open_findings": "The latest licence scan has unresolved findings on two bundled components.",
                      "not_run": "No licence scan has been run on the current release.",
                      None: "Licence scan status: not recorded."}[f["oss_licence_scan"]])
        if f["onsite_staff"]:
            c.append(f"About {f['onsite_staff']} of our people would work on the customer's sites during rollout.")
        if f["prior_indemnity_claims_5y"]:
            c.append(f"We have had {f['prior_indemnity_claims_5y']} indemnity claim(s) made against us in the last "
                     "five years.")
        c.append({"none": "We have not given this indemnity to any customer before.",
                  "similar_but_capped": f"We gave {org(rng)} a similar indemnity last year, but it was capped.",
                  "same_terms_granted": f"Near-identical wording was accepted for {org(rng)} and one other account."}[
            f["precedent"]])
        if f["customer_insists"]:
            c.append("Their procurement lead says the clause is non-negotiable.")
        return c

    def render(self, f, rng, surface):
        clause = self._clause(f)
        ctx = self._context(f, rng)
        if surface == "json":
            return dumps(prune({
                "matter": {"id": f["matter"], "customer": f["customer"], "customer_type": f["customer_type"],
                           "product": f["product"], "annual_contract_value_usd": f["acv_usd"],
                           "assigned_counsel": f["counsel"]},
                "requested_clause": {"section": "14 Indemnification", "text": " ".join(clause),
                                     "reciprocal": f["mutual"]},
                "background": {"oss_licence_scan": f["oss_licence_scan"],
                               "staff_on_customer_sites": f["onsite_staff"] or None,
                               "indemnity_claims_last_5_years": f["prior_indemnity_claims_5y"],
                               "precedent": f["precedent"],
                               "customer_says_non_negotiable": f["customer_insists"]},
                "notes": noisy(join_sentences(ctx), rng, 0.15),
            }))
        if surface == "counsel_memo":
            head = [f"TO: {f['senior_counsel']}", f"FROM: {f['counsel']}", f"DATE: {fmt_date(f['date'], rng)}",
                    f"RE: {f['customer']} -- requested indemnity ({f['matter']})", ""]
            body = [f"{f['customer']}, a {f['customer_type']} buying our {f['product']} "
                    f"(~{money(f['acv_usd'], rng)}/yr), wants us to indemnify them for "
                    f"{TRIGGER_SHORT[f['trigger']]}."]
            if f["covers_first_party_losses"]:
                body.append("The draft reaches their own direct losses, not only third-party claims.")
            body.append({"uncapped": "It sits outside the liability cap.", "general_cap": "It stays under the "
                         "general cap.", "super_cap_3x": "It has its own cap at 3x annual fees."}[f["cap_treatment"]])
            if f["standard_exclusions"] == "deleted":
                body.append("They removed our usual exclusions for modifications and combinations.")
            if f["defence_control"] == "customer_controls":
                body.append("They would run the defence and could settle on our dime.")
            if f["mutual"]:
                body.append("They offer a reciprocal indemnity for claims from their data.")
            body += ctx
            return ("\n".join(head) + "\n" + noisy(join_sentences(body), rng, 0.15) + "\n\nClause as proposed:\n"
                    + "\n".join(clause))
        # tracked_changes: the customer's insertions against our template wording, with reviewer comments
        marks = []
        for line in clause:
            if ", including losses suffered directly by Customer" in line:
                line = line.replace(", including losses suffered directly by Customer",
                                    " [+, including losses suffered directly by Customer+]")
            if line.startswith("[14.2"):
                line = ("[-14.2 Supplier has no obligation under 14.1 to the extent a claim results from Customer's "
                        "modification of the Service or its combination with items Supplier did not provide.-]")
            elif line.startswith("14.4 Customer may conduct"):
                line = "[-14.4 Supplier shall have sole control of the defence and settlement of any claim.-] [+" + line + "+]"
            elif line.startswith("14.5") and f["cap_treatment"] == "uncapped" and f["trigger"] != "ip_infringement":
                line = "14.5 Obligations under this Section 14 are [-subject to-] [+not subject to+] Section 12."
            elif line.startswith("14.6"):
                line = "[+" + line + "+]"
            marks.append(line)
        ini = "".join(w[0] for w in f["counsel"].split()[:2])
        comments = [f"[Comment {ini}{i + 1}]: {c}" for i, c in enumerate(ctx[:3])]
        trig = ("" if f["trigger"] == "ip_infringement"
                else "\n(14.1 trigger rewritten by Customer; our template covers IP infringement claims only)")
        return (f"MSA_{_slug(f['customer'])}_v3_customer-markup.docx -- Section 14 (tracked changes shown)\n"
                f"Annual fees: {money(f['acv_usd'], rng)} | {f['customer_type']} | {f['product']}{trig}\n\n"
                + "\n".join(marks) + "\n\n" + "\n".join(comments))

    def questions(self):
        return {
            "indemnity_position": choice("What position should the company take on the requested indemnity?", {
                "accept_as_drafted": "Agree to it as drafted: a routine third-party indemnity (IP infringement or "
                                     "on-site negligence) with the usual exclusions, limited to third-party claims "
                                     "and with the supplier running the defence.",
                "cap_and_restore_protections": "Keep the routine trigger but restore what the draft strips out: the "
                                               "usual exclusions, control of the defence, or the limit to "
                                               "third-party claims, under a cap where needed.",
                "narrow_the_trigger": "Give an indemnity only for third-party claims, cutting out the regulatory "
                                      "inquiries, fines and penalties the trigger reaches.",
                "refuse_indemnity": "Decline to give it: it would make any breach of the agreement a guarantee "
                                    "against loss; leave the ordinary liability terms to govern.",
            }),
            "scope_breadth": score("How far does the requested indemnity reach beyond what a software supplier "
                                   "normally stands behind?", [
                "Market standard: third-party claims of a routine kind (IP infringement, on-site negligence) with "
                "the usual exclusions, and nothing more.",
                "Somewhat wide: one element goes past that: the customer's own losses, deleted exclusions, "
                "regulatory fines, or no cap on an indemnity other than IP.",
                "Wide: two or more of those elements together.",
                "Open-ended: it covers any breach of the agreement.",
            ]),
            "claim_likelihood": score("How likely is a claim under this indemnity during the contract term?", [
                "Remote: nothing points to such claims: a clean licence scan, no staff on site, no past claims.",
                "Unlikely: conceivable, with no warning sign beyond the nature of the service.",
                "Plausible: one sign points that way: an unscanned release, several staff on site, a regulated "
                "customer's data, or a trigger any breach would meet.",
                "Probable: a known issue already points to a claim: open licence-scan findings, or two or more "
                "past indemnity claims.",
            ]),
            "defence_control_adequate": noul(
                "The clause leaves the supplier adequate control over how claims are defended and settled.",
                yes="The supplier runs the defence, and any customer consent right covers only settlements that "
                    "bind the customer.",
                no="The customer runs the defence or can settle while the supplier pays."),
            "matches_prior_concessions": noul(
                "Granting this indemnity would be in line with terms the company has already accepted for other "
                "customers."),
        }


# =========================================================================== schema 3
DPA_PRODUCT_CATS = {
    "patient scheduling portal": ["contact details", "health information", "government ID numbers"],
    "claims intake portal": ["contact details", "health information", "government ID numbers"],
    "payroll integration service": ["employee HR records", "government ID numbers", "contact details"],
    "field-service scheduling platform": ["contact details", "precise location history", "employee HR records"],
    "learning management system": ["contact details", "employee HR records"],
    "e-signature and document workflow service": ["contact details", "government ID numbers",
                                                  "employee HR records"],
    "subscription payments gateway": ["contact details", "payment card data"],
}
DPA_PRODUCT_CUSTOMERS = {
    "patient scheduling portal": ["hospital network"],
    "claims intake portal": ["insurance carrier", "hospital network"],
    "payroll integration service": CUSTOMER_TYPES,
    "field-service scheduling platform": ["logistics company", "manufacturer", "retail chain", "county government"],
    "learning management system": CUSTOMER_TYPES,
    "e-signature and document workflow service": ["regional bank", "insurance carrier", "county government",
                                                  "university"],
    "subscription payments gateway": ["retail chain", "university", "county government"],
}
REGIONS = ["EU-West", "EU-Central", "US-East", "US-West", "APAC-South", "Canada-Central"]


class DataProcessingAddendum(Schema):
    name = "data_processing_addendum"
    description = "A customer's redlined data-processing addendum: notice deadlines, audits, sub-processors, deletion."
    surfaces = {"json": 0.5, "redline_table": 0.25, "legal_intake_ticket": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        notice = pick(rng, [24, 36, 48, 72])
        confirm = maybe(rng, near(rng, 40, 20, lo=8, integer=True), 0.15)
        audit = pick_w(rng, {"report_only": 3, "onsite_customer_cost": 3, "onsite_our_cost": 2, "onsite_unlimited": 1})
        subproc = pick_w(rng, {"notice_and_objection": 3, "prior_consent": 3, "prior_consent_with_exit": 1})
        n_sub = pick(rng, [5, 9, 14, 22, 31])
        deletion = pick(rng, [30, 60, 90])
        backups = pick(rng, [35, 60, 90])
        deletion_incl_backups = chance(rng, 0.6)
        residency = pick_w(rng, {"not_requested": 3, "supported_region": 3, "unsupported_region": 2})
        product = pick(rng, list(DPA_PRODUCT_CATS))
        customer, ctype = _customer(rng, DPA_PRODUCT_CUSTOMERS[product])
        cats = rng.sample(DPA_PRODUCT_CATS[product], rng.randint(1, len(DPA_PRODUCT_CATS[product])))
        fines = pick_w(rng, {"within_cap": 3, "super_cap": 2, "uncapped": 1})
        regulated = ctype in ("regional bank", "insurance carrier", "hospital network") or chance(rng, 0.2)
        privacy_note = pick_w(rng, {"none": 3, "looks_fine": 2, "concerned": 2})
        lo_n, hi_n = SUBJECT_SCALE[ctype]
        subjects = int(round(lo_n * (hi_n / lo_n) ** rng.random(), -3))
        if reg == "clear":
            if chance(rng, 0.5):
                notice, confirm, audit, subproc = pick(rng, [48, 72]), near(rng, 18, 6, lo=6, integer=True), \
                    pick(rng, ["report_only", "onsite_customer_cost"]), "notice_and_objection"
                deletion, backups, residency, fines, cats = 90, 35, pick(rng, ["not_requested", "supported_region"]), \
                    "within_cap", ["contact details"]
            else:
                notice, confirm, audit, subproc, n_sub = 24, near(rng, 50, 10, lo=30, integer=True), \
                    "onsite_unlimited", "prior_consent_with_exit", 31
                deletion, backups, deletion_incl_backups = 30, 90, True
                residency, fines = "unsupported_region", "uncapped"
                product = pick(rng, ["patient scheduling portal", "claims intake portal"])
                customer, ctype = _customer(rng, DPA_PRODUCT_CUSTOMERS[product])
                lo_n, hi_n = SUBJECT_SCALE[ctype]
                subjects = int(round(lo_n * (hi_n / lo_n) ** rng.random(), -3))
                cats, regulated = ["health information", "government ID numbers"], True
        elif reg == "borderline":
            notice = pick(rng, [36, 48])
            confirm = near(rng, notice, 8, lo=8, integer=True)
            deletion = pick(rng, [60, 90])
            backups = pick(rng, [deletion - 5, deletion, deletion + 5])
            audit = pick(rng, ["onsite_customer_cost", "onsite_our_cost"])
        else:
            if chance(rng, 0.5):
                # the privacy analyst waves it through, but it asks for a region the product cannot serve
                privacy_note, residency = "looks_fine", "unsupported_region"
                notice, audit = 72, "report_only"
            else:
                # strict-looking terms on a small, low-sensitivity processing footprint
                notice, audit, subproc, fines = 24, "onsite_unlimited", "prior_consent", "uncapped"
                confirm = near(rng, 14, 4, lo=4, integer=True)
                cats, subjects, regulated = ["contact details"], max(1000, int(round(lo_n * 0.5, -3))), False
        return {"regime": reg, "customer": customer, "customer_type": ctype, "product": product,
                "breach_notice_hours": notice, "our_median_confirm_hours": confirm,
                "incidents_measured": pick(rng, [3, 4, 6, 9]), "audit_terms": audit,
                "subprocessor_terms": subproc, "subprocessor_count": n_sub, "deletion_days": deletion,
                "backup_retention_days": backups, "deletion_includes_backups": deletion_incl_backups,
                "residency": residency, "region": pick(rng, REGIONS), "data_categories": cats,
                "data_subjects": subjects, "fines_treatment": fines, "customer_regulated": regulated,
                "privacy_note": privacy_note, "analyst": person(rng), "requester": person(rng),
                "ticket": ident(rng, "LEGAL", 5, "-"), "days_to_target_signature": pick(rng, [3, 8, 15, 30, None]),
                "date": rand_date(rng)}

    AUDIT_TEXT = {"report_only": "annual SOC 2 Type II report; on-site only if a regulator requires it",
                  "onsite_customer_cost": "one on-site audit a year, at the customer's cost, 30 days' notice",
                  "onsite_our_cost": "one on-site audit a year, at our cost",
                  "onsite_unlimited": "on-site audits whenever the customer asks, at our cost"}
    SUB_TEXT = {"notice_and_objection": "30 days' notice of new sub-processors, customer may object",
                "prior_consent": "prior written consent for every new sub-processor",
                "prior_consent_with_exit": "prior written consent, and a right to terminate with refund if refused"}
    FINES_TEXT = {"within_cap": "subject to the general liability cap",
                  "super_cap": "separate cap of 3x annual fees for data-protection claims",
                  "uncapped": "regulatory fines and data-protection claims uncapped"}

    PRIVACY_POLICY = ("Privacy office review is required when the addendum covers health information, more than "
                      "500,000 data subjects, or uncapped regulatory fines; otherwise commercial legal signs off alone.")

    def _deletion(self, f):
        return f"{f['deletion_days']} days after termination" + (
            ", including backups" if f["deletion_includes_backups"] else ", backups roll off on their own schedule")

    def _residency(self, f):
        return {"not_requested": None, "supported_region": f"all processing in {f['region']} (we offer it)",
                "unsupported_region": f"all processing in {f['region']} (not an offered region for this product)"}[
            f["residency"]]

    def _privacy(self, f, rng):
        return {"none": None,
                "looks_fine": pick(rng, ["Privacy analyst skimmed it: 'mostly standard, should be ok'.",
                                         "Privacy team comment: looks broadly fine to me."]),
                "concerned": pick(rng, ["Privacy analyst: the notice window worries me.",
                                        "Privacy team flagged the audit and deletion wording as a concern."])}[
            f["privacy_note"]]

    def render(self, f, rng, surface):
        ops = (f"median {f['our_median_confirm_hours']} h from detection to confirmed personal-data breach "
               f"(last {f['incidents_measured']} incidents)") if f["our_median_confirm_hours"] else None
        if surface == "json":
            return dumps(prune({
                "request": {"ticket": f["ticket"], "customer": f["customer"], "customer_type": f["customer_type"],
                            "regulated_entity": f["customer_regulated"], "product": f["product"],
                            "requested_by": f["requester"],
                            "days_to_target_signature": f["days_to_target_signature"]},
                "processing": {"data_categories": f["data_categories"], "approx_data_subjects": f["data_subjects"]},
                "customer_redlines": {"breach_notice": f"within {f['breach_notice_hours']} hours of awareness",
                                      "audit": self.AUDIT_TEXT[f["audit_terms"]],
                                      "sub_processors": self.SUB_TEXT[f["subprocessor_terms"]],
                                      "deletion": self._deletion(f), "data_location": self._residency(f),
                                      "liability": self.FINES_TEXT[f["fines_treatment"]]},
                "our_template": {"breach_notice": "without undue delay, max 72 hours",
                                 "audit": self.AUDIT_TEXT["report_only"],
                                 "sub_processors": self.SUB_TEXT["notice_and_objection"],
                                 "deletion": "90 days after termination; backups expire on rotation"},
                "operations": {"breach_confirmation": ops, "backup_retention_days": f["backup_retention_days"],
                               "current_sub_processors": f["subprocessor_count"]},
                "privacy_comment": self._privacy(f, rng),
                "review_policy": self.PRIVACY_POLICY,
            }))
        if surface == "redline_table":
            rows = [["Breach notice", "72 h max", f"{f['breach_notice_hours']} h",
                     show(f["our_median_confirm_hours"], rng, lambda v: f"confirm takes ~{v} h (median)")],
                    ["Audit", "SOC 2 report", self.AUDIT_TEXT[f["audit_terms"]], "-"],
                    ["Sub-processors", "notice + objection", self.SUB_TEXT[f["subprocessor_terms"]],
                     f"{f['subprocessor_count']} in use today"],
                    ["Deletion", "90 days", self._deletion(f), f"backups kept {f['backup_retention_days']} days"],
                    ["Data location", "any offered region", self._residency(f) or "no requirement", "-"],
                    ["Liability", "general cap", self.FINES_TEXT[f["fines_treatment"]], "-"]]
            tail = kv_lines([("Customer", f"{f['customer']} ({f['customer_type']})"), ("Product", f["product"]),
                             ("Data", ", ".join(f["data_categories"])), ("Data subjects", f"~{f['data_subjects']:,}"),
                             ("Regulated", "yes" if f["customer_regulated"] else "no"),
                             ("Target signature", show(f["days_to_target_signature"], rng, lambda v: f"in {v} days"))],
                            rng)
            pc = self._privacy(f, rng)
            return (f"DPA redline review -- {f['ticket']}\n\n"
                    f"{table(['Clause', 'Our template', 'Customer redline', 'Ops note'], rows, rng)}\n\n{tail}"
                    + (f"\n\n{pc}" if pc else "") + f"\n\n{self.PRIVACY_POLICY}")
        # legal_intake_ticket
        desc = [f"{f['customer']} returned our DPA with their own markup for the {f['product']} deal."]
        desc.append(f"They want breach notice within {f['breach_notice_hours']} hours, "
                    f"{self.AUDIT_TEXT[f['audit_terms']]}, and {self.SUB_TEXT[f['subprocessor_terms']]}.")
        desc.append(f"Deletion: {self._deletion(f)}.")
        if self._residency(f):
            desc.append(f"Location: {self._residency(f)}.")
        desc.append(f"Liability: {self.FINES_TEXT[f['fines_treatment']]}.")
        if f["days_to_target_signature"]:
            desc.append(f"Sales wants to sign in {f['days_to_target_signature']} days.")
        pc = self._privacy(f, rng)
        return "\n".join([
            f"[{f['ticket']}] DPA review request",
            kv_lines([("Requester", f["requester"]), ("Opened", fmt_date(f["date"], rng)),
                      ("Queue", "Commercial Legal / Privacy"), ("Data categories", ", ".join(f["data_categories"])),
                      ("Approx. data subjects", f"{f['data_subjects']:,}"),
                      ("Customer regulated", "Y" if f["customer_regulated"] else "N")], rng),
            "", "Description:", noisy(join_sentences(desc), rng, 0.15), "",
            "Ops facts (from security team wiki):",
            bullets([f"Breach confirmation: {show(ops, rng)}", f"Backups retained {f['backup_retention_days']} days",
                     f"{f['subprocessor_count']} sub-processors currently listed"], rng)]
            + (["", f"Comment: {pc}"] if pc else []) + ["", f"Queue rule: {self.PRIVACY_POLICY}"])

    def questions(self):
        return {
            "dpa_response": choice("How should the company respond to the customer's data-processing addendum?", {
                "limit_scope": "Sign it only for the parts of the service that can meet the terms, because it asks "
                               "for a data location this product does not offer, with at most one other problem.",
                "revert_to_our_template": "Send the company's own addendum back: four or more of its commitments "
                                          "either cannot be met or go well past the template.",
                "sign_as_redlined": "Sign the customer's addendum with all of their changes: every commitment fits "
                                    "current operations and none goes well past the template.",
                "sign_with_targeted_edits": "Accept most of it, but rewrite the one to three commitments the "
                                            "company cannot meet or should not take.",
            }),
            "terms_operable": noul(
                "The company could meet every commitment in the redlined addendum with its current operations.",
                yes="Notice, audit, sub-processor, deletion and location commitments all fit how the service runs "
                    "today.",
                no="At least one commitment conflicts with current operations, such as a notice deadline shorter "
                   "than breach confirmation takes, deletion faster than backups expire, or a region not offered."),
            "operational_burden": score("How much ongoing work would the redlined commitments create for the "
                                        "operations and privacy teams?", [
                "Negligible: it matches what the company already does.",
                "Manageable: a few extra steps or reports each year.",
                "Heavy: recurring audits, consents or custom handling for this one customer.",
                "Unsustainable: commitments the teams could not reliably keep.",
            ]),
            "penalty_exposure": score("How much regulatory or contractual penalty exposure would the company take "
                                      "on by signing the redlined terms?", [
                "Minimal: low-sensitivity data and penalties inside the normal cap.",
                "Moderate: some sensitive data or a raised cap, but bounded.",
                "Significant: sensitive data at scale, or a high cap on fines.",
                "Extreme: uncapped fines over sensitive data for a regulated customer.",
            ]),
            "privacy_office_signoff": noul(
                "The privacy office has to review this addendum before anyone signs it.",
                yes="The addendum meets one of the triggers in the privacy review policy shown.",
                no="None of the policy's triggers applies, so commercial legal can sign off alone."),
        }


# =========================================================================== schema 4
TFC_REASONS = {
    "budget_uncertainty": "next year's budget has not been approved yet",
    "pending_merger": "they are mid-merger and do not know which systems survive integration",
    "pilot_evaluation": "they want a way out if the first rollout disappoints",
    "internal_policy": "their contracting policy requires an exit right in every multi-year software deal",
    "public_funding": "public funding rules require an exit if money is not appropriated",
}


class TerminationForConvenience(Schema):
    name = "termination_for_convenience"
    description = "A customer's request for a right to end a multi-year software commitment early, at will."
    surfaces = {"json": 0.5, "deal_desk_thread": 0.25, "order_form_excerpt": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        customer, ctype = _customer(rng)
        term = pick(rng, [2, 3, 3, 5])
        billing = pick_w(rng, {"annual_in_advance": 4, "multi_year_prepaid": 2, "quarterly_in_advance": 2})
        disc = pick(rng, [0, 8, 12, 18, 25])
        notice = pick(rng, [30, 60, 90, 180])
        refund = pick_w(rng, {"none": 3, "pro_rata_unused": 3, "pro_rata_plus_transition_credit": 1})
        earliest = pick(rng, [0, 6, 12, 24, 30])
        reason = pick(rng, list(TFC_REASONS))
        if reason == "public_funding" and ctype not in ("county government", "university"):
            reason = "internal_policy"
        renewal = chance(rng, 0.4)
        util = near(rng, 70, 25, lo=15, hi=100, integer=True) if renewal else None
        pilot = None if renewal else pick_w(rng, {"strong": 2, "mixed": 2, "not_run": 1})
        sponsor = pick_w(rng, {"engaged": 4, "quiet": 2, "departed": 1})
        finance = pick_w(rng, {"raised": 2, "not_reviewed": 3, "cleared": 1})
        competitor = chance(rng, 0.3)
        if reg == "clear":
            if chance(rng, 0.5):
                reason, earliest, refund, notice, disc = "public_funding" if ctype in ("county government",
                                                                                       "university") else \
                    "internal_policy", 24, "none", 180, 0
                sponsor, competitor = "engaged", False
                util = 88 if renewal else None
                pilot = None if renewal else "strong"
            else:
                reason, earliest, refund, notice = "pilot_evaluation", 0, "pro_rata_plus_transition_credit", 30
                billing, disc, sponsor, competitor = "multi_year_prepaid", 25, "departed", True
                util = 31 if renewal else None
                pilot = None if renewal else "mixed"
        elif reg == "borderline":
            earliest, notice = pick(rng, [12, 18, 24, 30]), pick(rng, [60, 90])
            refund = pick(rng, ["none", "pro_rata_unused"])
            disc = pick(rng, [8, 12])
        else:
            if chance(rng, 0.5):
                # a policy-driven ask on paper, but the sponsor has gone quiet and adoption is weak
                reason, sponsor, earliest, refund = "internal_policy", pick(rng, ["quiet", "departed"]), 12, "none"
                util = near(rng, 35, 10, lo=10, integer=True) if renewal else None
                pilot = None if renewal else "mixed"
            else:
                # a generous-looking exit on a customer who shows every sign of staying
                earliest, refund, billing, disc = 0, "pro_rata_unused", "multi_year_prepaid", 18
                sponsor, competitor = "engaged", False
                util = near(rng, 92, 5, hi=100, integer=True) if renewal else None
                pilot = None if renewal else "strong"
        acv = max(30000.0, _round_money(lognormal_amount(rng, 220000, 0.7)))
        earliest = min(earliest, term * 12 - 6)
        return {"regime": reg, "customer": customer, "customer_type": ctype, "product": pick(rng, PRODUCTS),
                "acv_usd": acv, "term_years": term, "billing": billing, "multi_year_discount_pct": disc,
                "notice_days": notice, "refund": refund, "earliest_exit_month": earliest, "reason": reason,
                "is_renewal": renewal, "seat_utilisation_pct": util, "pilot_result": pilot, "sponsor": sponsor,
                "finance_review": finance, "competitor_active": competitor,
                "days_to_quarter_end": pick(rng, [3, 9, 20, 40, 75]), "rep": person(rng), "desk": person(rng),
                "finance_partner": person(rng), "quote": ident(rng, "Q", 6, "-"), "date": rand_date(rng)}

    FINANCE_RULE = ("finance approves any termination term that refunds prepaid fees; other exit terms are settled by "
                    "the deal desk")

    @staticmethod
    def _exit_text(f):
        months = f["term_years"] * 12
        when = ("at any time" if f["earliest_exit_month"] == 0
                else f"after month {f['earliest_exit_month']} of the {months}-month term")
        ref = {"none": "no refund of fees already paid",
               "pro_rata_unused": "pro-rata refund of prepaid fees for the unused period",
               "pro_rata_plus_transition_credit": "pro-rata refund plus a transition-services credit"}[f["refund"]]
        return f"Customer may terminate for convenience {when} on {f['notice_days']} days' written notice, with {ref}."

    def _signals(self, f, rng):
        s = [f"Their reason: {TFC_REASONS[f['reason']]}."]
        if f["is_renewal"]:
            s.append(f"This is a renewal; seat utilisation is at {f['seat_utilisation_pct']}%.")
        else:
            s.append({"strong": "New logo; the pilot went well and they expanded scope during it.",
                      "mixed": "New logo; the pilot results were mixed and two sites never went live.",
                      "not_run": "New logo; there was no pilot."}[f["pilot_result"]])
        s.append({"engaged": "The exec sponsor is engaged and joined the last QBR.",
                  "quiet": "The exec sponsor has not replied in about six weeks.",
                  "departed": "The exec sponsor left the company last month."}[f["sponsor"]])
        if f["competitor_active"]:
            s.append("A competitor is still talking to them.")
        return s

    def render(self, f, rng, surface):
        tcv = f["acv_usd"] * f["term_years"]
        sig = self._signals(f, rng)
        fin = {"raised": "Finance flagged revenue-recognition impact.", "not_reviewed": None,
               "cleared": "Finance saw it and had no concerns."}[f["finance_review"]]
        if surface == "json":
            return dumps(prune({
                "quote": {"id": f["quote"], "customer": f["customer"], "customer_type": f["customer_type"],
                          "product": f["product"], "annual_fees_usd": f["acv_usd"], "term_years": f["term_years"],
                          "total_contract_value_usd": tcv, "billing": f["billing"],
                          "multi_year_discount_pct": f["multi_year_discount_pct"], "renewal": f["is_renewal"]},
                "requested_clause": {"text": self._exit_text(f), "customer_reason": TFC_REASONS[f["reason"]]},
                "playbook": {"standard": "no termination for convenience",
                             "fallback": "exit after month 12, 90+ days' notice, no refund of prepaid fees",
                             "never": "exit with refund of prepaid multi-year fees",
                             "finance_approval": self.FINANCE_RULE},
                "account_health": {"seat_utilisation_pct": f["seat_utilisation_pct"], "pilot": f["pilot_result"],
                                   "exec_sponsor": f["sponsor"], "competitor_in_deal": f["competitor_active"]},
                "finance": {"review": f["finance_review"]},
                "timing": {"days_to_quarter_end": f["days_to_quarter_end"]},
            }))
        if surface == "deal_desk_thread":
            rn, dn, fn = f["rep"].split()[0].lower(), f["desk"].split()[0].lower(), f["finance_partner"].split()[0].lower()
            msgs = [(rn, f"{f['customer']} wants a TfC clause on {f['quote']} ({f['term_years']}y, "
                         f"{money(f['acv_usd'], rng)}/yr, {f['billing'].replace('_', ' ')}, "
                         f"{f['multi_year_discount_pct']}% multi-year discount). their wording: \"{self._exit_text(f)}\""),
                    (rn, " ".join(sig)),
                    (dn, "playbook fallback is exit after month 12 with 90+ days notice and no refund of prepaid fees. "
                         "refunds on prepaid multi-year are a hard no without an exception. reminder: "
                         + self.FINANCE_RULE)]
            if fin:
                msgs.append((fn, fin.lower()))
            else:
                msgs.append((dn, f"@{fn} have you looked at this one?"))
            msgs.append((rn, f"quarter closes in {f['days_to_quarter_end']} days fwiw"))
            out, h, m = [], rng.randint(9, 15), rng.randint(0, 50)
            for who, text in msgs:
                m += rng.randint(1, 14)
                out.append(f"[{h + m // 60:02d}:{m % 60:02d}] {who}: {noisy(text, rng, 0.12)}")
            return f"#deal-desk -- {f['customer']}\n" + "\n".join(out)
        # order_form_excerpt
        head = kv_lines([("Order form", f["quote"]), ("Customer", f"{f['customer']} ({f['customer_type']})"),
                         ("Product", f["product"]), ("Subscription term", f"{f['term_years']} years"),
                         ("Annual fees", money(f["acv_usd"], rng)), ("Total contract value", money(tcv, rng)),
                         ("Billing", f["billing"].replace("_", " ")),
                         ("Multi-year discount", f"{f['multi_year_discount_pct']}%"),
                         ("Type", "renewal" if f["is_renewal"] else "new business")], rng)
        return (f"{head}\n\nSpecial terms (customer insert, section 9.4):\n  {self._exit_text(f)}\n\n"
                f"Deal desk notes:\n" + bullets(sig + ([fin] if fin else ["Finance: not yet reviewed."])
                                                + [f"Quarter end in {f['days_to_quarter_end']} days."], rng)
                + "\n\nPlaybook: standard = no TfC; fallback = after month 12, >= 90 days' notice, no refund; "
                + self.FINANCE_RULE + ".")

    def questions(self):
        return {
            "exit_right_position": choice("How should the company answer the request for a termination-for-"
                                          "convenience right?", {
                "grant_as_requested": "Give the exit right as asked: it already sits inside the playbook fallback "
                                      "(month 12 or later, 90+ days' notice, no refund).",
                "grant_with_protections": "Give an exit right, but pulled back to the playbook fallback: month 12 or "
                                          "later, 90+ days' notice, no refund of prepaid fees.",
                "offer_shorter_commitment": "Decline the exit right and offer a shorter or ramped commitment instead, "
                                            "because the customer's worry is whether the first rollout works.",
                "refuse_exit_right": "Decline the exit right and hold the full term: the account looks fragile "
                                     "(sponsor gone, or a competitor still in the deal and the sponsor not engaged).",
                "funding_out_only": "Offer only a narrow exit that applies if the customer's budget or public "
                                    "funding is formally withdrawn, since that is the reason given.",
            }),
            "revenue_at_risk": score("How much of the committed contract value would the requested exit right "
                                     "put at risk?", [
                "Little: the exit opens only in the last quarter of the term, and fees already paid are kept.",
                "Some: the exit opens in the second half of the term, or later than that but with a refund of "
                "prepaid fees.",
                "Much: the exit opens in the first half of the term, so most of the commitment could walk away.",
                "Nearly all: the customer could leave at any time and take prepaid money back.",
            ]),
            "exit_likelihood": score("How likely is the customer to use an exit right if it is granted?", [
                "Very unlikely: the clause is a formality for their paperwork.",
                "Unlikely: adoption and sponsorship look solid.",
                "Plausible: there are signs the relationship could wobble.",
                "Likely: the account already looks fragile.",
            ]),
            "discount_still_justified": noul(
                "The multi-year discount on the order form still makes sense if the exit right is granted.",
                yes="The discount is 8% or less, or the exit cannot be used before month 24 and fees already paid "
                    "are kept.",
                no="A discount above 8% stays while the customer could leave before month 24 or get prepaid fees "
                   "back: a multi-year price with the freedom of a shorter deal."),
            "finance_must_approve": noul(
                "Finance must approve the termination terms before they go to the customer.",
                yes="The requested exit refunds prepaid fees, which the approval rule sends to finance.",
                no="No prepaid fees would be refunded, so the deal desk can settle the exit terms without finance."),
        }


# =========================================================================== schema 5
RELATIONS = {"competitor": "a company that competes with us in two product lines",
             "potential_acquirer": "a larger company in our space",
             "supplier_candidate": "a vendor we are evaluating",
             "prospective_customer": "a prospect in late-stage evaluation",
             "channel_partner": "a regional systems integrator",
             "investor": "a growth-equity fund"}
REL_PURPOSES = {
    "competitor": ["a joint response to a public tender", "exploring a reseller partnership",
                   "a possible technology licensing deal"],
    "potential_acquirer": ["a possible strategic relationship", "exploring a commercial partnership",
                           "a technical integration pilot"],
    "supplier_candidate": ["evaluating them as a supplier", "a technical integration pilot"],
    "prospective_customer": ["a technical integration pilot", "a product deep-dive ahead of their purchase decision"],
    "channel_partner": ["exploring a reseller partnership", "a joint response to a public tender"],
    "investor": ["a possible strategic relationship", "an introductory meeting with their portfolio team"],
}
INFO = ["product roadmap", "pricing and discount structure", "source code excerpts", "customer list",
        "general capability deck", "financial statements", "security architecture documents"]
NON_SENSITIVE_INFO = ["general capability deck", "published case studies", "public API documentation",
                      "standard security questionnaire answers"]


class InboundNdaTriage(Schema):
    name = "inbound_nda_triage"
    description = "A non-disclosure agreement received on the other side's paper, triaged by legal operations."
    surfaces = {"json": 0.5, "intake_form": 0.25, "forwarded_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rel = pick(rng, list(RELATIONS))
        mna_hint = rel in ("potential_acquirer", "investor", "competitor") and chance(rng, 0.35)
        direction = pick_w(rng, {"mutual": 5, "we_disclose_only": 2, "they_disclose_only": 2})
        term = pick(rng, ["1 year", "2 years", "3 years", "5 years", "perpetual"])
        residuals = pick_w(rng, {"none": 4, "narrow": 2, "broad": 2})
        non_solicit = pick(rng, [None, None, 12, 24])
        standstill = rel in ("potential_acquirer", "investor") and chance(rng, 0.3)
        law = pick_w(rng, {"our_home_state": 3, "neutral_state": 3, "their_home_country": 2})
        oral = chance(rng, 0.4)
        info = rng.sample(INFO, rng.randint(1, 3))
        meeting = pick(rng, [1, 2, 5, 10, None])
        if reg == "clear":
            if chance(rng, 0.5):
                rel, direction, term, residuals, non_solicit, standstill = "supplier_candidate", "mutual", \
                    "2 years", "none", None, False
                law, oral, info, mna_hint = "our_home_state", False, ["general capability deck"], False
            else:
                rel, direction, residuals, term = "competitor", "we_disclose_only", "broad", "1 year"
                law, oral, non_solicit = "their_home_country", True, 24
                info = ["product roadmap", "pricing and discount structure"]
        elif reg == "borderline":
            residuals = pick(rng, ["narrow", "none"])
            term = pick(rng, ["2 years", "3 years"])
            direction = "mutual"
            law = pick(rng, ["neutral_state", "our_home_state"])
            non_solicit = pick(rng, [None, 12])
        else:
            if chance(rng, 0.5):
                # an innocuous stated purpose, with talk of a data room and valuation in the email
                rel, mna_hint = pick(rng, ["potential_acquirer", "investor"]), True
                residuals, direction = "none", "mutual"
            else:
                # clean paper but the counterparty is a competitor and the pack includes pricing
                rel, residuals, direction, law = "competitor", "none", "mutual", "our_home_state"
                info = ["pricing and discount structure"] + rng.sample(["general capability deck",
                                                                        "security architecture documents"], 1)
        purpose = ("evaluating them as a supplier" if rel == "supplier_candidate" and reg == "clear"
                   else pick(rng, REL_PURPOSES[rel]))
        counterparty = org(rng)
        f = {"regime": reg, "counterparty": counterparty, "relation": rel, "stated_purpose": purpose,
                "transaction_hints": mna_hint, "direction": direction, "term": term, "residuals": residuals,
                "non_solicit_months": non_solicit, "standstill": standstill, "governing_law": law,
                "oral_needs_written_confirmation": oral, "info_to_share": info, "meeting_in_days": meeting,
                "requester": person(rng), "requester_role": pick(rng, ["account executive", "engineering manager",
                                                                        "VP of partnerships", "product director",
                                                                        "chief of staff to the CEO"]),
                "request_id": ident(rng, "NDA", 5, "-"), "date": rand_date(rng)}
        # many exchanges carry only material that would not help a rival
        if rel != "competitor" and chance(rng, 0.45):
            f["info_to_share"] = rng.sample(NON_SENSITIVE_INFO, rng.randint(1, 2))
        return f

    RESID = {"none": "no residuals clause",
             "narrow": "residuals clause limited to general know-how retained in unaided memory, excluding "
                       "source code and pricing",
             "broad": "recipient may freely use any information retained in the memory of its personnel"}
    LAW = {"our_home_state": "our home state", "neutral_state": "a neutral state", "their_home_country":
           "the counterparty's home country, courts there exclusive"}
    DIRECTION = {"mutual": "mutual", "we_disclose_only": "one-way: only we disclose",
                 "they_disclose_only": "one-way: only they disclose"}

    def _terms(self, f):
        t = [f"Type: {self.DIRECTION[f['direction']]}", f"Term: {f['term']}", f"Residuals: {self.RESID[f['residuals']]}",
             f"Governing law: {self.LAW[f['governing_law']]}"]
        if f["non_solicit_months"]:
            t.append(f"Non-solicitation of employees: {f['non_solicit_months']} months")
        if f["standstill"]:
            t.append("Standstill: we may not acquire their securities for 18 months")
        if f["oral_needs_written_confirmation"]:
            t.append("Oral disclosures protected only if confirmed in writing within 30 days")
        return t

    def _hint(self, f, rng):
        if not f["transaction_hints"]:
            return None
        return pick(rng, ["They mentioned setting up a data room after the first meeting.",
                          "Their side asked early for our revenue by segment and a cap table.",
                          "Their corp dev lead will join the call and asked about valuation expectations."])

    def render(self, f, rng, surface):
        hint = self._hint(f, rng)
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "requested_by": f["requester"], "role": f["requester_role"],
                            "submitted": f["date"].isoformat(), "first_meeting_in_days": f["meeting_in_days"]},
                "counterparty": {"name": f["counterparty"], "relationship": RELATIONS[f["relation"]]},
                "purpose": {"stated": f["stated_purpose"], "requester_comment": hint},
                "information_we_expect_to_share": f["info_to_share"],
                "nda_terms_extracted": {"direction": f["direction"], "term": f["term"],
                                        "residuals": self.RESID[f["residuals"]],
                                        "non_solicit_months": f["non_solicit_months"],
                                        "standstill": f["standstill"] or None,
                                        "governing_law": self.LAW[f["governing_law"]],
                                        "oral_disclosure_needs_written_confirmation":
                                            f["oral_needs_written_confirmation"]},
                "policy": {"pre_approved_paper": "our template or counterparty paper matching it",
                           "signature_matrix": "director+ on template; legal on third-party paper with deviations; "
                                               "corp dev for anything touching a transaction"},
            }))
        if surface == "intake_form":
            rows = [("Request", f["request_id"]), ("Submitted", fmt_date(f["date"], rng)),
                    ("Requester", f"{f['requester']} ({f['requester_role']})"),
                    ("Counterparty", f"{f['counterparty']} -- {RELATIONS[f['relation']]}"),
                    ("Purpose", f["stated_purpose"]), ("Info to share", ", ".join(f["info_to_share"])),
                    ("First meeting", show(f["meeting_in_days"], rng, lambda v: f"in {v} day(s)")),
                    ("Paper", "counterparty's")]
            extra = f"\nRequester comment: {noisy(hint, rng)}" if hint else ""
            return ("NDA INTAKE (legal ops)\n" + kv_lines(rows, rng, skip_none=False)
                    + "\n\nAuto-extracted terms:\n" + bullets(self._terms(f), rng) + extra)
        # forwarded_email
        body = [f"Hi legal, {f['counterparty']} sent over their NDA ahead of {f['stated_purpose']}."]
        body.append(pick(rng, [f"For context, they are {RELATIONS[f['relation']]}.",
                               f"Background: {RELATIONS[f['relation']]}.",
                               f"(They're {RELATIONS[f['relation']]}.)"]))
        body.append(f"We would be sharing our {_and_list(f['info_to_share'])}.")
        if hint:
            body.append(hint)
        body.append(f"First meeting is in {f['meeting_in_days']} day(s), so a quick turnaround would help."
                    if f["meeting_in_days"] else "No meeting booked yet.")
        return "\n".join([
            f"Subject: FW: NDA -- {f['counterparty']}", f"From: {f['requester']} ({f['requester_role']})",
            "To: legal-ops@company.example", f"Date: {fmt_date(f['date'], rng)}", "",
            noisy(join_sentences(body), rng), "", "---------- Forwarded message ----------",
            f"Please find attached our standard mutual NDA. -- {person(rng)}, {f['counterparty']}", "",
            "[CLM extraction of attachment]", bullets(self._terms(f), rng)])

    def questions(self):
        return {
            "nda_route": choice("How should legal operations handle this NDA?", {
                "replace_with_our_template": "Decline the other side's paper and send the company's own template.",
                "route_to_counsel": "Send it to a lawyer for a full review before anything is signed.",
                "sign_as_is": "Let it be signed without changes.",
                "sign_with_stock_edits": "Apply the standard edits legal operations keeps on file and sign once "
                                         "the other side accepts them.",
            }),
            "signing_authority": choice("Who should sign this NDA for the company?", {
                "business_unit_vp": "A vice-president of the requesting business unit.",
                "corporate_development": "Corporate development, because the talks may lead to a transaction.",
                "legal_signatory": "An authorised signatory in the legal department.",
                "team_director": "A director in the requesting team, under standing delegated authority.",
            }),
            "disclosure_exposure": score("How exposed would the company's confidential information be under this "
                                         "NDA as written?", [
                "Well protected: the terms match the company's own template.",
                "Mostly protected: small gaps that matter little for what is being shared.",
                "Partly exposed: a gap such as a short term or a residuals clause touches what is being shared.",
                "Badly exposed: the recipient could lawfully reuse what it learns.",
            ]),
            "negotiation_effort": score("How much negotiation will this NDA probably need before both sides sign?", [
                "None: it can be signed as it stands.",
                "Light: one round of routine edits.",
                "Moderate: several points will need back-and-forth.",
                "Heavy: basic terms are in dispute and lawyers will need to talk.",
            ]),
            "competitively_sensitive": noul(
                "The information to be exchanged under this NDA would be valuable to a competitor.",
                yes="What will be shared, such as pricing, roadmap, code or customer names, would help a rival, "
                    "or the recipient is a rival.",
                no="The material is general or public-facing and the recipient does not compete with the company."),
        }


WORKFLOW = Workflow(
    name="contract_clause_review",
    title="Commercial contract clause review",
    description="How a supplier's deal and legal teams should answer the other side's redlines on a commercial "
                "contract, and who must approve the position.",
    domain="legal / commercial contracting",
    schemas=[LiabilityCapRedline(), IndemnityRequest(), DataProcessingAddendum(), TerminationForConvenience(),
             InboundNdaTriage()],
    notes="Sell-side contract negotiation and NDA intake with fictional parties. No bills, payments or purchase-"
          "order reconciliation, no support threads, no security-alert triage; breach-notice terms appear only as "
          "contract commitments.",
)
