"""Revenue recognition judgments: a technical-accounting team applies the group's written revenue policy to a
customer contract -- which promises are separate performance obligations, whether revenue is earned over time or
at a point in time, how much variable consideration may be booked, how a contract modification is accounted for,
and whether the company reports revenue gross as principal or net as agent.

Every state quotes the (fictional) group policy paragraphs that decide the case, so each case is decidable in
principle from the state alone. Sellers, customers, products and numbers are invented and internally consistent
(prices, discounts, progress and adjustments are computed from the figures shown).

This is the SELLER's revenue accounting. There are no vendor bills, purchase orders, payment runs or procurement
approvals anywhere: nothing here is accounts payable or invoice processing.

Each schema carries `intended(f)`: the answer the written policy points to for each question (choice key,
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
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    regime,
    score,
    table,
)

CURRENCIES = ["USD", "USD", "EUR", "GBP", "CAD", "AUD"]
SYM = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$"}
SELLER_A = ["Arclight", "Bellmont", "Cindervale", "Dawnridge", "Everhold", "Flintwick", "Glassmoor", "Hearthstone",
            "Ironvale", "Jadeport", "Kilnworth", "Lumenfield", "Mosswater", "Nightingale", "Oakhurst", "Pinemoor",
            "Quillfeather", "Riverstead", "Silverbirch", "Thistledown", "Umberleigh", "Verity", "Willowmere",
            "Ambergate", "Blackthorn", "Clearspring", "Deepwell", "Elmbridge", "Fairlight", "Goldcrest", "Highmoor",
            "Kestrel", "Lodestar", "Marlowe", "Northwind", "Oldcastle", "Primrose", "Redwing", "Stonecroft",
            "Tenterden"]
SELLER_B = ["Systems", "Technologies", "Solutions", "Engineering", "Software", "Industries", "Group", "Labs",
            "Equipment", "Services"]
CUSTOMER_A = ["Harrowgate", "Brisbane Road", "Cobham", "Delmore", "Eastleigh", "Fenwick", "Garside", "Hilltop",
              "Ingram", "Juno", "Kinross", "Lambeth", "Mayfair", "Newbury", "Ormond", "Parkside", "Queensway",
              "Rutherford", "Seaview", "Trent", "Upton", "Valemount", "Westgate", "Yardley", "Ashby", "Bexley",
              "Chandos", "Dalton", "Ellery", "Foxton"]
CUSTOMER_B = ["Hospital Trust", "Foods", "Logistics", "Retail Group", "University", "Manufacturing", "Utilities",
              "Housing Association", "Pharma", "Hotels", "Council", "Motors", "Airport", "Clinics", "Distribution"]


def _seller(rng):
    return f"{pick(rng, SELLER_A)} {pick(rng, SELLER_B)}"


def _customer(rng):
    return f"{pick(rng, CUSTOMER_A)} {pick(rng, CUSTOMER_B)}"


def kamt(k, cur, style):
    """An amount held in THOUSANDS, in one house style per state: $4.21m, $4,210k or $4,210,000."""
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


def _qend(d):
    """The calendar quarter end on or after date d."""
    m = ((d.month - 1) // 3 + 1) * 3
    return date(d.year, m, calendar.monthrange(d.year, m)[1])


def _r10(x):
    return int(round(x, -1)) if x >= 100 else int(round(x))


# =========================================================================== schema 1
# (seller kind, main item, questioned service, support element, noun for the item)
# (seller kind, main item, questioned service, support element, item noun, plural?, main price median k, customers)
BUNDLES = [
    ("enterprise software vendor", "a perpetual licence for its warehouse-management software",
     "implementation and configuration services", "two years of support and updates", "licence", False, 600,
     ["Logistics", "Distribution", "Retail Group"]),
    ("industrial equipment maker", "a high-speed packaging line", "installation and commissioning",
     "a three-year maintenance plan", "line", False, 2200, ["Foods", "Pharma", "Manufacturing"]),
    ("medical imaging supplier", "a CT scanner", "site preparation and installation", "a five-year service contract",
     "scanner", False, 1500, ["Hospital Trust", "Clinics"]),
    ("SaaS provider", "a three-year subscription to its scheduling platform", "data migration and onboarding",
     "a premium support tier", "subscription", False, 360, ["Hospital Trust", "Council", "Housing Association"]),
    ("laboratory instrument maker", "a mass spectrometer", "method development for the customer's assays",
     "a year of consumables", "instrument", False, 700, ["Pharma", "University", "Utilities"]),
    ("building-services contractor", "a rooftop chiller unit", "design and integration into the building "
     "management system", "an annual service plan", "chiller", False, 480, ["Hotels", "University", "Hospital Trust"]),
    ("robotics integrator", "four robotic welding cells", "programming and line integration",
     "a spare-parts and support package", "cells", True, 1600, ["Motors", "Manufacturing"]),
    ("point-of-sale systems vendor", "300 store terminals with software licences", "store-by-store installation",
     "helpdesk support", "terminals", True, 900, ["Retail Group", "Hotels", "Foods"]),
    ("telecoms equipment vendor", "core network hardware", "network design and integration", "an extended warranty",
     "hardware", False, 3200, ["Utilities", "Airport", "Council"]),
    ("e-learning company", "a licence to its compliance-course library", "customisation of 40 courses to the "
     "client's procedures", "hosting and learner support", "library licence", False, 220,
     ["Pharma", "Utilities", "Housing Association"]),
    ("agricultural-technology firm", "a fleet of 600 soil sensors", "field calibration and mapping",
     "a data-platform subscription", "sensors", True, 380, ["Foods", "Utilities"]),
    ("printing-press manufacturer", "a digital printing press", "operator training and workflow set-up",
     "a service contract", "press", False, 1100, ["Manufacturing", "Retail Group", "University"]),
    ("dental-equipment supplier", "six treatment-room packages", "installation and plumbing connection",
     "a maintenance agreement", "packages", True, 420, ["Clinics", "Hospital Trust"]),
    ("energy-storage company", "a 2 MWh battery system", "grid-connection engineering", "a performance-monitoring "
     "service", "battery system", False, 1300, ["Utilities", "Airport", "Council"]),
]
SETUP_ONLY_TEXT = ["creating the customer's tenant and user accounts", "issuing licence keys and portal log-ins",
                   "internal order set-up and credit checks", "registering the equipment serial numbers"]


class PerformanceObligationIdentification(Schema):
    name = "performance_obligation_identification"
    description = ("A bundled customer contract: is the service promised alongside the main item a separate "
                   "performance obligation, and how is the bundle price allocated?")
    surfaces = {"json": 0.5, "deal_abstract": 0.25, "accounting_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        b = pick(rng, BUNDLES)
        cust = f"{pick(rng, CUSTOMER_A)} {pick(rng, b[7])}"
        mod = pick_w(rng, {"none": 4, "configuration": 4, "significant": 2})
        others = chance(rng, 0.6)
        sold_sep = chance(rng, 0.45)
        integrated = chance(rng, 0.22)
        setup_only = chance(rng, 0.14)
        scope_clear = not chance(rng, 0.12)
        if reg == "clear":
            if chance(rng, 0.5):
                mod, others, sold_sep, integrated = "none", True, True, False
            else:
                mod, others, sold_sep, integrated = "significant", False, False, True
            setup_only, scope_clear = False, True
        elif reg == "conflicting":
            # standard-looking service that the contract nevertheless welds into one deliverable, or the reverse
            if chance(rng, 0.5):
                mod, others, integrated = "configuration", True, True
            else:
                mod, others, sold_sep, integrated = "significant", True, True, False
            if chance(rng, 0.3):
                scope_clear = False
        main = _r10(lognormal_amount(rng, b[6], 0.45))
        svc = _r10(main * rng.uniform(0.12, 0.6))
        sup = _r10(main * rng.uniform(0.1, 0.3))
        main_ev = pick_w(rng, {"observable_tight": 5, "highly_variable": 2, "never_sold_alone": 1})
        svc_sales = rng.randint(25, 140) if sold_sep and chance(rng, 0.6) else (rng.randint(2, 12) if sold_sep else 0)
        svc_spread = pick(rng, [6, 9, 12, 18, 30]) if sold_sep else None
        partner_rates = chance(rng, 0.5) if others else False
        cost_data = chance(rng, 0.6)
        disc_kind = pick_w(rng, {"none": 3, "general": 4, "specific": 3})
        sup_disc = pick(rng, [15, 20, 25, 30])
        if disc_kind == "none":
            total = _r10((main + svc + sup) * rng.uniform(0.995, 1.03))
        elif disc_kind == "general":
            total = _r10((main + svc + sup) * rng.uniform(0.78, 0.94))
        else:
            total = _r10(main + svc + sup * (1 - sup_disc / 100))
        return {"regime": reg, "seller": _seller(rng), "seller_kind": b[0], "customer": cust,
                "main_item": b[1], "service": b[2], "support": b[3], "item_noun": b[4], "plural": b[5],
                "modification": mod, "others_can_perform": others, "sold_separately": sold_sep,
                "integrated_output": integrated, "setup_only": setup_only,
                "setup_text": pick(rng, SETUP_ONLY_TEXT), "scope_clear": scope_clear,
                "item_works_without_service": mod != "significant" and chance(rng, 0.7),
                "ssp_main_k": main, "ssp_service_k": svc, "ssp_support_k": sup, "main_price_evidence": main_ev,
                "service_standalone_sales": svc_sales, "service_price_spread_pct": svc_spread,
                "partner_rate_card": partner_rates, "service_cost_tracking": cost_data,
                "discount_kind": disc_kind, "support_habitual_discount_pct": sup_disc if disc_kind == "specific" else
                (sup_disc if chance(rng, 0.25) else None), "contract_total_k": total,
                "currency": pick(rng, CURRENCIES), "contract_id": ident(rng, "CT", 6, "-"),
                "analyst": person(rng), "controller": person(rng), "signed": rand_date(rng),
                "sales_view": pick(rng, ["one_deal", "separate", None])}

    @staticmethod
    def ssp_level(f):
        if f["sold_separately"] and f["service_standalone_sales"] >= 20 and (f["service_price_spread_pct"] or 99) <= 15:
            return 3
        if f["sold_separately"] or f["partner_rate_card"]:
            return 2
        if f["service_cost_tracking"]:
            return 1
        return 0

    def intended(self, f):
        capable = (f["others_can_perform"] or f["sold_separately"]) and not f["setup_only"]
        separable = f["modification"] != "significant" and not f["integrated_output"]
        if not f["scope_clear"]:
            treat = "need_scope_clarification"
        elif f["setup_only"]:
            treat = "not_a_promise"
        elif capable and separable:
            treat = "separate_obligation"
        else:
            treat = "combine_with_item"
        integ = {"none": 0, "configuration": 1, "significant": 2}[f["modification"]] + (1 if f["integrated_output"] else 0)
        integ = min(3, integ)
        ssum = f["ssp_main_k"] + f["ssp_service_k"] + f["ssp_support_k"]
        disc = 1 - f["contract_total_k"] / ssum
        main_uncertain = f["main_price_evidence"] in ("highly_variable", "never_sold_alone")
        if main_uncertain and self.ssp_level(f) >= 2:
            alloc = "residual_for_main_item"
        elif disc <= 0.02:
            alloc = "no_discount_to_allocate"
        elif f["discount_kind"] == "specific":
            alloc = "discount_to_support_only"
        else:
            alloc = "allocate_proportionally"
        return {"service_treatment": treat, "integration_degree": integ, "service_price_evidence": self.ssp_level(f),
                "capable_of_being_distinct": "yes" if capable else "no", "price_allocation": alloc}

    POLICY = {
        "RR-2.1": "A promised good or service is a separate performance obligation only if (a) the customer can "
                  "benefit from it on its own or with resources readily available, for example because the company "
                  "or others sell it separately, and (b) it is separately identifiable: the company does not "
                  "significantly modify the other item with it or combine them into a single specified output.",
        "RR-2.2": "Administrative set-up activities that transfer nothing to the customer are not promises.",
        "RR-3.1": "Allocate the price on relative standalone selling prices. A discount goes to the whole contract "
                  "unless observable evidence shows it belongs to one item, such as that item being regularly sold "
                  "alone at that discount.",
        "RR-3.2": "The residual approach is allowed only for an item whose standalone price is highly variable or "
                  "never established, when the other items have observable or market-based prices.",
    }

    def _service_desc(self, f):
        if f["setup_only"]:
            return f"The '{f['service']}' line covers {f['setup_text']}; no work is done on the customer's side."
        n = f["item_noun"]
        mod = {"none": f"follows the standard published procedure and leaves the {n} exactly as supplied",
               "configuration": f"sets the {n} up to the customer's parameters using standard options, without "
                                f"changing the {n} itself",
               "significant": f"substantially modifies the {n} itself -- custom engineering that changes how it "
                              f"works -- to fit the customer's process"}
        s = [f"The service ({f['service']}) {mod[f['modification']]}."]
        s.append("Certified third-party partners routinely perform this work." if f["others_can_perform"] else
                 "Only the seller's engineers can perform it, using proprietary tools.")
        if f["integrated_output"]:
            s.append(f"The contract specifies one deliverable: a fully working, integrated {f['item_noun']} accepted as a whole.")
        if f["item_works_without_service"]:
            s.append(f"The {f['item_noun']} can be used without this service.")
        if not f["scope_clear"]:
            s.append("The statement of work for the service is marked 'to be agreed after kick-off'.")
        return " ".join(s)

    def _evidence(self, f, cur, st):
        n = f["item_noun"]
        main_ev = {"observable_tight": f"{n}: sold separately on a regular basis at about {kamt(f['ssp_main_k'], cur, st)} "
                                       f"(prices within 10%)",
                   "highly_variable": f"{n}: sold separately at prices ranging from "
                                      f"{kamt(_r10(f['ssp_main_k'] * 0.55), cur, st)} to "
                                      f"{kamt(_r10(f['ssp_main_k'] * 1.45), cur, st)}",
                   "never_sold_alone": f"{n}: never sold without services; list price "
                                       f"{kamt(f['ssp_main_k'], cur, st)}"}[f["main_price_evidence"]]
        if f["sold_separately"]:
            svc_ev = (f"service: sold on its own {f['service_standalone_sales']} times in the last two years, "
                      f"average {kamt(f['ssp_service_k'], cur, st)}, prices within {f['service_price_spread_pct']}%")
        elif f["partner_rate_card"]:
            svc_ev = f"service: never sold alone; partners' published rates imply about {kamt(f['ssp_service_k'], cur, st)}"
        elif f["service_cost_tracking"]:
            svc_ev = (f"service: never sold alone; tracked cost about {kamt(_r10(f['ssp_service_k'] / 1.3), cur, st)}, "
                      f"usual services margin 30%")
        else:
            svc_ev = "service: never sold alone; no rate card and no cost tracking"
        sup_ev = f"support: renewals sold at {kamt(f['ssp_support_k'], cur, st)}"
        if f["support_habitual_discount_pct"]:
            sup_ev += f"; support is routinely sold alone at {f['support_habitual_discount_pct']}% below that"
        return main_ev, svc_ev, sup_ev

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        main_ev, svc_ev, sup_ev = self._evidence(f, cur, st)
        sdesc = self._service_desc(f)
        sales = {"one_deal": "Sales sees this as one solution sale and wants it booked that way.",
                 "separate": "Sales priced each element separately in the quote.", None: None}[f["sales_view"]]
        if surface == "json":
            return dumps(prune({
                "contract": {"id": f["contract_id"], "seller": f["seller"], "customer": f["customer"],
                             "signed": f["signed"].isoformat(), "total_price": kamt(f["contract_total_k"], cur, st),
                             "promises": [f["main_item"], f["service"], f["support"]]},
                "service_facts": noisy(sdesc, rng, 0.05),
                "price_evidence": {"main_item": main_ev.split(": ", 1)[1], "service": svc_ev.split(": ", 1)[1],
                                   "support": sup_ev.split(": ", 1)[1]},
                "context": {"sales_comment": sales, "prepared_by": f["analyst"]},
                "policy": self.POLICY,
            }))
        if surface == "deal_abstract":
            return "\n".join([
                f"DEAL ABSTRACT {f['contract_id']} -- {f['seller']} ({f['seller_kind']})",
                kv_lines([("Customer", f["customer"]), ("Signed", fmt_date(f["signed"], rng, ds)),
                          ("Contract price", kamt(f["contract_total_k"], cur, st))], rng),
                "", "Promises in the contract:",
                bullets([f["main_item"], f["service"], f["support"]], rng),
                "", "About the service: " + noisy(sdesc, rng, 0.1),
                "", "Standalone price evidence:",
                bullets([main_ev, svc_ev, sup_ev], rng),
                "" if not sales else "\nSales note: " + sales,
                "", "Policy extract:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        return "\n".join([
            f"Subject: {f['contract_id']} {f['customer']} -- obligations and allocation",
            f"From: {f['analyst']}", f"To: {f['controller']}", "",
            f"Hi {_first(f['controller'])},", "",
            noisy(join_sentences([
                f"We signed {f['customer']} on {fmt_date(f['signed'], rng, ds)} for {kamt(f['contract_total_k'], cur, st)}: "
                f"{f['main_item']}, {f['service']} and {f['support']}.", sdesc, sales]), rng, 0.12), "",
            "What we have on standalone prices: " + "; ".join([main_ev, svc_ev, sup_ev]) + ".", "",
            "Applying the policy (" + " ".join(f"{k}: {v}" for k, v in self.POLICY.items()) + ")", "",
            "Can you confirm how you want it treated before I set up the revenue schedule?", "",
            _first(f["analyst"]),
        ])

    def questions(self):
        return {
            "service_treatment": choice("How should the service promised alongside the main item be accounted for?", {
                "combine_with_item": "Combine it with the main item into one performance obligation.",
                "need_scope_clarification": "Get the service scope defined first; the contract does not yet say what "
                                            "will be delivered.",
                "not_a_promise": "Treat it as an administrative set-up activity, not a promise to the customer.",
                "separate_obligation": "Account for it as a separate performance obligation.",
            }),
            "integration_degree": score("How tightly is the service bound up with the main item?", [
                "Independent: each could be delivered and used without the other.",
                "Loosely linked: standard configuration, nothing changed in the item.",
                "Interdependent: the service changes the item or the contract ties them closely.",
                "Fully integrated: the two are combined into one specified deliverable.",
            ]),
            "service_price_evidence": score("How good is the evidence of the service's standalone selling price?", [
                "None usable: no sales, rates or cost records.",
                "Estimates only: built up from cost plus a margin.",
                "Indirect: limited own sales or other providers' rates.",
                "Observable: frequent standalone sales at consistent prices.",
            ]),
            "capable_of_being_distinct": noul(
                "The customer could benefit from the service on its own or with resources readily available to it.",
                yes="The service, or one like it, is sold separately by the company or available from other providers, "
                    "and it transfers something to the customer.",
                no="Only the company can provide it and no one sells it separately, or it transfers nothing."),
            "price_allocation": choice("How should the contract price be allocated across the promises?", {
                "allocate_proportionally": "Spread the bundle discount across all items in proportion to their "
                                           "standalone prices.",
                "discount_to_support_only": "Put the whole discount on the support element, which is regularly sold "
                                            "alone at that discount.",
                "no_discount_to_allocate": "Allocate at standalone prices; the contract price shows no real discount.",
                "residual_for_main_item": "Price the other items at their standalone prices and give the main item "
                                          "the residual.",
            }),
        }


# =========================================================================== schema 2
# (description, seller kind, deliverable noun, plural?, series of units?, work on customer's site?, price median k,
#  customer kinds)
CONTRACTS_T = [
    ("design and build of a bespoke bottling machine", "machine builder", "machine", False, False, False, 1800,
     ["Foods", "Pharma"]),
    ("fit-out of three office floors at the customer's headquarters", "interiors contractor", "fit-out", False, False,
     True, 2600, ["Logistics", "Utilities", "Council"]),
    ("development of a claims-handling application to the customer's specification", "software house",
     "application", False, False, False, 1400, ["Housing Association", "Council", "Utilities"]),
    ("supply of 40,000 units of a standard catalogue valve", "valve manufacturer", "valves", True, True, False, 900,
     ["Utilities", "Manufacturing"]),
    ("a market-entry study delivered as a final report", "consulting firm", "report", False, False, False, 350,
     ["Retail Group", "Foods", "Hotels"]),
    ("construction of a 24-metre motor yacht", "boatyard", "yacht", False, False, False, 6500, ["Hotels"]),
    ("routine testing of water samples, about 300 a month for two years", "testing laboratory", "test results",
     True, True, False, 520, ["Utilities", "Council"]),
    ("production of 12 custom train-seat sets over nine months", "seating manufacturer", "seat sets", True, True,
     False, 1700, ["Motors", "Manufacturing"]),
    ("a modular classroom block built off-site and delivered complete", "modular builder", "classroom block", False,
     False, False, 2100, ["University", "Council"]),
    ("extension of the customer's warehouse, built on its own land", "construction contractor", "extension", False,
     False, True, 4800, ["Logistics", "Distribution"]),
    ("conversion of 20 vans into mobile clinics", "vehicle converter", "converted vans", True, True, False, 1300,
     ["Clinics", "Hospital Trust"]),
    ("printing of custom-branded packaging in monthly batches", "packaging printer", "packaging batches", True, True,
     False, 600, ["Foods", "Retail Group"]),
    ("an upgrade of the control system in the customer's plant, carried out on site", "automation integrator",
     "control system", False, False, True, 1500, ["Utilities", "Manufacturing", "Pharma"]),
    ("design of a new distribution centre, delivered as drawings in stages", "architecture practice", "design",
     False, False, False, 750, ["Logistics", "Distribution"]),
    ("manufacture of a bespoke turbine rotor", "heavy engineering firm", "rotor", False, False, False, 3900,
     ["Utilities"]),
]
PAY_RIGHT = {
    "cost_plus_margin": "If the customer terminates for convenience, it must pay for all work performed to date at "
                        "the contract rates, including margin.",
    "costs_only": "On termination for convenience the customer reimburses documented costs incurred, with no margin.",
    "deposit_only": "The customer pays a 15% deposit, which is non-refundable; nothing further is payable if it "
                    "terminates before delivery.",
    "none": "The customer may terminate before delivery without paying for work in progress.",
}
ACCEPT = {
    "objective_verified": "Acceptance is against the agreed specification, confirmed by tests the customer witnesses "
                          "before handover.",
    "objective_unverified": "Acceptance depends on objective performance tests that can only be run after handover, "
                            "in the customer's own environment.",
    "subjective": "Acceptance is at the customer's sole discretion once it is satisfied with the result.",
    "trial_period": "The customer may reject the deliverable for a full refund within 60 days of handover.",
}


class TimingOverTimeVsPoint(Schema):
    name = "timing_over_time_vs_point"
    description = ("Does control of the deliverable pass to the customer over time or at a point in time under the "
                   "group policy's criteria, and how should progress be measured?")
    surfaces = {"json": 0.5, "contract_extract": 0.25, "draft_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        desc, kind, noun, plural, series, on_site, med, ckinds = pick(rng, CONTRACTS_T)
        site = "customer_site" if on_site else "seller_premises"
        consumed = "testing of water samples" in desc
        custom = pick_w(rng, {"standard": 2, "configured": 3, "bespoke": 4})
        if "standard catalogue" in desc:
            custom = "standard"
        restriction = chance(rng, 0.18)
        pay = pick_w(rng, {"cost_plus_margin": 4, "costs_only": 2, "deposit_only": 2, "none": 2, None: 1})
        scope = pick_w(rng, {"throughout": 4, "after_milestone": 1})
        accept = pick_w(rng, {"objective_verified": 4, "objective_unverified": 2, "subjective": 2, "trial_period": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                custom, pay, scope, restriction = "bespoke", "cost_plus_margin", "throughout", chance(rng, 0.5)
            else:
                custom, pay, restriction, site, consumed = "standard", "none", False, "seller_premises", False
        elif reg == "conflicting":
            # bespoke work with a weak payment right, or a strong payment right on something resaleable
            if chance(rng, 0.5):
                custom, restriction, pay = "bespoke", chance(rng, 0.4), pick(rng, ["costs_only", "deposit_only", None])
            else:
                custom, restriction, pay = pick(rng, ["standard", "configured"]), False, "cost_plus_margin"
        if pay is None or pay == "none":
            scope = "throughout"
        uninstalled = chance(rng, 0.25) and not series
        price = _r10(lognormal_amount(rng, med, 0.4))
        progress_cost = round(rng.uniform(0.15, 0.85), 2)
        return {"regime": reg, "seller": _seller(rng), "seller_kind": kind,
                "customer": f"{pick(rng, CUSTOMER_A)} {pick(rng, ckinds)}",
                "contract": desc, "deliverable": noun, "plural": plural, "series_of_units": series, "work_location": site,
                "consumed_as_performed": consumed, "customisation": custom, "contractual_restriction": restriction,
                "payment_right": pay, "payment_right_scope": scope, "acceptance": accept,
                "uninstalled_materials_significant": uninstalled, "price_k": price,
                "cost_progress_pct": int(progress_cost * 100),
                "units_progress_pct": int(min(0.98, max(0.05, progress_cost + rng.uniform(-0.2, 0.2))) * 100),
                "currency": pick(rng, CURRENCIES), "analyst": person(rng), "reviewer": person(rng),
                "contract_id": ident(rng, "SO", 5, "-"), "quarter_end": _qend(rand_date(rng)),
                "pm_view": pick(rng, ["over_time", "on_delivery", None])}

    @staticmethod
    def alt_use(f):
        if f["contractual_restriction"]:
            return 3
        return {"standard": 0, "configured": 1, "bespoke": 2}[f["customisation"]]

    @staticmethod
    def pay_strength(f):
        base = {"cost_plus_margin": 3, "costs_only": 2, "deposit_only": 1, "none": 0, None: 0}[f["payment_right"]]
        if f["payment_right_scope"] == "after_milestone" and base > 0:
            base -= 1
        return base

    def criteria(self, f):
        c1 = f["consumed_as_performed"]
        c2 = f["work_location"] == "customer_site"
        c3 = self.alt_use(f) >= 2 and self.pay_strength(f) == 3
        return c1 or c2 or c3

    def intended(self, f):
        ot = self.criteria(f)
        if ot:
            pattern = "over_time_output_measure" if f["series_of_units"] else "over_time_input_measure"
        else:
            pattern = ("point_in_time_on_acceptance" if f["acceptance"] in ("subjective", "objective_unverified")
                       else "point_in_time_on_delivery")
        return {"recognition_pattern": pattern, "alternative_use_limits": self.alt_use(f),
                "payment_right_strength": self.pay_strength(f), "over_time_criteria_met": "yes" if ot else "no",
                "acceptance_is_formality": "yes" if f["acceptance"] == "objective_verified" else "no"}

    POLICY = {
        "RR-5.1": "Revenue is recognised over time if any one of these holds: (a) the customer receives and consumes "
                  "the benefit as the company performs; (b) the work creates or enhances an asset the customer "
                  "controls as it is created, such as work on the customer's own site; (c) the asset has no "
                  "alternative use to the company AND the company has an enforceable right, at all times, to payment "
                  "for work completed to date including a reasonable margin.",
        "RR-5.2": "Otherwise revenue is recognised at the point control passes: on delivery, or on acceptance if "
                  "acceptance cannot be judged objectively before delivery.",
        "RR-5.3": "Over-time revenue is measured by outputs (units delivered, milestones) where these reflect the "
                  "transfer of value, otherwise by costs incurred, excluding uninstalled materials.",
    }

    def _alt_use_text(self, f):
        d, v = f["deliverable"], ("are" if f["plural"] else "is")
        t = {"standard": f"The {d} {v} the seller's standard offering, supplied the same way to many customers.",
             "configured": f"The {d} {v} configured to the customer's options; adapting {'them' if f['plural'] else 'it'} "
                           f"for another buyer would take a few days.",
             "bespoke": f"The {d} {v} made to the customer's own requirements and would need heavy, costly rework "
                        f"before anyone else could use {'them' if f['plural'] else 'it'}."}[f["customisation"]]
        if f["contractual_restriction"]:
            t += " The contract forbids the seller from redirecting it to another customer."
        return t

    def _pay_text(self, f):
        if f["payment_right"] is None:
            return "Termination terms: refers to the master agreement, which is not in the file."
        t = PAY_RIGHT[f["payment_right"]]
        if f["payment_right_scope"] == "after_milestone":
            t += " This right applies only after the design-freeze milestone; before it, only costs are reimbursed."
        return t

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        loc = ("Work is carried out on the customer's premises, on assets the customer owns." if
               f["work_location"] == "customer_site" else
               f"Work is carried out at the seller's own facilities; the customer receives the {f['deliverable']} "
               f"when {'they are' if f['plural'] else 'it is'} finished.")
        alt = self._alt_use_text(f)
        pay = self._pay_text(f)
        acc = ACCEPT[f["acceptance"]]
        prog = (f"At {fmt_date(f['quarter_end'], rng, ds)}: {f['cost_progress_pct']}% of forecast cost incurred"
                + (", including significant materials delivered but not yet installed" if f["uninstalled_materials_significant"] else "")
                + (f"; {f['units_progress_pct']}% of units delivered" if f["series_of_units"] else "") + ".")
        pm = {"over_time": "The project manager wants revenue booked as the work progresses.",
              "on_delivery": "The project manager assumes revenue is booked on delivery, as in the old system.",
              None: None}[f["pm_view"]]
        if surface == "json":
            return dumps(prune({
                "contract": {"id": f["contract_id"], "seller": f"{f['seller']} ({f['seller_kind']})",
                             "customer": f["customer"], "scope": f["contract"], "price": kamt(f["price_k"], cur, st)},
                "terms": {"work_location": loc, "alternative_use": alt, "termination_payment": pay, "acceptance": acc},
                "status": {"progress": prog, "project_manager_view": pm},
                "policy": self.POLICY,
            }))
        if surface == "contract_extract":
            clauses = [("2.1 Scope", f["contract"][0].upper() + f["contract"][1:] + "."), ("2.4 Location", loc),
                       ("6.2 Specification", alt), ("14.3 Termination for convenience", pay), ("9.1 Acceptance", acc)]
            rng.shuffle(clauses)
            return "\n".join([
                f"CONTRACT EXTRACT -- {f['contract_id']} -- {f['seller']} / {f['customer']}",
                f"Contract price: {kamt(f['price_k'], cur, st)}", "",
                "\n".join(f"Clause {k}: {v}" for k, v in clauses), "",
                "Status note: " + prog + (" " + pm if pm else ""), "",
                "Group revenue policy:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        return "\n".join([
            f"DRAFT -- revenue timing memo -- {f['contract_id']}", f"Prepared by {f['analyst']}; reviewer {f['reviewer']}",
            "", "Background", noisy(join_sentences([
                f"{f['seller']}, a {f['seller_kind']}, contracted with {f['customer']} for {f['contract']} at "
                f"{kamt(f['price_k'], cur, st)}.", loc, alt]), rng, 0.1),
            "", "Key terms", bullets([pay, acc], rng),
            "", "Progress", prog + (" " + pm if pm else ""),
            "", "Policy applied", "\n".join(f"{k}: {v}" for k, v in self.POLICY.items()),
            "", "Analysis: [to be completed by reviewer]",
        ])

    def questions(self):
        return {
            "recognition_pattern": choice("When and how should revenue on this contract be recognised?", {
                "over_time_input_measure": "Over time, measured by the share of forecast cost incurred.",
                "over_time_output_measure": "Over time, measured by units delivered or milestones reached.",
                "point_in_time_on_acceptance": "At a point in time, when the customer formally accepts the deliverable.",
                "point_in_time_on_delivery": "At a point in time, when the deliverable is handed over.",
            }),
            "alternative_use_limits": score("How restricted is the seller from using the deliverable for another "
                                            "customer?", [
                "Not at all: it could go to another buyer as it is.",
                "Slightly: modest rework would make it saleable elsewhere.",
                "Practically: redirecting it would mean heavy rework or a big loss.",
                "Legally: the contract itself prevents redirecting it.",
            ]),
            "payment_right_strength": score("How strong is the seller's right to be paid for work done if the "
                                            "customer walks away?", [
                "None: no payment for work in progress, or the terms are not in the file.",
                "Weak: a deposit at most.",
                "Partial: costs recovered without margin, or full rights only after a milestone.",
                "Full: paid for all work to date, including margin, at every stage.",
            ]),
            "over_time_criteria_met": noul(
                "At least one of the policy's criteria for recognising revenue over time is met.",
                yes="The customer consumes the work as it happens, controls the asset as it is built, or the asset "
                    "has no alternative use and the seller can always recover cost plus margin.",
                no="None of the three criteria holds, so revenue is recognised at a point in time."),
            "acceptance_is_formality": noul(
                "The customer's acceptance clause is a formality that does not hold back the transfer of control."),
        }


# =========================================================================== schema 3
# (label, natural outcome shape, external driver?, description template)
VC_TYPES = {
    "volume_rebate": ("many", True, "a tiered rebate of up to {pct}% if the customer's purchases pass volume "
                                    "thresholds during the year"),
    "early_completion_bonus": ("binary", False, "a bonus of {amt} if the project completes before the target date"),
    "on_time_incentive": ("binary", False, "an incentive payment of {amt} if the final delivery is made by the "
                                           "contract date"),
    "approval_success_fee": ("binary", True, "a success fee of {amt} payable only if the regulator approves the "
                                             "customer's application"),
    "price_concessions": ("many", True, "no contractual variable element, but the company has a practice of "
                                        "granting price concessions to distributors after delivery"),
    "returns_right": ("many", True, "a right for retailers to return unsold stock within 120 days"),
    "commodity_index": ("many", True, "a price adjustment linked to an aluminium price index, settled at year end"),
    "service_credits": ("many", False, "service-level credits of up to {pct}% of fees if monthly availability "
                                       "targets are missed"),
    "award_fee": ("binary", True, "an award fee of up to {amt} decided by the customer's evaluation panel"),
    "weather_bonus": ("binary", True, "a bonus of {amt} if the outdoor works finish without weather-related delays"),
    "milestone_bonus": ("binary", False, "a bonus of {amt} for passing the customer's go-live test at the first attempt"),
    "efficiency_share": ("many", False, "a share of up to {amt} of savings the company achieves in running the "
                                        "customer's fleet maintenance"),
}


class VariableConsiderationConstraint(Schema):
    name = "variable_consideration_constraint"
    description = ("A contract with variable consideration: is the estimate made the right way, and how much may be "
                   "included in revenue under the constraint?")
    surfaces = {"json": 0.5, "estimate_workpaper": 0.25, "review_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        vt = pick(rng, list(VC_TYPES))
        shape, external, _ = VC_TYPES[vt]
        fixed = _r10(lognormal_amount(rng, 3000, 0.8))
        vmax = _r10(fixed * rng.uniform(0.04, 0.25))
        hist_n = pick_w(rng, {0: 2, 4: 2, 9: 1, 18: 2, 35: 2, 60: 1})
        relevance = pick_w(rng, {"same_product_and_customers": 3, "similar": 2, "different_market": 1}) if hist_n else None
        months = pick(rng, [1, 2, 3, 4, 6, 9, 11, 14, 20, 30])
        rng_width = pick_w(rng, {"narrow": 3, "wide": 1})
        correct = "most_likely_amount" if shape == "binary" else "expected_value"
        method = correct if chance(rng, 0.72) else ("expected_value" if correct == "most_likely_amount" else
                                                    "most_likely_amount")
        if reg == "clear":
            if chance(rng, 0.5):
                hist_n, relevance, months, rng_width, method = 60, "same_product_and_customers", 2, "narrow", correct
            else:
                hist_n, relevance, months, rng_width = 0, None, 24, "wide"
        elif reg == "conflicting":
            # a long, relevant history but a driver the company cannot influence, or vice versa
            if chance(rng, 0.5):
                hist_n, relevance = 60, "same_product_and_customers"
                months, rng_width = pick(rng, [14, 20, 30]), "wide"
            else:
                hist_n, relevance, months = pick(rng, [0, 4]), None if chance(rng, 0.5) else "different_market", 2
        if hist_n == 0:
            relevance = None
        elif relevance is None:
            relevance = "different_market"
        hit = rng.randint(35, 95) if hist_n else None
        est = _r10(vmax * (hit / 100 if hit else rng.uniform(0.4, 0.9)))
        if method == "most_likely_amount" and shape == "many":
            est = vmax
        if method == "expected_value" and shape == "binary":
            est = _r10(vmax * (hit / 100 if hit else 0.6))
        if method == "most_likely_amount" and shape == "binary":
            est = vmax if (hit or 60) >= 50 else 0
        trade = vt in ("volume_rebate", "price_concessions", "returns_right")
        cust = f"{pick(rng, CUSTOMER_A)} {pick(rng, ['Retail Group', 'Distribution', 'Foods'])}" if trade else _customer(rng)
        return {"regime": reg, "seller": _seller(rng), "customer": cust, "vc_type": vt,
                "outcome_shape": shape, "external_driver": external, "fixed_fee_k": fixed, "variable_max_k": vmax,
                "history_contracts": hist_n, "history_relevance": relevance, "history_hit_rate_pct": hit,
                "months_to_resolution": months, "outcome_range": rng_width, "method_used": method,
                "estimate_k": est, "pct": pick(rng, [3, 4, 5, 6, 8]), "currency": pick(rng, CURRENCIES),
                "analyst": person(rng), "controller": person(rng), "contract_id": ident(rng, "RC", 5, "-"),
                "quarter_end": _qend(rand_date(rng)),
                "sales_pressure": pick(rng, ["book_it_all", "conservative", None])}

    @staticmethod
    def risk(f):
        pts = 0
        n = f["history_contracts"]
        pts += 2 if n == 0 else 1 if (n < 10 or f["history_relevance"] == "different_market") else 0
        pts += 1 if f["external_driver"] else 0
        pts += 1 if f["months_to_resolution"] > 12 else 0
        pts += 1 if f["outcome_range"] == "wide" else 0
        return min(4, pts)

    def intended(self, f):
        r = self.risk(f)
        correct = "most_likely_amount" if f["outcome_shape"] == "binary" else "expected_value"
        if f["method_used"] != correct:
            dec = "redo_estimate_with_other_method"
        elif r <= 1:
            dec = "include_full_estimate"
        elif r == 2:
            dec = "include_constrained_portion"
        else:
            dec = "exclude_until_resolved"
        n, rel = f["history_contracts"], f["history_relevance"]
        sup = 0 if n == 0 else 1 if (n < 10 or rel == "different_market") else 2 if (n < 30 or rel == "similar") else 3
        return {"inclusion_decision": dec, "reversal_risk": r, "estimate_support": sup,
                "highly_probable_no_reversal": "yes" if r <= 1 else "no",
                "outside_company_influence": "yes" if f["external_driver"] else "no"}

    POLICY = {
        "RR-6.1": "Estimate variable consideration by the expected value when there are many possible outcomes or a "
                  "portfolio of similar contracts, and by the single most likely amount when the outcome is "
                  "essentially either/or.",
        "RR-6.2": "Include an estimate only to the extent it is highly probable that a significant reversal will not "
                  "occur. Factors that raise the risk: outcomes driven by things outside the company's influence "
                  "(markets, weather, third-party decisions), little or irrelevant experience with similar "
                  "contracts, a long time until the uncertainty resolves, and a broad range of outcomes.",
        "RR-6.3": "Where only part of the estimate meets that test, include that part.",
    }

    def _vc_text(self, f, cur, st):
        t = VC_TYPES[f["vc_type"]][2].format(pct=f["pct"], amt=kamt(f["variable_max_k"], cur, st))
        return t

    def _history_text(self, f):
        n = f["history_contracts"]
        if n == 0:
            return "No comparable contracts in the company's history."
        rel = {"same_product_and_customers": "same product line and customer type",
               "similar": "similar products, mostly different customers",
               "different_market": "contracts in a different market with different terms"}[f["history_relevance"]]
        return f"{n} comparable contracts ({rel}); the variable amount was earned or retained in {f['history_hit_rate_pct']}% of them."

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        vc = self._vc_text(f, cur, st)
        hist = self._history_text(f)
        method = {"expected_value": "probability-weighted expected value",
                  "most_likely_amount": "single most likely outcome"}[f["method_used"]]
        resolve = f"The amount will be known in about {f['months_to_resolution']} months."
        width = ("Outcomes could fall anywhere between nil and the maximum." if f["outcome_range"] == "wide"
                 else "Realistic outcomes sit in a narrow band.")
        press = {"book_it_all": "Sales leadership wants the full estimate in this quarter's revenue.",
                 "conservative": "The commercial lead suggests leaving it out until it is certain.", None: None}[f["sales_pressure"]]
        if surface == "json":
            return dumps(prune({
                "contract": {"id": f["contract_id"], "seller": f["seller"], "customer": f["customer"],
                             "fixed_consideration": kamt(f["fixed_fee_k"], cur, st), "variable_element": vc,
                             "maximum_variable_amount": kamt(f["variable_max_k"], cur, st)},
                "estimate": {"method_used": method, "amount_proposed": kamt(f["estimate_k"], cur, st),
                             "prepared_by": f["analyst"], "for_quarter_ending": f["quarter_end"].isoformat()},
                "evidence": {"history": hist, "resolution": resolve, "range": width},
                "commercial_view": press,
                "policy": self.POLICY,
            }))
        if surface == "estimate_workpaper":
            return "\n".join([
                f"VARIABLE CONSIDERATION ESTIMATE -- {f['contract_id']} ({f['customer']})",
                f"Quarter ending {fmt_date(f['quarter_end'], rng, ds)}; preparer {f['analyst']}", "",
                table(["Item", "Detail"], [["Fixed consideration", kamt(f["fixed_fee_k"], cur, st)],
                                           ["Variable element", vc],
                                           ["Maximum variable", kamt(f["variable_max_k"], cur, st)],
                                           ["Method used", method],
                                           ["Amount proposed for revenue", kamt(f["estimate_k"], cur, st)]], rng), "",
                "Evidence:", bullets([hist, resolve, width], rng),
                ("" if not press else "\nCommercial view: " + press), "",
                "Policy:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        t0 = rng.randint(8, 16)
        msgs = [
            (f["analyst"], f"{f['contract_id']} with {f['customer']}: fixed {kamt(f['fixed_fee_k'], cur, st)} plus {vc}"),
            (f["analyst"], f"I've estimated the variable part at {kamt(f['estimate_k'], cur, st)} using the {method}"),
            (f["controller"], "what's the history behind that?"),
            (f["analyst"], hist),
            (f["analyst"], f"{resolve} {width}"),
        ]
        if press:
            msgs.append((f["analyst"], press))
        msgs.append((f["controller"], "ok, check it against RR-6 before we lock the quarter"))
        lines = [f"[{t0 + i // 3:02d}:{(7 * i + 3) % 60:02d}] {_first(w).lower()}: {noisy(x, rng, 0.1)}"
                 for i, (w, x) in enumerate(msgs)]
        return "#revenue-close\n" + "\n".join(lines) + "\n\nPinned -- group policy:\n" + \
            "\n".join(f"{k}: {v}" for k, v in self.POLICY.items())

    def questions(self):
        return {
            "inclusion_decision": choice("How much of the variable consideration should go into revenue now?", {
                "exclude_until_resolved": "None of it until the uncertainty is resolved.",
                "include_constrained_portion": "Only the part of the estimate that is safe from a significant reversal.",
                "include_full_estimate": "The full estimate as proposed.",
                "redo_estimate_with_other_method": "None yet; the estimate must first be redone with the method the "
                                                   "policy requires.",
            }),
            "reversal_risk": score("How likely is a significant reversal of revenue if the estimate is included?", [
                "Remote.",
                "Low.",
                "Moderate.",
                "High.",
                "Very high.",
            ]),
            "estimate_support": score("How well does the company's experience support the estimate?", [
                "Not at all: no comparable contracts.",
                "Weakly: few contracts, or experience from a different market.",
                "Reasonably: a fair number of broadly similar contracts.",
                "Strongly: long experience with the same product and customer type.",
            ]),
            "highly_probable_no_reversal": noul(
                "It is highly probable that including the proposed estimate would not lead to a significant reversal.",
                yes="Experience is relevant, the outcome is largely in the company's hands, it resolves soon and the "
                    "range is narrow.",
                no="Several risk factors in the policy apply, so a significant reversal cannot be ruled out."),
            "outside_company_influence": noul(
                "The variable amount depends mainly on factors outside the company's influence."),
        }


# =========================================================================== schema 4
MOD_CONTRACTS = [
    # (original scope, added scope, remaining work is a series of distinct items?, customer kinds)
    ("supply of 20 identical packing machines delivered monthly", "eight more machines of the same model", True,
     ["Foods", "Pharma", "Distribution"]),
    ("a three-year facilities-management service, billed monthly", "a third building added to the same service", True,
     ["University", "Council", "Hospital Trust"]),
    ("construction of a single-storey warehouse", "a mezzanine floor integrated into the warehouse design", False,
     ["Logistics", "Distribution", "Retail Group"]),
    ("development of a bespoke booking platform", "an extra payments module built into the same platform", False,
     ["Hotels", "Airport", "Council"]),
    ("supply of 50,000 school laptops in batches", "10,000 additional laptops of the same specification", True,
     ["Council", "University"]),
    ("a two-year IT support service for 400 users", "support for a further 150 users", True,
     ["Housing Association", "Motors", "Pharma"]),
    ("refurbishment of a hospital ward", "an extended scope covering the adjoining corridor, done as one job", False,
     ["Hospital Trust", "Clinics"]),
    ("an ERP implementation project", "a redesign of the finance module part-way through the build", False,
     ["Manufacturing", "Utilities", "Retail Group"]),
    ("annual cleaning services for six depots", "cleaning of two new depots", True, ["Logistics", "Distribution"]),
    ("manufacture of a bespoke test rig", "an additional measurement station built into the rig", False,
     ["Motors", "Manufacturing", "Airport"]),
    ("supply of 300 street-lighting columns", "120 further columns for a second district", True, ["Council"]),
    ("a payroll outsourcing service for one country", "the same service extended to a second country", True,
     ["Retail Group", "Hotels", "Foods"]),
]
APPROVALS = {
    "signed_amendment": "Amendment No. {n} signed by both parties' authorised signatories.",
    "email_authorised": "Approved by the customer's operations director by email, which the master agreement "
                        "accepts as written variation.",
    "unpriced_scope_approved": "Change order approved for scope; price still being negotiated. The customer has paid "
                               "for every unpriced variation in the past at roughly the proposed rates.",
    "verbal_site": "Agreed verbally with the customer's site manager, who has no authority to vary the contract.",
    "under_negotiation": "Scope and price under negotiation; nothing approved yet.",
}


class ContractModificationReview(Schema):
    name = "contract_modification_review"
    description = ("A change to a customer contract part-way through: separate contract, prospective treatment, "
                   "cumulative catch-up, or nothing until it is approved?")
    surfaces = {"json": 0.5, "change_order_form": 0.25, "email_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        orig, added, series, ckinds = pick(rng, MOD_CONTRACTS)
        approval = pick_w(rng, {"signed_amendment": 4, "email_authorised": 2, "unpriced_scope_approved": 2,
                                "verbal_site": 1, "under_negotiation": 1})
        added_distinct = series or chance(rng, 0.15)
        p0 = _r10(lognormal_amount(rng, 4000, 0.8))
        progress = round(rng.uniform(0.2, 0.8), 2)
        ssp_add = _r10(p0 * rng.uniform(0.1, 0.5))
        disc = pick_w(rng, {"none": 3, "slight": 3, "material": 2, "deep": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                approval, disc, added_distinct = "signed_amendment", "none", True
                if not series:
                    orig, added, series, ckinds = pick(rng, [m for m in MOD_CONTRACTS if m[2]])
            else:
                approval = pick(rng, ["verbal_site", "under_negotiation"])
        elif reg == "borderline":
            disc = pick(rng, ["slight", "slight", "material"])
        else:
            # a signed, distinct add-on at a deep discount, or a well-priced change agreed with the wrong person
            if chance(rng, 0.5):
                approval, disc = "signed_amendment", "deep"
            else:
                approval, disc = pick(rng, ["verbal_site", "unpriced_scope_approved"]), "none"
        d = {"none": rng.uniform(-0.04, 0.02), "slight": rng.uniform(0.03, 0.09), "material": rng.uniform(0.11, 0.28),
             "deep": rng.uniform(0.32, 0.5)}[disc]
        add_price = _r10(ssp_add * (1 - d))
        qe = _qend(rand_date(rng))
        already = chance(rng, 0.12)
        days_left = pick_w(rng, {(1, 7): 3, (8, 30): 4, (31, 80): 3})
        today = qe - timedelta(days=rng.randint(*days_left))
        if already:
            today = qe + timedelta(days=rng.randint(10, 30))
        return {"regime": reg, "seller": _seller(rng), "customer": f"{pick(rng, CUSTOMER_A)} {pick(rng, ckinds)}",
                "original_scope": orig,
                "added_scope": added, "remaining_series": series, "added_distinct": added_distinct,
                "approval": approval, "original_price_k": p0, "progress_pct": int(progress * 100),
                "revenue_to_date_k": _r10(p0 * progress), "added_ssp_k": ssp_add, "added_price_k": add_price,
                "discount_band": disc, "today": today, "quarter_end": qe, "results_released": already,
                "discount_reason": pick(rng, [None, None, "customer threatened to re-tender",
                                              "saves the seller mobilisation costs it would incur on a new deal"]),
                "currency": pick(rng, CURRENCIES), "analyst": person(rng), "pm": person(rng),
                "co_number": rng.randint(2, 14), "contract_id": ident(rng, "PRJ", 4, "-")}

    @staticmethod
    def discount(f):
        return 1 - f["added_price_k"] / f["added_ssp_k"]

    def intended(self, f):
        d = self.discount(f)
        enforceable = f["approval"] in ("signed_amendment", "email_authorised", "unpriced_scope_approved")
        tol = 0.10 if f["discount_reason"] and "mobilisation" in f["discount_reason"] else 0.05
        if not enforceable:
            acc = "no_accounting_until_approved"
        elif f["added_distinct"] and d <= tol:
            acc = "separate_contract"
        elif f["remaining_series"]:
            acc = "prospective_treatment"
        else:
            acc = "cumulative_catch_up"
        depth = 0 if d <= 0.0 else 1 if d < 0.1 else 2 if d < 0.3 else 3
        if f["results_released"]:
            press = 3
        else:
            days = (f["quarter_end"] - f["today"]).days
            press = 2 if days <= 7 else 1 if days <= 30 else 0
        return {"modification_accounting": acc, "pricing_discount_depth": depth, "close_pressure": press,
                "added_scope_distinct": "yes" if f["added_distinct"] else "no",
                "modification_enforceable": "yes" if enforceable else "no"}

    POLICY = {
        "RR-7.1": "A modification exists once it is approved in a way that creates enforceable rights; an unpriced "
                  "change order counts if the scope is approved and the price can be estimated.",
        "RR-7.2": "Account for it as a separate contract if it adds distinct goods or services priced at their "
                  "standalone selling price (within 5%, or 10% where the discount reflects selling costs the "
                  "company saves).",
        "RR-7.3": "Otherwise, if the remaining goods or services are distinct from those already transferred, treat "
                  "it as ending the old contract and starting a new one (prospectively).",
        "RR-7.4": "If the remaining work is part of a single obligation partly satisfied, adjust revenue to date by "
                  "a cumulative catch-up.",
    }

    def render(self, f, rng, surface):
        cur, st, ds = f["currency"], _st(rng), _ds(rng)
        appr = APPROVALS[f["approval"]].format(n=f["co_number"])
        distinct_note = ("The added items are the same stand-alone units as the original scope." if f["added_distinct"]
                         and f["remaining_series"] else
                         "The added work can be used independently of the original deliverable." if f["added_distinct"]
                         else "The added work is built into the original deliverable and is not usable on its own.")
        why = f"Reason given for the price: {f['discount_reason']}." if f["discount_reason"] else None
        status = (f"Results for the quarter ended {fmt_date(f['quarter_end'], rng, ds)} were released last week on the "
                  f"original terms." if f["results_released"] else
                  f"Today is {fmt_date(f['today'], rng, ds)}; the quarter closes {fmt_date(f['quarter_end'], rng, ds)}.")
        if surface == "json":
            return dumps(prune({
                "contract": {"id": f["contract_id"], "seller": f["seller"], "customer": f["customer"],
                             "original_scope": f["original_scope"], "original_price": kamt(f["original_price_k"], cur, st),
                             "progress_pct": f["progress_pct"], "revenue_recognised_to_date":
                                 kamt(f["revenue_to_date_k"], cur, st)},
                "modification": {"change_order": f"CO-{f['co_number']}", "added_scope": f["added_scope"],
                                 "added_price": kamt(f["added_price_k"], cur, st),
                                 "standalone_price_of_added_scope": kamt(f["added_ssp_k"], cur, st),
                                 "relationship": distinct_note, "approval": appr, "pricing_note": why},
                "timing": status, "policy": self.POLICY,
            }))
        if surface == "change_order_form":
            return "\n".join([
                f"CHANGE ORDER CO-{f['co_number']} -- contract {f['contract_id']}",
                kv_lines([("Seller", f["seller"]), ("Customer", f["customer"]), ("Original scope", f["original_scope"]),
                          ("Original price", kamt(f["original_price_k"], cur, st)),
                          ("Complete to date", f"{f['progress_pct']}%"),
                          ("Revenue recognised to date", kamt(f["revenue_to_date_k"], cur, st)),
                          ("Change", f["added_scope"]), ("Change price", kamt(f["added_price_k"], cur, st)),
                          ("List / standalone price of change", kamt(f["added_ssp_k"], cur, st)),
                          ("Approval status", appr)], rng),
                "", "Notes: " + noisy(join_sentences([distinct_note, why]), rng, 0.1), status, "",
                "Revenue policy:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        return "\n".join([
            f"Subject: RE: CO-{f['co_number']} {f['customer']} -- revenue treatment?", "",
            f"From: {f['pm']}",
            noisy(join_sentences([f"Quick one on {f['contract_id']} ({f['original_scope']}, "
                                  f"{kamt(f['original_price_k'], cur, st)}, {f['progress_pct']}% done).",
                                  f"The customer wants {f['added_scope']} for {kamt(f['added_price_k'], cur, st)}; "
                                  f"our list price for that would be {kamt(f['added_ssp_k'], cur, st)}.",
                                  appr, distinct_note, why]), rng, 0.15), "",
            f"From: {f['analyst']}",
            f"Thanks. Revenue recognised to date is {kamt(f['revenue_to_date_k'], cur, st)}. {status} "
            f"I'll apply RR-7 -- pasting it here so we're all looking at the same text:",
            "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
        ])

    def questions(self):
        return {
            "modification_accounting": choice("How should this contract modification be accounted for?", {
                "cumulative_catch_up": "As part of the existing contract, with a cumulative catch-up to revenue "
                                       "recognised so far.",
                "no_accounting_until_approved": "Not at all yet; it is not approved in an enforceable way.",
                "prospective_treatment": "As ending the old contract and starting a new one for the remaining work.",
                "separate_contract": "As a separate new contract alongside the original.",
            }),
            "pricing_discount_depth": score("How far below its standalone selling price is the added scope priced?", [
                "Not below: at or above the standalone price.",
                "Slightly below: under a tenth off.",
                "Well below: a tenth to under a third off.",
                "Far below: a third or more off.",
            ]),
            "close_pressure": score("How pressing is it to settle the accounting for this change?", [
                "Not pressing: the quarter close is more than a month away.",
                "Moderate: it must be settled before this quarter's close, a few weeks out.",
                "High: the quarter closes within a week.",
                "Urgent: results on the old terms are already out and may need correcting.",
            ]),
            "added_scope_distinct": noul(
                "The goods or services added by the change are distinct from those in the original contract.",
                yes="They could be used on their own and are not built into the original deliverable.",
                no="They are integrated into, or modify, the original deliverable."),
            "modification_enforceable": noul(
                "The change has been approved in a way that creates enforceable rights and obligations.",
                yes="Signed, approved in writing by someone with authority, or an approved-scope change order whose "
                    "price can be estimated.",
                no="Agreed only by someone without authority, or still under negotiation."),
        }


# =========================================================================== schema 5
ARRANGEMENTS = [
    # (arrangement, what the customer gets, supplier)
    ("online hotel booking", "hotel rooms", "independent hotels"),
    ("an online marketplace for handmade furniture", "furniture", "independent makers"),
    ("resale of advertising slots on partner websites", "ad impressions", "publisher websites"),
    ("resale of third-party design-software licences", "software licences", "the software publisher"),
    ("provision of contract drivers to logistics customers", "drivers' services", "self-employed drivers"),
    ("international freight forwarding", "transport of goods", "shipping lines and airlines"),
    ("ticket sales for concerts", "concert tickets", "promoters"),
    ("sale of third-party gift cards in its stores", "gift cards", "card issuers"),
    ("a meal-delivery app", "restaurant meals", "partner restaurants"),
    ("print brokerage for corporate clients", "printed materials", "partner print shops"),
    ("managed IT services including hardware supply", "laptops and set-up", "hardware distributors"),
    ("package holidays combining flights and hotels", "holiday packages", "airlines and hotels"),
    ("event catering subcontracted to caterers", "catering", "local caterers"),
    ("a translation platform using freelance translators", "translations", "freelance translators"),
]


class PrincipalAgentAssessment(Schema):
    name = "principal_agent_assessment"
    description = ("An arrangement involving a third-party supplier: does the company control the good or service "
                   "before transfer and report revenue gross, or arrange it and report a net fee?")
    surfaces = {"json": 0.5, "arrangement_questionnaire": 0.25, "review_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        arr, good, supplier = pick(rng, ARRANGEMENTS)
        fulfil = pick_w(rng, {"company": 3, "shared": 2, "supplier": 3})
        inv = pick_w(rng, {"full": 2, "commitment": 2, "return_rights": 2, "none": 3})
        pricing = pick_w(rng, {"full": 3, "band": 2, "none": 3})
        integ = chance(rng, 0.15)
        if reg == "clear":
            if chance(rng, 0.5):
                fulfil, inv, pricing = "company", "full", "full"
            else:
                fulfil, inv, pricing, integ = "supplier", "none", "none", False
        elif reg == "conflicting":
            # the company sets prices and fronts the customer, but takes no inventory risk (or the reverse)
            if chance(rng, 0.5):
                fulfil, pricing, inv = "company", "full", "none"
            else:
                fulfil, pricing, inv = "supplier", "none", pick(rng, ["full", "commitment"])
        own_service = chance(rng, 0.3)
        contract_on_file = not chance(rng, 0.1)
        gross_k = _r10(lognormal_amount(rng, 9000, 0.9))
        take = pick(rng, [8, 10, 12, 15, 18, 22, 25])
        total_rev = _r10(gross_k * rng.uniform(4, 60))
        return {"regime": reg, "company": _seller(rng), "arrangement": arr, "good": good, "supplier": supplier,
                "fulfilment": fulfil, "inventory_risk": inv, "pricing": pricing, "integration_service": integ,
                "own_service_component": own_service, "supplier_terms_on_file": contract_on_file,
                "gross_billings_k": gross_k, "margin_pct": take, "group_revenue_k": total_rev,
                "current_presentation": pick(rng, ["gross", "net"]), "new_arrangement": chance(rng, 0.25),
                "precedent_conflict": chance(rng, 0.15), "currency": pick(rng, CURRENCIES),
                "analyst": person(rng), "bu_head": person(rng), "fy": pick(rng, [2025, 2026, 2027])}

    @staticmethod
    def points(f):
        p = {"company": 2, "shared": 1, "supplier": 0}[f["fulfilment"]]
        p += {"full": 2, "commitment": 1, "return_rights": 1, "none": 0}[f["inventory_risk"]]
        p += {"full": 2, "band": 1, "none": 0}[f["pricing"]]
        p += 2 if f["integration_service"] else 0
        return p

    def intended(self, f):
        p = self.points(f)
        bal = 0 if p <= 1 else 1 if p == 2 else 2 if p == 3 else 3 if p == 4 else 4
        principal = bal >= 3 or (bal == 2 and f["fulfilment"] == "company")
        if not f["supplier_terms_on_file"]:
            pres = "obtain_supplier_terms"
        elif f["own_service_component"] and not principal:
            pres = "split_by_component"
        else:
            pres = "gross_as_principal" if principal else "net_as_agent"
        inv = {"none": 0, "return_rights": 1, "commitment": 2, "full": 3}[f["inventory_risk"]]
        new_pres = "gross" if principal else "net"
        share = f["gross_billings_k"] / f["group_revenue_k"]
        if share >= 0.05 and new_pres != f["current_presentation"] and pres != "obtain_supplier_terms":
            signoff = "audit_committee_briefing"
        elif f["precedent_conflict"]:
            signoff = "group_policy_ruling"
        elif bal in (1, 2) or f["new_arrangement"]:
            signoff = "technical_accounting_memo"
        else:
            signoff = "controller_signoff"
        return {"revenue_presentation": pres, "control_indicator_balance": bal, "inventory_risk_level": inv,
                "controls_before_transfer": "yes" if principal else "no", "signoff_level": signoff}

    POLICY = {
        "RR-9.1": "The company is principal, and reports revenue gross, if it controls the good or service before it "
                  "passes to the customer. Indicators: it is primarily responsible for fulfilment and acceptability; "
                  "it bears inventory risk before transfer or after return; it has discretion over price. Combining "
                  "third-party inputs into its own integrated service is strong evidence of control.",
        "RR-9.2": "Otherwise it is an agent and reports only its fee or commission. Where a contract contains the "
                  "company's own service and an arranged third-party good, assess each separately.",
        "RR-9.3": "Sign-off: a change of presentation affecting 5% or more of group revenue is briefed to the audit "
                  "committee; treatments that conflict with another unit's precedent go to the group policy "
                  "committee; new arrangements or mixed indicators need a technical accounting memo; otherwise the "
                  "controller signs off.",
    }
    FULFIL = {"company": "The company handles all complaints and is responsible for remedying defects or failures.",
              "shared": "Complaints come to the company, but remedies are the supplier's responsibility under its own terms.",
              "supplier": "The supplier is responsible for delivering and for any defects; the company's terms say so."}
    INV = {"full": "The company buys and holds stock (or capacity) before any customer orders it.",
           "commitment": "The company commits to minimum volumes in advance and pays for unsold allocations.",
           "return_rights": "The company buys only after a customer order but bears the loss if customers return items.",
           "none": "The company never takes title or pays for anything unsold; unsold items stay with the supplier."}
    PRICE = {"full": "The company sets the price the customer pays.",
             "band": "The company may set the price within a band the supplier fixes.",
             "none": "The supplier sets the price; the company earns a fixed percentage."}

    def render(self, f, rng, surface):
        cur, st = f["currency"], _st(rng)
        facts = [self.FULFIL[f["fulfilment"]], self.INV[f["inventory_risk"]], self.PRICE[f["pricing"]]]
        if f["integration_service"]:
            facts.append(f"The company combines the {f['good']} with its own planning and support into one service.")
        if f["own_service_component"]:
            facts.append("Customers also buy the company's own premium service add-on in the same order.")
        terms = None if f["supplier_terms_on_file"] else "The supplier agreement has not been located; only an expired draft is on file."
        nums = (f"FY{f['fy']} gross customer billings {kamt(f['gross_billings_k'], cur, st)}; company keeps about "
                f"{f['margin_pct']}%; group revenue {kamt(f['group_revenue_k'], cur, st)}.")
        status = (f"Currently reported {f['current_presentation']}."
                  + (" The arrangement started this year." if f["new_arrangement"] else "")
                  + (" Another business unit reports a similar arrangement the other way." if f["precedent_conflict"] else ""))
        if surface == "json":
            return dumps(prune({
                "arrangement": {"company": f["company"], "description": f["arrangement"], "customer_receives": f["good"],
                                "supplied_by": f["supplier"]},
                "indicators": facts, "supplier_terms": terms,
                "figures": {"amounts_in": f"{cur} thousands", "gross_billings": f["gross_billings_k"],
                            "company_margin_pct": f["margin_pct"], "group_revenue": f["group_revenue_k"]},
                "reporting": status, "policy": self.POLICY,
            }))
        if surface == "arrangement_questionnaire":
            qa = [("Who is responsible to the customer if something goes wrong?", facts[0]),
                  ("Who holds the risk of unsold or returned items?", facts[1]),
                  ("Who sets the customer price?", facts[2]),
                  ("Anything else the company adds?", " ".join(facts[3:]) or "No."),
                  ("Is the signed supplier agreement on file?", "Yes." if f["supplier_terms_on_file"] else terms)]
            return "\n".join([
                f"PRINCIPAL / AGENT QUESTIONNAIRE -- {f['company']}: {f['arrangement']}",
                f"Completed by {f['bu_head']} (business unit); reviewed by {f['analyst']}", "",
                "\n".join(f"Q{i}. {q}\nA{i}. {noisy(a, rng, 0.08)}" for i, (q, a) in enumerate(qa, 1)), "",
                nums, status, "", "Policy:", "\n".join(f"  {k} {v}" for k, v in self.POLICY.items()),
            ])
        return "\n".join([
            f"MEMO -- gross or net: {f['arrangement']} ({f['company']})", f"From {f['analyst']} to {f['bu_head']}", "",
            noisy(join_sentences([f"Customers receive {f['good']} supplied by {f['supplier']}."] + facts + [terms]),
                  rng, 0.1), "",
            nums, status, "", "Relevant policy:", "\n".join(f"{k}: {v}" for k, v in self.POLICY.items()),
        ])

    def questions(self):
        return {
            "revenue_presentation": choice("How should revenue from this arrangement be presented?", {
                "gross_as_principal": "Gross: the full amount billed to customers, with the supplier's cost in "
                                      "cost of sales.",
                "net_as_agent": "Net: only the company's commission or margin.",
                "obtain_supplier_terms": "No conclusion until the signed supplier agreement is obtained and read.",
                "split_by_component": "Net for the third-party item, gross for the company's own service in the "
                                      "same order.",
            }),
            "control_indicator_balance": score("Taken together, where do the control indicators point?", [
                "Clearly agent.",
                "Leaning agent.",
                "Evenly balanced.",
                "Leaning principal.",
                "Clearly principal.",
            ]),
            "inventory_risk_level": score("How much inventory risk does the company carry before the customer takes "
                                          "the good or service?", [
                "None: unsold items stay with the supplier.",
                "Slight: only the risk of customer returns.",
                "Meaningful: advance volume commitments it must pay for.",
                "Full: it buys and holds stock before any order.",
            ]),
            "controls_before_transfer": noul(
                "The company controls the good or service before it is transferred to the customer.",
                yes="On balance the indicators in the policy show the company directs the good or service and "
                    "obtains its benefits before the customer does.",
                no="The supplier keeps control until the customer receives it; the company only arranges it."),
            "signoff_level": choice("Who needs to sign off the conclusion under the policy?", {
                "audit_committee_briefing": "Brief the audit committee: the presentation change is large.",
                "controller_signoff": "The controller can sign off; the case is routine.",
                "group_policy_ruling": "Refer to the group policy committee: it conflicts with another unit's treatment.",
                "technical_accounting_memo": "Write a technical accounting memo: the arrangement is new or the "
                                             "indicators are mixed.",
            }),
        }


WORKFLOW = Workflow(
    name="revenue_recognition_judgment",
    title="Revenue recognition judgments",
    description="Technical-accounting judgments on customer contracts: performance obligations and allocation, "
                "timing of recognition, variable consideration, contract modifications and principal versus agent, "
                "against the group's written revenue policy.",
    domain="accounting / revenue recognition",
    schemas=[PerformanceObligationIdentification(), TimingOverTimeVsPoint(), VariableConsiderationConstraint(),
             ContractModificationReview(), PrincipalAgentAssessment()],
    notes="Seller-side revenue accounting with fictional companies and policy paragraphs. No vendor bills, purchase "
          "orders, payment approvals or procurement decisions.",
)
