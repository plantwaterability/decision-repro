"""Supplier onboarding: should a prospective supplier be set up, and on what terms?

Third-party due diligence BEFORE a supplier exists in the vendor master: financial stability, quality
certificates, restricted-party screening hits, inherent-risk tiering, adverse media and conflicts of interest.
The decisions are about the supplier as a counterparty. No bill, purchase order, payment, goods receipt or
purchase approval appears anywhere (see the exclusion list: accounts payable and procurement exceptions).
"""
from __future__ import annotations

import re
from datetime import timedelta

from .base import (
    ORG_A,
    REAL_NAME_CLASHES,
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    first_name,
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    last_name,
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

COUNTRIES = ["Poland", "Portugal", "Mexico", "Vietnam", "Malaysia", "Czechia", "Romania", "India", "Brazil",
             "Morocco", "Thailand", "Spain", "Canada", "Slovakia", "Colombia", "Philippines", "Hungary", "Tunisia"]

# (category, kind, criticality: 1 = production-critical .. 3 = easily replaced)
CATEGORIES = [
    ("machined aluminium housings", "direct", 1), ("printed circuit board assembly", "direct", 1),
    ("cast iron valve bodies", "direct", 1), ("industrial adhesives", "direct", 2),
    ("injection-moulded clips", "direct", 2), ("powder coating", "direct", 2),
    ("laboratory reagents", "direct", 2), ("forklift maintenance", "service", 2),
    ("temporary warehouse staff", "service", 2), ("corrugated shipping cartons", "packaging", 3),
    ("stretch film and pallet wrap", "packaging", 3), ("contract cleaning", "service", 3),
]
ALT_WEEKS = {1: [10, 16, 26, 40], 2: [4, 8, 12, 18], 3: [1, 2, 3, 6]}


SUFFIX = {"direct": ["Industries", "Manufacturing", "Components", "Precision", "Engineering", "Works", "Industrial"],
          "packaging": ["Packaging", "Converting", "Industries", "Pack"],
          "service": ["Services", "Solutions", "Facilities", "Group", "Partners"],
          "software": ["Software", "Systems", "Technologies", "Digital"]}


def _name(rng, suffixes):
    """ORG_A prefix + one of `suffixes`, skipping combinations that name a real business (as base.org does)."""
    a, i = rng.choice(ORG_A), rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        n = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if n not in REAL_NAME_CLASHES:
            return n
    return n


def _supplier(rng, kind="direct"):
    return _name(rng, SUFFIX[kind]), pick(rng, COUNTRIES), ident(rng, "SUP", 6, "-")


def _people(rng, n):
    """n fictional people with distinct first names (chat handles and greetings use first names)."""
    out, seen = [], []
    while len(out) < n:
        p = person(rng)
        if p.split()[0] not in seen:
            seen.append(p.split()[0])
            out.append(p)
    return out


def _a_num(n):
    return ("an " if str(n).startswith("8") or n in (11, 18) else "a ") + str(n)


ACTIVITY = {"direct": ["recurring production supply", "a one-off tooling order", "trial samples only"],
            "packaging": ["recurring packaging supply", "trial samples only"],
            "service": ["a one-year service contract at two of our sites", "a three-month pilot at one site"],
            "software": ["a one-year subscription", "a three-month pilot"]}
SITE_WORD = {"direct": "the plant that would make our parts", "packaging": "the plant that would make our packaging",
             "service": "the branch that would serve our sites", "software": "the office that would run our service"}


def _money(x, rng, cur):
    return money(x, rng, cur, style=pick(rng, ["sym", "sym", "code"]))


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _article(word):
    return ("an " if word[:1].lower() in "aeiou" or word.startswith("NGO") else "a ") + word


def _poss(name):
    return name + ("'" if name.endswith("s") else "'s")


def _big_money(x, rng, cur):
    """Revenue-sized amounts: full figures or 'EUR 12.4m' style."""
    if chance(rng, 0.5):
        return _money(round(x, -3), rng, cur)
    return f"{cur} {x / 1e6:.1f}m"


# =========================================================================== schema 1
class FinancialStabilityReview(Schema):
    name = "financial_stability_review"
    description = "A prospective supplier's financial due-diligence file: approve, add safeguards, limit its role or decline."
    surfaces = {"json": 0.5, "analyst_memo": 0.25, "credit_extract": 0.25}

    EVENTS = {
        "lost_major_customer": "trade press reports the supplier lost its largest customer this spring",
        "refinanced": "the supplier refinanced its bank facilities four months ago",
        "new_owner": "a private-equity fund bought a majority stake last quarter",
        "cfo_resigned": "the finance director resigned within the last two months",
        "plant_fire": "a fire closed one of its two plants for six weeks last year",
    }
    OWNER_COMMENTS = [
        "Engineering has already signed off their samples and wants them live for the next build.",
        "We have used them for small jobs through a distributor for two years without a problem.",
        "They quoted well below the incumbent and can start within a month.",
        "They are the only shop in the region that can do this at our volumes.",
        "Honestly we just need a second option, the current supplier is stretched.",
    ]

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, kind, crit = pick(rng, CATEGORIES)
        sup, country, sid = _supplier(rng, kind)
        cur = pick(rng, ["USD", "EUR", "EUR", "GBP"])
        floor, cap, max_age = pick(rng, [40, 45, 50]), pick(rng, [15, 20, 25]), pick(rng, [12, 15, 18])
        revenue = round(lognormal_amount(rng, 16e6, 0.9), -5)
        years = pick(rng, [3, 5, 8, 12, 19, 31])
        bureau = between(rng, 38, 78, integer=True)
        share = between(rng, 2, cap * 1.1, digits=1)
        accounts = pick_w(rng, {"audited": 5, "reviewed": 2, "management_only": 2})
        age = between(rng, 3, max_age + 3, integer=True)
        gc = chance(rng, 0.08)
        dbt = pick(rng, [0, 2, 5, 9, 14, 21])
        trend = between(rng, -15, 20, integer=True)
        margin = between(rng, -3, 10, digits=1)
        cr = between(rng, 0.8, 2.2)
        judg = pick(rng, [0, 0, 0, 1, 2])
        guarantee = pick_w(rng, {"offered": 2, "refused": 1, "no_parent": 3, "not_asked": 2})
        event = pick_w(rng, {None: 6, "lost_major_customer": 1, "refinanced": 1, "new_owner": 1, "cfo_resigned": 1,
                             "plant_fire": 1})
        single = chance(rng, {1: 0.45, 2: 0.25, 3: 0.08}[crit])
        if reg == "clear":
            if chance(rng, 0.5):   # sound on every measure
                bureau, share, accounts, age = between(rng, 68, 92, integer=True), between(rng, 1, cap * 0.4, digits=1), \
                    "audited", pick(rng, [4, 6, 9])
                gc, dbt, trend, margin, cr, judg, event = False, pick(rng, [0, 2, 4]), between(rng, 2, 18, integer=True), \
                    between(rng, 4, 11, digits=1), between(rng, 1.4, 2.6), 0, None
            else:                  # weak on every measure
                bureau, gc, dbt = between(rng, 12, 30, integer=True), chance(rng, 0.6), pick(rng, [28, 35, 48])
                trend, margin, cr = between(rng, -30, -12, integer=True), between(rng, -9, -2, digits=1), \
                    between(rng, 0.5, 0.85)
                judg, accounts = pick(rng, [1, 2, 3]), pick(rng, ["management_only", "reviewed"])
        elif reg == "borderline":
            bureau = near(rng, floor, 8, lo=1, hi=100, integer=True)
            share = near(rng, cap, cap * 0.3, lo=0.5, digits=1)
            age = near(rng, max_age, 3, lo=2, integer=True)
            cr = near(rng, 1.0, 0.2, lo=0.3)
            margin = near(rng, 1.0, 2.5, digits=1)
        else:  # conflicting
            if chance(rng, 0.5):
                # a healthy score that has not yet caught up with a warning sign
                bureau = between(rng, 62, 85, integer=True)
                gc, event = pick(rng, [(True, None), (False, "lost_major_customer"), (True, "cfo_resigned"),
                                       (False, "new_owner")])
                dbt = pick(rng, [18, 26, 40])
            else:
                # weak figures on paper, strong backing or momentum behind them
                bureau = between(rng, 24, floor - 3, integer=True)
                accounts, guarantee, trend, years = "management_only", "offered", between(rng, 15, 45, integer=True), \
                    pick(rng, [1, 2])
            if chance(rng, 0.5):
                accounts = None    # the statements were never supplied
        spend = round(revenue * share / 100, -3)
        return {"regime": reg, "supplier": sup, "supplier_id": sid, "country": country, "category": cat,
                "criticality": crit, "currency": cur, "years_trading": years, "revenue": revenue,
                "revenue_change_pct": trend, "net_margin_pct": margin, "current_ratio": cr,
                "accounts_basis": accounts, "accounts_age_months": age if accounts else None,
                "going_concern_note": gc, "bureau_score": bureau, "score_floor": floor,
                "days_beyond_terms": dbt, "judgments_24m": judg, "planned_annual_spend": spend,
                "share_of_revenue_pct": share, "share_cap_pct": cap, "max_accounts_age_months": max_age,
                "parent_guarantee": guarantee, "recent_event": event, "single_source": single,
                "alt_qualification_weeks": pick(rng, ALT_WEEKS[crit]),
                "owner_comment": pick(rng, self.OWNER_COMMENTS + [None]), "analyst": _people(rng, 2)[0],
                "requester": person(rng), "file_id": f"DD-{rng.randint(2025, 2027)}-{rng.randint(100, 999):04d}",
                "review_date": rand_date(rng)}

    def _accounts_text(self, f):
        if not f["accounts_basis"]:
            return "No statutory or management accounts supplied yet; revenue is the bureau's estimate."
        basis = {"audited": "audited statutory accounts", "reviewed": "accountant-reviewed accounts",
                 "management_only": "management accounts only (not reviewed)"}[f["accounts_basis"]]
        return f"Latest figures are {basis}, {f['accounts_age_months']} months old."

    def _gc_text(self, f):
        return {"audited": "The auditor's report carries a material-uncertainty paragraph on going concern.",
                "reviewed": "The accountant's review report draws attention to going-concern uncertainty.",
                "management_only": "The management accounts note that continued trading depends on renewing a bank "
                                   "facility."}[f["accounts_basis"]]

    def render(self, f, rng, surface):
        cur = f["currency"]
        has_acc = f["accounts_basis"] is not None
        rev = _big_money(f["revenue"], rng, cur)
        spend = _money(f["planned_annual_spend"], rng, cur)
        g = {"offered": "offered by the parent", "refused": "requested, refused", "no_parent": "no parent company",
             "not_asked": None}[f["parent_guarantee"]]
        if surface == "json":
            return dumps(prune({
                "supplier": {"name": f["supplier"], "id": f["supplier_id"], "country": f["country"],
                             "years_trading": f["years_trading"], "category": f["category"],
                             "proposed_role": "sole source" if f["single_source"] else "one of several sources",
                             "weeks_to_qualify_alternative": f["alt_qualification_weeks"]},
                "financials": {"accounts_basis": f["accounts_basis"] or "not supplied",
                               "accounts_age_months": f["accounts_age_months"],
                               "revenue" if has_acc else "revenue_bureau_estimate": rev,
                               "revenue_change_pct": f["revenue_change_pct"] if has_acc else None,
                               "net_margin_pct": f["net_margin_pct"] if has_acc else None,
                               "current_ratio": f["current_ratio"] if has_acc else None,
                               "going_concern_emphasis": f["going_concern_note"] if has_acc else None},
                "credit": {"bureau_score": f["bureau_score"], "scale": "1-100",
                           "days_beyond_terms": f["days_beyond_terms"], "judgments_last_24m": f["judgments_24m"]},
                "relationship": {"planned_annual_spend": spend, "share_of_supplier_revenue_pct": f["share_of_revenue_pct"],
                                 "parent_guarantee": g},
                "policy": {"bureau_score_floor": f["score_floor"], "max_share_of_supplier_revenue_pct": f["share_cap_pct"],
                           "max_accounts_age_months": f["max_accounts_age_months"]},
                "notes": {"recent_event": self.EVENTS.get(f["recent_event"]),
                          "business_owner": noisy(f["owner_comment"], rng, 0.15) if f["owner_comment"] else None},
            }))
        if surface == "analyst_memo":
            p1 = [f"{f['supplier']} ({f['country']}, trading {_plural(f['years_trading'], 'year')}) has applied to supply "
                  f"{f['category']}" + (", and would be our only source." if f["single_source"]
                                        else " alongside our existing suppliers."),
                  f"Qualifying an alternative would take about {_plural(f['alt_qualification_weeks'], 'week')}."]
            p2 = [self._accounts_text(f)]
            if has_acc:
                p2.append(f"Revenue {rev} ({f['revenue_change_pct']:+d}% year on year), net margin "
                          f"{f['net_margin_pct']}%, current ratio {f['current_ratio']}.")
                if f["going_concern_note"]:
                    p2.append(self._gc_text(f))
            else:
                p2.append(f"Bureau revenue estimate: {rev}.")
            p3 = [f"Bureau score {f['bureau_score']} against our floor of {f['score_floor']}.",
                  f"They pay their own suppliers {f['days_beyond_terms']} days beyond terms on average"
                  + (f" and have {_plural(f['judgments_24m'], 'court judgment')} in the last two years." if f["judgments_24m"]
                     else "; no judgments on file."),
                  f"Our planned spend of roughly {spend} a year is about {f['share_of_revenue_pct']}% of their revenue "
                  f"(policy ceiling {f['share_cap_pct']}%)."]
            if f["recent_event"]:
                p3.append(f"Note that {self.EVENTS[f['recent_event']]}.")
            p3.append({"offered": "Their parent has offered a written guarantee.",
                       "refused": "We asked for a parent guarantee; they declined.",
                       "no_parent": "There is no parent company.", "not_asked": ""}[f["parent_guarantee"]])
            out = [f"SUPPLIER FINANCIAL REVIEW  {f['file_id']}",
                   f"Prepared by {f['analyst']} for {f['requester']} -- {fmt_date(f['review_date'], rng)}", "",
                   join_sentences(p1), "", noisy(join_sentences(p2), rng, 0.1), "", noisy(join_sentences(p3), rng, 0.1)]
            if f["owner_comment"]:
                out += ["", f"Business owner's comment: \"{noisy(f['owner_comment'], rng, 0.2)}\""]
            return "\n".join(out)
        # credit_extract: a bureau printout with the buyer's reviewer notes underneath
        rows = [("Subject", f"{f['supplier'].upper()} ({f['country']})"), ("Supplier ref", f["supplier_id"]),
                ("Years trading", f["years_trading"]), ("Score (1-100)", f["bureau_score"]),
                ("Days beyond terms", f["days_beyond_terms"]), ("Judgments, 24 months", f["judgments_24m"]),
                ("Filed accounts", (f"{f['accounts_basis'].replace('_', ' ')}, {f['accounts_age_months']} months old"
                                    if has_acc else "none on file")),
                ("Turnover", rev + ("" if has_acc else " (estimate)"))]
        if has_acc:
            rows += [("Turnover change", f"{f['revenue_change_pct']:+d}%"), ("Net margin", f"{f['net_margin_pct']}%"),
                     ("Current ratio", f["current_ratio"]),
                     ("Going-concern note", self._gc_text(f) if f["going_concern_note"] else "none")]
        notes = [f"Category: {f['category']}; " + ("sole source" if f["single_source"] else "multi-sourced")
                 + f"; alternative needs ~{f['alt_qualification_weeks']} "
                 + ("wk" if f["alt_qualification_weeks"] == 1 else "wks") + " to qualify",
                 f"Planned spend {spend}/yr = {f['share_of_revenue_pct']}% of their turnover (cap {f['share_cap_pct']}%)",
                 f"Score floor {f['score_floor']}; accounts must be <= {f['max_accounts_age_months']} months old",
                 ("Parent guarantee: " + g) if g else None,
                 self.EVENTS[f["recent_event"]].capitalize() if f["recent_event"] else None,
                 (f"Owner: {noisy(f['owner_comment'], rng, 0.2)}" if f["owner_comment"] else None)]
        return ("BUSINESS CREDIT SUMMARY\n" + kv_lines(rows, rng, style=pick(rng, ["aligned", "colon"])) +
                "\n\nReviewer notes:\n" + bullets([n for n in notes if n], rng))

    def questions(self):
        return {
            "onboarding_recommendation": choice("What should the due-diligence review recommend for this supplier?", {
                "approve": "Approve the supplier for the proposed role on standard terms.",
                "approve_with_safeguards": "Approve, but only with protections in place such as a parent guarantee, "
                                           "a volume cap or quarterly financial checks.",
                "approve_as_secondary_source": "Approve it only as a secondary source, so the business never depends "
                                               "on it alone.",
                "defer_for_current_financials": "Hold the decision until current, independently prepared financial "
                                                "statements are supplied.",
                "decline": "Decline the supplier; its finances are too weak for any role.",
            }),
            "primary_safeguard": choice("Which safeguard would do most to protect the business if this supplier is used?", {
                "dual_sourcing": "Qualify a second supplier so production never rests on this one.",
                "financial_monitoring": "Track the supplier's filings, credit score and payment behaviour every quarter.",
                "parent_guarantee": "Obtain a written performance guarantee from the supplier's parent company.",
                "volume_cap": "Cap our volume so the supplier does not become dependent on our business.",
                "none_needed": "Nothing special; ordinary supplier management is enough.",
            }),
            "distress_likelihood": score("How likely is the supplier to fall into serious financial difficulty "
                                         "within the next twelve months?", [
                "Remote: profitable, liquid and paying its own suppliers on time.",
                "Low: a few soft spots, none that threaten the business.",
                "Elevated: several indicators point to financial strain.",
                "High: it may not get through the year without new money.",
            ]),
            "supply_exposure": score("If this supplier failed, how exposed would our own operations be?", [
                "Hardly: other suppliers could absorb the volume straight away.",
                "Somewhat: switching would take a few weeks of effort.",
                "Considerably: a switch would take months and disrupt output.",
                "Severely: a production line would stop until a replacement was qualified.",
            ]),
            "statements_sufficient": noul(
                "The financial statements on file are recent and independent enough to base a decision on.",
                yes="Audited or reviewed statements are on file and fall within, or right at, the policy's age limit.",
                no="Statements are missing, prepared only by management, or clearly older than the policy allows."),
        }


# =========================================================================== schema 2
CERT_CATS = [
    ("brake caliper brackets", "automotive", "IATF 16949", ["stamping", "machining", "zinc-nickel plating"]),
    ("seat-belt anchor bolts", "automotive", "IATF 16949", ["cold heading", "heat treatment", "plating"]),
    ("infusion-pump housings", "medical devices", "ISO 13485", ["injection moulding", "ultrasonic welding",
                                                                 "clean-room assembly"]),
    ("catheter extrusion tubing", "medical devices", "ISO 13485", ["extrusion", "cutting", "EO sterilisation"]),
    ("food-contact trays", "food packaging", "FSSC 22000", ["thermoforming", "printing", "palletising"]),
    ("printed pouches for dry foods", "food packaging", "BRCGS Packaging Materials", ["gravure printing",
                                                                                      "lamination", "pouch making"]),
    ("titanium fasteners", "aerospace", "AS9100D", ["thread rolling", "heat treatment", "passivation"]),
    ("cabin-lighting wiring harnesses", "aerospace", "AS9100D", ["crimping", "harness assembly", "continuity test"]),
    ("control-panel enclosures", "industrial equipment", "ISO 9001", ["laser cutting", "bending", "powder coating"]),
    ("conveyor rollers", "industrial equipment", "ISO 9001", ["tube cutting", "bearing press-fit", "balancing"]),
]
DECLARATION = {"automotive": "material data submission", "medical devices": "biocompatibility test report",
               "food packaging": "food-contact migration test", "aerospace": "first article inspection report",
               "industrial equipment": "RoHS / REACH declaration"}


def _cert_body(rng):
    return f"{pick(rng, ['Norland', 'Veritrust', 'Castellan', 'Brightmark', 'Oriel', 'Summit Line'])} " \
           f"{pick(rng, ['Certification', 'Registrar Services', 'Assurance', 'Quality Registrar'])}"


def _cap(s):
    return s[0].upper() + s[1:] if s else s


class CertificationVerification(Schema):
    name = "certification_verification"
    description = "A prospective supplier's quality certificates checked against what the supplied category requires."
    surfaces = {"json": 0.5, "cert_checklist": 0.25, "sqe_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, sector, std, procs = pick(rng, CERT_CATS)
        sup, country, sid = _supplier(rng, "packaging" if sector == "food packaging" else "direct")
        status = pick_w(rng, {"valid": 6, "expiring": 2, "expired": 1, "suspended": 1})
        register = pick_w(rng, {"listed_valid": 6, "not_checked": 2, "not_found": 1, "number_mismatch": 1})
        accred = pick_w(rng, {"accredited": 7, "unknown": 2, "not_accredited": 1})
        scope = pick_w(rng, {"full": 6, "excludes_process": 3, "different_product": 1})
        site = pick_w(rng, {"same_site": 6, "hq_only": 2, "other_site": 2})
        majors, minors = pick(rng, [0, 0, 0, 1, 2]), pick(rng, [0, 1, 2, 3, 5, 8])
        decl = pick_w(rng, {"current": 5, "outdated": 2, "missing": 2})
        sample = pick_w(rng, {"passed": 4, "passed_with_deviations": 3, "failed": 1, "not_yet": 2})
        weeks = pick(rng, [2, 4, 6, 10, 16, 24])
        if reg == "clear":
            if chance(rng, 0.5):
                status, register, accred, scope, site, majors, decl, sample = (
                    "valid", "listed_valid", "accredited", "full", "same_site", 0, "current", "passed")
            else:
                status, register, accred, site = pick(rng, ["expired", "suspended"]), "not_found", "not_accredited", \
                    "other_site"
                majors, sample = pick(rng, [1, 2]), pick(rng, ["failed", "not_yet"])
        elif reg == "borderline":
            status = pick(rng, ["expiring", "expiring", "valid"])
            majors = pick(rng, [0, 1, 1])
            scope = pick(rng, ["full", "excludes_process"])
            site = pick(rng, ["same_site", "hq_only"])
            sample = pick(rng, ["passed_with_deviations", "passed", "not_yet"])
        else:  # conflicting: the paper looks fine and one check says otherwise, or the reverse
            if chance(rng, 0.5):
                status, accred, scope, site, decl = "valid", "accredited", "full", "same_site", "current"
                register, sample = pick(rng, [("listed_valid", "failed"), ("number_mismatch", "passed"),
                                              ("not_found", "passed"), ("listed_valid", "passed_with_deviations")])
                if register == "listed_valid":
                    majors = 2
            else:
                accred, register, site = pick(rng, ["not_accredited", "unknown"]), "not_checked", "same_site"
                sample, majors, decl = "passed", 0, "current"
        days = {"valid": between(rng, 150, 900, integer=True), "expiring": near(rng, 35, 30, lo=3, integer=True),
                "expired": -between(rng, 5, 70, integer=True), "suspended": between(rng, 60, 500, integer=True)}[status]
        if status == "suspended" and register == "listed_valid":
            register = "listed_suspended"
        claim = None
        if status == "expiring":
            claim = pick(rng, ["Recertification audit is booked; the registrar says the new certificate follows within "
                               "a month.", "Our renewal audit is next week.", None])
        elif status == "expired":
            claim = pick(rng, ["The registrar has granted a grace period while the recertification report is finalised.",
                               "Renewal is in progress, we are waiting on paperwork."])
        elif site == "hq_only":
            claim = "Our group certificate covers all our plants, so the second site is included."
        elif majors:
            claim = pick(rng, ["The open findings are documentation issues and will be closed this month.",
                               "Corrective actions for the audit findings are under way.", None])
        return {"regime": reg, "supplier": sup, "supplier_id": sid, "country": country, "category": cat,
                "sector": sector, "standard": std, "processes": procs, "cert_status": status,
                "days_to_expiry": days, "register_check": register, "accreditation": accred,
                "scope_fit": scope, "missing_process": pick(rng, procs[1:]) if scope == "excludes_process" else None,
                "site_coverage": site, "majors_open": majors, "minors_open": minors, "declaration": decl,
                "sample_result": sample, "weeks_to_supply_start": weeks, "supplier_claim": claim,
                "certifier": _cert_body(rng), "cert_no": f"{pick(rng, ['QC', 'CERT', 'RG'])}-{rng.randint(10000, 99999)}",
                "sqe": person(rng), "check_date": rand_date(rng)}

    def _scope_text(self, f):
        procs = ", ".join(f["processes"][:-1]) + " and " + f["processes"][-1]
        if f["scope_fit"] == "full":
            return f"Manufacture of {f['category']} including {procs}"
        if f["scope_fit"] == "excludes_process":
            kept = [p for p in f["processes"] if p != f["missing_process"]]
            return f"Manufacture of {f['category']} ({', '.join(kept)}); {f['missing_process']} subcontracted"
        return "Distribution and warehousing of industrial components"

    def _site_text(self, f):
        return {"same_site": f"{f['country']} production site (the site that would make our parts)",
                "hq_only": "head office only; the production site is not named",
                "other_site": "a different plant from the one quoting our parts"}[f["site_coverage"]]

    def render(self, f, rng, surface):
        exp = f["check_date"] + timedelta(days=f["days_to_expiry"])
        reg_txt = {"listed_valid": "listed, valid", "listed_suspended": "listed, SUSPENDED", "not_found": "not found",
                   "number_mismatch": "listed under a different certificate number", "not_checked": None}
        dec = DECLARATION[f["sector"]]
        if surface == "json":
            return dumps(prune({
                "supplier": {"name": f["supplier"], "id": f["supplier_id"], "country": f["country"],
                             "category": f["category"], "sector": f["sector"],
                             "weeks_until_supply_starts": f["weeks_to_supply_start"]},
                "certificate": {"standard": f["standard"], "number": f["cert_no"], "issued_by": f["certifier"],
                                "certifier_accreditation": None if f["accreditation"] == "unknown" else f["accreditation"],
                                "expires": fmt_date(exp, style="iso"),
                                "scope": self._scope_text(f), "sites_covered": self._site_text(f)},
                "verification": {"online_register": reg_txt[f["register_check"]],
                                 "last_audit_major_findings_open": f["majors_open"],
                                 "last_audit_minor_findings_open": f["minors_open"]},
                "evidence": {dec.lower().replace(" / ", "_").replace(" ", "_"): f["declaration"],
                             "sample_inspection": f["sample_result"]},
                "supplier_statement": f["supplier_claim"],
                "checked_on": fmt_date(f["check_date"], style="iso"),
            }))
        if surface == "cert_checklist":
            rows = [["Certificate", f"{f['standard']} no. {f['cert_no']}", ""],
                    ["Issuer", f["certifier"], show(None if f["accreditation"] == "unknown" else f["accreditation"],
                                                    rng, lambda v: v.replace("_", " "))],
                    ["Expiry", fmt_date(exp, rng), f"{f['days_to_expiry']} days left" if f["days_to_expiry"] >= 0
                     else f"lapsed {-f['days_to_expiry']} days ago"],
                    ["Register check", show(reg_txt[f["register_check"]], rng), ""],
                    ["Scope", self._scope_text(f), ""],
                    ["Site", self._site_text(f), ""],
                    ["Open audit findings", f"{f['majors_open']} major / {f['minors_open']} minor", ""],
                    [_cap(dec), f["declaration"].replace("_", " "), ""],
                    ["Sample inspection", f["sample_result"].replace("_", " "), ""]]
            head = (f"SUPPLIER QUALIFICATION CHECKLIST -- {f['supplier']} ({f['supplier_id']})\n"
                    f"Category: {f['category']} | supply due to start in ~{f['weeks_to_supply_start']} weeks | "
                    f"checked {fmt_date(f['check_date'], rng)}\n")
            tail = f"\n\nSupplier says: \"{noisy(f['supplier_claim'], rng, 0.15)}\"" if f["supplier_claim"] else ""
            return head + "\n" + table(["Item", "Evidence", "Note"], rows, rng) + tail
        # sqe_email
        mgr = [p for p in _people(rng, 2) if p.split()[0] != f["sqe"].split()[0]][0]
        s = [f"I've been through the paperwork from {f['supplier']} ({f['country']}) for the {f['category']} package.",
             f"They hold {f['standard']} certificate {f['cert_no']} from {f['certifier']}"
             + {"accredited": ", an accredited body", "not_accredited": ", which I can't find on any accreditation "
                "body's list", "unknown": ""}[f["accreditation"]] + ".",
             ("It lapsed " + str(-f["days_to_expiry"]) + " days ago." if f["days_to_expiry"] < 0 else
              f"It runs to {fmt_date(exp, rng)}."),
             {"listed_valid": "The online register shows it as valid.",
              "listed_suspended": "The register lists it as suspended.",
              "not_found": "I couldn't find it on the certifier's online register.",
              "number_mismatch": "The register shows them under a different certificate number.",
              "not_checked": "I haven't checked the register yet."}[f["register_check"]],
             f"Scope reads: \"{self._scope_text(f)}\". Site coverage: {self._site_text(f)}.",
             (f"Their last surveillance audit left {f['majors_open']} major and {f['minors_open']} minor findings open."
              if f["majors_open"] or f["minors_open"] else "No open findings from their last audit."),
             f"{_cap(dec)}: {f['declaration'].replace('_', ' ')}. Samples: {f['sample_result'].replace('_', ' ')}.",
             (f"Their comment: {f['supplier_claim']}" if f["supplier_claim"] else ""),
             f"The programme wants supply to start in about {_plural(f['weeks_to_supply_start'], 'week')}."]
        return "\n".join([f"Subject: {f['supplier']} -- certificate review", f"From: {f['sqe']}",
                          f"To: {mgr}", f"Date: {fmt_date(f['check_date'], rng)}", "",
                          f"Hi {mgr.split()[0]},", "",
                          noisy(join_sentences(s), rng, 0.12), "", f["sqe"].split()[0]])

    def questions(self):
        return {
            "certificate_acceptance": choice("How should supplier quality treat the certificates on file?", {
                "accept": "Accept them as sufficient evidence for this category.",
                "accept_pending_renewal": "Accept provisionally, on condition that proof of renewal or of closed "
                                          "findings arrives by a set date.",
                "accept_for_limited_scope": "Accept only for the parts and processes the certificates clearly cover.",
                "require_onsite_audit": "Do not rely on them; our own auditor must assess the site first.",
                "reject_for_category": "Reject; the supplier cannot be qualified for this category on this evidence.",
            }),
            "quality_system_confidence": score("How much confidence do the certificates and audit history give in "
                                               "the supplier's quality system?", [
                "Little: the paperwork is lapsed, unverifiable or beside the point.",
                "Some: a system exists, but with gaps that matter for our parts.",
                "Reasonable: a working system with minor gaps.",
                "High: a verified, current system covering exactly what we would buy.",
            ]),
            "auditor_visit_timing": score("How soon does our own supplier-quality auditor need to be on site?", [
                "No visit beyond the routine multi-year cycle: a current certificate, valid on the register, from an "
                "accredited body, covering our site and full scope, no open major findings, a current declaration "
                "and passed samples.",
                "Some time within the first year: the same, except for a minor point (certificate expiring soon, "
                "register not yet checked, accreditation unknown, declaration outdated or missing, or samples passed "
                "with deviations or not yet inspected).",
                "Before the first production parts ship: open major findings, a scope that leaves out one of our "
                "processes, a site that is not named or is a different plant, or a failed sample.",
                "Straight away: the certificate is expired or suspended, missing from the register or listed under "
                "another number, or the issuer is not accredited.",
            ]),
            "certificates_cover_supply": noul(
                "The certificates on file cover the site and the processes that would make our parts.",
                yes="The named site and the certified scope both include the work we would buy.",
                no="The site differs or is not named, or the scope leaves out a process our parts need."),
            "certificate_doubt": noul("There is reason to doubt that a certificate on file is genuine or in force."),
        }


# =========================================================================== schema 3
LISTS = {
    "asset_freeze": "Consolidated Asset-Freeze List",
    "export_denied": "Export Denied Parties Register",
    "debarment": "Development Bank Debarment List",
    "internal": "Group Do-Not-Engage List",
}
SPELL = {"Yusuf": "Youssef", "Dmitri": "Dmitriy", "Farid": "Fareed", "Hana": "Hanna", "Idris": "Idriss",
         "Olga": "Olha", "Sanjay": "Sanjai", "Viktor": "Victor", "Julia": "Yulia", "Pedro": "Petro", "Zofia": "Sofia",
         "Jonas": "Jonah", "Oskar": "Oscar", "Rafael": "Raphael", "Tomas": "Thomas", "Felix": "Feliks",
         "Marek": "Mark", "Carmen": "Karmen", "Esther": "Ester", "Ivo": "Iwo", "Noor": "Nour", "Emeka": "Emeca"}
DEACCENT = str.maketrans({"ø": "o", "ü": "u", "ö": "o", "é": "e", "á": "a", "í": "i", "ć": "c", "ř": "r", "ä": "a",
                          "ñ": "n", "ž": "z", "š": "s", "č": "c"})


def _person_variant(first, last, rng, closeness):
    last_u = last.translate(DEACCENT).upper()
    if closeness == "exact":
        return f"{last_u}, {first}"
    if closeness == "close":
        kind = pick(rng, ["spelling", "initial", "deaccent"])
        if kind == "spelling" and first in SPELL:
            return f"{last_u}, {SPELL[first]}"
        if kind == "initial":
            return f"{last_u}, {first} {pick(rng, 'ABDEHKMRS')}."
        return f"{last_u}, {first.translate(DEACCENT)}"
    if chance(rng, 0.4):
        return f"{last_u}, {first[0]}."
    others = [n for n in ("Adel", "Boris", "Carla", "Dario", "Elena", "Faisal", "Gennady", "Helga", "Ilya", "Jana",
                          "Karim", "Lena", "Milan", "Nadia", "Omar", "Petra", "Radu", "Selim", "Tereza", "Vadim")
              if n[0] == first[0] and n != first]
    return f"{last_u}, {pick(rng, others or ['Anton', 'Mira', 'Goran'])}"


def _entity_variant(name, rng, closeness):
    a, b = name.split(" ", 1)
    if closeness == "exact":
        return f"{name.upper()} {pick(rng, ['LLC', 'LTD', 'S.A.', 'SP. Z O.O.', 'S.R.L.'])}"
    if closeness == "close":
        b2 = {"Industries": "Industrial", "Group": "Group Holdings", "Components": "Component Trading",
              "Works": "Engineering Works", "Precision": "Precision Tools", "Engineering": "Engineering Group",
              "Plastics": "Plastic Products", "Packaging": "Packaging Group", "Services": "Service Co",
              "Solutions": "Solutions International", "Manufacturing": "Mfg"}.get(b, b + " International")
        return f"{a.upper()} {b2.upper()}"
    return f"{a.upper()} {pick(rng, ['TRADING FZE', 'MARITIME LTD', 'IMPEX', 'GENERAL TRADING'])}"


class SanctionsScreeningMatch(Schema):
    name = "sanctions_screening_match"
    description = "A restricted-party screening hit raised while onboarding a supplier: discount, ask, escalate or block."
    surfaces = {"json": 0.5, "screening_alert": 0.25, "case_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, kind, _crit = pick(rng, CATEGORIES)
        sup, country, sid = _supplier(rng, kind)
        role = pick_w(rng, {"supplier_entity": 3, "director": 4, "beneficial_owner": 3})
        lst = pick_w(rng, {"asset_freeze": 3, "export_denied": 3, "debarment": 2, "internal": 1})
        closeness = pick_w(rng, {"exact": 2, "close": 5, "loose": 3})
        fn, ln = first_name(rng), last_name(rng)
        yob_ours = maybe(rng, between(rng, 1952, 1990, integer=True), 0.35)
        yob_delta = pick(rng, [0, 0, 1, 2, 7, 15])
        country_listed = country if chance(rng, 0.5) else pick(rng, [c for c in COUNTRIES if c != country])
        regno_match = pick_w(rng, {"match": 1, "mismatch": 2, "not_listed": 3})
        alias = chance(rng, 0.3)
        own_pct = between(rng, 5, 70, integer=True) if role == "beneficial_owner" else None
        our_regno = f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"
        explanation = pick_w(rng, {"none_yet": 3, "documents": 3, "denial_only": 2, "resigned_recently": 1})
        if reg == "clear":
            if chance(rng, 0.5):   # obvious false positive: identifiers contradict the listing
                closeness, yob_delta, regno_match = "loose", pick(rng, [12, 18, 25]), "mismatch"
                yob_ours = yob_ours or between(rng, 1960, 1990, integer=True)
                country_listed = pick(rng, [c for c in COUNTRIES if c != country])
                explanation = "documents"
            else:                  # obvious true match
                closeness, yob_delta, regno_match, country_listed = "exact", 0, "match", country
                yob_ours = yob_ours or between(rng, 1955, 1985, integer=True)
                explanation = pick(rng, ["none_yet", "denial_only"])
        elif reg == "borderline":
            closeness = "close"
            yob_delta = pick(rng, [0, 1, 1, 2, 3])
            yob_ours = maybe(rng, yob_ours or between(rng, 1955, 1985, integer=True), 0.5)
            regno_match = "not_listed"
        else:  # conflicting
            if chance(rng, 0.5):
                closeness, yob_delta = "exact", 0
                yob_ours = yob_ours or between(rng, 1955, 1985, integer=True)
                country_listed = pick(rng, [c for c in COUNTRIES if c != country])
                explanation = pick(rng, ["documents", "denial_only"])
                regno_match = pick(rng, ["mismatch", "not_listed"])
            else:
                closeness, alias = "loose", True
                country_listed = country
                explanation = "resigned_recently" if role != "supplier_entity" else "denial_only"
        if role == "supplier_entity" and explanation == "resigned_recently":
            explanation = "denial_only"
        yob_listed = None if yob_ours is None and chance(rng, 0.3) else \
            ((yob_ours or between(rng, 1955, 1985, integer=True)) + pick(rng, [-1, 1]) * yob_delta)
        score_tool = {"exact": between(rng, 96, 100, integer=True), "close": between(rng, 85, 95, integer=True),
                      "loose": between(rng, 75, 86, integer=True)}[closeness]
        listed_regno = {"match": our_regno, "not_listed": None,
                        "mismatch": f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"}[regno_match]
        return {"regime": reg, "supplier": sup, "supplier_id": sid, "country": country, "category": cat,
                "matched_role": role, "list": lst, "closeness": closeness, "first": fn, "last": ln,
                "listed_name": (_entity_variant(sup, rng, closeness) if role == "supplier_entity"
                                else _person_variant(fn, ln, rng, closeness)),
                "alias_hit": alias, "tool_score": score_tool, "yob_ours": yob_ours, "yob_listed": yob_listed,
                "country_listed": country_listed, "regno_match": regno_match, "our_regno": our_regno,
                "listed_regno": listed_regno, "ownership_pct": own_pct, "explanation": explanation,
                "activity": pick(rng, ACTIVITY[kind]), "kind": kind,
                "alert_id": ident(rng, "RPS", 7, ""), "analyst": person(rng), "raised": rand_date(rng),
                "list_added_year": between(rng, 2014, 2026, integer=True)}

    def _screened(self, f):
        if f["matched_role"] == "supplier_entity":
            return f"{f['supplier']} (the applicant)"
        who = f"{f['first']} {f['last']}"
        if f["matched_role"] == "director":
            return f"{who}, director of {f['supplier']}"
        return f"{who}, holds {f['ownership_pct']}% of {f['supplier']}"

    def _explain(self, f):
        return {"none_yet": None,
                "documents": ("Supplier sent a passport copy and a residence certificate for the individual."
                              if f["matched_role"] != "supplier_entity"
                              else "Supplier sent its registry extract and articles of association."),
                "denial_only": "Supplier replied by email that they are 'not connected to that party'; no documents.",
                "resigned_recently": f"Supplier says {f['first']} {f['last']} "
                                     + ("sold the shareholding" if f["matched_role"] == "beneficial_owner"
                                        else "stepped down from the board")
                                     + " last week, after our screening questions were sent."}[f["explanation"]]

    def render(self, f, rng, surface):
        ind = f["matched_role"] != "supplier_entity"
        ours = {"year_of_birth": f["yob_ours"], "country": f["country"]} if ind else \
            {"registration_no": f["our_regno"], "country": f["country"]}
        listed = {"name": f["listed_name"], "list": LISTS[f["list"]], "listed_since": f["list_added_year"],
                  "year_of_birth": f["yob_listed"] if ind else None, "address_country": f["country_listed"],
                  "registration_no": None if ind else f["listed_regno"],
                  "matched_on_alias": f["alias_hit"] or None}
        if surface == "json":
            return dumps(prune({
                "alert": {"id": f["alert_id"], "raised": fmt_date(f["raised"], style="iso"), "match_score": f["tool_score"],
                          "screened_record": self._screened(f), "status": "open"},
                "listed_party": listed,
                "screened_details": ours,
                "onboarding": {"supplier_id": f["supplier_id"], "category": f["category"],
                               "intended_activity": f["activity"]},
                "supplier_response": self._explain(f),
            }))
        if surface == "screening_alert":
            rows = [("Alert", f["alert_id"]), ("Raised", fmt_date(f["raised"], rng)),
                    ("Screened record", self._screened(f)), ("List", LISTS[f["list"]]),
                    ("Listed name", f["listed_name"] + ("  [alias entry]" if f["alias_hit"] else "")),
                    ("Match score", f["tool_score"]), ("Listed since", f["list_added_year"])]
            if ind:
                rows += [("Listed YOB", show(f["yob_listed"], rng)), ("Screened YOB", show(f["yob_ours"], rng))]
            else:
                rows += [("Listed reg. no.", f["listed_regno"] or "not given on list"),
                         ("Screened reg. no.", f["our_regno"])]
            rows += [("Listed address country", f["country_listed"]), ("Screened country", f["country"]),
                     ("Category", f["category"]), ("Intended activity", f["activity"])]
            out = "RESTRICTED-PARTY SCREENING -- POTENTIAL MATCH\n" + kv_lines(rows, rng, style="aligned")
            ex = self._explain(f)
            return out + (f"\n\nAnalyst log: {noisy(ex, rng, 0.1)}" if ex else "\n\nAnalyst log: (empty)")
        # case_note
        s = [f"Screening of {f['supplier']} ({f['supplier_id']}) returned {_a_num(f['tool_score'])}-point match against the "
             f"{LISTS[f['list']]}. They would supply {f['category']}.",
             f"The hit is on {self._screened(f)}; the listed entry reads \"{f['listed_name']}\""
             + (" (an alias line, not the primary name)" if f["alias_hit"] else "") + f", listed since {f['list_added_year']}."]
        if ind:
            if f["yob_ours"] is None and f["yob_listed"] is None:
                s.append("Neither side gives a year of birth.")
            elif f["yob_ours"] is None:
                s.append(f"The listing gives {f['yob_listed']} as year of birth; we hold no date of birth for our person.")
            elif f["yob_listed"] is None:
                s.append(f"Our person was born in {f['yob_ours']}; the listing has no date of birth.")
            else:
                s.append(f"Years of birth: listing {f['yob_listed']}, ours {f['yob_ours']}.")
        else:
            s.append({"match": f"The listing's registration number, {f['listed_regno']}, is the one on the "
                               f"applicant's registry extract.",
                      "mismatch": f"The listing carries registration number {f['listed_regno']}; the applicant's is "
                                  f"{f['our_regno']}.",
                      "not_listed": "The listing gives no registration number."}[f["regno_match"]])
        s.append(f"Listed address is in {f['country_listed']}; the supplier is in {f['country']}.")
        s.append(f"The onboarding request covers {f['activity']}.")
        ex = self._explain(f)
        s.append(ex or "We have not heard back from the supplier yet.")
        return (f"CASE NOTE {f['alert_id']} -- {f['analyst']}, {fmt_date(f['raised'], rng)}\n\n"
                + noisy(join_sentences(s), rng, 0.1))

    def questions(self):
        return {
            "screening_resolution": choice("How should the compliance analyst resolve this screening hit?", {
                "block_onboarding": "Stop onboarding and record the supplier as a party we will not engage.",
                "discount_as_false_positive": "Record the hit as a false positive, with reasons, and let onboarding "
                                              "continue.",
                "escalate_to_trade_counsel": "Hand the case to trade-compliance counsel for a legal view before anything "
                                             "else happens.",
                "request_identifiers": "Ask the supplier for identifying documents, such as dates of birth or registry "
                                       "papers, before deciding.",
            }),
            "link_strength": score("How strongly does the evidence tie the supplier to the listed party?", [
                "Not at all: the identifiers contradict the listing.",
                "Weakly: a resemblance in name and little else.",
                "Moderately: some details fit, none of them decisive.",
                "Strongly: most identifiers line up with the listing.",
                "Conclusively: it is the same party.",
            ]),
            "consequence_if_genuine": score("If the hit proved genuine, how serious would dealing with this supplier be?", [
                "Minor: the hit is on the Group Do-Not-Engage List, an internal policy with no legal restriction "
                "behind it.",
                "Moderate: the hit is on the Development Bank Debarment List, a problem with lenders or customers who "
                "fund the work.",
                "Serious: the hit is on the Export Denied Parties Register, a probable breach of trade restrictions "
                "needing disclosure.",
                "Grave: the hit is on the Consolidated Asset-Freeze List, a prohibited dealing with a party whose "
                "assets are frozen.",
            ]),
            "identifiers_settle_it": noul(
                "The identifying details already on file are enough to settle whether the hit is genuine.",
                yes="For a company, the listing gives a registration number that either matches or differs from the "
                    "applicant's; for a person, both years of birth are known and differ by 5 or more years, or "
                    "match exactly with the same country.",
                no="The listing gives no registration number, a year of birth is missing on either side, or the "
                   "years differ by 1 to 4, or match only with a different country."),
            "supplier_account_credible": noul("The supplier's account of the hit is credible and backed by documents."),
        }


# =========================================================================== schema 4
RT_CATS = [
    ("payroll outsourcing", "service", "employee_personal", "none"),
    ("employee engagement survey platform", "software", "employee_personal", "none"),
    ("direct-mail printing", "service", "customer_personal", "none"),
    ("CAD file conversion", "service", "confidential_designs", "none"),
    ("HVAC maintenance", "service", "none", "hazardous_work"),
    ("scaffolding hire and erection", "service", "none", "hazardous_work"),
    ("industrial gases", "direct", "none", "visits"), ("speciality resin", "direct", "none", "none"),
    ("fleet telematics software", "software", "system_access", "none"),
    ("staff canteen catering", "service", "none", "regular_staff"),
    ("e-waste and laptop recycling", "service", "customer_personal", "visits"),
    ("label and barcode printers", "direct", "none", "visits"),
    ("building management system support", "service", "system_access", "visits"),
    ("corrugated shipping cartons", "packaging", "none", "none"),
]
DATA_TXT = {"none": "no access to our data", "confidential_designs": "our confidential drawings and designs",
            "employee_personal": "personal data about our employees", "customer_personal": "personal data about our customers",
            "system_access": "remote access into our operational systems"}
ONSITE_TXT = {"none": "no work on our sites", "visits": "occasional delivery and service visits",
              "regular_staff": "their staff on our premises every day", "hazardous_work": "hazardous work on our sites "
              "(height, hot work or confined spaces)"}
ESG = ["no modern-slavery statement", "conflict-minerals template not returned", "environmental permit renewal overdue",
       "no whistleblowing channel", "standard working week above 60 hours"]


class RiskTierAssignment(Schema):
    name = "risk_tier_assignment"
    description = "A new supplier's inherent-risk questionnaire: set its oversight tier and decide whether it may go live."
    surfaces = {"json": 0.45, "intake_form": 0.3, "chat_thread": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        quiet = [c for c in RT_CATS if c[2] == "none" and c[3] in ("none", "visits")]
        exposed = [c for c in RT_CATS if c[2] != "none" or c[3] == "hazardous_work"]
        sub_reg = pick(rng, ["a", "b"])
        pool = {"clear": quiet if sub_reg == "a" else [c for c in RT_CATS if c[2] != "none"],
                "conflicting": exposed if sub_reg == "a" else quiet}.get(reg, RT_CATS)
        cat, kind, data, onsite = pick(rng, pool)
        sup, country, sid = _supplier(rng, kind)
        spend = round(lognormal_amount(rng, 180000, 1.0), -3)
        single = chance(rng, 0.3)
        alt = pick(rng, [1, 3, 6, 12, 24])
        sub = pick_w(rng, {"none": 5, "disclosed": 3, "suspected_undisclosed": 1})
        qpct = pick(rng, [55, 70, 80, 90, 100, 100])
        cyber = between(rng, 40, 90, integer=True) if data != "none" else None
        cyber_status = "scored" if cyber is not None else "not_applicable"
        esg = rng.sample(ESG, pick(rng, [0, 0, 0, 1, 1, 2]))
        insurance = pick_w(rng, {"meets_minimum": 6, "below_minimum": 2, "not_provided": 2})
        go_live = pick(rng, [3, 5, 10, 14, 30, 45])
        if reg == "clear":
            if sub_reg == "a":
                spend, single, qpct, cyber, cyber_status, esg, insurance, sub = \
                    round(between(rng, 8000, 40000), -3), False, 100, None, "not_applicable", [], "meets_minimum", "none"
                alt = pick(rng, [1, 2, 3])
            else:
                single, alt, cyber, cyber_status = True, pick(rng, [16, 24, 30]), between(rng, 25, 45, integer=True), "scored"
                esg, spend = rng.sample(ESG, 2), round(between(rng, 600000, 2500000), -3)
        elif reg == "borderline":
            spend = round(near(rng, 250000, 90000, lo=20000), -3)
            qpct = pick(rng, [70, 75, 80, 85])
            if data != "none":
                cyber = near(rng, 60, 8, lo=0, hi=100, integer=True)
        else:  # conflicting: the tool's view and the facts diverge
            if sub_reg == "a":     # small spend hides a big exposure
                spend = round(between(rng, 12000, 60000), -3)
                cyber, cyber_status = None, pick(rng, ["not_sent", "sent_no_reply"])
                qpct = pick(rng, [55, 60, 70])
            else:                  # big spend, commodity, many alternatives, nothing sensitive
                spend = round(between(rng, 700000, 3000000), -3)
                single, alt, cyber, cyber_status = False, pick(rng, [1, 2]), None, "not_applicable"
        if data == "none" and cyber_status in ("scored", "not_sent", "sent_no_reply"):
            cyber, cyber_status = None, "not_applicable"
        if data != "none" and cyber_status == "not_applicable":
            cyber_status = "scored" if cyber is not None else "not_sent"
        # the tool only looks at spend and at questionnaire answers that were actually filled in
        auto = "High" if spend >= 500000 else "Medium" if spend >= 150000 else "Low"
        if data in ("customer_personal", "system_access") and qpct == 100 and auto == "Low":
            auto = "Medium"
        return {"regime": reg, "variant": sub_reg, "supplier": sup, "supplier_id": sid, "country": country,
                "category": cat, "annual_spend": spend, "currency": pick(rng, ["USD", "EUR", "GBP"]), "single_source": single,
                "alt_weeks": alt, "data_access": data, "onsite": onsite, "subcontracting": sub,
                "questionnaire_pct": qpct, "cyber_score": cyber, "cyber_status": cyber_status, "cyber_pass_mark": 60,
                "esg_flags": esg, "insurance": insurance, "auto_tier": auto, "go_live_days": go_live,
                "requester": None, "analyst": None,
                "requester_line": pick(rng, ["They're a small vendor, can we keep this light?",
                                             "We need them for the launch, please don't let this drag.",
                                             "Same setup as the supplier we had before, just cheaper.",
                                             "Our director already agreed terms with them.", None]),
                "intake_id": ident(rng, "TPR", 5, "-"), "date": rand_date(rng)} | \
            dict(zip(("requester", "analyst"), _people(rng, 2)))

    def _cyber_txt(self, f):
        if f["cyber_status"] == "scored":
            return f"{f['cyber_score']}/100 (pass mark {f['cyber_pass_mark']})"
        return {"not_sent": "questionnaire not sent", "sent_no_reply": "sent, no reply yet",
                "not_applicable": "not required"}[f["cyber_status"]]

    def render(self, f, rng, surface):
        spend = _money(f["annual_spend"], rng, f["currency"])
        if surface == "json":
            return dumps(prune({
                "supplier": {"name": f["supplier"], "id": f["supplier_id"], "country": f["country"],
                             "category": f["category"], "annual_spend_estimate": spend,
                             "sole_source": f["single_source"], "weeks_to_replace": f["alt_weeks"]},
                "exposure": {"data_access": f["data_access"], "onsite_work": f["onsite"],
                             "subcontracting": f["subcontracting"]},
                "questionnaires": {"core_completion_pct": f["questionnaire_pct"], "cyber": self._cyber_txt(f),
                                   "esg_flags": f["esg_flags"] or None, "insurance_certificate": f["insurance"]},
                "tool": {"auto_tier": f["auto_tier"], "inputs": "spend band + completed questionnaire answers"},
                "request": {"id": f["intake_id"], "requested_by": f["requester"],
                            "wanted_live_in_days": f["go_live_days"],
                            "requester_comment": noisy(f["requester_line"], rng, 0.2) if f["requester_line"] else None},
            }))
        if surface == "intake_form":
            rows = [("Intake", f["intake_id"]), ("Date", fmt_date(f["date"], rng)),
                    ("Supplier", f"{f['supplier']} ({f['country']})"), ("Category", f["category"]),
                    ("Annual spend (est.)", spend), ("Sole source", "Yes" if f["single_source"] else "No"),
                    ("Time to replace", f"~{_plural(f['alt_weeks'], 'week')}"), ("Data access", DATA_TXT[f["data_access"]]),
                    ("On-site work", ONSITE_TXT[f["onsite"]]),
                    ("Subcontracting", f["subcontracting"].replace("_", " ")),
                    ("Core questionnaire", f"{f['questionnaire_pct']}% complete"),
                    ("Cyber questionnaire", self._cyber_txt(f)),
                    ("ESG flags", "; ".join(f["esg_flags"]) or "none"),
                    ("Insurance certificate", f["insurance"].replace("_", " ")),
                    ("Auto tier", f["auto_tier"]), ("Wanted live in", f"{f['go_live_days']} days"),
                    ("Requested by", f["requester"])]
            out = "THIRD-PARTY RISK INTAKE\n" + kv_lines(rows, rng, style=pick(rng, ["colon", "aligned", "dash"]))
            if f["requester_line"]:
                out += f"\n\nRequester comment: {noisy(f['requester_line'], rng, 0.2)}"
            return out
        # chat_thread
        a, r = f["analyst"].split()[0].lower(), f["requester"].split()[0].lower()
        msgs = [(r, f"hi, need {f['supplier']} set up for {f['category']}, about {spend} a year. "
                    f"can they be live in {f['go_live_days']} days?"),
                (a, f"tool has them at {f['auto_tier']}. core questionnaire is {f['questionnaire_pct']}% done"),
                (r, f"they'd have {DATA_TXT[f['data_access']]}, and {ONSITE_TXT[f['onsite']]}")]
        if f["single_source"]:
            msgs.append((r, f"no one else does this for us, replacing them would take ~{_plural(f['alt_weeks'], 'week')}"))
        msgs.append((a, f"cyber questionnaire: {self._cyber_txt(f)}"))
        if f["esg_flags"]:
            msgs.append((a, "esg flags: " + "; ".join(f["esg_flags"])))
        if f["subcontracting"] != "none":
            msgs.append((a, {"disclosed": "they subcontract part of the work, subcontractor named",
                             "suspected_undisclosed": "their answers suggest some work goes to a subcontractor they "
                                                      "haven't named"}[f["subcontracting"]]))
        if f["insurance"] != "meets_minimum":
            msgs.append((a, "insurance cert " + f["insurance"].replace("_", " ")))
        if f["requester_line"]:
            msgs.append((r, f["requester_line"]))
        t = rng.randint(8, 15) * 60 + rng.randint(0, 59)
        out = []
        for who, text in msgs:
            t += rng.randint(1, 20)
            out.append(f"[{t // 60:02d}:{t % 60:02d}] {who}: {noisy(text, rng, 0.1)}")
        return f"#supplier-onboarding  {f['intake_id']}\n" + "\n".join(out)

    def questions(self):
        return {
            "oversight_level": score("What level of ongoing oversight does this supplier warrant?", [
                "Light: a self-declaration renewed every few years.",
                "Standard: a periodic questionnaire refresh and a spend review.",
                "Enhanced: a yearly evidence-based review by a named relationship owner.",
                "Intensive: continuous monitoring, on-site assessments and senior ownership.",
            ]),
            "diligence_focus": choice("Which area should the next round of diligence concentrate on?", {
                "continuity_of_supply": "How the business would cope if the supplier stopped delivering.",
                "financial_resilience": "Whether the supplier is financially strong enough for the relationship.",
                "information_security": "How the supplier protects the data or systems it would be able to reach.",
                "labour_and_environment": "Working conditions, environmental permits and human-rights commitments.",
                "site_safety_and_insurance": "The safety of the supplier's people on our sites and the insurance behind it.",
            }),
            "disruption_impact": score("If the supplier stopped delivering tomorrow, how badly would operations suffer?", [
                "Negligible: a like-for-like replacement is available within days.",
                "Manageable: a few weeks of workarounds.",
                "Serious: months of disruption to an important activity.",
                "Critical: a core operation stops until a replacement is qualified.",
            ]),
            "go_live_handling": choice("How should the business's request to start using the supplier be handled?", {
                "allow_go_live": "Let the supplier start now; the diligence done so far is proportionate.",
                "escalate_to_risk_committee": "Refer the case to the third-party risk committee for a decision.",
                "go_live_with_interim_limits": "Let it start with limits, such as no data access or no on-site work, "
                                               "until diligence finishes.",
                "hold_until_diligence_complete": "Do not let it start until the missing questionnaires and evidence "
                                                 "are in.",
            }),
            "auto_tier_understates": noul(
                "The screening tool's automatic tier understates the supplier's real risk.",
                yes="Facts in the file, such as data access, hazardous on-site work or sole sourcing, put the supplier "
                    "above the tool's tier.",
                no="The tool's tier is about right, or it overstates the risk."),
        }


# =========================================================================== schema 5
# per topic: claims (reporting only), probe (an official inquiry is open), outcome (penalty or ruling),
# cleared (the matter was dropped). {e} = entity named, {ep} = its possessive, {c} = country.
MEDIA_TOPICS = {
    "wage_violations": {
        "claim": ["workers at a {e} site say overtime went unpaid for months",
                  "union alleges {e} docked pay for toilet breaks", "{e} staff protest outside plant gates over back pay"],
        "probe": ["labour inspectorate opens inquiry into shift pay at {e}"],
        "outcome": ["labour inspectors fine {e} over unpaid overtime at its {c} plant"],
        "cleared": ["labour inspectorate closes {e} pay inquiry without penalty"]},
    "workplace_injury": {
        "claim": ["safety campaigners question conditions at {e} after second accident",
                  "family of injured {e} worker calls for inquiry", "former employees describe missing guards at {e}"],
        "probe": ["safety regulator investigating serious injury on a {e} press line"],
        "outcome": ["{e} fined after a worker was seriously injured on a press line"],
        "cleared": ["regulator finds no breach by {e} over press-line accident"]},
    "environmental_discharge": {
        "claim": ["residents link fish deaths to effluent from a {e} facility",
                  "anglers blame {e} for discoloured stream", "campaign group posts drone footage of {e} outfall"],
        "probe": ["environment agency samples outfall near {e} works"],
        "outcome": ["{e} penalised in river discharge case in {c}"],
        "cleared": ["environment agency clears {e} after outfall sampling"]},
    "bribery_probe": {
        "claim": ["local officials questioned over permits granted to {e}",
                  "opposition councillor alleges favours in {e} land deal",
                  "anonymous letter accuses {e} of paying for permits"],
        "probe": ["prosecutors examine contract awards involving a {e} director"],
        "outcome": ["{e} director convicted in permit bribery case"],
        "cleared": ["prosecutors drop inquiry into {e} contract awards"]},
    "counterfeit_parts": {
        "claim": ["buyers report relabelled components traced to {e}", "forum thread alleges {e} resells rejected stock",
                  "distributor drops {e} after authenticity dispute"],
        "probe": ["customs seize suspect parts shipped by {e} and open an inquiry"],
        "outcome": ["{e} fined for supplying relabelled components"],
        "cleared": ["customs release {e} shipment after authenticity checks"]},
    "insolvency_rumour": {
        "claim": ["lender said to be reviewing {ep} credit line", "suppliers complain of slow payment from {e}",
                  "{e} said to be seeking a buyer for its second plant"],
        "probe": ["{e} confirms talks with lenders over its borrowing"],
        "outcome": ["{e} files a restructuring plan with creditors"],
        "cleared": ["{e} secures new financing and files results on time"]},
}
CODE_GRADES = ("Supplier code of conduct: bribery is zero-tolerance; serious worker injury, polluting discharges and "
               "counterfeit parts are serious breaches; unpaid or docked wages are a breach the supplier must put "
               "right; financial difficulty is not a conduct matter.")
STATUS_KIND = {"allegation": "claim", "under_investigation": "probe", "fined_or_settled": "outcome",
               "convicted": "outcome", "dismissed": "cleared"}
SOURCES = {"court_record": "court record", "regulator_notice": "regulator's published notice",
           "national_newspaper": "national newspaper", "trade_press": "trade journal",
           "ngo_report": "NGO report", "blog": "personal blog", "social_media": "social media post"}


class AdverseMediaReview(Schema):
    name = "adverse_media_review"
    description = "Adverse-media hits found during supplier due diligence: dismiss, monitor, ask, pause or decline."
    surfaces = {"json": 0.5, "media_digest": 0.25, "analyst_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, kind, _crit = pick(rng, CATEGORIES)
        sup, country, sid = _supplier(rng, kind)
        topic = pick(rng, list(MEDIA_TOPICS))
        n_hits = pick(rng, [1, 1, 2, 3, 4])
        best = pick_w(rng, {"court_record": 1, "regulator_notice": 2, "national_newspaper": 3, "trade_press": 3,
                            "ngo_report": 2, "blog": 2, "social_media": 2})
        age = pick(rng, [1, 3, 8, 14, 26, 40, 70])
        status = pick_w(rng, {"allegation": 4, "under_investigation": 3, "fined_or_settled": 2, "convicted": 1,
                              "dismissed": 1})
        entity = pick_w(rng, {"exact_entity": 4, "parent_group": 2, "sister_company": 2, "namesake_possible": 2})
        site_ours = chance(rng, 0.5)
        response = pick_w(rng, {"not_asked": 3, "denied_no_evidence": 2, "explained_with_evidence": 2,
                                "remediated_with_audit": 2})
        if reg == "clear":
            if chance(rng, 0.5):   # stale, weak, not them
                best, n_hits, age, entity, status = pick(rng, ["blog", "social_media"]), 1, pick(rng, [40, 70]), \
                    "namesake_possible", "allegation"
            else:                  # recent, official, them
                best, n_hits, age, entity, status, site_ours = pick(rng, ["court_record", "regulator_notice"]), \
                    pick(rng, [3, 4]), pick(rng, [1, 3]), "exact_entity", pick(rng, ["convicted", "fined_or_settled"]), True
                response = pick(rng, ["not_asked", "denied_no_evidence"])
        elif reg == "borderline":
            best = pick(rng, ["national_newspaper", "trade_press", "ngo_report"])
            status = pick(rng, ["allegation", "under_investigation"])
            entity = pick(rng, ["exact_entity", "parent_group", "sister_company"])
            age = pick(rng, [8, 14, 26])
        else:  # conflicting
            if chance(rng, 0.5):   # strong source, but remediated and old
                best, status, response, age = "regulator_notice", "fined_or_settled", "remediated_with_audit", \
                    pick(rng, [26, 40])
            else:                  # weak source, but many hits, recent, and a flat denial
                best, n_hits, age, response = pick(rng, ["blog", "social_media", "ngo_report"]), pick(rng, [3, 4]), \
                    pick(rng, [1, 3]), "denied_no_evidence"
        if status in ("convicted", "fined_or_settled") and best in ("blog", "social_media"):
            status = "allegation"
        if best in ("court_record", "regulator_notice") and status == "allegation":
            status = "under_investigation"
        if topic == "insolvency_rumour" and best == "court_record" and status == "allegation":
            best = "national_newspaper"
        return {"regime": reg, "supplier": sup, "supplier_id": sid, "country": country, "topic": topic,
                "n_hits": n_hits, "best_source": best, "age_months": age, "status": status, "entity_match": entity,
                "affects_supplying_site": site_ours, "supplier_response": response,
                "category": cat, "kind": kind,
                "group_name": org(rng, "company").split()[0] + " " + pick(rng, ["Holdings", "Group", "Industrial Group"]),
                "date": rand_date(rng), "case_id": ident(rng, "AM", 5, "-")} | \
            dict(zip(("analyst", "owner"), _people(rng, 2)))

    def _entity_name(self, f):
        base_name = f["supplier"]
        return {"exact_entity": base_name, "parent_group": f["group_name"],
                "sister_company": base_name.split()[0] + " " + pick_static(f, ["Components", "Metals", "Plastics"]),
                "namesake_possible": base_name.split()[0] + " " + pick_static(f, ["Trading", "Services", "Enterprises"])
                }[f["entity_match"]]

    def _hits(self, f, rng):
        e, c = self._entity_name(f), f["country"]
        t = MEDIA_TOPICS[f["topic"]]
        heads = [pick(rng, t[STATUS_KIND[f["status"]]])] + rng.sample(t["claim"], min(3, f["n_hits"] - 1))
        heads = list(dict.fromkeys(heads))
        out = []
        for i, h in enumerate(heads):
            src = f["best_source"] if i == 0 else pick(rng, ["trade_press", "blog", "social_media", "ngo_report",
                                                            "national_newspaper"])
            text = re.sub(r"\ba (?=[AEIOU])", "an ", h.format(e=e, ep=_poss(e), c=c))
            out.append({"source": SOURCES[src], "months_ago": f["age_months"] + (0 if i == 0 else rng.randint(1, 10)),
                        "headline": text[0].upper() + text[1:]})
        return out

    def _status_txt(self, f):
        if f["topic"] == "insolvency_rumour":
            return {"allegation": "market talk only; nothing filed", "under_investigation": "the company has confirmed "
                    "lender talks", "fined_or_settled": "a restructuring plan has been filed",
                    "convicted": "a restructuring plan has been filed", "dismissed": "new financing announced"}[f["status"]]
        return {"allegation": "allegations only; no official action reported",
                "under_investigation": "an official investigation is reported to be open",
                "fined_or_settled": "a fine or settlement is on record", "convicted": "a conviction is on record",
                "dismissed": "the official case was reported closed without action"}[f["status"]]

    def _response_txt(self, f):
        return {"not_asked": None,
                "denied_no_evidence": "The supplier denies the reports and has offered nothing further.",
                "explained_with_evidence": "The supplier sent a written explanation with supporting documents.",
                "remediated_with_audit": "The supplier sent an independent audit report confirming corrective "
                                         "actions at the site."}[f["supplier_response"]]

    def render(self, f, rng, surface):
        hits = self._hits(f, rng)
        site = (SITE_WORD[f["kind"]] if f["affects_supplying_site"] else
                "a different site from the one that would supply us")
        rel = {"exact_entity": "the applicant itself", "parent_group": f"the applicant's parent, {f['group_name']}",
               "sister_company": "a sister company in the same group",
               "namesake_possible": "a company with a similar name; no ownership link found so far"}[f["entity_match"]]
        if surface == "json":
            return dumps(prune({
                "applicant": {"name": f["supplier"], "id": f["supplier_id"], "country": f["country"],
                              "category": f["category"]},
                "screening": {"case": f["case_id"], "hits": hits, "entity_named": rel, "location": site,
                              "legal_status": self._status_txt(f)},
                "supplier_response": self._response_txt(f),
                "policy": CODE_GRADES,
            }))
        if surface == "media_digest":
            lines = [f"ADVERSE MEDIA DIGEST {f['case_id']} -- applicant: {f['supplier']} ({f['country']})", ""]
            for i, h in enumerate(hits, 1):
                lines.append(f"{i}. [{h['source']}, {h['months_ago']} mo ago] {h['headline']}")
            lines += ["", kv_lines([("Entity named", rel), ("Location", site), ("Status", self._status_txt(f)),
                                    ("Supplier response", self._response_txt(f) or "not yet requested")], rng,
                                   style=pick(rng, ["colon", "aligned", "dash"])), "", CODE_GRADES]
            return "\n".join(lines)
        s = [f"Our screening of {f['supplier']} for the {f['category']} category turned up "
             + ("one adverse item." if len(hits) == 1 else f"{len(hits)} adverse items."),
             f"The strongest is {_article(hits[0]['source'])} from {_plural(hits[0]['months_ago'], 'month')} ago: "
             f"\"{hits[0]['headline']}\".",
             f"The entity named is {rel}, and the events concern {site}.",
             f"Status: {self._status_txt(f)}.", self._response_txt(f) or "We have not put this to the supplier yet."]
        return "\n".join([f"Subject: Adverse media -- {f['supplier']} ({f['case_id']})", f"From: {f['analyst']}",
                          f"To: {f['owner']}", "", noisy(join_sentences(s), rng, 0.12), "", f"({CODE_GRADES})", "",
                          f"-- {f['analyst'].split()[0]}"])

    def questions(self):
        return {
            "media_hit_handling": choice("What should happen to the supplier application in light of the adverse media?", {
                "decline_supplier": "Decline the supplier on the strength of what the reports show.",
                "dismiss_as_not_relevant": "Set the hits aside; they do not concern this supplier or are not material.",
                "pause_pending_investigation": "Pause onboarding until our own review or an independent audit reports back.",
                "record_and_monitor": "Note the hits on file, carry on with onboarding and watch for new reporting.",
                "request_supplier_response": "Put the reports to the supplier and ask for its explanation and evidence "
                                             "before deciding.",
            }),
            "reporting_credibility": score("How credible is the adverse reporting, taken as a whole?", [
                "Low: unsourced claims, a single post or a likely name collision.",
                "Limited: one secondary report with little detail.",
                "Substantial: several independent reports or an official source.",
                "High: an official finding such as a court judgment or a regulator's penalty.",
            ]),
            "code_breach_gravity": score("If the reports are accurate, how grave a breach of our supplier code of "
                                         "conduct would they be?", [
                "Negligible: the code does not treat it as misconduct (financial difficulty).",
                "Moderate: a breach the supplier must put right, such as unpaid or docked wages.",
                "Serious: a serious breach (worker injury, polluting discharge or counterfeit parts).",
                "Severe: zero-tolerance conduct (bribery) that no remediation plan could make acceptable.",
            ]),
            "reports_concern_applicant": noul(
                "The reports concern the same business that has applied to become a supplier.",
                yes="The entity named, or the site that would supply us, is the applicant's own.",
                no="The reports concern a different company, a distant affiliate or a namesake."),
            "issue_still_open": noul(
                "The problem described in the reports is still unresolved today.",
                yes="Nothing has closed it: allegations or an open investigation without a supporting explanation, or "
                    "a fine or conviction without an independent audit of corrective action.",
                no="The official case was closed without action, an independent audit confirms corrective action, or "
                   "a bare allegation has been answered with a documented explanation."),
        }


def pick_static(f, options):
    """A deterministic pick keyed on the case (render-time names must not depend on extra rng draws)."""
    return options[sum(map(ord, f["supplier_id"])) % len(options)]


# =========================================================================== schema 6
RELATIONSHIPS = {
    "spouse_or_partner": "is the spouse of one of the supplier's directors",
    "sibling": "has a brother or sister who is a director of the supplier",
    "former_employee": "worked for the supplier until {m} months ago",
    "shareholding": "holds {p}% of the supplier's shares",
    "close_friend": "is a close personal friend of the supplier's owner",
    "cousin_sales_rep": "has a cousin who is the supplier's sales representative for our account",
}
ROLES = {"decision_maker": "signed off the supplier shortlist", "technical_evaluator": "scored the technical bids",
         "requester": "raised the original request for a new supplier", "budget_holder": "owns the budget for the work",
         "no_role": "had no part in selecting suppliers for this work"}


class ConflictOfInterestCheck(Schema):
    name = "conflict_of_interest_check"
    description = "An employee's link to a prospective supplier, declared or discovered during onboarding."
    surfaces = {"json": 0.5, "declaration_form": 0.25, "ethics_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        cat, kind, _crit = pick(rng, CATEGORIES)
        sup, country, sid = _supplier(rng, kind)
        rel = pick(rng, list(RELATIONSHIPS))
        role = pick_w(rng, {"decision_maker": 2, "technical_evaluator": 3, "requester": 2, "budget_holder": 1,
                            "no_role": 2})
        declared = pick_w(rng, {"before_selection": 4, "after_shortlist": 3, "not_declared": 2})
        recused = chance(rng, 0.4)
        route = pick_w(rng, {"open_tender": 4, "panel_shortlist": 3, "introduced_by_employee": 2})
        bidders = pick(rng, [2, 3, 4, 5]) if route != "introduced_by_employee" else 1
        premium = between(rng, -8, 14, integer=True) if bidders > 1 else None
        consensus = pick_w(rng, {"all_ranked_first": 3, "split": 3, "only_employee_ranked_first": 1}) if bidders > 1 \
            else None
        months = pick(rng, [2, 5, 9, 18, 40])
        pct = pick(rng, [0.3, 1, 4, 12, 30])
        trained = chance(rng, 0.7)
        if reg == "clear":
            if chance(rng, 0.5):
                declared, role, recused, route, rel = "before_selection", pick(rng, ["no_role", "requester"]), True, \
                    "open_tender", pick(rng, ["former_employee", "close_friend"])
                bidders, premium, consensus, months = 4, between(rng, -8, -2, integer=True), "all_ranked_first", 40
            else:
                declared, role, recused, route, rel = "not_declared", "decision_maker", False, \
                    "introduced_by_employee", pick(rng, ["spouse_or_partner", "shareholding"])
                bidders, premium, consensus, pct = 1, None, None, pick(rng, [12, 30])
        elif reg == "borderline":
            declared = pick(rng, ["after_shortlist", "before_selection"])
            role = pick(rng, ["technical_evaluator", "requester", "budget_holder"])
            rel = pick(rng, ["former_employee", "shareholding", "cousin_sales_rep", "close_friend"])
            months, pct = pick(rng, [9, 12, 14]), pick(rng, [0.3, 1, 4])
            if bidders > 1:
                premium = near(rng, 3, 4, integer=True)
        else:  # conflicting: open process and good price, but concealed; or declared early but the employee drove it
            if chance(rng, 0.5):
                declared, route, bidders, consensus = "not_declared", "open_tender", 5, "all_ranked_first"
                premium = between(rng, -9, -3, integer=True)
            else:
                declared, recused, role = "before_selection", False, pick(rng, ["decision_maker", "technical_evaluator"])
                if bidders > 1:
                    consensus, premium = "only_employee_ranked_first", between(rng, 6, 18, integer=True)
        if consensus == "only_employee_ranked_first" and role not in ("technical_evaluator", "decision_maker"):
            role = pick(rng, ["technical_evaluator", "decision_maker"])
        emp, reviewer = _people(rng, 2)
        return {"regime": reg, "supplier": sup, "supplier_id": sid, "country": country, "relationship": rel,
                "months_since_left": months if rel == "former_employee" else None,
                "share_pct": pct if rel == "shareholding" else None, "employee": emp,
                "employee_title": pick(rng, ["Maintenance Manager", "Senior Buyer", "Site Engineer", "Plant Controller",
                                             "Logistics Lead", "Facilities Coordinator", "Head of Operations"]),
                "selection_role": role, "declared": declared, "self_recused": recused, "route": route,
                "bidders": bidders, "premium_vs_median_pct": premium, "evaluator_consensus": consensus,
                "coi_training_done": trained, "annual_value": round(lognormal_amount(rng, 150000, 0.9), -3),
                "currency": pick(rng, ["USD", "EUR", "GBP"]), "category": cat,
                "case_id": ident(rng, "COI", 4, "-"), "date": rand_date(rng), "reviewer": reviewer}

    def _rel_txt(self, f):
        return RELATIONSHIPS[f["relationship"]].format(m=f["months_since_left"], p=f["share_pct"])

    def _declared_txt(self, f):
        return {"before_selection": "declared by the employee before any bids were evaluated",
                "after_shortlist": "declared by the employee after the supplier had been shortlisted",
                "not_declared": "not declared; found by our related-party screening"}[f["declared"]]

    def _route_txt(self, f):
        return {"open_tender": f"open tender, {f['bidders']} bids received",
                "panel_shortlist": f"shortlist from the approved panel, {f['bidders']} bids",
                "introduced_by_employee": "supplier introduced by the employee; no competing bids"}[f["route"]]

    def _consensus_txt(self, f):
        return {"all_ranked_first": "every evaluator ranked this supplier first",
                "split": "evaluators were split between this supplier and one other",
                "only_employee_ranked_first": "only the employee concerned ranked this supplier first",
                None: None}[f["evaluator_consensus"]]

    def render(self, f, rng, surface):
        val = _money(f["annual_value"], rng, f["currency"])
        prem = (f"{f['premium_vs_median_pct']:+d}% vs " + ("the other bid" if f["bidders"] == 2 else
                                                              "the median of the other bids")
                if f["premium_vs_median_pct"] is not None else None)
        if surface == "json":
            return dumps(prune({
                "case": {"id": f["case_id"], "opened": fmt_date(f["date"], style="iso"),
                         "how_it_arose": self._declared_txt(f)},
                "employee": {"name": f["employee"], "title": f["employee_title"], "link_to_supplier": self._rel_txt(f),
                             "part_in_selection": ROLES[f["selection_role"]],
                             "stepped_back_voluntarily": f["self_recused"],
                             "conflicts_training_completed": f["coi_training_done"]},
                "supplier": {"name": f["supplier"], "id": f["supplier_id"], "category": f["category"],
                             "expected_annual_value": val},
                "selection": {"route": self._route_txt(f), "price_position": prem,
                              "evaluation": self._consensus_txt(f)},
            }))
        if surface == "declaration_form":
            rows = [("Case", f["case_id"]), ("Employee", f"{f['employee']}, {f['employee_title']}"),
                    ("Supplier", f"{f['supplier']} ({f['supplier_id']})"), ("Nature of interest", self._rel_txt(f)),
                    ("Role in this sourcing", ROLES[f["selection_role"]]),
                    ("Disclosure", self._declared_txt(f)),
                    ("Stepped back voluntarily", "yes" if f["self_recused"] else "no"),
                    ("Selection route", self._route_txt(f)), ("Price position", prem),
                    ("Evaluation", self._consensus_txt(f)), ("Expected annual value", val),
                    ("Conflicts training", "completed" if f["coi_training_done"] else "overdue")]
            return "CONFLICT OF INTEREST -- SUPPLIER ONBOARDING\n" + kv_lines(
                rows, rng, style=pick(rng, ["colon", "aligned", "dash"]), skip_none=True)
        s = [f"While onboarding {f['supplier']} for {f['category']} (about {val} a year) we found that "
             f"{f['employee']}, our {f['employee_title']}, {self._rel_txt(f)}.",
             f"The link was {self._declared_txt(f)}.",
             f"{f['employee'].split()[0]} {ROLES[f['selection_role']]}"
             + (" and has stepped back from the decision." if f["self_recused"] else "; there has been no recusal so far."),
             f"Selection route: {self._route_txt(f)}."]
        if prem:
            s.append(f"Their price is {prem}.")
        if self._consensus_txt(f):
            s.append(self._consensus_txt(f).capitalize() + ".")
        if not f["coi_training_done"]:
            s.append(f"{f['employee'].split()[0]}'s annual conflicts training is overdue.")
        return "\n".join([f"Subject: Possible conflict -- {f['supplier']} ({f['case_id']})", f"From: {f['reviewer']}",
                          "To: ethics-office@corp.example", "", noisy(join_sentences(s), rng, 0.12), "",
                          f"{f['reviewer'].split()[0]}"])

    def questions(self):
        return {
            "conflict_handling": choice("How should the onboarding team deal with the employee's link to the supplier?", {
                "halt_onboarding": "Stop onboarding while it is investigated: a material interest was not declared, "
                                   "and the employee introduced the supplier or signed off the shortlist.",
                "proceed_no_action": "Carry on: the interest is not material.",
                "proceed_with_recusal": "Carry on with the employee formally removed: a material interest was "
                                        "declared, and the employee either did not evaluate or decide, or stepped back.",
                "re_evaluate_independently": "Repeat the evaluation without the employee: a material interest was "
                                             "declared, but the employee scored bids or signed off without stepping "
                                             "back.",
                "refer_to_ethics_office": "Refer to the ethics office: a material interest was not declared, but the "
                                          "employee neither introduced the supplier nor signed off the shortlist.",
            }),
            "influence_extent": score("How much could the employee have shaped the choice of this supplier?", [
                "None: no part in the selection at all.",
                "Slight: a minor voice among several.",
                "Considerable: a scorer or requester whose view carried weight.",
                "Decisive: the employee effectively made or steered the choice.",
            ]),
            "handling_seriousness": score("How seriously should the employee's own handling of the conflict be viewed?", [
                "Exemplary: declared early and stepped back unprompted.",
                "Acceptable: declared, if not at the ideal moment.",
                "Careless: late or incomplete disclosure while still involved.",
                "Serious: concealed while taking part in the selection.",
            ]),
            "interest_is_material": noul(
                "The employee's interest is material enough to require a formal recusal.",
                yes="A spouse, partner, brother or sister is a director; a close personal friendship with the owner; "
                    "a shareholding of 5% or more; or employment there that ended less than 12 months ago.",
                no="A cousin in the supplier's sales team, a shareholding under 5%, or employment that ended 12 or "
                   "more months ago."),
            "selection_looks_favoured": noul(
                "The evaluation appears to have favoured this supplier beyond what its offer merits."),
        }


WORKFLOW = Workflow(
    name="supplier_onboarding",
    title="Supplier onboarding due diligence",
    description="Whether a prospective supplier should be approved into the supplier base, and under what oversight.",
    domain="supply chain / third-party risk management",
    schemas=[FinancialStabilityReview(), CertificationVerification(), SanctionsScreeningMatch(), RiskTierAssignment(),
             AdverseMediaReview(), ConflictOfInterestCheck()],
    notes="Counterparty due diligence before a supplier is set up. No invoices, purchase orders, payments, "
          "goods receipts or purchase approvals: nothing is billed, matched or paid.",
)
