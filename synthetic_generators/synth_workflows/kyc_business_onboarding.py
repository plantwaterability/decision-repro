"""KYC business onboarding: can a bank take on (or keep) a business customer, and on what terms?

Each case is one know-your-customer file for a business client: an ownership chain to be traced to beneficial
owners, a sanctions-screening name hit to be resolved, a politically exposed person among the owners, an expected
activity profile to be checked against the business, or a periodic refresh of an existing client. The state
carries the file as an analyst would see it: registry extracts, screening console output, application answers,
relationship-manager notes and the bank's written thresholds.

Fairness and fiction:
- Jurisdictions and cities are FICTIONAL. Country risk applies only to where companies are registered and where
  payments flow, through the bank's internal list; it is never a person's nationality.
- No person's race, religion, nationality, sex, age or disability is rendered or used. A date of birth appears
  only in the sanctions schema, and only as an identifier to compare with a list entry.
- Companies and people are invented (base.org, base.person).

Structure follows it_change_approval.py: each schema draws a `regime` first and lets it steer the factors.
"""
from __future__ import annotations

from datetime import date, timedelta

from .base import (
    FIRST,
    ORG_A,
    REAL_NAME_CLASHES,
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

# fictional jurisdictions -> the bank's internal country-risk tier (company registration and payment corridors only)
JURIS = {"domestic": "standard", "Calverra": "standard", "Northmere": "standard", "Isle of Brenn": "elevated",
         "Ostravia": "elevated", "Port Selan Free Zone": "high", "Veradia": "high"}
CITIES = {"domestic": ["Harrowgate", "Amberley", "Norwick", "Easton Vale"], "Calverra": ["Velmont"],
          "Northmere": ["Kestrel Bay"], "Isle of Brenn": ["St. Aldric"], "Ostravia": ["Sulden"],
          "Port Selan Free Zone": ["Port Selan"], "Veradia": ["Maravel"]}
FOREIGN = [j for j in JURIS if j != "domestic"]
HOLD_SUFFIX = {"domestic": "Holdings Ltd", "Calverra": "Beteiligungen AG", "Northmere": "Holdings Ltd",
               "Isle of Brenn": "Investments Ltd", "Ostravia": "Capital SA", "Port Selan Free Zone": "FZE",
               "Veradia": "Participaciones SA"}
# industry -> inherent risk tier used by the bank's risk model
INDUSTRIES = {
    "software consultancy": "standard", "commercial cleaning contractor": "standard",
    "medical supplies wholesaler": "standard", "construction subcontractor": "standard",
    "used-vehicle dealer": "elevated", "restaurant group": "elevated", "freight forwarder": "elevated",
    "electronics import-export": "elevated", "real-estate holding company": "elevated",
    "precious-metals trader": "high", "payment facilitator": "high", "crypto-asset brokerage": "high",
}
LICENSED = ("payment facilitator", "crypto-asset brokerage")
LEGAL_FORMS = ["private limited company", "limited liability partnership", "company limited by shares",
               "general partnership"]


def _company(rng, industry: str) -> str:
    tail = {"software consultancy": ["Consulting Ltd", "Digital Ltd"], "commercial cleaning contractor":
            ["Facility Services Ltd", "Cleaning Co. Ltd"], "medical supplies wholesaler": ["Medical Supply Ltd"],
            "construction subcontractor": ["Build Ltd", "Contracting Ltd"], "used-vehicle dealer": ["Motors Ltd",
            "Autos Ltd"], "restaurant group": ["Hospitality Ltd", "Dining Group Ltd"], "freight forwarder":
            ["Freight Ltd", "Logistics Ltd"], "electronics import-export": ["Trading Ltd", "Components Ltd"],
            "real-estate holding company": ["Property Ltd", "Estates Ltd"], "precious-metals trader":
            ["Bullion Ltd", "Metals Ltd"], "payment facilitator": ["Pay Ltd", "Payments Ltd"],
            "crypto-asset brokerage": ["Digital Assets Ltd", "Exchange Ltd"]}[industry]
    return _safe_name(rng, tail)


def _holding(rng, juris: str) -> str:
    return _safe_name(rng, [HOLD_SUFFIX[juris]])


def _safe_name(rng, suffixes) -> str:
    """ORG_A prefix + one of `suffixes`, skipping combinations that name a real business (base.REAL_NAME_CLASHES),
    the same way base.org does: deterministic, no extra rng draw."""
    a, i = rng.choice(ORG_A), rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        name = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if not any(name == c or name.startswith(c + " ") for c in REAL_NAME_CLASHES):
            return name
    return name


def _juris_label(j: str) -> str:
    return "domestic" if j == "domestic" else f"{j} (bank list: {JURIS[j]})"


# =========================================================================== schema 1
class BeneficialOwnershipTrace(Schema):
    name = "beneficial_ownership_trace"
    description = "A new business applicant's ownership chain, traced to the natural persons behind it."
    surfaces = {"json": 0.5, "ownership_chart": 0.25, "analyst_case_notes": 0.25}

    UNIDENT = {"discretionary_trust": "a discretionary trust (beneficiaries not named)",
               "nominee": "a nominee shareholder acting for an undisclosed principal",
               "fund": "an investment fund whose investors are not disclosed",
               "bearer_shares": "bearer shares (holders not recorded)"}

    def sample_factors(self, rng):
        reg = regime(rng)
        industry = pick(rng, list(INDUSTRIES))
        layers = pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1})
        inter = [pick_w(rng, {"domestic": 3, "Calverra": 1, "Northmere": 1, "Isle of Brenn": 2, "Ostravia": 1,
                              "Port Selan Free Zone": 1, "Veradia": 1}) for _ in range(layers)]
        direct = pick(rng, [0, 0, 20, 30, 40]) if layers else 0
        unident = pick(rng, [0, 0, 0, 10, 20, 35])
        unident_kind = pick(rng, list(self.UNIDENT))
        n_people = rng.randint(1, 3)
        registry_age = pick(rng, [12, 40, 70, 95, 140, 400])
        nominee_director = chance(rng, 0.15)
        mismatch = chance(rng, 0.15)
        poa_controller = chance(rng, 0.12)
        rationale = pick_w(rng, {"acquisition_legacy": 2, "investor_requirement": 2, "tax_planning": 2,
                                 "asset_protection": 1, None: 2})
        near_threshold = False
        if reg == "clear":
            if chance(rng, 0.5):
                layers, inter, direct, unident, registry_age = pick(rng, [0, 1]), [], 0, 0, pick(rng, [12, 40])
                inter = ["domestic"] * layers
                nominee_director, mismatch, poa_controller = False, False, False
                rationale = "acquisition_legacy" if layers else None
            else:
                layers, inter = 3, ["Isle of Brenn", "Ostravia", "Port Selan Free Zone"]
                unident, unident_kind, nominee_director = 35, pick(rng, ["bearer_shares", "nominee"]), True
                registry_age, rationale, direct = 400, None, 0
        elif reg == "borderline":
            near_threshold = True
            registry_age = pick(rng, [85, 95, 100])
            unident = pick(rng, [0, 10])
        else:
            if chance(rng, 0.5):
                # a neat, well-explained structure whose registry filing disagrees with the declaration
                rationale, mismatch, unident, nominee_director = "investor_requirement", True, 0, False
                registry_age = 20
            else:
                # simple direct ownership, but someone outside the share register gives the instructions
                layers, inter, direct, unident, poa_controller = 0, [], 0, 0, True
        if layers == 0 and not (reg == "conflicting" and rationale == "investor_requirement"):
            rationale = None
        elif layers == 0:
            layers, inter = 1, ["domestic"]
        hold_pct = 100 - direct
        # natural persons at the top of the chain
        remaining = 100 - (unident if unident else 0)
        cuts = sorted(rng.sample(range(5, 96, 5), n_people - 1)) if n_people > 1 else []
        shares, prev = [], 0
        for c in cuts + [100]:
            shares.append(c - prev)
            prev = c
        top = [round(remaining * s / 100, 1) for s in shares]
        if near_threshold:
            # one owner's EFFECTIVE stake sits just either side of the 25% threshold
            target = near(rng, 25, 2.5, lo=21, hi=29, digits=1)
            if layers:
                t = min(round(target * 100 / hold_pct, 1), remaining - 1)
            else:
                t = min(target, remaining - 1)
            rest = round(remaining - t, 1)
            top = [t] + ([rest] if n_people == 1 else [round(rest / 2, 1), round(rest - round(rest / 2, 1), 1)])
        people = []
        for i, p in enumerate(top):
            eff = round(p * hold_pct / 100, 1) if layers else p
            people.append({"name": person(rng), "top_pct": p, "effective_pct": eff,
                           "id_check": pick_w(rng, {"verified": 6, "pending": 2, "document_expired": 1})})
        if near_threshold and chance(rng, 0.7):
            people[0]["id_check"] = pick(rng, ["pending", "document_expired"])
        if reg == "clear" and layers <= 1 and unident == 0:
            for p in people:
                p["id_check"] = "verified"
        direct_owner = ({"name": person(rng), "pct": direct, "id_check": pick_w(rng, {"verified": 4, "pending": 1})}
                        if direct else None)
        juris = pick_w(rng, {"domestic": 5, "Calverra": 1, "Northmere": 1})
        form = pick(rng, LEGAL_FORMS[:3])
        applicant = _company(rng, industry)
        if form == "limited liability partnership":
            applicant = applicant.replace(" Ltd", " LLP")
        return {"regime": reg, "applicant": applicant, "industry": industry,
                "industry_risk": INDUSTRIES[industry], "legal_form": form,
                "registered_in": juris, "layers": layers, "intermediate_jurisdictions": inter,
                "holding_names": [_holding(rng, j) for j in inter], "holding_share_pct": hold_pct if layers else None,
                "direct_owner": direct_owner, "people": people,
                "unidentified_pct": unident, "unidentified_kind": unident_kind if unident else None,
                "registry_extract_age_days": registry_age, "nominee_director": nominee_director,
                "registry_mismatch": mismatch, "poa_controller": person(rng) if poa_controller else None,
                "structure_rationale": rationale, "analyst": person(rng), "case_id": ident(rng, "KYC", 6, "-"),
                "opened": rand_date(rng), "products": pick(rng, ["business current account",
                                                                 "current account + international payments",
                                                                 "current account + merchant acquiring"])}

    RATIONALE = {"acquisition_legacy": "The holding layer is left over from an acquisition in 2019.",
                 "investor_requirement": "The holding company was required by an outside investor.",
                 "tax_planning": "The structure was set up by the owners' tax advisers.",
                 "asset_protection": "The client says the structure protects family assets.",
                 None: None}

    def _mismatch_text(self, f):
        if not f["registry_mismatch"]:
            return None
        who = f["people"][0]["name"]
        return f"Registry filing lists a shareholder not on the declaration and omits {who}."

    def render(self, f, rng, surface):
        ubo_rows = [{"name": p["name"], "holding_in_top_entity_pct": p["top_pct"] if f["layers"] else None,
                     "direct_pct": None if f["layers"] else p["top_pct"], "id_verification": p["id_check"]}
                    for p in f["people"]]
        chain = [{"entity": n, "jurisdiction": _juris_label(j)} for n, j in zip(f["holding_names"],
                                                                                 f["intermediate_jurisdictions"])]
        extras = {"nominee_director": f["nominee_director"] or None,
                  "registry_discrepancy": self._mismatch_text(f),
                  "account_instructions_from": (f"{f['poa_controller']} (holds power of attorney; not a "
                                                "shareholder or director)") if f["poa_controller"] else None}
        if surface == "json":
            return dumps(prune({
                "applicant": {"name": f["applicant"], "legal_form": f["legal_form"],
                              "registered_in": _juris_label(f["registered_in"]), "industry": f["industry"],
                              "products_requested": f["products"], "case": f["case_id"]},
                "ownership": {"direct_shareholder": (f"{f['direct_owner']['name']} {f['direct_owner']['pct']}% "
                                                     f"(ID {f['direct_owner']['id_check']})")
                              if f["direct_owner"] else None,
                              "holding_chain_top_down": list(reversed(chain)) or None,
                              "chain_holds_pct_of_applicant": f["holding_share_pct"],
                              "natural_persons": ubo_rows,
                              "untraced_pct": f["unidentified_pct"] or None,
                              "untraced_held_by": self.UNIDENT.get(f["unidentified_kind"])},
                "documents": {"registry_extract_age_days": f["registry_extract_age_days"],
                              "client_explanation": self.RATIONALE[f["structure_rationale"]]},
                "flags": extras,
                "policy": {"ubo_threshold_pct": 25, "registry_extract_max_age_days": 90,
                           "note": "owners are traced through every layer; effective stake = product of holdings",
                           "stale_extract": "the analyst re-pulls an out-of-date extract; that alone does not hold "
                                            "the application"},
            }))
        if surface == "ownership_chart":
            lines = [f"{f['applicant']} ({f['legal_form']}, {_juris_label(f['registered_in'])}) -- {f['industry']}"]
            pad = " "
            if f["direct_owner"]:
                d = f["direct_owner"]
                lines.append(f"{pad}├─ {d['pct']}% {d['name']} (direct)  ID: {d['id_check']}")
            names = list(zip(f["holding_names"], f["intermediate_jurisdictions"]))
            for i, (n, j) in enumerate(names):
                pct = f["holding_share_pct"] if i == 0 else 100
                lines.append(f"{pad}└─ {pct}% {n} ({_juris_label(j)})")
                pad += "    "
            leaves = [f"{p['top_pct']}% {p['name']}  ID: {p['id_check'].replace('_', ' ')}" for p in f["people"]]
            if f["unidentified_pct"]:
                leaves.append(f"{f['unidentified_pct']}% {self.UNIDENT[f['unidentified_kind']]}")
            for i, leaf in enumerate(leaves):
                lines.append(f"{pad}{'└─' if i == len(leaves) - 1 else '├─'} {leaf}")
            notes = [x for x in [self.RATIONALE[f["structure_rationale"]], self._mismatch_text(f),
                                 "Sole director is a professional nominee." if f["nominee_director"] else None,
                                 extras["account_instructions_from"] and
                                 f"Account instructions to come from {extras['account_instructions_from']}."] if x]
            return ("OWNERSHIP STRUCTURE -- " + f["case_id"] + "\n" + "\n".join(lines) + "\n\n"
                    + kv_lines([("Registry extract dated", f"{f['registry_extract_age_days']} days ago"),
                                ("Products", f["products"]), ("Threshold", "25% effective, traced through layers"),
                                ("Extract over 90 days", "analyst re-pulls it; not by itself a reason to hold")],
                               rng)
                    + ("\n\nNotes:\n" + bullets(notes, rng) if notes else ""))
        # analyst_case_notes
        s = [f"{f['applicant']} is a {f['industry']} applying for a {f['products']}."]
        if f["layers"]:
            s.append(f"Shares are held {'partly ' if f['direct_owner'] else ''}through {f['layers']} holding "
                     f"layer(s): " + " -> ".join(f"{n} ({j})" for n, j in zip(f["holding_names"],
                                                                               f["intermediate_jurisdictions"]))
                     + f", which holds {f['holding_share_pct']}% of the applicant.")
            if f["direct_owner"]:
                s.append(f"{f['direct_owner']['name']} holds the other {f['direct_owner']['pct']}% directly.")
            s.append("Top entity owned by " + ", ".join(f"{p['name']} {p['top_pct']}%" for p in f["people"])
                     + (f", and {f['unidentified_pct']}% by {self.UNIDENT[f['unidentified_kind']]}"
                        if f["unidentified_pct"] else "") + ".")
        else:
            s.append("Owned directly by " + ", ".join(f"{p['name']} {p['top_pct']}%" for p in f["people"])
                     + (f", and {f['unidentified_pct']}% by {self.UNIDENT[f['unidentified_kind']]}"
                        if f["unidentified_pct"] else "") + ".")
        for p in f["people"]:
            if p["id_check"] != "verified":
                s.append(f"ID for {p['name']} is {p['id_check'].replace('_', ' ')}.")
        for x in [self.RATIONALE[f["structure_rationale"]], self._mismatch_text(f)]:
            if x:
                s.append(x)
        if f["nominee_director"]:
            s.append("The sole director is a corporate-services nominee.")
        if f["poa_controller"]:
            s.append(f"RM says day-to-day instructions will come from {f['poa_controller']} under a power of "
                     "attorney; not on the share register.")
        s.append(f"Registry extract on file is {f['registry_extract_age_days']} days old (limit 90; I re-pull a "
                 "stale one myself, so that alone does not hold the file).")
        return (f"Case {f['case_id']} -- analyst notes ({f['analyst']}, {fmt_date(f['opened'], rng)})\n\n"
                + noisy(join_sentences(s), rng, 0.15))

    def questions(self):
        return {
            "onboarding_decision": choice("What should the bank decide on this application at this stage?", {
                "approve_after_enhanced_review": "Approve after enhanced due diligence is recorded: everyone at 25% "
                                                 "or more is verified, but the chain runs through an elevated- or "
                                                 "high-risk jurisdiction or two or more layers, the director is a "
                                                 "nominee, or a share under 25% is untraced.",
                "approve_standard": "Approve under standard due diligence: everyone at 25% or more is verified, any "
                                    "holding layer is a single standard-risk one, and nothing else is flagged.",
                "decline_relationship": "Decline: 25% or more of the ownership sits behind bearer shares or a "
                                        "nominee acting for an undisclosed principal.",
                "pend_for_documents": "Hold and ask the client for documents: an owner at 25% or more (effective) "
                                      "is not yet verified or their ID has expired, or 25% or more sits with a "
                                      "trust or fund whose people are not named.",
                "refer_to_mlro": "Refer to the money-laundering reporting officer: the registry filing contradicts "
                                 "the declared owners, or someone outside the register gives the instructions.",
            }),
            "ownership_opacity": score("How hard is it to see who ultimately owns and controls this business?", [
                "Transparent: owners and control are plain from the documents.",
                "Mostly clear: a layer or two, fully traced.",
                "Partly obscured: some ownership or control rests on unverified or untraced parties.",
                "Opaque: the people behind the business cannot be established from the file.",
            ]),
            "documentation_gaps": score("How much of the ownership documentation is missing, stale or "
                                        "inconsistent?", [
                "None: everything is current and consistent.",
                "Minor: one item is slightly out of date or pending.",
                "Material: key items are missing, expired or out of date.",
                "Severe: the documents contradict each other or leave major owners undocumented.",
            ]),
            "ubo_threshold_satisfied": noul(
                "Every natural person with an effective stake of 25% or more has been identified and verified.",
                yes="Everyone at or above the threshold, directly or through intermediate entities, is named with "
                    "a completed identity check.",
                no="Someone at or above the threshold is unnamed, unverified, hidden behind a nominee, trust, fund "
                   "or bearer shares, or the records disagree about who they are."),
            "control_outside_register": noul(
                "Someone other than the recorded shareholders appears to exercise control over the business."),
        }


# =========================================================================== schema 2
BASIS = {
    "dual_use_exports": ("export of military-use electronic components", ["electronics import-export",
                                                                          "freight forwarder"]),
    "armed_group_finance": ("financing a designated armed group", ["precious-metals trader", "payment facilitator"]),
    "state_embezzlement": ("misappropriation of state funds (asset freeze)", ["real-estate holding company",
                                                                               "construction subcontractor"]),
    "port_services": ("providing services to a designated port operator", ["freight forwarder"]),
    "sanctions_evasion": ("facilitating payments for designated entities", ["payment facilitator",
                                                                            "crypto-asset brokerage"]),
}
# bank typology: besides BASIS's typical channels, businesses that are also exposed to the listed conduct
NEXUS_ALSO = {"dual_use_exports": ["software consultancy", "payment facilitator"],
              "armed_group_finance": ["crypto-asset brokerage", "used-vehicle dealer"],
              "state_embezzlement": ["precious-metals trader", "restaurant group"],
              "port_services": ["electronics import-export", "construction subcontractor"],
              "sanctions_evasion": ["precious-metals trader", "electronics import-export"]}


def _typology(basis: str) -> str:
    return (f"typical channels: {', '.join(BASIS[basis][1])}; also exposed: {', '.join(NEXUS_ALSO[basis])}")


SPELL = [("i", "y"), ("k", "c"), ("v", "w"), ("ph", "f"), ("ou", "u"), ("e", "a"), ("o", "u"), ("a", "e")]


def _ascii(s: str) -> str:
    table_ = str.maketrans("øüäöéáíñćčšžřěåÅÖÜØ", "ouaoeainccszreaAOUO")
    return s.translate(table_)


def _variant(rng, name: str, kind: str) -> str:
    first, last = name.split()[0], name.split()[-1]
    if kind == "spelling":
        tok = pick(rng, ["first", "last"])
        w = first if tok == "first" else last
        for a, b in rng.sample(SPELL, len(SPELL)):
            if a in w[1:].lower():
                i = w.lower().index(a, 1)
                w = w[:i] + b + w[i + len(a):]
                break
        else:
            w = w + "e"
        first, last = (w, last) if tok == "first" else (first, w)
        return f"{_ascii(last).upper()}, {_ascii(first)}"
    if kind == "initial":
        return f"{_ascii(last).upper()}, {first[0]}."
    if kind == "extra_names":
        return f"{_ascii(last).upper()}, {_ascii(first)} {pick(rng, FIRST)}"
    return f"{_ascii(last).upper()}, {_ascii(first)}"


class SanctionsNameHit(Schema):
    name = "sanctions_name_hit"
    description = "A sanctions-screening hit on someone connected to a business applicant, to be resolved."
    surfaces = {"json": 0.5, "screening_console": 0.25, "l1_analyst_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        basis = pick(rng, list(BASIS))
        linked = BASIS[basis][1]
        industry = pick(rng, linked) if chance(rng, 0.35) else pick(rng, list(INDUSTRIES))
        role = pick_w(rng, {"director": 3, "beneficial_owner": 3, "authorised_signatory": 2})
        variant = pick_w(rng, {"exact": 3, "spelling": 3, "initial": 1, "extra_names": 2, "alias": 1})
        dob_cmp = pick_w(rng, {"exact": 2, "same_year": 2, "differs": 3, "not_on_list": 2})
        loc_cmp = pick_w(rng, {"same_city": 2, "same_jurisdiction": 2, "different": 3, "not_on_list": 2})
        id_cmp = pick_w(rng, {"match": 1, "differs": 2, "not_available": 5})
        common = chance(rng, 0.35)
        l1 = pick_w(rng, {"likely_false": 3, "cannot_rule_out": 2, None: 2})
        if reg == "clear":
            if chance(rng, 0.6):
                variant, dob_cmp, loc_cmp, id_cmp = pick(rng, ["spelling", "initial"]), "differs", "different", \
                    pick(rng, ["differs", "not_available"])
                industry = pick(rng, [i for i in INDUSTRIES if i not in linked])
                common, l1 = True, "likely_false"
            else:
                variant, dob_cmp, loc_cmp, id_cmp = "exact", "exact", "same_city", "match"
                industry, l1 = pick(rng, linked), "cannot_rule_out"
        elif reg == "borderline":
            variant = pick(rng, ["exact", "spelling", "extra_names"])
            dob_cmp = pick(rng, ["same_year", "not_on_list"])
            loc_cmp = pick(rng, ["same_jurisdiction", "not_on_list"])
            id_cmp = "not_available"
        else:
            if chance(rng, 0.5):
                # strong name, place and business fit, but the listed date of birth is a few years off
                variant, loc_cmp, dob_cmp, id_cmp = "exact", "same_city", "differs", "not_available"
                industry, l1 = pick(rng, linked), "likely_false"
            else:
                # the date of birth matches exactly, but the name only partly and the place not at all
                variant, dob_cmp, loc_cmp, l1 = pick(rng, ["initial", "spelling"]), "exact", "different", \
                    "likely_false"
        name = person(rng)
        dob = date(rng.randint(1952, 1994), rng.randint(1, 12), rng.randint(1, 28))
        gap = pick(rng, [2, 3, 4]) if (reg == "conflicting" and dob_cmp == "differs") else pick(rng, [3, 6, 9, 14])
        list_dob = {"exact": dob.isoformat(), "same_year": str(dob.year), "not_on_list": None,
                    "differs": date(dob.year + pick(rng, [-1, 1]) * gap, rng.randint(1, 12), rng.randint(1, 28)).isoformat()
                    }[dob_cmp]
        home_j = pick_w(rng, {"domestic": 6, "Calverra": 1, "Northmere": 1})
        our_city = pick(rng, CITIES[home_j])
        list_j = {"same_city": home_j, "same_jurisdiction": home_j, "different": pick(rng, FOREIGN),
                  "not_on_list": None}[loc_cmp]
        list_city = (our_city if loc_cmp == "same_city" else
                     pick(rng, [c for c in CITIES[list_j] if c != our_city] or ["(city not given)"]) if list_j else None)
        if list_j == home_j and loc_cmp == "same_jurisdiction" and list_city == "(city not given)":
            list_city = None
        score_ = {"exact": 1.0, "alias": 1.0, "spelling": between(rng, 0.86, 0.95), "initial": between(rng, 0.78, 0.86),
                  "extra_names": between(rng, 0.84, 0.93)}[variant]
        list_name = (f"{_ascii(person(rng).split()[-1]).upper()}, {_ascii(person(rng).split()[0])}"
                     if variant == "alias" else _variant(rng, name, variant))
        return {"regime": reg, "applicant": _company(rng, industry), "industry": industry, "party_name": name,
                "party_role": role, "party_dob": dob, "party_city": our_city, "party_jurisdiction": home_j,
                "list_name": list_name, "alias_hit": variant == "alias", "name_variant": variant,
                "match_score": score_, "list_dob": list_dob, "dob_comparison": dob_cmp, "list_city": list_city,
                "list_jurisdiction": list_j, "location_comparison": loc_cmp, "id_comparison": id_cmp,
                "designation_basis": basis, "listed_since": rand_date(rng, date(2012, 1, 1), date(2024, 12, 31)),
                "common_name": common, "prior_hits_same_name_12m": pick(rng, [6, 11, 23]) if common else pick(rng, [0, 0, 1]),
                "l1_view": l1, "entry_id": ident(rng, "CSL", 5, "-"), "alert_ref": ident(rng, "SCR", 7, ""),
                "analyst": person(rng), "screened_on": rand_date(rng)}

    def _id_text(self, f):
        return {"match": "passport number on the entry ends in the same four digits as the one we hold",
                "differs": "passport number on the entry differs from the one we hold",
                "not_available": "no document number on the entry"}[f["id_comparison"]]

    def render(self, f, rng, surface):
        basis = BASIS[f["designation_basis"]][0]
        role = f["party_role"].replace("_", " ")
        l1 = {"likely_false": pick(rng, ["Looks like a different person to me.", "Probably a false match."]),
              "cannot_rule_out": pick(rng, ["Cannot rule this out on what we have.", "Not comfortable closing this."]),
              None: None}[f["l1_view"]]
        list_loc = (f"{f['list_city']}, {f['list_jurisdiction']}" if f["list_city"] and f["list_jurisdiction"]
                    else (f["list_jurisdiction"] or None))
        if surface == "json":
            return dumps(prune({
                "hit": {"reference": f["alert_ref"], "screened_on": f["screened_on"].isoformat(),
                        "match_score": f["match_score"], "matched_on": "alias" if f["alias_hit"] else "primary name"},
                "screened_party": {"name": f["party_name"], "role": f"{role} of {f['applicant']}",
                                   "date_of_birth": f["party_dob"].isoformat(),
                                   "residence": f"{f['party_city']}, {f['party_jurisdiction']}"},
                "list_entry": {"id": f["entry_id"], "name": f["list_name"] if not f["alias_hit"] else
                               f"{f['list_name']} (a.k.a. {_ascii(f['party_name'].split()[-1]).upper()}, "
                               f"{_ascii(f['party_name'].split()[0])})",
                               "date_of_birth": f["list_dob"], "address": list_loc, "identifiers": self._id_text(f),
                               "designation": basis, "listed_since": f["listed_since"].isoformat(),
                               "bank_typology": _typology(f["designation_basis"])},
                "applicant": {"name": f["applicant"], "business": f["industry"],
                              "bank_industry_tier": INDUSTRIES[f["industry"]]},
                "screening_history": {"hits_on_same_name_last_12m": f["prior_hits_same_name_12m"]},
                "level1_comment": l1,
            }))
        if surface == "screening_console":
            rows = [["Name", f["party_name"], f["list_name"] + (" (alias)" if f["alias_hit"] else "")],
                    ["Date of birth", fmt_date(f["party_dob"], rng, "iso"), f["list_dob"] or "not given on entry"],
                    ["Location", f"{f['party_city']}, {f['party_jurisdiction']}", list_loc or "not given on entry"],
                    ["ID document", "held on file", self._id_text(f)]]
            return "\n".join([
                f"SCREENING HIT {f['alert_ref']}   score {f['match_score']:.2f}   status: OPEN",
                f"Party: {f['party_name']} -- {role}, {f['applicant']} ({f['industry']}; industry tier "
                f"{INDUSTRIES[f['industry']]})", "",
                table(["Field", "Our record", f"List entry {f['entry_id']}"], rows, rng), "",
                kv_lines([("Designation", basis), ("Bank typology", _typology(f["designation_basis"])),
                          ("Listed since", fmt_date(f["listed_since"], rng, pick(rng, ["iso", "us", "long", "eu"]))),
                          ("Same-name hits closed in last 12 months", f["prior_hits_same_name_12m"])], rng)]
                + ([f"L1 comment: {l1}"] if l1 else []))
        # l1_analyst_note
        s = [f"Hit {f['alert_ref']} on {f['party_name']} ({role} of {f['applicant']}, a {f['industry']}) against "
             f"list entry {f['entry_id']} ({f['list_name']}{', alias match' if f['alias_hit'] else ''}; score "
             f"{f['match_score']:.2f})."]
        s.append({"exact": "Date of birth on the entry is identical to ours.",
                  "same_year": f"Entry gives only a birth year, {f['list_dob']}, which fits ours.",
                  "differs": f"Entry DOB is {f['list_dob']}; ours is {f['party_dob'].isoformat()}.",
                  "not_on_list": "Entry has no date of birth."}[f["dob_comparison"]])
        s.append({"same_city": f"Entry address is {f['list_city']}, the same city our party lives in.",
                  "same_jurisdiction": f"Entry address is in {f['list_jurisdiction']}"
                                       + (f" ({f['list_city']})" if f["list_city"] else "") + "; our party lives in "
                                       f"{f['party_city']}.",
                  "different": f"Entry address is in {list_loc}; our party lives in {f['party_city']}.",
                  "not_on_list": "Entry has no address."}[f["location_comparison"]])
        s.append(self._id_text(f).capitalize() + ".")
        s.append(f"Designated for {basis}. Bank typology for that: {_typology(f['designation_basis'])}. Our "
                 f"applicant's industry tier is {INDUSTRIES[f['industry']]}.")
        if f["prior_hits_same_name_12m"] > 3:
            s.append(f"This name has produced {f['prior_hits_same_name_12m']} hits in the last year, all closed.")
        if l1:
            s.append(l1)
        return f"L1 review note -- {f['analyst']}, {fmt_date(f['screened_on'], rng)}\n\n" + noisy(join_sentences(s), rng, 0.12)

    def questions(self):
        return {
            "hit_resolution": choice("How should this screening hit be resolved?", {
                "close_as_false_match": "Close the hit: the listed party is a different person.",
                "escalate_possible_match": "Escalate as a possible match and hold the application meanwhile.",
                "request_identifiers": "Ask the applicant for more identifying details, such as a full date of "
                                       "birth or ID document, before deciding.",
                "treat_as_confirmed_match": "Treat it as a confirmed match: stop onboarding, freeze anything held "
                                            "and follow the reporting procedure.",
            }),
            "identity_match_strength": score("How strongly do the details support this being the same person as "
                                             "the list entry?", [
                "Not at all: the details point to a different person.",
                "Weakly: only a partial name resemblance.",
                "Moderately: the name fits and nothing contradicts it, but little confirms it.",
                "Strongly: several independent details line up.",
                "Conclusively: the details leave no real doubt.",
            ]),
            "business_nexus": score("How closely does the applicant's business connect to the conduct behind the "
                                    "listing?", [
                "No connection: the business is not in the bank's typology for the listed conduct, and its "
                "industry tier is standard.",
                "Remote: not in the typology, but the bank rates the industry elevated or high risk.",
                "Relevant: the typology lists this kind of business as also exposed to the listed conduct.",
                "Direct: the business is one of the typology's typical channels for the listed conduct.",
            ]),
            "identifier_contradiction": noul(
                "At least one reliable secondary identifier contradicts the list entry.",
                yes="A date of birth, document number or location that is dependable on both sides clearly "
                    "differs.",
                no="The identifiers agree, are missing on one side, or are too vague to rule the match out."),
            "best_next_evidence": choice("What would most help settle whether the screened person is the listed "
                                         "one?", {
                "address_history": "The person's address history, to test the location given on the list entry.",
                "document_number_comparison": "A comparison of identity-document numbers held on each side.",
                "full_date_of_birth": "A full date of birth on both sides, where one side has only a year or "
                                      "nothing.",
                "nothing_further": "Nothing further; the details already on file settle the question.",
            }),
        }


# =========================================================================== schema 3
PEP_ROLES = {
    "deputy_minister_transport": ("deputy minister for transport", ["freight forwarder", "construction subcontractor"]),
    "regional_assembly": ("member of a regional assembly", ["real-estate holding company", "restaurant group"]),
    "port_authority_board": ("board member of the state port authority", ["freight forwarder",
                                                                          "electronics import-export"]),
    "mayor": ("mayor of a mid-sized city", ["construction subcontractor", "commercial cleaning contractor"]),
    "defence_procurement": ("head of procurement at the defence ministry", ["electronics import-export",
                                                                            "software consultancy"]),
    "health_agency": ("director of the national health purchasing agency", ["medical supplies wholesaler"]),
    "utility_ceo": ("chief executive of a state-owned utility", ["construction subcontractor",
                                                                 "software consultancy"]),
}
WEALTH = {"salary_and_investments": "career salary and investment income",
          "business_sale": "sale of a family logistics business in 2014",
          "inheritance": "an inheritance",
          "consulting_fees": "consulting fees earned since leaving office",
          "property": "rental property portfolio"}


class PepOwnerExposure(Schema):
    name = "pep_owner_exposure"
    description = "A business applicant with a politically exposed person, or a close associate of one, among its owners."
    surfaces = {"json": 0.5, "edd_memo": 0.25, "screening_summary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        role_key = pick(rng, list(PEP_ROLES))
        linked = PEP_ROLES[role_key][1]
        industry = pick(rng, linked) if chance(rng, 0.45) else pick(rng, list(INDUSTRIES))
        link = pick_w(rng, {"self": 4, "close_relative": 2, "business_associate": 2})
        company_role = pick_w(rng, {"beneficial_owner": 4, "director": 2, "authorised_signatory": 1})
        stake = near(rng, 40, 25, lo=5, hi=100, integer=True) if company_role == "beneficial_owner" else None
        years_out = pick_w(rng, {None: 3, 1: 2, 2: 2, 3: 2, 6: 1, 12: 1})     # None: still in office
        gov_share = pick(rng, [0, 0, 5, 15, 35, 60])
        wealth_src = pick(rng, list(WEALTH))
        wealth_doc = pick_w(rng, {"documented": 3, "partially_documented": 3, "declared_only": 2})
        funds_doc = pick_w(rng, {"documented": 3, "not_documented": 2})
        media = pick_w(rng, {"none": 5, "old_allegation_dismissed": 2, "recent_allegation": 1})
        turnover = _round_k(lognormal_amount(rng, 2_400_000, 0.8))
        expected_monthly = _round_k(turnover / 12 * pick(rng, [0.6, 0.9, 1.1, 1.8, 3.0]))
        if reg == "clear":
            if chance(rng, 0.5):
                years_out, gov_share, wealth_doc, funds_doc, media = 12, 0, "documented", "documented", "none"
                industry = pick(rng, [i for i in INDUSTRIES if i not in linked and INDUSTRIES[i] == "standard"])
                expected_monthly = _round_k(turnover / 12 * 0.9)
            else:
                years_out, gov_share, wealth_doc, media, link = None, 60, "declared_only", "recent_allegation", "self"
                industry = pick(rng, linked)
                expected_monthly = _round_k(turnover / 12 * 3.0)
        elif reg == "borderline":
            years_out = pick(rng, [1, 2, 3])
            wealth_doc = pick(rng, ["partially_documented", "documented"])
            gov_share = pick(rng, [5, 15, 35])
        else:
            if chance(rng, 0.5):
                # long out of office and well documented, but the company now wins contracts in their old portfolio
                years_out, wealth_doc, funds_doc, media = pick(rng, [6, 12]), "documented", "documented", "none"
                industry, gov_share = pick(rng, linked), pick(rng, [35, 60])
            else:
                # still in office and thinly documented, but the business has no public-sector exposure at all
                years_out, wealth_doc, gov_share = None, "partially_documented", 0
                industry = pick(rng, [i for i in INDUSTRIES if i not in linked])
                link, company_role, stake = "close_relative", "beneficial_owner", pick(rng, [10, 15, 20])
        return {"regime": reg, "applicant": _company(rng, industry), "industry": industry,
                "pep_role": role_key, "pep_link": link, "pep_name": person(rng), "party_name": person(rng),
                "company_role": company_role, "stake_pct": stake, "years_since_left_office": years_out,
                "public_contract_revenue_pct": gov_share, "wealth_source": wealth_src, "wealth_evidence": wealth_doc,
                "funds_evidence": funds_doc, "adverse_media": media, "declared_turnover": turnover,
                "expected_monthly_inflow": expected_monthly, "case_id": ident(rng, "EDD", 5, "-"),
                "analyst": person(rng), "rm": person(rng), "date": rand_date(rng)}

    def _who(self, f):
        pos = PEP_ROLES[f["pep_role"]][0]
        tenure = ("currently in office" if f["years_since_left_office"] is None
                  else f"left office {f['years_since_left_office']} year(s) ago")
        if f["pep_link"] == "self":
            return f"{f['party_name']} is a former {pos}" if f["years_since_left_office"] else \
                f"{f['party_name']} is the serving {pos}", tenure
        rel = "a close relative" if f["pep_link"] == "close_relative" else "a long-standing business associate"
        return f"{f['party_name']} is {rel} of {f['pep_name']}, {pos}", tenure

    def _role(self, f):
        return (f"beneficial owner ({f['stake_pct']}%)" if f["company_role"] == "beneficial_owner"
                else f["company_role"].replace("_", " "))

    MEDIA = {"none": "no adverse media found",
             "old_allegation_dismissed": "press reports from years ago alleged favouritism in a tender; the "
                                         "inquiry was closed with no finding",
             "recent_allegation": "a news report this year alleges contracts were steered to connected firms; "
                                  "no charges so far"}
    EVID = {"documented": "documented (statements and deeds on file)",
            "partially_documented": "partly documented (some statements missing)",
            "declared_only": "declared by the client, no evidence yet"}

    def render(self, f, rng, surface):
        who, tenure = self._who(f)
        if surface == "json":
            return dumps(prune({
                "applicant": {"name": f["applicant"], "business": f["industry"],
                              "declared_annual_turnover": f["declared_turnover"],
                              "expected_monthly_inflow": f["expected_monthly_inflow"],
                              "public_sector_revenue_pct": f["public_contract_revenue_pct"], "case": f["case_id"]},
                "pep_match": {"party": f["party_name"], "role_in_applicant": self._role(f),
                              "exposure": who, "status": tenure},
                "due_diligence": {"source_of_wealth": WEALTH[f["wealth_source"]],
                                  "source_of_wealth_evidence": self.EVID[f["wealth_evidence"]],
                                  "initial_funds_evidence": f["funds_evidence"],
                                  "adverse_media": self.MEDIA[f["adverse_media"]]},
                "policy": {"pep_treatment": "enhanced due diligence and senior-management approval while in office "
                                            "and for 24 months after; afterwards risk-based",
                           "after_24_months": "senior approval still needed if half or more of revenue is from "
                                              "public-sector contracts or there is adverse media from the last year",
                           "applies_to": "PEPs, close relatives and known close associates"},
            }))
        if surface == "edd_memo":
            s = [f"{who} ({tenure}); {self._role(f)} of {f['applicant']}, a {f['industry']}."]
            s.append(f"{f['public_contract_revenue_pct']}% of revenue comes from public-sector contracts."
                     if f["public_contract_revenue_pct"] else "The company reports no public-sector revenue.")
            s.append(f"Declared turnover {money(f['declared_turnover'], rng)}; expected inflows about "
                     f"{money(f['expected_monthly_inflow'], rng)} a month.")
            s.append(f"Source of wealth given as {WEALTH[f['wealth_source']]}: {self.EVID[f['wealth_evidence']]}.")
            s.append("Initial deposit source " + ("evidenced." if f["funds_evidence"] == "documented"
                                                  else "not yet evidenced."))
            s.append("Media: " + self.MEDIA[f["adverse_media"]] + ".")
            s.append(pick(rng, [f"RM ({f['rm']}) describes the client as well known locally and keen to start.",
                                f"RM {f['rm']} would like a decision this week.", ""]))
            return (f"ENHANCED DUE DILIGENCE MEMO -- {f['case_id']}\nPrepared by {f['analyst']}, "
                    f"{fmt_date(f['date'], rng)}\n\n" + noisy(join_sentences(s), rng, 0.12)
                    + "\n\nPolicy: PEP measures apply in office and for 24 months after leaving; then risk-based, "
                      "with senior approval still needed where half or more of revenue is public-sector or there is "
                      "adverse media from the last year.")
        rows = [("Case", f["case_id"]), ("Applicant", f"{f['applicant']} ({f['industry']})"),
                ("Matched party", f"{f['party_name']} -- {self._role(f)}"), ("PEP exposure", who),
                ("Status", tenure), ("Public-sector revenue", f"{f['public_contract_revenue_pct']}%"),
                ("Turnover (declared)", money(f["declared_turnover"], rng)),
                ("Expected monthly inflow", money(f["expected_monthly_inflow"], rng)),
                ("Source of wealth", f"{WEALTH[f['wealth_source']]} -- {self.EVID[f['wealth_evidence']]}"),
                ("Initial funds evidenced", "yes" if f["funds_evidence"] == "documented" else "no"),
                ("Adverse media", self.MEDIA[f["adverse_media"]])]
        return ("PEP SCREENING SUMMARY\n" + kv_lines(rows, rng) + "\n\n(PEP treatment: in office + 24 months; "
                "after that, senior sign-off only if public-sector revenue >= 50% or adverse media in the last year.)")

    def questions(self):
        return {
            "pep_relationship_decision": choice("What should the bank do with this application?", {
                "accept_standard_terms": "Accept on standard terms: out of office more than 24 months, under half "
                                         "of revenue public-sector, no adverse media from the last year, and the "
                                         "source of wealth is at least partly evidenced.",
                "accept_with_senior_approval": "Accept with enhanced due diligence and senior-management approval: "
                                               "the policy requires senior sign-off, and the wealth and initial "
                                               "funds are both evidenced.",
                "decline_application": "Decline: a recent allegation of steering contracts, and 25% or more of the "
                                       "business's revenue comes from public-sector contracts.",
                "pend_for_wealth_evidence": "Hold until evidence arrives: the source of wealth is only declared, or "
                                            "senior sign-off is required and the initial funds are not evidenced.",
            }),
            "corruption_exposure": score("How exposed is this relationship to the proceeds of corruption or abuse "
                                         "of office?", [
                "Negligible: no realistic path from the public role to the business's money.",
                "Low: a political link exists, but it is distant or long past.",
                "Material: the public role and the business's revenue could plausibly be connected.",
                "High: the business profits from the very area the person controls or controlled.",
            ]),
            "monitoring_intensity": score("How closely should the relationship be monitored if it goes ahead?", [
                "Standard review cycle with automated monitoring only.",
                "Annual enhanced review.",
                "Semi-annual enhanced review with sampling of larger payments.",
                "Quarterly review with a named senior owner and payment-level scrutiny.",
            ]),
            "wealth_origin_established": noul(
                "The origin of the politically exposed party's wealth has been adequately established.",
                yes="The stated source is backed by documents and fits the size of the money involved.",
                no="The source is only declared, partly evidenced, or does not explain the amounts."),
            "senior_approval_needed": noul(
                "Senior management must approve this relationship before it starts.",
                yes="The person is in office or left within the last 24 months, or half or more of revenue is from "
                    "public-sector contracts, or there is adverse media from the last year.",
                no="The person left office more than 24 months ago, public-sector revenue is under half, and there "
                   "is no adverse media from the last year."),
        }


def _round_k(x: float) -> float:
    return round(x, -3) if x < 1_000_000 else round(x, -4)


# =========================================================================== schema 4
IND_PURPOSE = {
    "software consultancy": ["supplier_payments"], "commercial cleaning contractor": ["supplier_payments"],
    "medical supplies wholesaler": ["supplier_payments", "trade_settlement"],
    "construction subcontractor": ["supplier_payments"], "used-vehicle dealer": ["retail_takings", "supplier_payments"],
    "restaurant group": ["retail_takings"], "freight forwarder": ["trade_settlement", "supplier_payments"],
    "electronics import-export": ["trade_settlement"], "real-estate holding company": ["rent_collection"],
    "precious-metals trader": ["trade_settlement", "retail_takings"], "payment facilitator": ["client_funds"],
    "crypto-asset brokerage": ["client_funds"],
}
REV_PER_HEAD = {"software consultancy": 150_000, "commercial cleaning contractor": 45_000,
                "medical supplies wholesaler": 400_000, "construction subcontractor": 120_000,
                "used-vehicle dealer": 500_000, "restaurant group": 70_000, "freight forwarder": 300_000,
                "electronics import-export": 600_000, "real-estate holding company": 800_000,
                "precious-metals trader": 1_500_000, "payment facilitator": 300_000, "crypto-asset brokerage": 400_000}
PURPOSES = {"supplier_payments": "paying suppliers and staff", "client_funds": "receiving funds on behalf of clients",
            "trade_settlement": "settling cross-border trade", "retail_takings": "depositing daily takings",
            "rent_collection": "collecting rents and paying contractors"}


class ActivityProfileCheck(Schema):
    name = "activity_profile_check"
    description = "A business applicant's declared expected account activity, checked against what the business is."
    surfaces = {"json": 0.5, "application_form": 0.25, "rm_call_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        industry = pick(rng, list(INDUSTRIES))
        years = pick(rng, [0, 1, 3, 7, 15])
        turnover = _round_k(lognormal_amount(rng, 1_800_000, 0.9))
        ratio = pick(rng, [0.5, 0.8, 1.0, 1.3, 2.2, 4.0])            # expected monthly inflow / (turnover / 12)
        purpose = pick(rng, IND_PURPOSE[industry])
        if chance(rng, 0.12):
            purpose = "client_funds"                                  # an odd purpose for the trade
        premises = pick_w(rng, {"operating_site": 4, "shared_office": 2, "formation_agent_address": 1})
        web = (pick_w(rng, {"established": 1, "new_basic": 4, "none": 2}) if years == 0
               else pick_w(rng, {"established": 5, "new_basic": 1, "none": 1}))
        licence = None
        if industry in LICENSED:
            licence = pick_w(rng, {"licence_number_verified": 2, "application_pending": 2, "not_mentioned": 1})
        staff = None
        if reg == "clear":
            if chance(rng, 0.5):
                industry = pick(rng, [i for i in INDUSTRIES if INDUSTRIES[i] == "standard"])
                years, ratio, premises, web, licence = pick(rng, [7, 15]), 1.0, "operating_site", "established", None
                purpose = IND_PURPOSE[industry][0]
            else:
                industry = pick(rng, ["payment facilitator", "precious-metals trader", "electronics import-export"])
                years, ratio, premises, web, staff = 0, 4.0, "formation_agent_address", "none", 1
                purpose = "client_funds"
                licence = "not_mentioned" if industry in LICENSED else None
        elif reg == "borderline":
            ratio = near(rng, 1.5, 0.4, lo=0.8, digits=1)
            years = pick(rng, [0, 1, 3])
            if industry in LICENSED:
                licence = "application_pending"
        else:
            if chance(rng, 0.5):
                # long-established and well staffed, but inflows far above anything the turnover supports
                years, premises, web, ratio = 15, "operating_site", "established", pick(rng, [2.2, 4.0])
                purpose = pick(rng, ["client_funds", IND_PURPOSE[industry][0]])
            else:
                # a brand-new, thin-looking company whose numbers are modest and consistent
                industry = pick(rng, ["software consultancy", "commercial cleaning contractor"])
                years, staff, web, premises, ratio = 0, 1, "new_basic", "formation_agent_address", 0.8
                licence, purpose = None, "supplier_payments"
                turnover = _round_k(between(rng, 60_000, 240_000))
        if industry not in LICENSED:
            licence = None
        if years == 0 and staff is None:
            turnover = _round_k(turnover * rng.uniform(0.2, 0.45))      # a first-year projection is smaller
        cash_pct = pick(rng, [20, 35, 70]) if purpose == "retail_takings" else pick_w(rng, {0: 5, 10: 2})
        if purpose == "trade_settlement" or (reg == "clear" and purpose == "client_funds"):
            corridors = rng.sample(FOREIGN, pick(rng, [1, 2]))
            intl_pct = pick(rng, [30, 45, 60])
        else:
            corridors = rng.sample(FOREIGN, pick(rng, [0, 0, 0, 1]))
            intl_pct = pick(rng, [5, 10, 20]) if corridors else 0
        if reg == "clear" and years == 0:
            corridors, intl_pct = ["Port Selan Free Zone", "Veradia"], 60
        if reg == "clear" and years >= 7:
            corridors, intl_pct = [], 0
        if staff is None:
            staff = max(1, int(round(turnover / REV_PER_HEAD[industry] * rng.uniform(0.7, 1.3))))
            if years == 0:
                staff = max(1, min(staff, pick(rng, [1, 2, 4, 6])))
        expected = _round_k(turnover / 12 * ratio)
        juris = pick_w(rng, {"domestic": 6, "Calverra": 1, "Isle of Brenn": 1})
        return {"regime": reg, "applicant": _company(rng, industry), "industry": industry,
                "industry_risk": INDUSTRIES[industry], "registered_in": juris, "years_trading": years,
                "declared_turnover": turnover if years else None, "projected_turnover": None if years else turnover,
                "inflow_ratio": ratio, "expected_monthly_inflow": expected, "cash_share_pct": cash_pct,
                "payment_corridors": corridors, "international_share_pct": intl_pct, "staff": staff,
                "premises": premises, "web_presence": web, "account_purpose": purpose, "licence": licence,
                "financials_provided": maybe(rng, pick(rng, ["audited", "management_accounts"]), 0.3) if years else None,
                "rm": person(rng), "director": person(rng), "app_id": ident(rng, "APP", 6, "-"), "date": rand_date(rng)}

    PREMISES = {"operating_site": "own operating premises (visited by RM)",
                "shared_office": "desk in a shared office",
                "formation_agent_address": "registered at a company-formation agent's address"}
    WEB = {"established": "established website with client references", "new_basic": "a new one-page website",
           "none": "no website or online presence found"}
    LIC = {"licence_number_verified": "licence number provided and checked on the regulator's register",
           "application_pending": "says a licence application is pending with the regulator",
           "not_mentioned": "no licence mentioned"}

    def _corr(self, f):
        if not f["payment_corridors"]:
            return "domestic only"
        return (f"{f['international_share_pct']}% international, mainly "
                + ", ".join(f"{c} ({JURIS[c]})" for c in f["payment_corridors"]))

    def render(self, f, rng, surface):
        turn_lbl = "declared_annual_turnover" if f["years_trading"] else "projected_first_year_turnover"
        turn = f["declared_turnover"] or f["projected_turnover"]
        if surface == "json":
            return dumps(prune({
                "applicant": {"id": f["app_id"], "name": f["applicant"], "business": f["industry"],
                              "registered_in": _juris_label(f["registered_in"]), "years_trading": f["years_trading"],
                              "staff": f["staff"], "premises": self.PREMISES[f["premises"]],
                              "online_presence": self.WEB[f["web_presence"]]},
                "financials": {turn_lbl: turn, "statements": f["financials_provided"]},
                "expected_activity": {"monthly_inflow": f["expected_monthly_inflow"],
                                      "inflow_vs_one_month_of_turnover": f"{f['inflow_ratio']:.1f}x",
                                      "cash_deposits_pct": f["cash_share_pct"], "payments": self._corr(f),
                                      "account_purpose": PURPOSES[f["account_purpose"]]},
                "regulatory": {"licence": self.LIC[f["licence"]] if f["licence"] else None},
                "risk_model_inputs": {"industry_tier": f["industry_risk"]},
            }))
        if surface == "application_form":
            rows = [("Application", f["app_id"]), ("Business name", f["applicant"]),
                    ("Nature of business", f["industry"]), ("Registered in", _juris_label(f["registered_in"])),
                    ("Years trading", f["years_trading"] or "new company"), ("Employees", f["staff"]),
                    ("Annual turnover" if f["years_trading"] else "Projected turnover (yr 1)", money(turn, rng)),
                    ("Financial statements", show(f["financials_provided"], rng, lambda v: v.replace("_", " "))
                     if f["years_trading"] else "n/a (new company)"),
                    ("Expected monthly credits", money(f["expected_monthly_inflow"], rng)),
                    ("Credits vs one month of turnover", f"{f['inflow_ratio']:.1f}x"),
                    ("Cash deposits (% of credits)", f"{f['cash_share_pct']}%"),
                    ("International payments", self._corr(f)),
                    ("Main purpose of account", PURPOSES[f["account_purpose"]]),
                    ("Regulatory licence", self.LIC[f["licence"]] if f["licence"] else "not applicable")]
            return "BUSINESS ACCOUNT APPLICATION -- expected activity\n" + kv_lines(rows, rng)
        s = [f"Call with {f['director']} (director) of {f['applicant']}, {fmt_date(f['date'], rng)}."]
        s.append(f"They run a {f['industry']}, " + (f"trading {f['years_trading']} years"
                                                    if f["years_trading"] else "newly incorporated")
                 + f", {f['staff']} staff, {self.PREMISES[f['premises']]}.")
        s.append(("Turnover about " if f["years_trading"] else "Projects first-year turnover of ")
                 + money(turn, rng) + f"; expects roughly {money(f['expected_monthly_inflow'], rng)} a month "
                 f"through the account ({f['inflow_ratio']:.1f}x a month of turnover), mainly for "
                 f"{PURPOSES[f['account_purpose']]}.")
        if f["cash_share_pct"]:
            s.append(f"Around {f['cash_share_pct']}% would be cash.")
        s.append("Payments: " + self._corr(f) + ".")
        s.append("Online: " + self.WEB[f["web_presence"]] + ".")
        if f["licence"]:
            s.append("Licence: " + self.LIC[f["licence"]] + ".")
        s.append(pick(rng, ["Came across as knowledgeable about the business.",
                            "Was vague when asked who the main clients are.",
                            "Keen to open quickly; mentioned a large contract starting next month.", ""]))
        return f"RM call note -- {f['rm']}\n\n" + noisy(join_sentences(s), rng, 0.15)

    def questions(self):
        return {
            "profile_decision": choice("How should the bank proceed with this application?", {
                "decline_application": "Decline; the profile is outside the bank's risk appetite.",
                "open_account": "Open the account on the profile as declared.",
                "open_with_restrictions": "Open the account with limits, such as no international payments or "
                                          "capped volumes, and review after the first months of activity.",
                "request_supporting_evidence": "Ask for evidence, such as contracts, bank statements or a licence, "
                                               "before deciding.",
            }),
            "edd_priority": choice("Which additional check would do most to resolve the doubts in this file?", {
                "licence_confirmation": "Confirm with the regulator that the activity is properly licensed.",
                "no_further_check": "None; the file already supports a decision.",
                "premises_verification": "Verify that the business actually operates where and how it says.",
                "source_of_funds_evidence": "Obtain evidence of where the expected money will come from.",
                "trade_counterparty_checks": "Identify and screen the main foreign trading counterparties.",
            }),
            "onboarding_risk_rating": score("What customer risk rating should this business receive at "
                                            "onboarding?", [
                "Low: an ordinary business with activity that fits it.",
                "Medium: some inherent risk, but the picture is consistent.",
                "High: inherent risk plus gaps or inconsistencies that need close watching.",
                "Unacceptable: outside what the bank will take on.",
            ]),
            "activity_plausibility": score("How plausible is the expected account activity for a business of this "
                                           "kind and size?", [
                "Implausible: expected monthly credits are more than 2.5x a month of turnover.",
                "Doubtful: over 1.6x and up to 2.5x a month of turnover, or an account purpose (such as receiving "
                "clients' money) that this trade does not have.",
                "Plausible: over 1.2x and up to 1.6x, or under 0.7x, a month of turnover, for a purpose that fits "
                "the trade.",
                "Entirely plausible: 0.7x to 1.2x a month of turnover, for a purpose that fits the trade.",
            ]),
            "licensing_gap": noul(
                "The business appears to carry on an activity that needs a licence it has not shown.",
                yes="It holds or moves money for others, or runs a regulated service, without a verified licence.",
                no="Its activity needs no licence, or a verified licence covers it."),
        }


# =========================================================================== schema 5
class PeriodicReviewRefresh(Schema):
    name = "periodic_review_refresh"
    description = "A scheduled KYC refresh of an existing business client whose circumstances may have changed."
    surfaces = {"json": 0.5, "review_worksheet": 0.25, "outreach_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        industry = pick(rng, list(INDUSTRIES))
        rating = pick_w(rng, {"low": 3, "medium": 4, "high": 2})
        since = pick(rng, [2, 4, 7, 11, 18])
        turnover_change = pick(rng, [-40, -10, 5, 20, 60, 180])
        own_change = pick_w(rng, {"none": 5, "new_owner_below_threshold": 2, "new_owner_above_threshold": 2})
        new_corr = rng.sample(FOREIGN, pick(rng, [0, 0, 0, 1, 2]))
        referrals = pick(rng, [0, 0, 1, 2, 4])
        open_ref = referrals > 0 and chance(rng, 0.3)
        id_expired = chance(rng, 0.3)
        extract_age = pick(rng, [30, 200, 500, 900])
        response = pick_w(rng, {"complete": 3, "partial": 3, "none": 2})
        days_since_request = pick(rng, [10, 25, 45, 70, 110])
        reminders = pick(rng, [0, 1, 2, 3])
        due_offset = pick(rng, [-20, 5, 25, 60, 120])          # days past the review due date (negative: not yet due)
        grace = 60
        if reg == "clear":
            if chance(rng, 0.5):
                turnover_change, own_change, new_corr, referrals, open_ref = 5, "none", [], 0, False
                id_expired, extract_age, response, due_offset = False, 30, "complete", pick(rng, [-20, 5])
            else:
                turnover_change, own_change, new_corr, referrals, open_ref = 180, "new_owner_above_threshold", \
                    ["Veradia", "Port Selan Free Zone"], 4, True
                id_expired, response, days_since_request, reminders, due_offset = True, "none", 110, 3, 120
        elif reg == "borderline":
            due_offset = near(rng, grace, 12, integer=True)
            response = "partial"
            turnover_change = pick(rng, [20, 60])
            referrals, open_ref = pick(rng, [1, 2]), False
        else:
            if chance(rng, 0.5):
                # fully cooperative and on time, but the business has visibly changed shape
                response, due_offset, id_expired = "complete", -20, False
                turnover_change, new_corr, own_change = 180, ["Isle of Brenn", "Veradia"], "new_owner_above_threshold"
            else:
                # nothing has changed in the business, but the client ignores the review and it is overdue
                turnover_change, own_change, new_corr, referrals, open_ref = 5, "none", [], 0, False
                response, reminders, days_since_request, due_offset = "none", 3, 110, pick(rng, [60, 120])
                rating = pick(rng, ["low", "medium"])
        if response == "complete":
            reminders = min(reminders, 1)
            extract_age, id_expired = 30, False
            days_since_request = min(days_since_request, pick(rng, [10, 25, 45]))
        # outreach starts about a month before the due date
        days_since_request = max(days_since_request, due_offset + pick(rng, [20, 30, 45]))
        reminders = min(reminders, days_since_request // 14)
        return {"regime": reg, "client": _company(rng, industry), "industry": industry, "current_rating": rating,
                "client_since_years": since, "turnover_change_pct": turnover_change, "ownership_change": own_change,
                "new_owner": person(rng) if own_change != "none" else None,
                "new_owner_pct": (pick(rng, [10, 15, 20]) if own_change == "new_owner_below_threshold"
                                  else pick(rng, [30, 45, 60]) if own_change != "none" else None),
                "new_corridors": new_corr, "unusual_activity_referrals_12m": referrals,
                "referral_still_open": open_ref, "director_id_expired": id_expired,
                "registry_extract_age_days": extract_age, "client_response": response,
                "days_since_first_request": days_since_request, "reminders_sent": reminders,
                "days_past_due": due_offset, "grace_days": grace, "extension_approved": chance(rng, 0.15),
                "reviewer": person(rng), "rm": person(rng), "client_ref": ident(rng, "CIF", 7, ""),
                "review_id": ident(rng, "PR", 6, "-"), "date": rand_date(rng)}

    def _due(self, f):
        d = f["days_past_due"]
        return f"due in {-d} days" if d < 0 else f"{d} days past due date"

    def _changes(self, f):
        c = [f"Turnover {'+' if f['turnover_change_pct'] >= 0 else ''}{f['turnover_change_pct']}% vs last review"]
        if f["ownership_change"] != "none":
            c.append(f"New shareholder {f['new_owner']} ({f['new_owner_pct']}%)")
        if f["new_corridors"]:
            c.append("New payment corridors: " + ", ".join(f"{x} ({JURIS[x]})" for x in f["new_corridors"]))
        if f["unusual_activity_referrals_12m"]:
            c.append(f"{f['unusual_activity_referrals_12m']} internal unusual-activity referral(s) in 12 months"
                     + ("; one still open" if f["referral_still_open"] else ", all closed with no further action"))
        if f["director_id_expired"]:
            c.append("Director's ID document on file has expired")
        return c

    def render(self, f, rng, surface):
        resp = {"complete": "returned all requested documents", "partial": "returned some documents; ownership "
                "confirmation and updated accounts still outstanding", "none": "no response"}[f["client_response"]]
        if surface == "json":
            return dumps(prune({
                "client": {"ref": f["client_ref"], "name": f["client"], "business": f["industry"],
                           "client_since_years": f["client_since_years"], "current_risk_rating": f["current_rating"]},
                "review": {"id": f["review_id"], "status": self._due(f), "grace_period_days": f["grace_days"],
                           "extension_approved": f["extension_approved"] or None, "reviewer": f["reviewer"]},
                "changes_since_last_review": self._changes(f),
                "documents": {"registry_extract_age_days": f["registry_extract_age_days"],
                              "director_id_expired": f["director_id_expired"]},
                "outreach": {"days_since_first_request": f["days_since_first_request"],
                             "reminders_sent": f["reminders_sent"], "client_response": resp},
            }))
        if surface == "review_worksheet":
            head = kv_lines([("Review", f["review_id"]), ("Client", f"{f['client']} ({f['client_ref']})"),
                             ("Business", f["industry"]), ("Client since", f"{f['client_since_years']} years"),
                             ("Current rating", f["current_rating"].upper()), ("Review status", self._due(f)),
                             ("Grace period", f"{f['grace_days']} days"
                              + (" (extension approved)" if f["extension_approved"] else "")),
                             ("Registry extract", f"{f['registry_extract_age_days']} days old")], rng)
            return (f"PERIODIC KYC REVIEW WORKSHEET\n{head}\n\nChanges identified:\n"
                    + bullets(self._changes(f), rng) + f"\n\nClient outreach: {resp} "
                    f"({f['reminders_sent']} reminder(s), first request {f['days_since_first_request']} days ago).")
        start = f["date"] - timedelta(days=f["days_since_first_request"])
        log = [(start, "Refresh pack sent to client (ownership confirmation, accounts, director ID)")]
        for i in range(f["reminders_sent"]):
            log.append((start + timedelta(days=14 * (i + 1)), f"Reminder {i + 1} sent by {f['rm'].split()[0]}"))
        if f["client_response"] != "none":
            log.append((start + timedelta(days=rng.randint(5, max(6, f["days_since_first_request"] - 2))),
                        "Client " + resp))
        log.sort(key=lambda x: x[0])
        lines = [f"{fmt_date(d, rng, 'iso')}  {t}" for d, t in log]
        return (f"Outreach log -- {f['client']} ({f['client_ref']}), rating {f['current_rating']}, "
                f"review {self._due(f)}\n" + "\n".join(lines) + "\n\nReviewer notes:\n"
                + noisy(join_sentences(c + "." for c in self._changes(f)), rng, 0.12)
                + (" Extension to the review deadline was approved." if f["extension_approved"] else ""))

    def questions(self):
        return {
            "refresh_outcome": choice("How should this periodic review be concluded?", {
                "lower_risk_rating": "Close the review and lower the client's risk rating.",
                "raise_risk_rating": "Close the review and raise the client's risk rating.",
                "restrict_until_documents": "Restrict new activity on the account until the outstanding "
                                            "information arrives.",
                "retain_current_rating": "Close the review and keep the current rating.",
                "start_exit_process": "Begin exiting the relationship.",
            }),
            "risk_drift": score("How much has the client's risk profile changed since the last review?", [
                "Unchanged: the business looks as it did.",
                "Slightly: normal growth or a minor change.",
                "Noticeably: new owners, markets or activity that alter the risk.",
                "Fundamentally: the client is effectively a different business now.",
            ]),
            "deadline_pressure": score("How pressing is the review deadline for completing this file?", [
                "Not pressing: the review is not yet due.",
                "Approaching: due now, with the grace period still ahead.",
                "Pressing: the grace period is close to running out.",
                "Breached: past the grace period, a compliance exception that must be reported.",
            ]),
            "profile_current": noul(
                "The information on file still describes how the client's business operates today.",
                yes="Turnover has moved by 50% or less either way, no new shareholder holds 25% or more, and no "
                    "new payment corridor runs to an elevated- or high-risk jurisdiction.",
                no="Turnover has moved by more than 50%, a new shareholder holds 25% or more, or a new corridor "
                   "runs to an elevated- or high-risk jurisdiction."),
            "client_cooperating": noul("The client is cooperating adequately with the review."),
        }


WORKFLOW = Workflow(
    name="kyc_business_onboarding",
    title="KYC business onboarding",
    description="Whether a bank should take on or keep a business client, after tracing its owners, resolving "
                "screening hits and checking its expected activity, and on what terms.",
    domain="banking / financial-crime compliance",
    schemas=[BeneficialOwnershipTrace(), SanctionsNameHit(), PepOwnerExposure(), ActivityProfileCheck(),
             PeriodicReviewRefresh()],
    notes="Customer due diligence on business clients with fictional jurisdictions, companies and people. "
          "Country risk attaches only to company registration and payment corridors, never to a person. No "
          "transaction-fraud alerts, no machine-security or credential questions, no support threads.",
)
