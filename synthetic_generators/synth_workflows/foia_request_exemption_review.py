"""Public-records request review against a written, FICTIONAL records-access statute.

Six schemas: the exemption review of a request's responsive records, a fee-waiver request, an expedited-processing
request, a business's objection to the release of information it submitted, an administrative appeal of a partial
denial, and a request for records about a public employee's conduct. Every state quotes the statute's own words
(section numbers, deadlines, thresholds and sunset periods are sampled per case), so each answer comes from
comparing the facts with the rule: an investigation that closed last month, a draft memo one year short of the
sunset, a fee estimate just over the advance-payment line, a price list the Act makes public anyway.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'). Borderline cases sit near a stated
threshold; conflicting cases carry a staff opinion or a requester claim that the facts contradict, or leave a key
fact out. `intended(f)` writes down the answer logic a careful records officer would apply to the RENDERED facts;
the builder does not use it, it exists for the spread audit.

All jurisdictions, agencies, statutes, people, outlets and businesses are fictional. Requesters are judged only on
what they ask for and how they will use it (fee category, dissemination, urgency), never on who they are; no
protected attribute appears as a factor.
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
)

JURIS = ["State of Westmarch", "Commonwealth of Calder", "Province of Northvale", "Territory of Ashcombe",
         "State of Lindmoor", "Commonwealth of Brayfield", "Province of Eastholm", "City of Port Elden",
         "Ravensmoor County", "City of Kestrel Falls", "Dunmore Valley Regional District", "State of Orrinford",
         "Tallis County", "City of New Carrow"]
ACTS = ["Public Records Access Act", "Open Government Records Act", "Access to Public Information Act",
        "Government Records Transparency Act", "Public Information Disclosure Act", "Records of Public Bodies Act",
        "Open Records and Meetings Act"]
AGENCIES = ["Department of Transportation", "Department of Environmental Quality", "Housing Authority",
            "Department of Public Works", "Parks and Recreation Department", "Water and Sewer Board",
            "Office of Procurement", "Department of Health Services", "Transit Authority", "Port Authority",
            "Department of Labour Standards", "Planning and Zoning Office", "Fire Marshal's Office",
            "Department of Agriculture", "Economic Development Office", "Office of the Treasurer",
            "Building Inspections Division", "Animal Services Bureau", "Library Board",
            "Department of Consumer Protection", "Fisheries and Wildlife Division", "Solid Waste Authority"]
OUTLET_B = ["Courier", "Ledger", "Gazette", "Dispatch", "Sentinel", "Review", "Bulletin", "Chronicle", "Tribune",
            "Record"]


def _outlet(rng):
    return f"The {pick(rng, ORG_A)} {pick(rng, OUTLET_B)}"


def _act(rng):
    """The statute a case is judged under: jurisdiction, name, citation style and section numbering."""
    return {"jurisdiction": pick(rng, JURIS), "act": pick(rng, ACTS), "sec": pick(rng, ["s. ", "§ ", "Sec. "]),
            "part": rng.randint(5, 14)}


def _cite(a, sub, letter=None):
    """A citation like 's. 9(1)(c)' or '§ 11'. sub is an offset from the act's base part number."""
    return f"{a['sec']}{a['part'] + sub}" + ("" if letter is None else f"(1)({letter})")


def _agency(rng, juris, base_name=None):
    """'<jurisdiction short name> <agency>'; base_name ties the agency to the topic, else one is drawn."""
    base_name = base_name or pick(rng, AGENCIES)
    return f"{juris.split(' of ')[-1]} {base_name}" if " of " in juris else f"{juris} {base_name}"


# which agency plausibly holds the records on each subject (subjects are unique across the pools below)
AGENCY_FOR = {
    # schema 1: privacy
    "noise complaints about the Quarry Road events venue": "Environmental Health Office",
    "injuries reported at municipal swimming pools": "Parks and Recreation Department",
    "dog-bite reports in the Fernhill district": "Animal Services Bureau",
    "the lead-pipe replacement grant scheme": "Water and Sewer Board",
    "public comments on the landfill expansion": "Solid Waste Authority",
    "tenant complaints at the Larkspur Court housing block": "Housing Authority",
    "water shut-offs for unpaid bills": "Water and Sewer Board",
    "the youth summer-jobs scheme": "Office of Workforce Development",
    "parking-permit appeals on the heritage streets": "Department of Transportation",
    "complaints about missed bin collections on the east routes": "Solid Waste Authority",
    "the emergency rental-assistance programme": "Housing Authority",
    "injury claims at the road-maintenance depot": "Department of Public Works",
    "visitor incidents at the botanical gardens": "Parks and Recreation Department",
    "residents relocated for the Millrace flood works": "Department of Public Works",
    # enforcement
    "the inspection of a poultry-processing plant on Route 9": "Department of Agriculture",
    "illegal dumping at the old Saltmarsh rail yard": "Department of Environmental Quality",
    "the licence case against a towing operator": "Department of Consumer Protection",
    "wage-theft complaints against a regional cleaning contractor": "Department of Labour Standards",
    "fire-code violations at the Ember Street warehouse": "Fire Marshal's Office",
    "price-gouging complaints after the January ice storm": "Department of Consumer Protection",
    "air-permit exceedances at an asphalt plant": "Department of Environmental Quality",
    "a home-solar seller accused of misleading contracts": "Department of Consumer Protection",
    "pesticide drift complaints near the Ashgrove orchards": "Department of Agriculture",
    "elevator-safety failures at the Driftwood Tower": "Building Inspections Division",
    "a food-labelling inquiry into a regional bakery chain": "Department of Health Services",
    "a licensed child-care centre's staffing ratios": "Office of Child Care Licensing",
    "unpermitted tree removal on the Valewood ridge": "Planning and Zoning Office",
    "a scrap-metal dealer's record-keeping violations": "Business Licensing Office",
    # deliberative
    "the bus-depot relocation": "Transit Authority", "the proposed water-rate increase": "Water and Sewer Board",
    "library branch closures": "Library Board", "the downtown parking strategy": "Department of Transportation",
    "the school-zone speed camera pilot": "Department of Transportation",
    "changes to park opening hours": "Parks and Recreation Department",
    "the stormwater fee redesign": "Department of Public Works", "merging two fire stations": "Fire Department",
    "the snow-clearing priority map": "Department of Public Works",
    "scoring for the arts micro-grants round": "Arts and Culture Office",
    "the transit fare restructure": "Transit Authority", "new procurement approval thresholds": "Office of Procurement",
    "relocating the animal shelter": "Animal Services Bureau", "closing the Pinecrest recycling depot": "Solid Waste Authority",
    # commercial
    "the parking-meter replacement contract": "Department of Transportation",
    "the school-meals catering contract": "Department of Education",
    "the fleet-fuel supply contract": "Office of Procurement",
    "the quarry expansion permit": "Department of Environmental Quality",
    "the economic-development grant to a battery maker": "Economic Development Office",
    "the permit-portal software licence": "Department of Information Technology",
    "the waste-hauling franchise renewal": "Solid Waste Authority",
    "the solar-farm lease on county land": "Office of Public Lands",
    "the ambulance billing contract": "Emergency Medical Services Agency",
    "the bridge-inspection services contract": "Department of Transportation",
    "the stadium concession agreement": "Economic Development Office",
    "the tourism-marketing grant to a hotel group": "Economic Development Office",
    "the street-lighting LED retrofit": "Department of Public Works",
    # schema 2: fee subjects
    "overtime payments in the road-maintenance division": "Department of Public Works",
    "the agency's contract with an outside debt collector": "Office of the Treasurer",
    "inspection results for the city's playground equipment": "Parks and Recreation Department",
    "spending on the new permit-portal software": "Department of Information Technology",
    "complaints about the regional bus contractor's reliability": "Transit Authority",
    "legal settlements paid by the agency since 2022": "Law Department",
    "the safety record of a privately owned marina the agency licenses": "Port Authority",
    "historic building permits for one private house": "Building Inspections Division",
    "a private company's environmental filings for its own site": "Department of Environmental Quality",
    "water-quality testing at public beaches": "Department of Health Services",
    "the agency's travel and hospitality expenses": "Office of the Treasurer",
    "a list of all licensed food trucks and their addresses": "Department of Health Services",
    "vacant-property registrations in the old mill district": "Planning and Zoning Office",
    "the consultants' fees on the stadium project": "Economic Development Office",
    "emergency-shelter capacity reports during last winter": "Housing Authority",
    "commercial lease terms the agency granted a cafe operator": "Parks and Recreation Department",
    # schema 3: expedite topics
    "the agency's inspections of a chemical storage depot after a fire nearby": "Fire Marshal's Office",
    "contamination test results for a neighbourhood's drinking-water wells": "Department of Environmental Quality",
    "the structural survey of the Ironbark footbridge": "Department of Transportation",
    "the transit authority's plan to cut night bus routes": "Transit Authority",
    "council members' meetings with the stadium developer": "Office of the Council Clerk",
    "overtime claims by senior staff at the water board": "Water and Sewer Board",
    "the contract awarded for the new jail food service": "Office of Procurement",
    "the agency's handling of a heat-wave cooling-centre shortage": "Department of Health Services",
    "the zoning file for a neighbour's garage extension": "Planning and Zoning Office",
    "historic land records for a family property": "Land Registry Office",
    "a business's own licence history, for its sale": "Business Licensing Office",
    "the agency's records of a car crash the requester was in": "Department of Transportation",
    "the evacuation planning for the river levee upgrade": "Office of Emergency Management",
    "the procurement of body-worn cameras for parking wardens": "Department of Transportation",
}


FIRM_B = ["Systems", "Services", "Contracting", "Technologies", "Partners", "Works", "Solutions", "Engineering"]


def _firm(rng):
    """A contractor name that fits a public tender (base.org's pool includes foods / outfitters)."""
    a, i = pick(rng, ORG_A), rng.randrange(len(FIRM_B))
    for k in range(len(FIRM_B)):
        name = f"{a} {FIRM_B[(i + k) % len(FIRM_B)]}"
        if name not in REAL_NAME_CLASHES:
            return name
    return name


def _yrs(n):
    return f"{n} year{'' if n == 1 else 's'}"


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


# =========================================================================== schema 1
KIND_LETTER = {"privacy": "a", "enforcement": "b", "deliberative": "c", "commercial": "d"}
KIND_KEY = {"privacy": "personal_privacy", "enforcement": "open_enforcement_matter",
            "deliberative": "predecisional_advice", "commercial": "confidential_commercial"}

# (subject, records, who the personal details belong to if private, if officials, topic public-interest 0-3)
PRIVACY_TOPICS = [
    ("noise complaints about the Quarry Road events venue", "complaint forms and call logs", "the residents who complained", "the duty officers who logged the calls", 1),
    ("injuries reported at municipal swimming pools", "incident reports", "the injured visitors and witnesses", "the lifeguards and duty managers on shift", 2),
    ("dog-bite reports in the Fernhill district", "bite reports", "the dog owners and the people bitten", "the animal-control officers who attended", 1),
    ("the lead-pipe replacement grant scheme", "the grant register", "the households that received grants", "the programme officers who approved each grant", 2),
    ("public comments on the landfill expansion", "comment letters and the comment log", "members of the public who wrote in", "the planners who summarised each letter", 2),
    ("tenant complaints at the Larkspur Court housing block", "complaint files", "the tenants who complained", "the housing officers assigned to each file", 2),
    ("water shut-offs for unpaid bills", "shut-off lists and account notes", "the account holders", "the billing staff who authorised each shut-off", 2),
    ("the youth summer-jobs scheme", "the participant roster and timesheets", "the participants", "the scheme coordinators", 1),
    ("parking-permit appeals on the heritage streets", "appeal letters and decisions", "the residents who appealed", "the hearing officers who decided them", 1),
    ("complaints about missed bin collections on the east routes", "the complaints database extract", "the residents who called in", "the route supervisors named on each entry", 1),
    ("the emergency rental-assistance programme", "application files", "the applicants", "the caseworkers who assessed them", 2),
    ("injury claims at the road-maintenance depot", "claim forms and supervisors' reports", "the injured workers' families and witnesses", "the depot supervisors who filed the reports", 2),
    ("visitor incidents at the botanical gardens", "security incident logs", "the visitors involved", "the security staff on duty", 1),
    ("residents relocated for the Millrace flood works", "relocation agreements", "the relocated households", "the property officers who negotiated them", 3),
]
PRIV_SENS = {"names_only": "names only", "contact": "names, home addresses and phone numbers",
             "sensitive": "names together with income, health or family details"}
ENFORCEMENT_TOPICS = [
    ("the inspection of a poultry-processing plant on Route 9", "the inspection file: inspector notes, photographs and lab results", 2),
    ("illegal dumping at the old Saltmarsh rail yard", "the investigation file: site photos, witness statements and haulier records", 2),
    ("the licence case against a towing operator", "the case file: complaint letters, invoices and interview notes", 1),
    ("wage-theft complaints against a regional cleaning contractor", "the investigation file: payroll extracts and worker interviews", 2),
    ("fire-code violations at the Ember Street warehouse", "the enforcement file: inspection reports and correspondence", 3),
    ("price-gouging complaints after the January ice storm", "the inquiry file: complaints, receipts and seller responses", 2),
    ("air-permit exceedances at an asphalt plant", "the compliance file: monitoring data, notices and engineer memos", 3),
    ("a home-solar seller accused of misleading contracts", "the consumer-fraud file: contracts, recordings and complaint summaries", 2),
    ("pesticide drift complaints near the Ashgrove orchards", "the investigation file: sample results and applicator logs", 2),
    ("elevator-safety failures at the Driftwood Tower", "the enforcement file: inspection certificates and incident notes", 3),
    ("a food-labelling inquiry into a regional bakery chain", "the inquiry file: product samples, lab reports and emails", 1),
    ("a licensed child-care centre's staffing ratios", "the compliance file: visit reports and roster extracts", 3),
    ("unpermitted tree removal on the Valewood ridge", "the case file: aerial images, arborist notes and notices", 1),
    ("a scrap-metal dealer's record-keeping violations", "the case file: ledger copies and inspection notes", 1),
]
ENF_STAGE = {"fieldwork": "inspectors are still interviewing witnesses and collecting samples",
             "awaiting_results": "lab results and an expert report are still awaited",
             "charging_decision": "the file is with the agency's counsel for a decision on charges",
             "settlement_talks": "the agency and the operator are negotiating a settlement"}
DELIB_TOPICS = [
    ("the bus-depot relocation", "the staff memo recommending three candidate sites, with cost tables", 3),
    ("the proposed water-rate increase", "the draft rate study with the director's margin comments", 3),
    ("library branch closures", "briefing notes ranking branches for closure", 3),
    ("the downtown parking strategy", "the options paper and the consultants' draft recommendations", 2),
    ("the school-zone speed camera pilot", "the evaluation draft and staff recommendation", 2),
    ("changes to park opening hours", "the staff recommendation memo and attendance tables", 1),
    ("the stormwater fee redesign", "draft fee models and the finance team's advice", 2),
    ("merging two fire stations", "the deputy chief's options memo and response-time modelling", 3),
    ("the snow-clearing priority map", "draft route rankings and the operations manager's notes", 1),
    ("scoring for the arts micro-grants round", "the panel's draft scoring notes and recommendation", 1),
    ("the transit fare restructure", "fare-model drafts and the board briefing", 3),
    ("new procurement approval thresholds", "the policy team's draft and legal comments", 1),
    ("relocating the animal shelter", "the site-assessment draft and the manager's recommendation", 2),
    ("closing the Pinecrest recycling depot", "the cost-benefit draft and staff advice", 2),
]
COMM_TOPICS = [
    ("the parking-meter replacement contract", "the winning bid: unit prices, staffing plan and technical design", 2),
    ("the school-meals catering contract", "the caterer's proposal: menu costings, supplier list and staffing", 2),
    ("the fleet-fuel supply contract", "the supplier's price-adjustment formula and volume discounts", 1),
    ("the quarry expansion permit", "the operator's process diagrams and production volumes", 1),
    ("the economic-development grant to a battery maker", "the company's financial statements and hiring projections", 3),
    ("the permit-portal software licence", "the vendor's pricing schedule and system architecture", 1),
    ("the waste-hauling franchise renewal", "the hauler's cost breakdown and route data", 2),
    ("the solar-farm lease on county land", "the developer's financial model and yield estimates", 2),
    ("the ambulance billing contract", "the contractor's fee schedule and collection-rate data", 2),
    ("the bridge-inspection services contract", "the engineering firm's methodology and hourly rates", 1),
    ("the stadium concession agreement", "the concessionaire's revenue projections and pricing", 2),
    ("the tourism-marketing grant to a hotel group", "the applicant's occupancy data and business plan", 1),
    ("the street-lighting LED retrofit", "the contractor's product specifications and installation rates", 1),
]
TOPICS = {"privacy": PRIVACY_TOPICS, "enforcement": ENFORCEMENT_TOPICS, "deliberative": DELIB_TOPICS,
          "commercial": COMM_TOPICS}
REQ_TYPES = ["reporter", "resident", "neighbourhood association", "law firm", "graduate student", "advocacy group",
             "business", "retired engineer", "council candidate"]


def _exemption_text(a, kind, sunset):
    c = _cite(a, 0, KIND_LETTER[kind])
    return {
        "privacy": f"{c}: information about an identifiable private individual where release would be an unreasonable "
                   "invasion of privacy, weighed against the public interest in release. Names and work details of "
                   "public employees acting in their official role are not covered.",
        "enforcement": f"{c}: records compiled for an enforcement matter that is still open, where release could "
                       "reasonably be expected to interfere with it. This exemption ends when the matter closes.",
        "deliberative": f"{c}: advice, recommendations and draft analysis prepared for a decision-maker before a "
                        "decision. Not covered: factual material; advice the decision-maker publicly adopts as the "
                        f"reasons for the decision; any record more than {sunset} years old.",
        "commercial": f"{c}: commercial or financial information supplied in confidence by a third party (marked or "
                      "expressly stated confidential when submitted) whose release could cause it substantial "
                      "competitive harm. Information the agency itself prepares is not covered.",
    }[kind]


def _harm_text(a):
    return (f"{_cite(a, 2)}: withhold under {_cite(a, 0)} only where release would cause a specific, foreseeable "
            "harm to the protected interest; age and the passing of events count against harm. Release non-exempt parts.")


def _timing_text(a, days, ext):
    return (f"{_cite(a, -2)}: respond within {days} business days; one extension of up to {ext} business days if the "
            "request is over 400 pages or third parties must be consulted; instalments only with the requester's agreement.")


class RequestExemptionReview(Schema):
    name = "request_exemption_review"
    description = "A records officer's review of a request's responsive records: which exemption fits, how much to redact, and how to meet the deadline."
    surfaces = {"json": 0.5, "analyst_worksheet": 0.25, "request_and_file_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        a = _act(rng)
        kind = pick(rng, list(KIND_LETTER))
        topic = pick(rng, TOPICS[kind])
        # target: is the exemption's condition met, and is there a specific harm? (steering only; intended()
        # recomputes from the facts)
        if reg == "clear":
            want_cond = chance(rng, 0.6)
            want_harm = want_cond
        else:
            want_cond = chance(rng, 0.65)
            want_harm = want_cond and chance(rng, 0.55)
        sunset = pick(rng, [5, 8, 10, 12, 15, 20])
        kf = {"subjects": None, "sensitivity": None, "enf_status": None, "enf_stage": None, "closed_days_ago": None,
              "subject_has_copy": None, "decision_status": None, "record_age_years": None, "reopen_planned": None,
              "supplier_status": None, "already_public": None, "info_age_years": None, "rebid_soon": None}
        if kind == "privacy":
            kf["subjects"] = "private" if want_cond else pick(rng, ["officials", "published"])
            kf["sensitivity"] = (pick(rng, ["contact", "sensitive"]) if want_harm
                                 else pick(rng, ["names_only", "names_only", "contact"]))
        elif kind == "enforcement":
            kf["enf_status"] = "open" if want_cond else "closed"
            if want_cond:
                kf["enf_stage"] = pick(rng, list(ENF_STAGE))
                kf["subject_has_copy"] = not want_harm
            else:
                kf["closed_days_ago"] = (near(rng, 20, 15, lo=2, integer=True) if reg == "borderline"
                                         else pick(rng, [12, 45, 90, 200, 400]))
                kf["subject_has_copy"] = chance(rng, 0.4)
        elif kind == "deliberative":
            if want_cond:
                kf["decision_status"] = pick(rng, ["pending", "made"]) if want_harm else "made"
                kf["record_age_years"] = (sunset - pick(rng, [1, 1, 2]) if reg == "borderline"
                                          else pick(rng, [0, 1, 2, 3]))
                kf["reopen_planned"] = want_harm and kf["decision_status"] == "made"
                if kf["decision_status"] == "pending":
                    kf["record_age_years"] = min(kf["record_age_years"], 1)
            else:
                if chance(rng, 0.5):
                    kf["decision_status"], kf["record_age_years"] = "adopted", pick(rng, [0, 1, 2, 4])
                else:
                    kf["decision_status"] = "made"
                    kf["record_age_years"] = sunset + (pick(rng, [0, 1]) if reg == "borderline" else pick(rng, [1, 3, 6]))
                kf["reopen_planned"] = chance(rng, 0.25)
        else:
            if want_cond:
                kf["supplier_status"], kf["already_public"] = "marked_confidential", False
                kf["info_age_years"] = (pick(rng, [0, 1, 2, 3]) if want_harm
                                        else (near(rng, 4, 1, lo=4, integer=True) if reg == "borderline" else pick(rng, [5, 6, 8])))
                kf["rebid_soon"] = want_harm and chance(rng, 0.5)
            else:
                kf["supplier_status"] = pick(rng, ["unmarked", "agency_prepared", "marked_confidential"])
                kf["already_public"] = kf["supplier_status"] == "marked_confidential" or chance(rng, 0.3)
                kf["info_age_years"] = pick(rng, [0, 1, 2, 4, 6])
                kf["rebid_soon"] = chance(rng, 0.4)
        # how much of the batch carries the exempt material
        share = {"privacy": rng.uniform(0.08, 0.9), "enforcement": rng.uniform(0.55, 1.0),
                 "deliberative": rng.uniform(0.2, 0.85), "commercial": rng.uniform(0.1, 0.65)}[kind]
        days = pick(rng, [10, 15, 20])
        ext = pick(rng, [10, 15, 20])
        elapsed = rng.randint(1, days - 2)
        left = days - elapsed
        per_day = pick(rng, [60, 100, 150])
        consult = kind == "commercial" or (kind == "enforcement" and chance(rng, 0.3))
        load = pick_w(rng, {"fits": 3, "over_small": 4, "over_big": 2})
        if reg == "borderline":
            est = near(rng, left, 2, lo=1, integer=True)
        elif load == "fits":
            est = max(1, left - rng.randint(0, 4))
        elif load == "over_small":
            est = left + rng.randint(1, ext)
        else:
            est = left + ext + rng.randint(2, 25)
        pages = max(6, (est - (5 if consult else 0) - rng.randint(0, 1)) * per_day + rng.randint(-per_day // 3, per_day // 3))
        instalments_ok = chance(rng, 0.3)
        if reg == "conflicting":
            # the programme lead's view runs against the facts
            note_lean = "withhold_all" if not (want_cond and want_harm) else "release_all"
        else:
            note_lean = pick_w(rng, {"none": 3, "withhold_all": 1, "release_all": 1, "careful": 2})
        received = rand_date(rng)
        return {
            "regime": reg, "act": a, "agency": _agency(rng, a["jurisdiction"], AGENCY_FOR[topic[0]]), "kind": kind,
            "subject": topic[0], "records": topic[1],
            "private_who": topic[2] if kind == "privacy" else None,
            "official_who": topic[3] if kind == "privacy" else None,
            "topic_pi": topic[-1], "coverage": pick_w(rng, {"none": 3, "local": 3, "wide": 2}),
            "agency_failure_alleged": chance(rng, 0.35), "sunset_years": sunset, **kf,
            "exempt_share": round(share, 2), "pages": pages, "incidental_pii": chance(rng, 0.4),
            "statutory_days": days, "extension_days": ext, "days_elapsed": elapsed, "pages_per_day": per_day,
            "consultation_needed": consult, "review_estimate_days": est, "instalments_ok": instalments_ok,
            "distractor": pick(rng, [k for k in KIND_LETTER if k != kind]) if chance(rng, 0.45) else None,
            "note_lean": note_lean, "requester": person(rng), "requester_type": pick(rng, REQ_TYPES),
            "officer": person(rng), "request_id": ident(rng, "PRR", 6, "-"), "received": received,
        }

    # ------------------------------------------------------------------ derived (shared by render and intended)
    def _cond(self, f):
        k = f["kind"]
        if k == "privacy":
            return f["subjects"] == "private"
        if k == "enforcement":
            return f["enf_status"] == "open"
        if k == "deliberative":
            return f["decision_status"] != "adopted" and f["record_age_years"] < f["sunset_years"]
        return f["supplier_status"] == "marked_confidential" and not f["already_public"]

    def _pi(self, f):
        return _clamp(f["topic_pi"] + (f["coverage"] == "wide") + f["agency_failure_alleged"] - 1)

    def _harm(self, f):
        if not self._cond(f):
            return False
        k = f["kind"]
        if k == "privacy":
            return f["sensitivity"] != "names_only" or self._pi(f) <= 1
        if k == "enforcement":
            return not f["subject_has_copy"]
        if k == "deliberative":
            return f["decision_status"] == "pending" or bool(f["reopen_planned"])
        return f["info_age_years"] <= 3 or bool(f["rebid_soon"])

    def _flagged_pages(self, f):
        return max(1, round(f["pages"] * f["exempt_share"]))

    def _facts(self, f):
        """Rendered facts about the records' status (the exemption's conditions), as short sentences."""
        k, s = f["kind"], []
        if k == "privacy":
            who = f["private_who"] if f["subjects"] != "officials" else f["official_who"]
            s.append(f"Personal details in the records: {PRIV_SENS[f['sensitivity']]} of {who}.")
            if f["subjects"] == "published":
                s.append("The same names were printed in a list the agency published on its website last year.")
        elif k == "enforcement":
            if f["enf_status"] == "open":
                s.append(f"The matter is open: {ENF_STAGE[f['enf_stage']]}.")
            else:
                s.append(f"The matter closed {f['closed_days_ago']} days ago with a "
                         f"{pick_from(f, ['warning letter', 'consent order and fine', 'no-further-action notice'])}.")
            if f["subject_has_copy"]:
                s.append("The operator's lawyer has already been given a full copy of the file.")
        elif k == "deliberative":
            age = f["record_age_years"]
            s.append(f"The documents are {_yrs(age)} old." if age else "The documents are from this year.")
            s.append({"pending": "No decision has been made yet; it is due at a board meeting later this quarter.",
                      "made": "The decision was made; the final resolution does not mention this advice.",
                      "adopted": "The decision was made, and the final resolution states that it adopts the memo's "
                                 "analysis and recommendation as its reasons."}[f["decision_status"]])
            if f["reopen_planned"]:
                s.append("The board has said it will revisit the question next year.")
        else:
            s.append({"marked_confidential": "The submitter marked the material 'confidential - commercial' on every page when it was submitted.",
                      "unmarked": "The submitter sent it in without any confidentiality marking or request.",
                      "agency_prepared": "The figures were compiled by agency staff from site visits and invoices."}[f["supplier_status"]])
            if f["already_public"]:
                s.append("Most of the same figures appear in the award notice on the procurement portal.")
            s.append(f"The information is about {_yrs(f['info_age_years'])} old"
                     + ("; the contract is due to be re-tendered within a year." if f["rebid_soon"] else "."))
        return s

    def _note(self, f, rng):
        lean = {"withhold_all": pick(rng, ["The programme lead wants the whole batch withheld.",
                                          "Programme staff say none of this should go out.",
                                          "Our director's view: withhold everything, it is too sensitive."]),
                "release_all": pick(rng, ["The programme lead sees no problem releasing everything.",
                                         "Programme staff say just send it all, nothing in here matters."]),
                "careful": "Programme staff ask for a line-by-line review before anything goes out.",
                "none": ""}[f["note_lean"]]
        extra = []
        if f["incidental_pii"]:
            extra.append("Some pages carry staff mobile numbers and signatures.")
        if f["agency_failure_alleged"]:
            extra.append("The request follows public claims that the agency mishandled this.")
        return noisy(join_sentences([lean] + extra), rng)

    def _statute(self, f):
        a = f["act"]
        secs = [_exemption_text(a, f["kind"], f["sunset_years"])]
        if f["distractor"]:
            secs.append(_exemption_text(a, f["distractor"], f["sunset_years"]))
        secs += [_harm_text(a), _timing_text(a, f["statutory_days"], f["extension_days"])]
        return secs

    def render(self, f, rng, surface):
        a = f["act"]
        facts = self._facts(f)
        note = self._note(f, rng)
        cov = {"none": None, "local": "covered by local news", "wide": "widely reported"}[f["coverage"]]
        days_left = f["statutory_days"] - f["days_elapsed"]
        statute = self._statute(f)
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "received": fmt_date(f["received"], style="iso"),
                            "jurisdiction": a["jurisdiction"], "requester_type": f["requester_type"], "seeks": f"records about {f['subject']}",
                            "media_attention": cov,
                            "instalments_acceptable": True if f["instalments_ok"] else None},
                "records": {"description": f["records"], "pages": f["pages"],
                            "pages_with_flagged_material": self._flagged_pages(f), "status_facts": facts},
                "statute": {"act": a["act"], "provisions": statute},
                "processing": {"business_days_elapsed": f["days_elapsed"], "business_days_remaining": days_left,
                               "review_estimate_business_days": f["review_estimate_days"],
                               "third_party_consultation": f["consultation_needed"]},
                "staff_note": note or None,
            }))
        if surface == "analyst_worksheet":
            head = kv_lines([
                ("Request", f"{f['request_id']} ({f['agency']})"), ("Received", fmt_date(f["received"], rng)),
                ("Requester", f"{f['requester']} ({f['requester_type']})"), ("Subject", f["subject"]),
                ("Responsive records", f"{f['records']}, {f['pages']} pages"),
                ("Pages with flagged material", self._flagged_pages(f)),
                ("Media attention", cov or "none noted"),
                ("Clock", f"day {f['days_elapsed']} of {f['statutory_days']}"),
                ("Review estimate", f"{f['review_estimate_days']} business days"
                 + (" incl. third-party consultation" if f["consultation_needed"] else "")),
                ("Instalments", "requester agreed" if f["instalments_ok"] else "not discussed"),
            ], rng)
            return (f"RECORDS REQUEST WORKSHEET\n{head}\n\nStatus of the records:\n{bullets(facts, rng)}\n\n"
                    f"{a['act']}:\n" + "\n".join(statute) + (f"\n\nFile note: {note}" if note else ""))
        # request_and_file_note
        greet = pick(rng, ["Dear Records Officer,", "To the public records unit:", "Hello,"])
        ask = pick(rng, [f"Under the {a['act']} I request all {f['records'].split(':')[0]} relating to {f['subject']}.",
                         f"Please send me copies of the {f['records'].split(':')[0]} concerning {f['subject']}.",
                         f"I am asking for the records the agency holds on {f['subject']}, in particular the {f['records'].split(':')[0]}."])
        inst = " I am happy to receive the records in batches as they are ready." if f["instalments_ok"] else ""
        letter = "\n".join([f"Subject: Records request - {f['subject']}", f"From: {f['requester']}", "", greet, "",
                            noisy(ask + inst, rng, 0.15), "", "Regards,", f["requester"]])
        file_note = join_sentences([
            f"File note ({f['officer']}, {f['request_id']}): {f['pages']} pages found, about "
            f"{self._flagged_pages(f)} with material to consider.", " ".join(facts),
            f"Clock: {days_left} business days left of {f['statutory_days']}; review will take about "
            f"{f['review_estimate_days']} days" + (" including notice to the third party." if f["consultation_needed"] else "."),
            (f"The topic has been {cov}." if cov else ""), note])
        return f"{letter}\n\n---\n{file_note}\n\nProvisions:\n" + "\n".join(statute)

    def questions(self):
        return {
            "applicable_exemption": choice("Which exemption, if any, supports withholding material from these records?", {
                "confidential_commercial": "The exemption for business information supplied in confidence by a third party.",
                "no_exemption_applies": "None: no exemption's conditions are met, so nothing may be withheld on that basis.",
                "open_enforcement_matter": "The exemption for records of an enforcement matter that is still open.",
                "personal_privacy": "The exemption for personal information about private individuals.",
                "predecisional_advice": "The exemption for advice and drafts prepared before a decision.",
            }),
            "redaction_extent": score("How much of the responsive material should be withheld from the requester?", [
                "None: release everything as it stands.",
                "A little: redact a few names, numbers or passages and release the rest.",
                "A lot: withhold large portions but release the non-exempt parts.",
                "All of it: withhold the records in full.",
            ]),
            "public_interest_weight": score("How strong is the public interest in seeing these records?", [
                "Slight: they shed little light on how the agency works.",
                "Some: of interest to the people directly involved and a few others.",
                "Considerable: they bear on how public money or authority is being used.",
                "Compelling: they bear directly on public safety or a contested government failure.",
            ]),
            "deadline_handling": choice("How should the agency handle the response deadline for this request?", {
                "ask_requester_to_narrow": "Ask the requester to narrow the request, because even an extension would not be enough.",
                "extend_deadline": "Take the one statutory extension, because the request is voluminous or needs consultation.",
                "release_in_rolling_batches": "Release the records in instalments as they are reviewed, as the requester has agreed.",
                "respond_within_deadline": "Complete the review and respond within the original statutory period.",
            }),
            "specific_harm_foreseeable": noul(
                "Releasing the material the exemption covers would cause a specific, foreseeable harm.",
                yes="An exemption applies and release would concretely damage what it protects, such as an open case, "
                    "a pending decision, a business's current prices or a private person's sensitive details.",
                no="No exemption applies, or the harm is speculative because events have moved on, the material is "
                   "stale or already known, or the public interest outweighs a slight intrusion."),
        }

    def intended(self, f):
        cond, harm = self._cond(f), self._harm(f)
        if not cond or not harm:
            extent = 1 if (f["incidental_pii"] or (cond and f["kind"] == "privacy")) else 0
        elif f["exempt_share"] >= 0.75:
            extent = 3
        elif f["exempt_share"] >= 0.3:
            extent = 2
        else:
            extent = 1
        left = f["statutory_days"] - f["days_elapsed"]
        est = f["review_estimate_days"]
        if est <= left:
            timing = "respond_within_deadline"
        elif f["instalments_ok"]:
            timing = "release_in_rolling_batches"
        elif est <= left + f["extension_days"] and (f["pages"] > 400 or f["consultation_needed"]):
            timing = "extend_deadline"
        else:
            timing = "ask_requester_to_narrow"
        return {"applicable_exemption": KIND_KEY[f["kind"]] if cond else "no_exemption_applies",
                "redaction_extent": extent, "public_interest_weight": self._pi(f),
                "deadline_handling": timing, "specific_harm_foreseeable": "yes" if harm else "no"}


def pick_from(f, options):
    """A deterministic choice keyed on the request id (render-time flavour that must not consume the rng)."""
    return options[sum(map(ord, f["request_id"])) % len(options)]


# =========================================================================== schema 2
# fee categories: what each may be charged (search, review, duplication) and the free allowances
FEE_RULES = {"commercial": "search, review and copying fees, with no free allowance",
             "educational": "copying fees only, after the first 100 pages",
             "news_media": "copying fees only, after the first 100 pages",
             "other": "search fees after the first 2 hours and copying fees after the first 100 pages"}
# (claimed status, evidence of who the requester is, true category, dissemination shown?, own stated use or None)
REQUESTER_PROFILES = [
    ("news media", "staff reporter at {outlet}, which publishes daily in print and online", "news_media", True, None),
    ("news media", "freelance journalist with a written commission from {outlet} for a feature", "news_media", True, None),
    ("news media", "runs a personal blog with about 200 subscribers and no regular publishing schedule", "other", False, None),
    ("news media", "hosts a weekly local-affairs podcast with about 6,000 listeners", "news_media", True, None),
    ("news media", "writes a newsletter for a property-investment subscription service", "commercial", True,
     "Our paying subscribers rely on this kind of figure to time their purchases."),
    ("educational", "doctoral researcher at {univ}; the records are for a thesis chapter", "educational", True, None),
    ("educational", "lecturer at {univ} preparing a peer-reviewed article", "educational", True, None),
    ("educational", "researcher at {univ} doing a paid study for a private developer", "commercial", True,
     "The study is commissioned by a developer, who receives the full findings before any publication."),
    ("educational", "student at {univ} gathering material for a class presentation", "other", False, None),
    ("other", "resident of the affected street", "other", False, None),
    ("other", "volunteer coordinator of a residents' association with a 3,000-reader newsletter", "other", True, None),
    ("other", "policy analyst at a nonprofit watchdog that publishes reports and briefs reporters", "other", True, None),
    ("other", "paralegal at a law firm acting for a client in a business dispute with the agency", "commercial", False,
     "The records will support our client's damages claim against the agency."),
    ("other", "analyst at a data broker that resells permit data to contractors", "commercial", False,
     "The data will be loaded into our subscription database."),
    ("other", "business-development manager at a firm planning to bid on the next contract", "commercial", False,
     "We need the records to prepare our bid."),
    ("other", "retired civil engineer who writes letters to the local paper", "other", False, None),
]
FEE_SUBJECTS = [
    ("overtime payments in the road-maintenance division", 2), ("the agency's contract with an outside debt collector", 2),
    ("inspection results for the city's playground equipment", 2), ("spending on the new permit-portal software", 2),
    ("complaints about the regional bus contractor's reliability", 2), ("legal settlements paid by the agency since 2022", 2),
    ("the safety record of a privately owned marina the agency licenses", 1), ("historic building permits for one private house", 0),
    ("a private company's environmental filings for its own site", 1), ("water-quality testing at public beaches", 2),
    ("the agency's travel and hospitality expenses", 2), ("a list of all licensed food trucks and their addresses", 0),
    ("vacant-property registrations in the old mill district", 1), ("the consultants' fees on the stadium project", 2),
    ("emergency-shelter capacity reports during last winter", 2), ("commercial lease terms the agency granted a cafe operator", 1),
]
UNIVS = ["Northvale University", "Calder Institute of Technology", "Lindmoor College", "Eastholm University",
         "Ashcombe Polytechnic", "Brayfield University"]


class FeeWaiverRequest(Schema):
    name = "fee_waiver_request"
    description = "A records requester asks for a fee waiver or reduced fee category; the agency rules on category, waiver and advance payment."
    surfaces = {"json": 0.5, "requester_letter": 0.25, "fee_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        a = _act(rng)
        profile = pick(rng, REQUESTER_PROFILES, [3 if p[3] and p[2] != "commercial" else 1 for p in REQUESTER_PROFILES])
        subject, link = pick(rng, FEE_SUBJECTS)
        public_share = pick_w(rng, {"none": 4, "some": 3, "most": 2})
        commercial_use = None
        search_h = pick(rng, [1, 2, 3, 5, 8, 14])
        review_h = pick(rng, [0, 1, 3, 6, 10])
        pages = pick(rng, [30, 80, 150, 400, 900, 2000])
        rate = pick(rng, [25, 30, 40, 48])
        per_page = pick(rng, [0.10, 0.15, 0.25])
        threshold = pick(rng, [250, 300, 500, 750, 1000])
        unpaid = 0.0
        if reg == "clear":
            if chance(rng, 0.5):
                profile = pick(rng, [p for p in REQUESTER_PROFILES if p[3] and p[2] in ("news_media", "educational")])
                link, public_share, commercial_use = 2, "none", 0
            else:
                profile = pick(rng, [p for p in REQUESTER_PROFILES if p[2] == "commercial"])
                commercial_use = 3
        elif reg == "borderline":
            # the advance-payment line is nearly met
            pass
        else:
            # a claimed status the evidence contradicts, or a strong case with an old unpaid bill
            if chance(rng, 0.5):
                profile = pick(rng, [p for p in REQUESTER_PROFILES if p[0] != {"news_media": "news media"}.get(p[2], p[2])])
            else:
                unpaid = round(rng.uniform(40, 600), 2)
        if chance(rng, 0.15) and not unpaid:
            unpaid = round(rng.uniform(30, 300), 2)
        if commercial_use is None:
            commercial_use = {"commercial": 2 if "subscription service" in profile[1] or "paid study" in profile[1] else 3,
                              "news_media": pick(rng, [0, 0, 1]),
                              "educational": pick(rng, [0, 1]), "other": pick(rng, [0, 0, 1, 2])}[profile[2]]
        f = {"regime": reg, "act": a, "agency": _agency(rng, a["jurisdiction"], AGENCY_FOR[subject]),
             "claimed_category": profile[0], "requester_profile": profile[1].format(outlet=_outlet(rng), univ=pick(rng, UNIVS)),
             "true_category": profile[2], "dissemination_shown": profile[3] if chance(rng, 0.8) else None,
             "subject": subject, "operations_link": link, "already_public": public_share,
             "commercial_use": commercial_use, "search_hours": search_h, "review_hours": review_h, "pages": pages,
             "hourly_rate": rate, "per_page": per_page, "advance_threshold": threshold, "unpaid_prior_fees": unpaid,
             "requester": person(rng), "request_id": ident(rng, "FOI", 5, "-"), "received": rand_date(rng),
             "own_use": profile[4], "use_statement": None}
        if reg == "borderline":
            # move the copy volume so the chargeable estimate lands near the advance-payment threshold
            est_no_copy = self._estimate(f, pages_override=0)
            room = threshold - est_no_copy
            free = 0 if f["true_category"] == "commercial" else 100
            if room > 0:
                f["pages"] = int(max(1, round(near(rng, room, room * 0.2) / per_page)) + free)
        f["use_statement"] = self._use_statement(f, rng)
        return f

    PRIVATE_USE = ["I want to understand the decision that affected my street.",
                   "This is for my own information and to raise at a residents' meeting.",
                   "I am trying to understand how the programme works.",
                   "I need this for a class assignment and my own interest."]
    USE = {0: ["The records will inform a public report; nobody is paying for this work.",
               "I intend to publish what I find, free to read, and I have no financial stake in the matter.",
               "This is for an article and a public talk; I have no business interest in it."],
           1: ["I will publish a summary; my employer may also use it in a paid training course.",
               "Mostly for a public write-up, though a client asked me about the same topic."],
           2: ["The findings will go into a report for paying clients, and a short public summary.",
               "A client in the industry has asked for this analysis; I will publish an extract."],
           3: ["We will use the data to identify sales prospects.", "The records support our client's bid preparation.",
               "The data will be loaded into our subscription database."]}

    def _use_statement(self, f, rng):
        if f["own_use"]:
            return f["own_use"]
        if f["commercial_use"] == 0 and f["dissemination_shown"] is False:
            return pick(rng, self.PRIVATE_USE)
        return pick(rng, self.USE[f["commercial_use"]])

    def _estimate(self, f, pages_override=None):
        """Chargeable fees under the requester's TRUE category (before any waiver)."""
        cat = f["true_category"]
        pages = f["pages"] if pages_override is None else pages_override
        if cat == "commercial":
            return round((f["search_hours"] + f["review_hours"]) * f["hourly_rate"] + pages * f["per_page"], 2)
        copy = max(0, pages - 100) * f["per_page"]
        if cat in ("educational", "news_media"):
            return round(copy, 2)
        return round(max(0, f["search_hours"] - 2) * f["hourly_rate"] + copy, 2)

    def _contribution(self, f):
        c = f["operations_link"] + (1 if f["dissemination_shown"] else 0)
        c -= {"none": 0, "some": 0, "most": 2}[f["already_public"]]
        return _clamp(c)

    def _rules(self, f):
        a = f["act"]
        return [f"{_cite(a, 0)} Fees by requester category: commercial use - {FEE_RULES['commercial']}; educational or "
                f"scientific institutions and news media - {FEE_RULES['news_media']}; all others - {FEE_RULES['other']}. "
                f"Search and review: {money(f['hourly_rate'])}/hour; copies: {money(f['per_page'], style='sym')}/page.",
                f"{_cite(a, 1)} Fees are waived or reduced where disclosure is likely to contribute significantly to public "
                "understanding of government operations and is not primarily in the requester's commercial interest. "
                "Commercial use means a use that furthers the trade or profit interests of the requester or of the "
                "person on whose behalf the request is made.",
                f"{_cite(a, 2)} Advance payment may be required where chargeable fees are likely to exceed "
                f"{money(f['advance_threshold'])}, or where the requester has unpaid fees from an earlier request."]

    def render(self, f, rng, surface):
        a = f["act"]
        rules = self._rules(f)
        diss = {True: "describes where and how the results will be published", False: "says nothing about publishing",
                None: None}[f["dissemination_shown"]]
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "agency": f["agency"], "subject": f["subject"],
                            "jurisdiction": a["jurisdiction"], "fee_category_claimed": f["claimed_category"], "waiver_requested": True},
                "requester": {"description": f["requester_profile"], "stated_use": f["use_statement"],
                              "dissemination": diss,
                              "unpaid_fees_from_earlier_requests": f["unpaid_prior_fees"] or None},
                "workload_estimate": {"search_hours": f["search_hours"], "review_hours": f["review_hours"],
                                      "pages": f["pages"]},
                "records_context": {"already_public": {"none": "none of it", "some": "some of it, in council papers",
                                                       "most": "most of it, on the open-data portal"}[f["already_public"]]},
                "rules": {"act": a["act"], "provisions": rules},
            }))
        if surface == "requester_letter":
            body = [f"Subject: Fee waiver - request {f['request_id']}", "", "Dear Records Officer,", "",
                    noisy(join_sentences([
                        f"I am requesting records on {f['subject']} and ask that all fees be waived.",
                        f"I should be treated as {f['claimed_category']} for fee purposes: I am a {f['requester_profile']}.",
                        f["use_statement"],
                        {True: f"The results will be published and shared with {pick(rng, ['local outlets', 'our readers', 'council members and the public'])}.",
                         False: "", None: ""}[f["dissemination_shown"]],
                        "These records show how public money is being spent." if f["operations_link"] == 2 else ""]), rng),
                    "", "Sincerely,", f["requester"]]
            ctx = kv_lines([("Estimated search", f"{f['search_hours']} h"), ("Estimated review", f"{f['review_hours']} h"),
                            ("Estimated pages", f["pages"]),
                            ("Already public", {"none": "no", "some": "partly (council papers)", "most": "mostly (open-data portal)"}[f["already_public"]]),
                            ("Unpaid earlier fees", money(f["unpaid_prior_fees"], rng) if f["unpaid_prior_fees"] else "none")], rng)
            return "\n".join(body) + f"\n\n[Records unit notes]\n{ctx}\n\n" + "\n".join(rules)
        rows = [("Request", f["request_id"]), ("Requester", f["requester"]), ("Profile", f["requester_profile"]),
                ("Category claimed", f["claimed_category"]), ("Stated use", f["use_statement"]),
                ("Dissemination", show(diss, rng)), ("Subject", f["subject"]),
                ("Work estimate", f"search {f['search_hours']} h, review {f['review_hours']} h, {f['pages']} pages"),
                ("Already public", f["already_public"]),
                ("Prior unpaid fees", money(f["unpaid_prior_fees"], rng) if f["unpaid_prior_fees"] else "none")]
        return "FEE DETERMINATION MEMO\n" + kv_lines(rows, rng) + "\n\nApplicable rules:\n" + bullets(rules, rng)

    def questions(self):
        return {
            "waiver_ruling": choice("How should the agency rule on the fee-waiver request?", {
                "deny_waiver": "Refuse the waiver and charge the fees the requester's category allows.",
                "grant_full_waiver": "Waive all fees, because the public-interest test is met and the use is not commercial.",
                "grant_partial_waiver": "Reduce the fees, for example for the records not already public or to copying costs only.",
                "request_more_information": "Ask the requester to explain how the information will reach the public before ruling.",
            }),
            "public_understanding_contribution": score("How much would releasing these records add to public understanding of government operations?", [
                "Little or nothing: they concern private matters or are already public.",
                "Some: they touch on agency activity but add little that is new.",
                "A meaningful amount: they reveal how a programme or contract is run.",
                "A great deal: they expose spending or performance that the public cannot see otherwise.",
            ]),
            "commercial_interest_level": score("How far is the request driven by the requester's own commercial interest?", [
                "Not at all: no financial stake in the records.",
                "Incidentally: some side benefit to the requester's work.",
                "Significantly: paying clients or business plans are part of the purpose.",
                "Primarily: the records serve a business purpose such as sales, bids or resale.",
            ]),
            "advance_payment_allowed": noul(
                "The agency may require the requester to pay in advance before it processes this request.",
                yes="The fees chargeable in the requester's real category exceed the advance-payment threshold, or "
                    "earlier fees are unpaid.",
                no="The chargeable fees stay under the threshold and nothing is owed from earlier requests."),
            "dissemination_ability_shown": noul(
                "The requester has shown a credible way of getting the information to a broad public audience."),
        }

    def intended(self, f):
        contrib = self._contribution(f)
        comm = f["commercial_use"]
        if f["true_category"] == "commercial" and comm >= 3:
            ruling = "deny_waiver"
        elif contrib >= 2 and comm <= 1 and f["dissemination_shown"]:
            ruling = "grant_full_waiver" if f["already_public"] == "none" else "grant_partial_waiver"
        elif contrib >= 2 and comm == 2:
            ruling = "grant_partial_waiver"
        elif f["dissemination_shown"] is None and contrib >= 1 and comm <= 1:
            ruling = "request_more_information"
        elif contrib >= 1 and comm <= 1 and f["dissemination_shown"] and f["already_public"] != "most":
            ruling = "grant_partial_waiver"
        else:
            ruling = "deny_waiver"
        adv = self._estimate(f) > f["advance_threshold"] or f["unpaid_prior_fees"] > 0
        return {"waiver_ruling": ruling, "public_understanding_contribution": contrib,
                "commercial_interest_level": comm, "advance_payment_allowed": "yes" if adv else "no",
                "dissemination_ability_shown": "yes" if f["dissemination_shown"] else "no"}


# =========================================================================== schema 3
# (topic, whether it concerns government activity, typical safety angle or None)
EXPEDITE_TOPICS = [
    ("the agency's inspections of a chemical storage depot after a fire nearby", True, "residents near the depot may be exposed to stored chemicals"),
    ("contamination test results for a neighbourhood's drinking-water wells", True, "families are drinking the well water now"),
    ("the structural survey of the Ironbark footbridge", True, "the bridge is still open to pedestrians"),
    ("the transit authority's plan to cut night bus routes", True, None),
    ("council members' meetings with the stadium developer", True, None),
    ("overtime claims by senior staff at the water board", True, None),
    ("the contract awarded for the new jail food service", True, None),
    ("the agency's handling of a heat-wave cooling-centre shortage", True, "another heat wave is forecast next week"),
    ("the zoning file for a neighbour's garage extension", False, None),
    ("historic land records for a family property", False, None),
    ("a business's own licence history, for its sale", False, None),
    ("the agency's records of a car crash the requester was in", False, None),
    ("the evacuation planning for the river levee upgrade", True, "the levee works start before the spring floods"),
    ("the procurement of body-worn cameras for parking wardens", True, None),
]
EXPEDITE_REQUESTERS = {
    "reporter": "a reporter at {outlet}", "podcaster": "the host of a local-affairs podcast",
    "advocacy_group": "the director of a public-interest advocacy group", "resident": "a local resident",
    "litigant": "a party to a pending licensing hearing", "attorney": "an attorney representing a client",
    "student": "a university student",
}
DISSEMINATORS = {"reporter", "podcaster", "advocacy_group"}


class ExpeditedProcessingRequest(Schema):
    name = "expedited_processing_request"
    description = "A requester asks to jump the records queue; the agency rules on whether the Act's expedited-processing grounds are met."
    surfaces = {"json": 0.5, "requester_email": 0.25, "intake_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        a = _act(rng)
        topic, gov, safety = pick(rng, EXPEDITE_TOPICS)
        rtype = pick_w(rng, {"reporter": 3, "podcaster": 2, "advocacy_group": 2, "resident": 2, "litigant": 2,
                             "attorney": 2, "student": 1})
        queue_days = pick(rng, [35, 60, 90, 140, 200])
        coverage = pick(rng, [0, 1, 3, 8, 20])
        certified = chance(rng, 0.7)
        specificity = pick_w(rng, {"vague": 3, "some_detail": 4, "documented": 3})
        broad = chance(rng, 0.4)
        days = None
        if reg == "clear":
            branch = pick(rng, ["safety", "proceeding", "no_ground"])
            if branch == "safety":
                topic, gov, safety = pick(rng, [t for t in EXPEDITE_TOPICS if t[2]])
                rtype = pick(rng, ["reporter", "advocacy_group", "resident"])
                certified, specificity, days = True, "documented", pick(rng, [4, 9, 12])
            elif branch == "proceeding":
                # a party whose hearing comes long before the queue would clear
                topic, gov, safety = pick(rng, EXPEDITE_TOPICS)
                rtype, certified, specificity = pick(rng, ["litigant", "attorney"]), True, pick(rng, ["some_detail", "documented"])
                days = rng.randint(5, max(6, queue_days // 3))
            else:
                topic, gov, safety = pick(rng, [t for t in EXPEDITE_TOPICS if not t[1]])
                rtype, coverage, specificity = pick(rng, ["resident", "student"]), 0, "vague"
        elif reg == "conflicting":
            if chance(rng, 0.5):
                # a disseminator on live government business, without the certification the Act requires
                topic, gov, safety = pick(rng, [t for t in EXPEDITE_TOPICS if t[1]])
                rtype, coverage, certified = pick(rng, sorted(DISSEMINATORS)), pick(rng, [3, 8, 20]), False
            else:
                # a disseminator, certified and insistent, but the records concern a private matter
                topic, gov, safety = pick(rng, [t for t in EXPEDITE_TOPICS if not t[1]])
                rtype, certified, coverage = pick(rng, sorted(DISSEMINATORS)), True, 0
        safety_claimed = safety is not None and chance(rng, 0.75)
        # the dated event follows the requester and the topic
        if rtype in ("litigant", "attorney") and (chance(rng, 0.75) or (reg == "clear" and days)):
            event = "hearing"
        elif safety_claimed and chance(rng, 0.7):
            event = "season"
        elif gov:
            event = pick_w(rng, {"vote": 3, None: 2})
        else:
            event = pick_w(rng, {"personal_deadline": 2, None: 2})
        if event is not None:
            days = days or pick(rng, [4, 9, 20, 45, 120, 300])
            if reg == "borderline":
                days = near(rng, queue_days, queue_days * 0.3, lo=3, integer=True)
        else:
            days = None
        return {"regime": reg, "act": a, "agency": _agency(rng, a["jurisdiction"], AGENCY_FOR[topic]), "topic": topic,
                "government_activity": gov, "safety_angle": safety if safety_claimed else None,
                "requester_type": rtype, "requester": person(rng), "outlet": _outlet(rng),
                "queue_days": queue_days, "event": event, "days_to_event": days,
                "news_items_30d": coverage, "certified": certified, "specificity": specificity,
                "request_is_broad": broad, "decide_within_days": pick(rng, [5, 10]),
                "request_id": ident(rng, "XR", 5, "-"), "received": rand_date(rng)}

    def _event_text(self, f):
        if f["event"] is None:
            return None
        d = f["days_to_event"]
        return {"vote": f"the council votes on this in {d} days", "hearing": f"my hearing is in {d} days",
                "season": f"the risk period starts in {d} days",
                "personal_deadline": f"my own deadline (a property sale and paperwork) falls in {d} days"}[f["event"]]

    def _ground(self, f):
        if f["safety_angle"] and f["specificity"] != "vague":
            return "imminent_safety_threat"
        if f["requester_type"] in DISSEMINATORS and f["government_activity"] and f["news_items_30d"] >= 1:
            return "urgency_to_inform_public"
        if f["requester_type"] in ("litigant", "attorney") and f["event"] == "hearing" and f["days_to_event"] is not None \
                and f["days_to_event"] < f["queue_days"]:
            return "due_process_rights"
        return "no_ground_shown"

    def _time(self, f):
        if f["safety_angle"] and f["days_to_event"] is not None and f["days_to_event"] <= 15:
            return 3
        if f["event"] is None:
            return 0
        if f["event"] == "personal_deadline":
            return 1 if f["days_to_event"] < f["queue_days"] else 0
        d, q = f["days_to_event"], f["queue_days"]
        if d >= q:
            return 1 if d < q * 1.5 else 0
        return 3 if d <= 15 else 2

    def _substantiation(self, f):
        s = {"vague": 0, "some_detail": 1, "documented": 2}[f["specificity"]] + (1 if f["certified"] else 0)
        return _clamp(s)

    def _rules(self, f):
        a = f["act"]
        return [f"{_cite(a, 0)} Expedited processing is granted where the requester shows a compelling need: (i) delay "
                "could reasonably be expected to pose an imminent threat to someone's life or physical safety; (ii) "
                "the requester's main activity is informing the public and there is urgency to inform it about actual "
                "or alleged government activity; or (iii) delay would cost a party substantial rights in a pending "
                "proceeding.",
                f"{_cite(a, 1)} The requester must certify the statement of need as true and correct. The agency decides "
                f"within {f['decide_within_days']} days and may expedite part of a request."]

    def render(self, f, rng, surface):
        a = f["act"]
        who = EXPEDITE_REQUESTERS[f["requester_type"]].format(outlet=f["outlet"])
        ev = self._event_text(f)
        need = join_sentences([
            f"I need these records quickly: {ev}." if ev else "I would like these records as soon as possible.",
            f"There is a real danger here: {f['safety_angle']}." if f["safety_angle"] else "",
            {"vague": "", "some_detail": "I have attached the meeting agenda.",
             "documented": "I attach the notice, the agenda and the relevant correspondence."}[f["specificity"]],
            "I certify that this statement is true and correct to the best of my knowledge." if f["certified"] else ""])
        cov = f"{f['news_items_30d']} news items in the last 30 days" if f["news_items_30d"] else "no recent news coverage"
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "agency": f["agency"], "records_sought": f["topic"],
                            "scope": "broad (several years, all departments)" if f["request_is_broad"] else "narrow",
                            "received": fmt_date(f["received"], style="iso")},
                "requester": {"described_as": who},
                "expedite_claim": {"statement": need, "certified": f["certified"],
                                   "supporting_documents": f["specificity"].replace("_", " ")},
                "context": {"public_attention": cov, "normal_queue_days": f["queue_days"]},
                "rules": {"act": a["act"], "provisions": self._rules(f)},
            }))
        if surface == "requester_email":
            body = [f"Subject: Request for expedited processing - {f['request_id']}", f"From: {f['requester']}", "",
                    "To the records office,", "",
                    noisy(join_sentences([f"I am {who}, and I have requested records about {f['topic']}.",
                                          "The request covers several years and every department." if f["request_is_broad"] else "",
                                          need]), rng), "", f["requester"]]
            ctx = kv_lines([("Normal queue", f"about {f['queue_days']} days"), ("Public attention", cov),
                            ("Decision due within", f"{f['decide_within_days']} days")], rng)
            return "\n".join(body) + "\n\n-- records office context --\n" + ctx + "\n\n" + "\n".join(self._rules(f))
        lines = [f"{fmt_date(f['received'], rng)}  {f['request_id']}  received from {f['requester']} ({who})",
                 f"  records sought: {f['topic']}" + (" [broad scope]" if f["request_is_broad"] else ""),
                 f"  expedite claim: {need}",
                 f"  certification: {'signed' if f['certified'] else 'missing'}",
                 f"  queue estimate: {f['queue_days']} days; media: {cov}"]
        return f"EXPEDITE INTAKE LOG ({a['act']})\n" + "\n".join(lines) + "\n\nRules:\n" + "\n".join(self._rules(f))

    def questions(self):
        return {
            "expedite_ruling": choice("How should the agency rule on the request for expedited processing?", {
                "deny_expedite": "Refuse expedition and process the request in the normal queue.",
                "grant_expedite": "Move the whole request to the front of the queue.",
                "grant_in_part": "Expedite only the part of the request tied to the urgent need; queue the rest.",
                "request_certification": "Ask the requester for the certified statement of need before deciding.",
            }),
            "qualifying_ground": choice("Which statutory ground, if any, does the requester's need fit?", {
                "due_process_rights": "Delay would cost a party substantial rights in a pending proceeding.",
                "imminent_safety_threat": "Delay could threaten someone's life or physical safety.",
                "no_ground_shown": "None of the grounds is shown; the need is convenience or personal interest.",
                "urgency_to_inform_public": "A person whose main work is informing the public needs to report on current government activity.",
            }),
            "delay_cost": score("How much would waiting for normal processing undercut the purpose of the request?", [
                "Not at all: the records would be just as useful after the normal queue.",
                "Slightly: the normal queue finishes around the time the records are needed.",
                "Seriously: the queue runs past the vote, hearing or deadline the requester names.",
                "Critically: the date is days away or people could be hurt in the meantime.",
            ]),
            "need_substantiation": score("How well has the requester substantiated the claimed need?", [
                "Not at all: bare assertion, uncertified.",
                "Thinly: some detail but no certification or documents.",
                "Adequately: a certified statement with some supporting detail.",
                "Thoroughly: certified, specific and backed by documents.",
            ]),
            "concerns_current_government_activity": noul(
                "The urgency concerns actual or alleged government activity that is a matter of current public attention.",
                yes="The records are about what a public body is doing or deciding, and that is being discussed publicly now.",
                no="The records concern a private matter, old history, or something nobody outside the requester is following."),
        }

    def intended(self, f):
        g = self._ground(f)
        sub = self._substantiation(f)
        if g == "no_ground_shown":
            ruling = "deny_expedite"
        elif not f["certified"]:
            ruling = "request_certification"
        elif f["request_is_broad"]:
            ruling = "grant_in_part"
        else:
            ruling = "grant_expedite"
        current = f["government_activity"] and f["news_items_30d"] >= 1
        return {"expedite_ruling": ruling, "qualifying_ground": g, "delay_cost": self._time(f),
                "need_substantiation": sub, "concerns_current_government_activity": "yes" if current else "no"}


# =========================================================================== schema 4
# (item, sensitivity 0-3 for competitive harm, is it pricing the Act may make public)
OBJ_ITEMS = [
    ("total contract price", 1, "total"), ("unit rates and hourly prices", 2, "unit"),
    ("the pricing methodology and margin assumptions", 3, None), ("the technical approach and process design", 3, None),
    ("the staffing plan and labour mix", 2, None), ("audited financial statements", 2, None),
    ("client reference list", 1, None), ("insurance certificates", 0, None), ("the project schedule", 0, None),
    ("subcontractor quotes", 2, None), ("the quality-control manual", 2, None), ("equipment specifications", 1, None),
    ("the maintenance cost model", 3, None), ("volume-discount tiers", 2, "unit"),
]
CONTRACTS = ["street-lighting maintenance", "school transport", "data-centre hosting", "road-salt supply",
             "public-library e-book platform", "transit-fare system", "park landscaping", "emergency-radio upgrade",
             "wastewater sludge hauling", "court-interpreter services", "fleet telematics", "municipal-pool operations",
             "election-ballot printing", "tree-planting programme"]
OBJ_REQUESTERS = ["a competing bidder", "a reporter at {outlet}", "a taxpayers' watchdog group", "a trade union",
                  "a council member's office", "a university procurement researcher"]


class SubmitterObjectionReview(Schema):
    name = "submitter_objection_review"
    description = "A contractor objects to the release of information it submitted; the agency weighs the objection against the commercial-information exemption."
    surfaces = {"json": 0.5, "objection_letter": 0.25, "release_review_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        a = _act(rng)
        n = pick(rng, [1, 2, 3, 4])
        items = rng.sample(OBJ_ITEMS, n)
        marked = chance(rng, 0.82)
        quality = pick_w(rng, {"boilerplate": 3, "general_letter": 3, "item_by_item": 3, "affidavit": 2})
        age = pick(rng, [0, 1, 1, 2, 3, 4, 6])
        retender = pick_w(rng, {"within_year": 3, "in_three_years": 3, "not_planned": 2})
        public_items = [it[0] for it in items if chance(rng, 0.3)]
        price_rule = pick(rng, ["total_only", "total_and_unit"])
        if reg == "clear":
            if chance(rng, 0.5):
                items = rng.sample([it for it in OBJ_ITEMS if it[1] >= 2 and it[2] is None], pick(rng, [1, 2]))
                marked, quality, age, retender, public_items = True, "affidavit", pick(rng, [0, 1]), "within_year", []
            else:
                items = rng.sample([it for it in OBJ_ITEMS if it[1] <= 1], pick(rng, [2, 3]))
                quality, age = "boilerplate", pick(rng, [5, 7])
        elif reg == "borderline":
            age = pick(rng, [2, 3, 4])
            quality = pick(rng, ["general_letter", "item_by_item"])
        else:
            # strong items mixed with ones the Act makes public, or a detailed objection over stale data
            if chance(rng, 0.5):
                strong = [it for it in OBJ_ITEMS if it[1] == 3]
                pricey = [it for it in OBJ_ITEMS if it[2] is not None]
                items = [pick(rng, strong), pick(rng, pricey)]
                quality = pick(rng, ["item_by_item", "affidavit"])
            else:
                quality, age, retender = "affidavit", pick(rng, [5, 7, 9]), "not_planned"
        return {"regime": reg, "act": a, "contract": pick(rng, CONTRACTS),
                "agency": _agency(rng, a["jurisdiction"], pick(rng, ["Office of Procurement", "Purchasing Division",
                                                                     "Contracts and Procurement Office"])),
                "submitter": _firm(rng), "items": [it[0] for it in items], "item_sensitivity": [it[1] for it in items],
                "item_price_kind": [it[2] for it in items], "marked_confidential": marked,
                "objection_quality": quality, "info_age_years": age, "retender": retender,
                "public_items": [x for x in public_items if x in [it[0] for it in items]],
                "price_rule": price_rule, "competitors": pick(rng, [2, 3, 5, 9, 15]),
                "requester": pick(rng, OBJ_REQUESTERS).format(outlet=_outlet(rng)),
                "objection_days": pick(rng, [7, 10, 14]), "signatory": person(rng),
                "request_id": ident(rng, "PRR", 6, "-"), "date": rand_date(rng)}

    def _price_items(self, f):
        hit = []
        for name, kind in zip(f["items"], f["item_price_kind"]):
            if kind == "total" or (kind == "unit" and f["price_rule"] == "total_and_unit"):
                hit.append(name)
        return hit

    def _protectable(self, f):
        """Items that could still be withheld: not price-rule items, not already public, sensitivity >= 2."""
        price, pub = self._price_items(f), f["public_items"]
        return [n for n, s in zip(f["items"], f["item_sensitivity"]) if n not in price and n not in pub and s >= 2]

    def _harm(self, f):
        prot = [s for n, s in zip(f["items"], f["item_sensitivity"]) if n not in self._price_items(f) and n not in f["public_items"]]
        if not prot:
            return 0
        h = max(prot)
        if f["info_age_years"] >= 5:
            h -= 2
        elif f["info_age_years"] >= 3:
            h -= 1
        if f["retender"] == "within_year":
            h += 1
        if f["competitors"] <= 3:
            h += 0
        return _clamp(h)

    def _rules(self, f):
        a = f["act"]
        price = ("the total amount payable under a public contract" if f["price_rule"] == "total_only"
                 else "the total amount payable and the unit prices of a public contract")
        return [f"Under {_cite(a, 0, 'd')}, a supplier's trade or financial data is exempt if the supplier flagged it as "
                "confidential at the time it was handed over and disclosure would be likely to damage the supplier's "
                "position against its rivals. Data already lawfully in the public domain is never exempt.",
                f"{_cite(a, 3)} Despite {_cite(a, 0, 'd')}, {price} is always public.",
                f"{_cite(a, 4)} A supplier objecting to release must say, item by item, why release would cause it "
                f"competitive harm, within {f['objection_days']} days of notice; generalised claims carry little weight."]

    def render(self, f, rng, surface):
        a = f["act"]
        q = {"boilerplate": "states that the entire proposal is proprietary and confidential",
             "general_letter": "explains in general terms that competitors could undercut it",
             "item_by_item": "explains for each item how a competitor would use it",
             "affidavit": "gives item-by-item reasons in a sworn statement from its finance director, with examples of past lost bids"}[f["objection_quality"]]
        retender = {"within_year": "the contract is re-tendered within a year",
                    "in_three_years": "the next tender is about three years away", "not_planned": "no re-tender is planned"}[f["retender"]]
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "agency": f["agency"], "requester": f["requester"],
                            "jurisdiction": a["jurisdiction"], "records": f"{f['submitter']}'s proposal for the {f['contract']} contract"},
                "objection": {"submitter": f["submitter"], "items_objected_to": f["items"],
                              "marked_confidential_at_submission": f["marked_confidential"], "reasoning": q},
                "facts": {"information_age_years": f["info_age_years"], "re_tender": retender,
                          "competitors_in_market": f["competitors"],
                          "items_already_public": f["public_items"] or None},
                "rules": {"act": a["act"], "provisions": self._rules(f)},
            }))
        if surface == "objection_letter":
            pubs = f"; note that {', '.join(f['public_items'])} also appeared in the award notice" if f["public_items"] else ""
            first_person = {
                "boilerplate": "Our entire proposal is proprietary and confidential, and none of it should be released.",
                "general_letter": "Competitors could use this material to undercut us on future bids.",
                "item_by_item": "For each item, the attached schedule explains how a competitor would use it against us.",
                "affidavit": "Our finance director's sworn statement (attached) explains item by item how each would be "
                             "used against us, with examples of bids we lost after earlier disclosures."}[f["objection_quality"]]
            body = [f"Subject: Objection to release - {f['request_id']}", f"From: {f['signatory']}, {f['submitter']}", "",
                    "Dear Records Officer,", "",
                    noisy(join_sentences([
                        f"We object to the release of the following from our {f['contract']} proposal: {', '.join(f['items'])}.",
                        "Each page was marked confidential when we submitted it." if f["marked_confidential"] else "",
                        first_person,
                        pick(rng, ["Release would put us at a serious disadvantage.", "These are trade secrets in all but name.",
                                   "Our competitors would gain an unfair insight."])]), rng),
                    "", "Yours faithfully,", f["signatory"]]
            tail = kv_lines([("Requester", f["requester"]), ("Age of the information", _yrs(f["info_age_years"])),
                             ("Re-tender", retender), ("Competitors in this market", f["competitors"]),
                             ("Already public", ", ".join(f["public_items"]) or "nothing identified")], rng)
            return "\n".join(body) + f"\n\n[agency review notes]\n{tail}{pubs}\n\n" + "\n".join(self._rules(f))
        rows = [("Request", f"{f['request_id']} from {f['requester']}"), ("Submitter", f["submitter"]),
                ("Contract", f["contract"]), ("Items objected to", "; ".join(f["items"])),
                ("Marked confidential", "yes" if f["marked_confidential"] else "no"), ("Objection reasoning", q),
                ("Information age", _yrs(f["info_age_years"])), ("Re-tender", retender),
                ("Already public", "; ".join(f["public_items"]) or "none found")]
        return "RELEASE REVIEW NOTE - third-party objection\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._rules(f))

    def questions(self):
        return {
            "objection_ruling": choice("How should the agency resolve the supplier's objection?", {
                "release_over_objection": "Release everything requested; the objection does not establish the exemption.",
                "release_with_targeted_redactions": "Release the records but withhold the specific items the exemption genuinely protects.",
                "seek_further_substantiation": "Give the supplier a short deadline to justify each item before deciding.",
                "uphold_objection": "Withhold all the items the supplier objects to.",
            }),
            "competitive_harm_likelihood": score("How likely is release of the contested items to cause the supplier real competitive harm?", [
                "Negligible: the items are routine, stale or already known.",
                "Low: a competitor would learn little it could use.",
                "Moderate: some items would help a rival price or design against the supplier.",
                "High: current, sensitive information that rivals could exploit at the next tender.",
            ]),
            "objection_specificity": score("How well does the supplier's objection meet the Act's demand for item-by-item reasons?", [
                "Not at all: a blanket claim that everything is confidential.",
                "Weakly: general concerns about competitors.",
                "Adequately: reasons given for each item.",
                "Fully: itemised, sworn and supported by evidence.",
            ]),
            "objected_items_already_public": noul(
                "At least one item the supplier wants withheld is already lawfully public.",
                yes="Some contested item has already appeared in a public notice or an earlier release.",
                no="Nothing the supplier objects to has been made public."),
            "price_rule_overrides_part": noul(
                "Part of what the supplier wants withheld is contract pricing that the Act makes public regardless."),
        }

    def intended(self, f):
        harm = self._harm(f)
        spec = {"boilerplate": 0, "general_letter": 1, "item_by_item": 2, "affidavit": 3}[f["objection_quality"]]
        prot = self._protectable(f)
        if not f["marked_confidential"] or not prot or harm <= 1:
            ruling = "release_over_objection"
        elif spec <= 1:
            ruling = "seek_further_substantiation"
        elif len(prot) < len(f["items"]):
            ruling = "release_with_targeted_redactions"
        else:
            ruling = "uphold_objection"
        return {"objection_ruling": ruling, "competitive_harm_likelihood": harm, "objection_specificity": spec,
                "objected_items_already_public": "yes" if f["public_items"] else "no",
                "price_rule_overrides_part": "yes" if self._price_items(f) else "no"}


# =========================================================================== schema 5
# (subject, custodians, agency, exemptions the agency could plausibly have claimed)
APPEAL_SUBJECTS = [
    ("emails about the closure of the Westbrook clinic", ["the director", "the deputy director", "the communications lead"],
     "Department of Health Services", ["deliberative", "privacy"]),
    ("records of complaints about the new waste contractor", ["the contracts manager", "the customer-care team lead"],
     "Solid Waste Authority", ["privacy", "commercial"]),
    ("correspondence with the developer of the riverside towers", ["the planning director", "two senior planners"],
     "Planning and Zoning Office", ["deliberative", "commercial"]),
    ("the investigation file on the collapsed retaining wall", ["the chief building inspector", "the case inspector"],
     "Building Inspections Division", ["enforcement", "privacy"]),
    ("records of overtime approvals at the transit depot", ["the depot manager", "the payroll supervisor"],
     "Transit Authority", ["privacy", "deliberative"]),
    ("drafts and comments on the road-safety audit", ["the traffic engineer", "the audit consultant's liaison"],
     "Department of Transportation", ["deliberative"]),
    ("texts and emails about the cancelled music festival permit", ["the events manager", "the deputy mayor's adviser"],
     "Special Events Office", ["deliberative", "commercial"]),
    ("records about the sale of the Elmstead depot site", ["the property manager", "the finance director"],
     "Office of Public Lands", ["commercial", "deliberative"]),
    ("the agency's records on the algae outbreak at the reservoir", ["the water-quality lead", "the operations manager"],
     "Water and Sewer Board", ["enforcement", "deliberative"]),
    ("communications about the hiring freeze exemption list", ["the HR director", "the budget officer"],
     "Human Resources Department", ["deliberative", "privacy"]),
    ("the inspection file on the Copperline food warehouse", ["the lead inspector", "the district manager"],
     "Department of Health Services", ["enforcement", "commercial"]),
    ("records of the tender for the harbour dredging contract", ["the procurement officer", "the harbour engineer"],
     "Port Authority", ["commercial", "deliberative"]),
]
SEARCH_LOCATIONS = ["the shared drive", "the email accounts of the named staff", "the records-management system",
                    "paper files at the regional office", "text messages on agency phones", "the project-management tool"]
APPEAL_EXEMPTIONS = {"enforcement": "an open enforcement matter", "deliberative": "pre-decision advice",
                     "privacy": "personal privacy", "commercial": "confidential commercial information"}


class DenialAppealReview(Schema):
    name = "denial_appeal_review"
    description = "An administrative appeal of a partial denial: the appeals officer reviews the search and the exemption claimed."
    surfaces = {"json": 0.5, "appeal_letter": 0.25, "appeals_officer_brief": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        a = _act(rng)
        subject, custodians, agency_name, exemptions = pick(rng, APPEAL_SUBJECTS)
        window = pick(rng, [20, 30, 45, 60, 90])
        filed = near(rng, window, window * 0.3, lo=2, integer=True)
        rights_notice = chance(rng, 0.85)
        n_loc = len(SEARCH_LOCATIONS)
        searched = rng.sample(SEARCH_LOCATIONS, pick(rng, [1, 2, 3, 4]))
        custodians_searched = rng.randint(0, len(custodians))
        missed_evidence = pick_w(rng, {None: 4, "referenced_memo": 2, "calendar_entry": 1, "news_quote": 1})
        exemption = pick(rng, exemptions)
        lapsed = pick_w(rng, {"still_holds": 4, "partly_lapsed": 2, "fully_lapsed": 2})
        segregated = chance(rng, 0.45)
        contest = pick_w(rng, {"both": 4, "withholding": 3, "search": 3})
        if reg == "clear":
            branch = pick(rng, ["sound_response", "bad_search", "plainly_late"])
            if branch == "sound_response":
                filed, rights_notice = rng.randint(2, max(3, window // 2)), True
                searched, custodians_searched, missed_evidence = rng.sample(SEARCH_LOCATIONS, 4), len(custodians), None
                lapsed, segregated = "still_holds", True
            elif branch == "plainly_late":
                filed, rights_notice = window + rng.randint(10, 60), True
            else:
                filed = rng.randint(2, max(3, window - 3))
                searched, custodians_searched, missed_evidence = rng.sample(SEARCH_LOCATIONS, 1), 0, "referenced_memo"
                contest = "search"
        elif reg == "conflicting":
            twist = pick(rng, ["late_no_notice", "lapsed", "unsegregated"])
            if twist == "late_no_notice":
                # a strong appeal, filed late, but the denial never mentioned appeal rights
                filed, rights_notice = window + rng.randint(3, 40), False
            elif twist == "unsegregated":
                # a thorough search and an exemption that still holds, but whole pages withheld
                searched, custodians_searched, missed_evidence = rng.sample(SEARCH_LOCATIONS, 4), len(custodians), None
                lapsed, segregated, contest = "still_holds", False, pick(rng, ["withholding", "both"])
                filed = rng.randint(2, max(3, window - 2))
            else:
                # a thorough search, but the exemption has since lapsed
                searched, custodians_searched, missed_evidence = rng.sample(SEARCH_LOCATIONS, 4), len(custodians), None
                lapsed, contest = "fully_lapsed", pick(rng, ["withholding", "both"])
        lapse_fact = {"enforcement": {"still_holds": "the enforcement case remains open",
                                      "partly_lapsed": "one of the two linked cases has since closed",
                                      "fully_lapsed": "the enforcement case closed after the denial was issued"},
                      "deliberative": {"still_holds": "the decision is still pending",
                                       "partly_lapsed": "the decision was made, but a second phase is still under discussion",
                                       "fully_lapsed": "the decision was made and announced after the denial"},
                      "privacy": {"still_holds": "the withheld pages concern private residents",
                                  "partly_lapsed": "some withheld names belong to staff acting in their jobs",
                                  "fully_lapsed": "the people concerned have since published their own accounts and consented to release"},
                      "commercial": {"still_holds": "the supplier's pricing is current and the contract is being re-tendered",
                                     "partly_lapsed": "part of the pricing was later published in the award notice",
                                     "fully_lapsed": "the contract expired and all of its prices were published"}}[exemption][lapsed]
        return {"regime": reg, "act": a, "agency": _agency(rng, a["jurisdiction"], agency_name), "subject": subject,
                "custodians": custodians, "custodians_searched": custodians_searched, "locations_searched": searched,
                "missed_evidence": missed_evidence, "exemption": exemption, "lapse_state": lapsed,
                "lapse_fact": lapse_fact, "segregated": segregated, "contest": contest,
                "appeal_window_days": window, "filed_after_days": filed, "rights_notice_given": rights_notice,
                "pages_released": pick(rng, [0, 12, 40, 150]), "pages_withheld": pick(rng, [6, 30, 85, 300]),
                "requester": person(rng), "appeal_id": ident(rng, "APL", 4, "-"), "denial_date": rand_date(rng)}

    def _search_score(self, f):
        s = len(f["locations_searched"]) - 1 + (f["custodians_searched"] == len(f["custodians"]))
        if f["missed_evidence"]:
            s -= 2
        if f["custodians_searched"] == 0:
            s -= 1
        return _clamp(s)

    def _exemption_score(self, f):
        s = {"still_holds": 3, "partly_lapsed": 1, "fully_lapsed": 0}[f["lapse_state"]]
        if not f["segregated"] and s > 0:
            s -= 1
        return s

    def _timely(self, f):
        return f["filed_after_days"] <= f["appeal_window_days"] or not f["rights_notice_given"]

    def _rules(self, f):
        a = f["act"]
        return [f"{_cite(a, 6)} A requester may appeal a refusal within {f['appeal_window_days']} days of the decision. "
                "A late appeal must be accepted if the decision did not tell the requester of the right to appeal.",
                f"{_cite(a, 7)} The appeals officer reviews whether the search was reasonably designed to find all "
                "responsive records and whether each exemption applies at the time of the appeal, and may order a new "
                "search or the release of records.",
                f"{_cite(a, 2)} Non-exempt parts of a record must be released once exempt parts are removed."]

    def render(self, f, rng, surface):
        a = f["act"]
        miss = {None: None,
                "referenced_memo": "a released email refers to a memo that was neither released nor listed as withheld",
                "calendar_entry": "a released calendar shows a meeting on the topic, but no notes or emails from it were found",
                "news_quote": "a news article quotes an internal email that the search did not turn up"}[f["missed_evidence"]]
        search = (f"searched {', '.join(f['locations_searched'])}; custodians searched: "
                  f"{f['custodians_searched']} of {len(f['custodians'])} named ({', '.join(f['custodians'])})")
        withheld = (f"{f['pages_withheld']} pages withheld under the exemption for {APPEAL_EXEMPTIONS[f['exemption']]}"
                    + ("; partly redacted pages released" if f["segregated"] else "; withheld in full, no partial release"))
        contest = {"both": "the search and the withholding", "withholding": "the withholding only",
                   "search": "the adequacy of the search only"}[f["contest"]]
        if surface == "json":
            return dumps(prune({
                "appeal": {"id": f["appeal_id"], "agency": f["agency"], "jurisdiction": a["jurisdiction"],
                           "records_sought": f["subject"],
                           "challenges": contest, "filed_days_after_decision": f["filed_after_days"]},
                "original_decision": {"date": fmt_date(f["denial_date"], style="iso"),
                                      "pages_released": f["pages_released"], "withholding": withheld,
                                      "told_requester_of_appeal_right": f["rights_notice_given"]},
                "search": {"description": search, "requester_evidence_of_missed_records": miss},
                "since_the_decision": f["lapse_fact"],
                "rules": {"act": a["act"], "provisions": self._rules(f)},
            }))
        if surface == "appeal_letter":
            sent = f["denial_date"] + timedelta(days=f["filed_after_days"])
            body = [f"Subject: Appeal - {f['subject']}", f"Date: {fmt_date(sent, rng)}", "", "Dear Appeals Officer,", "",
                    noisy(join_sentences([
                        f"I appeal the decision of {fmt_date(f['denial_date'], rng)} on my request for {f['subject']}.",
                        f"I challenge {contest}.", f"I was sent {f['pages_released']} pages; {withheld}.",
                        (f"I believe records are missing: {miss}." if miss else ""),
                        (f"I also understand that {f['lapse_fact']}." if f["contest"] != "search" else ""),
                        ("" if f["rights_notice_given"] else "The decision letter did not say I could appeal; I only learned of this recently.")]), rng),
                    "", f["requester"]]
            note = kv_lines([("Agency search record", search), ("Appeal window", f"{f['appeal_window_days']} days"),
                             ("Filed", f"{f['filed_after_days']} days after decision")], rng)
            return "\n".join(body) + f"\n\n[appeals office]\n{note}\n\n" + "\n".join(self._rules(f))
        rows = [("Appeal", f["appeal_id"]), ("Agency", f["agency"]), ("Records sought", f["subject"]),
                ("Challenge", contest), ("Decision date", fmt_date(f["denial_date"], rng)),
                ("Filed", f"{f['filed_after_days']} days after the decision (window {f['appeal_window_days']})"),
                ("Appeal rights stated in decision", "yes" if f["rights_notice_given"] else "no"),
                ("Search", search), ("Requester's evidence of missing records", miss or "none offered"),
                ("Withholding", withheld), ("Change since decision", f["lapse_fact"])]
        return "APPEALS OFFICER BRIEF\n" + kv_lines(rows, rng) + "\n\n" + "\n".join(self._rules(f))

    def questions(self):
        return {
            "appeal_ruling": choice("How should the appeals officer decide this appeal?", {
                "affirm": "Uphold the original response in full.",
                "dismiss_as_untimely": "Dismiss the appeal without reaching the merits because it was filed too late.",
                "release_in_part": "Order release of some withheld material, such as segregable parts or material no longer exempt.",
                "remand_for_new_search": "Send the request back for a new, wider search.",
                "reverse_and_release": "Overturn the withholding and order the withheld records released.",
            }),
            "main_defect": choice("What is the main defect, if any, in the agency's original response?", {
                "exemption_no_longer_fits": "The exemption claimed does not, or no longer does, cover the withheld material.",
                "no_defect": "There is no real defect; the search and the withholding both hold up.",
                "search_too_narrow": "The search missed places or people likely to hold responsive records.",
                "segregation_failure": "Whole records were withheld when exempt parts could have been cut out and the rest released.",
            }),
            "search_adequacy": score("How adequate was the agency's search for responsive records?", [
                "Inadequate: obvious sources were never searched or records are known to be missing.",
                "Weak: a narrow search with gaps the requester has pointed to.",
                "Reasonable: the main sources were covered.",
                "Thorough: every likely location and named custodian was searched.",
            ]),
            "exemption_support": score("How well does the exemption claimed justify the withholding as things stand now?", [
                "Not at all: its conditions are no longer met.",
                "Partly: it covers some of the material at most.",
                "Mostly: it covers the material, though the handling could be finer-grained.",
                "Fully: it still covers everything withheld.",
            ]),
            "appeal_filed_in_time": noul(
                "The appeal must be considered on its merits rather than rejected as late.",
                yes="It was filed within the appeal window, or the decision failed to state the right to appeal.",
                no="It was filed after the window and the decision did state the right to appeal."),
        }

    def intended(self, f):
        timely = self._timely(f)
        srch = self._search_score(f)
        ex = self._exemption_score(f)
        challenged_search = f["contest"] in ("both", "search")
        challenged_hold = f["contest"] in ("both", "withholding")
        if challenged_search and srch <= 1:
            defect = "search_too_narrow"
        elif challenged_hold and f["lapse_state"] != "still_holds":
            defect = "exemption_no_longer_fits"
        elif challenged_hold and not f["segregated"]:
            defect = "segregation_failure"
        else:
            defect = "no_defect"
        if not timely:
            ruling = "dismiss_as_untimely"
        elif defect == "search_too_narrow":
            ruling = "remand_for_new_search"
        elif defect == "exemption_no_longer_fits":
            ruling = "reverse_and_release" if f["lapse_state"] == "fully_lapsed" else "release_in_part"
        elif defect == "segregation_failure":
            ruling = "release_in_part"
        else:
            ruling = "affirm"
        return {"appeal_ruling": ruling, "main_defect": defect, "search_adequacy": srch, "exemption_support": ex,
                "appeal_filed_in_time": "yes" if timely else "no"}


# =========================================================================== schema 6
ROLES = {"senior": [("the deputy director of public works", "Department of Public Works"),
                    ("the chief financial officer", "Office of the Treasurer"),
                    ("the head of the licensing division", "Business Licensing Office"),
                    ("the operations director", "Transit Authority"), ("the deputy fire marshal", "Fire Marshal's Office"),
                    ("the director of housing services", "Housing Authority")],
         "supervisor": [("a road-depot supervisor", "Department of Public Works"),
                        ("an inspections team leader", "Building Inspections Division"),
                        ("a parks maintenance supervisor", "Parks and Recreation Department"),
                        ("a procurement team lead", "Office of Procurement"),
                        ("a bus-garage shift manager", "Transit Authority"), ("a branch manager", "Library Board")],
         "frontline": [("a water-meter technician", "Water and Sewer Board"),
                       ("a parks groundskeeper", "Parks and Recreation Department"),
                       ("a permit clerk", "Planning and Zoning Office"), ("a bus operator", "Transit Authority"),
                       ("a library assistant", "Library Board"), ("a building inspector", "Building Inspections Division"),
                       ("a housing caseworker", "Housing Authority")]}
ALLEGATIONS = [
    ("personal use of an agency vehicle", 1), ("steering a contract to a relative's firm", 3),
    ("falsified overtime sheets", 2), ("approving permits without the required inspection", 3),
    ("skipping mandatory safety checks on a work site", 3), ("rude conduct toward members of the public", 0),
    ("using an agency credit card for personal purchases", 2), ("accepting gifts from a regulated business", 3),
    ("repeated lateness and missed shifts", 0), ("disclosing confidential bid information", 2),
    ("misusing the agency's email for a side business", 1), ("ignoring complaints about a hazardous playground", 2),
    ("running personal errands during paid shifts", 0), ("losing a box of applicants' files", 1),
    ("leaving a work site unsecured overnight", 1), ("taking home surplus agency equipment", 1),
    ("waiving fees for friends without authority", 2), ("posting about a pending case on social media", 0),
]
FINDINGS = {"sustained": "sustained", "not_sustained": "not sustained (insufficient evidence)",
            "unfounded": "unfounded", "pending": "still under investigation"}
DISCIPLINE = {"termination": "dismissal", "suspension": "unpaid suspension", "reprimand": "written reprimand",
              "counselling": "informal counselling", "none": "no action"}


class EmployeeConductRecords(Schema):
    name = "employee_conduct_records"
    description = "A request for an internal investigation file about a public employee's conduct, weighed under the personnel-records rule."
    surfaces = {"json": 0.5, "records_review_memo": 0.25, "request_with_file_summary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        a = _act(rng)
        level = pick_w(rng, {"senior": 3, "supervisor": 3, "frontline": 4})
        allegation, gravity = pick(rng, ALLEGATIONS)
        finding = pick_w(rng, {"sustained": 4, "not_sustained": 2, "unfounded": 2, "pending": 2})
        discipline = (pick_w(rng, {"termination": 2, "suspension": 3, "reprimand": 3, "counselling": 2})
                      if finding == "sustained" else "none")
        named_in_news = chance(rng, 0.3)
        third_party = pick_w(rng, {"none": 4, "witness_names": 3, "complainant_details": 2, "medical_note": 2})
        threshold = pick(rng, ["suspension", "reprimand"])  # minimum discipline at which identities are released
        if reg == "clear":
            if chance(rng, 0.5):
                level, finding, discipline, gravity = "senior", "sustained", pick(rng, ["termination", "suspension"]), 3
                allegation = pick(rng, [x for x, g in ALLEGATIONS if g == 3])
            else:
                level, finding, discipline, named_in_news = "frontline", pick(rng, ["unfounded", "not_sustained"]), "none", False
                allegation, gravity = pick(rng, [x for x in ALLEGATIONS if x[1] <= 1])
        elif reg == "borderline":
            # sustained with discipline right at the release threshold, or an unproven allegation
            if chance(rng, 0.6):
                finding = "sustained"
                discipline = pick(rng, ["reprimand", "suspension", "counselling"])
            else:
                finding, discipline = pick(rng, ["not_sustained", "unfounded"]), "none"
        else:
            # high public interest but an unproven allegation, or a minor matter already in the news
            if chance(rng, 0.5):
                level, finding, discipline = "senior", pick(rng, ["not_sustained", "pending"]), "none"
                allegation, gravity = pick(rng, [x for x in ALLEGATIONS if x[1] >= 2])
            else:
                named_in_news, finding = True, pick(rng, ["sustained", "unfounded"])
                discipline = "reprimand" if finding == "sustained" else "none"
        role, agency_name = pick(rng, ROLES[level])
        return {"regime": reg, "act": a, "agency": _agency(rng, a["jurisdiction"], agency_name), "employee_level": level,
                "role": role, "allegation": allegation, "gravity": gravity, "finding": finding,
                "discipline": discipline, "release_threshold": threshold, "named_in_news": named_in_news,
                "third_party_content": third_party, "report_pages": pick(rng, [8, 15, 30, 60]),
                "requester": person(rng), "requester_type": pick(rng, ["reporter", "resident", "union steward",
                                                                      "council member", "watchdog group"]),
                "request_id": ident(rng, "PRR", 6, "-"), "year": rng.randint(2022, 2026)}

    def _identity_withheld(self, f):
        if f["finding"] != "sustained":
            return not f["named_in_news"] or f["finding"] == "pending"
        order = ["none", "counselling", "reprimand", "suspension", "termination"]
        return order.index(f["discipline"]) < order.index(f["release_threshold"]) and not f["named_in_news"]

    def _privacy(self, f):
        p = {"sustained": 0 if f["discipline"] in ("termination", "suspension") else 1,
             "pending": 2, "not_sustained": 2, "unfounded": 3}[f["finding"]]
        p += {"senior": -1, "supervisor": 0, "frontline": 1}[f["employee_level"]]
        if f["third_party_content"] == "medical_note":
            p += 1
        if f["named_in_news"]:
            p -= 1
        return _clamp(p)

    def _accountability(self, f):
        a = f["gravity"] + {"senior": 1, "supervisor": 0, "frontline": -1}[f["employee_level"]]
        if f["finding"] in ("unfounded",):
            a -= 1
        return _clamp(a)

    def _rules(self, f):
        a = f["act"]
        thr = {"suspension": "suspension or dismissal", "reprimand": "a written reprimand or more serious discipline"}[f["release_threshold"]]
        return [f"{_cite(a, 0, 'e')} Personnel records are exempt, except that the final outcome of a complaint against a "
                f"public employee is public, including the employee's name and role, where the complaint was sustained and "
                f"the discipline imposed was {thr}.",
                f"{_cite(a, 0, 'a')} Personal information about private individuals (complainants, witnesses, members of "
                "the public) and any medical information must be redacted.",
                f"{_cite(a, 0, 'b')} Records of an investigation that is still open are exempt until it closes."]

    def render(self, f, rng, surface):
        a = f["act"]
        tp = {"none": "no details about other people", "witness_names": "names and statements of two co-worker witnesses",
              "complainant_details": "the complainant's name, address and account", "medical_note": "a medical note the employee submitted"}[f["third_party_content"]]
        disc = "not yet decided" if f["finding"] == "pending" else DISCIPLINE[f["discipline"]]
        news = "the employee was named in local news coverage of the allegation" if f["named_in_news"] else None
        if surface == "json":
            return dumps(prune({
                "request": {"id": f["request_id"], "agency": f["agency"], "requester_type": f["requester_type"],
                            "jurisdiction": a["jurisdiction"], "seeks": f"the {f['year']} investigation file about {f['role']}"},
                "file": {"allegation": f["allegation"], "finding": FINDINGS[f["finding"]], "discipline_imposed": disc,
                         "report_pages": f["report_pages"], "also_contains": tp},
                "context": {"public_reporting": news},
                "rules": {"act": a["act"], "provisions": self._rules(f)},
            }))
        if surface == "records_review_memo":
            para = noisy(join_sentences([
                f"Request {f['request_id']} ({f['requester_type']}) seeks the {f['year']} investigation file about {f['role']}.",
                f"The allegation was {f['allegation']}; the finding: {FINDINGS[f['finding']]}.",
                f"Discipline imposed: {disc}." if f["finding"] == "sustained" else "",
                f"The {f['report_pages']}-page file also contains {tp}.", (news.capitalize() + ".") if news else "",
                pick(rng, ["HR asks that the file not be released.", "The employee's union has asked to be notified.", ""])]), rng)
            return f"MEMO - personnel records request\n\n{para}\n\nProvisions:\n" + "\n".join(self._rules(f))
        req = noisy(f"Please provide the investigation report and outcome concerning {f['role']} and the allegation of "
                    f"{f['allegation']}.", rng, 0.15)
        summ = kv_lines([("Finding", FINDINGS[f["finding"]]), ("Discipline", disc), ("Pages", f["report_pages"]),
                         ("Other content", tp), ("Public reporting", news or "none found")], rng)
        return (f"From: {f['requester']} ({f['requester_type']})\nRe: records request {f['request_id']}\n\n{req}\n\n"
                f"[File summary prepared by records unit]\n{summ}\n\n" + "\n".join(self._rules(f)))

    def questions(self):
        return {
            "conduct_release_scope": choice("What should the agency release in response to this request?", {
                "release_outcome_summary_only": "Release only a statement of the allegation's outcome, without the file itself.",
                "release_with_identity": "Release the report and outcome with the employee's name and role, redacting other people's details.",
                "release_with_identity_redacted": "Release the report and outcome with the employee's identity removed.",
                "withhold_in_full": "Withhold the file entirely for now.",
            }),
            "privacy_intrusion": score("How intrusive would releasing this file be for the employee?", [
                "Minimal: it records proven official misconduct by someone in a public role.",
                "Modest: the matter is work-related and partly known already.",
                "Significant: it would publicise an unproven or minor matter tied to a named person.",
                "Severe: it would expose an unfounded allegation or private health details.",
            ]),
            "accountability_interest": score("How strong is the public's interest in this file as a check on the agency?", [
                "Weak: a minor workplace matter with no bearing on public services.",
                "Moderate: it shows how the agency deals with everyday conduct issues.",
                "Strong: it concerns public money, safety or a position of trust.",
                "Very strong: it concerns serious abuse of office by a senior official.",
            ]),
            "employee_identity_protected": noul(
                "The employee's name must be kept out of whatever is released.",
                yes="The complaint was not sustained, or the discipline falls below the level the Act makes public, and "
                    "the name is not already public.",
                no="The Act makes the name public for this outcome, or the employee was already publicly named."),
            "third_party_redaction_needed": noul(
                "The file holds information about people other than the employee that must be redacted before any release."),
        }

    def intended(self, f):
        ident_withheld = self._identity_withheld(f)
        acc = self._accountability(f)
        if f["finding"] == "pending":
            scope = "withhold_in_full"
        elif not ident_withheld:
            scope = "release_with_identity"
        elif f["finding"] == "sustained":
            scope = "release_with_identity_redacted"
        elif acc >= 2:
            scope = "release_outcome_summary_only"
        else:
            scope = "withhold_in_full" if f["finding"] == "unfounded" else "release_outcome_summary_only"
        return {"conduct_release_scope": scope, "privacy_intrusion": self._privacy(f), "accountability_interest": acc,
                "employee_identity_protected": "yes" if ident_withheld else "no",
                "third_party_redaction_needed": "no" if f["third_party_content"] == "none" else "yes"}


WORKFLOW = Workflow(
    name="foia_request_exemption_review",
    title="Public-records request review",
    description="How a public body should answer a records-access request under a written records statute: exemptions, "
                "redaction, fees, expedition, third-party objections and appeals.",
    domain="public administration / records access",
    schemas=[RequestExemptionReview(), FeeWaiverRequest(), ExpeditedProcessingRequest(), SubmitterObjectionReview(),
             DenialAppealReview(), EmployeeConductRecords()],
    notes="Every statute, jurisdiction, agency, outlet and person is fictional; each state quotes the provisions it is "
          "judged against. Requesters are judged on use and need, never identity; no protected attribute is a factor. "
          "No helpdesk threads, no security alerts, no bills.",
)
