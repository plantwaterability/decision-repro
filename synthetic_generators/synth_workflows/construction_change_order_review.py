"""Construction change-order review: whether a contractor is entitled to a change, whether its price is reasonable,
and how much time it earns, judged against the contract's own written change, notice and time clauses.

Every state quotes the governing clause with case-specific numbers (notice periods, markup caps, quantity-variation
bands, response periods), so the reader has to apply the quoted terms. Projects, owners, contractors, contracts and
people are fictional. This is entitlement and pricing of CONTRACT MODIFICATIONS before they are signed: no
progress billing, invoices, pay applications or payment release, and nothing is matched against a purchase order
or a delivery.

Each schema exposes `intended(f)`: the answer a careful reviewer would reach from the factors (choice key, "yes" /
"no", or a score level index). It is never shown to the model; it exists so the spread of answers across options
can be checked before labelling.
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

# =========================================================================== shared pools
STEMS = ["Alder", "Basalt", "Cobalt", "Driftwood", "Ember", "Fernhill", "Ironbark", "Juniper", "Lakeshore", "Oakridge",
         "Pinecrest", "Quarry", "Redfern", "Saltmarsh", "Upland", "Valewood", "Westbrook", "Yarrow", "Ashgrove",
         "Bluestem", "Copperline", "Dunmore", "Elmstead", "Harrowgate", "Kingsfold", "Larkspur", "Moraine", "Nettlebed",
         "Orrin", "Rushford", "Stannard", "Wexcombe", "Whitlow", "Brackett", "Calloway", "Denholm", "Fairbairn",
         "Glenmoor", "Hatherley", "Lindell", "Quillhaven", "Pellbridge", "Marrowdale", "Corvina", "Halden",
         "Tamberlane", "Wrexley", "Cinder Bay", "Sorrel Bend", "Ridgemont"]
# (project name, kind, owner forms): the owner always fits the facility, and the stem is shared, as in practice
FACILITIES = [
    ("{s} Middle School addition", "building", ["{s} Unified School District"]),
    ("{s} Elementary School modernization", "building", ["{s} Unified School District"]),
    ("{s} Wastewater Treatment Plant upgrade", "civil", ["{s} Water Authority", "City of {s}"]),
    ("{s} County Courthouse HVAC replacement", "building", ["{s} County"]),
    ("{s} Public Library renovation", "building", ["City of {s}"]),
    ("{s} Fire Station No. {n}", "building", ["City of {s}"]),
    ("{s} Transit Center", "building", ["{s} Transit Authority"]),
    ("{s} Regional Airport apron rehabilitation", "civil", ["{s} Airport Authority"]),
    ("Water Main Replacement, Phase {n}", "civil", ["{s} Water Authority", "City of {s}"]),
    ("{s} Senior Housing", "building", ["{s} Housing Authority"]),
    ("{s} Community Recreation Center", "building", ["City of {s}", "{s} Parks and Recreation District"]),
    ("{s} Police Headquarters", "building", ["City of {s}"]),
    ("Stormwater Pump Station No. {n}", "civil", ["City of {s}", "{s} County"]),
    ("{s} Public Works operations building", "building", ["City of {s}"]),
    ("{s} Health Clinic expansion", "building", ["{s} County"]),
    ("{s} Reservoir outlet works", "civil", ["{s} Water Authority"]),
    ("Sewer Interceptor rehabilitation, Contract {n}", "civil", ["City of {s}"]),
    ("Main Street reconstruction, {s}", "civil", ["City of {s}"]),
    ("{s} Parking Structure", "building", ["City of {s}"]),
    ("{s} Aquatic Center", "building", ["{s} Parks and Recreation District"]),
    ("{s} Science Building renovation", "building", ["{s} Community College District"]),
    ("Route {n}0 widening, {s} segment", "civil", ["{s} County"]),
    ("{s} Creek bridge replacement", "civil", ["{s} County"]),
    ("{s} Bus Maintenance Facility", "building", ["{s} Transit Authority"]),
    ("{s} Water Treatment Plant filter rehabilitation", "civil", ["{s} Water Authority"]),
]
AREAS = {"building": ["north wing footings", "elevator pit", "gymnasium foundations", "main entrance canopy",
                      "east retaining wall", "loading dock", "stair tower 2", "service yard", "mechanical room addition"],
         "civil": ["pump station wet well", "Line A sewer trench", "storm outfall", "clarifier No. 2", "east abutment",
                   "cut at station 3+400", "intake structure", "new valve vault", "east approach embankment"]}
GC_SUFFIX = ["Constructors", "Builders", "Contracting", "Civil", "Construction Co.", "General Contractors",
             "Construction Group", "Builders & Engineers", "Heavy Civil", "Building Company"]
SUB_TRADES = ["electrical", "mechanical", "plumbing", "concrete", "earthwork", "steel erection", "drywall",
              "roofing", "fire protection", "paving", "masonry", "site utilities"]
CLAUSE_REFS = ["GC 7.3", "General Conditions Art. 10", "Section 00 72 00, 12.1", "GC 4.3 and 7.2", "Supplementary "
               "Conditions SC-9", "Article 15", "GC 11.2"]


def _project(rng, kind=None):
    """(project name, owner, general contractor, kind); `kind` restricts to 'building' or 'civil' projects."""
    name, k, owners = pick(rng, [x for x in FACILITIES if kind is None or x[1] == kind])
    stem = pick(rng, STEMS)
    gc = f"{pick(rng, STEMS)} {pick(rng, GC_SUFFIX)}"
    return name.format(s=stem, n=rng.randint(2, 9)), pick(rng, owners).format(s=stem), gc, k


def _contract_no(rng):
    return pick(rng, [ident(rng, "C", 4, "-"), f"{rng.randint(2024, 2027)}-{rng.randint(10, 99)}",
                      ident(rng, "PW", 5, "-"), f"Bid No. {rng.randint(100, 999)}"])


def _ds(rng):
    return pick(rng, ["iso", "us", "long", "eu"])


def _ms(rng):
    return pick(rng, ["sym", "sym", "code"])


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _gap_level(ratio):
    g = abs(ratio - 1.0)
    return 0 if g <= 0.05 else 1 if g <= 0.15 else 2 if g <= 0.35 else 3


# =========================================================================== schema 1: differing site conditions
class DifferingSiteConditions(Schema):
    name = "differing_site_conditions_claim"
    description = "A contractor's notice and price for a differing site condition, reviewed against the contract clause."
    surfaces = {"json": 0.5, "contractor_letter": 0.25, "resident_engineer_memo": 0.25}

    SCEN = {
        "rock": {"found": "rock at about {d} m at the {area}, needing hydraulic breaking", "unit": "m3", "cost": (40000, 260000),
                 "unit_item": "Item 203-04 Rock excavation", "unusual": False,
                 "differs": "boring logs B-3 and B-4 show dense sand and gravel to 6 m with no refusal",
                 "similar": "boring B-4 records auger refusal at 1.9 m and the geotechnical report warns of shallow, "
                            "irregular bedrock",
                 "silent": "no borings were taken in this part of the site"},
        "groundwater": {"found": "groundwater at {d} m at the {area}, flooding the excavation; well points needed",
                        "unit": "day", "cost": (30000, 180000),
                        "unit_item": None, "unusual": False,
                        "differs": "the geotechnical report gives seasonal high groundwater at 4.5 m",
                        "similar": "the geotechnical report notes perched water at 1-2 m after wet periods and makes "
                                   "dewatering the contractor's responsibility",
                        "silent": "the documents say nothing about groundwater"},
        "old_foundation": {"found": "an abandoned concrete foundation and slab, about {q} m3, under the {area}",
                           "cost": (15000, 90000),
                           "unit": "m3", "unit_item": "Item 202-10 Removal of existing structures", "unusual": True,
                           "differs": "the survey and demolition plan show the site cleared to grade",
                           "similar": "the demolition plan notes 'foundations of former warehouse remain below grade; "
                                      "contractor to remove as required'",
                           "silent": "the drawings show existing grade only"},
        "water_main": {"found": "an unmarked 300 mm cast-iron water main crossing the excavation at the {area}, {d} m "
                               "down", "cost": (12000, 70000),
                       "unit": "ea", "unit_item": None, "unusual": True,
                       "differs": "the utility plan shows no utilities in this block",
                       "similar": "the utility plan shows a 'water main, location approximate' within 3 m of the trench",
                       "silent": "the utility plan covers only the street, not the easement"},
        "fuel_soil": {"found": "petroleum-stained soil with a strong odour, about {q} m3, near the {area}", "unit": "t",
                      "cost": (20000, 150000),
                      "unit_item": None, "unusual": True,
                      "differs": "the environmental site assessment found no recognised environmental conditions",
                      "similar": "the Phase I assessment lists a former fuel island and recommends testing during "
                                 "excavation",
                      "silent": "no environmental report was included in the bid documents"},
        "soft_clay": {"found": "a soft clay pocket at the {area}, over-excavated {d} m and replaced with lean "
                               "concrete", "unit": "m3", "cost": (15000, 90000), "unit_item": "Item 206-02 Unsuitable material removal",
                      "unusual": False,
                      "differs": "the report gives 200 kPa allowable bearing on medium-dense sand at footing level",
                      "similar": "boring B-7 logs soft clay to 2.4 m and the report says to over-excavate where found",
                      "silent": "the report has no boring within 40 m of these footings"},
        "boulders": {"found": "cobbles and boulders up to 1.2 m across at the {area}", "unit": "m", "cost": (30000, 200000),
                     "unit_item": None, "unusual": False,
                     "differs": "the borings along the alignment log silty sand with occasional gravel",
                     "similar": "the report says 'cobbles and boulders should be expected throughout the glacial till'",
                     "silent": "no borings were taken along this reach"},
        "buried_debris": {"found": "buried demolition debris (brick, timber, rebar) in the fill to {d} m under the {area}",
                          "cost": (10000, 80000),
                          "unit": "m3", "unit_item": "Item 206-02 Unsuitable material removal", "unusual": True,
                          "differs": "test pits TP-2 and TP-3 found clean granular fill",
                          "similar": "test pit TP-3 notes 'brick fragments and debris in fill'",
                          "silent": "no test pits were dug in this area"},
    }
    def sample_factors(self, rng):
        reg = regime(rng)
        key = pick(rng, list(self.SCEN))
        sc = self.SCEN[key]
        docs = pick_w(rng, {"differs": 6, "similar": 2, "silent": 2})
        n_days = pick(rng, [2, 3, 5, 7])
        notice_days = pick(rng, [0, 1, 1, 2, 3, 5, 8, 12])
        form = pick_w(rng, {"letter": 4, "email": 3, "verbal_then_letter": 2, "verbal_only": 1})
        disturbed = chance(rng, 0.25)
        owner_saw = chance(rng, 0.35)
        pending = chance(rng, 0.18) or (key == "fuel_soil" and chance(rng, 0.5))
        partial = chance(rng, 0.4)
        unusual = sc["unusual"] if chance(rng, 0.8) else not sc["unusual"]
        extent = pick_w(rng, {"measured": 4, "estimated": 3, "unknown": 2})
        backup = pick_w(rng, {"detailed": 4, "summary": 3, "partial": 2})
        ratio = round(between(rng, 0.9, 1.9), 2)
        float_days = pick(rng, [0, 0, 3, 8, 15, 25])
        req_days = pick(rng, [3, 5, 8, 10, 14, 20, 30])
        tia = chance(rng, 0.5)
        if reg == "clear":
            if chance(rng, 0.5):
                docs, notice_days, form, disturbed, pending, partial = "differs", 0, "letter", False, False, False
                ratio, backup, float_days, tia = round(between(rng, 0.95, 1.08), 2), "detailed", 0, True
            else:
                docs = pick(rng, ["similar", "similar", "differs"])
                if docs == "differs":
                    notice_days, form, disturbed, owner_saw = n_days + pick(rng, [6, 10, 20]), "verbal_only", True, False
                ratio, backup = round(between(rng, 1.6, 2.4), 2), "summary"
        elif reg == "borderline":
            notice_days = n_days + pick(rng, [-1, 0, 0, 1, 2])
            notice_days = max(0, notice_days)
            ratio = near(rng, 1.15, 0.12, lo=0.9)
            float_days = pick(rng, [0, req_days // 2, req_days - 2, req_days + 3])
        else:  # conflicting: entitlement looks good on the documents but the notice went wrong, or vice versa
            if chance(rng, 0.5):
                docs, disturbed, form = "differs", True, pick(rng, ["verbal_only", "verbal_then_letter"])
                notice_days = n_days + pick(rng, [2, 4, 9])
            else:
                docs, notice_days, form, disturbed = pick(rng, ["similar", "silent", "differs"]), 0, "letter", False
                partial = chance(rng, 0.6)
            ratio = round(between(rng, 0.95, 1.6), 2)
        unit_item = sc["unit_item"] if (sc["unit_item"] and chance(rng, 0.75)) else None
        bid_qty = pick(rng, [50, 120, 200, 400, 800]) if unit_item else None
        found_qty = round(bid_qty * between(rng, 0.4, 2.2)) if unit_item else None
        est = round(between(rng, *sc["cost"]), -2)
        proj = _project(rng)
        return {"regime": reg, "scenario": key, "area": pick(rng, AREAS[proj[3]]), "depth_m": round(between(rng, 0.8, 3.5), 1),
                "qty_text": pick(rng, [60, 140, 300, 650]), "docs": docs, "unusual_for_area": unusual,
                "clause_notice_days": n_days, "notice_days": notice_days, "notice_form": form,
                "disturbed_before_notice": disturbed, "owner_rep_saw": owner_saw, "evidence_pending": pending,
                "partial_overlap": partial, "extent": extent, "backup": backup, "price_ratio": ratio,
                "owner_estimate": est, "requested": round(est * ratio, -2), "unit_item": unit_item,
                "bid_qty": bid_qty, "found_qty": found_qty, "band_pct": pick(rng, [15, 20, 25]),
                "requested_days": req_days, "float_days": float_days, "tia_submitted": tia,
                "project": proj, "contract": _contract_no(rng), "clause": pick(rng, CLAUSE_REFS),
                "pm": person(rng), "re": person(rng), "date": rand_date(rng)}

    # ------------------------------------------------------------------ answers
    def _notice_ok(self, f):
        written_in_time = (f["notice_form"] != "verbal_only" and f["notice_days"] <= f["clause_notice_days"]
                           and not f["disturbed_before_notice"])
        return written_in_time or f["owner_rep_saw"]

    def intended(self, f):
        foreseeable = f["docs"] == "similar" or (f["docs"] == "silent" and not f["unusual_for_area"])
        if foreseeable:
            pos = "deny_condition_foreseeable"
        elif not self._notice_ok(f):
            pos = "deny_notice_defective"
        elif f["evidence_pending"]:
            pos = "investigate_before_deciding"
        elif f["partial_overlap"]:
            pos = "accept_in_part"
        else:
            pos = "accept_entitlement"
        if f["unit_item"]:
            r = f["found_qty"] / f["bid_qty"]
            basis = "apply_contract_unit_price" if r <= 1 + f["band_pct"] / 100 else "unit_price_with_renegotiated_rate"
        elif f["extent"] == "measured" or (f["extent"] == "estimated" and f["backup"] == "detailed"):
            basis = "negotiate_lump_sum"
        else:
            basis = "force_account_tracking"
        lvl = 3 if f["price_ratio"] <= 1.1 else 2 if f["price_ratio"] <= 1.3 else 1 if f["price_ratio"] <= 1.6 else 0
        if f["backup"] == "summary":
            lvl -= 1
        cost = _clamp(lvl)
        crit = max(0, f["requested_days"] - f["float_days"])
        share = crit / f["requested_days"]
        sup = 0 if share == 0 else 1 if share < 0.5 else 2 if share < 0.9 else 3
        if not f["tia_submitted"] and sup >= 2:
            sup -= 1
        return {"dsc_position": pos, "notice_requirement_met": "yes" if self._notice_ok(f) else "no",
                "pricing_basis": basis, "cost_claim_reasonableness": cost, "schedule_impact_support": sup}

    # ------------------------------------------------------------------ rendering
    def _found(self, f):
        return self.SCEN[f["scenario"]]["found"].format(d=f["depth_m"], area=f["area"], q=f["qty_text"])

    def _docs_text(self, f):
        return self.SCEN[f["scenario"]][f["docs"]]

    def _clause(self, f):
        return (f"{f['clause']} (Differing site conditions): the Contractor shall give written notice promptly, and "
                f"in any event within {f['clause_notice_days']} days of encountering the condition and before it is "
                f"disturbed, of (1) subsurface or latent conditions differing materially from those indicated in the "
                f"Contract Documents, or (2) unknown physical conditions of an unusual nature. Unit prices apply where "
                f"an applicable item exists; quantities beyond {100 + f['band_pct']}% of the bid quantity may be "
                f"repriced. Time is adjusted only for delay to the critical path.")

    def _notice_text(self, f):
        form = {"letter": "letter", "email": "email to the owner's project manager", "verbal_then_letter":
                "verbal report to the inspector, followed by a letter", "verbal_only": "verbal report to the inspector "
                "only"}[f["notice_form"]]
        t = f"{form}, {f['notice_days']} day(s) after the condition was found"
        if f["disturbed_before_notice"]:
            t += "; work continued through the condition before the notice was given"
        return t

    def _qty(self, f):
        if not f["unit_item"]:
            return None
        u = self.SCEN[f["scenario"]]["unit"]
        return f"{f['unit_item']}: bid quantity {f['bid_qty']} {u}; quantity found so far {f['found_qty']} {u}"

    def _float(self, f):
        return (f"affected activity had {f['float_days']} days of total float in the last accepted update"
                if f["float_days"] else "affected activity is on the critical path (zero float)")

    def render(self, f, rng, surface):
        ms, ds = _ms(rng), _ds(rng)
        proj, owner, gc, _kind = f["project"]
        found_date = f["date"]
        if surface == "json":
            return dumps(prune({
                "project": {"name": proj, "owner": owner, "contractor": gc, "contract": f["contract"]},
                "claim": {"condition_found": self._found(f), "date_found": found_date.isoformat(),
                          "notice": self._notice_text(f),
                          "owner_representative_observed": ("resident engineer photographed it before removal"
                                                            if f["owner_rep_saw"] else None),
                          "requested_amount": money(f["requested"], rng, style=ms),
                          "cost_backup": {"detailed": "crew, equipment and hours by day, with supplier quotes",
                                          "summary": "single lump-sum figure", "partial": "equipment hours only, no "
                                          "labour breakdown"}[f["backup"]],
                          "requested_days": f["requested_days"], "time_impact_analysis": f["tia_submitted"]},
                "documents": {"what_they_show": self._docs_text(f), "applicable_unit_price": self._qty(f)},
                "owner_review": {"independent_estimate": money(f["owner_estimate"], rng, style=ms),
                                 "schedule": self._float(f),
                                 "open_items": ("lab results / owner's test pits not yet back" if f["evidence_pending"]
                                                else None),
                                 "extent": {"measured": "quantity measured jointly", "estimated": "quantity estimated "
                                            "from partial excavation", "unknown": "extent unknown; work continuing"}[
                                     f["extent"]],
                                 "overlap": ("part of the claimed quantity lies in an area the documents already "
                                             "flagged for this condition" if f["partial_overlap"] else None)},
                "contract_clause": self._clause(f),
            }))
        if surface == "contractor_letter":
            body = [f"Re: {proj}, Contract {f['contract']} -- Notice of differing site condition",
                    f"On {fmt_date(found_date, rng, ds)} our crew encountered {self._found(f)}.",
                    f"The contract documents show that {self._docs_text(f)}.",
                    f"Notice given: {self._notice_text(f)}.",
                    f"We request {money(f['requested'], rng, style=ms)} and {f['requested_days']} calendar days.",
                    {"detailed": "Daily crew and equipment records and supplier quotes are attached.",
                     "summary": "Our cost is summarised in the attached lump sum.",
                     "partial": "Equipment logs are attached; labour records to follow."}[f["backup"]],
                    "A time impact analysis is enclosed." if f["tia_submitted"] else
                    "The schedule impact will be shown in the next update."]
            if self._qty(f):
                body.append(f"For reference, {self._qty(f)}.")
            tail = kv_lines([("Owner's estimate (RE)", money(f["owner_estimate"], rng, style=ms)),
                             ("Schedule", self._float(f)),
                             ("RE observed condition", "yes, photographed" if f["owner_rep_saw"] else "no"),
                             ("Extent", f["extent"]),
                             ("Open items", "lab / test-pit results pending" if f["evidence_pending"] else "none"),
                             ("Overlap with flagged zone", "partial" if f["partial_overlap"] else "none noted")], rng)
            return (f"From: {gc}\nTo: {owner}, attn. {f['pm']}\n\n" + noisy("\n".join(body), rng, 0.1)
                    + f"\n\n--- owner review notes ---\n{tail}\n\n{self._clause(f)}")
        para = [f"{proj} ({owner}); contractor {gc}.",
                f"{fmt_date(found_date, rng, ds)}: {self._found(f)}.",
                f"Documents: {self._docs_text(f)}.", f"Notice: {self._notice_text(f)}.",
                "I saw and photographed the condition before it was removed." if f["owner_rep_saw"]
                else "I did not see the condition before it was removed." if f["disturbed_before_notice"] else "",
                f"Contractor asks {money(f['requested'], rng, style=ms)} ({ {'detailed': 'itemised', 'summary': 'lump sum only', 'partial': 'partly itemised'}[f['backup']] }) "
                f"and {f['requested_days']} days; my estimate is {money(f['owner_estimate'], rng, style=ms)}.",
                self._qty(f) + "." if self._qty(f) else "",
                f"Schedule: {self._float(f)}" + ("; TIA submitted." if f["tia_submitted"] else "; no TIA yet."),
                "Lab results and our test pits are still outstanding." if f["evidence_pending"] else "",
                "Part of the claimed quantity sits in an area the documents already flagged for this condition."
                if f["partial_overlap"] else ""]
        return (f"MEMO -- {f['re']}, resident engineer\nSubject: DSC notice, Contract {f['contract']}\n\n"
                + noisy(join_sentences(para), rng, 0.1) + "\n\n" + self._clause(f))

    def questions(self):
        return {
            "dsc_position": choice("What position should the owner take on entitlement?", {
                "accept_entitlement": "Accept that a differing site condition occurred and that the change is owed.",
                "accept_in_part": "Accept for part of the claimed work and refuse the rest.",
                "deny_condition_foreseeable": "Refuse: the documents indicated the condition, or a reasonable bidder "
                                              "should have expected it.",
                "deny_notice_defective": "Refuse: the contractor's notice did not meet the clause and the owner lost the "
                                         "chance to verify.",
                "investigate_before_deciding": "Reserve the decision until outstanding tests or investigations are back.",
            }),
            "notice_requirement_met": noul(
                "The contractor's notice satisfied the clause's notice requirement.",
                yes="Written notice reached the owner within the stated period and before the condition was disturbed, "
                    "or the owner's representative saw and recorded the condition first.",
                no="Notice was late, verbal only, or came after the condition was removed without the owner seeing it."),
            "pricing_basis": choice("If the change is priced, on what basis should it be priced?", {
                "apply_contract_unit_price": "Measure the quantity and apply the existing contract unit price.",
                "force_account_tracking": "Track actual labour, equipment and material daily because the extent is "
                                          "still unknown.",
                "negotiate_lump_sum": "Agree a fixed lump sum now, since the scope can be defined.",
                "unit_price_with_renegotiated_rate": "Use the unit price item, but renegotiate the rate for the quantity "
                                                     "beyond the variation band.",
            }),
            "cost_claim_reasonableness": score("How reasonable is the requested amount on the evidence submitted?", [
                "Unsupported: far above the owner's estimate, or a bare figure with no backup.",
                "Overstated: well above the estimate, or thinly supported.",
                "Within negotiating range of the estimate and reasonably supported.",
                "Well supported and close to the owner's estimate.",
            ]),
            "schedule_impact_support": score("How well does the record support the time requested?", [
                "Not at all: available float absorbs the delay.",
                "Weakly: only a small part of the request reaches the critical path.",
                "Partly: most of it reaches the critical path, but the analysis is incomplete.",
                "Fully: a time impact analysis shows the whole request on the critical path.",
            ]),
        }


# =========================================================================== schema 2: owner-directed change pricing
class OwnerDirectedChangePricing(Schema):
    name = "owner_directed_change_pricing"
    description = "A contractor's change-order proposal for owner-requested work, priced against the clause's markup rules."
    surfaces = {"json": 0.5, "cop_form": 0.3, "pm_review_email": 0.2}

    # (scope, trade, {overlap: what the bid documents already say}, size range, direct cost per unit of size)
    CHANGES = [
        ("add {n} floor boxes with power and data in the open office", "electrical", {
            "none": "the bid drawings show no floor boxes in the open office",
            "minor": "the bid drawings show the floor-box conduit stubs already; the boxes, devices and wiring are new",
            "substantial": "the bid drawings show {m} of the {n} boxes; Spec 26 05 33 requires boxes 'where shown'",
            "full": "the bid drawings already show all {n} boxes in the open office, and Spec 26 05 33 covers them"},
         (8, 24), 1450),
        ("upgrade the lobby flooring from VCT to terrazzo, {n} m2", "flooring", {
            "none": "the finish schedule shows VCT in the lobby",
            "minor": "the finish schedule shows VCT, but the terrazzo border strip at the entrance was already in the bid",
            "substantial": "Addendum 3 already changed the lobby to terrazzo except a {m} m2 vestibule",
            "full": "Addendum 3 already changed the lobby finish to terrazzo before bid"}, (60, 240), 260),
        ("relocate RTU-{n} about 4 m east to clear a new skylight", "mechanical", {
            "none": "drawing M-201 fixes RTU-{n} at grid C/5",
            "minor": "M-201 shows RTU-{n} at grid C/5; the curb was to be field-located, but the ductwork was not",
            "substantial": "M-201 notes 'final RTU location to be coordinated with skylights'; only the extra duct is new",
            "full": "M-201 notes 'contractor to coordinate RTU and skylight locations at no extra cost'"}, (2, 6), 5200),
        ("add card readers and electric strikes at {n} doors", "electrical", {
            "none": "the door hardware schedule lists keyed locks at these doors",
            "minor": "the hardware schedule lists keyed locks; empty conduit to the frames was already specified",
            "substantial": "the hardware schedule shows card access at {m} of these {n} doors already",
            "full": "the hardware schedule already shows card readers and strikes at all {n} doors"}, (4, 14), 3100),
        ("add seismic bracing to {n} m of ductwork in the mechanical room", "mechanical", {
            "none": "the project was bid under the pre-2024 code edition, and Spec 23 05 48 exempts ducts this size",
            "minor": "Spec 23 05 48 requires restraint of the larger ducts only; most of this run is the smaller size",
            "substantial": "Spec 23 05 48 requires seismic restraint of ductwork 'per the governing code' for most sizes",
            "full": "Spec 23 05 48 requires seismic restraint of all ductwork per the governing code"}, (20, 90), 140),
        ("extend the chain-link fence by {n} m and add a vehicle gate", "site", {
            "none": "the site plan shows fence on three sides only",
            "minor": "the site plan shows fence on three sides; the gate posts at the new opening were already shown",
            "substantial": "the site plan shows the fence on all four sides; only the vehicle gate is new",
            "full": "the site plan already shows the fence and a vehicle gate on the fourth side"}, (30, 160), 95),
        ("change the exterior coating to an elastomeric system on {n} m2", "painting", {
            "none": "Spec 09 91 13 calls for two coats of acrylic on exterior concrete",
            "minor": "Spec 09 91 13 calls for acrylic, with elastomeric only at the {m} m2 parapet caps",
            "substantial": "Spec 09 91 13 calls for elastomeric on all exterior concrete except the north elevation",
            "full": "Spec 09 91 13 already calls for an elastomeric system on all exterior concrete"}, (300, 1400), 38),
        ("add {n} m of 150 mm storm drain to a new yard inlet", "site utilities", {
            "none": "civil sheet C-301 shows the yard draining overland",
            "minor": "C-301 shows the yard draining overland; the inlet structure itself was bid as an alternate",
            "substantial": "C-301 shows the inlet and most of the pipe; the owner wants it extended {m} m",
            "full": "C-301 already shows the inlet and the full pipe run"}, (25, 120), 310),
        ("thicken the mechanical room slab from 150 mm to 200 mm over {n} m2", "concrete", {
            "none": "the structural notes give 150 mm everywhere on this level",
            "minor": "the structural notes give 150 mm; thickened edges at the equipment pads were already shown",
            "substantial": "section 4/S-301 shows 200 mm under the chillers, about two thirds of the room",
            "full": "section 4/S-301 already shows 200 mm for the whole mechanical room"}, (40, 160), 135),
        ("add blocking and backing for {n} wall-mounted monitors", "drywall", {
            "none": "no monitors are shown or specified anywhere in the documents",
            "minor": "the documents show {m} monitors elsewhere; these locations are new",
            "substantial": "the AV drawings show most of these monitors; Spec 06 10 00 requires blocking for items shown",
            "full": "the AV drawings show all {n} monitors and Spec 06 10 00 requires blocking for every item shown"},
         (10, 40), 220),
    ]

    def sample_factors(self, rng):
        reg = regime(rng)
        ci = rng.randrange(len(self.CHANGES))
        lo, hi = self.CHANGES[ci][3]
        n = rng.randint(lo, hi)
        overlap = pick_w(rng, {"none": 5, "minor": 2, "substantial": 2, "full": 1})
        own_cap, sub_cap, bond_cap = pick(rng, [10, 12, 15]), pick(rng, [5, 7.5, 10]), pick(rng, [1, 1.5, 2])
        own_mk = own_cap if chance(rng, 0.6) else pick(rng, [own_cap + 3, own_cap + 5, own_cap - 2])
        sub_mk = sub_cap if chance(rng, 0.6) else pick(rng, [sub_cap + 2.5, sub_cap + 5])
        bond = bond_cap if chance(rng, 0.8) else bond_cap + 1
        rate = pick(rng, [68.5, 74.0, 81.25, 88.0, 95.5])
        rate_prop = rate if chance(rng, 0.7) else round(rate * between(rng, 1.12, 1.3), 2)
        true_cost = round(max(1800.0, n * self.CHANGES[ci][4] * rng.uniform(0.75, 1.3)), -1)
        self_perf = chance(rng, 0.4)
        ratio = round(between(rng, 0.95, 1.6), 2)
        backup = pick_w(rng, {"complete": 5, "partial": 3, "none": 1})
        urgent = chance(rng, 0.35)
        critical = chance(rng, 0.4)
        days = pick(rng, [2, 3, 5, 8, 12, 20])
        if reg == "clear":
            if chance(rng, 0.5):
                overlap, own_mk, sub_mk, bond, rate_prop, ratio, backup = "none", own_cap, sub_cap, bond_cap, rate, \
                    round(between(rng, 0.97, 1.04), 2), "complete"
            else:
                overlap = pick(rng, ["full", "full", "none"])
                if overlap == "none":
                    backup, ratio = "none", round(between(rng, 1.4, 2.0), 2)
        elif reg == "borderline":
            ratio = near(rng, 1.15, 0.1, lo=0.9)
            own_mk = near(rng, own_cap, 1.5, lo=5, digits=1)
        else:  # conflicting: a tidy, compliant proposal for work the spec already requires; or urgent and far apart
            if chance(rng, 0.5):
                overlap, own_mk, sub_mk, bond, backup = pick(rng, ["full", "substantial"]), own_cap, sub_cap, bond_cap, "complete"
                ratio = round(between(rng, 0.98, 1.06), 2)
            else:
                urgent, ratio, overlap, backup = True, round(between(rng, 1.3, 1.9), 2), "none", "complete"
        labour_h = max(4, round(true_cost * (0.35 if self_perf else 0.07) / rate))
        prop_total = round(true_cost * ratio, -1)
        return {"regime": reg, "change_index": ci, "n": n, "m": max(1, round(n * between(rng, 0.3, 0.7))),
                "trade": self.CHANGES[ci][1],
                "overlap": overlap, "own_cap": own_cap, "sub_cap": sub_cap, "bond_cap": bond_cap,
                "own_markup": own_mk, "sub_markup": sub_mk, "bond_pct": bond, "contract_rate": rate,
                "proposed_rate": rate_prop, "labour_hours": labour_h, "owner_estimate": true_cost,
                "proposal_total": prop_total, "price_ratio": ratio, "backup": backup, "work_needed_now": urgent,
                "on_critical_path": critical, "days_requested": days, "self_performed": self_perf,
                "project": _project(rng, None if self.CHANGES[ci][1] in ("site", "site utilities") else "building"), "contract": _contract_no(rng), "clause": pick(rng, CLAUSE_REFS),
                "cop_no": f"COP-{rng.randint(3, 88):03d}", "bulletin": pick(rng, ["Bulletin", "ASI", "Owner request",
                                                                                 "PR"]) + f" {rng.randint(1, 30)}",
                "pm": person(rng), "date": rand_date(rng)}

    def _markups_ok(self, f):
        sub_ok = f["self_performed"] or f["sub_markup"] <= f["sub_cap"]
        return f["own_markup"] <= f["own_cap"] and sub_ok and f["bond_pct"] <= f["bond_cap"]

    def intended(self, f):
        gap = _gap_level(f["price_ratio"])
        ov = {"none": 0, "minor": 1, "substantial": 2, "full": 3}[f["overlap"]]
        mk_ok = self._markups_ok(f)
        rate_ok = f["proposed_rate"] <= f["contract_rate"] + 0.01
        if f["overlap"] == "full":
            disp = "reject_base_scope"
        elif f["backup"] == "none" or (f["backup"] == "partial" and gap >= 2):
            disp = "request_backup"
        elif f["work_needed_now"] and gap >= 2:
            disp = "issue_directive_and_negotiate"
        elif gap == 0 and mk_ok and rate_ok and f["overlap"] == "none":
            disp = "accept_proposal"
        else:
            disp = "negotiate_corrections"
        time_ok = f["on_critical_path"] and f["overlap"] != "full"
        return {"cop_disposition": disp, "markups_within_clause": "yes" if mk_ok else "no",
                "price_gap_to_estimate": gap, "base_scope_overlap": ov,
                "time_request_warranted": "yes" if time_ok else "no"}

    def _change(self, f):
        c = self.CHANGES[f["change_index"]]
        return c[0].format(n=f["n"]), c[2][f["overlap"]].format(n=f["n"], m=f["m"])

    def _clause(self, f):
        return (f"{f['clause']} (Changes, pricing): overhead and profit on self-performed work up to {f['own_cap']}%; "
                f"the contractor's markup on subcontracted work up to {f['sub_cap']}%; bond and insurance up to "
                f"{f['bond_cap']}%. Labour at the rates in Exhibit L. Work already required by the Contract Documents "
                f"is not a change. The owner may direct changed work by construction change directive pending price "
                f"agreement.")

    def _lines(self, f):
        sub = f["proposal_total"] * (0 if f["self_performed"] else 0.55)
        labour = f["labour_hours"] * f["proposed_rate"]
        base = f["proposal_total"] / (1 + f["own_markup"] / 100 + f["bond_pct"] / 100)
        material = max(0.0, base - labour - sub)
        rows = [("Labour", f"{f['labour_hours']} h @ {f['proposed_rate']:.2f}", labour),
                ("Material and equipment", "per attached quotes" if f["backup"] == "complete" else "lump sum", material)]
        if not f["self_performed"]:
            rows.append((f"{f['trade'].capitalize()} subcontractor", f"incl. sub markup {f['sub_markup']}%", sub))
        rows += [("Overhead and profit", f"{f['own_markup']}%", base * f["own_markup"] / 100),
                 ("Bond and insurance", f"{f['bond_pct']}%", base * f["bond_pct"] / 100)]
        return rows

    def render(self, f, rng, surface):
        ms, ds = _ms(rng), _ds(rng)
        proj, owner, gc, _kind = f["project"]
        what, ref = self._change(f)
        lines = self._lines(f)
        backup = {"complete": "itemised quantities, crew hours and supplier quotes", "partial": "summary by trade, no "
                  "quotes", "none": "single total only"}[f["backup"]]
        if surface == "json":
            return dumps(prune({
                "project": {"name": proj, "owner": owner, "contractor": gc, "contract": f["contract"]},
                "change": {"source": f["bulletin"], "description": what, "contract_documents_say": ref,
                           "work_needed_now": f["work_needed_now"] or None},
                "proposal": {"number": f["cop_no"], "total": money(f["proposal_total"], rng, style=ms),
                             "labour_rate": f["proposed_rate"], "labour_hours": f["labour_hours"],
                             "overhead_profit_pct": f["own_markup"],
                             "sub_markup_pct": None if f["self_performed"] else f["sub_markup"],
                             "bond_insurance_pct": f["bond_pct"], "backup": backup,
                             "days_requested": f["days_requested"]},
                "owner_review": {"independent_estimate": money(f["owner_estimate"], rng, style=ms),
                                 "exhibit_l_rate": f["contract_rate"],
                                 "schedule": "affected work is on the critical path" if f["on_critical_path"]
                                 else "affected work has float in the current schedule"},
                "contract_clause": self._clause(f),
            }))
        if surface == "cop_form":
            rows = [[a, b, money(round(c, 2), rng, style=ms)] for a, b, c in lines]
            rows.append(["TOTAL", "", money(f["proposal_total"], rng, style=ms)])
            return "\n".join([
                f"CHANGE ORDER PROPOSAL {f['cop_no']} -- {proj}",
                kv_lines([("Contractor", gc), ("Owner", owner), ("Contract", f["contract"]),
                          ("Reference", f["bulletin"]), ("Date", fmt_date(f["date"], rng, ds)),
                          ("Scope", what), ("Time requested", f"{f['days_requested']} calendar days"),
                          ("Backup", backup)], rng), "",
                table(["Item", "Basis", "Amount"], rows, rng), "",
                "Owner review notes:",
                bullets([f"Documents: {ref}.", f"Independent estimate {money(f['owner_estimate'], rng, style=ms)}.",
                         f"Exhibit L labour rate {f['contract_rate']:.2f}/h.",
                         "Area is on the critical path." if f["on_critical_path"] else "Area has float.",
                         "Owner wants the work started this week." if f["work_needed_now"] else None], rng), "",
                self._clause(f)])
        para = [f"Proposal {f['cop_no']} from {gc} on {proj}: {what} ({f['bulletin']}).",
                f"They are asking {money(f['proposal_total'], rng, style=ms)} and {f['days_requested']} days; our "
                f"estimate is {money(f['owner_estimate'], rng, style=ms)}.",
                f"Markups as submitted: OH&P {f['own_markup']}%"
                + ("" if f["self_performed"] else f", sub markup {f['sub_markup']}%") + f", bond {f['bond_pct']}%.",
                f"Labour priced at {f['proposed_rate']:.2f}/h for {f['labour_hours']} h (Exhibit L: {f['contract_rate']:.2f}).",
                f"Backup: {backup}.", f"On the documents: {ref}.",
                "The work is on the critical path." if f["on_critical_path"] else "There is float on this work.",
                "The principal wants it started before the ceiling close-in next week." if f["work_needed_now"] else ""]
        return (f"Subject: {f['cop_no']} review -- {proj}\nFrom: {f['pm']}\nTo: owner's project team\n\n"
                + noisy(join_sentences(para), rng, 0.1) + "\n\n" + self._clause(f))

    def questions(self):
        return {
            "cop_disposition": choice("How should the owner respond to this change-order proposal?", {
                "accept_proposal": "Accept the proposal as priced and issue the change order.",
                "issue_directive_and_negotiate": "Direct the work to proceed now under a change directive and settle "
                                                 "the price afterwards.",
                "negotiate_corrections": "Accept the change in principle but return the price for specific corrections "
                                         "before signing.",
                "reject_base_scope": "Reject it: the work is already required by the contract.",
                "request_backup": "Return it for the breakdown or quotes needed to evaluate the price.",
            }),
            "markups_within_clause": noul(
                "Every markup in the proposal is within the limits the clause allows.",
                yes="Overhead and profit, subcontract markup and bond are each at or below the clause's caps.",
                no="At least one markup exceeds the cap the clause sets for it."),
            "price_gap_to_estimate": score("How far is the proposed price from the owner's independent estimate?", [
                "Negligible: within about 5%.",
                "Modest: roughly 5 to 15% apart.",
                "Large: roughly 15 to 35% apart.",
                "Very large: more than about a third apart.",
            ]),
            "base_scope_overlap": score("How much of the proposed work do the contract documents already require?", [
                "None of it: entirely new work.",
                "A small part overlaps with work already shown or specified.",
                "A substantial part is already shown or specified.",
                "All of it is already required.",
            ]),
            "time_request_warranted": noul("The contractor's request for added time is warranted."),
        }


# =========================================================================== schema 3: delay / time extension
class TimeExtensionRequest(Schema):
    name = "time_extension_request"
    description = "A contractor's request for more contract time after a delay, reviewed under the time clause."
    surfaces = {"json": 0.5, "request_letter": 0.25, "scheduler_review": 0.25}

    CAUSES = {
        "late_submittal_review": ("owner", "structural steel shop drawings were returned after {x} working days against "
                                           "the {r}-day review period"),
        "owner_furnished_late": ("owner", "owner-furnished switchgear arrived {x} days after the date in the contract"),
        "owner_decision": ("owner", "the owner's decision on the lobby finish was {x} days late"),
        "design_reissue": ("owner", "the foundation drawings had to be re-issued after a design error, stopping the work "
                                    "for {x} days"),
        "weather": ("neutral", "rain cost {x} working days in {month}; the baseline allows {allow} weather days for "
                               "that month"),
        "utility_relocation": ("neutral", "the power utility relocated its pole line {x} days later than it had "
                                          "committed"),
        "supplier_strike": ("neutral", "a strike at the ready-mix supplier halted concrete deliveries for {x} days"),
        "inspection_backlog": ("neutral", "the city's inspection backlog delayed the cover inspection by {x} days"),
        "crane_breakdown": ("contractor", "the erector's crane broke down and was out for {x} days"),
        "sub_default": ("contractor", "the drywall subcontractor left the job and a replacement took {x} days to "
                                      "mobilise"),
        "rebar_misdetail": ("contractor", "the rebar fabricator mis-detailed the slab bars; re-fabrication took {x} days"),
    }
    MONTHS = ["March", "April", "May", "October", "November", "January"]
    BUILDING_ONLY = {"late_submittal_review", "owner_furnished_late", "owner_decision", "sub_default"}

    def sample_factors(self, rng):
        reg = regime(rng)
        cause = pick_w(rng, {k: (1 if v[0] == "contractor" else 2) for k, v in self.CAUSES.items()})
        proj = _project(rng)
        resp = self.CAUSES[cause][0]
        delay = pick(rng, [4, 6, 9, 12, 15, 21, 30])
        allow = pick(rng, [3, 4, 5, 6])
        lost = delay + allow if cause == "weather" else None
        float_days = pick(rng, [0, 0, 0, 2, 4, 7, 12, 25])
        concurrent = pick_w(rng, {0: 6, 3: 2, 6: 1, 10: 1})
        tia = chance(rng, 0.6)
        current = chance(rng, 0.8)
        n_days = pick(rng, [7, 10, 14, 21])
        notice = pick(rng, [2, 5, 7, 9, 12, 16, 25])
        requested = delay + pick(rng, [0, 0, 0, 2, 5, 10])
        if reg == "clear":
            if chance(rng, 0.5):
                cause = pick(rng, ["late_submittal_review", "owner_furnished_late", "design_reissue"])
                float_days, concurrent, tia, current, notice, requested = 0, 0, True, True, min(notice, n_days - 2), delay
            else:
                cause = pick(rng, ["crane_breakdown", "sub_default", "rebar_misdetail"])
                notice = n_days + pick(rng, [5, 12])
        elif reg == "borderline":
            notice = n_days + pick(rng, [-1, 0, 0, 1, 2])
            float_days = max(0, delay + pick(rng, [-9, -6, -4, -2, -1, 0, 3]))
            if cause == "weather":
                lost = allow + pick(rng, [-1, 0, 1, 2, 4])
        else:  # conflicting: owner-caused delay with the contractor's own delay running alongside, or no analysis
            cause = pick(rng, ["late_submittal_review", "owner_decision", "utility_relocation", "weather"])
            if chance(rng, 0.6):
                concurrent = pick(rng, [4, 7, 12])
            else:
                tia, current = False, False
        resp = self.CAUSES[cause][0]
        if cause == "weather":
            lost = lost if lost is not None else delay + allow
            delay = max(0, lost - allow)
        if resp == "contractor":
            concurrent = 0
        if cause in self.BUILDING_ONLY and proj[3] != "building":
            proj = _project(rng, "building")
        return {"regime": reg, "cause": cause, "responsibility": resp, "delay_days": delay, "weather_lost": lost,
                "weather_allowed": allow, "month": pick(rng, self.MONTHS), "review_days": pick(rng, [10, 14, 15]),
                "float_days": float_days, "concurrent_days": concurrent, "tia": tia, "schedule_current": current,
                "clause_notice_days": n_days, "notice_days": notice, "requested_days": max(1, requested),
                "response_period": pick(rng, [14, 21, 30]), "days_since_request": pick(rng, [3, 9, 15, 22, 31, 40]),
                "accelerating": chance(rng, 0.25), "days_to_completion": pick(rng, [18, 45, 90, 160, 300]),
                "ld_per_day": pick(rng, [750, 1500, 2500, 4000, 8000]), "project": proj,
                "contract": _contract_no(rng), "clause": pick(rng, CLAUSE_REFS), "scheduler": person(rng),
                "date": rand_date(rng)}

    def _critical_days(self, f):
        return max(0, f["delay_days"] - f["float_days"])

    def intended(self, f):
        resp = f["responsibility"]
        crit = self._critical_days(f)
        weather_within = f["cause"] == "weather" and f["weather_lost"] <= f["weather_allowed"]
        if resp == "contractor" or weather_within:
            cat = "not_excusable"
        elif not f["tia"] and not f["schedule_current"]:
            cat = "cannot_classify_yet"
        elif resp == "owner" and f["concurrent_days"] == 0:
            cat = "excusable_and_compensable"
        else:
            cat = "excusable_time_only"
        if cat in ("not_excusable", "cannot_classify_yet") or crit == 0:
            sup = 0
        else:
            share = crit / f["requested_days"]
            sup = 1 if share < 0.5 else 2 if share < 0.9 else 3
        if cat == "cannot_classify_yet":
            step = "request_time_impact_analysis"
        elif sup == 0:
            step = "deny_request"
        elif sup == 3:
            step = "grant_in_full"
        else:
            step = "grant_in_part"
        p = 0
        left = f["response_period"] - f["days_since_request"]
        if left < 0:
            p += 2
        elif left <= 5:
            p += 1
        if f["accelerating"]:
            p += 1
        if f["days_to_completion"] <= 45:
            p += 1
        notice_ok = f["notice_days"] <= f["clause_notice_days"]
        return {"delay_category": cat, "supported_extension": sup, "extension_next_step": step,
                "decision_time_pressure": _clamp(p), "notice_timely": "yes" if notice_ok else "no"}

    def _cause_text(self, f):
        return self.CAUSES[f["cause"]][1].format(x=f["delay_days"] if f["cause"] != "weather" else f["weather_lost"],
                                                 r=f["review_days"], month=f["month"], allow=f["weather_allowed"])

    def _clause(self, f):
        return (f"{f['clause']} (Time): written notice of delay within {f['clause_notice_days']} days of its start. "
                f"Extensions only for delay to the critical path, shown by a time impact analysis on the current "
                f"accepted schedule; float belongs to the project. Owner-caused delay: time and cost. Delay beyond both "
                f"parties' control, including weather beyond the baseline allowance: time only. Where the contractor's "
                f"own delay runs concurrently: time only. The owner responds within {f['response_period']} days.")

    def _sched(self, f):
        if not f["schedule_current"] and not f["tia"]:
            return "last accepted schedule update is three months old; no time impact analysis submitted"
        s = (f"affected activity had {f['float_days']} days of float" if f["float_days"] else
             "affected activity was on the critical path")
        if f["concurrent_days"]:
            s += f"; the contractor's own late formwork delivery overlapped the same window for {f['concurrent_days']} days"
        s += "; TIA submitted" if f["tia"] else "; no TIA, float taken from the last update"
        return s

    def render(self, f, rng, surface):
        ds = _ds(rng)
        proj, owner, gc, _kind = f["project"]
        start = f["date"]
        if surface == "json":
            return dumps(prune({
                "project": {"name": proj, "owner": owner, "contractor": gc, "contract": f["contract"],
                            "days_to_contract_completion": f["days_to_completion"],
                            "liquidated_damages_per_day": f["ld_per_day"]},
                "request": {"cause": self._cause_text(f), "delay_started": start.isoformat(),
                            "written_notice_after_days": f["notice_days"], "days_requested": f["requested_days"],
                            "submitted_days_ago": f["days_since_request"],
                            "contractor_note": "second shift added at our cost pending your decision"
                            if f["accelerating"] else None},
                "schedule_review": self._sched(f),
                "contract_clause": self._clause(f),
            }))
        if surface == "request_letter":
            head = f"Re: {proj} -- Request for extension of time No. {rng.randint(1, 9)}"
            body = [f"Beginning {fmt_date(start, rng, ds)}, {self._cause_text(f)}.",
                    f"We gave written notice {f['notice_days']} days after the delay began.",
                    f"We request {f['requested_days']} calendar days.",
                    "Our time impact analysis is attached." if f["tia"] else "We will quantify the impact in a later update.",
                    "We have added a second shift at our own cost to protect the completion date and reserve our "
                    "rights." if f["accelerating"] else ""]
            return (f"From: {gc}\nTo: {owner}\n{head}\n\n" + noisy(join_sentences(body), rng, 0.1)
                    + f"\n\n[Owner log: received {f['days_since_request']} days ago; {f['days_to_completion']} days to "
                      f"contract completion; LDs {money(f['ld_per_day'], rng)}/day]\n\n{self._clause(f)}")
        rows = [["Cause", self._cause_text(f)], ["Delay start", fmt_date(start, rng, ds)],
                ["Notice", f"{f['notice_days']} days after start"], ["Requested", f"{f['requested_days']} days"],
                ["Schedule", self._sched(f)], ["Received", f"{f['days_since_request']} days ago"],
                ["Completion", f"{f['days_to_completion']} days left; LDs {money(f['ld_per_day'], rng)}/day"],
                ["Acceleration", "contractor running a second shift at its own cost" if f["accelerating"] else "none"]]
        return "\n".join([f"SCHEDULE REVIEW -- {proj} ({gc})", f"Reviewer: {f['scheduler']}", "",
                          table(["Item", "Detail"], rows, rng), "", self._clause(f)])

    def questions(self):
        return {
            "delay_category": choice("How should the delay be classified under the time clause?", {
                "cannot_classify_yet": "It cannot be classified yet; the schedule analysis the clause relies on is missing.",
                "excusable_and_compensable": "Excusable and compensable: the owner caused it, so time and cost are due.",
                "excusable_time_only": "Excusable but not compensable: time only, no added cost.",
                "not_excusable": "Not excusable: the contractor bears this delay.",
            }),
            "supported_extension": score("How much of the requested time does the record support?", [
                "None of it.",
                "Less than half of the days requested.",
                "Most of the days requested, but not all.",
                "All, or very nearly all, of the days requested.",
            ]),
            "extension_next_step": choice("What should the owner do with the request now?", {
                "deny_request": "Deny the request and confirm the existing completion date.",
                "grant_in_full": "Grant the days requested by change order.",
                "grant_in_part": "Grant the days the record supports and deny the rest.",
                "request_time_impact_analysis": "Ask for a time impact analysis on a current schedule before ruling.",
            }),
            "decision_time_pressure": score("How pressing is it for the owner to rule on this request?", [
                "Not pressing: well inside the response period, completion far off.",
                "Some pressure: the response period is closing.",
                "Pressing: the period has run out, or the contractor is spending to accelerate.",
                "Urgent: overdue, with acceleration under way or completion and damages close.",
            ]),
            "notice_timely": noul(
                "The contractor gave written notice of the delay within the period the clause requires.",
                yes="Written notice arrived within the stated number of days from the start of the delay.",
                no="Notice came after the stated period."),
        }


# =========================================================================== schema 4: quantity variation repricing
class QuantityVariationRepricing(Schema):
    name = "quantity_variation_repricing"
    description = "A unit-price item whose quantity has moved far from the bid quantity: whether and how to reprice it."
    surfaces = {"json": 0.5, "quantity_memo": 0.25, "item_ledger": 0.25}

    ITEMS = [("Rock excavation", "m3", (55, 140)), ("Unsuitable subgrade removal and replacement", "m3", (28, 60)),
             ("Full-depth pavement patching", "m2", (70, 160)), ("Concrete curb and gutter", "m", (45, 95)),
             ("Crushed stone pipe bedding", "t", (30, 55)), ("Trench rock excavation", "m3", (90, 210)),
             ("Footing concrete", "m3", (380, 650)), ("Shotcrete slope protection", "m2", (85, 170)),
             ("Asbestos pipe-insulation abatement", "m", (40, 90)), ("Sidewalk replacement", "m2", (60, 120)),
             ("Topsoil and seeding", "m2", (4, 9)), ("Sewer service reconnection", "ea", (1800, 4200)),
             ("Geotextile stabilisation fabric", "m2", (3, 7)), ("Cement-treated base", "m2", (14, 30))]
    CAUSES = {
        "design_error": "the engineer's quantity take-off missed the {where}",
        "owner_added": "the owner added the {where} to the work by field order",
        "overbreak": "the contractor's measured quantity includes material removed or placed beyond the neat lines at "
                     "the {where}",
        "changed_character": "{character}",
        "underrun_design": "the {where} turned out far smaller than the plans assumed",
    }
    WHERES = {"civil": ["east approach", "Line B trench", "station 4+200 to 4+560", "north parking lot",
                        "pump station site", "school frontage", "Phase 2 blocks", "outfall channel", "retaining wall RW-3",
                        "west ramp", "detention basin"],
              "building": ["basement level", "east wing", "boiler room", "stair cores", "level 2 corridors",
                           "loading dock", "gymnasium"]}
    BUILDING_ITEMS = {"Footing concrete", "Asbestos pipe-insulation abatement"}

    def sample_factors(self, rng):
        reg = regime(rng)
        ii = rng.randrange(len(self.ITEMS))
        name, unit, (plo, phi) = self.ITEMS[ii]
        proj = _project(rng, "building" if name in self.BUILDING_ITEMS else "civil")
        band = pick(rng, [15, 20, 25])
        cause = pick_w(rng, {"design_error": 4, "owner_added": 2, "overbreak": 4, "changed_character": 2,
                             "underrun_design": 2})
        ratio = round(between(rng, 1.3, 2.4), 2) if cause != "underrun_design" else round(between(rng, 0.35, 0.8), 2)
        measure = pick_w(rng, {"joint_survey": 3, "scale_tickets": 3, "contractor_survey": 3, "disputed_sections": 2})
        if unit not in ("m3", "t") and measure == "scale_tickets":
            measure = pick(rng, ["joint_survey", "contractor_survey", "disputed_sections"])
        demand = pick_w(rng, {"owner_lower": 3, "contractor_higher": 3, "none": 2})
        unbal = round(pick_w(rng, {between(rng, 0.85, 1.15): 4, between(rng, 1.2, 1.6): 3, between(rng, 1.7, 2.6): 2}), 2)
        cost_evidence = pick_w(rng, {"lower": 3, "higher": 3, "none": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                cause, ratio, measure, demand = "design_error", round(between(rng, 1.03, 1 + band / 100 - 0.04), 2), \
                    "joint_survey", "none"
            else:
                cause, ratio, measure = pick(rng, ["design_error", "owner_added"]), round(between(rng, 1.6, 2.4), 2), \
                    "joint_survey"
                demand, cost_evidence = "owner_lower", "lower"
        elif reg == "borderline":
            if cause == "underrun_design":
                ratio = near(rng, 1 - band / 100, 0.06, lo=0.3)
            else:
                ratio = near(rng, 1 + band / 100, 0.06, lo=0.8)
        else:  # conflicting: the trigger is plainly met but the extra is not "more of the same"
            cause = pick(rng, ["overbreak", "changed_character", "owner_added"])
            if cause == "owner_added":
                unbal, demand = round(between(rng, 1.8, 2.6), 2), "contractor_higher"
            measure = pick(rng, ["contractor_survey", "scale_tickets", "joint_survey"] if unit in ("m3", "t")
                           else ["contractor_survey", "joint_survey"])
        bid_price = round(between(rng, plo, phi), 2)
        eng_price = round(bid_price / unbal, 2)
        bid_qty = pick(rng, [40, 90, 150, 300, 600, 1200, 4000]) if unit != "ea" else pick(rng, [12, 20, 35, 60])
        actual = round(bid_qty * ratio) if unit != "ea" else max(1, round(bid_qty * ratio))
        return {"regime": reg, "item": name, "unit": unit, "item_no": f"{rng.randint(201, 699)}-{rng.randint(1, 40):02d}",
                "band_pct": band, "cause": cause, "where": pick(rng, self.WHERES[proj[3]]), "qty_ratio": ratio,
                "bid_qty": bid_qty, "actual_qty": actual, "measurement": measure, "demand": demand,
                "bid_price": bid_price, "engineer_price": eng_price, "unbalance_ratio": unbal,
                "cost_evidence": cost_evidence, "overbreak_share": round(between(rng, 0.12, 0.35), 2)
                if cause == "overbreak" else 0.0, "after_backfill": unit in ("m3", "t") and chance(rng, 0.25),
                "project": proj, "contract": _contract_no(rng), "clause": pick(rng, CLAUSE_REFS),
                "engineer": person(rng), "date": rand_date(rng)}

    def _trigger(self, f):
        b = f["band_pct"] / 100
        return f["qty_ratio"] > 1 + b or f["qty_ratio"] < 1 - b

    def intended(self, f):
        trig = self._trigger(f)
        if f["cause"] == "changed_character":
            price = "price_as_separate_extra_work"
        elif f["cause"] == "overbreak":
            price = "exclude_quantity_outside_lines"
        elif not trig:
            price = "keep_bid_unit_price"
        elif f["qty_ratio"] > 1 and (f["demand"] == "owner_lower" or f["cost_evidence"] == "lower"
                                     or f["unbalance_ratio"] >= 1.7):
            price = "negotiate_lower_unit_price"
        elif f["qty_ratio"] < 1 or f["demand"] == "contractor_higher" or f["cost_evidence"] == "higher":
            price = "negotiate_higher_unit_price"
        else:
            price = "keep_bid_unit_price"
        rel = {"joint_survey": 3, "scale_tickets": 2, "contractor_survey": 1, "disputed_sections": 0}[f["measurement"]]
        if f["after_backfill"] and rel > 0:
            rel -= 1
        ub = 0 if f["unbalance_ratio"] <= 1.15 else 1 if f["unbalance_ratio"] <= 1.6 else 2 if f["unbalance_ratio"] <= 2.0 else 3
        if f["cause"] == "design_error" and ub >= 1 and f["qty_ratio"] > 1:
            ub = _clamp(ub + 1)
        if f["qty_ratio"] < 1:
            ub = max(0, ub - 1)
        inside = f["cause"] != "overbreak" and not (f["measurement"] == "scale_tickets" and f["unit"] in ("m3", "t")) \
            and not (f["after_backfill"] and f["measurement"] == "contractor_survey")
        return {"repricing_trigger_met": "yes" if trig else "no", "excess_quantity_pricing": price,
                "measurement_reliability": rel, "bid_unbalancing_concern": ub,
                "quantity_inside_measurement_lines": "yes" if inside else "no"}

    def _clause(self, f):
        b = f["band_pct"]
        return (f"{f['clause']} (Variations in estimated quantities): where the measured quantity of a unit-price item "
                f"exceeds {100 + b}% or falls below {100 - b}% of the bid quantity, either party may demand an "
                f"equitable adjustment of the unit price for the quantity outside that band, based on the change in "
                f"unit cost. Quantities are measured to the neat lines shown; material beyond them is not measured. "
                f"Work of a different character from the item is a change, not a variation.")

    def _measure_text(self, f):
        count = f["unit"] == "ea"
        t = {"joint_survey": "joint count signed by both parties" if count else "joint survey signed by both parties",
             "scale_tickets": "certified scale tickets, loose volume converted by the contractor",
             "contractor_survey": "contractor's own count, not yet checked by the owner" if count
             else "contractor's own survey, not yet checked by the owner",
             "disputed_sections": "counts disputed; the parties' figures differ by about 18%" if count
             else "cross-sections disputed; the parties' figures differ by about 18%"}[f["measurement"]]
        if f["after_backfill"]:
            t += "; taken after backfill, from photographs"
        return t

    def _cause_text(self, f):
        if "excavation" in f["item"].lower() or "unsuitable" in f["item"].lower():
            character = f"the material at the {f['where']} needs drilling and blasting rather than ripping"
        else:
            character = (f"the added work at the {f['where']} has to be done by hand at night in short, scattered "
                         f"sections, unlike the continuous machine work the item was bid for")
        return self.CAUSES[f["cause"]].format(where=f["where"], character=character)

    def render(self, f, rng, surface):
        ms, ds = _ms(rng), _ds(rng)
        proj, owner, gc, _kind = f["project"]
        dem = {"owner_lower": f"{owner} has demanded a lower unit price for the added quantity",
               "contractor_higher": f"{gc} has demanded a higher unit price",
               "none": "neither party has made a demand yet"}[f["demand"]]
        cost = {"lower": "contractor's cost records show the unit cost of the extra quantity below the bid price "
                         "(mobilisation and set-up already recovered)",
                "higher": "contractor's cost records show the unit cost rising (smaller, scattered work areas)",
                "none": None}[f["cost_evidence"]]
        if surface == "json":
            return dumps(prune({
                "project": {"name": proj, "owner": owner, "contractor": gc, "contract": f["contract"]},
                "item": {"number": f["item_no"], "description": f["item"], "unit": f["unit"], "bid_quantity": f["bid_qty"],
                         "measured_or_projected_quantity": f["actual_qty"],
                         "bid_unit_price": money(f["bid_price"], rng, style=ms),
                         "engineers_estimate_unit_price": money(f["engineer_price"], rng, style=ms)},
                "variation": {"reason": self._cause_text(f), "measurement": self._measure_text(f),
                              "overbreak_share_claimed_by_owner": f"{round(f['overbreak_share'] * 100)}%"
                              if f["overbreak_share"] else None,
                              "demand": dem, "cost_evidence": cost},
                "contract_clause": self._clause(f),
            }))
        if surface == "quantity_memo":
            para = [f"Item {f['item_no']} {f['item']} on {proj} ({gc}).",
                    f"Bid quantity {f['bid_qty']:,} {f['unit']} at {money(f['bid_price'], rng, style=ms)}/{f['unit']} "
                    f"(engineer's estimate {money(f['engineer_price'], rng, style=ms)}); measured/projected "
                    f"{f['actual_qty']:,} {f['unit']}.",
                    f"Reason: {self._cause_text(f)}.", f"Measurement: {self._measure_text(f)}.",
                    (f"Our inspector puts about {round(f['overbreak_share'] * 100)}% of the volume outside the "
                     f"neat lines." if f["overbreak_share"] else ""),
                    dem[0].upper() + dem[1:] + ".", (cost[0].upper() + cost[1:] + ".") if cost else ""]
            return (f"MEMO -- {f['engineer']}, project engineer, {fmt_date(f['date'], rng, ds)}\n\n"
                    + noisy(join_sentences(para), rng, 0.1) + "\n\n" + self._clause(f))
        rows = [["Item", f"{f['item_no']} {f['item']}"], ["Unit", f["unit"]], ["Bid qty", f"{f['bid_qty']:,}"],
                ["To date / projected", f"{f['actual_qty']:,}"], ["Bid price", money(f["bid_price"], rng, style=ms)],
                ["Engineer's est. price", money(f["engineer_price"], rng, style=ms)],
                ["Reason", self._cause_text(f)], ["Measured by", self._measure_text(f)],
                ["Demand", dem], ["Cost evidence", cost or "none submitted"]]
        if f["overbreak_share"]:
            rows.append(["Owner's view", f"~{round(f['overbreak_share'] * 100)}% outside neat lines"])
        return "\n".join([f"UNIT PRICE ITEM LEDGER -- {proj}", f"Contract {f['contract']}; {owner} / {gc}", "",
                          table(["Field", "Value"], rows, rng), "", self._clause(f)])

    def questions(self):
        return {
            "repricing_trigger_met": noul(
                "The quantity has moved far enough from the bid quantity for the variation clause to apply.",
                yes="The measured or projected quantity is outside the clause's band around the bid quantity.",
                no="The quantity is still within the band, so the bid unit price stands."),
            "excess_quantity_pricing": choice("How should the quantity outside the bid quantity be priced?", {
                "exclude_quantity_outside_lines": "Leave out the volume taken beyond the neat lines; it is not measured.",
                "keep_bid_unit_price": "Keep the bid unit price for all of it.",
                "negotiate_higher_unit_price": "Negotiate a higher unit price because the unit cost has risen.",
                "negotiate_lower_unit_price": "Negotiate a lower unit price because the extra quantity costs less per unit.",
                "price_as_separate_extra_work": "Price it as separate extra work; it is a different kind of work.",
            }),
            "measurement_reliability": score("How reliable is the reported quantity?", [
                "Unreliable: disputed or reconstructed after the fact.",
                "Weak: one party's figures, not yet checked.",
                "Fair: an objective record with some conversion or gaps.",
                "Strong: jointly measured and signed by both parties.",
            ]),
            "bid_unbalancing_concern": score("How strongly does the bid price look unbalanced in a way this variation exploits?", [
                "Not at all: the bid price is in line with the estimate.",
                "Slightly: somewhat above the estimate, nothing more.",
                "Considerably: well above the estimate on an item that grew.",
                "Strongly: far above the estimate on an item whose quantity was visibly understated.",
            ]),
            "quantity_inside_measurement_lines": noul(
                "All of the reported quantity lies within the contract's measurement lines."),
        }


# =========================================================================== schema 5: constructive change
class ConstructiveChangeClaim(Schema):
    name = "constructive_change_claim"
    description = "A contractor claims an informal direction changed the work; entitlement under the change clause."
    surfaces = {"json": 0.5, "claim_letter": 0.25, "owner_file_note": 0.25}

    DIRECTIONS = [
        ("required a Class A form finish on all exposed walls of the lower-level service corridor",
         "Section 03 30 00 requires Class A finish on 'exposed architectural concrete' and Class B elsewhere",
         "the corridor walls are not designated architectural on any drawing", (14000, 60000)),
        ("required a second coat of intumescent fireproofing on all columns",
         "the fire-resistance schedule asks for 2-hour protection on roof-supporting columns only",
         "the columns in question support floors, not the roof", (22000, 95000)),
        ("required crushed-stone pipe bedding instead of screened native material",
         "Section 33 05 05 allows native material meeting the bedding gradation",
         "the native material tested within the gradation", (9000, 45000)),
        ("required the arterial lane closures to be done at night",
         "the traffic specification allows closures from 9:00 to 15:00 on weekdays", "", (18000, 80000)),
        ("required full-height ceramic tile in all washrooms",
         "the finish schedule shows tile to 1.2 m with paint above", "", (12000, 48000)),
        ("required 40 m of curb to be removed and replaced as out of tolerance",
         "the tolerance is 10 mm in 3 m", "the inspector's readings ranged from 6 to 13 mm", (6000, 26000)),
        ("required ultrasonic testing of every complete-penetration weld",
         "Section 05 12 00 calls for testing 10% of complete-penetration welds", "", (15000, 70000)),
        ("required stainless-steel anchors in place of galvanised ones for the handrails",
         "the handrail detail calls for hot-dip galvanised anchors", "", (4000, 15000)),
        ("required the trench to be shored with steel boxes instead of sloped back",
         "the contract leaves the excavation support method to the contractor, subject to the safety plan", "",
         (20000, 90000)),
    ]
    DIRECTION_KIND = ["building", "building", "civil", "civil", "building", "civil", None, None, "civil"]
    DIRECTORS = {"owner_pm": "the owner's project manager (the authorised representative under GC 2.1)",
                 "resident_engineer": "the resident engineer (authorised to interpret the documents, not to change them)",
                 "inspector": "the field inspector (no authority to change the contract)",
                 "architect_observer": "the architect's site observer, in a field report"}

    def sample_factors(self, rng):
        reg = regime(rng)
        di = rng.randrange(len(self.DIRECTIONS))
        reading = pick_w(rng, {"contractor_clear": 4, "ambiguous_latent": 3, "ambiguous_patent": 2, "owner_clear": 1})
        director = pick_w(rng, {"owner_pm": 5, "resident_engineer": 3, "inspector": 2, "architect_observer": 1})
        ratified = chance(rng, 0.35)
        n_days = pick(rng, [5, 7, 14, 21])
        protest = pick_w(rng, {"written_before": 4, "written_late": 3, "after_completion": 2, "none_until_claim": 1})
        protest_days = pick(rng, [1, 2, 4, 9, 15, 30])
        partial = chance(rng, 0.35)
        lo, hi = self.DIRECTIONS[di][3]
        est = round(between(rng, lo, hi), -2)
        ratio = round(between(rng, 0.95, 1.8), 2)
        backup = pick_w(rng, {"daily_records": 4, "estimate_only": 3, "none": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                reading, director, protest, protest_days, ratio, backup, partial = "contractor_clear", "owner_pm", \
                    "written_before", 1, round(between(rng, 0.97, 1.08), 2), "daily_records", False
            else:
                reading = pick(rng, ["owner_clear", "ambiguous_patent"])
        elif reg == "borderline":
            protest, protest_days = "written_late", n_days + pick(rng, [-1, 0, 1, 2])
            protest_days = max(1, protest_days)
            if protest_days <= n_days:
                protest = "written_before"
            ratio = near(rng, 1.15, 0.12, lo=0.9)
        else:  # conflicting: a strong reading of the spec but a direction from someone without authority, or late
            reading = pick(rng, ["contractor_clear", "ambiguous_latent"])
            if chance(rng, 0.5):
                director, ratified = pick(rng, ["inspector", "architect_observer", "resident_engineer"]), chance(rng, 0.3)
            else:
                protest = pick(rng, ["after_completion", "none_until_claim"])
        if protest == "written_before":
            protest_days = min(protest_days, n_days)
        return {"regime": reg, "direction_index": di, "spec_reading": reading, "director": director,
                "ratified": ratified, "clause_notice_days": n_days, "protest": protest, "protest_days": protest_days,
                "partial": partial, "owner_estimate": est, "claimed": round(est * ratio, -2), "price_ratio": ratio,
                "backup": backup, "project": _project(rng, self.DIRECTION_KIND[di]), "contract": _contract_no(rng),
                "clause": pick(rng, CLAUSE_REFS), "pm": person(rng), "director_name": person(rng),
                "date": rand_date(rng)}

    def _authority(self, f):
        return f["director"] == "owner_pm" or f["ratified"]

    def _protest_ok(self, f):
        return (f["protest"] == "written_before" and f["protest_days"] <= f["clause_notice_days"]) or f["ratified"]

    def intended(self, f):
        clarity = {"owner_clear": 0, "ambiguous_patent": 1, "ambiguous_latent": 2, "contractor_clear": 3}[f["spec_reading"]]
        if clarity <= 1:
            ruling = "no_change_contract_required"
        elif not self._authority(f):
            ruling = "deny_direction_lacked_authority"
        elif not self._protest_ok(f):
            ruling = "deny_late_protest"
        elif f["partial"] or clarity == 2 and f["price_ratio"] > 1.3:
            ruling = "recognise_in_part"
        else:
            ruling = "recognise_change"
        lvl = 3 if f["price_ratio"] <= 1.1 else 2 if f["price_ratio"] <= 1.3 else 1 if f["price_ratio"] <= 1.6 else 0
        if f["backup"] == "estimate_only":
            lvl -= 1
        elif f["backup"] == "none":
            lvl = 0
        sup = _clamp(lvl)
        if ruling.startswith("deny") or ruling == "no_change_contract_required":
            follow = "close_without_change"
        elif f["backup"] == "none" or (f["backup"] == "estimate_only" and sup <= 1):
            follow = "obtain_cost_records"
        elif sup >= 3 and ruling == "recognise_change":
            follow = "issue_change_order_now"
        else:
            follow = "negotiate_price_first"
        return {"constructive_change_ruling": ruling, "spec_requirement_clarity": clarity,
                "direction_had_authority": "yes" if self._authority(f) else "no",
                "cost_claim_support": sup, "owner_follow_up": follow}

    def _d(self, f):
        return self.DIRECTIONS[f["direction_index"]]

    def _spec_text(self, f):
        what, spec, extra, _ = self._d(f)
        r = f["spec_reading"]
        if r == "contractor_clear":
            return f"{spec}" + (f"; {extra}" if extra else "") + "."
        if r == "owner_clear":
            return (f"{spec}; but Addendum 2 extended the requirement to this work, and the contractor acknowledged "
                    f"the addendum on its bid form.")
        if r == "ambiguous_latent":
            return (f"{spec}; a general note elsewhere can be read the other way, and the conflict only shows when "
                    f"the two sheets are read together.")
        return (f"{spec}; the drawings show the opposite on the same sheet, an obvious conflict that no bidder asked "
                f"about before bid.")

    def _protest_text(self, f):
        return {"written_before": f"written protest {f['protest_days']} day(s) after the direction, before doing the work",
                "written_late": f"written protest {f['protest_days']} days after the direction, with the work under way",
                "after_completion": "first written objection after the work was finished",
                "none_until_claim": "no objection until this claim"}[f["protest"]]

    def _clause(self, f):
        return (f"{f['clause']} (Changes): only the Owner's Authorised Representative may change the Work. If the "
                f"Contractor considers any direction, interpretation or inspection decision to change the Work, it "
                f"shall give written notice within {f['clause_notice_days']} days and before performing the affected "
                f"work, unless the Owner directs otherwise in writing; failing that, the work is at its own cost.")

    def render(self, f, rng, surface):
        ms, ds = _ms(rng), _ds(rng)
        proj, owner, gc, _kind = f["project"]
        what = self._d(f)[0]
        rat = ("the owner's project manager confirmed the direction in the meeting minutes the following week"
               if f["ratified"] else None)
        backup = {"daily_records": "daily crew and material records", "estimate_only": "contractor's estimate only",
                  "none": "no cost breakdown"}[f["backup"]]
        if surface == "json":
            return dumps(prune({
                "project": {"name": proj, "owner": owner, "contractor": gc, "contract": f["contract"]},
                "direction": {"given_by": self.DIRECTORS[f["director"]], "what": what,
                              "date": f["date"].isoformat(), "later_confirmation": rat},
                "contract_documents": self._spec_text(f),
                "contractor_response": {"protest": self._protest_text(f),
                                        "claimed_amount": money(f["claimed"], rng, style=ms), "cost_backup": backup,
                                        "part_of_work_beyond_direction": "claim also covers adjacent work nobody "
                                        "directed" if f["partial"] else None},
                "owner_review": {"independent_estimate": money(f["owner_estimate"], rng, style=ms)},
                "contract_clause": self._clause(f),
            }))
        if surface == "claim_letter":
            head = f"Re: {proj}, Contract {f['contract']} -- Claim for changed work"
            body = [f"On {fmt_date(f['date'], rng, ds)}, {self.DIRECTORS[f['director']]} {what}.",
                    f"In our reading of the contract, {self._spec_text(f)}",
                    f"Our response: {self._protest_text(f)}.",
                    f"We claim {money(f['claimed'], rng, style=ms)}, supported by {backup}.",
                    "The claim includes the adjoining areas we completed to match." if f["partial"] else "",
                    (rat[0].upper() + rat[1:] + ".") if rat else ""]
            return (f"From: {gc}\nTo: {owner}, attn. {f['pm']}\n{head}\n\n" + noisy(join_sentences(body), rng, 0.1)
                    + f"\n\n[Owner's estimate: {money(f['owner_estimate'], rng, style=ms)}]\n\n{self._clause(f)}")
        para = [f"File note, {proj}.", f"Claim from {gc}: {self.DIRECTORS[f['director']]} {what} "
                f"({fmt_date(f['date'], rng, ds)}).", f"Documents: {self._spec_text(f)}",
                f"Contractor's response at the time: {self._protest_text(f)}.",
                (rat[0].upper() + rat[1:] + ".") if rat else "No owner confirmation of the direction on file.",
                f"Claimed {money(f['claimed'], rng, style=ms)} ({backup}); our estimate "
                f"{money(f['owner_estimate'], rng, style=ms)}.",
                "Part of the claimed work goes beyond what was directed." if f["partial"] else ""]
        return (f"OWNER FILE NOTE -- {f['pm']}\n\n" + noisy(join_sentences(para), rng, 0.1) + "\n\n" + self._clause(f))

    def questions(self):
        return {
            "constructive_change_ruling": choice("How should the owner rule on this claim?", {
                "deny_direction_lacked_authority": "Deny: the direction came from someone without authority and was "
                                                   "never confirmed by someone with it.",
                "deny_late_protest": "Deny: the contractor did not object in writing in time, as the clause requires.",
                "no_change_contract_required": "No change: read correctly, the contract already required this work.",
                "recognise_change": "Recognise the claim: the direction changed the work and the contract was followed.",
                "recognise_in_part": "Recognise part of the claim and reject the rest.",
            }),
            "spec_requirement_clarity": score("How strongly do the contract documents support the contractor's reading?", [
                "Not at all: the documents clearly required the directed work.",
                "Weakly: the conflict was obvious at bid and the contractor never asked.",
                "Reasonably: a hidden ambiguity that the contractor read sensibly.",
                "Clearly: the documents plainly required less than was directed.",
            ]),
            "direction_had_authority": noul(
                "The direction came from, or was confirmed by, someone with authority to change the contract.",
                yes="The owner's authorised representative gave the direction or confirmed it afterwards.",
                no="Only someone without change authority gave it, and nobody with authority confirmed it."),
            "cost_claim_support": score("How well is the claimed amount supported?", [
                "Unsupported: no breakdown, or far above the owner's estimate.",
                "Thin: an estimate only, or well above the owner's figure.",
                "Fair: records exist but the amount needs adjusting.",
                "Strong: contemporaneous records and close to the owner's estimate.",
            ]),
            "owner_follow_up": choice("What should the owner do next on this claim?", {
                "close_without_change": "Close the claim without a change order.",
                "issue_change_order_now": "Issue a change order now at the claimed amount.",
                "negotiate_price_first": "Accept entitlement but negotiate the amount before issuing a change order.",
                "obtain_cost_records": "Ask for contemporaneous cost records before discussing the amount.",
            }),
        }


WORKFLOW = Workflow(
    name="construction_change_order_review",
    title="Construction change-order review",
    description="Entitlement, pricing reasonableness and schedule impact of construction changes, judged against the "
                "contract's written change, notice and time clauses.",
    domain="construction / contract administration",
    schemas=[DifferingSiteConditions(), OwnerDirectedChangePricing(), TimeExtensionRequest(),
             QuantityVariationRepricing(), ConstructiveChangeClaim()],
    notes="Contract modifications BEFORE they are agreed: entitlement, price and time. No invoices, pay applications, "
          "progress payments, retainage or payment release; nothing is reconciled against a purchase order or a "
          "delivery. Fictional projects, parties and contract clauses, quoted in every state.",
)
