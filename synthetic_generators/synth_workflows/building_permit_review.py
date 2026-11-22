"""Building permit review: does a project in a (fictional) municipality get its permit, inspection sign-off or
certificate of occupancy, and on what terms?

Plan review against invented zoning districts, application intake, field inspections at hold points, variance
staff reports and occupancy certificates. Jurisdictions, codes, districts and people are invented; the numbers
(setbacks, coverage, heights, bonds) follow the usual shape of a small-city zoning and building code. Each schema
draws a `regime` first: most cases sit on a dimensional limit, carry a designer's claim the drawings contradict,
or leave a document or a measurement method unstated.
"""
from __future__ import annotations

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

CITIES = ["City of Valewood", "Town of Brackenmoor", "City of Quillhaven", "Marrowfield Township", "City of Pellbridge",
          "Town of Wexmoor", "City of Dunhollow", "Village of Corriston"]
STREETS = ["Linden Ave", "Quarry Rd", "Heron St", "Maple Ct", "Birchwood Dr", "Harbor View Ln", "Orchard Way",
           "Kestrel St", "Millrace Rd", "Tamarack Pl", "Sorrel Ave", "Juniper Ln"]
# district: front, side, rear setbacks (ft), max height (ft), max lot coverage (%), description
DISTRICTS = {
    "R-1": (25, 7, 25, 30, 35, "single-family residential"),
    "R-2": (20, 5, 20, 32, 40, "small-lot residential"),
    "RM-3": (15, 8, 15, 45, 55, "multi-family residential"),
    "MU-2": (0, 0, 10, 55, 80, "mixed use"),
    "C-2": (10, 0, 15, 40, 70, "general commercial"),
}
RES = ["R-1", "R-2", "RM-3"]


def _address(rng):
    return f"{rng.randint(12, 4980)} {pick(rng, STREETS)}"


def _city(rng):
    return pick(rng, CITIES)


def _usd(x, rng):
    return money(x, rng, style=pick(rng, ["sym", "sym", "code"]))


# =========================================================================== schema 1
class ZoningPlanCheck(Schema):
    name = "zoning_plan_check"
    description = "A residential project's drawings checked against the zoning district's dimensional standards."
    surfaces = {"json": 0.5, "plan_review_sheet": 0.25, "designer_letter": 0.25}

    PROJECTS = {"rear_addition": "one-storey rear addition", "second_storey": "second-storey addition",
                "detached_garage": "detached two-car garage", "detached_adu": "detached accessory dwelling unit",
                "new_house": "new single-family house", "covered_patio": "roofed patio and outdoor kitchen"}

    def sample_factors(self, rng):
        reg = regime(rng)
        dist = pick(rng, RES)
        front, side, rear, hmax, cov_max, _ = DISTRICTS[dist]
        proj = pick(rng, list(self.PROJECTS))
        lot = pick(rng, [4800, 5500, 6250, 7200, 8400, 10000])
        existing = 0 if proj == "new_house" else int(lot * rng.uniform(0.18, 0.3))
        size = {"rear_addition": (250, 600), "second_storey": (0, 90), "detached_garage": (420, 640),
                "detached_adu": (450, 800), "covered_patio": (0, 0)}
        new_fp = rng.randint(*size[proj]) if proj != "new_house" else int(lot * rng.uniform(0.24, 0.34))
        patio = pick(rng, [0, 0, 180, 240, 320]) if proj != "covered_patio" else int(lot * rng.uniform(0.04, 0.07))
        side_prop = round(side + rng.uniform(0.5, 4), 1)
        rear_prop = round(rear + rng.uniform(1, 12), 1)
        hmax = {"detached_garage": 16, "covered_patio": 14, "detached_adu": 22}.get(proj, hmax)   # accessory limits
        height_avg = round(hmax - rng.uniform(0.5, 6), 1) if proj != "rear_addition" else round(rng.uniform(12.5, 17), 1)
        grade_drop = pick(rng, [0.0, 0.0, 0.8, 1.6, 2.4])
        method = pick_w(rng, {"average_grade": 4, "lowest_grade": 3, None: 3})
        survey = pick_w(rng, {"stamped_survey": 5, "assessor_map": 3, None: 2})
        attached = proj in ("rear_addition", "second_storey")
        nonconf = attached and chance(rng, 0.25)     # an existing house wall already inside the side setback
        historic = chance(rng, 0.12)
        claim_complies = chance(rng, 0.7)
        cov_rule = pick_w(rng, {"all_roofed": 4, "enclosed_only": 2, None: 4})
        if reg == "clear":
            if chance(rng, 0.5):
                existing = min(existing, int(lot * cov_max / 100 * 0.55)) if existing else 0
                if proj == "new_house":
                    new_fp = int(lot * cov_max / 100 * 0.75)
                patio, nonconf, survey, method, grade_drop = 0, False, "stamped_survey", "average_grade", 0.0
                side_prop, height_avg = round(side + rng.uniform(2, 5), 1), round(min(height_avg, hmax - 4), 1)
            else:
                side_prop = round(side - rng.uniform(1.5, 3), 1)
                over = int(lot * (cov_max + rng.uniform(4, 9)) / 100)
                if proj == "new_house":
                    new_fp = over
                else:
                    existing = max(existing, over - new_fp)
        elif reg == "borderline":
            target = near(rng, cov_max, 1.2)          # coverage lands within a point or so of the maximum
            if proj == "new_house":
                new_fp = max(900, int(lot * target / 100) - patio)
            else:
                existing = max(600, int(lot * target / 100) - new_fp - patio)
            if chance(rng, 0.5):
                side_prop = near(rng, side, 0.4, digits=1)
            if proj != "rear_addition" and chance(rng, 0.6):
                height_avg = near(rng, hmax - 1.0, 0.8, digits=1)
                grade_drop = pick(rng, [0.8, 1.6, 2.4])
        else:
            # the designer's tidy compliance table against what the drawings actually count
            claim_complies = True
            if not attached or chance(rng, 0.5):
                patio = patio or pick(rng, [220, 300, 360])
                below = int(lot * (cov_max - rng.uniform(0.3, 2.5)) / 100)
                if proj == "new_house":
                    new_fp = below
                else:
                    existing = max(600, below - new_fp)
                cov_rule = pick_w(rng, {"all_roofed": 3, None: 2, "enclosed_only": 1})
            else:
                nonconf = True
                side_prop = round(side - rng.uniform(0.6, 2.2), 1)
            if chance(rng, 0.5):
                survey = None
        cov_counted = round(100 * (existing + new_fp) / lot, 1)
        cov_all = round(100 * (existing + new_fp + patio) / lot, 1)
        f = {"regime": reg, "city": _city(rng), "district": dist, "project": proj, "lot_sqft": lot,
             "existing_footprint_sqft": existing, "new_footprint_sqft": new_fp, "roofed_patio_sqft": patio,
             "coverage_designer_pct": cov_counted, "coverage_all_roofed_pct": cov_all, "coverage_max_pct": cov_max,
             "coverage_rule": cov_rule,
             "side_setback_ft": side_prop, "side_required_ft": side, "rear_setback_ft": rear_prop,
             "rear_required_ft": rear, "front_required_ft": front, "height_avg_grade_ft": height_avg,
             "grade_drop_ft": grade_drop, "height_max_ft": hmax, "code_height_method": method,
             "survey_basis": survey, "existing_nonconforming_wall": nonconf, "historic_district": historic,
             "designer_claims_compliance": claim_complies, "designer": person(rng), "owner": person(rng),
             "address": _address(rng), "permit": ident(rng, "BLD", 6, "-"), "submitted": rand_date(rng),
             "corner_lot": chance(rng, 0.2)}
        # what the drawing set itself contains: spot grade elevations, and whether every yard is dimensioned
        f["spot_grades_shown"] = chance(rng, 0.5)
        f["rear_yard_dimensioned"] = not chance(rng, 0.12)
        return f

    def _submittal(self, f):
        """The intake checklist of what the drawing set contains."""
        return [("boundary basis", {"stamped_survey": "stamped boundary survey", "assessor_map": "scaled from the assessor's "
                                    "parcel map (no survey)", None: "missing: no survey or boundary source given"}[f["survey_basis"]]),
                ("spot grade elevations", "shown on the site plan" if f["spot_grades_shown"] else "not shown"),
                ("yard dimensions", "all yards dimensioned" if f["rear_yard_dimensioned"] else "rear yard not dimensioned")]

    def _height_lines(self, f):
        h = f["height_avg_grade_ft"]
        out = [f"{h} ft above average finished grade"]
        if f["grade_drop_ft"]:
            out.append(f"lowest adjacent grade is {f['grade_drop_ft']} ft below average grade at the rear wall")
        elif f["code_height_method"] == "lowest_grade":
            out.append("grade is level around the structure")
        return out

    def _cov_rule(self, f):
        return {"all_roofed": "all roofed structures count, enclosed or not",
                "enclosed_only": "enclosed buildings only; open-sided roofs excluded", None: None}[f["coverage_rule"]]

    def _method(self, f):
        return {"average_grade": "measured from average finished grade to the highest point of the roof",
                "lowest_grade": "measured from the lowest adjacent finished grade to the highest point of the roof",
                None: None}[f["code_height_method"]]

    def render(self, f, rng, surface):
        proj = self.PROJECTS[f["project"]]
        patio_note = (f"{f['roofed_patio_sqft']} sq ft roofed patio (open sides)" if f["roofed_patio_sqft"] else None)
        side_txt = f"{f['side_setback_ft']} ft" + (" (existing wall at this line retained and extended)"
                                                  if f["existing_nonconforming_wall"] else "")
        survey = {"stamped_survey": "stamped boundary survey", "assessor_map": "scaled from assessor's parcel map",
                  None: None}[f["survey_basis"]]
        if surface == "json":
            return dumps(prune({
                "application": {"permit": f["permit"], "address": f["address"], "jurisdiction": f["city"],
                                "scope": proj, "designer": f["designer"], "submitted": f["submitted"].isoformat(),
                                "designer_statement": "project complies with all zoning standards"
                                if f["designer_claims_compliance"] else None},
                "zoning": {"district": f["district"], "historic_overlay": f["historic_district"] or None,
                           "corner_lot": f["corner_lot"] or None,
                           "standards": {"side_setback_ft": f["side_required_ft"], "rear_setback_ft": f["rear_required_ft"],
                                         "max_height_ft": f["height_max_ft"], "max_lot_coverage_pct": f["coverage_max_pct"],
                                         "height_method": self._method(f), "coverage_counts": self._cov_rule(f)}},
                "drawings": {"lot_area_sqft": f["lot_sqft"], "existing_footprint_sqft": f["existing_footprint_sqft"] or None,
                             "new_footprint_sqft": f["new_footprint_sqft"], "other_roofed_structures": patio_note,
                             "lot_coverage_shown_pct": f["coverage_designer_pct"], "side_setback": side_txt,
                             "rear_setback_ft": f["rear_setback_ft"] if f["rear_yard_dimensioned"] else None,
                             "height": self._height_lines(f), "boundary_basis": survey},
                "submittal_checklist": {k.replace(" ", "_"): v for k, v in self._submittal(f)},
            }))
        if surface == "plan_review_sheet":
            rows = [["Side setback", f"{f['side_required_ft']} ft min", side_txt],
                    ["Rear setback", f"{f['rear_required_ft']} ft min",
                     f"{f['rear_setback_ft']} ft" if f["rear_yard_dimensioned"] else "not dimensioned"],
                    ["Height", f"{f['height_max_ft']} ft max", "; ".join(self._height_lines(f))],
                    ["Lot coverage", f"{f['coverage_max_pct']}% max", f"{f['coverage_designer_pct']}% (per applicant)"]]
            head = kv_lines([("Permit", f["permit"]), ("Address", f["address"]), ("Jurisdiction", f["city"]),
                             ("District", f"{f['district']}" + (", historic overlay" if f["historic_district"] else "")),
                             ("Scope", proj), ("Lot", f"{f['lot_sqft']:,} sq ft"),
                             ("Boundaries", show(survey, rng))], rng)
            tail = []
            if patio_note:
                tail.append(f"Site plan also shows: {patio_note}.")
            if f["existing_footprint_sqft"]:
                tail.append(f"Existing footprint {f['existing_footprint_sqft']:,} sq ft; new {f['new_footprint_sqft']:,} sq ft.")
            m = self._method(f)
            tail.append(f"Code height method: {m}." if m else "Code height method: not cited on the sheet.")
            if self._cov_rule(f):
                tail.append(f"Coverage definition: {self._cov_rule(f)}.")
            tail.append("Submittal checklist: " + "; ".join(f"{k}: {v}" for k, v in self._submittal(f)) + ".")
            return "ZONING PLAN CHECK\n" + head + "\n\n" + table(["standard", "required", "proposed"], rows, rng) + \
                "\n\n" + "\n".join(tail)
        # designer_letter: the applicant's own cover letter
        s = [f"Re: {f['permit']}, {f['address']} ({f['district']})", "",
             f"To the {f['city']} plan reviewer,", "",
             noisy(join_sentences([
                 f"Enclosed are revised drawings for a {proj}.",
                 "In our view the project complies with every zoning standard." if f["designer_claims_compliance"] else
                 "We believe the project is close to compliant and would welcome comments.",
                 f"Lot coverage works out at {f['coverage_designer_pct']}% of the {f['lot_sqft']:,} sq ft lot",
                 (f"(the {f['roofed_patio_sqft']} sq ft patio roof is open-sided, so we have not counted it)."
                  if f["roofed_patio_sqft"] else "."),
                 f"The side yard is {f['side_setback_ft']} ft",
                 ("where the new work continues the line of the existing house wall." if f["existing_nonconforming_wall"]
                  else "."),
                 (f"Rear yard {f['rear_setback_ft']} ft." if f["rear_yard_dimensioned"] else "")
                 + f" Height is {f['height_avg_grade_ft']} ft from average grade"
                 + (f"; the lot falls about {f['grade_drop_ft']} ft toward the rear." if f["grade_drop_ft"] else "."),
                 {"stamped_survey": "Dimensions are from the attached stamped survey.",
                  "assessor_map": "Boundary dimensions were taken from the parcel map.",
                  None: ""}[f["survey_basis"]],
             ]).replace(" .", ".").replace(" (", " ("), rng), "",
             "Regards,", f"{f['designer']}, for the owner {f['owner']}", "",
             f"[Reviewer: {f['district']} standards -- side {f['side_required_ft']} ft, rear {f['rear_required_ft']} ft, "
             f"height {f['height_max_ft']} ft" + (f" ({self._method(f)})" if self._method(f) else "")
             + f", coverage {f['coverage_max_pct']}%" + (f" ({self._cov_rule(f)})" if self._cov_rule(f) else "")
             + (". Historic overlay applies." if f["historic_district"] else ".") + "]",
             "[Intake checklist -- " + "; ".join(f"{k}: {v}" for k, v in self._submittal(f)) + "]"]
        return "\n".join(s)

    def questions(self):
        return {
            "plan_check_result": choice("What should the plan reviewer do with this submission?", {
                "approve_as_drawn": "Approve the drawings as submitted.",
                "approve_with_plan_notes": "Approve, with conditions written onto the plans for the inspector to "
                                           "verify on site.",
                "issue_correction_letter": "Send a correction letter; the designer must revise and resubmit.",
                "route_to_zoning_board": "Stop the review; the design as drawn cannot be approved without zoning relief.",
            }),
            "dimensional_conformity": score("How well does the design meet the district's dimensional standards?", [
                "Fully: every standard is met with room to spare.",
                "Narrowly: everything is met, some standards by very little.",
                "Short by a small amount that a minor redesign would fix.",
                "Materially noncompliant: meeting the standards would mean a different project.",
            ]),
            "information_sufficiency": score("How fully do the submitted documents let the reviewer verify "
                                             "compliance?", [
                "Poorly: the boundary basis is missing or a required yard is not dimensioned.",
                "Partly: everything is dimensioned, but boundaries are scaled from a parcel map and no spot grades "
                "are shown.",
                "Mostly: one gap remains: boundaries scaled from a parcel map, or no spot grades.",
                "Completely: a stamped survey, spot grades and every yard dimensioned.",
            ]),
            "coverage_within_limit": noul(
                "The project's lot coverage, counted the way the code counts it, is within the district maximum.",
                yes="Counting every roofed structure the code includes, coverage is at or below the maximum.",
                no="Once everything the code counts is included, coverage exceeds the maximum or cannot be shown to "
                   "be within it."),
            "height_complies": noul(
                "The building height, measured by the code's method, is within the district limit."),
        }


# =========================================================================== schema 2
PERMIT_TYPES = {
    "interior_remodel": ("interior remodel of an existing house", "residential"),
    "reroof": ("tear-off and re-roof", "residential"),
    "solar_pv": ("rooftop solar PV system", "residential"),
    "tenant_improvement": ("commercial tenant improvement", "commercial"),
    "new_house": ("new single-family dwelling", "residential"),
    "change_of_use": ("change of use, retail to restaurant", "commercial"),
    "retaining_wall": ("site retaining wall", "residential"),
}


class PermitIntakeScreening(Schema):
    name = "permit_intake_screening"
    description = "A building permit application at intake: complete enough to start review, and on which track."
    surfaces = {"json": 0.5, "intake_checklist": 0.25, "portal_record": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ptype = pick(rng, list(PERMIT_TYPES))
        kind = PERMIT_TYPES[ptype][1]
        valuation = {"interior_remodel": 45000, "reroof": 18000, "solar_pv": 26000, "tenant_improvement": 160000,
                     "new_house": 520000, "change_of_use": 210000, "retaining_wall": 30000}[ptype]
        valuation = int(round(valuation * rng.uniform(0.5, 1.6), -2))
        area = {"interior_remodel": 900, "tenant_improvement": 3200, "new_house": 2600, "change_of_use": 2400}.get(ptype)
        area = int(area * rng.uniform(0.6, 1.5)) if area else None
        structural = pick_w(rng, {"none": 4, "minor": 3, "major": 2})
        if ptype in ("new_house", "retaining_wall"):
            structural = "major" if ptype == "new_house" else pick(rng, ["minor", "major"])
        wall_ft = round(rng.uniform(2.5, 6.5), 1) if ptype == "retaining_wall" else None
        kw = round(rng.uniform(5, 14), 1) if ptype == "solar_pv" else None
        stamp = pick_w(rng, {"engineer": 3, "architect": 2, "none": 4, "expired": 1})
        docs = {"site_plan": chance(rng, 0.85), "structural_calcs": chance(rng, 0.7), "energy_forms": chance(rng, 0.75),
                "owner_authorization": chance(rng, 0.85)}
        not_applicable = {"retaining_wall": ["energy_forms"], "solar_pv": ["energy_forms"], "reroof": [],
                          "interior_remodel": [], "new_house": [], "tenant_improvement": [], "change_of_use": []}[ptype]
        licence = pick_w(rng, {"active": 6, "expired_recently": 1, "renewal_pending": 1, "owner_builder": 2})
        fees = pick_w(rng, {"paid": 6, "partial": 2, "unpaid": 1})
        days = rng.randint(1, 30)
        clock = pick(rng, [15, 30])
        resub = pick(rng, [0, 0, 0, 1, 2])
        prior_approval = pick_w(rng, {"not_needed": 6, "pending": 2, "granted": 2}) if kind == "commercial" or ptype == "retaining_wall" \
            else "not_needed"
        if reg == "clear":
            if chance(rng, 0.5):
                docs = {k: True for k in docs}
                licence, fees, days, resub, prior_approval = "active", "paid", rng.randint(1, 4), 0, "not_needed" \
                    if prior_approval == "not_needed" else "granted"
                stamp = "engineer" if structural != "none" else pick(rng, ["none", "architect"])
            else:
                docs["site_plan"], docs["structural_calcs"], fees = False, False, "unpaid"
                stamp = "none" if structural == "major" else stamp
        elif reg == "borderline":
            days = near(rng, clock - 2, 2, lo=1, integer=True)
            if ptype == "solar_pv":
                kw = near(rng, 10.0, 0.6, lo=4)
            elif ptype == "retaining_wall":
                wall_ft = near(rng, 4.0, 0.4, lo=2)
            else:
                valuation = int(round(near(rng, 50000 if kind == "residential" else 150000, 6000), -2))
            if chance(rng, 0.5):
                docs[pick(rng, list(docs))] = False
        else:
            # a polished package with one problem an intake clerk can miss, or the reverse
            if chance(rng, 0.5):
                docs = {k: True for k in docs}
                fees = "paid"
                pick_bad = pick(rng, ["licence", "stamp", "prior"])
                if pick_bad == "licence":
                    licence = pick(rng, ["expired_recently", "renewal_pending"])
                elif pick_bad == "stamp":
                    stamp, structural = "expired", pick(rng, ["minor", "major"])
                else:
                    prior_approval = "pending"
            else:
                docs["energy_forms"], docs["owner_authorization"] = False, pick(rng, [True, False])
                stamp, licence = "engineer", "active"
                days = rng.randint(clock - 3, clock + 2)
        if structural == "none":
            not_applicable = not_applicable + ["structural_calcs"]
        for k in not_applicable:
            docs[k] = None
        return {"regime": reg, "city": _city(rng), "permit_type": ptype, "occupancy": kind,
                "valuation_usd": valuation, "floor_area_sqft": area, "structural_scope": structural,
                "retaining_wall_height_ft": wall_ft, "solar_kw": kw, "design_stamp": stamp, "documents": docs,
                "contractor_licence": licence, "fees": fees, "days_since_submitted": days,
                "completeness_clock_days": clock, "resubmittal_number": resub,
                "prior_approval": prior_approval, "applicant": person(rng), "contractor": person(rng),
                "address": _address(rng), "application": ident(rng, "APP", 6, "-"), "received": rand_date(rng)}

    def _scope(self, f):
        s = PERMIT_TYPES[f["permit_type"]][0]
        if f["solar_kw"]:
            s += f", {f['solar_kw']} kW DC"
        if f["retaining_wall_height_ft"]:
            s += f", {f['retaining_wall_height_ft']} ft exposed height"
        if f["floor_area_sqft"]:
            s += f", {f['floor_area_sqft']:,} sq ft"
        return s

    STRUCT = {
        "interior_remodel": ("no structural work", "one new header and a relocated post",
                             "a bearing wall removed and replaced with a beam"),
        "reroof": ("no structural work", "decking and two damaged rafters replaced", "new roof framing over the rear half"),
        "solar_pv": ("ballasted racking, no roof penetrations", "racking attached to existing rafters",
                     "rafters sistered to carry the array"),
        "tenant_improvement": ("non-structural partitions only", "a new opening in a masonry wall",
                               "a mezzanine added over the stockroom"),
        "new_house": ("-", "-", "new foundations and framing throughout"),
        "change_of_use": ("no structural work", "new rooftop exhaust fan curb", "floor strengthened for walk-in cooler"),
        "retaining_wall": ("gravity block wall, no footing design", "reinforced block wall on a standard footing",
                           "engineered footing and wall design"),
    }

    def _structural(self, f):
        return self.STRUCT[f["permit_type"]][["none", "minor", "major"].index(f["structural_scope"])]

    def _stamp(self, f):
        return {"engineer": "sealed by a licensed engineer", "architect": "sealed by a licensed architect",
                "none": "no professional seal", "expired": "sealed, but the seal's licence lapsed last month"}[f["design_stamp"]]

    def _licence(self, f):
        return {"active": "contractor licence active", "expired_recently": "contractor licence expired 9 days ago",
                "renewal_pending": "contractor licence renewal filed, not yet processed",
                "owner_builder": "owner-builder (no contractor)"}[f["contractor_licence"]]

    def _prior(self, f):
        return {"not_needed": None, "pending": "planning / zoning approval applied for, hearing not yet held",
                "granted": "planning approval granted"}[f["prior_approval"]]

    RULE = ("Express counter review: re-roofs, solar up to 10 kW, interior work with no structural change and "
            "valuation under $50,000. Walls over 4 ft, structural alterations and all commercial work need a sealed "
            "design.")

    def render(self, f, rng, surface):
        d = f["documents"]
        if surface == "json":
            return dumps(prune({
                "application": {"id": f["application"], "address": f["address"], "jurisdiction": f["city"],
                                "scope": self._scope(f), "structural": self._structural(f),
                                "valuation_usd": f["valuation_usd"], "applicant": f["applicant"],
                                "resubmittal": f["resubmittal_number"] or None},
                "documents": {"drawings": self._stamp(f), "site_plan": d["site_plan"],
                              "structural_calculations": "n/a" if d["structural_calcs"] is None else d["structural_calcs"],
                              "energy_compliance": "n/a" if d["energy_forms"] is None else d["energy_forms"],
                              "owner_authorization": d["owner_authorization"]},
                "contractor": {"name": None if f["contractor_licence"] == "owner_builder" else f["contractor"],
                               "status": self._licence(f)},
                "fees": f["fees"], "prior_approvals": self._prior(f),
                "clock": {"days_since_submitted": f["days_since_submitted"],
                          "completeness_determination_due_within_days": f["completeness_clock_days"]},
                "intake_rules": self.RULE if chance(rng, 0.6) else None,
            }))
        if surface == "intake_checklist":
            mark = lambda b: "[n/a]" if b is None else ("[x]" if b else "[ ]")  # noqa: E731
            lines = [f"PERMIT INTAKE CHECKLIST -- {f['city']}",
                     kv_lines([("Application", f["application"]), ("Received", fmt_date(f["received"], rng)),
                               ("Address", f["address"]), ("Scope", self._scope(f)),
                               ("Valuation", _usd(f["valuation_usd"], rng)), ("Structural", self._structural(f))], rng),
                     "",
                     f"{mark(f['design_stamp'] in ('engineer', 'architect'))} Drawings: {self._stamp(f)}",
                     f"{mark(d['site_plan'])} Site plan", f"{mark(d['structural_calcs'])} Structural calculations",
                     f"{mark(d['energy_forms'])} Energy compliance forms",
                     f"{mark(d['owner_authorization'])} Owner authorization",
                     f"{mark(f['contractor_licence'] in ('active', 'owner_builder'))} {self._licence(f)}",
                     f"{mark(f['fees'] == 'paid')} Fees: {f['fees']}"]
            if self._prior(f):
                lines.append(f"[?] {self._prior(f)}")
            lines += ["", f"Day {f['days_since_submitted']} of {f['completeness_clock_days']} for the completeness "
                          f"determination." + (f" Resubmittal #{f['resubmittal_number']}." if f["resubmittal_number"] else "")]
            return "\n".join(lines)
        missing = [k.replace("_", " ") for k, v in d.items() if v is False]
        note = noisy(join_sentences([
            f"Applicant {f['applicant']} uploaded a {self._scope(f)} package.",
            f"Drawings {self._stamp(f)}; {self._structural(f)}.",
            (f"Not uploaded: {', '.join(missing)}." if missing else "All listed documents uploaded."),
            f"{self._licence(f).capitalize()}.", f"Fees {f['fees']}.",
            (self._prior(f).capitalize() + "." if self._prior(f) else ""),
        ]), rng, 0.15)
        return "\n".join([f"[{f['city']} ePermits] {f['application']}  status: PENDING INTAKE",
                          f"Submitted {f['days_since_submitted']} days ago | completeness due in "
                          f"{f['completeness_clock_days']} days | valuation {_usd(f['valuation_usd'], rng)}", "",
                          note, "", "Intake rules: " + self.RULE])

    def questions(self):
        return {
            "intake_ruling": choice("How should intake staff treat this application?", {
                "accept_start_review": "Deem it complete and start the plan-review clock.",
                "accept_with_deferred_items": "Accept it for review and list items that must arrive before the permit "
                                              "is issued.",
                "hold_for_prior_approval": "Hold it until a separate approval it depends on has been granted.",
                "return_incomplete": "Return it as incomplete; review does not start until it is resubmitted.",
            }),
            "review_track": choice("Which review path should the application go on?", {
                "express_counter": "Same-day counter review by one plans examiner.",
                "standard_single": "Standard review by the building discipline only.",
                "multi_discipline": "Full review routed to building, fire, planning and engineering.",
                "major_project_team": "A coordinated major-project team with a pre-review meeting.",
            }),
            "missing_item_weight": score("How much do the gaps in this submission matter to the review?", [
                "Not at all: nothing material is missing.",
                "A little: clerical items that do not affect the technical review.",
                "Considerably: documents the reviewers need for part of the review.",
                "Critically: the review cannot meaningfully start without them.",
            ]),
            "completeness_deadline_pressure": score("How close is the department to its deadline for telling the "
                                                    "applicant whether the application is complete?", [
                "No pressure: it arrived within the last few days.",
                "Some: past the halfway point of the completeness period.",
                "High: within a few days of the deadline.",
                "Past due or due today: it may be deemed complete by default.",
            ]),
            "sealed_design_required": noul(
                "The drawings for this work must carry the seal of a licensed design professional.",
                yes="The scope includes structural design, commercial work or another trigger in the local rules.",
                no="The scope is simple enough that the local rules accept unsealed drawings."),
        }


# =========================================================================== schema 3
STAGES = {
    "footing": ("footing and foundation", "concrete pour", "buried"),
    "underground_plumbing": ("underground plumbing", "slab pour over the lines", "buried"),
    "rough_framing": ("rough framing", "insulation and drywall", "enclosed"),
    "shear_wall": ("shear wall nailing", "sheathing covered by siding", "enclosed"),
    "insulation": ("insulation", "drywall", "enclosed"),
    "rough_electrical": ("rough electrical", "drywall", "enclosed"),
}
# stage -> deficiency class -> findings an inspector might write at that stage
DEFICIENCIES = {
    "footing": {
        "cosmetic": ["inspection card not posted on site", "loose soil and debris in the trench corners"],
        "minor": ["two rebar chairs missing along the north footing", "form boards not braced at one corner"],
        "significant": ["rebar lap length short at the south-west corner",
                        "footing 10 in deep where the plans show 12 in",
                        "anchor bolt spacing over the plan maximum on the garage wall"],
        "hazard": ["trench walls over 5 ft deep with no shoring while workers were in it",
                   "open excavation beside the public sidewalk with no barrier"]},
    "underground_plumbing": {
        "cosmetic": ["inspection card not posted on site", "pipe ends not capped against debris"],
        "minor": ["test gauge reading not written on the card", "one cleanout cap loose",
                  "bedding sand thin under one run"],
        "significant": ["kitchen drain slope under 1/4 in per ft", "test pressure dropped during the 15-minute hold",
                        "a short-radius fitting at a horizontal change of direction"],
        "hazard": ["trench over 5 ft deep with no shoring while a worker was in it"]},
    "rough_framing": {
        "cosmetic": ["address numbers not posted", "lumber stacked against the neighbour's fence"],
        "minor": ["two missing nail plates over pipes in the studs", "one closet header missing a jack stud"],
        "significant": ["a joist notched beyond the allowed depth under the bath", "fire blocking missing at the soffit",
                        "beam bearing shorter than the plans at the stair opening"],
        "hazard": ["a temporary shore removed under a loaded beam", "no guardrail at the open second-floor edge"]},
    "shear_wall": {
        "cosmetic": ["approved plans not on site, inspector used the city copy"],
        "minor": ["a few overdriven nails at one panel edge", "one hold-down nut only finger tight"],
        "significant": ["hold-down straps at two shear wall ends not installed",
                        "edge nailing at 6 in where the schedule calls for 3 in", "panel joints not blocked"],
        "hazard": ["two-storey wall panels standing unbraced with high winds forecast"]},
    "insulation": {
        "cosmetic": ["insulation certificate not posted"],
        "minor": ["depth markers missing in the attic", "gaps around two window rough openings"],
        "significant": ["R-13 batts where the energy forms show R-21", "no insulation at the rim joist on the east wall"],
        "hazard": ["insulation packed against recessed lights not rated for contact"]},
    "rough_electrical": {
        "cosmetic": ["panel directory not started"],
        "minor": ["one outlet box not flush with the stud face", "cables not stapled within 12 in of two boxes"],
        "significant": ["bathroom receptacles on a shared lighting circuit", "no bonding jumper at the water heater",
                        "kitchen counter circuits without the required protection"],
        "hazard": ["an energised splice with no box in the crawlspace",
                   "service panel energised with the dead-front cover off"]},
}


class FieldInspectionHoldPoint(Schema):
    name = "field_inspection_hold_point"
    description = "A field inspection at a construction hold point: pass, pass with corrections, partial, fail or stop work."
    surfaces = {"json": 0.45, "inspection_report": 0.3, "inspector_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        stage = pick(rng, list(STAGES))
        worst = pick_w(rng, {"none": 2, "cosmetic": 2, "minor": 4, "significant": 3, "hazard": 1})
        n_items = 0 if worst == "none" else pick(rng, [1, 2, 3])
        covered_early = chance(rng, 0.45)
        special = pick_w(rng, {"not_required": 5, "received": 2, "missing": 2, "pending_lab": 1})
        if not self.SPECIAL[stage]:
            special = "not_required"
        partial_ready = chance(rng, 0.25)
        attempt = pick(rng, [1, 1, 1, 2, 3])
        next_step_hours = pick(rng, [12, 20, 48, 120])
        access_limited = chance(rng, 0.15)
        contractor_note = pick_w(rng, {"none": 4, "will_fix_later": 2, "disputes": 1, "already_fixed": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                worst, n_items, covered_early, special, partial_ready, access_limited = "none", 0, False, \
                    pick(rng, ["not_required", "received"]), False, False
                contractor_note = "none"
            else:
                worst, n_items = pick(rng, ["significant", "hazard"]), pick(rng, [2, 3])
        elif reg == "borderline":
            worst = pick(rng, ["minor", "significant"])
            n_items = pick(rng, [1, 2])
            next_step_hours = pick(rng, [12, 20])
            if chance(rng, 0.4):
                special = "pending_lab"
        else:
            # a clean-looking stage with something hidden, or a failing list the contractor says is done
            if chance(rng, 0.5):
                worst, n_items = pick(rng, ["cosmetic", "minor"]), 1
                covered_early = True
            else:
                worst, n_items, contractor_note = "significant", pick(rng, [1, 2]), "already_fixed"
            if chance(rng, 0.4):
                access_limited = True
        if worst in ("none",):
            n_items = 0
        f = {"regime": reg, "city": _city(rng), "stage": stage, "worst_deficiency": worst,
             "n_deficiencies": n_items, "work_covered_before_inspection": covered_early,
             "special_inspection": special, "partially_ready": partial_ready, "attempt": attempt,
             "hours_to_next_step": next_step_hours, "access_limited": access_limited,
             "contractor_note": contractor_note, "permit": ident(rng, "BLD", 6, "-"), "inspector": person(rng),
             "contractor": person(rng), "address": _address(rng), "date": rand_date(rng),
             "special_type": pick(rng, self.SPECIAL[stage]) if self.SPECIAL[stage] else None}
        # for work covered early: did anyone document it before it was covered?
        f["covered_documented"] = covered_early and chance(rng, 0.35)
        return f

    SPECIAL = {"footing": ["high-strength concrete cylinders", "soil compaction under the footings"],
               "underground_plumbing": ["trench backfill compaction"],
               "rough_framing": ["structural steel moment-frame welding", "epoxy anchor installation"],
               "shear_wall": ["shear wall nailing", "epoxy hold-down anchors"],
               "insulation": [], "rough_electrical": []}
    ACCESS = {"footing": "north trench partly flooded, bottom not visible",
              "underground_plumbing": "a section of trench already partly backfilled",
              "rough_framing": "attic access blocked by stored material",
              "shear_wall": "second-floor panels only viewed from the ground",
              "insulation": "vaulted ceiling not reachable without a lift",
              "rough_electrical": "attic access blocked by stored material"}

    def _items(self, f, rng):
        worst = f["worst_deficiency"]
        if worst == "none":
            return []
        order = ["cosmetic", "minor", "significant", "hazard"]
        table_ = DEFICIENCIES[f["stage"]]
        items = [pick(rng, table_[worst])]
        pool = [x for c in order[:order.index(worst) + 1] for x in table_[c] if x not in items]
        rng.shuffle(pool)
        return items + pool[:max(0, f["n_deficiencies"] - 1)]

    def _special(self, f):
        return {"not_required": None, "received": f"special inspection report for {f['special_type']} received",
                "missing": f"special inspection report for {f['special_type']} not on file",
                "pending_lab": f"{f['special_type']}: special inspector attended, written report not yet received"}[f["special_inspection"]]

    def _extras(self, f, rng):
        stage_name, next_step, _ = STAGES[f["stage"]]
        out = []
        if f["work_covered_before_inspection"]:
            out.append(pick(rng, [f"part of the work was already covered before the call ({next_step} started on "
                                  f"one side)", f"{next_step} already under way in one area"]))
            out.append("contractor produced dated photos of that area taken before it was covered, showing it complete"
                       if f["covered_documented"] else "no photos or other record of the covered area before it was covered")
        if f["partially_ready"]:
            out.append("the garage portion is not ready and was not inspected")
        if f["access_limited"]:
            out.append(self.ACCESS[f["stage"]])
        cn = {"none": None, "will_fix_later": "contractor says items will be corrected during the next phase",
              "disputes": "contractor disputes one item and asked for a supervisor review",
              "already_fixed": "contractor says the items were corrected after the photos were taken"}[f["contractor_note"]]
        if cn:
            out.append(cn)
        return out

    def render(self, f, rng, surface):
        stage_name, next_step, _ = STAGES[f["stage"]]
        items = self._items(f, rng)
        extras = self._extras(f, rng)
        sp = self._special(f)
        nxt = f"{next_step} scheduled in {f['hours_to_next_step']} h"
        if surface == "json":
            return dumps(prune({
                "inspection": {"permit": f["permit"], "address": f["address"], "jurisdiction": f["city"],
                               "type": stage_name, "attempt": f["attempt"], "inspector": f["inspector"],
                               "date": f["date"].isoformat()},
                "findings": {"items": items or None, "observations": extras or None,
                             "special_inspection": sp},
                "schedule": {"next_step": nxt},
            }))
        if surface == "inspection_report":
            rows = [("Permit", f["permit"]), ("Site", f["address"]), ("Inspection", f"{stage_name} (attempt {f['attempt']})"),
                    ("Date", fmt_date(f["date"], rng)), ("Inspector", f["inspector"]), ("Contractor", f["contractor"])]
            body = "INSPECTION RECORD -- " + f["city"] + "\n" + kv_lines(rows, rng) + "\n\nCorrections noted:\n"
            body += (bullets(items, rng) if items else "  none") + "\n"
            if extras or sp:
                body += "\nRemarks:\n" + bullets(extras + ([sp] if sp else []), rng) + "\n"
            body += f"\nNext: {nxt}.\nResult: ______"
            return body
        s = [f"{f['address']}, {stage_name}, attempt {f['attempt']}."]
        s.append(("Found " + "; ".join(items) + ".") if items else "Nothing written up.")
        s += [x[0].upper() + x[1:] + "." for x in extras]
        if sp:
            s.append(sp[0].upper() + sp[1:] + ".")
        s.append(f"Crew wants to proceed: {nxt}.")
        return f"Inspector field notes ({f['inspector']}, {fmt_date(f['date'], rng)})\n" + noisy(join_sentences(s), rng, 0.2)

    def questions(self):
        return {
            "inspection_ruling": choice("What result should the inspector record for this inspection?", {
                "approved": "Approve the stage: nothing needs correcting and the next phase of work may proceed.",
                "approved_with_correction_notice": "Approve, because the only items are cosmetic or minor code items "
                                                   "that can be checked at the next inspection.",
                "not_approved_reinspect": "Do not approve: a structural, fire or egress item, or a missing special "
                                          "inspection report, must be dealt with and reinspected.",
                "partial_approval": "Approve only the areas that are ready, compliant and visible; an area that was "
                                    "not ready, or was covered up before inspection, waits for a reinspection.",
                "stop_work_order": "Stop all work on site because a condition is dangerous to workers or the public "
                                   "right now.",
            }),
            "concealment_consequence": score("How much would be hidden from later inspection if work moved on now?", [
                "Little: the next step leaves this work visible.",
                "Some: parts become hard to reach.",
                "Considerable: most of it will be enclosed or buried.",
                "Irreversible: correcting it afterwards would mean demolition or breaking out concrete.",
            ]),
            "deficiency_weight": score("How serious are the problems the inspector found?", [
                "None, or cosmetic only.",
                "Minor code items that do not affect safety.",
                "Significant: structure, fire separation or egress is affected.",
                "Dangerous: a hazard to workers or future occupants exists now.",
            ]),
            "special_inspection_gap": noul(
                "A required third-party special inspection result is still missing for this stage.",
                yes="The stage needs a special inspection and its report or lab result is not yet on file.",
                no="No special inspection is needed, or its report is on file."),
            "concealed_work_must_be_opened": noul(
                "Work already covered up must be opened so that it can be inspected.",
                yes="Part of the work was covered before the inspection and nothing, such as dated photos, shows how "
                    "it was built.",
                no="Nothing was covered early, or the covered area is documented by dated photos taken before it was "
                   "covered."),
        }


# =========================================================================== schema 4
STANDARDS = {
    "side_setback": ("side setback", "ft", 1),
    "rear_setback": ("rear setback", "ft", 1),
    "height": ("building height", "ft", -1),
    "fence_height": ("front-yard fence height", "ft", -1),
    "lot_coverage": ("lot coverage", "%", -1),
}


class VarianceStaffReport(Schema):
    name = "variance_staff_report"
    description = "A dimensional variance application on which planning staff must write a recommendation."
    surfaces = {"json": 0.5, "staff_memo": 0.25, "application_form": 0.25}

    CONDITIONS = {"irregular_lot": "wedge-shaped lot narrowing to 38 ft at the rear",
                  "steep_slope": "lot falls 14 ft from front to back", "easement": "10 ft utility easement across the rear",
                  "narrow_lot": "lot is 32 ft wide where the district assumes 50 ft", "none": None}
    PURPOSES = {"side_setback": ["a two-car garage", "a covered porch", "a family room addition"],
                "rear_setback": ["a family room addition", "a primary-suite addition", "a sunroom"],
                "height": ["a second-storey addition", "a rooftop deck with stair enclosure"],
                "fence_height": ["a taller privacy fence along the street"],
                "lot_coverage": ["a two-car garage", "a covered patio", "a family room addition"]}

    def sample_factors(self, rng):
        reg = regime(rng)
        dist = pick(rng, RES)
        std = pick(rng, list(STANDARDS))
        front, side, rear, hmax, cov, _ = DISTRICTS[dist]
        required = {"side_setback": side, "rear_setback": rear, "height": hmax, "fence_height": 4.0,
                    "lot_coverage": cov}[std]
        dev_pct = round(rng.uniform(4, 45), 1) if std != "height" else round(rng.uniform(3, 25), 1)
        condition = pick_w(rng, {"irregular_lot": 2, "steep_slope": 2, "easement": 2, "narrow_lot": 2, "none": 3})
        self_created = pick_w(rng, {"no": 6, "built_without_permit": 2, "bought_knowing": 1})
        support, oppose = rng.randint(0, 6), rng.randint(0, 5)
        abutter = pick_w(rng, {"supports": 3, "opposes": 2, "silent": 3})
        precedents = pick(rng, [0, 0, 1, 2, 4])
        alternative = pick_w(rng, {"none_feasible": 2, "costly": 3, "feasible": 3, None: 2})
        if reg == "clear":
            if chance(rng, 0.5):
                dev_pct, condition, self_created, oppose, abutter, alternative = round(rng.uniform(4, 9), 1), \
                    pick(rng, ["irregular_lot", "steep_slope", "narrow_lot"]), "no", 0, "supports", "none_feasible"
            else:
                dev_pct, condition, self_created, abutter, alternative = round(rng.uniform(22, 30 if std == "height" else 50), 1), "none", \
                    pick(rng, ["built_without_permit", "no"]), "opposes", "feasible"
                oppose = max(oppose, 3)
        elif reg == "borderline":
            dev_pct = near(rng, 10, 2.5, lo=2)              # around the administrative-adjustment cap
            if chance(rng, 0.5):
                support, oppose = 2, 2
        else:
            if chance(rng, 0.5):
                condition, self_created = pick(rng, ["steep_slope", "irregular_lot"]), "built_without_permit"
            else:
                condition, alternative, abutter = "none", "costly", "supports"
                support, oppose = rng.randint(4, 8), 0
        sign = STANDARDS[std][2]
        requested = round(required * (1 - sign * dev_pct / 100), 1)
        if std == "lot_coverage":
            requested = round(required * (1 + dev_pct / 100), 1)
        f = {"regime": reg, "city": _city(rng), "district": dist, "standard": std, "required": required,
             "requested": requested, "deviation_pct": dev_pct, "lot_condition": condition,
             "self_created": self_created, "letters_support": support, "letters_oppose": oppose,
             "abutting_neighbour": abutter, "similar_variances_on_street": precedents,
             "compliant_alternative": alternative, "purpose": pick(rng, self.PURPOSES[std]),
             "applicant": person(rng), "planner": person(rng), "case": ident(rng, "VAR", 4, "-"),
             "address": _address(rng), "hearing": rand_date(rng)}
        f["abutter_concern"] = pick(rng, ["light", "privacy", "space"])   # what an objecting abutter says
        return f

    CONCERN = {"light": "says it would shade their kitchen and back garden for much of the day",
               "privacy": "says it would look straight into their bedroom and yard",
               "space": "says it would leave a narrow, dark gap along their side wall"}

    def _abut(self, f):
        if f["abutting_neighbour"] == "opposes":
            return f"abutting owner opposes: {self.CONCERN[f['abutter_concern']]}"
        return {"supports": "abutting owner supports", "silent": "abutting owner has not responded"}[f["abutting_neighbour"]]

    def _dev(self, f):
        return f"a {f['deviation_pct']}% deviation from the standard"

    def _std(self, f):
        name, unit, _ = STANDARDS[f["standard"]]
        return name, f"{f['required']}{' ' + unit if unit == 'ft' else unit}", f"{f['requested']}{' ' + unit if unit == 'ft' else unit}"

    def _alt(self, f):
        return {"none_feasible": "applicant states no compliant layout fits the lot",
                "costly": "a compliant layout exists but would cost about 40% more",
                "feasible": "staff sketched a compliant layout of similar size", None: None}[f["compliant_alternative"]]

    def _history(self, f):
        return {"no": None, "built_without_permit": "part of the work was built before any permit was sought",
                "bought_knowing": "the lot was bought last year with the constraint disclosed"}[f["self_created"]]

    RULE = "Staff may grant an administrative adjustment of up to 10% of a dimensional standard without a hearing."

    def render(self, f, rng, surface):
        name, req, ask = self._std(f)
        cond = self.CONDITIONS[f["lot_condition"]]
        abut = self._abut(f)
        if surface == "json":
            return dumps(prune({
                "application": {"case": f["case"], "address": f["address"], "jurisdiction": f["city"],
                                "district": f["district"], "applicant": f["applicant"], "purpose": f["purpose"],
                                "relief_sought": {"standard": name, "required": req, "proposed": ask,
                                                  "deviation": f"{f['deviation_pct']}%"}},
                "site": {"physical_condition": cond or "no unusual site condition identified",
                         "history": self._history(f)},
                "public_input": {"letters_in_support": f["letters_support"], "letters_opposed": f["letters_oppose"],
                                 "abutting_owner": abut},
                "staff_analysis": {"compliant_alternative": self._alt(f),
                                   "similar_variances_granted_on_street": f["similar_variances_on_street"],
                                   "adjustment_rule": self.RULE if chance(rng, 0.6) else None},
            }))
        if surface == "staff_memo":
            s = [f"Case {f['case']} -- {f['address']} ({f['district']}).",
                 f"The applicant, {f['applicant']}, seeks relief from the {name} standard ({req} required, {ask} proposed, "
                 f"{self._dev(f)}) to build {f['purpose']}.",
                 (f"Site: {cond}." if cond else "Staff did not identify an unusual physical condition on the site."),
                 (self._history(f)[0].upper() + self._history(f)[1:] + "." if self._history(f) else ""),
                 (self._alt(f)[0].upper() + self._alt(f)[1:] + "." if self._alt(f) else ""),
                 f"Letters received: {f['letters_support']} in support, {f['letters_oppose']} opposed; {abut}.",
                 (f"{f['similar_variances_on_street']} similar variance(s) have been granted on this street."
                  if f["similar_variances_on_street"] else ""),
                 self.RULE]
            return (f"MEMORANDUM\nTo: Zoning Board of Appeals\nFrom: {f['planner']}, Planning Division\n"
                    f"Hearing: {fmt_date(f['hearing'], rng)}\n\n" + noisy(join_sentences(s), rng, 0.1) +
                    "\n\nRecommendation: [to be completed]")
        rows = [("Case no.", f["case"]), ("Property", f"{f['address']}, {f['city']}"), ("Zoning district", f["district"]),
                ("Owner/applicant", f["applicant"]), ("Proposed work", f["purpose"]),
                ("Standard to be varied", name), ("Code requirement", req), ("Requested", f"{ask} ({f['deviation_pct']}% deviation)"),
                ("What makes your property different?", cond or "(left blank)"),
                ("Was the condition caused by you?", "no" if f["self_created"] == "no" else show(self._history(f), rng)),
                ("Alternatives considered", show(self._alt(f), rng)),
                ("Neighbour letters", f"{f['letters_support']} for / {f['letters_oppose']} against ({abut})"),
                ("Similar variances granted on this street", f["similar_variances_on_street"])]
        return "VARIANCE APPLICATION\n" + kv_lines(rows, rng, skip_none=False)

    def questions(self):
        return {
            "staff_recommendation": choice("What should planning staff recommend to the zoning board?", {
                "recommend_approval": "Recommend granting the variance as requested.",
                "recommend_approval_with_conditions": "Recommend granting it with conditions such as screening, "
                                                      "materials or a recorded covenant.",
                "recommend_denial": "Recommend refusing the variance.",
                "recommend_lesser_relief": "Recommend granting a smaller deviation than the one requested.",
                "request_continuance": "Ask the board to continue the case until missing information is supplied.",
            }),
            "hearing_route": choice("How should the request be processed?", {
                "administrative_adjustment": "Staff can grant it as a minor adjustment without a hearing.",
                "consent_agenda": "Place it on the board's consent agenda; nobody is expected to contest it.",
                "full_public_hearing": "Hold a full public hearing; it is contested or substantial.",
                "refer_to_rezoning": "It is too large for a variance and belongs in a rezoning or code amendment.",
            }),
            "hardship_basis": score("How well does the application tie its hardship to the physical conditions of "
                                    "the lot?", [
                "Not at all: no unusual site condition is identified; the reason is preference or convenience.",
                "Weakly: there is a site condition, but staff found a compliant layout of similar size.",
                "Reasonably: a site condition makes compliance harder, and a compliant layout would cost much more "
                "or has not been explored.",
                "Strongly: a site condition leaves no compliant layout that fits the lot.",
            ]),
            "neighbourhood_effect": score("How much would granting the variance affect neighbouring properties "
                                          "and the street?", [
                "Negligible: a small deviation that no neighbour objects to.",
                "Slight: visible next door or from the street, but no abutting owner objects.",
                "Noticeable: an abutting owner objects that it takes light, privacy or space from their property.",
                "Substantial: a large deviation in height, bulk or a street-facing feature, with no similar cases "
                "granted on the street.",
            ]),
            "hardship_self_created": noul(
                "The hardship the applicant relies on was created by the applicant's own actions.",
                yes="The owner built, bought or designed their way into the problem.",
                no="The problem comes from the land itself or from circumstances the owner did not create."),
        }


# =========================================================================== schema 5
BUILDINGS = {
    "office": ("three-storey office building", 42000),
    "restaurant": ("restaurant tenant space", 4800),
    "apartments": ("12-unit apartment building", 14500),
    "warehouse": ("single-storey warehouse", 60000),
    "clinic": ("outpatient clinic fit-out", 7200),
}
FINALS = ["building", "electrical", "plumbing", "mechanical", "fire_alarm", "sprinkler", "fire_marshal"]
LIFE_SAFETY_OPEN = ["emergency lighting failed its 90-minute test in the rear corridor",
                    "one exit sign missing at the rear corridor", "fire alarm acceptance test not yet scheduled",
                    "handrail missing on the east stair", "fire door closer missing at the electrical room",
                    "sprinkler monitoring not yet connected to the alarm company"]
SITE_OPEN = ["landscaping not planted", "final lift of asphalt and striping", "public sidewalk panel replacement",
             "street trees not planted", "monument sign foundation", "stormwater pond seeding"]


class OccupancyCertificateRequest(Schema):
    name = "occupancy_certificate_request"
    description = "A request for a certificate of occupancy with some final inspections or site work still open."
    surfaces = {"json": 0.5, "co_checklist": 0.25, "developer_request": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        btype = pick(rng, list(BUILDINGS))
        finals = {k: pick_w(rng, {"passed": 7, "open": 2, "failed": 1}) for k in FINALS}
        n_ls = pick(rng, [0, 0, 1, 1, 2])
        n_site = pick(rng, [0, 1, 2, 3])
        site_cost = round(rng.uniform(15000, 140000), -3) if n_site else 0
        bond = round(site_cost * rng.uniform(0.6, 1.4), -3) if n_site and chance(rng, 0.8) else 0
        move_in = pick(rng, [2, 5, 9, 21, 45])
        prior_tco = pick(rng, [0, 0, 1, 2])
        occupied_already = chance(rng, 0.08)
        partial_complete = btype in ("office", "apartments") and chance(rng, 0.45)
        if reg == "clear":
            if chance(rng, 0.5):
                finals = {k: "passed" for k in FINALS}
                n_ls, occupied_already = 0, False
                n_site = pick(rng, [0, 1])
                site_cost = round(rng.uniform(15000, 40000), -3) if n_site else 0
                bond = round(site_cost * 1.25, -3) if n_site else 0
            else:
                finals["fire_alarm"], finals["sprinkler"] = "failed", "open"
                n_ls = 2
        elif reg == "borderline":
            n_ls = pick(rng, [0, 1])
            finals = {k: ("passed" if k in ("building", "electrical", "plumbing") else v) for k, v in finals.items()}
            if n_site:
                bond = round(site_cost * rng.uniform(0.9, 1.08), -3)
            move_in = pick(rng, [2, 5, 9])
        else:
            # everything signed except one life-safety item, with tenants arriving; or a bond short on a clean building
            if chance(rng, 0.5):
                finals = {k: "passed" for k in FINALS}
                finals[pick(rng, ["fire_alarm", "fire_marshal", "sprinkler"])] = "open"
                n_ls, move_in = 1, pick(rng, [1, 2, 3])
                occupied_already = chance(rng, 0.3)
            else:
                finals = {k: "passed" for k in FINALS}
                n_ls, n_site = 0, pick(rng, [2, 3])
                site_cost = round(rng.uniform(60000, 140000), -3)
                bond = round(site_cost * rng.uniform(0.5, 0.8), -3)
        if n_site == 0:
            site_cost, bond = 0, 0
        site_cost, bond = int(site_cost), int(bond)
        return {"regime": reg, "city": _city(rng), "building": btype, "finals": finals,
                "life_safety_items": n_ls, "site_items": n_site, "site_cost_estimate_usd": site_cost,
                "bond_usd": bond, "days_to_move_in": move_in, "prior_temporary_cos": prior_tco,
                "occupied_without_co": occupied_already, "part_complete_separable": partial_complete,
                "developer": person(rng), "official": person(rng), "permit": ident(rng, "BLD", 6, "-"),
                "address": _address(rng), "date": rand_date(rng)}

    def _open(self, f, rng):
        ls = rng.sample(LIFE_SAFETY_OPEN, f["life_safety_items"])
        site = rng.sample(SITE_OPEN, f["site_items"])
        return ls, site

    def render(self, f, rng, surface):
        desc, sqft = BUILDINGS[f["building"]]
        ls, site = self._open(f, rng)
        fin = {k.replace("_", " "): v for k, v in f["finals"].items()}
        guarantee = (_usd(f["bond_usd"], rng) + " performance bond" if f["bond_usd"] else
                     ("none posted" if f["site_items"] else None))
        est = _usd(f["site_cost_estimate_usd"], rng) if f["site_cost_estimate_usd"] else None
        extra = []
        if f["occupied_without_co"]:
            extra.append("inspector saw staff and furniture already moved into part of the building")
        if f["part_complete_separable"]:
            extra.append("ground floor is complete and separated from the upper floors by rated construction")
        if f["prior_temporary_cos"]:
            extra.append(f"{f['prior_temporary_cos']} temporary certificate(s) already issued on this permit")
        if surface == "json":
            return dumps(prune({
                "request": {"permit": f["permit"], "address": f["address"], "jurisdiction": f["city"],
                            "building": f"{desc}, {sqft:,} sq ft", "requested_by": f["developer"],
                            "occupancy_planned_in_days": f["days_to_move_in"]},
                "final_inspections": fin,
                "open_items": {"life_safety": ls or None, "site": site or None,
                               "site_work_estimate_usd": f["site_cost_estimate_usd"] or None},
                "financial_guarantee": guarantee,
                "notes": extra or None,
            }))
        if surface == "co_checklist":
            rows = [[k, v.upper() if v != "passed" else "passed"] for k, v in fin.items()]
            lines = [f"CERTIFICATE OF OCCUPANCY CHECKLIST -- {f['city']}",
                     kv_lines([("Permit", f["permit"]), ("Address", f["address"]), ("Building", f"{desc} ({sqft:,} sq ft)"),
                               ("Requested", fmt_date(f["date"], rng)), ("Planned occupancy", f"in {f['days_to_move_in']} days")], rng),
                     "", table(["final", "status"], rows, rng), ""]
            if ls or site:
                lines.append("Open items:")
                lines.append(bullets(ls + site, rng))
            lines.append(f"Site work estimate: {show(est, rng)}; guarantee: {show(guarantee, rng)}")
            if extra:
                lines += ["", "Notes: " + "; ".join(extra) + "."]
            return "\n".join(lines)
        not_passed = [k for k, v in fin.items() if v != "passed"]
        s = [f"We are requesting a certificate of occupancy for the {desc} at {f['address']} (permit {f['permit']}).",
             {"office": f"Our tenants are scheduled to move in {f['days_to_move_in']} days from now and the leases "
                        f"carry penalties for late delivery.",
              "apartments": f"Residents have signed leases starting in {f['days_to_move_in']} days.",
              "restaurant": f"The tenant plans to open in {f['days_to_move_in']} days; staff are hired and trained.",
              "warehouse": f"The operator's first inbound freight is booked for {f['days_to_move_in']} days from now.",
              "clinic": f"The clinic has patients booked from {f['days_to_move_in']} days from now."}[f["building"]],
             ("All final inspections have passed." if not not_passed else
              f"Finals still open or failed: {', '.join(not_passed)}; we expect these to clear shortly."),
             (f"Remaining items: {', '.join(ls + site)}." if ls or site else ""),
             (f"We have posted a {guarantee} against the site work, estimated at {est}." if f["bond_usd"] and est else
              (f"The remaining site work is estimated at {est}." if est else "")),
             (" ".join(x[0].upper() + x[1:] + "." for x in extra) if extra else "")]
        return "\n".join([f"Subject: CO request -- {f['address']}", f"From: {f['developer']}",
                          f"To: {f['official']}, Building Official", "", noisy(join_sentences(s), rng), "", "Thank you,",
                          f["developer"]])

    def questions(self):
        return {
            "occupancy_ruling": choice("What should the building official issue?", {
                "final_certificate": "A final certificate of occupancy for the whole building.",
                "partial_occupancy": "A certificate for only the completed, separable part of the building.",
                "temporary_certificate": "A temporary certificate with an expiry date and conditions.",
                "withhold_certificate": "Nothing yet; the building may not be occupied.",
            }),
            "life_safety_gap": score("How much do the open items affect the safety of people inside the building?", [
                "Not at all: only site or cosmetic work remains.",
                "Slightly: a minor item with an easy interim measure.",
                "Materially: alarm, egress or fire protection is incomplete in part of the building.",
                "Seriously: occupants could not be warned or get out safely in an emergency.",
            ]),
            "move_in_pressure": score("How pressing is the planned occupancy date?", [
                "Not pressing: occupancy is more than a few weeks away.",
                "Moderate: a couple of weeks remain to finish the open items.",
                "High: tenants or operations are booked to move in within days.",
                "Immediate: people are due in now or are already inside without a certificate.",
            ]),
            "guarantee_covers_site_work": noul(
                "The financial guarantee on file covers the cost of finishing the outstanding site work.",
                yes="A bond or similar guarantee is posted for at least the estimated cost of the open site work.",
                no="There is no guarantee, or it is smaller than the estimated cost of the open site work."),
            "open_items_safe_while_occupied": noul(
                "The remaining items can be finished safely while people occupy the building."),
        }


WORKFLOW = Workflow(
    name="building_permit_review",
    title="Building permit review",
    description="Whether a construction project in a fictional municipality gets its permit, inspection sign-off or "
                "occupancy certificate, and on what terms.",
    domain="municipal planning and building control",
    schemas=[ZoningPlanCheck(), PermitIntakeScreening(), FieldInspectionHoldPoint(), VarianceStaffReport(),
             OccupancyCertificateRequest()],
    notes="Invented jurisdictions, districts and codes; applicants' and designers' statements appear as evidence, "
          "never as the verdict. No customer-support threads, bills or security alerts.",
)
