"""Food safety inspection: what a local food-safety authority does after inspecting a food business.

An environmental health officer's findings: routine inspection results, food-safety (HACCP) records at a small
manufacturer, a cluster of illness reports linked to a premises, a revisit after a formal notice, and an
allergen-control failure. The decisions are the authority's: follow-up, enforcement level, closure, exclusion,
product withdrawal. Authorities, businesses and people are fictional; limits are generic good practice, not any
one country's code. No conversation with a member of the public is drafted or triaged.
"""
from __future__ import annotations

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
    fmt_date,
    ident,
    join_sentences,
    kv_lines,
    near,
    noisy,
    noul,
    numbered,
    person,
    pick,
    pick_w,
    prune,
    rand_date,
    regime,
    score,
    table,
)

AUTHORITY_SUFFIX = ["District Council", "Borough Council", "County Health Department", "City Environmental Health"]
# business type -> (name suffixes, risk band)
BIG_KITCHENS = {"restaurant", "school kitchen", "care-home kitchen", "butcher", "pub kitchen"}
BUSINESS = {
    "takeaway": (["Kebab House", "Fried Chicken", "Noodle Bar", "Pizza"], "high"),
    "restaurant": (["Grill", "Bistro", "Kitchen", "Brasserie"], "high"),
    "school kitchen": (["Primary School Kitchen", "Academy Kitchen"], "high"),
    "care-home kitchen": (["Care Home", "Nursing Home"], "high"),
    "bakery": (["Bakery", "Bakehouse"], "medium"),
    "café": (["Café", "Coffee House"], "medium"),
    "butcher": (["Butchers", "Meats"], "high"),
    "food truck": (["Street Kitchen", "Food Truck"], "medium"),
    "sushi bar": (["Sushi", "Sushi Bar"], "high"),
    "pub kitchen": (["Arms", "Tavern"], "medium"),
}


def _name(rng, suffixes):
    """ORG_A prefix + one of `suffixes`, skipping combinations that name a real business (as base.org does)."""
    a, i = rng.choice(ORG_A), rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        n = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if n not in REAL_NAME_CLASHES:
            return n
    return n


def _business(rng, kinds=None):
    kind = pick(rng, kinds or list(BUSINESS))
    return kind, _name(rng, BUSINESS[kind][0]), BUSINESS[kind][1]


def _authority(rng):
    return _name(rng, AUTHORITY_SUFFIX)


def _temp(c, unit):
    return f"{c * 9 / 5 + 32:.0f}°F" if unit == "F" else f"{c:.1f}°C"


def _plural(n, word):
    if n == 1:
        return f"1 {word}"
    return f"{n} " + (word[:-1] + "ies" if word.endswith("y") and word[-2] not in "aeiou" else word + "s")


def _date_style(rng):
    """One date style per document: real artefacts do not mix formats line to line."""
    return pick(rng, ["iso", "iso", "us", "long", "eu"])


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _people(rng, n):
    out, seen = [], []
    while len(out) < n:
        p = person(rng)
        if p.split()[0] not in seen:
            seen.append(p.split()[0])
            out.append(p)
    return out


# =========================================================================== schema 1
# finding -> (text, is it a priority (critical) item?)
FINDINGS = {
    "raw_over_rte": ("raw chicken stored above ready-to-eat desserts in the upright fridge", True),
    "handwash_blocked": ("kitchen hand-wash basin blocked with stored equipment", True),
    "no_soap": ("no soap or paper towels at the kitchen hand-wash basin", True),
    "dirty_slicer": ("meat slicer not cleaned between raw and cooked products", True),
    "chemicals_over_food": ("degreaser stored on the shelf above a food preparation surface", True),
    "cooling_uncontrolled": ("cooked rice left to cool at room temperature, no cooling record", True),
    "date_labels": ("prepared foods in the fridge without date labels", False),
    "floor_damage": ("damaged floor tiles in the wash-up area", False),
    "no_supervisor_cert": ("no one on shift holds a food-hygiene supervisor certificate", False),
    "no_probe": ("no working probe thermometer on site", False),
    "records_gap": ("fridge temperature records not kept for the last three weeks", False),
    "extraction_grease": ("grease build-up on the extraction canopy", False),
}
PESTS = {"none": None, "old_droppings": "old mouse droppings behind the dry-store shelving",
         "fresh_droppings": "fresh mouse droppings on dry-store shelving and in an open flour sack",
         "live_activity": "live cockroaches seen under the dishwasher and around the wash-up sink"}
UTILITY = {"none": None, "no_hot_water": "no hot water to the kitchen since the boiler failed two days ago",
           "sewage_backup": "sewage backing up through the floor drain in the wash-up area"}
ATTITUDE = {"cooperative": "manager cooperative and started putting things right during the visit",
            "defensive": "manager disputed several findings and said previous officers had never raised them",
            "absent": "manager not on site; the shift leader could not answer questions about procedures"}


class RoutineInspectionOutcome(Schema):
    name = "routine_inspection_outcome"
    description = "A routine inspection of a food business: follow-up and enforcement after the findings are in."
    surfaces = {"json": 0.5, "inspection_report": 0.25, "officer_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind, bname, risk = _business(rng)
        unit = pick(rng, ["C", "C", "F"])
        crit_keys = [k for k, v in FINDINGS.items() if v[1]]
        minor_keys = [k for k, v in FINDINGS.items() if not v[1]]
        n_crit, n_minor = pick(rng, [0, 1, 1, 2, 3]), pick(rng, [0, 1, 2, 3])
        chiller = round(between(rng, 3.0, 9.0), 1)
        pests = pick_w(rng, {"none": 6, "old_droppings": 2, "fresh_droppings": 1, "live_activity": 1})
        utility = pick_w(rng, {"none": 9, "no_hot_water": 1, "sewage_backup": 0.5})
        corrected = chance(rng, 0.5)
        repeats = pick(rng, [0, 0, 1, 2])
        prior_actions = pick(rng, [0, 0, 1, 2])
        attitude = pick_w(rng, {"cooperative": 5, "defensive": 2, "absent": 2})
        if reg == "clear":
            if chance(rng, 0.5):   # a well-run kitchen
                n_crit, n_minor, chiller, pests, utility, repeats, prior_actions, attitude = \
                    0, pick(rng, [0, 1]), round(between(rng, 2.0, 4.5), 1), "none", "none", 0, 0, "cooperative"
            else:                  # a kitchen that should not be open
                n_crit, pests, utility, repeats, prior_actions = pick(rng, [3, 4]), pick(rng, ["fresh_droppings",
                                                                                                "live_activity"]), \
                    pick(rng, ["none", "sewage_backup", "no_hot_water"]), 2, pick(rng, [1, 2])
                chiller, corrected = round(between(rng, 9.0, 13.0), 1), False
        elif reg == "borderline":
            chiller = near(rng, 5.5, 1.5, lo=1.0, digits=1)
            n_crit, n_minor, repeats = pick(rng, [1, 2, 2, 3]), pick(rng, [1, 2, 3]), pick(rng, [0, 1, 1])
            pests = pick(rng, ["none", "old_droppings", "fresh_droppings"])
            utility = "none"
        else:  # conflicting: a strong record with one alarming finding, or many findings all fixed on the spot
            if chance(rng, 0.5):
                n_crit, n_minor, repeats, prior_actions, attitude = 0, 1, 0, 0, "cooperative"
                pests, utility = pick(rng, [("live_activity", "none"), ("none", "sewage_backup"),
                                            ("fresh_droppings", "none")])
            else:
                n_crit, corrected, pests, utility, attitude = pick(rng, [2, 3]), True, "none", "none", "cooperative"
        crit = rng.sample(crit_keys, n_crit)
        minor = rng.sample(minor_keys, n_minor)
        deductions = sum(8 if FINDINGS[k][1] else 3 for k in crit + minor) + 4 * repeats + \
            {"none": 0, "old_droppings": 4, "fresh_droppings": 10, "live_activity": 18}[pests] + \
            (0 if utility == "none" else 15) + max(0, int(round((chiller - 5.0) * 3)))
        return {"regime": reg, "business": bname, "business_type": kind, "risk_band": risk, "unit": unit,
                "critical_findings": crit, "minor_findings": minor, "chiller_c": chiller, "pests": pests,
                "utility": utility, "criticals_corrected_on_site": corrected and bool(crit), "repeat_findings": repeats,
                "enforcement_actions_24m": prior_actions, "attitude": attitude,
                "score": max(0, 100 - deductions), "pass_mark": 70, "last_score": pick(rng, [68, 74, 81, 88, 95]),
                "customers_per_day": pick(rng, [40, 90, 200, 450]), "authority": _authority(rng),
                "fridge": "walk-in chiller" if kind in BIG_KITCHENS else "main upright fridge",
                "officer": person(rng), "visit": rand_date(rng), "ref": ident(rng, "FHI", 6, "-"),
                "operator_line": pick(rng, [None, "the chiller had just been restocked after a delivery",
                                            "the regular manager is on holiday this week",
                                            "we have booked a pest contractor already", "a new chef started on Monday"])}

    def _finding_lines(self, f):
        out = [FINDINGS[k][0] for k in f["critical_findings"]]
        limit = 5.0
        out.append(f"{f['fridge']} air temperature {_temp(f['chiller_c'], f['unit'])} (limit {_temp(limit, f['unit'])})")
        if f["pests"] != "none":
            out.append(PESTS[f["pests"]])
        if f["utility"] != "none":
            out.append(UTILITY[f["utility"]])
        out += [FINDINGS[k][0] for k in f["minor_findings"]]
        return out

    def render(self, f, rng, surface):
        lines = self._finding_lines(f)
        if surface == "json":
            return dumps(prune({
                "premises": {"name": f["business"], "type": f["business_type"], "risk_band": f["risk_band"],
                             "customers_per_day": f["customers_per_day"], "authority": f["authority"]},
                "inspection": {"ref": f["ref"], "date": fmt_date(f["visit"], style="iso"), "officer": f["officer"],
                               "score": f["score"], "pass_mark": f["pass_mark"], "previous_score": f["last_score"]},
                "findings": {"priority": [FINDINGS[k][0] for k in f["critical_findings"]] or None,
                             "temperatures": {f["fridge"].replace(" ", "_").replace("-", "_"): _temp(f["chiller_c"], f["unit"]),
                                              "cold_holding_limit": _temp(5.0, f["unit"])},
                             "pests": PESTS[f["pests"]], "water_and_drainage": UTILITY[f["utility"]],
                             "other": [FINDINGS[k][0] for k in f["minor_findings"]] or None,
                             "priority_items_corrected_during_visit": f["criticals_corrected_on_site"]
                             if f["critical_findings"] else None,
                             "repeated_from_last_inspection": f["repeat_findings"]},
                "history": {"formal_actions_last_24_months": f["enforcement_actions_24m"]},
                "management": {"observed": ATTITUDE[f["attitude"]], "operator_comment": f["operator_line"]},
            }))
        if surface == "inspection_report":
            head = kv_lines([("Premises", f"{f['business']} ({f['business_type']}, {f['risk_band']} risk)"),
                             ("Authority", f["authority"]), ("Inspection", f"{f['ref']}, {fmt_date(f['visit'], rng)}"),
                             ("Officer", f["officer"]), ("Customers per day", f"~{f['customers_per_day']}"),
                             ("Score", f"{f['score']} / 100 (pass {f['pass_mark']}; last inspection {f['last_score']})"),
                             ("Formal actions, 24 months", f["enforcement_actions_24m"])], rng,
                            style=pick(rng, ["colon", "aligned"]))
            tail = [f"Priority items corrected during visit: {'yes' if f['criticals_corrected_on_site'] else 'no'}"
                    if f["critical_findings"] else None,
                    f"Repeated from last inspection: {f['repeat_findings']}",
                    f"Management: {ATTITUDE[f['attitude']]}",
                    f"Operator says: {f['operator_line']}" if f["operator_line"] else None]
            return (f"FOOD HYGIENE INSPECTION REPORT\n{head}\n\nFindings:\n{numbered(lines)}\n\n"
                    + "\n".join(x for x in tail if x))
        s = [f"Routine visit to {f['business']}, {f['business_type']}, about {f['customers_per_day']} customers a day.",
             f"Found: {'; '.join(lines)}."]
        if f["critical_findings"]:
            s.append("Priority items were put right while I was there." if f["criticals_corrected_on_site"] else
                     "None of the priority items was corrected during the visit.")
        if f["repeat_findings"]:
            s.append(f"{_plural(f['repeat_findings'], 'item')} repeated from the last inspection.")
        s.append(_cap(ATTITUDE[f["attitude"]]) + ".")
        if f["operator_line"]:
            s.append(f"Operator's explanation: {f['operator_line']}.")
        s.append(f"Scored {f['score']} (pass mark {f['pass_mark']}, {f['last_score']} last time); "
                 + (f"{_plural(f['enforcement_actions_24m'], 'formal action')} in the last two years." if
                    f["enforcement_actions_24m"] else "no formal action in the last two years."))
        return (f"Officer notes -- {f['ref']} -- {f['officer']}, {fmt_date(f['visit'], rng)}\n"
                + noisy(join_sentences(s), rng, 0.12))

    def questions(self):
        return {
            "follow_up_visit": choice("What follow-up should the authority arrange after this inspection?", {
                "next_routine_inspection": "Nothing extra; the premises is seen again at its next routine inspection.",
                "evidence_by_email": "Ask the operator to send photos or receipts showing the items were fixed.",
                "revisit_within_two_weeks": "Revisit within about two weeks to check the work has been done.",
                "revisit_within_two_days": "Revisit within a day or two, because conditions cannot wait.",
            }),
            "enforcement_level": choice("What level of enforcement should follow the inspection?", {
                "advice_only": "Verbal or written advice, with no formal action.",
                "closure_order": "Close the premises, or ask it to close, until the risk is removed.",
                "formal_notice": "A formal notice listing the work required and a deadline.",
                "penalty_or_prosecution": "A penalty or a prosecution file for the breaches found.",
                "written_warning": "A written warning letter setting out what must improve.",
            }),
            "immediate_consumer_risk": score("How great is the immediate risk to people eating food from these "
                                             "premises?", [
                "Negligible: food is being handled safely.",
                "Low: lapses that are unlikely to make anyone ill today.",
                "Moderate: real chances of contamination or bacterial growth.",
                "High: food is likely to be unsafe unless things change now.",
                "Severe: an outbreak could happen at any service.",
            ]),
            "confidence_in_management": score("How much confidence does the business's food-safety management "
                                              "inspire?", [
                "None: nobody is managing food safety.",
                "Little: procedures exist on paper but are not followed.",
                "Fair: sound procedures with lapses.",
                "Good: a well-run system with only minor slips.",
            ]),
            "imminent_health_risk": noul(
                "Conditions at the premises amount to an imminent risk to health that justifies stopping food "
                "operations.",
                yes="Live pests or fresh droppings in food areas, sewage backing up, or no hot water for "
                    "handwashing: one of these is present now.",
                no="None of those is present. Warm fridges, poor storage, other priority items or old droppings "
                   "need putting right, but do not by themselves make trading unsafe today."),
        }


# =========================================================================== schema 2
# producer type -> (critical control point, critical limit text, what a failure looks like, product)
PRODUCERS = {
    "cooked meats": ("cooking", "core temperature at least 75°C", "core temperature below 75°C", "sliced ham and roast beef"),
    "ready meals": ("chilling", "below 8°C within 90 minutes of cooking", "product still above 8°C after 90 minutes",
                    "chilled lasagne and curries"),
    "soft cheese": ("pasteurisation", "72°C for 15 seconds", "holding temperature below 72°C", "fresh soft cheese"),
    "cold-pressed juice": ("high-pressure processing", "600 MPa for 3 minutes", "pressure below 600 MPa",
                           "bottled juices"),
    "sandwiches": ("chilled storage", "at or below 5°C", "store room above 5°C", "sandwiches for petrol-station chillers"),
    "smoked fish": ("brining", "salt in the water phase at least 3.5%", "salt below 3.5%", "cold-smoked salmon"),
}
DISTRIBUTION = {"local": "sold at {p} own shop and two local cafés",
                "regional": "supplied to about 40 shops across the region",
                "national": "sold online and dispatched nationwide"}


def _dist(f, p="its"):
    return DISTRIBUTION[f["distribution"]].format(p=p)


class HaccpRecordReview(Schema):
    name = "haccp_record_review"
    description = "A small food manufacturer's critical-control-point records reviewed by an officer: accept, advise, review or hold product."
    surfaces = {"json": 0.5, "record_extract": 0.25, "audit_letter": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ptype = pick(rng, list(PRODUCERS))
        days = pick(rng, [14, 21, 28])
        deviations = pick(rng, [0, 0, 1, 2, 4])
        actions = pick_w(rng, {"all": 3, "some": 2, "none": 2})
        pattern = pick_w(rng, {"plausible": 5, "identical_values": 2, "filled_ahead": 1, "gaps": 2})
        calib = pick_w(rng, {"in_date": 5, "overdue": 2, "no_record": 1})
        spot = pick_w(rng, {"within_limit": 5, "failed": 2, "not_taken": 2})
        dist = pick_w(rng, {"local": 3, "regional": 3, "national": 2})
        batches = pick(rng, [2, 6, 15, 40])
        if reg == "clear":
            if chance(rng, 0.5):
                deviations, actions, pattern, calib, spot = pick(rng, [0, 1]), "all", "plausible", "in_date", \
                    "within_limit"
            else:
                deviations, actions, pattern, calib, spot = pick(rng, [3, 5]), "none", pick(rng, ["identical_values",
                                                                                               "filled_ahead"]), \
                    "no_record", "failed"
        elif reg == "borderline":
            deviations, actions = pick(rng, [1, 2]), pick(rng, ["some", "all"])
            pattern = pick(rng, ["plausible", "gaps"])
            spot = pick(rng, ["within_limit", "not_taken"])
        else:  # conflicting: spotless records contradicted by the officer's own check, or messy records, good product
            if chance(rng, 0.5):
                deviations, pattern, spot = 0, pick(rng, ["identical_values", "plausible"]), "failed"
            else:
                deviations, actions, pattern, spot, calib = pick(rng, [2, 3]), "all", "gaps", "within_limit", "in_date"
        if deviations == 0:
            actions = "not_applicable"
        return {"regime": reg, "producer_type": ptype, "business": _name(rng, ["Foods", "Kitchen", "Fine Foods",
                                                                             "Producers"]),
                "ccp": PRODUCERS[ptype][0], "limit": PRODUCERS[ptype][1], "failure": PRODUCERS[ptype][2],
                "product": PRODUCERS[ptype][3], "days_reviewed": days, "deviations": deviations,
                "corrective_actions": actions, "record_pattern": pattern, "thermometer_calibration": calib,
                "officer_check": spot, "distribution": dist, "batches_since_first_deviation": batches,
                "authority": _authority(rng), "officer": person(rng), "operator": person(rng),
                "visit": rand_date(rng), "ref": ident(rng, "HAC", 5, "-"),
                "operator_line": pick(rng, [None, "the records are filled in at the end of each shift",
                                            "the probe was replaced last month", "one deviation was a data-entry slip"])}

    def _pattern_text(self, f):
        return {"plausible": "entries vary from day to day and are signed by different staff",
                "identical_values": "every entry shows exactly the same value, in the same pen, for weeks",
                "filled_ahead": "tomorrow's entry was already filled in when the officer arrived",
                "gaps": "several days have no entries at all"}[f["record_pattern"]]

    def _actions_text(self, f):
        return {"all": "each deviation has a written corrective action and product decision",
                "some": "some deviations have a corrective action recorded, others none",
                "none": "no corrective action or product decision is recorded for any deviation",
                "not_applicable": None}[f["corrective_actions"]]

    def _check_text(self, f):
        return {"within_limit": f"officer's own check during the visit met the limit ({f['limit']})",
                "failed": f"officer's own check during the visit failed the limit: {f['failure']}",
                "not_taken": "no production running during the visit, so no check was taken"}[f["officer_check"]]

    def render(self, f, rng, surface):
        calib = {"in_date": "in date", "overdue": "overdue", "no_record": "no calibration record"}[f["thermometer_calibration"]]
        if surface == "json":
            return dumps(prune({
                "business": {"name": f["business"], "activity": f"{f['producer_type']} producer",
                             "products": f["product"], "distribution": _dist(f)},
                "ccp": {"step": f["ccp"], "critical_limit": f["limit"]},
                "records": {"days_reviewed": f["days_reviewed"], "limit_not_met_entries": f["deviations"],
                            "corrective_actions": self._actions_text(f), "how_entries_look": self._pattern_text(f),
                            "monitoring_instrument_calibration": calib},
                "verification": {"officer_check": self._check_text(f),
                                 "batches_made_since_first_deviation": f["batches_since_first_deviation"]
                                 if f["deviations"] else None},
                "visit": {"ref": f["ref"], "date": fmt_date(f["visit"], style="iso"), "officer": f["officer"],
                          "authority": f["authority"]},
                "operator_comment": f["operator_line"],
            }))
        if surface == "record_extract":
            start = f["visit"] - timedelta(days=f["days_reviewed"])
            rows, base_val = [], {"plausible": None, "identical_values": "OK", "filled_ahead": None, "gaps": None}
            dev_days = sorted(rng.sample(range(f["days_reviewed"]), f["deviations"])) if f["deviations"] else []
            gap_days = set(rng.sample(range(f["days_reviewed"]), 4)) if f["record_pattern"] == "gaps" else set()
            for d in range(0, f["days_reviewed"], max(1, f["days_reviewed"] // 7)):
                day = start + timedelta(days=d)
                if d in gap_days:
                    rows.append([fmt_date(day, style="iso"), "", "", ""])
                    continue
                bad = any(abs(d - x) < max(1, f["days_reviewed"] // 7) and d <= x for x in dev_days)
                val = base_val[f["record_pattern"]] or ("NOT MET" if bad else "met")
                act = ("see note" if bad and f["corrective_actions"] in ("all", "some") and chance(rng, 0.7) else "")
                rows.append([fmt_date(day, style="iso"), val, pick(rng, ["JK", "MS", "AB", "TL"])
                             if f["record_pattern"] != "identical_values" else "JK", act])
            if f["record_pattern"] == "filled_ahead":
                rows.append([fmt_date(f["visit"] + timedelta(days=1), style="iso"), "met", "JK", ""])
            head = (f"{f['business']} -- {f['ccp']} CCP log (limit: {f['limit']}), extract\n"
                    f"Officer {f['officer']}, {f['authority']}, visit {fmt_date(f['visit'], rng)}\n")
            notes = bullets([f"{f['deviations']} entries not meeting the limit in {f['days_reviewed']} days",
                             self._actions_text(f), f"Entries: {self._pattern_text(f)}",
                             f"Probe/instrument calibration: {calib}", _cap(self._check_text(f)),
                             f"Products {_dist(f)}",
                             (f"{f['batches_since_first_deviation']} batches made since the first deviation"
                              if f["deviations"] else None),
                             (f"Operator: {f['operator_line']}" if f["operator_line"] else None)], rng)
            return head + "\n" + table(["date", "result", "initials", "action"], rows, rng) + "\n\n" + notes
        s = [f"Thank you for your time during my visit on {fmt_date(f['visit'], style=_date_style(rng))}.",
             f"I reviewed {f['days_reviewed']} days of your {f['ccp']} records against your critical limit "
             f"({f['limit']}).",
             (f"{_plural(f['deviations'], 'entry')} did not meet the limit." if f["deviations"] else
              "All entries record the limit as met."),
             (_cap(self._actions_text(f)) + ".") if self._actions_text(f) else "",
             f"I noted that {self._pattern_text(f)}.",
             f"Calibration of the monitoring instrument: {calib}.", _cap(self._check_text(f)) + ".",
             f"I understand your {f['product']} are {_dist(f, 'your')}."]
        if f["operator_line"]:
            s.append(f"You told me {f['operator_line']}.")
        body = noisy(join_sentences(s), rng, 0.05)
        return "\n".join([f"{f['authority']} -- Food Safety", f"Our ref: {f['ref']}", "",
                          f"Dear {f['operator']},", "", body, "", "Yours sincerely,", f["officer"],
                          "Environmental Health Officer"])

    def questions(self):
        return {
            "records_outcome": choice("What should the officer require after reviewing these records?", {
                "advice_and_monitor": "Give written advice and check the records again at the next visit.",
                "formal_review_and_revisit": "Require a formal review of the food-safety plan and revisit to check it.",
                "hold_product_and_formal_action": "Require product made during the failures to be held or recalled, "
                                                  "with formal action.",
                "records_satisfactory": "Accept the records as showing the process is under control.",
            }),
            "record_trustworthiness": score("How far can the monitoring records be trusted as a true account of "
                                            "what happened?", [
                "Not at all: they look invented or filled in without measuring.",
                "Doubtful: gaps or odd patterns undermine them.",
                "Mostly: minor gaps, but they ring true.",
                "Fully: consistent, verified and backed by the officer's own check.",
            ]),
            "shipped_product_concern": score("How concerned should the officer be that unsafe product has already "
                                             "left the premises?", [
                "Not concerned: the process has been under control.",
                "Slightly: brief lapses, dealt with at the time.",
                "Concerned: lapses without a clear product decision.",
                "Very concerned: repeated or hidden failures on product already sold.",
            ]),
            "ccp_controlled_in_practice": noul(
                "The critical control point is under control in practice, not just on paper.",
                yes="Deviations are rare and handled, the records ring true and the officer's own check agrees.",
                no="Deviations are unhandled, the records are doubtful, or the officer's own check failed."),
            "trace_affected_batches": noul(
                "Batches made during the recorded failures must be traced and held back from sale.",
                yes="Failures were recorded without a corrective action and product decision for each, or the "
                    "officer's own check failed, so affected product may be on sale.",
                no="No failure was recorded, or every failure has a recorded corrective action and product decision, "
                   "and the officer's own check did not fail."),
        }


# =========================================================================== schema 3
# suspected agent -> (typical incubation hours lo, hi, dominant symptoms, typical food vehicle)
AGENTS = {
    "norovirus": (12, 48, "vomiting and diarrhoea, short illness", "salad and sandwiches handled by an ill worker"),
    "salmonella": (6, 72, "diarrhoea, fever and stomach cramps", "undercooked chicken or raw-egg desserts"),
    "bacillus_cereus": (1, 6, "vomiting within hours of eating", "egg-fried rice kept warm for hours"),
    "c_perfringens": (8, 16, "diarrhoea and cramps, little vomiting", "gravy or stew cooled slowly"),
    "campylobacter": (48, 120, "diarrhoea, sometimes bloody, with fever", "chicken liver pâté"),
}
FOODS = {"norovirus": "mixed salad", "salmonella": "tiramisu made with raw egg", "bacillus_cereus": "egg-fried rice",
         "c_perfringens": "beef gravy", "campylobacter": "chicken liver pâté"}
# the premises each agent plausibly turns up at, with a dish override where the default does not fit
AGENT_PREMISES = {"norovirus": ["takeaway", "restaurant", "café", "sushi bar", "pub kitchen", "care-home kitchen"],
                  "salmonella": ["restaurant", "café", "pub kitchen", "care-home kitchen"],
                  "bacillus_cereus": ["takeaway", "restaurant"],
                  "c_perfringens": ["pub kitchen", "care-home kitchen", "restaurant"],
                  "campylobacter": ["restaurant", "pub kitchen", "café"]}
FOOD_OVERRIDE = {("norovirus", "sushi bar"): "salmon nigiri", ("norovirus", "takeaway"): "salad garnish",
                 ("salmonella", "pub kitchen"): "chicken burgers served pink", ("salmonella", "care-home kitchen"):
                 "egg mayonnaise sandwiches", ("c_perfringens", "care-home kitchen"): "beef stew"}


class IllnessReportInvestigation(Schema):
    name = "illness_report_investigation"
    description = "Reports of illness after eating food from one premises: how strong the link is and what to do next."
    surfaces = {"json": 0.5, "case_line_list": 0.25, "investigation_note": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        agent = pick(rng, list(AGENTS))
        kind, bname, _risk = _business(rng, AGENT_PREMISES[agent])
        lo, hi = AGENTS[agent][0], AGENTS[agent][1]
        cases = pick(rng, [1, 2, 3, 5, 8, 12])
        households = max(1, min(cases, pick(rng, [1, 1, 2, 3, 4])))
        onset_fit = pick_w(rng, {"fits": 6, "partly": 2, "does_not_fit": 1})
        positives, submitted = 0, pick(rng, [0, 0, 1, 2, 4])
        if submitted:
            positives = pick(rng, [0, 0, 1, submitted])
        common_food = chance(rng, 0.6)
        other_exposure = pick_w(rng, {"none": 5, "same_party_elsewhere": 1, "household_ill_before": 1})
        staff_ill = pick_w(rng, {"none_reported": 4, "one_off_sick": 2, "worked_while_ill": 1, "not_asked": 2})
        leftovers = chance(rng, 0.3)
        visit = pick_w(rng, {"not_yet": 3, "no_issues": 2, "temperature_issues": 2, "poor_handwashing": 1})
        if reg == "clear":
            if chance(rng, 0.5):   # one person, unrelated timing, another likely source
                cases, households, onset_fit, submitted, positives, common_food, other_exposure = \
                    1, 1, "does_not_fit", 0, 0, False, pick(rng, ["household_ill_before", "same_party_elsewhere"])
                staff_ill = "none_reported"
            else:                  # textbook point-source outbreak
                cases, households, onset_fit, common_food, other_exposure = pick(rng, [8, 12]), pick(rng, [3, 4]), \
                    "fits", True, "none"
                submitted = 4
                positives = pick(rng, [3, 4])
                staff_ill = pick(rng, ["worked_while_ill", "one_off_sick"])
        elif reg == "borderline":
            cases, households = pick(rng, [2, 3]), pick(rng, [1, 2])
            onset_fit = pick(rng, ["fits", "partly"])
            submitted, positives = pick(rng, [0, 1]), 0
        else:  # conflicting: several households and a good fit, but a spotless premises and another common exposure
            cases, households, onset_fit = pick(rng, [4, 5]), pick(rng, [2, 3]), "fits"
            other_exposure, visit = "same_party_elsewhere", "no_issues"
            if chance(rng, 0.5):
                submitted, positives = 2, 0
        households = min(households, cases)
        if other_exposure == "same_party_elsewhere" and cases < 2:
            other_exposure = "household_ill_before"
        onsets = {"fits": (lo + (hi - lo) * 0.2, lo + (hi - lo) * 0.7),
                  "partly": (max(0.5, lo * 0.6), hi * 1.3),
                  "does_not_fit": (hi * 1.6, hi * 2.5) if lo > 2 else (hi * 3, hi * 6)}[onset_fit]
        return {"regime": reg, "business": bname, "business_type": kind, "suspected_agent": agent,
                "symptoms": AGENTS[agent][2], "cases": cases, "households": households,
                "onset_hours_min": round(onsets[0]), "onset_hours_max": round(onsets[1]), "onset_fit": onset_fit,
                "samples_submitted": submitted, "samples_positive": positives, "common_food": common_food,
                "food": FOOD_OVERRIDE.get((agent, kind), FOODS[agent]), "other_exposure": other_exposure, "staff_illness": staff_ill,
                "leftovers_available": leftovers, "visit_findings": visit, "authority": _authority(rng),
                "officer": person(rng), "meal_date": rand_date(rng), "ref": ident(rng, "ILL", 5, "-"),
                "event": pick(rng, ["a birthday meal for 14 on Saturday", "takeaway orders on Friday evening",
                                    "Sunday lunch service", "a works lunch delivered on Tuesday"])}

    def _staff_text(self, f):
        return {"none_reported": "manager reports no staff illness",
                "one_off_sick": "one kitchen worker went home with vomiting the day before the meal",
                "worked_while_ill": "a kitchen worker had diarrhoea but worked the shift in question",
                "not_asked": "staff illness not yet asked about"}[f["staff_illness"]]

    def _other_text(self, f):
        return {"none": None, "same_party_elsewhere": "most of the cases also ate together at a separate event "
                                                       "the same weekend",
                "household_ill_before": "a member of the case's household was ill before the meal"}[f["other_exposure"]]

    def _visit_text(self, f):
        return {"not_yet": "premises not yet visited", "no_issues": "visit found no significant problems",
                "temperature_issues": "visit found food held warm below safe temperatures",
                "poor_handwashing": "visit found poor hand-washing practice"}[f["visit_findings"]]

    def _lab_text(self, f):
        if not f["samples_submitted"]:
            return "no stool samples submitted yet"
        return f"{f['samples_positive']} of {f['samples_submitted']} stool samples positive for the suspected organism"

    def render(self, f, rng, surface):
        agent = {"norovirus": "norovirus", "salmonella": "Salmonella", "campylobacter": "Campylobacter",
                 "bacillus_cereus": "Bacillus cereus", "c_perfringens": "Clostridium perfringens"}[f["suspected_agent"]]
        food = (f"most cases ate the {f['food']}" if f["common_food"] else "no single dish eaten by all cases")
        lo, hi = AGENTS[f["suspected_agent"]][0], AGENTS[f["suspected_agent"]][1]
        incub = f"typical incubation for {agent}: {lo}-{hi} hours"
        if surface == "json":
            return dumps(prune({
                "report": {"ref": f["ref"], "premises": f["business"], "type": f["business_type"],
                           "exposure": f["event"], "meal_date": fmt_date(f["meal_date"], style="iso")},
                "cases": {"ill": f["cases"], "households": f["households"], "symptoms": f["symptoms"],
                          "onset_hours_after_meal": f"{f['onset_hours_min']}-{f['onset_hours_max']}",
                          "suspected_agent": agent, "typical_incubation_hours": f"{lo}-{hi}",
                          "laboratory": self._lab_text(f)},
                "food": {"common_item": food, "leftovers_kept": f["leftovers_available"]},
                "premises": {"staff_illness": self._staff_text(f), "visit": self._visit_text(f)},
                "other_exposures": self._other_text(f),
            }))
        if surface == "case_line_list":
            rows = []
            for i in range(min(f["cases"], 8)):
                onset = rng.randint(f["onset_hours_min"], max(f["onset_hours_min"], f["onset_hours_max"]))
                ate = "yes" if (f["common_food"] and (i < max(1, int(f["cases"] * 0.8)))) else pick(rng, ["yes", "no",
                                                                                                       "unsure"])
                sample = ("positive" if i < f["samples_positive"] else "negative" if i < f["samples_submitted"]
                          else "-")
                rows.append([f"C{i + 1}", f"H{min(i, f['households'] - 1) + 1}", f"{onset} h", ate, sample])
            head = (f"{f['authority']} -- illness reports {f['ref']}\nPremises: {f['business']} ({f['business_type']}); "
                    f"exposure: {f['event']}; symptoms: {f['symptoms']}\n")
            extra = "" if f["cases"] <= 8 else f"\n(+{f['cases'] - 8} further cases, details pending)"
            notes = bullets([f"Suspected agent from symptoms: {agent} ({incub})", self._lab_text(f),
                             _cap(self._staff_text(f)), _cap(self._visit_text(f)),
                             "Leftovers kept by a case" if f["leftovers_available"] else "No leftovers available",
                             self._other_text(f)], rng)
            return head + "\n" + table(["case", "household", "onset", f"ate {f['food']}", "stool sample"], rows, rng) \
                + extra + "\n\n" + notes
        s = [("One person" if f["cases"] == 1 else f"{f['cases']} people") + f" from "
             f"{_plural(f['households'], 'household')} reported illness after {f['event']} from {f['business']} "
             f"({f['business_type']}).",
             f"Symptoms: {f['symptoms']}, starting {f['onset_hours_min']} to {f['onset_hours_max']} hours after eating.",
             f"The symptoms suggest {agent} ({incub}).", _cap(self._lab_text(f)) + ".", _cap(food) + ".",
             _cap(self._staff_text(f)) + ".", _cap(self._visit_text(f)) + "."]
        if f["cases"] == 1:
            s[4] = s[4].replace("most cases ate", "the case ate").replace("no single dish eaten by all cases",
                                                                            "the case cannot name a single dish")
        if self._other_text(f):
            s.append(f"Note: {self._other_text(f)}.")
        s.append("A case has kept leftovers." if f["leftovers_available"] else "No leftovers are available.")
        return (f"INVESTIGATION NOTE {f['ref']} -- {f['officer']}, {f['authority']}\n\n"
                + noisy(join_sentences(s), rng, 0.1))

    def questions(self):
        return {
            "next_investigation_step": choice("What should the authority do next about these reports?", {
                "exclude_ill_staff_and_sample": "Exclude any ill food handlers and take food and surface samples at once.",
                "gather_case_details": "Collect fuller food histories and samples from the cases before acting on "
                                       "the premises.",
                "record_and_close": "Record the reports and close; they do not point to the premises.",
                "unannounced_inspection": "Make an unannounced inspection of the premises this week.",
                "voluntary_closure_and_clean": "Ask the business to close voluntarily for a deep clean until the "
                                               "source is found.",
            }),
            "premises_link_strength": score("How strongly does the evidence link the illnesses to this premises?", [
                "Not at all: the timing or another exposure explains them.",
                "Weakly: a single report or a loose fit.",
                "Moderately: several reports that fit, not yet confirmed.",
                "Strongly: several households, a common dish and a good fit.",
                "Conclusively: laboratory results and the food history agree.",
            ]),
            "further_cases_risk": score("How likely are more people to fall ill if nothing more is done?", [
                "Very unlikely: a single case, onsets that do not fit, or another shared exposure explains it.",
                "Unlikely: the premises is not ruled out, but no ill handler or unsafe practice has been found.",
                "Likely: a handler was ill around the meal or hand-washing was poor, so the source may still be active.",
                "Very likely: a handler worked while ill, or food is still being held at unsafe temperatures.",
            ]),
            "exclude_food_handler": noul(
                "A food handler must be kept off work until cleared.",
                yes="A food handler has had symptoms of gastrointestinal illness recently, or is linked to the cases.",
                no="No food handler is reported ill or linked to the cases."),
            "timing_fits_agent": noul(
                "The timing of the illnesses fits the germ that the symptoms suggest.",
                yes="Every reported onset time falls inside the germ's typical incubation period.",
                no="Some or all of the onset times fall outside the typical incubation period."),
        }


# =========================================================================== schema 4
# notice item -> (text, priority 3 = most important, answer key for the priority question)
NOTICE_ITEMS = {
    "pest_proofing": ("seal gaps around pipework and fit door brush strips to keep out mice", 3, "pest_proofing"),
    "replace_fridge": ("replace the upright fridge that cannot hold 5°C", 3, "cold_storage"),
    "handwash_basin": ("install a dedicated hand-wash basin with hot water in the kitchen", 3, "handwashing"),
    "walls_floor": ("repair the damaged floor and wall surfaces in the wash-up area", 2, "structure_or_paperwork"),
    "canopy_clean": ("deep-clean the extraction canopy and filters", 1, "structure_or_paperwork"),
    "food_safety_plan": ("write and implement a documented food-safety management plan", 2, "structure_or_paperwork"),
    "training": ("train all food handlers to basic food-hygiene level", 1, "structure_or_paperwork"),
}


class NoticeComplianceRevisit(Schema):
    name = "notice_compliance_revisit"
    description = "A revisit after a formal improvement notice expired: sign off, extend, escalate or prohibit."
    surfaces = {"json": 0.5, "revisit_checklist": 0.25, "revisit_letter": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind, bname, _risk = _business(rng)
        n_items = pick(rng, [3, 4, 5])
        items = rng.sample(list(NOTICE_ITEMS), n_items)
        status = {k: pick_w(rng, {"done": 5, "partial": 3, "not_started": 2}) for k in items}
        overdue = pick(rng, [0, 0, 3, 10, 21])
        interim = pick_w(rng, {"none": 3, "some": 3})
        reason = pick_w(rng, {"contractor_delay": 3, "cost": 2, "new_owner": 1, "none_given": 2})
        extensions = pick(rng, [0, 0, 1, 2])
        new_issue = pick_w(rng, {"none": 5, "minor": 3, "serious": 1})
        contacted = chance(rng, 0.5)
        if reg == "clear":
            if chance(rng, 0.5):
                status = {k: "done" for k in items}
                overdue, new_issue, extensions = 0, "none", 0
            else:
                top = max(items, key=lambda k: NOTICE_ITEMS[k][1])
                status = {k: pick(rng, ["not_started", "partial"]) for k in items}
                status[top] = "not_started"
                overdue, reason, contacted, extensions = pick(rng, [21, 35]), "none_given", False, pick(rng, [1, 2])
        elif reg == "borderline":
            ks = sorted(items, key=lambda k: NOTICE_ITEMS[k][1])
            status = {k: "done" for k in items}
            status[pick(rng, ks[:2])] = pick(rng, ["partial", "not_started"])
            overdue = pick(rng, [0, 3, 7])
        else:  # conflicting: most work done but the key item missing; or little done but a credible reason and cover
            top = max(items, key=lambda k: NOTICE_ITEMS[k][1])
            if chance(rng, 0.5):
                status = {k: "done" for k in items}
                status[top] = pick(rng, ["partial", "not_started"])
                interim = "none"
            else:
                status = {k: pick(rng, ["partial", "not_started"]) for k in items}
                reason, contacted, interim, extensions = "contractor_delay", True, "some", 0
        return {"regime": reg, "business": bname, "business_type": kind, "items": items, "status": status,
                "days_past_deadline": overdue, "interim_measures": interim, "reason": reason,
                "extensions_already": extensions, "new_issue": new_issue, "contacted_before_deadline": contacted,
                "authority": _authority(rng), "officer": person(rng), "operator": person(rng),
                "notice_served": rand_date(rng), "notice_weeks": pick(rng, [4, 6, 8]), "ref": ident(rng, "IN", 5, "-")}

    def _interim_text(self, f):
        if f["interim_measures"] == "none":
            return None
        opts = {"pest_proofing": "pest contractor visiting weekly with bait stations in place",
                "replace_fridge": "high-risk food moved to a second fridge that holds temperature",
                "handwash_basin": "portable hand-wash unit with hot water in the kitchen",
                "walls_floor": "damaged area sealed off with temporary covering"}
        open_items = [k for k in f["items"] if f["status"][k] != "done" and k in opts]
        return opts[open_items[0]] if open_items else None

    def _reason_text(self, f):
        return {"contractor_delay": "the contractor postponed the work twice",
                "cost": "the owner says the work is unaffordable this quarter",
                "new_owner": "the business changed hands a month ago",
                "none_given": None}[f["reason"]]

    def _new_text(self, f):
        return {"none": None, "minor": "new at this visit: prepared foods not date-labelled",
                "serious": "new at this visit: raw meat stored above ready-to-eat food"}[f["new_issue"]]

    def render(self, f, rng, surface):
        served = f["notice_served"]
        deadline = served + timedelta(weeks=f["notice_weeks"])
        visit = deadline + timedelta(days=f["days_past_deadline"])
        st = {"done": "complete", "partial": "partly done", "not_started": "not started"}
        if surface == "json":
            return dumps(prune({
                "notice": {"ref": f["ref"], "served": fmt_date(served, style="iso"),
                           "deadline": fmt_date(deadline, style="iso"),
                           "extensions_already_granted": f["extensions_already"]},
                "revisit": {"date": fmt_date(visit, style="iso"), "officer": f["officer"],
                            "items": [{"requirement": NOTICE_ITEMS[k][0], "status": st[f["status"][k]]}
                                      for k in f["items"]],
                            "interim_measure_seen": self._interim_text(f), "new_findings": self._new_text(f)},
                "operator": {"contacted_officer_before_deadline": f["contacted_before_deadline"],
                             "reason_given": self._reason_text(f)},
                "premises": {"name": f["business"], "type": f["business_type"], "authority": f["authority"]},
            }))
        ds = _date_style(rng)
        if surface == "revisit_checklist":
            rows = [[str(i + 1), NOTICE_ITEMS[k][0], st[f["status"][k]]] for i, k in enumerate(f["items"])]
            head = (f"NOTICE REVISIT -- {f['ref']} -- {f['business']} ({f['business_type']})\n"
                    f"Served {fmt_date(served, style=ds)}; deadline {fmt_date(deadline, style=ds)}; revisit "
                    f"{fmt_date(visit, style=ds)}; extensions already granted: {f['extensions_already']}\n")
            notes = bullets([("Interim measure: " + self._interim_text(f)) if self._interim_text(f) else None,
                             _cap(self._new_text(f)) if self._new_text(f) else None,
                             ("Operator contacted us before the deadline" if f["contacted_before_deadline"] else
                              "No contact from the operator before the deadline"),
                             ("Reason given: " + self._reason_text(f)) if self._reason_text(f) else "No reason given"],
                            rng)
            return head + "\n" + table(["#", "requirement", "status"], rows, rng) + "\n\n" + notes
        s = [f"I revisited {f['business']} on {fmt_date(visit, style=ds)} to check compliance with improvement "
             f"notice {f['ref']}, served on {fmt_date(served, style=ds)} with a deadline of "
             f"{fmt_date(deadline, style=ds)}."]
        for k in f["items"]:
            s.append(f"Requirement to {NOTICE_ITEMS[k][0]}: {st[f['status'][k]]}.")
        if self._interim_text(f):
            s.append(f"In the meantime I saw a {self._interim_text(f)}.")
        if self._new_text(f):
            s.append(_cap(self._new_text(f)) + ".")
        s.append(("You contacted me before the deadline" if f["contacted_before_deadline"] else
                  "I received no contact from you before the deadline")
                 + (f" and told me {self._reason_text(f)}." if self._reason_text(f) else "."))
        if f["extensions_already"]:
            s.append(f"The deadline has already been extended {_plural(f['extensions_already'], 'time')}."
                     .replace("1 time", "once").replace("2 times", "twice"))
        head = f["authority"] if f["authority"].endswith("Environmental Health") else \
            f"{f['authority']} -- Environmental Health"
        return "\n".join([head, "", f"Dear {f['operator']},", "",
                          noisy(join_sentences(s), rng, 0.05), "", "Yours faithfully,", f["officer"]])

    def questions(self):
        return {
            "notice_outcome": choice("What should the officer do about the improvement notice now?", {
                "emergency_prohibition": "Stop food operations at once because of an imminent risk to health.",
                "grant_extension": "Extend the deadline so that the remaining work can be finished.",
                "prosecution_file": "Prepare a prosecution file for failing to comply with the notice.",
                "serve_further_notice": "Serve a fresh notice on the outstanding work with a new, shorter deadline.",
                "sign_off_notice": "Record the notice as complied with and close it.",
            }),
            "most_pressing_open_item": choice("Which outstanding matter poses the greatest risk to food safety?", {
                "cold_storage": "Refrigeration that cannot keep food cold enough.",
                "handwashing": "Hand-washing facilities for food handlers.",
                "nothing_material_open": "Nothing material; the work is done or only trivial items remain.",
                "pest_proofing": "Proofing the building against pests.",
                "structure_or_paperwork": "Surfaces, cleaning, training or written procedures.",
            }),
            "compliance_progress": score("How much progress has the business made on the notice?", [
                "None: nothing has been done.",
                "Little: minor items only, the important work untouched.",
                "Substantial: most of the work done, something important left.",
                "Complete: everything required has been done.",
            ]),
            "next_visit_timing": score("How soon should the officer visit the premises again?", [
                "Not before the next routine inspection, months away.",
                "In about a month.",
                "Within a week or two.",
                "Within a day or two.",
            ]),
            "more_time_justified": noul(
                "Giving the business more time to comply is justified.",
                yes="Real progress, a credible reason and interim protection make a short extension reasonable.",
                no="Little has been done, no good reason is given, or the risk cannot wait."),
        }


# =========================================================================== schema 5
# business type -> [(product, allergens it can involve, findings that fit it)]
ALLERGEN_PRODUCTS = {
    "bakery": [("seeded rolls", ["sesame"], ["label_missing", "test_purchase", "recipe_changed"]),
               ("flapjacks", ["tree nuts", "peanut"], ["label_missing", "test_purchase", "recipe_changed"]),
               ("custard slices", ["egg", "milk"], ["label_missing", "test_purchase"]),
               ("'gluten-free' loaves", ["gluten"], ["lab_positive"])],
    "takeaway": [("chicken korma", ["tree nuts", "milk"], ["test_purchase", "recipe_changed"]),
                 ("satay skewers", ["peanut"], ["test_purchase", "label_missing"]),
                 ("chips sold as gluten-free", ["gluten"], ["shared_fryer"]),
                 ("'nut-free' children's curry", ["peanut"], ["lab_positive"])],
    "café": [("pesto sandwiches", ["tree nuts"], ["label_missing", "test_purchase", "recipe_changed"]),
             ("pesto sandwiches", ["milk"], ["label_missing", "test_purchase"]),
             ("'dairy-free' brownies", ["milk"], ["lab_positive"]),
             ("house granola pots", ["tree nuts", "sesame"], ["label_missing", "recipe_changed"])],
    "restaurant": [("house salad dressing", ["mustard", "egg"], ["test_purchase", "recipe_changed"]),
                   ("mushroom risotto", ["celery"], ["test_purchase", "recipe_changed"]),
                   ("mushroom risotto", ["milk"], ["test_purchase"]),
                   ("'dairy-free' dessert of the day", ["milk"], ["lab_positive", "test_purchase"])],
    "sushi bar": [("crab-style rolls", ["crustaceans", "egg"], ["test_purchase", "label_missing"]),
                  ("sesame seaweed salad", ["sesame"], ["label_missing", "recipe_changed"]),
                  ("vegetable tempura sold as shellfish-free", ["crustaceans"], ["shared_fryer"])],
}
PROTEIN = {"peanut": "peanut protein", "sesame": "sesame protein", "milk": "milk protein", "egg": "egg protein",
           "gluten": "gluten", "mustard": "mustard protein", "celery": "celery protein",
           "crustaceans": "crustacean protein", "tree nuts": "tree-nut protein"}
FRY_SOURCE = {"gluten": "battered fish", "crustaceans": "prawn tempura"}
SALE_MODE = {"loose": "sold loose over the counter; allergen information may be given in writing or verbally",
             "prepacked_on_site": "packed on the premises before sale; a full ingredient list with allergens "
                                  "emphasised is required on the pack",
             "prepacked_retail": "prepacked and sold through other shops; a full label with allergens emphasised is "
                                 "required"}


class AllergenControlFinding(Schema):
    name = "allergen_control_finding"
    description = "An allergen-control failure found at a food business: advise, correct, withdraw, alert or take formal action."
    surfaces = {"json": 0.5, "sampling_report": 0.25, "case_summary": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        kind, bname, _risk = _business(rng, list(ALLERGEN_PRODUCTS))
        product, allergens, findings = pick(rng, ALLERGEN_PRODUCTS[kind])
        allergen, finding = pick(rng, allergens), pick(rng, findings)
        mode = pick_w(rng, {"loose": 3, "prepacked_on_site": 3, "prepacked_retail": 1})
        reaction = pick_w(rng, {"none": 6, "mild": 2, "severe": 1})
        units = pick(rng, [20, 80, 300, 1500])
        matrix = pick_w(rng, {"current": 3, "out_of_date": 3, "none": 2})
        training = pick_w(rng, {"all": 3, "some": 3, "none": 2})
        fixed = pick_w(rng, {"none": 3, "withdrew": 2, "relabelled": 2})
        other_lines = pick(rng, [0, 2, 6, 15])
        threshold = 20 if allergen == "gluten" else 5
        ppm = between(rng, 3, 90, integer=True)
        if reg == "clear":
            if chance(rng, 0.5):   # a slip, already put right, at a well-run business
                reaction, matrix, training, fixed, units = "none", "current", "all", pick(rng, ["relabelled", "withdrew"]), 20
                ppm = pick(rng, [threshold + 2, threshold + 4])
            else:                  # widespread exposure, harm, nothing in place
                reaction, matrix, training, fixed, units = "severe", "none", "none", "none", 1500
                mode = "prepacked_retail" if finding in ("label_missing", "lab_positive") else mode
                ppm = pick(rng, [threshold * 5, threshold * 10])
        elif reg == "borderline":
            reaction = "none"
            ppm = near(rng, threshold, threshold * 0.5, lo=1, integer=True)
            units = pick(rng, [80, 300])
        else:  # conflicting: a severe reaction at a business with good controls, or no harm and no controls
            if chance(rng, 0.5):
                reaction, matrix, training = "severe", "current", "all"
            else:
                reaction, matrix, training, fixed = "none", "none", "none", "none"
                units = pick(rng, [300, 1500])
        if finding == "label_missing":
            mode = pick(rng, ["prepacked_on_site", "prepacked_retail"])
        elif finding in ("test_purchase", "shared_fryer"):
            mode = "loose"
        if kind != "bakery" and mode == "prepacked_retail":
            mode = "prepacked_on_site"
        return {"regime": reg, "business": bname, "business_type": kind, "product": product, "allergen": allergen,
                "sale_mode": mode, "finding": finding, "ppm": ppm if finding == "lab_positive" else None,
                "ppm_reference": threshold, "reaction": reaction, "units_sold_estimate": units,
                "allergen_matrix": matrix, "staff_training": training, "operator_fix": fixed,
                "other_products_same_line": other_lines, "authority": _authority(rng), "officer": person(rng),
                "date": rand_date(rng), "ref": ident(rng, "ALG", 5, "-"),
                # for cross-contact findings: does the menu or label already warn about it?
                "precaution_stated": finding in ("shared_fryer", "lab_positive") and chance(rng, 0.5)}

    def _precaution_text(self, f):
        if f["finding"] not in ("shared_fryer", "lab_positive"):
            return None
        a = f["allergen"]
        if f["finding"] == "shared_fryer":
            return (f"the menu states that fried items share oil with {FRY_SOURCE[a]}" if f["precaution_stated"]
                    else "nothing on the menu mentions the shared fryer oil")
        return (f"the label carries a 'may contain {a}' warning" if f["precaution_stated"]
                else f"the label gives no 'may contain {a}' warning")

    def _finding_text(self, f):
        a, p = f["allergen"], f["product"]
        if f["finding"] == "lab_positive":
            ref = ("the 'gluten-free' limit is 20 ppm" if a == "gluten" else
                   f"the laboratory's reporting level is {f['ppm_reference']} ppm")
            return f"a sample of the {p} contained {f['ppm']} ppm of {PROTEIN[a]} ({ref})"
        if f["finding"] == "shared_fryer":
            return f"the {p} are fried in the same oil as {FRY_SOURCE[a]}"
        it = "them" if a in ("tree nuts", "crustaceans") else "it"
        return {"test_purchase": f"at a test purchase, staff said the {p} contained no {a}, but the recipe includes {it}",
                "label_missing": f"the pack label for the {p} lists ingredients but does not declare {a}, which is "
                                 f"in the recipe",
                "recipe_changed": f"the {p} recipe was changed to include {a} but the label and menu were not "
                                  f"updated"}[f["finding"]]

    def _reaction_text(self, f):
        return {"none": "no reactions reported",
                "mild": f"one customer reported a mild reaction (itching, rash) after eating the {f['product']}",
                "severe": f"one customer needed emergency treatment for anaphylaxis after eating the {f['product']}"
                }[f["reaction"]]

    def _controls_text(self, f):
        return (f"allergen matrix: {f['allergen_matrix'].replace('_', ' ')}; staff allergen training: "
                f"{ {'all': 'all staff', 'some': 'some staff', 'none': 'none'}[f['staff_training']] }")

    def _fix_text(self, f):
        return {"none": "no action taken by the business yet",
                "withdrew": "the business has taken the product off sale",
                "relabelled": "the business has corrected the label and menu"}[f["operator_fix"]]

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "premises": {"name": f["business"], "type": f["business_type"], "authority": f["authority"]},
                "product": {"name": f["product"], "how_sold": SALE_MODE[f["sale_mode"]],
                            "estimated_units_sold": f["units_sold_estimate"],
                            "other_products_made_on_same_line": f["other_products_same_line"]},
                "finding": {"ref": f["ref"], "date": fmt_date(f["date"], style="iso"), "allergen": f["allergen"],
                            "what_was_found": self._finding_text(f), "precautionary_information": self._precaution_text(f),
                            "reactions": self._reaction_text(f)},
                "controls": {"allergen_matrix": f["allergen_matrix"], "staff_training": f["staff_training"],
                             "business_response": self._fix_text(f)},
            }))
        if surface == "sampling_report":
            rows = [("Reference", f["ref"]), ("Date", fmt_date(f["date"], rng)),
                    ("Premises", f"{f['business']} ({f['business_type']})"), ("Product", f["product"]),
                    ("How sold", SALE_MODE[f["sale_mode"]]), ("Allergen", f["allergen"]),
                    ("Result", self._finding_text(f)), ("Precautionary information", self._precaution_text(f)),
                    ("Reactions", self._reaction_text(f)),
                    ("Estimated units sold", f["units_sold_estimate"]),
                    ("Other products on same line", f["other_products_same_line"]),
                    ("Controls", self._controls_text(f)), ("Business response", self._fix_text(f))]
            return f"{f['authority'].upper()} -- ALLERGEN CONTROL\n" + kv_lines(
                rows, rng, style=pick(rng, ["colon", "aligned", "dash"]))
        s = [f"{f['business']} ({f['business_type']}): {self._finding_text(f)}"
             + (f"; {self._precaution_text(f)}." if self._precaution_text(f) else "."),
             f"The {f['product']} are {SALE_MODE[f['sale_mode']].split(';')[0]}; about {f['units_sold_estimate']} sold "
             f"since the problem began.", _cap(self._reaction_text(f)) + ".",
             f"Controls: {self._controls_text(f)}.", _cap(self._fix_text(f)) + ".",
             f"{f['other_products_same_line']} other products are made on the same line or with the same staff."
             if f["other_products_same_line"] else "No other products share the line."]
        return (f"CASE SUMMARY {f['ref']} -- {f['officer']}, {fmt_date(f['date'], rng)}\n\n"
                + noisy(join_sentences(s), rng, 0.08))

    def questions(self):
        return {
            "allergen_response": choice("How should the authority respond to the allergen finding?", {
                "advice_only": "Advise the business and record the matter; no further action.",
                "formal_notice_on_controls": "Serve a formal notice requiring the business to overhaul its allergen "
                                             "controls.",
                "public_allergy_alert": "Have the product recalled from customers and publish an allergy alert.",
                "require_label_or_menu_fix": "Require the label, menu or allergen information to be corrected.",
                "withdraw_from_sale": "Require the product to be taken off sale until it is safe.",
            }),
            "allergic_harm_likelihood": score("How likely is an allergic customer to be harmed by the food as it is "
                                              "currently sold?", [
                "Remote: the problem is fixed or the exposure is trivial.",
                "Low: an allergic customer would probably be warned or unaffected.",
                "Real: an allergic customer could easily eat it unknowingly.",
                "High: allergic customers are being exposed now, or have already been harmed.",
            ]),
            "allergen_control_maturity": score("How mature are the business's allergen controls overall?", [
                "Absent: no allergen matrix and no staff allergen training.",
                "Weak: no matrix or no training, or an out-of-date matrix with only some staff trained.",
                "Adequate: a matrix and training in place, but the matrix is out of date or not all staff are trained.",
                "Strong: a current matrix and all staff trained; this was an isolated slip.",
            ]),
            "information_meets_rules": noul(
                "The allergen information given for this product meets the rules for how it is sold.",
                yes="The allergen is now declared in the form the sale requires (the business has corrected the label "
                    "and menu, or a precautionary warning covers the cross-contact, or a lab result is below the "
                    "reporting level).",
                no="The allergen is missing, wrong or given in a form the sale does not allow, and nothing has corrected it."),
            "other_products_affected": noul("The failure is likely to affect other products made by the business."),
        }


WORKFLOW = Workflow(
    name="food_safety_inspection",
    title="Food safety inspection outcomes",
    description="What a local food-safety authority does after inspecting a food business: follow-up, enforcement, "
                "exclusion, withdrawal or closure.",
    domain="public health / food safety regulation",
    schemas=[RoutineInspectionOutcome(), HaccpRecordReview(), IllnessReportInvestigation(), NoticeComplianceRevisit(),
             AllergenControlFinding()],
    notes="The authority's decisions on fictional premises, from the officer's side. Illness reports arrive as case "
          "records; no reply to a member of the public is drafted or triaged.",
)
