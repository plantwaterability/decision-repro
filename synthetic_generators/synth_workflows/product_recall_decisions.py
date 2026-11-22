"""Product recall decisions: what a manufacturer should do about a safety signal from product already sold.

The manufacturer's own product-safety committee reads field-failure data, lot traceability, single incident
reports, recall progress and upstream component notices, and decides between monitoring, a field fix, a stop-ship
or a recall, what to tell the product-safety regulator, and how wide the action must reach. All products,
brands, suppliers and regulators are fictional; the regulator is always "the product-safety regulator".
No customer conversation, reply drafting or churn question appears anywhere.
"""
from __future__ import annotations

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
    lognormal_amount,
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
    table,
)

# key -> (product, model prefix, hazard, components that can fail, failure description, typical misuse claim)
PRODUCTS = {
    "air_fryer": ("countertop air fryer", "AF", "fire", ["thermal cut-off", "heating-element terminal"],
                  "overheats and scorches the rear of the housing", "drawer blocked with baking paper"),
    "tool_battery": ("18 V cordless tool battery pack", "BP", "fire", ["cell batch", "battery management board"],
                     "swells and vents smoke while charging", "packs charged on non-original chargers"),
    "ebike_charger": ("e-bike charger", "EC", "fire", ["charging PCB", "output connector"],
                      "casing melts where the output lead enters", "chargers left out in the rain"),
    "oil_radiator": ("portable oil-filled radiator", "OR", "fire", ["cord strain relief", "cord crimp terminal"],
                     "cord overheats at the strain relief", "heaters run from coiled extension reels"),
    "pressure_washer": ("electric pressure washer", "PW", "shock", ["motor shaft seal", "housing gasket"],
                        "water tracks into the motor housing", "units stored outdoors over winter"),
    "kettle": ("stainless steel kettle", "KT", "scald", ["lid hinge", "lid catch spring"],
               "lid springs open while pouring", "kettles filled past the maximum line"),
    "step_ladder": ("aluminium step ladder", "SL", "fall", ["brace latch spring", "locking brace"],
                    "locking brace disengages under load", "ladders used on uneven ground"),
    "magnet_tiles": ("magnetic building tiles for children", "MT", "ingestion", ["tile edge weld", "magnet pocket"],
                     "small magnets come loose from the tiles", "tiles chewed by toddlers below the age grade"),
    "desk_frame": ("height-adjustable desk frame", "DF", "crush", ["column motor controller", "anti-collision sensor"],
                   "desk keeps moving after the button is released", "frames loaded beyond the rated weight"),
    "baby_monitor": ("video baby monitor", "BM", "strangulation", ["bracket wall anchor", "wall-mount bracket"],
                     "wall bracket loosens and the camera and cord drop towards the cot", "cameras fixed with the wrong wall plugs"),
}
# worst credible outcome of each hazard, 1 (minor) .. 5 (life-threatening)
HAZARD_CEILING = {"fire": 4, "shock": 5, "scald": 3, "fall": 4, "ingestion": 5, "crush": 3, "strangulation": 5}
HAZARD_WORDS = {"fire": "fire / overheating", "shock": "electric shock", "scald": "scalding", "fall": "fall from height",
                "ingestion": "magnet ingestion", "crush": "crush / pinch", "strangulation": "cord entanglement"}
BRAND_SUFFIX = {"air_fryer": ["Home", "Appliances", "Kitchen"], "tool_battery": ["Tools", "Power Tools"],
                "ebike_charger": ["Cycles", "E-Mobility"], "oil_radiator": ["Home", "Electrics", "Appliances"],
                "pressure_washer": ["Garden", "Tools", "Outdoor"], "kettle": ["Home", "Kitchen", "Appliances"],
                "step_ladder": ["Tools", "Home"], "magnet_tiles": ["Kids", "Toys", "Play"],
                "desk_frame": ["Office", "Furniture", "Living"], "baby_monitor": ["Baby", "Kids", "Home"]}


def _name(rng, suffixes):
    """ORG_A prefix + one of `suffixes`, skipping combinations that name a real business (as base.org does)."""
    a, i = rng.choice(ORG_A), rng.randrange(len(suffixes))
    for k in range(len(suffixes)):
        n = f"{a} {suffixes[(i + k) % len(suffixes)]}"
        if n not in REAL_NAME_CLASHES:
            return n
    return n


def _brand(rng, key):
    return _name(rng, BRAND_SUFFIX[key])


def _product(rng, keys=None):
    key = pick(rng, keys or list(PRODUCTS))
    name, prefix, hazard, comps, failure, misuse = PRODUCTS[key]
    return key, name, f"{prefix}-{rng.randint(100, 999)}", hazard, comps, failure, misuse


def _spread(total, weights):
    """Split an integer total over weights (largest remainder), so the parts add back up exactly."""
    tw = sum(weights) or 1
    raw = [total * w / tw for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[:total - sum(out)]:
        out[i] += 1
    return out


def _people(rng, n):
    out, seen = [], []
    while len(out) < n:
        p = person(rng)
        if p.split()[0] not in seen:
            seen.append(p.split()[0])
            out.append(p)
    return out


def _plural(n, word):
    if n == 1:
        return f"1 {word}"
    return f"{n} " + (word[:-1] + "ies" if word.endswith("y") and word[-2] not in "aeiou" else word + "s")


def _thousands(n):
    return f"{n:,}"


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def maybe_attrition(rng):
    return pick(rng, [None, None, None, 20, 35])


def _money(x, rng, cur):
    return money(x, rng, cur, style=pick(rng, ["sym", "sym", "code"]))


# =========================================================================== schema 1
class FieldFailureSignal(Schema):
    name = "field_failure_signal"
    description = "A cluster of field failures on a product already sold: monitor, fix in the field, stop shipping or recall."
    surfaces = {"json": 0.5, "signal_report": 0.25, "engineering_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        key, pname, model, hazard, comps, failure, misuse = _product(rng)
        units = int(round(lognormal_amount(rng, 150000, 0.7), -3))
        baseline = pick(rng, [2.0, 3.5, 5.0, 8.0])
        trigger = pick(rng, [1.5, 2.0, 2.0, 3.0])
        ratio = between(rng, 0.8, 4.0)
        minor, serious, damage = pick(rng, [0, 0, 1, 2, 4]), pick(rng, [0, 0, 0, 1]), pick(rng, [0, 0, 1, 3])
        cause = pick_w(rng, {"unknown": 3, "suspected": 4, "confirmed": 2})
        lab = pick_w(rng, {"reproduced": 2, "in_progress": 4, "not_reproduced": 2, "not_started": 1})
        misuse_pct = pick(rng, [5, 10, 20, 35, 50, 65])
        trend = pick_w(rng, {"rising": 3, "flat": 4, "falling": 2})
        shipping = chance(rng, 0.7)
        if reg == "clear":
            if chance(rng, 0.5):   # noise: near baseline, no harm, mostly misuse
                ratio, minor, serious, damage, cause, lab, misuse_pct, trend = between(rng, 0.7, 1.2), 0, 0, 0, \
                    "unknown", "not_reproduced", pick(rng, [60, 70, 80]), pick(rng, ["flat", "falling"])
            else:                  # an unmistakable defect with harm
                ratio, serious, damage, cause, lab, misuse_pct, trend = between(rng, 4.5, 9.0), pick(rng, [1, 2, 3]), \
                    pick(rng, [3, 6, 9]), "confirmed", "reproduced", pick(rng, [0, 5, 10]), "rising"
                minor = max(minor, 2)
        elif reg == "borderline":
            ratio = near(rng, trigger, trigger * 0.25, lo=0.5)
            serious = 0
            minor = pick(rng, [0, 1, 2])
            cause, lab = "suspected", pick(rng, ["in_progress", "not_started", "not_reproduced"])
        else:  # conflicting
            if chance(rng, 0.5):   # a high rate with no harm and heavy misuse claims
                ratio, minor, serious, damage, misuse_pct, lab = between(rng, 3.0, 6.0), 0, 0, pick(rng, [0, 1]), \
                    pick(rng, [45, 55, 65]), pick(rng, ["not_reproduced", "in_progress"])
            else:                  # a quiet rate but one serious injury, reproduced in the lab
                ratio, serious, lab, cause, trend = between(rng, 0.9, 1.6), 1, "reproduced", \
                    pick(rng, ["suspected", "confirmed"]), pick(rng, ["flat", "falling"])
            if chance(rng, 0.4):
                cause = "unknown"
        rate = round(baseline * ratio, 1)
        reports = max(1, int(round(rate * units / 1e5)))
        rate = round(reports / units * 1e5, 1)
        return {"regime": reg, "product_key": key, "product": pname, "model": model, "hazard": hazard,
                "brand": _brand(rng, key), "units_in_field": units, "months_on_market": pick(rng, [4, 8, 14, 22, 36]),
                "still_shipping": shipping, "reports": reports, "rate_per_100k": rate, "baseline_per_100k": baseline,
                "trigger_multiple": trigger, "trend_8wk": trend, "minor_injuries": minor, "serious_injuries": serious,
                "property_damage_reports": damage, "cause_status": cause, "suspect_component": pick(rng, comps),
                "failure": failure, "misuse_claim": misuse, "lab_status": lab, "misuse_pct": misuse_pct,
                "days_since_first_report": pick(rng, [6, 12, 25, 40, 75, 120]),
                "reporting_rule_days": pick(rng, [10, 10, 20]),
                "pm_comment": pick(rng, ["Return rate is still inside the category norm.",
                                         "Retail partners have not raised anything with us.",
                                         "A recall would land in the middle of peak season.",
                                         "Most of these came through one retailer's returns desk.", None]),
                "engineer": None, "pm": None, "case_id": ident(rng, "PSR", 5, "-"), "date": rand_date(rng)} | \
            dict(zip(("engineer", "pm"), _people(rng, 2)))

    def _cause_text(self, f):
        return {"unknown": "No cause identified yet.",
                "suspected": f"Suspected cause: the {f['suspect_component']}; not yet proven.",
                "confirmed": f"Cause confirmed: the {f['suspect_component']}."}[f["cause_status"]]

    def _lab_text(self, f):
        return {"reproduced": "the lab has reproduced the failure on returned units",
                "in_progress": "lab testing of returned units is under way",
                "not_reproduced": "the lab could not reproduce the failure on returned units",
                "not_started": "no returned units have been tested yet"}[f["lab_status"]]

    def render(self, f, rng, surface):
        harm = []
        if f["serious_injuries"]:
            harm.append(f"{_plural(f['serious_injuries'], 'injury')} needing hospital treatment")
        if f["minor_injuries"]:
            harm.append(f"{f['minor_injuries']} minor (first-aid) injur{'y' if f['minor_injuries'] == 1 else 'ies'}")
        if f["property_damage_reports"]:
            harm.append(f"{_plural(f['property_damage_reports'], 'report')} of property damage")
        harm_txt = "; ".join(harm) or "no injuries or property damage reported"
        if surface == "json":
            return dumps(prune({
                "product": {"brand": f["brand"], "name": f["product"], "model": f["model"],
                            "units_in_field": f["units_in_field"], "months_on_market": f["months_on_market"],
                            "still_shipping": f["still_shipping"]},
                "signal": {"field_reports": f["reports"], "failure_mode": f["failure"],
                           "reports_per_100k_units": f["rate_per_100k"],
                           "family_baseline_per_100k": f["baseline_per_100k"],
                           "review_trigger": f"{f['trigger_multiple']}x baseline", "trend_last_8_weeks": f["trend_8wk"],
                           "days_since_first_report": f["days_since_first_report"]},
                "harm": {"serious_injuries": f["serious_injuries"], "minor_injuries": f["minor_injuries"],
                         "property_damage_reports": f["property_damage_reports"]},
                "investigation": {"cause": self._cause_text(f), "lab": self._lab_text(f),
                                  "reports_alleging_misuse_pct": f["misuse_pct"],
                                  "misuse_alleged": f["misuse_claim"] if f["misuse_pct"] >= 20 else None},
                "policy": {"regulator_reporting_window_working_days": f["reporting_rule_days"]},
                "comments": {"product_manager": noisy(f["pm_comment"], rng, 0.15) if f["pm_comment"] else None},
            }))
        if surface == "signal_report":
            d = f["days_since_first_report"]
            recent = f["reports"] if d <= 56 else max(1, round(f["reports"] * 56 / d))
            live = min(8, -(-d // 7))                     # weeks since the first report, capped at 8
            w = [0.0] * (8 - live) + [max(0.2, 1 + {"rising": (i - live / 2) * 0.25, "flat": 0.0,
                                                   "falling": (live / 2 - i) * 0.2}[f["trend_8wk"]]
                                              + rng.uniform(-0.3, 0.3)) for i in range(live)]
            weeks = _spread(recent, w)
            head = kv_lines([("Case", f["case_id"]), ("Product", f"{f['brand']} {f['product']} {f['model']}"),
                             ("Units in field", _thousands(f["units_in_field"])),
                             ("On market", f"{f['months_on_market']} months"
                              + (", still shipping" if f["still_shipping"] else ", shipments paused")),
                             ("Failure mode", f["failure"]), ("Hazard class", HAZARD_WORDS[f["hazard"]]),
                             ("Rate", f"{f['rate_per_100k']} per 100k (family baseline {f['baseline_per_100k']}; "
                                      f"review trigger {f['trigger_multiple']}x)"),
                             ("Harm", harm_txt)], rng, style="aligned")
            tbl = table(["week", *[f"W-{8 - i}" for i in range(8)]], [["reports", *weeks]], rng, style="fixed")
            notes = bullets([self._cause_text(f), self._lab_text(f).capitalize() + ".",
                             f"{f['misuse_pct']}% of reports allege misuse" + (f" ({f['misuse_claim']})"
                                                                               if f["misuse_pct"] >= 20 else "") + ".",
                             f"First report {f['days_since_first_report']} days ago; reporting window is "
                             f"{f['reporting_rule_days']} working days once a serious risk is suspected.",
                             f"PM: {f['pm_comment']}" if f["pm_comment"] else None], rng)
            return f"FIELD SIGNAL REVIEW\n{head}\n\nWeekly field reports (last 8 weeks):\n{tbl}\n\n{notes}"
        s = [f"Quick summary on the {f['model']} {f['product']} ({_thousands(f['units_in_field'])} units out there, "
             f"{f['months_on_market']} months on sale" + (", still shipping)." if f["still_shipping"] else ", shipments on hold)."),
             f"We now have {_plural(f['reports'], 'field report')} of the same failure ({f['failure']}), which is "
             f"{f['rate_per_100k']} per 100k against a family baseline of {f['baseline_per_100k']} "
             f"(committee trigger {f['trigger_multiple']}x).",
             f"The trend over the last eight weeks is {f['trend_8wk']}.",
             f"Harm so far: {harm_txt}.", self._cause_text(f), f"Also, {self._lab_text(f)}."]
        if f["misuse_pct"] >= 20:
            s.append(f"About {f['misuse_pct']}% of the reports mention {f['misuse_claim']}.")
        s.append(f"First report came in {f['days_since_first_report']} days ago.")
        if f["pm_comment"]:
            s.append(f"{f['pm'].split()[0]} points out: {f['pm_comment'][0].lower() + f['pm_comment'][1:]}")
        return "\n".join([f"Subject: {f['model']} field reports -- for Thursday's safety committee",
                          f"From: {f['engineer']}", "To: product-safety-committee@corp.example", "",
                          noisy(join_sentences(s), rng, 0.12), "", f["engineer"].split()[0]])

    def questions(self):
        return {
            "field_action": choice("What should the product-safety committee do about the product in the field?", {
                "close_without_action": "Close the review; the reports do not point to a product problem.",
                "continue_monitoring": "Keep watching the field data at the next reviews without acting yet.",
                "service_fix_on_request": "Issue a service bulletin and repair affected units free of charge when "
                                          "owners come forward.",
                "stop_shipments_and_investigate": "Stop shipping the product until the cause is found, without "
                                                  "recalling sold units yet.",
                "voluntary_recall": "Recall sold units from consumers and stop sales.",
            }),
            "regulator_step": choice("What should the company do about the product-safety regulator at this point?", {
                "file_reasoned_no_report": "Record a written decision, with reasons, that the matter is not reportable.",
                "full_report_with_plan": "Report now together with a proposed corrective action.",
                "no_contact_needed": "Nothing; the signal is too weak even to need a documented assessment.",
                "preliminary_report_now": "Send a preliminary report now and follow up once the investigation ends.",
            }),
            "worst_credible_harm": score("What is the worst harm a user could credibly suffer from this failure?", [
                "Inconvenience: the product stops working, nothing more.",
                "Minor: a small burn, cut or bruise treated at home.",
                "Moderate: an injury that needs medical treatment.",
                "Serious: hospital admission or a house fire.",
                "Life-threatening: death or permanent disability is plausible.",
            ]),
            "signal_strength": score("How clearly do the field data show a real product problem rather than noise?", [
                "Not at all: within normal variation, with ready explanations.",
                "Faintly: slightly above normal, but explainable.",
                "Clearly: above the trigger and not explained away.",
                "Unmistakably: well above normal and reproduced or confirmed.",
            ]),
            "product_at_fault": noul(
                "The failures come from a fault in the product itself rather than from misuse, installation or wear.",
                yes="The cause is confirmed in a component or the lab reproduced the failure; or a component is "
                    "suspected, the lab has not failed to reproduce it, and fewer than half of reports allege misuse.",
                no="Nothing is confirmed or reproduced, and the lab could not reproduce it, half or more of reports "
                   "allege misuse, or no cause has even been suspected."),
        }


# =========================================================================== schema 2
class LotScopeDetermination(Schema):
    name = "lot_scope_determination"
    description = "Deciding which production lots a corrective action must cover once a faulty component is traced."
    surfaces = {"json": 0.45, "lot_table": 0.3, "scope_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        key, pname, model, hazard, comps, failure, misuse = _product(rng)
        comp = pick(rng, comps)
        n_suspect = pick(rng, [3, 4, 5, 6])
        year = pick(rng, [25, 26])
        start_wk = rng.randint(4, 36)
        per_lot = pick(rng, [2400, 4000, 6000, 9000])
        trace = pick_w(rng, {"serial_level": 2, "lot_level": 4, "date_code_only": 2, "mixed_bins": 1})
        overlap = chance(rng, 0.35)          # the component batch was still in use after the suspect window
        bfail = pick(rng, [0, 0, 1, 2])      # field failures in boundary lots
        btested = pick(rng, [0, 30, 50, 80])
        bsample_fail = 0 if btested == 0 else pick(rng, [0, 0, 0, 1, 2])
        bmode_same = chance(rng, 0.7)
        proposed = pick_w(rng, {"suspect_only": 3, "suspect_plus_boundary": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                trace, overlap, bfail, btested, bsample_fail, proposed = "serial_level", False, 0, 80, 0, "suspect_only"
            else:
                trace, overlap, bfail, btested, bsample_fail = pick(rng, ["date_code_only", "mixed_bins"]), True, \
                    pick(rng, [2, 3]), 50, pick(rng, [2, 3])
                bmode_same, proposed = True, "suspect_only"
        elif reg == "borderline":
            trace, bfail, btested = "lot_level", pick(rng, [0, 1]), pick(rng, [30, 50])
            bsample_fail = pick(rng, [0, 1])
            overlap = chance(rng, 0.3)
        else:  # conflicting: clean samples but a supply overlap, or boundary failures of a different mode
            if chance(rng, 0.5):
                btested, bsample_fail, overlap, bfail = 80, 0, True, 0
            else:
                bfail, bmode_same, btested, bsample_fail = pick(rng, [2, 3]), False, pick(rng, [0, 30]), 0
            if chance(rng, 0.4):
                trace = "date_code_only"
        lots = []
        for i in range(n_suspect + 2):
            wk = start_wk + i - 1
            status = "boundary" if i in (0, n_suspect + 1) else "suspect"
            units = int(per_lot * rng.uniform(0.8, 1.2)) // 10 * 10
            if status == "suspect":
                fails = max(1, int(units * rng.uniform(0.0006, 0.004)))
                tested, tfail = pick(rng, [(20, 3), (30, 5), (40, 4), (0, 0)])
            else:
                fails = 0
                tested, tfail = btested // 2, 0
            lots.append({"lot": f"L{year}{wk:02d}", "build_week": f"20{year}-W{wk:02d}", "units": units,
                         "field_failures": fails, "samples_tested": tested, "samples_failed": tfail,
                         "status": status})
        # spread the boundary findings over the two boundary lots, later lot first
        for j in range(bfail):
            lots[-1 if j % 2 == 0 else 0]["field_failures"] += 1
        for j in range(bsample_fail):
            lots[-1 if j % 2 == 0 else 0]["samples_failed"] += 1
        suspect_units = sum(x["units"] for x in lots if x["status"] == "suspect")
        return {"regime": reg, "product": pname, "model": model, "hazard": hazard, "brand": _brand(rng, key),
                "component": comp, "failure": failure, "component_supplier": _name(rng, ["Components"]),
                "lots": lots, "suspect_units": suspect_units, "traceability": trace, "component_overlap": overlap,
                "boundary_field_failures": bfail, "boundary_failure_mode_same": bmode_same,
                "boundary_samples_tested": btested, "boundary_samples_failed": bsample_fail,
                "proposed_scope": proposed, "registration_pct": pick(rng, [6, 12, 25, 40, 65, 85]),
                "channel": pick_w(rng, {"own_webshop": 2, "big_box_retail": 3, "trade_installers": 2, "mixed": 3}),
                "remedy_cost_per_unit": pick(rng, [14, 28, 45, 90, 160]), "currency": pick(rng, ["USD", "EUR", "GBP"]),
                "author": person(rng), "case_id": ident(rng, "LSD", 4, "-"), "date": rand_date(rng)}

    TRACE = {"serial_level": "serial numbers are linked to component batches in the build records",
             "lot_level": "build records link each production lot to the component batches used in it",
             "date_code_only": "only the build date code is printed on the unit; component batches were not "
                               "recorded per lot",
             "mixed_bins": "components were drawn from a shared line-side bin, so batches mixed across lots"}
    CHANNEL = {"own_webshop": "sold mostly through our own web shop", "big_box_retail": "sold mostly through large "
               "retail chains", "trade_installers": "sold through trade installers who keep job records",
               "mixed": "sold through a mix of retailers, marketplaces and our web shop"}

    def _proposal(self, f):
        s = [x["lot"] for x in f["lots"] if x["status"] == "suspect"]
        if f["proposed_scope"] == "suspect_only":
            return f"lots {s[0]} to {s[-1]} ({_thousands(f['suspect_units'])} units)"
        allu = sum(x["units"] for x in f["lots"])
        return f"lots {f['lots'][0]['lot']} to {f['lots'][-1]['lot']} ({_thousands(allu)} units)"

    def _boundary_note(self, f):
        if not f["boundary_field_failures"]:
            return "No field failures traced to the neighbouring lots."
        mode = ("with the same failure mode" if f["boundary_failure_mode_same"]
                else "but the returned units show a different failure (cracked housings from drops)")
        return f"{_plural(f['boundary_field_failures'], 'field failure')} traced to the neighbouring lots, {mode}."

    def render(self, f, rng, surface):
        cost = _money(f["remedy_cost_per_unit"], rng, f["currency"])
        overlap = ("The component supplier's delivery records show the same component batch was also delivered for "
                   "the weeks after the suspect window." if f["component_overlap"] else
                   "The component batch was fully consumed within the suspect weeks, per the supplier's delivery records.")
        if surface == "json":
            return dumps(prune({
                "case": {"id": f["case_id"], "product": f"{f['brand']} {f['product']}", "model": f["model"],
                         "faulty_component": f["component"], "failure": f["failure"],
                         "component_supplier": f["component_supplier"]},
                "lots": [{k: v for k, v in x.items() if k != "status"} | {"flag": x["status"]} for x in f["lots"]],
                "traceability": self.TRACE[f["traceability"]],
                "evidence": {"component_batch_note": overlap, "neighbouring_lots": self._boundary_note(f)},
                "proposal": {"quality_lead_scope": self._proposal(f), "remedy_cost_per_unit": cost},
                "reach": {"owner_registration_pct": f["registration_pct"], "channel": self.CHANNEL[f["channel"]]},
            }))
        if surface == "lot_table":
            rows = [[x["lot"], x["build_week"], _thousands(x["units"]), x["field_failures"],
                     (f"{x['samples_failed']}/{x['samples_tested']}" if x["samples_tested"] else "not tested"),
                     x["status"]] for x in f["lots"]]
            out = [f"{f['case_id']}  {f['brand']} {f['product']} {f['model']} -- {f['component']} ({f['failure']})", "",
                   table(["lot", "build week", "units", "field failures", "retained samples failed", "flag"], rows, rng),
                   "", bullets([f"Traceability: {self.TRACE[f['traceability']]}.", overlap, self._boundary_note(f),
                                f"Quality lead proposes: {self._proposal(f)}; remedy about {cost} per unit.",
                                f"Owner registration {f['registration_pct']}%; {self.CHANNEL[f['channel']]}."], rng)]
            return "\n".join(out)
        s = [f"Scope question for the {f['model']} {f['component']} action.",
             f"The fault ({f['failure']}) is tied to {f['component_supplier']} parts built into "
             f"{sum(1 for x in f['lots'] if x['status'] == 'suspect')} production lots.",
             f"Traceability: {self.TRACE[f['traceability']]}.", overlap, self._boundary_note(f)]
        if f["boundary_samples_tested"]:
            s.append(f"We pulled {f['boundary_samples_tested']} retained samples from the neighbouring lots; "
                     f"{f['boundary_samples_failed']} failed the bench test.")
        else:
            s.append("Retained samples from the neighbouring lots have not been tested.")
        s += [f"My proposal is {self._proposal(f)}, at roughly {cost} per unit.",
              f"{f['registration_pct']}% of owners registered their product; it was {self.CHANNEL[f['channel']]}."]
        return (f"MEMO {f['case_id']} -- recall scope\nFrom: {f['author']}, Quality\nDate: {fmt_date(f['date'], rng)}\n\n"
                + noisy(join_sentences(s), rng, 0.1))

    def questions(self):
        return {
            "action_scope": choice("Which units should the corrective action cover?", {
                "all_units_ever_made": "Every unit of the model ever made, because the affected ones cannot be isolated.",
                "suspect_and_neighbouring_lots": "The suspect lots plus the lots built just before and after them.",
                "suspect_lots_only": "Only the lots where the faulty component is known to have been used.",
                "whole_component_window": "Every lot built while the supplier's faulty component batch could have been "
                                          "on the line, however long that was.",
            }),
            "traceability_confidence": score("How confidently can affected units be told apart from unaffected ones?", [
                "Not at all: components came from a shared line-side bin, so batches mixed across lots.",
                "Poorly: only a build date code is on the unit; component batches were not recorded per lot.",
                "Reasonably: build records link each production lot to the component batches used in it.",
                "Precisely: serial numbers are linked to component batches.",
            ]),
            "escape_likelihood": score("How likely is it that faulty units exist outside the lots the quality lead "
                                       "proposes?", [
                "Very unlikely: the evidence confines the fault to the proposed lots.",
                "Unlikely, though not ruled out.",
                "Quite possible: some evidence points beyond the proposed lots.",
                "Likely: faulty units almost certainly exist outside them.",
            ]),
            "neighbours_implicated": noul(
                "The evidence implicates lots beyond the ones where the fault was first found.",
                yes="Field failures of the same kind, failed retained samples or component records reach into other lots.",
                no="Nothing so far ties the fault to lots outside the suspect range."),
            "owners_reachable": noul("Most owners of the affected units can be contacted directly."),
        }


# =========================================================================== schema 3
# per product: what happened, and {injury level: consequence}. Only levels that fit the hazard are listed.
INCIDENTS = {
    "air_fryer": ("the air fryer caught fire on a kitchen worktop",
                  {"none": "scorching to the worktop and wall", "first_aid": "mild smoke inhalation",
                   "medical": "burns to the hand while putting out the flames",
                   "hospital": "smoke inhalation needing an overnight stay"}),
    "tool_battery": ("the battery pack vented and ignited on its charger in a garage",
                     {"none": "a burnt workbench", "first_aid": "a small burn to the fingers",
                      "medical": "burns to the forearm", "hospital": "burns and smoke inhalation"}),
    "ebike_charger": ("the charger melted and caught fire in a hallway overnight",
                      {"none": "a scorched floor and smoke damage", "first_aid": "mild smoke inhalation",
                       "medical": "a burn while unplugging it", "hospital": "smoke inhalation, two nights in hospital"}),
    "oil_radiator": ("the power cord melted and started a fire behind a sofa",
                     {"none": "a burnt sofa and smoke damage", "first_aid": "mild smoke inhalation",
                      "medical": "burns to the hand", "hospital": "smoke inhalation needing admission"}),
    "pressure_washer": ("the user got an electric shock while holding the lance",
                        {"none": "a jolt with no lasting effect", "first_aid": "a tingling arm for an hour",
                         "medical": "numbness in the arm, checked at a clinic",
                         "hospital": "heart monitoring overnight after the shock"}),
    "kettle": ("the lid sprang open while pouring and boiling water spilled",
               {"first_aid": "a small scald to the fingers", "medical": "a scald to the back of the hand",
                "hospital": "scalds to the forearm needing specialist dressing"}),
    "step_ladder": ("the ladder folded while the user stood on the third tread",
                    {"first_aid": "bruising", "medical": "a sprained wrist", "hospital": "a fractured hip"}),
    "magnet_tiles": ("a two-year-old swallowed magnets from a broken tile",
                     {"medical": "an X-ray at the emergency department; the magnets passed naturally",
                      "hospital": "surgery to remove two magnets that had joined through the bowel wall"}),
    "desk_frame": ("the desk kept lowering after release and trapped a child's fingers against a chair arm",
                   {"first_aid": "bruised fingers", "medical": "a fractured fingertip"}),
    "baby_monitor": ("the camera fell into the cot and its cord lay across the baby",
                     {"none": "the parent found it before any harm", "medical": "red marks on the baby's neck, "
                                                                            "checked by a doctor"}),
}
INJURY = {"none": "no injury", "first_aid": "treated at home", "medical": "treated by a doctor, not admitted",
          "hospital": "admitted to hospital"}
INJURY_ORDER = ["none", "first_aid", "medical", "hospital"]


def _fit_injury(level, allowed):
    """The allowed level closest to the one the regime asked for."""
    i = INJURY_ORDER.index(level)
    return min(allowed, key=lambda x: (abs(INJURY_ORDER.index(x) - i), -INJURY_ORDER.index(x)))


class IncidentReportability(Schema):
    name = "incident_reportability"
    description = "A single serious incident report on a product: whether, and how fast, it must go to the regulator."
    surfaces = {"json": 0.5, "incident_intake": 0.25, "counsel_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        key, pname, model, hazard, comps, failure, misuse = _product(rng)
        injury = pick_w(rng, {"none": 2, "first_aid": 3, "medical": 3, "hospital": 2})
        source = pick(rng, ["consumer phone call", "retailer returns desk", "insurer's loss adjuster",
                            "social-media video flagged by marketing", "consumer's solicitor"]
                      + (["fire service report"] if hazard == "fire" else ["hospital injury-surveillance report"]))
        ident_status = pick_w(rng, {"serial_and_photos": 4, "description_only": 3, "unit_destroyed": 1})
        conditions = pick_w(rng, {None: 4, "misuse": 3, "modified": 1})
        prior = pick(rng, [0, 0, 1, 2, 4, 7])
        rule_days = pick(rng, [2, 5, 10])
        elapsed = pick(rng, [1, 2, 3, 5, 7, 9])
        exam = pick_w(rng, {"not_started": 3, "unit_requested": 3, "examining": 2, "cause_found": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                injury, ident_status, conditions, prior, exam = "hospital", "serial_and_photos", None, pick(rng, [2, 4, 7]), \
                    "cause_found"
                elapsed = max(elapsed, rule_days - 1)
            else:
                injury, ident_status, conditions, prior = "none", "description_only", "misuse", 0
                source = pick(rng, ["social-media video flagged by marketing", "consumer phone call"])
        elif reg == "borderline":
            injury = pick(rng, ["first_aid", "medical"])
            elapsed = near(rng, rule_days, max(1, rule_days * 0.4), lo=1, integer=True)
            prior = pick(rng, [0, 1, 2])
        else:  # conflicting: serious harm with a doubtful link, or a clear link and little harm
            if chance(rng, 0.5):
                injury, ident_status, conditions = "hospital", pick(rng, ["description_only", "unit_destroyed"]), \
                    pick(rng, ["misuse", "modified"])
            else:
                injury, ident_status, conditions, prior = pick(rng, ["none", "first_aid"]), "serial_and_photos", None, \
                    pick(rng, [3, 5, 8])
            elapsed = pick(rng, [rule_days - 1, rule_days, rule_days + 2])
        event, harms = INCIDENTS[key]
        injury = _fit_injury(injury, list(harms))
        return {"regime": reg, "product": pname, "model": model, "hazard": hazard, "brand": _brand(rng, key),
                "event": event, "consequence": harms[injury], "injury": injury,
                "reported_via": source, "identification": ident_status, "use_conditions": conditions,
                "misuse_claim": misuse, "similar_on_file": prior, "rule_working_days": rule_days,
                "working_days_elapsed": max(0, elapsed), "examination": exam,
                "product_age_months": pick(rng, [1, 3, 7, 13, 26]),
                "insurer_line": chance(rng, 0.3), "reference": ident(rng, "INC", 6, "-"),
                "received": rand_date(rng), "handler": None, "counsel": None} | \
            dict(zip(("handler", "counsel"), _people(rng, 2)))

    def _ident_text(self, f):
        return {"serial_and_photos": "serial number and photos received; model confirmed",
                "description_only": "consumer's description only; no photos or serial number yet",
                "unit_destroyed": "unit reported destroyed; model taken from the consumer's receipt"}[f["identification"]]

    def _cond_text(self, f):
        return {None: None, "misuse": f"consumer mentions {f['misuse_claim']}",
                "modified": "unit had been repaired by a third party before the incident"}[f["use_conditions"]]

    def _exam_text(self, f):
        return {"not_started": "not started", "unit_requested": "unit requested from the consumer",
                "examining": "unit in our lab, examination under way",
                "cause_found": "examination found a product fault consistent with the report"}[f["examination"]]

    def render(self, f, rng, surface):
        rule = (f"report within {f['rule_working_days']} working days of information reasonably suggesting a "
                f"serious risk")
        if surface == "json":
            return dumps(prune({
                "incident": {"reference": f["reference"], "received": fmt_date(f["received"], style="iso"),
                             "reported_via": f["reported_via"], "what_happened": f["event"],
                             "outcome": f["consequence"], "injury": INJURY[f["injury"]]},
                "product": {"brand": f["brand"], "name": f["product"], "model": f["model"],
                            "identification": self._ident_text(f), "age_months": f["product_age_months"],
                            "use_conditions": self._cond_text(f)},
                "history": {"similar_incidents_on_file": f["similar_on_file"]},
                "clock": {"company_rule": rule, "working_days_since_received": f["working_days_elapsed"]},
                "investigation": {"examination": self._exam_text(f),
                                  "insurer_note": ("insurer asks that nothing be filed before its adjuster reports"
                                                   if f["insurer_line"] else None)},
            }))
        if surface == "incident_intake":
            rows = [("Ref", f["reference"]), ("Received", fmt_date(f["received"], rng)), ("Via", f["reported_via"]),
                    ("Product", f"{f['brand']} {f['product']} {f['model']}"), ("Identification", self._ident_text(f)),
                    ("Product age", _plural(f["product_age_months"], "month")), ("What happened", f["event"]),
                    ("Outcome", f["consequence"]), ("Injury", INJURY[f["injury"]]),
                    ("Use conditions", self._cond_text(f)), ("Similar incidents on file", f["similar_on_file"]),
                    ("Examination", self._exam_text(f)),
                    ("Working days since received", f"{f['working_days_elapsed']} (rule: {f['rule_working_days']})")]
            out = "PRODUCT INCIDENT INTAKE\n" + kv_lines(rows, rng, style=pick(rng, ["colon", "aligned", "dash"]),
                                                          skip_none=False)
            if f["insurer_line"]:
                out += "\n\nNote: insurer asks that nothing be filed before its adjuster reports."
            return out
        s = [f"Incident {f['reference']} on the {f['brand']} {f['model']} {f['product']} came in "
             f"{_plural(f['working_days_elapsed'], 'working day')} ago via {f['reported_via']}.",
             f"Reported: {f['event']}; outcome: {f['consequence']} ({INJURY[f['injury']]}).",
             f"Identification: {self._ident_text(f)}."]
        if self._cond_text(f):
            s.append(f"Note the {self._cond_text(f).replace('consumer mentions', 'consumer also mentions')}.")
        s += [(f"We have {_plural(f['similar_on_file'], 'similar incident')} on file for this model."
               if f["similar_on_file"] else "Nothing similar is on file for this model."),
              f"Examination status: {self._exam_text(f)}.",
              f"Our rule is to {rule}."]
        if f["insurer_line"]:
            s.append("The insurer has asked us not to file anything before its adjuster reports.")
        return "\n".join([f"Subject: {f['reference']} -- do we report?", f"From: {f['handler']}", f"To: {f['counsel']}",
                          "", f"{f['counsel'].split()[0]},", "", noisy(join_sentences(s), rng, 0.12), "",
                          f["handler"].split()[0]])

    def questions(self):
        return {
            "report_decision": choice("What should the company do about reporting this incident to the regulator?", {
                "file_note_not_reportable": "Record why it is not reportable: no injury beyond home treatment and no "
                                            "product link.",
                "hold_for_unit_examination": "Wait for the unit examination and log why: an injury treated by a "
                                             "doctor or a product link, but not both, and no pattern on file.",
                "report_before_deadline": "Report within the window: the serious-risk test is met (a doctor-treated "
                                          "injury with a product link, or a pattern) and working days remain.",
                "report_immediately": "Report today: the serious-risk test is met and the deadline is today or has "
                                      "passed.",
            }),
            "criteria_fit": score("How clearly does the incident meet a written test of a serious risk to users, "
                                  "where a product link means the unit is identified by serial number and photos "
                                  "with no misuse or third-party repair mentioned, or examination found a product "
                                  "fault?", [
                "Not at all: no injury beyond home treatment, and no product link.",
                "Weakly: an injury treated by a doctor or in hospital, or a product link, but not both.",
                "Substantially: a doctor- or hospital-treated injury with a product link.",
                "Fully: a doctor- or hospital-treated injury where examination found a product fault, or 3+ similar "
                "incidents on file for an identified unit with no misuse or repair mentioned.",
            ]),
            "clock_pressure": score("How close is the company to missing its reporting deadline?", [
                "No real pressure: more than half of the reporting window remains.",
                "Some: two or more working days remain, but no more than half the window.",
                "High: exactly one working day remains.",
                "Critical: the deadline is today or has passed (working days elapsed at or beyond the rule).",
            ]),
            "product_involvement_established": noul(
                "The company's product was involved in the incident as described.",
                yes="The unit is identified by serial number, photos or examination, and the account is consistent.",
                no="The model is unconfirmed, the unit is unavailable, or the account does not fit the product."),
            "wider_problem_indicated": noul("The incident points to a problem that could affect other units of the product."),
        }


# =========================================================================== schema 4
class RecallEffectiveness(Schema):
    name = "recall_effectiveness"
    description = "An open recall's progress report: close it, keep going, broaden outreach, change the remedy or reopen scope."
    surfaces = {"json": 0.5, "status_report": 0.25, "committee_minutes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        key, pname, model, hazard, comps, failure, misuse = _product(rng)
        units = int(round(lognormal_amount(rng, 60000, 0.8), -2))
        weeks = pick(rng, [6, 12, 20, 30, 45, 60])
        target = pick(rng, [60, 70, 80])
        corr = between(rng, 15, 85, integer=True)
        trend = pick_w(rng, {"rising": 3, "plateau": 4, "falling": 2})
        reg_pct = pick(rng, [8, 15, 30, 55, 80])
        remedy = pick(rng, ["repair_kit", "replacement", "refund"])
        remedy_fail = pick(rng, [0, 0, 0, 1, 2])
        new_unremedied = pick(rng, [0, 0, 1, 2, 4])
        attrition = None                     # share of units estimated to be out of use anyway
        outside_scope = pick(rng, [0, 0, 0, 0, 1])
        retail_cleared = pick(rng, [72, 85, 94, 99, 100])
        listings = pick(rng, [0, 2, 6, 14, 30])
        regulator = pick_w(rng, {"no_comment": 4, "asked_for_plan": 2, "warned_of_escalation": 1})
        if reg == "clear":
            if chance(rng, 0.5):   # finished
                weeks, corr, trend, remedy_fail, new_unremedied, outside_scope, retail_cleared, listings, regulator = \
                    pick(rng, [45, 60]), between(rng, target + 5, 95, integer=True), "plateau", 0, 0, 0, 100, 0, "no_comment"
            else:                  # failing
                corr, trend, remedy_fail, new_unremedied, regulator = between(rng, 10, 25, integer=True), "plateau", \
                    pick(rng, [2, 3]), pick(rng, [3, 5]), "warned_of_escalation"
                weeks = max(weeks, 20)
        elif reg == "borderline":
            corr = near(rng, target, 8, lo=1, hi=99, integer=True)
            weeks = pick(rng, [20, 30, 45])
            trend = "plateau"
        else:  # conflicting
            if chance(rng, 0.5):   # headline rate looks fine, remedy itself failing
                corr, remedy_fail = between(rng, target, 92, integer=True), pick(rng, [1, 2, 3])
            else:                  # headline rate poor, but the hazard is almost gone from the market
                corr, retail_cleared, listings, new_unremedied = between(rng, 25, 45, integer=True), 100, 0, 0
                reg_pct, attrition = pick(rng, [8, 15]), pick(rng, [40, 50, 60])
            if chance(rng, 0.3):
                outside_scope = 1
        channels = ["retailer notices"]
        if reg_pct >= 15:
            channels.insert(0, "direct letters and emails to registered owners")
        if chance(rng, 0.6):
            channels.append("press release")
        if chance(rng, 0.4):
            channels.append("paid social media")
        return {"regime": reg, "product": pname, "model": model, "hazard": hazard, "brand": _brand(rng, key),
                "failure": failure, "units_in_scope": units, "weeks_open": weeks, "target_pct": target,
                "correction_pct": corr, "trend": trend, "registered_pct": reg_pct, "remedy": remedy,
                "remedy_failures": remedy_fail, "incidents_unremedied": new_unremedied,
                "incidents_outside_scope": outside_scope, "retail_stock_cleared_pct": retail_cleared,
                "est_out_of_use_pct": attrition if attrition is not None else maybe_attrition(rng),
                "resale_listings": listings, "regulator": regulator, "channels": channels,
                "spend_to_date": round(units * rng.uniform(4, 30), -3), "currency": pick(rng, ["USD", "EUR", "GBP"]),
                "recall_no": ident(rng, "RCL", 4, "-"), "owner": person(rng), "date": rand_date(rng)}

    REMEDY = {"repair_kit": "free repair kit fitted by the owner", "replacement": "free replacement unit",
              "refund": "full refund on return of the unit"}
    REGULATOR = {"no_comment": None, "asked_for_plan": "regulator has asked for an improvement plan by month end",
                 "warned_of_escalation": "regulator has said it may issue its own public warning if the rate "
                                         "does not improve"}

    TREND = {"rising": "still rising", "plateau": "flat for several weeks", "falling": "falling"}

    def render(self, f, rng, surface):
        corrected = int(f["units_in_scope"] * f["correction_pct"] / 100)
        spend = _money(f["spend_to_date"], rng, f["currency"])
        att = (f"an estimated {f['est_out_of_use_pct']}% of units in scope are already out of use (warranty-curve "
               f"estimate)" if f["est_out_of_use_pct"] else None)
        issues = []
        if f["remedy_failures"]:
            issues.append(f"{_plural(f['remedy_failures'], 'incident')} on units that had already received the remedy")
        if f["incidents_unremedied"]:
            issues.append(f"{_plural(f['incidents_unremedied'], 'incident')} on units not yet remedied")
        if f["incidents_outside_scope"]:
            issues.append(f"{_plural(f['incidents_outside_scope'], 'incident')} on a unit from outside the recalled lots")
        if surface == "json":
            return dumps(prune({
                "recall": {"number": f["recall_no"], "product": f"{f['brand']} {f['product']} {f['model']}",
                           "failure": f["failure"], "remedy": self.REMEDY[f["remedy"]], "weeks_open": f["weeks_open"]},
                "progress": {"units_in_scope": f["units_in_scope"], "units_corrected": corrected,
                             "correction_rate_pct": f["correction_pct"], "plan_target_pct": f["target_pct"],
                             "weekly_trend": f["trend"], "retail_stock_cleared_pct": f["retail_stock_cleared_pct"],
                             "resale_listings_found": f["resale_listings"], "attrition_note": att},
                "outreach": {"owner_registration_pct": f["registered_pct"], "channels": f["channels"]},
                "incidents_since_launch": issues or ["none"],
                "regulator": self.REGULATOR[f["regulator"]],
                "cost_to_date": spend,
            }))
        if surface == "status_report":
            rows = [("Recall", f["recall_no"]), ("Product", f"{f['brand']} {f['product']} {f['model']}"),
                    ("Remedy", self.REMEDY[f["remedy"]]), ("Weeks open", f["weeks_open"]),
                    ("Corrected", f"{_thousands(corrected)} of {_thousands(f['units_in_scope'])} "
                                  f"({f['correction_pct']}%; plan {f['target_pct']}%)"),
                    ("Weekly rate", self.TREND[f["trend"]]), ("Retail stock cleared", f"{f['retail_stock_cleared_pct']}%"),
                    ("Resale listings found", f["resale_listings"]),
                    ("Registered owners", f"{f['registered_pct']}%"), ("Channels", ", ".join(f["channels"])),
                    ("Attrition", att), ("Cost to date", spend)]
            out = "RECALL STATUS REPORT\n" + kv_lines(rows, rng, style=pick(rng, ["colon", "aligned"]))
            out += "\n\nIncidents since launch:\n" + (bullets(issues, rng) if issues else "- none")
            if self.REGULATOR[f["regulator"]]:
                out += f"\n\nRegulator: {self.REGULATOR[f['regulator']]}."
            return out
        s = [f"{f['owner']} presented recall {f['recall_no']} ({f['model']} {f['product']}, week {f['weeks_open']}).",
             f"{f['correction_pct']}% of the {_thousands(f['units_in_scope'])} units are corrected against a plan of "
             f"{f['target_pct']}%; the weekly rate is {self.TREND[f['trend']]}.",
             f"Remedy offered: {self.REMEDY[f['remedy']]}.",
             f"Retailers report {f['retail_stock_cleared_pct']}% of shelf stock removed; "
             f"{_plural(f['resale_listings'], 'resale listing')} found online.",
             ("Only " if f["registered_pct"] < 30 else "") + f"{f['registered_pct']}% of owners are registered; "
             f"outreach so far: {', '.join(f['channels'])}."]
        if att:
            s.append(_cap(att) + ".")
        s.append(("Incidents since launch: " + "; ".join(issues) + ".") if issues else "No incidents since launch.")
        if self.REGULATOR[f["regulator"]]:
            s.append(self.REGULATOR[f["regulator"]].capitalize() + ".")
        s.append(f"Spend to date {spend}.")
        return (f"PRODUCT SAFETY COMMITTEE -- minutes, {fmt_date(f['date'], rng)}\nItem 3: recall progress\n\n"
                + noisy(join_sentences(s), rng, 0.1))

    def questions(self):
        return {
            "recall_next_step": choice("What should the committee decide about the recall at this review?", {
                "broaden_outreach": "Add outreach channels: corrected units plus units estimated out of use are below "
                                    "the plan target and the weekly rate is flat or falling.",
                "close_recall": "Close: corrected plus out-of-use units reach the target, shelf stock is fully "
                                "cleared, no resale listings remain and no unremedied unit has had an incident.",
                "continue_current_plan": "Carry on and review next month: none of the other conditions holds (for "
                                         "example below target but still rising, or at target with stock, listings "
                                         "or incidents still open).",
                "reopen_scope": "Re-examine which units are covered: a unit from outside the recalled lots has had "
                                "an incident, and no remedied unit has failed.",
                "replace_the_remedy": "Change the remedy: a unit that had already received the remedy has had an "
                                      "incident.",
            }),
            "recall_effectiveness": score("How effective has the recall been so far at taking the hazard out of use?", [
                "Ineffective: most affected units are still in use, unremedied.",
                "Partly effective: progress is real but well short of plan.",
                "Largely effective: close to plan, with gaps.",
                "Effective: the hazard is essentially gone from use.",
            ]),
            "residual_hazard": score("How much hazard remains among units in use today?", [
                "Negligible.",
                "Low: some unremedied units, with no recent incidents.",
                "Material: many unremedied units or recent incidents.",
                "High: incidents continue and the remedy or reach is failing.",
            ]),
            "biggest_outreach_gap": choice("Where is the largest remaining gap in reaching affected units?", {
                "no_significant_gap": "There is no significant gap left to close.",
                "registered_owners": "Registered owners who have been contacted but have not acted.",
                "resale_market": "Units being resold second-hand or through online marketplaces.",
                "retail_stock": "Units still on shop shelves or in retailer warehouses.",
                "unregistered_owners": "Owners who never registered and cannot be contacted directly.",
            }),
            "remedy_works": noul(
                "The remedy being offered removes the hazard from units that receive it.",
                yes="No incident has occurred on any unit after it received the remedy.",
                no="At least one unit that had already received the remedy has had an incident since."),
        }


# =========================================================================== schema 5
# key -> (component, plural for the letter, problem, an independent safeguard our design might have)
NOTICES = {
    "capacitor_leak": ("electrolytic capacitor", "electrolytic capacitors",
                       "electrolyte leakage in capacitors from a contaminated batch",
                       "a separate thermal fuse on the mains side"),
    "cell_separator": ("lithium-ion cell", "lithium-ion cells",
                       "metal particles found in the separator of cells from two production days",
                       "cell-level temperature cut-off in the battery management board"),
    "resin_fr": ("flame-retardant housing resin", "housing resin",
                 "flame-retardant additive under-dosed in several resin lots",
                 "a metal inner enclosure around the live and hot parts"),
    "fastener_embrittlement": ("hardened steel fastener", "hardened steel fasteners",
                               "hydrogen embrittlement after a plating-line fault",
                               "a second locking pin that carries the load if the fastener breaks"),
    "cord_insulation": ("mains power cord", "mains power cords",
                        "insulation thinner than specification on part of one reel batch",
                        "an earth-leakage cut-out in the plug"),
    "magnet_adhesive": ("magnet adhesive", "magnet adhesive",
                        "adhesive cure failure in one batch, magnets may detach",
                        "a moulded pocket that holds each magnet even without adhesive"),
}
NOTICE_SUPPLIER = {"capacitor_leak": "Electronics", "cell_separator": "Energy", "resin_fr": "Polymers",
                   "fastener_embrittlement": "Fasteners", "cord_insulation": "Cable", "magnet_adhesive": "Adhesives"}
NOTICE_PRODUCTS = {"capacitor_leak": ["air_fryer", "ebike_charger", "desk_frame", "baby_monitor"],
                   "cell_separator": ["tool_battery"],
                   "resin_fr": ["air_fryer", "ebike_charger", "oil_radiator"],
                   "fastener_embrittlement": ["step_ladder", "desk_frame"],
                   "cord_insulation": ["oil_radiator", "pressure_washer", "kettle"],
                   "magnet_adhesive": ["magnet_tiles"]}


class SupplierDefectNotice(Schema):
    name = "supplier_defect_notice"
    description = "A component supplier's defect notice: whether it forces action on our own shipped product."
    surfaces = {"json": 0.45, "supplier_letter": 0.3, "quality_ticket": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        nkey = pick(rng, list(NOTICES))
        comp, comp_pl, problem, guard = NOTICES[nkey]
        pkey = pick(rng, NOTICE_PRODUCTS[nkey])
        pname, prefix, hazard = PRODUCTS[pkey][:3]
        severity_claim = pick_w(rng, {"quality_notice": 3, "safety_notice": 2})
        lot_info = pick_w(rng, {"lot_list_complete": 3, "lot_list_partial": 3, "date_range_only": 2})
        shipped = int(round(lognormal_amount(rng, 20000, 1.0), -2))
        warehouse = int(round(lognormal_amount(rng, 4000, 0.8), -2))
        protection = pick_w(rng, {"independent_protection": 3, "partial": 3, "none": 2})
        field = pick_w(rng, {"no_related_returns": 4, "some_related_returns": 2, "not_checked": 2})
        related = pick(rng, [1, 2, 3, 5]) if field == "some_related_returns" else 0
        advice = pick_w(rng, {"none": 2, "inspect_stock": 3, "replace_in_field": 1})
        tested = pick_w(rng, {"not_tested": 3, "tested_clean": 2, "tested_failures": 1})
        if reg == "clear":
            if chance(rng, 0.5):
                severity_claim, lot_info, protection, field, related, tested = "quality_notice", "lot_list_complete", \
                    "independent_protection", "no_related_returns", 0, "tested_clean"
                shipped = pick(rng, [0, 300, 1200])
            else:
                severity_claim, protection, field, related, tested, advice = "safety_notice", "none", \
                    "some_related_returns", pick(rng, [3, 5]), "tested_failures", "replace_in_field"
        elif reg == "borderline":
            protection = "partial"
            lot_info = pick(rng, ["lot_list_partial", "date_range_only"])
            tested = pick(rng, ["not_tested", "tested_clean"])
        else:  # conflicting: the supplier plays it down but our data disagree, or the reverse
            if chance(rng, 0.5):
                severity_claim, advice, field, related = "quality_notice", "none", "some_related_returns", pick(rng, [2, 3])
            else:
                severity_claim, advice, protection, tested, field, related = "safety_notice", "replace_in_field", \
                    "independent_protection", "tested_clean", "no_related_returns", 0
        return {"regime": reg, "notice_key": nkey, "component": comp, "problem": problem, "product": pname,
                "model": f"{prefix}-{rng.randint(100, 999)}", "hazard": hazard, "brand": _brand(rng, pkey),
                "supplier": _name(rng, [NOTICE_SUPPLIER[nkey]]),
                "notice_class": severity_claim, "lot_info": lot_info, "units_shipped_suspect": shipped,
                "units_in_warehouse": warehouse, "design_protection": protection, "field_data": field,
                "related_returns": related, "supplier_advice": advice, "retained_test": tested,
                "notice_ref": ident(rng, "SN", 5, "-"), "received": rand_date(rng),
                "days_since_notice": pick(rng, [1, 3, 6, 12]), "engineer": person(rng),
                "protection_detail": guard, "component_plural": comp_pl}

    ADVICE = {"none": "No action required by customers; for information only.",
              "inspect_stock": "We recommend customers inspect and quarantine stock from the listed lots.",
              "replace_in_field": "We recommend customers replace the part in products already delivered."}
    LOTINFO = {"lot_list_complete": "a full list of affected lot numbers",
               "lot_list_partial": "a partial lot list (further lots 'under review')",
               "date_range_only": "a production date range only, no lot numbers"}

    def _protection_text(self, f):
        return {"independent_protection": f"our design has an independent safeguard: {f['protection_detail']}",
                "partial": f"our design has {f['protection_detail']}, but it covers only part of the failure described",
                "none": "nothing in our design would contain the failure if the part behaves as described"
                }[f["design_protection"]]

    def _field_text(self, f):
        return {"no_related_returns": "no returns matching the failure described",
                "some_related_returns": f"{_plural(f['related_returns'], 'return')} that may match the failure described",
                "not_checked": "returns data not yet checked"}[f["field_data"]]

    def _test_text(self, f):
        return {"not_tested": "retained samples not yet tested", "tested_clean": "retained samples tested, no defects",
                "tested_failures": "retained samples tested, defects found"}[f["retained_test"]]

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "notice": {"ref": f["notice_ref"], "from": f["supplier"], "received": fmt_date(f["received"], style="iso"),
                           "classification": f["notice_class"].replace("_", " "), "component": f["component"],
                           "problem": f["problem"], "affected_parts_identified_by": self.LOTINFO[f["lot_info"]],
                           "supplier_recommendation": self.ADVICE[f["supplier_advice"]]},
                "our_exposure": {"product": f"{f['brand']} {f['product']} {f['model']}",
                                 "units_shipped_possibly_affected": f["units_shipped_suspect"],
                                 "units_in_our_warehouse": f["units_in_warehouse"],
                                 "design_protection": self._protection_text(f)},
                "our_checks": {"field_returns": self._field_text(f), "retained_samples": self._test_text(f),
                               "days_since_notice": f["days_since_notice"]},
            }))
        if surface == "supplier_letter":
            letter = [f"{f['supplier'].upper()}", f"{f['notice_class'].replace('_', ' ').upper()} {f['notice_ref']}",
                      f"Date: {fmt_date(f['received'], rng)}", "", "Dear Customer,", "",
                      noisy(join_sentences([
                          f"We are writing to inform you of {f['problem']}, affecting {f['component_plural']} we "
                          f"supplied to you.",
                          f"The affected parts are identified by {self.LOTINFO[f['lot_info']]} (attached).",
                          self.ADVICE[f["supplier_advice"]],
                          "We apologise for any inconvenience."]), rng, 0.05),
                      "", "--- internal annotations (product safety) ---",
                      bullets([f"Used in: {f['brand']} {f['product']} {f['model']}",
                               f"Possibly affected units shipped: {_thousands(f['units_shipped_suspect'])}; "
                               f"in our warehouse: {_thousands(f['units_in_warehouse'])}",
                               _cap(self._protection_text(f)),
                               f"Field data: {self._field_text(f)}", f"Lab: {self._test_text(f)}",
                               f"{_plural(f['days_since_notice'], 'day')} since the notice arrived"], rng)]
            return "\n".join(letter)
        rows = [("Ticket", ident(rng, "QT", 5, "-")), ("Opened by", f["engineer"]),
                ("Trigger", f"supplier {f['notice_class'].replace('_', ' ')} {f['notice_ref']} from {f['supplier']}"),
                ("Component", f"{f['component']}: {f['problem']}"),
                ("Lot information", self.LOTINFO[f["lot_info"]]),
                ("Supplier recommends", self.ADVICE[f["supplier_advice"]]),
                ("Affected product", f"{f['product']} {f['model']}"),
                ("Shipped units possibly affected", _thousands(f["units_shipped_suspect"])),
                ("Warehouse units", _thousands(f["units_in_warehouse"])),
                ("Design protection", self._protection_text(f)), ("Field data", self._field_text(f)),
                ("Retained samples", self._test_text(f)), ("Age of notice", f"{f['days_since_notice']} d")]
        return "QUALITY TICKET -- supplier notification\n" + kv_lines(rows, rng, style=pick(rng, ["colon", "aligned"]))

    def questions(self):
        return {
            "our_response": choice("What should the company do about its own product in response to the notice?", {
                "field_fix_shipped_units": "Offer a fix for shipped units on top of holding warehouse stock.",
                "quarantine_stock_only": "Hold unshipped stock containing the part; take no action on shipped units.",
                "recall_shipped_units": "Recall shipped units that may contain the part.",
                "record_only": "Record the notice; our product does not need any action.",
                "test_samples_before_deciding": "Hold stock and test retained samples before deciding anything about "
                                                "shipped units.",
            }),
            "field_exposure": score("How exposed are units already in customers' hands to the supplier's defect?", [
                "Not at all: no possibly affected units have shipped.",
                "Slightly: fewer than 1,000 shipped, or an independent safeguard in our design contains the failure.",
                "Materially: 1,000 or more shipped without an independent safeguard, short of the Heavily case.",
                "Heavily: 1,000 or more shipped, nothing in the design contains the failure, and there are matching "
                "returns or defective retained samples.",
            ]),
            "design_containment": score("If the part fails as the supplier describes, how well does our design keep "
                                        "users safe?", [
                "Not at all: nothing in the design would contain the failure.",
                "Partly: the design has a safeguard, but it covers only part of the failure.",
                "Mostly: an independent protective device (a fuse, a cut-off or an earth-leakage trip) stops the "
                "failure once it starts.",
                "Fully: an independent physical feature (a metal enclosure, a second locking pin or a moulded "
                "pocket) makes the failure harmless.",
            ]),
            "quarantine_now": noul(
                "Unshipped stock containing the suspect part must be quarantined now.",
                yes="Our design has no independent safeguard for this failure (none, or one covering only part of "
                    "it), or retained samples showed defects.",
                no="An independent safeguard in our design contains the failure, and retained samples showed no "
                   "defects or have not been tested."),
            "notice_detail_sufficient": noul("The supplier's notice gives enough detail to identify every affected unit."),
        }


WORKFLOW = Workflow(
    name="product_recall_decisions",
    title="Product recall decisions",
    description="What a manufacturer should do about a safety signal on product already sold: monitor, fix, stop "
                "shipping or recall, and what to tell the regulator.",
    domain="product safety / quality",
    schemas=[FieldFailureSignal(), LotScopeDetermination(), IncidentReportability(), RecallEffectiveness(),
             SupplierDefectNotice()],
    notes="Manufacturer-side safety decisions on fictional products and brands. No conversation with a consumer is "
          "drafted or triaged; incidents arrive as records, not threads.",
)
