"""Water quality advisories: public-health decisions on drinking water and recreational water, made against written action levels.

A (fictional) state Office of Drinking Water and county Environmental Health Divisions decide what the public must
be told and done after a sample result: repeat sampling and boil-water notices after a coliform-positive sample,
compliance with a running-average or acute chemical standard, posting or closing a bathing beach on bacteria
results, posting a lake for a cyanobacterial bloom, and lifting a boil-water notice after a main break. Every state
quotes the rule that governs the decision (an invented Drinking Water Rule "DWR-14" and Recreational Water Code
"RWC-6"; their numbers follow the usual shape of such rules but belong to this fictional code), so the answer
follows from the written action levels plus the results, which are often borderline, late, or contradicted by a
field observation.

This is public-health advisory work on water systems and bathing waters. It is not a facility's discharge-permit or
air-permit compliance reporting (see environmental_compliance_reports), and no conversation with a member of the
public is drafted or triaged. Utilities, towns, beaches, lakes and people are fictional.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

from .base import (
    ORG_A,
    Schema,
    Workflow,
    between,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    fmt_ts,
    ident,
    join_sentences,
    kv_lines,
    maybe,
    near,
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    rand_ts,
    regime,
    score,
    table,
)

TOWNS = ["Alder Springs", "Bramble Creek", "Cotter's Ford", "Delmar Flats", "Easton Mills", "Fallow Ridge", "Glenhaven",
         "Harrow Point", "Ivy Bend", "Juniper Hollow", "Larkspur", "Millbrook", "Northcote", "Oakvale", "Pinebluff",
         "Quinton", "Rookwood", "Thornbury", "Underhill", "Vesper Lake", "Willowmere", "Yarrowby", "Ashcombe",
         "Bellhaven", "Corriston", "Dunwick"]
COUNTY = ["Marlow", "Kessler", "Tamberlane", "Hollins", "Brackett", "Ferris", "Gilmore", "Ostley", "Prentiss", "Ravenna"]
# system name style -> plausible populations served
SYSTEM_STYLE = {"{t} Water District": [2600, 7400, 9500, 18000], "City of {t} Water Utility": [9500, 24000, 52000, 61000],
                "{t} Mutual Water Company": [450, 900, 1200, 2600], "{o} Heights Water Association": [180, 220, 450, 900],
                "{t} Public Utilities": [3800, 7400, 18000, 24000], "{t} Rural Water Cooperative": [450, 1200, 2600, 3800],
                "{o} Park Mobile Home Community water system": [120, 180, 220, 350],
                "{t} Regional Water Authority": [24000, 52000, 61000, 88000]}


def _system_pop(rng):
    style = pick(rng, list(SYSTEM_STYLE))
    return style.format(t=pick(rng, TOWNS), o=pick(rng, ORG_A)), pick(rng, SYSTEM_STYLE[style])


def _system(rng):
    return _system_pop(rng)[0]


def _county(rng):
    return f"{pick(rng, COUNTY)} County"


def _ds(rng):
    return pick(rng, ["iso", "iso", "us", "long"])


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _gm(vals):
    vals = [max(1.0, v) for v in vals]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# =========================================================================== schema 1
TAPS = {"station": "dedicated sampling station", "kitchen": "kitchen tap (aerator removed)",
        "hose_bib": "outside hose bib", "swivel": "swivel faucet in a restaurant prep sink",
        "utility_sink": "utility sink in a school boiler room"}
REPEAT_LOC = ["original tap", "upstream (3 connections)", "downstream (4 connections)", "upstream (2 connections)",
              "downstream (5 connections)"]
STREETS = ["Elm St", "Ridge Rd", "Mill Ln", "3rd Ave", "Orchard Way", "Station Rd", "Kingfisher Ct", "Water St",
           "Hilltop Dr", "Cedar Loop", "Quarry Rd", "Lakeview Ter", "School St", "Pump House Rd"]


class ColiformPositiveFollowup(Schema):
    name = "coliform_positive_followup"
    description = "A coliform-positive routine sample in a distribution system and its repeat samples: what to tell customers and what notice tier applies."
    surfaces = {"json": 0.5, "lab_results_table": 0.25, "operator_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        routine_ec = chance(rng, 0.3)
        tap = pick_w(rng, {"station": 3, "kitchen": 3, "hose_bib": 2, "swivel": 1, "utility_sink": 1})
        resid = round(near(rng, 0.35, 0.3, lo=0.02, hi=1.4), 2)
        pressure = pick_w(rng, {"none": 6, "main_break": 2, "low_pressure": 2})
        rep_hours = rng.randint(4, 30)
        repeats = [pick_w(rng, {"neg": 6, "tc": 2, "ec": 1, "pending": 1}) for _ in range(3)]
        holding = chance(rng, 0.1)
        overdue = chance(rng, 0.15)
        other_pos = pick(rng, [0, 0, 0, 1, 2])
        if reg == "clear":
            if chance(rng, 0.5):
                routine_ec, repeats, rep_hours, tap, resid, pressure = False, ["neg"] * 3, rng.randint(4, 20), "station", round(between(rng, 0.6, 1.2), 2), "none"
                holding = False
            else:
                repeats, pressure, resid = ["ec", pick(rng, ["tc", "ec", "neg"]), "tc"], "main_break", round(between(rng, 0.02, 0.15), 2)
        elif reg == "borderline":
            rep_hours = near(rng, 24, 6, lo=3, integer=True)
            resid = round(near(rng, 0.2, 0.1, lo=0.02), 2)
            if chance(rng, 0.2):
                repeats, tap = ["tc", "neg", "neg"], pick(rng, ["hose_bib", "swivel", "utility_sink", "kitchen"])
        twist = None
        if reg == "conflicting":
            twist = pick(rng, ["bad_tap", "bad_tap", "pending_ec", "late_repeats", "good_residual_ec"])
            if twist == "bad_tap":
                tap, repeats = pick(rng, ["hose_bib", "swivel", "utility_sink"]), ["tc", "neg", "neg"]
                resid, routine_ec, pressure = round(between(rng, 0.5, 1.1), 2), False, "none"
                rep_hours = rng.randint(4, 22)
            elif twist == "pending_ec":
                routine_ec, repeats = True, ["pending", "pending", "neg"]
                rep_hours = rng.randint(4, 18)
            elif twist == "late_repeats":
                rep_hours = rng.randint(27, 52)
                repeats = [pick(rng, ["neg", "tc"]) for _ in range(3)]
            else:
                repeats, resid, tap = [pick(rng, ["ec", "tc"]), "neg", "neg"], round(between(rng, 0.6, 1.0), 2), "station"
        if twist != "bad_tap":
            rng.shuffle(repeats)
        locs = ["original tap", pick(rng, REPEAT_LOC[1::2]), pick(rng, REPEAT_LOC[2::2])]
        t0 = rand_ts(rng, hours=(7, 15))
        sysname, pop = _system_pop(rng)
        return {"regime": reg, "routine_ecoli": routine_ec, "tap": tap, "residual_mg_l": resid, "pressure_event": pressure,
                "repeat_hours_after_notice": rep_hours, "repeats": repeats, "repeat_locations": locs,
                "holding_time_exceeded": holding, "assessment_overdue": overdue, "other_tc_pos_month": other_pos,
                "system": sysname, "population": pop,
                "pwsid": ident(rng, "PWS", 7, ""), "street": pick(rng, STREETS), "operator": person(rng),
                "notified": t0, "operator_view": pick(rng, ["calm", "calm", "dismissive", "worried"])}

    @staticmethod
    def _ecoli_violation(f):
        reps = f["repeats"]
        late = f["repeat_hours_after_notice"] > 24
        if f["routine_ecoli"]:
            if any(r in ("tc", "ec") for r in reps):
                return True
            if late:
                return True          # failure to take all repeats in time after an E. coli-positive routine
            return False
        return any(r == "ec" for r in reps)

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        viol = self._ecoli_violation(f)
        reps = f["repeats"]
        late = f["repeat_hours_after_notice"] > 24
        orig_only = reps[0] == "tc" and all(r == "neg" for r in reps[1:])
        risky = f["residual_mg_l"] < 0.2 or f["pressure_event"] != "none"
        if viol:
            adv = "mandatory_boil_water_notice"
        elif f["routine_ecoli"] or (any(r == "tc" for r in reps) and risky):
            adv = "precautionary_boil_advisory"
        elif orig_only and f["tap"] in ("hose_bib", "swivel", "utility_sink"):
            adv = "investigate_sample_site_first"
        elif any(r in ("tc", "pending") for r in reps) or late or f["holding_time_exceeded"]:
            adv = "additional_repeat_sampling"
        else:
            adv = "close_out_no_further_action"
        if viol:
            tier = "tier_1_within_24_hours"
        elif f["assessment_overdue"]:
            tier = "tier_2_within_30_days"
        elif late or f["holding_time_exceeded"]:
            tier = "tier_3_annual_report"
        else:
            tier = "no_notice_required"
        c = 0
        if f["routine_ecoli"] or "ec" in reps:
            c += 2
        if sum(r in ("tc", "ec") for r in reps) >= 2:
            c += 1
        if f["residual_mg_l"] < 0.2:
            c += 1
        if f["pressure_event"] != "none":
            c += 1
        if all(r == "neg" for r in reps):
            c -= 1
        if orig_only and f["residual_mg_l"] >= 0.4:
            c -= 1
        v = {"station": 3, "kitchen": 2, "utility_sink": 1, "swivel": 1, "hose_bib": 1}[f["tap"]]
        if late:
            v -= 1
        if f["holding_time_exceeded"]:
            v -= 1
        return {"advisory_decision": adv, "public_notice_tier": tier, "contamination_likelihood": max(0, min(3, c)),
                "sampling_validity": max(0, v), "ecoli_violation_established": "yes" if viol else "no"}

    def _rules(self, rng):
        return [
            pick(rng, ["DWR-14 §4.2 After a total-coliform-positive routine sample, take at least three repeats within "
                       "24 hours of notification: the original tap, one within five connections upstream and one "
                       "within five downstream.",
                       "DWR-14 §4.2 Three repeat samples (original tap, upstream and downstream within five connections) "
                       "are due within 24 h of the lab's notice of a TC-positive routine sample."]),
            pick(rng, ["DWR-14 §4.3 E. coli violation: an E. coli-positive repeat after a TC-positive routine; a "
                       "TC-positive repeat after an E. coli-positive routine; or failure to take all repeats on time "
                       "after an E. coli-positive routine.",
                       "DWR-14 §4.3 The E. coli MCL is violated by EC+ repeat following TC+ routine, TC+ repeat following "
                       "EC+ routine, or missing/late repeats following an EC+ routine."]),
            "DWR-14 §4.6 Notice tiers: E. coli violation, Tier 1 within 24 h with boil-water advice; an overdue "
            "assessment or corrective action, Tier 2 within 30 days; late repeats or samples past holding time, Tier 3 "
            "in the annual report.",
            "DWR-14 §4.8 A positive confined to the original tap with clean upstream and downstream repeats may reflect "
            "the tap or premises plumbing; the state may require the site to be investigated before further action.",
        ]

    def _res(self, r):
        return {"neg": "TC absent", "tc": "TC present / E. coli absent", "ec": "TC present / E. coli PRESENT",
                "pending": "pending"}[r]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        routine = "TC present / E. coli " + ("PRESENT" if f["routine_ecoli"] else "absent")
        rep_time = f["notified"] + timedelta(hours=f["repeat_hours_after_notice"])
        press = {"none": None, "main_break": "main break on the same pressure zone two days before the routine sample",
                 "low_pressure": "low-pressure complaints from the street the week of sampling"}[f["pressure_event"]]
        extra = []
        if f["holding_time_exceeded"]:
            extra.append("lab notes one repeat bottle arrived 33 h after collection (holding time 30 h)")
        if f["assessment_overdue"]:
            extra.append("a Level 1 assessment from an earlier coliform trigger is 3 weeks overdue")
        if f["other_tc_pos_month"]:
            extra.append(f"{f['other_tc_pos_month']} other TC-positive routine sample(s) this month at other sites")
        view = {"calm": None, "dismissive": "Operator thinks the hit is a dirty tap and wants to close it out.",
                "worried": "Operator is worried and asks whether to warn customers now."}[f["operator_view"]]
        if surface == "json":
            return dumps(prune({
                "system": {"name": f["system"], "pwsid": f["pwsid"], "population_served": f["population"]},
                "routine_sample": {"site": f"{f['street']}, {TAPS[f['tap']]}", "result": routine,
                                   "lab_notified_operator": fmt_ts(f["notified"], style="iso"),
                                   "free_chlorine_mg_l": f["residual_mg_l"]},
                "repeat_samples": {"collected": fmt_ts(rep_time, style="iso"),
                                   "results": [{"location": l, "result": self._res(r)}
                                               for l, r in zip(f["repeat_locations"], f["repeats"])]},
                "context": {"pressure_events": press, "notes": extra or None, "operator_comment": view},
                "rule_excerpts": self._rules(rng),
            }))
        if surface == "lab_results_table":
            rows = [["routine", f"{f['street']} ({TAPS[f['tap']]})", fmt_ts(f["notified"], style="log"), routine]]
            rows += [[f"repeat {i + 1}", l, fmt_ts(rep_time, style="log"), self._res(r)]
                     for i, (l, r) in enumerate(zip(f["repeat_locations"], f["repeats"]))]
            head = kv_lines([("System", f"{f['system']} ({f['pwsid']})"), ("Population", f"{f['population']:,}"),
                             ("Chlorine residual at routine site", f"{f['residual_mg_l']} mg/L free")], rng)
            notes = [press] + extra if press else extra
            return (f"COLIFORM RESULTS -- follow-up review\n{head}\n\n"
                    + table(["sample", "location", "time (collected / lab notice)", "result"], rows, rng)
                    + ("\n\nNotes:\n" + bullets(notes, rng) if notes else "") + (f"\n{view}" if view else "")
                    + "\n\n" + "\n".join(self._rules(rng)))
        para = [f"Lab called us {fmt_ts(f['notified'], rng)} with a {routine} result on the routine sample at {f['street']} "
                f"({TAPS[f['tap']]}); free chlorine there was {f['residual_mg_l']} mg/L.",
                f"Repeats were collected {fmt_ts(rep_time, rng)}: "
                + "; ".join(f"{l}: {self._res(r)}" for l, r in zip(f["repeat_locations"], f["repeats"])) + "."]
        if press:
            para.append(_cap(press) + ".")
        para += [_cap(x) + "." for x in extra]
        if view:
            para.append(pick(rng, ["Honestly I think it's the tap.", "Should we be telling people to boil?"])
                        if f["operator_view"] != "calm" else "")
        sanit = person(rng)
        return "\n".join([f"Subject: {f['pwsid']} coliform follow-up -- {f['system']}", f"From: {f['operator']} (operator)",
                          f"To: {sanit}, Office of Drinking Water", "", f"{sanit.split()[0]},", "",
                          noisy(join_sentences(para), rng, 0.12), "", f"{f['operator'].split()[0]}", "", "---",
                          "Rule references:", bullets(self._rules(rng), rng)])

    def questions(self):
        return {
            "advisory_decision": choice("What should the water system do now about this coliform result?", {
                "close_out_no_further_action": "Close the event out; nothing more is required.",
                "additional_repeat_sampling": "Take further samples before deciding anything else.",
                "precautionary_boil_advisory": "Issue a precautionary boil-water advisory while the picture firms up.",
                "mandatory_boil_water_notice": "Issue a mandatory boil-water notice now.",
                "investigate_sample_site_first": "Inspect the sampling tap and premises plumbing before acting system-wide.",
            }),
            "public_notice_tier": choice("Which public-notice tier, if any, does this event require?", {
                "no_notice_required": "No public notice is required.",
                "tier_3_annual_report": "Report it in the annual water quality report.",
                "tier_2_within_30_days": "Notify customers within 30 days.",
                "tier_1_within_24_hours": "Notify customers within 24 hours.",
            }),
            "contamination_likelihood": score("How likely is it that contaminated water is actually reaching customers?", [
                "Remote: the evidence points to a sampling artefact.",
                "Possible: a positive without supporting signs in the system.",
                "Probable: positives plus a weakness in the system, such as low chlorine or a pressure loss.",
                "Near certain: E. coli confirmed with supporting signs in the system.",
            ]),
            "sampling_validity": score("How much weight can the sample results bear?", [
                "Little: poor sampling site and procedural failures.",
                "Limited: a poor site or a procedural failure.",
                "Reasonable: an ordinary tap, procedure followed.",
                "Strong: a dedicated sampling station, procedure followed on time.",
            ]),
            "ecoli_violation_established": noul(
                "The results establish an E. coli violation under the rule.",
                yes="One of the rule's E. coli violation conditions is met on the results and timing in the file.",
                no="None of those conditions is met yet: the results are negative, coliform-only or still pending."),
        }


# =========================================================================== schema 2
# contaminant -> (MCL, unit, kind, label)
CHEMS = {
    "TTHM": (80.0, "ug/L", "chronic", "total trihalomethanes (locational running annual average)"),
    "HAA5": (60.0, "ug/L", "chronic", "haloacetic acids (locational running annual average)"),
    "arsenic": (10.0, "ug/L", "chronic", "arsenic (running annual average at the entry point)"),
    "PFOS": (4.0, "ng/L", "chronic", "PFOS (running annual average at the entry point)"),
    "uranium": (30.0, "ug/L", "chronic", "uranium (running annual average)"),
    "nitrate": (10.0, "mg/L as N", "acute", "nitrate (single confirmed result)"),
    "nitrite": (1.0, "mg/L as N", "acute", "nitrite (single confirmed result)"),
}
SOURCES = ["Well 2", "Well 4", "the river intake", "the east wellfield", "the purchased-water connection", "Well 7A",
           "the spring box", "the reservoir outlet"]


class ChemicalRunningAverage(Schema):
    name = "chemical_running_average"
    description = "A chemical monitoring result judged against a running-average or acute standard: what response the rule requires."
    surfaces = {"json": 0.5, "compliance_worksheet": 0.25, "sanitarian_notes": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        chem = pick_w(rng, {"TTHM": 2, "HAA5": 1, "arsenic": 2, "PFOS": 1, "uranium": 1, "nitrate": 4, "nitrite": 1})
        mcl, unit, kind, _ = CHEMS[chem]
        target = near(rng, 1.0, 0.35, lo=0.2)
        qualifier = pick_w(rng, {None: 6, "J (estimated; matrix interference)": 1, "received above 10 C": 1,
                                 "duplicate RPD 28%": 1})
        pending = False
        confirm = None
        conf_hours = None
        if kind == "chronic":
            prev = [round(mcl * target * between(rng, 0.75, 1.25), 1) for _ in range(3)]
            cur = round(mcl * target * between(rng, 0.7, 1.4), 1)
            if reg == "clear":
                t = pick(rng, [0.45, 1.6])
                prev = [round(mcl * t * between(rng, 0.85, 1.15), 1) for _ in range(3)]
                cur = round(mcl * t * between(rng, 0.85, 1.15), 1)
            elif reg == "conflicting":
                twist = pick(rng, ["one_spike", "pending", "qualifier", "messy"])
                if twist == "one_spike":
                    prev = [round(mcl * between(rng, 0.5, 0.8), 1) for _ in range(3)]
                    cur = round(mcl * between(rng, 1.4, 2.6), 1)
                elif twist == "pending":
                    pending = True
                elif twist == "qualifier":
                    qualifier = pick(rng, ["received above 10 C", "duplicate RPD 28%"])
                else:
                    qualifier, pending = pick(rng, ["received above 10 C", "duplicate RPD 28%"]), True
            if chance(rng, 0.12):
                pending = True
            return {"regime": reg, "chem": chem, "mcl": mcl, "unit": unit, "kind": kind, "prev_quarters": prev,
                    "current": None if pending else cur, "current_pending": pending, "original": None,
                    "confirmation": None, "confirm_hours": None, "qualifier": qualifier, **self._ctx(rng, chem)}
        orig = round(mcl * near(rng, 1.0, 0.45, lo=0.2), 2)
        status = pick_w(rng, {"taken": 6, "pending": 2, "not_taken": 1})
        if reg == "clear":
            orig = round(mcl * pick(rng, [between(rng, 0.2, 0.45), between(rng, 1.6, 2.4)]), 2)
            status = "taken"
        elif reg == "conflicting":
            twist = pick(rng, ["confirm_low", "late_confirm", "pending_high"])
            if twist == "confirm_low":
                orig, status = round(mcl * between(rng, 1.1, 1.6), 2), "taken"
            elif twist == "late_confirm":
                orig, status = round(mcl * between(rng, 1.05, 1.4), 2), "not_taken"
            else:
                orig, status = round(mcl * between(rng, 1.2, 2.2), 2), "pending"
        if status == "taken":
            confirm = round(orig * between(rng, 0.55, 1.2), 2) if orig > mcl else round(orig * between(rng, 0.8, 1.2), 2)
            if reg == "conflicting" and confirm >= orig * 0.9 and chance(rng, 0.5):
                confirm = round(orig * between(rng, 0.4, 0.7), 2)
            conf_hours = rng.randint(6, 22)
        elif status == "not_taken":
            conf_hours = rng.randint(30, 70)
        if orig <= mcl:
            status, confirm, conf_hours = "not_required", None, None
        ctx = self._ctx(rng, chem)
        if reg == "conflicting" and chance(rng, 0.5):
            ctx["change_note"] = "Operator says a bypass valve was left open during one sampling event"
            qualifier = qualifier or "received above 10 C"
        return {"regime": reg, "chem": chem, "mcl": mcl, "unit": unit, "kind": kind, "prev_quarters": None,
                "current": None, "current_pending": False, "original": orig, "confirmation": confirm,
                "confirm_status": status, "confirm_hours": conf_hours, "qualifier": qualifier, **ctx}

    def _ctx(self, rng, chem):
        sysname, pop = _system_pop(rng)
        locs = (["DBP site 3 (far end of the north zone)", "DBP site 1 (storage tank outlet)",
                 "DBP site 2 (dead end on the east loop)", "DBP site 4 (highest water age)"] if chem in ("TTHM", "HAA5")
                else ["entry point EP-1", "entry point EP-2", "entry point EP-3 (blended)"])
        return {"system": sysname, "population": pop,
                "pwsid": ident(rng, "PWS", 7, ""), "location": pick(rng, locs),
                "source": pick(rng, SOURCES), "operator": person(rng), "sanitarian": person(rng),
                "change_note": pick(rng, [None, None, "Blending ratio changed in spring after {s} was taken offline",
                                          "Chlorine dose raised in summer to hold residuals",
                                          "A new well came online in the last quarter",
                                          "Operator says a bypass valve was left open during one sampling event"]),
                "date": rand_ts(rng).date()}

    @staticmethod
    def _compliance_value(f):
        """(value, established) for the compliance basis."""
        if f["kind"] == "chronic":
            prev = f["prev_quarters"]
            if f["current_pending"]:
                return sum(prev) / 4.0, False
            return (sum(prev) + f["current"]) / 4.0, True
        st = f["confirm_status"]
        if st == "taken":
            return (f["original"] + f["confirmation"]) / 2.0, True
        if st == "not_taken":
            return f["original"], True
        if st == "pending":
            return f["original"], False
        return f["original"], True

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        mcl = f["mcl"]
        val, est = self._compliance_value(f)
        if f["kind"] == "chronic":
            exceeded = val > mcl                     # with a pending quarter, val is the floor (pending counted as 0)
            if exceeded:
                resp = "tier_2_notice_and_corrective_plan"
            elif f["current_pending"]:
                resp = "confirm_before_deciding"
            elif f["current"] > mcl or val > 0.8 * mcl:
                resp = "increased_monitoring"
            else:
                resp = "continue_routine_monitoring"
            r = val / mcl
            expo = 3 if r >= 1.5 else 2 if r > 1.0 else 1 if r > 0.8 else 0
            advice = False
        else:
            st = f["confirm_status"]
            exceeded = (st in ("taken", "not_taken") and val > mcl) or (st == "not_taken")
            if exceeded:
                resp = "immediate_do_not_drink_advisory"
            elif st == "pending":
                resp = "confirm_before_deciding"
            elif f["original"] > 0.5 * mcl:
                resp = "increased_monitoring"
            else:
                resp = "continue_routine_monitoring"
            r = (f["original"] if st == "pending" else val) / mcl
            expo = 3 if (exceeded and r > 1.0) or r >= 1.5 else 2 if r >= 0.9 else 1 if r >= 0.5 else 0
            advice = exceeded or (st == "pending" and f["original"] > 1.5 * mcl)
        conf = 3
        if f["qualifier"]:
            conf -= 1
        if f["current_pending"] or (f["kind"] == "acute" and f["confirm_status"] in ("pending", "not_taken")):
            conf -= 1
        if f["change_note"] and "bypass" in f["change_note"]:
            conf -= 1
        if f["kind"] == "acute" and f["confirm_status"] == "taken" and abs(f["confirmation"] - f["original"]) > 0.35 * f["original"]:
            conf -= 1
        return {"regulatory_response": resp, "exposure_concern": expo, "data_confidence": max(0, conf),
                "standard_exceeded": "yes" if exceeded else "no", "alternative_water_advice_now": "yes" if advice else "no"}

    def _rules(self, rng, f):
        mcl, unit = f["mcl"], f["unit"]
        if f["kind"] == "chronic":
            return [
                pick(rng, [f"DWR-14 §6.1 MCL for {f['chem']}: {mcl:g} {unit}, as the average of the four most recent "
                           "quarterly results at the location. A quarter still at the lab counts as zero until reported; a "
                           "violation is established once the average exceeds the MCL on that basis.",
                           f"DWR-14 §6.1 {f['chem']} complies while the four-quarter running average at the location is at "
                           f"or below {mcl:g} {unit}; an unreported quarter is taken as zero, so a violation can be "
                           "established before it arrives."]),
                "DWR-14 §6.3 A single quarterly result above the MCL, or a running average above 80% of the MCL, moves the "
                "location to monthly monitoring.",
                "DWR-14 §6.5 An MCL violation of a running-average standard requires a Tier 2 notice within 30 days and a "
                "corrective action plan.",
            ]
        return [
            pick(rng, [f"DWR-14 §7.1 {f['chem'].capitalize()} MCL {mcl:g} {unit}. A result above the MCL must be confirmed "
                       "by a second sample within 24 hours; compliance is the average of the original and the "
                       "confirmation. If no confirmation is taken within 24 hours, the original result stands as a violation.",
                       f"DWR-14 §7.1 Acute standard for {f['chem']}: {mcl:g} {unit}. Confirm any exceedance within 24 h and "
                       "average the two; a confirmation not collected within 24 h means the original is the "
                       "compliance value and the MCL is violated."]),
            f"DWR-14 §7.2 Where the original result exceeds 1.5 times the MCL, customers are advised at once to use "
            "another source for drinking and food preparation, without waiting for confirmation.",
            "DWR-14 §7.4 Violation: Tier 1 notice within 24 hours with do-not-drink advice. A result above 50% of the MCL "
            "moves the source to quarterly monitoring.",
        ]

    def _qual(self, f):
        return f" (lab flag: {f['qualifier']})" if f["qualifier"] else ""

    def render(self, f, rng, surface):
        ds = _ds(rng)
        unit = f["unit"]
        note = f["change_note"].format(s=f["source"]) if f["change_note"] else None
        if f["kind"] == "chronic":
            qs = [f"{v:g}" for v in f["prev_quarters"]]
            cur = "at the lab, not reported" if f["current_pending"] else f"{f['current']:g}"
            results = {"Q-3": qs[0], "Q-2": qs[1], "Q-1": qs[2], "current quarter": cur}
        else:
            results = {"original": f"{f['original']:g}"}
            st = f["confirm_status"]
            results["confirmation"] = {"taken": f"{f['confirmation']:g} (collected {f['confirm_hours']} h after the lab's call)"
                                       if f["confirmation"] is not None else None,
                                       "pending": "collected, result pending",
                                       "not_taken": f"not collected until {f['confirm_hours']} h after the lab's call; result void",
                                       "not_required": "not taken"}[st]
        if surface == "json":
            return dumps(prune({
                "system": {"name": f["system"], "pwsid": f["pwsid"], "population_served": f["population"],
                           "source": f["source"]},
                "monitoring": {"contaminant": f["chem"], "location": f["location"], "unit": unit,
                               "results": results, "lab_qualifier": f["qualifier"]},
                "operations": {"note": note},
                "rule_excerpts": self._rules(rng, f),
            }))
        if surface == "compliance_worksheet":
            rows = [[k, v] for k, v in results.items() if v is not None]
            head = kv_lines([("System", f"{f['system']} ({f['pwsid']})"), ("Population", f"{f['population']:,}"),
                             ("Contaminant", f"{CHEMS[f['chem']][3]}"), ("Location", f["location"]),
                             ("Lab flag", f["qualifier"]), ("Operations note", note)], rng)
            return (f"COMPLIANCE WORKSHEET -- {fmt_date(f['date'], style=ds)}\n{head}\n\n"
                    + table(["sample", f"result ({unit})"], rows, rng) + "\n\n" + "\n".join(self._rules(rng, f)))
        lines = [f"{f['system']} ({f['pwsid']}, pop. {f['population']:,}), {f['chem']} at {f['location']}."]
        if f["kind"] == "chronic":
            lines.append("Results: " + "; ".join(f"{k} {v}" for k, v in results.items() if v is not None) + f" {unit}"
                         + self._qual(f) + ".")
        else:
            res = dict(results)
            res["original"] = f"{res['original']} {unit}"
            lines.append("Results: " + "; ".join(f"{k} {v}" for k, v in res.items() if v is not None) + self._qual(f) + ".")
        if note:
            lines.append(_cap(note) + ".")
        lines.append(pick(rng, [f"Spoke with {f['operator']}; they want to know what they have to do.",
                                f"{f['operator']} (operator) asks whether this is a violation.",
                                f"Operator {f['operator']} thinks the last result is a lab error."]))
        return (f"Sanitarian notes -- {f['sanitarian']} -- {fmt_date(f['date'], style=ds)}\n\n"
                + noisy(join_sentences(lines), rng, 0.1) + "\n\nRule:\n" + bullets(self._rules(rng, f), rng))

    def questions(self):
        return {
            "regulatory_response": choice("What response does the rule require of this water system now?", {
                "continue_routine_monitoring": "Nothing beyond the routine monitoring schedule.",
                "increased_monitoring": "Move the location or source to more frequent monitoring.",
                "tier_2_notice_and_corrective_plan": "Notify customers within 30 days and submit a corrective action plan.",
                "immediate_do_not_drink_advisory": "Tell customers at once not to drink the water, with Tier 1 notice.",
                "confirm_before_deciding": "Wait for the outstanding confirmation or quarterly result before deciding.",
            }),
            "exposure_concern": score("How concerning is the customers' exposure on these results?", [
                "Low: comfortably below the standard.",
                "Moderate: near the standard, or a single high result.",
                "High: the standard is, or very likely will be, exceeded.",
                "Serious: well above the standard, or an acute contaminant over its limit.",
            ]),
            "data_confidence": score("How much confidence do the data deserve as a basis for the decision?", [
                "Little: several gaps or quality problems.",
                "Limited: two issues, such as a pending result and a lab flag.",
                "Fair: one issue that does not change the picture much.",
                "High: complete, unflagged results.",
            ]),
            "standard_exceeded": noul(
                "On the compliance basis the rule prescribes, the standard is exceeded.",
                yes="The running average, or the confirmed acute result, is above the MCL as the rule computes it.",
                no="On that basis the result is at or below the MCL, or not yet established."),
            "alternative_water_advice_now": noul(
                "Customers should be told now to use another water source for drinking and food preparation."),
        }


# =========================================================================== schema 3
MARINE_BEACHES = ["Gull Point Beach", "North Cove", "Sandpiper Strand", "Kestrel Bay", "Old Ferry Landing", "Pebble Reach",
                  "Twin Rocks", "Sunset Shoals", "Driftwood Flats", "Cormorant Beach", "Lantern Bay", "Seal Rock Beach"]
FRESH_BEACHES = ["Larch Lake swim area", "Mill Pond beach", "Heron Lake public beach", "Crescent Lake beach",
                 "Willow Lake day-use beach", "Pike Lake town beach", "Clearwater Flowage beach", "Osprey Lake swim area",
                 "Riverbend Park river beach", "Mirror Lake beach"]


def _season_ts(rng, months=(5, 9), hours=(6, 11)):
    d = date(pick(rng, [2025, 2026]), rng.randint(*months), rng.randint(1, 28))
    return datetime(d.year, d.month, d.day, rng.randint(*hours), rng.randrange(60), rng.randrange(60))
STATIONS = ["north end", "centre", "south end", "near the creek mouth", "by the stormwater outfall", "by the lifeguard tower",
            "at the boat ramp"]


class BeachBacteriaPosting(Schema):
    name = "beach_bacteria_posting"
    description = "Bathing-beach bacteria results, rainfall and sewage reports: post, close, lift or resample."
    surfaces = {"json": 0.5, "monitoring_table": 0.25, "duty_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        marine = chance(rng, 0.55)
        bav, gm_std = (70, 35) if marine else (190, 126)
        n_st = pick(rng, [2, 3, 3])
        level = near(rng, 1.0, 0.7, lo=0.1)
        method = pick_w(rng, {"qpcr": 2, "culture": 3})
        age_h = rng.randint(4, 30) if method == "qpcr" else rng.randint(20, 60)
        rain = pick_w(rng, {0: 4, 6: 2, 18: 2, 31: 2, 52: 1})
        sewage = pick_w(rng, {"none": 6, "upstream_contained": 1, "reaching_beach": 1, "unconfirmed": 1})
        existing = pick_w(rng, {"none": 5, "advisory": 3, "closure": 1})
        gm30 = round(gm_std * near(rng, 0.9, 0.5, lo=0.2), 1)
        birds = chance(rng, 0.2)
        source_fixed = chance(rng, 0.5)
        if reg == "clear":
            if chance(rng, 0.5):
                level, rain, sewage, gm30, existing = between(rng, 0.1, 0.4), 0, "none", round(gm_std * 0.4, 1), "none"
                age_h = min(age_h, 30)
            else:
                level, sewage = between(rng, 2.0, 6.0), pick(rng, ["reaching_beach", "none"])
        elif reg == "conflicting":
            twist = pick(rng, ["rain_clean_sample", "outfall_only", "stale_lift", "unconfirmed_spill"])
            if twist == "rain_clean_sample":
                rain, level, method, age_h = rng.randint(26, 60), between(rng, 0.2, 0.8), "qpcr", rng.randint(3, 10)
            elif twist == "outfall_only":
                level = between(rng, 0.6, 1.2)
            elif twist == "stale_lift":
                existing, level, method, age_h = "advisory", between(rng, 0.2, 0.8), "culture", rng.randint(50, 80)
            else:
                sewage, level = "unconfirmed", between(rng, 0.4, 1.1)
        vals = [max(1, int(round(bav * level * math.exp(rng.gauss(0, 0.45))))) for _ in range(n_st)]
        names = rng.sample(STATIONS, n_st)
        if reg == "conflicting" and "by the stormwater outfall" not in names and chance(rng, 0.5):
            names[-1] = "by the stormwater outfall"
            vals[-1] = int(vals[-1] * between(rng, 3, 8))
        return {"regime": reg, "marine": marine, "bav": bav, "gm_standard": gm_std, "stations": names, "values": vals,
                "method": method, "result_age_hours": age_h, "rain_mm_24h": rain, "sewage": sewage,
                "existing_posting": existing, "gm30": gm30, "birds_or_dogs": birds, "sewage_source_fixed": source_fixed,
                "beach": pick(rng, MARINE_BEACHES if marine else FRESH_BEACHES), "county": _county(rng),
                "holiday": chance(rng, 0.3), "sampler": person(rng), "officer": person(rng), "date": _season_ts(rng)}

    def _day_gm(self, f):
        return _gm(f["values"])

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        gm = self._day_gm(f)
        bav = f["bav"]
        over = gm > bav
        stale = f["method"] == "culture" and f["result_age_hours"] > 48
        rain_trig = f["rain_mm_24h"] >= 25 and not (f["method"] == "qpcr" and not over)
        if f["sewage"] == "reaching_beach" or gm > 4 * bav:
            post = "close_beach"
        elif stale and f["existing_posting"] != "none":
            post = "resample_before_posting"
        elif over or f["gm30"] > f["gm_standard"] or rain_trig:
            post = "post_advisory"
        elif f["sewage"] == "unconfirmed":
            post = "resample_before_posting"
        elif f["existing_posting"] == "advisory":
            post = "lift_existing_advisory"
        elif f["existing_posting"] == "closure":
            post = "lift_existing_advisory" if f["sewage_source_fixed"] else "resample_before_posting"
        else:
            post = "no_posting"
        hot = [v for s, v in zip(f["stations"], f["values"]) if v > bav]
        outfall_hot = any(s in ("by the stormwater outfall", "near the creek mouth") and v > bav
                          for s, v in zip(f["stations"], f["values"]))
        if f["sewage"] in ("reaching_beach", "upstream_contained"):
            src = "sewage_infrastructure"
        elif f["rain_mm_24h"] >= 18 or (outfall_hot and len(hot) == 1):
            src = "stormwater_runoff"
        elif f["birds_or_dogs"]:
            src = "wildlife_or_pets"
        else:
            src = "undetermined"
        r = gm / bav
        ill = 0 if r < 0.5 else 1 if r <= 1.0 else 2 if r <= 3.0 else 3
        if f["sewage"] == "reaching_beach":
            ill = 3
        if f["holiday"] and 0 < ill < 3:
            ill += 1
        rel = 3 if f["method"] == "qpcr" else 2
        if f["method"] == "culture" and f["result_age_hours"] > 48:
            rel -= 1
        if len(f["values"]) < 3:
            rel -= 1
        return {"beach_posting": post, "likely_source": src, "illness_risk": ill, "result_reliability": max(0, rel),
                "action_value_exceeded": "yes" if over else "no"}

    def _rules(self, rng, f):
        org_name = "Enterococcus" if f["marine"] else "E. coli"
        return [
            pick(rng, [f"RWC-6 §3.1 {org_name} beach action value {f['bav']} per 100 mL, applied to the geometric mean "
                       f"of all stations sampled at the beach that day; 30-day geometric mean standard {f['gm_standard']}.",
                       f"RWC-6 §3.1 Post an advisory when the day's geometric mean across stations exceeds {f['bav']} "
                       f"{org_name}/100 mL, or the 30-day geometric mean exceeds {f['gm_standard']}."]),
            "RWC-6 §3.3 Close the beach when sewage is known to be reaching the water, or the day's geometric mean "
            "exceeds four times the action value.",
            pick(rng, ["RWC-6 §3.4 Rainfall of 25 mm or more in 24 hours triggers a 72-hour advisory unless a same-day "
                       "qPCR result shows the geometric mean below the action value.",
                       "RWC-6 §3.4 25 mm+ of rain in 24 h: advisory for 72 h, unless same-day qPCR results are below the "
                       "action value."]),
            "RWC-6 §3.5 Lift an advisory on a valid result below the action value; a closure needs the sewage source "
            "repaired first. Culture results older than 48 hours may not be used to lift; resample instead. An "
            "unconfirmed sewage report is resampled before posting.",
        ]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        org_name = "Enterococcus" if f["marine"] else "E. coli"
        method = ("qPCR, same-day" if f["method"] == "qpcr" else "culture (IDEXX), next-day")
        sew = {"none": None, "upstream_contained": "a sewer overflow 3 km upstream was contained before reaching the creek",
               "reaching_beach": "the sewer utility reports a pump-station overflow discharging to the creek that drains onto the beach",
               "unconfirmed": "a caller reported a sewage smell near the outfall; the utility has not confirmed any release"}[f["sewage"]]
        posting = {"none": "not posted", "advisory": "advisory posted since last week",
                   "closure": "closed since a sewage release four days ago"}[f["existing_posting"]]
        if f["existing_posting"] == "closure":
            posting += "; " + ("utility reports the pump station repaired" if f["sewage_source_fixed"] else "repair not yet confirmed")
        extra = []
        if f["birds_or_dogs"]:
            extra.append(pick(rng, ["large gull flock roosting on the sand", "off-leash dogs on the beach daily",
                                    "geese flock resident at the swim area"]))
        if f["holiday"]:
            extra.append("holiday weekend: heavy use expected")
        if surface == "json":
            return dumps(prune({
                "beach": {"name": f["beach"], "county": f["county"], "water": "marine" if f["marine"] else "freshwater",
                          "current_status": posting},
                "samples": {"indicator": org_name, "method": method, "hours_since_collection": f["result_age_hours"],
                            "per_100ml_by_station": dict(zip(f["stations"], f["values"])),
                            "thirty_day_geometric_mean": f["gm30"]},
                "conditions": {"rain_mm_24h": f["rain_mm_24h"], "sewage_reports": sew, "observations": extra or None},
                "code": self._rules(rng, f),
            }))
        if surface == "monitoring_table":
            rows = [[s, str(v)] for s, v in zip(f["stations"], f["values"])]
            head = kv_lines([("Beach", f"{f['beach']} ({f['county']})"), ("Status", posting),
                             ("Indicator / method", f"{org_name} / {method}"),
                             ("Collected", fmt_ts(f["date"], rng)), ("Result age", f"{f['result_age_hours']} h"),
                             ("30-day geometric mean", f["gm30"]), ("Rain, last 24 h", f"{f['rain_mm_24h']} mm"),
                             ("Sewage reports", sew or "none"), ("Notes", "; ".join(extra) or None)], rng)
            return (f"BEACH MONITORING -- {f['county']} Environmental Health\n{head}\n\n"
                    + table(["station", f"{org_name}/100 mL"], rows, rng) + "\n\n" + "\n".join(self._rules(rng, f)))
        t = f["date"]
        msgs = [(f["sampler"], f"{f['beach']} results in ({method}, collected {f['result_age_hours']}h ago): "
                               + ", ".join(f"{s} {v}" for s, v in zip(f["stations"], f["values"]))),
                (f["officer"], f"status right now: {posting}. 30-day GM is {f['gm30']}"),
                (f["sampler"], f"rain gauge shows {f['rain_mm_24h']} mm in the last 24h")]
        if sew:
            msgs.append((f["officer"], sew))
        for x in extra:
            msgs.append((f["sampler"], x))
        msgs.append((f["officer"], pick(rng, ["what do we put on the sign?", "parks wants an answer before 10",
                                              "need a call on posting"])))
        out = []
        for who, text in msgs:
            t += timedelta(minutes=rng.randint(1, 12))
            out.append(f"[{t.strftime('%H:%M')}] {who.split()[0].lower()}: {noisy(text, rng, 0.1)}")
        return f"#beach-monitoring ({f['county']})\n" + "\n".join(out) + "\n\nCode refs:\n" + bullets(self._rules(rng, f), rng)

    def questions(self):
        return {
            "beach_posting": choice("What should the beach's status be today?", {
                "no_posting": "Leave the beach open with no advisory.",
                "post_advisory": "Post (or keep) a swimming advisory.",
                "close_beach": "Close the beach to water contact.",
                "resample_before_posting": "Resample before changing the beach's status.",
                "lift_existing_advisory": "Lift the posting now in place.",
            }),
            "likely_source": choice("What is the most likely source of the bacteria at this beach?", {
                "stormwater_runoff": "Stormwater runoff after rain, or through an outfall.",
                "sewage_infrastructure": "A sewer overflow or other sewage infrastructure.",
                "wildlife_or_pets": "Birds, dogs or other animals using the beach.",
                "undetermined": "Nothing in the file points to a particular source.",
            }),
            "illness_risk": score("How great is the risk of illness for people swimming here today?", [
                "Low: indicator levels well below the action value.",
                "Moderate: levels approaching the action value.",
                "High: levels above the action value.",
                "Very high: sewage in the water, or levels far above the action value with heavy use.",
            ]),
            "result_reliability": score("How reliable are today's results as a basis for the posting decision?", [
                "Unreliable: stale results from too few stations.",
                "Limited: stale results, or too few stations.",
                "Adequate: next-day culture results from enough stations.",
                "Good: same-day results from enough stations.",
            ]),
            "action_value_exceeded": noul(
                "The day's results exceed the beach action value.",
                yes="The geometric mean of the stations sampled today is above the action value.",
                no="The geometric mean is at or below the action value, even if one station is above it."),
        }


# =========================================================================== schema 4
LAKES = ["Larch Lake", "Copper Reservoir", "Fenwick Pond", "Lake Marrow", "Heron Lake", "Blackwater Reservoir",
         "Willow Lake", "Stoney Creek Reservoir", "Lake Tamsin", "Pike Lake", "Osprey Lake", "Clearwater Flowage",
         "Mirror Lake", "Badger Reservoir"]
LEVELS = ["none", "caution", "warning", "danger"]


class AlgalBloomPosting(Schema):
    name = "algal_bloom_posting"
    description = "A cyanobacterial bloom on a lake: which posting level the code requires and whether a drinking-water intake is at risk."
    surfaces = {"json": 0.5, "field_report": 0.25, "lab_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        visual = pick_w(rng, {"none": 2, "streaks": 3, "scum_open_water": 2, "scum_swim_area": 2})
        cells = int(round(near(rng, 60000, 55000, lo=500), -2))
        mc_lab = maybe(rng, round(near(rng, 8, 9, lo=0.1), 1), 0.25)
        strip = pick_w(rng, {None: 3, "low": 1, "moderate": 1, "high": 1})
        age = pick(rng, [1, 2, 3, 5, 8, 12])
        current = pick_w(rng, {"none": 3, "caution": 3, "warning": 2, "danger": 1})
        below_weeks = pick(rng, [0, 0, 1, 2])
        animal = pick_w(rng, {"none": 7, "unconfirmed": 2, "confirmed": 1})
        source = chance(rng, 0.5)
        raw_mc = round(near(rng, 1.0, 0.8, lo=0.0), 2) if source else None
        if reg == "clear":
            if chance(rng, 0.5):
                visual, cells, mc_lab, strip, animal, current = "none", rng.randint(500, 6000), round(between(rng, 0.1, 1.5), 1), None, "none", "none"
                raw_mc = 0.05 if source else None
            else:
                visual, cells, mc_lab = "scum_swim_area", rng.randint(200000, 900000), round(between(rng, 25, 90), 1)
        elif reg == "conflicting":
            twist = pick(rng, ["strip_high_lab_low", "old_danger_stepdown", "dog_death", "open_water_only"])
            if twist == "strip_high_lab_low":
                strip, mc_lab = "high", round(between(rng, 2, 7.5), 1)
            elif twist == "old_danger_stepdown":
                current, mc_lab, below_weeks = "danger", round(between(rng, 1, 7), 1), pick(rng, [1, 2])
            elif twist == "dog_death":
                animal, mc_lab = pick(rng, ["confirmed", "unconfirmed"]), maybe(rng, round(between(rng, 3, 15), 1), 0.4)
            else:
                visual, mc_lab, cells = "scum_open_water", round(between(rng, 9, 30), 1), rng.randint(80000, 400000)
        return {"regime": reg, "visual": visual, "cells_per_ml": cells, "microcystin_lab": mc_lab, "strip_test": strip,
                "sample_age_days": age, "current_level": current, "weeks_below_current": below_weeks,
                "animal_illness": animal, "drinking_source": source, "raw_water_microcystin": raw_mc,
                "intake_depth_m": pick(rng, [3, 5, 8, 12]) if source else None, "lake": pick(rng, LAKES),
                "county": _county(rng), "wind": pick(rng, ["light and variable", "onshore toward the swim beach",
                                                           "offshore", "strong from the north"]),
                "sampler": person(rng), "officer": person(rng), "date": _season_ts(rng, (6, 10), (7, 16)),
                "lab_id": ident(rng, "HAB", 5, "-")}

    def _rule_level(self, f):
        """The level the code's criteria point to today, before the step-down rule."""
        mc = f["microcystin_lab"]
        if (mc is not None and mc >= 20) or f["animal_illness"] == "confirmed":
            return 3
        if (mc is not None and mc >= 8) or (f["visual"] == "scum_swim_area" and f["cells_per_ml"] >= 100000):
            return 2
        if f["visual"] != "none" or f["cells_per_ml"] >= 20000 or f["strip_test"] in ("moderate", "high"):
            return 1
        return 0

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        lvl = self._rule_level(f)
        cur = LEVELS.index(f["current_level"])
        if lvl < cur:
            lvl = cur - 1 if f["weeks_below_current"] >= 2 else cur      # step down one level after two clean weeks
        if f["strip_test"] == "high" and lvl < 2:
            lvl = max(lvl, 1)
        post = ["no_posting", "caution_watch", "warning_avoid_contact", "danger_closed"][lvl]
        mc = f["microcystin_lab"]
        risk = 0
        if mc is not None:
            risk = 3 if mc >= 20 else 2 if mc >= 8 else 1 if mc >= 2 else 0
        elif f["strip_test"] in ("moderate", "high"):
            risk = 2
        if f["visual"] == "scum_swim_area" or f["wind"].startswith("onshore"):
            risk = min(3, risk + 1)
        if f["animal_illness"] == "confirmed":
            risk = 3
        cur_ev = 3
        if f["sample_age_days"] > 7:
            cur_ev -= 2
        elif f["sample_age_days"] > 3:
            cur_ev -= 1
        if mc is None:
            cur_ev -= 1
        danger = (mc is not None and mc >= 20) or f["animal_illness"] == "confirmed"
        intake = f["drinking_source"] and ((f["raw_water_microcystin"] or 0) >= 1.0 or
                                           (f["visual"] != "none" and f["intake_depth_m"] <= 5))
        return {"bloom_posting_level": post, "toxin_exposure_risk": risk, "evidence_currency": max(0, cur_ev),
                "danger_threshold_reached": "yes" if danger else "no",
                "drinking_intake_threatened": "yes" if intake else "no"}

    def _rules(self, rng):
        return [
            pick(rng, ["RWC-6 §7.2 Caution: visible bloom, cyanobacteria at 20,000 cells/mL or more, or a positive field "
                       "strip. Warning: lab microcystin 8 ug/L or more, or scum in the swim area with 100,000 cells/mL "
                       "or more. Danger (closed): lab microcystin 20 ug/L or more, or a confirmed animal death or illness.",
                       "RWC-6 §7.2 Levels: caution (visible bloom / >= 20,000 cells per mL / positive strip), warning "
                       "(microcystin >= 8 ug/L by lab, or swim-area scum >= 100,000 cells/mL), danger (microcystin >= 20 "
                       "ug/L by lab, or a vet-confirmed animal poisoning)."]),
            "RWC-6 §7.3 Field strip tests may raise a posting but never lower one; only laboratory results count "
            "toward the warning and danger toxin thresholds.",
            "RWC-6 §7.4 A posting comes down one level at a time, and only after two consecutive weekly samples below "
            "the criteria for the current level.",
            "RWC-6 §7.6 Where the lake is a drinking-water source, notify the water system when raw-water microcystin "
            "reaches 1 ug/L, or a bloom is visible over an intake 5 m deep or shallower.",
        ]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        vis = {"none": "no visible bloom", "streaks": "green streaks along the shoreline",
               "scum_open_water": "surface scum in open water away from the beach",
               "scum_swim_area": "paint-like scum inside the swim area buoys"}[f["visual"]]
        strip = {None: None, "low": "field strip: faint / low", "moderate": "field strip: moderate",
                 "high": "field strip: strong positive"}[f["strip_test"]]
        lab = (f"{f['microcystin_lab']} ug/L microcystin (ELISA)" if f["microcystin_lab"] is not None
               else "lab toxin result not yet available")
        animal = {"none": None, "unconfirmed": "a dog owner reports her dog vomited after swimming; no vet report",
                  "confirmed": "vet necropsy confirms a dog died of microcystin poisoning after swimming here"}[f["animal_illness"]]
        cur = {"none": "not posted", "caution": "CAUTION sign up", "warning": "WARNING sign up",
               "danger": "DANGER / closed"}[f["current_level"]]
        if f["current_level"] != "none" and f["weeks_below_current"]:
            cur += f"; {f['weeks_below_current']} consecutive weekly sample(s) since then below this level's criteria"
        intake = (f"lake supplies a treatment plant; intake at {f['intake_depth_m']} m; raw-water microcystin "
                  f"{f['raw_water_microcystin']} ug/L" if f["drinking_source"] else None)
        if surface == "json":
            return dumps(prune({
                "site": {"lake": f["lake"], "county": f["county"], "current_posting": cur, "drinking_water": intake},
                "observations": {"visual": vis, "wind": f["wind"], "animal_reports": animal,
                                 "cells_per_ml": f["cells_per_ml"], "field_strip": strip},
                "lab": {"sample_id": f["lab_id"], "result": lab, "sample_age_days": f["sample_age_days"]},
                "code": self._rules(rng),
            }))
        if surface == "field_report":
            head = kv_lines([("Lake", f"{f['lake']} ({f['county']})"), ("Visited", fmt_ts(f["date"], rng)),
                             ("Sampler", f["sampler"]), ("Current posting", cur), ("Visual", vis), ("Wind", f["wind"]),
                             ("Cell count", f"{f['cells_per_ml']:,} cells/mL"), ("Strip", strip),
                             ("Lab", f"{lab}, sample {f['sample_age_days']} day(s) old"), ("Animal reports", animal),
                             ("Drinking water", intake)], rng)
            return f"HAB FIELD REPORT\n{head}\n\n" + "\n".join(self._rules(rng))
        para = [f"Results for {f['lab_id']} ({f['lake']}): {lab}; {f['cells_per_ml']:,} cells/mL. The sample is "
                f"{f['sample_age_days']} day(s) old.",
                f"Field crew saw {vis}, wind {f['wind']}." + (f" {_cap(strip)}." if strip else ""),
                f"Current status: {cur}."]
        if animal:
            para.append(_cap(animal) + ".")
        if intake:
            para.append(_cap(intake) + ".")
        return "\n".join([f"Subject: {f['lake']} cyanobacteria results", f"From: lab@{f['county'].split()[0].lower()}-health.example",
                          f"To: {f['officer']}", "", noisy(join_sentences(para), rng, 0.1), "",
                          "Code:", bullets(self._rules(rng), rng)])

    def questions(self):
        return {
            "bloom_posting_level": choice("What posting level does the code require at this lake today?", {
                "no_posting": "No posting.",
                "caution_watch": "A caution sign telling people to watch for scum and keep pets out of it.",
                "warning_avoid_contact": "A warning sign telling people to avoid contact with the water.",
                "danger_closed": "A danger posting closing the water to all contact.",
            }),
            "toxin_exposure_risk": score("How much toxin exposure risk do people and pets face at the swim area?", [
                "Minimal: little or no toxin and no scum near people.",
                "Low: some toxin or bloom, but away from the swim area.",
                "Elevated: toxin near the warning level, or scum where people swim.",
                "High: danger-level toxin or a confirmed poisoning.",
            ]),
            "evidence_currency": score("How current and complete is the evidence behind the posting decision?", [
                "Stale: old samples and no lab toxin result.",
                "Dated: a week-old sample, or no lab result.",
                "Fairly current: a few days old, lab-backed.",
                "Current: a fresh sample with a lab toxin result.",
            ]),
            "danger_threshold_reached": noul(
                "The evidence meets the code's threshold for a danger posting.",
                yes="A lab microcystin result at or above the danger level, or a confirmed animal poisoning.",
                no="Neither a lab result at the danger level nor a confirmed poisoning."),
            "drinking_intake_threatened": noul(
                "The bloom threatens the treated drinking water drawn from this lake."),
        }


# =========================================================================== schema 5
class BoilNoticeLiftRequest(Schema):
    name = "boil_notice_lift_request"
    description = "A utility's request to lift a boil-water notice after a main break, judged against the lift criteria."
    surfaces = {"json": 0.5, "clearance_table": 0.25, "utility_request_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        n_zones = pick(rng, [1, 2, 2, 3])
        isolated = chance(rng, 0.6) and n_zones > 1
        sysname, pop = _system_pop(rng)
        if pop < 1500:                       # a small system has one pressure zone
            n_zones, isolated = 1, False
        total_conn = max(40, pop / 2.5 * between(rng, 0.15, 0.7))
        spacing = near(rng, 31, 7, lo=6, integer=True)
        zones = []
        znames = rng.sample(["Zone A", "Zone B", "Zone C", "Hilltop", "Lower town", "North loop", "Industrial park",
                             "School hill", "Riverside", "Pump station 2 area"], n_zones)
        for z in range(n_zones):
            conns = max(20, int(round(total_conn / n_zones * between(rng, 0.6, 1.4), -1)))
            req = max(3, math.ceil(conns / 250))
            sites = [req + pick(rng, [0, 0, 0, 1, 1, 2]) for _ in range(2)]
            res = [round(between(rng, 0.28, 0.95), 2) for _ in range(2)]
            pos = [0, 0]
            zones.append({"name": znames[z] if n_zones > 1 else "notice area",
                          "connections": conns, "sites": sites, "min_residual": res, "tc_positive": pos})
        rounds = 2
        flushed = chance(rng, 0.88)
        pressure_ok = chance(rng, 0.94)
        outside_pos = False
        turbidity = round(near(rng, 0.6, 0.5, lo=0.1), 1)
        if reg == "clear":
            if chance(rng, 0.5):
                spacing, flushed, pressure_ok = rng.randint(24, 40), True, True
                for z in zones:
                    z["sites"] = [max(3, math.ceil(z["connections"] / 250)) + 1] * 2
                    z["min_residual"] = [round(between(rng, 0.4, 1.0), 2)] * 2
                turbidity = 0.3
            else:
                zones[0]["tc_positive"] = [0, rng.randint(1, 3)]
                zones[0]["min_residual"][1] = round(between(rng, 0.02, 0.12), 2)
        elif reg == "borderline":
            z = pick(rng, zones)
            k = rng.randrange(2)
            z["min_residual"][k] = round(near(rng, 0.23, 0.06, lo=0.05), 2)
            if chance(rng, 0.35):
                z["sites"][k] = max(1, max(3, math.ceil(z["connections"] / 250)) - pick(rng, [0, 1]))
        else:
            twist = pick(rng, ["one_zone_bad", "one_zone_bad", "one_round", "one_round", "outside_positive",
                               "outside_positive", "short_spacing"])
            if twist == "one_zone_bad":
                z = zones[-1]
                z["tc_positive"] = [0, 1]
                isolated = len(zones) > 1 and chance(rng, 0.75)
            elif twist == "one_round":
                rounds = 1
                if chance(rng, 0.75):
                    zones[0]["min_residual"][0] = round(between(rng, 0.05, 0.18), 2)
            elif twist == "outside_positive":
                outside_pos = True
                zones[0]["tc_positive"] = [pick(rng, [0, 1]), 1]
            else:
                spacing = rng.randint(10, 20)
        return {"regime": reg, "zones": zones, "zones_isolated": isolated, "rounds": rounds, "round_spacing_h": spacing,
                "flushing_documented": flushed, "pressure_restored": pressure_ok, "positive_outside_area": outside_pos,
                "turbidity_ntu": turbidity, "system": sysname, "population": pop, "pwsid": ident(rng, "PWS", 7, ""),
                "hospital_in_area": chance(rng, 0.25), "manager": person(rng), "sanitarian": person(rng),
                "street": pick(rng, STREETS), "notice_issued": rand_ts(rng, hours=(5, 20)),
                "pressure_from": pick(rng, [None, None, "the mayor's office", "a food processor in the area",
                                            "the school district"])}

    def _zone_ok(self, f, z):
        req = max(3, math.ceil(z["connections"] / 250))
        rounds = f["rounds"]
        return (rounds >= 2 and f["round_spacing_h"] >= 24
                and all(s >= req for s in z["sites"][:rounds])
                and all(p == 0 for p in z["tc_positive"][:rounds])
                and all(r >= 0.2 for r in z["min_residual"][:rounds]))

    def expected(self, f):
        """Intended answers from the latent factors (spread audit only; never rendered)."""
        oks = [self._zone_ok(f, z) for z in f["zones"]]
        common = f["flushing_documented"] and f["pressure_restored"]
        any_pos = any(p for z in f["zones"] for p in z["tc_positive"][:f["rounds"]])
        sampling_met = (f["rounds"] >= 2 and f["round_spacing_h"] >= 24 and all(
            all(s >= max(3, math.ceil(z["connections"] / 250)) for s in z["sites"][:f["rounds"]])
            and all(p == 0 for p in z["tc_positive"][:f["rounds"]]) for z in f["zones"]))
        if f["positive_outside_area"]:
            dec = "keep_and_expand_area"
        elif all(oks) and common:
            dec = "lift_whole_area"
        elif any(oks) and f["zones_isolated"] and common:
            dec = "lift_clean_zones_only"
        else:
            dec = "keep_notice_another_round"
        min_res = min(r for z in f["zones"] for r in z["min_residual"][:f["rounds"]])
        if any_pos:
            follow = "sanitary_assessment"
        elif not f["flushing_documented"] or min_res < 0.2 or not f["pressure_restored"]:
            follow = "flushing_plan_review"
        elif f["hospital_in_area"] or f["turbidity_ntu"] > 1.0 or min_res < 0.3:
            follow = "extra_monitoring_week"
        else:
            follow = "no_followup"
        ev = 3
        if f["rounds"] < 2:
            ev -= 2
        elif f["round_spacing_h"] < 24:
            ev -= 1
        if any(s < max(3, math.ceil(z["connections"] / 250)) for z in f["zones"] for s in z["sites"][:f["rounds"]]):
            ev -= 1
        if min_res < 0.2:
            ev -= 1
        if any_pos:
            ev -= 1
        risk = 0
        if any_pos:
            risk += 2
        if min_res < 0.2:
            risk += 1
        if not f["flushing_documented"] or not f["pressure_restored"]:
            risk += 1
        if f["rounds"] < 2 or f["round_spacing_h"] < 24:
            risk += 1
        if f["positive_outside_area"]:
            risk = 3
        return {"lift_decision": dec, "followup_requirement": follow, "clearance_evidence": max(0, ev),
                "residual_risk_if_lifted": min(3, risk), "lift_sampling_rule_met": "yes" if sampling_met else "no"}

    def _rules(self, rng):
        return [
            pick(rng, ["DWR-14 §9.2 Lift only after two consecutive sampling rounds at least 24 hours apart with every "
                       "sample total-coliform negative.",
                       "DWR-14 §9.2 Two rounds, collected 24 h or more apart, all TC-negative, are required to lift."]),
            pick(rng, ["DWR-14 §9.3 Each round: one site per 250 service connections in the area (minimum 3), "
                       "including the repair site and dead ends.",
                       "DWR-14 §9.3 Sites per round: 1 per 250 connections, never fewer than 3, repair site and dead ends included."]),
            "DWR-14 §9.4 Free chlorine of at least 0.2 mg/L at every site in both rounds; repairs complete, mains flushed "
            "and disinfected, pressure restored throughout.",
            "DWR-14 §9.6 A zone that is valved off from the rest may be lifted on its own if it meets every criterion. A "
            "positive outside the notice boundary requires the area to be widened.",
        ]

    def render(self, f, rng, surface):
        ds = _ds(rng)
        r2 = f["notice_issued"] + timedelta(hours=30)
        rows = []
        for z in f["zones"]:
            for k in range(f["rounds"]):
                rows.append([z["name"], f"{z['connections']:,}", f"R{k + 1}", str(z["sites"][k]),
                             str(z["tc_positive"][k]), f"{z['min_residual'][k]:.2f}"])
        flags = []
        flags.append("flushing and disinfection log attached" if f["flushing_documented"] else "flushing log not provided")
        flags.append("pressure restored to all services" if f["pressure_restored"] else "two streets still on low pressure")
        if f["positive_outside_area"]:
            flags.append(f"one TC-positive at a routine site on {f['street']}, just outside the notice boundary")
        if f["hospital_in_area"]:
            flags.append("a dialysis clinic is inside the notice area")
        flags.append(f"turbidity {f['turbidity_ntu']} NTU at the entry point")
        iso = ("zones can be valved off from each other" if f["zones_isolated"] else
               "zones share mains and cannot be isolated") if len(f["zones"]) > 1 else None
        spacing = (f"round 2 collected {f['round_spacing_h']} h after round 1" if f["rounds"] >= 2
                   else "only one round collected so far")
        if surface == "json":
            return dumps(prune({
                "notice": {"system": f["system"], "pwsid": f["pwsid"], "issued": fmt_ts(f["notice_issued"], style="iso"),
                           "cause": f"main break on {f['street']}"},
                "sampling": {"rounds": spacing, "zone_isolation": iso,
                             "by_zone": [{"zone": z["name"], "connections": z["connections"],
                                          "sites_per_round": z["sites"][:f["rounds"]],
                                          "tc_positive_per_round": z["tc_positive"][:f["rounds"]],
                                          "min_free_chlorine_mg_l": z["min_residual"][:f["rounds"]]} for z in f["zones"]]},
                "status": flags,
                "requested_by": {"manager": f["manager"], "pressure_from": f["pressure_from"]},
                "rule_excerpts": self._rules(rng),
            }))
        if surface == "clearance_table":
            head = kv_lines([("System", f"{f['system']} ({f['pwsid']})"), ("Notice issued", fmt_ts(f["notice_issued"], rng)),
                             ("Cause", f"main break on {f['street']}"), ("Rounds", spacing), ("Zones", iso)], rng)
            return (f"BOIL-WATER NOTICE CLEARANCE REVIEW\n{head}\n\n"
                    + table(["zone", "connections", "round", "sites", "TC+", "min Cl2 (mg/L)"], rows, rng)
                    + "\n\n" + bullets(flags, rng) + "\n\n" + "\n".join(self._rules(rng)))
        para = [f"We're requesting to lift the boil-water notice issued {fmt_ts(f['notice_issued'], rng)} after the main "
                f"break on {f['street']}.", _cap(spacing) + "."]
        for z in f["zones"]:
            para.append(f"{z['name']} ({z['connections']:,} connections): sites " + "/".join(map(str, z["sites"][:f["rounds"]]))
                        + ", TC-positives " + "/".join(map(str, z["tc_positive"][:f["rounds"]]))
                        + ", lowest chlorine " + "/".join(f"{x:.2f}" for x in z["min_residual"][:f["rounds"]]) + " mg/L.")
        if iso:
            para.append(_cap(iso) + ".")
        para += [_cap(x) + "." for x in flags]
        if f["pressure_from"]:
            para.append(f"We are getting a lot of pressure from {f['pressure_from']} to lift today.")
        return "\n".join([f"Subject: Request to lift BWN -- {f['system']}", f"From: {f['manager']}",
                          f"To: {f['sanitarian']}, Office of Drinking Water", "", noisy(join_sentences(para), rng, 0.1),
                          "", f["manager"], "", "(ODW reviewer's rule excerpt)", bullets(self._rules(rng), rng)])

    def questions(self):
        return {
            "lift_decision": choice("How should the request to lift the boil-water notice be answered?", {
                "lift_whole_area": "Lift the notice for the whole area.",
                "lift_clean_zones_only": "Lift it only for the zones that meet every criterion.",
                "keep_notice_another_round": "Keep the notice and require another round of sampling.",
                "keep_and_expand_area": "Keep the notice and widen the area it covers.",
            }),
            "followup_requirement": choice("What follow-up should go with the decision?", {
                "no_followup": "No follow-up beyond routine monitoring.",
                "extra_monitoring_week": "A week of extra sampling in the area after the notice comes down.",
                "flushing_plan_review": "A review of the flushing and disinfection work.",
                "sanitary_assessment": "A sanitary assessment to find how coliforms got into the system.",
            }),
            "clearance_evidence": score("How strong is the sampling evidence that the water is safe again?", [
                "Weak: missing rounds or positives in the results.",
                "Partial: more than one shortfall against the criteria.",
                "Nearly complete: one criterion narrowly missed.",
                "Complete: every round, site and residual criterion met.",
            ]),
            "residual_risk_if_lifted": score("If the whole notice were lifted today, how much risk would customers face?", [
                "Negligible.",
                "Low: a gap in the paperwork rather than in the water.",
                "Moderate: a weakness that could let contamination persist.",
                "High: evidence of contamination still in or beyond the area.",
            ]),
            "lift_sampling_rule_met": noul(
                "The sampling meets the rule's requirements for lifting the notice across the whole area.",
                yes="Two rounds at least 24 hours apart, with enough sites in every zone and every sample TC-negative.",
                no="A round is missing or too soon, a zone has too few sites, or a sample was positive."),
        }


WORKFLOW = Workflow(
    name="water_quality_advisory",
    title="Water quality advisories",
    description="Public-health advisory decisions for drinking-water systems and bathing waters (boil-water and do-not-"
                "drink notices, notice tiers, beach and bloom postings), judged against written action levels.",
    domain="environment / public water health",
    schemas=[ColiformPositiveFollowup(), ChemicalRunningAverage(), BeachBacteriaPosting(), AlgalBloomPosting(),
             BoilNoticeLiftRequest()],
    notes="Fictional Drinking Water Rule (DWR-14) and Recreational Water Code (RWC-6). Advisories to the public about "
          "water systems and bathing waters only; no facility discharge-permit reporting, no customer threads.",
)
