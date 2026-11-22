"""Hospital capacity escalation: bed-board and surge decisions (escalation level, elective surgery, ambulance
diversion, ICU space, nurse staffing, transfer acceptance) against a hospital's written capacity plan.

Everything here is fictional: the hospitals, units, capacity plans and every trigger in them, the staff and the
patients. Each state quotes the part of its hospital's plan the case turns on: escalation triggers on occupancy,
ED boarding and ICU space, a regional diversion agreement, ICU step-down criteria, nurse-to-patient ratios and a
transfer-acceptance policy. The thresholds vary between the fictional hospitals. None of it is real operational or
clinical guidance, and the level names are invented, not any real system's scheme.

Equity is part of the plans the way an equity-aware hospital would write it: discharge barriers (transport, no safe
place to go, an interpreter for discharge teaching, medicines a patient without coverage cannot afford) bring in the
discharge-support team and never justify an unsafe discharge; transfer acceptance turns on capability and capacity
only, never on payer, immigration status or ability to pay, and a finance-clearance request is not a step in it.
Payer, language and coverage are recorded where a real transfer log records them.

Each schema draws a `regime` first ('clear' / 'borderline' / 'conflicting'): borderline cases sit on a trigger
(94.6% occupancy against a 95% line, a nurse who would reach the overtime cap exactly), conflicting cases carry a
manager's reassurance against the numbers, a stale figure, or pressure that the written plan overrides. Every schema
defines `intended(f) -> {qid: answer}` (choice key, "yes"/"no", or score level index), the plan applied to the
factors; it is used only to check that answers spread across options and is never rendered.
"""
from __future__ import annotations

import math
from datetime import timedelta

from .base import (
    Schema,
    Workflow,
    bullets,
    chance,
    choice,
    dumps,
    fmt_date,
    join_sentences,
    kv_lines,
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

HOSP_STEMS = ["Larchwood", "Cresthaven", "Marlbrook", "Ashwick", "Pemberly Vale", "Rowanmere", "Stonebridge Hollow",
              "Thornfield Cross", "Kestrel Point", "Wyvern Bay", "Calloway", "Briarcliff", "Holloway Park", "Quillmoor",
              "Fennimore", "Blackthorn Rise", "Silverdale Fen", "Merriton", "Gullhaven", "Eastmere", "Brookhallow",
              "Tamberlane", "Cindervale", "Northwick Glen", "Harrowgate Mill", "Lindenfold"]
HOSP_KINDS = ["Medical Center", "General Hospital", "Regional Hospital", "University Medical Center",
              "Community Hospital", "Memorial Medical Center"]
MEDSURG = ["4 North Med/Surg", "5 West Med/Surg", "6 East Surgical", "3 South Medicine", "7 North Oncology",
           "2 West Orthopaedics", "5 East Renal", "6 West Neurosciences", "4 East General Medicine",
           "8 North Respiratory", "3 West Gastroenterology", "7 South Vascular", "9 East Acute Medicine",
           "2 South Short Stay", "8 West Geriatrics"]
TELE = ["3 North Telemetry", "4 West Cardiac Tele", "6 South Telemetry"]
STEPDOWN = ["Progressive Care Unit", "5 South Step-Down", "Cardiac Step-Down"]
ICUS = ["MICU", "SICU", "CVICU", "Neuro ICU", "Mixed ICU"]


def _hospital(rng):
    return f"{pick(rng, HOSP_STEMS)} {pick(rng, HOSP_KINDS)}"


def _hm(t):
    return t.strftime("%H:%M")


def _cap(text):
    return text[:1].upper() + text[1:] if text else text


# =========================================================================== schema 1: house-wide huddle
class HouseWideEscalation(Schema):
    name = "house_wide_escalation"
    description = ("The morning bed huddle: the escalation level the written plan sets from occupancy, ED boarding "
                   "and ICU space, and what follows for elective surgery, boarders and blocked discharges.")
    surfaces = {"json": 0.55, "bed_board": 0.225, "huddle_minutes": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"{rng.randint(2, 7)}.{rng.randint(0, 9)}", "t2": pick(rng, [88, 90, 92]),
             "t3": pick(rng, [94, 95, 96]), "b2": pick(rng, [5, 6, 8]), "b3": pick(rng, [10, 12, 14]),
             "b4": pick(rng, [18, 20, 24]), "h3": pick(rng, [6, 8, 12]), "barrier_n": pick(rng, [2, 3, 4])}
        units = rng.sample(MEDSURG, rng.randint(4, 7)) + rng.sample(TELE, 1)
        staffed = [rng.randint(18, 34) for _ in units]
        total = sum(staffed)
        if reg == "clear":
            pct = pick(rng, [rng.uniform(72, 86), rng.uniform(72, 86), rng.uniform(100.5, 105)])
        elif reg == "borderline":
            pct = near(rng, pick(rng, [g["t2"], g["t2"], g["t3"], 100]), 1.4)
        else:
            pct = rng.uniform(84, 99)
        occupied = int(round(total * pct / 100))
        surge = max(0, occupied - total)
        # spread the occupied beds over the units (surge beds count on the busiest units)
        occ = [min(s, int(s * occupied / total)) for s in staffed]
        i = 0
        while sum(occ) < occupied - surge:
            if occ[i % len(occ)] < staffed[i % len(occ)]:
                occ[i % len(occ)] += 1
            i += 1
        occ[0] += surge
        boarders = (near(rng, pick(rng, [g["b2"], g["b3"]]), 2, lo=0, integer=True) if reg == "borderline"
                    else pick_w(rng, {rng.randint(0, 4): 3.5, rng.randint(5, 12): 3, rng.randint(13, 26): 1.3}))
        longest = None
        if boarders:
            hi = 5 if boarders <= 4 else 10 if boarders <= 12 else 30      # long waits come with a full ED
            longest = round(near(rng, g["h3"], 1.5) if reg == "borderline" and chance(rng, 0.4)
                            else rng.uniform(0.5, hi), 1)
        icu_staffed = rng.randint(12, 28)
        icu_free = pick_w(rng, {0: 1.3, 1: 2, 2: 3, 3: 3, 4: 2})
        icu_boarders = pick(rng, [1, 1, 2]) if icu_free == 0 and chance(rng, 0.6) else 0
        dc_conf = rng.randint(0, 28)
        if reg == "conflicting":
            dc_conf = max(dc_conf, boarders + rng.randint(0, 6))
        barriers = {"transport": pick_w(rng, {0: 4, 1: 2, 2: 1.5, 3: 1, 5: 0.5}),
                    "no_safe_destination": pick_w(rng, {0: 6, 1: 2, 2: 0.6}),
                    "interpreter_for_teaching": pick_w(rng, {0: 6, 1: 2, 2: 0.8}),
                    "medicines_without_coverage": pick_w(rng, {0: 6, 1: 2, 2: 0.7})}
        manager = None
        if reg == "conflicting":
            manager = pick(rng, ["night manager: 'we got through last night, no need to escalate'",
                                 "bed manager expects 'loads of discharges' but has no names yet",
                                 "surgical lead asks to keep the full elective list running",
                                 "a director suggests discharging the patients with transport problems by taxi at "
                                 "their own cost"])
        t = rand_ts(rng, hours=(7, 7)).replace(minute=pick(rng, [0, 15, 30]), second=0)
        return {"regime": reg, "plan": g, "hospital": _hospital(rng), "huddle_time": t, "lead": person(rng),
                "units": [{"unit": u, "staffed": s, "occupied": o} for u, s, o in zip(units, staffed, occ)],
                "staffed_total": total, "occupied_total": occupied, "surge_in_use": surge, "ed_boarders": boarders,
                "longest_boarding_h": longest, "icu_unit": pick(rng, ICUS), "icu_staffed": icu_staffed,
                "icu_free": icu_free, "icu_boarders_in_ed": icu_boarders, "discharges_confirmed_before_11": dc_conf,
                "blocked_discharges": barriers, "electives_needing_bed": rng.randint(2, 16),
                "electives_needing_icu": pick_w(rng, {0: 2, 1: 2, 2: 1.2, 3: 0.5}), "manager_remark": manager}

    @staticmethod
    def _level(f):
        g = f["plan"]
        pct = 100.0 * f["occupied_total"] / f["staffed_total"]
        lo = 4 if pct >= 100 else 3 if pct >= g["t3"] else 2 if pct >= g["t2"] else 1
        if f["ed_boarders"] and f["discharges_confirmed_before_11"] >= f["ed_boarders"]:
            lo = max(1, lo - 1)
        b = f["ed_boarders"]
        lb = 4 if b >= g["b4"] else 3 if b >= g["b3"] else 2 if b >= g["b2"] else 1
        h = f["longest_boarding_h"] or 0
        lw = 4 if h >= 24 else 3 if h >= g["h3"] else 1
        li = 4 if (f["icu_free"] == 0 and f["icu_boarders_in_ed"]) else 3 if f["icu_free"] == 0 else 1
        return max(lo, lb, lw, li)

    def intended(self, f):
        g = f["plan"]
        lvl = self._level(f)
        if lvl == 4:
            plan = "cancel_non_emergent"
        elif lvl == 3:
            plan = "defer_bed_dependent_electives"
        elif f["icu_free"] <= 1 and f["electives_needing_icu"] > 0:
            plan = "defer_icu_booked_cases"
        else:
            plan = "proceed_as_scheduled"
        h = f["longest_boarding_h"] or 0
        if h >= 24 or f["icu_boarders_in_ed"]:
            urg = 3
        elif h >= g["h3"]:
            urg = 2
        elif f["ed_boarders"] >= g["b2"]:
            urg = 1
        else:
            urg = 0
        b = f["blocked_discharges"]
        support = sum(b.values()) >= g["barrier_n"] or b["no_safe_destination"] > 0
        command = lvl >= 3 or (lvl == 2 and f["discharges_confirmed_before_11"] * 2 < f["ed_boarders"])
        return {"escalation_level": lvl - 1, "elective_surgery_plan": plan, "boarder_move_urgency": urg,
                "discharge_support_activation": "yes" if support else "no",
                "command_centre_opens": "yes" if command else "no"}

    def _plan(self, f, rng):
        g = f["plan"]
        return {
            "title": pick(rng, [f"{f['hospital']} capacity and surge plan, rev {g['rev']}",
                                f"Escalation triggers ({f['hospital']}, plan {g['rev']})"]),
            "triggers": f"The level is the highest any trigger reaches. Adult occupancy (occupied / staffed beds): "
                        f"Level 2 at {g['t2']}%, Level 3 at {g['t3']}%, Level 4 at 100% or more. ED boarders: Level "
                        f"2 at {g['b2']}, 3 at {g['b3']}, 4 at {g['b4']}. Longest boarder: Level 3 at {g['h3']} h, "
                        f"Level 4 at 24 h. ICU: Level 3 with no free bed, Level 4 with none free and an ICU patient "
                        f"boarding in the ED.",
            "discharge_credit": "If discharges confirmed for before 11:00 at least equal the ED boarders, the "
                                "occupancy trigger counts one level lower.",
            "electives": "Level 4: cancel all non-emergent surgery. Level 3: defer electives that need an inpatient "
                         "bed. Level 1-2 with at most one ICU bed free: defer only the ICU-booked cases.",
            "boarders": f"Move boarders at once if anyone has waited 24 h or a critical-care patient is boarding; "
                        f"within 2 h once the longest wait passes {g['h3']} h; by the late-morning round at "
                        f"{g['b2']}+ boarders.",
            "discharge_support": f"Activate the discharge-support team (transport vouchers, social work, interpreter "
                                 f"for discharge teaching, bridging medicines for patients without coverage) when "
                                 f"{g['barrier_n']}+ ready patients are blocked by such barriers or anyone lacks a "
                                 f"safe place to go. No one is discharged with a barrier unresolved.",
            "command_centre": "Open the capacity command centre at Level 3-4, or at Level 2 when discharges "
                              "confirmed for before 11:00 are fewer than half the ED boarders.",
        }

    def render(self, f, rng, surface):
        pl = self._plan(f, rng)
        b = f["blocked_discharges"]
        blocked = {"transport home": b["transport"], "no safe place to go": b["no_safe_destination"],
                   "waiting for an interpreter for discharge teaching": b["interpreter_for_teaching"],
                   "cannot afford discharge medicines (no coverage)": b["medicines_without_coverage"]}
        t = f["huddle_time"]
        if surface == "json":
            return dumps(prune({
                "huddle": {"hospital": f["hospital"], "time": _hm(t), "led_by": f["lead"],
                           "remark": f["manager_remark"]},
                "adult_beds_occupied_of_staffed": {u["unit"]: f"{u['occupied']} of {u['staffed']}" for u in f["units"]},
                "surge_beds_in_use": f["surge_in_use"] or None,
                "ed": {"admitted_patients_boarding": f["ed_boarders"],
                       "longest_boarding_hours": f["longest_boarding_h"]},
                "icu": {"unit": f["icu_unit"], "staffed_beds": f["icu_staffed"], "free_beds": f["icu_free"],
                        "icu_patients_boarding_in_ed": f["icu_boarders_in_ed"]},
                "discharges": {"confirmed_before_11": f["discharges_confirmed_before_11"],
                               "ready_but_blocked": {k: v for k, v in blocked.items() if v}},
                "theatres_today": {"electives_needing_inpatient_bed": f["electives_needing_bed"],
                                   "electives_booked_for_icu": f["electives_needing_icu"]},
                "plan": pl,
            }))
        if surface == "bed_board":
            rows = [[u["unit"], u["staffed"], u["occupied"], u["staffed"] - u["occupied"] if u["occupied"] <=
                     u["staffed"] else f"+{u['occupied'] - u['staffed']} surge"] for u in f["units"]]
            rows.append([f["icu_unit"], f["icu_staffed"], f["icu_staffed"] - f["icu_free"], f["icu_free"]])
            extra = kv_lines([
                ("ED boarders", f"{f['ed_boarders']}" + (f" (longest {f['longest_boarding_h']} h)"
                                                         if f["longest_boarding_h"] else "")),
                ("ICU patients waiting in ED", f["icu_boarders_in_ed"]),
                ("Discharges confirmed before 11:00", f["discharges_confirmed_before_11"]),
                ("Ready but blocked", ", ".join(f"{v} {k}" for k, v in blocked.items() if v) or "none"),
                ("Electives today", f"{f['electives_needing_bed']} need a bed, {f['electives_needing_icu']} booked "
                                    f"for ICU"),
                ("Note", f["manager_remark"])], rng)
            return (f"BED BOARD {f['hospital']} -- {fmt_date(t.date(), rng)} {_hm(t)}\n"
                    + table(["unit", "staffed", "occupied", "free"], rows, rng) + "\n\n" + extra
                    + f"\n\n{pl['title']}\n" + "\n".join(f"- {v}" for k, v in pl.items() if k != "title"))
        units = "; ".join(f"{u['unit']} {u['occupied']}/{u['staffed']}" for u in f["units"])
        para = [f"Huddle at {_hm(t)}, chaired by {f['lead']}.", f"Adult wards (occupied/staffed): {units}.",
                f"{f['ed_boarders']} admitted patients boarding in the ED"
                + (f", longest {f['longest_boarding_h']} h." if f["longest_boarding_h"] else "."),
                f"{f['icu_unit']}: {f['icu_free']} of {f['icu_staffed']} beds free"
                + (f", {f['icu_boarders_in_ed']} ICU patient(s) waiting in the ED." if f["icu_boarders_in_ed"]
                   else "."),
                f"{f['discharges_confirmed_before_11']} discharges confirmed for before 11:00.",
                "Ready but blocked: " + (", ".join(f"{v} {k}" for k, v in blocked.items() if v) or "none") + ".",
                f"Theatres: {f['electives_needing_bed']} electives need a bed, {f['electives_needing_icu']} booked "
                f"for ICU."]
        if f["manager_remark"]:
            para.append(_cap(f["manager_remark"]) + ".")
        return (f"HUDDLE MINUTES -- {f['hospital']}\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n"
                + pl["title"] + ":\n" + bullets([v for k, v in pl.items() if k != "title"], rng))

    def questions(self):
        return {
            "escalation_level": score("What escalation level does the plan set at this huddle?", [
                "Level 1: routine operations.",
                "Level 2: pressure building; heightened bed management.",
                "Level 3: major pressure; elective and flow restrictions apply.",
                "Level 4: critical; the hospital is over its staffed capacity or critical care is exhausted.",
            ]),
            "elective_surgery_plan": choice("What should happen to today's elective surgery?", {
                "cancel_non_emergent": "Cancel every non-emergent operation today.",
                "defer_bed_dependent_electives": "Defer the electives that need an inpatient bed; day cases go ahead.",
                "defer_icu_booked_cases": "Run the list but defer the cases booked for an ICU bed.",
                "proceed_as_scheduled": "Run the elective list as scheduled.",
            }),
            "boarder_move_urgency": score("How soon must admitted patients boarding in the ED be moved to beds?", [
                "As beds turn over through the day.",
                "By the late-morning discharge round.",
                "Within the next two hours.",
                "At once: someone has boarded 24 hours or a critical-care patient is waiting in the ED.",
            ]),
            "discharge_support_activation": noul(
                "The plan calls for activating the discharge-support team at this huddle.",
                yes="Enough ready patients are blocked by transport, interpreter, medicine-cost or housing barriers, "
                    "or someone has no safe place to go.",
                no="Blocked discharges are below the plan's threshold and everyone has a safe destination."),
            "command_centre_opens": noul("The capacity command centre must open after this huddle."),
        }


# =========================================================================== schema 2: ED diversion request
class EdDiversionRequest(Schema):
    name = "ed_diversion_request"
    description = ("The ED charge nurse asks the administrator on call for ambulance diversion: decide against the "
                   "regional diversion agreement and the hospital's own first steps.")
    surfaces = {"json": 0.55, "request_page": 0.225, "admin_log": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"RDA-{rng.randint(10, 40)}", "board_trigger": pick(rng, [8, 10, 12]),
             "max_hours": pick(rng, [4, 6, 8]), "wait_points": pick(rng, [15, 20, 25]),
             "longest_points_h": pick(rng, [3, 4, 6])}
        bays = rng.randint(18, 44)
        full = chance(rng, 0.42)
        occupied_bays = bays if full else bays - pick_w(rng, {1: 2.5, 2: 3, 3: 2.5, 5: 2, 8: 1.5})
        boarders = (near(rng, g["board_trigger"], 2, lo=0, integer=True) if reg == "borderline"
                    else pick_w(rng, {rng.randint(2, g["board_trigger"] - 1): 1, rng.randint(g["board_trigger"], 24): 1.4}))
        ct_down = chance(rng, 0.14)
        waiting = rng.randint(4, 40)
        longest_wait_h = round(rng.uniform(0.5, 9), 1)
        steps = {"inpatient_pull": chance(rng, 0.7), "surge_beds_open": chance(rng, 0.65),
                 "discharge_lounge_open": chance(rng, 0.75)}
        if reg == "clear" and chance(rng, 0.5):
            steps = {k: True for k in steps}
        neighbours = pick_w(rng, {0: 4, 1: 3, 2: 1.5, 3: 0.6})
        used = pick_w(rng, {0: 4, 1: 2, 2: 2, 3: 1.5, 4: 1.2, 6: 0.8})
        block = pick(rng, [1, 2])
        if reg == "borderline" and chance(rng, 0.4):
            used = max(0, g["max_hours"] - block + pick(rng, [-1, 0, 0, 1]))
        arrivals = rng.randint(1, 6)
        pressure = None
        if reg == "conflicting":
            pressure = pick(rng, ["ED director: 'we have to go on diversion, staff are exhausted'",
                                  "charge nurse: 'the neighbours are coping, they won't mind'",
                                  "administrator: 'we never divert, find another way'"])
        t = rand_ts(rng)
        return {"regime": reg, "agreement": g, "hospital": _hospital(rng), "time": t, "bays": bays,
                "bays_occupied": occupied_bays, "boarders": boarders, "ct_down": ct_down, "waiting_room": waiting,
                "longest_wait_h": longest_wait_h, "internal_steps": steps, "neighbours_diverting": neighbours,
                "diversion_hours_last_24h": used, "requested_block_h": block, "ambulances_next_hour": arrivals,
                "pressure_remark": pressure, "charge_nurse": person(rng), "admin_on_call": person(rng)}

    def intended(self, f):
        g = f["agreement"]
        full = f["bays_occupied"] >= f["bays"]
        trigger = (full and f["boarders"] >= g["board_trigger"]) or f["ct_down"]
        regional = f["neighbours_diverting"] < 2 and f["diversion_hours_last_24h"] + f["requested_block_h"] <= \
            g["max_hours"]
        internal = all(f["internal_steps"].values())
        if not trigger:
            dec = "deny_trigger_not_met"
        elif not regional:
            dec = "deny_regional_limit"
        elif not internal:
            dec = "deny_internal_steps_first"
        else:
            dec = "approve_time_limited_diversion"
        pts = (2 if full else 0) + (2 if f["boarders"] >= g["board_trigger"] else 0) + \
              (1 if f["waiting_room"] >= g["wait_points"] else 0) + (1 if f["longest_wait_h"] > g["longest_points_h"]
                                                                     else 0)
        crowd = 0 if pts <= 1 else 1 if pts <= 3 else 2 if pts <= 5 else 3
        free = max(0, f["bays"] - f["bays_occupied"])
        if free == 0:
            sat = 3
        else:
            hrs = free / f["ambulances_next_hour"]
            sat = 2 if hrs <= 1 else 1 if hrs <= 2 else 0
        return {"diversion_decision": dec, "crowding_index_band": crowd, "time_to_no_monitored_space": sat,
                "regional_terms_allow": "yes" if regional else "no",
                "internal_steps_complete": "yes" if internal else "no"}

    def _agreement(self, f, rng):
        g = f["agreement"]
        return {
            "title": pick(rng, [f"Regional diversion agreement {g['rev']}", f"Ambulance diversion terms ({g['rev']}), "
                                                                            f"with local first steps"]),
            "trigger": f"Diversion may be requested only when every monitored bay is full and {g['board_trigger']}+ "
                       f"admitted patients are boarding, or when the CT scanner is down.",
            "first_steps": "Before any request: inpatient units pull their boarders, surge beds open and the "
                           "discharge lounge opens.",
            "regional_limits": f"Refused while two or more neighbouring hospitals are diverting, or if the request "
                               f"would take this hospital past {g['max_hours']} diversion hours in 24 h.",
            "exemptions": "Cardiac arrest, STEMI, stroke and major trauma are never diverted.",
            "crowding_index": f"Crowding index: 2 points if all monitored bays are full, 2 if boarders reach "
                              f"{g['board_trigger']}, 1 if {g['wait_points']}+ are in the waiting room, 1 if the "
                              f"longest wait exceeds {g['longest_points_h']} h. 0-1 low, 2-3 moderate, 4-5 high, "
                              f"6 severe.",
        }

    def render(self, f, rng, surface):
        ag = self._agreement(f, rng)
        st = f["internal_steps"]
        steps = {"inpatient units pulling boarders": st["inpatient_pull"], "surge beds open": st["surge_beds_open"],
                 "discharge lounge open": st["discharge_lounge_open"]}
        t = f["time"]
        if surface == "json":
            return dumps(prune({
                "request": {"from": f["charge_nurse"], "to": f["admin_on_call"], "time": _hm(t),
                            "hours_requested": f["requested_block_h"], "comment": f["pressure_remark"]},
                "ed_now": {"monitored_bays": f["bays"], "bays_occupied": f["bays_occupied"],
                           "admitted_boarders": f["boarders"], "waiting_room": f["waiting_room"],
                           "longest_wait_hours": f["longest_wait_h"], "ct_scanner": "down" if f["ct_down"] else "working",
                           "ambulances_expected_next_hour": f["ambulances_next_hour"]},
                "internal_steps": {k.replace(" ", "_"): v for k, v in steps.items()},
                "region": {"neighbours_on_diversion": f["neighbours_diverting"],
                           "our_diversion_hours_last_24h": f["diversion_hours_last_24h"]},
                "agreement": ag,
            }))
        if surface == "request_page":
            body = noisy(join_sentences([
                f"Requesting {f['requested_block_h']} h of ambulance diversion.",
                f"All {f['bays']} monitored bays " + ("are full" if f["bays_occupied"] >= f["bays"] else
                                                     f"but {f['bays'] - f['bays_occupied']} are in use or free") + ".",
                f"{f['boarders']} admitted patients are boarding; {f['waiting_room']} in the waiting room, longest "
                f"wait {f['longest_wait_h']} h.", "CT is down." if f["ct_down"] else "",
                f"Expecting about {f['ambulances_next_hour']} ambulances in the next hour.",
                "Done so far: " + (", ".join(k for k, v in steps.items() if v) or "nothing yet") + ".",
                "Not done: " + (", ".join(k for k, v in steps.items() if not v) or "none") + ".",
                f["pressure_remark"] or ""]), rng, 0.08)
            body = body.replace("but 0 are in use or free", "with none free")
            return (f"PAGE {_hm(t)} to {f['admin_on_call']} from {f['charge_nurse']} (ED charge)\n{body}\n\n"
                    f"Regional status: {f['neighbours_diverting']} neighbour(s) on diversion; we have used "
                    f"{f['diversion_hours_last_24h']} h in the last 24 h.\n\n{ag['title']}\n"
                    + "\n".join(f"- {v}" for k, v in ag.items() if k != "title"))
        log = [(t, f"diversion request from {f['charge_nurse']}: {f['requested_block_h']} h"),
               (t + timedelta(minutes=3), f"ED: {f['bays_occupied']}/{f['bays']} monitored bays occupied, "
                                          f"{f['boarders']} boarders, {f['waiting_room']} waiting (longest "
                                          f"{f['longest_wait_h']} h), CT {'down' if f['ct_down'] else 'up'}"),
               (t + timedelta(minutes=5), f"~{f['ambulances_next_hour']} ambulances due next hour"),
               (t + timedelta(minutes=7), "internal steps: " + "; ".join(f"{k} {'yes' if v else 'no'}"
                                                                         for k, v in steps.items())),
               (t + timedelta(minutes=9), f"region: {f['neighbours_diverting']} diverting; our hours last 24 h "
                                          f"{f['diversion_hours_last_24h']}")]
        if f["pressure_remark"]:
            log.append((t + timedelta(minutes=11), f["pressure_remark"]))
        return (f"ADMINISTRATOR ON CALL LOG -- {f['hospital']}\n" + "\n".join(f"{_hm(a)}  {b}" for a, b in log)
                + f"\n\n{ag['title']}\n" + bullets([v for k, v in ag.items() if k != "title"], rng))

    def questions(self):
        return {
            "diversion_decision": choice("How should the administrator answer the diversion request?", {
                "approve_time_limited_diversion": "Approve diversion for the requested block, exemptions kept.",
                "deny_internal_steps_first": "Refuse for now; the hospital's own first steps are not all in place.",
                "deny_regional_limit": "Refuse; the regional agreement's limits rule diversion out.",
                "deny_trigger_not_met": "Refuse; the department does not meet the agreement's trigger.",
            }),
            "crowding_index_band": score("Where does the department fall on the agreement's crowding index?", [
                "Low.", "Moderate.", "High.", "Severe.",
            ]),
            "time_to_no_monitored_space": score(
                "How soon will the department have no monitored bay for a new emergency?", [
                    "Not within the next two hours at the expected arrival rate.",
                    "Within about two hours.",
                    "Within the hour.",
                    "Already: every monitored bay is occupied.",
                ]),
            "regional_terms_allow": noul(
                "The regional terms (neighbours already diverting, hours used) leave room for this request.",
                yes="Fewer than two neighbours are diverting and the requested block keeps the hospital within its "
                    "24-hour diversion allowance.",
                no="Two or more neighbours are diverting, or the block would exceed the allowance."),
            "internal_steps_complete": noul("Every first step the agreement requires was in place before the request."),
        }


# =========================================================================== schema 3: ICU capacity huddle
PENDING_KINDS = {
    "ed_boarder": ("patient in the ED needing ICU now", 0),
    "ward_deterioration": ("deteriorating ward patient, rapid-response team asking for ICU now", 0),
    "post_op_elective": ("elective {proc} post-op, list starts {t}", None),
    "post_op_urgent": ("urgent {proc} post-op (cannot wait), expected {t}", None),
    "transfer_in": ("accepted transfer from {site}, arriving about {t}", None),
}
PROCS = ["oesophagectomy", "cardiac bypass", "aortic valve replacement", "major liver resection",
         "Whipple procedure", "craniotomy", "lung lobectomy", "abdominal aortic aneurysm repair"]


class IcuCapacityHuddle(Schema):
    name = "icu_capacity_huddle"
    description = ("An ICU near or at capacity with patients waiting for beds: which step the plan takes first, who "
                   "moves out to step-down, and how tight things are.")
    surfaces = {"json": 0.55, "icu_board": 0.225, "intensivist_handover": 0.225}

    LABELS = ["A", "B", "C", "D"]

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"CC-{rng.randint(1, 9)}.{rng.randint(0, 9)}", "pressor_free_h": pick(rng, [6, 12]),
             "fio2_max": pick(rng, [40, 50]), "surge_ratio": 2}
        now = rand_ts(rng, hours=(7, 10)).replace(second=0)
        staffed = rng.randint(10, 26)
        free = pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1})
        n_pending = max(0, free + pick_w(rng, {-2: 0.9, -1: 0.8, 0: 1.2, 1: 2.5, 2: 1.8, 3: 0.7}))
        pending = []
        for _ in range(n_pending):
            k = pick_w(rng, {"ed_boarder": 1.2, "ward_deterioration": 0.6, "post_op_elective": 2.5, "post_op_urgent": 1,
                             "transfer_in": 1.5})
            due = 0 if PENDING_KINDS[k][1] == 0 else pick(rng, [1, 2, 3, 4, 6, 8, 10])
            pending.append({"kind": k, "due_h": due, "proc": pick(rng, PROCS),
                            "site": f"{pick(rng, HOSP_STEMS)} {pick(rng, ['Community Hospital', 'Regional Hospital'])}"})
        # four step-down candidates; at least one meets every written criterion
        cands = []
        for lab in self.LABELS:
            pf = pick(rng, [None, 2, 5, 8, 11, 14, 20, 30])        # hours since last vasopressor (None: still on)
            fio2 = pick(rng, [30, 35, 40, 45, 50, 60])
            titr = chance(rng, 0.3)
            if reg == "borderline":
                pf = pick(rng, [g["pressor_free_h"] - 1, g["pressor_free_h"], g["pressor_free_h"] + 2])
                fio2 = pick(rng, [g["fio2_max"] - 5, g["fio2_max"], g["fio2_max"] + 5])
            cands.append({"label": lab, "bed": f"Bed {rng.randint(1, staffed)}", "pressor_free_h": pf,
                          "fio2": fio2, "titrated_infusion": titr, "ready_h": round(rng.uniform(0.5, 18), 1)})
        if not any(self._eligible(g, c) for c in cands):
            c = pick(rng, cands)
            c["pressor_free_h"], c["fio2"], c["titrated_infusion"] = g["pressor_free_h"] + 4, g["fio2_max"] - 10, False
        stepdown_free = pick_w(rng, {0: 4, 1: 3, 2: 1})
        surge_beds = pick_w(rng, {0: 2, 1: 2, 2: 2, 4: 1})
        surge_nurses = pick_w(rng, {0: 3, 1: 3, 2: 1.5})
        intensivist = chance(rng, 0.8)
        remark = None
        if reg == "conflicting":
            opts = ["charge nurse: 'we can squeeze one more in without opening anything'",
                    "bed manager: 'step-down says they're full but someone is going home later'",
                    "night registrar: 'patient C still looks a bit wobbly to me, I'd keep them'"]
            if any(p["kind"] == "post_op_elective" for p in pending):
                opts.append("surgeon: 'my case is quick, the patient won't need ICU for long'")
            remark = pick(rng, opts)
        return {"regime": reg, "plan": g, "hospital": _hospital(rng), "icu_unit": pick(rng, ICUS), "now": now,
                "staffed": staffed, "free": free, "pending": pending, "candidates": cands,
                "stepdown_unit": pick(rng, STEPDOWN), "stepdown_free": stepdown_free, "surge_beds": surge_beds,
                "surge_nurses": surge_nurses, "intensivist_for_surge": intensivist, "remark": remark,
                "intensivist": "Dr. " + person(rng)}

    @staticmethod
    def _eligible(g, c):
        return (c["pressor_free_h"] is not None and c["pressor_free_h"] >= g["pressor_free_h"]
                and c["fio2"] <= g["fio2_max"] and not c["titrated_infusion"])

    def intended(self, f):
        g = f["plan"]
        need = len(f["pending"])
        deficit = need - f["free"]
        elig = [c for c in f["candidates"] if self._eligible(g, c)]
        first = max(elig, key=lambda c: (c["ready_h"], c["label"]))
        surge_ok = f["surge_beds"] >= 1 and f["surge_nurses"] >= 1 and f["intensivist_for_surge"]
        if deficit <= 0:
            act = "no_action_needed"
        elif f["stepdown_free"] > 0:
            act = "move_ready_patient_to_stepdown"
        elif surge_ok:
            act = "open_surge_beds"
        elif any(p["kind"] == "post_op_elective" for p in f["pending"]):
            act = "defer_icu_dependent_surgery"
        else:
            act = "request_partner_icu_transfer"
        margin = f["free"] - need
        strain = 0 if margin >= 2 else 1 if margin >= 0 else 2 if margin == -1 else 3
        dues = [p["due_h"] for p in f["pending"]]
        soonest = min(dues) if dues else None
        when = 0 if soonest is None or soonest > 6 else 1 if soonest > 2 else 2 if soonest > 0 else 3
        return {"first_capacity_step": act, "stepdown_candidate": "candidate_" + first["label"].lower(),
                "icu_strain": strain, "bed_needed_by": when,
                "surge_beds_can_open": "yes" if surge_ok else "no"}

    def _plan(self, f, rng):
        g = f["plan"]
        return {
            "title": pick(rng, [f"Critical care capacity procedure {g['rev']}", f"{f['hospital']} ICU flow rules "
                                                                                  f"({g['rev']})"]),
            "order": "When pending ICU admissions outnumber free beds, take the first step that is possible: move a "
                     "ready patient to step-down (if a step-down bed is free); open surge beds; defer elective cases "
                     "that need ICU; ask a partner ICU to take a patient.",
            "stepdown_ready": f"Ready for step-down: off vasopressors for {g['pressor_free_h']} h or more, FiO2 "
                              f"{g['fio2_max']}% or less, no titrated infusion. Among ready patients, the one ready "
                              f"longest moves first.",
            "surge": "Surge beds open only with a physical surge space, at least one extra ICU nurse (one per two "
                     "surge patients) and an intensivist covering them.",
        }

    @staticmethod
    def _pend_text(p, now):
        txt, _ = PENDING_KINDS[p["kind"]]
        t = _hm(now + timedelta(hours=p["due_h"]))
        return txt.format(proc=p["proc"], t=t, site=p["site"])

    def render(self, f, rng, surface):
        pl = self._plan(f, rng)
        now = f["now"]

        def cand(c):
            pf = "still on vasopressor" if c["pressor_free_h"] is None else f"off vasopressors {c['pressor_free_h']} h"
            return pf, f"FiO2 {c['fio2']}%", "titrated drip running" if c["titrated_infusion"] else "no titrated drip"
        if surface == "json":
            return dumps(prune({
                "icu": {"hospital": f["hospital"], "unit": f["icu_unit"], "time": _hm(now), "staffed_beds": f["staffed"],
                        "free_beds": f["free"], "comment": f["remark"]},
                "waiting_for_icu": [self._pend_text(p, now) for p in f["pending"]] or None,
                "stepdown_candidates": [{"patient": c["label"], "bed": c["bed"], "status": ", ".join(cand(c)),
                                         "judged_ready_hours_ago": c["ready_h"]} for c in f["candidates"]],
                "step_down": {"unit": f["stepdown_unit"], "free_beds": f["stepdown_free"]},
                "surge": {"physical_surge_beds": f["surge_beds"], "extra_icu_nurses_available": f["surge_nurses"],
                          "intensivist_cover_for_surge": f["intensivist_for_surge"]},
                "procedure": pl,
            }))
        if surface == "icu_board":
            rows = [[c["label"], c["bed"], *cand(c), f"{c['ready_h']} h"] for c in f["candidates"]]
            head = (f"{f['icu_unit']} -- {f['hospital']} -- {_hm(now)}\nStaffed {f['staffed']}, free {f['free']}. "
                    f"{f['stepdown_unit']}: {f['stepdown_free']} free.\nSurge: {f['surge_beds']} space(s), "
                    f"{f['surge_nurses']} extra nurse(s), intensivist cover "
                    f"{'yes' if f['intensivist_for_surge'] else 'no'}.")
            pend = "\n".join(f"  - {self._pend_text(p, now)}" for p in f["pending"]) or "  (none)"
            return (f"{head}\n\nWaiting for ICU:\n{pend}\n\nStep-down candidates:\n"
                    + table(["pt", "bed", "pressors", "oxygen", "drips", "ready for"], rows, rng)
                    + (f"\n\n{f['remark']}" if f["remark"] else "") + f"\n\n{pl['title']}\n"
                    + "\n".join(f"- {v}" for k, v in pl.items() if k != "title"))
        para = [f"{_hm(now)} handover from {f['intensivist']}, {f['icu_unit']}: {f['free']} of {f['staffed']} beds "
                f"free."]
        if f["pending"]:
            para.append("Waiting: " + "; ".join(self._pend_text(p, now) for p in f["pending"]) + ".")
        else:
            para.append("Nobody waiting for a bed right now.")
        for c in f["candidates"]:
            pf, ox, dr = cand(c)
            para.append(f"Patient {c['label']} ({c['bed']}): {pf}, {ox}, {dr}, judged ready {c['ready_h']} h ago.")
        para.append(f"{f['stepdown_unit']} has {f['stepdown_free']} free bed(s). Surge: {f['surge_beds']} space(s), "
                    f"{f['surge_nurses']} extra nurse(s), intensivist cover "
                    f"{'available' if f['intensivist_for_surge'] else 'not available'}.")
        if f["remark"]:
            para.append(_cap(f["remark"]) + ".")
        return ("ICU HANDOVER NOTE\n" + noisy(join_sentences(para), rng, 0.08) + f"\n\n{pl['title']}:\n"
                + bullets([v for k, v in pl.items() if k != "title"], rng))

    def questions(self):
        return {
            "first_capacity_step": choice("What should the ICU do first about its capacity?", {
                "defer_icu_dependent_surgery": "Postpone the elective operation that would need an ICU bed.",
                "move_ready_patient_to_stepdown": "Move a patient who meets the step-down criteria out to step-down.",
                "no_action_needed": "Nothing yet; free beds cover everyone waiting.",
                "open_surge_beds": "Open surge ICU beds with the extra staff.",
                "request_partner_icu_transfer": "Ask a partner hospital's ICU to take a patient.",
            }),
            "stepdown_candidate": choice("Which candidate should be the first to move to step-down?", {
                "candidate_a": "Patient A on the candidate list.",
                "candidate_b": "Patient B on the candidate list.",
                "candidate_c": "Patient C on the candidate list.",
                "candidate_d": "Patient D on the candidate list.",
            }),
            "icu_strain": score("How tight is ICU capacity against the patients waiting?", [
                "Comfortable: free beds exceed those waiting by two or more.",
                "Tight: free beds just cover those waiting.",
                "Short by one bed.",
                "Short by two or more beds.",
            ]),
            "bed_needed_by": score("How soon is the next ICU bed needed?", [
                "Not for at least six hours, or no one is waiting.",
                "Within six hours.",
                "Within two hours.",
                "Now: a patient needing ICU is already waiting in the ED or on a ward.",
            ]),
            "surge_beds_can_open": noul(
                "Surge ICU beds could be opened today under the procedure.",
                yes="A physical surge space, at least one extra ICU nurse and intensivist cover are all available.",
                no="Space, an extra nurse or intensivist cover is missing."),
        }


# =========================================================================== schema 4: nurse staffing
UNIT_TYPES = {"medsurg": "med/surg", "tele": "telemetry", "stepdown": "step-down", "icu": "ICU"}


class StaffingBedClosure(Schema):
    name = "staffing_bed_closure"
    description = ("The staffing office before a shift change: are units within their written nurse ratios, and what "
                   "should cover the gap for the coming shift?")
    surfaces = {"json": 0.55, "staffing_grid": 0.225, "staffing_email": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"SP-{rng.randint(2, 9)}", "ratio": {"medsurg": pick(rng, [5, 6]), "tele": 4, "stepdown": 3,
                                                          "icu": 2},
             "holdover_cap_h": pick(rng, [14, 16]), "incentive_lead_h": 4}
        kinds = ["medsurg", "medsurg", "tele", "stepdown", "icu"]
        n = rng.randint(3, 5)
        picks = rng.sample(kinds, n)
        units = []
        for k in picks:
            name = pick(rng, MEDSURG if k == "medsurg" else TELE if k == "tele" else STEPDOWN if k == "stepdown"
                        else ICUS)
            r = g["ratio"][k]
            census_now = rng.randint(r * 2, r * 7)
            nurses_now = math.ceil(census_now / r) + pick_w(rng, {0: 5, 1: 2, -1: 1.6 if reg != "clear" else 0.6})
            adm = rng.randint(0, 2 if k == "icu" else 5)
            dc = rng.randint(0, max(1, census_now // (4 if k == "icu" else 3)))
            census_next = max(r, census_now + adm - dc)
            sched = max(1, math.ceil(census_next / r) + pick_w(rng, {0: 4, 1: 2.5, -1: 1}))
            callouts = min(sched - 1, pick_w(rng, {0: 5, 1: 2.5, 2: 0.8}))
            units.append({"unit": name, "kind": k, "census_now": census_now, "nurses_now": max(1, nurses_now),
                          "admits": adm, "discharges": dc, "scheduled_next": max(1, sched), "callouts": callouts})
        floats = pick_w(rng, {0: 3, 1: 3, 2: 2, 3: 1})
        agency = pick_w(rng, {0: 4, 1: 2, 2: 1})
        now = rand_ts(rng)
        shift = now.replace(hour=19 if now.hour < 19 and now.hour >= 7 else 7, minute=0, second=0)
        if shift <= now:
            shift += timedelta(days=1)
        hours_to = round((shift - now).total_seconds() / 3600, 1)
        if reg == "borderline":
            hours_to = round(near(rng, pick(rng, [2, 4, 6]), 0.6, lo=0.3), 1)
            now = shift - timedelta(hours=hours_to)
        worked = round(near(rng, g["holdover_cap_h"] - 4, 1.5) * 2) / 2 if reg != "clear" else pick(rng, [8, 9, 12.5])
        willing = pick_w(rng, {"agreed": 5, "not asked yet": 2, "declined": 1.5})
        remark = None
        if reg == "conflicting":
            remark = pick(rng, ["unit manager: 'we always manage short, no need to close beds'",
                                "director: 'just mandate the night staff to stay'",
                                "charge nurse: 'two of our patients are going home, count them out already'"])
        return {"regime": reg, "policy": g, "hospital": _hospital(rng), "units": units, "float_pool": floats,
                "agency_confirmed": agency, "now": now, "shift_start": shift, "hours_to_shift": hours_to,
                "holdover_nurse": person(rng), "holdover_worked_h": worked, "holdover_willing": willing,
                "holdover_hours_proposed": 4, "remark": remark, "coordinator": person(rng)}

    @staticmethod
    def _gap(f, when):
        g = f["policy"]
        gap = 0
        for u in f["units"]:
            r = g["ratio"][u["kind"]]
            if when == "now":
                gap += max(0, math.ceil(u["census_now"] / r) - u["nurses_now"])
            else:
                census = u["census_now"] + u["admits"] - u["discharges"]
                gap += max(0, math.ceil(max(r, census) / r) - (u["scheduled_next"] - u["callouts"]))
        return gap

    def intended(self, f):
        g = f["policy"]
        gap = self._gap(f, "next")
        if gap == 0:
            resp = "no_change_needed"
        elif f["float_pool"] + f["agency_confirmed"] >= gap:
            resp = "deploy_float_and_agency"
        elif f["hours_to_shift"] >= g["incentive_lead_h"]:
            resp = "offer_incentive_shifts"
        else:
            resp = "close_beds_to_admissions"
        size = 0 if gap == 0 else 1 if gap == 1 else 2 if gap <= 3 else 3
        h = f["hours_to_shift"]
        press = 0 if gap == 0 else 1 if h >= 6 else 2 if h >= 2 else 3
        ok_hold = (f["holdover_willing"] == "agreed"
                   and f["holdover_worked_h"] + f["holdover_hours_proposed"] <= g["holdover_cap_h"])
        return {"staffing_response": resp, "next_shift_shortfall": size, "fill_deadline_pressure": press,
                "ratio_breach_now": "yes" if self._gap(f, "now") > 0 else "no",
                "holdover_within_policy": "yes" if ok_hold else "no"}

    def _policy(self, f, rng):
        g, r = f["policy"], f["policy"]["ratio"]
        return {
            "title": pick(rng, [f"Nurse staffing policy {g['rev']}", f"{f['hospital']} safe staffing standard "
                                                                       f"({g['rev']})"]),
            "ratios": f"One registered nurse per {r['medsurg']} med/surg, {r['tele']} telemetry, {r['stepdown']} "
                      f"step-down and {r['icu']} ICU patients; round up.",
            "next_shift": "Next-shift need uses current census plus expected admissions minus confirmed discharges; "
                          "available nurses are those scheduled minus call-outs.",
            "cover_order": f"Cover a gap first with float pool and confirmed agency nurses; if they cannot cover it, "
                           f"offer incentive shifts when {g['incentive_lead_h']} h or more remain before the shift; "
                           f"otherwise close beds to new admissions on the short unit.",
            "holdover": f"Holding a nurse over is voluntary only, and never past {g['holdover_cap_h']} h of "
                        f"continuous work. Mandated overtime is not used.",
        }

    def render(self, f, rng, surface):
        pl = self._policy(f, rng)
        hold = (f"Proposal: hold over {f['holdover_nurse']} ({f['holdover_worked_h']} h worked so far) for "
                f"{f['holdover_hours_proposed']} h; nurse has {f['holdover_willing']}.")
        if surface == "json":
            return dumps(prune({
                "now": {"hospital": f["hospital"], "time": _hm(f["now"]), "next_shift_starts": _hm(f["shift_start"]),
                        "coordinator": f["coordinator"], "comment": f["remark"]},
                "units": [{"unit": u["unit"], "type": UNIT_TYPES[u["kind"]], "patients_now": u["census_now"],
                           "nurses_now": u["nurses_now"], "expected_admissions": u["admits"],
                           "confirmed_discharges": u["discharges"], "nurses_scheduled_next": u["scheduled_next"],
                           "call_outs_next": u["callouts"]} for u in f["units"]],
                "cover_available": {"float_pool_nurses": f["float_pool"], "agency_confirmed": f["agency_confirmed"]},
                "holdover_proposal": hold,
                "policy": pl,
            }))
        if surface == "staffing_grid":
            rows = [[u["unit"], UNIT_TYPES[u["kind"]], u["census_now"], u["nurses_now"], f"+{u['admits']}/-"
                     f"{u['discharges']}", u["scheduled_next"], u["callouts"]] for u in f["units"]]
            return (f"STAFFING GRID {f['hospital']} -- {_hm(f['now'])} (next shift {_hm(f['shift_start'])})\n"
                    + table(["unit", "type", "pts now", "RNs now", "adm/dc", "RNs sched", "call-outs"], rows, rng)
                    + f"\nFloat pool: {f['float_pool']}   Agency confirmed: {f['agency_confirmed']}\n{hold}"
                    + (f"\n{f['remark']}" if f["remark"] else "") + f"\n\n{pl['title']}\n"
                    + "\n".join(f"- {v}" for k, v in pl.items() if k != "title"))
        lines = [f"Subject: Staffing for the {_hm(f['shift_start'])} shift", f"From: {f['coordinator']} (staffing office)",
                 "", noisy(join_sentences(
                     [f"It is {_hm(f['now'])}."] +
                     [f"{u['unit']} ({UNIT_TYPES[u['kind']]}): {u['census_now']} patients with {u['nurses_now']} RNs "
                      f"now; {u['admits']} admissions expected and {u['discharges']} confirmed discharges; "
                      f"{u['scheduled_next']} RNs scheduled next with {u['callouts']} call-out(s)." for u in f["units"]]
                     + [f"Float pool has {f['float_pool']} nurse(s); agency confirmed {f['agency_confirmed']}.", hold,
                        f["remark"] or ""]), rng, 0.08), "", pl["title"] + ":",
                 bullets([v for k, v in pl.items() if k != "title"], rng)]
        return "\n".join(lines)

    def questions(self):
        return {
            "staffing_response": choice("How should the staffing office cover the coming shift?", {
                "close_beds_to_admissions": "Close beds to new admissions on the short unit.",
                "deploy_float_and_agency": "Send float-pool and confirmed agency nurses to the short units.",
                "no_change_needed": "Nothing; every unit is covered.",
                "offer_incentive_shifts": "Offer incentive shifts to fill the gap before the shift starts.",
            }),
            "next_shift_shortfall": score("How many nurses short is the coming shift across these units?", [
                "None.", "One nurse.", "Two or three nurses.", "Four or more nurses.",
            ]),
            "fill_deadline_pressure": score("How pressed is the office for time to fill the gap?", [
                "No gap to fill.",
                "A gap, with six hours or more before the shift.",
                "A gap, with two to six hours left.",
                "A gap, with under two hours left.",
            ]),
            "ratio_breach_now": noul(
                "At least one unit is outside its written nurse ratio right now.",
                yes="A unit's current patients need more nurses, at the written ratio rounded up, than it has now.",
                no="Every unit's current nurses meet the written ratio for its current patients."),
            "holdover_within_policy": noul("The proposed holdover of a nurse is allowed under the policy."),
        }


# =========================================================================== schema 5: transfer acceptance
SERVICES = {
    "cath_lab": ("primary angioplasty for a heart attack", True),
    "neurosurgery": ("neurosurgical evacuation of a brain bleed", True),
    "thrombectomy": ("stroke clot retrieval", True),
    "burns": ("major burns care", True),
    "vascular": ("vascular surgery for a leaking aneurysm", True),
    "high_risk_obstetrics": ("high-risk obstetric care, 31 weeks pregnant with severe pre-eclampsia", True),
    "dialysis_ward": ("inpatient dialysis and renal ward care", False),
    "general_medicine": ("general medical admission for pneumonia", False),
    "oncology": ("inpatient oncology care", False),
    "cardiology_ward": ("cardiology ward care for heart failure", False),
    "hip_fracture": ("orthopaedic surgery for a hip fracture", False),
    "inpatient_rehab": ("inpatient stroke rehabilitation", False),
}
BED_NEEDS = {"icu": ["ventilated", "on a vasopressor infusion", "intubated and sedated"],
             "step_down": ["on BiPAP", "on a titrated nitroglycerin infusion", "needing hourly neuro checks"],
             "telemetry": ["needing cardiac monitoring", "with intermittent arrhythmia"],
             "general_ward": ["stable on room air", "stable on oxygen 2 L/min"]}
SERVICE_NEEDS = {"cath_lab": ["icu", "step_down", "telemetry"], "neurosurgery": ["icu"],
                 "thrombectomy": ["icu", "step_down"], "burns": ["icu", "step_down"], "vascular": ["icu"],
                 "high_risk_obstetrics": ["step_down", "icu"], "dialysis_ward": ["general_ward", "telemetry"],
                 "general_medicine": ["general_ward", "telemetry", "step_down"], "oncology": ["general_ward", "step_down"],
                 "cardiology_ward": ["telemetry", "step_down", "general_ward"], "hip_fracture": ["general_ward",
                                                                                                "telemetry"],
                 "inpatient_rehab": ["general_ward"]}
PAYERS_T = ["commercial plan", "public plan", "older-adult public plan", "uninsured",
            "emergency coverage only (immigration-related eligibility)"]


class TransferCenterAcceptance(Schema):
    name = "transfer_center_acceptance"
    description = ("An outside hospital asks the transfer center to take a patient: accept, accept pending a bed, "
                   "redirect, or decline, on capability and capacity under the written acceptance policy.")
    surfaces = {"json": 0.55, "transfer_call_log": 0.225, "acceptance_form": 0.225}

    def sample_factors(self, rng):
        reg = regime(rng)
        g = {"rev": f"TC-{rng.randint(1, 9)}{pick(rng, ['', 'a', 'b'])}", "pending_window_h": pick(rng, [2, 4, 6])}
        svc = pick(rng, list(SERVICES))
        offered = chance(rng, 0.78)
        on_call = offered and chance(rng, 0.82)
        need = pick(rng, SERVICE_NEEDS[svc])
        free_now = pick_w(rng, {0: 5, 1: 2.5, 2: 1.5, 4: 1})
        next_bed_h = None if free_now else pick_w(rng, {1: 1.5, 3: 1.5, 5: 1.5, 8: 2, None: 2.5})
        if reg == "borderline" and not free_now:
            next_bed_h = max(1, near(rng, g["pending_window_h"], 1.5, integer=True))
        level = pick_w(rng, {1: 2, 2: 3, 3: 3, 4: 2})
        network = pick_w(rng, {None: 2, "capability and a free bed": 4, "capability, no free bed": 1})
        time_critical = SERVICES[svc][1] and need in ("icu", "step_down")
        payer = pick(rng, PAYERS_T)
        language = "English" if chance(rng, 0.78) else pick(rng, ["Spanish", "Vietnamese", "Arabic", "Haitian Creole",
                                                                 "Somali", "Mandarin", "Polish", "Dari"])
        finance = None
        if reg == "conflicting" or chance(rng, 0.1):
            finance = pick(rng, ["finance office asks for insurance verification before accepting",
                                 "admin note: 'uninsured, suggest they try the county hospital first'",
                                 "sending clerk asks whether we take patients with emergency-only coverage"])
        age = rng.randint(19, 88) if svc != "high_risk_obstetrics" else rng.randint(17, 44)
        sex = "F" if svc == "high_risk_obstetrics" else pick(rng, ["F", "M"])
        return {"regime": reg, "policy": g, "hospital": _hospital(rng), "service": svc, "service_offered": offered,
                "service_on_call": on_call, "bed_need": need, "need_detail": pick(rng, BED_NEEDS[need]),
                "free_beds_of_type": free_now, "next_bed_h": next_bed_h, "escalation_level": level,
                "network_option": network, "time_critical": time_critical, "payer": payer, "language": language,
                "finance_remark": finance, "age": age, "sex": sex, "time": rand_ts(rng),
                "sender": f"{pick(rng, HOSP_STEMS)} {pick(rng, ['Community Hospital', 'Rural Hospital', 'Medical Center'])}",
                "sending_md": "Dr. " + person(rng), "coordinator": person(rng)}

    def intended(self, f):
        g = f["policy"]
        capable = f["service_offered"] and f["service_on_call"]
        soon = f["next_bed_h"] is not None and f["next_bed_h"] <= g["pending_window_h"]
        if not capable:
            dec = "decline_no_capability"
        elif f["free_beds_of_type"] > 0:
            dec = "accept_now"
        elif soon and f["escalation_level"] < 4:
            dec = "accept_pending_bed"
        elif not f["time_critical"] and f["network_option"] == "capability and a free bed":
            dec = "redirect_within_network"
        elif soon:
            dec = "accept_pending_bed"
        else:
            dec = "decline_no_capacity"
        avail = (3 if f["free_beds_of_type"] >= 2 else 2 if f["free_beds_of_type"] == 1 else 1 if soon else 0)
        crit = (3 if f["time_critical"] and f["bed_need"] == "icu" else 2 if f["time_critical"] else
                1 if f["bed_need"] in ("icu", "step_down") else 0)
        return {"acceptance_decision": dec, "bed_type_needed": f["bed_need"], "transfer_time_criticality": crit,
                "suitable_bed_availability": avail, "capability_match": "yes" if capable else "no"}

    def _policy(self, f, rng):
        g = f["policy"]
        return {
            "title": pick(rng, [f"Transfer acceptance policy {g['rev']}", f"{f['hospital']} transfer center rules "
                                                                            f"({g['rev']})"]),
            "capability": "Decline for capability only when the service is not offered here or nobody covering it "
                          "is on call; give the sender the network hospital that offers it.",
            "capacity": f"Accept now with a suitable free bed. With none, accept pending a bed expected within "
                        f"{g['pending_window_h']} h, unless the hospital is at escalation Level 4. Patients who are "
                        f"not time-critical may be redirected to a network hospital with the capability and a free "
                        f"bed. Otherwise decline for capacity and help find a bed elsewhere.",
            "bed_type": "Bed type by support: ventilation or vasopressors ICU; BiPAP, titrated drips or hourly neuro "
                        "checks step-down; cardiac monitoring telemetry; otherwise a general ward.",
            "equity": "Payer, immigration status and ability to pay are never asked about or considered; no "
                      "financial clearance is requested before acceptance. Use an interpreter for any family call.",
        }

    def render(self, f, rng, surface):
        pl = self._policy(f, rng)
        svc_txt = SERVICES[f["service"]][0]
        cap_txt = ("offered here" if f["service_offered"] else "not offered here") + (
            ", on call today" if f["service_on_call"] else (", nobody on call today" if f["service_offered"] else ""))
        bed_txt = (f"{f['free_beds_of_type']} free" if f["free_beds_of_type"] else
                   ("none free; next expected in about " + f"{f['next_bed_h']} h" if f["next_bed_h"] is not None
                    else "none free; no discharge expected today"))
        bed_label = {"icu": "ICU", "step_down": "step-down", "telemetry": "telemetry",
                     "general_ward": "general ward"}[f["bed_need"]]
        if surface == "json":
            return dumps(prune({
                "request": {"from": f["sender"], "physician": f["sending_md"], "time": _hm(f["time"]),
                            "needs": svc_txt, "patient": f"{f['age']} {f['sex']}, {f['need_detail']}",
                            "time_critical_per_sender": f["time_critical"], "payer": f["payer"],
                            "family_language": f["language"]},
                "our_status": {"service": cap_txt, "escalation_level": f["escalation_level"],
                               "beds_" + bed_label.replace(" ", "_").replace("-", "_"): bed_txt},
                "network": {"other_hospitals": f["network_option"] or "none with the service and a free bed"},
                "notes": f["finance_remark"],
                "policy": pl,
            }))
        if surface == "transfer_call_log":
            t = f["time"]
            log = [f"{_hm(t)} {f['sending_md']} at {f['sender']} requests transfer: {svc_txt}",
                   f"{_hm(t + timedelta(minutes=2))} patient {f['age']}{f['sex']}, {f['need_detail']}; "
                   f"{'time-critical' if f['time_critical'] else 'stable for now'}",
                   f"{_hm(t + timedelta(minutes=4))} registration: {f['payer']}; family speaks {f['language']}",
                   f"{_hm(t + timedelta(minutes=6))} our service: {cap_txt}",
                   f"{_hm(t + timedelta(minutes=8))} {bed_label} beds: {bed_txt}; hospital at escalation Level "
                   f"{f['escalation_level']}",
                   f"{_hm(t + timedelta(minutes=9))} network: {f['network_option'] or 'no other hospital with the service and a free bed'}"]
            if f["finance_remark"]:
                log.append(f"{_hm(t + timedelta(minutes=10))} {f['finance_remark']}")
            return (f"TRANSFER CENTER -- {f['hospital']} ({f['coordinator']})\n" + "\n".join(log) + f"\n\n{pl['title']}\n"
                    + bullets([v for k, v in pl.items() if k != "title"], rng))
        rows = [("Sending", f"{f['sender']} / {f['sending_md']}"), ("Requested", svc_txt),
                ("Patient", f"{f['age']} {f['sex']}; {f['need_detail']}"),
                ("Time-critical (sender)", "yes" if f["time_critical"] else "no"), ("Payer", f["payer"]),
                ("Family language", f["language"]), ("Service here", cap_txt),
                (f"{_cap(bed_label)} beds", bed_txt), ("Escalation level", f["escalation_level"]),
                ("Network", f["network_option"] or "none suitable"), ("Note", f["finance_remark"])]
        return ("TRANSFER ACCEPTANCE FORM\n" + kv_lines(rows, rng) + f"\n\n{pl['title']}\n"
                + "\n".join(f"- {v}" for k, v in pl.items() if k != "title"))

    def questions(self):
        return {
            "acceptance_decision": choice("How should the transfer center answer this request?", {
                "accept_now": "Accept the patient now into a free bed of the right type.",
                "accept_pending_bed": "Accept, with the patient to come when the expected bed frees up.",
                "decline_no_capability": "Decline: the service is not available here today.",
                "decline_no_capacity": "Decline for capacity and help the sender find a bed elsewhere.",
                "redirect_within_network": "Redirect the patient to the network hospital with the service and a "
                                           "free bed.",
            }),
            "bed_type_needed": choice("What kind of bed does this patient need?", {
                "general_ward": "A general ward bed.",
                "icu": "An intensive care bed.",
                "step_down": "A step-down bed.",
                "telemetry": "A telemetry (cardiac-monitored) bed.",
            }),
            "transfer_time_criticality": score("How time-critical is this transfer?", [
                "Not time-critical: a ward-level patient.",
                "Needs a higher-level bed, but the timing is flexible.",
                "Time-critical: the treatment window is short.",
                "Time-critical and needing intensive care on arrival.",
            ]),
            "suitable_bed_availability": score("How available is a bed of the right type?", [
                "None within the policy's waiting window.",
                "None now, but one expected within the window.",
                "One free now.",
                "Two or more free now.",
            ]),
            "capability_match": noul(
                "This hospital can provide the requested service today.",
                yes="The service is offered here and someone covering it is on call.",
                no="The service is not offered here, or nobody covering it is on call today."),
        }


WORKFLOW = Workflow(
    name="hospital_capacity_escalation",
    title="Hospital capacity escalation",
    description="Bed-board and surge decisions in a hospital (escalation level, electives, diversion, ICU space, nurse "
                "staffing, transfer acceptance) against its written capacity plan.",
    domain="health-care operations / hospital patient flow",
    schemas=[HouseWideEscalation(), EdDiversionRequest(), IcuCapacityHuddle(), StaffingBedClosure(),
             TransferCenterAcceptance()],
    notes="All hospitals, plans, triggers, units and people are fictional; level names and thresholds are invented and "
          "are not real operational or clinical guidance. Equity is written into the plans: discharge barriers bring "
          "in support and never justify an unsafe discharge, and acceptance never depends on payer, immigration "
          "status or ability to pay. No ED triage acuity and no EMS call priority.",
)
