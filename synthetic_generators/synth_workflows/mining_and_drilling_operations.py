"""Mining and drilling operations: safety-critical calls in underground and surface mines and on drilling rigs,
judged against a (fictional) site standard or trigger-action response plan (TARP).

Five schemas: a methane or airflow reading in an underground coal section against power-off and withdrawal
limits, a roof or pit-wall displacement reading against ground-control triggers, a surface blast's clearance before
firing, kick indicators on a drilling rig against the well-control standard's shut-in rule, and a tailings dam's
surveillance readings against its TARP. Artefacts are examiner reports, control-room radio logs, TARP forms,
geotechnical emails, blast countdown logs, driller's tour logs, rig chat and tailings surveillance reports.

Every rule is invented: the standards (MVS-11, GC-04, BP-12, WCS-7, TSM-6), their numbers and the mines, rigs and
people are fictional, and thresholds are sampled per case so the answer always needs a comparison. Only the one to
three clauses that govern a case are quoted. Midstream pipeline integrity lives in pipeline_integrity_dig_decision
and is not repeated here.

Each schema also has `intended(f)`: the designer's reading of the rule for that case. It is never rendered and
never a label; it exists so the answer-spread check can confirm every question moves with the factors.
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

MINE_NAMES = ["Alderbrook", "Bramwell", "Carrow", "Dunlin", "Fallowmere", "Gorsedale", "Hartsell", "Inchcombe",
              "Kelsing", "Larkhollow", "Nethercote", "Orrin Creek", "Pellham", "Quenby", "Selwick", "Thornby",
              "Wexcombe", "Yarrowdale", "Cobbett Hill", "Marlow Ridge", "Ashcombe", "Brindle Flat"]


def _mine(rng, kind):
    return f"{pick(rng, MINE_NAMES)} {kind}"


def _first(name):
    return name.split()[0]


def _kv(pairs, rng, skip_none=True):
    """kv_lines without the snake_case 'equals' house style, which mangles labels such as 'Site / sector'."""
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]), skip_none=skip_none)


def _shift(ts):
    return "day" if 7 <= ts.hour < 15 else "afternoon" if 15 <= ts.hour < 23 else "night"


# =========================================================================== schema 1
CM_EVENTS = {"none": None,
             "stopping_damaged": "a scoop hit a stopping two crosscuts outby; temporary repair only",
             "curtain_down": "line curtain at the face found torn down by the shuttle car",
             "fan_trip": "main fan tripped and was restarted",
             "regulator_change": "section regulator adjusted on the previous shift",
             "fault_zone": "the miner cut into the mapped fault zone this shift",
             "baro_drop": "barometer falling fast ahead of a front"}
LW_EVENTS = {"none": None,
             "stopping_damaged": "a stopping in the maingate was damaged by a supply vehicle",
             "fan_trip": "main fan tripped and was restarted",
             "regulator_change": "tailgate regulator opened two notches yesterday",
             "goaf_fall": "a large goaf fall was heard behind the shields",
             "baro_drop": "barometer falling fast ahead of a front"}
VENT_EVENTS = {"stopping_damaged", "curtain_down", "fan_trip", "regulator_change"}
STRATA_EVENTS = {"fault_zone", "goaf_fall"}


class UndergroundGasWithdrawal(Schema):
    name = "underground_gas_withdrawal"
    description = ("A methane or airflow reading in an underground coal section, judged against the ventilation "
                   "standard's power-off and withdrawal limits.")
    surfaces = {"json": 0.5, "examiner_report": 0.25, "control_radio": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        lw = chance(rng, 0.4)
        power_off = pick(rng, [0.8, 1.0, 1.0, 1.25])
        withdraw = 2.0 if power_off >= 1.25 else pick(rng, [1.5, 2.0])
        air_min = pick(rng, [25.0, 30.0, 35.0]) if lw else pick(rng, [5.0, 6.0, 7.5, 9.0])
        fan_rule = pick(rng, [10, 15, 20])
        events = LW_EVENTS if lw else CM_EVENTS
        event = pick_w(rng, {k: (5 if k == "none" else 0.7 if k in VENT_EVENTS else 2.0 if k == "baro_drop" else 1.3)
                             for k in events})
        cal = pick_w(rng, {"in_date": 6, "overdue": 2, "bump_failed": 1, None: 1})
        trend = pick_w(rng, {"rising": 3, "steady": 4, "falling": 2})
        air = round(air_min * rng.uniform(0.9, 1.5), 1)
        ch4 = round(rng.uniform(0.2, withdraw + 0.5), 2)
        confirm = "auto"  # drawn after the regime branch, from the final primary reading
        fan_min = None
        note_kind = pick(rng, ["production", "gassy", "settle", "none", "none"])
        k = None
        if reg == "clear":
            cal = "in_date"
            if chance(rng, 0.5):
                ch4 = between(rng, 0.1, power_off * 0.5)
                confirm = maybe(rng, round(ch4 + rng.uniform(-0.04, 0.06), 2), 0.5)
                air = round(air_min * rng.uniform(1.15, 1.6), 1)
                event, trend = pick(rng, ["none", "none", "regulator_change"]), pick(rng, ["steady", "falling"])
            else:
                ch4 = between(rng, withdraw + 0.2, withdraw + 1.1)
                confirm = round(ch4 + rng.uniform(-0.1, 0.1), 2)
                trend = "rising"
        elif reg == "borderline":
            k = pick(rng, ["power_off", "withdraw", "air", "fan"])
            if k == "power_off":
                ch4 = near(rng, power_off, 0.18, lo=0.05)
                confirm = maybe(rng, near(rng, ch4, 0.12, lo=0.05), 0.45)
            elif k == "withdraw":
                ch4 = near(rng, withdraw, 0.22, lo=0.05)
                confirm = maybe(rng, near(rng, ch4, 0.12, lo=0.05), 0.45)
            elif k == "air":
                ch4 = between(rng, 0.2, power_off * 0.85)
                confirm = maybe(rng, round(ch4 + rng.uniform(-0.05, 0.05), 2), 0.5)
                air = near(rng, air_min, air_min * 0.08, digits=1)
                event = pick(rng, ["none", "regulator_change", "stopping_damaged"])
            else:
                event = "fan_trip"
                fan_min = near(rng, fan_rule, 4, lo=1, integer=True)
                ch4 = between(rng, 0.2, power_off * 0.9)
        else:
            k = pick(rng, ["bad_sensor_high", "air_short_quiet", "note_dismisses", "fan_long_quiet"])
            if k == "bad_sensor_high":
                ch4 = between(rng, withdraw - 0.1, withdraw + 0.8)
                cal = pick(rng, ["bump_failed", "overdue"])
                confirm = between(rng, 0.2, power_off * 1.1)
                note_kind = "drift"
            elif k == "air_short_quiet":
                ch4 = between(rng, 0.2, power_off * 0.7)
                air = round(air_min * rng.uniform(0.6, 0.92), 1)
                event = pick(rng, ["stopping_damaged", "curtain_down", "none"]) if not lw else pick(rng, ["stopping_damaged", "none"])
                note_kind = "settle"
            elif k == "note_dismisses":
                ch4 = between(rng, power_off, withdraw + 0.4)
                confirm = round(ch4 + rng.uniform(-0.1, 0.1), 2)
                cal = "in_date"
                note_kind = "drift"
            else:
                event = "fan_trip"
                fan_min = fan_rule + rng.randint(2, 25)
                ch4 = between(rng, 0.1, 0.6)
                note_kind = "fan_ok"
        if confirm == "auto":
            confirm = maybe(rng, round(max(0.05, ch4 + rng.uniform(-0.15, 0.15)), 2), 0.4)
        if event == "fan_trip" and fan_min is None:
            fan_min = rng.randint(2, 30)
        crew = rng.randint(12, 22) if lw else rng.randint(6, 14)
        loc = pick(rng, ["tailgate end", "shearer", "longwall return"] if lw else ["face", "section return", "last open crosscut"])
        inst = "machine methanometer" if loc in ("face", "shearer") else "fixed monitor"
        return {"regime": reg, "case_kind": k, "longwall": lw, "mine": _mine(rng, pick(rng, ["Colliery", "Mine", "No. 2 Mine"])),
                "district": (f"LW{rng.randint(101, 412)}" if lw else f"{rng.randint(1, 9)} {pick(rng, ['Left', 'Right', 'North', 'South', 'East', 'West'])}"),
                "location": loc, "instrument": inst, "ch4_pct": ch4, "check_pct": confirm, "calibration": cal,
                "trend": trend, "air_m3s": maybe(rng, air, 0.1), "air_min_m3s": air_min, "event": event,
                "fan_stopped_min": fan_min, "power_off_pct": power_off, "withdraw_pct": withdraw,
                "fan_rule_min": fan_rule, "crew": crew, "energised": chance(rng, 0.7), "note_kind": note_kind,
                "examiner": person(rng), "foreman": person(rng),
                "ts": rand_ts(rng), "std": pick(rng, ["MVS-11", "MVS-14", "SVS-3"]),
                "co_ppm": rng.randint(0, 9), "o2_pct": round(rng.uniform(20.4, 20.9), 1),
                "belt_running": chance(rng, 0.7), "rock_dust": pick(rng, ["adequate", "adequate", "light at the last crosscut"])}

    # ------------------------------------------------------------------ the designer's reading
    def _governing(self, f):
        p, c = f["ch4_pct"], f["check_pct"]
        if c is None:
            return p
        if f["calibration"] in ("bump_failed", "overdue") and c < p:
            return c
        return max(p, c)

    def intended(self, f):
        g = self._governing(f)
        po, wd = f["power_off_pct"], f["withdraw_pct"]
        air = f["air_m3s"]
        air_short = air is not None and air < f["air_min_m3s"]
        fan_long = f["fan_stopped_min"] is not None and f["fan_stopped_min"] >= f["fan_rule_min"]
        if fan_long:
            resp = "evacuate_mine"
        elif g >= wd:
            resp = "withdraw_from_district"
        elif g >= po or air_short:
            resp = "deenergise_and_correct"
        else:
            resp = "continue_with_checks"
        p, c = f["ch4_pct"], f["check_pct"]
        ev = f["event"]
        if f["calibration"] in ("bump_failed", "overdue") and c is not None and c < p - 0.3:
            cause = "instrument_fault"
        elif air_short or ev in VENT_EVENTS:
            cause = "ventilation_disruption"
        elif ev in STRATA_EVENTS or (g >= po and ev != "baro_drop"):
            cause = "increased_strata_emission"
        elif ev == "baro_drop":
            cause = "barometric_pressure_drop"
        else:
            cause = "normal_background_make"
        rel = {"in_date": 2, "overdue": 1, "bump_failed": 0, None: 1}[f["calibration"]]
        if c is not None and abs(c - p) <= 0.15:
            rel += 1
        elif c is not None and abs(c - p) >= 0.4:
            rel -= 1
        rel = max(0, min(3, rel))
        hz = 0 if g < 0.5 * po else 1 if g < po else 2 if g < wd else 3
        if f["trend"] == "rising" and hz in (1, 2) and f["energised"]:
            hz += 1
        if fan_long:
            hz = max(hz, 2)
        return {"district_response": resp, "likely_cause": cause, "reading_reliability": rel,
                "ignition_hazard": min(3, hz),
                "air_quantity_meets_plan": "yes" if air is not None and not air_short else "no"}

    # ------------------------------------------------------------------ rendering
    def _rule(self, f):
        s = (f"{f['std']} s4.2: power off face equipment at >= {f['power_off_pct']}% CH4; withdraw persons at >= "
             f"{f['withdraw_pct']}%; minimum {f['air_min_m3s']} m3/s at the "
             + ("tailgate" if f["longwall"] else "last open crosscut"))
        if f["event"] == "fan_trip":
            s += f"; main fan stopped >= {f['fan_rule_min']} min: withdraw all persons from the mine"
        return s

    def _note(self, f, rng):
        t = {"production": "Production wants this cut finished before shift change.",
             "gassy": "This panel has always been gassy at the face.",
             "settle": ("Air will come back once the outby stopping is patched." if f["longwall"] else
                        pick(rng, ["Readings like this usually settle once the curtain is hung properly.",
                                   "Air will come back once the outby stopping is patched."])),
             "drift": pick(rng, ["I reckon it's sensor drift, it's been reading high all week.",
                                 "Probably the head on the miner, it's due for a service."]),
             "fan_ok": "Fan is back up and the gas never moved, so I'd keep going.",
             "none": ""}[f["note_kind"]]
        return noisy(t, rng, 0.15) if t else None

    def _cal(self, v):
        return {"in_date": "calibrated, bump test passed", "overdue": "calibration overdue",
                "bump_failed": "failed bump test at start of shift", None: "calibration status not recorded"}[v]

    def render(self, f, rng, surface):
        events = LW_EVENTS if f["longwall"] else CM_EVENTS
        ev = events.get(f["event"])
        note = self._note(f, rng)
        ts = f["ts"]
        if surface == "json":
            return dumps(prune({
                "reading": {"location": f["location"], "instrument": f["instrument"], "ch4_pct": f["ch4_pct"],
                            "instrument_status": self._cal(f["calibration"]), "trend_30min": f["trend"],
                            "handheld_check_pct": f["check_pct"], "time": fmt_ts(ts, style="iso")},
                "section": {"mine": f["mine"], "district": f["district"], "crew_inby": f["crew"],
                            "face_equipment_energised": f["energised"], "shift": _shift(f["ts"])},
                "other_gases": {"co_ppm": f["co_ppm"], "o2_pct": f["o2_pct"]},
                "ventilation": {"air_m3s": f["air_m3s"], "recent_event": ev,
                                "main_fan_down_min": f["fan_stopped_min"]},
                "rule": self._rule(f),
                "foreman_note": note,
            }))
        if surface == "examiner_report":
            t = ts
            lines = [f"{f['mine'].upper()} -- EXAMINER'S REPORT, {f['district']}, {_shift(ts)} shift {fmt_date(ts.date(), rng)}"]
            lines.append(f"{t.strftime('%H:%M')}  CH4 {f['ch4_pct']}% at {f['location']} ({f['instrument']}), trend {f['trend']}")
            if f["check_pct"] is not None:
                t += timedelta(minutes=rng.randint(3, 9))
                lines.append(f"{t.strftime('%H:%M')}  handheld check {f['check_pct']}%")
            t += timedelta(minutes=rng.randint(3, 12))
            lines.append(f"{t.strftime('%H:%M')}  air " + ("not measured yet" if f["air_m3s"] is None else f"{f['air_m3s']} m3/s"))
            if ev:
                lines.append(f"Noted: {ev}" + (f" (down {f['fan_stopped_min']} min)" if f["event"] == "fan_trip" else ""))
            lines.append(f"CO {f['co_ppm']} ppm, O2 {f['o2_pct']}%. Rock dust {f['rock_dust']}. Belt "
                         + ("running." if f["belt_running"] else "stopped."))
            lines.append(f"Instrument: {self._cal(f['calibration'])}. Crew inby: {f['crew']}. Face equipment "
                         + ("energised." if f["energised"] else "isolated."))
            lines.append(f"Examiner: {f['examiner']}")
            if note:
                lines.append(f"Foreman ({f['foreman']}): {note}")
            lines.append(f"Std: power-off {f['power_off_pct']}% / withdraw {f['withdraw_pct']}% / air min {f['air_min_m3s']} m3/s"
                         + (f" / fan stop {f['fan_rule_min']} min" if f["event"] == "fan_trip" else ""))
            return "\n".join(lines)
        # control_radio
        cr, fm = "CONTROL", _first(f["foreman"]).upper()
        msgs = [(fm, f"{f['district']} {f['location']}, methane {f['ch4_pct']} percent and {f['trend']}")]
        if f["check_pct"] is not None:
            msgs.append((fm, f"handheld says {f['check_pct']}"))
        msgs.append((cr, {"in_date": "copy. Head was bump tested this morning",
                          "overdue": "copy. Note that head is overdue for calibration",
                          "bump_failed": "copy. That head failed its bump test at start of shift",
                          None: "copy. No calibration record here for that head"}[f["calibration"]]))
        if f["air_m3s"] is not None:
            msgs.append((fm, f"air reading {f['air_m3s']} cubes"))
        else:
            msgs.append((cr, "no air reading logged yet this shift"))
        if ev:
            msgs.append((cr, ev + (f", down {f['fan_stopped_min']} minutes" if f["event"] == "fan_trip" else "")))
        if note:
            msgs.append((fm, note))
        msgs.append((cr, f"{f['crew']} on your section, " + ("shearer" if f["longwall"] else "miner") + " "
                     + ("running" if f["energised"] else "isolated") + f". CO {f['co_ppm']} ppm on the section monitor"))
        t = ts
        out = []
        for who, m in msgs:
            out.append(f"[{t.strftime('%H:%M')}] {who}: {m}")
            t += timedelta(minutes=rng.randint(1, 4))
        return (f"{f['mine']} control room radio log\n" + "\n".join(out) + f"\n-- {self._rule(f)}")

    def questions(self):
        return {
            "district_response": choice("What does the ventilation standard require the section to do now?", {
                "continue_with_checks": "Keep producing, with more frequent gas and air checks this shift.",
                "deenergise_and_correct": "Cut power to face equipment and fix the gas or airflow problem before restarting.",
                "withdraw_from_district": "Withdraw everyone from the affected district except those correcting the problem.",
                "evacuate_mine": "Bring every person out of the mine.",
            }),
            "likely_cause": choice("What most likely explains the reading?", {
                "barometric_pressure_drop": "Falling barometric pressure drawing gas out of old workings or the goaf.",
                "increased_strata_emission": "More gas coming off the strata, for example from a fault or a goaf fall.",
                "instrument_fault": "A faulty or out-of-calibration instrument giving a false reading.",
                "normal_background_make": "Ordinary background gas from the coal, nothing out of the ordinary.",
                "ventilation_disruption": "Less air reaching the area because of damaged or changed ventilation controls.",
            }),
            "reading_reliability": score("How far can the gas reading be trusted?", [
                "Hardly at all: the instrument is suspect and nothing confirms it.",
                "Partly: one doubtful instrument or readings that disagree.",
                "Mostly: a working instrument, but no independent check.",
                "Fully: a calibrated instrument confirmed by a second reading.",
            ]),
            "ignition_hazard": score("How close is the section to an explosive-atmosphere hazard?", [
                "Remote: gas well below the power-off level and steady.",
                "Low: some gas, but below the power-off level.",
                "Elevated: at or above the power-off level, or rising with power on.",
                "Serious: at the withdrawal level or rising fast with ventilation lost.",
            ]),
            "air_quantity_meets_plan": noul(
                "Airflow at the measuring point met the ventilation plan minimum when last measured.",
                yes="A current measurement is at or above the minimum the standard sets.",
                no="The measurement is below the minimum, or no current measurement shows it is met."),
        }


# =========================================================================== schema 2
UG_AREAS = {"main travelway intersection": 2, "working section, last row of bolts": 2, "belt drive": 1,
            "return airway, examined weekly": 0, "crib room access": 2, "pump station": 1}
PIT_AREAS = {"haul ramp below the wall": 2, "shovel working at the toe": 2, "inactive bench": 0,
             "dewatering pump access": 1, "crusher pad below": 2, "light-vehicle road on the bench": 1}
UG_SIGNS = {"none": 0, "roof drummy when sounded": 1, "fresh rib spall": 1, "water dripping from bolt holes": 1,
            "bolt plates dished": 2, "mesh bagging with broken rock": 2}
PIT_SIGNS = {"none": 0, "ravelling from the face": 1, "seepage at the toe": 1,
             "new tension crack on the crest": 2, "bench crest loss": 2}


class GroundControlTrigger(Schema):
    name = "ground_control_trigger"
    description = ("A roof or pit-wall displacement reading compared with the site's ground-control trigger levels "
                   "(TARP), with visual signs and the people working below.")
    surfaces = {"json": 0.5, "tarp_form": 0.25, "geotech_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        pit = chance(rng, 0.45)
        trend = pick_w(rng, {"accelerating": 2, "steady": 4, "decelerating": 3})
        status = pick_w(rng, {"ok": 7, "damaged": 1, "stale": 1})
        signs = PIT_SIGNS if pit else UG_SIGNS
        sign = pick_w(rng, {k: (9 if k == "none" else 1) for k in signs})
        if pit:
            amber = pick(rng, [0.5, 1.0, 2.0])
            red = round(amber * pick(rng, [3, 4, 5]), 1)
            rate_amber = rate_red = None
            val = round(amber * rng.uniform(0.2, 2.5), 2)
            rate = None
        else:
            amber = pick(rng, [10, 15, 20])
            red = amber + pick(rng, [10, 15, 20])
            rate_amber, rate_red = pick(rng, [1, 2]), pick(rng, [3, 4, 5])
            val = round(amber * rng.uniform(0.2, 1.6), 1)
            rate = round(rng.uniform(0.1, rate_red * 0.9), 1)
        k = None
        if reg == "clear":
            status = "ok"
            if chance(rng, 0.6):
                val = round(amber * rng.uniform(0.1, 0.5), 2 if pit else 1)
                rate = None if pit else round(rng.uniform(0.0, rate_amber * 0.4), 1)
                sign, trend = "none", pick(rng, ["steady", "decelerating"])
            else:
                val = round(red * rng.uniform(1.05, 1.6), 2 if pit else 1)
                rate = None if pit else round(rate_red * rng.uniform(1.0, 2.0), 1)
                trend = "accelerating"
                sign = pick(rng, [s for s, v in signs.items() if v == 2])
        elif reg == "borderline":
            k = pick(rng, ["amber", "amber", "red", "rate"]) if not pit else pick(rng, ["amber", "amber", "red"])
            if k == "amber":
                val = near(rng, amber, amber * 0.15, lo=0, digits=2 if pit else 1)
                if not pit:
                    rate = round(rng.uniform(0.1, rate_amber * 0.9), 1)
            elif k == "red":
                val = near(rng, red, red * 0.12, lo=0, digits=2 if pit else 1)
                if not pit:
                    rate = round(rng.uniform(rate_amber, rate_red * 0.9), 1)
            else:
                val = round(amber * rng.uniform(0.5, 0.95), 1)
                rate = near(rng, pick(rng, [rate_amber, rate_red]), 0.6, lo=0.1, digits=1)
        else:
            k = pick(rng, ["quiet_numbers_bad_sign", "loud_numbers_slowing", "instrument_down", "quiet_numbers_worried_crew"])
            if k == "quiet_numbers_bad_sign":
                val = round(amber * rng.uniform(0.5, 1.2), 2 if pit else 1)
                rate = None if pit else round(rng.uniform(0.2, rate_amber * 1.2), 1)
                sign, trend = pick(rng, [s for s, v in signs.items() if v == 2]), pick(rng, ["accelerating", "steady"])
            elif k == "loud_numbers_slowing":
                val = round(red * rng.uniform(0.85, 1.15), 2 if pit else 1)
                rate = None if pit else round(rng.uniform(0.1, rate_amber * 0.8), 1)
                sign, trend = "none", "decelerating"
            elif k == "instrument_down":
                status = pick(rng, ["damaged", "stale"])
                sign = pick(rng, [s for s, v in signs.items() if v >= 1])
            else:
                val = round(amber * rng.uniform(0.15, 0.6), 2 if pit else 1)
                rate = None if pit else round(rng.uniform(0.0, rate_amber * 0.5), 1)
                sign, trend, status = "none", pick(rng, ["steady", "decelerating"]), "ok"
        areas = PIT_AREAS if pit else UG_AREAS
        area = pick(rng, list(areas))
        prev = round(val * {"accelerating": rng.uniform(0.4, 0.75), "steady": rng.uniform(0.8, 0.97),
                            "decelerating": rng.uniform(0.85, 1.0)}[trend], 2 if pit else 1)
        return {"regime": reg, "case_kind": k, "pit": pit, "site": _mine(rng, "Open Pit" if pit else pick(rng, ["Mine", "Colliery", "Underground"])),
                "location": (f"{pick(rng, ['North', 'East', 'South', 'West'])} wall, sector {rng.randint(2, 14)}" if pit
                             else f"{pick(rng, ['Main', 'Panel', 'Drive'])} {rng.randint(2, 18)} at XC {rng.randint(3, 40)}"),
                "instrument": pick(rng, ["slope radar", "prism survey"]) if pit else pick(rng, ["dual-height telltale", "sonic extensometer"]),
                "value": val, "rate": rate, "amber": amber, "red": red, "rate_amber": rate_amber, "rate_red": rate_red,
                "trend": trend, "status": status, "stale_hours": rng.randint(20, 72) if status == "stale" else None,
                "sign": sign, "area": area, "area_use": areas[area],
                "people": rng.randint(*{0: (0, 3), 1: (2, 9), 2: (5, 18)}[areas[area]]),
                "recent": pick(rng, (["production blast 80 m away yesterday", "38 mm of rain overnight", None, None] if pit
                                     else ["mining within 15 m this shift", "pillar extraction in the next panel", None, None])),
                "reporter": person(rng), "geotech": person(rng), "date": rand_ts(rng),
                "opinion": "worried" if k == "quiet_numbers_worried_crew" else pick(rng, ["keep_going", "worried", "none", "none"]),
                "tarp": pick(rng, ["GC-04", "GCMP-7", "TARP-G2"]),
                "support": (pick(rng, ["no mesh, 12 m catch bench", "rockfall fence at the toe", "16 m catch bench, no fence"]) if pit
                            else pick(rng, ["1.8 m bolts at 1.2 m x 1.0 m, mesh", "2.4 m bolts at 1.5 m x 1.5 m, straps",
                                            "1.8 m bolts at 1.0 m x 1.0 m, cable bolts at intersection"])),
                "prev_value": prev}

    def _level(self, f):
        bad_inst = f["status"] != "ok"
        sev = (PIT_SIGNS if f["pit"] else UG_SIGNS)[f["sign"]]
        if bad_inst:
            red_hit, amber_hit, near_red = False, sev >= 1, False
        elif f["pit"]:
            red_hit, amber_hit = f["value"] >= f["red"], f["value"] >= f["amber"]
            near_red = f["value"] >= 0.8 * f["red"]
        else:
            red_hit = f["value"] >= f["red"] or f["rate"] >= f["rate_red"]
            amber_hit = f["value"] >= f["amber"] or f["rate"] >= f["rate_amber"]
            near_red = f["value"] >= 0.8 * f["red"] or f["rate"] >= 0.8 * f["rate_red"]
        lvl = 2 if red_hit else 1 if amber_hit else 0
        if lvl == 1 and f["trend"] == "accelerating" and sev == 2:
            lvl = 2
        if lvl == 0 and sev == 2:
            lvl = 1
        return lvl, sev, near_red, bad_inst

    def intended(self, f):
        lvl, sev, near_red, bad_inst = self._level(f)
        if lvl == 2:
            resp = "barricade_and_withdraw"
        elif lvl == 1 and (sev == 2 or near_red or f["trend"] == "accelerating"):
            resp = "stop_and_support"
        elif lvl == 1 or bad_inst:
            resp = "restrict_and_monitor"
        else:
            resp = "continue_normal"
        if lvl == 0:
            low = (f["value"] < 0.5 * f["amber"]) and f["trend"] != "accelerating" and sev == 0 and not bad_inst
            inst = 0 if low else 1
        else:
            inst = lvl + 1
        expo = min(3, f["area_use"] + (1 if f["people"] >= 8 else 0))
        return {"tarp_response": resp, "instability_level": inst, "exposure_of_people": expo,
                "red_trigger_reached": "yes" if lvl == 2 else "no",
                "geotech_inspection_required": "yes" if (lvl >= 1 or bad_inst) else "no"}

    def _rule(self, f):
        if f["pit"]:
            return (f"{f['tarp']}: amber at {f['amber']} mm/h, red at {f['red']} mm/h (radar or prism velocity); "
                    "amber plus accelerating trend and a crest crack or crest loss = red")
        return (f"{f['tarp']}: amber at {f['amber']} mm or {f['rate_amber']} mm/day; red at {f['red']} mm or "
                f"{f['rate_red']} mm/day; amber plus accelerating trend and dished plates or bagging mesh = red")

    def _reading(self, f):
        if f["status"] == "damaged":
            return "instrument damaged, no valid reading"
        base_txt = (f"{f['value']} mm/h" if f["pit"] else f"{f['value']} mm total, {f['rate']} mm/day")
        if f["status"] == "stale":
            return f"{base_txt} (last read {f['stale_hours']} h ago)"
        return base_txt

    def _prev(self, f):
        if f["status"] == "damaged":
            return None
        return f"{f['prev_value']} mm/h" if f["pit"] else f"{f['prev_value']} mm total"

    def _opinion(self, f, rng):
        t = {"keep_going": pick(rng, ["Shift boss says it has looked like this for weeks.",
                                      "Crew are happy to keep working, nothing has changed visually."]),
             "worried": pick(rng, ["Operators reported small rocks coming off the wall overnight." if f["pit"]
                                   else "Crew reported noises from the roof overnight.",
                                   "Operator asked for the area to be looked at before next shift."]),
             "none": ""}[f["opinion"]]
        return noisy(t, rng, 0.15) if t else None

    def render(self, f, rng, surface):
        op = self._opinion(f, rng)
        reading = self._reading(f)
        if surface == "json":
            return dumps(prune({
                "location": {"site": f["site"], "area": f["location"], "use": f["area"],
                             "people_per_shift": f["people"]},
                "monitoring": {"instrument": f["instrument"], "reading": reading,
                               "previous_reading": self._prev(f),
                               "trend_48h": f["trend"] if f["status"] != "damaged" else None},
                "installed_support": f["support"],
                "inspection": {"visible_signs": f["sign"], "recent_activity": f["recent"],
                               "inspected_by": f["reporter"], "date": fmt_date(f["date"].date(), style="iso")},
                "tarp": self._rule(f),
                "comment": op,
            }))
        if surface == "tarp_form":
            body = _kv([("Site / area", f"{f['site']} / {f['location']}"), ("Use of area", f["area"]),
                             ("People per shift", f["people"]), ("Instrument", f["instrument"]),
                             ("Reading", reading), ("Previous", self._prev(f)), ("Support", f["support"]),
                             ("Trend (48 h)", f["trend"] if f["status"] != "damaged" else None),
                             ("Visual signs", f["sign"]), ("Recent activity", f["recent"]),
                             ("Inspector", f"{f['reporter']}, {fmt_date(f['date'].date(), rng)}")], rng, skip_none=False)
            return (f"GROUND CONTROL TARP CHECK\n{body}\nComment: {op or '-'}\n"
                    f"Trigger levels: {self._rule(f).split(': ', 1)[1]}")
        s = [f"Readings from the {f['instrument']} at {f['location']} ({f['site']}): {reading}"
             + (f", trend {f['trend']}." if f["status"] != "damaged" else "."),
             (f"Previous reading {self._prev(f)}." if self._prev(f) else ""),
             f"Support: {f['support']}.",
             f"Visual: {f['sign']}." if f["sign"] != "none" else "Nothing visible on inspection.",
             f"{f['recent'][0].upper() + f['recent'][1:]}." if f["recent"] else "",
             f"Area is the {f['area']} with about {f['people']} people a shift.", op or ""]
        return "\n".join([f"Subject: {f['location']} -- {f['tarp']} readings",
                          f"From: {f['reporter']}", f"To: {f['geotech']} (geotechnical)", "",
                          join_sentences(s), "", f"For reference, {self._rule(f)}.", "", _first(f["reporter"])])

    def questions(self):
        return {
            "tarp_response": choice("Which response does the ground-control plan call for in this area?", {
                "barricade_and_withdraw": "Barricade the area and withdraw everyone from the zone that could be affected.",
                "continue_normal": "Carry on working with routine inspections.",
                "restrict_and_monitor": "Keep working but limit access and read the instruments more often until "
                                        "a geotechnical inspection.",
                "stop_and_support": "Stop work in the area until extra support or remediation is installed and checked.",
            }),
            "instability_level": score("How unstable does the ground look on the readings and the inspection?", [
                "Stable: readings small and settling, nothing visible.",
                "Watch: some movement or a minor sign, still under the first trigger.",
                "Moving: past the first trigger, or a serious visible sign.",
                "Failing: at the red trigger, or accelerating with support distress.",
            ]),
            "exposure_of_people": score("How exposed are people if the ground there gives way?", [
                "Minimal: an area hardly anyone enters.",
                "Limited: occasional access by a few people.",
                "Significant: a regularly used area or several people nearby.",
                "High: a busy route or working place with many people.",
            ]),
            "red_trigger_reached": noul(
                "The area has reached the plan's red trigger level.",
                yes="A valid reading, or the plan's combined amber-plus-signs rule, puts the area at red.",
                no="The readings are below red, or there is no valid reading showing red."),
            "geotech_inspection_required": noul(
                "A geotechnical engineer has to inspect the area before normal work goes on there."),
        }


# =========================================================================== schema 3
INTRUDERS = ["an unidentified light vehicle seen near the east ramp", "the water-cart driver has not checked out",
             "two walkers seen on the ridge track above the pit", "a fitter's ute still parked at the crusher pad"]


class BlastFiringClearance(Schema):
    name = "blast_firing_clearance"
    description = "A surface blast in its countdown: whether the shotfirer can fire, given guard posts, sweep, lightning and loading."
    surfaces = {"json": 0.5, "countdown_log": 0.25, "blast_checklist": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        R = pick(rng, [300, 400, 500, 600])
        L = pick(rng, [8, 10, 15])
        notify_m = pick(rng, [800, 1000, 1500])
        notify_h = pick(rng, [24, 48])
        n_posts = rng.randint(3, 5)
        posts = [[f"Post {i + 1}", int(R * rng.uniform(1.0, 1.4)), True, True] for i in range(n_posts)]
        sweep, intruder = "confirmed", None
        light = maybe(rng, round(rng.uniform(L + 3, 45), 1), 0.6)
        structure = int(R * rng.uniform(0.9, 3.5))
        road_in_zone = chance(rng, 0.3)
        closure = True
        neighbours = rng.randint(0, 12)
        notice_h = rng.randint(notify_h, notify_h + 72) if neighbours else None
        sunset = rng.randint(40, 300)
        short_holes = 0
        k = None
        if reg != "clear":
            # background imperfections any real countdown can carry
            if chance(rng, 0.2):
                posts[rng.randrange(n_posts)][1] = int(R * rng.uniform(0.75, 0.97))
            if chance(rng, 0.1):
                posts[rng.randrange(n_posts)][2] = False
            if chance(rng, 0.1):
                posts[rng.randrange(n_posts)][3] = False
            if chance(rng, 0.3):
                short_holes = rng.randint(1, 4)
            if neighbours and chance(rng, 0.45):
                notice_h = rng.randint(2, notify_h - 1)
            if road_in_zone and chance(rng, 0.25):
                closure = False
        if reg == "clear":
            if chance(rng, 0.5):
                light = None
            elif chance(rng, 0.5):
                light = round(rng.uniform(1.0, L * 0.6), 1)
            else:
                short_holes, structure = rng.randint(2, 5), int(R * rng.uniform(0.9, 1.15))
                posts[rng.randrange(n_posts)][1] = int(R * rng.uniform(0.6, 0.85))
        elif reg == "borderline":
            k = pick(rng, ["post_distance", "lightning", "stemming", "notice", "sunset"])
            if k == "post_distance":
                posts[rng.randrange(n_posts)][1] = near(rng, R, R * 0.06, lo=50, integer=True)
            elif k == "lightning":
                light = near(rng, L, 2.5, lo=0.5, digits=1)
            elif k == "stemming":
                short_holes = pick(rng, [1, 2])
                structure = int(R * rng.uniform(0.9, 2.2))
            elif k == "notice":
                neighbours = rng.randint(1, 8)
                notice_h = near(rng, notify_h, 6, lo=1, integer=True)
            else:
                posts[rng.randrange(n_posts)][3] = False
                sunset = rng.randint(15, 45)
        else:
            k = pick(rng, ["intruder_late", "unstaffed_post", "light_clear_sky", "stemming_near_road"])
            if k == "intruder_late":
                intruder = pick(rng, INTRUDERS)
                sweep = pick(rng, ["confirmed", "partial"])
                sunset = rng.randint(20, 120)
            elif k == "unstaffed_post":
                posts[rng.randrange(n_posts)][2] = False
                sweep = pick(rng, ["confirmed", None])
            elif k == "light_clear_sky":
                light = round(rng.uniform(1.0, L), 1)
            else:
                short_holes = rng.randint(2, 5)
                structure = int(R * rng.uniform(0.9, 1.3))
                road_in_zone, closure = True, chance(rng, 0.4)
        return {"regime": reg, "case_kind": k, "site": _mine(rng, pick(rng, ["Quarry", "Open Pit", "Mine"])),
                "blast_id": ident(rng, "B", 4, "-"), "holes": (holes := rng.randint(24, 160)),
                "explosive_t": round(holes * rng.uniform(40, 220) / 1000, 1), "exclusion_m": R, "lightning_rule_km": L,
                "posts": posts, "sweep": sweep, "intruder": intruder, "lightning_km": light,
                "short_stemmed_holes": short_holes, "design_stemming_m": pick(rng, [2.5, 3.0, 3.5, 4.0]),
                "structure_m": structure, "structure": pick(rng, ["farmhouse", "public road", "rail siding", "workshop", "power line"]),
                "road_in_zone": road_in_zone, "road_closed": closure if road_in_zone else None,
                "neighbours_in_notice_radius": neighbours, "notice_hours_before": notice_h,
                "notice_rule_m": notify_m, "notice_rule_h": notify_h, "minutes_to_sunset": sunset,
                "shotfirer": person(rng), "supervisor": person(rng), "ts": rand_ts(rng, hours=(10, 16)),
                "code": pick(rng, ["BP-12", "SBP-4", "DBM-9"])}

    def _flags(self, f):
        light = f["lightning_km"] is not None and f["lightning_km"] <= f["lightning_rule_km"]
        people = f["intruder"] is not None or f["sweep"] != "confirmed"
        guard = (any((not st) or (not radio) or d < f["exclusion_m"] for _, d, st, radio in f["posts"])
                 or (f["road_in_zone"] and not f["road_closed"]))
        loading = f["short_stemmed_holes"] > 0
        return light, people, guard, loading

    def _notice_gap_h(self, f):
        """Hours still to wait before the neighbour notice period is met (0 if met or nobody to notify)."""
        if f["neighbours_in_notice_radius"] == 0 or f["notice_hours_before"] is None:
            return 0
        return max(0, f["notice_rule_h"] - f["notice_hours_before"])

    def intended(self, f):
        light, people, guard, loading = self._flags(f)
        sun = f["minutes_to_sunset"]
        gap_h = self._notice_gap_h(f)
        if light:
            call = "stand_down_and_guard"
        elif people:
            call = "stand_down_and_guard" if sun < 30 else "re_sweep_zone"
        elif gap_h * 60 > sun - 20:
            call = "stand_down_and_guard"
        elif guard or loading or gap_h > 0:
            call = "stand_down_and_guard" if sun < 25 else "hold_short_delay"
        else:
            call = "fire_as_scheduled"
        gap = 0
        posts = f["posts"]
        if any(not radio for _, _, _, radio in posts) or any(f["exclusion_m"] * 0.97 <= d < f["exclusion_m"] for _, d, _, _ in posts):
            gap = 1
        if any(not st for _, _, st, _ in posts) or any(d < f["exclusion_m"] * 0.97 for _, d, _, _ in posts) \
                or (f["road_in_zone"] and not f["road_closed"]):
            gap = 2
        if people:
            gap = 3
        fly = 0
        ratio = f["structure_m"] / f["exclusion_m"]
        if f["short_stemmed_holes"]:
            fly = 2 if ratio >= 1.2 else 3
            if f["short_stemmed_holes"] == 1 and ratio >= 2:
                fly = 1
        else:
            fly = 0 if ratio >= 2 else 1
        notices = (f["neighbours_in_notice_radius"] == 0
                   or (f["notice_hours_before"] is not None and f["notice_hours_before"] >= f["notice_rule_h"]))
        return {"firing_call": call, "clearance_gap_severity": gap, "flyrock_exposure": fly,
                "exclusion_zone_met": "yes" if not any((not st) or d < f["exclusion_m"] for _, d, st, _ in posts) else "no",
                "required_notices_complete": "yes" if notices else "no"}

    def _rule(self, f):
        return (f"{f['code']}: guards at >= {f['exclusion_m']} m; no firing with lightning within "
                f"{f['lightning_rule_km']} km; neighbours within {f['notice_rule_m']} m notified >= "
                f"{f['notice_rule_h']} h ahead")

    def _stem(self, f):
        if not f["short_stemmed_holes"]:
            return "all holes stemmed to design"
        return f"{f['short_stemmed_holes']} hole(s) under {f['design_stemming_m']} m design stemming"

    def render(self, f, rng, surface):
        ts = f["ts"]
        light = "no strikes on detector" if f["lightning_km"] is None else f"nearest strike {f['lightning_km']} km (last 30 min)"
        if surface == "json":
            return dumps(prune({
                "blast": {"id": f["blast_id"], "site": f["site"], "holes": f["holes"], "explosive_t": f["explosive_t"],
                          "loading": self._stem(f), "fire_time": fmt_ts(ts, style="iso"),
                          "minutes_to_sunset": f["minutes_to_sunset"]},
                "clearance": {"guard_posts": [{"post": p, "distance_m": d, "staffed": st, "radio_check": rc}
                                              for p, d, st, rc in f["posts"]],
                              "sweep": f["sweep"] or "not reported", "sighting": f["intruder"],
                              "public_road_in_zone": f["road_in_zone"], "road_closed": f["road_closed"]},
                "conditions": {"lightning": light,
                               "nearest_structure": f"{f['structure']} at {f['structure_m']} m"},
                "notices": {"neighbours_in_radius": f["neighbours_in_notice_radius"],
                            "notified_hours_before": f["notice_hours_before"]},
                "rule": self._rule(f),
            }))
        if surface == "countdown_log":
            t = ts - timedelta(minutes=rng.randint(25, 45))
            out = [f"{f['site']} blast {f['blast_id']} -- shotfirer {f['shotfirer']}"]
            out.append(f"{t.strftime('%H:%M')} sweep " + {"confirmed": "complete, all clear reported",
                                                       "partial": "east side not completed", None: "status not called in"}[f["sweep"]])
            for p, d, st, rc in f["posts"]:
                t += timedelta(minutes=rng.randint(1, 3))
                out.append(f"{t.strftime('%H:%M')} {p} at {d} m: " + ("in position" if st else "NOT STAFFED")
                           + ("" if (rc or not st) else ", no radio response"))
            if f["intruder"]:
                out.append(f"{t.strftime('%H:%M')} report: {f['intruder']}")
            if f["road_in_zone"]:
                out.append(f"{t.strftime('%H:%M')} road block " + ("in place" if f["road_closed"] else "not yet in place"))
            out.append(f"{t.strftime('%H:%M')} weather: {light}")
            out.append(f"Loading: {self._stem(f)}; nearest {f['structure']} {f['structure_m']} m.")
            out.append(f"Notices: {f['neighbours_in_notice_radius']} neighbour(s) in radius"
                       + (f", notified {f['notice_hours_before']} h before" if f["notice_hours_before"] is not None else ""))
            out.append(f"Scheduled fire {ts.strftime('%H:%M')}, sunset in {f['minutes_to_sunset']} min. {self._rule(f)}.")
            return "\n".join(out)
        rows = [[p, f"{d} m", "yes" if st else "no", "ok" if rc else "-"] for p, d, st, rc in f["posts"]]
        return "\n".join([
            f"PRE-FIRING CHECKLIST  {f['blast_id']}  {f['site']}  {fmt_date(ts.date(), rng)} {ts.strftime('%H:%M')}",
            f"Exclusion zone required: {f['exclusion_m']} m   Lightning stand-down: {f['lightning_rule_km']} km",
            table(["post", "distance", "staffed", "radio"], rows, rng),
            _kv([("Sweep", f["sweep"] or "not recorded"), ("Sightings", f["intruder"] or "none"),
                      ("Road in zone", ("yes, closed" if f["road_closed"] else "yes, NOT closed") if f["road_in_zone"] else "no"),
                      ("Lightning", light), ("Loading", self._stem(f)),
                      ("Nearest structure", f"{f['structure']}, {f['structure_m']} m"),
                      ("Neighbour notice", f"{f['neighbours_in_notice_radius']} in {f['notice_rule_m']} m"
                       + (f", sent {f['notice_hours_before']} h ahead (rule {f['notice_rule_h']} h)" if f["notice_hours_before"] is not None else "")),
                      ("Minutes to sunset", f["minutes_to_sunset"])], rng),
        ])

    def questions(self):
        return {
            "firing_call": choice("What should the shotfirer do with this blast?", {
                "fire_as_scheduled": "Fire at the scheduled time.",
                "hold_short_delay": "Hold the countdown, close the specific gap, then fire this shift.",
                "re_sweep_zone": "Stop the countdown and sweep the whole exclusion zone again before anything else.",
                "stand_down_and_guard": "Stand the blast down and keep the loaded pattern guarded until it can be fired safely.",
            }),
            "clearance_gap_severity": score("How serious are the gaps in clearing the exclusion zone?", [
                "None: every post is staffed at distance and the sweep is complete.",
                "Minor: a check-in is late or a post sits just inside the line.",
                "Significant: a post or road block is missing or well short of the required distance.",
                "Critical: someone may still be inside the zone.",
            ]),
            "flyrock_exposure": score("If fired now, how exposed is anything outside the pit to flyrock?", [
                "Negligible: loading to design and the nearest structure far beyond the zone.",
                "Low: loading to design, structures not far beyond the zone.",
                "Elevated: under-stemmed holes facing a structure outside the zone.",
                "High: under-stemmed holes with a structure close to or inside the zone.",
            ]),
            "exclusion_zone_met": noul(
                "Every guard post is staffed at or beyond the blast plan's exclusion distance.",
                yes="All posts are staffed and none is closer than the required distance.",
                no="A post is unstaffed or sits inside the required distance."),
            "required_notices_complete": noul(
                "Every neighbour notice the blast plan requires went out in time."),
        }


# =========================================================================== schema 4
OP_PHRASE = {"drilling": "drilling ahead", "connection": "making a connection", "tripping_out": "tripping out"}
SURFACE_CAUSES = {"transfer": "mud transfer from the reserve tank into the active system",
                  "dump": "shakers bypassed and sand trap dumped",
                  "sensor": "pit-volume sensor recalibrated mid-tour"}


class WellKickIndicators(Schema):
    name = "well_kick_indicators"
    description = "Kick indicators on a drilling rig (pit gain, flow, fill, gas) judged against the well-control standard's shut-in rule."
    surfaces = {"json": 0.5, "driller_log": 0.25, "rig_chat": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        op = pick(rng, ["drilling", "drilling", "connection", "tripping_out"])
        G = pick(rng, [5, 10])
        flow_rule = pick(rng, [10, 15, 20])
        trip_rule = pick(rng, [2, 3, 5])
        tol = pick(rng, [15, 20, 25, 30, 40])
        gain = round(rng.uniform(-1.0, G * 0.6), 1)
        flow = rng.randint(-5, int(flow_rule * 0.7))
        brk = op == "drilling" and chance(rng, 0.2)
        deficit = round(rng.uniform(0, trip_rule * 0.6), 1) if op == "tripping_out" else None
        check = pick_w(rng, {"not_done": 5, "static": 3, "flowing": 1, "slight_flow_tapering": 1})
        surf = pick_w(rng, {None: 6, "transfer": 2, "dump": 1, "sensor": 1})
        surf_bbl = None
        balloon = chance(rng, 0.25)
        bg = rng.randint(15, 80)
        gas = bg + rng.randint(0, 60)
        k = None
        if reg == "clear":
            if chance(rng, 0.5):
                gain, flow, brk, check, surf = round(rng.uniform(-0.5, 1.0), 1), rng.randint(-3, 3), False, pick(rng, ["static", "not_done"]), None
                deficit = round(rng.uniform(0, 0.5), 1) if op == "tripping_out" else None
            else:
                gain, flow, check = round(G * rng.uniform(1.3, 3.0), 1), rng.randint(flow_rule, flow_rule + 30), "flowing"
        elif reg == "borderline":
            k = pick(rng, ["gain", "flow", "trip"]) if op == "tripping_out" else pick(rng, ["gain", "flow", "break"])
            if k == "gain":
                gain = near(rng, G, G * 0.25, lo=0, digits=1)
                check = pick(rng, ["not_done", "not_done", "static"])
            elif k == "flow":
                flow = near(rng, flow_rule, 4, integer=True)
                check = "not_done"
            elif k == "trip":
                deficit = near(rng, trip_rule, 1.2, lo=0, digits=1)
                check = "not_done"
            else:
                brk, check = True, "not_done"
                gain = round(rng.uniform(0, G * 0.5), 1)
        else:
            k = pick(rng, ["transfer_masks", "ballooning", "gas_no_gain", "taper_flow_dismissed"])
            if k == "transfer_masks":
                surf = "transfer"
                gain = round(G * rng.uniform(0.9, 2.2), 1)
                surf_bbl = round(gain - rng.choice([rng.uniform(-1.5, 1.5), rng.uniform(3, 8)]), 1)
                check = "not_done"
            elif k == "ballooning":
                op, balloon = "connection", True
                gain, check = round(rng.uniform(1, G * 1.2), 1), pick(rng, ["slight_flow_tapering", "static"])
            elif k == "gas_no_gain":
                gain, gas, check = round(rng.uniform(-0.5, 1.5), 1), bg + rng.randint(150, 600), "static"
            else:
                check, balloon = "slight_flow_tapering", False
                gain = round(rng.uniform(1, G * 0.8), 1)
        if surf == "transfer" and surf_bbl is None:
            surf_bbl = round(max(0.5, gain + rng.uniform(-1.0, 1.0)), 1)
        if op != "drilling":
            brk = False
        if op != "tripping_out":
            deficit = None
        return {"regime": reg, "case_kind": k, "rig": f"Rig {rng.randint(11, 98)}",
                "well": f"{pick(rng, MINE_NAMES)} {rng.randint(1, 40)}-{rng.randint(1, 36)}H",
                "operation": op, "depth_ft": rng.randrange(6500, 17500, 10), "mud_ppg": round(rng.uniform(9.2, 14.8), 1),
                "pit_gain_bbl": gain, "flow_out_change_pct": flow, "drilling_break": brk, "trip_deficit_bbl": deficit,
                "flow_check": check, "surface_cause": surf, "surface_bbl": surf_bbl, "ballooning_history": balloon,
                "background_gas_u": bg, "max_gas_u": gas, "gain_rule_bbl": G, "flow_rule_pct": flow_rule,
                "trip_rule_bbl": trip_rule, "kick_tolerance_bbl": tol, "driller": person(rng), "company_man": person(rng),
                "ts": rand_ts(rng), "note": pick(rng, ["balloon", "trust", "rush", "none", "none"]),
                "spp_psi": rng.randrange(1800, 4200, 10), "spp_change_psi": rng.randint(-60, 20),
                "rop_fph": rng.randint(25, 140),
                "std": pick(rng, ["WCS-7", "WCM-3", "DWC-12"])}

    def _unexplained(self, f):
        g = f["pit_gain_bbl"]
        if f["surface_cause"] == "transfer" and f["surface_bbl"] is not None:
            g -= f["surface_bbl"]
        elif f["surface_cause"] in ("dump", "sensor"):
            g -= 0  # a dump or recalibration does not add mud; it only muddies the reading
        return round(g, 1)

    def intended(self, f):
        G = f["gain_rule_bbl"]
        ug = self._unexplained(f)
        chk = f["flow_check"]
        flow_hit = f["flow_out_change_pct"] >= f["flow_rule_pct"]
        trip_hit = f["trip_deficit_bbl"] is not None and f["trip_deficit_bbl"] >= f["trip_rule_bbl"]
        gas_high = f["max_gas_u"] >= 3 * f["background_gas_u"] and f["max_gas_u"] >= 120
        rule = chk in ("flowing", "slight_flow_tapering") or (ug >= G and chk != "static")
        if rule:
            act = "shut_in_now"
        elif chk == "not_done" and (flow_hit or trip_hit or f["drilling_break"] or ug >= 0.5 * G):
            act = "flow_check_first"
        elif gas_high or (ug >= G and chk == "static"):
            act = "circulate_bottoms_up"
        else:
            act = "continue_with_watch"
        if f["surface_cause"] == "transfer" and abs(ug) <= 2 and chk != "flowing":
            expl = "surface_mud_handling"
        elif chk == "flowing" or ug >= G or (f["drilling_break"] and flow_hit) or (trip_hit and chk == "not_done"):
            expl = "formation_influx"
        elif f["ballooning_history"] and chk in ("slight_flow_tapering", "static") and f["operation"] == "connection":
            expl = "wellbore_ballooning"
        elif f["surface_cause"] in ("dump", "sensor"):
            expl = "surface_mud_handling"
        elif chk == "slight_flow_tapering" or flow_hit or gas_high:
            expl = "formation_influx"
        else:
            expl = "measurement_noise"
        if chk == "flowing" and (ug >= G or flow_hit):
            lik = 4
        elif (chk == "flowing" or (ug >= G and chk == "not_done") or (f["drilling_break"] and flow_hit)
              or (trip_hit and ug >= 0.5 * G) or (chk == "slight_flow_tapering" and not f["ballooning_history"])):
            lik = 3
        elif chk == "slight_flow_tapering" or flow_hit or trip_hit or f["drilling_break"] or ug >= 0.5 * G or gas_high:
            lik = 2
        elif ug >= 0.25 * G or f["flow_out_change_pct"] >= 0.5 * f["flow_rule_pct"]:
            lik = 1
        else:
            lik = 0
        r = max(ug, 0) / f["kick_tolerance_bbl"]
        margin = 0 if r < 0.08 else 1 if r < 0.25 else 2 if r < 0.5 else 3
        if chk == "flowing" and margin < 3:
            margin += 1
        return {"well_control_action": act, "gain_explanation": expl, "influx_likelihood": lik,
                "kick_tolerance_concern": margin, "shut_in_rule_triggered": "yes" if rule else "no"}

    def _rule(self, f):
        s = (f"{f['std']} 3.1: shut in on any flow on a flow check, or an unexplained pit gain >= {f['gain_rule_bbl']} bbl; "
             f"flow-check on flow-out up >= {f['flow_rule_pct']}% or a drilling break")
        if f["operation"] == "tripping_out":
            s += f"; trip: flow-check if hole fill is short by >= {f['trip_rule_bbl']} bbl"
        return s + f". Kick tolerance this section: {f['kick_tolerance_bbl']} bbl."

    def _check_txt(self, v):
        return {"not_done": "not done yet", "static": "static (well not flowing)", "flowing": "well flowing",
                "slight_flow_tapering": "slight flow, tapering off"}[v]

    def _note(self, f, rng):
        t = {"balloon": "Probably the formation giving back mud like last connection.",
             "trust": "Mud loggers say gas is just connection gas.",
             "rush": "We're close to section TD, want to get there before the bit change.", "none": ""}[f["note"]]
        return noisy(t, rng, 0.15) if t else None

    def render(self, f, rng, surface):
        note = self._note(f, rng)
        sc = SURFACE_CAUSES.get(f["surface_cause"])
        if sc and f["surface_cause"] == "transfer" and f["surface_bbl"] is not None:
            sc += f" (~{f['surface_bbl']} bbl)"
        if surface == "json":
            return dumps(prune({
                "well": {"rig": f["rig"], "well": f["well"], "operation": f["operation"], "depth_ft": f["depth_ft"],
                         "mud_weight_ppg": f["mud_ppg"], "standpipe_psi": f["spp_psi"],
                         "rop_fph": f["rop_fph"] if f["operation"] == "drilling" else None},
                "indicators": {"pit_gain_bbl": f["pit_gain_bbl"], "flow_out_change_pct": f["flow_out_change_pct"],
                               "drilling_break": f["drilling_break"] if f["operation"] == "drilling" else None,
                               "hole_fill_short_bbl": f["trip_deficit_bbl"],
                               "gas_units": {"background": f["background_gas_u"], "max": f["max_gas_u"]},
                               "flow_check": self._check_txt(f["flow_check"])},
                "context": {"surface_activity": sc, "losses_or_ballooning_seen_this_section": f["ballooning_history"],
                            "driller_comment": note},
                "standard": self._rule(f),
            }))
        if surface == "driller_log":
            t = f["ts"]
            out = [f"{f['rig']} / {f['well']}  TOUR LOG  {fmt_date(t.date(), rng)}   depth {f['depth_ft']:,} ft, MW {f['mud_ppg']} ppg"]
            ev = [OP_PHRASE[f["operation"]]]
            if f["drilling_break"]:
                ev.append("drilling break, ROP doubled")
            ev.append(f"PVT gain {f['pit_gain_bbl']} bbl, flow out {f['flow_out_change_pct']:+d}%, SPP {f['spp_psi']} psi "
                      f"({f['spp_change_psi']:+d})")
            if f["trip_deficit_bbl"] is not None:
                ev.append(f"trip tank: hole took {f['trip_deficit_bbl']} bbl less than calculated")
            if sc:
                ev.append(sc)
            ev.append(f"gas {f['background_gas_u']} u bkg, peak {f['max_gas_u']} u")
            ev.append(f"flow check: {self._check_txt(f['flow_check'])}")
            for e in ev:
                out.append(f"{t.strftime('%H:%M')}  {e}")
                t += timedelta(minutes=rng.randint(1, 6))
            if f["ballooning_history"]:
                out.append("Remark: losses / mud returning on connections seen earlier in this section.")
            if note:
                out.append(f"Driller: {note}")
            out.append(self._rule(f))
            return "\n".join(out)
        msgs = [(f["driller"], f"PVT up {f['pit_gain_bbl']} bbl, flow out {f['flow_out_change_pct']:+d}% while "
                               f"{OP_PHRASE[f['operation']]} at {f['depth_ft']:,} ft")]
        if f["drilling_break"]:
            msgs.append((f["driller"], "had a drilling break just before"))
        if f["trip_deficit_bbl"] is not None:
            msgs.append((f["driller"], f"hole fill short {f['trip_deficit_bbl']} bbl over the last stands"))
        if sc:
            msgs.append((f["driller"], f"derrickhand says: {sc}"))
        msgs.append((f["company_man"], f"flow check? gas?"))
        msgs.append((f["driller"], f"{self._check_txt(f['flow_check'])}. gas peak {f['max_gas_u']} u on {f['background_gas_u']} bkg"))
        if f["ballooning_history"]:
            msgs.append((f["company_man"], "we saw the hole give mud back on connections yesterday"))
        if note:
            msgs.append((f["driller"], note))
        t = f["ts"]
        out = []
        for who, m in msgs:
            out.append(f"{t.strftime('%H:%M')} {_first(who)}: {m}")
            t += timedelta(minutes=rng.randint(1, 3))
        return f"#{f['rig'].lower().replace(' ', '-')}-ops\n" + "\n".join(out) + f"\n[pinned] {self._rule(f)}"

    def questions(self):
        return {
            "well_control_action": choice("What should the driller do now?", {
                "circulate_bottoms_up": "Circulate bottoms-up through the gas buster with the well watched closely.",
                "continue_with_watch": "Carry on with the operation and keep watching the indicators.",
                "flow_check_first": "Stop, space out, pumps off and flow-check before deciding anything else.",
                "shut_in_now": "Shut the well in now and record the pressures.",
            }),
            "gain_explanation": choice("What best explains the indicators?", {
                "formation_influx": "Formation fluid entering the wellbore.",
                "measurement_noise": "Normal fluctuation; nothing out of the ordinary is happening.",
                "surface_mud_handling": "Mud moved, dumped or re-measured at surface rather than anything downhole.",
                "wellbore_ballooning": "The formation giving back mud it took earlier.",
            }),
            "influx_likelihood": score("How likely is it that an influx is under way?", [
                "Very unlikely: indicators are flat.",
                "Unlikely: one weak sign that is probably explained.",
                "Possible: a trigger or an unexplained sign that needs checking.",
                "Likely: strong signs without a confirming check.",
                "Confirmed: the well flows on a check with a real gain.",
            ]),
            "kick_tolerance_concern": score("How much of the section's kick tolerance could the possible influx already use up?", [
                "Negligible: a small fraction at most.",
                "Moderate: well under half.",
                "Large: around half or more.",
                "Critical: near or past the tolerance.",
            ]),
            "shut_in_rule_triggered": noul(
                "The well-control standard requires the well to be shut in on these indicators.",
                yes="A flow check showed flow, or the unexplained gain meets the shut-in volume.",
                no="Neither the flow-check result nor the unexplained gain meets the shut-in rule."),
        }


# =========================================================================== schema 5
DAM_SIGNS = {"none": 0, "minor crest cracking": 1, "wet patch on the downstream face": 1,
             "depression on the crest": 2, "sand boil at the toe": 2}


class TailingsDamAlert(Schema):
    name = "tailings_dam_alert"
    description = "A tailings dam's surveillance readings (piezometers, seepage, freeboard, beach) against its trigger-action response plan."
    surfaces = {"json": 0.5, "surveillance_report": 0.25, "rtfe_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        n = rng.randint(2, 3)
        piezos = []
        for i in range(n):
            alert = round(rng.uniform(380, 460), 1)
            alarm = round(alert + pick(rng, [0.8, 1.0, 1.5]), 1)
            piezos.append([f"VWP-{rng.randint(1, 24):02d}", round(alert - rng.uniform(0.3, 2.5), 1), alert, alarm, "ok"])
        seep_base = round(rng.uniform(0.8, 6.0), 1)
        seep = round(seep_base * rng.uniform(0.85, 1.3), 1)
        clarity = pick_w(rng, {"clear": 7, "slightly cloudy": 2, "turbid": 1})
        fb_min = pick(rng, [1.0, 1.5, 2.0])
        fb = round(fb_min + rng.uniform(0.3, 1.5), 2)
        rise = round(rng.uniform(0.05, 0.6), 2)
        beach_min = pick(rng, [50, 80, 100])
        beach = int(beach_min * rng.uniform(1.0, 1.8))
        sign = pick_w(rng, {"none": 6, "minor crest cracking": 1, "wet patch on the downstream face": 1,
                            "depression on the crest": 0.3, "sand boil at the toe": 0.3})
        k = None
        if reg == "clear":
            if chance(rng, 0.55):
                clarity, sign = "clear", "none"
                seep = round(seep_base * rng.uniform(0.9, 1.1), 1)
            else:
                i = rng.randrange(n)
                piezos[i][1] = round(piezos[i][3] + rng.uniform(0.2, 1.0), 1)
                clarity, seep = "turbid", round(seep_base * rng.uniform(1.6, 2.5), 1)
                sign = pick(rng, ["depression on the crest", "sand boil at the toe"])
        elif reg == "borderline":
            k = pick(rng, ["piezo", "freeboard", "beach", "seep"])
            i = rng.randrange(n)
            if k == "piezo":
                thr = pick(rng, [piezos[i][2], piezos[i][3]])
                piezos[i][1] = near(rng, thr, 0.25, digits=1)
            elif k == "freeboard":
                fb = near(rng, fb_min + rise, 0.2, lo=0.2)
            elif k == "beach":
                beach = near(rng, beach_min, beach_min * 0.1, lo=5, integer=True)
            else:
                seep = round(seep_base * rng.uniform(1.35, 1.7), 1)
                clarity = pick(rng, ["clear", "slightly cloudy"])
        else:
            k = pick(rng, ["faulty_high_piezo", "clear_but_rising", "storm_coming", "sign_only"])
            i = rng.randrange(n)
            if k == "faulty_high_piezo":
                piezos[i][1] = round(piezos[i][3] + rng.uniform(0.5, 3.0), 1)
                piezos[i][4] = pick(rng, ["erratic since lightning strike", "reads high after cable repair"])
            elif k == "clear_but_rising":
                seep = round(seep_base * rng.uniform(1.6, 2.6), 1)
                clarity = "clear"
                piezos[i][1] = round(piezos[i][2] + rng.uniform(-0.2, 0.6), 1)
            elif k == "storm_coming":
                fb = round(fb_min + rng.uniform(0.1, 0.5), 2)
                rise = round(rng.uniform(0.3, 0.9), 2)
            else:
                sign = pick(rng, ["depression on the crest", "sand boil at the toe", "wet patch on the downstream face"])
        return {"regime": reg, "case_kind": k, "facility": f"{pick(rng, MINE_NAMES)} TSF {pick(rng, ['1', '2', 'North', 'South'])}",
                "raise_method": pick(rng, ["upstream", "centreline", "downstream"]), "piezos": piezos,
                "seep_lps": seep, "seep_baseline_lps": seep_base, "clarity": clarity, "freeboard_m": fb,
                "freeboard_min_m": fb_min, "forecast_rise_m": rise, "forecast_mm": int(rise * rng.uniform(140, 260)),
                "beach_m": beach, "beach_min_m": beach_min, "sign": sign,
                "spigots": pick(rng, ["north wall", "east wall", "south wall", "west abutment"]),
                "inspector": person(rng), "rtfe": person(rng), "eor": org_like(rng), "date": rand_ts(rng),
                "manual": pick(rng, ["TSM-6", "TARP-T3", "TSF-OMS 4"]),
                "opinion": pick(rng, ["ops_fine", "rain_worry", "none", "none"])}

    def _state(self, f):
        rel = [p for p in f["piezos"] if p[4] == "ok"]
        alarm = any(p[1] >= p[3] for p in rel)
        alert = any(p[1] >= p[2] for p in rel)
        rise_pct = f["seep_lps"] / f["seep_baseline_lps"] - 1
        turbid = f["clarity"] == "turbid"
        sev = DAM_SIGNS[f["sign"]]
        margin = f["freeboard_m"] - f["freeboard_min_m"] - f["forecast_rise_m"]
        return alarm, alert, rise_pct, turbid, sev, margin

    def intended(self, f):
        alarm, alert, rise_pct, turbid, sev, margin = self._state(f)
        below_min = f["freeboard_m"] < f["freeboard_min_m"]
        beach_short = f["beach_m"] < f["beach_min_m"]
        if turbid and (alarm or sev == 2) and rise_pct >= 0.3:
            lvl = "red_emergency_response"
        elif alarm or turbid or below_min or margin < 0 or sev == 2:
            lvl = "orange_engineering_response"
        elif alert or rise_pct >= 0.5 or beach_short or margin < 0.3 or sev == 1 or f["clarity"] == "slightly cloudy":
            lvl = "yellow_heightened_surveillance"
        else:
            lvl = "green_normal"
        if lvl == "red_emergency_response" or alarm or turbid or sev == 2:
            dep = "stop_all_deposition"
        elif below_min or margin < 0.3:
            dep = "reduce_and_lower_pond"
        elif beach_short or alert:
            dep = "relocate_discharge"
        else:
            dep = "continue_current_plan"
        sc = 0
        if rise_pct >= 0.2 or f["clarity"] == "slightly cloudy" or f["sign"] == "wet patch on the downstream face":
            sc = 1
        if rise_pct >= 0.5 or (rise_pct >= 0.2 and f["clarity"] == "slightly cloudy"):
            sc = 2
        if turbid or f["sign"] == "sand boil at the toe":
            sc = 3
        sm = 0 if margin < 0 else 1 if margin < 0.3 else 2 if margin < 0.8 else 3
        return {"tarp_level": lvl, "deposition_instruction": dep, "seepage_concern": sc, "storm_margin": sm,
                "piezometer_alert_exceeded": "yes" if alert else "no"}

    def _rule(self, f):
        return (f"{f['manual']}: freeboard >= {f['freeboard_min_m']} m after the design storm; beach >= "
                f"{f['beach_min_m']} m; piezometer at alert = yellow, at alarm = orange; turbid seepage = orange")

    def _opinion(self, f, rng):
        t = {"ops_fine": f"Operations would like to keep the {f['spigots']} spigots running to hold throughput.",
             "rain_worry": "Plant water balance is already tight before this rain.", "none": ""}[f["opinion"]]
        return noisy(t, rng, 0.15) if t else None

    def render(self, f, rng, surface):
        op = self._opinion(f, rng)
        pz_rows = [[p[0], p[1], p[2], p[3], p[4]] for p in f["piezos"]]
        if surface == "json":
            return dumps(prune({
                "facility": {"name": f["facility"], "raise_method": f["raise_method"], "active_spigots": f["spigots"],
                             "inspection_date": fmt_date(f["date"].date(), style="iso")},
                "piezometers_m_rl": [{"id": p[0], "reading": p[1], "alert": p[2], "alarm": p[3], "status": p[4]}
                                     for p in f["piezos"]],
                "seepage": {"toe_weir_lps": f["seep_lps"], "baseline_lps": f["seep_baseline_lps"], "clarity": f["clarity"]},
                "pond": {"freeboard_m": f["freeboard_m"], "forecast_mm_72h": f["forecast_mm"],
                         "modelled_rise_m": f["forecast_rise_m"], "beach_width_m": f["beach_m"]},
                "visual": f["sign"], "ops_note": op, "tarp": self._rule(f),
            }))
        if surface == "surveillance_report":
            return "\n".join([
                f"{f['facility']} -- weekly surveillance, {fmt_date(f['date'].date(), rng)} ({f['inspector']})",
                table(["piezometer", "m RL", "alert", "alarm", "status"], pz_rows, rng),
                _kv([("Toe weir", f"{f['seep_lps']} L/s (baseline {f['seep_baseline_lps']}), {f['clarity']}"),
                          ("Freeboard", f"{f['freeboard_m']} m; 72 h forecast {f['forecast_mm']} mm, model rise {f['forecast_rise_m']} m"),
                          ("Beach width", f"{f['beach_m']} m"), ("Visual", f["sign"]),
                          ("Deposition", f"spigots on {f['spigots']}")], rng),
                f"Ops: {op}" if op else "", self._rule(f)]).strip()
        pz = "; ".join(f"{p[0]} {p[1]} (alert {p[2]}, alarm {p[3]}" + ("" if p[4] == "ok" else f", {p[4]}") + ")"
                       for p in f["piezos"])
        s = [f"Piezometers: {pz}.",
             f"Toe weir {f['seep_lps']} L/s against a {f['seep_baseline_lps']} L/s baseline, {f['clarity']}.",
             f"Freeboard {f['freeboard_m']} m with {f['forecast_mm']} mm forecast over 72 h (model says +{f['forecast_rise_m']} m).",
             f"Beach {f['beach_m']} m; spigots on the {f['spigots']}.",
             (f"Also noted: {f['sign']}." if f["sign"] != "none" else "No new visual issues."), op or ""]
        return "\n".join([f"Subject: {f['facility']} surveillance -- {fmt_date(f['date'].date(), rng)}",
                          f"From: {f['inspector']}", f"To: {f['rtfe']} (RTFE); cc {f['eor']}", "",
                          join_sentences(s), "", f"Triggers per {self._rule(f)}."])

    def questions(self):
        return {
            "tarp_level": choice("Which TARP level should the dam be placed at?", {
                "green_normal": "Green: normal operation and routine surveillance.",
                "orange_engineering_response": "Orange: notify the engineer of record and start contingency measures.",
                "red_emergency_response": "Red: activate the emergency plan, including warning people downstream.",
                "yellow_heightened_surveillance": "Yellow: more frequent readings and inspections, responsible engineer informed.",
            }),
            "deposition_instruction": choice("What should happen to tailings deposition?", {
                "continue_current_plan": "Keep depositing from the current spigots.",
                "reduce_and_lower_pond": "Cut deposition and pump water off the pond to recover freeboard.",
                "relocate_discharge": "Move discharge to other spigots to rebuild the beach away from the problem area.",
                "stop_all_deposition": "Stop all deposition into the facility until engineers clear it.",
            }),
            "seepage_concern": score("How concerning is the seepage picture?", [
                "Low: flow near baseline and clear.",
                "Notable: flow somewhat up, cloudiness or a wet patch.",
                "High: flow well above baseline.",
                "Severe: turbid flow or a sand boil suggesting internal erosion.",
            ]),
            "storm_margin": score("After the forecast rain, how much margin will the pond keep above its minimum freeboard?", [
                "None: the forecast storm would take it below the minimum.",
                "Thin: less than about 0.3 m to spare.",
                "Adequate: some room to spare.",
                "Ample: comfortably above the minimum.",
            ]),
            "piezometer_alert_exceeded": noul(
                "A piezometer that can be relied on is reading at or above its alert level.",
                yes="At least one working piezometer is at or above its alert level.",
                no="All working piezometers are below alert; any high reading comes from a faulty instrument."),
        }


def org_like(rng):
    """An invented engineer-of-record consultancy."""
    return f"{pick(rng, ['Halden', 'Corrie', 'Strand', 'Merrow', 'Tavish', 'Ebbing'])} {pick(rng, ['Geotechnical', 'Dam Engineering', 'Geo Consultants'])}"


WORKFLOW = Workflow(
    name="mining_and_drilling_operations",
    title="Mining and drilling operations",
    description="Safety calls in underground and surface mines and on drilling rigs, against fictional site standards and TARPs.",
    domain="mining, quarrying and oil and gas drilling (NAICS 21)",
    schemas=[UndergroundGasWithdrawal(), GroundControlTrigger(), BlastFiringClearance(), WellKickIndicators(),
             TailingsDamAlert()],
    notes="Standards, sites and people are fictional; thresholds are sampled per case. Midstream pipelines are excluded "
          "(pipeline_integrity_dig_decision).",
)
