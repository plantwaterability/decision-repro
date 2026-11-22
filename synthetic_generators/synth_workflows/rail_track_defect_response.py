"""Rail track defect response: how a (fictional) railway's track engineers respond to a track fault, judged
against the railway's own written track standards.

Track-geometry exceedances from a recording vehicle, ultrasonic rail-flaw findings, hot-weather buckling
precautions, foot-patrol findings and requests to lift a post-works speed restriction. The questions are about
speed restrictions, train holds, remediation timing and verification, always against an extract of the standard
quoted in the state, so every case is decidable in principle. Railways, routes, places, people, recording
vehicles and every limit in the standards (TS-410, RD-220, HW-115, TP-305, TW-520) are invented for this
generator; they are plausible in shape but are not any real railway's figures. Aircraft maintenance and freight
exceptions live in other workflows; nothing here is a customer conversation.

Each schema draws a `regime` first: 'clear' cases sit well inside one response band, 'borderline' cases put a
measured value or a tonnage right at a limit in the quoted table, and 'conflicting' cases pair the numbers with
an opinion or a data-quality flag that points the other way (a section manager calling a reading a glitch, a
contractor rating a defect low risk) or leave a key record missing. Every schema also defines `intended(f)`,
the answer the quoted rules give for each question; it documents the logic and drives the spread check.
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
    near,
    noisy,
    noul,
    org,
    person,
    pick,
    pick_w,
    prune,
    rand_ts,
    regime,
    score,
    table,
)

# =========================================================================== shared pools
# (railway, abbreviation, units)
RAILWAYS = [
    ("Wrexford Vale Railway", "WVR", "imperial"), ("Fenmoor Regional Rail", "FRR", "imperial"),
    ("Kestrel Coast Lines", "KCL", "imperial"), ("Northfold Rail Infrastructure", "NRI", "metric"),
    ("Lowther Fell Railway", "LFR", "imperial"), ("Saltmere Junction Railways", "SJR", "metric"),
    ("Brackenridge Rail Network", "BRN", "metric"), ("Ostwick & Harl Railway", "OHR", "metric"),
    ("Corrie Moss Railway", "CMR", "imperial"), ("Tarrow Estuary Lines", "TEL", "metric"),
]
PLACES = ["Hollin Cross", "Marlow Fen", "Ketterby", "Brayford Junction", "Corrie Moss", "Dunhallow", "Eskerby",
          "Garthmoor", "Hob Lane", "Inchbrook", "Jessop's Bridge", "Kilnsey Road", "Lanthorn", "Mossgill",
          "Orrin Vale", "Pellow Sidings", "Ravenstall", "Scarth Gap", "Thornyhaugh", "Ulverby", "Wadsley Moor",
          "Yeldham Cross", "Ashlin Parkway", "Bexbury", "Cauldwell", "Dorrington Loop", "Elsmere Road",
          "Fallow Green", "Gilsby", "Harnham Bank", "Ivelet", "Keld Cutting", "Lindale Viaduct", "Merrow Halt",
          "Nunthorpe Fold", "Oxlease", "Penhallow", "Rookwith", "Sallow Beck", "Tolverne", "Wenning Bridge",
          "Arkle Town", "Bramwith Lock", "Coldharbour Lane"]
TRACKS = {"imperial": ["Up Main", "Down Main", "Up Fast", "Down Fast", "Up Slow", "Down Slow", "Single line",
                       "Up Relief", "Down Goods"],
          "metric": ["Track 1", "Track 2", "Track 3", "northbound main", "southbound main", "single line",
                     "eastbound main", "westbound main"]}
# line-speed bands, lowest first
BAND_SPEEDS = {"imperial": [[20, 25, 30, 40], [50, 60, 70, 75], [80, 85, 90, 100], [110, 115, 125]],
               "metric": [[30, 40, 50, 60], [80, 90, 100, 120], [130, 140, 150, 160], [170, 180, 200]]}
BAND_LABEL = {"imperial": ["up to 40 mph", "45-75 mph", "80-100 mph", "105-125 mph"],
              "metric": ["up to 60 km/h", "70-120 km/h", "130-160 km/h", "170-200 km/h"]}
BAND_TOP = {"imperial": [40, 75, 100, 125], "metric": [60, 120, 160, 200]}
CAUTION = {"imperial": 20, "metric": 30}
STEP = {"imperial": 50, "metric": 80}
UNIT = {"imperial": "mph", "metric": "km/h"}
TRAFFIC = {"passenger": "passenger only", "mixed": "mixed passenger and freight",
           "heavy_freight": "heavy freight (25 t axle loads) with some passenger"}


def _route(rng, units=None):
    rw, ab, u = pick(rng, [r for r in RAILWAYS if units is None or r[2] == units])
    a, b = rng.sample(PLACES, 2)
    return {"railway": rw, "abbr": ab, "units": u, "line_code": f"{ab[:2]}{pick(rng, 'MLNRS')}{rng.randint(1, 9)}",
            "section": f"{a} - {b}", "track": pick(rng, TRACKS[u])}


def _mileage(rng, units):
    if units == "imperial":
        return f"{rng.randint(2, 180)}m {rng.randint(0, 79):02d}ch"
    return f"km {rng.randint(2, 290)}.{rng.randint(0, 999):03d}"


def _spd(v, units):
    return f"{v} {UNIT[units]}"


def _curve_text(radius, cant=None):
    if radius is None:
        return "straight"
    return f"{radius} m radius curve" + (f", cant {cant} mm" if cant else "")


def _lvl_name(i):
    return ["Alert", "Intervention", "Immediate Action", "Block"][i]


def _kv(pairs, rng):
    """kv_lines without the snake_case 'equals' style (labels here carry slashes and commas)."""
    return kv_lines(pairs, rng, style=pick(rng, ["colon", "colon", "aligned", "dash"]))


def _std(rng, ref, clauses, json_ok=False):
    """The standard's extract in one of several presentations: prose, numbered clauses or bullets. For the json
    surface (json_ok) it is either a clause list or a single string."""
    style = pick(rng, ["prose", "numbered", "bullets", "prose"])
    if json_ok:
        return list(clauses) if style in ("numbered", "bullets") else f"{ref} extract. " + " ".join(clauses)
    if style == "numbered":
        return f"{ref} (extract)\n" + "\n".join(f"{ref.split('-')[1]}.{i} {c}" for i, c in enumerate(clauses, 1))
    if style == "bullets":
        return f"Extract from {ref}:\n" + bullets(clauses, rng)
    return f"{ref} extract. " + " ".join(clauses)


def _ucfirst(t):
    return t[:1].upper() + t[1:] if t else t


def _plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


# =========================================================================== schema 1
# Geometry limits by parameter: one row per line-speed band (lowest band first): (alert, intervention, IA, block), mm
GEOM = {
    "twist": ("twist, 3 m base", [(15, 22, 27, 30), (12, 18, 22, 27), (10, 14, 18, 24), (8, 11, 14, 22)]),
    "gauge": ("gauge spread over nominal", [(19, 28, 33, 38), (15, 23, 28, 33), (12, 18, 23, 29), (10, 14, 18, 26)]),
    "top": ("vertical top, 35 m chord", [(17, 26, 31, 35), (14, 21, 26, 31), (11, 16, 21, 27), (9, 13, 16, 24)]),
    "alignment": ("lateral alignment, 35 m chord", [(16, 25, 30, 34), (13, 20, 25, 30), (11, 15, 20, 26),
                                                     (9, 12, 15, 23)]),
}
DATA_FLAGS = ["channel dropout on the left vertical sensor", "recording speed under 15 mph through the site",
              "lateral sensor saturation", "position mismatch against the line reference", "laser obscured by spray",
              "inertial unit reset just before the site"]
ARTEFACT_SITES = ["bridge transition", "switch toe", "level crossing", "station platform ramp",
                  "under-bridge deck end", "rail joint"]
VEHICLES = ["Wayfarer", "Lodestar", "Merlin", "Pathfinder", "Sentinel", "Kittiwake", "Curlew", "Osprey", "Harrier",
            "Nightjar"]


class GeometryExceedance(Schema):
    name = "geometry_exceedance"
    description = ("A track-recording run reports a geometry exceedance: what restriction, verification and repair "
                   "deadline the track-geometry standard calls for.")
    surfaces = {"json": 0.5, "trc_report": 0.25, "control_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rt = _route(rng)
        units = rt["units"]
        band = pick(rng, [0, 1, 2, 3], [2, 3, 3, 2])
        speed = pick(rng, BAND_SPEEDS[units][band])
        param = pick(rng, list(GEOM))
        row = GEOM[param][1][band]
        if reg == "clear":
            lvl = pick(rng, [0, 0, 1, 1, 2, 3])
            lo, hi = ([row[0]] + list(row[1:]) + [row[3] + 6])[lvl], ([row[0]] + list(row[1:]) + [row[3] + 6])[lvl + 1]
            value = round(between(rng, lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)), 1)
        elif reg == "borderline":
            cands = [row[1], row[2], row[3], 0.9 * row[2]] + [GEOM[param][1][b][1] for b in range(band)]
            value = near(rng, pick(rng, cands), 1.4, lo=row[0] + 0.2, digits=1)
        else:
            lvl = pick(rng, [0, 1, 1, 2, 2, 3])
            edges = [row[0], row[1], row[2], row[3], row[3] + 6]
            value = round(between(rng, edges[lvl] + 0.1, edges[lvl + 1] - 0.2), 1)
        radius = pick_w(rng, {None: 4, 450: 1, 600: 2, 800: 2, 1200: 2, 2000: 1})
        cant = None if radius is None else pick(rng, [40, 65, 90, 110, 140])
        raw = self._level(param, band, value)
        p_second = (0.8 if raw <= 1 else 0.3) * (0.7 if reg == "clear" else 1.0)
        second = None
        if chance(rng, p_second):
            p2 = pick(rng, [p for p in GEOM if p != param])
            lvl2 = pick_w(rng, {0: 4, 1: 5, 2: 1})
            dist = (near(rng, 50, 7, lo=20, integer=True) if reg == "borderline" and chance(rng, 0.6)
                    else pick(rng, [9, 14, 18, 23, 26, 31, 34, 38, 41, 45, 63, 80, 110, 150]))
            second = {"param": p2, "level": lvl2, "distance_m": dist}
        flag = None
        if chance(rng, {"clear": 0.12, "borderline": 0.35, "conflicting": 0.6}[reg]):
            flag = pick(rng, DATA_FLAGS)
        prior_weeks = pick(rng, [2, 4, 4, 8, 13])
        prior = None if chance(rng, 0.35) else round(value * rng.uniform(0.55, 0.97), 1)
        drivers = pick_w(rng, {0: 8, 1: 2, 2: 1}) if reg != "clear" else pick(rng, [0, 0, 0, 0, 0, 1])
        site_feature = pick(rng, ARTEFACT_SITES) if chance(rng, 0.45) else None
        opinion = None
        if reg == "conflicting" or chance(rng, 0.2):
            opinion = pick(rng, ["glitch", "rides_fine", "known_site", "genuine", "worse_than_shown"])
        return {
            "regime": reg, **rt, "mileage": _mileage(rng, units), "line_speed": speed, "band": band,
            "parameter": param, "value_mm": value, "curve_radius_m": radius, "cant_mm": cant,
            "traffic": pick_w(rng, {"passenger": 4, "mixed": 4, "heavy_freight": 2}),
            "fastenings": pick_w(rng, {"good": 6, "clips_missing": 2, "timber_decayed": 2}),
            "second_fault": second, "data_flag": flag, "prior_run_weeks": prior_weeks, "prior_value_mm": prior,
            "rough_ride_reports_72h": drivers, "site_feature": site_feature, "manager_opinion": opinion,
            "manager": person(rng), "vehicle": pick(rng, VEHICLES), "run_id": ident(rng, "TRR", 5, "-"),
            "recorded": rand_ts(rng, hours=(0, 5)), "next_train_min": pick(rng, [6, 12, 19, 27, 40, 55]),
            "show_level": chance(rng, 0.5),
        }

    # ------------------------------------------------------------------ the standard's logic
    @staticmethod
    def _level(param, band, v):
        row = GEOM[param][1][band]
        return 0 if v < row[1] else 1 if v < row[2] else 2 if v < row[3] else 3

    def _eff(self, f):
        """(effective level 0-3 for the primary site, seven_day flag)."""
        lvl = self._level(f["parameter"], f["band"], f["value_mm"])
        sec = f["second_fault"]
        near50 = sec is not None and sec["distance_m"] <= 50
        seven = False
        if lvl == 1 and near50 and sec["level"] >= 1:
            return 2, False
        if lvl == 1 and (f["value_mm"] >= 0.9 * GEOM[f["parameter"]][1][f["band"]][2] or (near50 and sec["level"] == 0)):
            seven = True
        if lvl == 0 and near50 and sec["level"] >= 1:
            seven = True
        return lvl, seven

    def _restriction(self, f):
        lvl, _ = self._eff(f)
        rows = GEOM[f["parameter"]][1]
        if lvl == 3:
            return "block_line"
        if lvl < 2:
            return "keep_line_speed"
        for b in range(f["band"] - 1, -1, -1):
            if f["value_mm"] < rows[b][1]:
                return "esr_lower_band"
        return "caution_esr"

    def intended(self, f):
        raw = self._level(f["parameter"], f["band"], f["value_mm"])
        lvl, seven = self._eff(f)
        restr = self._restriction(f)
        if restr in ("block_line", "caution_esr") or (f["rough_ride_reports_72h"] > 0 and lvl >= 1):
            verify = "inspect_before_next_train"
        elif raw >= 1 and (f["data_flag"] or f["prior_value_mm"] is None):
            verify = "hand_measure_within_24h"
        elif f["data_flag"]:
            verify = "confirm_on_next_recording_run"
        else:
            verify = "no_verification_needed"
        deadline = 3 if lvl >= 2 else 2 if seven else lvl
        pts = raw
        if f["parameter"] == "twist" and raw >= 1:
            pts += 1 + (f["traffic"] == "heavy_freight")
        if f["parameter"] == "gauge" and f["fastenings"] != "good":
            pts += 1
        if f["parameter"] in ("gauge", "alignment") and f["curve_radius_m"] and f["curve_radius_m"] <= 600:
            pts += 1
        sec = f["second_fault"]
        if sec and sec["distance_m"] <= 50:
            pts += 1
        exposure = 0 if pts == 0 else 1 if pts == 1 else 2 if pts <= 3 else 3
        combined = bool(sec and sec["distance_m"] <= 50 and raw <= 1 and max(raw, sec["level"]) >= 1)
        return {"restriction_call": restr, "verification_step": verify, "repair_deadline": deadline,
                "derailment_exposure": exposure, "combined_fault_rule": "yes" if combined else "no"}

    # ------------------------------------------------------------------ rendering
    def _rules(self, f):
        u = f["units"]
        return [
            "Alert: correct in the next 13-week maintenance cycle.",
            "Intervention: correct within 28 days, or within 7 days if the value is within 10% of the Immediate "
            "Action limit or an Alert-level fault lies within 50 m.",
            "Two faults at Intervention or worse within 50 m of each other: treat both as Immediate Action.",
            "Immediate Action: correct within 36 h; before the next train impose an ESR at the top speed of the "
            "highest lower band whose Intervention limit the value is below; if no band qualifies, a "
            f"{_spd(CAUTION[u], u)} caution ESR with the site examined before the next train.",
            "At or above Block: no trains until the site has been examined.",
            "Verification: flagged data at Intervention or worse, and first-time readings at Intervention or worse "
            "with no trace on the previous run, are measured by hand within 24 h; flagged Alert-level data may be "
            "checked on the next recording run; rough-ride reports within 72 h at a site at Intervention or worse "
            "require examination before the next train.",
        ]

    def _limits(self, f):
        u = f["units"]
        return {BAND_LABEL[u][b]: {"alert": r[0], "intervention": r[1], "immediate_action": r[2], "block": r[3]}
                for b, r in enumerate(GEOM[f["parameter"]][1])}

    def _limits_table(self, f, rng):
        u = f["units"]
        return table(["band", "alert", "interv.", "IA", "block"],
                     [[BAND_LABEL[u][b], *r] for b, r in enumerate(GEOM[f["parameter"]][1])], rng)

    def _second_text(self, f):
        s = f["second_fault"]
        if not s:
            return None
        return f"{GEOM[s['param']][0]} at {_lvl_name(s['level'])} level, {s['distance_m']} m away"

    def _opinion(self, f):
        o = f["manager_opinion"]
        if o is None:
            return None
        site = f["site_feature"] or "that location"
        return {"glitch": f"The car always throws a spike at the {site}. I'd treat this as a glitch.",
                "rides_fine": "Cab ride last week was fine. I would keep line speed and fix it on Thursday's shift.",
                "known_site": f"Known soft spot by the {site}; it's been like this for months and never got worse.",
                "genuine": "Drivers have been mentioning a lurch here. I think it's real.",
                "worse_than_shown": "If anything it will be worse on the ground than the car shows."}[o]

    def render(self, f, rng, surface):
        u = f["units"]
        pname = GEOM[f["parameter"]][0]
        raw = self._level(f["parameter"], f["band"], f["value_mm"])
        prior = (f"{f['prior_value_mm']} mm" if f["prior_value_mm"] is not None else "no exceedance recorded")
        curve = _curve_text(f["curve_radius_m"], f["cant_mm"])
        if surface == "json":
            return dumps(prune({
                "exceedance": {"run": f["run_id"], "vehicle": f["vehicle"], "recorded_utc": fmt_ts(f["recorded"], style="iso"),
                               "line": f["line_code"], "track": f["track"], "section": f["section"],
                               "location": f["mileage"], "parameter": pname, "value_mm": f["value_mm"],
                               "reported_level": _lvl_name(raw) if f["show_level"] else None,
                               "data_quality_flag": f["data_flag"], "site_feature": f["site_feature"]},
                "nearby": {"second_fault": self._second_text(f)} if f["second_fault"] else None,
                "site": {"line_speed": _spd(f["line_speed"], u), "speed_band": BAND_LABEL[u][f["band"]],
                         "alignment": curve, "traffic": TRAFFIC[f["traffic"]],
                         "fastenings": f["fastenings"].replace("_", " ")},
                "history": {"previous_run": f"{f['prior_run_weeks']} weeks ago: {prior}",
                            "rough_ride_reports_72h": f["rough_ride_reports_72h"]},
                "remarks": {"section_manager": self._opinion(f)} if f["manager_opinion"] else None,
                "standard": {"railway": f["railway"], "limits_mm": self._limits(f),
                             "rules": _std(rng, "TS-410", self._rules(f), json_ok=True)},
            }))
        if surface == "trc_report":
            head = (f"{f['railway'].upper()} - TRACK RECORDING EXCEEDANCE REPORT\n"
                    f"Run {f['run_id']}   Vehicle '{f['vehicle']}'   Recorded {fmt_ts(f['recorded'], rng)}\n"
                    f"Line {f['line_code']} {f['track']}, {f['section']}. Line speed {_spd(f['line_speed'], u)}.")
            rows = [[f["mileage"], pname, f"{f['value_mm']:.1f}", _lvl_name(raw) if f["show_level"] else ""]]
            if f["second_fault"]:
                s = f["second_fault"]
                rows.append([f"+{s['distance_m']} m", GEOM[s["param"]][0], "", _lvl_name(s["level"])])
            body = table(["location", "parameter", "mm", "level"], rows, rng, style="fixed")
            notes = _kv([("Data quality", f["data_flag"] or "no flags"), ("Site", f["site_feature"]),
                              ("Alignment", curve), ("Traffic", TRAFFIC[f["traffic"]]),
                              ("Fastenings (last patrol)", f["fastenings"].replace("_", " ")),
                              (f"Previous run ({f['prior_run_weeks']} wk)", prior),
                              ("Rough-ride reports, last 72 h", f["rough_ride_reports_72h"])], rng)
            out = [head, "", body, "", notes, "", f"Limits for {pname} (mm):", self._limits_table(f, rng), "",
                   _std(rng, "TS-410", self._rules(f))]
            if f["manager_opinion"]:
                out.insert(5, f"Section manager note: {noisy(self._opinion(f), rng)}")
            return "\n".join(out)
        # control_log
        t = f["recorded"] + timedelta(minutes=rng.randint(15, 70))
        mgr = f["manager"]
        lines = [f"{f['railway']} route control log, {fmt_date(t.date(), rng)}"]

        def add(mins, who, text):
            nonlocal t
            t += timedelta(minutes=mins)
            lines.append(f"{t.strftime('%H:%M')}  {who}: {text}")
        add(0, "Control", f"TRC exceedance received from run {f['run_id']}: {pname} {f['value_mm']} mm at "
                          f"{f['line_code']} {f['track']} {f['mileage']} ({f['section']}), line speed "
                          f"{_spd(f['line_speed'], u)}, {curve}." + (f" Car reports it as {_lvl_name(raw)}."
                                                                     if f["show_level"] else ""))
        if f["data_flag"]:
            add(2, "Control", f"Data flagged by the car: {f['data_flag']}.")
        if f["second_fault"]:
            add(1, "Control", f"Also on the run: {self._second_text(f)}.")
        add(rng.randint(3, 9), f"Control", f"Previous run {f['prior_run_weeks']} weeks ago: {prior}. Rough-ride "
                                           f"reports in the last 72 h: {f['rough_ride_reports_72h']}. "
                                           f"Traffic {TRAFFIC[f['traffic']]}.")
        if f["manager_opinion"]:
            add(rng.randint(4, 15), f"{mgr} (section mgr)", noisy(self._opinion(f), rng))
        add(rng.randint(1, 6), "Control", f"Next train over the site in about {f['next_train_min']} min. Response "
                                          f"per TS-410 required.")
        return ("\n".join(lines) + f"\n\nLimits for {pname} (mm):\n{self._limits_table(f, rng)}\n\n"
                + _std(rng, "TS-410", self._rules(f)))

    def questions(self):
        return {
            "restriction_call": choice("What should happen to train speeds over this site until the fault is put right?", {
                "keep_line_speed": "Leave trains at the normal line speed; the reading does not call for a temporary restriction.",
                "esr_lower_band": "Impose an emergency speed restriction at the top speed of a lower line-speed band.",
                "caution_esr": "Impose the standard's caution-speed restriction and have the site examined before the next train.",
                "block_line": "Stop trains over the site until a track inspector has examined it.",
            }),
            "verification_step": choice("What check of the reading should follow before or alongside the repair?", {
                "no_verification_needed": "None: the reading is consistent with earlier runs and can be acted on as recorded.",
                "hand_measure_within_24h": "Have the site measured by hand within 24 hours to confirm or clear the reading.",
                "inspect_before_next_train": "Send an inspector to examine the site before the next train passes over it.",
                "confirm_on_next_recording_run": "Look at the site again on the next scheduled recording run; no visit is needed.",
            }),
            "repair_deadline": score("How soon does the standard require the fault to be corrected?", [
                "In the next planned 13-week maintenance cycle.",
                "Within 28 days.",
                "Within 7 days.",
                "Within 36 hours, with the site protected in the meantime.",
            ]),
            "derailment_exposure": score("If trains kept running at line speed until the repair, how exposed would they be to a derailment?", [
                "Negligible: a ride-quality matter only.",
                "Low: worth correcting, but wheel-climb or gauge loss is not credible yet.",
                "Material: the fault, alone or with others, could contribute to a derailment.",
                "High: the kind of fault that derails trains if left at speed.",
            ]),
            "combined_fault_rule": noul(
                "A second fault near this one changes the response the standard requires at this site.",
                yes="Another recorded fault lies within the standard's distance and raises the urgency of the response.",
                no="There is no second fault close enough, or it does not change what the standard already requires."),
        }


# =========================================================================== schema 2
# defect type: (label, size unit, class boundaries C|B, B|A, A|X)
RAIL_DEFECTS = {
    "transverse": ("transverse defect in the rail head", "% of head area", (10, 25, 45)),
    "bolt_hole": ("bolt-hole crack", "mm", (12, 25, 38)),
    "horizontal_split": ("horizontal split head", "mm length", (100, 250, 400)),
    "vertical_split": ("vertical split head", "mm length", (80, 180, 300)),
    "weld": ("defective aluminothermic weld", "mm indication", (8, 18, 30)),
    "squat": ("squat (rolling-contact fatigue)", "mm depth", (2.0, 4.0, 6.0)),
}
COLD_TYPES = ("transverse", "weld", "bolt_hole")
CLASSES = ["C", "B", "A", "X"]
QUALITY_FLAGS = ["poor probe coupling over the indication", "possible geometric echo from the weld collar",
                 "signal noise from a wet railhead", "indication partly masked by head checking",
                 "operator marked the size as an estimate"]
RAIL_SECTIONS = ["60 kg/m flat-bottom", "56 kg/m flat-bottom", "54 kg/m flat-bottom", "50 kg/m flat-bottom",
                 "bullhead 95 lb"]
SITES = {"plain": "plain line", "switch": "in a switch (points) unit", "crossing": "in a crossing nose",
         "bridge_exp": "at a bridge expansion device", "tunnel": "in a tunnel", "lc": "at a level crossing"}


class UltrasonicRailDefect(Schema):
    name = "ultrasonic_rail_defect"
    description = ("An ultrasonic test finds an internal rail defect: how trains should run over it and how soon the "
                   "rail must be replaced under the rail-defect standard.")
    surfaces = {"json": 0.5, "test_unit_printout": 0.25, "contractor_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rt = _route(rng)
        units = rt["units"]
        cold_case = chance(rng, 0.5)
        dtype = pick(rng, list(COLD_TYPES) if cold_case else list(RAIL_DEFECTS))
        bounds = RAIL_DEFECTS[dtype][2]
        edges = [0.3 * bounds[0], bounds[0], bounds[1], bounds[2], bounds[2] * 1.4]
        if reg == "borderline":
            size = near(rng, pick(rng, bounds), bounds[0] * 0.35, lo=edges[0])
        else:
            cls = pick(rng, [0, 1, 2, 3], [3, 3, 2, 1] if reg == "clear" else [3, 3, 3, 2])
            size = between(rng, edges[cls] + 0.1 * (edges[cls + 1] - edges[cls]),
                           edges[cls + 1] - 0.15 * (edges[cls + 1] - edges[cls]))
        size = round(size, 1) if dtype == "squat" else int(round(size))
        cwr = dtype != "bolt_hole" or cold_case or chance(rng, 0.3)   # bolt holes: jointed track, or insulated joints in CWR
        site = pick_w(rng, {"plain": 6, "switch": 1, "crossing": 1, "bridge_exp": 1, "tunnel": 1, "lc": 1})
        if dtype == "bolt_hole" and site in ("crossing", "bridge_exp"):
            site = "plain"
        season = "winter" if cold_case else pick(rng, ["winter", "spring", "autumn", "summer"])
        fmin = {"winter": near(rng, -4, 4, integer=True), "autumn": between(rng, -4, 6, integer=True),
                "spring": between(rng, -2, 8, integer=True), "summer": between(rng, 8, 16, integer=True)}[season]
        if reg == "borderline" and cold_case:
            fmin = near(rng, -3, 2, integer=True)
        elif reg == "conflicting" and chance(rng, 0.5):
            fmin, season = pick(rng, [-3, -4, -6, -8]), "winter"
        quality = None
        if chance(rng, {"clear": 0.1, "borderline": 0.3, "conflicting": 0.55}[reg]):
            quality = pick(rng, [q for q in QUALITY_FLAGS if dtype == "weld" or "weld collar" not in q])
        tonnage = pick(rng, [4, 8, 12, 18, 25, 35])
        heavy = rt["units"] == "imperial" and chance(rng, 0.3) or chance(rng, 0.15)
        opinion = None
        if reg == "conflicting" or chance(rng, 0.15):
            opinion = pick(rng, ["low_risk", "replace_now", "probably_spurious", "seen_worse"])
        return {
            "regime": reg, **rt, "mileage": _mileage(rng, units),
            "line_speed": pick(rng, BAND_SPEEDS[units][pick(rng, [1, 2, 2, 3])]),
            "defect_type": dtype, "size": size, "cwr": cwr, "site": site, "season": season,
            "forecast_min_c": fmin, "quality_flag": quality, "annual_mgt": tonnage, "heavy_axle": heavy,
            "rail_section": pick(rng, RAIL_SECTIONS[:4] if cwr else RAIL_SECTIONS),
            "rail_rolled": rng.randint(1979, 2021), "rail": pick(rng, ["left", "right", "cess", "six-foot"]),
            "other_indications_same_rail": pick_w(rng, {0: 6, 1: 2, 2: 1}),
            "contractor": org(rng, "company"), "operator": person(rng), "engineer": person(rng),
            "opinion": opinion, "unit": f"UTU-{rng.randint(2, 9)} '{pick(rng, VEHICLES)}'",
            "tested": self._season_ts(rng, season), "report": ident(rng, "RF", 6, "-"),
        }

    @staticmethod
    def _season_ts(rng, season):
        t = rand_ts(rng, hours=(0, 6))
        month = pick(rng, {"winter": [12, 1, 2], "spring": [3, 3, 4], "summer": [6, 7, 8], "autumn": [10, 11, 11]}[season])
        return t.replace(month=month, day=min(t.day, 28))

    @staticmethod
    def _base_class(f):
        b = RAIL_DEFECTS[f["defect_type"]][2]
        s = f["size"]
        return 0 if s < b[0] else 1 if s < b[1] else 2 if s < b[2] else 3

    def _cold(self, f):
        return f["cwr"] and f["defect_type"] in COLD_TYPES and f["forecast_min_c"] <= -3 and self._base_class(f) < 3

    def intended(self, f):
        base_c = self._base_class(f)
        cls = min(3, base_c + (1 if self._cold(f) else 0))
        run = cls
        if f["site"] in ("switch", "crossing", "bridge_exp") and cls in (1, 2):
            run = cls + 1
        running = ["normal_running", "clamp_and_cap", "restrict_to_lowest", "stop_traffic"][run]
        window = {0: 1 if f["heavy_axle"] else 0, 1: 1, 2: 2, 3: 3}[cls]
        b = RAIL_DEFECTS[f["defect_type"]][2]
        near_b = any(abs(f["size"] - x) <= 0.1 * x for x in b)
        pts = cls + (1 if f["forecast_min_c"] <= -3 and f["defect_type"] in COLD_TYPES else 0)
        pts += (f["heavy_axle"] or f["annual_mgt"] >= 25) + (f["other_indications_same_rail"] > 0)
        pts += 1 if f["rail_rolled"] <= 1986 else 0
        brk = 0 if pts <= 0 else 1 if pts <= 2 else 2 if pts <= 3 else 3
        return {"interim_running": running, "replacement_window": window, "break_likelihood": brk,
                "hand_test_needed": "yes" if (f["quality_flag"] or near_b) else "no",
                "cold_upgrade_applies": "yes" if self._cold(f) else "no"}

    def _rules(self, f):
        u = f["units"]
        lab, unit, b = RAIL_DEFECTS[f["defect_type"]]
        return [
            f"{_ucfirst(lab)} ({unit}): class C below {b[0]}, B {b[0]} to under {b[1]}, A {b[1]} to under {b[2]}, "
            f"X {b[2]} or more.",
            "Class C: normal running, retest within 4 weeks, replace at the next planned renewal (within 13 weeks; "
            "within 28 days on routes with 25 t axle loads).",
            f"Class B: emergency clamps within 24 h, speed capped at {_spd(STEP[u], u)}, replace within 28 days.",
            f"Class A: clamps and a {_spd(CAUTION[u], u)} ESR before the next train, replace within 7 days.",
            "Class X: no trains until the rail is replaced, or clamped and examined by the section engineer; replace "
            "within 36 h.",
            "In CWR, transverse, weld and bolt-hole defects move up one class when the forecast minimum air "
            "temperature in the next 7 days is -3 C or lower.",
            "Where clamps cannot be fitted (switch and crossing units, bridge expansion devices), apply the running "
            "restriction of the next class up.",
            "Confirm the size by hand test where the test unit flags the reading or the size is within 10% of a "
            "class boundary.",
        ]

    def _opinion_text(self, f):
        return {"low_risk": "From what we see on the trace this is low risk; we would not restrict.",
                "replace_now": "Our operator would not want trains on this; we recommend replacement tonight.",
                "probably_spurious": "Our view is the indication is probably spurious, but it has to be reported.",
                "seen_worse": "We have seen far worse left in track for months without trouble."}.get(f["opinion"])

    def render(self, f, rng, surface):
        u = f["units"]
        lab, unit, _ = RAIL_DEFECTS[f["defect_type"]]
        size = f"{f['size']} {unit}"
        joint = "continuous welded rail (CWR)" if f["cwr"] else "jointed track"
        fc = f"forecast minimum {f['forecast_min_c']} C over the next 7 days"
        if surface == "json":
            return dumps(prune({
                "finding": {"report": f["report"], "test_unit": f["unit"], "tested_utc": fmt_ts(f["tested"], style="iso"),
                            "defect": lab, "size": size, "rail": f["rail"], "test_flag": f["quality_flag"],
                            "other_indications_in_same_rail": f["other_indications_same_rail"],
                            "contractor_comment": self._opinion_text(f)},
                "location": {"railway": f["railway"], "line": f["line_code"], "track": f["track"],
                             "section": f["section"], "mileage": f["mileage"], "site": SITES[f["site"]]},
                "track": {"construction": joint, "rail": f"{f['rail_section']}, rolled {f['rail_rolled']}",
                          "line_speed": _spd(f["line_speed"], u), "annual_tonnage_mgt": f["annual_mgt"],
                          "axle_loads_25t": f["heavy_axle"]},
                "weather": {"forecast_min_air_c_next_7_days": f["forecast_min_c"]},
                "standard": _std(rng, "RD-220", self._rules(f), json_ok=True),
            }))
        if surface == "test_unit_printout":
            rows = kv_lines([("Report", f["report"]), ("Unit", f["unit"]), ("Operator", f["operator"]),
                             ("Tested", fmt_ts(f["tested"], rng)),
                             ("Line / track", f"{f['line_code']} {f['track']} ({f['section']})"),
                             ("Mileage", f["mileage"]), ("Rail", f"{f['rail']} rail, {f['rail_section']}, rolled {f['rail_rolled']}"),
                             ("Site", SITES[f["site"]]), ("Construction", joint),
                             ("Indication", lab), ("Size", size), ("Test flag", f["quality_flag"] or "none"),
                             ("Other indications this rail length", f["other_indications_same_rail"]),
                             ("Line speed", _spd(f["line_speed"], u)),
                             ("Traffic", f"{f['annual_mgt']} MGT/yr" + (", 25 t axle loads" if f["heavy_axle"] else "")),
                             ("Weather", fc)], rng, style=pick(rng, ["aligned", "colon"]))
            return f"{f['railway'].upper()} RAIL FLAW DETECTION - DEFECT REPORT\n{rows}\n\n{_std(rng, 'RD-220', self._rules(f))}"
        # contractor_email
        para = [f"Our {f['unit']} run on {fmt_date(f['tested'].date(), rng)} found a {lab} in the {f['rail']} rail at "
                f"{f['line_code']} {f['track']} {f['mileage']} ({f['section']}), {SITES[f['site']]}.",
                f"Sized at {size}." + (f" Note: {f['quality_flag']}." if f["quality_flag"] else ""),
                (f"The same rail length has {_plural(f['other_indications_same_rail'], 'further smaller indication')}."
                 if f["other_indications_same_rail"] else ""),
                f"Track is {joint}, {f['rail_section']} rail rolled {f['rail_rolled']}, line speed "
                f"{_spd(f['line_speed'], u)}, about {f['annual_mgt']} MGT a year"
                + (" with 25 t axle loads." if f["heavy_axle"] else "."),
                f"The weather desk forecasts a minimum of {f['forecast_min_c']} C over the next 7 days.",
                self._opinion_text(f) or ""]
        return "\n".join([f"Subject: Rail defect {f['report']} - {f['line_code']} {f['mileage']}",
                          f"From: {f['operator']} <{f['operator'].split()[0].lower()}@{f['contractor'].split()[0].lower()}-ndt.example>",
                          f"To: {f['engineer']} (track engineer, {f['railway']})", "",
                          f"{f['engineer'].split()[0]},", "", noisy(join_sentences(para), rng), "",
                          _std(rng, "RD-220", self._rules(f)), "", f"{f['operator']}", f"{f['contractor']}"])

    def questions(self):
        return {
            "interim_running": choice("Until the rail is replaced, how should trains run over the defect?", {
                "normal_running": "Keep running at line speed and retest the rail on the standard's schedule.",
                "clamp_and_cap": "Fit emergency clamps and cap speed at the standard's intermediate limit.",
                "restrict_to_lowest": "Impose the standard's lowest emergency speed before the next train, with clamps where they fit.",
                "stop_traffic": "Stop traffic over the rail until it is replaced, or made safe and examined.",
            }),
            "replacement_window": score("How soon must the defective rail be replaced?", [
                "At the next planned rail renewal, within 13 weeks.",
                "Within 28 days.",
                "Within 7 days.",
                "Within 36 hours.",
            ]),
            "break_likelihood": score("How likely is the rail to break before the replacement is done?", [
                "Remote.",
                "Unlikely, but worth watching.",
                "A real possibility given the size, loading or weather.",
                "Likely unless the rail is dealt with straight away.",
            ]),
            "hand_test_needed": noul(
                "The reported size must be confirmed by a hand-held ultrasonic test before the defect is classed.",
                yes="The test unit flagged the reading, or the size sits close enough to a class boundary that the standard asks for a hand test.",
                no="The reading is unflagged and clearly inside one class."),
            "cold_upgrade_applies": noul(
                "The forecast cold weather moves this defect into a more urgent class under the standard."),
        }


# =========================================================================== schema 3
SFT_BASIS = {
    "stress_test": "rail stress test", "destress_cert": "destressing certificate",
    "assumed": "design assumption (no site record)", "predates_repair": "destressing certificate",
}
WORK_KINDS = ["tamping", "ballast cleaning", "sleeper renewal", "a rail replacement", "drainage works with a track lift"]


class HeatBucklingPrecaution(Schema):
    name = "heat_buckling_precaution"
    description = ("A hot-weather forecast for a CWR site: which buckling precaution the hot-weather standard calls "
                   "for, what drives the risk, and how soon it must be in place.")
    surfaces = {"json": 0.5, "heat_watch_sheet": 0.25, "route_chat": 0.25}

    ALLOW = {"concrete": 30, "timber": 25, "steel": 27}

    def sample_factors(self, rng):
        reg = regime(rng)
        rt = _route(rng)
        # forecast offset from the site's CRT, steered by regime around the precaution boundaries
        if reg == "borderline":
            off = near(rng, pick(rng, [-10, -10, -5, 5]), 2.5, integer=True)
        elif reg == "clear":
            off = pick(rng, [-17, -14, -13, -8, -7, 0, 2, 8, 10])
        else:
            off = between(rng, -16, 9, integer=True)
        # a heat watch is only raised in hot weather (rail 39 C+), so a site with no precaution due must be sound
        strong = off <= -10
        sleeper = pick_w(rng, {"concrete": 6, "timber": 3, "steel": 1} if not strong else {"concrete": 5, "steel": 1})
        basis = pick_w(rng, {"stress_test": 4, "destress_cert": 3, "assumed": 2, "predates_repair": 2})
        if reg == "clear":
            sft = pick(rng, [between(rng, 22, 27, integer=True), between(rng, 12, 17, integer=True)])
        elif reg == "borderline":
            sft = near(rng, 21, 3, integer=True)
        else:
            sft = between(rng, 19, 27, integer=True)
            basis = pick(rng, ["predates_repair", "assumed", "stress_test"])
        if strong:
            basis, sft = pick(rng, ["stress_test", "destress_cert"]), between(rng, 22, 27, integer=True)
        test_year = None
        if basis in ("stress_test", "destress_cert", "predates_repair"):
            test_year = pick(rng, [2016, 2018, 2020, 2022, 2023, 2024, 2025] if strong
                             else [2012, 2014, 2016, 2018, 2020, 2022, 2023, 2024, 2025])
        if basis == "assumed":
            sft = 21
        ballast = pick_w(rng, {"full": 5, "one_side": 2, "deficient": 2} if not strong else {"full": 3, "one_side": 1})
        work = None
        if chance(rng, 0.45):
            kt = (near(rng, 100, 30, lo=5, integer=True) if reg == "borderline"
                  else pick(rng, [12, 30, 55, 80, 140, 220, 400]))
            if strong:
                kt = max(kt, pick(rng, [104, 130, 180, 260]))
            work = {"kind": pick(rng, WORK_KINDS), "days_ago": pick(rng, [2, 4, 6, 9, 14, 21, 35]), "kt_since": kt}
        radius = pick_w(rng, {None: 5, 500: 1, 700: 2, 1000: 2, 1600: 1})
        f = {"regime": reg, **rt, "mileage": _mileage(rng, rt["units"]), "sleeper": sleeper, "sft_recorded": sft,
             "sft_basis": basis, "sft_record_year": test_year, "ballast": ballast, "recent_work": work,
             "curve_radius_m": radius, "misalignment_history": pick_w(rng, {0: 6, 1: 2, 2: 1}),
             "line_speed": pick(rng, BAND_SPEEDS[rt["units"]][pick(rng, [1, 2, 2, 3])]),
             "forecast_day": pick_w(rng, {0: 3, 1: 4, 2: 2, 3: 1}), "peak_hour": pick(rng, [13, 14, 15, 16, 17]),
             "issued": self._summer_ts(rng), "engineer": person(rng), "controller": person(rng),
             "repair_year": None, "forecast_offset": off}
        if basis == "predates_repair":
            f["repair_year"] = min(2026, (test_year or 2018) + pick(rng, [1, 2, 3, 4]))
        rtemp = min(63, max(39, self._crt(f) + off))
        f["forecast_rail_c"] = rtemp
        f["forecast_air_c"] = int(round(rtemp - rng.uniform(11, 16)))
        f["opinion"] = (pick(rng, ["fine_last_year", "worried", "no_staff", "ballast_done"])
                        if reg == "conflicting" or chance(rng, 0.2) else None)
        return f

    @staticmethod
    def _summer_ts(rng):
        d = rand_ts(rng, hours=(5, 11))
        return d.replace(month=pick(rng, [5, 6, 6, 7, 7, 7, 8, 8, 9]), day=min(d.day, 28))

    @staticmethod
    def _reliable(f):
        return f["sft_basis"] in ("stress_test", "destress_cert") and (f["sft_record_year"] or 0) >= 2016

    def _eff_sft(self, f):
        return f["sft_recorded"] if self._reliable(f) else min(f["sft_recorded"], 18)

    def _penalties(self, f):
        w = f["recent_work"]
        work = 10 if w and w["kt_since"] < 100 else 0
        ballast = {"full": 0, "one_side": 3, "deficient": 6}[f["ballast"]]
        curve = 2 if f["curve_radius_m"] and f["curve_radius_m"] < 800 else 0
        return work, ballast, curve

    def _crt(self, f):
        work, ballast, curve = self._penalties(f)
        return self._eff_sft(f) + self.ALLOW[f["sleeper"]] - work - ballast - curve

    def intended(self, f):
        m = f["forecast_rail_c"] - self._crt(f)
        cls = 0 if m < -10 else 1 if m < -5 else 2 if m < 5 else 3
        measure = ["no_precaution", "patrol_watch", "heat_esr", "trains_off_or_restress"][cls]
        work, ballast, _ = self._penalties(f)
        sft_short = max(0, 21 - self._eff_sft(f))
        cand = {"recent_track_work": work, "stress_free_temperature": sft_short, "ballast_shoulder": ballast}
        top = max(cand.values())
        weakness = "no_particular_weakness" if top < 3 else next(k for k, v in cand.items() if v == top)
        lik = 0 if m < -9 else 1 if m < -2 else 2 if m < 6 else 3
        if f["misalignment_history"] and m >= -9:
            lik = min(3, lik + 1)
        lead = 0 if cls == 0 else {0: 3, 1: 2}.get(f["forecast_day"], 1)
        return {"heat_precaution": measure, "main_weakness": weakness, "buckle_likelihood": lik,
                "precaution_lead_time": lead, "sft_value_reliable": "yes" if self._reliable(f) else "no"}

    RULES = [
        "Critical rail temperature CRT = SFT + 30 C on concrete sleepers (25 C timber, 27 C steel); minus 6 C for a "
        "deficient ballast shoulder (3 C where only one side is short); minus 10 C where the track was disturbed and "
        "has carried under 100 kt since; minus 2 C on curves under 800 m.",
        "Use the recorded SFT only if it comes from a destressing certificate or stress test dated 2016 or later "
        "with no rail cut or repair at the site since; otherwise take 18 C, or the recorded value if lower. Target "
        "SFT 21-27 C.",
        "On forecast peak rail temperature RT: below CRT-10 no precaution; CRT-10 to CRT-5 patrol the site through "
        "the hottest hours; CRT-5 to CRT+5 heat ESR for the hot hours; CRT+5 or above no trains at the peak unless "
        "the site is restressed first.",
        "Precautions must be in force before the first train on the day of the forecast peak.",
    ]

    def _basis_text(self, f):
        b = f["sft_basis"]
        if b == "assumed":
            return "21 C design value; no site record"
        txt = f"{f['sft_recorded']} C from {SFT_BASIS[b]} ({f['sft_record_year']})"
        if b == "predates_repair":
            txt += f"; rail cut and replaced at this site after a broken rail in {f['repair_year']}, not restressed"
        return txt

    def _work_text(self, f):
        w = f["recent_work"]
        if not w:
            return "no track work in the last 90 days"
        return f"{w['kind']} {w['days_ago']} days ago, about {w['kt_since']} kt carried since"

    def _day(self, f):
        return {0: "today", 1: "tomorrow"}.get(f["forecast_day"], f"in {f['forecast_day']} days")

    def _opinion(self, f):
        return {"fine_last_year": "This site was fine through last summer's heat.",
                "worried": "I don't like this one, the cess side looked thin when I walked it.",
                "no_staff": "We're short of patrollers this week if it comes to that.",
                "ballast_done": "Ballast regulator went through last month so the shoulder should be fine now."
                }.get(f["opinion"])

    def render(self, f, rng, surface):
        u = f["units"]
        curve = _curve_text(f["curve_radius_m"])
        if surface == "json":
            return dumps(prune({
                "site": {"railway": f["railway"], "line": f["line_code"], "track": f["track"], "section": f["section"],
                         "mileage": f["mileage"], "alignment": curve, "sleepers": f["sleeper"],
                         "ballast_shoulder": {"full": "full profile", "one_side": "short on the cess side",
                                              "deficient": "short on both sides"}[f["ballast"]],
                         "line_speed": _spd(f["line_speed"], u),
                         "misalignments_reported_last_3_summers": f["misalignment_history"]},
                "stress_free_temperature": self._basis_text(f),
                "recent_work": self._work_text(f),
                "forecast": {"issued_utc": fmt_ts(f["issued"], style="iso"), "peak_day": self._day(f),
                             "peak_air_c": f["forecast_air_c"], "peak_rail_c": f["forecast_rail_c"],
                             "peak_hour_local": f"{f['peak_hour']}:00"},
                "remarks": self._opinion(f),
                "standard": _std(rng, "HW-115", self.RULES, json_ok=True),
            }))
        if surface == "heat_watch_sheet":
            rows = _kv([("Site", f"{f['line_code']} {f['track']} {f['mileage']} ({f['section']})"),
                             ("Alignment", curve), ("Sleepers", f["sleeper"]),
                             ("Ballast shoulder", {"full": "full", "one_side": "cess side short",
                                                   "deficient": "short both sides"}[f["ballast"]]),
                             ("SFT", self._basis_text(f)), ("Recent work", self._work_text(f)),
                             ("Misalignments, last 3 summers", f["misalignment_history"]),
                             ("Line speed", _spd(f["line_speed"], u)),
                             ("Forecast peak", f"{self._day(f)} around {f['peak_hour']}:00, air {f['forecast_air_c']} C, "
                                               f"rail {f['forecast_rail_c']} C"),
                             ("Engineer's note", self._opinion(f))], rng)
            return (f"{f['railway']} HOT WEATHER WATCH SHEET (issued {fmt_ts(f['issued'], rng)})\n{rows}\n\n"
                    + _std(rng, "HW-115", self.RULES))
        # route_chat
        eng, ctl = f["engineer"], f["controller"]
        msgs = [(ctl, f"Heat alert: rail {f['forecast_rail_c']} C forecast {self._day(f)} (air {f['forecast_air_c']} C), "
                      f"peak about {f['peak_hour']}:00. Need a call on {f['line_code']} {f['mileage']} {f['track']}."),
                (eng, f"{curve}, {f['sleeper']} sleepers, ballast "
                      + {"full": "at full profile", "one_side": "short on the cess side",
                         "deficient": "short both sides"}[f["ballast"]] + "."),
                (eng, f"SFT: {self._basis_text(f)}."),
                (eng, f"Recent work: {self._work_text(f)}.")]
        if f["misalignment_history"]:
            msgs.append((ctl, f"Log shows {f['misalignment_history']} misalignment report(s) here in the last 3 summers."))
        if f["opinion"]:
            msgs.append((eng, self._opinion(f)))
        t = f["issued"]
        out = []
        for who, text in msgs:
            t += timedelta(minutes=rng.randint(1, 12))
            out.append(f"[{t.strftime('%H:%M')}] {who.split()[0]}: {noisy(text, rng, 0.12) if text[:2].istitle() else text}")
        return f"#route-control-heat ({f['railway']})\n" + "\n".join(out) + "\n\npinned: " + _std(rng, "HW-115", self.RULES)

    def questions(self):
        return {
            "heat_precaution": choice("Which heat precaution should apply to this site for the forecast peak?", {
                "no_precaution": "None: the forecast stays well below what this site can take.",
                "patrol_watch": "Patrol the site through the hottest hours, without a speed restriction.",
                "heat_esr": "Impose the heat speed restriction for the hours the rail is forecast to be hottest.",
                "trains_off_or_restress": "Keep trains off the site at the peak unless it is restressed before then.",
            }),
            "main_weakness": choice("What most reduces this site's resistance to buckling?", {
                "stress_free_temperature": "A stress-free temperature that is low or cannot be relied on.",
                "recent_track_work": "Recent track work whose ballast has not yet consolidated under traffic.",
                "ballast_shoulder": "A ballast shoulder short of the standard profile.",
                "no_particular_weakness": "Nothing in particular: the site is about as strong as the standard assumes.",
            }),
            "buckle_likelihood": score("How likely is the track to buckle at the forecast peak if no precaution is taken?", [
                "Very unlikely.",
                "Unlikely, though the margin is thinner than usual.",
                "A real possibility.",
                "Likely.",
            ]),
            "precaution_lead_time": score("How soon must the precaution be in force?", [
                "Nothing needs to be in force for this forecast.",
                "Before the first train on a hot day two or more days away.",
                "Before tomorrow's first train.",
                "Before the next train today.",
            ]),
            "sft_value_reliable": noul(
                "The site's recorded stress-free temperature can be used as it stands in the CRT calculation.",
                yes="The record is a recent enough certificate or stress test, and the rail has not been cut or repaired since.",
                no="The value is assumed, too old, or predates a later rail repair, so the standard's default applies."),
        }


# =========================================================================== schema 4
# primary finding: (report label, unit singular, unit plural, how a patroller says it)
PRIMARY = {
    "fastenings": ("ineffective fastenings (clips missing or broken)", "sleeper with an ineffective fastening",
                   "consecutive sleepers with an ineffective fastening", "clips missing or broken on {n} sleepers in a row"),
    "voids": ("voided sleepers (pumping, visible movement under traffic)", "voided sleeper",
              "consecutive voided sleepers", "{n} sleepers in a row pumping and moving under trains"),
    "cracked": ("cracked or broken sleepers", "cracked sleeper", "consecutive cracked sleepers",
                "{n} cracked sleepers in a row"),
}


class TrackPatrolFinding(Schema):
    name = "track_patrol_finding"
    description = ("A foot patrol reports defective fastenings, sleepers or joints: how to respond under the track "
                   "patrol standard and what will put the site right.")
    surfaces = {"json": 0.5, "patrol_report": 0.25, "phone_report": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rt = _route(rng)
        radius = pick_w(rng, {None: 4, 450: 2, 550: 1, 700: 2, 1000: 2, 1400: 1, 2200: 1})
        lim = self._limit(radius)
        primary = pick(rng, list(PRIMARY))
        if reg == "clear":
            n = pick(rng, [max(1, lim - 2), lim - 1 if lim > 1 else 1, lim + 2, lim + 3])
        elif reg == "borderline":
            n = lim + pick(rng, [-1, 0, 0, 1])
        else:
            n = lim + pick(rng, [-1, 0, 1, 2])
        n = max(1, n)
        joint = None
        if chance(rng, 0.3):
            joint = pick_w(rng, {3: 2, 2: 3, 1: 3, 0: 1})        # effective bolts in the worse rail end
        wet = pick_w(rng, {0: 5, 2: 2, 4: 2, 6: 1})            # sleepers over a wet bed
        drain = pick_w(rng, {"clear": 5, "partly_blocked": 2, "blocked": 2})
        timber = pick_w(rng, {"concrete": 6, "timber": 4})
        if primary == "voids" and reg != "clear":
            wet = max(wet, pick(rng, [2, 3, 4, 6]))
        prev = max(0, n - pick(rng, [0, 0, 1, 2, 3, 4, 6])) if not chance(rng, 0.15) else 0
        materials = chance(rng, 0.55)
        lookout_ok = chance(rng, 0.6)
        return {"regime": reg, **rt, "mileage": _mileage(rng, rt["units"]), "curve_radius_m": radius,
                "line_speed": pick(rng, BAND_SPEEDS[rt["units"]][pick(rng, [1, 1, 2, 3])]),
                "primary": primary, "count": n, "joint_bolts": joint, "wet_sleepers": wet, "drainage": drain,
                "sleepers": timber, "timber_decay": timber == "timber" and chance(rng, 0.45),
                "prev_count": prev, "weeks_since_last": pick(rng, [1, 1, 2]),
                "rain_48h": pick_w(rng, {"dry": 4, "showers": 3, "heavy": 3}),
                "materials_carried": materials, "lookout_adequate": lookout_ok,
                "patroller": person(rng), "supervisor": person(rng),
                "patrol_date": rand_ts(rng, hours=(7, 15)), "ref": ident(rng, "PTL", 5, "-"),
                "vegetation": chance(rng, 0.3), "fence_damage": chance(rng, 0.15)}

    @staticmethod
    def _limit(radius):
        if radius is None or radius >= 1500:
            return 4
        return 3 if radius >= 600 else 2

    def _severity(self, f):
        lim = self._limit(f["curve_radius_m"])
        n = f["count"]
        s_primary = 0 if n < lim else 1 if n == lim else 2 if n == lim + 1 else 3
        s_joint = 0 if f["joint_bolts"] is None or f["joint_bolts"] >= 2 else (2 if f["joint_bolts"] == 1 else 3)
        s_wet = 1 if f["wet_sleepers"] >= 3 else 0
        return s_primary, s_joint, s_wet

    def intended(self, f):
        sp, sj, sw = self._severity(f)
        worst = max(sp, sj, sw)
        worst_kind = "joint" if sj == worst and sj > sp else ("primary" if sp == worst else "wet")
        fixable = (f["materials_carried"] and f["lookout_adequate"]
                   and ((worst_kind == "primary" and f["primary"] == "fastenings" and not f["timber_decay"])
                        or worst_kind == "joint"))
        if worst == 3:
            resp = "block_and_call_out"
        elif worst in (1, 2) and fixable:
            resp = "fix_during_patrol"
        else:
            resp = ["record_for_planned_work", "repair_within_week", "esr_and_urgent_repair"][worst]
        # the work that removes the finding driving the response (the worst one; the primary run wins ties)
        if worst_kind == "joint":
            work = "joint_overhaul"
        elif worst_kind == "wet" or f["primary"] == "voids":
            work = "drainage_and_packing"
        elif f["primary"] == "cracked" or f["timber_decay"]:
            work = "sleeper_renewal"
        else:
            work = "fastening_renewal"
        rate = (f["count"] - f["prev_count"]) / f["weeks_since_last"]
        trend = 0 if rate <= 0 else 1 if rate <= 1 else 2 if rate <= 2.5 else 3
        return {"patrol_response": resp, "remedial_work": work, "remedial_deadline": worst,
                "deterioration_trend": trend,
                "cluster_limit_reached": "yes" if f["count"] >= self._limit(f["curve_radius_m"]) else "no"}

    def _rules(self, f):
        u = f["units"]
        return [
            "Consecutive defective assemblies on one rail (ineffective fastenings, voided or cracked sleepers): limit "
            "L = 4 on straight track and curves of 1500 m or more, 3 on curves of 600-1499 m, 2 on curves under 600 m.",
            f"Below L: record for planned maintenance (13 weeks). At L: repair within 7 days. L+1: ESR "
            f"{_spd(STEP[u], u)} and repair within 36 h. L+2 or more: stop trains and call out.",
            "Fishplated joints: one effective bolt in a rail end, ESR and repair within 36 h; none, stop trains.",
            "Wet beds over 3 or more sleepers: repair within 7 days.",
            "A patroller carrying the materials, with lookout protection adequate for the work, may refit fastenings "
            "(not into decayed timber) or joint bolts on the spot; a defect made good during the patrol needs no "
            "restriction.",
        ]

    def _count_text(self, f, n):
        _, one, many, _ = PRIMARY[f["primary"]]
        return f"{n} {one if n == 1 else many}"

    def _findings(self, f, speech=False):
        lab, _, _, said = PRIMARY[f["primary"]]
        first = said.format(n=f["count"]) if speech else f"{lab}: {self._count_text(f, f['count'])}"
        out = [first + (" (timber decayed, spike holes enlarged)" if f["timber_decay"] and f["primary"] != "voids" else "")]
        if f["joint_bolts"] is not None:
            out.append(f"fishplated joint at the same site: {f['joint_bolts']} effective bolt(s) in the worse rail end")
        if f["wet_sleepers"]:
            out.append(f"wet bed with slurry over {f['wet_sleepers']} sleepers")
        out.append({"clear": "cess drain running clear", "partly_blocked": "cess drain partly silted",
                    "blocked": "catchpit and cess drain blocked, standing water"}[f["drainage"]])
        if f["vegetation"]:
            out.append("vegetation encroaching on the cess walking route")
        if f["fence_damage"]:
            out.append("boundary fence damaged (reported separately)")
        return out

    def render(self, f, rng, surface):
        u = f["units"]
        curve = _curve_text(f["curve_radius_m"])
        wk = _plural(f["weeks_since_last"], "week")
        prev = (f"{self._count_text(f, f['prev_count'])}, {wk} ago" if f["prev_count"]
                else f"nothing recorded at this site, {wk} ago")
        if surface == "json":
            return dumps(prune({
                "patrol": {"ref": f["ref"], "patroller": f["patroller"], "date": fmt_date(f["patrol_date"].date(), style="iso"),
                           "materials_carried": f["materials_carried"],
                           "protection": "lookout, adequate sighting for work" if f["lookout_adequate"]
                           else "lookout, sighting only adequate for walking"},
                "site": {"railway": f["railway"], "line": f["line_code"], "track": f["track"], "section": f["section"],
                         "mileage": f["mileage"], "alignment": curve, "sleepers": f["sleepers"],
                         "line_speed": _spd(f["line_speed"], u)},
                "findings": self._findings(f),
                "previous_patrol": prev,
                "weather_48h": {"dry": "dry", "showers": "showers", "heavy": "heavy rain"}[f["rain_48h"]],
                "standard": _std(rng, "TP-305", self._rules(f), json_ok=True),
            }))
        if surface == "patrol_report":
            head = _kv([("Patrol ref", f["ref"]), ("Patroller", f["patroller"]),
                             ("Date", fmt_date(f["patrol_date"].date(), rng)),
                             ("Line / track", f"{f['line_code']} {f['track']}"), ("Section", f["section"]),
                             ("Mileage", f["mileage"]), ("Alignment", curve), ("Sleepers", f["sleepers"]),
                             ("Line speed", _spd(f["line_speed"], u)),
                             ("Materials carried", "yes" if f["materials_carried"] else "no"),
                             ("Protection", "lookout; sighting adequate for work" if f["lookout_adequate"]
                              else "lookout; sighting adequate for walking only"),
                             ("Last patrol", prev),
                             ("Weather, last 48 h", f["rain_48h"])], rng)
            return (f"{f['railway'].upper()} TRACK PATROL REPORT\n{head}\n\nFindings:\n"
                    f"{bullets(self._findings(f), rng)}\n\n" + _std(rng, "TP-305", self._rules(f)))
        # phone_report
        said = [f"It's {f['patroller'].split()[0]} on the {f['line_code']} patrol, I'm at {f['mileage']} {f['track']}, "
                f"{f['section']}, on the {curve}.",
                "I've got " + "; ".join(self._findings(f, speech=True)) + ".",
                f"Last patrol: {prev}.",
                ("I've got clips and bolts in the bag" if f["materials_carried"] else "I've not got any materials with me")
                + (" and my lookout has good sighting here." if f["lookout_adequate"]
                   else ", and the sighting here is poor, my lookout can only cover us walking."),
                {"dry": "", "showers": "Been showery.", "heavy": "We've had a lot of rain since Tuesday."}[f["rain_48h"]]]
        return (f"Control phone log {f['ref']}, {fmt_ts(f['patrol_date'], rng)}. Line speed {_spd(f['line_speed'], u)}.\n"
                f"Caller (patroller): \"{noisy(join_sentences(said), rng, 0.15)}\"\n"
                f"Supervisor on call: {f['supervisor']}\n\n" + _std(rng, "TP-305", self._rules(f)))

    def questions(self):
        return {
            "patrol_response": choice("How should the findings be dealt with now?", {
                "record_for_planned_work": "Record them for the next planned maintenance visit; nothing more is needed now.",
                "fix_during_patrol": "Make the defect good on the spot during the patrol and record it as fixed.",
                "repair_within_week": "Book a maintenance gang to repair it within seven days, with no speed restriction.",
                "esr_and_urgent_repair": "Impose an emergency speed restriction and get it repaired within 36 hours.",
                "block_and_call_out": "Stop trains over the site and call out a maintenance gang now.",
            }),
            "remedial_work": choice("What work will actually put this site right?", {
                "fastening_renewal": "Replace the missing or broken fastenings.",
                "sleeper_renewal": "Renew the defective sleepers.",
                "drainage_and_packing": "Clear the drainage softening the bed, then lift and pack the voided sleepers.",
                "joint_overhaul": "Overhaul the fishplated joint with new bolts, and new fishplates if needed.",
            }),
            "remedial_deadline": score("Under the patrol standard, how soon must the worst defect found be put right?", [
                "At the next planned maintenance visit, within 13 weeks.",
                "Within 7 days.",
                "Within 36 hours, under a speed restriction until then.",
                "Before any further train passes over it.",
            ]),
            "deterioration_trend": score("How fast is the site deteriorating since the last patrol?", [
                "Not at all: no worse than last time.",
                "Slowly: a little worse than last time.",
                "Markedly: clearly worse within a week or two.",
                "Rapidly: defects are multiplying from patrol to patrol.",
            ]),
            "cluster_limit_reached": noul(
                "The run of defective assemblies reaches the standard's limit for this curve band.",
                yes="The consecutive count is at or above the limit the standard sets for this radius.",
                no="The consecutive count is below the limit for this radius."),
        }


# =========================================================================== schema 5
# work type: (label, step kt, full kt, new welds?)
WORKS = {
    "tamping": ("plain-line tamping", 10, 25, False),
    "ballast": ("ballast renewal", 40, 100, False),
    "sleepers": ("sleeper renewal (spot, 12 sleepers)", 20, 50, False),
    "renewal": ("plain-line track renewal with new welds", 60, 150, True),
    "lift": ("drainage works with a track lift", 25, 60, False),
}
GEOM_CHECKS = {
    "clean": "post-work recording: nothing at Alert level or worse",
    "alert": "post-work recording: one Alert-level top fault",
    "intervention": "post-work recording: an Intervention-level twist",
    "hand_clean": "hand measurement by the supervisor: nothing at Alert level or worse",
    "none_yet": "no post-work recording or hand measurement yet",
}


class SpeedRestrictionRemoval(Schema):
    name = "speed_restriction_removal"
    description = ("A works supervisor asks to raise or lift the temporary speed restriction left after track work: "
                   "what the post-works standard allows now.")
    surfaces = {"json": 0.5, "tsr_form": 0.25, "works_email": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        rt = _route(rng)
        units = rt["units"]
        work = pick(rng, list(WORKS))
        _, step_kt, full_kt, welds = WORKS[work]
        current = pick(rng, ["caution", "caution", "step"]) if work != "tamping" else pick(rng, ["caution", "step", "step"])
        hot = chance(rng, 0.5)
        if reg == "borderline":
            anchor = pick(rng, [full_kt, 1.2 * full_kt, step_kt, 1.5 * full_kt] if hot else [full_kt, full_kt, step_kt])
            kt = near(rng, anchor, 0.15 * full_kt, lo=1, integer=True)
        elif reg == "clear":
            kt = pick(rng, [int(step_kt * 0.5), int(full_kt * 2.2), int(full_kt * 1.25), int(full_kt * 1.3)] if hot
                      else [int(step_kt * 0.5), int(full_kt * 2.2), int(full_kt * 1.7), int((step_kt + full_kt) / 2)])
        else:
            kt = between(rng, step_kt * 0.6, full_kt * 1.8, integer=True)
            if hot and chance(rng, 0.6):
                kt = between(rng, full_kt, full_kt * 1.45, integer=True)
        if current == "step":
            kt = max(kt, step_kt)
        geom = pick_w(rng, {"clean": 6, "hand_clean": 2, "alert": 2, "intervention": 1.5, "none_yet": 0.6}
                      if reg != "clear" else {"clean": 6, "intervention": 1, "none_yet": 0.5})
        signed = pick_w(rng, {"signed": 12, "pending": 1})
        ballast = pick_w(rng, {"full": 6, "short": 2})
        certs = None
        if welds:
            certs = pick_w(rng, {"all": 6, "missing_some": 2})
        if reg == "conflicting":
            twist = pick(rng, ["pushed_without_records", "fault_found", "none"])
            if twist == "pushed_without_records":       # asks for line speed with most of the paperwork outstanding
                signed, ballast = "pending", "short"
                geom = pick(rng, ["none_yet", "alert"])
                certs = "missing_some" if welds else None
            elif twist == "fault_found":                # a restriction already raised, then the recording finds a fault
                current, geom = "step", "intervention"
                kt = max(kt, step_kt)
        return {"regime": reg, **rt, "mileage": _mileage(rng, units),
                "line_speed": pick(rng, BAND_SPEEDS[units][pick(rng, [1, 2, 2, 3])]), "work": work,
                "current": current, "kt_since": kt, "days_since": max(1, int(kt / pick(rng, [9, 15, 22, 30]))),
                "geometry_check": geom, "handback": signed, "ballast_profile": ballast, "weld_certs": certs,
                "forecast_rail_max_c": pick(rng, [46, 47, 49, 52]) if hot else pick(rng, [24, 31, 36, 41, 44]),
                "supervisor": person(rng), "engineer": person(rng), "tsr_ref": ident(rng, "TSR", 4, "-"),
                "request_ts": self._ts(rng, hot),
                "ask": pick(rng, ["line_speed", "line_speed", "step"]) if current == "caution" else "line_speed",
                "pressure": pick(rng, [None, None, "timetable", "complaints", "possession_overrun"])}

    @staticmethod
    def _ts(rng, hot):
        t = rand_ts(rng, hours=(6, 19))
        return t.replace(month=pick(rng, [6, 7, 7, 8]), day=min(t.day, 28)) if hot else t

    def _allowed(self, f):
        """('caution' | 'step' | 'line' | 'current') the standard allows now."""
        _, step_kt, full_kt, welds = WORKS[f["work"]]
        if f["geometry_check"] == "intervention":
            return "caution"
        if f["handback"] != "signed" or f["geometry_check"] == "none_yet" or (welds and f["weld_certs"] != "all"):
            return f["current"]
        need = full_kt * (1.5 if f["forecast_rail_max_c"] > 45 else 1)
        if f["kt_since"] >= need and f["geometry_check"] in ("clean", "hand_clean") and f["ballast_profile"] == "full":
            return "line"
        if f["kt_since"] >= step_kt:
            return "step"
        return f["current"]

    def intended(self, f):
        _, step_kt, full_kt, _ = WORKS[f["work"]]
        allowed = self._allowed(f)
        order = {"caution": 0, "step": 1, "line": 2}
        if order[allowed] < order[f["current"]]:
            dec = "tighten_restriction"
        elif allowed == f["current"]:
            dec = "keep_restriction"
        elif allowed == "line":
            dec = "lift_to_line_speed"
        else:
            dec = "step_up"
        ev = ((f["handback"] == "signed") + (f["geometry_check"] != "none_yet")
              + (f["ballast_profile"] == "full") + (f["weld_certs"] in (None, "all")))
        evidence = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}[ev]
        ratio = f["kt_since"] / full_kt
        pts = (2 if ratio < 0.5 else 1 if ratio < 1 else 0)
        pts += {"clean": 0, "hand_clean": 0, "alert": 1, "intervention": 3, "none_yet": 1}[f["geometry_check"]]
        pts += (f["ballast_profile"] == "short") + (f["forecast_rail_max_c"] > 45 and ratio < 1.5)
        pts += (f["weld_certs"] == "missing_some")
        risk = 0 if pts == 0 else 1 if pts <= 1 else 2 if pts <= 3 else 3
        heat = f["forecast_rail_max_c"] > 45 and full_kt <= f["kt_since"] < 1.5 * full_kt
        return {"line_speed_decision": dec, "handback_evidence": evidence, "risk_if_lifted_now": risk,
                "consolidation_met": "yes" if f["kt_since"] >= full_kt else "no",
                "heat_rule_bites": "yes" if heat else "no"}

    def _rules(self, f):
        u = f["units"]
        lab, step_kt, full_kt, _ = WORKS[f["work"]]
        return [
            f"Consolidation tonnage for {lab}: step {step_kt} kt, full {full_kt} kt.",
            "Line speed may be restored only when the site has carried the full tonnage, a post-work recording or "
            "hand measurement shows nothing at Alert or worse, the ballast shoulder is at full profile, the handback "
            "certificate is signed and, where there are new welds, every weld has a test certificate.",
            "With the step tonnage carried but another line-speed condition unmet (a single Alert-level fault, a "
            f"short shoulder), the restriction may be raised to the step speed of {_spd(STEP[u], u)}.",
            "No raise of any kind without a signed handback, a post-work check and, for new welds, all weld "
            "certificates.",
            f"An Intervention-level fault after the work: lower the restriction to {_spd(CAUTION[u], u)} until "
            "corrected.",
            "If rail temperature is forecast above 45 C in the next 72 h, line speed needs 1.5 times the full tonnage.",
        ]

    def _speed_now(self, f):
        u = f["units"]
        return _spd(CAUTION[u] if f["current"] == "caution" else STEP[u], u)

    def _pressure(self, f):
        return {"timetable": "Operations want the restriction gone before the timetable change on Sunday.",
                "complaints": "Train operator has complained twice about the delay minutes this week.",
                "possession_overrun": "The possession overran on Saturday so we are already behind."}.get(f["pressure"])

    def render(self, f, rng, surface):
        u = f["units"]
        lab = WORKS[f["work"]][0]
        ask = "full line speed" if f["ask"] == "line_speed" else f"{_spd(STEP[u], u)}"
        certs = (None if f["weld_certs"] is None else
                 "all weld test certificates received" if f["weld_certs"] == "all"
                 else "test certificates outstanding for 2 of the new welds")
        if surface == "json":
            return dumps(prune({
                "request": {"tsr": f["tsr_ref"], "requested_by": f["supervisor"], "asks_for": ask,
                            "submitted_utc": fmt_ts(f["request_ts"], style="iso"), "note": self._pressure(f)},
                "site": {"railway": f["railway"], "line": f["line_code"], "track": f["track"], "section": f["section"],
                         "mileage": f["mileage"], "line_speed": _spd(f["line_speed"], u),
                         "current_restriction": self._speed_now(f)},
                "work": {"type": lab, "completed_days_ago": f["days_since"], "tonnage_since_kt": f["kt_since"],
                         "geometry": GEOM_CHECKS[f["geometry_check"]],
                         "ballast_shoulder": "full profile" if f["ballast_profile"] == "full" else "short, top-up booked",
                         "handback_certificate": f["handback"], "welds": certs},
                "forecast": {"max_rail_temperature_next_72h_c": f["forecast_rail_max_c"]},
                "standard": _std(rng, "TW-520", self._rules(f), json_ok=True),
            }))
        if surface == "tsr_form":
            rows = _kv([("TSR", f["tsr_ref"]), ("Line / track", f"{f['line_code']} {f['track']}"),
                             ("Section / mileage", f"{f['section']}, {f['mileage']}"),
                             ("Line speed", _spd(f["line_speed"], u)), ("Restriction now", self._speed_now(f)),
                             ("Raise requested to", ask), ("Work", lab),
                             ("Completed", f"{_plural(f['days_since'], 'day')} ago"), ("Tonnage since", f"{f['kt_since']} kt"),
                             ("Geometry", GEOM_CHECKS[f["geometry_check"]]),
                             ("Ballast shoulder", "full" if f["ballast_profile"] == "full" else "short (top-up booked)"),
                             ("Handback certificate", f["handback"]), ("Welds", certs),
                             ("Max rail temp forecast, 72 h", f"{f['forecast_rail_max_c']} C"),
                             ("Requested by", f["supervisor"])], rng)
            return f"TSR CHANGE REQUEST - {f['railway']}\n{rows}\n\n" + _std(rng, "TW-520", self._rules(f))
        para = [f"Requesting the {f['tsr_ref']} restriction on {f['line_code']} {f['track']} {f['mileage']} "
                f"({f['section']}) goes up to {ask}; it's currently {self._speed_now(f)} against a line speed of "
                f"{_spd(f['line_speed'], u)}.",
                f"Work was {lab}, finished {_plural(f['days_since'], 'day')} ago, and we make it about {f['kt_since']} kt since.",
                _ucfirst(GEOM_CHECKS[f["geometry_check"]]) + ".",
                "Shoulder is back to full profile." if f["ballast_profile"] == "full"
                else "Shoulder is still a bit short, top-up is booked for next week.",
                "Handback certificate is signed." if f["handback"] == "signed" else "Handback cert still to be signed off.",
                (_ucfirst(certs) + ".") if certs else "",
                f"Forecast has rail temps up to {f['forecast_rail_max_c']} C in the next three days.",
                self._pressure(f) or ""]
        return "\n".join([f"Subject: {f['tsr_ref']} - raise request", f"From: {f['supervisor']}",
                          f"To: {f['engineer']} (track engineer)", "", noisy(join_sentences(para), rng), "",
                          _std(rng, "TW-520", self._rules(f))])

    def questions(self):
        return {
            "line_speed_decision": choice("What should happen to the temporary speed restriction now?", {
                "lift_to_line_speed": "Lift the restriction and restore full line speed.",
                "step_up": "Raise it to the intermediate step speed and review again after more traffic.",
                "keep_restriction": "Keep the current restriction unchanged for now.",
                "tighten_restriction": "Lower the restriction further, because the checks after the work found a worse condition.",
            }),
            "handback_evidence": score("How complete is the evidence that the work was finished to standard?", [
                "Thin: most of the required records are missing.",
                "Partial: some key records are still outstanding.",
                "Mostly complete: one item is still missing.",
                "Complete: every record the standard asks for is in place.",
            ]),
            "risk_if_lifted_now": score("How much risk would restoring full line speed today carry?", [
                "Negligible.",
                "Low.",
                "Moderate.",
                "High.",
            ]),
            "consolidation_met": noul(
                "The site has carried the full consolidation tonnage the standard sets for this type of work.",
                yes="The tonnage since the work is at or above the full figure in the table for this work type.",
                no="The tonnage since the work is short of the full figure for this work type."),
            "heat_rule_bites": noul(
                "The forecast heat is the only reason the tonnage carried falls short of what line speed needs."),
        }


WORKFLOW = Workflow(
    name="rail_track_defect_response",
    title="Rail track defect response",
    description=("How a railway's track engineers respond to a track fault: speed restrictions, train holds, "
                 "verification and remediation timing against written track standards."),
    domain="transport operations / railway infrastructure maintenance",
    schemas=[GeometryExceedance(), UltrasonicRailDefect(), HeatBucklingPrecaution(), TrackPatrolFinding(),
             SpeedRestrictionRemoval()],
    notes="Fictional railways and standards; every limit is quoted in the state. No aircraft, no freight claims, "
          "no customer conversation.",
)
