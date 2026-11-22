"""Forestry and fisheries management: resource agencies apply written (fictional) codes to field and catch data.

Five schemas: a forest district's review of a timber harvest plan against riparian-buffer, slope, opening-size,
retention and hazard rules; a fire-season ruling on a logging operation under an operations fire-restriction level;
a salvage proposal after windthrow, wildfire, ice storm or bark-beetle attack; an in-season fishery closure decision
from landings and catch-rate projections against the quota trigger; and a bycatch-limit decision from observer
estimates against hotspot and sector caps. Each state quotes only the clauses that decide the case, with thresholds
sampled per case: a 26 m buffer on a stream that needs 30, a 13:00 humidity reading one point under the waiver bar,
a projection that reaches the trigger nine days before the season ends.

Forest types, disturbance agents, fishery seasons and gear types are kept consistent with each other (spruce beetle
breeds in spruce windthrow, crab seasons open in winter, fire-restriction levels apply in summer). Agencies,
districts, codes and people are invented. `intended(f)` records the answer logic a careful forester, fire officer or
fishery manager would apply to the rendered facts; the builder never uses it.
"""
from __future__ import annotations

from datetime import date, timedelta

from .base import (
    ORG_A,
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
    near,
    noisy,
    noul,
    person,
    pick,
    pick_w,
    prune,
    regime,
    score,
    table,
)

JURIS = ["Province of Northvale", "State of Westmarch", "Calder Territory", "Ashcombe Region", "State of Lindmoor",
         "Province of Eastholm", "Brayfield Commonwealth", "State of Orrinford"]


def _clamp(x, lo=0, hi=3):
    return max(lo, min(hi, x))


def _district(rng):
    return f"{pick(rng, ORG_A)} {pick(rng, ['Forest District', 'Resource District', 'Natural Resource District'])}"


# =========================================================================== schema 1: timber harvest plan
# (forest type, operating window text, wet-season window?)
FOREST_TYPES = [("coastal western hemlock-Douglas-fir", "Nov 1 - Mar 31", True), ("coastal western hemlock-Douglas-fir", "Jun 15 - Oct 15", False),
                ("interior spruce-subalpine fir", "Jul 1 - Oct 31", False), ("northern hardwoods (sugar maple, yellow birch)", "Dec 15 - Mar 15 (frozen ground)", False),
                ("northern hardwoods (sugar maple, yellow birch)", "Apr 1 - May 31 (spring thaw)", True), ("loblolly pine plantation", "year-round", True),
                ("ponderosa pine", "Jun 1 - Oct 31", False), ("lodgepole pine", "Jul 1 - Nov 15", False),
                ("oak-hickory", "Sep 1 - Feb 28", False), ("Sitka spruce-western hemlock", "Oct 15 - Apr 15", True)]
SYSTEMS = ["clearcut", "clearcut", "variable retention", "shelterwood", "group selection"]
CLASS_NAMES = {1: "Class 1 (fish-bearing)", 2: "Class 2 (perennial, no fish)", 3: "Class 3 (seasonal)"}


class TimberHarvestPlanReview(Schema):
    name = "timber_harvest_plan_review"
    description = "A forest district's review of a timber harvest plan against the forest practices code's buffer, slope, opening and hazard rules."
    surfaces = {"json": 0.5, "plan_summary": 0.25, "field_check": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        ftype, window, wet = pick(rng, FOREST_TYPES)
        system = pick(rng, SYSTEMS)
        b = {1: pick(rng, [20, 30, 50]), 2: pick(rng, [10, 15, 20]), 3: pick(rng, [5, 10])}
        slope_lim, open_max, ret_min = pick(rng, [30, 35, 40]), pick(rng, [40, 60, 80]), pick(rng, [10, 15])
        nest_m = pick(rng, [200, 400])
        scen = pick_w(rng, {"clean": 2.6, "buffer": 3.2, "slope": 1.8, "retention": 1.6, "hazard": 2})
        n_streams = pick(rng, [1, 1, 2, 2, 3])
        streams = []
        for _ in range(n_streams):
            c = pick_w(rng, {1: 2, 2: 3, 3: 3})
            streams.append([c, b[c] + pick(rng, [0, 0, 5, 10])])
        area = round(rng.uniform(8, open_max * 1.0), 1)
        ret = ret_min + rng.randint(0, 8)
        max_slope = slope_lim - rng.randint(2, 12)
        steep_ha = 0.0
        unstable, geotech = chance(rng, 0.25), "complete"
        nest = None
        severity = "major" if reg in ("clear", "conflicting") else pick(rng, ["minor", "minor", "major"])
        if scen == "buffer":
            i = rng.randrange(n_streams)
            c = streams[i][0]
            if severity == "minor":
                short = rng.uniform(0.04, 0.1) if c == 1 else rng.uniform(0.1, 0.25)
            else:
                short = rng.uniform(0.15, 0.5) if c == 1 else rng.uniform(0.3, 0.6)
            streams[i][1] = max(0, int(round(b[c] * (1 - short))))
        elif scen == "slope":
            if severity == "minor":
                max_slope, steep_ha = slope_lim + rng.randint(1, 5), round(rng.uniform(0.3, 2.0), 1)
            else:
                max_slope, steep_ha = slope_lim + rng.randint(6, 20), round(rng.uniform(2.5, 12), 1)
        elif scen == "retention":
            if system in ("clearcut", "variable retention"):
                if severity == "minor":
                    ret = ret_min - pick(rng, [1, 2])
                else:
                    ret = max(0, ret_min - rng.randint(4, 9))
            else:
                area = round(open_max * (rng.uniform(1.02, 1.1) if severity == "minor" else rng.uniform(1.2, 1.6)), 1)
                system = "clearcut"
        elif scen == "hazard":
            if chance(rng, 0.5):
                unstable, geotech = True, pick(rng, ["pending", "not done"])
            else:
                nest = rng.randint(int(nest_m * 0.3), nest_m - 20)
        if unstable and scen != "hazard":
            geotech = "complete"
        if (reg == "conflicting" and chance(rng, 0.6)) or (reg == "clear" and scen not in ("clean", "hazard") and chance(rng, 0.4)):
            # a second issue riding along with the first
            if scen != "buffer":
                j = rng.randrange(n_streams)
                c = streams[j][0]
                streams[j][1] = int(round(b[c] * (1 - pick(rng, [0.08, 0.2, 0.4]))))
            else:
                max_slope, steep_ha = slope_lim + pick(rng, [2, 8, 14]), round(rng.uniform(0.5, 6), 1)
        if system not in ("clearcut", "variable retention"):
            ret = None
        if nest is None and chance(rng, 0.15):
            nest = rng.randint(nest_m + 50, nest_m * 3)
        crossings = pick(rng, [0, 1, 1, 2, 3, 4])
        return {
            "regime": reg, "scenario": scen, "forest_type": ftype, "window": window, "wet_season": wet, "system": system,
            "buffers": b, "slope_limit": slope_lim, "open_max": open_max, "retention_min": ret_min, "nest_buffer": nest_m,
            "streams": streams, "area_ha": area, "retention_pct": ret, "max_ground_slope": max_slope, "steep_ground_ha": steep_ha,
            "unstable": unstable, "geotech": geotech if unstable else None, "nest_m": nest, "crossings": crossings,
            "juris": pick(rng, JURIS), "district": _district(rng), "licensee": f"{pick(rng, ORG_A)} {pick(rng, ['Timber', 'Forest Products', 'Lumber', 'Woodlands'])}",
            "forester": person(rng), "unit": f"CB-{rng.randint(100, 899)}-{pick(rng, 'ABCDE')}", "code": f"FPC-{rng.randint(2018, 2025)}",
        }

    def _issues(self, f):
        out = []
        b = f["buffers"]
        for c, w in f["streams"]:
            if w < b[c]:
                short = (b[c] - w) / b[c]
                major = short > 0.1 if c == 1 else short > 0.25
                out.append(("riparian_buffers", "major" if major else "minor"))
        if f["steep_ground_ha"] > 0 and f["max_ground_slope"] > f["slope_limit"]:
            major = f["max_ground_slope"] - f["slope_limit"] > 5 or f["steep_ground_ha"] > 2
            out.append(("steep_slope_ground_skidding", "major" if major else "minor"))
        if f["retention_pct"] is not None and f["retention_pct"] < f["retention_min"]:
            out.append(("retention_or_opening_size", "major" if f["retention_min"] - f["retention_pct"] > 2 else "minor"))
        if f["system"] == "clearcut" and f["area_ha"] > f["open_max"]:
            out.append(("retention_or_opening_size", "major" if f["area_ha"] > f["open_max"] * 1.1 else "minor"))
        if f["unstable"] and f["geotech"] != "complete":
            out.append(("unassessed_hazard", "hazard"))
        if f["nest_m"] is not None and f["nest_m"] < f["nest_buffer"]:
            out.append(("unassessed_hazard", "hazard"))
        return out

    def intended(self, f):
        iss = self._issues(f)
        kinds = [k for k, _ in iss]
        sev = [s for _, s in iss]
        if "hazard" in sev:
            dec, main = "hold_for_specialist_assessment", "unassessed_hazard"
        elif "major" in sev:
            dec, main = "return_for_revision", [k for k, s in iss if s == "major"][0]
        elif iss:
            dec, main = "approve_with_conditions", kinds[0]
        else:
            dec, main = "approve_as_submitted", "no_material_deficiency"
        b = f["buffers"]
        pts = sum(2 if (c <= 2 and (b[c] - w) / b[c] > (0.1 if c == 1 else 0.25)) else 1
                  for c, w in f["streams"] if w < b[c] and c <= 2)
        pts += (f["steep_ground_ha"] > 0 and f["max_ground_slope"] > f["slope_limit"]) + (f["crossings"] >= 3)
        pts += bool(f["wet_season"]) + (f["unstable"] and f["geotech"] != "complete")
        n_major = sev.count("major") + sev.count("hazard")
        dep = 0 if not iss else 1 if (len(iss) == 1 and n_major == 0) else 2 if n_major <= 1 else 3
        buffers_ok = all(w >= b[c] for c, w in f["streams"])
        return {"plan_decision": dec, "principal_deficiency": main, "sediment_delivery_risk": _clamp(pts),
                "departure_from_code": dep, "buffers_meet_code": "yes" if buffers_ok else "no"}

    def _rules(self, f, rng, short=False):
        b = f["buffers"]
        if short:
            return (f"{f['code']}: buffers C1 {b[1]} m / C2 {b[2]} m / C3 {b[3]} m; ground-based <= {f['slope_limit']}% slope; "
                    f"clearcut <= {f['open_max']} ha with >= {f['retention_min']}% retention; mapped unstable terrain needs a "
                    f"geotechnical assessment; no operations within {f['nest_buffer']} m of an active raptor nest.")
        r = [f"{f['code']} s.12: no-harvest riparian buffers of {b[1]} m (Class 1, fish-bearing), {b[2]} m (Class 2) and "
             f"{b[3]} m machine-free (Class 3)."]
        if f["scenario"] in ("slope", "retention") or chance(rng, 0.5):
            r.append(f"s.18: ground-based skidding only on slopes up to {f['slope_limit']}%; openings up to {f['open_max']} ha "
                     f"with at least {f['retention_min']}% of the area retained.")
        if f["unstable"] or f["nest_m"] is not None or chance(rng, 0.4):
            r.append(f"s.21: mapped unstable terrain needs a completed geotechnical assessment; no operations within "
                     f"{f['nest_buffer']} m of an active raptor nest without a specialist's timing plan.")
        return r

    def _stream_txt(self, f):
        return [f"{CLASS_NAMES[c]}: {w} m buffer" for c, w in f["streams"]]

    def render(self, f, rng, surface):
        geo = None if not f["unstable"] else f"mapped unstable terrain; geotechnical assessment {f['geotech']}"
        nest = None if f["nest_m"] is None else f"active raptor nest {f['nest_m']} m from the unit boundary"
        ret = f"{f['retention_pct']}%" if f["retention_pct"] is not None else None
        steep = (f"{f['steep_ground_ha']} ha ground-based at up to {f['max_ground_slope']}%" if f["steep_ground_ha"]
                 else f"max ground-based slope {f['max_ground_slope']}%")
        if surface == "json":
            return dumps(prune({
                "plan": {"unit": f["unit"], "licensee": f["licensee"], "forest_type": f["forest_type"], "system": f["system"],
                         "area_ha": f["area_ha"], "operating_window": f["window"], "retention": ret},
                "streams": self._stream_txt(f), "terrain": {"ground_based": steep, "stability": geo, "road_crossings": f["crossings"]},
                "wildlife": nest, "code": self._rules(f, rng),
            }))
        if surface == "plan_summary":
            rows = [("Unit", f"{f['unit']} ({f['licensee']})"), ("District", f["district"]), ("Forest type", f["forest_type"]),
                    ("System / area", f"{f['system']}, {f['area_ha']} ha"), ("Retention", ret), ("Operating window", f["window"]),
                    ("Streams", "; ".join(self._stream_txt(f))), ("Ground-based", steep), ("Stability", geo),
                    ("Road crossings", f["crossings"]), ("Wildlife", nest)]
            return "HARVEST PLAN SUMMARY\n" + kv_lines(rows, rng) + "\n\nCode limits: " + self._rules(f, rng, short=True)
        para = [f"Walked {f['unit']} ({f['area_ha']} ha {f['system']}, {f['forest_type']}).",
                "Streams flagged: " + "; ".join(self._stream_txt(f)) + ".", f"Ground-based: {steep}.",
                (f"Retention patches total {ret}." if ret else ""), (f"Stability: {geo}." if geo else ""),
                (f"{nest.capitalize()}." if nest else ""), f"{f['crossings'] or 'No'} road crossing{'' if f['crossings'] == 1 else 's'}; planned window {f['window']}."]
        return (f"Field check - {f['forester']}, {f['district']}\n\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n"
                + bullets(self._rules(f, rng), rng))

    def questions(self):
        return {
            "plan_decision": choice("What should the district decide on this harvest plan?", {
                "approve_as_submitted": "Approve the plan as submitted.",
                "approve_with_conditions": "Approve it with conditions that fix small departures from the code.",
                "return_for_revision": "Return it for revision; it departs materially from the code.",
                "hold_for_specialist_assessment": "Hold it until a specialist assesses an unassessed hazard.",
            }),
            "principal_deficiency": choice("What is the plan's most serious problem?", {
                "no_material_deficiency": "None: the plan meets the code.",
                "riparian_buffers": "Stream buffers narrower than the code requires.",
                "steep_slope_ground_skidding": "Ground-based skidding on slopes steeper than allowed.",
                "retention_or_opening_size": "Too little retention or an opening larger than allowed.",
                "unassessed_hazard": "Unstable terrain or an active nest that has not been dealt with.",
            }),
            "sediment_delivery_risk": score("How much risk does the plan pose of delivering sediment to streams?", [
                "Low.",
                "Moderate.",
                "High.",
                "Very high.",
            ]),
            "departure_from_code": score("How far does the plan depart from the forest practices code?", [
                "Not at all.",
                "Slightly: one minor departure.",
                "Materially: one serious departure or several minor ones.",
                "Fundamentally: several serious departures.",
            ]),
            "buffers_meet_code": noul(
                "Every proposed riparian buffer meets the code's minimum width for its stream class.",
                yes="Each stream's buffer is at least the width the code sets for its class.",
                no="At least one buffer is narrower than its class requires."),
        }


# =========================================================================== schema 2: fire-season operations
# activity -> (label, rule by level: "any" | "restricted" | "prohibited"), spark potential 0-2
ACTIVITIES = {
    "power_saw": ("power-saw felling in the unit", {1: "any", 2: "restricted", 3: "prohibited"}, 2),
    "mechanised": ("feller-buncher and skidder operations", {1: "any", 2: "any", 3: "restricted"}, 1),
    "cable_yarding": ("cable yarding", {1: "any", 2: "restricted", 3: "prohibited"}, 2),
    "welding": ("welding repairs at the landing", {1: "any", 2: "restricted", 3: "restricted"}, 2),
    "road_grading": ("road grading and rock crushing", {1: "any", 2: "restricted", 3: "restricted"}, 1),
    "hauling": ("loading and hauling", {1: "any", 2: "any", 3: "any"}, 0),
}
HOURS = {"06:00-17:00": (6, 17), "05:00-13:00": (5, 13), "04:00-12:30": (4, 12.5), "07:00-15:00": (7, 15),
         "06:00-14:30": (6, 14.5), "20:00-04:00": (20, 28)}


class FireSeasonOperations(Schema):
    name = "fire_season_operations"
    description = "Whether a logging-site activity may run, and when, under the operations fire-restriction level in force."
    surfaces = {"json": 0.5, "district_notice": 0.25, "site_inspection": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        level = pick_w(rng, {1: 2, 2: 4, 3: 3, 4: 0.8})
        act = pick_w(rng, {"power_saw": 2, "mechanised": 2, "cable_yarding": 1.5, "welding": 1.2, "road_grading": 1.2, "hauling": 0.8})
        hours = pick(rng, list(HOURS))
        rh_min, tender_min = pick(rng, [30, 35, 40]), pick(rng, [1500, 2000, 3000])
        rh = rng.randint(rh_min - 12, rh_min + 18)
        tender = pick_w(rng, {0: 1.7, tender_min - 500: 1.2, tender_min: 2, tender_min + 1000: 2.4})
        if reg == "borderline":
            rh = near(rng, rh_min, 3, integer=True)
            if chance(rng, 0.4):
                tender = pick(rng, [tender_min - 500, tender_min])
        elif reg == "clear":
            if chance(rng, 0.5):
                level, hours = pick(rng, [1, 2]), pick(rng, ["05:00-13:00", "04:00-12:30"])
            else:
                level = pick(rng, [3, 4])
                act = pick(rng, ["power_saw", "cable_yarding"]) if level == 3 else act
        else:
            twist = pick(rng, ["humid_but_prohibited", "dry_with_tender", "night_shift"])
            if twist == "humid_but_prohibited":
                level, act, rh, tender = 3, pick(rng, ["power_saw", "cable_yarding"]), rh_min + rng.randint(5, 20), tender_min + 1000
            elif twist == "dry_with_tender":
                level, rh, tender = pick(rng, [2, 3]), rh_min - rng.randint(1, 8), tender_min + 1000
                hours = pick(rng, ["06:00-17:00", "07:00-15:00"])
            else:
                level, hours = pick(rng, [2, 3]), "20:00-04:00"
        if level in (2, 3) and act != "hauling" and chance(rng, 0.5):
            hours = pick(rng, ["06:00-17:00", "07:00-15:00", "06:00-14:30"])
        watch_req = {1: 1, 2: 2, 3: 3, 4: 3}[level]
        watch = pick_w(rng, {watch_req: 4, watch_req + 1: 1.5, max(0, watch_req - 1): 1.5})
        day = date(rng.choice([2025, 2026]), pick(rng, [6, 7, 7, 8, 8, 9]), rng.randint(1, 28))
        return {
            "regime": reg, "level": level, "activity": act, "hours": hours, "rh_min": rh_min, "tender_min": tender_min,
            "rh_1300": rh, "tender_l": tender, "watch_hours": watch, "watch_required": watch_req,
            "wind_kmh": pick(rng, [5, 10, 15, 20, 30, 40]), "arresters_ok": chance(rng, 0.7), "response_km": pick(rng, [4, 12, 25, 45, 60]),
            "recent_ignition": chance(rng, 0.2), "date": day, "juris": pick(rng, JURIS), "district": _district(rng),
            "operator": f"{pick(rng, ORG_A)} {pick(rng, ['Logging', 'Contracting', 'Timber Services', 'Yarding'])}",
            "officer": person(rng), "permit": ident(rng, "OP", 5, "-"), "code": f"OFR-{rng.randint(2, 9)}",
        }

    def _status(self, f):
        if f["level"] == 4:
            return "shutdown"
        return ACTIVITIES[f["activity"]][1][f["level"]]

    def _fits_window(self, f):
        a, b = HOURS[f["hours"]]
        return b <= 13 or a >= 20

    def _waiver_ok(self, f):
        return f["level"] in (2, 3) and f["rh_1300"] >= f["rh_min"] and f["tender_l"] >= f["tender_min"]

    def intended(self, f):
        st = self._status(f)
        if st == "shutdown":
            rul = "shut_down_site"
            ok = False
        elif st == "prohibited":
            rul, ok = "prohibit_activity", False
        elif st == "any" or self._fits_window(f):
            rul, ok = "allow_as_requested", True
        elif self._waiver_ok(f):
            rul, ok = "allow_under_waiver", False
        else:
            rul, ok = "allow_restricted_hours", False
        lvl_pts = {1: 0, 2: 1, 3: 2, 4: 3}[f["level"]]
        spark = ACTIVITIES[f["activity"]][2]
        hz = lvl_pts + (spark == 2) - (spark == 0) + (f["wind_kmh"] >= 30) + (f["rh_1300"] < 25) - (f["rh_1300"] >= 45)
        hz = _clamp(hz, 0, 4)
        rd = (f["tender_l"] >= f["tender_min"]) + (f["watch_hours"] >= f["watch_required"]) + f["arresters_ok"]
        rd = rd - (f["response_km"] >= 40) if rd > 0 else 0
        return {"operations_ruling": rul, "ignition_hazard": hz, "suppression_readiness": _clamp(rd),
                "activity_permitted_as_requested": "yes" if ok else "no",
                "waiver_conditions_met": "yes" if self._waiver_ok(f) else "no"}

    def _rules(self, f):
        a = ACTIVITIES[f["activity"]]
        return [f"{f['code']} Level {f['level']}: " + {
                    1: "fire-season precautions; all activities at normal hours; fire watch 1 h after shutdown.",
                    2: "power saws (outside landings), cable yarding, welding and road grading only 20:00-13:00; fire watch 2 h.",
                    3: "power-saw felling and cable yarding prohibited; mechanised felling, welding and grading only 20:00-13:00; fire watch 3 h.",
                    4: "general shutdown of all operations."}[f["level"]] + " Loading and hauling may continue below Level 4.",
                f"{f['code']} waiver: at Levels 2-3 the district may lift the 13:00 limit (not a prohibition) for the day when "
                f"site relative humidity at 13:00 is >= {f['rh_min']}% and a staffed water tender of >= {f['tender_min']:,} L is on site."]

    def _site(self, f):
        return {"rh_at_1300": f"{f['rh_1300']}%", "wind": f"{f['wind_kmh']} km/h",
                "water_tender": f"{f['tender_l']:,} L, staffed" if f["tender_l"] else "none on site",
                "fire_watch_planned": f"{f['watch_hours']} h after shutdown" if f["watch_hours"] else "none planned",
                "spark_arresters": "inspected" if f["arresters_ok"] else "not inspected this week",
                "nearest_suppression_crew": f"{f['response_km']} km"}

    def render(self, f, rng, surface):
        act = ACTIVITIES[f["activity"]][0]
        site = self._site(f)
        ign = "small roadside ignition 3 km away last week" if f["recent_ignition"] else None
        if surface == "json":
            return dumps(prune({
                "request": {"permit": f["permit"], "operator": f["operator"], "activity": act, "hours": f["hours"],
                            "date": f["date"].isoformat()},
                "restriction": {"level_in_force": f["level"], "district": f["district"]}, "site": site,
                "recent_fire_activity": ign, "rules": self._rules(f),
            }))
        if surface == "district_notice":
            head = (f"{f['district'].upper()} - OPERATIONS FIRE RESTRICTION LEVEL {f['level']} from "
                    f"{fmt_date(f['date'], rng, 'long')}\n{self._rules(f)[0]}\nWaiver: >= {f['rh_min']}% RH at 13:00 "
                    f"and >= {f['tender_min']:,} L staffed tender (Levels 2-3, hour limits only).")
            req = kv_lines([("Operator request", f"{f['operator']} - {act}, {f['hours']}"), ("Permit", f["permit"])] +
                           [(k.replace("_", " ").capitalize(), v) for k, v in site.items()] + [("Recent fire", ign)], rng)
            return head + "\n\n" + req
        para = [f"Visited {f['operator']} ({f['permit']}) on {fmt_date(f['date'], rng)}: they want to run {act}, {f['hours']}.",
                f"Level {f['level']} is in force.", f"13:00 RH {f['rh_1300']}%, wind {f['wind_kmh']} km/h.",
                f"Water tender: {site['water_tender']}.", f"Fire watch after shutdown: {site['fire_watch_planned']}; spark arresters {site['spark_arresters']}.",
                f"Nearest crew {f['response_km']} km.", (f"Note: {ign}." if ign else "")]
        return (f"Site inspection - {f['officer']}\n\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n"
                + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "operations_ruling": choice("How should the fire officer rule on the requested operation?", {
                "allow_as_requested": "Allow it as requested.",
                "allow_restricted_hours": "Allow it only within the hours the restriction level permits.",
                "allow_under_waiver": "Allow the requested hours under a humidity-and-water-supply waiver.",
                "prohibit_activity": "Prohibit this activity at the current level; other permitted work may continue.",
                "shut_down_site": "Shut the whole site down.",
            }),
            "ignition_hazard": score("How likely is this operation to start a fire under today's conditions?", [
                "Low.",
                "Moderate.",
                "High.",
                "Very high.",
                "Extreme.",
            ]),
            "suppression_readiness": score("How ready is the site to catch and suppress an ignition?", [
                "Unprepared.",
                "Thin: one safeguard in place.",
                "Adequate: most safeguards in place.",
                "Well prepared: water, watch and equipment all meet the rules.",
            ]),
            "activity_permitted_as_requested": noul(
                "Under the restriction level in force, this activity may run during the hours requested without a waiver.",
                yes="The level allows the activity, and the requested hours fall within the allowed window.",
                no="The level prohibits the activity or the requested hours fall outside the allowed window."),
            "waiver_conditions_met": noul(
                "The site meets every condition the rules set for an hours waiver."),
        }


# =========================================================================== schema 3: salvage after disturbance
# (disturbance, stand, beetle-prone?, burned?, months to serious value loss)
DISTURBANCES = [("wildfire", "ponderosa pine", False, True, 18), ("wildfire", "lodgepole pine", False, True, 18),
                ("wildfire", "mixed conifer (Douglas-fir, grand fir)", False, True, 15),
                ("windthrow", "Engelmann spruce-subalpine fir", True, False, 14), ("windthrow", "Sitka spruce-western hemlock", False, False, 12),
                ("windthrow", "white spruce-aspen", True, False, 14), ("ice storm", "northern hardwoods (sugar maple, yellow birch)", False, False, 10),
                ("spruce beetle attack", "Engelmann spruce", True, False, 24), ("mountain pine beetle attack", "lodgepole pine", False, False, 30)]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


class SalvageAfterDisturbance(Schema):
    name = "salvage_after_disturbance"
    description = "A salvage-logging proposal after wildfire, windthrow, ice storm or beetle attack, checked against the salvage directive."
    surfaces = {"json": 0.5, "salvage_proposal": 0.25, "silviculture_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        dist, stand, beetle_prone, burned, decay = pick(rng, DISTURBANCES)
        win = pick(rng, [12, 18, 24])
        snags_min, slope_lim = pick(rng, [5, 8, 10]), pick(rng, [30, 35, 40])
        flight = pick(rng, ["May", "June"])
        scen = pick_w(rng, {"full": 3, "partial": 2.5, "sanitation": 1.6 if beetle_prone else 0, "defer": 2.2 if beetle_prone else 0,
                            "late": 2.8, "uneconomic": 1})
        months = rng.randint(2, win - 3)
        if scen == "late":
            months = win + (pick(rng, [1, 2]) if reg == "borderline" else rng.randint(2, 14))
        elif reg == "borderline" and chance(rng, 0.4):
            months = win - pick(rng, [0, 1, 2])
        net = round(rng.uniform(8, 45), 0)
        if scen == "uneconomic":
            net = round(-rng.uniform(1, 12), 0) if reg != "borderline" else round(rng.uniform(-3, -1), 0)
        net = net - round(max(0, months - decay) * 1.5)
        brood = None
        if beetle_prone:
            brood = {"sanitation": "live brood in down logs", "defer": "not surveyed"}.get(scen, pick(rng, ["none found", "none found", "not surveyed"]) if scen != "full" else "none found")
        steep_burn = 0.0
        riparian = 0.0
        snags = snags_min + rng.randint(0, 6)
        if scen == "partial":
            which = pick(rng, ["steep", "riparian", "snags"] if burned else ["riparian", "snags"])
            if which == "steep":
                steep_burn = round(rng.uniform(3, 25), 1)
            elif which == "riparian":
                riparian = round(rng.uniform(1, 8), 1)
            else:
                snags = max(1, snags_min - (pick(rng, [1, 2]) if reg == "borderline" else rng.randint(2, 4)))
        sev_high = round(rng.uniform(0.2, 0.8), 2) if burned else None
        soil = pick_w(rng, {"well-drained gravelly loam": 3, "fine silty soils": 2, "wet clay soils": 1.5})
        if reg == "conflicting":
            twist = pick(rng, ["valuable_but_late", "brood_and_steep", "cheap_but_clean"])
            if twist == "valuable_but_late":
                months, net = win + rng.randint(1, 4), round(rng.uniform(30, 60), 0)
            elif twist == "brood_and_steep" and beetle_prone:
                brood, riparian = "live brood in down logs", round(rng.uniform(1, 5), 1)
            else:
                net = round(rng.uniform(2, 8), 0)
        now = pick(rng, MONTHS)
        return {
            "regime": reg, "scenario": scen, "disturbance": dist, "stand": stand, "beetle_prone": beetle_prone, "burned": burned,
            "window_months": win, "snags_min": snags_min, "slope_limit": slope_lim, "flight_month": flight,
            "months_since": months, "net_value": net, "brood": brood, "steep_burned_ha": steep_burn, "riparian_ha": riparian,
            "snags": snags, "high_severity_share": sev_high, "soil": soil, "now_month": now,
            "affected_ha": (aff := rng.randint(60, 900)), "proposed_ha": int(aff * rng.uniform(0.25, 0.9)), "volume_m3_ha": rng.randint(90, 420),
            "decay_note": ("checking and blue stain throughout" if months > decay else "sapwood still sound; light checking"),
            "district": _district(rng), "juris": pick(rng, JURIS), "licensee": f"{pick(rng, ORG_A)} {pick(rng, ['Timber', 'Forest Products', 'Woodlands'])}",
            "author": person(rng), "ref": ident(rng, "SAL", 4, "-"), "code": f"SD-{rng.randint(2, 9)}",
        }

    def intended(self, f):
        within = f["months_since"] <= f["window_months"]
        brood = f["brood"]
        if not within or (f["net_value"] <= 0 and brood != "live brood in down logs"):
            dec = "decline_salvage"
        elif brood == "not surveyed":
            dec = "defer_and_reassess"
        elif brood == "live brood in down logs":
            dec = "approve_priority_sanitation"
        elif f["steep_burned_ha"] > 0 or f["riparian_ha"] > 0 or f["snags"] < f["snags_min"]:
            dec = "approve_partial_salvage"
        else:
            dec = "approve_full_salvage"
        if dec == "decline_salvage":
            win = "no_operations_this_season"
        elif f["beetle_prone"] and brood != "none found":
            win = "before_next_beetle_flight"
        elif f["soil"] == "wet clay soils" or (f["burned"] and (f["high_severity_share"] or 0) >= 0.6):
            win = "frozen_ground_winter_only"
        else:
            win = "start_immediately"
        ero = (f["steep_burned_ha"] > 0) + {"well-drained gravelly loam": 0, "fine silty soils": 1, "wet clay soils": 2}[f["soil"]]
        ero += f["riparian_ha"] > 0
        ero += (f["high_severity_share"] or 0) >= 0.5
        nv = f["net_value"]
        val = 0 if nv <= 0 or not within else 1 if nv < 12 else 2 if nv < 28 else 3
        return {"salvage_decision": dec, "operating_window": win, "erosion_hazard": _clamp(ero), "recoverable_value": val,
                "within_salvage_window": "yes" if within else "no"}

    def _rules(self, f, rng):
        r = [f"{f['code']} §2: salvage may proceed only within {f['window_months']} months of the disturbance and where net "
             f"log value is positive; retain at least {f['snags_min']} large snags or logs per hectare; riparian reserves excluded."]
        if f["burned"]:
            r.append(f"§3: no ground-based salvage on burned slopes over {f['slope_limit']}% with high burn severity.")
        if f["beetle_prone"]:
            r.append(f"§4: spruce with live beetle brood must be removed or debarked before the {f['flight_month']} flight; "
                     "stands not surveyed for brood are surveyed before any decision.")
        return r

    def render(self, f, rng, surface):
        sev = f"{round(f['high_severity_share'] * 100)}% high severity" if f["high_severity_share"] is not None else None
        items = {"disturbance": f"{f['disturbance']} in {f['stand']}, {f['months_since']} months ago",
                 "affected_ha": f["affected_ha"], "proposed_salvage_ha": f["proposed_ha"], "volume_m3_per_ha": f["volume_m3_ha"],
                 "wood_condition": f["decay_note"], "net_log_value_per_m3": f"${f['net_value']:.0f}", "burn_severity": sev,
                 "beetle_brood_survey": f["brood"], "steep_burned_ground_in_proposal_ha": f["steep_burned_ha"] or None,
                 "riparian_reserve_inside_proposal_ha": f["riparian_ha"] or None, "snags_retained_per_ha": f["snags"],
                 "soils": f["soil"], "current_month": f["now_month"]}
        if surface == "json":
            return dumps(prune({"proposal": {"ref": f["ref"], "licensee": f["licensee"], "district": f["district"]},
                                "site": items, "directive": self._rules(f, rng)}))
        if surface == "salvage_proposal":
            labels = {"affected_ha": "Area affected (ha)", "proposed_salvage_ha": "Proposed salvage (ha)",
                      "volume_m3_per_ha": "Volume (m3/ha)", "net_log_value_per_m3": "Net log value ($/m3)",
                      "steep_burned_ground_in_proposal_ha": "Steep burned ground in proposal (ha)",
                      "riparian_reserve_inside_proposal_ha": "Riparian reserve inside proposal (ha)",
                      "snags_retained_per_ha": "Snags retained (per ha)"}
            rows = [(labels.get(k, k.replace("_", " ").capitalize()), v) for k, v in items.items()]
            return (f"SALVAGE PROPOSAL {f['ref']} - {f['licensee']}\n" + kv_lines(rows, rng)
                    + "\n\nDirective: " + " ".join(self._rules(f, rng)))
        para = [f"{f['licensee']} proposes salvaging {f['proposed_ha']} ha of the {f['affected_ha']} ha hit by {f['disturbance']} "
                f"in {f['stand']} {f['months_since']} months ago.", f"Standing volume about {f['volume_m3_ha']} m3/ha; {f['decay_note']}; "
                f"net log value ${f['net_value']:.0f}/m3.", (f"Burn severity {sev}." if sev else ""),
                (f"Beetle brood survey: {f['brood']}." if f["brood"] else ""),
                (f"{f['steep_burned_ha']} ha of the proposal is steep burned ground." if f["steep_burned_ha"] else ""),
                (f"{f['riparian_ha']} ha falls inside a riparian reserve." if f["riparian_ha"] else ""),
                f"Snag retention planned at {f['snags']}/ha. Soils: {f['soil']}. It is now {f['now_month']}."]
        return (f"Silviculture memo - {f['author']}, {f['district']}\n\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n"
                + bullets(self._rules(f, rng), rng))

    def questions(self):
        return {
            "salvage_decision": choice("What should the district decide on this salvage proposal?", {
                "approve_full_salvage": "Approve the salvage as proposed.",
                "approve_partial_salvage": "Approve it with the non-compliant ground or retention shortfall taken out or fixed.",
                "approve_priority_sanitation": "Approve it as urgent sanitation: remove or treat infested wood first.",
                "defer_and_reassess": "Defer until the missing survey is done.",
                "decline_salvage": "Decline it.",
            }),
            "operating_window": choice("When should salvage operations run?", {
                "start_immediately": "As soon as possible, before more value is lost.",
                "before_next_beetle_flight": "Before the next beetle flight.",
                "frozen_ground_winter_only": "Only on frozen ground in winter, to protect the soil.",
                "no_operations_this_season": "Not at all this season.",
            }),
            "erosion_hazard": score("How serious is the erosion and sediment hazard from salvaging this area?", [
                "Low.",
                "Moderate.",
                "High.",
                "Severe.",
            ]),
            "recoverable_value": score("How much timber value can salvage still recover under the directive?", [
                "None: salvage is barred or would lose money.",
                "Marginal.",
                "Moderate.",
                "Substantial.",
            ]),
            "within_salvage_window": noul(
                "The proposed salvage falls within the period the directive allows after this disturbance.",
                yes="The time since the disturbance is within the directive's salvage window.",
                no="The salvage window has already closed."),
        }


# =========================================================================== schema 4: in-season fishery closure
# (fishery, gear, season start (month, day), season end (month, day), TAC range t)
FISHERIES = [("Northern Gulf Pacific cod A-season", "trawl", (1, 20), (6, 10), (8000, 25000)),
             ("Kestrel Bay Atlantic herring", "purse seine", (6, 1), (10, 31), (12000, 40000)),
             ("coastal Dungeness crab", "pot", (12, 1), (8, 15), (2500, 9000)),
             ("Western Shelf sablefish", "longline", (3, 15), (11, 15), (4000, 12000)),
             ("summer flounder, quarter 1", "otter trawl", (1, 1), (3, 31), (1500, 4500)),
             ("Gulf pink shrimp", "shrimp trawl", (4, 1), (10, 31), (6000, 15000)),
             ("Ashcombe Pacific halibut", "longline", (3, 20), (12, 7), (3000, 9000)),
             ("southern shortfin squid", "jig", (5, 1), (9, 30), (10000, 30000))]


class InseasonFisheryClosure(Schema):
    name = "inseason_fishery_closure"
    description = "An in-season decision on a quota-managed commercial fishery: stay open, slow the catch, schedule or order a closure."
    surfaces = {"json": 0.5, "landings_dashboard": 0.25, "manager_briefing": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        fish, gear, s0, s1, (tlo, thi) = pick(rng, FISHERIES)
        tac = round(rng.uniform(tlo, thi), -1)
        trig = pick(rng, [90, 95, 100])
        notice_h = pick(rng, [48, 72])
        year = pick(rng, [2025, 2026])
        start = date(year, *s0)
        end = date(year + (1 if s1 < s0 else 0), *s1)
        length = (end - start).days
        rep = start + timedelta(days=rng.randint(int(length * 0.35), int(length * 0.85)))
        days_left = (end - rep).days
        lag = pick(rng, [2, 3, 5, 7, 10])
        rate = tac * rng.uniform(0.03, 0.08)
        days_left = max(21, days_left)
        wl = days_left / 7
        nw = notice_h / 24 / 7
        scen = pick_w(rng, {"keep": 3.4, "slow": 2.0, "soon": 1.8, "now": 1.5, "data": 1.6})
        if scen == "keep":
            wk = wl * (near(rng, 1.06, 0.07) if reg == "borderline" else rng.uniform(1.12, 1.6))
        elif scen == "slow":
            wk = rng.uniform(3.4, max(3.6, wl * 0.92))
        elif scen == "soon":
            wk = near(rng, 3.0, 0.5) if reg == "borderline" else rng.uniform(nw + 0.6, 2.9)
        elif scen == "now":
            wk = near(rng, nw + 0.3, 0.3) if reg == "borderline" else rng.uniform(-0.6, nw + 0.2)
        else:
            wk = wl * rng.uniform(0.88, 1.15)
        trig_t = tac * trig / 100
        landed = max(tac * 0.1, trig_t - wk * rate - rate * lag / 7)
        rates = [round(rate * rng.uniform(0.8, 1.2)), round(rate * rng.uniform(0.85, 1.15)), round(rate)]
        missing = pick_w(rng, {0: 4, 4: 2, 8: 1.5, 15: 0.6, 25: 0.4})
        disc = pick_w(rng, {1: 4, 3: 2, 6: 0.6, 12: 0.4})
        if scen == "data" or (reg == "conflicting" and chance(rng, 0.3)):
            missing, disc = pick(rng, [(15, 2), (25, 3), (4, 8), (12, 12)])
        storm = chance(rng, 0.2)
        rep = end - timedelta(days=days_left)
        return {
            "regime": reg, "scenario": scen, "fishery": fish, "gear": gear, "tac_t": tac, "trigger_pct": trig, "notice_h": notice_h,
            "season_end": end, "report_date": rep, "days_left": days_left, "lag_days": lag, "landed_t": int(round(landed)),
            "weekly_t": rates, "missing_dealer_pct": missing, "logbook_discrepancy_pct": disc, "storm_forecast": storm,
            "vessels": rng.randint(12, 140), "trip_limit": round(rate / rng.randint(20, 60), 1), "overage_rule": True,
            "agency": f"{pick(rng, ['Northern', 'Western', 'Gulf', 'Coastal', 'Eastern'])} Fisheries Administration",
            "manager": person(rng), "fmp": f"FMP-{rng.randint(10, 60)}", "notice": ident(rng, "IFN", 4, "-"),
        }

    def _proj(self, f):
        rate = f["weekly_t"][-1]
        trig = f["tac_t"] * f["trigger_pct"] / 100
        now = f["landed_t"] + rate * f["lag_days"] / 7
        end = now + rate * f["days_left"] / 7
        wk = (trig - now) / rate if rate else 99
        return trig, now, end, wk

    def intended(self, f):
        trig, now, end, wk = self._proj(f)
        poor = f["missing_dealer_pct"] > 10 or f["logbook_discrepancy_pct"] > 5
        reach = end >= trig
        if now >= trig or wk <= f["notice_h"] / 24 / 7 + 0.3:
            act = "close_at_earliest_notice"
        elif reach and poor and end / trig < 1.12:
            act = "verify_landings_before_deciding"
        elif reach and wk <= 3:
            act = "announce_closure_date"
        elif reach:
            act = "reduce_trip_limits"
        elif poor and end / trig >= 0.93:
            act = "verify_landings_before_deciding"
        else:
            act = "keep_open"
        r = end / trig
        ov = 3 if (now >= trig or r >= 1.25) else 2 if r >= 1.0 else 1 if r >= 0.92 else 0
        rel = (f["lag_days"] <= 3) + (f["missing_dealer_pct"] <= 5) + (f["logbook_discrepancy_pct"] <= 3)
        ws = f["weekly_t"]
        rel -= (max(ws) - min(ws)) / max(1, sum(ws) / 3) > 0.3
        return {"inseason_action": act, "overage_risk": ov, "projection_reliability": _clamp(rel),
                "trigger_reached_before_season_end": "yes" if reach else "no",
                "landings_data_adequate": "no" if poor else "yes"}

    def _rules(self, f):
        return [f"{f['fmp']} §6.3: the fishery closes when landings are projected to reach {f['trigger_pct']}% of the "
                f"{f['tac_t']:,.0f} t TAC; a closure takes effect no sooner than {f['notice_h']} h after notice. Trip limits may be "
                "cut to slow the catch when the trigger would otherwise be reached more than three weeks out.",
                "§6.7: projections need dealer reports from at least 90% of active vessels and logbook totals within 5% of "
                "dealer totals; overages are deducted from next year's TAC."]

    def render(self, f, rng, surface):
        w = f["weekly_t"]
        dq = (f"{f['missing_dealer_pct']}% of active vessels' dealer reports missing; logbooks differ from dealer totals by "
              f"{f['logbook_discrepancy_pct']}%")
        if surface == "json":
            return dumps(prune({
                "fishery": {"name": f["fishery"], "gear": f["gear"], "season_ends": f["season_end"].isoformat(),
                            "active_vessels": f["vessels"], "current_trip_limit_t": f["trip_limit"]},
                "landings": {"reported_through": f["report_date"].isoformat(), "reporting_lag_days": f["lag_days"],
                             "cumulative_t": f["landed_t"], "last_three_weeks_t": w},
                "data_quality": dq, "weather": "gale warning for the next 5 days" if f["storm_forecast"] else None,
                "plan": self._rules(f),
            }))
        if surface == "landings_dashboard":
            rows = [["week -2", str(w[0])], ["week -1", str(w[1])], ["last week", str(w[2])]]
            head = kv_lines([("Fishery", f"{f['fishery']} ({f['gear']})"), ("TAC / trigger", f"{f['tac_t']:,.0f} t / {f['trigger_pct']}% "
                             f"({f['fmp']} §6.3, {f['notice_h']} h notice)"), ("Landed", f"{f['landed_t']:,.0f} t through "
                             f"{fmt_date(f['report_date'], rng)} (lag {f['lag_days']} d)"), ("Season ends", fmt_date(f["season_end"], rng)),
                             ("Data quality", dq + " (need >= 90% reporting, <= 5% gap)"),
                             ("Weather", "gale warning, 5 days" if f["storm_forecast"] else None)], rng)
            return "IN-SEASON LANDINGS DASHBOARD\n" + head + "\n\n" + table(["week", "landings (t)"], rows, rng)
        para = [f"{f['fishery']} ({f['gear']}): {f['landed_t']:,.0f} t landed through {fmt_date(f['report_date'], rng)}, "
                f"with reports lagging about {f['lag_days']} days.", f"Weekly landings over the last three weeks: {w[0]}, {w[1]} and {w[2]} t.",
                f"The season ends {fmt_date(f['season_end'], rng)}; {f['vessels']} vessels active.", f"Data: {dq}.",
                ("A gale warning is out for the next five days." if f["storm_forecast"] else "")]
        return (f"Briefing for the regional administrator - {f['manager']}, {f['agency']}\n\n"
                + noisy(join_sentences(para), rng, 0.08) + "\n\n" + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "inseason_action": choice("What in-season action should the administrator take?", {
                "keep_open": "Keep the fishery open on current measures.",
                "reduce_trip_limits": "Cut trip limits to slow the catch.",
                "announce_closure_date": "Announce a closure for the date the trigger is projected to be reached.",
                "close_at_earliest_notice": "Close the fishery as soon as the notice period allows.",
                "verify_landings_before_deciding": "Get the missing or conflicting landings data resolved first.",
            }),
            "overage_risk": score("How likely is the fleet to exceed the quota trigger this season?", [
                "Unlikely.",
                "Possible.",
                "Likely.",
                "Near certain, or already happening.",
            ]),
            "projection_reliability": score("How reliable is the catch projection behind this decision?", [
                "Unreliable.",
                "Shaky.",
                "Reasonable.",
                "Solid.",
            ]),
            "trigger_reached_before_season_end": noul(
                "At the current catch rate, landings will reach the closure trigger before the season ends.",
                yes="Projected landings by the season's end meet or pass the trigger tonnage.",
                no="Projected landings stay below the trigger tonnage."),
            "landings_data_adequate": noul(
                "The landings data meet the plan's standard for an in-season projection."),
        }


# =========================================================================== schema 5: bycatch limits
# (fishery, bycatch species, cap, unit, rate unit, typical rate trigger)
BYCATCH = [("Northern Gulf pollock trawl", "Chinook salmon", (25000, 45000), "fish", "fish per t", (0.03, 0.08)),
           ("Eastern Shelf groundfish trawl", "Pacific halibut", (900, 2200), "t", "kg per t", (8, 20)),
           ("coastal monkfish gillnet", "harbour porpoise", (40, 90), "animals", "per 100 net-days", (0.5, 2.0)),
           ("pelagic swordfish longline", "loggerhead sea turtle", (60, 140), "animals", "per 1,000 hooks", (0.02, 0.08)),
           ("southern shrimp trawl", "juvenile red snapper", (900, 2500), "thousand fish", "fish per tow", (5, 15)),
           ("Western Shelf sablefish longline", "black-footed albatross", (120, 300), "birds", "per 1,000 hooks", (0.03, 0.1))]


class BycatchLimitTrigger(Schema):
    name = "bycatch_limit_trigger"
    description = "An in-season bycatch decision from observer-based estimates against a hotspot trigger and a sector cap, and the monitoring that should follow."
    surfaces = {"json": 0.5, "observer_summary": 0.25, "council_memo": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng, clear=0.25, borderline=0.45, conflicting=0.30)
        fish, sp, (clo, chi), unit, runit, (rlo, rhi) = pick(rng, BYCATCH)
        cap = int(round(rng.uniform(clo, chi), -1 if chi < 1000 else -2))
        pc = pick(rng, [90, 100])
        r_trig = round(rng.uniform(rlo, rhi), 3 if rhi < 1 else 1)
        cov_min = pick(rng, [20, 30])
        closure_days = pick(rng, [7, 14, 21])
        scen = pick_w(rng, {"quiet": 2.4, "advisory": 2, "hotspot": 3.1, "sector": 2, "confirm": 1.3})
        share = {"quiet": rng.uniform(0.2, 0.6), "advisory": rng.uniform(0.76, pc / 100 - 0.03),
                 "hotspot": rng.uniform(0.3, 0.8), "sector": rng.uniform(pc / 100, pc / 100 + 0.15),
                 "confirm": rng.uniform(pc / 100, pc / 100 + 0.1)}[scen]
        rate = r_trig * {"quiet": rng.uniform(0.2, 0.6), "advisory": rng.uniform(0.72, 0.98), "hotspot": rng.uniform(1.05, 2.0),
                         "sector": rng.uniform(0.3, 1.5), "confirm": rng.uniform(0.3, 1.3)}[scen]
        if reg == "borderline":
            if scen in ("hotspot", "advisory"):
                rate = r_trig * near(rng, 1.0, 0.08)
            elif scen in ("sector", "confirm"):
                share = near(rng, pc / 100, 0.03)
        coverage = rng.randint(cov_min, cov_min + 40)
        if scen == "confirm" or (reg == "conflicting" and chance(rng, 0.3)):
            coverage = rng.randint(max(3, cov_min - 15), cov_min - 1)
        ci = round(share * rng.uniform(0.08, 0.35) * (1.6 if coverage < cov_min else 1.0), 3)
        disc = pick_w(rng, {rng.randint(0, 10): 5, rng.randint(26, 60): 2.2})
        hi_vessels = pick_w(rng, {0: 3, 1: 2, rng.randint(2, 5): 2.5})
        em = chance(rng, 0.6)
        return {
            "regime": reg, "scenario": scen, "fishery": fish, "species": sp, "cap": cap, "unit": unit, "rate_unit": runit,
            "sector_pct": pc, "rate_trigger": r_trig, "coverage_min": cov_min, "closure_days": closure_days,
            "estimate": int(round(cap * share)), "ci_half": max(1, int(round(cap * ci))),
            "area_rate": round(rate, 3 if rate < 1 else 1), "area": f"statistical area {rng.randint(510, 650)}",
            "coverage": coverage, "logbook_gap_pct": disc, "high_rate_vessels": hi_vessels, "em_available": em,
            "weeks_left": rng.randint(2, 16), "council": f"{pick(rng, ['North', 'South', 'Western', 'Gulf', 'Atlantic'])} Fishery Management Council",
            "analyst": person(rng), "bmp": f"BMP-{rng.randint(2, 9)}", "week": rng.randint(8, 40),
        }

    def intended(self, f):
        share = f["estimate"] / f["cap"]
        sector = share * 100 >= f["sector_pct"]
        cov_ok = f["coverage"] >= f["coverage_min"]
        hot = f["area_rate"] > f["rate_trigger"]
        if sector:
            meas = "close_sector" if cov_ok else "confirm_estimate_before_closure"
        elif hot:
            meas = "close_hotspot_area"
        elif f["area_rate"] >= 0.7 * f["rate_trigger"] or share >= 0.75:
            meas = "fleet_move_on_advisory"
        else:
            meas = "continue_monitoring"
        if not cov_ok:
            mon = "raise_observer_coverage"
        elif f["logbook_gap_pct"] > 25:
            mon = "audit_logbook_reporting"
        elif f["high_rate_vessels"] >= 2 and f["em_available"]:
            mon = "electronic_monitoring_for_high_rate_vessels"
        else:
            mon = "no_change"
        cp = 0 if share < 0.5 else 1 if share < 0.75 else 2 if share * 100 < f["sector_pct"] else 3
        rel = 3 if f["coverage"] >= f["coverage_min"] + 15 else 2 if cov_ok else 1
        if f["ci_half"] / max(1e-9, f["estimate"]) > 0.3:
            rel -= 1
        if not cov_ok and f["coverage"] < f["coverage_min"] - 10:
            rel = 0
        return {"bycatch_measure": meas, "monitoring_change": mon, "cap_pressure": cp, "estimate_reliability": _clamp(rel),
                "hotspot_trigger_met": "yes" if hot else "no"}

    def _rules(self, f):
        return [f"{f['bmp']} §3: when the 14-day {f['species']} bycatch rate in a statistical area exceeds {f['rate_trigger']} "
                f"{f['rate_unit']}, that area closes for {f['closure_days']} days.",
                f"§4: the sector closes when estimated bycatch reaches {f['sector_pct']}% of the {f['cap']:,} {f['unit']} cap; "
                f"estimates from observer coverage under {f['coverage_min']}% must be confirmed before a sector closure."]

    def render(self, f, rng, surface):
        est = f"{f['estimate']:,} {f['unit']} (90% CI +/- {f['ci_half']:,})"
        n = f["high_rate_vessels"]
        vs = (("no vessels" if n == 0 else "1 vessel" if n == 1 else f"{n} vessels") + " with rates well above the fleet; "
              f"EM {'available to fit' if f['em_available'] else 'not available'}")
        if surface == "json":
            return dumps(prune({
                "fishery": f["fishery"], "species": f["species"], "week_of_season": f["week"], "weeks_left": f["weeks_left"],
                "estimate_to_date": est, "observer_coverage_pct": f["coverage"],
                "hotspot": {"area": f["area"], "rate_last_14_days": f"{f['area_rate']} {f['rate_unit']}"},
                "fleet": {"high_rate_vessels": vs, "logbook_vs_observer_gap_pct": f["logbook_gap_pct"]},
                "rules": self._rules(f),
            }))
        if surface == "observer_summary":
            rows = [[f["area"], f"{f['area_rate']}", f"{f['rate_trigger']}"]]
            head = kv_lines([("Fishery", f"{f['fishery']}, week {f['week']} ({f['weeks_left']} weeks left)"),
                             ("Bycatch estimate", est + f" of {f['cap']:,} cap; sector closes at {f['sector_pct']}%"),
                             ("Observer coverage", f"{f['coverage']}% (confirm estimate if under {f['coverage_min']}%)"),
                             ("Logbook vs observer", f"{f['logbook_gap_pct']}% lower in logbooks"), ("Vessels", vs)], rng)
            return (f"OBSERVER PROGRAM SUMMARY - {f['species']}\n{head}\n\n"
                    + table(["area", f"14-day rate ({f['rate_unit']})", f"area trigger ({f['bmp']} §3)"], rows, rng))
        para = [f"Week {f['week']} of the {f['fishery']}: estimated {f['species']} bycatch is {est}, against a cap of "
                f"{f['cap']:,} {f['unit']}.", f"Observer coverage is {f['coverage']}%.",
                f"In {f['area']} the last 14 days ran at {f['area_rate']} {f['rate_unit']}.",
                f"Logbook-reported bycatch is {f['logbook_gap_pct']}% below observer estimates.", vs[0].upper() + vs[1:] + ".",
                f"{f['weeks_left']} weeks remain in the season."]
        return (f"Memo to the {f['council']} - {f['analyst']}\n\n" + noisy(join_sentences(para), rng, 0.08) + "\n\n"
                + bullets(self._rules(f), rng))

    def questions(self):
        return {
            "bycatch_measure": choice("What bycatch measure should the agency take now?", {
                "continue_monitoring": "Continue monitoring; no measure is warranted.",
                "fleet_move_on_advisory": "Issue a fleet advisory asking vessels to move away from high-bycatch grounds.",
                "close_hotspot_area": "Close the hotspot statistical area for the set period.",
                "close_sector": "Close the sector for the rest of the season.",
                "confirm_estimate_before_closure": "Confirm the estimate before acting, as the rules require for low coverage.",
            }),
            "monitoring_change": choice("What change to monitoring should follow?", {
                "no_change": "No change to monitoring.",
                "raise_observer_coverage": "Raise observer coverage.",
                "electronic_monitoring_for_high_rate_vessels": "Require electronic monitoring on the high-rate vessels.",
                "audit_logbook_reporting": "Audit the fleet's logbook reporting of bycatch.",
            }),
            "cap_pressure": score("How close is the fishery to its bycatch cap?", [
                "Far: under half the cap.",
                "Approaching: half to three quarters.",
                "Close: past three quarters but below the closure point.",
                "At or past the closure point.",
            ]),
            "estimate_reliability": score("How reliable is the bycatch estimate?", [
                "Unreliable: coverage far below the minimum.",
                "Weak: coverage below the minimum, or a wide interval.",
                "Adequate.",
                "Strong: coverage well above the minimum and a tight interval.",
            ]),
            "hotspot_trigger_met": noul(
                "The bycatch rate in the statistical area meets the area-closure trigger.",
                yes="The area's 14-day rate exceeds the trigger rate.",
                no="The area's 14-day rate is at or below the trigger rate."),
        }


WORKFLOW = Workflow(
    name="forestry_and_fisheries_management",
    title="Forestry and fisheries management",
    description="Forest-district and fishery-agency decisions on harvest plans, fire-season operations, salvage logging, in-season fishery closures and bycatch limits, judged against written (fictional) codes and management plans.",
    domain="natural resource management (forestry, fisheries)",
    schemas=[TimberHarvestPlanReview(), FireSeasonOperations(), SalvageAfterDisturbance(), InseasonFisheryClosure(),
             BycatchLimitTrigger()],
    notes="Regulatory and management decisions on field and catch data; species, forest types, seasons and gear are kept "
          "consistent. No enforcement actions against individuals.",
)
