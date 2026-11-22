"""Grid outage triage: restoration priority, re-energising, crew dispatch and switching on a (fictional) distribution grid.

Control-room and dispatch decisions at invented utilities: where an outage event sits in the restoration queue,
whether a locked-out feeder may be test-closed under the written operating procedure, which crew a single-premises
trouble call needs, whether a switching order is safe to release, and how to support a critical facility running
on backup power. Substations, feeders, device numbers, procedures and limits are invented, shaped like real
outage-management (OMS) and SCADA artefacts; no real utility, procedure text or person is used. Each schema draws
a `regime` first so most cases sit near a limit, carry signals that disagree (a caller's account against the
meter pings), or miss a key fact.

Every schema also has `intended(f)`: the answer a careful reviewer would most likely give from what the state
shows. It is never rendered and never used as a label; it exists so that the answer spread of every question can
be audited before labelling (no question may collapse onto one option).
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
    show,
    table,
)

# --------------------------------------------------------------------------- shared pools
SUB_A = ["Alder", "Birch", "Cedar", "Dogwood", "Elm", "Fir", "Garnet", "Heron", "Iris", "Jasper", "Kestrel", "Larch",
         "Maple", "Nettle", "Osprey", "Pine", "Quail", "Raven", "Sumac", "Tansy", "Umber", "Vetch", "Willow", "Yew",
         "Basin", "Coulee", "Delta", "Ferry", "Granger", "Harlow", "Lomax", "Prescott", "Tolland", "Wexford"]
SUB_B = ["Creek", "Flats", "Hill", "Junction", "Ridge", "Hollow", "Road", "Bend", "Park", "Mill", "Crossing", "Gap"]
STREETS = ["Birch Rd", "Old Mill Rd", "Larkspur Ln", "County Rd 14", "Quarry St", "Hawthorn Ave", "Station St",
           "Pickett Ln", "Ridgeview Dr", "Sawmill Rd", "Orchard Way", "Canal St", "Fenwick Ave", "Juniper Ct",
           "Route 9 frontage road", "Kessler Rd", "Mulberry St", "Tanner Ln", "Wicklow Dr", "Granary Rd",
           "Foxglove Ct", "Cutler Ave", "Spring Valley Rd", "Ashby Ln", "Weir St", "Holloway Rd", "Pasture Ln",
           "Brook Hollow Dr", "Depot St", "Lantern Way", "Copper Hill Rd", "Ellery St", "Thistle Ln", "Marsh Rd"]
UTIL_SUFFIX = ["Power & Light", "Electric Cooperative", "Public Power District", "Municipal Light & Power",
               "Rural Electric Association", "Electric Utility District"]
PRIMARY_KV = [4.16, 12.47, 12.47, 13.2, 13.8, 24.9, 34.5]
EXTREME = ("extreme_heat", "extreme_cold")


def _utility(rng):
    return f"{pick(rng, SUB_A)} {pick(rng, UTIL_SUFFIX)}"


def _substation(rng):
    a, b = pick(rng, SUB_A), pick(rng, SUB_B)
    return f"{a} {b}", (a[:2] + b[0]).upper()


def _feeder(rng, code):
    return f"{code}-{rng.randint(1, 16):02d}"


def _address(rng):
    return f"{rng.randint(100, 9899)} {pick(rng, STREETS)}"


def _temp_for(rng, weather):
    return {"extreme_heat": rng.randint(99, 113), "extreme_cold": rng.randint(-18, 9),
            "storm_active": rng.randint(41, 78), "post_storm": rng.randint(38, 80),
            "clear": rng.randint(45, 84), "high_wind": rng.randint(35, 72),
            "thunderstorm": rng.randint(62, 88), "clear_calm": rng.randint(40, 86),
            "rain": rng.randint(38, 66), "ice_snow": rng.randint(18, 31)}.get(weather, rng.randint(40, 80))


# =========================================================================== schema 1
FACILITY_TYPES = {
    # type: (name templates, life-safety?)
    "hospital": (["{a} Regional Medical Center", "{a} Community Hospital", "{a} Valley Hospital"], True),
    "nursing_home": (["{a} Care Center", "{a} Senior Living", "{a} Rehabilitation and Nursing"], True),
    "dialysis_clinic": (["{a} Dialysis Center", "{a} Kidney Care Clinic"], True),
    "dispatch_911": (["{a} County 911 Center", "{a} Emergency Communications Center"], True),
    "water_pump_station": (["{a} Booster Pump Station", "{a} Well Field Pumps"], True),
    "wastewater_lift_station": (["Lift Station {n} ({a} Rd)", "{a} Sewage Lift Station"], True),
    "cell_site": (["wireless site {code}", "cell site {code}"], False),
    "cooling_warming_center": (["{a} Community Center (designated shelter)", "{a} Middle School shelter"], False),
    "grocery_distribution": (["{a} Foods distribution center", "{a} cold-storage warehouse"], False),
}
DEVICE_LABEL = {"feeder_breaker": "feeder breaker", "line_recloser": "line recloser", "lateral_fuse": "lateral fuse",
                "transformer": "distribution transformer"}
CAUSES = {"tree_contact": "tree/limb on line", "vehicle_pole": "vehicle hit pole", "equipment_failure": "equipment failure",
          "animal": "animal contact", "lightning": "lightning", "unknown": "unknown - not patrolled"}
REPAIR_H = {"tree_contact": (2.0, 3.5), "vehicle_pole": (5.0, 9.0), "equipment_failure": (2.5, 5.0),
            "animal": (1.0, 1.8), "lightning": (1.5, 4.0), "unknown": (2.0, 5.0)}
WIRE_LOC = {"roadway": "across the roadway", "school_frontage": "along the sidewalk in front of the elementary school",
            "residential_yard": "in a front yard, near the driveway", "rear_easement": "in the rear-lot easement behind the houses",
            "open_field": "in an open pasture well back from the road", "parking_lot": "across a store parking lot"}
PUBLIC_LOCS = ("roadway", "school_frontage", "residential_yard", "parking_lot")


class OutageEventPriority(Schema):
    name = "outage_event_priority"
    description = "An outage event in the outage-management system during a busy period: where it goes in the restoration queue."
    surfaces = {"json": 0.5, "oms_screen": 0.25, "dispatch_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        device = pick_w(rng, {"feeder_breaker": 1, "line_recloser": 2, "lateral_fuse": 4, "transformer": 3})
        customers = {"feeder_breaker": rng.randint(900, 3800), "line_recloser": rng.randint(180, 1400),
                     "lateral_fuse": rng.randint(12, 260), "transformer": rng.randint(1, 14)}[device]
        weather = pick_w(rng, {"storm_active": 3, "post_storm": 3, "clear": 2, "extreme_heat": 2, "extreme_cold": 2})
        cause = pick_w(rng, {"tree_contact": 4, "vehicle_pole": 1, "equipment_failure": 2, "animal": 1,
                             "lightning": 1, "unknown": 3})
        n_fac = 0 if device == "transformer" and chance(rng, 0.6) else pick_w(rng, {0: 5, 1: 4, 2: 1})
        facilities = []
        for _ in range(n_fac):
            ftype = pick(rng, list(FACILITY_TYPES))
            backup = pick_w(rng, {"generator_running": 4, "no_backup": 2, "unknown": 2, "generator_failed": 1})
            facilities.append({"type": ftype, "backup": backup,
                               "name": pick(rng, FACILITY_TYPES[ftype][0]).format(
                                   a=pick(rng, SUB_A), n=rng.randint(2, 31), code=f"{pick(rng, SUB_A)[:3].upper()}{rng.randint(10, 99)}")})
        med = pick(rng, [0, 0, 0, 0, 1, 1, 2, 3, 5]) if customers > 8 else pick(rng, [0, 0, 0, 1])
        wire = pick_w(rng, {None: 5, "customer_call": 3, "police_fire_on_scene": 1, "crew_confirmed": 1})
        loc = pick_w(rng, {"roadway": 2, "school_frontage": 1, "residential_yard": 2, "rear_easement": 3, "open_field": 2,
                            "parking_lot": 1}) if wire else None
        wire_state = pick_w(rng, {"unknown": 4, "arcing_reported": 2, "upstream_device_open": 2}) if wire else None
        entrapment = cause == "vehicle_pole" and chance(rng, 0.25)
        crews_free = pick_w(rng, {0: 5, 1: 3, 2: 1})
        nearby = maybe(rng, {"distance_mi": round(rng.uniform(0.3, 6.5), 1),
                             "job": pick(rng, ["fuse replacement", "tree off secondary", "pole transformer change",
                                               "service drop reattach", "recloser reset after patrol"]),
                             "remaining_min": pick(rng, [15, 30, 45, 60, 90, 150])}, 0.25)
        if reg == "clear":
            if chance(rng, 0.5):    # obvious top of the queue
                wire, loc, wire_state = pick(rng, ["crew_confirmed", "police_fire_on_scene"]), pick(rng, ["roadway", "school_frontage"]), "arcing_reported"
                weather = pick(rng, ["storm_active", "extreme_heat", "extreme_cold"])
            else:                   # obvious ordinary job
                device, customers, facilities, med, wire, loc, wire_state = "transformer", rng.randint(2, 9), [], 0, None, None, None
                weather, entrapment = pick(rng, ["clear", "post_storm"]), False
        elif reg == "conflicting":
            # a big, noisy event with nothing dangerous in it, against a small one with a hazard or a fragile customer
            if chance(rng, 0.5):
                device, customers, wire, loc, wire_state = "line_recloser", rng.randint(700, 1400), None, None, None
                facilities = [f for f in facilities if f["backup"] == "generator_running"]
            else:
                device, customers = pick(rng, ["lateral_fuse", "transformer"]), rng.randint(3, 40)
                if chance(rng, 0.5):
                    wire, loc = "customer_call", pick(rng, ["residential_yard", "rear_easement", "roadway"])
                    wire_state = pick(rng, ["unknown", "upstream_device_open"])
                else:
                    med = pick(rng, [2, 3, 4])
                    weather = pick(rng, list(EXTREME))
        out_min = rng.randint(25, 560)
        lo, hi = REPAIR_H[cause]
        repair_h = round(rng.uniform(lo, hi), 1)
        earliest_free_h = 0.0 if crews_free else round(rng.uniform(0.5, 6.0), 1)
        travel_h = round(rng.uniform(0.3, 1.3), 1)
        expected_h = earliest_free_h + travel_h + repair_h
        etr_kind = pick_w(rng, {"crew_estimate": 2, "storm_default": 3, "system_predicted": 3})
        etr_h = round(max(0.5, expected_h + rng.uniform(-3.5, 3.0)), 1)
        return {"regime": reg, "utility": _utility(rng), "event_id": ident(rng, "OE", 7, "-"),
                "substation": _substation(rng), "feeder_no": rng.randint(1, 16), "device": device,
                "device_id": {"feeder_breaker": "CB-", "line_recloser": "R-", "lateral_fuse": "F-", "transformer": "T-"}[device]
                + str(rng.randint(1000, 98999)), "customers_out": customers, "facilities": facilities,
                "medical_registry": med, "weather": weather, "temp_f": _temp_for(rng, weather), "cause": cause,
                "wire_down": wire, "wire_location": loc, "wire_state": wire_state, "entrapment": entrapment,
                "street": pick(rng, STREETS), "out_min": out_min, "crews_free": crews_free,
                "earliest_free_h": earliest_free_h, "travel_h": travel_h, "repair_h": repair_h,
                "nearby_crew": nearby, "open_events": rng.randint(4, 72), "etr_kind": etr_kind, "etr_h": etr_h,
                "now": rand_ts(rng), "dispatcher": person(rng)}

    # ---------------------------------------------------------------- intended answers (audit only)
    def _hazard(self, f):
        if f["entrapment"]:
            return 3
        if not f["wire_down"]:
            return 1 if f["cause"] == "vehicle_pole" else 0
        public = f["wire_location"] in PUBLIC_LOCS
        lvl = 1
        if public:
            lvl = 2 if f["wire_down"] == "customer_call" else 3
        elif f["wire_down"] != "customer_call":
            lvl = 2
        if f["wire_state"] == "arcing_reported" and public:
            lvl = 3
        return lvl

    def _critical(self, f):
        extreme = f["weather"] in EXTREME
        lvl = 0
        if f["medical_registry"] >= 1:
            lvl = 2 if extreme else 1
        if f["medical_registry"] >= 3:
            lvl = 3 if extreme else 2
        if extreme and f["customers_out"] >= 300:
            lvl = max(lvl, 1)          # many homes without heat or cooling
        for fac in f["facilities"]:
            life = FACILITY_TYPES[fac["type"]][1] or (fac["type"] == "cooling_warming_center" and extreme)
            if fac["backup"] == "generator_running":
                lvl = max(lvl, 1)
            elif fac["backup"] == "unknown":
                lvl = max(lvl, 2)
            else:
                lvl = max(lvl, 3 if life else 2)
        return lvl

    def intended(self, f):
        hz, cr = self._hazard(f), self._critical(f)
        if hz == 3 or cr == 3:
            q = "assign_next_free_crew" if f["crews_free"] else "divert_crew_from_lower_job"
        elif hz == 2 or cr == 2 or f["customers_out"] >= 1500:
            q = "assign_next_free_crew"
        elif f["nearby_crew"] and f["nearby_crew"]["distance_mi"] <= 4 and f["device"] != "feeder_breaker":
            q = "add_to_nearby_crew_route"
        else:
            q = "hold_in_normal_queue"
        expected = f["earliest_free_h"] + f["travel_h"] + f["repair_h"]
        big = f["device"] in ("feeder_breaker", "line_recloser")
        if f["cause"] == "tree_contact" and (big or f["wire_down"]):
            crew = "line_crew_with_tree_crew"
        elif f["cause"] == "vehicle_pole" or f["wire_down"] in ("crew_confirmed", "police_fire_on_scene") or (
                f["device"] == "transformer" and f["cause"] == "equipment_failure"):
            crew = "line_crew"
        elif big and f["cause"] in ("unknown", "lightning", "equipment_failure"):
            crew = "damage_assessor_first"
        else:
            crew = "troubleman_first"
        return {"restoration_queue_call": q, "first_crew_type": crew, "public_safety_exposure": hz,
                "critical_load_weight": cr,
                "published_etr_unrealistic": "yes" if expected > f["etr_h"] + 0.75 else "no"}

    # ---------------------------------------------------------------- rendering
    def _wire_text(self, f, rng):
        if not f["wire_down"]:
            return None
        who = {"customer_call": pick(rng, ["Caller", "Customer on the phone", "A resident"]),
               "police_fire_on_scene": pick(rng, ["Police on scene", "Fire engine 12 on scene", "Sheriff's deputy"]),
               "crew_confirmed": pick(rng, ["Troubleman", "Line crew passing by", "Damage assessor"])}[f["wire_down"]]
        verb = {"customer_call": "says there is a line down", "police_fire_on_scene": "reports a conductor down",
                "crew_confirmed": "confirms a primary conductor down"}[f["wire_down"]]
        state = {"unknown": "", "arcing_reported": pick(rng, [", sparking and smoking", ", arcing on the ground", ", popping and flashing"]),
                 "upstream_device_open": pick(rng, [", fuse upstream looks open", "; upstream device shows open in OMS"])}[f["wire_state"]]
        return f"{who} {verb} {WIRE_LOC[f['wire_location']]} on {f['street']}{state}."

    def _notes(self, f, rng):
        s = []
        w = self._wire_text(f, rng)
        if w:
            s.append(w)
        if f["entrapment"]:
            s.append("Driver still in the vehicle, fire asking for the line to be confirmed dead before they extricate.")
        if f["cause"] == "unknown" and not w:
            s.append(pick(rng, ["No patrol yet, cause unknown.", "Nobody has looked at it yet.", "Cause TBD, no eyes on it."]))
        if f["cause"] == "tree_contact" and chance(rng, 0.5):
            s.append(pick(rng, ["Caller heard a crack and saw a big limb on the wires.", "Tree crew may be needed, large limb reported."]))
        for fac in f["facilities"]:
            b = {"generator_running": "on their generator", "no_backup": "has no backup generation",
                 "unknown": "has not answered on backup status", "generator_failed": "says their generator failed to start"}[fac["backup"]]
            s.append(f"{fac['name']} ({fac['type'].replace('_', ' ')}) {b}.")
        if f["medical_registry"]:
            s.append(f"{f['medical_registry']} medical-equipment registry customer(s) on this device.")
        if f["regime"] == "conflicting" and chance(rng, 0.5):
            s.append(pick(rng, ["Caller insists the whole street is out.", "Councilmember's office called asking for an update.",
                                "Several callers angry about the time out.", "Social media posts about this one are getting attention."]))
        return noisy(join_sentences(s), rng, 0.15) if s else None

    def render(self, f, rng, surface):
        sub, code = f["substation"]
        feeder = f"{code}-{f['feeder_no']:02d}"
        now = f["now"]
        etr = now + timedelta(hours=f["etr_h"])
        out_since = now - timedelta(minutes=f["out_min"])
        notes = self._notes(f, rng)
        crew = {"free_now": f["crews_free"],
                "earliest_crew_free_hours": None if f["crews_free"] else f["earliest_free_h"],
                "nearest_working_crew": (None if not f["nearby_crew"] else
                                         {"distance_miles": f["nearby_crew"]["distance_mi"], "job": f["nearby_crew"]["job"],
                                          "est_minutes_to_finish": f["nearby_crew"]["remaining_min"]}),
                "typical_drive_hours": f["travel_h"]}
        if surface == "json":
            return dumps(prune({
                "event": {"id": f["event_id"], "utility": f["utility"], "substation": sub, "feeder": feeder,
                          "predicted_device": {"type": f["device"], "id": f["device_id"]},
                          "customers_out": f["customers_out"], "out_since": fmt_ts(out_since, style="iso"),
                          "cause_code": CAUSES[f["cause"]],
                          "published_etr": {"time": fmt_ts(etr, style="iso"), "source": f["etr_kind"]},
                          "cause_model_repair_hours": f["repair_h"],
                          "street": f["street"]},
                "critical_customers": ({"facilities": [{"name": x["name"], "type": x["type"], "backup": x["backup"]}
                                                       for x in f["facilities"]] or None,
                                        "medical_registry_count": f["medical_registry"] or None}),
                "conditions": {"weather": f["weather"], "temperature_f": f["temp_f"],
                               "open_events_in_district": f["open_events"], "as_of": fmt_ts(now, style="iso")},
                "crews": crew,
                "call_notes": notes,
            }))
        if surface == "oms_screen":
            rows = [("Event", f"{f['event_id']}   ({f['utility']})"), ("Status", "UNASSIGNED" if f["crews_free"] == 0 else "UNASSIGNED - crew(s) free"),
                    ("Device", f"{DEVICE_LABEL[f['device']]} {f['device_id']}  {feeder} / {sub}"),
                    ("Customers out", f["customers_out"]), ("Out since", fmt_ts(out_since, rng)),
                    ("Cause", CAUSES[f["cause"]]), ("ETR", f"{fmt_ts(etr, rng)} ({f['etr_kind'].replace('_', ' ')})"),
                    ("Est. repair (cause model)", f"{f['repair_h']} h"),
                    ("Weather", f"{f['weather'].replace('_', ' ')}, {f['temp_f']} F"),
                    ("Open events (district)", f["open_events"]),
                    ("Crews free", f["crews_free"] if f["crews_free"] else f"0 (earliest ~{f['earliest_free_h']} h)"),
                    ("Nearest crew", None if not f["nearby_crew"] else
                     f"{f['nearby_crew']['distance_mi']} mi away on {f['nearby_crew']['job']}, ~{f['nearby_crew']['remaining_min']} min left"),
                    ("Drive time", f"~{f['travel_h']} h"),
                    ("Medical registry", f["medical_registry"] or None)]
            crit = [f"{x['name']} - {x['type'].replace('_', ' ')} - backup: {x['backup'].replace('_', ' ')}" for x in f["facilities"]]
            out = [f"OMS EVENT DETAIL  [as of {fmt_ts(now, style='log')}]", kv_lines(rows, rng)]
            if crit:
                out += ["", "Critical accounts:", bullets(crit, rng)]
            if notes:
                out += ["", "Trouble-call remarks:", notes]
            return "\n".join(out)
        # dispatch_log
        t = out_since
        lines = [f"# {f['utility']} distribution dispatch - {sub} district - {fmt_date(now.date(), rng)}"]
        lines.append(f"{t.strftime('%H:%M')}  OMS  {f['event_id']} predicted at {DEVICE_LABEL[f['device']]} {f['device_id']} ({feeder}), "
                     f"{f['customers_out']} cust")
        t2 = t + timedelta(minutes=rng.randint(3, 25))
        w = self._wire_text(f, rng)
        if w:
            lines.append(f"{t2.strftime('%H:%M')}  CALL {noisy(w, rng, 0.1)}")
        for fac in f["facilities"]:
            t2 += timedelta(minutes=rng.randint(2, 30))
            lines.append(f"{t2.strftime('%H:%M')}  KEY ACCT  {fac['name']} ({fac['type'].replace('_', ' ')}): "
                         + {"generator_running": "on generator", "no_backup": "no backup power", "unknown": "backup status unknown",
                            "generator_failed": "generator failed"}[fac["backup"]])
        if f["entrapment"]:
            lines.append(f"{t2.strftime('%H:%M')}  FIRE  occupant still in vehicle at pole, need line confirmed dead")
        if f["medical_registry"]:
            lines.append(f"{t2.strftime('%H:%M')}  OMS  {f['medical_registry']} medical registry flag(s) on event")
        lines.append(f"{now.strftime('%H:%M')}  WX   {f['weather'].replace('_', ' ')}, {f['temp_f']}F; {f['open_events']} open events in district")
        crew_line = (f"{f['crews_free']} crew(s) free now" if f["crews_free"] else f"no free crews, earliest ~{f['earliest_free_h']}h")
        if f["nearby_crew"]:
            crew_line += (f"; crew on {f['nearby_crew']['job']} {f['nearby_crew']['distance_mi']} mi away, "
                          f"~{f['nearby_crew']['remaining_min']} min to finish")
        lines.append(f"{now.strftime('%H:%M')}  DISP {crew_line}; drive ~{f['travel_h']}h")
        lines.append(f"{now.strftime('%H:%M')}  OMS  ETR {etr.strftime('%H:%M')} ({f['etr_kind'].replace('_', ' ')}), cause: {CAUSES[f['cause']]}, "
                     f"repair est {f['repair_h']} h")
        return "\n".join(lines)

    def questions(self):
        return {
            "restoration_queue_call": choice("Where should this outage event go in the restoration queue?", {
                "assign_next_free_crew": "Give it the next crew that comes free, ahead of the ordinary queue order.",
                "divert_crew_from_lower_job": "Pull a crew off a lower-priority job now and send it here.",
                "add_to_nearby_crew_route": "Hand it to the crew already working close by, to take on after their current job.",
                "hold_in_normal_queue": "Leave it in the queue in the usual order of customer count and time out.",
            }),
            "first_crew_type": choice("What kind of resource should be sent to this event first?", {
                "troubleman_first": "A one-person troubleman to refuse a fuse, reset a device or make a small repair.",
                "damage_assessor_first": "A patroller or damage assessor to find and scope the damage before a crew is committed.",
                "line_crew": "A full line crew equipped for a conductor, pole or transformer repair.",
                "line_crew_with_tree_crew": "A line crew together with a tree crew to clear vegetation before the repair.",
            }),
            "public_safety_exposure": score("How much danger does this event pose to the public until a crew gets there?", [
                "Minimal: nothing reported down and nothing within reach of the public.",
                "Limited: a possible hazard, or one in a place few people go.",
                "Serious: a reported downed or low conductor near a road, path or homes.",
                "Acute: a confirmed hazard where people are present, or someone trapped near the equipment.",
            ]),
            "critical_load_weight": score("How much do the customers on this event push it up the restoration order?", [
                "Little: ordinary homes and businesses only.",
                "Some: a few medical-registry customers or a critical site that is running on working backup.",
                "High: a critical site with uncertain backup, or several medical-registry customers.",
                "Overriding: a life-safety site with no working backup, or medical-registry customers in extreme temperatures.",
            ]),
            "published_etr_unrealistic": noul(
                "The restoration time currently published for this event can no longer be met.",
                yes="The wait for a crew, the drive and the expected repair time together run past the published time.",
                no="A crew could plausibly arrive and finish the repair by the published time."),
        }


# =========================================================================== schema 2
WEATHER2 = {"thunderstorm": "thunderstorms, lightning in the area", "high_wind": "high wind",
            "clear_calm": "clear and calm", "rain": "steady rain", "ice_snow": "wet snow / icing"}
FAULT_TYPES = ["A-phase to ground", "B-phase to ground", "C-phase to ground", "A-B phase to phase",
               "B-C phase to phase", "three-phase"]


class FeederLockoutReclose(Schema):
    name = "feeder_lockout_reclose"
    description = "A feeder breaker or line recloser has locked out after its reclose attempts; the operator decides how to re-energise."
    surfaces = {"json": 0.5, "scada_log": 0.25, "operator_handover": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        device = pick_w(rng, {"feeder_breaker": 3, "midline_recloser": 2})
        ug_limit = pick(rng, [30, 40, 50])
        weather = pick_w(rng, {"thunderstorm": 3, "high_wind": 3, "clear_calm": 3, "rain": 2, "ice_snow": 1})
        strikes = rng.randint(4, 140) if weather == "thunderstorm" else pick(rng, [0, 0, 0, 1, 3])
        ftype = pick(rng, FAULT_TYPES)
        fault_ka = round(rng.uniform(0.6, 4.5) if device == "midline_recloser" else rng.uniform(1.2, 9.5), 2)
        trend = pick_w(rng, {"steady": 4, "falling": 3, "erratic": 2})
        ug_pct = pick_w(rng, {rng.randint(0, 15): 4, rng.randint(16, 39): 3, rng.randint(40, 85): 3})
        hftd = chance(rng, 0.35)
        fire = pick_w(rng, {"none": 6, "fire_weather_watch": 2, "red_flag_warning": 2}) if hftd else "none"
        wire_calls = pick_w(rng, {0: 6, 1: 2, 2: 1})
        bang = chance(rng, 0.25)
        clearance = pick_w(rng, {None: 8, "active": 1, "released_by_radio_only": 1})
        fci = pick_w(rng, {"localised": 5, "none_tripped": 1, "no_data": 3})
        customers = rng.randint(400, 3600) if device == "feeder_breaker" else rng.randint(150, 1300)
        transfer_a = rng.randint(70, 320)
        adj_rating = pick(rng, [480, 540, 600, 720])
        adj_load = int(near(rng, adj_rating - transfer_a, 90, lo=80, hi=adj_rating - 10, integer=True))
        if reg == "clear":
            if chance(rng, 0.5):   # textbook temporary fault, procedure satisfied
                weather, strikes, trend, ug_pct, fire, wire_calls, bang, clearance = "thunderstorm", rng.randint(30, 140), "falling", rng.randint(0, 12), "none", 0, False, None
            else:                  # textbook keep-it-dead
                fire, hftd, wire_calls, trend, weather, clearance = "red_flag_warning", True, pick(rng, [1, 2]), "steady", "high_wind", None
        elif reg == "conflicting":
            if chance(rng, 0.5):   # lightning everywhere, but the current says solid fault and someone heard a bang
                weather, strikes, trend, bang = "thunderstorm", rng.randint(20, 120), "steady", True
                ftype = "three-phase"
                fault_ka = round(rng.uniform(5.0, 9.5) if device == "feeder_breaker" else rng.uniform(2.6, 4.6), 2)
            else:                  # procedure technically satisfied but the circuit is mostly cable
                ug_pct, wire_calls, fire = near(rng, ug_limit, 6, lo=5, hi=90, integer=True), 0, "none"
                clearance = pick(rng, [None, "released_by_radio_only"])
        wire_in_fci = wire_calls > 0 and fci == "localised" and chance(rng, 0.6)
        gust = rng.randint(38, 66) if weather == "high_wind" else rng.randint(5, 24)
        if weather != "thunderstorm":
            strikes = min(strikes, 3)
        feeder_no = rng.randint(1, 16)
        adj_no = pick(rng, [n for n in range(1, 17) if n != feeder_no])
        return {"regime": reg, "utility": _utility(rng), "substation": _substation(rng), "feeder_no": feeder_no,
                "device": device, "procedure_style": rng.randrange(3), "device_id": ("CB 52-" if device == "feeder_breaker" else "R-") + str(rng.randint(10, 9899)),
                "shots": pick(rng, [2, 3, 3]), "fault_type": ftype, "fault_ka": fault_ka, "trend": trend,
                "weather": weather, "strikes_30min": strikes, "gust_mph": gust, "temp_f": _temp_for(rng, weather),
                "ug_pct": ug_pct, "ug_limit_pct": ug_limit, "hftd": hftd, "fire_weather": fire,
                "wire_calls": wire_calls, "bang_reported": bang, "wire_in_fci_section": wire_in_fci,
                "clearance": clearance, "clearance_crew": person(rng), "fci": fci,
                "fci_pair": (f"SW-{rng.randint(1000, 4999)}", f"SW-{rng.randint(5000, 9999)}"),
                "customers_out": customers, "faulted_section_customers": int(customers * rng.uniform(0.15, 0.55)),
                "tie_switch": f"N.O. {rng.randint(100, 999)}", "adj_feeder_no": adj_no,
                "adj_load_a": adj_load, "adj_emerg_a": adj_rating, "transfer_a": transfer_a,
                "transfer_source": pick(rng, ["pre-fault SCADA", "pre-fault SCADA", "last month's peak (no SCADA on section)"]),
                "tripped_at": rand_ts(rng), "operator": person(rng)}

    def _allowed(self, f):
        return (f["wire_calls"] == 0 and f["clearance"] is None and not (f["hftd"] and f["fire_weather"] == "red_flag_warning")
                and f["ug_pct"] < f["ug_limit_pct"])

    def _permanent(self, f):
        p = 0.6
        p += {"thunderstorm": -1, "high_wind": -0.5, "clear_calm": 1, "rain": 0, "ice_snow": 0.5}[f["weather"]]
        p += {"steady": 1, "falling": -1, "erratic": 0}[f["trend"]]
        if f["fault_type"] == "three-phase" and f["fault_ka"] > 5:
            p += 1
        if f["bang_reported"]:
            p += 1
        if f["wire_calls"]:
            p += 1.5
        if f["ug_pct"] >= 50:
            p += 0.5
        return max(0, min(3, int(round(p - 0.25))))

    def intended(self, f):
        allowed, perm = self._allowed(f), self._permanent(f)
        tie_ok = f["adj_load_a"] + f["transfer_a"] <= f["adj_emerg_a"]
        fire_block = f["hftd"] and f["fire_weather"] == "red_flag_warning"
        if f["clearance"]:
            plan = "hold_for_clearance_release"
        elif allowed:
            plan = ("isolate_section_then_test" if ((perm >= 2 or f["bang_reported"]) and f["fci"] == "localised")
                    else "test_close_once")
        elif f["fci"] == "localised" and not fire_block and f["wire_in_fci_section"] and f["ug_pct"] < f["ug_limit_pct"]:
            plan = "isolate_section_then_test"
        elif f["fci"] == "localised" and tie_ok:
            plan = "transfer_via_tie"
        else:
            plan = "patrol_before_any_close"
        if fire_block and (perm >= 1 or f["fire_weather"] == "red_flag_warning"):
            harm = 3
        elif f["wire_calls"] and f["bang_reported"]:
            harm = 3
        elif f["wire_calls"] or f["ug_pct"] >= f["ug_limit_pct"]:
            harm = 2
        elif perm >= 2 or f["hftd"]:
            harm = 1
        else:
            harm = 0
        return {"reenergise_plan": plan, "permanent_fault_likelihood": perm, "reclose_harm_potential": harm,
                "test_close_allowed_by_procedure": "yes" if allowed else "no",
                "tie_can_carry_transfer": "yes" if tie_ok else "no"}

    def _procedure(self, f):
        u = f["ug_limit_pct"]
        return [
            (f"DOP-21 single test-close: permitted once from the control centre after lockout only if (1) no wire-down, "
             f"contact or pole-fire report is open on the section to be energised, (2) no crew holds a clearance or hold tag "
             f"on the circuit (a clearance counts as held until it is released in the switching log), (3) the circuit is not "
             f"under a red-flag reclose block in a high fire-threat district, and (4) less than {u}% of the circuit length is "
             f"underground cable. Otherwise patrol first."),
            (f"Operating Bulletin 14 (manual test after lockout): one remote test is allowed when the section to be energised "
             f"has no open wire-down or contact report, the circuit is under {u}% cable by length, no clearance is held on it "
             f"(radio releases do not count until logged), and no red-flag block applies (HFTD circuits). Any other case: patrol."),
            (f"SOP E-07 s.3: Do not test-close after lockout if any of the following hold: wire-down/contact/pole-fire report "
             f"open on the section to be energised; clearance or hold tag outstanding in the switching log; HFTD circuit under "
             f"red flag warning; underground cable {u}% or more of circuit length. If none hold, one test-close is permitted."),
        ][f["procedure_style"]]

    def _calls(self, f, rng):
        s = []
        if f["wire_calls"]:
            where = "between the indicators that tripped" if f["wire_in_fci_section"] else pick(rng, ["near the end of the main line", "on a tap off the main line", "location not clear from the call"])
            s.append(f"{f['wire_calls']} wire-down call(s) on this circuit ({where}).")
        if f["bang_reported"]:
            s.append(pick(rng, ["Caller heard a loud bang and saw a flash.", "Resident reports a bang then lights out.",
                                "Report of a transformer 'exploding' near the creek."]))
        if f["clearance"] == "active":
            s.append(f"{f['clearance_crew']}'s crew holds a hold tag on the circuit for tree work.")
        elif f["clearance"] == "released_by_radio_only":
            s.append(f"{f['clearance_crew']} radioed that his crew is clear, but the clearance is not yet released in the switching log.")
        return s

    def render(self, f, rng, surface):
        sub, code = f["substation"]
        feeder = f"{code}-{f['feeder_no']:02d}"
        adj = f"{code}-{f['adj_feeder_no']:02d}"
        t0 = f["tripped_at"]
        fci_text = {"localised": f"FCIs show fault current passed {f['fci_pair'][0]} but not {f['fci_pair'][1]}",
                    "none_tripped": "no faulted-circuit indicator tripped downstream of the device",
                    "no_data": "no remote FCI data on this circuit"}[f["fci"]]
        calls = self._calls(f, rng)
        if surface == "json":
            return dumps(prune({
                "lockout": {"utility": f["utility"], "substation": sub, "circuit": feeder,
                            "device": f["device"].replace("_", " "), "device_id": f["device_id"],
                            "locked_out_at": fmt_ts(t0, style="iso"), "reclose_attempts": f["shots"],
                            "fault_type": f["fault_type"], "fault_current_ka": f["fault_ka"],
                            "fault_current_on_successive_trips": f["trend"], "customers_out": f["customers_out"]},
                "circuit": {"underground_cable_pct": f["ug_pct"], "high_fire_threat_district": f["hftd"],
                            "fault_indicators": fci_text,
                            "customers_beyond_indicated_section": f["faulted_section_customers"] if f["fci"] == "localised" else None},
                "conditions": {"weather": WEATHER2[f["weather"]], "lightning_strikes_last_30_min": f["strikes_30min"],
                               "gust_mph": f["gust_mph"],
                               "fire_weather": None if f["fire_weather"] == "none" else f["fire_weather"].replace("_", " ")},
                "reports": calls or None,
                "tie": {"switch": f["tie_switch"], "neighbour_circuit": adj, "neighbour_load_a": f["adj_load_a"],
                        "neighbour_emergency_rating_a": f["adj_emerg_a"], "load_to_transfer_a": f["transfer_a"],
                        "transfer_estimate_basis": f["transfer_source"]},
                "procedure": self._procedure(f),
            }))
        if surface == "scada_log":
            lines = [f"{f['utility']} SCADA EVENT LOG - {sub} - {feeder}"]
            gaps = [pick(rng, [2, 5, 15, 30]) for _ in range(f["shots"])]
            t = t0 - timedelta(seconds=sum(gaps) + f["shots"] * 0.4 + 0.3, milliseconds=rng.randint(0, 999))
            ia = f["fault_ka"] * 1000

            def stamp(x):
                return x.strftime("%H:%M:%S") + f".{x.microsecond // 1000:03d}"
            for i in range(f["shots"] + 1):
                cur = {"steady": ia * rng.uniform(0.98, 1.02), "falling": ia * (1 - 0.18 * i),
                       "erratic": ia * rng.uniform(0.5, 1.1)}[f["trend"]]
                lines.append(f"{stamp(t)}  {f['device_id']}  TRIP  {f['fault_type']}  I={int(cur)} A")
                if i < f["shots"]:
                    t += timedelta(seconds=gaps[i])
                    lines.append(f"{stamp(t)}  {f['device_id']}  RECLOSE {i + 1}")
                    t += timedelta(milliseconds=rng.randint(60, 400))
            lines.append(f"{stamp(t + timedelta(milliseconds=rng.randint(40, 120)))}  {f['device_id']}  LOCKOUT  ({f['customers_out']} cust)")
            lines.append(f"{(t0 + timedelta(minutes=2)).strftime('%H:%M:%S')}  FCI  {fci_text}")
            lines.append("")
            lines.append(kv_lines([
                ("Weather", f"{WEATHER2[f['weather']]}, gusts {f['gust_mph']} mph, {f['strikes_30min']} strikes/30 min"),
                ("Fire weather", f["fire_weather"].replace("_", " ") + (" (HFTD circuit)" if f["hftd"] else " (not HFTD)")),
                ("Underground cable", f"{f['ug_pct']}% of circuit length"),
                ("Tie", f"{f['tie_switch']} to {adj}: {f['adj_load_a']} A now, emergency rating {f['adj_emerg_a']} A; "
                        f"transfer ~{f['transfer_a']} A ({f['transfer_source']})"),
            ], rng))
            if calls:
                lines += ["", "Trouble calls / field:", bullets(calls, rng)]
            lines += ["", self._procedure(f)]
            return "\n".join(lines)
        # operator_handover
        para = [f"{feeder} ({sub}) {f['device'].replace('_', ' ')} {f['device_id']} locked out at {fmt_ts(t0, rng)} after "
                f"{f['shots']} recloses, {f['fault_type']}, about {f['fault_ka']} kA, current {f['trend']} across the shots.",
                f"{f['customers_out']} customers out.", fci_text[0].upper() + fci_text[1:] + ".",
                f"Weather {WEATHER2[f['weather']]}" + (f", {f['strikes_30min']} strikes in the last half hour." if f["strikes_30min"] else "."),
                f"Circuit is {f['ug_pct']}% cable" + (", HFTD" if f["hftd"] else "") +
                ("" if f["fire_weather"] == "none" else f", {f['fire_weather'].replace('_', ' ')} in effect") + "."]
        para += calls
        para.append(f"Tie {f['tie_switch']} to {adj} is available: {adj} at {f['adj_load_a']} A, emergency rating {f['adj_emerg_a']} A, "
                    f"we'd move about {f['transfer_a']} A ({f['transfer_source']}).")
        para.append(pick(rng, ["Field supervisor asking if we can just try it once.", "Crews are 40 min out.",
                               "Patrol truck is available but slow in this weather.", "Customers calling in steadily."]))
        return ("CONTROL ROOM HANDOVER NOTE - " + f["operator"] + "\n\n" + noisy(join_sentences(para), rng, 0.1)
                + "\n\nRef: " + self._procedure(f))

    def questions(self):
        return {
            "reenergise_plan": choice("How should the operator go about re-energising this circuit?", {
                "test_close_once": "Test-close the locked-out device once from the control room.",
                "isolate_section_then_test": "Open a switch to cut off the indicated faulted section, then test-close to restore the rest.",
                "transfer_via_tie": "Pick up the unfaulted customers through the tie to the neighbouring circuit and leave the faulted part dead.",
                "patrol_before_any_close": "Keep the circuit dead until a patrol has looked at the line.",
                "hold_for_clearance_release": "Do no switching at all until the crew holding a clearance on the circuit has released it.",
            }),
            "permanent_fault_likelihood": score("How likely is it that the fault is still there and a re-close would trip again?", [
                "Unlikely: the signature fits a passing contact such as lightning or a blown branch.",
                "Even: the signs point both ways.",
                "Likely: the fault looks solid, though nothing confirms damage.",
                "Near certain: damage is reported or the signature leaves little doubt.",
            ]),
            "reclose_harm_potential": score("If the circuit were re-energised with the fault still present, how much harm could follow?", [
                "Little: a second trip on overhead line, in wet or open country.",
                "Moderate: added stress on equipment and a longer outage, with no public exposure.",
                "High: likely damage to underground cable, or a reported conductor near people.",
                "Severe: fire-ignition risk in dry, windy conditions, or a person possibly in contact.",
            ]),
            "test_close_allowed_by_procedure": noul(
                "The written operating procedure permits a single control-room test-close of this circuit right now.",
                yes="Every condition the procedure sets for a single test-close is met on the record shown.",
                no="At least one condition in the procedure is not met or cannot be confirmed."),
            "tie_can_carry_transfer": noul(
                "The neighbouring circuit can take the unfaulted load through the tie without going over its emergency rating.",
                yes="Its present load plus the load to be transferred stays at or under its emergency rating.",
                no="Its present load plus the transferred load would exceed its emergency rating."),
        }


# =========================================================================== schema 3
SYMPTOMS = {
    "no_power": "no power at all",
    "partial_power": "half the house is out, some outlets work",
    "bright_dim": "lights going bright and dim, a TV and a microwave quit",
    "flicker": "lights flickering on and off for the last few days",
    "low_voltage": "everything seems weak, AC won't start",
}
HAZ_NOTES = {
    None: None,
    "weatherhead_sparks": "sparks at the weatherhead where the wires come in",
    "low_drop": "service wire sagging over the driveway, maybe 9 ft up",
    "meter_hot": "meter base buzzing and warm to the touch",
    "tree_on_drop": "a limb is resting on the service wire",
}


class SinglePremiseTroubleCall(Schema):
    name = "single_premise_trouble_call"
    description = "A trouble call from one premises with smart-meter pings: where the fault probably is and whom to send."
    surfaces = {"json": 0.5, "trouble_ticket": 0.25, "call_transcript": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        true_loc = pick_w(rng, {"customer_wiring": 3, "service_drop_or_meter": 4, "distribution_transformer": 3,
                                "primary_line_upstream": 2})
        service = pick_w(rng, {"overhead": 3, "underground": 2})
        n_on_xfmr = rng.randint(2, 9)
        neutral = false_neutral = False
        others_out, xfmrs_out, neigh_unbal = 0, 0, 0
        v1 = v2 = 120
        meter_resp = True
        symptom = "no_power"
        hazard = None
        if true_loc == "customer_wiring":
            symptom = pick(rng, ["no_power", "partial_power", "flicker"])
            v1, v2 = rng.randint(117, 124), rng.randint(117, 124)
        elif true_loc == "service_drop_or_meter":
            kind = pick_w(rng, {"open": 2, "neutral": 4, "one_leg": 1})
            if kind == "open":
                symptom, meter_resp = "no_power", False
            elif kind == "neutral":
                neutral = True
                symptom = pick(rng, ["bright_dim", "flicker", "partial_power"])
                v1, v2 = rng.randint(84, 106), rng.randint(134, 156)
            else:
                symptom, v1, v2 = "partial_power", rng.randint(118, 124), pick(rng, [0, 0, 3, 41])
            hazard = pick_w(rng, {None: 3, "weatherhead_sparks": 1, "low_drop": 1, "tree_on_drop": 1}) if service == "overhead" else pick(rng, [None, None, "meter_hot"])
        elif true_loc == "distribution_transformer":
            symptom = pick(rng, ["no_power", "no_power", "low_voltage", "bright_dim", "bright_dim"])
            if symptom == "no_power":
                meter_resp, others_out = False, rng.randint(max(1, n_on_xfmr - 2), n_on_xfmr - 1) if n_on_xfmr > 2 else 1
            elif symptom == "low_voltage":
                v1, v2 = rng.randint(101, 109), rng.randint(101, 109)
                others_out = 0
            else:
                neutral = True
                v1, v2 = rng.randint(88, 104), rng.randint(133, 150)
                neigh_unbal = rng.randint(1, max(1, n_on_xfmr - 1))
        else:
            symptom, meter_resp = "no_power", False
            others_out = n_on_xfmr - 1
            xfmrs_out = rng.randint(2, 9)
        pings_done = pick_w(rng, {"all": 6, "partial": 2, "none": 2})
        breaker = pick_w(rng, {"main_on": 3, "reset_no_change": 2, "not_asked": 3, "no_access": 1})
        reported_by = pick_w(rng, {"customer": 6, "electrician": 2, "fire_department": 1})
        if reg == "clear":
            pings_done = "all"
            breaker = "main_on" if true_loc != "customer_wiring" else "reset_no_change"
        elif reg == "conflicting":
            # the caller's story and the meter data disagree
            if true_loc == "customer_wiring":
                reported_by = "customer"
                false_neutral = chance(rng, 0.5)      # caller says lights going bright and dim, but legs read balanced
                if false_neutral:
                    symptom = "bright_dim"
            elif true_loc in ("service_drop_or_meter", "distribution_transformer"):
                pings_done = pick(rng, ["partial", "none"])
            else:
                xfmrs_out = pick(rng, [1, 2])
        if meter_resp is False:
            v1 = v2 = None
        return {"regime": reg, "true_location": true_loc, "utility": _utility(rng), "premise": _address(rng),
                "service": service, "meters_on_transformer": n_on_xfmr, "neighbours_out_same_xfmr": others_out,
                "other_transformers_out": xfmrs_out, "neighbours_unbalanced": neigh_unbal, "meter_responded": meter_resp, "leg_v1": v1, "leg_v2": v2,
                "lost_neutral_physical": neutral, "caller_claims_bright_dim": false_neutral, "symptom": symptom,
                "pings": pings_done, "breaker_check": breaker, "reported_by": reported_by, "hazard": hazard,
                "medical_registry": chance(rng, 0.15),
                "weather": pick_w(rng, {"clear": 4, "extreme_heat": 2, "extreme_cold": 2, "post_storm": 2}),
                "calls_30d": pick(rng, [0, 0, 0, 1, 2, 4]), "xfmr_id": f"T-{rng.randint(10000, 99999)}",
                "pad_mount": service == "underground", "now": rand_ts(rng), "caller": person(rng),
                "ticket": ident(rng, "TC", 7, "-"), "troubleman_eta_min": pick(rng, [20, 35, 50, 75, 120, 180])}

    def _evidence_location(self, f):
        if f["other_transformers_out"] >= 2:
            return "primary_line_upstream"
        if f["pings"] != "none" and f["neighbours_out_same_xfmr"] >= 2:
            return "distribution_transformer"
        if f["meter_responded"]:
            v = sorted([f["leg_v1"], f["leg_v2"]])
            if v[0] < 60 or (v[1] - v[0]) > 20:
                if f["pings"] != "none" and f["neighbours_unbalanced"]:
                    return "distribution_transformer"
                return "service_drop_or_meter"
            if max(v) < 110:
                return "distribution_transformer"
            return "customer_wiring"
        return "service_drop_or_meter" if f["pings"] == "all" else ("distribution_transformer" if f["neighbours_out_same_xfmr"] else "service_drop_or_meter")

    def intended(self, f):
        loc = self._evidence_location(f)
        oh = f["service"] == "overhead"
        if loc == "customer_wiring":
            crew = "troubleman_single" if f["hazard"] else "refer_to_customer_electrician"
        elif loc == "service_drop_or_meter":
            crew = "troubleman_single" if oh else "underground_crew"
        elif loc == "distribution_transformer":
            crew = "underground_crew" if f["pad_mount"] else ("line_crew_bucket" if f["calls_30d"] >= 2 else "troubleman_single")
        else:
            crew = "line_crew_bucket" if oh else "underground_crew"
        neutral_seen = f["meter_responded"] and abs(f["leg_v1"] - f["leg_v2"]) > 20 and min(f["leg_v1"], f["leg_v2"]) > 60
        extreme = f["weather"] in EXTREME
        if f["hazard"] in ("weatherhead_sparks", "low_drop", "meter_hot") or (neutral_seen and f["symptom"] == "bright_dim"):
            prom = 3
        elif f["symptom"] == "no_power" and (f["medical_registry"] or extreme or loc != "customer_wiring"):
            prom = 2
        elif loc == "customer_wiring" and not f["hazard"]:
            prom = 0
        else:
            prom = 1
        conf = {"all": 3, "partial": 2, "none": 1}[f["pings"]]
        if f["caller_claims_bright_dim"] or f["breaker_check"] in ("not_asked", "no_access"):
            conf -= 1
        if f["other_transformers_out"] == 1:
            conf -= 1
        return {"probable_fault_point": loc, "first_resource": crew, "response_promptness": prom,
                "diagnosis_confidence": max(0, min(3, conf)), "lost_neutral_indicated": "yes" if neutral_seen else "no"}

    def _ping_text(self, f):
        if f["pings"] == "none":
            return "meter pings not run yet"
        n = f["meters_on_transformer"]
        unbal = (f"; {f['neighbours_unbalanced']} neighbour meter(s) also reading unbalanced legs"
                 if f["neighbours_unbalanced"] else "")
        if f["pings"] == "partial":
            return (f"pinged 2 of {n} meters on {f['xfmr_id']}: " + ("1 no response" if f["neighbours_out_same_xfmr"] else "both responded")
                    + ("; 1 reads unbalanced legs" if f["neighbours_unbalanced"] else ""))
        return (f"pinged all {n} meters on {f['xfmr_id']}: {f['neighbours_out_same_xfmr']} neighbour(s) not responding" + unbal
                + (f"; {f['other_transformers_out']} other transformer(s) on the same tap with no-response meters"
                   if f["other_transformers_out"] else ""))

    def _meter_text(self, f):
        if not f["meter_responded"]:
            return "premise meter: no response (last gasp logged)"
        return f"premise meter on-demand read: L1-N {f['leg_v1']} V, L2-N {f['leg_v2']} V"

    def _caller_story(self, f, rng):
        sym = "lights going bright and dim" if f["caller_claims_bright_dim"] else SYMPTOMS[f["symptom"]]
        s = [f"Caller reports {sym}."]
        if f["reported_by"] == "electrician":
            s.append(pick(rng, ["Their electrician is on site and says the problem is on the utility side.",
                                "Electrician says the panel checks out and it's our side."]))
        elif f["reported_by"] == "fire_department":
            s.append("Fire department called it in after a smell of burning was reported.")
        s.append({"main_on": "Main breaker is on.", "reset_no_change": "Customer reset the main breaker, no change.",
                  "not_asked": "", "no_access": "Customer can't get to the panel (rental, locked utility room)."}[f["breaker_check"]])
        if f["hazard"]:
            s.append(f"Also reports {HAZ_NOTES[f['hazard']]}.")
        if f["calls_30d"]:
            s.append(f"{f['calls_30d']} earlier call(s) from this address in the last 30 days about flickering.")
        if f["medical_registry"]:
            s.append("Premises is on the medical-equipment registry.")
        return s

    def render(self, f, rng, surface):
        if surface == "json":
            return dumps(prune({
                "ticket": {"id": f["ticket"], "utility": f["utility"], "premise": f["premise"],
                           "opened": fmt_ts(f["now"], style="iso"), "reported_by": f["reported_by"],
                           "caller_remarks": noisy(join_sentences(self._caller_story(f, rng)), rng, 0.15)},
                "ami": {"premise_meter": {"responded": f["meter_responded"], "l1_volts": f["leg_v1"], "l2_volts": f["leg_v2"]},
                        "transformer": f["xfmr_id"], "meters_on_transformer": f["meters_on_transformer"],
                        "neighbour_pings": self._ping_text(f)},
                "service": {"type": f["service"], "transformer_style": "pad-mount" if f["pad_mount"] else "pole-mount"},
                "conditions": {"weather": f["weather"].replace("_", " "), "medical_registry": f["medical_registry"] or None},
                "resources": {"troubleman_eta_minutes": f["troubleman_eta_min"]},
            }))
        if surface == "trouble_ticket":
            rows = [("Ticket", f["ticket"]), ("Premise", f["premise"]), ("Opened", fmt_ts(f["now"], rng)),
                    ("Service", f"{f['service']}, {'pad-mount' if f['pad_mount'] else 'pole-mount'} transformer {f['xfmr_id']}"),
                    ("Meter", self._meter_text(f)), ("Neighbour pings", self._ping_text(f)),
                    ("Weather", f["weather"].replace("_", " ")), ("Medical registry", "YES" if f["medical_registry"] else "no"),
                    ("Troubleman ETA", f"{f['troubleman_eta_min']} min")]
            return (f"{f['utility'].upper()} - TROUBLE TICKET\n{kv_lines(rows, rng)}\n\nRemarks:\n"
                    + noisy(join_sentences(self._caller_story(f, rng)), rng, 0.15))
        lines = [f"Call log {f['ticket']} - {fmt_ts(f['now'], rng)}",
                 f"AGENT: {f['utility']}, outage line, what's the address?",
                 f"CALLER: {f['premise']}."]
        for s in self._caller_story(f, rng):
            if s:
                lines.append(f"CALLER (summarised): {s}")
        lines.append(f"AGENT NOTE: {self._meter_text(f)}; {self._ping_text(f)}.")
        lines.append(f"AGENT NOTE: {f['service']} service; troubleman ETA about {f['troubleman_eta_min']} min; weather {f['weather'].replace('_', ' ')}.")
        return "\n".join(lines)

    def questions(self):
        return {
            "probable_fault_point": choice("Where is the fault behind this call most likely to be?", {
                "customer_wiring": "In the customer's own wiring or main panel, past the meter.",
                "service_drop_or_meter": "In the service conductors or meter that serve this premises alone.",
                "distribution_transformer": "In the transformer, or its fuse, that serves this premises and its neighbours.",
                "primary_line_upstream": "On the primary line or a device upstream that feeds several transformers.",
            }),
            "first_resource": choice("Who should be sent first?", {
                "troubleman_single": "A one-person troubleman to diagnose and make simple repairs.",
                "line_crew_bucket": "A full overhead line crew with a bucket truck.",
                "underground_crew": "An underground cable crew.",
                "refer_to_customer_electrician": "No utility crew; tell the customer to call their own electrician.",
            }),
            "response_promptness": score("How soon does someone need to reach this premises?", [
                "At the next routine service visit, within a few days.",
                "Some time during the current shift.",
                "Within about two hours.",
                "Straight away, as an emergency call.",
            ]),
            "diagnosis_confidence": score("How firmly do the call details and meter data point to one fault location?", [
                "Hardly at all: key readings are missing or contradict the caller.",
                "Loosely: one source of evidence, not cross-checked.",
                "Fairly firmly: the readings and the caller mostly agree.",
                "Firmly: complete meter data and the caller's account tell the same story.",
            ]),
            "lost_neutral_indicated": noul(
                "The readings and symptoms point to a lost or broken neutral serving this premises.",
                yes="Leg voltages are unbalanced, one high and one low, consistent with lights brightening and dimming.",
                no="Leg voltages are balanced, missing or near zero, or the symptoms fit some other cause."),
        }


# =========================================================================== schema 4
# (key, checker's note, applies to restore-via-tie orders too?)
DEFECTS = {
    "clerical": [("label", "step {a} names switch {d1} where the circuit map shows {d2}", True),
                 ("no_time", "no time or initials column for the switchman to fill in", True),
                 ("heading", "heading names the wrong substation", True),
                 ("vague_verify", "step {a} says 'verify' without saying what to verify", True)],
    "material": [("no_test_dead", "no test-for-absence-of-voltage before the grounds go on", False),
                 ("no_hold_tag", "open points are not hold-tagged", False),
                 ("no_ground_removal", "grounds are never listed for removal before the section is re-energised", False),
                 ("no_relay_step", "nothing confirms the receiving relay settings before the load is picked up", True)],
    "dangerous": [("disc_under_load", "step {a} opens {d1}, a non-load-break disconnect, while it carries roughly {amps} A", True),
                  ("grounds_first", "grounds go on before {d1} is opened", False)],
}


class SwitchingOrderReview(Schema):
    name = "switching_order_review"
    description = "A written switching order to transfer load or isolate equipment, checked by the switching coordinator before release."
    surfaces = {"json": 0.5, "switching_order": 0.3, "review_thread": 0.2}

    def sample_factors(self, rng):
        reg = regime(rng)
        purpose = pick_w(rng, {"isolate_for_repair": 3, "planned_maintenance": 3, "restore_via_tie": 2})
        same_sub = chance(rng, 0.5)
        same_bus = same_sub and chance(rng, 0.5)
        transition = pick_w(rng, {"open": 1, "closed": 1})
        angle = None if same_sub else maybe(rng, round(rng.uniform(0.5, 16.0), 1), 0.35)
        angle_limit = pick(rng, [8, 10, 12])
        normal = pick(rng, [400, 450, 500, 600])
        emerg = int(normal * pick(rng, [1.15, 1.2, 1.25]))
        transfer = rng.randint(60, 300)
        clpu_h = pick(rng, [0, 0, 1, 4, 12]) if purpose == "restore_via_tie" else 0
        clpu = {0: 1.0, 1: 1.3, 4: 1.7, 12: 2.0}[clpu_h]
        target_total = pick_w(rng, {normal * 0.8: 2, normal * 1.05: 2, emerg * 0.97: 3, emerg * 1.08: 2})
        recv_peak = max(80, int(target_total - transfer * clpu + rng.uniform(-25, 25)))
        pickup = int(emerg * pick(rng, [1.05, 1.15, 1.3, 1.5]))
        reach_mi = round(pick(rng, [rng.uniform(0.1, 1.2), rng.uniform(0.1, 1.2), rng.uniform(1.6, 4.5)]), 1)
        defect_level = pick_w(rng, {"none": 3, "clerical": 3, "material": 2, "dangerous": 2})
        if reg == "clear":
            if chance(rng, 0.5):
                defect_level, transition, recv_peak, reach_mi, clpu_h, clpu = "none", "open", int(normal * 0.5), 0.4, 0, 1.0
            else:
                defect_level = "dangerous"
        elif reg == "conflicting":
            # tidy paperwork on an overloaded transfer, or a sound plan with one dangerous step written by a veteran
            if chance(rng, 0.5):
                defect_level, recv_peak = "none", int(emerg * 1.05 - transfer * clpu)
            else:
                defect_level, recv_peak = "dangerous", int(normal * 0.6)
        defects = []
        if defect_level != "none":
            pools = ["clerical"] if defect_level == "clerical" else ([defect_level] + (["clerical"] if chance(rng, 0.4) else []))
            for lvl in pools:
                options = [d for d in DEFECTS[lvl] if d[2] or purpose != "restore_via_tie"]
                key, _, _ = pick(rng, options)
                defects.append((lvl, key))
        flagged = chance(rng, 0.6)      # a first-pass checker has written notes on some orders
        from_no = rng.randint(1, 16)
        return {"regime": reg, "utility": _utility(rng), "order_no": ident(rng, "SO", 6, "-"), "purpose": purpose,
                "substation_a": _substation(rng), "substation_b": _substation(rng), "same_substation": same_sub,
                "same_bus": same_bus, "transition": transition, "phase_angle_deg": angle, "angle_limit_deg": angle_limit,
                "from_no": from_no, "to_no": pick(rng, [n for n in range(1, 17) if n != from_no]), "transfer_a": transfer,
                "recv_peak_a": max(60, recv_peak), "recv_normal_a": normal, "recv_emerg_a": emerg, "outage_hours_before_pickup": clpu_h,
                "clpu_factor": clpu, "relay_pickup_a": pickup, "reach_extension_mi": reach_mi, "defects": defects, "checker_notes": flagged,
                "author": person(rng), "author_years": pick(rng, [1, 2, 4, 9, 17, 26]), "field_checked": chance(rng, 0.6),
                "tie_sw": f"S-{rng.randint(100, 999)}", "open_sw": f"S-{rng.randint(100, 999)}",
                "iso_sw": f"S-{rng.randint(100, 999)}", "disc": f"D-{rng.randint(10, 299)}",
                "date": rand_ts(rng, hours=(6, 15))}

    def _strain(self, f):
        total = f["recv_peak_a"] + f["transfer_a"] * f["clpu_factor"]
        if total <= 0.9 * f["recv_normal_a"]:
            return 0, total
        if total <= 0.93 * f["recv_emerg_a"]:
            return 1, total
        if total <= f["recv_emerg_a"]:
            return 2, total
        return 3, total

    def intended(self, f):
        strain, total = self._strain(f)
        # a defect the reader can see: anything written into the steps, or any defect when a checker left notes
        lvls = [lv for lv, k in f["defects"] if f["checker_notes"] or k not in ("no_time", "heading")]
        sev = 3 if "dangerous" in lvls else 2 if "material" in lvls else 1 if "clerical" in lvls else 0
        parallel = f["transition"] == "closed" and not f["same_bus"]
        prot = total >= 0.85 * f["relay_pickup_a"] or f["reach_extension_mi"] > 1.5
        bad_parallel = parallel and not f["same_substation"] and (f["phase_angle_deg"] is None or f["phase_angle_deg"] > f["angle_limit_deg"])
        if strain == 3:
            call = "reject_switching_plan"
        elif sev == 3 or bad_parallel:
            call = "return_for_rewrite"
        elif sev in (1, 2) or prot:
            call = "release_after_minor_corrections"
        else:
            call = "release_as_written"
        return {"order_release_call": call, "receiving_feeder_strain": strain, "step_defect_severity": sev,
                "sources_paralleled": "yes" if parallel else "no", "protection_check_needed": "yes" if prot else "no"}

    def _steps(self, f):
        """The written steps with the drawn defects applied, plus the first-pass checker's notes (if any)."""
        keys = {k for _, k in f["defects"]}
        sub_a = f["substation_a"][1]
        frm, to = f"{sub_a}-{f['from_no']:02d}", f"{(sub_a if f['same_substation'] else f['substation_b'][1])}-{f['to_no']:02d}"
        tie, opn, iso, disc = f["tie_sw"], f["open_sw"], f["iso_sw"], f["disc"]
        close_tie = f"Close tie {tie} ({frm}/{to})"
        open_pt = f"Open {opn} on {frm}" if "disc_under_load" not in keys else f"Open {disc} (disconnect) on {frm}"
        verify = "Verify" if "vague_verify" in keys else f"Verify {to} relay settings group 1 active, load {f['recv_peak_a']} A"
        steps = [] if "no_relay_step" in keys else [verify]
        steps += [close_tie, open_pt] if f["transition"] == "closed" else [open_pt, close_tie]
        if f["purpose"] != "restore_via_tie":
            grounds = "Apply personal protective grounds both sides of work location"
            iso_step = f"Open {iso} - verify open all three phases"
            body = [grounds, iso_step] if "grounds_first" in keys else [iso_step]
            if "no_hold_tag" not in keys:
                body.append(f"Hold tag {opn} and {iso}")
            if "no_test_dead" not in keys:
                body.append("Test for absence of voltage at work location")
            if "grounds_first" not in keys:
                body.append(grounds)
            body.append("Release to crew - work clearance issued")
            if "no_ground_removal" not in keys:
                body += ["On completion: remove grounds, release clearance", f"Close {iso}, remove tags"]
            else:
                body += [f"On completion: release clearance, close {iso}"]
            steps += body
        else:
            steps += [f"Confirm customers beyond {opn} restored via {to} (AMI)"]
        if "label" in keys:
            steps = [s.replace(iso, f"S-{int(iso[2:]) + 7}") if i == len(steps) - 2 else s for i, s in enumerate(steps)]
        notes = []
        if f["checker_notes"]:
            for lvl, k in f["defects"]:
                tmpl = next(t for kk, t, _ in DEFECTS[lvl] if kk == k)
                a = next((i for i, s in enumerate(steps, 1) if (disc in s or "grounds" in s.lower() or s == "Verify")), 2)
                notes.append(tmpl.format(a=a, d1=disc if k == "disc_under_load" else iso, d2=iso,
                                         amps=int(f["transfer_a"] * 0.8)))
        return frm, to, steps, notes

    def render(self, f, rng, surface):
        frm, to, steps, dtexts = self._steps(f)
        loads = {"receiving_circuit_peak_a": f["recv_peak_a"], "receiving_normal_rating_a": f["recv_normal_a"],
                 "receiving_emergency_rating_a": f["recv_emerg_a"], "load_to_transfer_a": f["transfer_a"],
                 "hours_out_before_pickup": f["outage_hours_before_pickup"] or None,
                 "receiving_relay_pickup_a": f["relay_pickup_a"], "added_line_beyond_receiving_end_mi": f["reach_extension_mi"]}
        src = ("same substation, same bus" if f["same_bus"] else "same substation, different transformer bank and bus"
               if f["same_substation"] else f"different substations ({f['substation_a'][0]} / {f['substation_b'][0]})")
        if surface == "json":
            return dumps(prune({
                "order": {"number": f["order_no"], "utility": f["utility"], "purpose": f["purpose"].replace("_", " "),
                          "from_circuit": frm, "to_circuit": to, "sources": src, "written_by": f["author"],
                          "author_years_switching": f["author_years"], "field_checked": f["field_checked"],
                          "planned_start": fmt_ts(f["date"], style="iso"),
                          "steps": [f"{i}. {s}" for i, s in enumerate(steps, 1)]},
                "loading": loads,
                "phase_angle": ({"measured_deg": f["phase_angle_deg"], "closed_transfer_limit_deg": f["angle_limit_deg"]}
                                if not f["same_substation"] else None),
                "reviewer_flags": dtexts or None,
            }))
        if surface == "switching_order":
            head = kv_lines([("Switching order", f["order_no"]), ("Purpose", f["purpose"].replace("_", " ")),
                             ("From / to", f"{frm} -> {to}"), ("Sources", src), ("Planned", fmt_ts(f["date"], rng)),
                             ("Written by", f"{f['author']} ({f['author_years']} yrs)"),
                             ("Field checked", "yes" if f["field_checked"] else "no")], rng)
            tbl = table(["#", "Step", "Time", "Init"], [[i, s, "", ""] for i, s in enumerate(steps, 1)], rng)
            ld = (f"Loading: {to} peak {f['recv_peak_a']} A; ratings {f['recv_normal_a']} A normal / {f['recv_emerg_a']} A emergency; "
                  f"transfer {f['transfer_a']} A" + (f" after {f['outage_hours_before_pickup']} h out" if f["outage_hours_before_pickup"] else "")
                  + f"; relay pickup {f['relay_pickup_a']} A; adds {f['reach_extension_mi']} mi beyond {to}'s normal end.")
            if not f["same_substation"]:
                ld += f" Phase angle across tie: {show(f['phase_angle_deg'], rng, lambda v: f'{v} deg')} (closed-transfer limit {f['angle_limit_deg']} deg)."
            tail = ("\nChecker notes:\n" + bullets(dtexts, rng)) if dtexts else ""
            return f"{f['utility']} - SWITCHING ORDER\n{head}\n\n{tbl}\n\n{ld}{tail}"
        rev = person(rng)
        msgs = [(f["author"], f"{f['order_no']} up for review: {f['purpose'].replace('_', ' ')}, {frm} to {to} ({src}). Steps: "
                 + "; ".join(f"({i}) {s}" for i, s in enumerate(steps, 1)))]
        msgs.append((rev, f"what's {to} carrying at peak? and rating?"))
        msgs.append((f["author"], f"{f['recv_peak_a']} A peak, {f['recv_normal_a']}/{f['recv_emerg_a']} A normal/emergency, we add ~{f['transfer_a']} A"
                     + (f" after {f['outage_hours_before_pickup']}h out" if f["outage_hours_before_pickup"] else "")
                     + f". relay pickup {f['relay_pickup_a']} A, reach goes {f['reach_extension_mi']} mi past the normal end"))
        if not f["same_substation"]:
            msgs.append((rev, "angle across the tie?"))
            msgs.append((f["author"], show(f["phase_angle_deg"], rng, lambda v: f"{v} deg on last reading") + f" (limit {f['angle_limit_deg']})"))
        for d in dtexts:
            msgs.append((rev, d))
        msgs.append((f["author"], pick(rng, ["been writing these for years, it's the standard sequence", "can we get it released today?",
                                             "field checked it last week", "happy to fix whatever"])))
        t = f["date"] - timedelta(days=1)
        out = []
        for who, text in msgs:
            t += timedelta(minutes=rng.randint(1, 19))
            out.append(f"[{t.strftime('%H:%M')}] {who.split()[0].lower()}: {text}")
        return f"#switching-review ({f['utility']})\n" + "\n".join(out)

    def questions(self):
        return {
            "order_release_call": choice("What should the switching coordinator do with this order?", {
                "release_as_written": "Release the order to the field exactly as written.",
                "release_after_minor_corrections": "Release it once small, specific corrections or an added check step are made.",
                "return_for_rewrite": "Send it back to be rewritten; the sequence itself is unsafe or unproven.",
                "reject_switching_plan": "Refuse this plan altogether; the receiving circuit cannot take the load and another approach is needed.",
            }),
            "receiving_feeder_strain": score("How close will the receiving circuit come to its limits once it carries the transferred load?", [
                "Comfortably inside its normal rating.",
                "Above normal rating, but well inside the emergency rating.",
                "Close to the emergency rating, especially once cold-load pickup is allowed for.",
                "Beyond the emergency rating at the expected peak.",
            ]),
            "step_defect_severity": score("How serious are the defects in the written switching steps?", [
                "None found: devices, actions and order are all correct.",
                "Clerical: labels or blanks that a switchman would catch.",
                "Material: a missing check or tag step that has to be added.",
                "Dangerous: a step that breaks load on a device not rated for it, or grounds or energises in the wrong order.",
            ]),
            "sources_paralleled": noul(
                "Carrying out the steps as written would tie two separate sources together, even briefly."),
            "protection_check_needed": noul(
                "The receiving circuit's protection settings must be checked before this transfer goes ahead.",
                yes="Combined load comes near the relay pickup, or the transfer stretches the circuit well past the reach its settings were coordinated for.",
                no="Peak load stays well under pickup and the added line is within the reach already covered."),
        }


# =========================================================================== schema 5
CRIT_SITES = {
    # type: (name templates, kW range, life-safety statement)
    "hospital": (["{a} Regional Medical Center", "{a} Community Hospital"], (900, 3200)),
    "nursing_home": (["{a} Care Center", "{a} Rehabilitation and Nursing"], (150, 480)),
    "dispatch_911": (["{a} County 911 Center", "{a} Emergency Communications"], (70, 240)),
    "dialysis_clinic": (["{a} Dialysis Center", "{a} Kidney Care"], (90, 260)),
    "water_treatment": (["{a} Water Treatment Plant", "{a} Booster Pump Station"], (180, 1400)),
    "wastewater_lift_station": (["Lift Station {n}", "{a} Sewage Lift Station"], (45, 260)),
    "telecom_office": (["{a} telephone central office", "{a} fibre hub"], (120, 600)),
    "cooling_warming_center": (["{a} Community Center shelter", "{a} High School shelter"], (60, 260)),
    "cold_storage_warehouse": (["{a} Cold Storage", "{a} Produce Terminal"], (250, 1200)),
    "data_center": (["{a} colocation data hall", "{a} county data center"], (300, 1500)),
    "county_admin_building": (["{a} County Administration Building", "{a} Courthouse Annex"], (120, 450)),
}
SITE_LABEL = {"hospital": "hospital", "nursing_home": "nursing home", "dispatch_911": "911 dispatch centre",
              "dialysis_clinic": "dialysis clinic", "water_treatment": "water treatment plant",
              "wastewater_lift_station": "sewage lift station", "telecom_office": "telecom central office",
              "cooling_warming_center": "public cooling/warming shelter", "cold_storage_warehouse": "cold-storage warehouse",
              "data_center": "data centre", "county_admin_building": "county office building"}
OCCUPANTS = {"hospital": (80, 400, "patients"), "nursing_home": (40, 180, "residents"), "dispatch_911": (6, 30, "staff on shift"),
             "dialysis_clinic": (12, 60, "patients"), "cooling_warming_center": (20, 300, "people sheltering")}
GEN_SIZES_KVA = [150, 300, 500, 800, 1000, 1500, 2000]
# how far a reviewer would stretch a published ETR, by how it was produced
ETR_FACTOR = {"crew_on_site_assessed": 1.0, "storm_default": 1.2, "not_yet_assessed": 1.4}


class CriticalFacilityBackup(Schema):
    name = "critical_facility_backup"
    description = "A critical facility is out of utility supply and running (or failing) on backup; how the utility supports it."
    surfaces = {"json": 0.5, "eoc_request": 0.25, "key_account_log": 0.25}

    def sample_factors(self, rng):
        reg = regime(rng)
        ftype = pick(rng, list(CRIT_SITES))
        kw_lo, kw_hi = CRIT_SITES[ftype][1]
        kw = rng.randint(kw_lo, kw_hi)
        backup = pick_w(rng, {"running": 6, "running_partial": 2, "failed": 1, "none_installed": 1})
        long_fuel = ftype in ("hospital", "dispatch_911", "telecom_office", "data_center")
        fuel_h = (None if backup in ("failed", "none_installed")
                  else maybe(rng, round(rng.uniform(20, 96) if long_fuel else rng.uniform(4, 48), 0), 0.15))
        refuel_eta_h = maybe(rng, round(rng.uniform(2, 30), 0), 0.3)
        etr_h = round(pick_w(rng, {rng.uniform(1.5, 8): 3, rng.uniform(8, 20): 2, rng.uniform(20, 40): 1}), 1)
        etr_basis = pick_w(rng, {"crew_on_site_assessed": 3, "storm_default": 3, "not_yet_assessed": 2})
        alt = pick_w(rng, {"none": 3, "second_feed_dead": 2, "second_feed_live": 5})
        alt_cap_kw = int(kw * rng.uniform(0.75, 1.9)) if alt == "second_feed_live" else None
        units = sorted(rng.sample(GEN_SIZES_KVA, rng.randint(1, 3)))
        units_free = pick_w(rng, {0: 1, 1: 3, 2: 2})
        if units_free == 0:
            units = []
        else:
            units = units[:units_free]
        hookup = pick_w(rng, {"docking_station": 3, "manual_transfer_switch": 2, "none": 2})
        travel_h = round(rng.uniform(1, 7), 1)
        competing = pick(rng, [0, 0, 1, 2, 4])
        weather = pick_w(rng, {"extreme_heat": 2, "extreme_cold": 2, "post_storm": 3, "storm_active": 2})
        if reg == "clear":
            if chance(rng, 0.5):   # site is fine on its own
                backup, fuel_h, etr_h, etr_basis, alt, alt_cap_kw = "running", round(rng.uniform(36, 72)), round(rng.uniform(2, 6), 1), "crew_on_site_assessed", "none", None
            else:                  # nothing working and a long outage
                backup, fuel_h, etr_h, etr_basis = pick(rng, ["failed", "none_installed"]), None, round(rng.uniform(14, 40), 1), "not_yet_assessed"
                units, hookup = [pick(rng, [1000, 1500, 2000])], "docking_station"
        elif reg == "borderline":
            if fuel_h is not None:
                eff = etr_h * ETR_FACTOR[etr_basis]
                fuel_h = max(2, round(near(rng, eff * 1.15, max(2, eff * 0.4), lo=1), 0))
        else:
            # live second feed that can't quite carry it, or a big unit with no place to plug in
            if chance(rng, 0.5):
                alt, alt_cap_kw = "second_feed_live", int(kw * rng.uniform(0.7, 0.98))
            else:
                units, hookup = [pick(rng, [1500, 2000])], "none"
                backup, fuel_h = "running", round(rng.uniform(2, 8), 0)
        a = pick(rng, SUB_A)
        return {"regime": reg, "utility": _utility(rng), "facility_type": ftype,
                "facility": pick(rng, CRIT_SITES[ftype][0]).format(a=a, n=rng.randint(2, 40)), "load_kw": kw,
                "backup": backup, "fuel_hours": fuel_h, "refuel_eta_h": refuel_eta_h, "etr_h": etr_h, "etr_basis": etr_basis,
                "alternate_feed": alt, "alt_capacity_kw": alt_cap_kw, "mobile_units_kva": units, "hookup": hookup,
                "unit_travel_h": travel_h, "competing_requests": competing, "weather": weather,
                "temp_f": _temp_for(rng, weather),
                "occupants": rng.randint(*OCCUPANTS[ftype][:2]) if ftype in OCCUPANTS else None,
                "on_life_support": rng.randint(0, 24) if ftype in ("hospital", "nursing_home") else 0,
                "feeder": _feeder(rng, _substation(rng)[1]), "contact": person(rng), "now": rand_ts(rng),
                "request_id": ident(rng, "EOC", 5, "-")}

    def _runout(self, f):
        if f["backup"] in ("failed", "none_installed"):
            return 3
        fh = f["fuel_hours"]
        etr = f["etr_h"] * ETR_FACTOR[f["etr_basis"]]
        if fh is None:
            return 2
        if f["refuel_eta_h"] is not None and f["refuel_eta_h"] < fh * 0.8:
            fh = fh + 24
        if fh < etr * 0.9:
            return 3
        if fh < etr * 1.3:
            return 2
        if fh < etr * 2.5:
            return 1
        return 0

    def _consequence(self, f):
        t = f["facility_type"]
        extreme = f["weather"] in EXTREME
        if t in ("hospital", "dispatch_911") or (t == "nursing_home" and f["on_life_support"] > 0):
            return 3
        if t in ("nursing_home", "dialysis_clinic", "wastewater_lift_station"):
            return 3 if extreme and t == "nursing_home" else 2
        if t == "cooling_warming_center":
            return 2 if extreme else 1
        if t == "water_treatment":
            return 2
        if t == "telecom_office":
            return 1
        return 0

    def intended(self, f):
        run = self._runout(f)
        need = f["load_kw"] / 0.8 * 1.25      # kVA at 0.8 pf, 25% margin for motor starting
        fits = any(u >= need for u in f["mobile_units_kva"]) and f["hookup"] != "none"
        alt_ok = f["alternate_feed"] == "second_feed_live" and (f["alt_capacity_kw"] or 0) >= f["load_kw"]
        if alt_ok:
            resp = "switch_to_alternate_feed"
        elif run >= 2 and fits:
            resp = "dispatch_mobile_generator"
        elif run >= 2:
            resp = "expedite_feeder_repair"
        else:
            resp = "monitor_and_arrange_fuel"
        return {"critical_site_response": resp, "runout_pressure": run, "loss_consequence": self._consequence(f),
                "mobile_unit_matches_load": "yes" if fits else "no", "alternate_feed_viable": "yes" if alt_ok else "no"}

    def render(self, f, rng, surface):
        bk = {"running": "generator running, full load", "running_partial": "generator running, life-safety branch only",
              "failed": "generator failed to start", "none_installed": "no permanent backup installed"}[f["backup"]]
        alt = {"none": "single utility feed", "second_feed_dead": "second utility feed, also out",
               "second_feed_live": f"second utility feed energised (manual throwover by a utility crew), about "
                                  f"{show(f['alt_capacity_kw'], rng)} kW of capacity available on it"}[f["alternate_feed"]]
        hook = {"docking_station": "generator docking station (cam-lock connection)",
                "manual_transfer_switch": "manual transfer switch, needs an electrician to terminate cables",
                "none": "no connection point for a portable unit"}[f["hookup"]]
        units = ", ".join(f"{u} kVA" for u in f["mobile_units_kva"]) or "none free"
        if surface == "json":
            return dumps(prune({
                "facility": {"name": f["facility"], "type": SITE_LABEL[f["facility_type"]], "peak_load_kw": f["load_kw"],
                             "occupants": (None if f["occupants"] is None else f"{f['occupants']} {OCCUPANTS[f['facility_type']][2]}"), "patients_on_powered_equipment": f["on_life_support"] or None,
                             "backup_status": bk, "fuel_hours_left": f["fuel_hours"], "refuel_eta_hours": f["refuel_eta_h"],
                             "portable_hookup": hook, "utility_supply": alt},
                "outage": {"utility": f["utility"], "feeder": f["feeder"], "etr_hours": f["etr_h"],
                           "etr_basis": f["etr_basis"].replace("_", " "), "as_of": fmt_ts(f["now"], style="iso")},
                "mobile_generation": {"free_units": units, "travel_hours": f["unit_travel_h"],
                                      "other_sites_requesting": f["competing_requests"]},
                "conditions": {"weather": f["weather"].replace("_", " "), "temperature_f": f["temp_f"]},
            }))
        if surface == "eoc_request":
            body = [f"Subject: {f['request_id']} power support request - {f['facility']}", f"From: {f['contact']}, facility manager", "",
                    noisy(join_sentences([
                        f"We are a {SITE_LABEL[f['facility_type']]} on {f['feeder']} and lost utility power.",
                        f"Status: {bk}.",
                        (f"Fuel on hand for roughly {int(f['fuel_hours'])} hours." if f["fuel_hours"] else
                         ("We can't tell how much fuel is left, the gauge is unreliable." if f["backup"].startswith("running") else "")),
                        (f"We have {f['occupants']} {OCCUPANTS[f['facility_type']][2]} on site." if f["occupants"] else ""),
                        ("" if not f["backup"].startswith("running") else
                         f"Our fuel vendor says {int(f['refuel_eta_h'])} hours for a delivery." if f["refuel_eta_h"] else
                         "Fuel vendor has not confirmed a delivery time."),
                        f"Peak load is about {f['load_kw']} kW. We have {hook}.",
                        f"Utility supply: {alt}.",
                        (f"{f['on_life_support']} residents/patients depend on powered equipment." if f["on_life_support"] else ""),
                        f"Your outage map says about {f['etr_h']} hours ({f['etr_basis'].replace('_', ' ')}).",
                        "Please advise what support you can provide."]), rng),
                    "", f"[utility note: free mobile units {units}; travel ~{f['unit_travel_h']} h; {f['competing_requests']} other site requests open; "
                        f"{f['weather'].replace('_', ' ')} {f['temp_f']}F]"]
            return "\n".join(body)
        t = f["now"]
        rows = [[fmt_ts(t - timedelta(hours=rng.randint(1, 4)), style="log"), "OMS", f"{f['feeder']} out, ETR {f['etr_h']} h ({f['etr_basis'].replace('_', ' ')})"],
                [fmt_ts(t - timedelta(minutes=rng.randint(30, 59)), style="log"), "KEY ACCT", f"{f['facility']}: {bk}"],
                [fmt_ts(t - timedelta(minutes=rng.randint(10, 29)), style="log"), "KEY ACCT",
                 f"fuel {show(f['fuel_hours'], rng, lambda v: f'{int(v)} h')}, refuel ETA {show(f['refuel_eta_h'], rng, lambda v: f'{int(v)} h')}, load {f['load_kw']} kW"],
                [fmt_ts(t, style="log"), "MOBILE GEN", f"free: {units}; travel {f['unit_travel_h']} h; site hookup: {hook}"],
                [fmt_ts(t, style="log"), "SUPPLY", alt]]
        if f["occupants"]:
            rows.insert(3, [fmt_ts(t - timedelta(minutes=rng.randint(1, 9)), style="log"), "KEY ACCT",
                            f"{f['occupants']} {OCCUPANTS[f['facility_type']][2]} on site"
                            + (f", {f['on_life_support']} on powered medical equipment" if f["on_life_support"] else "")])
        return (f"KEY ACCOUNT LOG - {f['utility']} - {f['facility']} ({SITE_LABEL[f['facility_type']]})\n"
                + table(["time", "src", "entry"], rows, rng)
                + f"\n\nWeather {f['weather'].replace('_', ' ')}, {f['temp_f']} F. Other critical-site requests open: {f['competing_requests']}.")

    def questions(self):
        return {
            "critical_site_response": choice("How should the utility support this facility now?", {
                "dispatch_mobile_generator": "Send a mobile generator to the site and connect it.",
                "switch_to_alternate_feed": "Switch the facility onto its alternate utility feed.",
                "expedite_feeder_repair": "Move the facility's feeder to the front of the repair queue while it rides on whatever it has.",
                "monitor_and_arrange_fuel": "Leave it on its own backup, help line up fuel, and check in on a schedule.",
            }),
            "runout_pressure": score("How close is the facility to losing its backup before normal supply is expected back?", [
                "Backup comfortably outlasts the expected restoration.",
                "Backup lasts past the restoration estimate, with a modest margin.",
                "Backup and the restoration estimate are about even, or the estimate is shaky.",
                "Backup runs out first, or there is no working backup at all.",
            ]),
            "loss_consequence": score("If the site lost all power, what would be at stake?", [
                "Inconvenience: service is interrupted but people are not put at risk.",
                "Public service: an essential service is degraded but can be worked around for a while.",
                "Health: vulnerable people or sanitation are harmed if it lasts for hours.",
                "Lives: patients on equipment, emergency calls or a dangerous release within hours.",
            ]),
            "mobile_unit_matches_load": noul(
                "One of the free mobile generators can carry this facility and be connected without major site work.",
                yes="A free unit is large enough for the site's load with room for motor starting, and the site has a connection point.",
                no="The free units are too small, none is free, or the site has no connection point for a portable unit."),
            "alternate_feed_viable": noul(
                "The facility's alternate utility feed can carry it right now."),
        }


WORKFLOW = Workflow(
    name="grid_outage_triage",
    title="Distribution grid outage triage",
    description="Restoration priority, re-energising, crew dispatch, switching and critical-site support decisions on a distribution grid.",
    domain="energy / electric utility operations",
    schemas=[OutageEventPriority(), FeederLockoutReclose(), SinglePremiseTroubleCall(), SwitchingOrderReview(),
             CriticalFacilityBackup()],
    notes="Invented utilities, substations and operating procedures; artefacts are OMS events, SCADA logs, trouble "
          "tickets, switching orders and key-account logs. No security alerts, no customer-service replies, no billing.",
)
